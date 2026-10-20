"""Explicit LightRAG 1.5.4 workspaces with durable catalog write fencing.

No environment variable is changed. Each backend operation holds the repository's
namespace lock until external IO completes; sealing takes that same lock.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import os
import re


from lightrag.kg.neo4j_impl import Neo4JStorage
from lightrag.kg.postgres_impl import (
    ClientManager, PGDocStatusStorage, PGKVStorage, PGVectorStorage, PostgreSQLDB,
    namespace_to_table_name,
)
from lightrag.kg.shared_storage import get_namespace_lock
from lightrag.utils import validate_workspace

_binding = ContextVar("catalog_storage_binding", default=None)


class BackendNotPrepared(RuntimeError):
    """Catalog runtime requires an explicitly prepared backend schema."""


class ConnectionOnlyPostgreSQLDB(PostgreSQLDB):
    async def configure_vector_extension(self, connection):
        # initdb and reconnect both dispatch through this hook. The upstream
        # implementation creates extensions; runtime must only validate them.
        row = await connection.fetchrow("SELECT extversion FROM pg_extension WHERE extname='vector'")
        if not row or not row["extversion"]:
            raise BackendNotPrepared("catalog backend requires a separately prepared vector extension")
        if self.vector_index_type == "HNSW_HALFVEC":
            match = re.match(r"^(\d+)\.(\d+)\.(\d+)", row["extversion"])
            if match is None or tuple(map(int, match.groups())) < (0, 7, 0):
                raise BackendNotPrepared("prepared pgvector >=0.7.0 is required for halfvec")
        if self.vector_index_type == "VCHORDRQ":
            extension = await connection.fetchrow("SELECT extversion FROM pg_extension WHERE extname='vchord'")
            if not extension:
                raise BackendNotPrepared("catalog backend requires a separately prepared vchord extension")


async def acquire_catalog_client():
    """Share raw upstream refcounts without its implicit schema/migration entry."""
    async with ClientManager._lock:
        config = ClientManager.get_config(vector_storage="PGVectorStorage")
        signature = ClientManager._build_vector_signature(config, "PGVectorStorage")
        current = ClientManager._instances["db"]
        if current is None:
            current = ConnectionOnlyPostgreSQLDB(config)
            try:
                await current.initdb()
            except BaseException:
                if current.pool is not None:
                    await current.pool.close()
                raise
            ClientManager._instances["db"] = current
            ClientManager._instances["ref_count"] = 0
            ClientManager._instances["vector_signature"] = signature
        else:
            ClientManager._assert_compatible_vector_signature(signature)
            if not isinstance(current, ConnectionOnlyPostgreSQLDB):
                raise BackendNotPrepared("drain the legacy PostgreSQL pool before opening catalog runtime")
        ClientManager._instances["ref_count"] += 1
        return current


async def verify_prepared_schema(db, table):
    """Read-only table/ownership-key preflight; runtime never repairs a schema."""
    if not table or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise BackendNotPrepared("catalog backend table identity is invalid")
    table = table.lower()
    verified = getattr(db, "_catalog_verified_tables", set())
    if table in verified:
        return
    try:
        await db.query(f'SELECT * FROM "{table}" LIMIT 0', [], multirows=True)
        key = await db.query(
            "SELECT EXISTS(SELECT 1 FROM pg_index i WHERE i.indrelid=to_regclass($1) "
            "AND i.indisunique AND i.indisvalid AND i.indpred IS NULL AND i.indexprs IS NULL "
            "AND i.indnkeyatts=2 AND i.indnatts=2 AND "
            "(SELECT array_agg(a.attname::text ORDER BY a.attname) "
            "FROM unnest(i.indkey::smallint[]) AS k(attnum) "
            "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum) "
            "=ARRAY['id','workspace']::text[]) AS prepared", [table])
        if not key or not key["prepared"]:
            raise BackendNotPrepared("prepared schema requires a unique (workspace,id) ownership key")
    except Exception as exc:
        raise BackendNotPrepared(f"catalog backend schema must be explicitly prepared: {table}") from exc
    verified.add(table)
    db._catalog_verified_tables = verified


@contextmanager
def storage_binding(repository, claim, *, read_only=False):
    token = _binding.set((repository, dict(claim), read_only))
    try:
        yield
    finally:
        _binding.reset(token)


class _CatalogStorage:
    def _bind(self):
        binding = _binding.get()
        if binding is None:
            raise RuntimeError("catalog storage requires an explicit namespace binding")
        self._catalog_repository, self._catalog_claim, self._catalog_read_only = binding
        if self.workspace != self._catalog_claim["namespace"]:
            raise ValueError("storage workspace differs from admitted namespace")
        validate_workspace(self.workspace)


class _PostgresStorage(_CatalogStorage):
    def __post_init__(self):
        super().__post_init__()
        self._bind()

    async def initialize(self):
        # Do not call upstream initialize: PostgreSQLDB.workspace overrides the
        # caller's workspace there. Keep the raw client for correct refcounts.
        if self.db is None:
            self.db = await acquire_catalog_client()
        try:
            table = getattr(self, "table_name", None) or namespace_to_table_name(self.namespace)
            await verify_prepared_schema(self.db, table)
        except BaseException:
            original, self.db = self.db, None
            await ClientManager.release_client(original)
            raise

    async def finalize(self):
        if self.db is not None:
            try:
                if isinstance(self, PGVectorStorage) and (
                    self._pending_vector_docs or self._pending_vector_deletes
                ):
                    await self._flush_pending_vector_ops()
            finally:
                original, self.db = self.db, None
                await ClientManager.release_client(getattr(original, "raw", original))


class CatalogPGKVStorage(_PostgresStorage, PGKVStorage):
    pass


class CatalogPGDocStatusStorage(_PostgresStorage, PGDocStatusStorage):
    pass


class CatalogPGVectorStorage(_PostgresStorage, PGVectorStorage):
    async def initialize(self):
        await super().initialize()
        if self._flush_lock is None:
            self._flush_lock = get_namespace_lock(self.namespace, workspace=self.workspace)


class CatalogNeo4JStorage(_CatalogStorage, Neo4JStorage):
    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(namespace, global_config, embedding_func, workspace)
        # Upstream's ctor replaces workspace from env, but does no IO. Restore
        # the explicit value before initialization or any graph access.
        self.workspace = workspace
        self._bind()

    async def initialize(self):
        if self._catalog_read_only:
            await self._initialize_reader()
            return
        async with self._catalog_repository.namespace_access(self._catalog_claim):
            await self._initialize_reader()

    async def _initialize_reader(self):
        # The upstream initializer creates databases and indexes. Published
        # readers only connect, preserving every upstream connection setting.
        from lightrag.kg import neo4j_impl as backend

        def value(environment, option, default=None):
            return os.environ.get(environment, backend.config.get("neo4j", option, fallback=default))

        self._driver = backend.AsyncGraphDatabase.driver(
            value("NEO4J_URI", "uri"),
            auth=(value("NEO4J_USERNAME", "username"), value("NEO4J_PASSWORD", "password")),
            max_connection_pool_size=int(value("NEO4J_MAX_CONNECTION_POOL_SIZE", "connection_pool_size", 100)),
            connection_timeout=float(value("NEO4J_CONNECTION_TIMEOUT", "connection_timeout", 30.0)),
            connection_acquisition_timeout=float(value("NEO4J_CONNECTION_ACQUISITION_TIMEOUT", "connection_acquisition_timeout", 30.0)),
            max_transaction_retry_time=float(value("NEO4J_MAX_TRANSACTION_RETRY_TIME", "max_transaction_retry_time", 30.0)),
            max_connection_lifetime=float(value("NEO4J_MAX_CONNECTION_LIFETIME", "max_connection_lifetime", 300.0)),
            liveness_check_timeout=float(value("NEO4J_LIVENESS_CHECK_TIMEOUT", "liveness_check_timeout", 300.0)),
            keep_alive=str(value("NEO4J_KEEP_ALIVE", "keep_alive", "true")).lower() in {"true", "1", "yes", "on"})
        database = os.environ.get("NEO4J_DATABASE", re.sub(r"[^a-zA-Z0-9-]", "-", self.namespace))
        for selected in (database, None):
            try:
                async with self._driver.session(database=selected, default_access_mode="READ") as session:
                    result = await session.run("MATCH (n) RETURN n LIMIT 0")
                    await result.consume()
                self._DATABASE = selected
                return
            except backend.neo4jExceptions.ClientError as exc:
                if selected is None or exc.code != "Neo.ClientError.Database.DatabaseNotFound":
                    raise

    async def _create_fulltext_index(self, driver, database, workspace_label):
        # Upstream also drops a global legacy index here. Catalog namespaces
        # must never modify another workspace's schema or graph.
        if self._catalog_read_only:
            return
        async with driver.session(database=database) as session:
            name = self._get_fulltext_index_name(workspace_label)
            result = await session.run(
                f"CREATE FULLTEXT INDEX `{name}` IF NOT EXISTS "
                f"FOR (n:`{workspace_label}`) ON EACH [n.entity_id]")
            await result.consume()


def _guard_method(method):
    @wraps(method)
    async def guarded(self, *args, **kwargs):
        if self._catalog_read_only:
            # Query keyword extraction can try to memoize; immutable publication
            # caches are readable but never extended by a reader.
            if method.__name__ == "upsert" and "llm_response_cache" in self.namespace:
                return None
            if method.__name__ in {"index_done_callback", "_flush_pending_vector_ops"}:
                if not (getattr(self, "_pending_vector_docs", None) or
                        getattr(self, "_pending_vector_deletes", None)):
                    return None
            raise RuntimeError("catalog publication storage is read-only")
        async with self._catalog_repository.namespace_access(self._catalog_claim):
            return await method(self, *args, **kwargs)
    return guarded


# Include buffering and flush methods: guarding only public upsert leaves a
# stale process able to flush a buffer after another process seals the receipt.
_MUTATORS = {
    "upsert", "delete", "drop", "index_done_callback", "drop_pending_index_ops",
    "_flush_pending_vector_ops", "delete_entity", "delete_entity_relation",
    "upsert_node", "upsert_edge", "upsert_nodes_batch", "upsert_edges_batch",
    "delete_node", "remove_nodes", "remove_edges",
}
for _cls in (CatalogPGKVStorage, CatalogPGDocStatusStorage,
             CatalogPGVectorStorage, CatalogNeo4JStorage):
    for _name in _MUTATORS:
        if hasattr(_cls, _name):
            setattr(_cls, _name, _guard_method(getattr(_cls, _name)))


def register_catalog_storages():
    from lightrag.kg import STORAGES, STORAGE_IMPLEMENTATIONS

    classes = {"KV_STORAGE": CatalogPGKVStorage,
               "VECTOR_STORAGE": CatalogPGVectorStorage,
               "DOC_STATUS_STORAGE": CatalogPGDocStatusStorage,
               "GRAPH_STORAGE": CatalogNeo4JStorage}
    for kind, cls in classes.items():
        STORAGES[cls.__name__] = __name__
        implementations = STORAGE_IMPLEMENTATIONS[kind]["implementations"]
        if cls.__name__ not in implementations:
            implementations.append(cls.__name__)


def resolve_layers(rows, layers, tombstones, role, key):
    """Resolve full replacement rows; never concatenate independently built summaries."""
    ranks = {namespace: index for index, namespace in enumerate(layers)}
    deleted = {}
    for row in tombstones:
        if row["role"] == role and row["namespace"] in ranks:
            output_id = row["output_id"]
            deleted[output_id] = max(deleted.get(output_id, -1), ranks[row["namespace"]])
    visible = {}
    for row in rows:
        namespace = row.get("workspace", row.get("_pv_namespace"))
        if namespace not in ranks:
            continue
        output_id = key(row)
        # A row recreated in the deleting layer supersedes its tombstone. A
        # delete physically removes that layer's row before sealing.
        if ranks[namespace] < deleted.get(output_id, -1):
            continue
        previous = visible.get(output_id)
        if previous is None or ranks[namespace] > ranks[previous.get("workspace", previous.get("_pv_namespace"))]:
            visible[output_id] = row
    return visible


class LayeredPostgresDB:
    """Apply manifest selection before any upstream predicate or vector ranking."""
    def __init__(self, raw, layers, tombstones, role):
        self.raw, self.layers, self.tombstones, self.role = raw, list(layers), tombstones, role

    def __getattr__(self, name):
        return getattr(self.raw, name)

    async def query(self, sql, params=None, multirows=False, **kwargs):
        params = list(params or [])
        workspace = re.search(r"\bworkspace\s*=\s*\$(\d+)", sql, re.I)
        table = re.search(r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.I)
        if workspace is None or table is None or not sql.lstrip().upper().startswith("SELECT"):
            raise RuntimeError("layered storage refuses an unscoped backend read")
        position = int(workspace.group(1))
        if position > len(params):
            raise RuntimeError("backend workspace parameter absent")
        params[position - 1] = self.layers
        ranks = {namespace: index for index, namespace in enumerate(self.layers, 1)}
        deleted = {}
        for row in self.tombstones:
            if row["role"] == self.role and row["namespace"] in ranks:
                deleted[row["output_id"]] = max(deleted.get(row["output_id"], 0), ranks[row["namespace"]])
        params.append(json.dumps(deleted))
        tombstone_position = len(params)
        parameter = f"${position}::varchar[]"
        # The visible CTE resolves overwrite/tombstone ownership BEFORE distance
        # filters and LIMIT in the original, unchanged LightRAG SQL query.
        visible = (
            f"WITH _pv_visible AS (SELECT DISTINCT ON (id) * FROM {table.group(1)} "
            f"WHERE workspace=ANY({parameter}) "
            f"AND array_position({parameter},workspace) >= "
            f"COALESCE((${tombstone_position}::jsonb ->> id)::int,0) "
            f"ORDER BY id,array_position({parameter},workspace) DESC) ")
        rewritten = sql[:table.start(1)] + "_pv_visible" + sql[table.end(1):]
        rewritten = re.sub(r"\bworkspace\s*=\s*\$" + str(position) + r"\b",
                           f"workspace=ANY({parameter})", rewritten, flags=re.I)
        return await self.raw.query(visible + rewritten, params=params, multirows=multirows, **kwargs)


def _edge_id(source, target):
    return json.dumps(sorted((source, target)), separators=(",", ":"))


def _properties(row):
    return {key: value for key, value in row.items() if key not in {"_pv_namespace", "workspace"}}


class LayeredGraphView:
    """Logical graph of the latest reducer-produced nodes and edges in a manifest."""
    def __init__(self, graph, layers, tombstones):
        self.graph, self.layers, self.tombstones = graph, list(layers), tombstones

    async def _records(self, query, **params):
        async with self.graph._driver.session(
            database=self.graph._DATABASE, default_access_mode="READ"
        ) as session:
            result = await session.run(query, namespaces=self.layers, **params)
            rows = [dict(record) async for record in result]
            await result.consume()
            return rows

    async def _nodes(self, node_ids=None):
        records = await self._records(
            "MATCH (n:PaperVaultCatalogEntity) "
            "WHERE n._pv_namespace IN $namespaces "
            "AND ($node_ids IS NULL OR n.entity_id IN $node_ids) "
            "RETURN properties(n) AS properties", node_ids=node_ids)
        return resolve_layers([row["properties"] for row in records], self.layers,
                              self.tombstones, "graph_nodes", lambda row: row["entity_id"])

    async def _edges(self, node_ids=None):
        records = await self._records(
            "MATCH (a:PaperVaultCatalogEntity)-[r:DIRECTED]->(b:PaperVaultCatalogEntity) "
            "WHERE r._pv_namespace IN $namespaces "
            "AND a._pv_namespace IN $namespaces AND b._pv_namespace IN $namespaces "
            "AND ($node_ids IS NULL OR a.entity_id IN $node_ids OR b.entity_id IN $node_ids) "
            "RETURN a.entity_id AS source,b.entity_id AS target,properties(r) AS properties",
            node_ids=node_ids)
        rows = [{**record["properties"], "source": record["source"], "target": record["target"]}
                for record in records]
        edges = resolve_layers(rows, self.layers, self.tombstones, "graph_edges",
                               lambda row: _edge_id(row["source"], row["target"]))
        endpoints = {value for row in edges.values() for value in (row["source"], row["target"])}
        nodes = await self._nodes(list(endpoints)) if endpoints else {}
        return {key: row for key, row in edges.items()
                if row["source"] in nodes and row["target"] in nodes}

    async def get_nodes_batch(self, node_ids):
        return {key: _properties(row) for key, row in (await self._nodes(node_ids)).items()}

    async def get_node(self, node_id):
        return (await self.get_nodes_batch([node_id])).get(node_id)

    async def has_node(self, node_id):
        return await self.get_node(node_id) is not None

    async def has_nodes_batch(self, node_ids):
        return set(await self.get_nodes_batch(node_ids))

    async def get_edges_batch(self, pairs):
        normalized = [(pair["src"], pair["tgt"]) if isinstance(pair, dict) else tuple(pair)
                      for pair in pairs]
        edges = await self._edges(list({node for pair in normalized for node in pair}))
        return {pair: _properties(edges[_edge_id(*pair)])
                for pair in normalized if _edge_id(*pair) in edges}

    async def get_edge(self, source, target):
        return (await self.get_edges_batch([(source, target)])).get((source, target))

    async def has_edge(self, source, target):
        return await self.get_edge(source, target) is not None

    async def get_nodes_edges_batch(self, node_ids):
        result = {node: [] for node in node_ids}
        for row in (await self._edges(node_ids)).values():
            edge = (row["source"], row["target"])
            for node in set(edge) & set(node_ids):
                result[node].append(edge)
        return result

    async def get_node_edges(self, node_id):
        if not await self.has_node(node_id):
            return None
        return (await self.get_nodes_edges_batch([node_id]))[node_id]

    async def node_degrees_batch(self, node_ids):
        return {node: len(edges) for node, edges in
                (await self.get_nodes_edges_batch(node_ids)).items()}

    async def node_degree(self, node_id):
        return (await self.node_degrees_batch([node_id]))[node_id]

    async def edge_degrees_batch(self, pairs):
        degrees = await self.node_degrees_batch(list({node for pair in pairs for node in pair}))
        return {tuple(pair): sum(degrees[node] for node in pair) for pair in pairs}

    async def edge_degree(self, source, target):
        return (await self.edge_degrees_batch([(source, target)]))[(source, target)]

    async def get_all_nodes(self):
        return [_properties(row) for row in (await self._nodes()).values()]

    async def get_all_edges(self):
        return [_properties(row) for row in (await self._edges()).values()]

    async def get_all_labels(self):
        return list(await self._nodes())


_PG_ROLES = ("full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks",
             "relation_chunks", "llm_response_cache", "doc_status", "chunks_vdb", "entities_vdb",
             "relationships_vdb")


def install_layered_storage(rag, repository, claim, *, read_only=False):
    """A private delta inherits one pinned aggregate and never mutates its bases."""
    layers = (list(claim["aggregate_layers"]) if read_only else
              list(claim.get("base_aggregate_layers", [])) + [claim["namespace"]])
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("aggregate requires a nonempty ordered set of namespace layers")
    tombstones = list(repository.list_tombstones(layers))
    rag._catalog_layers = layers
    rag._catalog_tombstones = tombstones

    def record(role, output_id):
        row = repository.record_tombstone(claim, role, str(output_id))
        if row not in tombstones:
            tombstones.append(row)

    def deleted(method, role):
        async def invoke(ids, *args, **kwargs):
            async with repository.namespace_access(claim):
                for output_id in ids:
                    record(role, output_id)
                return await method(ids, *args, **kwargs)
        return invoke

    for role in _PG_ROLES:
        store = getattr(rag, role, None)
        if store is None:
            continue
        store.db = LayeredPostgresDB(store.db, layers, tombstones, role)
        if not read_only and hasattr(store, "delete"):
            store.delete = deleted(store.delete, role)
        if not read_only and role in {"entities_vdb", "relationships_vdb"}:
            _install_vector_deletes(store, role, repository, claim, record)
    graph = getattr(rag, "chunk_entity_relation_graph", None)
    if graph is None:
        return
    view = LayeredGraphView(graph, layers, tombstones)
    raw = {}
    for name in ("get_nodes_batch", "get_node", "has_node", "has_nodes_batch", "get_edges_batch",
                 "get_edge", "has_edge", "get_nodes_edges_batch", "get_node_edges", "node_degrees_batch",
                 "node_degree", "edge_degrees_batch", "edge_degree", "get_all_nodes", "get_all_edges",
                 "get_all_labels"):
        raw[name] = getattr(graph, name)
        setattr(graph, name, getattr(view, name))
    graph._catalog_raw_reads = raw
    if not read_only:
        _install_graph_writes(graph, view, repository, claim, record)


def _install_vector_deletes(store, role, repository, claim, record):
    from lightrag.utils import compute_mdhash_id

    original_entity = store.delete_entity
    original_relations = store.delete_entity_relation

    async def delete_entity(entity):
        async with repository.namespace_access(claim):
            record(role, compute_mdhash_id(entity, prefix="ent-"))
            await original_entity(entity)

    async def delete_relations(entity):
        async with repository.namespace_access(claim):
            rows = await store.db.query(
                f"SELECT id FROM {store.table_name} WHERE workspace=$1 "
                "AND (source_id=$2 OR target_id=$2)", [claim["namespace"], entity], multirows=True)
            for row in rows:
                record(role, row["id"])
            await original_relations(entity)

    store.delete_entity = delete_entity
    store.delete_entity_relation = delete_relations


def _install_graph_writes(graph, view, repository, claim, record):
    upsert_node, upsert_nodes = graph.upsert_node, graph.upsert_nodes_batch
    upsert_edge, upsert_edges = graph.upsert_edge, graph.upsert_edges_batch
    delete_node, remove_edges = graph.delete_node, graph.remove_edges

    async def tag_nodes(ids):
        async with graph._driver.session(database=graph._DATABASE) as session:
            result = await session.run(
                f"MATCH (n:`{graph._get_workspace_label()}`) WHERE n.entity_id IN $ids "
                "SET n:PaperVaultCatalogEntity,n._pv_namespace=$namespace", ids=ids,
                namespace=claim["namespace"])
            await result.consume()

    async def node(node_id, node_data):
        async with repository.namespace_access(claim):
            await upsert_node(node_id, {**node_data, "_pv_namespace": claim["namespace"]})
            await tag_nodes([node_id])

    async def nodes(rows):
        async with repository.namespace_access(claim):
            await upsert_nodes([(key, {**value, "_pv_namespace": claim["namespace"]})
                                for key, value in rows])
            await tag_nodes([key for key, _ in rows])

    async def materialize_endpoints(ids):
        existing = await view.get_nodes_batch(list(ids))
        if set(existing) != set(ids):
            raise RuntimeError("graph edge references a missing logical endpoint")
        await nodes(list(existing.items()))

    async def edge(source_node_id, target_node_id, edge_data):
        async with repository.namespace_access(claim):
            await materialize_endpoints({source_node_id, target_node_id})
            await upsert_edge(source_node_id, target_node_id,
                              {**edge_data, "_pv_namespace": claim["namespace"]})

    async def edges(rows):
        async with repository.namespace_access(claim):
            await materialize_endpoints({endpoint for source, target, _ in rows
                                         for endpoint in (source, target)})
            await upsert_edges([(source, target, {**data, "_pv_namespace": claim["namespace"]})
                                for source, target, data in rows])

    async def delete(node_id):
        async with repository.namespace_access(claim):
            for source, target in await view.get_node_edges(node_id) or []:
                record("graph_edges", _edge_id(source, target))
            record("graph_nodes", node_id)
            await delete_node(node_id)

    async def remove(rows):
        async with repository.namespace_access(claim):
            for source, target in rows:
                record("graph_edges", _edge_id(source, target))
            await remove_edges(rows)

    graph.upsert_node, graph.upsert_nodes_batch = node, nodes
    graph.upsert_edge, graph.upsert_edges_batch = edge, edges
    graph.delete_node, graph.remove_edges = delete, remove
