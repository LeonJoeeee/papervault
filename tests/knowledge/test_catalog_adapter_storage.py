"""Catalog boundaries must hold even when dependency workspace env points elsewhere."""
from contextlib import asynccontextmanager

import pytest


class Gate:
    def __init__(self):
        self.sealed = False
        self.entries = 0

    @asynccontextmanager
    async def namespace_access(self, claim, **kwargs):
        if self.sealed:
            raise RuntimeError("sealed")
        self.entries += 1
        yield


@pytest.mark.asyncio
async def test_copy_on_write_graph_accepts_original_reducer_keyword_arguments():
    from types import SimpleNamespace
    from papervault.knowledge.store.catalog_storage import _install_graph_writes

    calls = []

    async def upsert(*args, **kwargs):
        calls.append((args, kwargs))

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def run(self, *args, **kwargs):
            return SimpleNamespace(consume=consume)

    async def consume():
        pass

    async def endpoints(ids):
        return {key: {"entity_id": key} for key in ids}

    graph = SimpleNamespace(upsert_node=upsert, upsert_nodes_batch=upsert, upsert_edge=upsert,
        upsert_edges_batch=upsert, delete_node=upsert, remove_edges=upsert, _DATABASE="test",
        _driver=SimpleNamespace(session=lambda **kwargs: Session()), _get_workspace_label=lambda: "attempt")
    _install_graph_writes(graph, SimpleNamespace(get_nodes_batch=endpoints), Gate(),
                          {"namespace": "attempt"}, lambda *args: None)
    await graph.upsert_node(node_id="A", node_data={"description": "original summary"})
    await graph.upsert_edge(source_node_id="A", target_node_id="B", edge_data={"weight": 1.0})
    assert calls[0][0][1]["description"] == "original summary"
    assert calls[-1][0][2]["weight"] == 1.0


@pytest.mark.asyncio
async def test_pg_kv_retains_explicit_workspace_and_releases_original_client(monkeypatch):
    from papervault.knowledge.store import catalog_storage
    from papervault.knowledge.store.catalog_storage import CatalogPGKVStorage, storage_binding
    from lightrag.kg.postgres_impl import ClientManager

    class DB:
        workspace = "production"

        async def query(self, *args, **kwargs):
            return {"prepared": True}

    db = DB()
    released = []

    async def acquire(**kwargs):
        db.vector_enabled = ClientManager.get_config(**kwargs)["enable_vector"]
        return db

    async def release(client):
        released.append(client)

    monkeypatch.setattr(catalog_storage, "acquire_catalog_client", acquire)
    monkeypatch.setattr(ClientManager, "release_client", release)
    gate = Gate()
    with storage_binding(gate, {"namespace": "pv_private"}):
        storage = CatalogPGKVStorage(namespace="full_docs", workspace="pv_private",
                                     global_config={"vector_storage": "CatalogPGVectorStorage"}, embedding_func=None)
    await storage.initialize()
    assert storage.workspace == "pv_private"
    assert db.workspace == "production"
    assert db.vector_enabled
    await storage.finalize()
    assert released == [db]


@pytest.mark.asyncio
async def test_sealed_namespace_blocks_kv_write_before_database_io():
    from papervault.knowledge.store.catalog_storage import CatalogPGKVStorage, storage_binding

    gate = Gate()
    with storage_binding(gate, {"namespace": "pv_private"}):
        storage = CatalogPGKVStorage(namespace="full_docs", workspace="pv_private",
                                     global_config={}, embedding_func=None)
    gate.sealed = True
    with pytest.raises(RuntimeError, match="sealed"):
        await storage.upsert({"doc": {"content": "unsafe"}})


def test_neo4j_constructor_does_not_inherit_environment_workspace(monkeypatch):
    from papervault.knowledge.store.catalog_storage import CatalogNeo4JStorage, storage_binding

    monkeypatch.setenv("NEO4J_WORKSPACE", "production")
    with storage_binding(Gate(), {"namespace": "pv_private"}):
        storage = CatalogNeo4JStorage(namespace="graph", workspace="pv_private",
                                      global_config={}, embedding_func=None)
    assert storage.workspace == "pv_private"
    assert storage._get_workspace_label() == "pv_private"


@pytest.mark.asyncio
async def test_read_only_cache_never_persists_query_side_effects():
    from papervault.knowledge.store.catalog_storage import CatalogPGKVStorage, storage_binding

    gate = Gate()
    gate.sealed = True
    with storage_binding(gate, {"namespace": "pv_private"}, read_only=True):
        storage = CatalogPGKVStorage(namespace="llm_response_cache", workspace="pv_private",
                                     global_config={}, embedding_func=None)
    await storage.upsert({"query": {"return": "answer"}})
    assert gate.entries == 0
    with pytest.raises(RuntimeError, match="read.only"):
        await storage.delete(["query"])


@pytest.mark.asyncio
async def test_cached_legacy_graph_rechecks_committed_authority(monkeypatch):
    import sys
    from types import SimpleNamespace
    from papervault.knowledge.store import graph

    def reject(mode):
        raise RuntimeError("catalog authority")

    monkeypatch.setitem(sys.modules, "papervault.store.knowledge_runtime",
                        SimpleNamespace(assert_knowledge_authority=reject))
    monkeypatch.setattr(graph, "_RAG", object())
    with pytest.raises(RuntimeError, match="catalog authority"):
        await graph.get_graph()


def test_layer_resolution_replaces_summary_before_traversal_and_honors_tombstones():
    from papervault.knowledge.store.catalog_storage import resolve_layers

    rows = [
        {"workspace": "base", "id": "Earth", "description": "old summary"},
        {"workspace": "delta", "id": "Earth", "description": "original reducer new summary"},
        {"workspace": "private", "id": "Earth", "description": "unpublished contamination"},
        {"workspace": "base", "id": "Mars", "description": "removed"},
    ]
    tombstones = [{"namespace": "delta", "role": "nodes", "output_id": "Mars"}]
    visible = resolve_layers(rows, ["base", "delta"], tombstones, "nodes", lambda row: row["id"])
    assert visible == {"Earth": rows[1]}


@pytest.mark.asyncio
async def test_vector_query_resolves_layer_replacements_before_distance_and_top_k():
    from papervault.knowledge.store.catalog_storage import LayeredPostgresDB

    class DB:
        async def query(self, sql, params=None, **kwargs):
            assert params[0] == ["base", "delta"]
            assert "DISTINCT ON (id)" in sql
            assert sql.index("DISTINCT ON (id)") < sql.index("content_vector <=>")
            assert "_pv_visible" in sql
            return [{"entity_name": "Earth"}]

    db = LayeredPostgresDB(DB(), ["base", "delta"], [], "entities_vdb")
    result = await db.query("SELECT entity_name FROM LIGHTRAG_VDB_ENTITY WHERE workspace=$1 "
                            "AND content_vector <=> $4::vector < $2 "
                            "ORDER BY content_vector <=> $4::vector LIMIT $3",
                            ["delta", 0.8, 10, [0.5, 0.2]], multirows=True)
    assert result == [{"entity_name": "Earth"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("table", [
    "LIGHTRAG_DOC_CHUNKS", "lightrag_llm_cache", "LIGHTRAG_VDB_ENTITY",
    "lightrag_vdb_entity_pv_acceptance_fixture_3d",
])
async def test_layered_query_preserves_extract_expressions_and_rewrites_real_table(table):
    from papervault.knowledge.store.catalog_storage import LayeredPostgresDB

    class DB:
        async def query(self, sql, params=None, **kwargs):
            return sql, params, kwargs

    query = ("SELECT EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at, "
             "EXTRACT(EPOCH FROM update_time)::BIGINT AS updated_at "
             f"FROM {table} WHERE workspace=$1 AND id=$2")
    db = LayeredPostgresDB(DB(), ["base", "delta"], [], "text_chunks")
    sql, params, options = await db.query(query, ["delta", "doc"], multirows=True)
    assert f"SELECT DISTINCT ON (id) * FROM {table} " in sql
    assert sql.endswith(
        "SELECT EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at, "
        "EXTRACT(EPOCH FROM update_time)::BIGINT AS updated_at "
        "FROM _pv_visible WHERE workspace=ANY($1::varchar[]) AND id=$2")
    assert params == [["base", "delta"], "doc", "{}"]
    assert options == {"multirows": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [
    "FROM LIGHTRAG_DOC_FULL AS a JOIN LIGHTRAG_DOC_FULL AS b ON a.id=b.id",
    "FROM (SELECT $2::varchar AS id) AS a JOIN LIGHTRAG_DOC_FULL AS b ON a.id=b.id",
])
async def test_layered_query_rewrites_genuine_from_and_join_references(source):
    from papervault.knowledge.store.catalog_storage import LayeredPostgresDB

    class DB:
        async def query(self, sql, **kwargs):
            return sql

    db = LayeredPostgresDB(DB(), ["base", "delta"], [], "full_docs")
    sql = await db.query(f"SELECT b.id {source} WHERE b.workspace=$1", ["delta", "doc"])
    assert "SELECT DISTINCT ON (id) * FROM LIGHTRAG_DOC_FULL " in sql
    assert "JOIN _pv_visible AS b ON a.id=b.id" in sql
    assert sql.count("LIGHTRAG_DOC_FULL") == 1
    assert sql.endswith("WHERE b.workspace=ANY($1::varchar[])")


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "SELECT EXTRACT(EPOCH FROM create_time) WHERE workspace=$1",
    "SELECT * FROM unrelated WHERE workspace=$1",
    "SELECT * FROM LIGHTRAG_DOC_FULL_extra WHERE workspace=$1",
    "SELECT * FROM LIGHTRAG_VDB_ENTITY_unqualified WHERE workspace=$1",
    "SELECT a.id FROM LIGHTRAG_DOC_FULL a JOIN LIGHTRAG_DOC_CHUNKS b ON a.id=b.id WHERE a.workspace=$1",
])
async def test_layered_query_refuses_unrecognized_or_mixed_table_reads_before_io(query):
    from papervault.knowledge.store.catalog_storage import LayeredPostgresDB

    class DB:
        async def query(self, *args, **kwargs):
            pytest.fail("unsupported reads must be refused before backend IO")

    db = LayeredPostgresDB(DB(), ["base", "delta"], [], "full_docs")
    with pytest.raises(RuntimeError):
        await db.query(query, ["delta"])


@pytest.mark.asyncio
async def test_graph_layer_view_keeps_cross_document_connectivity_and_original_summary():
    from papervault.knowledge.store.catalog_storage import LayeredGraphView

    class View(LayeredGraphView):
        async def _records(self, query, **params):
            if "RETURN properties(n)" in query:
                return [{"properties": row} for row in [
                    {"_pv_namespace": "base", "entity_id": "Earth", "description": "old"},
                    {"_pv_namespace": "delta", "entity_id": "Earth", "description": "reduced by LightRAG"},
                    {"_pv_namespace": "base", "entity_id": "Mars"},
                    {"_pv_namespace": "delta", "entity_id": "Sun"},
                ] if params.get("node_ids") is None or row["entity_id"] in params["node_ids"]]
            return [{"source": "Earth", "target": other, "properties": {"_pv_namespace": namespace}}
                    for other, namespace in [("Mars", "base"), ("Sun", "delta"), ("Secret", "private")]]

    view = View(None, ["base", "delta"], [])
    assert (await view.get_node("Earth"))["description"] == "reduced by LightRAG"
    assert set(await view.get_node_edges("Earth")) == {("Earth", "Mars"), ("Earth", "Sun")}
    assert await view.node_degree("Earth") == 2


@pytest.mark.asyncio
async def test_copy_on_write_delete_records_tombstone_before_private_backend_delete():
    from types import SimpleNamespace
    from papervault.knowledge.store.catalog_storage import install_layered_storage

    class Repository(Gate):
        def __init__(self):
            super().__init__()
            self.tombstones = []

        def list_tombstones(self, namespaces, role=None):
            return list(self.tombstones)

        def record_tombstone(self, claim, role, output_id):
            row = {"namespace": claim["namespace"], "role": role, "output_id": output_id}
            self.tombstones.append(row)
            return row

    repo = Repository()

    async def delete(ids):
        assert repo.tombstones == [{"namespace": "delta", "role": "full_docs", "output_id": "old"}]

    store = SimpleNamespace(db=object(), delete=delete)
    rag = SimpleNamespace(full_docs=store)
    install_layered_storage(rag, repo, {"namespace": "delta", "base_aggregate_layers": ["base"]})
    await rag.full_docs.delete(["old"])


@pytest.mark.asyncio
@pytest.mark.parametrize("read_only", [True, False])
async def test_graph_initialization_never_prepares_schema(monkeypatch, read_only):
    from lightrag.kg import neo4j_impl
    from papervault.knowledge.store.catalog_storage import CatalogNeo4JStorage, storage_binding

    queries = []

    class Result:
        async def consume(self):
            pass

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def run(self, query, **params):
            queries.append(query)
            return Result()

    class Driver:
        def session(self, **kwargs):
            return Session()

    monkeypatch.setattr(neo4j_impl.AsyncGraphDatabase, "driver", lambda *args, **kwargs: Driver())
    monkeypatch.setattr(neo4j_impl, "get_data_init_lock", lambda: Gate().namespace_access({}))
    with storage_binding(Gate(), {"namespace": "published"}, read_only=read_only):
        storage = CatalogNeo4JStorage(namespace="graph", workspace="published",
                                      global_config={}, embedding_func=None)
    await storage.initialize()
    assert queries
    assert all(query.strip().startswith("MATCH") for query in queries)


@pytest.mark.asyncio
async def test_catalog_factory_initializes_its_own_pipeline_status(monkeypatch, tmp_path):
    from lightrag.kg import shared_storage
    from papervault.knowledge.store import graph

    initialized = []

    async def status(**kwargs):
        initialized.append(kwargs.get("workspace"))

    class Rag:
        def __init__(self, **kwargs):
            self.workspace = kwargs["workspace"]

        async def initialize_storages(self):
            pass

        async def finalize_storages(self):
            pass

    class Repository(Gate):
        def list_tombstones(self, namespaces):
            return []

    monkeypatch.setattr(graph, "LightRAG", Rag)
    monkeypatch.setattr(shared_storage, "initialize_pipeline_status", status)
    rag = await graph.build_catalog_graph(Repository(), {"namespace": "private_attempt",
                                          "base_aggregate_layers": []}, working_dir=tmp_path)
    await rag.finalize_storages()
    assert initialized == ["private_attempt"]


@pytest.mark.asyncio
async def test_new_vector_namespace_never_runs_shared_legacy_migration_or_drop(monkeypatch):
    from lightrag.kg.postgres_impl import PGVectorStorage
    from lightrag.utils import EmbeddingFunc
    from papervault.knowledge.store import catalog_storage

    class DB:
        workspace = "production"

        async def query(self, *args, **kwargs):
            return {"prepared": True}

        async def _create_vector_index(self, table, dimension):
            raise AssertionError("runtime must not prepare vector indexes")

    async def acquire(**kwargs):
        return DB()

    async def unsafe_setup(*args, **kwargs):
        raise AssertionError("legacy setup may drop a shared table")

    monkeypatch.setattr(catalog_storage, "acquire_catalog_client", acquire)
    monkeypatch.setattr(PGVectorStorage, "setup_table", unsafe_setup)
    monkeypatch.setattr(PGVectorStorage, "_pg_create_table", unsafe_setup)
    monkeypatch.setattr(catalog_storage, "get_namespace_lock", lambda *args, **kwargs: object())
    with catalog_storage.storage_binding(Gate(), {"namespace": "private"}):
        storage = catalog_storage.CatalogPGVectorStorage(namespace="chunks", workspace="private",
            global_config={"embedding_batch_num": 32,
                           "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2}},
            embedding_func=EmbeddingFunc(embedding_dim=4, func=lambda texts: texts, model_name="fixture"))
    await storage.initialize()
    assert storage.workspace == "private"


def test_real_lightrag_constructor_binds_every_persistent_store_without_env_mutation(monkeypatch, tmp_path):
    from lightrag.utils import EmbeddingFunc
    from papervault.knowledge.store.build_pipeline import BuildAwareLightRAG
    from papervault.knowledge.store.catalog_storage import register_catalog_storages, storage_binding

    monkeypatch.setenv("NEO4J_WORKSPACE", "production")
    monkeypatch.setenv("POSTGRES_WORKSPACE", "production")
    register_catalog_storages()

    async def no_llm(*args, **kwargs):
        raise AssertionError("constructor must not perform inference")

    with storage_binding(Gate(), {"namespace": "private"}):
        rag = BuildAwareLightRAG(working_dir=str(tmp_path), workspace="private",
            llm_model_func=no_llm,
            kv_storage="CatalogPGKVStorage", vector_storage="CatalogPGVectorStorage",
            doc_status_storage="CatalogPGDocStatusStorage", graph_storage="CatalogNeo4JStorage",
            embedding_func=EmbeddingFunc(embedding_dim=4, func=lambda texts: texts, model_name="fixture"))
    assert len(rag._index_storages()) == 12
    assert {storage.workspace for storage in rag._index_storages()} == {"private"}
    import os
    assert os.environ["NEO4J_WORKSPACE"] == os.environ["POSTGRES_WORKSPACE"] == "production"
