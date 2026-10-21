"""Opt-in real PGvector + Neo4j acceptance, with no LLM/GPU dependency.

Set ALL PAPERVAULT_BACKEND_TEST_DSN and PAPERVAULT_BACKEND_TEST_NEO4J_
{URI,USER,PASSWORD,DATABASE}, then run:
  pytest -m integration tests/knowledge/test_catalog_backend_integration.py -q

Database names must contain a test/rehearsal token. Neo4j Community may use
"neo4j" only with PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE=1 and a different
host/port from the operator-supplied live NEO4J_URI. Preparation adds the pinned
tables/indexes; it never creates a Neo4j database. Cleanup deletes only this
test's random attempt namespaces and its disposable catalog schema. Shared
backend tables/indexes remain.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from ipaddress import ip_address
from socket import inet_aton
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

pytestmark = pytest.mark.integration

_MODEL = "pv_acceptance_fixture"
_DIM = 3
_LOOPBACK_ALIASES = (
    "127.1", "127.0.1", "127.000.000.001", "0177.0.0.1", "0x7f000001",
    "0x7f.0.0.1", "2130706433", "127.0.0.2", "127.255.255.255", "[::1]",
)


async def _fixture_embeddings(texts, **kwargs):
    import numpy as np

    vectors = {"vector:old": [1.0, 0.0, 0.0], "vector:new": [0.0, 1.0, 0.0],
               "vector:other": [0.8, 0.6, 0.0]}
    return np.asarray([vectors.get(text, [1.0, 0.0, 0.0]) for text in texts], dtype=np.float32)


def _embedding_contract():
    from importlib.metadata import version

    return {"schema_version": 1, "model": _MODEL, "dimension": _DIM,
        "max_length": 8192, "use_fp16": False, "representation": "fixture-float32-cosine-v1",
        "implementation_sha256": hashlib.sha256(inspect.getsource(_fixture_embeddings).encode()).hexdigest(),
        "packages": {"lightrag-hku": version("lightrag-hku"), "numpy": version("numpy")}}


def _neo4j_address(uri):
    """Compare endpoints independently of scheme, credentials and loopback spelling."""
    if not uri:
        raise ValueError("disposable Neo4j requires explicit target and configured live URIs")
    # urlsplit silently removes tabs/newlines; reject them before parsing.
    if any(character.isspace() for character in uri):
        raise ValueError("invalid Neo4j URI whitespace")
    try:
        parsed = urlsplit(uri)
        host, port = parsed.hostname, parsed.port
    except ValueError:
        raise ValueError("invalid Neo4j URI host/port") from None
    if parsed.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"} or not host or port == 0:
        raise ValueError("invalid Neo4j URI scheme or host/port")
    host = host.lower().removesuffix(".")
    try:
        address = ip_address(host)
    except ValueError:
        try:
            # Decode legacy numeric IPv4 spellings locally, without DNS resolution.
            address = ip_address(inet_aton(host))
        except OSError:
            address = None
    if address is not None:
        address = getattr(address, "ipv4_mapped", None) or address
        host = "localhost" if address.is_loopback else address.compressed
    elif len(host) > 253 or not all(
        re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
        for label in host.split(".")
    ):
        raise ValueError("invalid Neo4j URI hostname")
    return host, 7687 if port is None else port


def _guard_targets(targets):
    from psycopg.conninfo import conninfo_to_dict
    from scripts.prepare_knowledge_backend import _check_target

    pg = conninfo_to_dict(targets["PAPERVAULT_BACKEND_TEST_DSN"])
    if not all(pg.get(key) for key in ("dbname", "host", "user", "password")):
        raise ValueError("backend acceptance DSN requires explicit host/database/user/password")
    _check_target(pg["dbname"], allow_live=False)
    database = targets["PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE"]
    if database == "neo4j":
        if targets.get("PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE") != "1":
            raise ValueError("default neo4j requires explicit disposable opt-in; live administration is refused")
        if _neo4j_address(targets.get("PAPERVAULT_BACKEND_TEST_NEO4J_URI")) == _neo4j_address(targets.get("NEO4J_URI")):
            raise ValueError("disposable Neo4j target must differ from the configured live host/port")
    else:
        _check_target(database, allow_live=False)
    return pg


def _prepare_neo4j(driver, targets):
    from scripts.prepare_knowledge_backend import prepare_neo4j

    _guard_targets(targets)
    database = targets["PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE"]
    # Only this harness's validated disposable-default case may bypass the
    # preparation helper's name check. Production/CLI guards stay unchanged.
    return prepare_neo4j(driver, database=database, allow_live=database == "neo4j")


@pytest.mark.parametrize("pg_database,neo_database", [
    ("papervault", "backend-rehearsal"), ("papervault_catalog", "backend-rehearsal"),
    ("knowledge_system", "backend-rehearsal"), ("postgres", "backend-rehearsal"),
    ("backend_rehearsal", "neo4j"), ("backend_rehearsal", "system"),
    ("backend_rehearsal", "papervault"), ("unqualified", "backend-rehearsal"),
])
def test_backend_acceptance_refuses_live_or_unqualified_targets_before_connecting(pg_database, neo_database):
    with pytest.raises(ValueError, match="rehearsal|live"):
        _guard_targets({"PAPERVAULT_BACKEND_TEST_DSN":
            f"host=localhost dbname={pg_database} user=fixture password=fixture",
            "PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE": neo_database})


@pytest.mark.parametrize("disposable,target_uri,live_uri", [
    (None, "bolt://localhost:17687", "bolt://localhost:7687"),
    ("0", "bolt://localhost:17687", "bolt://localhost:7687"),
    ("true", "bolt://localhost:17687", "bolt://localhost:7687"),
    ("1", "bolt://localhost:7687", "bolt://localhost:7687"),
    ("1", "neo4j+s://LIVE.EXAMPLE:7687", "bolt://live.example"),
    ("1", "bolt://live.example.:7687", "neo4j://live.example:7687"),
    ("1", "bolt://127.0.0.1:7687", "bolt://localhost:7687"),
    ("1", "bolt://[::1]:7687", "bolt://localhost:7687"),
    ("1", "bolt://[::ffff:127.0.0.1]:7687", "bolt://localhost:7687"),
    *[("1", f"bolt://{host}:7687", "bolt://localhost:7687") for host in _LOOPBACK_ALIASES],
    *[("1", "bolt://localhost:7687", f"bolt://{host}:7687") for host in _LOOPBACK_ALIASES],
    ("1", "bolt://localhost:17687", None),
    ("1", "bolt://localhost:17687", ""),
    ("1", "bolt://localhost:17687", "not-a-uri"),
    ("1", "bolt://localhost:17687", "bolt://localhost:invalid"),
    ("1", None, "bolt://localhost:7687"),
    ("1", "not-a-uri", "bolt://localhost:7687"),
    ("1", "bolt://localhost:invalid", "bolt://localhost:7687"),
    ("1", "bolt://localhost:0", "bolt://localhost:7687"),
    ("1", "http://localhost:17687", "bolt://localhost:7687"),
])
def test_backend_acceptance_refuses_unsafe_disposable_neo4j_targets(disposable, target_uri, live_uri):
    with pytest.raises(ValueError):
        _guard_targets({"PAPERVAULT_BACKEND_TEST_DSN":
            "host=staging.invalid dbname=p4_rehearsal user=fixture password=fixture",
            "PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE": "neo4j",
            "PAPERVAULT_BACKEND_TEST_NEO4J_URI": target_uri,
            "PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE": disposable,
            "NEO4J_URI": live_uri})


@pytest.mark.parametrize("host", [
    "bad host", ".", "", " ", "bad\thost", "bad\nhost", "bad\rhost",
    "bad..host", "-bad.host", "bad-.host", "host..",
])
@pytest.mark.parametrize("invalid_side", ["PAPERVAULT_BACKEND_TEST_NEO4J_URI", "NEO4J_URI"])
def test_backend_acceptance_refuses_malformed_neo4j_hosts(host, invalid_side):
    targets = {"PAPERVAULT_BACKEND_TEST_DSN":
        "host=staging.invalid dbname=p4_rehearsal user=fixture password=fixture",
        "PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE": "neo4j",
        "PAPERVAULT_BACKEND_TEST_NEO4J_URI": "bolt://localhost:17687",
        "PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE": "1", "NEO4J_URI": "bolt://localhost:7687"}
    # A different port prevents endpoint equality from masking missing validation.
    port = 17687 if invalid_side == "PAPERVAULT_BACKEND_TEST_NEO4J_URI" else 7687
    targets[invalid_side] = f"bolt://{host}:{port}"
    with pytest.raises(ValueError, match="invalid Neo4j URI"):
        _guard_targets(targets)


@pytest.mark.parametrize("target_uri,live_uri", [
    ("bolt://localhost:17687", "bolt://localhost:7687"),
    ("neo4j://staging.invalid:7687", "bolt://live.invalid:7687"),
    ("bolt+ssc://[::1]:17687", "neo4j+s://localhost"),
    ("bolt://staging.invalid", "bolt://live.invalid:7687"),
    *[(f"bolt://{host}:17687", "bolt://localhost:7687") for host in _LOOPBACK_ALIASES],
    *[("bolt://localhost:17687", f"bolt://{host}:7687") for host in _LOOPBACK_ALIASES],
])
def test_backend_acceptance_accepts_disposable_default_on_separate_address(target_uri, live_uri):
    targets = {"PAPERVAULT_BACKEND_TEST_DSN":
        "host=staging.invalid dbname=p4_rehearsal user=fixture password=fixture",
        "PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE": "neo4j",
        "PAPERVAULT_BACKEND_TEST_NEO4J_URI": target_uri,
        "PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE": "1", "NEO4J_URI": live_uri}
    assert _guard_targets(targets)["dbname"] == "p4_rehearsal"

    class Driver:
        def __init__(self):
            self.statements = []

        @contextmanager
        def session(self, *, database):
            assert database == "neo4j"
            yield self

        def run(self, statement):
            self.statements.append(statement)
            return self

        def consume(self):
            pass

    driver = Driver()
    assert _prepare_neo4j(driver, targets) == {"indexes": 3}
    assert len(driver.statements) == 3
    assert all(query.startswith("CREATE INDEX IF NOT EXISTS") for query in driver.statements)

    # The preparation entry itself revalidates the target before opening a session.
    targets["NEO4J_URI"] = target_uri
    with pytest.raises(ValueError):
        _prepare_neo4j(driver, targets)
    assert len(driver.statements) == 3


@pytest.mark.parametrize("database", ["postgres", "papervault_catalog", "knowledge_system", "papervault"])
def test_backend_acceptance_refuses_postgres_defaults_with_disposable_neo4j(database):
    with pytest.raises(ValueError, match="rehearsal|live"):
        _guard_targets({"PAPERVAULT_BACKEND_TEST_DSN":
            f"host=staging.invalid dbname={database} user=fixture password=fixture",
            "PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE": "neo4j",
            "PAPERVAULT_BACKEND_TEST_NEO4J_URI": "bolt://localhost:17687",
            "PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE": "1", "NEO4J_URI": "bolt://localhost:7687"})


class BackendAcceptance:
    def __init__(self, repository, catalog, pg_connection):
        self.repository, self.catalog, self.pg_connection = repository, catalog, pg_connection
        self.namespaces, self.resources, self.llm_calls = [], [], []

    def claim(self, name):
        from papervault.store.repository import AuditEvent, Record

        k, r = self.repository, self.catalog
        unit = "paper:" + name
        r.upsert(Record.from_payload(name, {"key": name, "title": name}),
                 expected_revision=0, audit=AuditEvent("backend-test", "record"))
        k.command("register_unit", "unit:" + name,
                  {"unit_id": unit, "kind": "paper", "record_key": name, "identity": name})
        data = ("fixture:" + name).encode()
        digest = hashlib.sha256(data).hexdigest()
        k.command("register_input", "input:" + name, {"unit_id": unit,
            "consumed_input_revision": "input:" + name,
            "manifest": {"metadata_revision": "1", "artifacts": [{"artifact_id": digest,
                "sha256": digest, "byte_length": len(data)}], "upstream_derivation_ids": []}})
        recipe = k.command("register_recipe", "recipe:" + name, {"recipe": {
            "schema_version": 1, "code_version": "real-backend-acceptance",
            "embedding_contract": _embedding_contract(),
            "output_contract": {"fulltext": {"required": ["graph", "entities_vdb", "tombstones"],
                                             "not_applicable": []}}}})["recipe_digest"]
        k.command("set_requirement", "requirement:" + name, {"unit_id": unit,
            "consumed_input_revision": "input:" + name, "recipe_digest": recipe,
            "generation_id": "acceptance", "expected_revision": 0,
            "expected_record_revision": r.get(name).revision})
        job = k.command("enqueue", "enqueue:" + name, {"unit_id": unit,
            "consumed_input_revision": "input:" + name, "recipe_digest": recipe,
            "generation_id": "acceptance"})
        claim = k.command("claim", "claim:" + name, {"job_id": job["job_id"],
            "expected_revision": job["revision"], "authority_epoch": 1, "lease_seconds": 600})
        self.namespaces.append(claim["namespace"])
        return claim

    async def open(self, claim, *, read_only=False):
        from lightrag.kg.shared_storage import initialize_share_data
        from lightrag.utils import EmbeddingFunc
        from papervault.knowledge.store.catalog_storage import (
            CatalogNeo4JStorage, CatalogPGKVStorage, CatalogPGVectorStorage,
            install_layered_storage, storage_binding,
        )

        initialize_share_data()

        async def summarize(prompt, **kwargs):
            self.llm_calls.append(prompt)
            return "Fixture summary produced by the original LightRAG merge reducer."

        class Tokenizer:
            def encode(self, text):
                return list(text.encode())

            def decode(self, tokens):
                return bytes(tokens).decode()

        config = {"embedding_batch_num": 8, "vector_storage": "CatalogPGVectorStorage",
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": -0.01},
            "enable_llm_cache": True, "enable_llm_cache_for_entity_extract": True,
            "source_ids_limit_method": "KEEP", "max_source_ids_per_entity": 100,
            "max_source_ids_per_relation": 100, "tokenizer": Tokenizer(),
            "summary_context_size": 10000, "summary_max_tokens": 10000,
            "force_llm_summary_on_merge": 8, "summary_length_recommended": 100,
            "role_llm_funcs": {"extract": summarize}, "llm_model_name": "fixture-summary"}
        embedding = EmbeddingFunc(embedding_dim=_DIM, max_token_size=8192, func=_fixture_embeddings, model_name=_MODEL)
        with storage_binding(self.repository, claim, read_only=read_only):
            graph = CatalogNeo4JStorage("chunk_entity_relation", config, embedding, workspace=claim["namespace"])
            cache = CatalogPGKVStorage("llm_response_cache", claim["namespace"], config, embedding)
            chunks = CatalogPGKVStorage("entity_chunks", claim["namespace"], config, embedding)
            vectors = CatalogPGVectorStorage("entities", claim["namespace"], config, embedding)
        stores = [graph, cache, chunks, vectors]
        # Register before initialization so cleanup also closes a partially opened resource.
        self.resources.extend(stores)
        for store in stores:
            await store.initialize()
        assert graph._DATABASE == os.environ["NEO4J_DATABASE"]
        rag = SimpleNamespace(chunk_entity_relation_graph=graph, llm_response_cache=cache,
                              entity_chunks=chunks, entities_vdb=vectors, config=config)
        install_layered_storage(rag, self.repository, claim, read_only=read_only)
        return rag

    async def seal(self, claim, rag, *, publish=True):
        from papervault.knowledge.store.catalog_adapter import _graph_inventory, _inventory
        from papervault.store.repository import payload_hash

        await rag.entities_vdb.index_done_callback()
        stamp = datetime.now(timezone.utc).isoformat()
        graph = rag.chunk_entity_relation_graph
        # Inventory actual private rows, not the inherited logical graph.
        nodes = await graph._catalog_raw_reads["get_all_nodes"]()
        edges = await graph._catalog_raw_reads["get_all_edges"]()
        table = rag.entities_vdb.table_name
        rows = await rag.entities_vdb.db.raw.query(
            f'SELECT * FROM "{table.lower()}" WHERE workspace=$1 ORDER BY id',
            [claim["namespace"]], multirows=True)
        inventory = [_graph_inventory(claim["namespace"], nodes, edges, stamp),
            _inventory("entities_vdb", "postgres", claim["namespace"], [dict(row) for row in rows],
                       [row["id"] for row in rows], stamp),
            self.repository.tombstone_inventory(claim["namespace"])]
        inventory[1]["verification"]["evidence"]["backend_table"] = table.lower()
        receipt_id = "receipt:" + claim["attempt_id"]
        selected = {**claim["base_selected_unit_receipts"], claim["unit_id"]: receipt_id}
        receipt = {**claim, "receipt_id": receipt_id, "achieved_tier": "fulltext",
            "sealed": True, "durable": True, "output_inventory": inventory,
            "provider_observations": {"provider": "fixture", "model": "fixture-summary", "version": "1"},
            "origin_lineage": {}, "aggregate_layers": claim["base_aggregate_layers"] + [claim["namespace"]],
            "selected_unit_receipts": selected, "selected_unit_receipts_digest": payload_hash(selected)}
        async with self.repository.namespace_access(claim, seal=True):
            self.repository.command("seal_namespace", "seal:" + claim["attempt_id"],
                                    {"claim": claim, "receipt": receipt})
        if publish:
            self.repository.command("publish_derivation", "publish:" + claim["attempt_id"],
                {**claim, "receipt": receipt, "expected_revision": claim["revision"],
                 "expected_selection_revision": 0, "select": True})
        return receipt


@pytest.fixture
async def backend(monkeypatch, backend_test_targets):
    # This check precedes importing runtime storage or making ANY connection.
    targets = backend_test_targets
    optional = {"PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE", "NEO4J_URI"}
    missing = [key for key, value in targets.items() if key not in optional and not value]
    if missing:
        pytest.skip("real backend acceptance requires all explicit credentials: " + ", ".join(missing))
    pg = _guard_targets(targets)

    import psycopg
    from psycopg import sql
    from neo4j import GraphDatabase
    from scripts.prepare_knowledge_backend import prepare_postgres

    # Override only test configuration; storage classes, clients, queries and
    # catalog namespace gates below are the real implementations.
    for suffix, option in (("URI", "URI"), ("USER", "USERNAME"), ("PASSWORD", "PASSWORD"),
                           ("DATABASE", "DATABASE")):
        monkeypatch.setenv("NEO4J_" + option, targets["PAPERVAULT_BACKEND_TEST_NEO4J_" + suffix])
    monkeypatch.setenv("NEO4J_WORKSPACE", "deliberately_unrelated_test_workspace")
    from lightrag.kg.postgres_impl import ClientManager
    from papervault.store.repository import AuditEvent, PostgresCatalogRepository
    from papervault.store.knowledge_runtime import KnowledgeRepositoryProxy

    assert ClientManager._instances["db"] is None, "run backend acceptance in a process without a legacy pool"
    config = {"host": pg["host"], "port": int(pg.get("port", 5432)), "user": pg["user"],
        "password": pg["password"], "database": pg["dbname"], "workspace": "unrelated_test_workspace",
        "max_connections": 4, "ssl_mode": pg.get("sslmode"), "ssl_root_cert": pg.get("sslrootcert"),
        "ssl_cert": pg.get("sslcert"), "ssl_key": pg.get("sslkey"), "ssl_crl": pg.get("sslcrl"),
        "enable_vector": True, "vector_index_type": "HNSW", "hnsw_m": 16, "hnsw_ef": 64,
        "ivfflat_lists": 100, "vchordrq_build_options": "", "vchordrq_probes": "",
        "vchordrq_epsilon": 1.9, "server_settings": None, "statement_cache_size": 100,
        "connection_retry_attempts": 1, "connection_retry_backoff": 0.0,
        "connection_retry_backoff_max": 0.0, "pool_close_timeout": 5.0}
    monkeypatch.setattr(ClientManager, "get_config", staticmethod(lambda **kwargs: dict(config)))
    schema = "catalog_backend_acceptance_" + uuid.uuid4().hex
    harness, catalog_created = None, False
    with psycopg.connect(targets["PAPERVAULT_BACKEND_TEST_DSN"], autocommit=True, connect_timeout=5) as conn:
        plan = prepare_postgres(conn, embedding_dim=_DIM, embedding_model_name=_MODEL)
        with GraphDatabase.driver(targets["PAPERVAULT_BACKEND_TEST_NEO4J_URI"],
                auth=(targets["PAPERVAULT_BACKEND_TEST_NEO4J_USER"],
                      targets["PAPERVAULT_BACKEND_TEST_NEO4J_PASSWORD"])) as driver:
            # Explicit database must exist; fail here instead of runtime's
            # upstream-compatible missing-database fallback to the default DB.
            _prepare_neo4j(driver, targets)
            try:
                catalog = PostgresCatalogRepository(conn, schema=schema)
                catalog.ensure_schema(audit=AuditEvent("backend-test", "schema"))
                catalog_created = True
                catalog.ensure_knowledge_schema(audit=AuditEvent("backend-test", "knowledge-schema"))
                conn.execute(sql.SQL("UPDATE {} SET authority='catalog'").format(catalog._table("knowledge_runtime")))
                proxy = KnowledgeRepositoryProxy(targets["PAPERVAULT_BACKEND_TEST_DSN"], schema=schema)
                proxy.command("create_generation", "generation", {"generation_id": "acceptance", "candidate": False})
                harness = BackendAcceptance(proxy, catalog, conn)
                yield harness
            finally:
                cleanup_errors = []
                try:
                    if harness is not None:
                        for store in reversed(harness.resources):
                            try:
                                await store.finalize()
                            except Exception as exc:
                                cleanup_errors.append(exc)
                        for table in plan["tables"]:
                            try:
                                conn.execute(sql.SQL("DELETE FROM {} WHERE workspace=ANY(%s)").format(sql.Identifier(table)),
                                             (harness.namespaces,))
                            except Exception as exc:
                                cleanup_errors.append(exc)
                        try:
                            with driver.session(database=targets["PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE"]) as session:
                                session.run("MATCH (n) WHERE n._pv_namespace IN $namespaces DETACH DELETE n",
                                            namespaces=harness.namespaces).consume()
                        except Exception as exc:
                            cleanup_errors.append(exc)
                finally:
                    if catalog_created:
                        conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
                if cleanup_errors:
                    raise ExceptionGroup("backend acceptance cleanup failed", cleanup_errors)


async def _merge(rag, entity, descriptions, source_prefix):
    from lightrag.operate import _merge_nodes_then_upsert

    return await _merge_nodes_then_upsert(entity, [
        {"entity_type": "concept", "description": description,
         "source_id": f"{source_prefix}:{index}", "file_path": source_prefix}
        for index, description in enumerate(descriptions)], rag.chunk_entity_relation_graph,
        rag.entities_vdb, rag.config, llm_response_cache=rag.llm_response_cache,
        entity_chunks_storage=rag.entity_chunks)


@pytest.mark.asyncio
async def test_real_original_merge_summary_and_cross_document_traversal(backend):
    base_claim = backend.claim("first")
    base = await backend.open(base_claim)
    await _merge(base, "Shared", ["description from first document"], "first")
    await _merge(base, "FirstNeighbor", ["first neighbor"], "first")
    await base.chunk_entity_relation_graph.upsert_edge("Shared", "FirstNeighbor", {"weight": 1.0})
    await backend.seal(base_claim, base)

    private_claim = backend.claim("unpublished-graph")
    private = await backend.open(private_claim)
    await _merge(private, "Shared", ["unpublished poison"], "private")
    await _merge(private, "Secret", ["unpublished neighbor"], "private")
    await private.chunk_entity_relation_graph.upsert_edge("Shared", "Secret", {"weight": 99.0})
    await private.entities_vdb.index_done_callback()

    delta_claim = backend.claim("second")
    delta = await backend.open(delta_claim)
    descriptions = [f"additional document description {index}" for index in range(7)]
    merged = await _merge(delta, "Shared", descriptions, "second")
    assert merged["description"] == "Fixture summary produced by the original LightRAG merge reducer."
    assert len(backend.llm_calls) == 1
    assert "description from first document" in backend.llm_calls[0]
    assert "unpublished poison" not in backend.llm_calls[0]
    assert all(description in backend.llm_calls[0] for description in descriptions)
    await _merge(delta, "SecondNeighbor", ["second neighbor"], "second")
    await delta.chunk_entity_relation_graph.upsert_edge("Shared", "SecondNeighbor", {"weight": 2.0})
    receipt = await backend.seal(delta_claim, delta)
    reader = await backend.open({"namespace": delta_claim["namespace"],
                                 "aggregate_layers": receipt["aggregate_layers"]}, read_only=True)
    graph = reader.chunk_entity_relation_graph
    assert (await graph.get_node("Shared"))["description"] == merged["description"]
    assert set(await graph.get_node_edges("Shared")) == {
        ("Shared", "FirstNeighbor"), ("Shared", "SecondNeighbor")}
    assert await graph.node_degree("Shared") == 2
    assert not await graph.has_node("Secret")
    assert (await reader.entity_chunks.get_by_id("Shared"))["count"] == 8


@pytest.mark.asyncio
async def test_real_vectors_select_latest_before_distance_top_k_and_tombstones(backend):
    base_claim = backend.claim("vectors-base")
    base = await backend.open(base_claim)
    await base.entities_vdb.upsert({"same": {"entity_name": "Replaced", "content": "vector:old", "source_id": "a"},
        "other": {"entity_name": "Other", "content": "vector:other", "source_id": "b"}})
    await backend.seal(base_claim, base)
    private_claim = backend.claim("vectors-private")
    private = await backend.open(private_claim)
    await private.entities_vdb.upsert({"private": {"entity_name": "Unpublished", "content": "vector:old", "source_id": "d"}})
    await private.entities_vdb.index_done_callback()
    delta_claim = backend.claim("vectors-delta")
    delta = await backend.open(delta_claim)
    await delta.entities_vdb.upsert({"same": {"entity_name": "Replaced", "content": "vector:new", "source_id": "c"}})
    await delta.entities_vdb.index_done_callback()
    # The physically closest old version and unpublished vector must both be
    # excluded BEFORE LIMIT 1, leaving the next closest visible entity.
    assert [row["entity_name"] for row in await delta.entities_vdb.query("unused", 1, [1.0, 0.0, 0.0])] == ["Other"]
    await delta.entities_vdb.delete(["other"])
    await delta.entities_vdb.index_done_callback()
    assert [row["entity_name"] for row in await delta.entities_vdb.query("unused", 1, [1.0, 0.0, 0.0])] == ["Replaced"]
    receipt = await backend.seal(delta_claim, delta)
    reader = await backend.open({"namespace": delta_claim["namespace"],
                                 "aggregate_layers": receipt["aggregate_layers"]}, read_only=True)
    assert [row["entity_name"] for row in await reader.entities_vdb.query("unused", 1, [1.0, 0.0, 0.0])] == ["Replaced"]


@pytest.mark.asyncio
async def test_real_cache_isolation_and_catalog_rejects_sealed_or_stale_writes(backend):
    from psycopg import sql
    from papervault.store.knowledge import ClaimRejected

    claim = backend.claim("immutable")
    writer = await backend.open(claim)
    await writer.llm_response_cache.upsert({"cache-key": {"original_prompt": "fixture", "return": "published"}})
    receipt = await backend.seal(claim, writer, publish=False)
    with pytest.raises(ClaimRejected):
        await writer.llm_response_cache.upsert({"cache-key": {"original_prompt": "fixture", "return": "sealed mutation"}})
    with pytest.raises(ClaimRejected):
        await writer.chunk_entity_relation_graph.upsert_node("Forbidden", {"entity_id": "Forbidden"})
    with pytest.raises(ClaimRejected):
        await writer.entities_vdb.upsert({"forbidden": {"entity_name": "Forbidden", "content": "vector:old", "source_id": "x"}})
    reader = await backend.open({"namespace": claim["namespace"],
                                 "aggregate_layers": receipt["aggregate_layers"]}, read_only=True)
    private_claim = backend.claim("private-cache")
    private = await backend.open(private_claim)
    await private.llm_response_cache.upsert({"cache-key": {"original_prompt": "fixture", "return": "private"}})
    await reader.llm_response_cache.upsert({"cache-key": {"original_prompt": "fixture", "return": "query side effect"}})
    assert (await reader.llm_response_cache.get_by_id("cache-key"))["return"] == "published"
    assert (await private.llm_response_cache.get_by_id("cache-key"))["return"] == "private"
    await private.entities_vdb.upsert({"stale-buffer": {"entity_name": "StaleBuffered",
        "content": "vector:old", "source_id": "buffered-before-expiry"}})
    # Force DB-clock expiry only in this disposable catalog schema. Retain the
    # old live storage objects to exercise stale workers after a new claim.
    backend.pg_connection.execute(sql.SQL(
        "UPDATE {} SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE attempt_id=%s"
    ).format(backend.catalog._table("knowledge_attempts")), (private_claim["attempt_id"],))
    job = backend.repository.get_job(private_claim["job_id"])
    replacement = backend.repository.command("claim", "reclaim:private", {
        "job_id": job["job_id"], "expected_revision": job["revision"], "authority_epoch": 1,
        "lease_seconds": 600})
    backend.namespaces.append(replacement["namespace"])
    assert replacement["fencing_token"] > private_claim["fencing_token"]
    with pytest.raises(ClaimRejected):
        await private.llm_response_cache.upsert({"cache-key": {"original_prompt": "fixture", "return": "stale"}})
    with pytest.raises(ClaimRejected):
        await private.chunk_entity_relation_graph.upsert_node("Stale", {"entity_id": "Stale"})
    with pytest.raises(ClaimRejected):
        await private.entities_vdb.upsert({"stale": {"entity_name": "Stale", "content": "vector:old", "source_id": "x"}})
    with pytest.raises(ClaimRejected):
        await private.entities_vdb.index_done_callback()
    rows = await private.entities_vdb.db.raw.query(
        f'SELECT id FROM "{private.entities_vdb.table_name.lower()}" WHERE workspace=$1 AND id=$2',
        [private_claim["namespace"], "stale-buffer"], multirows=True)
    assert rows == []
    assert (await private.llm_response_cache.get_by_id("cache-key"))["return"] == "private"
    assert not await private.chunk_entity_relation_graph._catalog_raw_reads["has_node"]("Stale")
    assert not await writer.chunk_entity_relation_graph._catalog_raw_reads["has_node"]("Forbidden")
    # Finalization must still release its raw pool reference after the fenced
    # pending flush raises. A second cleanup finalize is consequently a no-op.
    with pytest.raises(ClaimRejected):
        await private.entities_vdb.finalize()
    assert (await reader.llm_response_cache.get_by_id("cache-key"))["return"] == "published"
    assert not await reader.chunk_entity_relation_graph.has_node("Forbidden")
    assert not await reader.chunk_entity_relation_graph.has_node("Stale")
