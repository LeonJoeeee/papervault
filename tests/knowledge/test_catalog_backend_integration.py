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
from collections import defaultdict
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

    def claim(self, name, *, input_store=None):
        from papervault.store.repository import AuditEvent, Record

        k, r = self.repository, self.catalog
        unit = "paper:" + name
        r.upsert(Record.from_payload(name, {"key": name, "title": name}),
                 expected_revision=0, audit=AuditEvent("backend-test", "record"))
        k.command("register_unit", "unit:" + name,
                  {"unit_id": unit, "kind": "paper", "record_key": name, "identity": name})
        data = ("fixture:" + name).encode()
        digest = hashlib.sha256(data).hexdigest()
        artifact = {"artifact_id": digest, "sha256": digest, "byte_length": len(data)}
        if input_store is not None:
            artifact = input_store.capture(data, role="fulltext_evidence")
        manifest = {"metadata_revision": "1", "artifacts": [artifact], "upstream_derivation_ids": []}
        if input_store is not None:
            manifest["tier"] = "fulltext"
        k.command("register_input", "input:" + name, {"unit_id": unit,
            "consumed_input_revision": "input:" + name,
            "manifest": manifest})
        recipe_data = {
            "schema_version": 1, "code_version": "real-backend-acceptance",
            "embedding_contract": _embedding_contract(),
            "output_contract": {"fulltext": {"required": ["graph", "entities_vdb", "tombstones"],
                                             "not_applicable": []}}}
        if input_store is not None:
            from lightrag.prompt import PROMPTS

            recipe_data["prompts"] = dict(PROMPTS)
        recipe = k.command("register_recipe", "recipe:" + name, {"recipe": recipe_data})["recipe_digest"]
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

    async def open(self, claim, *, read_only=False, for_deletion=False):
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
        if for_deletion:
            from functools import partial
            from lightrag import LightRAG
            from lightrag.kg.shared_storage import initialize_pipeline_status
            from papervault.knowledge.store.catalog_storage import CatalogPGDocStatusStorage

            specs = {"full_docs": (CatalogPGKVStorage, "full_docs"),
                "text_chunks": (CatalogPGKVStorage, "text_chunks"),
                "full_entities": (CatalogPGKVStorage, "full_entities"),
                "full_relations": (CatalogPGKVStorage, "full_relations"),
                "relation_chunks": (CatalogPGKVStorage, "relation_chunks"),
                "doc_status": (CatalogPGDocStatusStorage, "doc_status"),
                "chunks_vdb": (CatalogPGVectorStorage, "chunks"),
                "relationships_vdb": (CatalogPGVectorStorage, "relationships")}
            with storage_binding(self.repository, claim, read_only=read_only):
                additional = {role: cls(namespace, claim["namespace"], config, embedding)
                              for role, (cls, namespace) in specs.items()}
            self.resources.extend(additional.values())
            stores.extend(additional.values())
            for role, store in additional.items():
                await store.initialize()
                setattr(rag, role, store)
            rag.workspace = claim["namespace"]
            config["workspace"] = rag.workspace
            rag._index_storages = lambda: stores
            rag._build_global_config = lambda: config
            for method in ("adelete_by_doc_id", "_insert_done", "_update_delete_retry_state"):
                setattr(rag, method, partial(getattr(LightRAG, method), rag))
            await initialize_pipeline_status(workspace=rag.workspace)
        install_layered_storage(rag, self.repository, claim, read_only=read_only)
        return rag

    async def seal(self, claim, rag, *, publish=True, doc_id=None):
        from papervault.knowledge.store.catalog_adapter import (
            _canonical_graph, _document_graph_contribution, _graph_inventory, _inventory,
        )
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
        if doc_id is not None:
            from lightrag.kg.postgres_impl import namespace_to_table_name

            membership = {}
            for role in ("entity_chunks", "relation_chunks"):
                store = getattr(rag, role)
                table = getattr(store, "table_name", None) or namespace_to_table_name(store.namespace)
                rows = await store.db.raw.query(
                    f'SELECT * FROM "{table.lower()}" WHERE workspace=$1 ORDER BY id',
                    [claim["namespace"]], multirows=True)
                membership[role] = {row["id"]: dict(row) for row in rows}
            status = await rag.doc_status.get_by_id(doc_id)
            nodes, edges = _canonical_graph(nodes, edges)
            contribution = await _document_graph_contribution(
                rag, doc_id, status["chunks_list"], nodes, edges, membership)
            inventory[0]["verification"]["evidence"]["document_contribution"] = contribution
        receipt_id = "receipt:" + claim["attempt_id"]
        selected = {**claim["base_selected_unit_receipts"], claim["unit_id"]: receipt_id}
        receipt = {**claim, "receipt_id": receipt_id, "achieved_tier": "fulltext",
            "sealed": True, "durable": True, "output_inventory": inventory,
            "provider_observations": {"provider": "fixture", "model": "fixture-summary", "version": "1"},
            "origin_lineage": {}, "aggregate_layers": claim["base_aggregate_layers"] + [claim["namespace"]],
            "selected_unit_receipts": selected, "selected_unit_receipts_digest": payload_hash(selected)}
        if doc_id is not None:
            receipt["provider_observations"]["doc_id"] = doc_id
        if claim["retired_receipt_ids"]:
            from papervault.knowledge.store.catalog_adapter import verify_delta_outputs

            verified, _ = await verify_delta_outputs(rag, claim)
            verified_graph = next(row for row in verified if row["role"] == "graph")
            assert verified_graph["sha256"] == inventory[0]["sha256"]
            receipt["retirement_verification"] = {"version": 1,
                "retired_receipt_ids": claim["retired_receipt_ids"],
                "graph_sha256": verified_graph["sha256"]}
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
async def test_real_original_relationship_source_append_accepts_full_entity_vector_content(backend):
    from lightrag.operate import _merge_edges_then_upsert
    from lightrag.utils import compute_mdhash_id
    from papervault.knowledge.store.catalog_adapter import verify_delta_outputs

    claim = backend.claim("relationship-entity-content")
    rag = await backend.open(claim, for_deletion=True)
    rag.config["embedding_token_limit"] = 8192
    await _merge(rag, "Shared", ["d" * 9000], "A")
    for store in rag._index_storages():
        await store.index_done_callback()
    vector_id = compute_mdhash_id("Shared", prefix="ent-")
    initial = await rag.entities_vdb.get_by_id(vector_id)
    # The character tokenizer leaves 7936 characters after the reducer's
    # 256-token reserve; the seven-character entity prefix uses that budget.
    assert initial["content"] == "Shared\n" + "d" * 7929
    await verify_delta_outputs(rag, claim)

    await _merge_edges_then_upsert("Shared", "Neighbor", [{
        "description": "relationship evidence", "keywords": "related",
        "source_id": "B:0", "file_path": "B", "weight": 1.0}],
        rag.chunk_entity_relation_graph, rag.relationships_vdb, rag.entities_vdb, rag.config,
        entity_chunks_storage=rag.entity_chunks, relation_chunks_storage=rag.relation_chunks,
        llm_response_cache=rag.llm_response_cache)
    for store in rag._index_storages():
        await store.index_done_callback()
    updated = await rag.entities_vdb.get_by_id(vector_id)
    assert updated["content"] == "Shared\n" + "d" * 9000
    assert set(updated["chunk_ids"]) == {"A:0", "B:0"}
    assert (await rag.chunk_entity_relation_graph.get_node("Shared"))["description"] == "d" * 9000
    assert backend.llm_calls == []
    await verify_delta_outputs(rag, claim)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "entity", "no-cache", "mixed-cache", "no-description"])
async def test_real_retirement_rebuild_preserves_shared_evidence_and_parent_aggregate(backend, monkeypatch, failure, tmp_path):
    from lightrag.utils import compute_mdhash_id
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    old_claim = backend.claim("retirement-old")
    old = await backend.open(old_claim, for_deletion=True)
    await _merge(old, "Shared", ["description from retired document"], "old")
    await _merge(old, "Exclusive", ["only retired evidence"], "old")
    incomplete = failure in {"mixed-cache", "no-description"}
    if incomplete:
        await _merge(old, "CachedPeer", ["retired peer evidence"], "old")
    await old.full_docs.upsert({"old-doc": {"content": "old evidence", "file_path": "old"}})
    stamp = datetime.now(timezone.utc).isoformat()
    await old.doc_status.upsert({"old-doc": {"status": "processed", "content_summary": "old",
        "content_length": 12, "chunks_count": 1, "chunks_list": ["old:0"],
        "created_at": stamp, "updated_at": stamp, "file_path": "old"}})
    await old.text_chunks.upsert({"old:0": {"tokens": 2, "chunk_order_index": 0,
        "full_doc_id": "old-doc", "content": "old evidence", "file_path": "old"}})
    old_entities = ["Shared", "Exclusive"] + (["CachedPeer"] if incomplete else [])
    await old.full_entities.upsert({"old-doc": {"entity_names": old_entities, "count": len(old_entities)}})
    await old.full_relations.upsert({"old-doc": {"relation_pairs": [], "count": 0}})
    await backend.seal(old_claim, old, doc_id="old-doc")

    other_claim = backend.claim("retirement-other")
    other = await backend.open(other_claim, for_deletion=True)
    await _merge(other, "Shared", ["description from retained document"], "other")
    await other.text_chunks.upsert({"other:0": {"tokens": 4, "chunk_order_index": 0,
        "full_doc_id": "other-doc", "content": "retained evidence", "file_path": "other",
        "llm_cache_list": [] if failure in {"no-cache", "mixed-cache"} else ["retained-extraction"]}})
    await other.llm_response_cache.upsert({"retained-extraction": {"cache_type": "extract",
        "chunk_id": "other:0", "original_prompt": "fixture extraction", "return":
        "entity<|#|>Shared<|#|>concept<|#|>description from retained document\n<|COMPLETE|>"}})
    if incomplete:
        await _merge(other, "CachedPeer", ["retained peer evidence"], "other-peer")
        await other.text_chunks.upsert({"other-peer:0": {"tokens": 4, "chunk_order_index": 0,
            "full_doc_id": "other-doc", "content": "retained peer evidence", "file_path": "other",
            "llm_cache_list": ["retained-peer-extraction"]}})
        await other.llm_response_cache.upsert({"retained-peer-extraction": {"cache_type": "extract",
            "chunk_id": "other-peer:0", "original_prompt": "fixture peer extraction", "return":
            "entity<|#|>CachedPeer<|#|>concept<|#|>retained peer evidence\n<|COMPLETE|>"}})
        if failure == "no-description":
            await other.llm_response_cache.upsert({"retained-extraction": {"cache_type": "extract",
                "chunk_id": "other:0", "original_prompt": "fixture extraction", "return":
                "entity<|#|>Unrelated<|#|>concept<|#|>unrelated cached evidence\n<|COMPLETE|>"}})
    parent_receipt = await backend.seal(other_claim, other)

    k = backend.repository
    replacement_input = "input:retirement-replacement"
    data = b"replacement evidence"
    manifest = {"metadata_revision": "2", "artifacts": [{"artifact_id": hashlib.sha256(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(), "byte_length": len(data)}], "upstream_derivation_ids": []}
    replacement_recipe = old_claim["recipe_digest"]
    if incomplete:
        from copy import deepcopy
        from lightrag.prompt import PROMPTS
        from papervault.knowledge.scheduler import catalog_intake
        from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore

        input_store = ImmutableInputStore(tmp_path / "retirement-inputs")
        manifest = {"metadata_revision": "2", "tier": "metadata",
            "artifacts": [input_store.capture(b'{"title":"metadata downgrade"}', role="metadata")],
            "upstream_derivation_ids": []}
        recipe = deepcopy(k.get_recipe(old_claim["recipe_digest"]))
        recipe["prompts"] = deepcopy(dict(PROMPTS))
        recipe["output_contract"]["metadata"] = {"required": ["metadata", "graph", "tombstones"]}
        replacement_recipe = k.command("register_recipe", "retirement:metadata-recipe", {"recipe": recipe})["recipe_digest"]
        monkeypatch.setattr(catalog_intake, "effective_distill_recipe", lambda: deepcopy(recipe))
    k.command("register_input", "retirement:input", {"unit_id": old_claim["unit_id"],
        "consumed_input_revision": replacement_input, "manifest": manifest})
    requirement = k.get_requirement(old_claim["unit_id"], "acceptance")
    k.command("set_requirement", "retirement:requirement", {"unit_id": old_claim["unit_id"],
        "consumed_input_revision": replacement_input, "recipe_digest": replacement_recipe,
        "generation_id": "acceptance", "expected_revision": requirement["revision"],
        "expected_record_revision": backend.catalog.get("retirement-old").revision})
    job = k.command("enqueue", "retirement:enqueue", {"unit_id": old_claim["unit_id"],
        "consumed_input_revision": replacement_input, "recipe_digest": replacement_recipe,
        "generation_id": "acceptance"})
    claim = k.command("claim", "retirement:claim", {"job_id": job["job_id"],
        "expected_revision": job["revision"], "authority_epoch": 1, "lease_seconds": 600})
    backend.namespaces.append(claim["namespace"])
    assert claim["retired_receipt_ids"] == ["receipt:" + old_claim["attempt_id"]]
    delta = await backend.open(claim, for_deletion=True)
    if incomplete:
        from papervault.store.knowledge import RetirementFailure

        async def factory(*args, **kwargs):
            async def finalize():
                for store in delta._index_storages():
                    await store.finalize()
            delta.finalize_storages = finalize
            return delta

        with pytest.raises(RetirementFailure) as caught:
            await CatalogLightRAGAdapter(k, rag_factory=factory).submit_metadata(claim, manifest, input_store)
        assert "isolated rebuild required" in caught.value.evidence["rebuild_validation_error"]
        residual = await backend.open(claim, for_deletion=True)
        # The inherited graph/vector still carry the retired contribution. The
        # refusal protects serving until an isolated rebuild can repair it.
        assert "old:0" in (await residual.chunk_entity_relation_graph.get_node("Shared"))["source_id"]
        vector = await residual.entities_vdb.get_by_id(compute_mdhash_id("Shared", prefix="ent-"))
        assert "old:0" in vector["chunk_ids"]
        assert k.get_submission(claim["submission_id"])["receipt_id"] is None
        assert k.get_job(claim["job_id"])["state"] == "running"
        return
    if failure == "entity":
        async def reject_rebuild(*args, **kwargs):
            raise OSError("disposable backend entity write rejected")
        monkeypatch.setattr(delta.chunk_entity_relation_graph, "upsert_node", reject_rebuild)
    if failure is not None:
        from papervault.store.knowledge import RetirementFailure

        with pytest.raises(RetirementFailure) as caught:
            await CatalogLightRAGAdapter(k)._retire(delta, claim)
        if failure == "no-cache":
            # Prevalidation now stops deletion before the original reducer's
            # no-cache success path, retaining document retry state explicitly.
            assert caught.value.status == "fail"
            assert "every remaining chunk" in caught.value.evidence["rebuild_validation_error"]
            assert await delta.doc_status.get_by_id("old-doc") is not None
        else:
            # Actual backend errors are still swallowed by the original reducer;
            # operation history must reject its successful deletion result.
            assert caught.value.status == "success"
            assert caught.value.messages[0] == "Document old-doc successfully deleted"
            assert any("Failed" in message for message in caught.value.messages)
        assert k.get_submission(claim["submission_id"])["receipt_id"] is None
        assert k.get_job(claim["job_id"])["state"] == "running"
        return
    await CatalogLightRAGAdapter(k)._retire(delta, claim)

    shared = await delta.chunk_entity_relation_graph.get_node("Shared")
    assert shared["description"] == "description from retained document"
    assert shared["source_id"] == "other:0"
    assert (await delta.entity_chunks.get_by_id("Shared"))["chunk_ids"] == ["other:0"]
    assert not await delta.chunk_entity_relation_graph.has_node("Exclusive")
    assert await delta.doc_status.get_by_id("old-doc") is None
    assert await delta.text_chunks.get_by_id("old:0") is None
    assert compute_mdhash_id("Exclusive", prefix="ent-") not in await delta.entities_vdb.get_vectors_by_ids(
        [compute_mdhash_id("Exclusive", prefix="ent-")])
    await backend.seal(claim, delta, publish=False)

    parent = await backend.open({"namespace": other_claim["namespace"],
        "aggregate_layers": parent_receipt["aggregate_layers"]}, read_only=True, for_deletion=True)
    assert await parent.chunk_entity_relation_graph.has_node("Exclusive")
    assert "retired document" in (await parent.chunk_entity_relation_graph.get_node("Shared"))["description"]
    assert (await parent.doc_status.get_by_id("old-doc"))["status"] == "processed"
    assert k.get_job(claim["job_id"])["state"] == "running"


class _RetirementProcessDeath(BaseException):
    """Interrupt outside Exception handlers, as a killed worker would."""


def _retirement_vector(rag, kind):
    from lightrag.utils import compute_mdhash_id

    if kind == "entity":
        return rag.entities_vdb, compute_mdhash_id("Shared", prefix="ent-")
    return rag.relationships_vdb, compute_mdhash_id("NeighborShared", prefix="rel-")


async def _seed_private_ranking_probe(rag):
    """Give a failed namespace a distinguishable nearest-neighbor result."""
    from lightrag.utils import compute_mdhash_id

    await _merge(rag, "PrivateRetirementOnly", ["unpublished ranking evidence"], "private")
    await rag.relationships_vdb.upsert({compute_mdhash_id("PrivateRetirementOnlyShared", prefix="rel-"): {
        "src_id": "PrivateRetirementOnly", "tgt_id": "Shared", "source_id": "private:0",
        "content": "vector:old"}})
    await rag.entities_vdb.index_done_callback()
    await rag.relationships_vdb.index_done_callback()


async def _crash_before_retirement_vector(context, monkeypatch, kind, vector_state):
    await _seed_private_ranking_probe(context.delta)
    store, vector_id = _retirement_vector(context.delta, kind)
    original_upsert, original_delete = store.upsert, store.delete

    async def interrupt(data):
        if vector_id in data:
            if vector_state == "absent":
                await original_delete([vector_id])
            raise _RetirementProcessDeath("graph committed before replacement vector")
        return await original_upsert(data)

    async def preserve_stale(ids):
        return await original_delete([item for item in ids if item != vector_id])

    # The original relationship reducer deletes both orientations before its
    # replacement upsert. Preserve that inherited row only for the stale case.
    if kind == "relation" and vector_state == "stale":
        monkeypatch.setattr(store, "delete", preserve_stale)
    monkeypatch.setattr(store, "upsert", interrupt)
    adapter = _adapter_with_actual_stores(context.k, context.delta, context.inputs)
    with pytest.raises(_RetirementProcessDeath):
        await adapter.submit_metadata(context.claim, context.manifest, context.inputs)
    monkeypatch.setattr(store, "upsert", original_upsert)
    monkeypatch.setattr(store, "delete", original_delete)


async def _assert_selected_retirement_parent(backend, context):
    """A failed private graph and vector must never enter selected retrieval."""
    k, claim = context.k, context.claim
    manifest = k.get_manifest("acceptance")["manifest"]
    assert claim["namespace"] not in manifest["aggregate_layers"]
    reader = await backend.open({"namespace": context.other_claim["namespace"],
        "aggregate_layers": manifest["aggregate_layers"]}, read_only=True, for_deletion=True)
    assert "old:0" in (await reader.chunk_entity_relation_graph.get_node("Shared"))["source_id"]
    assert "old:0" in (await reader.chunk_entity_relation_graph.get_edge("Shared", "Neighbor"))["source_id"]
    assert not await reader.chunk_entity_relation_graph.has_node("PrivateRetirementOnly")
    for kind in ("entity", "relation"):
        store, vector_id = _retirement_vector(reader, kind)
        assert "old:0" in (await store.get_by_id(vector_id))["chunk_ids"]
        ranked = await store.query("unused", 100, [1.0, 0.0, 0.0])
        if kind == "entity":
            assert "Shared" in {row["entity_name"] for row in ranked}
            assert "PrivateRetirementOnly" not in {row["entity_name"] for row in ranked}
        else:
            assert ("Neighbor", "Shared") in {(row["src_id"], row["tgt_id"]) for row in ranked}
            assert all("PrivateRetirementOnly" not in (row["src_id"], row["tgt_id"]) for row in ranked)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["entity", "relation"])
@pytest.mark.parametrize("vector_state", ["stale", "absent"])
async def test_real_interrupted_retirement_requires_isolated_rebuild_on_same_lease(
        backend, monkeypatch, tmp_path, kind, vector_state):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.store.knowledge import ClaimRejected, RetirementFailure

    context = await _metadata_retirement_context(backend, monkeypatch, tmp_path)
    k, claim = context.k, context.claim
    before_manifest = k.get_manifest("acceptance")
    before_selection = k.get_selection(claim["unit_id"], "acceptance")
    await _crash_before_retirement_vector(context, monkeypatch, kind, vector_state)
    residual = await backend.open(claim, for_deletion=True)
    if kind == "entity":
        row = await residual.chunk_entity_relation_graph.get_node("Shared")
        tracker = await residual.entity_chunks.get_by_id("Shared")
        assert row["description"] == "other description for Shared"
    else:
        from lightrag.utils import make_relation_chunk_key

        row = await residual.chunk_entity_relation_graph.get_edge("Shared", "Neighbor")
        tracker = await residual.relation_chunks.get_by_id(make_relation_chunk_key("Shared", "Neighbor"))
        assert row["description"] == "other relationship evidence"
    assert row["source_id"] == "other:0"
    assert tracker["chunk_ids"] == ["other:0"]
    store, vector_id = _retirement_vector(residual, kind)
    vector = await store.get_by_id(vector_id)
    if vector_state == "stale":
        assert "old:0" in vector["chunk_ids"]
    else:
        assert vector is None

    adapter = _adapter_with_actual_stores(k, residual, context.inputs)
    scheduler = CatalogScheduler(k, adapter, context.inputs)
    delivery = next(row for row in k.list_outbox(kind="submission")
                    if row["submission_id"] == claim["submission_id"])
    summary = defaultdict(int)
    await scheduler._deliver(delivery, summary)

    assert summary["published"] == 0 and summary["repairs"] == summary["discrepancies"] == 1
    assert k.get_submission(claim["submission_id"])["receipt_id"] is None
    assert k.get_attempt(claim["attempt_id"])["sealed"] is False
    assert k.get_job(claim["job_id"])["state"] == "failed"
    assert k.get_attempt(claim["attempt_id"])["outcome"] == "failed"
    assert k.get_manifest("acceptance") == before_manifest
    assert k.get_selection(claim["unit_id"], "acceptance") == before_selection
    discrepancy = backend.pg_connection.execute(
        "SELECT reason,evidence FROM " + backend.catalog._table("knowledge_discrepancies").as_string(
            backend.pg_connection)).fetchone()
    assert discrepancy["reason"] == "retirement_rebuild_unverified"
    assert "isolated rebuild" in str(discrepancy["evidence"]).lower()
    repairs = k.list_outbox(kind="repair")
    assert len(repairs) == 1
    assert repairs[0]["payload"]["isolated_rebuild_required"] is True
    with pytest.raises((ClaimRejected, RetirementFailure)):
        await adapter.submit_metadata(claim, context.manifest, context.inputs)
    with pytest.raises(ClaimRejected):
        await backend.open(claim, for_deletion=True)
    with pytest.raises(ClaimRejected):
        await residual.chunk_entity_relation_graph.upsert_node("Poison", {"entity_id": "Poison"})
    with pytest.raises(ClaimRejected):
        async with k.namespace_access(claim, seal=True):
            pytest.fail("failed retirement namespace entered the sealing gate")
    again = defaultdict(int)
    await scheduler._deliver(delivery, again)
    assert again["published"] == 0
    assert k.get_submission(claim["submission_id"])["receipt_id"] is None
    await _assert_selected_retirement_parent(backend, context)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["entity", "relation"])
async def test_real_interrupted_retirement_lease_expiry_rebuilds_in_new_namespace(
        backend, monkeypatch, tmp_path, kind):
    from psycopg import sql
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.store.knowledge import ClaimRejected

    context = await _metadata_retirement_context(backend, monkeypatch, tmp_path)
    k, old = context.k, context.claim
    await _crash_before_retirement_vector(context, monkeypatch, kind, "stale")
    backend.pg_connection.execute(sql.SQL(
        "UPDATE {} SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE attempt_id=%s"
    ).format(backend.catalog._table("knowledge_attempts")), (old["attempt_id"],))
    job = k.get_job(old["job_id"])
    claim = k.command("claim", "retirement:after-death", {"job_id": job["job_id"],
        "expected_revision": job["revision"], "authority_epoch": 1, "lease_seconds": 600})
    backend.namespaces.append(claim["namespace"])
    assert claim["attempt_id"] != old["attempt_id"]
    assert claim["submission_id"] != old["submission_id"]
    assert claim["namespace"] != old["namespace"]
    assert claim["fencing_token"] > old["fencing_token"]
    assert old["namespace"] not in claim["base_aggregate_layers"]
    with pytest.raises(ClaimRejected):
        await context.delta.chunk_entity_relation_graph.upsert_node("Stale", {"entity_id": "Stale"})
    delta = await backend.open(claim, for_deletion=True)
    scheduler = CatalogScheduler(k, _adapter_with_actual_stores(k, delta, context.inputs), context.inputs)
    delivery = next(row for row in k.list_outbox(kind="submission")
                    if row["submission_id"] == claim["submission_id"])
    summary = defaultdict(int)
    await scheduler._deliver(delivery, summary)
    assert summary["published"] == 1 and summary["discrepancies"] == 0
    receipt = k.get_submission(claim["submission_id"])["receipt"]
    assert old["namespace"] not in receipt["aggregate_layers"]
    reader = await backend.open({"namespace": claim["namespace"],
        "aggregate_layers": receipt["aggregate_layers"]}, read_only=True, for_deletion=True)
    assert (await reader.chunk_entity_relation_graph.get_node("Shared"))["description"] == "other description for Shared"
    assert (await reader.chunk_entity_relation_graph.get_edge("Shared", "Neighbor"))["description"] == "other relationship evidence"
    for role in ("entity", "relation"):
        store, vector_id = _retirement_vector(reader, role)
        row = await store.get_by_id(vector_id)
        assert row["chunk_ids"] == ["other:0"]
        assert "old" not in row["content"]
    assert await reader.doc_status.get_by_id("old-doc") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,damage", [
    (kind, damage) for damage in (None, "content", "identity", "source")
    for kind in ("entity", "relation")
] + [("relation", "stale-reverse")])
async def test_real_retirement_graph_vector_agreement_gates_sealing(
        backend, monkeypatch, tmp_path, kind, damage):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler

    context = await _metadata_retirement_context(backend, monkeypatch, tmp_path)
    k, claim, delta = context.k, context.claim, context.delta
    if damage is not None:
        await _seed_private_ranking_probe(delta)
    store, vector_id = _retirement_vector(delta, kind)
    original_upsert = store.upsert

    async def corrupt_replacement(data):
        if vector_id in data and damage is not None:
            data = {key: dict(value) for key, value in data.items()}
            if damage == "content":
                data[vector_id]["content"] = "retired evidence survives only in embedding input"
            elif damage == "identity":
                data[vector_id]["entity_name" if kind == "entity" else "tgt_id"] = "WrongIdentity"
            elif damage == "source":
                data[vector_id]["source_id"] = "old:0<SEP>other:0"
        await original_upsert(data)
        if vector_id in data and damage == "stale-reverse":
            from lightrag.utils import compute_mdhash_id

            await original_upsert({compute_mdhash_id("SharedNeighbor", prefix="rel-"): {
                **data[vector_id], "src_id": "Shared", "tgt_id": "Neighbor",
                "source_id": "old:0<SEP>other:0",
                "content": "related\tShared\nNeighbor\nretired relationship evidence"}})

    monkeypatch.setattr(store, "upsert", corrupt_replacement)
    before_manifest = k.get_manifest("acceptance")
    scheduler = CatalogScheduler(k, _adapter_with_actual_stores(k, delta, context.inputs), context.inputs)
    delivery = next(row for row in k.list_outbox(kind="submission")
                    if row["submission_id"] == claim["submission_id"])
    summary = defaultdict(int)
    await scheduler._deliver(delivery, summary)
    if damage is None:
        assert summary["published"] == 1 and summary["discrepancies"] == 0
        assert k.get_attempt(claim["attempt_id"])["sealed"] is True
        receipt = k.get_submission(claim["submission_id"])["receipt"]
        reader = await backend.open({"namespace": claim["namespace"],
            "aggregate_layers": receipt["aggregate_layers"]}, read_only=True, for_deletion=True)
        vectors, vector_id = _retirement_vector(reader, kind)
        row = await vectors.get_by_id(vector_id)
        assert row["content"] == ("Shared\nother description for Shared" if kind == "entity"
            else "related\tNeighbor\nShared\nother relationship evidence")
        assert row["chunk_ids"] == ["other:0"]
        return
    assert summary["published"] == 0 and summary["discrepancies"] == 1
    assert k.get_submission(claim["submission_id"])["receipt_id"] is None
    assert k.get_attempt(claim["attempt_id"])["sealed"] is False
    assert k.get_manifest("acceptance") == before_manifest
    await _assert_selected_retirement_parent(backend, context)


async def _seed_indexed_document(rag, doc_id, prefix, text, track_id, *, exclusive=True):
    """Write deterministic extraction through original reducers and actual stores."""
    from lightrag.operate import _merge_edges_then_upsert

    entities = ["Shared", "Neighbor"] + (["Exclusive"] if exclusive else [])
    pairs = [["Shared", "Neighbor"]] + ([["Exclusive", "Neighbor"]] if exclusive else [])
    for entity in entities:
        await _merge(rag, entity, [f"{prefix} description for {entity}"], prefix)
    for source, target in pairs:
        await _merge_edges_then_upsert(source, target, [{
            "weight": 1.0, "keywords": "related", "description": f"{prefix} relationship evidence",
            "source_id": prefix + ":0", "file_path": prefix}], rag.chunk_entity_relation_graph,
            rag.relationships_vdb, rag.entities_vdb, rag.config,
            llm_response_cache=rag.llm_response_cache,
            relation_chunks_storage=rag.relation_chunks, entity_chunks_storage=rag.entity_chunks)
    stamp = datetime.now(timezone.utc).isoformat()
    await rag.full_docs.upsert({doc_id: {"content": text, "file_path": prefix}})
    await rag.doc_status.upsert({doc_id: {"status": "processed", "content_summary": text,
        "content_length": len(text), "chunks_count": 1, "chunks_list": [prefix + ":0"],
        "created_at": stamp, "updated_at": stamp, "file_path": prefix, "track_id": track_id}})
    await rag.text_chunks.upsert({prefix + ":0": {"tokens": 4, "chunk_order_index": 0,
        "full_doc_id": doc_id, "content": text, "file_path": prefix,
        "llm_cache_list": [prefix + "-extraction"]}})
    await rag.chunks_vdb.upsert({prefix + ":0": {"content": text,
        "tokens": 4, "chunk_order_index": 0, "full_doc_id": doc_id, "file_path": prefix}})
    extraction = [f"entity<|#|>{entity}<|#|>concept<|#|>{prefix} description for {entity}" for entity in entities]
    extraction.extend(f"relation<|#|>{source}<|#|>{target}<|#|>related<|#|>{prefix} relationship evidence"
                      for source, target in pairs)
    await rag.llm_response_cache.upsert({prefix + "-extraction": {"cache_type": "extract",
        "chunk_id": prefix + ":0", "original_prompt": "fixture extraction",
        "return": "\n".join([*extraction, "<|COMPLETE|>"])}})
    await rag.full_entities.upsert({doc_id: {"entity_names": entities, "count": len(entities)}})
    await rag.full_relations.upsert({doc_id: {"relation_pairs": pairs, "count": len(pairs)}})
    for store in rag._index_storages():
        await store.index_done_callback()


async def _damage_document_index(rag, doc_id, damage):
    if damage is None:
        return
    if damage == "empty-status":
        status = await rag.doc_status.get_by_id(doc_id)
        await rag.doc_status.upsert({doc_id: {**status, "chunks_list": [], "chunks_count": 0}})
        return
    role, mode = damage
    store = getattr(rag, role)
    if mode == "missing":
        await store.delete([doc_id])
    else:
        field = "entity_names" if role == "full_entities" else "relation_pairs"
        row = await store.get_by_id(doc_id)
        await store.upsert({doc_id: {**row, field: row[field][:1], "count": 1}})


def _adapter_with_actual_stores(repository, rag, input_store):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    async def factory(*args, **kwargs):
        async def finalize():
            for store in rag._index_storages():
                await store.finalize()
        rag.finalize_storages = finalize
        return rag

    return CatalogLightRAGAdapter(repository, rag_factory=factory, input_store=input_store)


_INDEX_DAMAGE = [None, ("full_entities", "missing"), ("full_entities", "partial"),
                 ("full_relations", "missing"), ("full_relations", "partial")]


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", _INDEX_DAMAGE)
async def test_real_document_indexes_gate_submission_sealing_and_publication(backend, monkeypatch, tmp_path, damage):
    from papervault.knowledge.scheduler import catalog_intake
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore

    inputs = ImmutableInputStore(tmp_path / "indexed-inputs")
    claim = backend.claim("indexed-build", input_store=inputs)
    k = backend.repository
    rag = await backend.open(claim, for_deletion=True)

    async def insert(text, *, ids, file_paths, track_id):
        await _seed_indexed_document(rag, ids, "build", text, track_id)
        await _damage_document_index(rag, ids, damage)

    rag.ainsert = insert
    monkeypatch.setattr(catalog_intake, "effective_distill_recipe", lambda: k.get_recipe(claim["recipe_digest"]))
    adapter = _adapter_with_actual_stores(k, rag, inputs)
    before_manifest = k.get_manifest("acceptance")
    scheduler = CatalogScheduler(k, adapter, inputs)
    summary = defaultdict(int)
    delivery = next(row for row in k.list_outbox(kind="submission")
                    if row["submission_id"] == claim["submission_id"])

    await scheduler._deliver(delivery, summary)

    if damage is None:
        receipt = k.get_submission(claim["submission_id"])["receipt"]
        graph = next(row for row in receipt["output_inventory"] if row["role"] == "graph")
        assert graph["verification"]["evidence"]["document_contribution"] == {
            "doc_id": "doc-" + hashlib.sha256(claim["submission_id"].encode()).hexdigest(),
            "chunk_ids": ["build:0"], "entity_ids": ["Exclusive", "Neighbor", "Shared"],
            "relation_pairs": [["Exclusive", "Neighbor"], ["Neighbor", "Shared"]]}
        assert summary["published"] == 1 and summary["discrepancies"] == 0
        assert k.get_selection(claim["unit_id"], "acceptance") is not None
        return
    assert summary["discrepancies"] == 1 and summary["published"] == 0
    assert k.get_submission(claim["submission_id"])["receipt_id"] is None
    assert k.get_job(claim["job_id"])["state"] == "running"
    assert k.get_selection(claim["unit_id"], "acceptance") is None
    assert k.get_manifest("acceptance") == before_manifest
    assert k.get_attempt(claim["attempt_id"])["sealed"] is False
    discrepancy = backend.pg_connection.execute(
        "SELECT reason,evidence FROM " + backend.catalog._table("knowledge_discrepancies").as_string(
            backend.pg_connection)).fetchone()
    assert discrepancy["reason"] == "submission_outcome_unknown"
    assert discrepancy["evidence"]["error_type"] == "RuntimeError"


async def _metadata_retirement_context(backend, monkeypatch, tmp_path):
    from copy import deepcopy
    from lightrag.prompt import PROMPTS
    from papervault.knowledge.scheduler import catalog_intake
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore

    old_claim = backend.claim("indexed-old")
    old = await backend.open(old_claim, for_deletion=True)
    await _seed_indexed_document(old, "old-doc", "old", "retired document", old_claim["submission_id"])
    old_receipt = await backend.seal(old_claim, old, doc_id="old-doc")
    other_claim = backend.claim("indexed-other")
    other = await backend.open(other_claim, for_deletion=True)
    await _seed_indexed_document(other, "other-doc", "other", "retained document",
                                 other_claim["submission_id"], exclusive=False)
    parent_receipt = await backend.seal(other_claim, other, doc_id="other-doc")
    inputs = ImmutableInputStore(tmp_path / "indexed-metadata")
    manifest = {"metadata_revision": "2", "tier": "metadata",
        "artifacts": [inputs.capture(b'{"title":"metadata downgrade"}', role="metadata")],
        "upstream_derivation_ids": []}
    k = backend.repository
    recipe = deepcopy(k.get_recipe(old_claim["recipe_digest"]))
    recipe["prompts"] = dict(PROMPTS)
    recipe["output_contract"]["metadata"] = {"required": ["metadata", "graph", "tombstones"]}
    recipe_digest = k.command("register_recipe", "indexed-metadata:recipe", {"recipe": recipe})["recipe_digest"]
    monkeypatch.setattr(catalog_intake, "effective_distill_recipe", lambda: deepcopy(recipe))
    desired = {"unit_id": old_claim["unit_id"], "consumed_input_revision": "indexed-metadata",
               "generation_id": "acceptance", "recipe_digest": recipe_digest}
    k.command("register_input", "indexed-metadata:input", {
        "unit_id": old_claim["unit_id"], "consumed_input_revision": "indexed-metadata", "manifest": manifest})
    requirement = k.get_requirement(old_claim["unit_id"], "acceptance")
    k.command("set_requirement", "indexed-metadata:requirement", {**desired,
        "expected_revision": requirement["revision"],
        "expected_record_revision": backend.catalog.get("indexed-old").revision})
    job = k.command("enqueue", "indexed-metadata:enqueue", desired)
    claim = k.command("claim", "indexed-metadata:claim", {"job_id": job["job_id"],
        "expected_revision": job["revision"], "authority_epoch": 1, "lease_seconds": 600})
    backend.namespaces.append(claim["namespace"])
    delta = await backend.open(claim, for_deletion=True)
    return SimpleNamespace(k=k, old_claim=old_claim, old_receipt=old_receipt,
        other_claim=other_claim, parent_receipt=parent_receipt, inputs=inputs,
        manifest=manifest, claim=claim, delta=delta)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [*_INDEX_DAMAGE, "empty-status"])
async def test_real_document_indexes_gate_metadata_retirement_and_preserve_selected_shared_material(
        backend, monkeypatch, tmp_path, damage):
    from lightrag.utils import compute_mdhash_id
    from papervault.knowledge.scheduler.catalog import CatalogScheduler

    context = await _metadata_retirement_context(backend, monkeypatch, tmp_path)
    k, claim, delta, inputs = context.k, context.claim, context.delta, context.inputs
    old_claim, old_receipt = context.old_claim, context.old_receipt
    other_claim, parent_receipt = context.other_claim, context.parent_receipt
    await _damage_document_index(delta, "old-doc", damage)
    before_manifest = k.get_manifest("acceptance")
    before_selection = k.get_selection(old_claim["unit_id"], "acceptance")
    adapter = _adapter_with_actual_stores(k, delta, inputs)
    scheduler = CatalogScheduler(k, adapter, inputs)
    summary = defaultdict(int)
    delivery = next(row for row in k.list_outbox(kind="submission")
                    if row["submission_id"] == claim["submission_id"])

    await scheduler._deliver(delivery, summary)

    if damage is None:
        assert summary["published"] == 1 and summary["discrepancies"] == 0 and summary["repairs"] == 0
        receipt = k.get_submission(claim["submission_id"])["receipt"]
        assert receipt["achieved_tier"] == "metadata"
        assert k.get_selection(old_claim["unit_id"], "acceptance") != before_selection
        reader = await backend.open({"namespace": claim["namespace"],
            "aggregate_layers": receipt["aggregate_layers"]}, read_only=True, for_deletion=True)
        assert (await reader.chunk_entity_relation_graph.get_node("Shared"))["source_id"] == "other:0"
        assert (await reader.chunk_entity_relation_graph.get_edge("Shared", "Neighbor"))["source_id"] == "other:0"
        assert (await reader.entities_vdb.get_by_id(compute_mdhash_id("Shared", prefix="ent-")))["chunk_ids"] == ["other:0"]
        assert not await reader.chunk_entity_relation_graph.has_node("Exclusive")
        assert await reader.doc_status.get_by_id("old-doc") is None
    else:
        assert summary["discrepancies"] == summary["repairs"] == 1 and summary["published"] == 0
        assert k.get_submission(claim["submission_id"])["receipt_id"] is None
        assert k.get_job(claim["job_id"])["state"] == "failed"
        assert k.get_attempt(claim["attempt_id"])["sealed"] is False
        assert k.get_manifest("acceptance") == before_manifest
        assert k.get_selection(old_claim["unit_id"], "acceptance") == before_selection
        repairs = [row for row in k.list_outbox(kind="repair")
                   if row["payload"]["publication_id"] == k.get_receipt_publications(old_receipt["receipt_id"])[0]["publication_id"]]
        assert len(repairs) == 1 and repairs[0]["payload"]["reason"] == "retirement_rebuild_unverified"
        discrepancy = backend.pg_connection.execute(
            "SELECT reason,evidence FROM " + backend.catalog._table("knowledge_discrepancies").as_string(
                backend.pg_connection)).fetchone()
        assert discrepancy["reason"] == "retirement_rebuild_unverified"
        assert "index" in discrepancy["evidence"]["retirement_evidence"]["index_validation_error"]
        reader = await backend.open({**claim,
            "aggregate_layers": claim["base_aggregate_layers"] + [claim["namespace"]]},
            read_only=True, for_deletion=True)
        assert "old:0" in (await reader.chunk_entity_relation_graph.get_node("Shared"))["source_id"]
        assert "old:0" in (await reader.chunk_entity_relation_graph.get_edge("Shared", "Neighbor"))["source_id"]
        assert "old:0" in (await reader.entities_vdb.get_by_id(compute_mdhash_id("Shared", prefix="ent-")))["chunk_ids"]
        assert "old:0" in (await reader.relationships_vdb.get_by_id(
            compute_mdhash_id("NeighborShared", prefix="rel-")))["chunk_ids"]
        assert await reader.chunk_entity_relation_graph.has_node("Exclusive")
        assert await reader.doc_status.get_by_id("old-doc") is not None
    parent = await backend.open({"namespace": other_claim["namespace"],
        "aggregate_layers": parent_receipt["aggregate_layers"]}, read_only=True, for_deletion=True)
    assert "old:0" in (await parent.chunk_entity_relation_graph.get_node("Shared"))["source_id"]
    assert "old:0" in (await parent.chunk_entity_relation_graph.get_edge("Shared", "Neighbor"))["source_id"]
    assert await parent.chunk_entity_relation_graph.has_node("Exclusive")
    assert (await parent.doc_status.get_by_id("old-doc"))["status"] == "processed"
