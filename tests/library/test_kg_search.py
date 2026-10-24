"""KG candidate retrieval against controlled model/database boundaries."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from papervault.library.models import Paper
from papervault.library.state_v2 import Merge, Policy, TerminalState


class Library:
    def __init__(self, *papers):
        self.papers = {paper.key: paper for paper in papers}

    def all_papers(self, *, include_quarantined=False):
        assert include_quarantined
        return list(self.papers.values())

    def get(self, key):
        return self.papers.get(key)


def paper(key, **kwargs):
    return Paper(key=key, title="Cosmic ray transport", authors=["A. Researcher"], **kwargs)


def search(library, concepts=None, **kwargs):
    from papervault.library.services.kg_search import kg_local_search

    return kg_local_search("full scientific question", SimpleNamespace(
        terms=["fallback phrase"], kg_concepts=concepts), library, **kwargs)


@pytest.fixture
def backends(monkeypatch):
    for name, value in {
        "PAPERVAULT_STORAGE_BACKEND": "postgres",
        "POSTGRES_HOST": "pg.invalid", "POSTGRES_USER": "reader",
        "POSTGRES_PASSWORD": "password", "POSTGRES_DB": "knowledge_system",
        "POSTGRES_WORKSPACE": "test-workspace", "NEO4J_URI": "bolt://neo.invalid",
        "NEO4J_USERNAME": "reader", "NEO4J_PASSWORD": "password",
    }.items():
        monkeypatch.setenv(name, value)
    state = SimpleNamespace(rows={}, edges=[], encoded=[], sql=[], pg_options=None,
                            neo_options=None, session_options=None, graph_query=None,
                            graph_params=None, pg_closed=False, neo_closed=False,
                            session_closed=False, fail=None, database_rows={},
                            missing_databases=set(), session_databases=[])

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            state.pg_closed = True

        def execute(self, sql, params=None):
            state.sql.append((sql, params))
            if state.fail == "pg":
                raise RuntimeError("PG unavailable")
            if sql.startswith("SET LOCAL"):
                self.rows = []
            else:
                assert sql.lstrip().startswith("SELECT")
                table = next(name for name in ("chunks", "entity", "relation")
                             if f"lightrag_vdb_{name}" in sql)
                workspace, vector, limit = params
                assert workspace == "test-workspace"
                assert 0 < limit <= 120
                index = int(json.loads(vector)[0])
                self.rows = state.rows.get((table, index), [])
            return self

        def fetchall(self):
            return self.rows

    def connect(**kwargs):
        state.pg_options = kwargs
        if state.fail == "connect":
            raise RuntimeError("connection unavailable")
        return Connection()

    class Session:
        def __init__(self, database):
            self.database = database

        def __enter__(self):
            return self

        def __exit__(self, *_):
            state.session_closed = True

        def run(self, query, **params):
            state.graph_query, state.graph_params = query, params
            if self.database in state.missing_databases:
                import neo4j
                raise neo4j.exceptions.Neo4jError._hydrate_neo4j(
                    code="Neo.ClientError.Database.DatabaseNotFound", message="Missing test database")
            if state.fail == "neo":
                raise RuntimeError("Neo4j unavailable")
            return state.database_rows.get(self.database, state.edges)

    class Driver:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            state.neo_closed = True

        def session(self, **kwargs):
            state.session_options = kwargs
            state.session_databases.append(kwargs["database"])
            return Session(kwargs["database"])

    def driver(uri, **kwargs):
        state.neo_options = (uri, kwargs)
        return Driver()

    def encode(texts):
        state.encoded.append(texts)
        if state.fail == "embed":
            raise RuntimeError("embedding unavailable")
        return [[float(index), 1.0] for index in range(len(texts))]

    monkeypatch.setitem(sys.modules, "papervault.knowledge.store.lightrag_init",
                        SimpleNamespace(_bge_encode_sync=encode))
    import psycopg
    import neo4j
    monkeypatch.setattr(psycopg, "connect", connect)
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", driver)
    return state


def test_vector_consensus_canonicalizes_paths_and_fences_retained_rows(backends):
    alias = paper("Old", terminal_state=replace(TerminalState(), merge=Merge.ALIAS,
                                               alias_key="Canon"))
    excluded = paper("Excluded", terminal_state=replace(TerminalState(), policy=Policy.EXCLUDED))
    library = Library(paper("Canon"), alias, paper("Chunk"), paper("Relation"), excluded)
    backends.rows = {
        ("chunks", 0): [{"file_path": "paper/Old<SEP>Excluded<SEP>missing"},
                         {"file_path": "Chunk"}],
        ("entity", 0): [{"entity_name": "full", "file_path": "Canon"}],
        ("entity", 1): [{"entity_name": "concept", "file_path": "paper/Canon"}],
        ("relation", 0): [{"file_path": "Relation"}],
    }
    assert search(library, ["canonical concept"]) == ["Canon", "Relation", "Chunk"]
    assert backends.encoded == [["full scientific question", "canonical concept"]]
    selects = [(sql, params) for sql, params in backends.sql if sql.startswith("SELECT")]
    assert len(selects) == 5  # chunks(query), entities(query/concept), relations(query/concept)
    assert sum("lightrag_vdb_chunks" in sql for sql, _ in selects) == 1


def test_concepts_fall_back_to_terms_and_backend_reads_are_scoped_and_bounded(backends, monkeypatch):
    monkeypatch.setenv("POSTGRES_DATABASE", "explicit_vector_db")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    monkeypatch.setenv("POSTGRES_WORKSPACE", "test-workspace")
    backends.rows[("entity", 1)] = [{"entity_name": "E", "file_path": "P"}]
    assert search(Library(paper("P"))) == ["P"]
    assert backends.encoded == [["full scientific question", "fallback phrase"]]
    assert backends.pg_options["dbname"] == "explicit_vector_db"
    assert "default_transaction_read_only=on" in backends.pg_options["options"]
    assert "statement_timeout=" in backends.pg_options["options"]
    assert 0 < backends.pg_options["connect_timeout"] <= 5
    assert any(sql == "SET LOCAL hnsw.ef_search = 120" for sql, _ in backends.sql)
    assert backends.session_options == {"database": "neo4j", "default_access_mode": "READ"}
    query = backends.graph_query
    assert query.timeout <= 5
    assert "`test-workspace`" in query.text and "LIMIT $limit" in query.text
    assert backends.graph_params["all_ents"] == ["E"]
    assert backends.pg_closed and backends.neo_closed and backends.session_closed


def test_topology_scores_cross_facet_connections_and_high_weight_hubs(backends):
    library = Library(*(paper(key) for key in ("Cross", "Hub", "Same", "Vector")))
    backends.rows = {
        ("entity", 1): [{"entity_name": "A", "file_path": "Vector"},
                         {"entity_name": "B", "file_path": ""}],
        ("entity", 2): [{"entity_name": "C", "file_path": ""}],
    }
    backends.edges = [
        {"source": "A", "target": "B", "weight": 0, "file_path": "Same"},
        {"source": "A", "target": "C", "weight": 0, "file_path": "Cross"},
        {"source": "B", "target": "A", "weight": 100, "file_path": "Hub"},
    ]
    result = search(library, ["facet one", "facet two"])
    assert set(result) == {"Vector", "Cross", "Hub"}
    assert result.index("Hub") < result.index("Cross")
    assert backends.graph_params["all_ents"] == ["A", "B", "C"]


@pytest.mark.parametrize("missing_preferred", [False, True])
def test_topology_uses_lightrag_default_database_and_read_only_missing_database_fallback(
        backends, missing_preferred):
    backends.rows[("entity", 1)] = [{"entity_name": "A", "file_path": "Vector"}]
    backends.rows[("entity", 2)] = [{"entity_name": "B", "file_path": "Vector"}]
    preferred_edge = {"source": "A", "target": "B", "weight": 1, "file_path": "Preferred"}
    fallback_edge = dict(preferred_edge, file_path="Fallback")
    backends.database_rows = {"chunk-entity-relation": [preferred_edge], None: [fallback_edge]}
    if missing_preferred:
        backends.missing_databases.add("chunk-entity-relation")
    result = search(Library(*(paper(key) for key in ("Vector", "Preferred", "Fallback"))),
                    ["facet one", "facet two"])
    assert set(result) == {"Vector", "Fallback" if missing_preferred else "Preferred"}
    assert backends.session_databases == (["chunk-entity-relation", None] if missing_preferred
                                          else ["chunk-entity-relation"])
    assert backends.session_options["default_access_mode"] == "READ"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_embedding_shares_s2_admission_and_cancellation_retains_permit_until_encode_finishes(
        backends, monkeypatch, cancelled):
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()  # An S2 forward pass already owns the GPU.
    requested, started, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release_encode = threading.Event()
    encoder_module = sys.modules["papervault.knowledge.store.lightrag_init"]
    original_encode = encoder_module._bge_encode_sync

    def get_semaphore():
        requested.set()
        return semaphore

    def encode(texts):
        loop.call_soon_threadsafe(started.set)
        assert release_encode.wait(timeout=3), "Test did not release the controlled forward pass"
        return original_encode(texts)

    monkeypatch.setattr(encoder_module, "_get_embed_sem", get_semaphore, raising=False)
    monkeypatch.setattr(encoder_module, "_bge_encode_sync", encode)
    backends.rows[("chunks", 0)] = [{"file_path": "P"}]

    def worker():
        try:
            return search(Library(paper("P")), ["concept"], embedding_loop=loop)
        finally:
            loop.call_soon_threadsafe(finished.set)

    task = asyncio.create_task(asyncio.to_thread(worker))
    requested_task = asyncio.create_task(requested.wait())
    try:
        done, _ = await asyncio.wait({task, requested_task}, timeout=1,
                                     return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task  # Surface an invalid helper interface instead of hiding it behind a timeout.
        assert requested_task in done
        assert backends.encoded == []
        semaphore.release()
        await asyncio.wait_for(started.wait(), timeout=1)
        assert semaphore.locked()
        if cancelled:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert semaphore.locked(), "Cancelling the awaiter must not release an active GPU pass"
        release_encode.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        if not cancelled:
            assert await task == ["P"]
        await asyncio.wait_for(semaphore.acquire(), timeout=1)
        semaphore.release()
        assert backends.encoded == [["full scientific question", "concept"]]
    finally:
        release_encode.set()
        requested_task.cancel()
        await asyncio.gather(requested_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_embedding_admission_timeout_does_not_leave_a_waiter_or_leak_a_permit(backends, monkeypatch):
    from papervault.library.services import kg_search

    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    encoder_module = sys.modules["papervault.knowledge.store.lightrag_init"]
    monkeypatch.setattr(encoder_module, "_get_embed_sem", lambda: semaphore, raising=False)
    monkeypatch.setattr(kg_search, "_TIMEOUT", 0.01)
    assert await asyncio.to_thread(search, Library(paper("P")), ["concept"], embedding_loop=loop) == []
    assert backends.encoded == [] and backends.pg_options is None
    semaphore.release()
    await asyncio.wait_for(semaphore.acquire(), timeout=1)
    semaphore.release()


@pytest.mark.asyncio
async def test_database_latency_does_not_hold_shared_gpu_admission(backends, monkeypatch):
    import psycopg

    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(1)
    database_started = asyncio.Event()
    release_database = threading.Event()
    encoder_module = sys.modules["papervault.knowledge.store.lightrag_init"]
    monkeypatch.setattr(encoder_module, "_get_embed_sem", lambda: semaphore, raising=False)
    original_connect = psycopg.connect

    def connect(**kwargs):
        loop.call_soon_threadsafe(database_started.set)
        assert release_database.wait(timeout=3)
        return original_connect(**kwargs)

    monkeypatch.setattr(psycopg, "connect", connect)
    backends.rows[("chunks", 0)] = [{"file_path": "P"}]
    task = asyncio.create_task(asyncio.to_thread(search, Library(paper("P")), ["concept"],
                                                embedding_loop=loop))
    try:
        await asyncio.wait_for(database_started.wait(), timeout=1)
        await asyncio.wait_for(semaphore.acquire(), timeout=1)
        semaphore.release()
    finally:
        release_database.set()
    assert await task == ["P"]


@pytest.mark.asyncio
async def test_failed_encoder_releases_shared_admission_without_querying_stores(backends, monkeypatch):
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(1)
    encoder_module = sys.modules["papervault.knowledge.store.lightrag_init"]
    monkeypatch.setattr(encoder_module, "_get_embed_sem", lambda: semaphore, raising=False)
    backends.fail = "embed"
    assert await asyncio.to_thread(search, Library(paper("P")), ["concept"], embedding_loop=loop) == []
    await asyncio.wait_for(semaphore.acquire(), timeout=1)
    semaphore.release()
    assert backends.pg_options is None


def test_path_duplicates_do_not_create_extra_rrf_votes(backends):
    backends.rows = {
        ("chunks", 0): [{"file_path": "A<SEP>A<SEP>paper/A"}, {"file_path": "A"},
                         {"file_path": "B"}],
        ("relation", 0): [{"file_path": "B"}],
    }
    assert search(Library(paper("A"), paper("B")), ["concept"]) == ["B", "A"]


@pytest.mark.parametrize("failure", ["embed", "connect", "pg", "neo"])
def test_backend_failure_discards_partial_candidates_and_closes_resources(backends, failure):
    backends.fail = failure
    backends.rows[("entity", 1)] = [{"entity_name": "E", "file_path": "P"}]
    assert search(Library(paper("P")), ["concept"]) == []
    if failure in {"pg", "neo"}:
        assert backends.pg_closed
    if failure == "neo":
        assert backends.neo_closed and backends.session_closed


@pytest.mark.parametrize("setting,value", [
    ("PAPERVAULT_SEARCH_KG", "0"), ("PAPERVAULT_STORAGE_BACKEND", "json"),
    ("POSTGRES_WORKSPACE", ""), ("POSTGRES_DB", ""), ("NEO4J_URI", ""),
    ("NEO4J_WORKSPACE", "different-workspace"),
])
def test_disabled_or_unconfigured_search_never_loads_model_or_connects(backends, monkeypatch, setting, value):
    monkeypatch.setenv(setting, value)
    assert search(Library(paper("P")), ["concept"]) == []
    assert backends.encoded == [] and backends.pg_options is None


def test_explicit_opt_in_works_with_json_catalog(backends, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_STORAGE_BACKEND", "json")
    monkeypatch.setenv("PAPERVAULT_SEARCH_KG", "1")
    backends.rows[("chunks", 0)] = [{"file_path": "P"}]
    assert search(Library(paper("P")), ["concept"]) == ["P"]


def test_missing_optional_dependency_fails_open_without_model_loading(backends, monkeypatch):
    monkeypatch.setitem(sys.modules, "neo4j", None)
    assert search(Library(paper("P")), ["concept"]) == []
    assert backends.encoded == []


def test_dangling_and_cyclic_aliases_cannot_be_returned(backends):
    def alias(key, target):
        return paper(key, terminal_state=replace(TerminalState(), merge=Merge.ALIAS,
                                                alias_key=target))
    backends.rows[("chunks", 0)] = [{"file_path": "Dangling<SEP>A<SEP>B<SEP>P"}]
    assert search(Library(alias("Dangling", "Missing"), alias("A", "B"),
                          alias("B", "A"), paper("P")), ["concept"]) == ["P"]
