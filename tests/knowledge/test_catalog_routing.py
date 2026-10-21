"""Catalog routing must precede every legacy schema, pipeline and singleton mutation."""
import asyncio
import sys
import types

import pytest
from click.testing import CliRunner

from papervault.store import knowledge_runtime as runtime


@pytest.fixture(autouse=True)
async def authority_cache():
    from papervault.knowledge.mcp import server
    server._authority_task = None
    yield
    if server._authority_task is not None:
        await asyncio.gather(server._authority_task, return_exceptions=True)
    server._authority_task = None


def forbidden(*args, **kwargs):
    raise AssertionError("legacy I/O reached")


@pytest.fixture
def catalog_service(monkeypatch):
    module = types.ModuleType("papervault.knowledge.scheduler.catalog_service")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "postgres")
    monkeypatch.setattr(runtime, "assert_knowledge_authority", lambda expected: {"authority": expected, "authority_epoch": 2})
    return module


async def test_postgres_round_routes_before_legacy_index_or_reconcile(monkeypatch, catalog_service):
    from papervault.knowledge.scheduler import round as rounds

    async def run_catalog_round(only_keys=None):
        return {"catalog": True, "keys": sorted(only_keys or [])}
    catalog_service.run_catalog_round = run_catalog_round
    monkeypatch.setattr(rounds, "load_clean_index", forbidden)
    assert await rounds.run_round(object(), only_keys=["B", "A"]) == {"catalog": True, "keys": ["A", "B"]}


async def test_postgres_loop_routes_before_legacy_schema(monkeypatch, catalog_service):
    from papervault.knowledge.scheduler import round as rounds

    class LoopReached(Exception):
        pass
    async def run_catalog_loop(interval=60):
        raise LoopReached(str(interval))
    catalog_service.run_catalog_loop = run_catalog_loop
    monkeypatch.setattr(rounds.ledger, "ensure_schema", forbidden)
    with pytest.raises(LoopReached, match="120"):
        await rounds.main_loop(object(), interval=120)


async def test_postgres_server_start_keeps_selfcheck_without_opening_legacy_graph(monkeypatch, catalog_service):
    from papervault.knowledge.mcp import server
    from papervault.knowledge.store import graph

    async def selfcheck():
        return True
    async def loop(interval=60):
        return interval
    catalog_service.run_catalog_loop = loop
    monkeypatch.setattr(server, "_embedding_stack_ok", selfcheck)
    monkeypatch.setattr(graph, "get_graph", forbidden)
    monkeypatch.setenv("KS_AUTO_INGEST_INTERVAL_SEC", "123")
    try:
        assert await server._start_once() is True
        assert await server._bg_task == 123
    finally:
        server._bg_task = None


async def test_background_start_rejects_committed_catalog_even_when_legacy_autoingest_disabled(monkeypatch):
    from papervault.knowledge.mcp import server
    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "legacy")
    monkeypatch.delenv("KS_AUTO_INGEST_ENABLED", raising=False)
    def refuse(expected):
        raise runtime.KnowledgeAuthorityError("committed catalog")
    monkeypatch.setattr(runtime, "assert_knowledge_authority", refuse)
    await server.start_background()
    with pytest.raises(runtime.KnowledgeAuthorityError, match="committed catalog"):
        await server.query("question")


@pytest.mark.parametrize("operation", ["schema", "upsert", "delete", "distill", "remove", "rollback",
                                      "markers", "enqueue", "purge", "reconcile_sections", "terminal", "healed"])
async def test_legacy_mutators_refuse_before_any_backend_or_ledger_access(monkeypatch, operation):
    from papervault.knowledge.ingest import distill, operator_docs
    from papervault.knowledge.ledger import store
    from papervault.knowledge.scheduler import round as rounds

    def refuse(expected):
        raise runtime.KnowledgeAuthorityError("legacy fenced")
    monkeypatch.setattr(runtime, "assert_knowledge_authority", refuse)
    monkeypatch.setattr(store, "_workspace", forbidden)
    monkeypatch.setattr(store, "load", forbidden)
    monkeypatch.setattr(store, "load_by_status", forbidden)
    rag = object()
    calls = {"schema": lambda: store.ensure_schema(),
        "upsert": lambda: store.upsert("paper", "A", doc_id="A", status="done"),
        "delete": lambda: store.delete("paper", "A"), "distill": lambda: distill.distill_batch(rag, []),
        "remove": lambda: distill.remove_one(rag, "A", delete_ledger=True),
        "rollback": lambda: distill.rollback_abstract_docs(rag, apply=True),
        "markers": lambda: distill._clear_insert_path(rag, "A"),
        "enqueue": lambda: operator_docs.enqueue_sections(rag, []),
        "purge": lambda: operator_docs._purge_sections(rag, "textbook", "A", "textbook:A"),
        "reconcile_sections": lambda: operator_docs._reconcile_sections(rag, "textbook", []),
        "terminal": lambda: rounds.reconcile_terminal(rag), "healed": lambda: rounds.reconcile_healed(rag)}
    with pytest.raises(runtime.KnowledgeAuthorityError, match="legacy fenced"):
        await calls[operation]()


async def test_operator_postgres_capture_routes_before_force_purge_or_legacy_ledger(monkeypatch, catalog_service):
    from papervault.knowledge.ingest import operator_docs
    async def capture_operator_document(kind, key, path, **kwargs):
        return {"catalog_queued": True, "queued": 2, "sections": 2,
                "done": 0, "error": 0, "generation_id": "g2"}
    catalog_service.capture_operator_document = capture_operator_document
    monkeypatch.setattr(operator_docs, "_purge_sections", forbidden)
    monkeypatch.setattr(operator_docs.ledger, "load", forbidden)
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    result = await operator_docs.ingest_document(object(), "textbook", "textbook:Book2020", "unused.md", force=True)
    assert result["catalog_queued"] is True and result["queued"] == 2
    assert result["done"] == result["error"] == 0


async def test_pickup_catalog_queued_source_moves_processed_without_build_success_or_failure(monkeypatch, tmp_path, catalog_service):
    from papervault.knowledge.scheduler import opdoc_pickup
    pending = tmp_path / "pending"
    pending.mkdir()
    path = pending / "textbook-Book2020.md"
    path.write_text("# Chapter\nbody")
    monkeypatch.setenv("KS_OPDOC_PENDING_DIR", str(pending))
    async def ingest(*args, **kwargs):
        return {"catalog_queued": True, "queued": 2, "sections": 2, "done": 0, "error": 0}
    monkeypatch.setattr(opdoc_pickup, "ingest_document", ingest)
    result = await opdoc_pickup.drain_pending(None)
    assert result["queued"] == 1
    assert result["ingested"] == result["failed"] == result["partial"] == 0
    assert (pending / "processed" / path.name).exists()
    assert not (pending / "failed").exists()


def test_postgres_cli_queues_without_legacy_singleton_or_cotenancy_gate(monkeypatch, tmp_path, catalog_service):
    import papervault.cli as cli
    from papervault.knowledge.store import graph
    from papervault import ops_guards
    path = tmp_path / "book.md"
    path.write_text("# Chapter\nbody")
    monkeypatch.setattr(graph, "get_graph", forbidden)
    monkeypatch.setattr(ops_guards, "require_services_stopped", forbidden)
    async def capture_operator_document(kind, key, path, **kwargs):
        return {"catalog_queued": True, "queued": 2, "sections": 2,
                "done": 0, "error": 0, "generation_id": "g2"}
    catalog_service.capture_operator_document = capture_operator_document
    result = CliRunner().invoke(cli.main, ["ingest-doc", str(path), "--kind", "textbook", "--key", "textbook:Book2020"],
                                env={"PAPERVAULT_OPERATOR_SOURCES": "1"})
    assert result.exit_code == 0, result.output
    assert "queued" in result.output and "g2" in result.output
    assert "ingested" not in result.output
