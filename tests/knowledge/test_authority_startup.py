"""Slow authority reads cannot block unified serving or reconnect per MCP session."""
import asyncio
import threading

import pytest

from papervault.knowledge.mcp import server
from papervault.store import knowledge_runtime


@pytest.fixture(autouse=True)
async def clean_startup(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "legacy")
    monkeypatch.delenv("KS_AUTO_INGEST_ENABLED", raising=False)
    monkeypatch.setenv("KS_STARTUP_AWAIT_SEC", "0.01")
    server._bg_task = server._startup_task = None
    server._authority_task = None
    yield
    if server._startup_task is not None:
        server._startup_task.cancel()
        try:
            await server._startup_task
        except (asyncio.CancelledError, Exception):
            pass
    server._startup_task = server._bg_task = server._authority_task = None


async def test_slow_authority_read_is_off_loop_bounded_and_shared_by_repeated_sessions(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    owner_thread = threading.get_ident()
    calls = []
    def slow_authority(expected):
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(0.5), "authority read blocked startup instead of waiting off-loop"
        return {"authority": expected}
    monkeypatch.setattr(knowledge_runtime, "assert_knowledge_authority", slow_authority)
    try:
        await server.start_background()
        assert await asyncio.to_thread(entered.wait, 0.2)
        assert calls == [calls[0]] and calls[0] != owner_thread
        assert not server._authority_task.done()
        await server.start_background()
        assert len(calls) == 1
        release.set()
        await asyncio.shield(server._authority_task)
        await server.start_background()
        assert len(calls) == 1
        assert server._bg_task is None
    finally:
        release.set()
        if getattr(server, "_authority_task", None) is not None:
            await asyncio.gather(server._authority_task, return_exceptions=True)


async def test_incompatible_authority_is_cached_and_only_knowledge_queries_are_refused(monkeypatch):
    calls = []
    def incompatible(expected):
        calls.append(expected)
        raise knowledge_runtime.KnowledgeAuthorityError("committed catalog authority")
    monkeypatch.setattr(knowledge_runtime, "assert_knowledge_authority", incompatible)
    async def forbidden_pipeline(**kwargs):
        raise AssertionError("incompatible knowledge query entered pipeline")
    monkeypatch.setattr("papervault.knowledge.query.aquery.query", forbidden_pipeline)
    await server.start_background()  # Unified library startup remains available.
    for _ in range(2):
        await server.start_background()
        with pytest.raises(knowledge_runtime.KnowledgeAuthorityError, match="committed catalog"):
            await server.query("question")
    assert calls == ["legacy"]
    assert server._bg_task is None


async def test_query_cancellation_does_not_cancel_or_duplicate_pending_authority_read(monkeypatch):
    release, entered = threading.Event(), threading.Event()
    calls = []
    def slow_authority(expected):
        calls.append(expected)
        entered.set()
        assert release.wait(0.5)
        return {"authority": expected}
    monkeypatch.setattr(knowledge_runtime, "assert_knowledge_authority", slow_authority)
    async def query_pipeline(**kwargs):
        return {"answer": "authority checked"}
    monkeypatch.setattr("papervault.knowledge.query.aquery.query", query_pipeline)
    try:
        await server.start_background()
        assert await asyncio.to_thread(entered.wait, 0.2)
        query = asyncio.create_task(server.query("first"))
        await asyncio.sleep(0)
        query.cancel()
        with pytest.raises(asyncio.CancelledError):
            await query
        assert not server._authority_task.cancelled()
        assert calls == ["legacy"]
        release.set()
        assert await server.query("second") == {"answer": "authority checked"}
    finally:
        release.set()
        if getattr(server, "_authority_task", None) is not None:
            await asyncio.gather(server._authority_task, return_exceptions=True)
