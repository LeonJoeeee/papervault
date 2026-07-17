"""S4 auto-ingest scheduler lifecycle (bug fixed 2026-06-19).

The scheduler must be a SERVER-LIFETIME singleton started ONCE at server boot — NOT per MCP session.
Under streamable-http the FastMCP `_lifespan` runs once PER CLIENT SESSION, so the old code (scheduler
in `_lifespan`, cancelled in its finally) never started at boot, duplicated per client, and died on
disconnect. These tests pin the fixed contract: start_background is idempotent + default-OFF, the
per-session `_lifespan` only idempotently ensures it (never tears down), stop_background cleans up once.
"""
import asyncio

import pytest

import papervault.knowledge.mcp.server as srv


@pytest.fixture(autouse=True)
def _reset_bg():
    srv._bg_task = None
    yield
    # cleanup any task a test created so it doesn't leak
    t = srv._bg_task
    if t is not None and not t.done():
        t.cancel()
    srv._bg_task = None


def _patch_enabled(monkeypatch, on: bool):
    monkeypatch.setenv("KS_AUTO_INGEST_ENABLED", "true" if on else "false")


def _stub_graph_and_loop(monkeypatch):
    """Make start_background's get_graph()/main_loop cheap + non-blocking-forever so a task is created."""
    import papervault.knowledge.scheduler.round as rnd
    import papervault.knowledge.store.graph as g

    async def _fake_get_graph():
        return object()

    async def _fake_main_loop(rag, *, interval):  # noqa: ARG001 — runs as the bg task; block until cancelled
        await asyncio.Event().wait()

    monkeypatch.setattr(g, "get_graph", _fake_get_graph)
    monkeypatch.setattr(rnd, "main_loop", _fake_main_loop)


def _run(coro):
    return asyncio.run(coro)


def test_disabled_starts_nothing(monkeypatch):
    _patch_enabled(monkeypatch, False)
    _run(srv.start_background())
    assert srv._bg_task is None


def test_enabled_starts_one_task_and_is_idempotent(monkeypatch):
    _patch_enabled(monkeypatch, True)
    _stub_graph_and_loop(monkeypatch)

    async def go():
        await srv.start_background()
        first = srv._bg_task
        assert first is not None and not first.done()
        await srv.start_background()          # second call = no-op (idempotent)
        assert srv._bg_task is first          # same task, NOT a duplicate
        await srv.stop_background()           # graceful cleanup
        assert srv._bg_task is None
        assert first.cancelled() or first.done()

    _run(go())


def test_per_session_lifespan_does_not_tear_down(monkeypatch):
    """Entering+exiting the per-session lifespan must NOT cancel the server-lifetime scheduler."""
    _patch_enabled(monkeypatch, True)
    _stub_graph_and_loop(monkeypatch)

    async def go():
        async with srv._lifespan(None):       # one "session"
            t = srv._bg_task
            assert t is not None and not t.done()
        # session exited — scheduler must SURVIVE (old bug: it got cancelled here)
        assert srv._bg_task is t and not t.done()
        async with srv._lifespan(None):       # a second "session" → still the same task
            assert srv._bg_task is t
        assert srv._bg_task is t and not t.done()
        await srv.stop_background()

    _run(go())
