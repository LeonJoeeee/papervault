"""S4 auto-ingest scheduler lifecycle (bug fixed 2026-06-19).

The scheduler must be a SERVER-LIFETIME singleton started ONCE at server boot — NOT per MCP session.
Under streamable-http the FastMCP `_lifespan` runs once PER CLIENT SESSION, so the old code (scheduler
in `_lifespan`, cancelled in its finally) never started at boot, duplicated per client, and died on
disconnect. These tests pin the fixed contract: start_background is idempotent + default-OFF, the
per-session `_lifespan` only idempotently ensures it (never tears down), stop_background cleans up once.
"""
import asyncio
import logging

import pytest

import papervault.knowledge.mcp.server as srv


@pytest.fixture(autouse=True)
def _reset_bg(monkeypatch):
    # These are scheduler-lifecycle tests, not embedding tests. start_background now runs a
    # boot-time embedding self-check (#84) before wiring the scheduler; with no GPU/model on CI
    # that check fails and start_background would never create the task under test. Bypass it so
    # these tests exercise the scheduler-wiring contract they're actually about.
    monkeypatch.setenv("KS_SKIP_EMBED_SELFCHECK", "1")
    monkeypatch.setenv("KS_STARTUP_AWAIT_SEC", "0.02")
    srv._bg_task = None
    srv._startup_task = None
    yield
    # cleanup any task a test created so it doesn't leak
    t = srv._bg_task
    if t is not None and not t.done():
        t.cancel()
    srv._bg_task = None
    startup = srv._startup_task
    if startup is not None and not startup.done():
        startup.cancel()
    srv._startup_task = None


def _patch_enabled(monkeypatch, on: bool):
    monkeypatch.setenv("KS_AUTO_INGEST_ENABLED", "true" if on else "false")


def _stub_graph_and_loop(monkeypatch, *, graph_ready=None):
    """Make start_background's get_graph()/main_loop cheap + non-blocking-forever so a task is created."""
    import papervault.knowledge.scheduler.round as rnd
    import papervault.knowledge.store.graph as g
    import papervault.knowledge.ledger.store as ledger

    calls = {"get_graph": 0, "scheduler": [], "loop": None}
    graph_entered = asyncio.Event()
    scheduler_started = asyncio.Event()
    rag = object()

    async def _fake_get_graph():
        calls["get_graph"] += 1
        graph_entered.set()
        if graph_ready is not None:
            await graph_ready.wait()
        return rag

    async def _fake_main_loop(rag, *, interval):  # noqa: ARG001 — runs as the bg task; block until cancelled
        calls["scheduler"].append(rag)
        calls["loop"] = asyncio.get_running_loop()
        scheduler_started.set()
        await asyncio.Event().wait()

    async def _noop():
        pass

    monkeypatch.setattr(g, "get_graph", _fake_get_graph)
    monkeypatch.setattr(rnd, "main_loop", _fake_main_loop)
    monkeypatch.setattr(g, "close_graph", _noop)
    monkeypatch.setattr(ledger, "close_pool", _noop)
    return calls, graph_entered, scheduler_started, rag


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


@pytest.mark.parametrize("blocked_stage", ["graph", "embedding"])
async def test_lifespan_ready_before_slow_startup_then_scheduler_recovers(
    monkeypatch, caplog, blocked_stage,
):
    """A slow initializer must neither hold the lifespan nor be cancelled/duplicated."""
    _patch_enabled(monkeypatch, True)
    release = asyncio.Event()
    calls, graph_entered, scheduler_started, rag = _stub_graph_and_loop(
        monkeypatch, graph_ready=release if blocked_stage == "graph" else None,
    )
    if blocked_stage == "embedding":
        async def slow_selfcheck():
            await release.wait()
            return True
        monkeypatch.setattr(srv, "_embedding_stack_ok", slow_selfcheck)

    async def enter_lifespan():
        async with srv._lifespan(None):
            return "ready"

    try:
        with caplog.at_level(logging.WARNING, logger="ks.mcp.server"):
            assert await asyncio.wait_for(enter_lifespan(), timeout=0.5) == "ready"
        assert srv._bg_task is None
        startup = srv._startup_task
        assert startup is not None and not startup.done()
        if blocked_stage == "graph":
            assert graph_entered.is_set()
        assert "KS_STARTUP_AWAIT_SEC" in caplog.text
        assert "background" in caplog.text

        # A client session must return promptly and retain the same pending initializer.
        assert await asyncio.wait_for(enter_lifespan(), timeout=0.5) == "ready"
        assert srv._startup_task is startup
        release.set()
        await asyncio.wait_for(scheduler_started.wait(), timeout=0.5)
        assert calls["get_graph"] == 1
        assert calls["scheduler"] == [rag]
        assert calls["loop"] is asyncio.get_running_loop()
        assert srv._startup_task is None
    finally:
        await srv.stop_background()


async def test_graph_startup_exceptions_retry_until_success(monkeypatch, caplog):
    """Both the first and later graph failures must recover without a client trigger."""
    import papervault.knowledge.store.graph as graph

    _patch_enabled(monkeypatch, True)
    monkeypatch.setenv("KS_EMBED_SELFCHECK_RETRY_SEC", "0")
    calls, _, scheduler_started, rag = _stub_graph_and_loop(monkeypatch)
    successful_get_graph = graph.get_graph
    release_retry = asyncio.Event()
    attempts = 0
    delays = []

    async def flaky_get_graph():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("graph temporarily unavailable")
        return await successful_get_graph()

    async def controlled_retry(delay):
        delays.append(delay)
        await release_retry.wait()

    monkeypatch.setattr(graph, "get_graph", flaky_get_graph)
    monkeypatch.setattr(srv, "_wait_selfcheck_retry", controlled_retry)
    try:
        with caplog.at_level(logging.WARNING, logger="ks.mcp.server"):
            await asyncio.wait_for(srv.start_background(), timeout=0.5)
        assert attempts == 1 and srv._bg_task is None
        startup = srv._startup_task
        assert startup is not None and not startup.done()
        assert "graph temporarily unavailable" in caplog.text
        assert "retry" in caplog.text.lower()
        await srv.start_background()
        assert attempts == 1 and srv._startup_task is startup
        release_retry.set()
        await asyncio.wait_for(scheduler_started.wait(), timeout=0.5)
        assert attempts == 3 and delays == [1.0, 1.0]
        assert calls["scheduler"] == [rag]
    finally:
        await srv.stop_background()


async def test_shutdown_cancels_pending_startup_before_scheduler_can_start(monkeypatch):
    _patch_enabled(monkeypatch, True)
    release = asyncio.Event()
    calls, _, _, _ = _stub_graph_and_loop(monkeypatch, graph_ready=release)
    try:
        await asyncio.wait_for(srv.start_background(), timeout=0.5)
        startup = srv._startup_task
        assert startup is not None and not startup.done()
        await srv.stop_background()
        assert startup.cancelled() and srv._startup_task is None
        release.set()
        await asyncio.sleep(0)
        assert srv._bg_task is None and calls["scheduler"] == []
    finally:
        await srv.stop_background()


async def test_cancelled_startup_wait_preserves_background_initialization(monkeypatch):
    _patch_enabled(monkeypatch, True)
    monkeypatch.setenv("KS_STARTUP_AWAIT_SEC", "30")
    release = asyncio.Event()
    calls, graph_entered, scheduler_started, rag = _stub_graph_and_loop(
        monkeypatch, graph_ready=release,
    )
    waiter = asyncio.create_task(srv.start_background())
    try:
        await asyncio.wait_for(graph_entered.wait(), timeout=0.5)
        startup = srv._startup_task
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert srv._startup_task is startup and not startup.done()
        release.set()
        await asyncio.wait_for(scheduler_started.wait(), timeout=0.5)
        assert calls["get_graph"] == 1 and calls["scheduler"] == [rag]
    finally:
        waiter.cancel()
        await srv.stop_background()


async def test_zero_startup_wait_serves_immediately_and_keeps_initializing(monkeypatch):
    _patch_enabled(monkeypatch, True)
    monkeypatch.setenv("KS_STARTUP_AWAIT_SEC", "0")
    release = asyncio.Event()
    calls, _, scheduler_started, rag = _stub_graph_and_loop(monkeypatch, graph_ready=release)
    try:
        await asyncio.wait_for(srv.start_background(), timeout=0.5)
        assert srv._startup_task is not None and not srv._startup_task.done()
        assert srv._bg_task is None
        release.set()
        await asyncio.wait_for(scheduler_started.wait(), timeout=0.5)
        assert calls["get_graph"] == 1 and calls["scheduler"] == [rag]
    finally:
        await srv.stop_background()


@pytest.mark.parametrize("value", ["invalid", "-1", "nan", "inf"])
async def test_invalid_startup_wait_logs_default_and_still_starts(monkeypatch, caplog, value):
    _patch_enabled(monkeypatch, True)
    monkeypatch.setenv("KS_STARTUP_AWAIT_SEC", value)
    calls, _, scheduler_started, rag = _stub_graph_and_loop(monkeypatch)
    try:
        with caplog.at_level(logging.WARNING, logger="ks.mcp.server"):
            await asyncio.wait_for(srv.start_background(), timeout=0.5)
        await asyncio.wait_for(scheduler_started.wait(), timeout=0.5)
        assert calls["scheduler"] == [rag]
        assert "KS_STARTUP_AWAIT_SEC" in caplog.text and "3" in caplog.text
    finally:
        await srv.stop_background()
