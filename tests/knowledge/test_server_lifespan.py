"""S4 runtime-wiring tests for the FastMCP `lifespan` (SDD §6.6 "S4 运行时接线").

Pure offline: get_graph / main_loop are monkeypatched, so no real LightRAG, no DB,
no network. Verifies the single-loop wiring contract:

- DEFAULT-OFF: KS_AUTO_INGEST_ENABLED unset → lifespan does NOT call get_graph and
  starts NO scheduler task. The default-OFF reason is now the §6.5 copy-throughput
  go/no-go + explicit human start — NOT pollution: ks_ledger is workspace-isolated
  (SDD §4.1), so run_round can no longer touch prod 'l0' ledger rows.
- OPT-IN: KS_AUTO_INGEST_ENABLED=true → lifespan awaits get_graph() (workspace-gated),
  create_task(main_loop(rag)) in the SAME loop, then cancels it on exit (graceful stop).
- The scheduler task runs in the SAME event loop as the lifespan body (the whole point
  of the single-loop constraint vs a separate thread + asyncio.run).
"""
from __future__ import annotations

import asyncio

import pytest

from papervault.knowledge.mcp import server


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("KS_AUTO_INGEST_ENABLED", raising=False)
    monkeypatch.delenv("KS_AUTO_INGEST_INTERVAL_SEC", raising=False)
    # Bypass the boot-time embedding self-check (issue #84): these tests pin the scheduler
    # WIRING (get_graph/main_loop are stubbed), not the GPU embed stack, so the self-check
    # (which would load a real BGE-M3 model) is orthogonal here. Its own coverage lives in
    # tests/knowledge/test_embed_selfcheck.py.
    monkeypatch.setenv("KS_SKIP_EMBED_SELFCHECK", "1")
    yield


def _patch_no_io(monkeypatch):
    """Stub out get_graph/main_loop and the close_* teardown so nothing touches IO."""
    import papervault.knowledge.store.graph as graph
    import papervault.knowledge.ledger.store as ledger_store
    import papervault.knowledge.scheduler.round as rnd

    calls: dict = {"get_graph": 0, "main_loop": [], "loop": None, "cancelled": False}
    sentinel_rag = object()

    async def fake_get_graph():
        calls["get_graph"] += 1
        return sentinel_rag

    async def fake_main_loop(rag, *, interval=60.0, **kw):
        calls["main_loop"].append((rag, interval))
        calls["loop"] = asyncio.get_running_loop()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            calls["cancelled"] = True
            raise

    async def noop():
        return None

    monkeypatch.setattr(graph, "get_graph", fake_get_graph)
    monkeypatch.setattr(graph, "close_graph", noop)
    monkeypatch.setattr(ledger_store, "close_pool", noop)
    monkeypatch.setattr(rnd, "main_loop", fake_main_loop)
    return calls, sentinel_rag


async def test_lifespan_default_off_starts_no_scheduler(monkeypatch):
    calls, _ = _patch_no_io(monkeypatch)
    # KS_AUTO_INGEST_ENABLED unset → default OFF
    async with server._lifespan(server.mcp):
        pass
    assert calls["get_graph"] == 0       # never opened the (gated) graph
    assert calls["main_loop"] == []      # no scheduler task spawned


async def test_lifespan_opt_in_runs_scheduler_in_same_loop_and_cancels(monkeypatch):
    monkeypatch.setenv("KS_AUTO_INGEST_ENABLED", "true")
    monkeypatch.setenv("KS_AUTO_INGEST_INTERVAL_SEC", "120")
    calls, sentinel_rag = _patch_no_io(monkeypatch)

    this_loop = asyncio.get_running_loop()
    async with server._lifespan(server.mcp):
        # give the scheduler task one tick to start
        await asyncio.sleep(0)
        assert calls["get_graph"] == 1               # opened the gated graph once
        assert calls["main_loop"] == [(sentinel_rag, 120.0)]  # got the SHARED rag + interval
        assert calls["loop"] is this_loop            # SAME event loop (single-loop constraint)
    # on exit the task is cancelled (graceful stop)
    assert calls["cancelled"] is True


async def test_lifespan_disabled_values(monkeypatch):
    # only 1/true/yes opt in; anything else stays OFF
    for val in ("false", "0", "no", "off", ""):
        monkeypatch.setenv("KS_AUTO_INGEST_ENABLED", val)
        calls, _ = _patch_no_io(monkeypatch)
        async with server._lifespan(server.mcp):
            await asyncio.sleep(0)
        assert calls["main_loop"] == [], f"value {val!r} should NOT enable auto-ingest"
