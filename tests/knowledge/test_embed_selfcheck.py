"""Boot-time embedding self-check (issue #84) — FAIL SAFE. Pure offline: no GPU, no model.

On 2026-07-23 a torch/torchvision ABI drift crashed BGE-M3 embedding on EVERY call, but the
KS service booted fine and the scheduler CHURNED for an hour — re-distilling then failing every
document — before anyone noticed. `start_background` now embeds ONE short string through the real
embedder path (`_bge_embed`) BEFORE spawning the scheduler `main_loop`; if it raises (or hangs),
the scheduler is NOT started (auto-ingest stays off) and a LOUD, actionable error is logged.

These tests patch the embedder (so no torch/GPU is touched) and the scheduler collaborators
(get_graph/main_loop, same shape as test_server_lifespan) to pin the contract:
  - embed RAISES  → main_loop NOT spawned (`_bg_task is None`) + loud ERROR fired;
  - embed SUCCEEDS → scheduler starts as normal;
  - embed returns EMPTY → treated as failure (fail safe);
  - KS_SKIP_EMBED_SELFCHECK=1 → check skipped, embedder never called, scheduler starts;
  - a HANG past KS_EMBED_SELFCHECK_TIMEOUT_SEC → fails safe (scheduler not started);
  - self-check does not even run when auto-ingest is disabled (default OFF).
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from papervault.knowledge.mcp import server


@pytest.fixture(autouse=True)
def _clean_env_and_task(monkeypatch):
    """Isolate each test: clear the relevant env knobs + reset the server-lifetime task global.

    `start_background` early-returns if `_bg_task is not None`, so a task leaked from a prior
    test would mask the behaviour under test — reset it before, and cancel/clear after.
    """
    for var in (
        "KS_AUTO_INGEST_ENABLED",
        "KS_AUTO_INGEST_INTERVAL_SEC",
        "KS_SKIP_EMBED_SELFCHECK",
        "KS_EMBED_SELFCHECK_TIMEOUT_SEC",
    ):
        monkeypatch.delenv(var, raising=False)
    server._bg_task = None
    yield
    task = server._bg_task
    server._bg_task = None
    if task is not None:
        task.cancel()


def _patch_scheduler(monkeypatch):
    """Stub get_graph/main_loop so the scheduler path never touches IO (mirrors test_server_lifespan)."""
    import papervault.knowledge.scheduler.round as rnd
    import papervault.knowledge.store.graph as graph

    calls: dict = {"get_graph": 0, "main_loop": []}
    sentinel_rag = object()

    async def fake_get_graph():
        calls["get_graph"] += 1
        return sentinel_rag

    async def fake_main_loop(rag, *, interval=60.0, **kw):
        calls["main_loop"].append((rag, interval))
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(graph, "get_graph", fake_get_graph)
    monkeypatch.setattr(rnd, "main_loop", fake_main_loop)
    return calls, sentinel_rag


def _patch_embed(monkeypatch, impl):
    """Patch the real embedder path `_bge_embed`; return a counter of how many times it ran."""
    import papervault.knowledge.store.lightrag_init as li

    calls = {"n": 0}

    async def fake_embed(texts):
        calls["n"] += 1
        return await impl(texts)

    monkeypatch.setattr(li, "_bge_embed", fake_embed)
    return calls


async def test_selfcheck_failure_does_not_start_scheduler(monkeypatch, caplog):
    monkeypatch.setenv("KS_AUTO_INGEST_ENABLED", "true")
    sched, _ = _patch_scheduler(monkeypatch)

    async def _boom(_texts):
        # The real incident shape: BGE-M3 import chain crashes on every call.
        raise RuntimeError("operator torchvision::nms does not exist")

    embed = _patch_embed(monkeypatch, _boom)

    with caplog.at_level(logging.ERROR, logger="ks.mcp.server"):
        await server.start_background()

    assert embed["n"] == 1                      # the smoke-test actually exercised the embedder
    assert server._bg_task is None              # scheduler NOT started — the whole point
    assert sched["main_loop"] == []             # main_loop never spawned
    assert sched["get_graph"] == 0              # failed fast, before opening the graph
    # LOUD + actionable log fired.
    msgs = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "embedding self-check FAILED" in msgs
    assert "scheduler NOT started" in msgs
    assert "torchvision::nms" in msgs           # the underlying error is surfaced, not swallowed


async def test_selfcheck_success_starts_scheduler(monkeypatch):
    monkeypatch.setenv("KS_AUTO_INGEST_ENABLED", "true")
    monkeypatch.setenv("KS_AUTO_INGEST_INTERVAL_SEC", "120")
    sched, sentinel_rag = _patch_scheduler(monkeypatch)

    async def _ok(texts):
        return np.zeros((len(texts), 1024), dtype=np.float32)

    embed = _patch_embed(monkeypatch, _ok)

    await server.start_background()
    await asyncio.sleep(0)                       # let the scheduler task tick once

    assert embed["n"] == 1
    assert server._bg_task is not None           # scheduler started as normal
    assert sched["get_graph"] == 1               # graph opened once (after the check passed)
    assert sched["main_loop"] == [(sentinel_rag, 120.0)]  # shared rag + interval wired through


async def test_selfcheck_empty_result_is_failure(monkeypatch, caplog):
    monkeypatch.setenv("KS_AUTO_INGEST_ENABLED", "true")
    sched, _ = _patch_scheduler(monkeypatch)

    async def _empty(_texts):
        return np.zeros((0, 1024), dtype=np.float32)  # silent empty → also a fault

    _patch_embed(monkeypatch, _empty)

    with caplog.at_level(logging.ERROR, logger="ks.mcp.server"):
        await server.start_background()

    assert server._bg_task is None
    assert sched["main_loop"] == []
    msgs = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "EMPTY result" in msgs


async def test_selfcheck_skipped_via_env(monkeypatch, caplog):
    monkeypatch.setenv("KS_AUTO_INGEST_ENABLED", "true")
    monkeypatch.setenv("KS_SKIP_EMBED_SELFCHECK", "1")
    sched, sentinel_rag = _patch_scheduler(monkeypatch)

    async def _boom(_texts):
        raise RuntimeError("must NOT be called when the self-check is skipped")

    embed = _patch_embed(monkeypatch, _boom)

    with caplog.at_level(logging.INFO, logger="ks.mcp.server"):
        await server.start_background()
    await asyncio.sleep(0)                        # let the scheduler task tick once

    assert embed["n"] == 0                       # embedder never touched under the escape hatch
    assert server._bg_task is not None           # scheduler still starts
    assert sched["main_loop"] == [(sentinel_rag, 60.0)]
    assert any("self-check SKIPPED" in r.getMessage() for r in caplog.records)


async def test_selfcheck_hang_fails_safe(monkeypatch, caplog):
    monkeypatch.setenv("KS_AUTO_INGEST_ENABLED", "true")
    monkeypatch.setenv("KS_EMBED_SELFCHECK_TIMEOUT_SEC", "0.05")
    # Re-read the module-level default so the tiny timeout takes effect.
    monkeypatch.setattr(server, "_EMBED_SELFCHECK_TIMEOUT_SEC", 0.05)
    sched, _ = _patch_scheduler(monkeypatch)

    async def _hang(_texts):
        await asyncio.sleep(10)                   # simulate a wedged embed stack

    _patch_embed(monkeypatch, _hang)

    with caplog.at_level(logging.ERROR, logger="ks.mcp.server"):
        await server.start_background()

    assert server._bg_task is None               # a hang must never block boot into churning
    assert sched["main_loop"] == []
    msgs = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "embedding self-check FAILED" in msgs


async def test_selfcheck_not_run_when_auto_ingest_disabled(monkeypatch):
    # Default OFF → we return before the self-check; the embedder must never be called.
    sched, _ = _patch_scheduler(monkeypatch)

    async def _boom(_texts):
        raise RuntimeError("self-check must not run when auto-ingest is disabled")

    embed = _patch_embed(monkeypatch, _boom)

    await server.start_background()

    assert embed["n"] == 0
    assert server._bg_task is None
    assert sched["main_loop"] == []
