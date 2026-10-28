"""Embed GPU-offload contract (issue #31 — MCP event-loop robustness). No GPU, no model.

`_bge_embed` historically ran BGE-M3 `model.encode` (a synchronous CUDA call) directly on the S4
single event loop, so under the reviewer fan-out those blocking bursts starved the MCP handshake
path. The fix offloads the load+encode to a worker thread under a GPU-concurrency semaphore. These
tests pin the observable contract with a fake model (records the thread it runs on):
  - default (KS_EMBED_OFFLOAD unset): encode runs OFF the event-loop thread;
  - KS_EMBED_OFFLOAD=0: legacy path, encode runs ON the calling (loop) thread;
  - the returned array still matches the model's dense_vecs;
  - the GPU-concurrency semaphore exists and is an asyncio.Semaphore.
"""
from __future__ import annotations

import asyncio
import importlib
import threading

import numpy as np
import pytest

from papervault.knowledge.store import lightrag_init


class _FakeBGE:
    """Stand-in for BGEM3FlagModel: records the OS thread encode() ran on."""

    def __init__(self):
        self.encode_thread = None

    def encode(self, texts, batch_size=32, max_length=8192):
        self.encode_thread = threading.get_ident()
        return {"dense_vecs": np.arange(len(texts) * 2, dtype=np.float32).reshape(len(texts), 2)}


@pytest.fixture
def _stub_model(monkeypatch):
    fake = _FakeBGE()
    monkeypatch.setattr(lightrag_init, "_get_bge_model", lambda: fake)
    return fake


def test_embed_offloads_encode_off_the_event_loop(_stub_model, monkeypatch):
    monkeypatch.setattr(lightrag_init, "_EMBED_OFFLOAD", True)
    # reset the lazily-bound semaphore so it binds to this test's loop
    monkeypatch.setattr(lightrag_init, "_EMBED_SEM", None)

    async def _drive():
        loop_thread = threading.get_ident()
        out = await lightrag_init._bge_embed(["a", "b"])
        return loop_thread, out

    loop_thread, out = asyncio.run(_drive())
    assert _stub_model.encode_thread is not None
    assert _stub_model.encode_thread != loop_thread, "encode must run OFF the event-loop thread"
    assert out.shape == (2, 2)


def test_embed_legacy_flag_runs_on_loop_thread(_stub_model, monkeypatch):
    monkeypatch.setattr(lightrag_init, "_EMBED_OFFLOAD", False)

    async def _drive():
        loop_thread = threading.get_ident()
        await lightrag_init._bge_embed(["x"])
        return loop_thread

    loop_thread = asyncio.run(_drive())
    assert _stub_model.encode_thread == loop_thread, "KS_EMBED_OFFLOAD=0 must run on the loop thread"


def test_embed_sem_is_asyncio_semaphore(monkeypatch):
    monkeypatch.setattr(lightrag_init, "_EMBED_SEM", None)

    async def _drive():
        return lightrag_init._get_embed_sem()

    sem = asyncio.run(_drive())
    assert isinstance(sem, asyncio.Semaphore)


def test_embed_offload_default_on():
    # Fresh import with no override → offload defaults ON (the fix is the default).
    mod = importlib.reload(lightrag_init)
    try:
        assert mod._EMBED_OFFLOAD is True
    finally:
        importlib.reload(mod)


@pytest.mark.parametrize("operation", ["embed", "rerank"])
@pytest.mark.parametrize("worker_fails", [False, True])
async def test_cancelled_gpu_call_keeps_permit_until_thread_finishes(
    monkeypatch, operation, worker_fails
):
    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(lightrag_init, "_EMBED_OFFLOAD", True)
    monkeypatch.setattr(lightrag_init, "_get_embed_sem", lambda: sem)
    monkeypatch.setattr(lightrag_init, "_get_rerank_sem", lambda: sem)
    monkeypatch.setattr(lightrag_init, "_RERANK_FAILURES", 0)
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    calls = []

    def compute(inputs):
        calls.append(inputs)
        if len(calls) == 1:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5), "test failed to release GPU worker"
            if worker_fails:
                raise ValueError("cancelled worker failed")
        return np.array([[1.0, 2.0]]) if operation == "embed" else [0.75]

    monkeypatch.setattr(lightrag_init, "_bge_encode_sync", compute)
    monkeypatch.setattr(lightrag_init, "_get_bge_reranker", lambda: object())
    monkeypatch.setattr(lightrag_init, "_predict_with_oom_retry", lambda model, pairs: compute(pairs))

    async def run():
        if operation == "embed":
            return await lightrag_init._bge_embed(["input"])
        return await lightrag_init._bge_rerank("query", ["document"])

    first = asyncio.create_task(run())
    second = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert sem.locked(), "cancelled caller released a running worker's permit"
        second = asyncio.create_task(run())
        await asyncio.sleep(0)
        assert len(calls) == 1 and not second.done()
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        if second is not None:
            result = await asyncio.wait_for(second, 2)
            if operation == "embed":
                np.testing.assert_array_equal(result, [[1.0, 2.0]])
            else:
                assert result == [{"index": 0, "relevance_score": 0.75}]
        await asyncio.wait_for(sem.acquire(), 2)
        sem.release()
    assert len(calls) == 2
    assert lightrag_init._RERANK_FAILURES == 0
