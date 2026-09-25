"""Opt-in idle model release, using CPU-only fakes for the two GPU models."""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest
import torch

from papervault.knowledge.store import lightrag_init as models


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return {"dense_vecs": np.ones((len(texts), 2), dtype=np.float32)}


class FakeReranker:
    def predict(self, pairs, **kwargs):
        return np.array([0.8] * len(pairs), dtype=np.float32)


@pytest.fixture
def idle_models(monkeypatch):
    monkeypatch.setattr(models, "_IDLE_UNLOAD_ENABLED", True, raising=False)
    monkeypatch.setattr(models, "_IDLE_UNLOAD_TIMEOUT", 0.02, raising=False)
    monkeypatch.setattr(models, "_BGE_MODEL", None)
    monkeypatch.setattr(models, "_BGE_RERANKER", None)
    loads = {"embed": 0, "rerank": 0}
    emptied = threading.Event()
    monkeypatch.setattr(torch.cuda, "empty_cache", emptied.set)

    def get_embed():
        if models._BGE_MODEL is None:
            loads["embed"] += 1
            models._BGE_MODEL = FakeEmbedder()
        return models._BGE_MODEL

    def get_rerank():
        if models._BGE_RERANKER is None:
            loads["rerank"] += 1
            models._BGE_RERANKER = FakeReranker()
        return models._BGE_RERANKER

    monkeypatch.setattr(models, "_get_bge_model", get_embed)
    monkeypatch.setattr(models, "_get_bge_reranker", get_rerank)
    yield loads, emptied
    timer = getattr(models, "_IDLE_UNLOAD_TIMER", None)
    if timer is not None:
        timer.cancel()


def test_idle_timer_unloads_both_models_and_next_calls_reload(idle_models):
    loads, emptied = idle_models
    assert models._bge_encode_sync(["one"]).shape == (1, 2)
    assert asyncio.run(models._bge_rerank("q", ["doc"]))[0]["index"] == 0
    assert loads == {"embed": 1, "rerank": 1}
    assert emptied.wait(2), "idle timer did not release the CUDA cache"
    assert models._BGE_MODEL is None
    assert models._BGE_RERANKER is None
    models._bge_encode_sync(["again"])
    asyncio.run(models._bge_rerank("q", ["again"]))
    assert loads == {"embed": 2, "rerank": 2}


def test_call_during_unload_waits_and_reloads(idle_models, monkeypatch):
    loads, _ = idle_models
    entered = threading.Event()
    release = threading.Event()

    def slow_empty_cache():
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr(torch.cuda, "empty_cache", slow_empty_cache)
    models._bge_encode_sync(["first"])
    assert entered.wait(2), "idle unload did not begin"
    result = []
    worker = threading.Thread(target=lambda: result.append(models._bge_encode_sync(["second"])))
    worker.start()
    try:
        time.sleep(0.02)
        assert worker.is_alive(), "call raced through an in-progress unload"
    finally:
        release.set()
        worker.join(2)
    assert len(result) == 1
    assert result[0].shape == (1, 2)
    assert loads["embed"] == 2


def test_async_rerank_waits_for_unload_before_reloading(idle_models, monkeypatch):
    loads, _ = idle_models
    entered = threading.Event()
    release = threading.Event()

    def slow_empty_cache():
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr(torch.cuda, "empty_cache", slow_empty_cache)
    assert asyncio.run(models._bge_rerank("first", ["doc"]))[0]["index"] == 0
    assert entered.wait(2), "idle unload did not begin"

    async def drive():
        task = asyncio.create_task(models._bge_rerank("second", ["doc"]))
        try:
            await asyncio.sleep(0.02)
            assert not task.done(), "rerank used a model during CUDA cache release"
        finally:
            release.set()
        return await asyncio.wait_for(task, 2)

    assert asyncio.run(drive())[0]["index"] == 0
    assert loads["rerank"] == 2


def test_default_off_keeps_models_resident(monkeypatch, idle_models):
    loads, emptied = idle_models
    monkeypatch.setattr(models, "_IDLE_UNLOAD_ENABLED", False)
    models._bge_encode_sync(["one"])
    asyncio.run(models._bge_rerank("q", ["doc"]))
    assert not emptied.wait(0.1)
    assert models._BGE_MODEL is not None
    assert models._BGE_RERANKER is not None
    assert loads == {"embed": 1, "rerank": 1}
