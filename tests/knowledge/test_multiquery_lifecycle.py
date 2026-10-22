"""Facet validation and task ownership at multi-query retrieval boundaries."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from papervault.knowledge.query import multiquery as mq


_FACETS = [
    {"query": "facet-a", "hl": ["theme"], "ll": []},
    {"query": "facet-b", "hl": [], "ll": ["instrument"]},
]


def _result(key="Anchor"):
    return {
        "status": "success",
        "data": {
            "chunks": [{"file_path": f"paper/{key}", "content": f"evidence {key}"}],
            "references": [{"file_path": f"paper/{key}"}],
            "entities": [],
            "relationships": [],
        },
        "metadata": {"processing_info": {"total_entities_found": 30}},
    }


class _Pending:
    """An operation whose cancellation cleanup requires an explicit release."""

    def __init__(self):
        self.started = asyncio.Event()
        self.cleaning = asyncio.Event()
        self.release = asyncio.Event()
        self.stopped = asyncio.Event()
        self.task = None

    async def __call__(self, *args, **kwargs):
        self.task = asyncio.current_task()
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cleaning.set()
            await self.release.wait()
            self.stopped.set()


async def _finish(*tasks):
    """Avoid leaking tasks even when exercising the unfixed implementation."""
    pending = [task for task in tasks if task is not None]
    for task in pending:
        if not task.done():
            task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.parametrize("decompose", [mq._decompose, mq._decompose_standard])
@pytest.mark.parametrize("payload", [
    [{"query": "facet", "hl": 1, "ll": []}],
    [{"query": "facet", "hl": [], "ll": 1}],
    [{"query": "facet", "hl": "theme", "ll": []}],
    [{"query": "facet", "hl": [], "ll": "term"}],
    [{"query": 42, "hl": ["theme"], "ll": []}],
    [{"query": ["facet"], "hl": ["theme"], "ll": []}],
    [{"query": "facet", "hl": [42], "ll": []}],
    [{"query": "facet", "hl": [], "ll": [None]}],
    [{"query": "facet", "hl": {"theme": True}, "ll": []}],
    [_FACETS[0], None],
    [_FACETS[0], ["nested"]],
])
async def test_malformed_facet_fields_fall_back_in_both_decomposers(monkeypatch, decompose, payload):
    async def complete(*args, **kwargs):
        return json.dumps(payload)

    monkeypatch.setattr(mq, "mimo_complete", complete)
    assert await decompose("intent") == []


@pytest.mark.parametrize("decompose", [mq._decompose, mq._decompose_standard])
async def test_valid_facets_trim_strings_and_keep_anchor_prefix(monkeypatch, decompose):
    async def complete(*args, **kwargs):
        return json.dumps([
            {"query": " anchor ", "hl": [" theme ", " "], "ll": []},
            _FACETS[1],
        ])

    monkeypatch.setattr(mq, "mimo_complete", complete)
    assert await decompose("intent", n=1) == [{"query": "anchor", "hl": ["theme"], "ll": []}]


@pytest.mark.parametrize("variant", ["fused", "standard"])
async def test_malformed_decomposition_keeps_healthy_anchor(monkeypatch, variant):
    base = _result()
    calls = []

    async def complete(*args, **kwargs):
        return '[{"query":"facet","hl":1,"ll":[]}]'

    async def retrieve(query, params):
        calls.append(query)
        return base

    monkeypatch.setattr(mq, "mimo_complete", complete)
    monkeypatch.setenv("KS_MQ_FANOUT", "0")
    result = await getattr(mq, f"retrieve_{variant}")("intent", SimpleNamespace(aquery_data=retrieve))
    assert [(c["file_path"], c["content"]) for c in result[0]["chunks"]] == [
        ("paper/Anchor", "evidence Anchor")]
    assert result[1] == base["metadata"]
    assert calls == ["intent"]
    if variant == "fused":
        assert result[2] is False


async def test_empty_decomposition_cancels_and_awaits_unneeded_estimator(monkeypatch):
    estimate = _Pending()
    base = _result()

    async def decompose(intent):
        await estimate.started.wait()
        return []

    async def retrieve(query, params):
        return base

    monkeypatch.setattr(mq, "_decompose", decompose)
    monkeypatch.setattr(mq, "_estimate_n_sources", estimate)
    monkeypatch.setenv("KS_MQ_FANOUT", "1")
    query = asyncio.create_task(mq.retrieve_fused("intent", SimpleNamespace(aquery_data=retrieve)))
    try:
        await asyncio.wait_for(estimate.cleaning.wait(), 0.5)
        assert not query.done(), "query returned before estimator cleanup completed"
        estimate.release.set()
        data, meta, fused = await asyncio.wait_for(query, 0.5)
        assert data is base["data"] and meta is base["metadata"] and not fused
        assert estimate.stopped.is_set() and estimate.task.done()
    finally:
        estimate.release.set()
        await _finish(query, estimate.task)


@pytest.mark.parametrize("failed", ["decomposition", "estimator"])
async def test_planning_failure_cancels_and_awaits_other_task(monkeypatch, failed):
    sibling = _Pending()

    async def fail(intent):
        await sibling.started.wait()
        raise RuntimeError("planning failed")

    async def retrieve(query, params):
        return _result()

    monkeypatch.setattr(mq, "_decompose", fail if failed == "decomposition" else sibling)
    monkeypatch.setattr(mq, "_estimate_n_sources", fail if failed == "estimator" else sibling)
    monkeypatch.setenv("KS_MQ_FANOUT", "1")
    query = asyncio.create_task(mq.retrieve_fused("intent", SimpleNamespace(aquery_data=retrieve)))
    try:
        await asyncio.wait_for(sibling.cleaning.wait(), 0.5)
        assert not query.done(), "planning error escaped before sibling cleanup"
        sibling.release.set()
        with pytest.raises(RuntimeError, match="planning failed"):
            await asyncio.wait_for(query, 0.5)
        assert sibling.stopped.is_set() and sibling.task.done()
    finally:
        sibling.release.set()
        await _finish(query, sibling.task)


async def test_caller_cancellation_drains_decomposition_and_estimator(monkeypatch):
    decomposition, estimator = _Pending(), _Pending()

    async def retrieve(query, params):
        return _result()

    monkeypatch.setattr(mq, "_decompose", decomposition)
    monkeypatch.setattr(mq, "_estimate_n_sources", estimator)
    monkeypatch.setenv("KS_MQ_FANOUT", "1")
    query = asyncio.create_task(mq.retrieve_fused("intent", SimpleNamespace(aquery_data=retrieve)))
    try:
        await asyncio.wait_for(asyncio.gather(decomposition.started.wait(), estimator.started.wait()), 0.5)
        query.cancel()
        await asyncio.wait_for(asyncio.gather(decomposition.cleaning.wait(), estimator.cleaning.wait()), 0.5)
        assert not query.done()
        decomposition.release.set()
        estimator.release.set()
        with pytest.raises(asyncio.CancelledError):
            await query
        assert decomposition.stopped.is_set() and estimator.stopped.is_set()
        assert decomposition.task.done() and estimator.task.done()
    finally:
        decomposition.release.set()
        estimator.release.set()
        await _finish(query, decomposition.task, estimator.task)


@pytest.mark.parametrize("variant", ["fused", "standard"])
async def test_caller_cancellation_awaits_every_retrieval_cleanup(monkeypatch, variant):
    pending = {facet["query"]: _Pending() for facet in _FACETS}

    async def decompose(intent):
        return _FACETS

    async def retrieve(query, params):
        if query == "intent":
            return _result()
        return await pending[query]()

    monkeypatch.setattr(mq, "_decompose" if variant == "fused" else "_decompose_standard", decompose)
    monkeypatch.setenv("KS_MQ_FANOUT", "0")
    query = asyncio.create_task(getattr(mq, f"retrieve_{variant}")("intent", SimpleNamespace(aquery_data=retrieve)))
    try:
        await asyncio.wait_for(asyncio.gather(*(operation.started.wait() for operation in pending.values())), 0.5)
        query.cancel()
        await asyncio.wait_for(asyncio.gather(*(operation.cleaning.wait() for operation in pending.values())), 0.5)
        assert not query.done()
        for operation in pending.values():
            operation.release.set()
        with pytest.raises(asyncio.CancelledError):
            await query
        assert all(operation.stopped.is_set() and operation.task.done() for operation in pending.values())
    finally:
        for operation in pending.values():
            operation.release.set()
        await _finish(query, *(operation.task for operation in pending.values()))


@pytest.mark.parametrize("variant", ["fused", "standard"])
async def test_ordinary_facet_failure_keeps_other_retrieval_results(monkeypatch, variant):
    async def decompose(intent):
        return _FACETS

    async def retrieve(query, params):
        if query == "facet-b":
            raise RuntimeError("one facet unavailable")
        return _result("Anchor" if query == "intent" else "Facet")

    monkeypatch.setattr(mq, "_decompose" if variant == "fused" else "_decompose_standard", decompose)
    monkeypatch.setenv("KS_MQ_FANOUT", "0")
    data, *_ = await getattr(mq, f"retrieve_{variant}")("intent", SimpleNamespace(aquery_data=retrieve))
    assert {chunk["file_path"] for chunk in data["chunks"]} == (
        {"paper/Anchor", "paper/Facet"} if variant == "fused" else {"paper/Facet"}
    )
