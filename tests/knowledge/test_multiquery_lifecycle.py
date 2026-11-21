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


async def test_empty_decomposition_never_launches_estimator(monkeypatch):
    estimate = _Pending()
    base = _result()

    async def decompose(intent):
        return []

    async def retrieve(query, params):
        return base

    monkeypatch.setattr(mq, "_decompose", decompose)
    monkeypatch.setattr(mq, "_estimate_n_sources", estimate)
    monkeypatch.setenv("KS_MQ_FANOUT", "1")
    data, meta, fused = await asyncio.wait_for(
        mq.retrieve_fused("intent", SimpleNamespace(aquery_data=retrieve)), 0.5
    )
    assert data is base["data"] and meta is base["metadata"] and not fused
    assert not estimate.started.is_set()


async def test_inline_n_sources_skips_estimator(monkeypatch):
    estimate = _Pending()
    seen_params = []

    async def decompose(intent):
        return mq._DecomposedFacets(_FACETS, n_sources=9)

    async def retrieve(query, params):
        seen_params.append((query, params))
        return _result("Anchor" if query == "intent" else query)

    monkeypatch.setattr(mq, "_decompose", decompose)
    monkeypatch.setattr(mq, "_estimate_n_sources", estimate)
    monkeypatch.setenv("KS_MQ_FANOUT", "1")
    data, _, fused = await asyncio.wait_for(
        mq.retrieve_fused("intent", SimpleNamespace(aquery_data=retrieve)), 0.5
    )
    assert fused is True
    assert not estimate.started.is_set()
    # n_sources=9 scaled the facet budget above the floor (32 -> 63).
    facet_params = [p for q, p in seen_params if q != "intent"]
    assert facet_params and facet_params[0].chunk_top_k == 63


@pytest.mark.parametrize("failed", ["decomposition", "estimator"])
async def test_planning_failure_propagates_cleanly(monkeypatch, failed):
    estimate_calls = []

    async def fail(intent):
        raise RuntimeError("planning failed")

    async def bare_facets(intent):
        return list(_FACETS)

    async def record_estimate(intent):
        estimate_calls.append(intent)
        return 6

    async def retrieve(query, params):
        return _result()

    monkeypatch.setattr(mq, "_decompose", fail if failed == "decomposition" else bare_facets)
    monkeypatch.setattr(mq, "_estimate_n_sources", fail if failed == "estimator" else record_estimate)
    monkeypatch.setenv("KS_MQ_FANOUT", "1")
    with pytest.raises(RuntimeError, match="planning failed"):
        await asyncio.wait_for(
            mq.retrieve_fused("intent", SimpleNamespace(aquery_data=retrieve)), 0.5
        )
    if failed == "decomposition":
        assert estimate_calls == []


@pytest.mark.parametrize("stage", ["decomposition", "estimator"])
async def test_caller_cancellation_drains_planning_stage(monkeypatch, stage):
    pending = _Pending()

    async def bare_facets(intent):
        return list(_FACETS)

    async def retrieve(query, params):
        return _result()

    monkeypatch.setattr(mq, "_decompose", pending if stage == "decomposition" else bare_facets)
    monkeypatch.setattr(mq, "_estimate_n_sources", pending if stage == "estimator" else _Pending())
    monkeypatch.setenv("KS_MQ_FANOUT", "1")
    query = asyncio.create_task(mq.retrieve_fused("intent", SimpleNamespace(aquery_data=retrieve)))
    try:
        await asyncio.wait_for(pending.started.wait(), 0.5)
        query.cancel()
        await asyncio.wait_for(pending.cleaning.wait(), 0.5)
        assert not query.done()
        pending.release.set()
        with pytest.raises(asyncio.CancelledError):
            await query
        assert pending.stopped.is_set() and pending.task.done()
    finally:
        pending.release.set()
        await _finish(query, pending.task)


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
