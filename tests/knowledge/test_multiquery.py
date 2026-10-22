"""Silent-failure observability on the live V-MQ path (audit 2026-06-18, siblings of #21/#29).

Two log-only degradation signals, both pure observability (no behavior change):
  - retrieve_fused: facet sub-queries that ERROR or return non-success are dropped from the RRF
    fusion (correct graceful degradation) but were previously SILENT — now logged LOUD, with the
    all-fail "collapsed to single-query baseline" case flagged.
  - _estimate_n_sources: an empty/whitespace-200 (S16 silent LLM failure) was conflated with a
    real "narrow question" estimate of 0 — now logged before taking the floor-budget fallback.
"""
import asyncio
import logging

import papervault.knowledge.query.multiquery as mq


def _run(coro):
    return asyncio.run(coro)


def _strong_base():
    """A base single-query result with strong coverage (so the fuse-gate passes) + one chunk."""
    return {
        "status": "success",
        "data": {
            "entities": [], "relationships": [],
            "chunks": [{"content": "base chunk", "file_path": "paper/Base2020", "reference_id": "1"}],
            "references": [{"file_path": "paper/Base2020", "reference_id": "1"}],
        },
        "metadata": {"processing_info": {"total_entities_found": 30}},  # >=20 → "strong"
    }


def _ok_facet(key):
    return {
        "status": "success",
        "data": {
            "entities": [], "relationships": [],
            "chunks": [{"content": f"facet chunk {key}", "file_path": f"paper/{key}", "reference_id": key}],
            "references": [{"file_path": f"paper/{key}", "reference_id": key}],
        },
        "metadata": {"processing_info": {"total_entities_found": 25}},
    }


class _StubRag:
    """First aquery_data call = base (intent); subsequent calls = facet sub-queries, served in order
    from `facet_results`. An Exception entry is RAISED (captured by gather's return_exceptions)."""

    def __init__(self, base, facet_results):
        self._base = base
        self._facets = list(facet_results)
        self.calls = 0

    async def aquery_data(self, q, param):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            return self._base
        nxt = self._facets.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _patch_decompose(monkeypatch, n):
    facets = [{"query": f"sub{i}", "hl": [], "ll": []} for i in range(n)]

    async def _fake_decompose(intent, *a, **k):  # noqa: ARG001
        return facets

    monkeypatch.setattr(mq, "_decompose", _fake_decompose)
    # KS_MQ_FANOUT=0 → skip _estimate_n_sources (floor budget), keep this test about the gather loop.
    monkeypatch.setenv("KS_MQ_FANOUT", "0")
    return facets


def test_partial_facet_failure_logs_degraded(monkeypatch, caplog):
    _patch_decompose(monkeypatch, 3)
    rag = _StubRag(_strong_base(), [_ok_facet("A"), RuntimeError("all keys failed"),
                                    {"status": "failure", "data": {}, "metadata": {}}])
    with caplog.at_level(logging.WARNING, logger="ks.query.multiquery"):
        data, _meta, fused_applied = _run(mq.retrieve_fused("broad intent", rag))
    assert fused_applied is True            # graceful: fusion still ran on base + the 1 ok facet
    assert "multiquery DEGRADED" in caplog.text
    assert "2/3" in caplog.text             # 2 of 3 facets failed (1 exc + 1 status!=success)
    assert "RuntimeError" in caplog.text    # exception type surfaced
    assert "COLLAPSED" not in caplog.text   # not all-fail → not the collapse message


def test_all_facets_fail_logs_collapse_to_baseline(monkeypatch, caplog):
    _patch_decompose(monkeypatch, 2)
    rag = _StubRag(_strong_base(), [RuntimeError("boom"), RuntimeError("boom")])
    with caplog.at_level(logging.WARNING, logger="ks.query.multiquery"):
        data, _meta, fused_applied = _run(mq.retrieve_fused("broad intent", rag))
    assert "multiquery DEGRADED" in caplog.text
    assert "2/2" in caplog.text
    assert "COLLAPSED to single-query baseline" in caplog.text
    # collapse = RRF over base alone → the base chunk survives, fusion silently == baseline
    assert any(c.get("file_path") == "paper/Base2020" for c in data["chunks"])


def test_all_facets_ok_no_degraded_log(monkeypatch, caplog):
    _patch_decompose(monkeypatch, 2)
    rag = _StubRag(_strong_base(), [_ok_facet("A"), _ok_facet("B")])
    with caplog.at_level(logging.WARNING, logger="ks.query.multiquery"):
        _run(mq.retrieve_fused("broad intent", rag))
    assert "multiquery DEGRADED" not in caplog.text


def test_success_but_empty_facet_is_not_a_failure(monkeypatch, caplog):
    # A facet returning status=success with empty data is a LEGITIMATELY empty retrieval, NOT a
    # failure — it must not trip the degradation log (the fairness nuance from the audit).
    _patch_decompose(monkeypatch, 2)
    empty_ok = {"status": "success", "data": {}, "metadata": {}}
    rag = _StubRag(_strong_base(), [_ok_facet("A"), empty_ok])
    with caplog.at_level(logging.WARNING, logger="ks.query.multiquery"):
        _run(mq.retrieve_fused("broad intent", rag))
    assert "multiquery DEGRADED" not in caplog.text


def test_estimate_n_sources_empty_200_logs_s16_and_returns_zero(monkeypatch, caplog):
    async def _whitespace_200(*a, **k):
        return "  \n \t "

    monkeypatch.setattr(mq, "mimo_complete", _whitespace_200)
    with caplog.at_level(logging.WARNING, logger="ks.query.multiquery"):
        n = _run(mq._estimate_n_sources("anything"))
    assert n == 0                           # floor-budget fallback preserved
    assert "S16" in caplog.text             # but no longer silent


def test_estimate_n_sources_real_estimate_passes_through(monkeypatch):
    async def _twelve(*a, **k):
        return "I think about 12 distinct papers."

    monkeypatch.setattr(mq, "mimo_complete", _twelve)
    assert _run(mq._estimate_n_sources("broad synthesis")) == 12


def test_anchor_and_decomposition_overlap(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        async def decompose(*args):
            started.set()
            return [{"query": "facet", "hl": [], "ll": []}]
        class Rag:
            async def aquery_data(self, query, param):
                if query == "anchor":
                    await asyncio.wait_for(started.wait(), 0.2)
                    return _strong_base()
                return _ok_facet("Facet")
        monkeypatch.setattr(mq, "_decompose", decompose)
        monkeypatch.setenv("KS_MQ_FANOUT", "0")
        data, _, fused = await mq.retrieve_fused("anchor", Rag())
        assert fused and any(c["file_path"] == "paper/Facet" for c in data["chunks"])
    _run(scenario())


def test_rejected_anchor_cancels_and_drains_speculative_decomposition(monkeypatch):
    async def scenario():
        started, stopped = asyncio.Event(), asyncio.Event()
        async def decompose(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        class Rag:
            async def aquery_data(self, query, param):
                assert query == "anchor", "gated anchor must not run facets"
                await asyncio.wait_for(started.wait(), 0.2)
                return {"status": "success", "data": {}, "metadata": {}}
        monkeypatch.setattr(mq, "_decompose", decompose)
        data, _, fused = await mq.retrieve_fused("anchor", Rag())
        assert not fused and not data.get("chunks")
        assert stopped.is_set()
    _run(scenario())


def test_fulltext_and_canon_survive_rrf_cut_and_per_paper_cap(monkeypatch):
    from papervault.knowledge.ingest.abstract_doc import ABSTRACT_HEADER
    _patch_decompose(monkeypatch, 1)
    monkeypatch.setattr(mq.aq, "_CHUNK_TOP_K", 3)
    monkeypatch.setattr(mq, "_MQ_MAX_CHUNKS_PER_PAPER", 1)
    base = _strong_base()
    base["data"]["chunks"] = [
        {"file_path": "paper/A", "content": ABSTRACT_HEADER + "\nA abstract"},
        {"file_path": "paper/Stub", "content": ABSTRACT_HEADER + "\nOther stub"},
        {"file_path": "A", "content": "Fulltext A"},
        {"file_path": "textbook:Book#s1", "content": "Textbook evidence"},
    ]
    data, _, fused = _run(mq.retrieve_fused("anchor", _StubRag(base, [_ok_facet("Facet")])))
    assert fused
    assert [c["file_path"] for c in data["chunks"]] == ["textbook:Book#s1", "paper/Facet", "A"]
    assert [c["source_tier"] for c in data["chunks"]] == ["canon", "fulltext", "fulltext"]
    assert [r["source_tier"] for r in data["references"]] == ["canon", "fulltext", "fulltext"]


def test_anchor_failure_drains_decomposition_without_swallowing_error(monkeypatch):
    import pytest
    async def scenario():
        started, stopped = asyncio.Event(), asyncio.Event()
        async def decompose(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        class Rag:
            async def aquery_data(self, query, param):
                await started.wait()
                raise RuntimeError("backend unavailable")
        monkeypatch.setattr(mq, "_decompose", decompose)
        with pytest.raises(RuntimeError, match="backend unavailable"):
            await mq.retrieve_fused("anchor", Rag())
        assert stopped.is_set()
    _run(scenario())


def test_query_cancellation_drains_decomposition(monkeypatch):
    import pytest
    async def scenario():
        started, stopped = asyncio.Event(), asyncio.Event()
        async def decompose(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        class Rag:
            async def aquery_data(self, query, param):
                await asyncio.Event().wait()
        monkeypatch.setattr(mq, "_decompose", decompose)
        query = asyncio.create_task(mq.retrieve_fused("anchor", Rag()))
        await started.wait()
        query.cancel()
        with pytest.raises(asyncio.CancelledError):
            await query
        assert stopped.is_set()
    _run(scenario())
