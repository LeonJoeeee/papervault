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
