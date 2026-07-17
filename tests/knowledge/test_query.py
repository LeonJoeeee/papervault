"""Pure unit tests for S3 query out-feed parsing (SDD §6.4). No LightRAG / no DB / no LLM.

Covers the two load-bearing pure functions over a canned aquery_data dict:
  _cited_papers   — strip 'paper/' from references[].file_path, dedupe, sort; ignore
                    textbook//web//unknown (F12: chunk-level refs, not entity-level).
  _assess_coverage — graph-signal kb_coverage from metadata.processing_info, None-safe
                     (failure/empty → empty, processing_info absent).
And the empty/failure guards on the `query()` entrypoint (rag stubbed, no LightRAG).
"""
import asyncio

from papervault.knowledge.query.aquery import _assess_coverage, _cited_papers, query


def _success(references, entities_found, n_chunks):
    return {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [],
            "relationships": [],
            "chunks": [{"content": "c", "file_path": "paper/X"} for _ in range(n_chunks)],
            "references": references,
        },
        "metadata": {
            "query_mode": "mix",
            "processing_info": {"total_entities_found": entities_found},
        },
    }


# ---- _cited_papers ----------------------------------------------------------

def test_cited_papers_strips_paper_prefix_only():
    refs = [
        {"reference_id": "1", "file_path": "paper/Reames2023"},
        {"reference_id": "2", "file_path": "textbook/9780521861014"},
        {"reference_id": "3", "file_path": "web/https://nasa.gov/x"},
        {"reference_id": "4", "file_path": "paper/Jokipii1966"},
    ]
    assert _cited_papers({"references": refs}) == ["Jokipii1966", "Reames2023"]  # sorted, paper/ only


def test_cited_papers_dedup_and_sort():
    refs = [
        {"reference_id": "1", "file_path": "paper/Bbb2020"},
        {"reference_id": "2", "file_path": "paper/Aaa2019"},
        {"reference_id": "3", "file_path": "paper/Bbb2020"},  # dup
    ]
    assert _cited_papers({"references": refs}) == ["Aaa2019", "Bbb2020"]


def test_cited_papers_empty_when_no_paper_refs():
    # naive mode / no graph signal → references may be empty or non-paper only
    assert _cited_papers({"references": []}) == []
    assert _cited_papers({"references": [{"reference_id": "1", "file_path": "web/x"}]}) == []
    assert _cited_papers({}) == []  # references key absent


def test_cited_papers_ignores_blank_or_malformed_paths():
    refs = [
        {"reference_id": "1", "file_path": "paper/"},      # empty key after prefix → skip
        {"reference_id": "2", "file_path": ""},            # blank
        {"reference_id": "3", "file_path": "paper/Good2021"},
    ]
    assert _cited_papers({"references": refs}) == ["Good2021"]


# ---- _assess_coverage (graph signal, None-safe) -----------------------------

def test_coverage_strong_thin_empty_thresholds():
    data = {"chunks": [{"content": "c"}]}
    assert _assess_coverage({"processing_info": {"total_entities_found": 25}}, data) == "strong"
    assert _assess_coverage({"processing_info": {"total_entities_found": 20}}, data) == "strong"
    assert _assess_coverage({"processing_info": {"total_entities_found": 19}}, data) == "thin"
    assert _assess_coverage({"processing_info": {"total_entities_found": 5}}, data) == "thin"
    assert _assess_coverage({"processing_info": {"total_entities_found": 4}}, data) == "empty"
    assert _assess_coverage({"processing_info": {"total_entities_found": 0}}, data) == "empty"


def test_coverage_empty_when_no_chunks():
    # chunks empty → empty regardless of entity count
    assert _assess_coverage({"processing_info": {"total_entities_found": 99}}, {"chunks": []}) == "empty"


def test_coverage_none_safe_on_missing_processing_info():
    data = {"chunks": [{"content": "c"}]}
    assert _assess_coverage(None, data) == "empty"                      # metadata absent
    assert _assess_coverage({}, data) == "empty"                        # processing_info absent
    assert _assess_coverage({"processing_info": {}}, data) == "empty"   # total_entities_found absent → None


# ---- query() guards (rag stubbed; no LightRAG) ------------------------------

class _StubRag:
    def __init__(self, res):
        self._res = res

    async def aquery_data(self, intent, param):  # noqa: ARG002
        return self._res


def _run(coro):
    return asyncio.run(coro)


def test_query_failure_status_returns_empty():
    # aquery_data failure: {status:'failure', data:{}, metadata:{failure_reason,mode}}
    res = {"status": "failure", "message": "no results", "data": {}, "metadata": {"failure_reason": "no_results"}}
    out = _run(query("anything", rag=_StubRag(res)))
    assert out == {"answer": "(KB 无相关知识)", "cited_papers": [], "kb_coverage": "empty"}


def test_query_empty_data_returns_empty():
    res = {"status": "success", "message": "ok", "data": {}}
    out = _run(query("anything", rag=_StubRag(res)))
    assert out["cited_papers"] == [] and out["kb_coverage"] == "empty"


def test_query_naive_empty_references_yields_no_cited():
    # naive-like: chunks present but references empty + no processing_info → empty coverage, no cites.
    # synth is stubbed out by making the LLM path unreachable: data has no chunks/ents so synth
    # produces a fallback string; we only assert the structured fields here.
    res = {
        "status": "success",
        "data": {"entities": [], "relationships": [], "chunks": [], "references": []},
        "metadata": {},
    }
    out = _run(query("anything", rag=_StubRag(res)))
    assert out["cited_papers"] == []
    assert out["kb_coverage"] == "empty"


# ---- synth empty/whitespace-200 → failure (S16 / #21) -----------------------

def test_synth_empty_200_treated_as_failure(monkeypatch):
    """S16 (SDD §6.4): a 200 with empty/whitespace content is a SILENT failure — synth must
    return the honesty fallback, NOT hand the out-feed a blank answer string."""
    import papervault.knowledge.query.synth as synth

    async def _whitespace_200(*a, **k):
        return "   \n  \t "

    monkeypatch.setattr(synth, "mimo_complete", _whitespace_200)
    out = _run(synth.synth_answer({"chunks": [{"content": "c"}]}, "anything"))
    assert out.startswith(synth.SYNTH_FAILED_PREFIX)


def test_synth_real_answer_passes_through(monkeypatch):
    import papervault.knowledge.query.synth as synth

    async def _real(*a, **k):
        return "  modulation is driven by drift [X2020]  "

    monkeypatch.setattr(synth, "mimo_complete", _real)
    out = _run(synth.synth_answer({"chunks": [{"content": "c"}]}, "anything"))
    assert out == "modulation is driven by drift [X2020]"
