"""S2: evidence-bound citations, honest coverage, and shared query receipts."""
from copy import deepcopy
import logging

import pytest

from papervault.knowledge.ingest.abstract_doc import ABSTRACT_HEADER
from papervault.knowledge.query import aquery as aq, synth


def evidence():
    return {
        "chunks": [
            {"file_path": "paper/Zeta", "content": "Z evidence", "reference_id": "9"},
            {"file_path": "Alpha", "content": "A evidence", "reference_id": "1"},
            {"file_path": "paper/Beta", "content": ABSTRACT_HEADER + "\nB evidence"},
            {"file_path": "textbook:Book#s2", "content": "Canon evidence"},
        ],
        # Reference IDs and ordering deliberately disagree with retrieval rank.
        "references": [{"file_path": fp} for fp in (
            "Alpha", "paper/Beta", "paper/Zeta", "textbook:Book#s2")],
        "entities": [], "relationships": [],
    }


META = {"query_mode": "mix", "processing_info": {"total_entities_found": 30}}


def test_outfeed_counts_only_verified_inline_citations_by_frequency_then_retrieval_rank():
    data = evidence()
    original = deepcopy(data)
    out = aq.build_outfeed(data, META,
        "[Alpha] [Beta] [Zeta] [Beta] [Invented] [textbook:Book] [1]",
        generation_id="g1", variant="multiquery", synthesis_model="pro", synthesis_latency_s=2.5)
    assert out["cited_papers"] == ["Beta", "Zeta", "Alpha"]
    assert out["retrieved_papers"] == ["Zeta", "Alpha", "Beta"]
    assert out["cited_sources"] == ["textbook:Book"]
    assert out["abstract_only_papers"] == ["Beta"]
    receipt = out["query_receipt"]
    assert receipt["generation_id"] == "g1"
    assert receipt["variant"] == "multiquery" and receipt["query_mode"] == "mix"
    assert receipt["kb_coverage"] == "strong"
    assert (receipt["retrieved_chunk_count"], receipt["retrieved_paper_count"], receipt["cited_paper_count"]) == (4, 3, 3)
    assert receipt["rerank_degraded"] is False
    assert receipt["synthesis_model"] == "pro" and receipt["synthesis_latency_s"] == 2.5
    assert data == original


@pytest.mark.parametrize("answer", ["No inline citations.", "", synth.SYNTH_FAILED_PREFIX + ")"])
def test_uncited_or_failed_synthesis_keeps_retrieval_separate(answer):
    out = aq.build_outfeed(evidence(), META, answer)
    assert out["cited_papers"] == []
    assert out["retrieved_papers"] == ["Zeta", "Alpha", "Beta"]
    assert out["query_receipt"]["cited_paper_count"] == 0


def test_fast_no_synth_retains_retrieval_rank_and_all_candidates():
    out = aq.build_outfeed(evidence(), META, None, no_synth=True)
    assert out["answer"] is None
    assert out["cited_papers"] == ["Zeta", "Alpha", "Beta"]
    assert out["query_receipt"]["synthesis_model"] is None
    assert out["query_receipt"]["synthesis_latency_s"] == 0


def test_abstract_only_coverage_is_weak_even_with_many_graph_entities():
    data = {"chunks": [{"file_path": "A", "content": ABSTRACT_HEADER + "\nabstract"}]}
    assert aq._assess_coverage(META, data) == "weak"
    data["chunks"].append({"file_path": "textbook:Book", "content": "Full evidence"})
    assert aq._assess_coverage(META, data) == "strong"


@pytest.mark.asyncio
@pytest.mark.parametrize("entities", [None, 0, 4])
async def test_empty_graph_coverage_skips_llm_and_preserves_receipt(monkeypatch, caplog, entities):
    monkeypatch.setattr(aq, "_QUERY_VARIANT", "single")
    async def forbidden(*args, **kwargs):
        pytest.fail("empty coverage must not invoke synthesis")
    monkeypatch.setattr(aq, "synth_answer", forbidden)
    class Rag:
        async def aquery_data(self, intent, param):
            return {"status": "success", "data": evidence(), "metadata": {
                "processing_info": {"total_entities_found": entities}}}
    with caplog.at_level(logging.INFO, logger="ks.query.aquery"):
        out = await aq.query("no graph support", rag=Rag())
    assert out["kb_coverage"] == "empty" and out["answer"] == aq._EMPTY["answer"]
    assert out["cited_papers"] == [] and out["retrieved_papers"] == ["Zeta", "Alpha", "Beta"]
    assert out["query_receipt"]["retrieved_chunk_count"] == 4
    assert out["query_receipt"]["synthesis_model"] is None
    assert any(getattr(r, "query_receipt", None) == out["query_receipt"] for r in caplog.records)


@pytest.mark.asyncio
async def test_weak_coverage_reaches_synth_and_discloses_abstract_limitation(monkeypatch):
    prompts = []
    async def complete(prompt, **kwargs):
        prompts.append(prompt)
        return "Abstract finding [A]."
    monkeypatch.setattr(synth, "mimo_complete", complete)
    answer = await synth.synth_answer({"chunks": [{"file_path": "A", "content":
        ABSTRACT_HEADER + "\nA finding"}]}, "question", kb_coverage="weak")
    assert "weak" in prompts[0].lower()
    assert "limited" in answer.lower() and "abstract" in answer.lower()
    assert "[A]" in answer


def test_source_tier_annotation_is_pure_and_abstract_overrides_canon():
    data = evidence()
    data["chunks"].extend([
        {"file_path": "Classic", "content": "Classic full text", "source_tier": "canon"},
        {"file_path": "Stub", "content": ABSTRACT_HEADER + "\nStub", "source_tier": "canon"},
    ])
    original = deepcopy(data)
    annotated = synth.annotate_sources(data)
    assert [c["source_tier"] for c in annotated["chunks"]] == [
        "fulltext", "fulltext", "abstract_only", "canon", "canon", "abstract_only"]
    assert {r["file_path"]: r["source_tier"] for r in annotated["references"]} == {
        "Alpha": "fulltext", "paper/Beta": "abstract_only", "paper/Zeta": "fulltext", "textbook:Book#s2": "canon"}
    assert data == original


@pytest.mark.asyncio
async def test_mcp_forwards_and_traces_observed_receipt(monkeypatch, caplog):
    from papervault.knowledge.mcp import server
    async def ready():
        pass
    async def run(**kwargs):
        return aq.build_outfeed(evidence(), META, "[Alpha]")
    monkeypatch.setattr(server, "_ensure_knowledge_authority", ready)
    monkeypatch.setattr(aq, "query", run)
    with caplog.at_level(logging.INFO):
        result = await server.query("query")
    assert result["retrieved_papers"] == ["Zeta", "Alpha", "Beta"]
    assert any(getattr(r, "query_receipt", None) == result["query_receipt"] for r in caplog.records)


def test_blank_chunks_cannot_establish_coverage_or_citations():
    data = {"chunks": [{"content": "  \n", "file_path": "A"}], "references": [{"file_path": "A"}]}
    out = aq.build_outfeed(data, META, "invented support [A]")
    assert out["kb_coverage"] == "empty" and out["cited_papers"] == []
    assert out["query_receipt"]["retrieved_chunk_count"] == 1


@pytest.mark.asyncio
async def test_live_receipt_records_rerank_degradation_and_synthesis_timing(monkeypatch):
    from papervault.knowledge.store import lightrag_init
    failures = [0]
    monkeypatch.setattr(lightrag_init, "rerank_failure_count", lambda: failures[0])
    monkeypatch.setattr(aq, "_QUERY_VARIANT", "single")
    async def answer(*args, **kwargs):
        return "[Alpha]"
    monkeypatch.setattr(aq, "synth_answer", answer)
    class Rag:
        async def aquery_data(self, intent, param):
            failures[0] += 1
            return {"status": "success", "data": evidence(), "metadata": META}
    out = await aq.query("question", rag=Rag())
    assert out["query_receipt"]["rerank_degraded"] is True
    assert out["query_receipt"]["synthesis_model"] == "pro"
    assert out["query_receipt"]["synthesis_latency_s"] >= 0
    assert out["cited_papers"] == ["Alpha"]
