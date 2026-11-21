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
        generation_id="g1", variant="multiquery", synthesis_model="gpt-6.1-terra@high", synthesis_latency_s=2.5)
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
    assert receipt["synthesis_model"] == "gpt-6.1-terra@high" and receipt["synthesis_latency_s"] == 2.5
    assert data == original


def test_receipt_retains_pinned_scope_and_available_freshness_without_mutation():
    snapshot = {"generation_id": "g1", "manifest_revision": 7, "revision": 9,
        "manifest": {"indexed_scope": ["paper:A"], "freshness": {"max_age_s": 300},
                     "indexed_at": "2026-10-20T12:00:00Z", "cohort_id": "frozen-corpus"}}
    before = deepcopy(snapshot)
    out = aq.build_outfeed(evidence(), META, "[Alpha]", serving_manifest=snapshot)
    receipt = out["query_receipt"]
    assert receipt["generation_id"] == "g1" and receipt["manifest_revision"] == 7
    assert receipt["indexed_scope"] == ["paper:A"]
    assert receipt["freshness"] == {"max_age_s": 300}
    assert receipt["indexed_at"] == "2026-10-20T12:00:00Z"
    assert receipt["cohort_id"] == "frozen-corpus"
    receipt["indexed_scope"].append("paper:Invented")
    receipt["freshness"]["max_age_s"] = 1
    assert snapshot == before


def test_receipt_declares_selected_scope_and_input_revisions_without_inventing_dates():
    snapshot = {"generation_id": "g1", "manifest_revision": 1,
        "manifest": {"selected_unit_receipts": {"paper:A": "receipt1"}, "entries": [
            {"unit_id": "paper:A", "consumed_input_revision": "input1", "record_revision": 3},
            {"unit_id": "paper:B", "consumed_input_revision": "input2", "excluded": True}]}}
    receipt = aq.build_outfeed(evidence(), META, "[Alpha]", serving_manifest=snapshot)["query_receipt"]
    assert receipt["indexed_scope"] == ["paper:A"]
    assert receipt["freshness"] == {"input_revisions": {"paper:A": "input1"}}
    assert "indexed_at" not in receipt
    legacy = aq.build_outfeed(evidence(), META, "[Alpha]")["query_receipt"]
    assert legacy["manifest_revision"] is None
    assert legacy["indexed_scope"] is None and legacy["freshness"] is None


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


@pytest.mark.parametrize("identity", [
    {"file_path": "A"}, {"full_doc_id": "abstract-A", "paper_key": "A"},
    {"paper_key": "A"},
])
def test_split_abstract_document_stays_weak_and_labels_every_prompt_chunk(identity):
    data = {"chunks": [
        {**identity, "content": ABSTRACT_HEADER + "\nFirst abstract chunk"},
        {**identity, "content": "Trailing abstract chunk"},
    ], "references": [{"file_path": "A"}]}
    before = deepcopy(data)
    annotated = synth.annotate_sources(data)
    assert [c["source_tier"] for c in annotated["chunks"]] == ["abstract_only", "abstract_only"]
    assert aq._assess_coverage(META, data) == "weak"
    prompt = synth._build_prompt(data, "question")
    assert prompt.count("(source tier: abstract_only)") == 2
    assert data == before


def test_split_abstract_outfeed_does_not_upgrade_abstract_only_paper():
    data = {"chunks": [
        {"file_path": "A", "content": ABSTRACT_HEADER + "\nFirst abstract chunk"},
        {"file_path": "A", "content": "Trailing abstract chunk"},
    ]}
    out = aq.build_outfeed(data, META, "Finding [A].")
    assert out["kb_coverage"] == "weak"
    assert out["abstract_only_papers"] == ["A"]


def test_abstract_document_with_only_paper_identity_keeps_outfeed_and_prompt_labels():
    data = {"chunks": [
        {"paper_key": "A", "content": ABSTRACT_HEADER + "\nAbstract start"},
        {"paper_key": "A", "content": "Abstract continuation"},
    ], "references": [{"file_path": "A"}]}
    out = aq.build_outfeed(data, META, "Finding [A].")
    assert out["abstract_only_papers"] == ["A"]
    assert out["kb_coverage"] == "weak"
    assert synth._build_prompt(data, "question").count("source key: A | credibility: abstract-only") == 2


def test_distinct_fulltext_document_for_same_paper_retains_fulltext_authority():
    data = {"chunks": [
        {"file_path": "A", "full_doc_id": "abstract-A", "content": ABSTRACT_HEADER + "\nabstract"},
        {"file_path": "A", "full_doc_id": "abstract-A", "content": "abstract continuation"},
        {"file_path": "A", "full_doc_id": "fulltext-A", "content": "Full text evidence"},
    ], "references": [{"file_path": "A"}]}
    annotated = synth.annotate_sources(data)
    assert [c["source_tier"] for c in annotated["chunks"]] == ["abstract_only", "abstract_only", "fulltext"]
    assert annotated["references"][0]["source_tier"] == "fulltext"
    out = aq.build_outfeed(data, META, "Finding [A].")
    assert out["kb_coverage"] == "strong"
    assert out["abstract_only_papers"] == []


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
    assert out["query_receipt"]["synthesis_model"] == "gpt-6.2-astra@xhigh"
    assert out["query_receipt"]["synthesis_latency_s"] >= 0
    assert out["cited_papers"] == ["Alpha"]


def test_multi_key_and_locator_citations_and_slash_operator_paths():
    data = evidence()
    data["chunks"].extend([
        {"file_path": "textbook/SlashBook#s1", "content": "Slash textbook"},
        {"file_path": "notebook/idea-23", "content": "Slash notebook"},
        {"file_path": "web/nasa-srag", "content": "Slash web"},
    ])
    answer = (
        "Consensus [Alpha, Zeta] and [Alpha; Beta] plus [FakeKey2025] "
        "with [textbook:Book, Ch. 5], [textbook/SlashBook, Sec. 3.2], "
        "[Zeta, pp. 12-15], [Zeta, Eq. (4)], [ Alpha ], [notebook:idea-23], [web:nasa-srag]."
    )
    out = aq.build_outfeed(data, META, answer)
    assert set(out["cited_papers"]) == {"Alpha", "Zeta", "Beta"}
    assert "textbook:Book" in out["cited_sources"]
    assert "textbook:SlashBook" in out["retrieved_sources"]
    assert "textbook:SlashBook" in out["cited_sources"]
    assert "notebook:idea-23" in out["cited_sources"]
    assert "web:nasa-srag" in out["cited_sources"]
    assert "FakeKey2025" in out["query_receipt"]["unverified_citations"]


def test_hallucinated_citations_recorded_and_logged_without_flagging_numeric_brackets(caplog):
    answer = (
        "Valid [Alpha], hallucinated [FakeAuthor2025] and [textbook:Unknown2020] and [Smith et al. 2024], "
        "non-citations [1] [0, 1] [km/s] and markdown [link](https://example.com)."
    )
    with caplog.at_level(logging.WARNING, logger="ks.query.aquery"):
        out = aq.build_outfeed(evidence(), META, answer)
    unverified = out["query_receipt"]["unverified_citations"]
    assert unverified == ["FakeAuthor2025", "textbook:Unknown2020", "Smith et al. 2024"]
    assert any("unverified inline citations" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_mcp_and_aquery_log_query_receipt_once(monkeypatch, caplog):
    from papervault.knowledge.mcp import server

    monkeypatch.setattr(server, "_queries_by_id", {})
    monkeypatch.setattr(server, "_queries_by_intent", {})
    monkeypatch.setattr(server, "_query_stopping", False)
    monkeypatch.setattr(aq, "_QUERY_VARIANT", "single")

    async def ready():
        pass

    class Rag:
        async def aquery_data(self, intent, param):
            return {"status": "success", "data": evidence(), "metadata": META}

    async def fake_synth(data, intent, **kw):
        return "Verified finding [Alpha]."

    async def fake_get_graph():
        return Rag()

    monkeypatch.setattr(server, "_ensure_knowledge_authority", ready)
    monkeypatch.setattr("papervault.knowledge.store.graph.get_graph", fake_get_graph)
    monkeypatch.setattr(aq, "synth_answer", fake_synth)

    with caplog.at_level(logging.INFO):
        result = await server.query("unique-dedup-receipt-check")
    receipt_logs = [r for r in caplog.records if getattr(r, "query_receipt", None) is not None]
    assert result["cited_papers"] == ["Alpha"]
    assert len(receipt_logs) == 1


@pytest.mark.asyncio
async def test_trimmed_prompt_chunks_excluded_from_verified_citations(monkeypatch, caplog):
    """A paper retrieved in data but trimmed out by KS_SYNTH_MAX_PROMPT_TOKENS must NOT pass as
    a verified cited_paper — citing it must trigger unverified_citations instead."""
    monkeypatch.setattr(synth, "_SYNTH_MAX_PROMPT_TOKENS", 220)

    async def fake_complete(prompt, **kwargs):
        assert "KeepPaper2023" in prompt
        assert "DropPaper2024" not in prompt
        return "Findings supported by [KeepPaper2023] and hallucinated [DropPaper2024]."

    monkeypatch.setattr(synth, "mimo_complete", fake_complete)
    data = {
        "chunks": [
            {"file_path": "paper/KeepPaper2023", "content": "Concise finding about SEP onset."},
            {"file_path": "paper/DropPaper2024", "content": " ".join(["overflow"] * 300)},
        ],
        "references": [{"file_path": "paper/KeepPaper2023"}, {"file_path": "paper/DropPaper2024"}],
        "entities": [],
        "relationships": [],
    }
    with caplog.at_level(logging.WARNING, logger="ks.query.aquery"):
        out = await aq.synthesize_outfeed(data, META, "Summarize SEP onset.")
    assert out["retrieved_papers"] == ["KeepPaper2023", "DropPaper2024"]
    assert out["cited_papers"] == ["KeepPaper2023"]
    assert out["query_receipt"]["unverified_citations"] == ["DropPaper2024"]
    assert any("unverified inline citations" in r.getMessage() for r in caplog.records)
