"""Pure unit tests for the deterministic eval backbone (experiments/eval/backbone.py).

No DB, no LLM, no LightRAG — hand-built result/gold fixtures only. Pins every metric in
the settled KS metric design: (1) paper-recall@12/hit@12/recall@5 (there is deliberately NO
references-recall / rerank-cut-gap metric — see SDD §6.10 A.(1) + backbone docstring: in
LightRAG mix mode data.references is a SUBSET of the already-cut data.chunks, so such a gap
is structurally a no-op; a guard test below pins that the fields are gone), (2) citation
integrity hallucinated_rate/phantom_rate, (6) gold-citation-recall, and the kb_coverage
guardrail (trap-empty + over-confidence rate), including the None semantics (traps -> recall
None; no inline cites -> rate None) and the sentinel handling (empty / synth-failed prose
carry no citations).

backbone.py lives under experiments/eval (not in the papervault.knowledge package), so we add
it to sys.path here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent.parent / "experiments" / "eval"
sys.path.insert(0, str(_EVAL_DIR))

import backbone as bb  # noqa: E402


# --- fixture helpers ---------------------------------------------------------

def _chunk(key: str) -> dict:
    return {"file_path": f"paper/{key}", "content": f"content of {key}"}


def _result(
    qid: str,
    *,
    chunk_keys: list[str],
    ref_keys: list[str] | None = None,
    cited_papers: list[str] | None = None,
    kb_coverage: str = "thin",
    answer: str = "",
    entities_found: int | None = None,
    extra_chunks: list[dict] | None = None,
) -> dict:
    """Build a saved per-question result dict.

    chunk_keys/ref_keys are paper keys (turned into 'paper/<key>' file_paths, rank order
    preserved). extra_chunks lets a test inject non-paper (textbook/web) chunks.
    """
    chunks = [_chunk(k) for k in chunk_keys]
    if extra_chunks:
        chunks = chunks + extra_chunks
    refs = [{"file_path": f"paper/{k}"} for k in (ref_keys if ref_keys is not None else chunk_keys)]
    res: dict = {
        "qid": qid,
        "data": {"chunks": chunks, "references": refs, "entities": [], "relationships": []},
        "cited_papers": cited_papers if cited_papers is not None else [],
        "kb_coverage": kb_coverage,
        "answer": answer,
    }
    if entities_found is not None:
        res["metadata"] = {"processing_info": {"total_entities_found": entities_found}}
    return res


def _gold(qid: str, gold_keys: list[str], band: str = "thin") -> dict:
    return {
        "qid": qid,
        "gold_keys": gold_keys,
        "nuggets": [] if not gold_keys else ["n1", "n2", "n3"],
        "per_paper_relevance": {k: "directly-answering" for k in gold_keys},
        "expected_coverage_band": band,
    }


# --- strip_paper_key / paper-set parsing -------------------------------------

def test_strip_paper_key():
    assert bb.strip_paper_key("paper/Corti2018") == "Corti2018"
    assert bb.strip_paper_key("textbook/978x") is None
    assert bb.strip_paper_key("web/https://x") is None
    assert bb.strip_paper_key("paper/") is None
    assert bb.strip_paper_key("") is None
    assert bb.strip_paper_key(None) is None


def test_retrieved_papers_dedup_and_rank_order():
    data = {"chunks": [_chunk("B"), _chunk("A"), _chunk("B"), _chunk("C")]}
    assert bb.retrieved_papers_from_chunks(data) == ["B", "A", "C"]  # dedup, first-seen order


def test_retrieved_papers_top_n_counts_chunks_not_papers():
    # top-5 means first 5 CHUNKS. Here the gold paper 'G' is the 6th chunk -> excluded at @5.
    data = {"chunks": [_chunk("X"), _chunk("X"), _chunk("Y"), _chunk("Z"), _chunk("W"), _chunk("G")]}
    assert "G" in bb.retrieved_papers_from_chunks(data)            # @12
    assert "G" not in bb.retrieved_papers_from_chunks(data, top_n=5)  # @5


def test_paper_set_ignores_non_paper_chunks():
    data = {"chunks": [_chunk("A"), {"file_path": "textbook/978"}, {"file_path": "web/x"}]}
    assert bb.retrieved_papers_from_chunks(data) == ["A"]


def test_at12_cut_is_enforced_in_module_not_delegated():
    # A dump with MORE than chunk_top_k chunks must be scored over the top-12 only — the @12
    # cut is enforced in backbone.py, not assumed to have been done upstream by LightRAG.
    # 14 chunks: papers C13/C14 are beyond the top-12 and must NOT appear in the default set.
    data = {"chunks": [_chunk(f"C{i}") for i in range(14)]}
    got = bb.retrieved_papers_from_chunks(data)  # default top_n = _CHUNK_TOP_K = 12
    assert len(got) == 12
    assert "C12" not in got and "C13" not in got   # 13th/14th chunk (0-indexed) excluded
    # explicit top_n=None opts out of the cut (scores every present chunk)
    assert len(bb.retrieved_papers_from_chunks(data, top_n=None)) == 14


def test_recall_at_12_excludes_gold_paper_beyond_top12():
    # gold {G}; G sits as the 13th chunk -> beyond the @12 cut -> recall@12 == 0 (not 1.0).
    chunks = [f"f{i}" for i in range(12)] + ["G"]
    r = _result("q", chunk_keys=chunks)
    m = bb.compute_question_metrics(r, _gold("q", ["G"]))
    assert m.paper_recall_at_12 == 0.0
    assert m.hit_at_12 == 0


def test_backbone_chunk_top_k_is_fixed_scoring_budget():
    # DECOUPLED 2026-06-14 (long-context frame): the backbone's @12-distinct SCORING budget is a
    # FIXED, denominator-fair constant (H4 anti-gaming) and is DELIBERATELY independent of how many
    # chunks the live path SERVES to synth (aquery._CHUNK_TOP_K is now 60 by default). The headline
    # retrieval term moved to @served_distinct; @12 stays a legacy/dashboard shadow at a fixed 12.
    # (Was: assert bb._CHUNK_TOP_K == aq._CHUNK_TOP_K — that mirror invariant no longer holds.)
    assert bb._CHUNK_TOP_K == 12


def test_backbone_sentinels_match_query_layer():
    # The backbone hardcodes the empty/synth-failed sentinels (so it stays import-free), but if
    # the query layer ever changes a sentinel the backbone would silently stop excluding those
    # answers from the prose metrics. Pin the two in lockstep (drill 2026-06-02e, tail 10).
    from papervault.knowledge.query import aquery as aq
    from papervault.knowledge.query import synth
    assert bb.EMPTY_ANSWER_SENTINEL == aq._EMPTY["answer"]
    assert bb.SYNTH_FAILED_PREFIX == synth.SYNTH_FAILED_PREFIX


# --- (1) retrieval recall ----------------------------------------------------

def test_recall_at_12_partial_and_hit():
    # gold {A,B,C}; chunks recall A and B only -> recall 2/3, hit 1
    r = _result("q", chunk_keys=["A", "B", "Z"])
    m = bb.compute_question_metrics(r, _gold("q", ["A", "B", "C"], band="strong"))
    assert m.paper_recall_at_12 == 2 / 3
    assert m.hit_at_12 == 1


def test_recall_at_12_zero_and_hit_zero():
    r = _result("q", chunk_keys=["X", "Y"])
    m = bb.compute_question_metrics(r, _gold("q", ["A"]))
    assert m.paper_recall_at_12 == 0.0
    assert m.hit_at_12 == 0


def test_recall_at_5_vs_12():
    # gold {G}. G is the 6th chunk: recalled @12, not @5.
    chunks = ["a", "b", "c", "d", "e", "G"]
    r = _result("q", chunk_keys=chunks)
    m = bb.compute_question_metrics(r, _gold("q", ["G"]))
    assert m.paper_recall_at_12 == 1.0
    assert m.paper_recall_at_5 == 0.0


def test_no_references_recall_or_rerank_cut_gap_metric():
    # The references-recall / rerank-cut-gap metric was REMOVED (SDD §6.10 A.(1), drill
    # 2026-06-02d): in LightRAG 1.4.16 mix mode data.references is built from the SAME
    # already-cut truncated_chunks as data.chunks (only dropping 'unknown_source'), so it is a
    # SUBSET of the chunk paper set, never a wider pre-cut pool. references-recall <= chunks-
    # recall always and the gap is <= 0 (never positive) -> a no-op that can never localize a
    # rerank cut. Pin that the fields no longer exist on QuestionMetrics so the dead metric
    # cannot silently creep back in.
    m = bb.compute_question_metrics(
        _result("q", chunk_keys=["A"]), _gold("q", ["A"])
    )
    assert not hasattr(m, "references_recall")
    assert not hasattr(m, "rerank_cut_gap")
    # the helper is gone too
    assert not hasattr(bb, "references_papers")


# --- (2) citation integrity --------------------------------------------------

def test_parse_prose_citations_basic_and_dedup_and_order():
    prose = "Claim one [Corti2018]. Claim two [Song2021] and again [Corti2018]. Tail [de2020]."
    assert bb.parse_prose_citations(prose) == ["Corti2018", "Song2021", "de2020"]


def test_parse_prose_citations_excludes_numeric_footnotes():
    prose = "A claim [12] with a real cite [A2015] and footnote [3]."
    assert bb.parse_prose_citations(prose) == ["A2015"]


def test_parse_prose_citations_leaves_bracketed_operator_sources_out_of_paper_tags():
    # #122: synth now brackets operator sources with their colon key; they are not paper tags,
    # so they must never be scored as hallucinated paper citations.
    prose = "A [Reames2023], [textbook:Baumjohann2012], [notebook:idea-scope], [web:nasa-srag]."
    assert bb.parse_prose_citations(prose) == ["Reames2023"]


def test_parse_prose_citations_skips_empty_and_synthfail_sentinels():
    assert bb.parse_prose_citations(bb.EMPTY_ANSWER_SENTINEL) == []
    assert bb.parse_prose_citations(bb.SYNTH_FAILED_PREFIX + "; see cited_papers.)") == []
    assert bb.is_real_prose(bb.EMPTY_ANSWER_SENTINEL) is False
    assert bb.is_real_prose("(synthesis LLM failed; ...)") is False
    assert bb.is_real_prose("Real prose [A].") is True


def test_hallucinated_rate():
    # retrieved (chunks) = {Aa,Bb}. prose cites Aa, Bb, and HALLUCINATED Hh (never retrieved).
    r = _result("q", chunk_keys=["Aa", "Bb"], answer="x [Aa] y [Bb] z [Hh].")
    m = bb.compute_question_metrics(r, _gold("q", ["Aa"]))
    assert m.n_prose_citations == 3
    assert m.hallucinated_rate == 1 / 3   # only Hh is not in {Aa,Bb}


def test_hallucinated_rate_none_when_no_prose_citations():
    r = _result("q", chunk_keys=["A"], answer="prose with no citation tags at all")
    m = bb.compute_question_metrics(r, _gold("q", ["A"]))
    assert m.n_prose_citations == 0
    assert m.hallucinated_rate is None


def test_phantom_rate():
    # cited_papers advertises {Aa,Bb,Pp}; prose only actually used {Aa,Bb} -> Pp is phantom.
    r = _result("q", chunk_keys=["Aa", "Bb", "Pp"], cited_papers=["Aa", "Bb", "Pp"],
                answer="used [Aa] and [Bb] only.")
    m = bb.compute_question_metrics(r, _gold("q", ["Aa"]))
    assert m.phantom_rate == 1 / 3


def test_phantom_rate_none_when_no_cited_papers():
    r = _result("q", chunk_keys=["A"], cited_papers=[], answer="prose [A].")
    m = bb.compute_question_metrics(r, _gold("q", ["A"]))
    assert m.phantom_rate is None


def test_phantom_rate_none_when_synth_failed():
    # synth failed -> no real prose -> phantom undefined even though cited_papers is non-empty.
    r = _result("q", chunk_keys=["A"], cited_papers=["A"],
                answer="(synthesis LLM failed; see cited_papers.)")
    m = bb.compute_question_metrics(r, _gold("q", ["A"]))
    assert m.has_real_prose is False
    assert m.phantom_rate is None


# --- (6) gold-citation-recall ------------------------------------------------

def test_gold_citation_recall():
    # gold {A,B,C}; cited_papers {A,B,Q} -> intersection {A,B} -> 2/3.
    r = _result("q", chunk_keys=["A", "B"], cited_papers=["A", "B", "Q"])
    m = bb.compute_question_metrics(r, _gold("q", ["A", "B", "C"], band="strong"))
    assert m.gold_citation_recall == 2 / 3


# --- kb_coverage guardrail ---------------------------------------------------

def test_trap_question_metrics_are_none_and_guardrail():
    # trap: gold empty. recall fields None; correct empty answer (kb 'empty') -> no violation.
    r = _result("trap", chunk_keys=[], ref_keys=[], cited_papers=[],
                kb_coverage="empty", answer=bb.EMPTY_ANSWER_SENTINEL)
    m = bb.compute_question_metrics(r, _gold("trap", []))
    assert m.is_trap is True
    assert m.paper_recall_at_12 is None
    assert m.hit_at_12 is None
    assert m.gold_citation_recall is None
    assert m.trap_violation is False
    assert m.overconfident is False  # traps can't be over-confident (no gold)


def test_trap_violation_when_not_empty_coverage():
    # trap but KS reported 'strong' coverage -> trap_violation True (over-confidence guardrail).
    r = _result("trap", chunk_keys=["X"], kb_coverage="strong", answer="overconfident [X].")
    m = bb.compute_question_metrics(r, _gold("trap", []))
    assert m.trap_violation is True


def test_overconfident_strong_but_zero_recall():
    # answerable, KS says 'strong', but recall@12 == 0 -> overconfident True.
    r = _result("q", chunk_keys=["X", "Y"], kb_coverage="strong", answer="confident [X].")
    m = bb.compute_question_metrics(r, _gold("q", ["A"]))
    assert m.paper_recall_at_12 == 0.0
    assert m.overconfident is True


def test_not_overconfident_when_recall_positive():
    r = _result("q", chunk_keys=["A"], kb_coverage="strong", answer="[A].")
    m = bb.compute_question_metrics(r, _gold("q", ["A"]))
    assert m.overconfident is False


def test_not_overconfident_when_coverage_thin():
    r = _result("q", chunk_keys=["X"], kb_coverage="thin", answer="[X].")
    m = bb.compute_question_metrics(r, _gold("q", ["A"]))
    assert m.overconfident is False


# --- qid mismatch guard ------------------------------------------------------

def test_qid_mismatch_raises():
    import pytest

    r = _result("qA", chunk_keys=["A"])
    with pytest.raises(ValueError):
        bb.compute_question_metrics(r, _gold("qB", ["A"]))


# --- aggregation -------------------------------------------------------------

def test_aggregate_drops_none_and_separates_populations():
    results = [
        # answerable, full recall, clean citations, correct coverage
        _result("q1", chunk_keys=["A"], cited_papers=["A"], kb_coverage="strong",
                answer="claim [A]."),
        # answerable, zero recall, over-confident
        _result("q2", chunk_keys=["X"], cited_papers=[], kb_coverage="strong",
                answer="claim [X]."),
        # trap, correct empty
        _result("t1", chunk_keys=[], cited_papers=[], kb_coverage="empty",
                answer=bb.EMPTY_ANSWER_SENTINEL),
        # trap, VIOLATION (non-empty coverage)
        _result("t2", chunk_keys=["Z"], cited_papers=[], kb_coverage="thin",
                answer="something [Z]."),
    ]
    gold_by_qid = {
        "q1": _gold("q1", ["A"]),
        "q2": _gold("q2", ["A"]),
        "t1": _gold("t1", []),
        "t2": _gold("t2", []),
    }
    out = bb.evaluate_run(results, gold_by_qid)
    agg = out["aggregate"]

    assert agg["n_questions"] == 4
    assert agg["n_traps"] == 2
    assert agg["n_answerable"] == 2
    # recall averages over answerable only: (1.0 + 0.0)/2
    assert agg["paper_recall_at_12"] == 0.5
    assert agg["hit_at_12"] == 0.5
    # over-confidence over answerable: q2 only -> 0.5
    assert agg["over_confidence_rate"] == 0.5
    # trap-violation over traps: t2 only -> 0.5
    assert agg["trap_violation_rate"] == 0.5
    # gold-citation-recall over answerable: q1=1.0 (cited A), q2=0.0 (cited none) -> 0.5
    assert agg["gold_citation_recall"] == 0.5


def test_aggregate_mean_is_none_when_no_defined_values():
    # one answerable question whose prose has NO citations -> hallucinated_rate undefined
    # for it -> corpus hallucinated_rate is None (averaged over zero defined values).
    results = [_result("q", chunk_keys=["A"], cited_papers=["A"], answer="no tags here")]
    out = bb.evaluate_run([results[0]], {"q": _gold("q", ["A"])})
    assert out["aggregate"]["hallucinated_rate"] is None


def test_evaluate_run_missing_gold_raises():
    import pytest

    with pytest.raises(KeyError):
        bb.evaluate_run([_result("q", chunk_keys=["A"])], {})
