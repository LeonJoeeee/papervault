"""Tests for experiments/eval/headline.py — headline composite + HARDENED red-line gates + verdict
(SDD §6.10 E/G; §8 adversarial-drill hardening H1-H7).

Pure-python, no DB/LLM/judge. compare_metric uses exact sign-flip enumeration at n<=20, so the
permutation p is deterministic; n_boot kept small for speed.

The §8 drill found the ORIGINAL metric not_trustworthy_fix_first: a 'winning' variant could make
KS worse (paired sign-flip permutation is near-inert for the deterministic poison rates at n=21;
single-Q-driven wins slip through; chunk-based recall lets a de-dup variant inflate mechanically;
trap/faithfulness/gold-cite weren't gated). These tests pin the fixes:
  · absolute gate fires on a deterministic poison regression the paired test would miss (H1)
  · trap / faithfulness / gold-cite gates veto a headline win (H2)
  · verdict bakes in the noise floor (H3 ii) + jackknife rejects single-Q-driven wins (H3 iii)
  · distinct-paper recall differs from chunk-based on a dup-heavy fixture (H4)
  · gaming scenario: recall 0 / nugget 1.0 from substitute papers → gold_citation_recall gate
    catches 'substance from the wrong papers' (the drill's slice20-q5 case).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from papervault.eval import headline as H

# backbone lives under experiments/eval (not in the package) — add it for the H4 recall tests.
_EVAL_DIR = Path(__file__).resolve().parent.parent / "experiments" / "eval"
sys.path.insert(0, str(_EVAL_DIR))
import backbone as bb  # noqa: E402

_KW = dict(n_boot=2000, n_perm=2000, seed=0)
# verdict needs the noise floor baked in; baseline 3-run wobble was run_mean_sd≈0.0075 (BASELINE.md).
# Most gate-isolation tests pass a 0 floor to take criterion (ii) out of the picture and probe the
# OTHER gates — that now requires the explicit allow_zero_noise_floor opt-out (H8d: verdict rejects
# a <=0 floor by default so production can't silently disable the over-noise check).
_VKW = dict(noise_floor_run_mean_sd=0.0, allow_zero_noise_floor=True, **_KW)


# --------------------------------------------------------------------------- #
# harmonic + headline table (core unchanged)
# --------------------------------------------------------------------------- #
def test_harmonic_basics():
    assert H.harmonic(1.0, 1.0) == 1.0
    assert H.harmonic(0.0, 0.9) == 0.0          # either 0 → 0
    assert H.harmonic(0.0, 0.0) == 0.0
    assert math.isclose(H.harmonic(0.5, 0.5), 0.5)
    # punishes lopsided: harmonic(0.8,0.2)=0.32 << arithmetic 0.5
    assert math.isclose(H.harmonic(0.8, 0.2), 0.32, abs_tol=1e-9)
    assert H.harmonic(0.8, 0.2) < (0.8 + 0.2) / 2


def test_headline_table_uses_distinct_recall_and_excludes_traps():
    # H4: the headline reads paper_recall_at_12_distinct, NOT chunk-based paper_recall_at_12.
    backbone = {
        "paper_recall_at_12": {"q1": 0.2, "q2": 0.2},            # chunk-based — must be IGNORED
        "paper_recall_at_12_distinct": {"q1": 1.0, "q2": 0.5},   # distinct — the one used
    }   # trap q3 absent (None→omitted)
    judge = {"nugget_recall": {"q1": 0.5, "q2": 0.5, "q4": 1.0}}  # q4 only in judge
    t = H.headline_table(backbone, judge)
    assert set(t) == {"q1", "q2"}                # only the intersection
    assert math.isclose(t["q1"], H.harmonic(1.0, 0.5))   # uses the DISTINCT recall (1.0), not 0.2
    assert math.isclose(H.headline_mean(t), (t["q1"] + t["q2"]) / 2)


def test_headline_mean_empty():
    assert H.headline_mean({}) == 0.0


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _tables(
    recall, nugget, halluc, overconf, citeprec,
    *, faith=None, goldcite=None,
):
    """Build {'backbone':..,'judge':..} tables from per-qid dicts.

    `recall` fills paper_recall_at_12_distinct (the headline's recall). faith/goldcite default to
    a clean (non-regressing) value matched to the baseline so unrelated tests don't trip the new
    gates. goldcite defaults to 1.0 (clean), faithfulness to 0.9 (clean).
    """
    qids = list(recall)
    if faith is None:
        faith = {q: 0.9 for q in qids}
    if goldcite is None:
        goldcite = {q: 1.0 for q in qids}
    return {
        "backbone": {
            "paper_recall_at_12_distinct": recall,
            "hallucinated_rate": halluc,
            "over_confidence_rate": overconf,
            "gold_citation_recall": goldcite,
        },
        "judge": {
            "nugget_recall": nugget,
            "citation_support_precision": citeprec,
            "faithfulness": faith,
        },
    }


# --------------------------------------------------------------------------- #
# absolute gate (H1): fires on a deterministic poison regression the paired test misses
# --------------------------------------------------------------------------- #
def test_absolute_gate_lower_is_better():
    qs = [f"q{i}" for i in range(8)]
    base = {q: 0.0 for q in qs}
    worse = {q: 0.05 for q in qs}                # hallucination went 0 → 0.05 everywhere
    g = H.absolute_gate("hallucinated_rate", False, base, worse)
    assert g["kind"] == "absolute"
    assert g["regressed"] is True                # mean worse by 0.05 > epsilon 0.0
    g2 = H.absolute_gate("hallucinated_rate", False, base, dict(base))
    assert g2["regressed"] is False              # identical → no regression


def test_absolute_gate_fires_where_paired_test_would_not():
    # THE H1 HOLE: a deterministic poison CONCENTRATED on a minority of the 21 Qs (18 unchanged,
    # 3 go 0 → 0.34). The paired sign-flip permutation is near-inert here — most deltas are 0, so
    # the permutation null overlaps the observed mean and p stays large (~0.26, NOT significant) →
    # the OLD paired-only gate would WAVE THIS THROUGH. The absolute mean-vs-mean gate catches it
    # (corpus hallucinated_rate rose 0 → ~0.049). This is the drill's poison-rate scenario.
    qs = [f"q{i}" for i in range(21)]
    base = {q: 0.0 for q in qs}
    var = dict(base)
    for q in ("q0", "q1", "q2"):
        var[q] = 0.34                            # 3/21 Qs newly fabricate citations
    # the absolute gate fails it:
    g_abs = H.absolute_gate("hallucinated_rate", False, base, var)
    assert g_abs["regressed"] is True
    # ...while the paired test, on its own, does NOT flag it (p not < 0.05):
    g_paired = H.gate_regressed("hallucinated_rate", False, base, var, **_KW)
    assert g_paired["p_value"] > 0.05
    assert g_paired["paired_regressed"] is False
    # (the paired-with-floor gate's absolute floor still catches it — that's the H1 tail)
    assert g_paired["floor_regressed"] is True
    assert g_paired["regressed"] is True


def test_absolute_gate_higher_is_better_gold_citation_recall():
    qs = [f"q{i}" for i in range(8)]
    base = {q: 0.9 for q in qs}
    worse = {q: 0.6 for q in qs}                 # gold_citation_recall dropped
    g = H.absolute_gate("gold_citation_recall", True, base, worse)
    assert g["regressed"] is True
    g2 = H.absolute_gate("gold_citation_recall", True, base, {q: 0.95 for q in qs})
    assert g2["regressed"] is False              # went UP → fine


# --------------------------------------------------------------------------- #
# paired-with-floor gate (judge metrics): keeps the paired test AND an absolute floor
# --------------------------------------------------------------------------- #
def test_gate_regressed_paired_significant():
    qs = [f"q{i}" for i in range(8)]
    base = {q: 0.9 for q in qs}
    worse = {q: 0.4 for q in qs}                 # citation precision dropped a lot, uniformly
    g = H.gate_regressed("citation_support_precision", True, base, worse, **_KW)
    assert g["regressed"] is True                # floor catches it (and paired may too)
    assert g["floor_regressed"] is True


def test_gate_regressed_clean_no_regression():
    qs = [f"q{i}" for i in range(8)]
    base = {q: 0.9 for q in qs}
    g = H.gate_regressed("citation_support_precision", True, base, dict(base), **_KW)
    assert g["regressed"] is False


# --------------------------------------------------------------------------- #
# verdict: clear win (passes all of i-iv)
# --------------------------------------------------------------------------- #
def test_verdict_clear_win():
    qs = [f"q{i}" for i in range(8)]
    base = _tables(
        recall={q: 0.4 for q in qs}, nugget={q: 0.4 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs})
    var = _tables(
        recall={q: 0.8 for q in qs}, nugget={q: 0.8 for q in qs},  # both core metrics up
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs})
    v = H.verdict(base, var, trap_refusal_baseline=0.5, trap_refusal_variant=0.75, **_VKW)
    assert v["variant_headline"] > v["baseline_headline"]
    assert v["headline"]["improved"] is True
    assert v["passes_noise_floor"] is True
    assert v["survives_jackknife"] is True
    assert all(not g["regressed"] for g in v["gates"].values())
    assert v["win"] is True


def test_verdict_poison_gate_vetoes_headline_win():
    # headline up, but hallucinated_rate crept up uniformly — the ABSOLUTE gate vetoes (H1).
    qs = [f"q{i}" for i in range(8)]
    base = _tables(
        recall={q: 0.4 for q in qs}, nugget={q: 0.4 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs})
    var = _tables(
        recall={q: 0.8 for q in qs}, nugget={q: 0.8 for q in qs},   # headline up...
        halluc={q: 0.05 for q in qs},                                # ...but now fabricates citations
        overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs})
    v = H.verdict(base, var, trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_VKW)
    assert v["headline"]["improved"] is True
    assert v["gates"]["hallucinated_rate"]["regressed"] is True
    assert v["gates"]["hallucinated_rate"]["kind"] == "absolute"
    assert v["win"] is False                     # poison gate vetoes the headline win


# --------------------------------------------------------------------------- #
# H2: new gates (trap / faithfulness / gold-cite) veto a headline win
# --------------------------------------------------------------------------- #
def test_verdict_trap_refusal_gate_vetoes(  ):
    qs = [f"q{i}" for i in range(8)]
    base = _tables(
        recall={q: 0.4 for q in qs}, nugget={q: 0.4 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs})
    var = _tables(
        recall={q: 0.8 for q in qs}, nugget={q: 0.8 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs})
    # headline clearly up & clean — but refusal honesty dropped 0.58 → 0.33 (H2a) → veto.
    v = H.verdict(base, var, trap_refusal_baseline=0.58, trap_refusal_variant=0.33, **_VKW)
    assert v["headline"]["improved"] is True
    assert v["gates"]["trap_correct_refusal_rate"]["regressed"] is True
    assert v["win"] is False


def test_verdict_faithfulness_gate_vetoes():
    qs = [f"q{i}" for i in range(8)]
    base = _tables(
        recall={q: 0.4 for q in qs}, nugget={q: 0.4 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs},
        faith={q: 0.9 for q in qs})
    var = _tables(
        recall={q: 0.8 for q in qs}, nugget={q: 0.8 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs},
        faith={q: 0.6 for q in qs})              # faithfulness dropped (H2b) → veto
    v = H.verdict(base, var, trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_VKW)
    assert v["headline"]["improved"] is True
    assert v["gates"]["faithfulness"]["regressed"] is True
    assert v["win"] is False


def test_verdict_gold_citation_recall_gate_vetoes():
    qs = [f"q{i}" for i in range(8)]
    base = _tables(
        recall={q: 0.4 for q in qs}, nugget={q: 0.4 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs},
        goldcite={q: 0.9 for q in qs})
    var = _tables(
        recall={q: 0.8 for q in qs}, nugget={q: 0.8 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs},
        goldcite={q: 0.6 for q in qs})           # gold_citation_recall dropped (H2c) → veto
    v = H.verdict(base, var, trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_VKW)
    assert v["headline"]["improved"] is True
    assert v["gates"]["gold_citation_recall"]["regressed"] is True
    assert v["gates"]["gold_citation_recall"]["kind"] == "absolute"
    assert v["win"] is False


# --------------------------------------------------------------------------- #
# H3 ii: noise floor baked in
# --------------------------------------------------------------------------- #
def test_verdict_noise_floor_vetoes_a_ci_win_inside_the_wobble():
    # A tiny but statistically-clean improvement that sits INSIDE the test-retest wobble. The CI
    # may say "improved", but mean_delta < k*run_mean_sd → not a win (H3 ii). Use a uniform +0.01
    # headline lift but a run_mean_sd of 0.05 (well above it).
    qs = [f"q{i}" for i in range(12)]
    base = _tables(
        recall={q: 0.80 for q in qs}, nugget={q: 0.80 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs})
    var = _tables(
        recall={q: 0.81 for q in qs}, nugget={q: 0.81 for q in qs},   # ~+0.01 headline
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs})
    v = H.verdict(base, var, noise_floor_run_mean_sd=0.05, k=1.0,
                  trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_KW)
    assert v["headline"]["improved"] is True     # CI/p say improved
    assert v["passes_noise_floor"] is False       # ...but inside the wobble
    assert v["win"] is False


# --------------------------------------------------------------------------- #
# H3 iii: jackknife rejects a win that hinges on ONE question
# --------------------------------------------------------------------------- #
def test_jackknife_rejects_single_q_driven_win():
    # 20 saturated Qs (delta exactly 0) + ONE big mover. Mirrors the drill's 18/21-saturated
    # landscape where one Q can carry the headline. Under the R-tail-1 redesign the jackknife
    # drops only MOVABLE questions; with a single mover n_movers==1 < 2 → not robust (a single-
    # Q-driven win is rejected up front, with an explanatory note rather than a per-drop entry).
    qs = [f"q{i}" for i in range(20)]
    base = {q: 0.9 for q in qs}
    var = dict(base)                              # 20 Qs identical → delta 0
    base["qX"] = 0.1
    var["qX"] = 0.95                              # one huge mover
    from papervault.eval import stats as S
    res = S.jackknife_headline(base, var, **_KW)
    assert res["n_movers"] == 1                   # only qX moves; the 20 saturated ties are dropped
    assert res["survives"] is False               # a single-mover win is not robust
    assert res["per_qid"] == {}                   # rejected before any per-drop pass


def test_jackknife_survives_a_broad_win():
    # a broad, every-question improvement survives ANY single drop.
    qs = [f"q{i}" for i in range(10)]
    base = {q: 0.3 for q in qs}
    var = {q: 0.7 for q in qs}
    from papervault.eval import stats as S
    res = S.jackknife_headline(base, var, **_KW)
    assert res["survives"] is True
    assert all(res["per_qid"].values())


def test_verdict_single_q_driven_win_is_no_win():
    # Wire the single-Q-driven headline through verdict end-to-end → win False via jackknife.
    qs = [f"q{i}" for i in range(20)]
    recall_b = {q: 0.9 for q in qs}
    nug_b = {q: 0.9 for q in qs}
    recall_v = dict(recall_b)
    nug_v = dict(nug_b)
    recall_b["qX"], nug_b["qX"] = 0.1, 0.1
    recall_v["qX"], nug_v["qX"] = 0.95, 0.95     # only qX moves
    clean = lambda d: {q: 0.0 for q in d}        # noqa: E731
    cite = lambda d: {q: 0.9 for q in d}         # noqa: E731
    base = _tables(recall_b, nug_b, clean(recall_b), clean(recall_b), cite(recall_b))
    var = _tables(recall_v, nug_v, clean(recall_v), clean(recall_v), cite(recall_v))
    v = H.verdict(base, var, trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_VKW)
    assert v["survives_jackknife"] is False
    assert v["win"] is False


# --------------------------------------------------------------------------- #
# H4: distinct-paper recall differs from chunk-based on a dup-heavy fixture
# --------------------------------------------------------------------------- #
def _chunk(key):
    return {"file_path": f"paper/{key}", "content": f"c-{key}"}


def test_distinct_paper_recall_differs_from_chunk_based_on_dup_heavy_fixture():
    # Dup-dominated: paper A fills the first 12 chunk slots, so chunk-based @12 only surfaces {A}
    # and the gold paper G (the 13th chunk) is excluded by the slot budget. But distinct-paper @12
    # walks the FULL reranked order, deduping to distinct papers, so G is the 2nd DISTINCT paper —
    # well within the first-12-distinct budget → distinct recall FINDS it where chunk-@12 misses.
    chunks = [_chunk("A")] * 12 + [_chunk("G")]   # 13 chunks; chunk-@12 = {A}; G is 13th
    data = {"chunks": chunks}
    gold = {"qid": "q", "gold_keys": ["G"], "nuggets": ["n1"],
            "per_paper_relevance": {"G": "directly-answering"}, "expected_coverage_band": "thin"}
    result = {"qid": "q", "data": data, "cited_papers": [], "kb_coverage": "thin", "answer": "x [A]."}
    m = bb.compute_question_metrics(result, gold)
    # chunk-based @12: only sees A among the first 12 chunks → recall 0.
    assert m.paper_recall_at_12 == 0.0
    # distinct @12: first 12 DISTINCT papers in reranked order = {A, G} → recall 1.0.
    assert m.paper_recall_at_12_distinct == 1.0
    # the two genuinely differ on this dup-heavy fixture (the point of H4).
    assert m.paper_recall_at_12 != m.paper_recall_at_12_distinct


def test_distinct_vs_chunk_recall_helpers():
    # 14 chunks: A repeated, then a tail of distinct papers. chunk-@12 cuts CHUNKS; distinct-@12
    # cuts distinct PAPERS, so it can reach papers that chunk-@12's slot budget excludes.
    chunks = [_chunk("A")] * 11 + [_chunk("B"), _chunk("C"), _chunk("D")]
    data = {"chunks": chunks}
    # chunk-@12 = first 12 chunks deduped = {A, B}
    assert set(bb.retrieved_papers_from_chunks(data)) == {"A", "B"}
    # distinct-@12 = first 12 DISTINCT papers = {A, B, C, D} (only 4 distinct exist)
    assert bb.distinct_papers_from_chunks(data) == ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# H7 gaming scenario: the drill's slice20-q5 case — recall 0 / nugget 1.0 from substitute papers
# --------------------------------------------------------------------------- #
def test_gaming_substance_from_wrong_papers_caught_by_gold_citation_gate():
    """Reproduce the drill's slice20-q5 gaming case: a variant answers with the RIGHT substance
    (nugget_recall 1.0) but sourced from the WRONG papers (gold paper not retrieved, recall 0) and
    cites those substitutes — so gold_citation_recall collapses. The hardened gate must catch this
    'substance from the wrong papers' rather than reward the high nugget score.

    Build the backbone scoring directly from a result/gold pair to show the deterministic signals,
    then confirm verdict()'s gold_citation_recall gate vetoes a headline that the nugget score
    alone would have waved through.
    """
    # --- deterministic backbone on the gaming question -----------------------
    # gold wants paper G; the variant retrieved/cites SUBSTITUTE papers S1,S2 (topically adjacent),
    # never G. recall@12 (distinct) = 0 (G absent); gold_citation_recall = 0 (cited S1,S2 ∉ gold).
    gaming_result = {
        "qid": "slice20-q5", "data": {"chunks": [_chunk("S1"), _chunk("S2")]},
        "cited_papers": ["S1", "S2"], "kb_coverage": "thin", "answer": "right idea [S1] [S2].",
    }
    gold = {"qid": "slice20-q5", "gold_keys": ["G"], "nuggets": ["n1"],
            "per_paper_relevance": {"G": "directly-answering"}, "expected_coverage_band": "thin"}
    m = bb.compute_question_metrics(gaming_result, gold)
    assert m.paper_recall_at_12_distinct == 0.0   # gold paper never retrieved
    assert m.gold_citation_recall == 0.0          # cited the substitutes, not gold

    # --- verdict: harmonic of (recall=0 via headline_table) already tanks this Q's headline, AND
    # the gold_citation_recall gate independently catches the corpus-level drop. Show a corpus
    # where the variant LOOKS like a nugget win but gold-cite collapses on the gamed slice.
    qs = [f"q{i}" for i in range(8)]
    # baseline: solid retrieval + gold citations everywhere.
    base = _tables(
        recall={q: 0.7 for q in qs}, nugget={q: 0.7 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs},
        goldcite={q: 0.9 for q in qs})
    # variant: nugget_recall climbs (answers sound great) but gold_citation_recall collapses —
    # substance sourced from the wrong papers, the slice20-q5 disease spread corpus-wide.
    var = _tables(
        recall={q: 0.7 for q in qs}, nugget={q: 0.95 for q in qs},   # nugget up → headline up
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs},
        goldcite={q: 0.4 for q in qs})           # gold citations collapsed → wrong-paper substance
    v = H.verdict(base, var, trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_VKW)
    assert v["variant_headline"] > v["baseline_headline"]   # nugget lift makes headline look like a win
    assert v["gates"]["gold_citation_recall"]["regressed"] is True   # ...but caught
    assert v["win"] is False


# --------------------------------------------------------------------------- #
# H8 (2026-06-03 reconcile drill): the §8 fix still had FOUR fail-OPEN paths in verdict() — a
# 'win' could slip through when (a) the mandatory trap gate's args were omitted, (b) a red-line
# gate's variant table was empty, (c) the variant headline table was empty (compare_metric RAISED),
# or (d) the noise floor was passed as 0. These pin the fail-CLOSED hardening: an un-evaluable
# gate vetoes the win exactly like a regressed one; a degenerate input is a clean non-win, never
# a crash and never a silent win.
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


def _winning_tables(qs):
    """A clean, broad headline win (every Q moves 0.4→0.8) with all per-Q gates non-regressing.
    Used as the 'looks like a win' base for the H8 fail-open scenarios — every gate that CAN be
    evaluated is clean, so only the fail-open hole under test can decide win."""
    base = _tables(
        recall={q: 0.4 for q in qs}, nugget={q: 0.4 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs})
    var = _tables(
        recall={q: 0.8 for q in qs}, nugget={q: 0.8 for q in qs},
        halluc={q: 0.0 for q in qs}, overconf={q: 0.0 for q in qs}, citeprec={q: 0.9 for q in qs})
    return base, var


# ---- H8a: the trap gate is MANDATORY — omitting its args must NOT silently disable it ---------
def test_verdict_trap_gate_mandatory_omitted_args_fail_closed():
    # drill 4b: headline clearly up & clean, but trap_refusal_* OMITTED. The OLD code skipped the
    # trap gate and returned win=True — silently disabling the gate that protects the measured TOP
    # weakness (~0.58 refusal honesty). Now the gate is recorded NOT EVALUABLE → fail-closed.
    qs = [f"q{i}" for i in range(8)]
    base, var = _winning_tables(qs)
    v = H.verdict(base, var, **_VKW)            # NOTE: no trap_refusal_baseline / _variant
    assert v["headline"]["improved"] is True     # the headline really is up...
    assert v["passes_noise_floor"] is True
    assert v["survives_jackknife"] is True
    g = v["gates"]["trap_correct_refusal_rate"]
    assert g["evaluable"] is False               # the gate could not be evaluated...
    assert g["regressed"] is True                # ...so it fails CLOSED (counts as a regression)
    assert v["all_gates_evaluable"] is False
    assert v["win"] is False                     # ...and the win is vetoed (was True in the hole)


def test_verdict_trap_gate_one_arg_missing_also_fail_closed():
    # only the baseline scalar supplied → still not evaluable → fail-closed.
    qs = [f"q{i}" for i in range(8)]
    base, var = _winning_tables(qs)
    v = H.verdict(base, var, trap_refusal_baseline=0.58, **_VKW)  # variant scalar missing
    assert v["gates"]["trap_correct_refusal_rate"]["evaluable"] is False
    assert v["win"] is False


# ---- H8b: an empty/absent red-line gate table fails CLOSED (reachable from production) ---------
def test_absolute_gate_empty_variant_table_fails_closed():
    # The H8b unit: a poison gate whose VARIANT table is empty must NOT pass. (Production path:
    # hallucinated_rate is None for any answerable Q with no inline [key], so a variant that stops
    # emitting inline cites empties the whole table — the deterministic hallucination red-line would
    # go inert EXACTLY when the variant changed its citation behaviour.)
    base = {f"q{i}": 0.0 for i in range(8)}
    g = H.absolute_gate("hallucinated_rate", False, base, {})  # variant table empty
    assert g["n"] == 0
    assert g["evaluable"] is False
    assert g["regressed"] is True                # fail-closed, NOT the old regressed=False


def test_paired_gate_empty_variant_table_fails_closed():
    base = {f"q{i}": 0.9 for i in range(8)}
    g = H.gate_regressed("faithfulness", True, base, {}, **_KW)  # no common qids
    assert g["evaluable"] is False
    assert g["regressed"] is True


def test_verdict_empty_hallucinated_rate_table_vetoes_win():
    # drill probe B: deleting the variant's hallucinated_rate table (the variant stopped emitting
    # inline cites) used to leave the poison gate inert → win=True. Now it fails closed.
    qs = [f"q{i}" for i in range(8)]
    base, var = _winning_tables(qs)
    var["backbone"]["hallucinated_rate"] = {}    # variant emits no inline cites → table empty
    v = H.verdict(base, var, trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_VKW)
    assert v["headline"]["improved"] is True
    g = v["gates"]["hallucinated_rate"]
    assert g["evaluable"] is False
    assert g["regressed"] is True
    assert v["all_gates_evaluable"] is False
    assert v["win"] is False


def test_verdict_empty_gold_citation_recall_table_vetoes_win():
    # drill probe A: deleting the variant's gold_citation_recall table used to leave that gate inert.
    qs = [f"q{i}" for i in range(8)]
    base, var = _winning_tables(qs)
    var["backbone"]["gold_citation_recall"] = {}
    v = H.verdict(base, var, trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_VKW)
    assert v["gates"]["gold_citation_recall"]["evaluable"] is False
    assert v["gates"]["gold_citation_recall"]["regressed"] is True
    assert v["win"] is False


# ---- H8c: an empty variant headline table → structured non-win, NEVER an uncaught raise --------
def test_verdict_empty_variant_headline_returns_clean_non_win_not_raise():
    # drill7: wiping the variant judge source makes headline_table(variant) == {} → the old code let
    # paired_deltas raise ValueError('no common qids') out of verdict(), crashing a loop scoring many
    # variants. Now verdict() returns a structured win=False with a note.
    qs = [f"q{i}" for i in range(8)]
    base, _var = _winning_tables(qs)
    var = {"backbone": {}, "judge": {}}          # variant produced nothing → empty headline table
    v = H.verdict(base, var, trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_VKW)
    assert v["win"] is False
    assert v["headline"] is None
    assert "empty" in v["note"]
    assert v["all_gates_evaluable"] is False


# ---- H8d: the noise floor must be POSITIVE — a 0 floor disables (ii) and is rejected -----------
def test_verdict_rejects_zero_noise_floor_by_default():
    # noise_floor_run_mean_sd=0 → k*0==0 → any positive mean_delta clears (ii): the sub-noise-win
    # hole H3(ii) closed re-opens. verdict() must REFUSE a <=0 floor unless explicitly opted out.
    qs = [f"q{i}" for i in range(8)]
    base, var = _winning_tables(qs)
    with pytest.raises(ValueError):
        H.verdict(base, var, noise_floor_run_mean_sd=0.0,
                  trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_KW)


def test_verdict_rejects_negative_noise_floor():
    qs = [f"q{i}" for i in range(8)]
    base, var = _winning_tables(qs)
    with pytest.raises(ValueError):
        H.verdict(base, var, noise_floor_run_mean_sd=-0.01,
                  trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_KW)


def test_verdict_zero_noise_floor_allowed_only_with_explicit_optout():
    # the test-only opt-out lets a test isolate the OTHER gates with a 0 floor (what _VKW does).
    qs = [f"q{i}" for i in range(8)]
    base, var = _winning_tables(qs)
    v = H.verdict(base, var, noise_floor_run_mean_sd=0.0, allow_zero_noise_floor=True,
                  trap_refusal_baseline=0.5, trap_refusal_variant=0.75, **_KW)
    assert v["win"] is True                       # with the opt-out, a clean broad win still wins


def test_canonical_noise_floor_constant_pinned():
    # the canonical baseline floor lives in ONE place so production can't pass 0. Updated to the
    # 49-Q grown-gold 3-run wobble (BASELINE.md 2026-06-03: 0.0127; was 0.0075 on the 25-Q set).
    assert H.BASELINE_NOISE_FLOOR_RUN_MEAN_SD == 0.0127


def test_verdict_uses_pinned_floor_end_to_end():
    # a clean broad win evaluated against the PINNED production floor (0.0127) still wins (the lift
    # 0.4→0.8 is far above the floor), and every gate is evaluable.
    qs = [f"q{i}" for i in range(8)]
    base, var = _winning_tables(qs)
    v = H.verdict(base, var, noise_floor_run_mean_sd=H.BASELINE_NOISE_FLOOR_RUN_MEAN_SD,
                  trap_refusal_baseline=0.5, trap_refusal_variant=0.75, **_KW)
    assert v["all_gates_evaluable"] is True
    assert v["win"] is True


# --------------------------------------------------------------------------- #
# R-tail 1 (reconcile drill 2026-06-03): the H8-hardened metric was structurally UN-WINNABLE on its
# own corpus. verdict()(i)/(iii) required the headline paired win to clear, and survive dropping
# each question under, the discrete sign-flip p<0.05 — whose floor is set ONLY by the number of
# MOVERS (3→0.25, 5→0.0625, 6→0.0312), independent of effect size. BASELINE.md documents 18/21 Qs
# saturated at recall 1.0 (only ~3 movable), so a GENUINE broad upstream win lifting the 3 movable
# Qs could never pass even (i). These pin the redesign: significance on the MOVABLE subset
# (ci_low>0, n_movers>=2), continuous leave-one-MOVER-out jackknife, noise floor carries real-vs-
# jitter. A genuine few-mover win now WINS; gaming / sub-noise / regression still lose.
# --------------------------------------------------------------------------- #
def _saturated_landscape_tables(mover_val):
    """Realistic 18-saturated / 3-movable headline landscape (BASELINE.md). 18 Qs sit at headline
    1.0 (recall=nugget=1.0) in both runs; 3 movers carry recall=nugget=mover_val so the per-qid
    headline = harmonic(mover_val, mover_val) = mover_val. All red-line gates clean.
    """
    sat = [f"s{i}" for i in range(18)]
    mov = ["m0", "m1", "m2"]
    qids = sat + mov
    recall = {q: 1.0 for q in sat}
    nug = {q: 1.0 for q in sat}
    for q in mov:
        recall[q] = mover_val
        nug[q] = mover_val
    return _tables(
        recall=recall, nugget=nug,
        halluc={q: 0.0 for q in qids}, overconf={q: 0.0 for q in qids},
        citeprec={q: 0.9 for q in qids})


def test_verdict_genuine_few_mover_win_passes_on_saturated_landscape():
    # R-tail 1 end-to-end: 18 saturated + 3 genuine movers 0.3→0.95. The OLD (i)/(iii) made this
    # impossible (3 movers → p floored at 0.25, full-sample ci_low pinned to 0). The redesign lets
    # it WIN, while head['full_sample_improved'] is False (the OLD criterion would have rejected it).
    base = _saturated_landscape_tables(0.30)
    var = _saturated_landscape_tables(0.95)
    v = H.verdict(base, var, noise_floor_run_mean_sd=H.BASELINE_NOISE_FLOOR_RUN_MEAN_SD,
                  trap_refusal_baseline=0.58, trap_refusal_variant=0.58, **_KW)
    assert v["n_movers"] == 3
    assert v["headline"]["improved"] is True             # movers-based significance
    assert v["headline"]["full_sample_improved"] is False  # the inert old criterion would reject it
    assert v["passes_noise_floor"] is True
    assert v["survives_jackknife"] is True
    assert v["all_gates_evaluable"] is True
    assert v["win"] is True                               # a TRUE win can finally register (R-tail 1)


def test_verdict_sub_noise_few_mover_win_is_no_win_on_saturated_landscape():
    # Same landscape but the 3 movers lift only 0.50→0.505 (well inside the run-to-run wobble). The
    # movers ci_low may be >0, but the FULL-SET mean_delta is below the noise floor → not a win.
    base = _saturated_landscape_tables(0.500)
    var = _saturated_landscape_tables(0.505)
    v = H.verdict(base, var, noise_floor_run_mean_sd=H.BASELINE_NOISE_FLOOR_RUN_MEAN_SD,
                  trap_refusal_baseline=0.58, trap_refusal_variant=0.58, **_KW)
    assert v["n_movers"] == 3
    assert v["passes_noise_floor"] is False               # the lift is sub-noise
    assert v["win"] is False


def test_verdict_single_mover_win_on_saturated_landscape_is_no_win():
    # 18 saturated + exactly ONE mover (0.1→0.95). n_movers<2 → not robust (single-Q-driven), and
    # the movers-based significance also requires >=2 movers → improved False → no win.
    base = _saturated_landscape_tables(0.30)
    var = _saturated_landscape_tables(0.30)
    var["backbone"]["paper_recall_at_12_distinct"]["m0"] = 0.95   # only m0 moves in the variant
    var["judge"]["nugget_recall"]["m0"] = 0.95
    v = H.verdict(base, var, noise_floor_run_mean_sd=H.BASELINE_NOISE_FLOOR_RUN_MEAN_SD,
                  trap_refusal_baseline=0.58, trap_refusal_variant=0.58, **_KW)
    assert v["n_movers"] == 1
    assert v["headline"]["improved"] is False
    assert v["survives_jackknife"] is False
    assert v["win"] is False


# --------------------------------------------------------------------------- #
# R-tail 2 (reconcile drill 2026-06-03): PARTIAL qid-drop evades the per-question red-line gates —
# H8b only closed the ALL-EMPTY-table case. A variant whose gate table is a strict SUBSET of the
# baseline's (the poison qid simply ABSENT, not the whole table empty) used to be scored only on the
# qid INTERSECTION → it read clean. Reachable from production: stats.per_question_scalars writes a
# row for every RESULT ROW, so a qid is absent only when the variant produced NO result for it (the
# query errored / was skipped on the hard Q — exactly the Qs most likely to be over-confident or
# hallucinate). These pin the fix: a gate whose variant is missing qids the baseline scored fails
# CLOSED (evaluable=False, regressed=True), not silently scored on the intersection.
# --------------------------------------------------------------------------- #
def test_absolute_gate_partial_qid_drop_fails_closed():
    # baseline scored q0..q7; variant is missing q0 (the over-confident poison Q it crashed on).
    base = {f"q{i}": (1.0 if i == 0 else 0.0) for i in range(8)}
    var = {f"q{i}": 0.0 for i in range(1, 8)}     # q0 absent — variant produced no result for it
    g = H.absolute_gate("over_confidence_rate", False, base, var)
    assert g["evaluable"] is False                # not fully evaluable (variant dropped a scored qid)
    assert g["regressed"] is True                 # fail-closed
    assert g["n_missing_qids"] == 1
    assert "q0" in g["missing_qids"]


def test_paired_gate_partial_qid_drop_fails_closed():
    base = {f"q{i}": 0.9 for i in range(8)}
    var = {f"q{i}": 0.9 for i in range(1, 8)}      # missing q0
    g = H.gate_regressed("faithfulness", True, base, var, **_KW)
    assert g["evaluable"] is False
    assert g["regressed"] is True
    assert g["n_missing_qids"] == 1


def test_verdict_partial_drop_of_poison_qid_vetoes_win():
    # THE R-tail-2 end-to-end gaming case: a broad headline win, but the variant's
    # hallucinated_rate table is MISSING exactly q0 — the poisoned Q. The baseline scored q0 as
    # hallucinating (1.0); a variant that crashed on that hard Q omits it, so the OLD code
    # scored the clean 7-qid intersection and returned win=True. Now the gate fails CLOSED.
    # (Fixture moved over_confidence_rate→hallucinated_rate when the former was de-gated on the
    # full corpus — 2026-06-08 drill D1; the fail-closed mechanics under test are unchanged.)
    qs = [f"q{i}" for i in range(8)]
    base, var = _winning_tables(qs)
    base["backbone"]["hallucinated_rate"] = {q: (1.0 if q == "q0" else 0.0) for q in qs}
    # variant dropped q0 from the hallucinated_rate table (errored on the hard, poisoned Q):
    var["backbone"]["hallucinated_rate"] = {q: 0.0 for q in qs if q != "q0"}
    v = H.verdict(base, var, trap_refusal_baseline=0.5, trap_refusal_variant=0.5, **_VKW)
    assert v["headline"]["improved"] is True       # the headline really is up on the survivors...
    g = v["gates"]["hallucinated_rate"]
    assert g["evaluable"] is False                 # ...but the poison gate can't be fully evaluated
    assert g["regressed"] is True
    assert v["all_gates_evaluable"] is False
    assert v["win"] is False                        # vetoed (was a silent WIN before R-tail 2)
