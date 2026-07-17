"""KS downstream HEADLINE score + RED-LINE gates (SDD §6.10 E/G) — the single-number-with-safety
layer on top of the metric vector. Settled 2026-06-02; HARDENED by the §8 adversarial drill
2026-06-02 (H1-H7: the original gates were near-inert for poison rates → a 'winning' variant
could make KS worse). RECONCILE-HARDENED 2026-06-03 (H8): the §8 fix still had four fail-OPEN
paths — verdict() could declare a win when (a) the mandatory trap gate's scalar args were omitted,
(b) a red-line gate's variant table was empty (reachable from production: a variant that stops
emitting inline cites empties hallucinated_rate's table → the poison gate went inert), (c) the
variant headline table was empty (compare_metric RAISED rather than returning a clean non-win),
or (d) the noise floor was passed as 0 (k*0==0 disables the over-noise check). This module now
truly fails CLOSED: an un-evaluable gate vetoes the win exactly like a regressed one.

WHY a headline AT ALL + WHY it can't be a naive blend:
  The 6-metric vector measures different diseases that can move in OPPOSITE directions (a cleaner
  graph can answer worse). A naive average would let a gain on one axis MASK a poison regression
  on another (the paper-library C/P→F lesson: one blended number lies). So:

  HEADLINE = per-question harmonic-mean(paper_recall@12-DISTINCT, nugget_recall), averaged over
  the NON-TRAP questions. Harmonic (not arithmetic) so a lopsided answer — found the right papers
  but missed the substance, or vice-versa — tanks it (BOTH must be decent).
    · paper_recall_at_12_distinct (backbone) = retrieval surfaced the right papers, scored over
        the first 12 DISTINCT papers in reranked order (H4: chunk-based @12 is dup-dominated, so
        a de-dup variant could inflate it mechanically; distinct-paper recall is denominator-fair)
    · nugget_recall              (Claude-judge overlay) = the answer conveyed the right substance
  These two ARE the quality core and point the same way; everything else is dashboard or gate.

  RED-LINE GATES — NOT blended into the headline; kept as must-not-regress checks so a headline
  gain can NEVER mask a poison regression. TWO kinds of gate, TWO kinds of check (H1):

  (a) ABSOLUTE gates on the DETERMINISTIC rates (noise≈0 → no paired test):
    · hallucinated_rate    (backbone, →0)  a [paper_key] in prose not in the retrieved chunks
    · over_confidence_rate (backbone, →0)  kb_coverage='strong' while recall==0
    · gold_citation_recall (backbone, ↑)   cited_papers landed on gold (H2c)
    · trap_correct_refusal_rate (judge guardrail rate over the 4 traps, ↑)  refusal honesty (H2a)
   These are pure set-math / a rate over 4 traps → run-to-run noise ~0. The n=21 paired sign-flip
   permutation is NEAR-INERT for such rates (the original H1 hole), so we DON'T paired-test them:
   a gate FAILS iff the variant's MEAN is worse than baseline's MEAN by more than a tiny epsilon
   (default 0.0), regardless of CI/p — see absolute_gate().

  (b) PAIRED gates on the JUDGE-NOISY metrics (seed/synth jitter → keep the paired test) but ALSO
   require an absolute floor (a stable small poison can sit under a paired test):
    · citation_support_precision (judge, ↑)  each cited key actually entails its sentence
    · faithfulness               (judge, ↑)  answer claims are backed by retrieved context (H2b)

  DECISION (variant vs baseline) — verdict() requires ALL of (H3; R-tail-1/2 redesign 2026-06-03):
    (i)   headline BCa ci_low>0 over the MOVABLE subset (|delta|>eps_move) AND n_movers>=2 — NOT
          the old full-sample "ci_low>0 ∧ p<0.05". 18/21 Qs sit saturated at headline 1.0 in BOTH
          runs, which pins the full-sample BCa ci_low to 0 and lifts the sign-flip p to its discrete
          mover-count floor (3 movers → 0.25, ≥5 → never <0.05) regardless of effect size — so a
          genuine broad win that truly lifts the ~3 movable Qs could NEVER clear the old (i). The
          significance is judged on the movers (effect-size-sensitive ci_low); the discrete p is
          reported but NOT gated. (Same principle by which traps are excluded from the paired tables.)
    (ii)  headline FULL-SET mean_delta > k * noise_floor.run_mean_sd  (k default 1.0; noise floor is
          BAKED IN as a verdict() arg — carries the "real, not jitter" burden the inert p can't)
    (iii) JACKKNIFE: the win survives dropping ANY single MOVABLE question, with a CONTINUOUS per-
          drop criterion (remaining movers stay net-positive). Dropping a saturated tie is a no-op;
          requiring survival after dropping a mover under the discrete p was un-winnable (R-tail 1).
    (iv)  no red-line gate regresses (absolute gates + paired-with-floor gates + trap gate), every
          gate is EVALUABLE, and the variant COVERS the baseline's scored qid set on each per-
          question gate (R-tail 2: a partial qid-drop — the poison qid simply absent — fails CLOSED,
          not just the all-empty-table case H8b).
  Pure functions over the SAME per-question tables stats.py builds → the headline rides the
  identical paired-difference machinery as every other metric (it is just one more table).
"""
from __future__ import annotations

from typing import Any

from papervault.eval import stats as _stats

# Canonical baseline noise floor — the 3-run headline run_mean_sd on the GROWN 49-Q gold (BASELINE.md
# 2026-06-03: 0.0127; was 0.0075 on the near-saturated 25-Q set — the 18 hard multi-paper Qs add real
# run-to-run synthesis variance, so the honest floor rose). Pinned HERE so the production caller can't
# accidentally pass 0 (H8d): verdict() rejects a <=0 floor unless allow_zero_noise_floor=True (a
# test-only opt-out). k*0 == 0 would let any positive mean_delta clear (ii) — re-opening hole H3(ii).
BASELINE_NOISE_FLOOR_RUN_MEAN_SD = 0.0127

# FULL-CORPUS frame noise floor (2026-06-11): 3 identical fullcorpus baseline repeats
# (fullcorpus_baseline/_r2/_r3, gold_v2, tightened contract, Opus judge, 1 seed/run) →
# per-run headline [0.4261, 0.4339, 0.4284], run_mean_sd = 0.0040 (range 0.0078).
# Much tighter than the retired test100 floor (0.0127): the mid-range (non-saturated)
# question landscape produces less run-to-run synthesis swing. USE THIS for all variant
# verdicts on the full-corpus frame (the 0.0127 above is the RETIRED test100 frame).
FULLCORPUS_NOISE_FLOOR_RUN_MEAN_SD = 0.0040

# LONG-CONTEXT frame noise floor (2026-06-14): the lctx config (multiquery + V-SR + MAX_TOTAL_TOKENS
# =300000 + facet-rerank-off, serving 60 chunks) is the NEW baseline. Its 3 identical repeats
# (lctx_r1/r2/r3, headline=harm(@served,nugget)) gave per-run [0.5868, 0.5832, 0.6128],
# run_mean_sd = 0.0132 — 5× the short-context floor (longer answers over more chunks = more synth
# run-to-run swing). USE THIS for verdicts on the long-context frame (verdict_lctx.py).
FULLCORPUS_LONGCTX_NOISE_FLOOR_RUN_MEAN_SD = 0.0132

# Headline retrieval term. Default = the legacy first-12-distinct cut (short-context frame, H4).
# The long-context frame passes recall_key="paper_recall_at_served_distinct" (coverage over ALL
# distinct papers served to synth) — see 2026-06-14-metric-adjustment-decision.md. Identical to @12
# whenever served<=12, so the short-context arc is unchanged.
DEFAULT_HEADLINE_RECALL_KEY = "paper_recall_at_12_distinct"

# ABSOLUTE gates: metric -> (table source, higher_is_better). Deterministic rates (noise≈0) —
# checked by absolute_gate (mean-vs-mean), NOT the paired permutation (H1). gold_citation_recall
# (H2c) and trap_correct_refusal_rate (H2a, a guardrail rate over the 4 traps) join the two
# original poison rates. trap_correct_refusal_rate is sourced NOT from a per-question table but
# from judge_aggregate.trap_correct_refusal_rate() → verdict() takes its scalar value directly.
ABSOLUTE_GATES: dict[str, tuple[str, bool]] = {
    "hallucinated_rate": ("backbone", False),
    # over_confidence_rate REMOVED as a gate (2026-06-08 full-corpus drill, D1 BLOCKER): on the
    # 227k-entity graph kb_coverage saturates at 'strong' for every query (incl. all 9 off-domain
    # traps), so the metric degenerates to exactly 1 − hit@12 — a dead calibration signal that
    # (a) double-counts retrieval misses the headline already prices (false veto at 1/48 per
    # flipped Q with epsilon=0), and (b) is trivially zeroed by any variant touching the
    # _assess_coverage bins (signal laundering). It remains a dashboard value in the backbone
    # vector. Refusal honesty is gated by the judge's trap_correct_refusal_rate (mandatory,
    # fail-closed in verdict()). Re-add only after a relevance-bearing kb_coverage redesign.
    "gold_citation_recall": ("backbone", True),
}

# PAIRED-WITH-FLOOR gates: judge-noisy metrics (H1 tail) — keep the paired significance test AND
# require an absolute floor (a stable small poison can sit under a paired test). faithfulness is
# the new H2b gate joining citation_support_precision.
PAIRED_GATES: dict[str, tuple[str, bool]] = {
    "citation_support_precision": ("judge", True),
    "faithfulness": ("judge", True),
}

# Back-compat alias (old name) — the union of both gate families' per-question metrics. Kept so
# any external reference to GATES still resolves; verdict() iterates the two dicts explicitly.
GATES = {**ABSOLUTE_GATES, **PAIRED_GATES}


def harmonic(a: float, b: float) -> float:
    """Harmonic mean of two [0,1] scores; 0 if either is 0 (lopsided → tanked)."""
    a = max(0.0, float(a))
    b = max(0.0, float(b))
    s = a + b
    return (2.0 * a * b / s) if s > 0.0 else 0.0


def headline_table(
    backbone_q: dict[str, dict[str, float]],
    judge_q: dict[str, dict[str, float]],
    recall_key: str = DEFAULT_HEADLINE_RECALL_KEY,
) -> dict[str, float]:
    """{qid: harmonic(<recall_key>, nugget_recall)} over qids in BOTH tables.

    recall_key default = paper_recall_at_12_distinct (H4: denominator-fair first-12-DISTINCT recall,
    short-context frame). The long-context frame passes "paper_recall_at_served_distinct" (coverage
    over all distinct papers served to synth). Either way it's a DISTINCT-paper recall (never the
    dup-dominated chunk recall). The intersection is naturally the non-trap answerable set:
    nugget_recall (judge) drops traps and the recall table is None→omitted for traps.
    """
    rec = backbone_q.get(recall_key, {})
    nug = judge_q.get("nugget_recall", {})
    return {q: harmonic(rec[q], nug[q]) for q in (set(rec) & set(nug))}


def headline_mean(table: dict[str, float]) -> float:
    return (sum(table.values()) / len(table)) if table else 0.0


def _means(
    baseline: dict[str, float],
    variant: dict[str, float],
) -> tuple[float, float, int]:
    """(mean_baseline, mean_variant, n) over the qids present in BOTH tables (paired support)."""
    common = sorted(set(baseline) & set(variant))
    if not common:
        return 0.0, 0.0, 0
    mb = sum(baseline[q] for q in common) / len(common)
    mv = sum(variant[q] for q in common) / len(common)
    return mb, mv, len(common)


def absolute_gate(
    metric: str,
    higher_is_better: bool,
    baseline: dict[str, float],
    variant: dict[str, float],
    *,
    epsilon: float = 0.0,
    **_ignored: Any,
) -> dict[str, Any]:
    """ABSOLUTE gate (H1): a DETERMINISTIC-rate gate FAILS iff the variant's MEAN is worse than the
    baseline's MEAN by more than `epsilon` (default 0.0) — regardless of CI/p.

    The deterministic backbone rates (hallucinated_rate, over_confidence_rate, gold_citation_recall)
    have run-to-run noise ~0, so the n=21 paired sign-flip permutation is near-inert for them (the
    H1 hole: a real poison regression in a rate could fail to clear the paired test). Mean-vs-mean
    with a tiny epsilon catches a poison regression the paired test would miss.
      lower-is-better → regressed if (mean_variant - mean_baseline) > epsilon  (rate went UP)
      higher-is-better → regressed if (mean_baseline - mean_variant) > epsilon  (rate went DOWN)
    """
    mb, mv, n = _means(baseline, variant)
    if n == 0:
        # H8b: empty/absent input = NOT EVALUABLE = fail-CLOSED. A red-line poison gate whose
        # variant table is empty (e.g. a variant that stopped emitting inline cites empties the
        # hallucinated_rate table → drill5, reachable from production) must NEVER pass — it goes
        # inert EXACTLY when the variant changed the behaviour the gate guards. Mark it regressed
        # AND not-evaluable so verdict() vetoes the win.
        return {
            "metric": metric, "kind": "absolute", "regressed": True, "evaluable": False,
            "n": 0, "note": "no common qids — gate input empty/absent (fail-closed)",
        }
    # R-tail 2 (reconcile 2026-06-03): a PARTIAL qid-drop also fails CLOSED. If the variant is
    # MISSING qids the baseline scored (set(baseline) - set(variant) non-empty), scoring only the
    # intersection silently drops exactly the hard Qs a crashing/skipping variant didn't answer —
    # which are the Qs most likely to be over-confident / hallucinate. The poison qid simply ABSENT
    # (not the whole table empty) used to read clean. Require the variant to COVER the baseline's
    # qid set; otherwise the gate is not fully evaluable → veto.
    missing = sorted(set(baseline) - set(variant))
    if missing:
        return {
            "metric": metric, "kind": "absolute", "regressed": True, "evaluable": False,
            "n": n, "n_missing_qids": len(missing), "missing_qids": missing[:10],
            "note": "variant missing qids the baseline scored — partial-drop fail-closed (R-tail 2)",
        }
    worse_by = (mv - mb) if not higher_is_better else (mb - mv)
    regressed = worse_by > epsilon
    return {
        "metric": metric, "kind": "absolute", "regressed": bool(regressed), "evaluable": True,
        "mean_baseline": mb, "mean_variant": mv, "mean_delta": mv - mb,
        "worse_by": worse_by, "epsilon": epsilon, "n": n,
    }


def gate_regressed(
    metric: str,
    higher_is_better: bool,
    baseline: dict[str, float],
    variant: dict[str, float],
    *,
    alpha: float = 0.05,
    epsilon: float = 0.0,
    **kw: Any,
) -> dict[str, Any]:
    """A PAIRED-WITH-FLOOR gate (H1 tail) for the JUDGE-noisy metrics.

    REGRESSES iff EITHER:
      (paired) the variant is SIGNIFICANTLY WORSE (whole CI on the bad side, p<alpha) — keeps the
        paired test because judge metrics carry seed/synth jitter;
      OR (absolute floor) the variant's mean is worse than baseline's by more than `epsilon` — a
        stable small poison can sit UNDER the paired test (the paired permutation is weak at n=21),
        so we ALSO require the mean not to drop.
    Uses the same paired machinery; reads compare_metric's CI (reported in original metric units):
      higher-is-better gate → paired-regressed if ci_high < 0 (variant−baseline entirely negative)
      lower-is-better  gate → paired-regressed if ci_low  > 0 (variant−baseline entirely positive)
    """
    if not (set(baseline) & set(variant)):
        # H8b: empty/absent input = NOT EVALUABLE = fail-CLOSED (same rationale as absolute_gate).
        return {
            "metric": metric, "kind": "paired", "regressed": True, "evaluable": False,
            "n": 0, "note": "no common qids — gate input empty/absent (fail-closed)",
        }
    # R-tail 2 (reconcile 2026-06-03): partial qid-drop fails CLOSED (same rationale as
    # absolute_gate — see there). The variant must COVER the baseline's scored qid set.
    missing = sorted(set(baseline) - set(variant))
    if missing:
        return {
            "metric": metric, "kind": "paired", "regressed": True, "evaluable": False,
            "n": len(set(baseline) & set(variant)),
            "n_missing_qids": len(missing), "missing_qids": missing[:10],
            "note": "variant missing qids the baseline scored — partial-drop fail-closed (R-tail 2)",
        }
    res = _stats.compare_metric(metric, baseline, variant, higher_is_better=higher_is_better, alpha=alpha, **kw)
    if higher_is_better:
        paired_regressed = (res.ci_high < 0.0) and (res.p_value < alpha)
        worse_by = res.mean_baseline - res.mean_variant
    else:
        paired_regressed = (res.ci_low > 0.0) and (res.p_value < alpha)
        worse_by = res.mean_variant - res.mean_baseline
    floor_regressed = worse_by > epsilon
    return {
        "metric": metric, "kind": "paired", "regressed": bool(paired_regressed or floor_regressed),
        "evaluable": True,
        "paired_regressed": bool(paired_regressed), "floor_regressed": bool(floor_regressed),
        "mean_baseline": res.mean_baseline, "mean_variant": res.mean_variant,
        "mean_delta": res.mean_delta, "worse_by": worse_by, "epsilon": epsilon,
        "ci_low": res.ci_low, "ci_high": res.ci_high,
        "p_value": res.p_value, "n": res.n,
    }


def run_summary(
    backbone_q: dict[str, dict[str, float]],
    judge_q: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Single-run report: the headline number + the full metric vector (means) + n."""
    h = headline_table(backbone_q, judge_q)
    vector = {m: (sum(t.values()) / len(t)) for m, t in {**backbone_q, **judge_q}.items() if t}
    return {"headline": headline_mean(h), "n_headline": len(h), "vector": vector}


def verdict(
    baseline_tables: dict[str, dict[str, dict[str, float]]],
    variant_tables: dict[str, dict[str, dict[str, float]]],
    *,
    noise_floor_run_mean_sd: float,
    k: float = 1.0,
    epsilon: float = 0.0,
    absolute_epsilons: dict[str, float] | None = None,
    recall_key: str = DEFAULT_HEADLINE_RECALL_KEY,
    trap_refusal_baseline: float | None = None,
    trap_refusal_variant: float | None = None,
    alpha: float = 0.05,
    seed: int = 0,
    allow_zero_noise_floor: bool = False,
    **kw: Any,
) -> dict[str, Any]:
    """Variant-vs-baseline decision (HARDENED, H3; fail-CLOSED reconcile, H8).
    *_tables = {'backbone':{metric:{qid:..}},'judge':{...}}.

    WIN iff ALL of:
      (i)   SIGNIFICANCE on the MOVABLE subset (R-tail 1): BCa ci_low>0 over the questions that
            actually move (|delta|>eps_move) AND n_movers>=2. NOT the old full-sample
            "ci_low>0 ∧ p<0.05" — with 18/21 Qs saturated the full-sample ci_low is pinned to 0
            and the sign-flip p to its mover-count floor (3 movers → 0.25), so a genuine broad win
            could never clear it. The discrete p is reported but not gated (provably inert here).
      (ii)  headline FULL-SET mean_delta > k * noise_floor_run_mean_sd   (noise floor BAKED IN —
            verdict takes run_mean_sd as an arg; carries the "real, not jitter" burden)
      (iii) JACKKNIFE: the win survives dropping ANY single MOVABLE question, continuous per-drop
            criterion (stats.jackknife_headline — leave-one-MOVER-out, R-tail 1 redesign)
      (iv)  no red-line gate regresses AND every gate is EVALUABLE AND the variant COVERS the
            baseline's scored qid set on each per-question gate (R-tail 2 partial-drop fail-closed):
              · ABSOLUTE gates (hallucinated_rate, over_confidence_rate, gold_citation_recall):
                absolute_gate — variant mean worse than baseline mean by > epsilon
              · PAIRED-WITH-FLOOR gates (citation_support_precision, faithfulness): gate_regressed
              · trap_correct_refusal_rate: a guardrail rate over the 4 traps (NOT a per-question
                table) — pass its scalar baseline/variant values; FAILS if it drops below baseline
                by > epsilon.

    FAIL-CLOSED HARDENING (H8 — the reconcile drill found four fail-OPEN paths a 'win' slipped
    through; a gate's strength must NOT live only in an argument the caller can omit):
      · (H8a) trap_correct_refusal_rate is MANDATORY. If EITHER trap_refusal_* is None the trap
        gate is recorded evaluable=False, regressed=True → win CANNOT be True. (This gate protects
        the measured TOP weakness ~0.58 refusal honesty; silently skipping it on a forgotten arg
        is the precise hole H2a existed to close.)
      · (H8b) any configured red-line gate whose variant table is empty/absent fails CLOSED
        (absolute_gate / gate_regressed return evaluable=False, regressed=True) — reachable from
        production (a variant that stops emitting inline cites empties hallucinated_rate's table).
      · (H8c) if the headline table (baseline ∩ variant qids) is empty, return a structured
        win=False with a note instead of letting compare_metric raise 'no common qids'.
      · (H8d) noise_floor_run_mean_sd <= 0 disables (ii); rejected (raise) unless
        allow_zero_noise_floor=True (a test-only opt-out). Production passes the pinned
        BASELINE_NOISE_FLOOR_RUN_MEAN_SD (0.0127, measured on the RETIRED 89-paper/49-Q frame —
        ⚠️ INVALID for full-corpus verdicts until re-measured there: 3 identical fullcorpus
        baseline repeats → run_mean_sd; see the 2026-06-08 drill D3 BLOCKER).

    Returns headline PairedResult + passes_noise_floor + survives_jackknife + each gate's verdict
    (+ evaluable) + all_gates_evaluable + win bool.
    """
    # (H8d) the noise floor must be POSITIVE or the over-noise check (ii) is a no-op.
    if float(noise_floor_run_mean_sd) <= 0.0 and not allow_zero_noise_floor:
        raise ValueError(
            f"noise_floor_run_mean_sd={noise_floor_run_mean_sd!r} <= 0 disables the over-noise "
            f"check (k*0==0 lets any positive mean_delta win). Pass the pinned "
            f"BASELINE_NOISE_FLOOR_RUN_MEAN_SD ({BASELINE_NOISE_FLOOR_RUN_MEAN_SD}), or set "
            f"allow_zero_noise_floor=True only to isolate other gates in a test."
        )

    bh = headline_table(baseline_tables.get("backbone", {}), baseline_tables.get("judge", {}), recall_key)
    vh = headline_table(variant_tables.get("backbone", {}), variant_tables.get("judge", {}), recall_key)

    # (H8c) empty headline support → structured non-win instead of a raised 'no common qids'.
    if not (set(bh) & set(vh)):
        return {
            "headline": None,
            "baseline_headline": headline_mean(bh),
            "variant_headline": headline_mean(vh),
            "passes_noise_floor": False,
            "noise_floor_run_mean_sd": float(noise_floor_run_mean_sd),
            "k": k,
            "survives_jackknife": False,
            "jackknife": {"survives": False, "per_qid": {}, "n_movers": 0, "n": 0,
                          "note": "variant headline table empty / no common qids"},
            "headline_movers": None,
            "n_movers": 0,
            "gates": {},
            "all_gates_evaluable": False,
            "win": False,
            "note": "variant headline table empty / no common qids",
        }

    # Full-sample paired comparison — REPORTED (mean_delta drives the noise floor) but no longer
    # the significance gate: with 18/21 Qs saturated at headline 1.0 the full-sample BCa ci_low is
    # pinned to 0 and the sign-flip p to its mover-count floor (3 movers → 0.25), so a genuine
    # broad win could never clear "ci_low>0 ∧ p<0.05" (R-tail 1, reconcile drill 2026-06-03).
    head = _stats.compare_metric("headline", bh, vh, higher_is_better=True, alpha=alpha, seed=seed, **kw)

    # (i) SIGNIFICANCE on the MOVABLE subset (R-tail 1): the headline really improved iff, over the
    # questions that actually move (|delta|>eps_move), the effect-size-sensitive BCa ci_low>0 AND
    # there are >=2 movers (no single-Q-driven win). The discrete sign-flip p is NOT a hard gate
    # here (provably inert at the few-mover counts this corpus operates at); it stays reported on
    # `head` for transparency. Noise/real is decided by (ii) the noise floor on the FULL-set delta.
    movers = _stats.movable_qids(bh, vh)
    n_movers = len(movers)
    if n_movers >= 2:
        head_movers = _stats.compare_metric(
            "headline_movers", {q: bh[q] for q in movers}, {q: vh[q] for q in movers},
            higher_is_better=True, alpha=alpha, seed=seed, **kw)
        movers_ci_low = float(head_movers.ci_low)
    else:
        head_movers = None
        movers_ci_low = 0.0
    improved = (n_movers >= 2) and (movers_ci_low > 0.0)

    # (ii) noise floor — baked in. The full-set mean_delta must clear the test-retest wobble; this
    # is what rejects a sub-noise lift on the movers (carries the "real, not jitter" burden that the
    # discrete p used to nominally — but inertly — carry).
    passes_noise_floor = head.mean_delta > k * float(noise_floor_run_mean_sd)

    # (iii) jackknife robustness — the win must survive dropping any single MOVABLE question, with
    # a continuous per-drop criterion (R-tail 1 redesign — see stats.jackknife_headline).
    jk = _stats.jackknife_headline(bh, vh, higher_is_better=True, alpha=alpha, seed=seed, **kw)
    survives_jackknife = bool(jk.get("survives"))

    # (iv) gates. absolute_gate / gate_regressed fail CLOSED on empty input (H8b).
    # absolute_epsilons (2026-06-14): a PER-METRIC noise band. The absolute rates are SYNTH-derived
    # (parsed from synth prose / cited_papers), so they carry real run-to-run noise — the ε=0 default
    # over-fired (a 0.0026 hallucinated swing vetoed lctx, vs its 0.016 run-sd). Pass ε = the metric's
    # measured baseline run-sd; falls back to the scalar `epsilon` when a metric isn't in the map.
    abs_eps = absolute_epsilons or {}
    gates: dict[str, Any] = {}
    for m, (src, hib) in ABSOLUTE_GATES.items():
        b = baseline_tables.get(src, {}).get(m, {})
        v = variant_tables.get(src, {}).get(m, {})
        gates[m] = absolute_gate(m, hib, b, v, epsilon=abs_eps.get(m, epsilon))
    for m, (src, hib) in PAIRED_GATES.items():
        b = baseline_tables.get(src, {}).get(m, {})
        v = variant_tables.get(src, {}).get(m, {})
        gates[m] = gate_regressed(m, hib, b, v, alpha=alpha, epsilon=epsilon, seed=seed, **kw)

    # trap_correct_refusal_rate — scalar guardrail rate over the 4 traps (H2a). MANDATORY (H8a):
    # if either scalar is missing the gate is NOT EVALUABLE → fail-CLOSED (regressed=True), so a
    # caller that forgets the args can never silently disable the most important new gate.
    if trap_refusal_baseline is not None and trap_refusal_variant is not None:
        worse_by = float(trap_refusal_baseline) - float(trap_refusal_variant)
        gates["trap_correct_refusal_rate"] = {
            "metric": "trap_correct_refusal_rate", "kind": "absolute_scalar",
            "regressed": bool(worse_by > epsilon), "evaluable": True,
            "mean_baseline": float(trap_refusal_baseline), "mean_variant": float(trap_refusal_variant),
            "worse_by": worse_by, "epsilon": epsilon,
        }
    else:
        gates["trap_correct_refusal_rate"] = {
            "metric": "trap_correct_refusal_rate", "kind": "absolute_scalar",
            "regressed": True, "evaluable": False,
            "note": "trap_refusal_baseline/variant not supplied — MANDATORY gate, fail-closed (H8a)",
        }

    any_gate_regressed = any(g.get("regressed") for g in gates.values())
    all_gates_evaluable = all(g.get("evaluable", False) for g in gates.values())
    headline_dict = head.as_dict()
    # Surface the R-tail-1 significance signal ON the headline dict so callers/tests read one place.
    # `improved` here is the MOVERS-based criterion (i), NOT head.improved (the inert full-sample
    # ci_low>0 ∧ p<0.05) — head.improved stays available on the raw PairedResult fields for audit.
    headline_dict["improved"] = bool(improved)
    headline_dict["full_sample_improved"] = bool(head.improved)
    headline_dict["n_movers"] = n_movers
    headline_dict["movers_ci_low"] = movers_ci_low
    return {
        "headline": headline_dict,
        "headline_movers": (head_movers.as_dict() if head_movers is not None else None),
        "n_movers": n_movers,
        "baseline_headline": headline_mean(bh),
        "variant_headline": headline_mean(vh),
        "passes_noise_floor": bool(passes_noise_floor),
        "noise_floor_run_mean_sd": float(noise_floor_run_mean_sd),
        "k": k,
        "survives_jackknife": survives_jackknife,
        "jackknife": jk,
        "gates": gates,
        "all_gates_evaluable": bool(all_gates_evaluable),
        "win": bool(
            improved
            and passes_noise_floor
            and survives_jackknife
            and not any_gate_regressed
            and all_gates_evaluable
        ),
    }
