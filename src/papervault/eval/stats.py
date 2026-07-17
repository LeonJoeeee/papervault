"""Paired-difference statistics for the downstream eval — the DECISION layer of the ruler.

Methodology (the SETTLED design, learned from the paper-library win-rate lesson):
  * NOT pairwise A/B win-rate. Win-rate flips sign below the noise floor and hides effect size.
  * Instead: absolute per-question metric for baseline AND variant on the SAME fixed 25 Qs →
    per-question deltas d_i = variant_i - baseline_i → summarize the delta distribution with a
    paired bootstrap BCa confidence interval + a sign-flip (paired) permutation p-value.
  * Improvement is declared iff (lower CI bound > 0)  AND  (permutation p < 0.05), for a metric
    where higher is better. For a "lower is better" metric (hallucinated_rate, phantom_rate,
    over_confidence_rate) pass higher_is_better=False — the same machinery runs on negated
    deltas so "improvement" still means a real reduction.
  * Noise floor: run the BASELINE 3-5x and measure the per-metric test-retest spread. The
    deterministic backbone's noise is ~0 (same inputs → same numbers); the judge overlay has
    seed noise. A delta inside the noise floor is not real regardless of the CI.

Pairing is BY QID: deltas are formed only over qids present in BOTH runs. Bootstrap resamples
QUESTIONS (rows), not metrics — the resampling unit is the question, preserving the paired
structure. BCa (bias-corrected and accelerated) is used over the percentile interval because
n=25 is small and the delta distribution can be skewed (recall is bounded in [0,1], many ties).

Pure python + numpy/scipy only (both confirmed installed). Deterministic given a seed → the
permutation/bootstrap parts of the tests pin a seed and assert exact behaviour on toy inputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats as _sps


# --------------------------------------------------------------------------- #
# pairing
# --------------------------------------------------------------------------- #
def paired_deltas(
    baseline: dict[str, float],
    variant: dict[str, float],
) -> tuple[list[str], np.ndarray]:
    """Form per-question deltas (variant - baseline) over the qids present in BOTH dicts.

    baseline/variant map qid -> the metric's per-question scalar for that run. A qid missing
    a value in either run (e.g. a query that errored) is dropped from the pairing (and the
    caller is told via the returned qid list, so n is explicit). Raises if there is no overlap.
    """
    qids = sorted(set(baseline) & set(variant))
    if not qids:
        raise ValueError("no common qids between baseline and variant runs")
    d = np.array([float(variant[q]) - float(baseline[q]) for q in qids], dtype=float)
    return qids, d


# --------------------------------------------------------------------------- #
# paired bootstrap BCa CI
# --------------------------------------------------------------------------- #
def _bca_ci(
    deltas: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """BCa confidence interval for the MEAN of a paired-delta sample.

    Bootstrap resamples the deltas (i.e. resamples questions with replacement — the paired
    structure is already baked into each delta). Bias-correction z0 from the fraction of
    bootstrap means below the observed mean; acceleration a from jackknife skewness. Falls
    back to the percentile interval if the sample is degenerate (all-equal deltas → 0 variance,
    z0/a undefined) so a no-op variant returns a sane CI=(mean,mean) rather than NaN.
    """
    n = len(deltas)
    theta_hat = float(np.mean(deltas))
    if n < 2 or np.allclose(deltas, deltas[0]):
        return theta_hat, theta_hat

    boot = np.array(
        [np.mean(rng.choice(deltas, size=n, replace=True)) for _ in range(n_boot)]
    )
    # bias-correction z0
    prop_less = np.mean(boot < theta_hat)
    prop_less = min(max(prop_less, 1.0 / n_boot), 1.0 - 1.0 / n_boot)  # guard 0/1 → ±inf
    z0 = _sps.norm.ppf(prop_less)

    # acceleration a via jackknife
    jack = np.array([np.mean(np.delete(deltas, i)) for i in range(n)])
    jack_mean = np.mean(jack)
    diff = jack_mean - jack
    denom = 6.0 * (np.sum(diff**2) ** 1.5)
    a = (np.sum(diff**3) / denom) if denom != 0 else 0.0

    z_lo, z_hi = _sps.norm.ppf(alpha / 2), _sps.norm.ppf(1 - alpha / 2)

    def _adj(z: float) -> float:
        return float(_sps.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))

    lo_p, hi_p = _adj(z_lo), _adj(z_hi)
    lo = float(np.quantile(boot, lo_p))
    hi = float(np.quantile(boot, hi_p))
    return lo, hi


# --------------------------------------------------------------------------- #
# sign-flip (paired) permutation p-value
# --------------------------------------------------------------------------- #
def _sign_flip_p(
    deltas: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided paired permutation p-value via sign-flipping.

    Under H0 (no difference) the sign of each paired delta is exchangeable → randomly flip
    signs, recompute the mean, count how often |perm mean| >= |observed mean|. For small n the
    exact 2^n sign assignments are enumerated; otherwise n_perm random sign vectors are drawn.
    The observed assignment is always included (the +1 standard correction) so p is never 0.
    All-zero deltas → p = 1.0 (no evidence of any difference).
    """
    n = len(deltas)
    obs = abs(float(np.mean(deltas)))
    if np.allclose(deltas, 0.0):
        return 1.0

    if n <= 20:  # exact enumeration of all 2^n sign flips
        count, total = 0, 0
        for mask in range(1 << n):
            signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)], dtype=float)
            total += 1
            if abs(float(np.mean(signs * deltas))) >= obs - 1e-12:
                count += 1
        return count / total

    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = np.abs((signs * deltas).mean(axis=1))
    count = int(np.sum(perm_means >= obs - 1e-12))
    return (count + 1) / (n_perm + 1)


# --------------------------------------------------------------------------- #
# top-level paired comparison
# --------------------------------------------------------------------------- #
@dataclass
class PairedResult:
    metric: str
    n: int
    mean_baseline: float
    mean_variant: float
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    higher_is_better: bool
    improved: bool
    qids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


def compare_metric(
    metric: str,
    baseline: dict[str, float],
    variant: dict[str, float],
    *,
    higher_is_better: bool = True,
    alpha: float = 0.05,
    n_boot: int = 10000,
    n_perm: int = 10000,
    seed: int = 0,
) -> PairedResult:
    """Paired comparison of ONE metric: baseline vs variant over common qids.

    `improved` = (CI strictly excludes 0 on the improving side) AND (p < alpha). For a
    higher-is-better metric that's ci_low > 0; for lower-is-better the deltas are negated
    before the CI/p so the SAME "ci_low > 0 and p<alpha" rule reads as "a real reduction".
    The reported ci_low/ci_high/mean_delta are always in the ORIGINAL metric direction (we
    un-negate them) so a human reads them naturally.
    """
    qids, d = paired_deltas(baseline, variant)
    rng = np.random.default_rng(seed)

    # Work on the "improvement-positive" orientation for the decision; report in original units.
    work = d if higher_is_better else -d
    lo_w, hi_w = _bca_ci(work, n_boot=n_boot, alpha=alpha, rng=rng)
    p = _sign_flip_p(work, n_perm=n_perm, rng=rng)

    if higher_is_better:
        ci_low, ci_high = lo_w, hi_w
    else:
        ci_low, ci_high = -hi_w, -lo_w  # map back to original direction

    improved = (lo_w > 0.0) and (p < alpha)

    mb = float(np.mean([baseline[q] for q in qids]))
    mv = float(np.mean([variant[q] for q in qids]))
    return PairedResult(
        metric=metric, n=len(qids),
        mean_baseline=mb, mean_variant=mv, mean_delta=float(np.mean(d)),
        ci_low=ci_low, ci_high=ci_high, p_value=p,
        higher_is_better=higher_is_better, improved=improved, qids=qids,
    )


# --------------------------------------------------------------------------- #
# movable (non-saturated) questions of a paired headline comparison
# --------------------------------------------------------------------------- #
# Default |delta| threshold below which a question is a SATURATED TIE (carries no signal). The
# headline is in [0,1]; a per-qid delta under 1e-9 is numerically a no-op (both runs identical
# on that Q, e.g. both saturated at recall 1.0). Movable = |variant - baseline| > eps_move.
EPS_MOVE = 1e-9


def movable_qids(
    baseline_tbl: dict[str, float],
    variant_tbl: dict[str, float],
    *,
    eps_move: float = EPS_MOVE,
) -> list[str]:
    """The common qids where the headline actually MOVES (|variant - baseline| > eps_move).

    The 18-of-21 saturated landscape (BASELINE.md) means most answerable Qs sit at headline 1.0
    in BOTH runs → delta exactly 0 → a SATURATED TIE that carries no signal. Including those ties
    in the paired bootstrap/permutation only dilutes effect-size, pins the BCa ci_low to 0 (the
    bootstrap frequently resamples a tie-only subset whose mean delta is 0), and lifts the discrete
    sign-flip p to its mover-count floor (3 movers → 0.25, ≥5 → never <0.05) REGARDLESS of how big
    the genuine lift on the movers is. The R-tail-1 redesign (reconcile drill 2026-06-03) therefore
    runs the headline significance test on the MOVABLE subset — the exact same principle by which
    judge_aggregate / backbone already exclude always-tied trap rows from the paired comparison.
    """
    return [q for q in sorted(set(baseline_tbl) & set(variant_tbl))
            if abs(float(variant_tbl[q]) - float(baseline_tbl[q])) > eps_move]


def _movers_positive(
    baseline_tbl: dict[str, float],
    variant_tbl: dict[str, float],
    movers: list[str],
    *,
    higher_is_better: bool,
    alpha: float,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> bool:
    """Continuous robustness criterion on a set of MOVERS (R-tail-1): the win is real on this
    subset iff the paired delta is net-positive in the improving direction.

    With >=2 movers we use the effect-size-sensitive BCa ci_low (compare_metric over the movers):
    a real lift has ci_low>0; the discrete sign-flip p is NOT used here (it is provably inert at
    the few-mover counts this whole layer operates at). With exactly 1 mover the BCa CI degenerates
    (n=1) → fall back to the plain sign check (the single mover's delta points the improving way).
    """
    if not movers:
        return False
    if len(movers) == 1:
        q = movers[0]
        d = float(variant_tbl[q]) - float(baseline_tbl[q])
        return (d > 0.0) if higher_is_better else (d < 0.0)
    b = {q: baseline_tbl[q] for q in movers}
    v = {q: variant_tbl[q] for q in movers}
    res = compare_metric(
        "headline_movers", b, v,
        higher_is_better=higher_is_better, alpha=alpha,
        n_boot=n_boot, n_perm=n_perm, seed=seed,
    )
    return bool(res.ci_low > 0.0)


# --------------------------------------------------------------------------- #
# jackknife robustness of a HEADLINE paired win (leave-one-MOVER-out, continuous)
# --------------------------------------------------------------------------- #
def jackknife_headline(
    baseline_tbl: dict[str, float],
    variant_tbl: dict[str, float],
    *,
    higher_is_better: bool = True,
    alpha: float = 0.05,
    n_boot: int = 10000,
    n_perm: int = 10000,
    seed: int = 0,
    eps_move: float = EPS_MOVE,
) -> dict[str, Any]:
    """Does the HEADLINE paired win survive dropping ANY single MOVABLE question? (R-tail-1 redesign)

    `baseline_tbl`/`variant_tbl` are the per-qid HEADLINE scalar tables (the harmonic-combined
    headline value per qid, e.g. headline.headline_table output).

    REDESIGN (reconcile drill 2026-06-03): the original jackknife dropped EVERY one of the 21 qids
    and required each leave-one-out to independently re-clear `ci_low>0 ∧ p<0.05`. That made the
    metric STRUCTURALLY UN-WINNABLE on its own corpus: the sign-flip p has a discrete floor set only
    by the number of MOVERS (6 movers → full-sample p=0.0312 passes, but every leave-one-out drops
    to 5 movers → p=0.0625 → fails; you'd need 8 movers, while BASELINE.md documents only ~3 are
    movable). A genuine broad upstream win that truly lifts the 3 movable Qs could NEVER survive.
    Two changes decouple robustness from that discrete small-n p:
      (1) DROP ONLY MOVERS. Dropping a saturated tie (|delta|<=eps_move) is a provable no-op — it
          is not in the movable subset the win rests on, so its removal cannot change the result.
          Requiring survival after dropping the very Qs that carry the signal is what was impossible.
      (2) CONTINUOUS criterion per drop. After dropping a mover, the remaining movers must stay
          net-positive (BCa ci_low>0 with >=2 left, or a sign check with exactly 1 left) — NOT
          re-clear the discrete p, which is the wrong instrument at these mover counts.

    Returns {survives, per_qid: {dropped_mover_qid: survived_bool}, n_movers, n}. `survives` is True
    iff there are >=2 movers AND every leave-one-mover-out is still net-positive. n_movers<2 → a
    single-question-driven win → survives False (the slice20-q5 single-Q gaming case).
    """
    common = sorted(set(baseline_tbl) & set(variant_tbl))
    movers = movable_qids(baseline_tbl, variant_tbl, eps_move=eps_move)
    n_movers = len(movers)
    if n_movers < 2:
        return {
            "survives": False, "per_qid": {}, "n_movers": n_movers, "n": len(common),
            "note": "fewer than 2 movable questions — a single-Q-driven win is not robust",
        }
    per_qid: dict[str, bool] = {}
    for drop in movers:
        remaining = [q for q in movers if q != drop]
        per_qid[drop] = _movers_positive(
            baseline_tbl, variant_tbl, remaining,
            higher_is_better=higher_is_better, alpha=alpha,
            n_boot=n_boot, n_perm=n_perm, seed=seed,
        )
    return {
        "survives": all(per_qid.values()),
        "per_qid": per_qid,
        "n_movers": n_movers,
        "n": len(common),
    }


# --------------------------------------------------------------------------- #
# noise floor from test-retest
# --------------------------------------------------------------------------- #
def noise_floor(
    runs: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Test-retest noise floor for ONE metric, from 3-5 repeats of the SAME (baseline) config.

    `runs` is a list of qid->scalar dicts (one per repeat). For each qid we take the spread of
    its value across repeats; the floor is summarized at the RUN level (the unit a delta is
    reported at). Returns:
      per_question_sd_mean : mean over qids of the per-qid std-dev across repeats
      run_mean_sd          : std-dev of the per-run MEANS (the run-level metric's wobble — the
                             quantity a mean-delta is compared against)
      run_mean_range       : max-min of the per-run means
      n_runs, n_questions
    A run-level mean-delta smaller than ~run_mean_sd (or inside run_mean_range) is within noise
    and must NOT be called an improvement even if the CI/p say so on a single pair. For the
    deterministic backbone all of these are ~0.

    VARIANCE SOURCES COMBINED (drill 2026-06-02 H5 — confirm, no code change needed): `runs` is a
    list of RUN-LEVEL per-question tables (one per baseline repeat). When the eval uses 3 judge
    seeds, each per-run table ALREADY averages those seeds per question (judge_aggregate averages
    the 3 seed JSONs into one per-question scalar before this layer ever sees it). So the run-to-
    run spread captured here is a COMBINED floor: synth temperature=0.2 jitter (changes prose_keys
    → the deterministic citation metrics) + retrieval/keyword jitter + judge-seed jitter (folded
    into each run's seed-averaged judge scalars). run_mean_sd is therefore the right single number
    for verdict()'s noise-floor test — it already rolls synth+keyword+judge-seed variance together.
    """
    if len(runs) < 2:
        raise ValueError("noise_floor needs >=2 repeats of the baseline")
    common = sorted(set.intersection(*[set(r) for r in runs]))
    if not common:
        raise ValueError("no common qids across the repeat runs")

    mat = np.array([[float(r[q]) for q in common] for r in runs])  # shape (n_runs, n_questions)
    per_q_sd = mat.std(axis=0, ddof=1)
    run_means = mat.mean(axis=1)
    return {
        "n_runs": len(runs),
        "n_questions": len(common),
        "per_question_sd_mean": float(per_q_sd.mean()),
        "run_mean_sd": float(run_means.std(ddof=1)),
        "run_mean_range": float(run_means.max() - run_means.min()),
    }


# --------------------------------------------------------------------------- #
# results-file → per-metric per-question scalar tables (via the deterministic backbone)
# --------------------------------------------------------------------------- #
# Which deterministic-backbone metrics are "lower is better" (poison/over-confidence rates).
LOWER_IS_BETTER = {"hallucinated_rate", "phantom_rate", "over_confidence_rate"}

# The deterministic-backbone scalars exposed for paired comparison. Pulled off
# backbone.QuestionMetrics; None-valued fields (traps' recall, undefined rates) are simply
# omitted from a metric's table — pairing then naturally happens only where BOTH runs define
# a value, which is the correct paired-difference behaviour (a question undefined in either
# run carries no delta). overconfident is a per-question bool → 0/1 scalar (always defined).
_BACKBONE_SCALARS = (
    "paper_recall_at_12", "paper_recall_at_12_distinct", "paper_recall_at_served_distinct",
    "paper_recall_at_5", "hit_at_12", "gold_citation_recall", "hallucinated_rate", "phantom_rate",
)


def per_question_scalars(results_path: str | Path, gold_path: str | Path) -> dict[str, dict[str, float]]:
    """Score a results/<tag>.jsonl with backbone.py → {metric: {qid: scalar}}.

    Produces the per-question scalar tables that compare_metric/noise_floor consume, one table
    per deterministic-backbone metric. Honours the backbone's None semantics: a None scalar
    (trap recall, undefined rate) is OMITTED from that metric's table (drop-not-zero), so it is
    never silently averaged or paired as a 0. The over_confidence_rate table carries the per-
    question 0/1 overconfident flag (defined for every question). Imported lazily so importing
    stats.py for the pure stat fns alone doesn't require backbone.py.
    """
    from papervault.eval import backbone  # type: ignore

    gold = {}
    for line in Path(gold_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            g = json.loads(line)
            gold[g["qid"]] = g

    tables: dict[str, dict[str, float]] = {m: {} for m in _BACKBONE_SCALARS}
    tables["over_confidence_rate"] = {}
    for line in Path(results_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        qid = rec.get("qid")
        if qid not in gold:
            continue
        qm = backbone.compute_question_metrics(rec, gold[qid])
        for m in _BACKBONE_SCALARS:
            v = getattr(qm, m)
            if v is not None:
                tables[m][qid] = float(v)
        # Denominator consistency (2026-06-08 drill D1): backbone.aggregate computes
        # over_confidence_rate over ANSWERABLE Qs only (a trap can never be overconfident —
        # its recall is None), so the table must skip traps too; including them as structural
        # 0.0 diluted the run mean (0.1875 vs the true 0.2308 on the 48-Q fullcorpus baseline).
        if not qm.is_trap:
            tables["over_confidence_rate"][qid] = 1.0 if qm.overconfident else 0.0
    return tables


def compare_runs(
    baseline_results: str | Path,
    variant_results: str | Path,
    gold_path: str | Path,
    *,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Convenience: score two result files and run compare_metric on every deterministic metric.

    Returns {metric: PairedResult.as_dict()}. lower-is-better metrics are flagged automatically
    from LOWER_IS_BETTER. (The judge metrics are compared the same way via judge_aggregate's
    per-question scalars; this helper covers the deterministic backbone.)
    """
    bt = per_question_scalars(baseline_results, gold_path)
    vt = per_question_scalars(variant_results, gold_path)
    out: dict[str, dict[str, Any]] = {}
    for metric, b in bt.items():
        v = vt.get(metric, {})
        if not (set(b) & set(v)):
            continue
        res = compare_metric(
            metric, b, v,
            higher_is_better=(metric not in LOWER_IS_BETTER),
            seed=seed,
        )
        out[metric] = res.as_dict()
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Paired-difference stats over two eval result files.")
    p.add_argument("--baseline", required=True, help="results/<baseline>.jsonl")
    p.add_argument("--variant", required=True, help="results/<variant>.jsonl")
    p.add_argument("--gold", required=True, help="gold.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = compare_runs(args.baseline, args.variant, args.gold, seed=args.seed)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
