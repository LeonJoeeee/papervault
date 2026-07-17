"""Aggregate the Claude-judge outputs (metrics 3/4/5) — validate, average over seeds, score.

The judge subagent (driven by judge_prompt.md) emits one JSON object per (question, seed). This
module is the pure-python consumer of those JSONs. It does NOT spawn judges or call any LLM —
that's the eval-run-time workflow. Here we:

  * validate each judge JSON against the contract's self-consistency rules (the schema's
    arithmetic identities — a judge that mis-sums its own checks is rejected, not silently
    averaged in). This is the guard that keeps the judge honest.
  * average the per-seed scalars over the 3 seeds (the contract runs 3 seeds per question) and
    record the per-seed spread (the judge-overlay noise — fed into the noise floor).
  * emit per-question scalar tables for the three judge metrics in the SAME shape stats.py's
    per-question tables use, so the judge metrics go through the identical paired-difference
    machinery as the deterministic backbone. TRAP questions (gold_keys==[]) are EXCLUDED from
    these quality tables — exactly as the backbone drops them — so judge paired-n == backbone
    paired-n; the trap correct-refusal credit is reported separately as a guardrail rate via
    trap_correct_refusal_rate() (SDD §6.10 B, tail 9).

Judge metrics produced (all in [0,1] except relevance, normalised to [0,1] as relevance/100 so
every metric the stats layer sees is on one scale; the raw 0-100 is kept too):
  citation_support_precision, citation_recall, nugget_recall, faithfulness, relevance(/100).
All are higher-is-better.

Tests (tests/test_judge_aggregate.py) exercise validation (good + each broken identity) and the
seed-averaging on canned judge JSONs — no judge, no LLM.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

_COVERAGE_SCORE = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
_VALID_VERDICTS = {"entail", "neutral", "contradict", "missing_chunk"}
_TOL = 1e-6

JUDGE_METRICS = (
    "citation_support_precision", "citation_recall", "nugget_recall",
    "faithfulness", "relevance",  # relevance stored normalised to [0,1]
)


class JudgeValidationError(ValueError):
    """A judge JSON violated the contract (bad arithmetic, out-of-range, malformed checks)."""


def validate_judge_json(j: dict[str, Any], *, n_gold_nuggets: int | None = None) -> None:
    """Enforce judge_prompt.md's self-consistency rules. Raise JudgeValidationError on any breach.

    Checks (mirrors the "Self-consistency requirements" block of the contract):
      * citation_support_precision == (#entail)/len(citation_checks), 0 checks → 1.0
      * citation_recall == n_claims_with_citation / n_substantive_claims, 0 claims → 1.0
      * nugget_recall == mean(nugget_judgements[].score); each coverage maps to its score
      * all [0,1] scores in range; relevance an int in [0,100]
      * verdicts in the allowed set; if n_gold_nuggets given, one judgement per nugget
    A trap_correct_refusal=true entry is allowed to carry empty check/nugget lists with all
    scores pinned to their 1.0 (or relevance 100) — the identities still hold (0/empty → 1.0).
    """
    def _in01(name: str) -> float:
        v = j.get(name)
        if not isinstance(v, (int, float)) or not (0.0 - _TOL <= v <= 1.0 + _TOL):
            raise JudgeValidationError(f"{name}={v!r} not a float in [0,1]")
        return float(v)

    checks = j.get("citation_checks")
    if not isinstance(checks, list):
        raise JudgeValidationError("citation_checks must be a list")
    for c in checks:
        if c.get("verdict") not in _VALID_VERDICTS:
            raise JudgeValidationError(f"bad citation verdict {c.get('verdict')!r}")

    prec = _in01("citation_support_precision")
    n_entail = sum(1 for c in checks if c.get("verdict") == "entail")
    expect_prec = (n_entail / len(checks)) if checks else 1.0
    if not math.isclose(prec, expect_prec, abs_tol=1e-3):
        raise JudgeValidationError(
            f"citation_support_precision {prec} != {n_entail}/{len(checks)}={expect_prec}"
        )

    rec = _in01("citation_recall")
    n_claims = j.get("n_substantive_claims")
    n_cited = j.get("n_claims_with_citation")
    if not isinstance(n_claims, int) or not isinstance(n_cited, int) or n_claims < 0 or n_cited < 0:
        raise JudgeValidationError("n_substantive_claims / n_claims_with_citation must be non-neg ints")
    if n_cited > n_claims:
        raise JudgeValidationError("n_claims_with_citation > n_substantive_claims")
    expect_rec = (n_cited / n_claims) if n_claims else 1.0
    if not math.isclose(rec, expect_rec, abs_tol=1e-3):
        raise JudgeValidationError(f"citation_recall {rec} != {n_cited}/{n_claims}={expect_rec}")

    nugs = j.get("nugget_judgements")
    if not isinstance(nugs, list):
        raise JudgeValidationError("nugget_judgements must be a list")
    if n_gold_nuggets is not None and len(nugs) != n_gold_nuggets:
        raise JudgeValidationError(
            f"nugget_judgements has {len(nugs)} entries, expected {n_gold_nuggets} (one per gold nugget)"
        )
    for nj in nugs:
        cov = nj.get("coverage")
        if cov not in _COVERAGE_SCORE:
            raise JudgeValidationError(f"bad nugget coverage {cov!r}")
        if not math.isclose(float(nj.get("score", -1)), _COVERAGE_SCORE[cov], abs_tol=1e-6):
            raise JudgeValidationError(f"nugget score {nj.get('score')} != {cov}->{_COVERAGE_SCORE[cov]}")

    nr = _in01("nugget_recall")
    if nugs:
        expect_nr = sum(_COVERAGE_SCORE[nj["coverage"]] for nj in nugs) / len(nugs)
    elif j.get("trap_correct_refusal") is False:
        # Failed trap (2026-06-08 drill D4): the contract scores a substantively-answered trap
        # HARSHLY — nugget_recall stays 0 with an empty judgement list (a trap has no gold
        # nuggets). The old identity (empty list -> expect 1.0) wrongly rejected exactly those
        # honest failed-trap outputs; only a CLEAN refusal earns 1.0-with-empty-list credit.
        expect_nr = 0.0
    else:
        expect_nr = 1.0
    if not math.isclose(nr, expect_nr, abs_tol=1e-3):
        raise JudgeValidationError(f"nugget_recall {nr} != mean(nugget scores)={expect_nr}")

    _in01("faithfulness")

    relv = j.get("relevance")
    if not isinstance(relv, int) or not (0 <= relv <= 100):
        raise JudgeValidationError(f"relevance={relv!r} not an int in [0,100]")


def _scalars(j: dict[str, Any]) -> dict[str, float]:
    """Pull the five judge metrics out of one validated judge JSON (relevance → /100)."""
    return {
        "citation_support_precision": float(j["citation_support_precision"]),
        "citation_recall": float(j["citation_recall"]),
        "nugget_recall": float(j["nugget_recall"]),
        "faithfulness": float(j["faithfulness"]),
        "relevance": float(j["relevance"]) / 100.0,
    }


def aggregate_seeds(
    seed_jsons: list[dict[str, Any]],
    *,
    n_gold_nuggets: int | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Average one question's judge metrics over its (3) seed JSONs + record seed spread.

    Every seed JSON is validated (unless validate=False) before it contributes. Returns:
      {qid, n_seeds, mean:{metric:..}, sd:{metric:..}}  — sd is the per-metric across-seed
    std-dev (the judge-overlay noise for this question; aggregated into the noise floor).
    """
    if not seed_jsons:
        raise JudgeValidationError("no seed JSONs to aggregate")
    if validate:
        for j in seed_jsons:
            validate_judge_json(j, n_gold_nuggets=n_gold_nuggets)

    per_seed = [_scalars(j) for j in seed_jsons]
    qids = {j.get("qid") for j in seed_jsons}
    mean: dict[str, float] = {}
    sd: dict[str, float] = {}
    n = len(per_seed)
    for m in JUDGE_METRICS:
        vals = [s[m] for s in per_seed]
        mu = sum(vals) / n
        mean[m] = mu
        sd[m] = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"qid": next(iter(qids)) if len(qids) == 1 else sorted(qids), "n_seeds": n, "mean": mean, "sd": sd}


def _is_trap(g: dict[str, Any]) -> bool:
    """A gold entry is a trap iff it has no gold_keys (the corpus answers nothing). Mirrors
    backbone.compute_question_metrics' is_trap so the two layers bucket questions identically."""
    return not (g.get("gold_keys") or [])


def per_question_scalars(
    judge_dir: str | Path,
    gold_path: str | Path,
    *,
    tag: str,
    n_seeds: int = 3,
) -> dict[str, dict[str, float]]:
    """Build {metric: {qid: seed-averaged scalar}} tables for the judge QUALITY metrics of ONE run.

    Expects judge outputs laid out as `<judge_dir>/<tag>/<qid>.seed<k>.json` (k = 0..n_seeds-1),
    each one JSON object matching the contract. This is the judge-side analogue of
    stats.per_question_scalars; the returned tables drop straight into stats.compare_metric.
    Missing seed files for a qid → that qid is skipped (with no silent partial-seed averaging
    unless fewer than all seeds are intentionally present).

    TRAP POLICY (SDD §6.10 B, drill 2026-06-02e, tail 9): trap questions (gold_keys==[]) are
    EXCLUDED from these five quality tables — exactly as backbone.aggregate drops them from the
    paired recall/citation comparison (so judge paired-n == backbone paired-n == 21, not 25).
    A trap's pinned 1.0 correct-refusal scores would otherwise inject always-tied delta-0 rows,
    diluting effect size and artificially tightening the bootstrap CI / shifting the permutation
    null. The trap correct-refusal credit is reported SEPARATELY as a guardrail via
    trap_correct_refusal_rate(), the judge-side analogue of the backbone's trap_violation_rate.
    """
    gold = {}
    for line in Path(gold_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            g = json.loads(line)
            gold[g["qid"]] = g

    base = Path(judge_dir) / tag
    tables: dict[str, dict[str, float]] = {m: {} for m in JUDGE_METRICS}
    for qid, g in gold.items():
        if _is_trap(g):
            continue  # traps go to the guardrail (trap_correct_refusal_rate), not the quality tables
        seeds = _load_seeds(base, qid, n_seeds)
        if not seeds:
            continue
        agg = aggregate_seeds(seeds, n_gold_nuggets=len(g.get("nuggets") or []) or None)
        for m in JUDGE_METRICS:
            tables[m][qid] = agg["mean"][m]
    return tables


def _load_seeds(base: Path, qid: str, n_seeds: int) -> list[dict[str, Any]]:
    seeds = []
    for k in range(n_seeds):
        f = base / f"{qid}.seed{k}.json"
        if f.exists():
            seeds.append(json.loads(f.read_text()))
    return seeds


def trap_correct_refusal_rate(
    judge_dir: str | Path,
    gold_path: str | Path,
    *,
    tag: str,
    n_seeds: int = 3,
) -> dict[str, Any]:
    """Guardrail: over the TRAP questions, the fraction the judge marked as a correct refusal.

    The judge-side analogue of the backbone's trap_violation_rate / over_confidence_rate — a
    guardrail rate, NOT a paired quality metric (so it never enters stats.compare_metric). For
    each trap qid we read its seed JSONs and treat trap_correct_refusal as True iff the MAJORITY
    of seeds agree (ties / all-True -> True); a trap with no seed files is skipped (not counted).
    Returns {n_traps_scored, trap_correct_refusal_rate, per_qid:{qid: bool}}.
    """
    gold = {}
    for line in Path(gold_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            g = json.loads(line)
            gold[g["qid"]] = g

    base = Path(judge_dir) / tag
    per_qid: dict[str, bool] = {}
    for qid, g in gold.items():
        if not _is_trap(g):
            continue
        seeds = _load_seeds(base, qid, n_seeds)
        if not seeds:
            continue
        n_true = sum(1 for j in seeds if bool(j.get("trap_correct_refusal")))
        per_qid[qid] = n_true * 2 >= len(seeds)  # majority (ties -> True)
    n = len(per_qid)
    rate = (sum(1 for v in per_qid.values() if v) / n) if n else None
    return {"n_traps_scored": n, "trap_correct_refusal_rate": rate, "per_qid": per_qid}
