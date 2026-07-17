"""Reference-free retrieval metric — PRECISION (per-chunk IMS) + PAIRWISE RECALL guard.

Pure-python consumer of the Claude-subagent judge outputs for the two reference-free judges
(precision_judge.md + pairwise_recall_judge.md). NO gold answers / nuggets / keys are EVER read
here — the metric runs on ANY question given only its intent + retrieved chunks. This module does
NOT spawn judges or call any LLM (that is the eval-run-time workflow; see
build_reffree_prompts.py + the runner snippet in this docstring). It mirrors judge_aggregate.py:
validate each judge JSON against the prompt's self-consistency rules, then aggregate to
per-question scalars + a run mean, so the reference-free metric rides the same paired-difference
machinery (stats.py) as every other metric.

Two parts:

  PART 1 — PRECISION (the primary optimization signal).
    Per (intent, chunk) the precision judge (precision_judge.md) emits one JSON with an integer
    IMS 0-100. precision(question) = mean(IMS over that question's served chunks) / 100. The run
    precision = mean over questions. Pure fns: validate_precision_json, precision_for_question,
    precision_per_question (the {qid: precision} table for stats.compare_metric), run_precision.

  PART 2 — PAIRWISE RECALL GUARD (variant vs baseline, reference-free).
    Per (intent, set A=baseline chunks, set B=variant chunks) the pairwise judge
    (pairwise_recall_judge.md) lists B's unique GAINS and LOSSES and a directional verdict
    {net_gain, net_loss, tie}. pairwise_tally counts net gains vs losses across the question set.
    Pure fns: validate_pairwise_json, pairwise_tally.

LAYOUT (mirrors judge_aggregate / build_judge_prompts):
  precision judge outputs: <judge_dir>/<tag>/<qid>.<chunk_id>.seed<k>.json
  pairwise  judge outputs: <judge_dir>/<tag>/<qid>.seed<k>.json    (one per question)
where chunk_id is "<paper_key>#<ordinal>" (ordinal = the chunk's index in the result record, so
duplicate paper_keys stay distinct — no dedup, per CHUNK_QUALITY_METRIC.md "no dedup by paper").

Tests (tests/test_metric_reffree.py) exercise validation (good + each broken identity) and the
aggregation on canned judge JSONs — no judge, no LLM.

----------------------------------------------------------------------------------------------
RUNNER (how to fan out the judging — mirror build_judge_prompts.py + the workflow judge pattern):

  # 1. build per-(intent,chunk) precision prompts + per-(intent,A,B) pairwise prompts from runs:
  uv run python experiments/eval/build_reffree_prompts.py precision <tag>
  uv run python experiments/eval/build_reffree_prompts.py pairwise  <baseline_tag> <variant_tag>

  # 2. fan out the Claude judge over every *.prompt.txt (one subagent call per file), writing the
  #    JSON next to it as <same-stem>.seed0.json (replays -> .seed1.json, ...). This is the SAME
  #    pattern judge_prompt.md uses: the subagent reads ONE prompt.txt and returns ONE JSON object;
  #    it never touches the live graph (all inputs are in the prompt). The judge MUST be a Claude
  #    model, NOT MiMo (MiMo is the pl production gate — avoid self-judging).

  # 3. score offline (pure, no LLM):
  from papervault.eval import metric_reffree as M
  pt = M.precision_per_question("experiments/eval/judge", tag=<tag>)      # {qid: precision}
  print(M.run_precision(pt))                                             # the run's mean precision
  tally = M.pairwise_tally("experiments/eval/judge", tag=<pairwise_tag>) # net gains vs losses
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# --- band caps (mirror precision_judge.md §3 / §7) ---------------------------------------------
_BANDS = ("90-100", "70-89", "40-69", "0-39")
_BAND_CAP = {"90-100": 100, "70-89": 89, "40-69": 55, "0-39": 39}
_BAND_FLOOR = {"90-100": 90, "70-89": 70, "40-69": 40, "0-39": 0}
_VALID_VERDICTS = {"net_gain", "net_loss", "tie"}
_VALID_IMPORTANCE = {"high", "medium", "low"}


class JudgeValidationError(ValueError):
    """A reference-free judge JSON violated its prompt contract (bad band/cap, bad counts, range)."""


# ================================================================================================
# PART 1 — PRECISION (per-chunk IMS)
# ================================================================================================

def validate_precision_json(j: dict[str, Any]) -> None:
    """Enforce precision_judge.md §7 self-consistency. Raise JudgeValidationError on any breach.

    Checks:
      * ims is an int in [0,100]
      * band is one of the four; ims lies within [floor, cap] of its band (40-69 cap 55, 0-39 cap 39)
      * pointer_only=true ⇒ ims <= 39 (a pointer cannot exceed the pointer cap)
    """
    ims = j.get("ims")
    if not isinstance(ims, int) or isinstance(ims, bool) or not (0 <= ims <= 100):
        raise JudgeValidationError(f"ims={ims!r} not an int in [0,100]")

    band = j.get("band")
    if band not in _BANDS:
        raise JudgeValidationError(f"band={band!r} not one of {_BANDS}")
    lo, cap = _BAND_FLOOR[band], _BAND_CAP[band]
    if not (lo <= ims <= cap):
        raise JudgeValidationError(
            f"ims={ims} outside band {band!r} range [{lo},{cap}] (band cap violated)"
        )

    if bool(j.get("pointer_only")) and ims > 39:
        raise JudgeValidationError(f"pointer_only=true but ims={ims} > 39 (pointer cap)")


def precision_for_question(chunk_jsons: list[dict[str, Any]], *, validate: bool = True) -> float:
    """precision(question) = mean(IMS over the question's served chunks) / 100, in [0,1].

    `chunk_jsons` = every precision-judge JSON for ONE qid (one per served chunk). An empty list
    (a question that served no chunks) → 0.0 (no useful material is the honest floor signal — see
    CHUNK_QUALITY_METRIC.md "below a floor of useful material = too little").
    """
    if validate:
        for j in chunk_jsons:
            validate_precision_json(j)
    if not chunk_jsons:
        return 0.0
    return sum(int(j["ims"]) for j in chunk_jsons) / (100.0 * len(chunk_jsons))


def precision_per_question(
    judge_dir: str | Path,
    *,
    tag: str,
    seed: int = 0,
    validate: bool = True,
) -> dict[str, float]:
    """Build the {qid: precision} table for ONE run from per-chunk precision-judge outputs.

    Reads `<judge_dir>/<tag>/<qid>.<chunk_id>.seed<seed>.json`, groups by qid, and averages each
    qid's chunk IMS /100. The returned table drops straight into stats.compare_metric — the
    reference-free precision rides the identical paired-difference machinery as every other metric.

    REFERENCE-FREE: no gold is read; EVERY qid that has judged chunks is scored (broad or specific,
    answerable or trap — there is no gold to bucket against). A caller that wants to drop traps must
    do so with its own qid list; this fn does not know what a trap is (no gold).
    """
    base = Path(judge_dir) / tag
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(base.glob(f"*.seed{seed}.json")):
        j = json.loads(f.read_text())
        qid = j.get("qid")
        if qid is None:
            raise JudgeValidationError(f"{f}: precision judge JSON missing 'qid'")
        by_qid.setdefault(qid, []).append(j)
    return {
        qid: precision_for_question(js, validate=validate)
        for qid, js in by_qid.items()
    }


def run_precision(precision_table: dict[str, float]) -> float:
    """Run-level precision = mean over the per-question precisions (0.0 if no questions)."""
    return (sum(precision_table.values()) / len(precision_table)) if precision_table else 0.0


# ================================================================================================
# PART 2 — PAIRWISE RECALL GUARD (variant B vs baseline A)
# ================================================================================================

def validate_pairwise_json(j: dict[str, Any]) -> None:
    """Enforce pairwise_recall_judge.md §4 self-consistency. Raise JudgeValidationError on breach.

    Checks:
      * n_gains == len(b_gains), n_losses == len(b_losses)
      * verdict in {net_gain, net_loss, tie}
      * every gain/loss importance in {high, medium, low}
      * net_gain with 0 gains, or net_loss with 0 losses, is contradictory → rejected
    """
    gains = j.get("b_gains")
    losses = j.get("b_losses")
    if not isinstance(gains, list) or not isinstance(losses, list):
        raise JudgeValidationError("b_gains / b_losses must be lists")

    ng, nl = j.get("n_gains"), j.get("n_losses")
    if ng != len(gains):
        raise JudgeValidationError(f"n_gains={ng!r} != len(b_gains)={len(gains)}")
    if nl != len(losses):
        raise JudgeValidationError(f"n_losses={nl!r} != len(b_losses)={len(losses)}")

    for pt in (*gains, *losses):
        imp = pt.get("importance")
        if imp not in _VALID_IMPORTANCE:
            raise JudgeValidationError(f"bad importance {imp!r} (must be high|medium|low)")

    verdict = j.get("verdict")
    if verdict not in _VALID_VERDICTS:
        raise JudgeValidationError(f"verdict={verdict!r} not one of {_VALID_VERDICTS}")
    if verdict == "net_gain" and len(gains) == 0:
        raise JudgeValidationError("verdict net_gain but b_gains is empty (ungrounded)")
    if verdict == "net_loss" and len(losses) == 0:
        raise JudgeValidationError("verdict net_loss but b_losses is empty (ungrounded)")


# weight by importance when tallying the substance moved (not just the per-question verdict).
_IMPORTANCE_W = {"high": 3.0, "medium": 2.0, "low": 1.0}


def pairwise_tally(
    judge_dir: str | Path,
    *,
    tag: str,
    seed: int = 0,
    validate: bool = True,
) -> dict[str, Any]:
    """Aggregate the per-question pairwise verdicts of ONE comparison into a directional summary.

    Reads `<judge_dir>/<tag>/<qid>.seed<seed>.json` (one per question). Returns BOTH a coarse
    verdict count (how many questions net_gain vs net_loss vs tie) AND an importance-weighted
    substance balance (Σ gain-weights − Σ loss-weights across questions), so a few high-importance
    gains are not washed out by many low ones. `net_verdict` is the directional summary used to
    compare the variant vs the baseline:
        net_gain  if (#net_gain Qs) > (#net_loss Qs)
        net_loss  if (#net_loss Qs) > (#net_gain Qs)
        tie       otherwise
    (Ties on the question count fall back to the sign of the weighted balance.)
    """
    base = Path(judge_dir) / tag
    per_qid: dict[str, dict[str, Any]] = {}
    n_gain = n_loss = n_tie = 0
    gain_w = loss_w = 0.0

    for f in sorted(base.glob(f"*.seed{seed}.json")):
        j = json.loads(f.read_text())
        if validate:
            validate_pairwise_json(j)
        qid = j.get("qid") or f.name
        v = j["verdict"]
        gw = sum(_IMPORTANCE_W[g["importance"]] for g in j["b_gains"])
        lw = sum(_IMPORTANCE_W[g["importance"]] for g in j["b_losses"])
        gain_w += gw
        loss_w += lw
        if v == "net_gain":
            n_gain += 1
        elif v == "net_loss":
            n_loss += 1
        else:
            n_tie += 1
        per_qid[qid] = {"verdict": v, "n_gains": j["n_gains"], "n_losses": j["n_losses"],
                        "gain_weight": gw, "loss_weight": lw}

    if n_gain > n_loss:
        net = "net_gain"
    elif n_loss > n_gain:
        net = "net_loss"
    else:  # question-count tie → break by weighted substance balance
        bal = gain_w - loss_w
        net = "net_gain" if bal > 0 else "net_loss" if bal < 0 else "tie"

    return {
        "tag": tag,
        "n_questions": n_gain + n_loss + n_tie,
        "n_net_gain": n_gain,
        "n_net_loss": n_loss,
        "n_tie": n_tie,
        "gain_weight": gain_w,
        "loss_weight": loss_w,
        "weighted_balance": gain_w - loss_w,
        "net_verdict": net,
        "per_qid": per_qid,
    }


# ================================================================================================
# chunk_id helper (shared with build_reffree_prompts.py) — keeps duplicate paper_keys distinct
# ================================================================================================

def chunk_id(paper_key: str | None, ordinal: int) -> str:
    """Stable per-chunk id: '<paper_key>#<ordinal>'. ordinal = position in the result record's
    chunk list, so two chunks from the same paper stay distinct (CHUNK_QUALITY_METRIC.md: no dedup
    by paper — if one paper supplies ten useful pieces, all ten are scored)."""
    return f"{paper_key or 'none'}#{ordinal}"


_SAFE = re.compile(r"[^A-Za-z0-9._#-]+")


def safe_stem(s: str) -> str:
    """Filesystem-safe stem for a chunk_id (paper_keys are usually safe, but slashes/spaces happen)."""
    return _SAFE.sub("_", s)
