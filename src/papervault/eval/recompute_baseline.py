"""Recompute the baseline headline from PERSISTED judge/results (no re-run, no LLM) under the
HARDENED backbone — confirm whether the recorded 0.864 is distinct-recall or chunk-based, and
report the hardened vector + run-level noise floor. Read-only over persisted data.

Run:  uv run python experiments/eval/recompute_baseline.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent.parent))
from papervault.eval import headline as HL  # noqa: E402
from papervault.eval import judge_aggregate as JA  # noqa: E402
from papervault.eval import stats as ST  # noqa: E402

TAGS = ["baseline_r1", "baseline_r2", "baseline_r3"]
gold_path = EVAL / "gold.jsonl"


def _hl(rec_table, nug_table):
    common = set(rec_table) & set(nug_table)
    return (sum(HL.harmonic(rec_table[q], nug_table[q]) for q in common) / len(common)) if common else 0.0


per_run = {}
for tag in TAGS:
    bbq = ST.per_question_scalars(EVAL / "results" / f"{tag}.jsonl", gold_path)
    jq = JA.per_question_scalars(EVAL / "judge", gold_path, tag=tag, n_seeds=1)
    summ = HL.run_summary(bbq, jq)
    trap = JA.trap_correct_refusal_rate(EVAL / "judge", gold_path, tag=tag, n_seeds=1)
    nug = jq.get("nugget_recall", {})
    hl_distinct = _hl(bbq.get("paper_recall_at_12_distinct", {}), nug)
    hl_chunk = _hl(bbq.get("paper_recall_at_12", {}), nug)
    per_run[tag] = {"summary": summ, "trap": trap, "hl_table": HL.headline_table(bbq, jq),
                    "hl_distinct": hl_distinct, "hl_chunk": hl_chunk}

d = [per_run[t]["hl_distinct"] for t in TAGS]
c = [per_run[t]["hl_chunk"] for t in TAGS]
print(f"HEADLINE (DISTINCT recall — the hardened/official one): mean={statistics.mean(d):.4f}  per-run={[round(x,4) for x in d]}  sd={statistics.pstdev(d):.4f}")
print(f"HEADLINE (chunk-based recall — old/secondary):          mean={statistics.mean(c):.4f}  per-run={[round(x,4) for x in c]}  sd={statistics.pstdev(c):.4f}")
print(f"run_summary().headline (uses distinct): {[round(per_run[t]['summary']['headline'],4) for t in TAGS]}")
nf = ST.noise_floor([per_run[t]["hl_table"] for t in TAGS])
print(f"noise floor (run-level): run_mean_sd={nf['run_mean_sd']:.4f}  range={nf['run_mean_range']:.4f}  n_q={nf['n_questions']}")
print("\n--- VECTOR (mean over 3 runs) ---")
metrics = sorted({m for t in TAGS for m in per_run[t]["summary"]["vector"]})
for m in metrics:
    vals = [per_run[t]["summary"]["vector"][m] for t in TAGS if m in per_run[t]["summary"]["vector"]]
    print(f"  {m:30} mean={statistics.mean(vals):.4f}  runs={[round(v,3) for v in vals]}")
print("\n--- trap_correct_refusal_rate ---")
for t in TAGS:
    print(f"  {t}: {per_run[t]['trap']['trap_correct_refusal_rate']}  ({per_run[t]['trap']['n_traps_scored']} traps)")
print(f"\n  headline n (non-trap Qs scored): {per_run['baseline_r1']['summary']['n_headline']}/21")
