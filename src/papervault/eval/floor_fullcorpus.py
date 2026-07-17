"""Compute the FULL-CORPUS baseline headline (3 runs, tightened judge contract) + noise floor.

The full-corpus analogue of recompute_baseline.py: 3 identical baseline repeats
(fullcorpus_baseline / _r2 / _r3) on gold_v2, each judged 1-seed under the 2026-06-08
tightened contract → per-run headline → run-level noise floor (stats.noise_floor run_mean_sd).
Pure aggregation over persisted artifacts (no re-run, no LLM).

  uv run python experiments/eval/floor_fullcorpus.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent.parent))
from papervault.eval import headline as HL  # noqa: E402
from papervault.eval import judge_aggregate as JA  # noqa: E402
from papervault.eval import stats as ST  # noqa: E402

TAGS = ["fullcorpus_baseline", "fullcorpus_r2", "fullcorpus_r3"]
GOLD = EVAL / "gold_v2.jsonl"


def _hl(rec_table: dict, nug_table: dict) -> float:
    common = set(rec_table) & set(nug_table)
    return (sum(HL.harmonic(rec_table[q], nug_table[q]) for q in common) / len(common)) if common else 0.0


def main() -> None:
    per_run = {}
    for tag in TAGS:
        bbq = ST.per_question_scalars(EVAL / "results" / f"{tag}.jsonl", GOLD)
        jq = JA.per_question_scalars(EVAL / "judge", GOLD, tag=tag, n_seeds=1)
        trap = JA.trap_correct_refusal_rate(EVAL / "judge", GOLD, tag=tag, n_seeds=1)
        per_run[tag] = {
            "headline": _hl(bbq.get("paper_recall_at_12_distinct", {}), jq.get("nugget_recall", {})),
            "hl_table": HL.headline_table(bbq, jq),
            "vector": HL.run_summary(bbq, jq)["vector"],
            "trap": trap["trap_correct_refusal_rate"],
        }
    heads = [per_run[t]["headline"] for t in TAGS]
    nf = ST.noise_floor([per_run[t]["hl_table"] for t in TAGS])
    out = {
        "tags": TAGS, "gold": GOLD.name, "contract": "tightened 2026-06-08",
        "per_run_headline": [round(h, 4) for h in heads],
        "HEADLINE_mean": round(statistics.mean(heads), 4),
        "noise_floor_run_mean_sd": round(nf["run_mean_sd"], 4),
        "run_mean_range": round(nf["run_mean_range"], 4),
        "trap_refusal_per_run": [per_run[t]["trap"] for t in TAGS],
        "vector_mean": {m: round(statistics.mean([per_run[t]["vector"][m] for t in TAGS]), 4)
                        for m in sorted(per_run[TAGS[0]]["vector"])},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
