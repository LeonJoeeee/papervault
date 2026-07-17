"""Compute the FULL-CORPUS downstream headline from the persisted baseline results + judge seeds.

Single-run (1-seed protocol) analogue of recompute_baseline.py, for tag=fullcorpus_baseline over
gold_v2.jsonl (39 answerable + 9 traps). No re-run, no LLM — pure aggregation over persisted data.

  uv run python experiments/eval/headline_fullcorpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent.parent))
from papervault.eval import headline as HL  # noqa: E402
from papervault.eval import judge_aggregate as JA  # noqa: E402
from papervault.eval import stats as ST  # noqa: E402

TAG = "fullcorpus_baseline"
GOLD = EVAL / "gold_v2.jsonl"
NSEEDS = 1


def _hl(rec_table: dict, nug_table: dict) -> float:
    common = set(rec_table) & set(nug_table)
    return (sum(HL.harmonic(rec_table[q], nug_table[q]) for q in common) / len(common)) if common else 0.0


def main() -> None:
    bbq = ST.per_question_scalars(EVAL / "results" / f"{TAG}.jsonl", GOLD)
    jq = JA.per_question_scalars(EVAL / "judge", GOLD, tag=TAG, n_seeds=NSEEDS)
    summ = HL.run_summary(bbq, jq)
    trap = JA.trap_correct_refusal_rate(EVAL / "judge", GOLD, tag=TAG, n_seeds=NSEEDS)
    nug = jq.get("nugget_recall", {})
    hl_distinct = _hl(bbq.get("paper_recall_at_12_distinct", {}), nug)
    hl_chunk = _hl(bbq.get("paper_recall_at_12", {}), nug)

    out = {
        "tag": TAG, "gold": GOLD.name, "n_seeds": NSEEDS,
        "HEADLINE_distinct": round(hl_distinct, 4),
        "HEADLINE_chunk": round(hl_chunk, 4),
        "run_summary_headline": round(summ["headline"], 4),
        "n_headline_questions": summ.get("n_headline"),
        "trap_correct_refusal_rate": trap["trap_correct_refusal_rate"],
        "n_traps_scored": trap["n_traps_scored"],
        "vector": {k: round(v, 4) for k, v in sorted(summ["vector"].items())},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
