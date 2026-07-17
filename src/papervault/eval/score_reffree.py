"""Aggregate the reference-free PRECISION judge outputs into the baseline number + a breakdown.

Pure consumer (no LLM, no graph): reads the per-(intent,chunk) precision-judge JSONs written by
the Claude judge fan-out (judge/<tag>/<qid>.<chunk_id>.seed<seed>.json), validates each against
precision_judge.md §7 (via metric_reffree.validate_precision_json), and prints:
  * run precision = mean over questions of mean(IMS/100 over that question's served chunks)
  * per-INTENT-TYPE means (broad vs cross_domain), joined from the intents file
  * a per-question table (precision, #chunks judged, #served) so under-judged questions are visible
  * JUDGE COVERAGE: judged-chunk count vs served-chunk count per qid (from results/<tag>.jsonl), so
    a partial fan-out can NEVER be silently read as "scored everything" (no-silent-caps).

Run (after build_reffree_prompts precision <tag> + the judge fan-out land the seed0.json files):
  uv run python experiments/eval/score_reffree.py <tag> [--intents intents_v1.jsonl] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))
import metric_reffree as M  # noqa: E402


def _load_jsonl(path: Path) -> dict:
    d = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            o = json.loads(line)
            d[o["qid"]] = o
    return d


def _served_counts(tag: str) -> dict[str, int]:
    """#chunks the run actually served per qid (from results/<tag>.jsonl), for coverage check."""
    p = EVAL / "results" / f"{tag}.jsonl"
    if not p.exists():
        return {}
    return {qid: len((rec.get("data") or {}).get("chunks") or []) for qid, rec in _load_jsonl(p).items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate reference-free precision judge outputs.")
    ap.add_argument("tag", help="run tag (judge/<tag>/ + results/<tag>.jsonl)")
    ap.add_argument("--intents", default=str(EVAL / "intents_v1.jsonl"))
    ap.add_argument("--judge-dir", default=str(EVAL / "judge"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    intents = _load_jsonl(Path(args.intents))
    type_of = {qid: o.get("type", "?") for qid, o in intents.items()}

    # per-question precision (validates each judge JSON; raises on contract breach)
    pq = M.precision_per_question(args.judge_dir, tag=args.tag, seed=args.seed)

    # judged-chunk counts per qid (group the seed files), for the coverage line
    base = Path(args.judge_dir) / args.tag
    judged = {}
    for f in sorted(base.glob(f"*.seed{args.seed}.json")):
        j = json.loads(f.read_text())
        judged[j.get("qid")] = judged.get(j.get("qid"), 0) + 1
    served = _served_counts(args.tag)

    overall = M.run_precision(pq)
    by_type: dict[str, list[float]] = {}
    for qid, p in pq.items():
        by_type.setdefault(type_of.get(qid, "?"), []).append(p)

    print(f"\n=== REFERENCE-FREE PRECISION | tag={args.tag} | seed={args.seed} ===")
    print(f"questions scored: {len(pq)} / {len(intents)} intents\n")
    print(f"{'qid':38} {'type':12} {'precision':>9} {'judged':>7} {'served':>7}")
    print("-" * 78)
    for qid in sorted(pq, key=lambda q: (type_of.get(q, '?'), q)):
        jc, sc = judged.get(qid, 0), served.get(qid, -1)
        flag = "  <-- UNDER-JUDGED" if (sc >= 0 and jc < sc) else ""
        print(f"{qid:38} {type_of.get(qid,'?'):12} {pq[qid]*100:8.1f} {jc:7d} {sc:7d}{flag}")
    print("-" * 78)
    for t in sorted(by_type):
        vals = by_type[t]
        print(f"  {t:12} mean precision = {100*sum(vals)/len(vals):5.1f}  (n={len(vals)})")
    print(f"\n  OVERALL run precision = {overall*100:.1f}  (mean over {len(pq)} questions)")

    # coverage guard (no-silent-caps)
    miss = [q for q in served if served[q] > judged.get(q, 0)]
    unscored = [q for q in intents if q not in pq]
    if miss:
        print(f"\n  ! UNDER-JUDGED ({len(miss)} qids served more chunks than were judged): {miss}")
    if unscored:
        print(f"  ! NOT SCORED ({len(unscored)} intents have no judge output): {unscored}")
    if not miss and not unscored:
        print("\n  coverage OK: every served chunk of every intent was judged.")


if __name__ == "__main__":
    main()
