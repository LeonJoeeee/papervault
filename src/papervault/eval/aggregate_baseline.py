"""Consume the ks-baseline-judge workflow output → write per-seed judge JSONs (recomputing the
derived scalars from the raw checks/judgements so judge_aggregate validation passes by
construction) → aggregate the baseline: per-run headline + vector (headline.run_summary),
baseline = mean over the 3 runs, noise floor = stats.noise_floor over the 3 runs' headline
tables, + trap_correct_refusal_rate. Prints the baseline report.

Run:  python -m papervault.eval.aggregate_baseline /tmp/.../<judge_workflow>.output
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent.parent))  # repo root → `from papervault.eval import ...`
from papervault.eval import headline as HL  # noqa: E402
from papervault.eval import judge_aggregate as JA  # noqa: E402
from papervault.eval import stats as ST  # noqa: E402

TAGS = ["baseline_r1", "baseline_r2", "baseline_r3"]
_COV = {"covered": 1.0, "partial": 0.5, "missing": 0.0}


def _recompute(j: dict) -> dict:
    """Force the self-consistency identities from the raw decomposed checks (so the judge's own
    arithmetic rounding can't fail judge_aggregate.validate)."""
    checks = j.get("citation_checks") or []
    n_ent = sum(1 for c in checks if c.get("verdict") == "entail")
    j["citation_support_precision"] = (n_ent / len(checks)) if checks else 1.0
    nc, ns = int(j.get("n_claims_with_citation", 0) or 0), int(j.get("n_substantive_claims", 0) or 0)
    nc = min(nc, ns)
    j["n_claims_with_citation"], j["n_substantive_claims"] = nc, ns
    j["citation_recall"] = (nc / ns) if ns else 1.0
    nugs = j.get("nugget_judgements") or []
    for nj in nugs:
        nj["score"] = _COV.get(nj.get("coverage"), 0.0)
    j["nugget_recall"] = (sum(nj["score"] for nj in nugs) / len(nugs)) if nugs else 1.0
    return j


def main() -> None:
    out = json.loads(Path(sys.argv[1]).read_text())
    res = (out.get("result", out) or {}).get("results", [])
    if not res:
        raise SystemExit(f"no results in {sys.argv[1]}")

    # write recomputed seed0.json
    wrote = 0
    for r in res:
        tag, qid, j = r["tag"], r["qid"], _recompute(dict(r["judge"]))
        j.setdefault("qid", qid)
        d = EVAL / "judge" / tag
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{qid}.seed0.json").write_text(json.dumps(j, ensure_ascii=False))
        wrote += 1
    print(f"wrote {wrote} judge seed files")

    gold_path = EVAL / "gold.jsonl"
    per_run = {}
    for tag in TAGS:
        bbq = ST.per_question_scalars(EVAL / "results" / f"{tag}.jsonl", gold_path)
        jq = JA.per_question_scalars(EVAL / "judge", gold_path, tag=tag, n_seeds=1)
        summ = HL.run_summary(bbq, jq)
        trap = JA.trap_correct_refusal_rate(EVAL / "judge", gold_path, tag=tag, n_seeds=1)
        per_run[tag] = {"summary": summ, "trap": trap, "hl_table": HL.headline_table(bbq, jq)}

    headlines = [per_run[t]["summary"]["headline"] for t in TAGS]
    print("\n================= KS BASELINE (3 runs, l0_probe, V2/V3/V7, reranker on) =================")
    print(f"HEADLINE (harmonic of paper_recall@12 × nugget_recall): "
          f"mean={statistics.mean(headlines):.4f}  per-run={[round(h, 4) for h in headlines]}  "
          f"sd={statistics.pstdev(headlines):.4f}")
    try:
        nf = ST.noise_floor([per_run[t]["hl_table"] for t in TAGS])
        print(f"HEADLINE noise floor (run-level): run_mean_sd={nf['run_mean_sd']:.4f} "
              f"range={nf['run_mean_range']:.4f}  n_q={nf['n_questions']}")
    except Exception as e:  # noqa: BLE001
        print(f"noise_floor: {e}")

    metrics = sorted({m for t in TAGS for m in per_run[t]["summary"]["vector"]})
    print("\n--- VECTOR (mean over 3 runs) ---")
    for m in metrics:
        vals = [per_run[t]["summary"]["vector"][m] for t in TAGS if m in per_run[t]["summary"]["vector"]]
        print(f"  {m:28} mean={statistics.mean(vals):.4f}  runs={[round(v, 3) for v in vals]}")
    print("\n--- trap_correct_refusal_rate (judge: did synth correctly refuse the 4 traps?) ---")
    for t in TAGS:
        print(f"  {t}: {per_run[t]['trap']['trap_correct_refusal_rate']}  ({per_run[t]['trap']['n_traps_scored']} traps)")
    print(f"\n  headline n (non-trap Qs scored): {per_run['baseline_r1']['summary']['n_headline']}/21")


if __name__ == "__main__":
    main()
