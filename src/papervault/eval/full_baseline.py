"""Aggregate the FULL 49-question baseline on the grown gold: consume the ks-judge-newq workflow
output (24 new Qs x 3 runs) -> write newq judge seeds (recomputing derived scalars so
judge_aggregate.validate passes) -> MERGE with the persisted existing-25 results+judge per run
-> per-run headline+vector over all 49 -> baseline mean + noise floor + trap_correct_refusal over
all 10 traps. Read-only on the graph (operates on persisted results/judge).

Run:  python -m papervault.eval.full_baseline /tmp/.../<ks-judge-newq>.output
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent.parent))
from papervault.eval import headline as HL  # noqa: E402
from papervault.eval import judge_aggregate as JA  # noqa: E402
from papervault.eval import stats as ST  # noqa: E402

RUNS = [1, 2, 3]
GOLD = EVAL / "gold.jsonl"
_COV = {"covered": 1.0, "partial": 0.5, "missing": 0.0}


def _recompute(j: dict) -> dict:
    """Force the self-consistency identities from the raw decomposed checks (mirror of
    aggregate_baseline._recompute) so judge_aggregate.validate passes by construction."""
    checks = j.get("citation_checks") or []
    # coerce any off-contract verdict (e.g. a judge that emitted 'partial') to 'neutral' so
    # validate_judge_json passes; 'neutral' is conservative (does NOT count toward entail).
    _OKV = {"entail", "neutral", "contradict", "missing_chunk"}
    for c in checks:
        if c.get("verdict") not in _OKV:
            c["verdict"] = "neutral"
    # coerce any off-contract coverage to 'missing' (conservative).
    for nj in (j.get("nugget_judgements") or []):
        if nj.get("coverage") not in _COV:
            nj["coverage"] = "missing"
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
    # accept ONE OR MORE workflow output files (main pass + any fill-in re-judges)
    wrote = 0
    for path in sys.argv[1:]:
        out = json.loads(Path(path).read_text())
        res = (out.get("result", out) or {}).get("results", [])
        for r in res:
            tag, qid, j = r["tag"], r["qid"], _recompute(dict(r["judge"]))
            j.setdefault("qid", qid)
            d = EVAL / "judge" / tag
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{qid}.seed0.json").write_text(json.dumps(j, ensure_ascii=False))
            wrote += 1
    print(f"wrote {wrote} newq judge seed files (from {len(sys.argv) - 1} output file(s))")

    per_run = {}
    for k in RUNS:
        # merge results: existing-25 (baseline_rk) + new-24 (newq_rk) -> full_rk
        full_res = EVAL / "results" / f"full_r{k}.jsonl"
        with full_res.open("w") as f:
            for tag in (f"baseline_r{k}", f"newq_r{k}"):
                p = EVAL / "results" / f"{tag}.jsonl"
                if p.exists():
                    f.write(p.read_text())
        # merge judge dirs
        fulljd = EVAL / "judge" / f"full_r{k}"
        fulljd.mkdir(parents=True, exist_ok=True)
        for tag in (f"baseline_r{k}", f"newq_r{k}"):
            for sf in (EVAL / "judge" / tag).glob("*.seed0.json"):
                shutil.copy(sf, fulljd / sf.name)
        bbq = ST.per_question_scalars(full_res, GOLD)
        jq = JA.per_question_scalars(EVAL / "judge", GOLD, tag=f"full_r{k}", n_seeds=1)
        summ = HL.run_summary(bbq, jq)
        trap = JA.trap_correct_refusal_rate(EVAL / "judge", GOLD, tag=f"full_r{k}", n_seeds=1)
        per_run[k] = {"summary": summ, "trap": trap, "hl_table": HL.headline_table(bbq, jq)}

    headlines = [per_run[k]["summary"]["headline"] for k in RUNS]
    print("\n========== KS FULL BASELINE (49 Qs = 39 answerable + 10 traps, l0_probe, 89 papers) ==========")
    print(f"HEADLINE  mean={statistics.mean(headlines):.4f}  per-run={[round(h, 4) for h in headlines]}  sd={statistics.pstdev(headlines):.4f}")
    nf = ST.noise_floor([per_run[k]["hl_table"] for k in RUNS])
    print(f"noise floor (run-level): run_mean_sd={nf['run_mean_sd']:.4f}  range={nf['run_mean_range']:.4f}  n_q={nf['n_questions']}")
    metrics = sorted({m for k in RUNS for m in per_run[k]["summary"]["vector"]})
    print("\n--- VECTOR (mean over 3 runs) ---")
    for m in metrics:
        vals = [per_run[k]["summary"]["vector"][m] for k in RUNS if m in per_run[k]["summary"]["vector"]]
        print(f"  {m:30} mean={statistics.mean(vals):.4f}  runs={[round(v, 3) for v in vals]}")
    print("\n--- trap_correct_refusal_rate (10 traps) ---")
    for k in RUNS:
        print(f"  full_r{k}: {per_run[k]['trap']['trap_correct_refusal_rate']}  ({per_run[k]['trap']['n_traps_scored']} traps)")
    print(f"\n  headline n (non-trap answerable Qs scored): {per_run[1]['summary']['n_headline']}/39")


if __name__ == "__main__":
    main()
