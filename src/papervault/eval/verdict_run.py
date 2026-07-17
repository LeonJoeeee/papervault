"""Variant-vs-baseline decision via headline.verdict, on the 49-Q gold (3-run protocol).

baseline side: full_r{1,2,3} (existing-25 + new-24 results + judge, already persisted).
variant side : mq_r{1,2,3} results (backbone) + judge where, per run, FUSED questions use the
  variant's own judge (judge/mq_r{k}) and NON-FUSED questions REUSE the baseline judge
  (judge/full_r{k}) — the variant is byte-identical to baseline on non-fused Qs, so reusing the
  baseline outcome is common-random-numbers variance reduction (the per-Q delta there is exactly 0,
  concentrating the test on where the variant actually acts). Merged into judge/mqfull_r{k}.

Per-question value fed to verdict = MEAN over the 3 runs. Noise floor = the pinned baseline
run_mean_sd (headline.BASELINE_NOISE_FLOOR_RUN_MEAN_SD = 0.0127). Trap refusal rate = mean over runs.

Run (AFTER the variant judge seeds exist):  uv run python experiments/eval/verdict_run.py
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


def _mean_tables(per_run: list[dict]) -> dict:
    """[{metric:{qid:v}}] over runs → {metric:{qid:mean-over-runs}}."""
    metrics: set = set().union(*[set(t) for t in per_run]) if per_run else set()
    out: dict = {}
    for m in metrics:
        qv: dict = {}
        for t in per_run:
            for q, v in (t.get(m) or {}).items():
                qv.setdefault(q, []).append(v)
        out[m] = {q: statistics.mean(vs) for q, vs in qv.items()}
    return out


def _side(results_fmt: str, judge_tag_fmt: str) -> tuple[dict, float]:
    """Build a {'backbone':..,'judge':..} mean-table side + mean trap_correct_refusal_rate."""
    bb_runs, j_runs, traps = [], [], []
    for k in RUNS:
        bb_runs.append(ST.per_question_scalars(EVAL / "results" / results_fmt.format(k=k), GOLD))
        j_runs.append(JA.per_question_scalars(EVAL / "judge", GOLD, tag=judge_tag_fmt.format(k=k), n_seeds=1))
        tr = JA.trap_correct_refusal_rate(EVAL / "judge", GOLD, tag=judge_tag_fmt.format(k=k), n_seeds=1)
        if tr.get("trap_correct_refusal_rate") is not None:
            traps.append(tr["trap_correct_refusal_rate"])
    return {"backbone": _mean_tables(bb_runs), "judge": _mean_tables(j_runs)}, (statistics.mean(traps) if traps else 0.0)


def _build_mqfull_judge_dirs() -> dict[int, int]:
    """Per run, build judge/mqfull_r{k} = baseline full_r{k} seeds, with FUSED qids overwritten by
    the variant's judge/mq_r{k} seeds. Returns {k: n_fused}."""
    nfused = {}
    for k in RUNS:
        recs = [json.loads(l) for l in (EVAL / "results" / f"mq_r{k}.jsonl").read_text().splitlines() if l.strip()]
        fused = {r["qid"] for r in recs if r["meta"].get("fused_applied")}
        dst = EVAL / "judge" / f"mqfull_r{k}"
        dst.mkdir(parents=True, exist_ok=True)
        for sf in (EVAL / "judge" / f"full_r{k}").glob("*.seed0.json"):
            shutil.copy(sf, dst / sf.name)
        applied = 0
        for q in fused:
            src = EVAL / "judge" / f"mq_r{k}" / f"{q}.seed0.json"
            if src.exists():
                shutil.copy(src, dst / f"{q}.seed0.json")
                applied += 1
        nfused[k] = applied
    return nfused


def main() -> None:
    nfused = _build_mqfull_judge_dirs()
    print(f"fused (variant judge applied) per run: {nfused}")

    base, base_trap = _side("full_r{k}.jsonl", "full_r{k}")
    var, var_trap = _side("mq_r{k}.jsonl", "mqfull_r{k}")

    v = HL.verdict(
        base, var,
        noise_floor_run_mean_sd=HL.BASELINE_NOISE_FLOOR_RUN_MEAN_SD,
        trap_refusal_baseline=base_trap, trap_refusal_variant=var_trap,
        n_boot=4000, n_perm=4000, seed=0,
    )

    bh = v.get("baseline_headline"); vh = v.get("variant_headline")
    print("\n================= VERDICT: V-MQ (multi-query+RRF) vs baseline =================")
    print(f"baseline headline = {bh:.4f}   variant headline = {vh:.4f}   delta = {(vh - bh):+.4f}")
    hd = v.get("headline") or {}
    print(f"(i)   movers significance: improved={hd.get('improved')}  n_movers={v.get('n_movers')}  movers_ci_low={hd.get('movers_ci_low')}")
    print(f"(ii)  passes_noise_floor  = {v.get('passes_noise_floor')}  (delta {hd.get('mean_delta'):+.4f} vs k*floor {HL.BASELINE_NOISE_FLOOR_RUN_MEAN_SD})")
    print(f"(iii) survives_jackknife  = {v.get('survives_jackknife')}")
    print(f"      trap_refusal: baseline={base_trap:.3f} variant={var_trap:.3f}")
    print("(iv)  red-line gates:")
    for m, g in (v.get("gates") or {}).items():
        print(f"        {m:30} regressed={g.get('regressed')} evaluable={g.get('evaluable')}"
              + (f"  base={g.get('mean_baseline'):.3f} var={g.get('mean_variant'):.3f}" if g.get('mean_baseline') is not None else ""))
    print(f"\n  all_gates_evaluable = {v.get('all_gates_evaluable')}")
    print(f"  *** WIN = {v.get('win')} ***")


if __name__ == "__main__":
    main()
