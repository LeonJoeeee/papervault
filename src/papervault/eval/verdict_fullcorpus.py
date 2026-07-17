"""Variant-vs-baseline verdict on the FULL-CORPUS frame (gold_v2, 3-run protocol).

  uv run python experiments/eval/verdict_fullcorpus.py <variant_prefix>   # e.g. vmq | vsr

baseline side: fullcorpus_baseline / fullcorpus_r2 / fullcorpus_r3 (results + Opus judge seeds).
variant side : <prefix>_r{1,2,3} results + judge/<prefix>_r{k} seeds (same tightened contract,
same Opus judge — the instrument must not change between sides).

Protocol (BASELINE_FULLCORPUS.md): per-question value fed to verdict = MEAN over 3 runs;
noise floor = headline.FULLCORPUS_NOISE_FLOOR_RUN_MEAN_SD (0.0040), k=1; per-question gates
strict (epsilon=0); the 9-trap scalar gate gets a separate epsilon=0.10 post-adjustment
(measured per-run swing 4/4/2 of 9 → a single judge flip must not decide a verdict, drill D1).
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

GOLD = EVAL / "gold_v2.jsonl"
BASE_TAGS = ["fullcorpus_baseline", "fullcorpus_r2", "fullcorpus_r3"]
TRAP_EPSILON = 0.10


def _mean_tables(per_run: list[dict]) -> dict:
    metrics: set = set().union(*[set(t) for t in per_run]) if per_run else set()
    out: dict = {}
    for m in metrics:
        qv: dict = {}
        for t in per_run:
            for q, v in (t.get(m) or {}).items():
                qv.setdefault(q, []).append(v)
        out[m] = {q: statistics.mean(vs) for q, vs in qv.items()}
    return out


def _side(tags: list[str]) -> tuple[dict, float]:
    bb, jj, traps = [], [], []
    for tag in tags:
        bb.append(ST.per_question_scalars(EVAL / "results" / f"{tag}.jsonl", GOLD))
        jj.append(JA.per_question_scalars(EVAL / "judge", GOLD, tag=tag, n_seeds=1))
        tr = JA.trap_correct_refusal_rate(EVAL / "judge", GOLD, tag=tag, n_seeds=1)
        if tr.get("trap_correct_refusal_rate") is not None:
            traps.append(tr["trap_correct_refusal_rate"])
    return ({"backbone": _mean_tables(bb), "judge": _mean_tables(jj)},
            statistics.mean(traps) if traps else 0.0)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: verdict_fullcorpus.py <variant_prefix e.g. vmq|vsr>")
    pfx = sys.argv[1]
    var_tags = [f"{pfx}_r{k}" for k in (1, 2, 3)]

    base, base_trap = _side(BASE_TAGS)
    var, var_trap = _side(var_tags)
    v = HL.verdict(
        base, var,
        noise_floor_run_mean_sd=HL.FULLCORPUS_NOISE_FLOOR_RUN_MEAN_SD,
        trap_refusal_baseline=base_trap, trap_refusal_variant=var_trap,
        n_boot=4000, n_perm=4000, seed=0,
    )

    # Trap-gate epsilon post-adjustment (drill D1): the scalar trap gate at epsilon=0 lets a
    # single 1/9 judge flip veto a win. Re-judge ONLY that gate at TRAP_EPSILON; every other
    # gate stays strict. win_trap_adjusted recomputes verdict()'s own conjunction with the
    # adjusted trap gate.
    gates = v["gates"]
    trap_g = gates.get("trap_correct_refusal_rate", {})
    trap_ok_adj = bool(trap_g.get("evaluable")) and (
        not trap_g.get("regressed") or float(trap_g.get("worse_by", 1.0)) <= TRAP_EPSILON
    )
    others_clean = all(
        (g.get("evaluable") and not g.get("regressed"))
        for name, g in gates.items() if name != "trap_correct_refusal_rate"
    )
    win_adj = bool(
        v["headline"]["improved"] and v["passes_noise_floor"] and v["survives_jackknife"]
        and others_clean and trap_ok_adj
    )

    bh, vh = v.get("baseline_headline"), v.get("variant_headline")
    out = {
        "variant": pfx,
        "baseline_headline": round(bh, 4), "variant_headline": round(vh, 4),
        "delta": round(vh - bh, 4),
        "headline_improved_on_movers": v["headline"]["improved"],
        "n_movers": v["headline"].get("n_movers"),
        "passes_noise_floor": v["passes_noise_floor"],
        "survives_jackknife": v["survives_jackknife"],
        "trap_refusal": {"baseline": round(base_trap, 4), "variant": round(var_trap, 4),
                         "gate_strict": not trap_g.get("regressed"),
                         "gate_at_eps_0.10": trap_ok_adj},
        "gates": {name: ("OK" if (g.get("evaluable") and not g.get("regressed"))
                         else f"REGRESSED(worse_by={g.get('worse_by')})" if g.get("evaluable")
                         else "NOT-EVALUABLE")
                  for name, g in gates.items()},
        "win_strict": v["win"],
        "win_trap_adjusted": win_adj,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
