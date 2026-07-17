"""Variant verdict on the LONG-CONTEXT frame — the NEW baseline is lctx@60 (user, 2026-06-14).

  python -m papervault.eval.verdict_lctx <variant_prefix>   # e.g. somevariant → somevariant_r{1,2,3}

The #5 loop promoted the long-context config (multiquery + V-SR + MAX_TOTAL_TOKENS=300000 +
KS_MQ_ENABLE_RERANK=false, serving 60 chunks) to the CURRENT baseline. New variants are judged
against it, in-regime:

  baseline  : lctx_r1/r2/r3 (results + MiMo judge seeds)
  headline  : harmonic(paper_recall@SERVED_distinct, nugget_recall) — @served, not @12 (the synth
              now sees 60 chunks; @12 is slack — see 2026-06-14-metric-adjustment-decision.md)
  noise floor: headline.FULLCORPUS_LONGCTX_NOISE_FLOOR_RUN_MEAN_SD (0.0132 — 5× the short-context
              floor; longer answers over more chunks swing more run-to-run)
  abs gates : NOISE-AWARE epsilon per metric (the synth-derived rates carry real run-noise):
              hallucinated_rate ε=0.016 (measured baseline run-sd); gold_citation_recall ε=0
              (genuinely deterministic, run-sd 0.0); trap gate ε=0.10 (per-run swing, drill D1).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent.parent))
from papervault.eval import headline as HL  # noqa: E402
from papervault.eval import judge_aggregate as JA  # noqa: E402
from papervault.eval import stats as ST  # noqa: E402

GOLD = EVAL / os.getenv("VERDICT_GOLD", "gold_v4.jsonl")  # gold_v4 frame (56 Qs, under-expansion-fixed + 8 new hard)
BASE_TAGS = os.getenv("VERDICT_BASE_TAGS", "lctx_v4_r1,lctx_v4_r2,lctx_v4_r3").split(",")
RECALL_KEY = "paper_recall_at_served_distinct"
NOISE_FLOOR = HL.FULLCORPUS_LONGCTX_NOISE_FLOOR_RUN_MEAN_SD
ABSOLUTE_EPSILONS = {"hallucinated_rate": 0.016, "gold_citation_recall": 0.0}
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
        raise SystemExit("usage: verdict_lctx.py <variant_prefix>")
    pfx = sys.argv[1]
    # variant seeds env-configurable (default 1,2,3 = unchanged); e.g. VERDICT_VAR_SEEDS=1,2,3,4,5
    _seeds = [int(x) for x in os.getenv("VERDICT_VAR_SEEDS", "1,2,3").split(",")]
    var_tags = [f"{pfx}_r{k}" for k in _seeds]

    base, base_trap = _side(BASE_TAGS)
    var, var_trap = _side(var_tags)
    v = HL.verdict(
        base, var,
        noise_floor_run_mean_sd=NOISE_FLOOR,
        recall_key=RECALL_KEY,
        absolute_epsilons=ABSOLUTE_EPSILONS,
        trap_refusal_baseline=base_trap, trap_refusal_variant=var_trap,
        n_boot=4000, n_perm=4000, seed=0,
    )

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
        "variant": pfx, "frame": "long-context (baseline=lctx@60, headline=harm(@served,nugget))",
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
