"""Dashboard recall curve for the long-context regime (#5 loop) — does NOT touch the locked ruler.

The headline retrieval term stays paper_recall@12_distinct (H4 anti-gaming, comparability with the
0.42→0.55 history). But when synth is served 26-60+ chunks, @12 UNDER-credits gold surfaced at
served-ranks 13-60. This tool reports a FIXED-K distinct-recall shadow curve (@12 / @24 / @36 /
@served) + served chunk count, purely from the result dumps (judge-independent). Pair it with
nugget_recall (from judge seeds, if present): @12 flat + @served & nugget rising = a real
long-context win the headline structurally under-credits; @served rising + nugget flat = the deep
chunks are noise. K is FIXED (not =n_served) so the shadow cannot be gamed by merely serving more.

  uv run python experiments/eval/recall_curve.py <prefix>          # 3-run mean over <prefix>_r{1,2,3}
  uv run python experiments/eval/recall_curve.py <tag1> <tag2> ... # explicit tags, each averaged in
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent.parent))
from papervault.eval import backbone as BB  # noqa: E402
from papervault.eval import judge_aggregate as JA  # noqa: E402

# gold_v4 is the CURRENT promotion frame (56 Qs / 232 keys). The hardcoded gold_v2 (48 Qs / 166
# keys) was silently superseded — it dropped all 8 new hard Qs + carried stale keys for 12
# re-expanded Qs, under-crediting the @served recall by ~2-3x the noise floor on exactly the
# high-fanout Qs this curve exists to read. Env-overridable (RECALL_CURVE_GOLD) so it can't rot
# again on the next gold refresh — mirrors verdict_lctx.py's VERDICT_GOLD pattern.
GOLD = EVAL / os.getenv("RECALL_CURVE_GOLD", "gold_v4.jsonl")
K_GRID = [12, 24, 36]


def _gold() -> dict:
    return {json.loads(l)["qid"]: set(json.loads(l).get("gold_keys") or [])
            for l in GOLD.read_text().splitlines() if l.strip()}


def _recall_at(data: dict, gk: set, k):
    dp = set(BB.distinct_papers_from_chunks(data, top_n=k)) if k is not None \
        else set(BB.distinct_papers_from_chunks(data, top_n=None))
    return len(gk & dp) / len(gk) if gk else None


def _one_tag(tag: str, gold: dict) -> dict | None:
    p = EVAL / "results" / f"{tag}.jsonl"
    if not p.exists():
        return None
    res = {json.loads(l)["qid"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}
    cols = {f"@{k}": [] for k in K_GRID}
    cols["@served"] = []
    served = []
    for qid, gk in gold.items():
        if not gk:
            continue
        r = res.get(qid)
        if not r:
            continue
        data = r.get("data") or {}
        served.append(len(data.get("chunks") or []))
        for k in K_GRID:
            cols[f"@{k}"].append(_recall_at(data, gk, k))
        cols["@served"].append(_recall_at(data, gk, None))
    out = {c: statistics.mean(v) for c, v in cols.items() if v}
    out["served_med"] = sorted(served)[len(served) // 2] if served else 0
    return out


def _nugget_trap(tags: list[str], gold: dict):
    nrs, traps = [], []
    for t in tags:
        try:
            jj = JA.per_question_scalars(EVAL / "judge", GOLD, tag=t, n_seeds=1)
            nr = jj.get("nugget_recall", {})
            if nr:
                nrs.append(sum(nr.values()) / len(nr))
            tr = JA.trap_correct_refusal_rate(EVAL / "judge", GOLD, tag=t, n_seeds=1)
            if tr.get("trap_correct_refusal_rate") is not None:
                traps.append(tr["trap_correct_refusal_rate"])
        except Exception:
            pass
    return (statistics.mean(nrs) if nrs else None), (statistics.mean(traps) if traps else None)


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit("usage: recall_curve.py <prefix> | <tag...>")
    groups: dict[str, list[str]] = {}
    if len(args) == 1 and not (EVAL / "results" / f"{args[0]}.jsonl").exists():
        pfx = args[0]
        groups[pfx] = [f"{pfx}_r{k}" for k in (1, 2, 3)]
    else:
        for a in args:
            groups[a] = [a]
    gold = _gold()
    hdr = f"{'group':22s} {'served':>6s} " + " ".join(f"{c:>8s}" for c in ["@12", "@24", "@36", "@served"]) + f" {'nugget':>7s} {'trap':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for g, tags in groups.items():
        per = [_one_tag(t, gold) for t in tags]
        per = [x for x in per if x]
        if not per:
            print(f"{g:22s}  (no results)")
            continue
        served = round(statistics.mean(x["served_med"] for x in per))
        row = {c: statistics.mean(x[c] for x in per if c in x) for c in ["@12", "@24", "@36", "@served"]}
        nug, trap = _nugget_trap(tags, gold)
        cells = " ".join(f"{row[c]:8.4f}" for c in ["@12", "@24", "@36", "@served"])
        print(f"{g:22s} {served:6d} {cells} {('%.4f'%nug) if nug is not None else '   -  ':>7s} {('%.3f'%trap) if trap is not None else '  -  ':>6s}")
    print("\n@12 = LOCKED headline retrieval term · @24/@36 = fixed-K distinct shadows · @served = all distinct papers reaching synth")


if __name__ == "__main__":
    main()
