"""Fill judge_prompt.md per question from a run's results + gold → judge/<tag>/<qid>.prompt.txt.

Deterministic mustache fill so the 3 judge seeds (and every variant's judge run) see identical,
well-formed inputs — the self-consistency the contract + judge_aggregate validation rely on.
The judge subagent then reads ONE <qid>.prompt.txt and returns the judge JSON (judge_prompt.md
output schema); it does NOT touch the live graph (all inputs are in the prompt).

Fills ALL gold questions present in the run (incl. traps — the judge scores trap_correct_refusal;
judge_aggregate drops traps from the quality tables but uses them for trap_correct_refusal_rate).

Run:  uv run python experiments/eval/build_judge_prompts.py <tag>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
_TEMPLATE = (EVAL / "judge_prompt.md").read_text()


def fill_prompt(rec: dict, g: dict) -> str:
    """Fill the judge_prompt.md mustache for ONE (result record, gold entry)."""
    chunks = "\n".join(
        f"--- chunk (paper_key={c.get('paper_key')}) ---\n{c.get('content', '') or ''}"
        for c in (rec.get("data", {}).get("chunks") or [])
    )
    nuggets = "\n".join(f"- (n{i}) {t}" for i, t in enumerate(g.get("nuggets") or []))
    is_trap = "true" if not (g.get("gold_keys") or []) else "false"

    out = _TEMPLATE
    # iteration blocks first (lambda replacement → content used verbatim, no backslash re-parsing)
    out = re.sub(r"\{\{#chunks\}\}.*?\{\{/chunks\}\}", lambda _m: chunks, out, flags=re.S)
    out = re.sub(r"\{\{#nuggets\}\}.*?\{\{/nuggets\}\}", lambda _m: nuggets, out, flags=re.S)
    scalars = {
        "{{qid}}": rec.get("qid", g.get("qid", "")),
        "{{intent}}": rec.get("intent", g.get("intent", "")),
        "{{answer}}": rec.get("answer") or "",
        "{{per_paper_relevance}}": json.dumps(g.get("per_paper_relevance") or {}, ensure_ascii=False),
        "{{expected_coverage_band}}": g.get("expected_coverage_band", ""),
        "{{is_trap}}": is_trap,
    }
    for k, v in scalars.items():
        out = out.replace(k, str(v))
    return out


def _load_jsonl(path: Path) -> dict:
    d = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            o = json.loads(line)
            d[o["qid"]] = o
    return d


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: build_judge_prompts.py <tag>")
    tag = sys.argv[1]
    gold_name = sys.argv[2] if len(sys.argv) > 2 else "gold.jsonl"  # optional gold override (e.g. gold_v2.jsonl)
    results = _load_jsonl(EVAL / "results" / f"{tag}.jsonl")
    gold = _load_jsonl(EVAL / (gold_name if "/" in gold_name else gold_name))
    outdir = EVAL / "judge" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped = []
    for qid, g in gold.items():
        rec = results.get(qid)
        if rec is None:
            skipped.append(qid)
            continue
        (outdir / f"{qid}.prompt.txt").write_text(fill_prompt(rec, g))
        n += 1
    print(f"wrote {n} judge prompts -> {outdir}" + (f" | skipped (no result): {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
