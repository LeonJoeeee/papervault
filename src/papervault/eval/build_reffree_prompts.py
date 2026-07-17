"""Fill the reference-free judge prompts per (intent, chunk) / (intent, A, B) → judge/<tag>/*.prompt.txt.

Deterministic mustache fill (same style as build_judge_prompts.py) so judge replays see identical,
well-formed inputs. The Claude subagent then reads ONE *.prompt.txt and returns the judge JSON
(precision_judge.md / pairwise_recall_judge.md output schema); it does NOT touch the live graph
(all inputs are in the prompt). REFERENCE-FREE: gold.jsonl is never read here — only the run's
results/<tag>.jsonl (intent + retrieved chunks).

Two modes:

  precision <tag>
    For each question in results/<tag>.jsonl, for EACH served chunk, fill precision_judge.md →
    judge/<tag>/<qid>.<chunk_id>.prompt.txt   (chunk_id = '<paper_key>#<ordinal>', no dedup).

  pairwise <baseline_tag> <variant_tag>
    For each qid present in BOTH runs, fill pairwise_recall_judge.md with A=baseline chunks,
    B=variant chunks → judge/<variant_tag>__vs__<baseline_tag>/<qid>.prompt.txt.

Run:
  python -m papervault.eval.build_reffree_prompts precision <tag>
  python -m papervault.eval.build_reffree_prompts pairwise  <baseline_tag> <variant_tag>

Then fan out the Claude judge over every *.prompt.txt (one subagent call per file), writing the
JSON next to it as <same-stem>.seed0.json (replays -> .seed1.json, ...). Score offline with
metric_reffree.precision_per_question / .pairwise_tally.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))
from metric_reffree import chunk_id, safe_stem  # noqa: E402  (sibling module)

_PRECISION_TEMPLATE = (EVAL / "precision_judge.md").read_text()
_PAIRWISE_TEMPLATE = (EVAL / "pairwise_recall_judge.md").read_text()


def _load_jsonl(path: Path) -> dict:
    d = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            o = json.loads(line)
            d[o["qid"]] = o
    return d


def _chunks(rec: dict) -> list[dict]:
    return rec.get("data", {}).get("chunks") or []


# ---- PRECISION: one prompt per (question, chunk) ----------------------------------------------

def fill_precision_prompt(rec: dict, c: dict, ordinal: int) -> str:
    """Fill precision_judge.md for ONE (result record, ONE chunk)."""
    cid = chunk_id(c.get("paper_key"), ordinal)
    out = _PRECISION_TEMPLATE
    # iteration-free template (single chunk): plain scalar replacement.
    scalars = {
        "{{qid}}": rec.get("qid", ""),
        "{{chunk_id}}": cid,
        "{{intent}}": rec.get("intent", ""),
        "{{paper_key}}": str(c.get("paper_key")),
        "{{chunk_text}}": c.get("content") or "",
    }
    for k, v in scalars.items():
        out = out.replace(k, str(v))
    return out


def build_precision(tag: str) -> int:
    results = _load_jsonl(EVAL / "results" / f"{tag}.jsonl")
    outdir = EVAL / "judge" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for qid, rec in results.items():
        for ordinal, c in enumerate(_chunks(rec)):
            cid = chunk_id(c.get("paper_key"), ordinal)
            (outdir / f"{qid}.{safe_stem(cid)}.prompt.txt").write_text(
                fill_precision_prompt(rec, c, ordinal)
            )
            n += 1
    print(f"wrote {n} precision prompts -> {outdir}")
    return n


# ---- PAIRWISE: one prompt per (question, A=baseline, B=variant) -------------------------------

def _render_chunks(chunks: list[dict]) -> str:
    return "\n".join(
        f"--- chunk (paper_key={c.get('paper_key')}) ---\n{c.get('content', '') or ''}"
        for c in chunks
    )


def fill_pairwise_prompt(qid: str, intent: str, a_chunks: list[dict], b_chunks: list[dict]) -> str:
    """Fill pairwise_recall_judge.md for ONE qid with A=baseline, B=variant served chunks."""
    out = _PAIRWISE_TEMPLATE
    a_block = _render_chunks(a_chunks)
    b_block = _render_chunks(b_chunks)
    out = re.sub(r"\{\{#a_chunks\}\}.*?\{\{/a_chunks\}\}", lambda _m: a_block, out, flags=re.S)
    out = re.sub(r"\{\{#b_chunks\}\}.*?\{\{/b_chunks\}\}", lambda _m: b_block, out, flags=re.S)
    out = out.replace("{{qid}}", qid).replace("{{intent}}", intent or "")
    return out


def build_pairwise(baseline_tag: str, variant_tag: str) -> int:
    base = _load_jsonl(EVAL / "results" / f"{baseline_tag}.jsonl")
    var = _load_jsonl(EVAL / "results" / f"{variant_tag}.jsonl")
    outdir = EVAL / "judge" / f"{variant_tag}__vs__{baseline_tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped = []
    for qid, vrec in var.items():
        brec = base.get(qid)
        if brec is None:
            skipped.append(qid)
            continue
        intent = vrec.get("intent") or brec.get("intent") or ""
        (outdir / f"{qid}.prompt.txt").write_text(
            fill_pairwise_prompt(qid, intent, _chunks(brec), _chunks(vrec))
        )
        n += 1
    print(f"wrote {n} pairwise prompts -> {outdir}"
          + (f" | skipped (not in baseline): {skipped}" if skipped else ""))
    return n


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("precision", "pairwise"):
        raise SystemExit(
            "usage:\n"
            "  build_reffree_prompts.py precision <tag>\n"
            "  build_reffree_prompts.py pairwise  <baseline_tag> <variant_tag>"
        )
    mode = sys.argv[1]
    if mode == "precision":
        if len(sys.argv) != 3:
            raise SystemExit("usage: build_reffree_prompts.py precision <tag>")
        build_precision(sys.argv[2])
    else:
        if len(sys.argv) != 4:
            raise SystemExit("usage: build_reffree_prompts.py pairwise <baseline_tag> <variant_tag>")
        build_pairwise(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
