#!/usr/bin/env python3
"""MiMo-judge: score the downstream-eval prompts via the LiteLLM gateway (NOT Claude subagents).

WHY THIS EXISTS (2026-06-14): the #5 variant loop was judged by Opus subagents, which kept dying
on Claude session/rate limits (220-wide fan-out → server-side throttle; never finished). The user
cut the judge over to MiMo through the local LiteLLM gateway (94/94 keys live, high concurrency,
zero Claude quota). RULER INVARIANT: the judge model must be identical on BOTH sides of every
verdict, so this script re-judges the 3 BASELINE tags too — the old Opus seeds are archived under
judge/_opusjudge_archive_*/ for audit, never silently mixed with MiMo seeds.

Each judge/<tag>/<qid>.prompt.txt is a SELF-CONTAINED filled contract (intent + answer + retrieved
chunks + gold nuggets + output schema). We send it verbatim to MiMo, extract the JSON object, then
RECOMPUTE every derived scalar from the check/nugget lists (citation_support_precision,
citation_recall, nugget scores+recall) so a model arithmetic slip can never fail the
self-consistency validator — the model only has to make the per-check/per-nugget JUDGEMENTS; the
identities are enforced here. Then validate via judge_aggregate.validate_judge_json and write
<qid>.seed0.json.

  uv run python experiments/eval/judge_mimo.py --all [--concurrency 60] [--force]
  uv run python experiments/eval/judge_mimo.py vmq_r1 vsr_r3 [--force]

--force overwrites existing seeds (the cutover run uses it); without it, a seed that already
parses+validates is skipped (idempotent resume after a partial run / rate-limit blip).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent.parent))  # repo root, for `papervault.eval`
from papervault.eval import judge_aggregate as JA  # noqa: E402

JUDGE = EVAL / "judge"
GOLD = EVAL / os.getenv("JUDGE_GOLD", "gold_v2.jsonl")  # env-override for new gold frames (gold_v4 / gold_newq)
ALL_TAGS = ["fullcorpus_baseline", "fullcorpus_r2", "fullcorpus_r3",
            "vmq_r1", "vmq_r2", "vmq_r3", "vsr_r1", "vsr_r2", "vsr_r3"]

GATEWAY = os.getenv("KS_GATEWAY_URL", "http://127.0.0.1:4000/v1")
VKEY = os.getenv("KS_VIRTUAL_KEY", "")
MODEL = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")
MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "32000"))
MAX_ATTEMPTS = int(os.getenv("JUDGE_MAX_ATTEMPTS", "2"))  # was 4; lowered so a stuck hard-Q judge fails fast → idempotent resume catches it (overnight robustness)

_SYSTEM = (
    "You are an impartial downstream-answer evaluation judge. Read the contract and the filled "
    "inputs, follow the contract EXACTLY (including its tightened rules: a compound claim is "
    "'entail' only if EVERY conjunct is supported by a retrieved chunk; a multi-component gold "
    "nugget missing a load-bearing component is at most 'partial'; a refuse-then-explain hybrid "
    "on a trap is NOT a clean refusal). Use ONLY the retrieved chunks as the evidence base. "
    "Output ONE single valid JSON object that matches the contract's Output schema — no markdown "
    "code fences, no prose before or after, no commentary. Just the JSON object."
)

_COV = {"covered": 1.0, "partial": 0.5, "missing": 0.0}


# ---------------------------------------------------------------- JSON extraction
def _extract_json(raw: str) -> dict:
    """Pull the judge JSON object out of a model reply that may carry fences or stray text."""
    s = raw.strip()
    # strip a leading ```json / ``` fence if present
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.DOTALL)
    if fence:
        s = fence.group(1)
    try:
        return json.loads(s)
    except Exception:
        pass
    # brace-match scan: find the widest balanced {...} that parses
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = s[start:i + 1]
                        try:
                            return json.loads(cand)
                        except Exception:
                            break
        start = s.find("{", start + 1)
    raise ValueError("no parseable JSON object in model reply")


# ---------------------------------------------------------------- scalar recompute
def _recompute(d: dict, *, is_trap: bool) -> dict:
    """Enforce the contract's self-consistency identities so the model's per-item JUDGEMENTS are
    authoritative and its arithmetic is irrelevant. Mirrors judge_aggregate.validate_judge_json."""
    # citation_support_precision = (#entail)/len(checks); 0 checks -> 1.0
    checks = d.get("citation_checks")
    if not isinstance(checks, list):
        checks = []
        d["citation_checks"] = checks
    n_entail = sum(1 for c in checks if c.get("verdict") == "entail")
    d["citation_support_precision"] = (n_entail / len(checks)) if checks else 1.0

    # citation_recall = n_claims_with_citation / n_substantive_claims; clamp cited<=claims; 0 -> 1.0
    n_claims = d.get("n_substantive_claims")
    n_cited = d.get("n_claims_with_citation")
    n_claims = n_claims if isinstance(n_claims, int) and n_claims >= 0 else 0
    n_cited = n_cited if isinstance(n_cited, int) and n_cited >= 0 else 0
    n_cited = min(n_cited, n_claims)
    d["n_substantive_claims"] = n_claims
    d["n_claims_with_citation"] = n_cited
    d["citation_recall"] = (n_cited / n_claims) if n_claims else 1.0

    # nugget scores forced to their coverage map; nugget_recall = mean (or trap identity on empty)
    nugs = d.get("nugget_judgements")
    if not isinstance(nugs, list):
        nugs = []
        d["nugget_judgements"] = nugs
    for nj in nugs:
        cov = nj.get("coverage")
        if cov in _COV:
            nj["score"] = _COV[cov]
    if nugs:
        d["nugget_recall"] = sum(_COV.get(nj.get("coverage"), 0.0) for nj in nugs) / len(nugs)
    else:
        # empty list: a CLEAN refusal earns 1.0; a substantively-answered (failed) trap earns 0.0
        d["nugget_recall"] = 1.0 if d.get("trap_correct_refusal") is True else 0.0

    # relevance must be an int 0..100
    relv = d.get("relevance", 0)
    try:
        d["relevance"] = max(0, min(100, int(round(float(relv)))))
    except Exception:
        d["relevance"] = 0

    # faithfulness clamp to [0,1]
    f = d.get("faithfulness", 0.0)
    try:
        d["faithfulness"] = max(0.0, min(1.0, float(f)))
    except Exception:
        d["faithfulness"] = 0.0
    return d


def _load_gold() -> dict:
    g = {}
    for line in GOLD.read_text().splitlines():
        line = line.strip()
        if line:
            o = json.loads(line)
            g[o["qid"]] = o
    return g


# ---------------------------------------------------------------- per-question judging
async def _judge_one(client, sem, tag, qid, gold_entry, force, counters):
    out = JUDGE / tag / f"{qid}.seed0.json"
    n_gold = (len(gold_entry.get("nuggets") or []) or None)
    is_trap = not (gold_entry.get("gold_keys") or [])
    if out.exists() and not force:
        try:
            JA.validate_judge_json(json.load(open(out)), n_gold_nuggets=n_gold)
            counters["skip"] += 1
            return (tag, qid, "skip", "")
        except Exception:
            pass  # invalid existing seed -> rejudge
    prompt = (JUDGE / tag / f"{qid}.prompt.txt").read_text()
    last_err = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": _SYSTEM},
                              {"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=MAX_TOKENS,
                )
            raw = resp.choices[0].message.content or ""
            d = _extract_json(raw)
            d.setdefault("qid", qid)
            d = _recompute(d, is_trap=is_trap)
            JA.validate_judge_json(d, n_gold_nuggets=n_gold)
            out.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            counters["ok"] += 1
            return (tag, qid, "ok", f"attempt={attempt}")
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            await asyncio.sleep(min(2.0 * attempt, 8.0))
    counters["fail"] += 1
    return (tag, qid, "fail", last_err)


async def _run(tags, concurrency, force):
    if not VKEY:
        raise SystemExit("KS_VIRTUAL_KEY not set — export it or run under the KS env")
    gold = _load_gold()
    qids = list(gold.keys())
    jobs = [(t, q) for t in tags for q in qids]
    client = AsyncOpenAI(base_url=GATEWAY, api_key=VKEY,
                         timeout=float(os.getenv("JUDGE_CALL_TIMEOUT", "300")), max_retries=0)
    sem = asyncio.Semaphore(concurrency)
    counters = {"ok": 0, "skip": 0, "fail": 0}
    print(f"judging {len(jobs)} (tag,qid) over {len(tags)} tags @ concurrency {concurrency} "
          f"via {GATEWAY} model={MODEL} force={force}", flush=True)
    tasks = [asyncio.create_task(_judge_one(client, sem, t, q, gold[q], force, counters))
             for (t, q) in jobs]
    done = 0
    fails = []
    for fut in asyncio.as_completed(tasks):
        tag, qid, status, note = await fut
        done += 1
        if status == "fail":
            fails.append((tag, qid, note))
        if done % 25 == 0 or status == "fail":
            print(f"  [{done}/{len(jobs)}] ok={counters['ok']} skip={counters['skip']} "
                  f"fail={counters['fail']}  last={tag}/{qid}:{status}", flush=True)
    print(f"DONE ok={counters['ok']} skip={counters['skip']} fail={counters['fail']}", flush=True)
    for t, q, n in fails:
        print(f"  FAIL {t}/{q}: {n}", flush=True)
    return counters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", help="tags to judge (or --all)")
    ap.add_argument("--all", action="store_true", help="judge all 9 frame tags")
    ap.add_argument("--concurrency", type=int, default=60)
    ap.add_argument("--force", action="store_true", help="overwrite existing seeds")
    a = ap.parse_args()
    tags = ALL_TAGS if a.all else a.tags
    if not tags:
        raise SystemExit("give tags or --all")
    bad = [t for t in tags if not (JUDGE / t).is_dir()]
    if bad:
        raise SystemExit(f"unknown tag dirs: {bad}")
    asyncio.run(_run(tags, a.concurrency, a.force))


if __name__ == "__main__":
    main()
