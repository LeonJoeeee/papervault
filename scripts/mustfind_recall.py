"""Must-find recall — the search coverage ruler (issue #22).

For each frozen gold pool (``gold_search_pools.jsonl``), run the standard live
``search_papers`` call and measure ``|served ∩ pool| / |pool|``. Matching is
DOI-first (normalized), title-fallback (normalized containment / high token
overlap). Sequential real calls — the caller-shaped measurement.

Usage:  .venv/bin/python scripts/mustfind_recall.py --tag mfr_$(date +%m%d)
Writes  src/papervault/eval/results/<tag>.jsonl + prints the summary card.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ROOT = Path(__file__).resolve().parent.parent


def _norm_doi(d: str | None) -> str | None:
    if not d:
        return None
    d = d.strip().lower()
    for pre in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(pre):
            d = d[len(pre):]
    return d or None


def _norm_title(t: str | None) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def _title_match(a: str, b: str) -> bool:
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    ta, tb = set(na.split()), set(nb.split())
    return len(ta & tb) / max(1, len(ta | tb)) >= 0.8


def _served_matches(pool_entry: dict, served: list[dict]) -> bool:
    pdoi = _norm_doi(pool_entry.get("doi"))
    for s in served:
        if pdoi and _norm_doi(s.get("doi")) == pdoi:
            return True
        if _title_match(pool_entry["title"], s.get("title")):
            return True
    return False


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pools", default=str(ROOT / "src/papervault/eval/gold_search_pools.jsonl"))
    ap.add_argument("--intents", default=str(ROOT / "src/papervault/eval/intents_v1.jsonl"))
    args = ap.parse_args()

    pools = {json.loads(l)["qid"]: json.loads(l) for l in open(args.pools) if l.strip()}
    intents = {json.loads(l)["qid"]: json.loads(l)["intent"] for l in open(args.intents) if l.strip()}
    out_path = ROOT / "src/papervault/eval/results" / f"{args.tag}.jsonl"

    recalls = []
    with out_path.open("w") as fh:
        for qid, pool_row in pools.items():
            t0 = time.monotonic()
            async with streamablehttp_client("http://127.0.0.1:8080/mcp",
                                             sse_read_timeout=timedelta(seconds=120)) as (r, w, _):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    res = await s.call_tool("search_papers", {"query": intents[qid]},
                                            read_timeout_seconds=timedelta(seconds=600))
                    served = json.loads(res.content[0].text).get("results") or []
            dt = time.monotonic() - t0
            pool = pool_row["pool"]
            hits = [p["title"] for p in pool if _served_matches(p, served)]
            misses = [p["title"] for p in pool if p["title"] not in hits]
            rec = len(hits) / len(pool)
            recalls.append(rec)
            fh.write(json.dumps({"qid": qid, "recall": round(rec, 4), "pool_n": len(pool),
                                 "served_n": len(served), "latency_s": round(dt, 1),
                                 "misses": misses,
                                 "served": [{"title": s0.get("title"), "doi": s0.get("doi")}
                                            for s0 in served]}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{qid:<28} recall={rec:.2f} ({len(hits)}/{len(pool)}) served={len(served)} {dt:.0f}s",
                  flush=True)

    print("\n=== MUST-FIND RECALL (search coverage) ===")
    print(f"pools: {len(recalls)}  mean={sum(recalls)/len(recalls):.4f}  min={min(recalls):.4f}")
    print(f"results: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
