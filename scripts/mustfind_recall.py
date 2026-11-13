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
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", d)
    d = re.sub(r"^doi:\s*", "", d).strip()
    return d or None


def _norm_title(t: str | None) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", t or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)  # punctuation/unicode -> SPACE (hyphens must not fuse words)
    return re.sub(r"\s+", " ", t).strip()


def _title_match(a: str, b: str) -> bool:
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return False
    # containment only when the shorter side is substantial (>=4 words) — a 3-word
    # normalized title inside an unrelated longer one is an overcount vector.
    if min(len(na.split()), len(nb.split())) >= 4 and (na in nb or nb in na):
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


async def _search_once(intent: str, *, timeout_s: float = 900.0) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    current_query = intent
    async with streamablehttp_client(
        "http://127.0.0.1:8080/mcp",
        sse_read_timeout=timedelta(seconds=max(630.0, timeout_s + 30.0)),
    ) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            while True:
                remaining = deadline - time.monotonic()
                try:
                    res = await s.call_tool(
                        "search_papers",
                        {"query": current_query},
                        read_timeout_seconds=timedelta(seconds=min(600.0, max(5.0, remaining))),
                    )
                except Exception as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"search_papers timed out after {timeout_s:.0f}s waiting for {current_query}"
                        ) from exc
                    raise
                if getattr(res, "isError", False):
                    detail = res.content[0].text if getattr(res, "content", None) else "MCP tool error"
                    raise RuntimeError(f"search_papers MCP error: {detail}")
                if not getattr(res, "content", None):
                    raise RuntimeError("search_papers returned empty MCP content")
                data = json.loads(res.content[0].text)
                status = data.get("status")
                if status == "ok":
                    return data.get("results") or []
                if status == "running":
                    search_id = data.get("search_id")
                    if not isinstance(search_id, str) or not search_id:
                        raise RuntimeError(
                            f"search_papers returned status='running' without search_id: {data}"
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"search_papers timed out after {timeout_s:.0f}s waiting for {search_id}"
                        )
                    retry_after = float(data.get("retry_after_s") or 5.0)
                    await asyncio.sleep(min(max(0.1, retry_after), remaining))
                    current_query = search_id
                    continue
                detail = data.get("reason") or data.get("message") or data
                raise RuntimeError(f"search_papers failed with status={status!r}: {detail}")


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
            err_label: str | None = None
            try:
                served: list[dict] | None = await _search_once(intents[qid])
            except Exception as exc:  # noqa: BLE001
                err_label = f"{type(exc).__name__}: {exc}"
                served = None
            dt = time.monotonic() - t0
            if served is None or not served:
                # An empty serve or transient non-ok status (e.g. busy) is retried once;
                # if still failing/empty, QUARANTINE the row (excluded from the mean).
                why = err_label or "EMPTY SERVE"
                print(f"{qid:<28} {why} ({dt:.0f}s) — retrying once", flush=True)
                await asyncio.sleep(30)
                t0 = time.monotonic()
                err_label = None
                try:
                    served = await _search_once(intents[qid])
                except Exception as exc:  # noqa: BLE001
                    err_label = f"{type(exc).__name__}: {exc}"
                    served = None
                dt = time.monotonic() - t0
            if served is None:
                fh.write(json.dumps({"qid": qid, "error": err_label,
                                     "latency_s": round(dt, 1)}) + "\n")
                fh.flush()
                print(f"{qid:<28} QUARANTINED ({err_label})", flush=True)
                continue
            if not served:
                fh.write(json.dumps({"qid": qid, "error": "empty_serve_after_retry",
                                     "latency_s": round(dt, 1)}) + "\n")
                fh.flush()
                print(f"{qid:<28} QUARANTINED (empty serve twice)", flush=True)
                continue
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
    if recalls:
        print(f"pools: {len(recalls)}  mean={sum(recalls)/len(recalls):.4f}  min={min(recalls):.4f}")
    else:
        print("pools: 0 (all quarantined)")
    print(f"results: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
