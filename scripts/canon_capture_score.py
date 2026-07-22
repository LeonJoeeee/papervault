"""Capture-once / score-offline arbitration for the issue #37 canon-slots lever.

WHY not 6 live runs: each ``search_papers`` call fans out to rate-limited external
backends (S2 429 under contention with the live service) — 72 sequential live searches is
hours of noisy wall-clock. But the lever is a PURE, DETERMINISTIC reorder of the ranked
pool before the top-15 cut. So we capture the full ranked pool ONCE per intent (limit high
enough to include the buried tail) and apply the REAL ``_reserve_canon_slots`` offline for
each N. The OFF-vs-ON coverage delta is then VARIANCE-FREE by construction (identical
underlying pool — the 3-rep protocol existed only to tame live single-shot variance, which
cannot enter a paired delta computed on one shared capture). Precision is judged on the
served sets with the canonical relevance judge (bar 0.7), reusing per-paper scores across N
so the delta carries no judge noise either.

Fidelity: imports the SHIPPED ``_reserve_canon_slots`` — offline application is byte-identical
to the server running ``PAPERVAULT_CANON_RESERVED_SLOTS=N`` at ``effective_limit=15`` (proved
by tests/library/test_search_canon_slots.py). Capture runs against the frozen snapshot server
(NO_INGEST=1) so nothing mutates.

    # phase 1 (network): capture pools
    .venv/bin/python scripts/canon_capture_score.py capture --port 8099 --tag cap1
    # phase 2 (offline): score coverage + precision for each N
    .venv/bin/python scripts/canon_capture_score.py score --tag cap1 --limit 15 --slots 0 3 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from mustfind_recall import _served_matches                       # noqa: E402
from search_relevance_baseline import RELEVANT_BAR, _judge        # noqa: E402
from papervault import config                                     # noqa: E402
from papervault.library.mcp.server import _reserve_canon_slots    # noqa: E402  SHIPPED lever

RESULTS = ROOT / "src/papervault/eval/results"
POOLS = ROOT / "src/papervault/eval/gold_search_pools.jsonl"
INTENTS = ROOT / "src/papervault/eval/intents_v1.jsonl"


async def _search(intent: str, port: int) -> list[dict]:
    async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp",
                                     sse_read_timeout=timedelta(seconds=630)) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool("search_papers", {"query": intent},
                                    read_timeout_seconds=timedelta(seconds=600))
            return json.loads(res.content[0].text).get("results") or []


async def capture(args) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pools = {json.loads(l)["qid"]: json.loads(l) for l in open(POOLS) if l.strip()}
    intents = {json.loads(l)["qid"]: json.loads(l)["intent"] for l in open(INTENTS) if l.strip()}
    out = RESULTS / f"{args.tag}_capture.jsonl"
    with out.open("w") as fh:
        for qid in pools:
            t0 = time.monotonic()
            served = await _search(intents[qid], args.port)
            if not served:
                await asyncio.sleep(20)
                served = await _search(intents[qid], args.port)
            dt = time.monotonic() - t0
            fh.write(json.dumps({"qid": qid, "captured_n": len(served), "latency_s": round(dt, 1),
                                 "pool": served}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{qid:<28} captured={len(served):>3}  {dt:.0f}s", flush=True)
    print(f"captures: {out}")


def _wrap(pool: list[dict]) -> list[tuple[float, dict]]:
    # Strictly-decreasing synthetic scores preserve the captured rank order exactly (no ties),
    # so _reserve_canon_slots sees the same relevance ordering the server would.
    return [(1.0 - i * 1e-4, p) for i, p in enumerate(pool)]


async def score(args) -> None:
    cap = {json.loads(l)["qid"]: json.loads(l) for l in open(RESULTS / f"{args.tag}_capture.jsonl") if l.strip()}
    pools = {json.loads(l)["qid"]: json.loads(l) for l in open(POOLS) if l.strip()}
    intents = {json.loads(l)["qid"]: json.loads(l)["intent"] for l in open(INTENTS) if l.strip()}
    limit, slots = args.limit, args.slots

    # served set per (qid, N) via the SHIPPED lever
    served_by = {}   # (qid, N) -> served list[dict]
    for qid, row in cap.items():
        pool = row["pool"]
        for N in slots:
            reordered = _reserve_canon_slots(_wrap(pool), limit=limit, n=N)
            served_by[(qid, N)] = [c for _s, c in reordered[:limit]]

    # precision: judge the UNION of served papers per qid ONCE (dedup by doi|title), reuse per N
    judge = AsyncOpenAI(base_url=config.GATEWAY_URL, api_key=config.GATEWAY_KEY or "sk-")
    pscore = {}   # qid -> {doi_or_title: score}
    for qid in cap:
        union, seen = [], set()
        for N in slots:
            for p in served_by[(qid, N)]:
                k = (p.get("doi") or "").lower() or (p.get("title") or "").lower()
                if k and k not in seen:
                    seen.add(k); union.append(p)
        scores = await _judge(judge, intents[qid], union) if union else []
        pscore[qid] = {}
        for p, sc in zip(union, scores):
            k = (p.get("doi") or "").lower() or (p.get("title") or "").lower()
            pscore[qid][k] = sc

    def _prec(qid, served):
        if not served:
            return None
        good = 0
        for p in served:
            k = (p.get("doi") or "").lower() or (p.get("title") or "").lower()
            if pscore[qid].get(k, 0.0) >= RELEVANT_BAR:
                good += 1
        return good / len(served)

    print(f"\n{'N':>3} {'cov_mean':>9} {'cov_min':>8} {'P@served':>9}  {'promoted_landmarks':>18}")
    summary = {}
    for N in slots:
        covs, precs, promoted_hits = [], [], 0
        base_served = {qid: served_by[(qid, 0)] for qid in cap}
        for qid in cap:
            gold = pools[qid]["pool"]
            served = served_by[(qid, N)]
            hits = [g["title"] for g in gold if _served_matches(g, served)]
            covs.append(len(hits) / len(gold))
            pr = _prec(qid, served)
            if pr is not None:
                precs.append(pr)
            # landmarks the lever ADDED vs OFF baseline
            base_hits = {g["title"] for g in gold if _served_matches(g, base_served[qid])}
            promoted_hits += len(set(hits) - base_hits)
        cm, cmin = sum(covs) / len(covs), min(covs)
        pm = sum(precs) / len(precs) if precs else float("nan")
        summary[N] = {"cov_mean": cm, "cov_min": cmin, "p_served": pm,
                      "added_landmarks": promoted_hits}
        print(f"{N:>3} {cm:>9.4f} {cmin:>8.4f} {pm:>9.4f}  {promoted_hits:>18}")

    out = RESULTS / f"{args.tag}_score.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("--port", type=int, default=8099); c.add_argument("--tag", required=True)
    s = sub.add_parser("score"); s.add_argument("--tag", required=True)
    s.add_argument("--limit", type=int, default=15); s.add_argument("--slots", type=int, nargs="+", default=[0, 3, 5])
    args = ap.parse_args()
    asyncio.run(capture(args) if args.cmd == "capture" else score(args))


if __name__ == "__main__":
    main()
