#!/usr/bin/env python3
"""MCP fan-out load test (issue #31) — reproduce + verify the handshake-starvation failure.

Mimics the 16-reviewer fan-out against the unified papervault MCP server: launches N concurrent
heavy `query` calls (each its OWN MCP session, exactly like a separate Codex/Claude reviewer)
while a PROBE loop repeatedly opens a FRESH session and times the `initialize`+`tools/list`
handshake — the thing a new reviewer needs before it gets any grounding tools. It also samples the
server process RSS. The metric that matters: does the handshake stay fast (single-digit seconds)
WHILE heavy queries are in flight, and does the server survive (no OOM/crash)?

Reads nothing but the running server on 127.0.0.1:8080. Prints a JSON summary. Compare two runs
(BEFORE the fix vs AFTER) — the handshake p90/max under load is the headline number.

    python scripts/loadtest_mcp.py --queries 8 --label before --pid <server_pid>

Query cost note: `query` runs the real KS retrieval (embed+rerank on the 3090) + synth via the
local gateway; retrieval alone keeps a session in-flight for minutes, which is what stresses the
loop — synth success/failure does not change the handshake measurement.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import timedelta

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = "http://127.0.0.1:8080/mcp"

# Grounding-style intents a reviewer would ask (rich intent, on-domain — real retrieval).
INTENTS = [
    "I'm reviewing a manuscript on PINN-based Parker transport inversion; what does the "
    "literature say about training stability of stiff-PDE physics-informed networks, especially "
    "adaptive collocation sampling?",
    "For a review of a GNN-PINN SEP forecasting paper, summarize known failure modes of "
    "multi-loss physics-constrained training where loss terms suppress each other, with mitigations.",
    "What is the cross-paper consensus on whether automatic-differentiation-only PINNs "
    "under-constrain solutions in sparse-collocation regimes? Supporting and contradicting evidence.",
    "Reviewing a multi-sensor space-weather inverse-problem method: recent work on graph neural "
    "networks combined with physics constraints, with method comparisons rather than a survey.",
    "Assessing a Voyager outer-heliosphere XPINN inversion cycle; what prior work exists on "
    "domain-decomposition PINNs for large-scale plasma transport, and their reported limitations?",
    "For a referee report on a magnetic reconnection ML surrogate, what are established benchmarks "
    "and known generalization pitfalls of data-driven reconnection-rate models?",
    "Reviewing a claim about turbulence-driven solar-wind heating; what is the observational and "
    "theoretical support for the proposed mechanism across the recent literature?",
    "Evaluating a paper on radiation-belt electron flux nowcasting with neural operators; what do "
    "prior studies report on out-of-distribution robustness during storm-time?",
]


async def _handshake_once(timeout_s: float = 120.0) -> tuple[float, str]:
    """Open a fresh session, time initialize()+list_tools(). Returns (seconds, 'ok'|'error:...')."""
    t0 = time.monotonic()
    try:
        async with streamablehttp_client(URL, sse_read_timeout=timedelta(seconds=timeout_s)) as (r, w, _):
            async with ClientSession(r, w) as s:
                await asyncio.wait_for(s.initialize(), timeout=timeout_s)
                tools = await asyncio.wait_for(s.list_tools(), timeout=timeout_s)
        return time.monotonic() - t0, ("ok" if tools.tools else "empty_tools")
    except Exception as e:  # noqa: BLE001
        return time.monotonic() - t0, f"error:{type(e).__name__}"


async def _query_once(intent: str, timeout_s: float = 900.0) -> tuple[float, str]:
    t0 = time.monotonic()
    try:
        async with streamablehttp_client(URL, sse_read_timeout=timedelta(seconds=timeout_s)) as (r, w, _):
            async with ClientSession(r, w) as s:
                await asyncio.wait_for(s.initialize(), timeout=120)
                res = await s.call_tool("query", {"intent": intent},
                                        read_timeout_seconds=timedelta(seconds=timeout_s))
                txt = res.content[0].text if res.content else ""
                busy = '"busy": true' in txt or '"busy":true' in txt
                return time.monotonic() - t0, ("busy" if busy else "ok")
    except Exception as e:  # noqa: BLE001
        return time.monotonic() - t0, f"error:{type(e).__name__}"


def _rss_mb(pid: int | None) -> float | None:
    if not pid:
        return None
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        return None


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    return {"n": len(xs), "min": round(xs[0], 3), "p50": round(p(0.5), 3),
            "p90": round(p(0.9), 3), "max": round(xs[-1], 3),
            "mean": round(statistics.mean(xs), 3)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=8, help="concurrent heavy query sessions (reviewers)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--pid", type=int, default=None, help="server PID for RSS sampling")
    ap.add_argument("--probe-interval", type=float, default=1.5)
    ap.add_argument("--idle-probes", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=600.0, help="cap the load window")
    ap.add_argument("--drain-seconds", type=float, default=20.0,
                    help="how long to wait for still-running queries after the window before reporting")
    args = ap.parse_args()

    # 1) idle baseline handshake
    idle = []
    for _ in range(args.idle_probes):
        dt, out = await _handshake_once()
        idle.append(dt)
        await asyncio.sleep(0.4)
    rss0 = _rss_mb(args.pid)

    # 2) launch N concurrent queries; probe handshake + sample RSS until they finish (or cap)
    q_tasks = [asyncio.create_task(_query_once(INTENTS[i % len(INTENTS)])) for i in range(args.queries)]
    launch = time.monotonic()
    probes: list[float] = []
    probe_errs: list[str] = []
    rss_peak = rss0 or 0.0
    while not all(t.done() for t in q_tasks):
        if time.monotonic() - launch > args.max_seconds:
            break
        dt, out = await _handshake_once()
        probes.append(dt)
        if out != "ok":
            probe_errs.append(out)
        r = _rss_mb(args.pid)
        if r:
            rss_peak = max(rss_peak, r)
        await asyncio.sleep(args.probe_interval)

    # drain queries (short bounded wait) so we can report outcomes without blocking on full synth
    drain_deadline = time.monotonic() + args.drain_seconds
    q_out = []
    for t in q_tasks:
        remaining = max(0.1, drain_deadline - time.monotonic())
        try:
            q_out.append(await asyncio.wait_for(asyncio.shield(t), timeout=remaining))
        except asyncio.TimeoutError:
            q_out.append((float("nan"), "still_running"))
        except Exception as e:  # noqa: BLE001
            q_out.append((float("nan"), f"error:{type(e).__name__}"))

    summary = {
        "label": args.label,
        "queries": args.queries,
        "handshake_idle_s": _stats(idle),
        "handshake_under_load_s": _stats(probes),
        "handshake_probe_errors": probe_errs,
        "query_outcomes": [o for _, o in q_out],
        "query_durations_s": [round(d, 1) for d, _ in q_out],
        "rss_start_mb": round(rss0, 1) if rss0 else None,
        "rss_peak_mb": round(rss_peak, 1) if rss_peak else None,
        "load_window_s": round(time.monotonic() - launch, 1),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
