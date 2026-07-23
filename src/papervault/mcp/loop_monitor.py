"""Event-loop lag monitor for the unified MCP server (issue #31).

The failure this guards against: heavy SYNCHRONOUS work executed on the single asyncio loop
(a GPU encode/predict, a big CPU section) blocks the loop for its whole duration, so the MCP
`initialize`/`tools/list` handshake a fresh client connection needs is starved — idle 0.2s
balloons to tens of seconds under the reviewer fan-out, and a systemd graceful-stop can't be
serviced (the loop is stuck) so the unit escalates to SIGKILL. That starvation is INVISIBLE in
the access log (which only times completed tool calls); this monitor makes it directly
observable.

Mechanism: a background task sleeps for a fixed ``interval`` and measures how much WALL time
actually elapsed. On a healthy loop the overshoot (``lag = elapsed - interval``) is ~0; when the
loop was blocked by sync work, the timer fires late by exactly the block duration, so the
overshoot IS the loop-block time. It logs LOUD on any single lag over ``warn_ms`` and emits a
periodic rollup (max/count) so an operator sees sustained starvation, not just spikes.

Cost is negligible (one short sleep per interval) and it is OFF by default — enable with
``PAPERVAULT_LOOP_LAG_MONITOR_MS`` = the warn threshold in ms (0/unset disables). Never raises
into the server: a monitor must not be able to take the process down.
"""
from __future__ import annotations

import asyncio
import logging
import os

log = logging.getLogger("papervault.mcp.loopmon")

_task: "asyncio.Task | None" = None


def _config() -> tuple[float, float, float]:
    """(warn_s, interval_s, rollup_s) from env. warn_s<=0 → disabled."""
    warn_ms = float(os.getenv("PAPERVAULT_LOOP_LAG_MONITOR_MS", "0"))
    interval_s = float(os.getenv("PAPERVAULT_LOOP_LAG_INTERVAL_MS", "50")) / 1000.0
    rollup_s = float(os.getenv("PAPERVAULT_LOOP_LAG_ROLLUP_S", "30"))
    return warn_ms / 1000.0, max(0.005, interval_s), max(1.0, rollup_s)


async def _run(warn_s: float, interval_s: float, rollup_s: float) -> None:
    loop = asyncio.get_running_loop()
    last = loop.time()
    window_start = last
    max_lag = 0.0
    over = 0          # samples over the warn threshold this window
    samples = 0
    log.info(
        "loop-lag monitor ON (warn=%.0fms interval=%.0fms rollup=%.0fs)",
        warn_s * 1000, interval_s * 1000, rollup_s,
    )
    while True:
        await asyncio.sleep(interval_s)
        now = loop.time()
        lag = now - last - interval_s      # overshoot beyond the intended sleep = loop-block time
        last = now
        samples += 1
        if lag > max_lag:
            max_lag = lag
        if lag >= warn_s:
            over += 1
            log.warning("LOOPLAG blocked=%.0fms (loop was stalled by sync work)", lag * 1000)
        if now - window_start >= rollup_s:
            if max_lag * 1000 >= 1.0:  # skip an all-quiet window
                log.info(
                    "LOOPLAG rollup window=%.0fs max=%.0fms over_%0.fms=%d/%d",
                    now - window_start, max_lag * 1000, warn_s * 1000, over, samples,
                )
            window_start, max_lag, over, samples = now, 0.0, 0, 0


async def start(log_=None) -> None:
    """Start the monitor once if PAPERVAULT_LOOP_LAG_MONITOR_MS>0. Idempotent + never raises."""
    global _task
    if _task is not None:
        return
    warn_s, interval_s, rollup_s = _config()
    if warn_s <= 0:
        return
    try:
        _task = asyncio.create_task(_run(warn_s, interval_s, rollup_s), name="mcp-loop-lag-monitor")
    except Exception:  # noqa: BLE001 — observability must never take the server down
        log.exception("loop-lag monitor failed to start (serving unaffected)")


async def stop() -> None:
    """Cancel the monitor (graceful shutdown). Never raises."""
    global _task
    if _task is None:
        return
    _task.cancel()
    try:
        await _task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass
    _task = None
