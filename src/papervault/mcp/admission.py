"""Load control for the heavy MCP tools (issue #28, layers 1+2).

Layer 1 — per-session in-flight cap: each MCP session may have at most
``PAPERVAULT_SESSION_INFLIGHT`` (default 2) heavy calls executing; excess calls
WAIT on that session's own semaphore (a client queues behind itself, never
starving other sessions of the shared GPU lanes).

Layer 2 — bounded admission with an honest busy answer: at entry, projected
wait is estimated as ``max(0, depth - lanes) * rolling_avg_service_time``.
Above ``PAPERVAULT_ADMISSION_MAX_WAIT_S`` (default 600; ``0`` disables the
layer) the call is NOT queued — it returns a STRUCTURED normal result::

    {"status": "busy", "busy": true, "reason": "...", "queue_depth": 7,
     "expected_wait_s": 780, "retry_after_s": 840}

Fairness (issue #99): ``depth`` counts only calls that are EXECUTING (holding
a session permit) plus the arriving call. A call still waiting on its own
session's semaphore is not counted, so one client's burst queues behind that
client and never raises the depth — or causes the refusal — another session
sees; each session adds at most ``PAPERVAULT_SESSION_INFLIGHT`` to it.

Own-session queueing (issue #137): layer 2 judges only a call whose session has
a free slot, i.e. one that would start EXECUTING and raise the depth. A call
whose session already has ``PAPERVAULT_SESSION_INFLIGHT`` heavy calls in the
wrapper waits behind that session and is never refused — it only takes a slot
its own session frees, so it never raises the depth beyond what that session
already contributes.

An LLM caller reschedules itself well on such an answer; an opaque 20-minute
stall it handles badly. The rolling average is fed by real completions (last
20 per tool, seeded conservatively before data exists).

Scope: ``query`` and ``search_papers`` only — ``get_paper`` is sub-second and
never load-controlled. Installed UNDER the access log (access log wraps this),
so MCPCALL durations remain the caller-experienced truth (queue wait included)
and busy answers still produce an MCPCALL line. All knobs env-tunable;
thresholds are expected to be re-set from MCPCALL production data (issue #28).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

log = logging.getLogger("papervault.mcp.admission")

HEAVY_TOOLS = ("query", "search_papers")

_SESSION_INFLIGHT = int(os.getenv("PAPERVAULT_SESSION_INFLIGHT", "2"))
_MAX_WAIT_S = float(os.getenv("PAPERVAULT_ADMISSION_MAX_WAIT_S", "600"))
_LANES = int(os.getenv("PAPERVAULT_ADMISSION_LANES", "2"))
# Conservative pre-data seeds (2026-07-18 MCPCALL observations).
_SEED_AVG_S = {"query": 300.0, "search_papers": 320.0}

# Units note (by design): layer 1 caps a session's TOTAL heavy budget (query +
# search_papers share the session's permits); layer 2's depth/busy model is per-tool.
_session_sems: dict[int, asyncio.Semaphore] = {}
_session_active: dict[int, int] = {}             # tasks inside the wrapper per session key
_depth: dict[str, int] = {}                      # per-tool EXECUTING (queued excluded, #99)
_recent: dict[str, deque] = {}                   # per-tool completed durations


def _session_key(kwargs: dict[str, Any]) -> int:
    for v in kwargs.values():
        if isinstance(v, Context):
            try:
                return id(v.session)
            except Exception:  # noqa: BLE001 — unbound Context: shared bucket
                return 0
    return 0


def _avg_service_s(tool: str) -> float:
    hist = _recent.get(tool)
    if hist:
        return sum(hist) / len(hist)
    return _SEED_AVG_S.get(tool, 300.0)


def _record(tool: str, dur_s: float) -> None:
    _recent.setdefault(tool, deque(maxlen=20)).append(dur_s)


def _busy_answer(tool: str, depth: int, expected_wait_s: float) -> dict[str, Any]:
    retry = int(expected_wait_s) + 60
    return {
        "status": "busy",
        "busy": True,
        "reason": (
            f"{tool} is load-limited right now: {depth - 1} calls are already running and the "
            f"projected wait (~{int(expected_wait_s // 60)} min) exceeds the service's "
            "honest-wait threshold. Nothing is wrong — do other work and retry after "
            f"retry_after_s ({retry}) seconds."
        ),
        "queue_depth": depth,
        "expected_wait_s": int(expected_wait_s),
        "retry_after_s": retry,
    }


def _wrap(name: str, fn):
    async def admitted(**kwargs):
        skey = _session_key(kwargs)
        # A session already at its cap queues this call behind ITSELF (layer 1): it
        # adds nothing to the executing depth, so layer 2 does not judge it (#137).
        if _MAX_WAIT_S > 0 and _session_active.get(skey, 0) < _SESSION_INFLIGHT:
            # Executing calls ahead + this one. Calls queued on a session semaphore
            # are deliberately absent: a backlog waits behind its own session (#99).
            depth = _depth.get(name, 0) + 1
            projected = max(0, depth - _LANES) * _avg_service_s(name)
            if projected > _MAX_WAIT_S:
                log.warning("ADMISSION busy tool=%s depth=%d expected_wait=%.0fs",
                            name, depth, projected)
                return _busy_answer(name, depth, projected)
        _session_active[skey] = _session_active.get(skey, 0) + 1
        t_enter = time.monotonic()
        try:
            sem = _session_sems.setdefault(skey, asyncio.Semaphore(_SESSION_INFLIGHT))
            async with sem:
                waited = time.monotonic() - t_enter
                if waited > 1.0:
                    log.info("ADMISSION wait tool=%s waited=%.0fs (session cap)", name, waited)
                _depth[name] = _depth.get(name, 0) + 1
                try:
                    t0 = time.monotonic()
                    result = await fn(**kwargs)
                    _record(name, time.monotonic() - t0)
                    return result
                finally:
                    _depth[name] = _depth.get(name, 1) - 1
        finally:
            # Evict the session's semaphore once NOTHING (holder or waiter) references
            # its key — long-lived servers must not accumulate one entry per dead session.
            n = _session_active.get(skey, 1) - 1
            if n <= 0:
                _session_active.pop(skey, None)
                _session_sems.pop(skey, None)
            else:
                _session_active[skey] = n

    return admitted


def install_admission(mcp: FastMCP) -> None:
    """Wrap the heavy tools with layers 1+2. Missing internals degrade to a warning."""
    try:
        tools = mcp._tool_manager._tools
    except AttributeError:
        log.warning("admission NOT installed: FastMCP tool-manager internals changed")
        return
    wrapped = []
    for name in HEAVY_TOOLS:
        tool = tools.get(name)
        if tool is None or not getattr(tool, "is_async", True):
            continue
        tool.fn = _wrap(name, tool.fn)
        wrapped.append(name)
    log.info("admission installed on %s (session cap %d, lanes %d, max wait %.0fs%s)",
             ", ".join(wrapped) or "NOTHING", _SESSION_INFLIGHT, _LANES, _MAX_WAIT_S,
             "" if _MAX_WAIT_S > 0 else " — layer 2 DISABLED")
