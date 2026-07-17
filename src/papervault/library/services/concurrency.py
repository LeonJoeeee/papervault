"""Async concurrency primitives shared across the MCP server.

This module owns the process-wide async coordination state used by the
MCP frontend: resource semaphores (network / LLM / GPU), a write-lock
for the Library, and a per-key in-flight registry that bridges
foreground requests with background workers so the same paper is never
fetched/extracted twice in parallel.

All primitives are module-level singletons; they live in the MCP server
process. Multi-process deployments would need a different coordination
layer (filesystem locks); we are intentionally single-process for v1.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Awaitable, Callable

log = logging.getLogger("papervault.library.concurrency")


# ---------- Queue priority constants (D11) ----------

# Foreground MCP-tool calls (get_paper) enqueue at PRIORITY_URGENT
# so a single user request jumps past a multi-hundred-paper backlog from a
# search-paper batch ingest. Background recovery scans and search candidates
# enqueue at PRIORITY_NORMAL. Lower number = higher priority (asyncio.PriorityQueue
# is min-heap).
PRIORITY_URGENT = 0
PRIORITY_NORMAL = 5
# Phase 22.1 (2026-05-22): long papers (>90 pages) get PRIORITY_LOW so they
# run only after all normal-sized papers finish. Avoids 1500-page books
# blocking the 75% short-paper backlog. ExtractQueue's recovery scan auto-
# detects pages and promotes appropriately. Foreground URGENT still jumps
# past everything.
PRIORITY_LOW = 10


# ---------- Extract dispatch pause-toggle via file (SDD §4.3) ----------
#
# 2026-06-06 (MinerU migration): the persistent MinerU2.5-Pro vLLM server owns
# its GPU 24/7, so ``ocr-pool.conf`` can no longer "free a card" by listing GPU
# indices — that is now done by ``systemctl --user stop
# paper-library-mineru.service`` (the server releases the card). ``ocr-pool.conf``
# is REDEFINED to a binary **pause-dispatch** toggle:
#
#   non-empty / absent → dispatch RUNS (the daemon admits new extractions)
#   empty ("")         → dispatch PAUSED (admit nothing; the queue holds, the
#                        in-flight set drains gracefully)
#
# The exact GPU-index list is no longer meaningful (the daemon doesn't select a
# GPU — the MinerU server unit owns ``CUDA_VISIBLE_DEVICES``); ANY non-empty
# content means "run", an empty file means "pause". An absent file → run (the
# default), matching the old env fallback.
#
# Examples:
#   echo "run" > <vault>/ocr-pool.conf   → dispatch runs
#   echo ""    > <vault>/ocr-pool.conf   → dispatch paused (drain + hold)
#   rm         <vault>/ocr-pool.conf      → dispatch runs (default)


def _vault_path() -> str:
    import os
    from papervault import config
    raw = (os.environ.get("PAPERVAULT_VAULT")
           or os.environ.get("PAPER_LIBRARY_PATH")
           or str(config.VAULT_PATH))
    return os.path.expanduser(raw)  # tilde-safe, matches Library.root + config.VAULT_PATH


def extract_dispatch_enabled() -> bool:
    """Whether the daemon should admit new extractions right now (SDD §4.3).

    Reads the redefined ``<vault>/ocr-pool.conf`` pause-dispatch toggle: an
    EMPTY file pauses dispatch (return False); a non-empty file or an absent
    file runs (return True). Content is no longer a GPU-index list — the
    persistent MinerU server owns its card; "free the GPU" is now ``systemctl
    stop`` of the server unit, not a conf edit.
    """
    import os
    pool_file = os.path.join(_vault_path(), "ocr-pool.conf")
    try:
        if os.path.exists(pool_file):
            return bool(open(pool_file).read().strip())
    except OSError:
        pass  # file unreadable → default to running
    return True


# ---------- Extract concurrency admission (SDD §4.1/§4.3) ----------
#
# Replaces the per-paper exclusive GPU pin (which would SERIALIZE what vLLM
# batches). A plain ``asyncio.Semaphore(_EXTRACT_CONCURRENCY)`` admits this many
# whole-PDF extractions in flight; ``acquire_extract_slot`` ALSO hands out a
# round-robin endpoint URL so multi-endpoint (3090 ↔ 5090) work spreads evenly.
# vLLM's server-side ``--max-num-seqs`` is the hard backstop; this knob plus the
# bounded ~3.5 GiB transient headroom (baked into the gmu choice) are the other
# two layers. Steady-state has ONE endpoint, so the round-robin degenerates to
# the same URL — harmless.

# Concurrency is read from the env ONCE here (mirrors ``extract._EXTRACT_CONCURRENCY``;
# the env var is the single source of truth, default 8 per SDD §4.1).
import os as _os  # noqa: E402

_EXTRACT_CONCURRENCY = int(
    _os.environ.get("PAPER_LIBRARY_EXTRACT_CONCURRENCY", "8"))

extract_slots = asyncio.Semaphore(_EXTRACT_CONCURRENCY)

# Round-robin endpoint cursor (process-wide; guarded by a plain lock — the
# acquire path is async but the counter bump is trivial + non-blocking).
_endpoint_rr_lock = threading.Lock()
_endpoint_rr_index = 0


async def acquire_extract_slot():
    """Acquire one extraction permit + a round-robin endpoint (SDD §4.1).

    Awaits a free semaphore permit (caps in-flight extractions at
    ``_EXTRACT_CONCURRENCY``), then returns the next endpoint in round-robin
    order so multi-endpoint backfill / burst work spreads evenly. The caller
    MUST call :func:`release_extract_slot` in a ``finally`` to return the permit.

    Returns the chosen :class:`papervault.library.mineru_client.Endpoint`, or ``None``
    when no endpoint is configured (the caller falls back to the env endpoint
    set inside ``extract_mineru``). The permit is held regardless, so the
    concurrency cap is enforced even with a single env-configured endpoint.
    """
    global _endpoint_rr_index
    await extract_slots.acquire()
    from ..mineru_client import endpoints_from_env
    endpoints = endpoints_from_env()
    if not endpoints:
        return None
    with _endpoint_rr_lock:
        ep = endpoints[_endpoint_rr_index % len(endpoints)]
        _endpoint_rr_index += 1
    return ep


def release_extract_slot(endpoint=None) -> None:
    """Return one extraction permit acquired via :func:`acquire_extract_slot`."""
    extract_slots.release()


# ---------- Resource semaphores ----------

# Cap on concurrent outbound HTTP fetches (SS / OpenAlex / arxiv / unpaywall).
# 4 is safe for free-tier rate limits.
network_sem = asyncio.Semaphore(4)

# Cap on concurrent text-LLM calls (MiMo via litellm). Used by the
# remaining LLM paths inside paper-library: resolver fuzzy-match,
# search rerank, and the extract clarity / completeness judges
# (``review_extract`` / ``completeness_gate``). The previous
# 24 ceiling was sized for the InsightQueue worker pool that ran the
# 5-Q digest at 12 per endpoint × 2 endpoints — that queue was removed
# in Phase 28 (2026-05-24, route B), and the cap could in principle be
# tightened. Kept at 24 for now as a defensive head-room knob — the
# resolver/rerank paths are intermittent and never approach the cap,
# so the value isn't load-bearing.
llm_sem = asyncio.Semaphore(24)

# Dedicated cap for the post-OCR LLM gates (``review_extract`` + ``completeness_gate``),
# which now run OFF the event loop via ``asyncio.to_thread`` (the fix for the
# blocking-litellm-call-freezes-the-whole-loop pathology that idled the GPU ~46%).
# Bounds concurrent gate HTTP calls to respect MiMo's plan-level 429 ceiling — kept
# SEPARATE from ``llm_sem`` (the to_thread gate path does not pass through it). Default 8.
_GATE_CONCURRENCY = int(_os.environ.get("PAPER_LIBRARY_GATE_CONCURRENCY", "8"))
gate_sem = asyncio.Semaphore(_GATE_CONCURRENCY)

# Extract concurrency is now an explicit asyncio.Semaphore (``extract_slots``
# above, SDD §4.1) decoupled from GPU count: the whole-PDF MinerU call fans out
# per-page sub-requests that vLLM batches server-side via ``--max-num-seqs``.
# The old per-engine (marker/dots/chandra) subprocess-pool serialization +
# per-paper GPU pin are gone with the cascade (2026-06-06 MinerU migration).

# Library write serialization at the asyncio level. Pairs with the
# filelock at the filesystem level (which serializes across processes);
# this lock prevents in-process re-entrancy on Library.save().
lib_write_lock = asyncio.Lock()


# ---------- Per-key in-flight registry ----------

_in_flight: dict[str, asyncio.Future] = {}


async def with_dedup(key: str, work: Callable[[], Awaitable[Any]]) -> Any:
    """Run ``work()`` exactly once for ``key``; concurrent callers share the result.

    Used by frontend tools and background workers that operate on a single
    paper at a time. The first caller to claim a key starts the work; any
    later caller (foreground request or background worker) discovers the
    in-flight Future and awaits its result, avoiding duplicate fetches.

    ``work`` is a zero-argument callable that returns a coroutine — passed
    by name (not by value), so the coroutine isn't created at all when a
    duplicate caller subscribes to an existing Future.

    Exceptions propagate to every waiter on the Future.
    """
    existing = _in_flight.get(key)
    if existing is not None:
        return await existing

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _in_flight[key] = fut

    try:
        result = await work()
        if not fut.done():
            fut.set_result(result)
        return result
    except Exception as exc:
        if not fut.done():
            fut.set_exception(exc)
        raise
    finally:
        _in_flight.pop(key, None)


def in_flight_keys() -> list[str]:
    """Snapshot of currently-in-flight canonical keys (diagnostic)."""
    return list(_in_flight.keys())


def is_in_flight(key: str) -> bool:
    return key in _in_flight
