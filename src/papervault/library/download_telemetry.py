"""Bounded, recording-only download spans; no URLs, credentials or response bodies.

UTC timestamps align independent actors; monotonic clocks measure elapsed time.
Browser spans describe fetch calls, not process creation/exit. Cancellation may
release a coroutine's permit before its executor thread actually ends.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from uuid import uuid4

log = logging.getLogger(__name__)
# Existing outcomes and new member-thread records must remain complete JSONL lines.
manifest_lock = threading.RLock()


@dataclass(frozen=True)
class Observation:
    library: object
    key: str
    run_id: str
    actor: str


@dataclass(frozen=True)
class Span:
    span_id: str
    started_at: str
    clock: float
    source: str
    member: str


_current: ContextVar[Observation | None] = ContextVar("download_observation", default=None)
_span: ContextVar[Span | None] = ContextVar("download_span", default=None)
_slot: ContextVar[Span | None] = ContextVar("download_slot", default=None)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_run_id() -> str:
    return uuid4().hex


@contextmanager
def observation(library, key: str = "", run_id: str | None = None, actor: str = "download"):
    token = _current.set(Observation(library, key, run_id or new_run_id(), actor))
    try:
        yield
    finally:
        _current.reset(token)


def emit(phase: str, mark: str = "point", **fields) -> None:
    """New recording failures never change acquisition/control-flow behavior."""
    current = _current.get()
    if current is None:
        return
    state = _span.get()
    event = {"event": "download_telemetry", "key": current.key[:128],
             "run_id": current.run_id[:32], "actor": current.actor[:32],
             "phase": phase[:64], "mark": mark[:16], "at": utc_now(),
             "span_id": state.span_id if state else "",
             "source": state.source if state else "",
             "member": state.member if state else "",
             "process_id": os.getpid(), "thread_id": threading.get_ident()}
    # Only scalar fields; labels/statuses bounded independently of upstream data.
    event.update({key: value[:64] if isinstance(value, str) else value
                  for key, value in fields.items()
                  if value is None or isinstance(value, (str, bool, int, float))})
    try:
        current.library.log(event)
    except Exception:
        log.warning("download telemetry record unavailable (%s/%s)", phase, mark)


def enrich(event: dict, library) -> dict:
    current = _current.get()
    if (current is None or current.library is not library
            or event.get("key") != current.key or event.get("event") == "download_telemetry"):
        return event
    timing = {"run_id": current.run_id, "at": utc_now()}
    slot = _slot.get()
    if slot is not None:
        timing.update(slot_id=slot.span_id, slot_started_at=slot.started_at,
                      slot_ended_at=timing["at"],
                      slot_duration_s=max(0.0, time.monotonic() - slot.clock))
    return {**event, "timing": timing}


@contextmanager
def span(phase: str, *, source: str = "", member: str = ""):
    """Emit balanced start/end marks even for exceptions; yield terminal fields."""
    fields = {}
    if _current.get() is None:
        yield fields
        return
    parent = _span.get()
    state = Span(uuid4().hex, utc_now(), time.monotonic(),
                 (source or (parent.source if parent else ""))[:64], member[:64])
    common = {"span_id": state.span_id, "parent_span_id": parent.span_id if parent else "",
              "source": state.source, "member": state.member, "started_at": state.started_at}
    token = _span.set(state)
    slot_token = _slot.set(state) if phase == "tier" else None
    emit(phase, "start", **common)
    status = "ok"
    try:
        yield fields
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
        raise
    finally:
        ended_at = utc_now()
        duration = max(0.0, time.monotonic() - state.clock)
        emit(phase, "end", **common, ended_at=ended_at, duration_s=duration,
             status=status, **fields)
        if slot_token is not None:
            _slot.reset(slot_token)
        _span.reset(token)


@asynccontextmanager
async def network_slot(semaphore, library, key: str = "", *, actor: str = "download"):
    """Observe the same acquire/release semantics as ``async with semaphore``."""
    current = _current.get()
    with observation(library, key, current.run_id if current else None, actor):
        with span("network_wait"):
            await semaphore.acquire()
        with span("network_hold"):
            try:
                yield
            finally:
                # Hold end is recorded after release, including cancellation/errors.
                semaphore.release()


def trace_download(fn):
    @wraps(fn)
    def traced(paper, library):
        # Preserve the established no-log/no-fetch idempotence path.
        if library.has_pdf(paper.key) and paper.arxiv_withdrawal is None:
            return fn(paper, library)
        current = _current.get()
        with observation(library, paper.key, current.run_id if current else None):
            with span("cascade") as terminal:
                try:
                    result = fn(paper, library)
                    terminal["pdf_returned"] = result
                    return result
                finally:
                    terminal["paper_status"] = paper.download_status
                    terminal["outcome_source"] = paper.download_source
    return traced


def browser_fetch(fetcher, *args, **kwargs):
    """The browser API owns lifecycle; observe its call without inspecting inputs."""
    with span("browser_call"):
        return fetcher.fetch(*args, **kwargs)
