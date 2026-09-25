"""Process-wide build-path breaker (#143): pause the background knowledge build while the upstream
LLM path is failing persistently, and resume it by itself.

2026-09-25 (#130): the relay ran out of balance for three hours and MaaS returned intermittent
400s; the build kept retrying at full rate — ~37k failed calls, roughly one a second, and a
flooded log. No document was lost (LightRAG re-queues FAILED-with-content docs every round and
`reconcile_healed` returns their ledger rows to `done`), so the harm is wasted calls and noise.
This breaker stops that: it watches the BUILD role only.

States:
  closed     every call goes through. Each upstream failure is stamped; when `threshold` of them
             fall inside the last `window_s` seconds (successes in between do not reset the count)
             the breaker OPENS.
  open       build-role calls fail fast with `BuildPausedError` without contacting the gateway,
             and the scheduler round starts no new build work. Results of calls still in flight
             from before the trip are ignored. After `cooldown_s` the next check moves it to
  half_open  one probe round: calls go through again; the first success CLOSES the breaker
             (window cleared), the first upstream failure RE-OPENS it for a fresh cool-down.

Every transition logs exactly one line (logger ``ks.store.build_breaker``); nothing is logged per
call. Counted as upstream failures: an HTTP 4xx/5xx status from the gateway/endpoint, a transport
error (connection refused, transport timeout, a cut stream), and the direct pool's exhaustion error
chained to one of those. NOT counted: a cancellation (the caller's own deadline — LightRAG's worker
timeout, shutdown), a caller-side bug, a missing-key configuration error, or the breaker's own
`BuildPausedError`.

Never on the query path: graph.build_llm consults it for the "build" route only — the query-path
keyword call (route "keyword") and decompose/synth (mimo_complete called directly) bypass it.

Env (read once, at first use): KS_BUILD_BREAKER_FAILURES (default 30; <=0 disables the breaker),
KS_BUILD_BREAKER_WINDOW_S (default 300), KS_BUILD_BREAKER_COOLDOWN_S (default 600).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Callable, Mapping, Optional

from openai import APIConnectionError

from papervault.knowledge.store.llm import StreamTruncated, _error_code

log = logging.getLogger("ks.store.build_breaker")

DEFAULT_FAILURES = 30
DEFAULT_WINDOW_S = 300.0
DEFAULT_COOLDOWN_S = 600.0

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class BuildPausedError(RuntimeError):
    """A build-role LLM call refused because the build-path breaker is open (#143)."""


def is_upstream_failure(exc: BaseException) -> bool:
    """True when ``exc`` (or an exception in its ``__cause__`` chain) is an upstream/gateway failure:
    an HTTP 4xx/5xx status, a transport error, or a stream cut short. Cancellations and every other
    error are the caller's, not the upstream's."""
    seen: set[int] = set()
    e: Optional[BaseException] = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, BuildPausedError):
            return False
        if isinstance(e, (APIConnectionError, StreamTruncated)):   # incl. APITimeoutError
            return True
        if isinstance(e, Exception):
            code = _error_code(e)
            if code is not None and 400 <= code <= 599:
                return True
        e = e.__cause__
    return False


def _env_number(env: Mapping[str, str], name: str, default: float, cast: Callable) -> float:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        log.warning("ignoring invalid %s=%r (using %s)", name, raw, default)
        return default


class BuildBreaker:
    def __init__(self, *, threshold: int = DEFAULT_FAILURES, window_s: float = DEFAULT_WINDOW_S,
                 cooldown_s: float = DEFAULT_COOLDOWN_S,
                 clock: Callable[[], float] = time.monotonic):
        self.threshold = int(threshold)
        self.window_s = float(window_s)
        self.cooldown_s = float(cooldown_s)
        self._clock = clock
        self._lock = threading.Lock()   # the build may run LLM calls from more than one thread
        self._failures: deque[float] = deque()
        self._state = CLOSED
        self._open_until = 0.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "BuildBreaker":
        return cls(
            threshold=int(_env_number(env, "KS_BUILD_BREAKER_FAILURES", DEFAULT_FAILURES, int)),
            window_s=_env_number(env, "KS_BUILD_BREAKER_WINDOW_S", DEFAULT_WINDOW_S, float),
            cooldown_s=_env_number(env, "KS_BUILD_BREAKER_COOLDOWN_S", DEFAULT_COOLDOWN_S, float),
        )

    @property
    def enabled(self) -> bool:
        return self.threshold > 0

    @property
    def state(self) -> str:
        with self._lock:
            self._poll()
            return self._state

    def allows(self) -> bool:
        """May build work start now? False only while open; an elapsed cool-down turns this check
        into the half-open probe."""
        return self.state != OPEN

    def before_call(self) -> None:
        """Gate for one build-role LLM call: raise ``BuildPausedError`` while open."""
        if not self.allows():
            raise BuildPausedError(
                "build paused: upstream LLM path failing persistently (build-path breaker open, "
                f"retry after the {self.cooldown_s:.0f}s cool-down)")

    def record_success(self) -> None:
        with self._lock:
            self._poll()
            if self._state == HALF_OPEN:
                self._state = CLOSED
                self._failures.clear()
                log.info("build breaker CLOSED — probe build call succeeded; background build resumes")

    def record_failure(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._poll()
            now = self._clock()
            if self._state == OPEN:
                return   # a call from before the trip finishing late: already open
            if self._state == HALF_OPEN:
                self._trip(now)
                log.warning("build breaker RE-OPENED — probe build call failed upstream; "
                            "background build paused another %.0fs", self.cooldown_s)
                return
            self._failures.append(now)
            while self._failures and now - self._failures[0] > self.window_s:
                self._failures.popleft()
            if len(self._failures) >= self.threshold:
                n = len(self._failures)
                self._trip(now)
                log.warning("build breaker OPEN — %d upstream build-call failures within %.0fs; "
                            "background build paused %.0fs (query path unaffected)",
                            n, self.window_s, self.cooldown_s)

    def _trip(self, now: float) -> None:
        self._state = OPEN
        self._open_until = now + self.cooldown_s
        self._failures.clear()

    def _poll(self) -> None:
        if self._state == OPEN and self._clock() >= self._open_until:
            self._state = HALF_OPEN
            log.info("build breaker HALF-OPEN — cool-down over; allowing one probe build round")


_BREAKER: Optional[BuildBreaker] = None
_BREAKER_LOCK = threading.Lock()


def get_breaker() -> BuildBreaker:
    """The process-wide build-path breaker (configured from env on first use)."""
    global _BREAKER
    if _BREAKER is None:
        with _BREAKER_LOCK:
            if _BREAKER is None:
                _BREAKER = BuildBreaker.from_env()
    return _BREAKER
