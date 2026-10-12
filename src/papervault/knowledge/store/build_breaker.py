"""Process-wide build-path breaker (#143): pause the background knowledge build while the upstream
LLM path is failing persistently or during configured peak-price windows, and resume by itself.

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

Off-peak schedule (#209; env also read once at first use):
  KS_BUILD_OFFPEAK_ONLY   default false; true/1/yes enables the schedule gate (false/0/no disables).
  KS_BUILD_PEAK_WINDOWS   default 09:00-12:00,14:00-18:00; comma-separated same-day HH:MM-HH:MM.
                         Start is inclusive, end exclusive; overlapping/adjacent windows merge.
  KS_BUILD_PEAK_DAYS      default Mon-Fri; comma-separated weekday names/ranges (e.g. Mon,Wed-Fri).
  KS_BUILD_PEAK_TZ        default Asia/Shanghai; an IANA timezone via zoneinfo.
  KS_BUILD_PEAK_HOLIDAYS  default empty; comma-separated YYYY-MM-DD dates excluded from peak.
Malformed schedule fields fall back individually to these documented defaults, with one WARNING
per configuration load listing the invalid fields. Invalid windows never mean an all-day pause.
The disabled default preserves current behavior; operators opt in via their service environment.
Schedule checks run BEFORE failure-state polling and never change failure counters or cooldowns.
Exactly one INFO logs each observed peak entry/resumption; repeated checks are silent.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from datetime import date, datetime, time as wall_time, timezone
from typing import Callable, Mapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from openai import APIConnectionError

from papervault.knowledge.store.llm import StreamTruncated, _error_code

log = logging.getLogger("ks.store.build_breaker")

DEFAULT_FAILURES = 30
DEFAULT_WINDOW_S = 300.0
DEFAULT_COOLDOWN_S = 600.0

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class BuildPausedError(RuntimeError):
    """A build-role LLM call deferred by the failure breaker (#143) or peak schedule (#209)."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_enabled(raw: str) -> bool:
    value = raw.strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise ValueError("expected true/1/yes or false/0/no")


def _parse_windows(raw: str) -> tuple[tuple[wall_time, wall_time], ...]:
    windows = []
    for item in raw.split(","):
        start, end = (part.strip() for part in item.strip().split("-"))
        if not all(re.fullmatch(r"[0-9]{2}:[0-9]{2}", part) for part in (start, end)):
            raise ValueError("expected HH:MM-HH:MM")
        start, end = wall_time.fromisoformat(start), wall_time.fromisoformat(end)
        if start >= end:
            raise ValueError("peak windows must start before they end on the same day")
        windows.append((start, end))
    merged: list[tuple[wall_time, wall_time]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _parse_days(raw: str) -> frozenset[int]:
    names = {name.lower(): i for i, name in enumerate(
        ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))}
    days = set()
    for item in raw.split(","):
        parts = [part.strip().lower() for part in item.split("-")]
        if len(parts) not in (1, 2) or any(part not in names for part in parts):
            raise ValueError("expected weekday names or ranges, e.g. Mon-Fri")
        start, end = names[parts[0]], names[parts[-1]]
        days.update((start + i) % 7 for i in range((end - start) % 7 + 1))
    return frozenset(days)


def _parse_holidays(raw: str) -> frozenset[date]:
    if not raw.strip():
        return frozenset()
    holidays = set()
    for item in raw.split(","):
        value = item.strip()
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValueError("expected YYYY-MM-DD")
        holidays.add(date.fromisoformat(value))
    return frozenset(holidays)


class BuildSchedule:
    """Local peak windows with an injectable aware wall clock, separate from failure state."""

    def __init__(self, *, enabled: bool = False,
                 windows: tuple[tuple[wall_time, wall_time], ...] = (
                     (wall_time(9), wall_time(12)), (wall_time(14), wall_time(18))),
                 days: frozenset[int] = frozenset(range(5)), tz: ZoneInfo | None = None,
                 holidays: frozenset[date] = frozenset(), now: Callable[[], datetime] | None = None):
        self.enabled = enabled
        self.windows = windows
        self.days = days
        self.tz = tz if tz is not None else ZoneInfo("Asia/Shanghai")
        self.holidays = holidays
        self._now = now if now is not None else _utc_now
        self._paused = False
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ, *,
                 now: Callable[[], datetime] | None = None) -> "BuildSchedule":
        fields = (
            ("enabled", "KS_BUILD_OFFPEAK_ONLY", "false", _parse_enabled),
            ("windows", "KS_BUILD_PEAK_WINDOWS", "09:00-12:00,14:00-18:00", _parse_windows),
            ("days", "KS_BUILD_PEAK_DAYS", "Mon-Fri", _parse_days),
            ("tz", "KS_BUILD_PEAK_TZ", "Asia/Shanghai", ZoneInfo),
            ("holidays", "KS_BUILD_PEAK_HOLIDAYS", "", _parse_holidays),
        )
        values, invalid = {}, []
        for key, name, default, parse in fields:
            raw = env.get(name, default)
            try:
                values[key] = parse(raw)
            except (ValueError, ZoneInfoNotFoundError):
                values[key] = parse(default)
                invalid.append(f"{name}={raw!r} (using {default!r})")
        if invalid:
            log.warning("invalid KS build schedule config: %s", "; ".join(invalid))
        return cls(**values, now=now)

    def pause_until(self) -> datetime | None:
        """Return the current peak's end, or None; log each observed transition just once."""
        if not self.enabled:
            return None
        with self._lock:
            local = self._now().astimezone(self.tz)
            resume = None
            if local.weekday() in self.days and local.date() not in self.holidays:
                for start, end in self.windows:
                    if start <= local.time() < end:
                        resume = datetime.combine(local.date(), end, tzinfo=self.tz).replace(fold=local.fold)
                        break
            paused = resume is not None
            if paused != self._paused:
                self._paused = paused
                if paused:
                    log.info("build schedule entered DeepSeek peak window; %s", self.message(resume))
                else:
                    log.info("build schedule off-peak resumed; background build schedule allows work")
            return resume

    @staticmethod
    def message(resume: datetime) -> str:
        return "build paused: DeepSeek peak window (off-peak resumes " + resume.strftime("%H:%M %Z") + ")"


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
                 clock: Callable[[], float] = time.monotonic, schedule: BuildSchedule | None = None):
        self.threshold = int(threshold)
        self.window_s = float(window_s)
        self.cooldown_s = float(cooldown_s)
        self._clock = clock
        self._lock = threading.Lock()   # the build may run LLM calls from more than one thread
        self._failures: deque[float] = deque()
        self._state = CLOSED
        self._open_until = 0.0
        self.schedule = schedule if schedule is not None else BuildSchedule()

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "BuildBreaker":
        return cls(
            threshold=int(_env_number(env, "KS_BUILD_BREAKER_FAILURES", DEFAULT_FAILURES, int)),
            window_s=_env_number(env, "KS_BUILD_BREAKER_WINDOW_S", DEFAULT_WINDOW_S, float),
            cooldown_s=_env_number(env, "KS_BUILD_BREAKER_COOLDOWN_S", DEFAULT_COOLDOWN_S, float),
            schedule=BuildSchedule.from_env(env),
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
        """May build work start? Check the schedule before polling the failure cooldown."""
        return self.schedule.pause_until() is None and self.state != OPEN

    def before_call(self) -> None:
        """Defer a build-role call during peak or while the failure breaker is open."""
        resume = self.schedule.pause_until()
        if resume is not None:
            raise BuildPausedError(self.schedule.message(resume))
        if self.state == OPEN:
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
