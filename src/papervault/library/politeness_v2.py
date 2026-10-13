"""Pure, process-wide A2/D-D accounting; no transport or persistence side effects.

Windows count SEND timestamps in (now - seconds, now]. Admissions are atomic,
and in-flight tickets are separate from rate accounting. Envelope increases
are explicit configuration decisions, never a consequence of a successful GET.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Mapping


def _positive(value):
    return not isinstance(value, bool) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class Scope:
    service: str
    credential: str = ""
    ip: str = ""
    endpoint: str = ""

    def __post_init__(self):
        if not self.service or self.service != self.service.strip().lower():
            raise ValueError("scope service must be a nonempty normalized name")


@dataclass(frozen=True)
class Window:
    requests: int
    seconds: float

    def __post_init__(self):
        if (type(self.requests) is not int or self.requests <= 0
                or not _positive(self.seconds)):
            raise ValueError("window requires positive integer requests and finite seconds")


def _windows(windows):
    if (not isinstance(windows, tuple) or not windows
            or not all(isinstance(w, Window) for w in windows)
            or len({w.seconds for w in windows}) != len(windows)):
        raise ValueError("configure each applicable window exactly once")


def _cooldown(quiet_seconds, timeout_cluster, timeout_window):
    if (not _positive(quiet_seconds) or not _positive(timeout_window)
            or type(timeout_cluster) is not int or timeout_cluster <= 0):
        raise ValueError("cooldown requires positive quiet/window seconds and cluster count")


@dataclass(frozen=True)
class Policy:
    windows: tuple[Window, ...]
    concurrency: int = 1
    quiet_seconds: float = 60
    timeout_cluster: int = 3
    timeout_window: float = 60

    def __post_init__(self):
        _windows(self.windows)
        if type(self.concurrency) is not int or self.concurrency <= 0:
            raise ValueError("concurrency must be a positive integer")
        _cooldown(self.quiet_seconds, self.timeout_cluster, self.timeout_window)


@dataclass(frozen=True)
class Envelope:
    analogue: tuple[Window, ...] = ()
    initial: tuple[Window, ...] = ()
    hard_cap: tuple[Window, ...] = ()
    authorization: str = ""
    quiet_seconds: float = 60
    timeout_cluster: int = 3
    timeout_window: float = 60

    @property
    def configured(self):
        return bool(self.analogue and self.initial and self.hard_cap and self.authorization.strip())

    def __post_init__(self):
        _cooldown(self.quiet_seconds, self.timeout_cluster, self.timeout_window)
        if not self.configured:
            return  # incomplete configuration disables the scope
        for windows in (self.analogue, self.initial, self.hard_cap):
            _windows(windows)
        analogue = sorted(self.analogue, key=lambda w: w.seconds)
        starts = sorted(self.initial, key=lambda w: w.seconds)
        caps = sorted(self.hard_cap, key=lambda w: w.seconds)
        if (len(analogue) != len(starts) or len(starts) != len(caps)
                or any(start.seconds < documented.seconds
                       or start.requests > documented.requests
                       or start.requests / start.seconds > 0.5 * documented.requests / documented.seconds
                       for start, documented in zip(starts, analogue))
                or any(start.requests > cap.requests or start.seconds < cap.seconds
                       for start, cap in zip(starts, caps))):
            raise ValueError("envelope needs every analogue window, initial <=50%, and explicit caps")


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    retry_after: float = 0


class AdmissionDenied(Exception):
    def __init__(self, decision: Decision):
        self.decision = decision
        super().__init__(f"politeness admission denied: {decision.reason}")


@dataclass(frozen=True)
class Ticket:
    scope: Scope
    sequence: int
    owner: object


@dataclass(frozen=True)
class Event:
    at: float
    kind: str
    windows: tuple[Window, ...]


@dataclass(frozen=True)
class Snapshot:
    windows: tuple[Window, ...]
    sent: int
    in_flight: int
    positive_completions: int
    evidence: int
    cooldown_until: float
    ramp_stopped: bool
    events: tuple[Event, ...]
    configuration: Policy | Envelope


@dataclass(frozen=True)
class QuotaSnapshot:
    at: float
    windows: tuple[Window, ...]
    sends: tuple[float, ...]
    cooldown_until: float
    violations: int = 0
    handled: int = 0


@dataclass
class _State:
    config: Policy | Envelope
    windows: tuple[Window, ...]
    last_increase: float
    sends: deque = field(default_factory=deque)
    timeouts: deque = field(default_factory=deque)
    active: set = field(default_factory=set)
    sent: int = 0
    positives: int = 0
    evidence: int = 0
    cooldown_until: float = 0
    cooling: bool = False
    stopped: bool = False
    events: list[Event] = field(default_factory=list)
    quota_watermarks: dict = field(default_factory=dict)
    quota_violations: int = 0
    quota_handled: int = 0


class Ledger:
    """One ledger registry shared by all callers in the single-server process.

    gate() is advisory. note_sent() rechecks under the same lock as the debit;
    callers must obtain a ticket immediately before invoking their transport.
    Failed/uncertain sends stay in the windows, even after the executor drains.
    """

    def __init__(self, configurations: Mapping[Scope, Policy | Envelope | None], *,
                 clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.RLock()
        self._last_time = -math.inf
        self._sequence = 0
        self._owner = object()
        now = self._now()
        self._states = {}
        for scope, config in configurations.items():
            if config is None or isinstance(config, Envelope) and not config.configured:
                continue
            if not isinstance(scope, Scope) or not isinstance(config, (Policy, Envelope)):
                raise ValueError("invalid scope configuration")
            windows = config.initial if isinstance(config, Envelope) else config.windows
            state = _State(config, windows, now)
            state.stopped = (isinstance(config, Envelope)
                             and set(windows) == set(config.hard_cap))
            state.events.append(Event(now, "configured", windows))
            self._states[scope] = state

    def _now(self):
        now = self._clock()
        if not math.isfinite(now) or now < self._last_time:
            raise ValueError("accounting clock must be finite and monotonic")
        self._last_time = now
        return now

    def _refresh(self, state, now):
        from papervault.config import stoprules_v2_enabled
        if stoprules_v2_enabled():
            from .stop_rules import quota_violation_observations
            # Retain evidence before normal window pruning. Observations are
            # watermarked, so retained sends cannot relatch an acknowledged fault.
            for seconds, at, count in quota_violation_observations(state.windows, state.sends):
                evidence = (at, count)
                if evidence > state.quota_watermarks.get(seconds, (-math.inf, 0)):
                    state.quota_watermarks[seconds] = evidence
                    state.quota_violations += 1
        horizon = max(w.seconds for w in state.windows)
        while state.sends and state.sends[0] <= now - horizon:
            state.sends.popleft()
        while state.timeouts and state.timeouts[0] <= now - state.config.timeout_window:
            state.timeouts.popleft()
        if state.cooling and now >= state.cooldown_until:
            state.cooling = False
            state.timeouts.clear()
            state.events.append(Event(now, "resume", state.windows))

    def _decision(self, scope, now):
        state = self._states.get(scope)
        if state is None:
            return Decision(False, "disabled")
        self._refresh(state, now)
        if state.cooling:
            return Decision(False, "cooldown", state.cooldown_until - now)
        concurrency = 1 if isinstance(state.config, Envelope) else state.config.concurrency
        if len(state.active) >= concurrency:
            return Decision(False, "concurrency")
        waits = []
        for window in state.windows:
            active = [stamp for stamp in state.sends if stamp > now - window.seconds]
            if len(active) >= window.requests:
                waits.append(active[-window.requests] + window.seconds - now)
        return Decision(False, "window", max(waits)) if waits else Decision(True)

    def gate(self, scope: Scope) -> Decision:
        with self._lock:
            return self._decision(scope, self._now())

    def note_sent(self, scope: Scope) -> Ticket:
        with self._lock:
            now = self._now()
            decision = self._decision(scope, now)
            if not decision.allowed:
                raise AdmissionDenied(decision)
            state = self._states[scope]
            self._sequence += 1
            ticket = Ticket(scope, self._sequence, self._owner)
            state.sends.append(now)
            state.active.add(ticket)
            state.sent += 1
            return ticket

    def _cool(self, state, now, reason):
        repeated = state.cooling and state.cooldown_until == now + state.config.quiet_seconds
        state.cooling = True
        state.cooldown_until = now + state.config.quiet_seconds
        state.stopped = True
        state.evidence = 0
        if not repeated:
            state.events.append(Event(now, reason, state.windows))

    def note_response(self, ticket: Ticket, *, status_code: int) -> None:
        """Observe headers before body consumption; a first 429 stops at once."""
        with self._lock:
            now = self._now()
            state = self._states.get(ticket.scope)
            if state is None or ticket not in state.active:
                raise ValueError("response requires an outstanding send ticket")
            if status_code == 429:
                self._cool(state, now, "429")

    def enter_cooldown(self, scope: Scope, reason: str) -> None:
        """Explicit detector actuator, without inventing a response/send ticket."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("cooldown requires a reason")
        with self._lock:
            now = self._now()
            state = self._states[scope]
            self._refresh(state, now)
            if not state.cooling:
                self._cool(state, now, reason)
            elif state.events[-1].kind != reason:
                # Already in the resulting state: preserve its quiet deadline.
                state.events.append(Event(now, reason, state.windows))
            state.quota_handled = state.quota_violations

    def note_result(self, ticket: Ticket, *, status_code: int | None = None,
                    timed_out: bool = False, completed: bool = True) -> None:
        with self._lock:
            now = self._now()
            state = self._states.get(ticket.scope)
            if state is None or ticket not in state.active:
                raise ValueError("completion requires an outstanding send ticket")
            self._refresh(state, now)
            state.active.remove(ticket)
            if status_code == 429:
                self._cool(state, now, "429")
            elif timed_out:
                state.timeouts.append(now)
                if state.cooling or len(state.timeouts) >= state.config.timeout_cluster:
                    self._cool(state, now, "timeout_cluster")
            elif completed and status_code is not None and 200 <= status_code < 300:
                state.positives += 1
                if not state.cooling and not state.stopped:
                    state.evidence += 1

    def increase(self, scope: Scope, windows: tuple[Window, ...]) -> bool:
        """Apply caller-chosen values; never choose or automatically ramp limits."""
        _windows(windows)
        with self._lock:
            now = self._now()
            state = self._states.get(scope)
            if state is None or not isinstance(state.config, Envelope):
                return False
            self._refresh(state, now)
            if (state.stopped or state.cooling or not state.evidence
                    or now - state.last_increase < 86400):
                return False
            old = sorted(state.windows, key=lambda w: w.seconds)
            new = sorted(windows, key=lambda w: w.seconds)
            caps = sorted(state.config.hard_cap, key=lambda w: w.seconds)
            if (len(new) != len(old)
                    or not any(n.requests / n.seconds > o.requests / o.seconds for n, o in zip(new, old))
                    or any(not o.requests <= n.requests <= cap.requests
                           or not cap.seconds <= n.seconds <= o.seconds
                           or n.requests / n.seconds > 1.25 * o.requests / o.seconds
                           for n, o, cap in zip(new, old, caps))):
                return False
            state.windows = windows
            state.last_increase = now
            state.evidence = 0
            state.stopped = new == caps
            state.events.append(Event(now, "cap" if state.stopped else "increase", windows))
            return True

    def snapshot(self, scope: Scope) -> Snapshot:
        with self._lock:
            now = self._now()
            state = self._states[scope]
            self._refresh(state, now)
            return Snapshot(state.windows, state.sent, len(state.active), state.positives,
                            state.evidence, state.cooldown_until, state.stopped, tuple(state.events),
                            state.config)

    def quota_snapshots(self) -> dict[Scope, QuotaSnapshot]:
        """Detached send evidence for pure detectors; denials never debit a window."""
        with self._lock:
            now = self._now()
            for state in self._states.values():
                self._refresh(state, now)
            return {scope: QuotaSnapshot(now, state.windows, tuple(state.sends), state.cooldown_until,
                                         state.quota_violations, state.quota_handled)
                    for scope, state in self._states.items()}
