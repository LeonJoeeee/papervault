"""Tests for the on-demand MinerU server controller (services/mineru_server.py).

The systemctl subprocess and the HTTP readiness probe are stubbed at the
``_run_systemctl`` / ``_ready`` seams so no real server or bus is touched. These
exercise the review-driven invariants: the transport-subclass contract, the
``ex:`` in-flight idle predicate, the cold-start wait + cooldown, the
``_stopping`` fast-path gate, and the OFF=no-op guarantee.
"""

from __future__ import annotations

import asyncio

import pytest

from papervault.library.mineru_client import MineruTransportError
from papervault.library.services import mineru_server as ms
from papervault.library.services.mineru_server import (
    MineruServerController,
    MineruServerUnavailable,
)


def _run(coro):
    return asyncio.run(coro)


class _FakeQueue:
    def __init__(self, n=0):
        self._n = n

    def qsize(self):
        return self._n


def _controller(ondemand=True, **kw):
    return MineruServerController(ondemand=ondemand, unit="x.service",
                                  base_url="http://127.0.0.1:30000", **kw)


def _stub_systemctl(c):
    """Record every systemctl verb; return rc=0, no stderr (bus OK)."""
    calls = []

    async def fake(verb):
        calls.append(verb)
        return (0, "")

    c._run_systemctl = fake
    return calls


def _stub_ready(c, seq):
    """_ready() yields each value of seq, then its last value forever."""
    it = iter(seq)
    last = [seq[-1]]

    async def fake():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    c._ready = fake


# --------------------------- OFF = strict no-op ----------------------------


def test_ensure_ready_offmode_is_noop():
    c = _controller(ondemand=False)
    calls = _stub_systemctl(c)
    _run(c.ensure_ready())                 # must not raise, must not touch systemctl
    assert calls == []


def test_monitor_loop_offmode_returns_immediately():
    c = _controller(ondemand=False)
    calls = _stub_systemctl(c)
    _run(c.monitor_loop(_FakeQueue(0)))    # returns at once, no bus probe / stop
    assert calls == []


def test_operator_ocr_session_offmode_is_noop():
    c = _controller(ondemand=False)
    calls = _stub_systemctl(c)

    async def drive():
        async with c.ocr_session():
            assert c._active_ocr == 0

    _run(drive())
    assert calls == []


# ----------------------------- ensure_ready --------------------------------


def test_ensure_ready_fast_path_when_already_ready():
    c = _controller()
    calls = _stub_systemctl(c)
    _stub_ready(c, [True])                  # already model-ready
    _run(c.ensure_ready())
    assert calls == []                      # never started anything


def test_ensure_ready_starts_then_waits_until_ready(monkeypatch):
    monkeypatch.setattr(ms, "_HEALTH_POLL", 0.0)
    c = _controller()
    calls = _stub_systemctl(c)
    # down on the fast-path + first slow-path check, then ready
    _stub_ready(c, [False, False, False, True])
    _run(c.ensure_ready())
    assert "is-active" in calls            # bus probe
    assert "start" in calls                # issued the start


def test_ensure_ready_timeout_raises_transport_and_sets_cooldown(monkeypatch):
    monkeypatch.setattr(ms, "_HEALTH_POLL", 0.0)
    monkeypatch.setattr(ms, "_READY_TIMEOUT", 0.05)   # never becomes ready in time
    c = _controller()
    _stub_systemctl(c)
    _stub_ready(c, [False])                # stays down forever
    with pytest.raises(MineruServerUnavailable) as ei:
        _run(c.ensure_ready())
    # CONTRACT (review fix #1): it IS a transport error → extract_md's C1 arm catches it.
    assert isinstance(ei.value, MineruTransportError)
    assert c._cooldown_until > 0.0          # cooldown armed


def test_cooldown_short_circuits_without_starting(monkeypatch):
    monkeypatch.setattr(ms, "_HEALTH_POLL", 0.0)
    monkeypatch.setattr(ms, "_READY_TIMEOUT", 0.05)
    c = _controller()
    calls = _stub_systemctl(c)
    _stub_ready(c, [False])
    with pytest.raises(MineruServerUnavailable):
        _run(c.ensure_ready())             # first attempt: starts, times out, arms cooldown
    starts_before = calls.count("start")
    with pytest.raises(MineruServerUnavailable):
        _run(c.ensure_ready())             # second: in cooldown → must NOT start again
    assert calls.count("start") == starts_before


def test_ensure_ready_no_bus_raises_transport(monkeypatch):
    monkeypatch.setattr(ms, "_HEALTH_POLL", 0.0)
    c = _controller()
    _stub_ready(c, [False])

    async def no_bus(verb):
        return (1, "Failed to connect to bus: No such file or directory")

    c._run_systemctl = no_bus
    with pytest.raises(MineruServerUnavailable):
        _run(c.ensure_ready())             # bus absent → can't start → transport


def test_failed_systemctl_start_enters_cooldown_immediately(monkeypatch):
    monkeypatch.setattr(ms, "_READY_TIMEOUT", 0.05)
    monkeypatch.setattr(ms, "_HEALTH_POLL", 0.0)
    c = _controller()
    _stub_ready(c, [False])

    async def failed_start(verb):
        return (5 if verb == "start" else 3, "unit not found" if verb == "start" else "")

    c._run_systemctl = failed_start
    with pytest.raises(MineruServerUnavailable, match="systemctl start failed"):
        _run(c.ensure_ready())
    assert c._cooldown_until > 0


# ------------------------------- idle logic --------------------------------


def test_idle_true_when_empty_and_no_extract_in_flight(monkeypatch):
    monkeypatch.setattr("papervault.library.services.concurrency.in_flight_keys",
                        lambda: ["dl:Foo2020"])           # a DOWNLOAD, not extract
    assert _controller()._idle(_FakeQueue(0)) is True


def test_not_idle_when_queue_nonempty(monkeypatch):
    monkeypatch.setattr("papervault.library.services.concurrency.in_flight_keys",
                        lambda: [])
    assert _controller()._idle(_FakeQueue(3)) is False


def test_not_idle_when_extract_in_flight(monkeypatch):
    # review fix #2: in_flight_keys is a FUNCTION returning ex:-prefixed keys.
    monkeypatch.setattr("papervault.library.services.concurrency.in_flight_keys",
                        lambda: ["ex:Bar2021"])
    assert _controller()._idle(_FakeQueue(0)) is False


# ----------------------- monitor stop decision -----------------------------


def test_monitor_stops_when_idle_past_timeout(monkeypatch):
    monkeypatch.setattr("papervault.library.services.concurrency.in_flight_keys", lambda: [])
    monkeypatch.setattr(ms, "_IDLE_TIMEOUT", 0.0)         # immediately "past idle"
    monkeypatch.setattr(ms, "_CHECK_INTERVAL", 0.0)
    c = _controller()
    calls = _stub_systemctl(c)
    _stub_ready(c, [True])                                # server is up
    c._bus_ok = True

    async def drive():
        task = asyncio.create_task(c.monitor_loop(_FakeQueue(0)))
        await asyncio.sleep(0.05)                         # let ≥1 iteration run
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _run(drive())
    assert "stop" in calls
    assert c._stopping is False                           # flag cleared after stop


def test_monitor_does_not_stop_while_extract_in_flight(monkeypatch):
    monkeypatch.setattr("papervault.library.services.concurrency.in_flight_keys",
                        lambda: ["ex:Live2024"])          # extraction running
    monkeypatch.setattr(ms, "_IDLE_TIMEOUT", 0.0)
    monkeypatch.setattr(ms, "_CHECK_INTERVAL", 0.0)
    c = _controller()
    calls = _stub_systemctl(c)
    _stub_ready(c, [True])
    c._bus_ok = True

    async def drive():
        task = asyncio.create_task(c.monitor_loop(_FakeQueue(0)))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _run(drive())
    assert "stop" not in calls                            # never stops under live work


def test_monitor_rechecks_idle_after_readiness_probe(monkeypatch):
    """An extract admitted during the monitor's probe must keep MinerU alive."""
    in_flight = []
    monkeypatch.setattr("papervault.library.services.concurrency.in_flight_keys",
                        lambda: in_flight.copy())
    monkeypatch.setattr(ms, "_IDLE_TIMEOUT", 0.0)
    monkeypatch.setattr(ms, "_CHECK_INTERVAL", 0.0)
    c = _controller()
    c._bus_ok = True
    c._last_activity = 0.0
    calls = _stub_systemctl(c)
    probe_entered = asyncio.Event()
    release_probe = asyncio.Event()
    next_cycle = asyncio.Event()
    real_sleep = asyncio.sleep
    cycles = 0

    async def sleep(delay):
        nonlocal cycles
        if delay == 0.0:
            cycles += 1
            if cycles > 1:
                next_cycle.set()
                await asyncio.Event().wait()
                return
        await real_sleep(delay)

    async def ready():
        if not probe_entered.is_set():
            probe_entered.set()
            await release_probe.wait()
        return True

    monkeypatch.setattr(ms.asyncio, "sleep", sleep)
    c._ready = ready

    async def drive():
        monitor = asyncio.create_task(c.monitor_loop(_FakeQueue(0)))
        try:
            await asyncio.wait_for(probe_entered.wait(), 1)
            in_flight.append("ex:NewPaper")
            await asyncio.wait_for(c.ensure_ready(), 1)
            release_probe.set()
            await asyncio.wait_for(next_cycle.wait(), 1)
        finally:
            monitor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await monitor

    _run(drive())
    assert "stop" not in calls


def test_monitor_forces_off_when_no_bus():
    c = _controller()

    async def no_bus(verb):
        return (1, "Failed to connect to bus")

    c._run_systemctl = no_bus
    _run(c.monitor_loop(_FakeQueue(0)))                  # returns after the bus probe
    assert c.ondemand is False                            # degraded to persistent


# ------------------------------- _stopping gate ----------------------------


def test_stopping_flag_blocks_fast_path(monkeypatch):
    """When a stop is in flight, ensure_ready must NOT take the lockless ready
    fast-path (teardown TOCTOU, review fix #6) — it falls through to the lock."""
    monkeypatch.setattr(ms, "_HEALTH_POLL", 0.0)
    c = _controller()
    c._stopping = True
    calls = _stub_systemctl(c)
    # _ready True everywhere; with _stopping set, the fast path is skipped and the
    # locked slow-path re-checks _ready()==True and returns WITHOUT starting.
    _stub_ready(c, [True])
    _run(c.ensure_ready())
    assert "start" not in calls


def test_fast_path_waits_when_stop_begins_during_probe():
    """A probe result from before teardown cannot admit a new OCR call."""
    c = _controller()
    entered = asyncio.Event()
    release_probe = asyncio.Event()

    async def ready():
        if not entered.is_set():
            entered.set()
            await release_probe.wait()
        return True

    c._ready = ready

    async def drive():
        task = asyncio.create_task(c.ensure_ready())
        await asyncio.wait_for(entered.wait(), 1)
        await c._lock.acquire()  # model a monitor that has started stopping
        c._stopping = True
        try:
            release_probe.set()
            await asyncio.sleep(0)
            assert not task.done(), "readiness escaped during teardown"
        finally:
            c._stopping = False
            c._lock.release()
        await asyncio.wait_for(task, 1)

    _run(drive())


def test_operator_ocr_session_keeps_server_active_until_ocr_finishes(monkeypatch):
    monkeypatch.setattr("papervault.library.services.concurrency.in_flight_keys", lambda: [])
    monkeypatch.setattr(ms, "_IDLE_TIMEOUT", 0.0)
    c = _controller()
    probes = []

    async def ready():
        probes.append(True)
        return True

    c._ready = ready

    async def drive():
        async with c.ocr_session():
            c._last_activity = 0.0
            assert c._past_idle(_FakeQueue(0)) is False
        c._last_activity = 0.0
        assert c._past_idle(_FakeQueue(0)) is True

    _run(drive())
    assert probes == [True]


# ------------- issue #134: self-heal on server-side image decode -------------
# A stale MinerU server (still running from a deleted venv) answers every page
# with 400 "Failed to load image". The client parks those papers uncharged; the
# controller restarts the unit once after N consecutive occurrences (on-demand
# ON) or logs one ERROR naming the cause and the unit (on-demand OFF).


def _record_systemctl(c):
    """Record (verb, _stopping-at-call-time); rc=0, bus OK."""
    calls = []

    async def fake(verb):
        calls.append((verb, c._stopping))
        return (0, "")

    c._run_systemctl = fake
    return calls


def _fail_n(c, n):
    """Report n image-decode failures from inside a loop; await any heal task."""
    async def drive():
        tasks = [c.note_image_decode_failure() for _ in range(n)]
        for t in {t for t in tasks if t is not None}:
            await t
    _run(drive())


def test_image_decode_self_heal_on_restarts_once_after_threshold(monkeypatch):
    monkeypatch.setattr(ms, "_IMAGE_DECODE_RESTART_AFTER", 3)
    monkeypatch.setattr(ms, "_HEALTH_POLL", 0.0)
    c = _controller()
    calls = _record_systemctl(c)
    _stub_ready(c, [True])

    _fail_n(c, 2)
    assert [v for v, _ in calls if v == "restart"] == []     # below threshold
    _fail_n(c, 1)
    restarts = [(v, stopping) for v, stopping in calls if v == "restart"]
    assert restarts == [("restart", True)]    # once, with the fast path closed
    assert c._stopping is False               # reopened after systemctl returned


def test_image_decode_self_heal_does_not_restart_again_without_a_success(
        monkeypatch, caplog):
    """The restart is attempted ONCE per unhealthy stretch: if the failures keep
    coming with no successful parse in between, log instead of a restart loop."""
    monkeypatch.setattr(ms, "_IMAGE_DECODE_RESTART_AFTER", 3)
    monkeypatch.setattr(ms, "_HEALTH_POLL", 0.0)
    c = _controller()
    calls = _record_systemctl(c)
    _stub_ready(c, [True])

    _fail_n(c, 3)
    with caplog.at_level("ERROR", logger=ms.log.name):
        _fail_n(c, 6)
    assert [v for v, _ in calls].count("restart") == 1
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "x.service" in errors[0].getMessage()


def test_image_decode_success_resets_streak_and_rearms_heal(monkeypatch):
    monkeypatch.setattr(ms, "_IMAGE_DECODE_RESTART_AFTER", 3)
    monkeypatch.setattr(ms, "_HEALTH_POLL", 0.0)
    c = _controller()
    calls = _record_systemctl(c)
    _stub_ready(c, [True])

    _fail_n(c, 2)
    c.note_parse_success()                    # a success breaks the streak
    _fail_n(c, 2)
    assert "restart" not in [v for v, _ in calls]
    _fail_n(c, 1)                             # 3 consecutive → restart
    c.note_parse_success()                    # healthy again → heal re-armed
    _fail_n(c, 3)
    assert [v for v, _ in calls].count("restart") == 2


def test_image_decode_self_heal_respects_start_failure_cooldown(monkeypatch):
    monkeypatch.setattr(ms, "_IMAGE_DECODE_RESTART_AFTER", 3)
    c = _controller()
    calls = _record_systemctl(c)
    _stub_ready(c, [True])
    c._cooldown_until = ms.time.monotonic() + 600.0

    _fail_n(c, 3)
    assert "restart" not in [v for v, _ in calls]


def test_image_decode_self_heal_failed_restart_arms_cooldown(monkeypatch):
    monkeypatch.setattr(ms, "_IMAGE_DECODE_RESTART_AFTER", 3)
    c = _controller()
    _stub_ready(c, [False])

    async def failing(verb):
        return (1 if verb == "restart" else 0, "")

    c._run_systemctl = failing
    _fail_n(c, 3)
    assert c._cooldown_until > ms.time.monotonic()


def test_image_decode_self_heal_waits_for_model_ready(monkeypatch):
    """After the restart the controller holds the lock until the model is ready,
    so ensure_ready callers wait instead of hitting a half-started server."""
    monkeypatch.setattr(ms, "_IMAGE_DECODE_RESTART_AFTER", 1)
    monkeypatch.setattr(ms, "_HEALTH_POLL", 0.0)
    c = _controller()
    _record_systemctl(c)
    _stub_ready(c, [False, False, True])
    _fail_n(c, 1)
    assert c._cooldown_until == 0.0           # became ready → no cooldown


def test_image_decode_offmode_logs_one_error_and_never_restarts(monkeypatch, caplog):
    monkeypatch.setattr(ms, "_IMAGE_DECODE_RESTART_AFTER", 3)
    c = _controller(ondemand=False)
    calls = _record_systemctl(c)
    with caplog.at_level("ERROR", logger=ms.log.name):
        _fail_n(c, 2)
        assert not [r for r in caplog.records if r.levelname == "ERROR"]
        _fail_n(c, 7)
    assert calls == []                        # OFF never touches systemctl
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    msg = errors[0].getMessage()
    assert "stale" in msg and "x.service" in msg and "restart" in msg


def test_image_decode_self_heal_task_error_is_logged_not_raised(monkeypatch, caplog):
    """An unexpected error inside the background restart must not surface as an
    unobserved task exception; it is logged and the fast path reopens."""
    monkeypatch.setattr(ms, "_IMAGE_DECODE_RESTART_AFTER", 1)
    c = _controller()
    _stub_ready(c, [True])

    async def boom(verb):
        if verb == "restart":
            raise OSError("fork failed")
        return (0, "")

    c._run_systemctl = boom
    with caplog.at_level("ERROR", logger=ms.log.name):
        _fail_n(c, 1)
    assert c._stopping is False
    assert any("self-heal" in r.getMessage() for r in caplog.records
               if r.levelname == "ERROR")
