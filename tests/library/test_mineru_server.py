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
