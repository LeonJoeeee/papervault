"""Cooperative leases guard real controller transitions; no live unit is touched."""

import asyncio
import json

import pytest

from papervault.library.services import mineru_server as ms


FUTURE = "2099-01-01T00:00:00Z"
EXPIRED = "2000-01-01T00:00:00+00:00"


def _write_lease(path, expires_at=FUTURE):
    path.write_text(json.dumps({
        "holder": "MacroInvest", "pid": 99999999, "expires_at": expires_at,
    }), encoding="utf-8")


@pytest.fixture
def lane(monkeypatch, tmp_path):
    lease = tmp_path / "paper-library-mineru.lease"
    monkeypatch.setenv("PAPER_LIBRARY_MINERU_LEASE_FILE", str(lease))
    monkeypatch.setattr(ms, "_IDLE_TIMEOUT", 0.0)
    monkeypatch.setattr(ms, "_IMAGE_DECODE_RESTART_AFTER", 1)
    monkeypatch.setattr("papervault.library.services.concurrency.in_flight_keys", lambda: [])
    controller = ms.MineruServerController(
        ondemand=True, unit="paper-library-mineru.service", base_url="http://127.0.0.1:30000",
    )
    controller._last_activity = 0.0
    calls = []

    async def systemctl(verb):
        if verb in {"stop", "restart"}:
            assert controller._lock.locked()
            assert controller._stopping
        calls.append(verb)
        return 0, ""

    async def ready():
        return True

    controller._run_systemctl = systemctl
    controller._ready = ready
    return controller, calls, lease


async def _act(controller, verb, monkeypatch):
    if verb == "restart":
        task = controller.note_image_decode_failure()
        assert task is not None
        await task
    else:
        # Exercise exactly one sweep, then cancel at the next interval.
        sweeps = 0

        async def tick(_delay):
            nonlocal sweeps
            sweeps += 1
            if sweeps > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(ms.asyncio, "sleep", tick)
        with pytest.raises(asyncio.CancelledError):
            await controller.monitor_loop(asyncio.Queue())


@pytest.mark.parametrize("verb", ["stop", "restart"])
@pytest.mark.parametrize("expires_at", [FUTURE, "2099-01-01T00:00:00+00:00"])
async def test_future_lease_skips_action_and_logs_once(lane, monkeypatch, caplog, verb, expires_at):
    controller, calls, lease = lane
    _write_lease(lease, expires_at)
    with caplog.at_level("INFO", logger=ms.log.name):
        await _act(controller, verb, monkeypatch)
    assert calls == ["is-active"]
    messages = [r.getMessage() for r in caplog.records if "lease" in r.getMessage()]
    assert messages == [f"lease held by MacroInvest until {expires_at}; skipping {verb}"]
    assert all(r.levelname == "INFO" for r in caplog.records if "lease" in r.getMessage())
    assert controller._stopping is False
    assert controller._cooldown_until == 0.0


@pytest.mark.parametrize("verb", ["stop", "restart"])
@pytest.mark.parametrize("lease_state", ["missing", "expired"])
async def test_missing_or_expired_lease_preserves_action(lane, monkeypatch, caplog, verb, lease_state):
    controller, calls, lease = lane
    if lease_state == "expired":
        _write_lease(lease, EXPIRED)
    with caplog.at_level("INFO", logger=ms.log.name):
        await _act(controller, verb, monkeypatch)
    assert calls == ["is-active", verb]
    assert not [r for r in caplog.records if "lease" in r.getMessage()]
    assert controller._stopping is False


@pytest.mark.parametrize("verb", ["stop", "restart"])
@pytest.mark.parametrize("payload", [
    b"{", b"", b"\xff", b"null", b"[]", b"{}",
    json.dumps({"holder": 42, "pid": 123, "expires_at": FUTURE}).encode(),
    json.dumps({"holder": "MacroInvest", "pid": "123", "expires_at": FUTURE}).encode(),
    json.dumps({"holder": "MacroInvest", "pid": True, "expires_at": FUTURE}).encode(),
    json.dumps({"holder": "MacroInvest", "expires_at": FUTURE}).encode(),
    json.dumps({"holder": "MacroInvest", "pid": 123, "expires_at": 42}).encode(),
    json.dumps({"holder": "MacroInvest", "pid": 123, "expires_at": "tomorrow"}).encode(),
    json.dumps({"holder": "MacroInvest", "pid": 123,
                "expires_at": "2099-01-01T00:00:00"}).encode(),
    json.dumps({"holder": "MacroInvest", "pid": 123,
                "expires_at": "2099-01-01T00:00:00+08:00"}).encode(),
    b'{"holder":"MacroInvest","pid":123,"expires_at":"2099-01-01T00:00:00Z","x":NaN}',
    b"[" * 2000 + b"]" * 2000,
], ids=[
    "bad-json", "empty", "bad-utf8", "null", "array", "empty-object", "holder-type",
    "pid-string", "pid-bool", "missing-pid", "expiry-type", "bad-expiry", "naive", "non-utc",
    "non-json-constant", "too-deep",
])
async def test_malformed_lease_warns_once_and_proceeds(lane, monkeypatch, caplog, verb, payload):
    controller, calls, lease = lane
    lease.write_bytes(payload)
    await _act(controller, verb, monkeypatch)
    assert calls == ["is-active", verb]
    warnings = [r for r in caplog.records if "lease" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelname == "WARNING"


@pytest.mark.parametrize("verb", ["stop", "restart"])
async def test_unreadable_lease_warns_and_proceeds(lane, monkeypatch, caplog, verb):
    controller, calls, lease = lane
    lease.mkdir()  # Reading a directory as JSON fails even when tests run as root.
    await _act(controller, verb, monkeypatch)
    assert calls == ["is-active", verb]
    warnings = [r for r in caplog.records if "lease" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelname == "WARNING"


@pytest.mark.parametrize("verb", ["stop", "restart"])
async def test_default_lease_path_uses_xdg_runtime_dir(lane, monkeypatch, verb):
    controller, calls, lease = lane
    monkeypatch.delenv("PAPER_LIBRARY_MINERU_LEASE_FILE")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(lease.parent))
    _write_lease(lease)
    await _act(controller, verb, monkeypatch)
    assert calls == ["is-active"]


@pytest.mark.parametrize("verb", ["stop", "restart"])
async def test_lease_appearing_during_probe_guards_next_action(lane, monkeypatch, verb):
    controller, calls, lease = lane
    if verb == "stop":
        async def ready():
            _write_lease(lease)
            return True

        controller._ready = ready
    else:
        async def systemctl(probe):
            calls.append(probe)
            if probe == "is-active":
                _write_lease(lease)
            return 0, ""

        controller._run_systemctl = systemctl
    await _act(controller, verb, monkeypatch)
    assert calls == ["is-active"]


@pytest.mark.parametrize("verb", ["stop", "restart"])
async def test_lease_appearing_while_waiting_for_lock_guards_action(lane, monkeypatch, verb):
    controller, calls, lease = lane
    real_sleep = asyncio.sleep
    await controller._lock.acquire()
    action = asyncio.create_task(_act(controller, verb, monkeypatch))
    try:
        await real_sleep(0)  # The transition now waits on the already-held lock.
        assert not action.done()
        _write_lease(lease)
    finally:
        controller._lock.release()
    await asyncio.wait_for(action, 1)
    assert calls == ["is-active"]


@pytest.mark.parametrize("verb", ["stop", "restart"])
@pytest.mark.parametrize("change", ["expire", "remove"])
async def test_lease_is_reread_and_deferred_restart_remains_eligible(lane, monkeypatch, verb, change):
    controller, calls, lease = lane
    _write_lease(lease)
    await _act(controller, verb, monkeypatch)
    assert calls == ["is-active"]
    if change == "expire":
        _write_lease(lease, EXPIRED)
    else:
        lease.unlink()
    await _act(controller, verb, monkeypatch)
    assert calls == ["is-active", verb]


async def test_start_does_not_consult_lease(lane, caplog):
    controller, calls, lease = lane
    lease.write_text("malformed", encoding="utf-8")
    readiness = iter([False, False, True])

    async def ready():
        return next(readiness)

    controller._ready = ready
    await controller.ensure_ready()
    assert calls == ["is-active", "start"]
    assert not [r for r in caplog.records if "lease" in r.getMessage()]


async def test_off_mode_does_not_consult_lease(lane, caplog):
    controller, calls, lease = lane
    controller._ondemand = False
    lease.write_text("malformed", encoding="utf-8")
    await controller.monitor_loop(asyncio.Queue())
    await controller.ensure_ready()
    assert controller.note_image_decode_failure() is None
    assert calls == []
    assert not [r for r in caplog.records if "lease" in r.getMessage()]
