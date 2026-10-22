"""Capacity admission must preserve assets and serialize concurrent work."""

import errno
import multiprocessing
from pathlib import Path

import pytest

from papervault.store import derivative_capacity as capacity


def test_plan_is_read_only_and_counts_nested_roots_once(tmp_path):
    derivatives = tmp_path / "derivatives"
    nested = derivatives / "extracts"
    nested.mkdir(parents=True)
    (nested / "current.md").write_bytes(b"12345")
    state = tmp_path / "missing" / "state"
    plan = capacity.capacity_plan(
        state, estimated_bytes=10, derivative_roots=(derivatives, nested),
        extra_used_bytes=3, soft_cap=100,
    )
    assert plan["used_bytes"] == 8
    assert plan["reserved_bytes"] == 0
    assert plan["projected_bytes"] == 18
    assert plan["admitted"] is True
    assert not state.parent.exists()
    assert (nested / "current.md").read_bytes() == b"12345"


@pytest.mark.parametrize("used,estimate,alert,admitted", [
    (69, 10, False, True), (70, 10, True, True), (90, 10, True, True),
    (91, 10, True, False), (100, 1, True, False),
])
def test_alert_and_admission_boundaries(tmp_path, used, estimate, alert, admitted):
    plan = capacity.capacity_plan(
        tmp_path / "state", estimated_bytes=estimate, derivative_roots=(),
        extra_used_bytes=used, soft_cap=100,
    )
    assert plan["alert"] is alert
    assert plan["admitted"] is admitted
    if not admitted:
        with pytest.raises(ValueError, match="capacity"):
            with capacity.reserve_capacity(
                tmp_path / "state", "blocked", estimated_bytes=estimate,
                derivative_roots=(), extra_used_bytes=used, soft_cap=100,
            ):
                pytest.fail("unfunded work executed")


def test_held_reservations_count_once_and_cleanup_only_owned_files(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    sentinel = state / "operator-note"
    sentinel.write_bytes(b"keep")
    with capacity.reserve_capacity(
        state, "outer", estimated_bytes=30, derivative_roots=(state,), soft_cap=100,
    ) as admitted:
        assert admitted["admitted"] is True
        plan = capacity.capacity_plan(
            state, estimated_bytes=20, derivative_roots=(state,), soft_cap=100,
        )
        assert plan["used_bytes"] == 4
        assert plan["reserved_bytes"] == 30
        assert plan["projected_bytes"] == 54
    assert sentinel.read_bytes() == b"keep"
    assert capacity.capacity_plan(
        state, estimated_bytes=1, derivative_roots=(state,), soft_cap=100,
    )["reserved_bytes"] == 0


def test_failure_in_work_cleans_reservation(tmp_path):
    state = tmp_path / "state"
    with pytest.raises(RuntimeError, match="work failed"):
        with capacity.reserve_capacity(
            state, "failure", estimated_bytes=16, derivative_roots=(), soft_cap=100,
        ):
            raise RuntimeError("work failed")
    assert capacity.capacity_plan(
        state, estimated_bytes=1, derivative_roots=(), soft_cap=100,
    )["reserved_bytes"] == 0


def test_enospc_does_not_execute_work_or_touch_assets(tmp_path, monkeypatch):
    derivatives = tmp_path / "extracts"
    derivatives.mkdir()
    current = derivatives / "current.md"
    current.write_bytes(b"current")
    source = tmp_path / "protected.pdf"
    source.write_bytes(b"source")

    def exhausted(*args):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(capacity.os, "posix_fallocate", exhausted)
    with pytest.raises(ValueError, match="capacity"):
        with capacity.reserve_capacity(
            tmp_path / "state", "no-space", estimated_bytes=16,
            derivative_roots=(derivatives,), soft_cap=100,
        ):
            pytest.fail("unfunded work executed")
    assert current.read_bytes() == b"current"
    assert source.read_bytes() == b"source"
    assert capacity.capacity_plan(
        tmp_path / "state", estimated_bytes=1, derivative_roots=(), soft_cap=100,
    )["reserved_bytes"] == 0


def test_disk_headroom_includes_work_and_physical_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity.shutil, "disk_usage", lambda _: (100, 81, 19))
    plan = capacity.capacity_plan(
        tmp_path / "state", estimated_bytes=10, derivative_roots=(), soft_cap=100,
    )
    assert plan["admitted"] is False
    assert plan["required_free_bytes"] == 20
    with pytest.raises(ValueError, match="capacity"):
        with capacity.reserve_capacity(
            tmp_path / "state", "headroom", estimated_bytes=10,
            derivative_roots=(), soft_cap=100,
        ):
            pytest.fail("insufficient headroom admitted")


@pytest.mark.parametrize("field,value", [
    ("estimated_bytes", 0), ("estimated_bytes", -1), ("estimated_bytes", True),
    ("estimated_bytes", 1.5), ("extra_used_bytes", -1), ("soft_cap", 0),
])
def test_invalid_sizes_are_rejected_without_writes(tmp_path, field, value):
    kwargs = {"estimated_bytes": 1, "derivative_roots": (), "soft_cap": 100}
    kwargs[field] = value
    with pytest.raises(ValueError):
        capacity.capacity_plan(tmp_path / "state", **kwargs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("reservation_id", ["", ".", "..", "../source", "a/b", "a\\b", "x" * 129])
def test_invalid_ids_cannot_escape_state(tmp_path, reservation_id):
    with pytest.raises(ValueError, match="reservation"):
        with capacity.reserve_capacity(
            tmp_path / "state", reservation_id, estimated_bytes=1, derivative_roots=(),
        ):
            pytest.fail("invalid identifier admitted")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("target", ["root", "derivative", "file", "hardlink"])
def test_aliases_are_rejected(tmp_path, target):
    state = tmp_path / "state"
    derivatives = tmp_path / "extracts"
    derivatives.mkdir()
    protected = tmp_path / "protected.pdf"
    protected.write_bytes(b"protected")
    if target == "root":
        state.symlink_to(derivatives, target_is_directory=True)
    elif target == "derivative":
        alias = tmp_path / "alias"
        alias.symlink_to(derivatives, target_is_directory=True)
        derivatives = alias
    elif target == "file":
        (derivatives / "alias").symlink_to(protected)
    else:
        (derivatives / "alias").hardlink_to(protected)
    with pytest.raises(ValueError, match="symlink|alias|link"):
        capacity.capacity_plan(state, estimated_bytes=1, derivative_roots=(derivatives,))
    assert protected.read_bytes() == b"protected"


def _hold_reservation(state, ready, release, output):
    try:
        with capacity.reserve_capacity(
            Path(state), "worker", estimated_bytes=60, derivative_roots=(), soft_cap=100,
        ):
            ready.set()
            if not release.wait(10):
                raise RuntimeError("test release timed out")
        output.put("ok")
    except Exception as exc:
        output.put(repr(exc))
        ready.set()


def test_concurrent_process_cannot_overcommit_held_capacity(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, release, output = context.Event(), context.Event(), context.Queue()
    state = tmp_path / "state"
    process = context.Process(target=_hold_reservation, args=(str(state), ready, release, output))
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(ValueError, match="capacity"):
            with capacity.reserve_capacity(
                state, "second", estimated_bytes=60, derivative_roots=(), soft_cap=100,
            ):
                pytest.fail("concurrent reservations overcommitted")
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
    assert output.get(timeout=2) == "ok"
    assert process.exitcode == 0


def test_crashed_reservation_remains_counted_and_cannot_be_reused(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    stale = state / ".capacity-reservation-crashed.reserve"
    stale.write_bytes(b"x" * 60)
    plan = capacity.capacity_plan(state, estimated_bytes=50, derivative_roots=(), soft_cap=100)
    assert plan["reserved_bytes"] == 60
    assert plan["admitted"] is False
    with pytest.raises(ValueError, match="reservation|capacity"):
        with capacity.reserve_capacity(
            state, "crashed", estimated_bytes=1, derivative_roots=(), soft_cap=100,
        ):
            pytest.fail("existing reservation overwritten")
    assert stale.read_bytes() == b"x" * 60


def test_root_rebinding_is_refused_and_cleanup_uses_original_directory(tmp_path, monkeypatch):
    state = tmp_path / "state"
    moved = tmp_path / "moved"
    protected = tmp_path / "protected"
    protected.mkdir()
    sentinel = protected / ".capacity-reservation-race.reserve"
    sentinel.write_bytes(b"protected")
    allocate = capacity.os.posix_fallocate

    def rebind(fd, offset, length):
        allocate(fd, offset, length)
        state.rename(moved)
        state.symlink_to(protected, target_is_directory=True)

    monkeypatch.setattr(capacity.os, "posix_fallocate", rebind)
    with pytest.raises(ValueError, match="root|symlink|changed"):
        with capacity.reserve_capacity(
            state, "race", estimated_bytes=10, derivative_roots=(), soft_cap=100,
        ):
            pytest.fail("rebound root admitted")
    assert sentinel.read_bytes() == b"protected"
    assert not (moved / ".capacity-reservation-race.reserve").exists()


def _contend_for_capacity(state, start, release, output, reservation_id):
    if not start.wait(10):
        output.put("timeout")
        return
    try:
        with capacity.reserve_capacity(
            Path(state), reservation_id, estimated_bytes=60, derivative_roots=(), soft_cap=100,
        ):
            output.put("held")
            release.wait(10)
    except ValueError:
        output.put("blocked")


def test_simultaneous_admission_is_atomic(tmp_path):
    context = multiprocessing.get_context("spawn")
    start, release, output = context.Event(), context.Event(), context.Queue()
    processes = [context.Process(
        target=_contend_for_capacity,
        args=(str(tmp_path / "state"), start, release, output, str(index)),
    ) for index in range(2)]
    for process in processes:
        process.start()
    try:
        start.set()
        outcomes = [output.get(timeout=10), output.get(timeout=10)]
        assert sorted(outcomes) == ["blocked", "held"]
    finally:
        release.set()
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()
    assert [process.exitcode for process in processes] == [0, 0]


def _crash_after_reservation(state):
    import os

    with capacity.reserve_capacity(
        Path(state), "crash", estimated_bytes=60, derivative_roots=(), soft_cap=100,
    ):
        os._exit(0)


def test_process_crash_retains_physical_reservation(tmp_path):
    context = multiprocessing.get_context("spawn")
    state = tmp_path / "state"
    process = context.Process(target=_crash_after_reservation, args=(str(state),))
    process.start()
    process.join(10)
    if process.is_alive():
        process.kill()
        process.join()
    assert process.exitcode == 0
    plan = capacity.capacity_plan(state, estimated_bytes=50, derivative_roots=(), soft_cap=100)
    assert plan["reserved_bytes"] == 60
    assert plan["admitted"] is False
    reservation = state / ".capacity-reservation-crash.reserve"
    assert reservation.stat().st_blocks > 0


def test_unsupported_physical_allocation_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delattr(capacity.os, "posix_fallocate")
    with pytest.raises(ValueError, match="capacity"):
        with capacity.reserve_capacity(
            tmp_path / "state", "unsupported", estimated_bytes=1, derivative_roots=(),
        ):
            pytest.fail("unreserved work executed")
    assert not (tmp_path / "state" / ".capacity-reservation-unsupported.reserve").exists()


def test_foreign_replacement_is_never_deleted_on_exit(tmp_path):
    state = tmp_path / "state"
    with pytest.raises(ValueError, match="replaced"):
        with capacity.reserve_capacity(
            state, "replaced", estimated_bytes=10, derivative_roots=(), soft_cap=100,
        ):
            file = state / ".capacity-reservation-replaced.reserve"
            file.unlink()
            file.write_bytes(b"foreign")
    assert file.read_bytes() == b"foreign"


def test_lock_alias_never_changes_protected_file(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    protected = tmp_path / "protected"
    protected.write_bytes(b"protected")
    (state / ".capacity.lock").symlink_to(protected)
    with pytest.raises(ValueError, match="capacity|alias|symlink"):
        with capacity.reserve_capacity(
            state, "locked", estimated_bytes=1, derivative_roots=(),
        ):
            pytest.fail("aliased lock admitted")
    assert protected.read_bytes() == b"protected"


def test_derivative_filesystem_must_match_reservation_filesystem(tmp_path, monkeypatch):
    import os

    derivatives = tmp_path / "derivatives"
    derivatives.mkdir()
    (derivatives / "current").write_bytes(b"current")
    real_fstat = capacity.os.fstat
    identity = derivatives.stat().st_ino

    def different_device(fd):
        info = real_fstat(fd)
        if info.st_ino == identity:
            fields = list(info)
            fields[2] = info.st_dev + 1
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(capacity.os, "fstat", different_device)
    with pytest.raises(ValueError, match="filesystem"):
        with capacity.reserve_capacity(
            tmp_path / "state", "cross-device", estimated_bytes=1,
            derivative_roots=(derivatives,),
        ):
            pytest.fail("reservation on unrelated filesystem admitted")
    assert (derivatives / "current").read_bytes() == b"current"


def test_work_io_failure_preserves_original_exception_and_releases_capacity(tmp_path):
    failure = OSError(errno.EIO, "output write failed")
    state = tmp_path / "state"
    with pytest.raises(OSError) as caught:
        with capacity.reserve_capacity(
            state, "work-io", estimated_bytes=10, derivative_roots=(), soft_cap=100,
        ):
            raise failure
    assert caught.value is failure
    assert capacity.capacity_plan(
        state, estimated_bytes=1, derivative_roots=(), soft_cap=100,
    )["reserved_bytes"] == 0


def test_missing_derivative_directory_on_another_filesystem_is_refused(tmp_path, monkeypatch):
    import os

    mount = tmp_path / "other-filesystem"
    mount.mkdir()
    real_fstat = capacity.os.fstat
    mount_inode = mount.stat().st_ino

    def different_device(fd):
        info = real_fstat(fd)
        if info.st_ino == mount_inode:
            fields = list(info)
            fields[2] = info.st_dev + 1
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(capacity.os, "fstat", different_device)
    with pytest.raises(ValueError, match="filesystem"):
        capacity.capacity_plan(
            tmp_path / "state", estimated_bytes=1, derivative_roots=(mount / "future",),
        )
    assert not (mount / "future").exists()


def test_pending_output_headroom_is_not_reused_by_concurrent_reservation(tmp_path, monkeypatch):
    free = 110
    allocate = capacity.os.posix_fallocate

    def allocation(fd, offset, length):
        nonlocal free
        allocate(fd, offset, length)
        free -= length

    monkeypatch.setattr(capacity.shutil, "disk_usage", lambda path: (110, 110 - free, free))
    monkeypatch.setattr(capacity.os, "posix_fallocate", allocation)
    with capacity.reserve_capacity(tmp_path / "state", "first", estimated_bytes=50,
                                   derivative_roots=(), soft_cap=1000):
        with pytest.raises(ValueError, match="capacity"):
            with capacity.reserve_capacity(tmp_path / "state", "second", estimated_bytes=30,
                                           derivative_roots=(), soft_cap=1000):
                pytest.fail("pending output headroom was promised to two jobs")
