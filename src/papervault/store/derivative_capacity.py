"""Conservative admission for disposable derivative work, without asset deletion.

Callers must choose a shared, dedicated maintenance state root and derivative-only
roots; source corpus paths must never be supplied as derivative roots. Adapter
reported database/snapshot usage belongs in ``extra_used_bytes``. Every writer in
the same capacity domain must share this root and the same accounting inputs.

Reservation files stay physically allocated while work runs. Admission therefore
requires twice the estimate in free space: one estimate for the reservation and
one for output, plus outstanding output headroom for every held reservation.
Crashed reservations remain charged until an operator reconciles
them; this module never automatically prunes unknown or abandoned objects.
"""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Iterator


DERIVATIVE_SOFT_CAP = 2_000_000_000_000
_PREFIX = ".capacity-reservation-"
_LOCK = ".capacity.lock"
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _path(path: Path) -> Path:
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("capacity path must not escape through '..'")
    path = path.absolute()
    if path == path.parent:
        raise ValueError("capacity root must not be a filesystem root")
    return path


def _open_directory(path: Path, *, create: bool = False) -> int | None:
    """Walk from / with no-follow descriptors, never resolving through aliases."""
    fd = os.open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    return None
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        result, fd = fd, -1
        return result
    except OSError as exc:
        raise ValueError("capacity path contains a symlink or non-directory alias") from exc
    finally:
        if fd != -1:
            os.close(fd)


def _check_binding(path: Path, fd: int) -> None:
    current = _open_directory(path)
    try:
        if current is None or _identity(os.fstat(current)) != _identity(os.fstat(fd)):
            raise ValueError("capacity root changed during reservation")
    finally:
        if current is not None:
            os.close(current)


def _regular(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("capacity inventory contains a symlink, hardlink alias or special file")


def _sizes(estimated_bytes: int, extra_used_bytes: int, soft_cap: int) -> None:
    for name, value, minimum in (
        ("estimated_bytes", estimated_bytes, 1),
        ("extra_used_bytes", extra_used_bytes, 0),
        ("soft_cap", soft_cap, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"capacity {name} must be an integer >= {minimum}")


def _reserved(root_fd: int | None) -> tuple[int, set[tuple[int, int]]]:
    total = 0
    identities = set()
    if root_fd is not None:
        for name in os.listdir(root_fd):
            if name.startswith(_PREFIX):
                info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                _regular(info)
                total += info.st_size
                identities.add(_identity(info))
    return total, identities


def _inventory(fd: int, seen: set[tuple[int, int]], device: int) -> int:
    info = os.fstat(fd)
    if info.st_dev != device:
        raise ValueError("derivative and capacity reservation must share a filesystem")
    identity = _identity(info)
    if identity in seen:
        return 0
    seen.add(identity)
    total = 0
    for name in os.listdir(fd):
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=fd)
            try:
                if _identity(os.fstat(child)) != _identity(info):
                    raise ValueError("capacity derivative directory changed during inventory")
                total += _inventory(child, seen, device)
            finally:
                os.close(child)
        else:
            _regular(info)
            if _identity(info) not in seen:
                seen.add(_identity(info))
                total += info.st_size
    return total


def _plan(
    root: Path, root_fd: int | None, estimated_bytes: int,
    derivative_roots: tuple[Path, ...], extra_used_bytes: int, soft_cap: int,
) -> dict:
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    capacity_fd = root_fd if root_fd is not None else _open_directory(ancestor)
    try:
        device = os.fstat(capacity_fd).st_dev
        reserved, seen = _reserved(root_fd)
        used = extra_used_bytes
        for path in derivative_roots:
            fd = _open_directory(path)
            if fd is None:
                # Future output still needs to be funded on its actual filesystem.
                ancestor = path.parent
                while not ancestor.exists():
                    ancestor = ancestor.parent
                fd = _open_directory(ancestor)
                try:
                    if os.fstat(fd).st_dev != device:
                        raise ValueError("derivative and capacity reservation must share a filesystem")
                finally:
                    os.close(fd)
            else:
                try:
                    used += _inventory(fd, seen, device)
                finally:
                    os.close(fd)
        # Query the bound directory; a changed pathname cannot redirect admission
        # to another filesystem after its descriptor was opened.
        free = shutil.disk_usage(f"/proc/self/fd/{capacity_fd}")[2]
    finally:
        if root_fd is None:
            os.close(capacity_fd)
    projected = used + reserved + estimated_bytes
    required_free = reserved + estimated_bytes * 2
    reason = None
    if used + reserved >= soft_cap or projected > soft_cap:
        reason = "derivative soft capacity exhausted"
    elif free < required_free:
        reason = "insufficient physical capacity for reservation and output"
    return {
        "used_bytes": used,
        "reserved_bytes": reserved,
        "estimated_bytes": estimated_bytes,
        "projected_bytes": projected,
        "soft_cap_bytes": soft_cap,
        "free_bytes": free,
        "required_free_bytes": required_free,
        "alert": projected * 5 >= soft_cap * 4,
        "admitted": reason is None,
        "reason": reason,
    }


def capacity_plan(
    root: Path, *, estimated_bytes: int, derivative_roots: tuple[Path, ...],
    extra_used_bytes: int = 0, soft_cap: int = DERIVATIVE_SOFT_CAP,
) -> dict:
    """Return a read-only advisory plan; admission is rechecked under a lock.

    Missing derivative directories contribute zero bytes. Overlapping derivative
    directories and reservations beneath them are counted only once. Symlinks and
    hardlinks are refused rather than risking inclusion of source corpus aliases.
    Derivatives must reside on the reservation filesystem, including nested mounts.
    """
    _sizes(estimated_bytes, extra_used_bytes, soft_cap)
    root = _path(root)
    derivatives = tuple(_path(path) for path in derivative_roots)
    root_fd = _open_directory(root)
    try:
        return _plan(root, root_fd, estimated_bytes, derivatives, extra_used_bytes, soft_cap)
    except OSError as exc:
        raise ValueError("capacity inventory could not be verified") from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)


def _verify_object(root_fd: int, name: str, fd: int) -> None:
    info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    _regular(info)
    if _identity(info) != _identity(os.fstat(fd)):
        raise ValueError("capacity reservation or lock was replaced")


class ReservationPlan(dict):
    """A reservation plan whose owner can retain the exact file for recovery.

    ``retain()`` transfers cleanup to the caller's durable reconciliation owner.
    It never grants permission to delete an unverified or unknown reservation.
    """

    retained = False

    def retain(self) -> None:
        self.retained = True


@contextmanager
def reserve_capacity(
    root: Path, reservation_id: str, *, estimated_bytes: int,
    derivative_roots: tuple[Path, ...], extra_used_bytes: int = 0,
    soft_cap: int = DERIVATIVE_SOFT_CAP,
) -> Iterator[dict]:
    """Atomically admit and physically reserve capacity for one work context.

    Only the exact reservation inode created here is removed on ordinary exit.
    Unsupported physical allocation fails closed. A process crash deliberately
    leaves its reservation charged; duplicate IDs never overwrite existing files.
    Plans include reservations already held, excluding the newly admitted one.
    A durable owner may call the yielded dict-compatible plan's ``retain()`` to
    keep its file charged after exit, then reconcile that exact inode itself.
    """
    _sizes(estimated_bytes, extra_used_bytes, soft_cap)
    if not isinstance(reservation_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", reservation_id,
    ):
        raise ValueError("invalid capacity reservation identifier")
    root = _path(root)
    derivatives = tuple(_path(path) for path in derivative_roots)
    # Verify derivative inventories before creating maintenance state.
    capacity_plan(root, estimated_bytes=estimated_bytes, derivative_roots=derivatives,
                  extra_used_bytes=extra_used_bytes, soft_cap=soft_cap)
    root_fd = _open_directory(root, create=True)
    lock_fd = None
    reservation_fd = None
    plan = None
    name = f"{_PREFIX}{reservation_id}.reserve"
    try:
        try:
            lock_fd = os.open(_LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                              0o600, dir_fd=root_fd)
            _regular(os.fstat(lock_fd))
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                _check_binding(root, root_fd)
                _verify_object(root_fd, _LOCK, lock_fd)
                plan = ReservationPlan(_plan(
                    root, root_fd, estimated_bytes, derivatives, extra_used_bytes, soft_cap))
                if not plan["admitted"]:
                    raise ValueError(plan["reason"])
                try:
                    reservation_fd = os.open(
                        name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600, dir_fd=root_fd,
                    )
                except FileExistsError as exc:
                    raise ValueError("capacity reservation identifier already exists") from exc
                allocate = getattr(os, "posix_fallocate", None)
                if allocate is None:
                    raise ValueError("physical capacity reservation is unavailable")
                allocate(reservation_fd, 0, estimated_bytes)
                os.fsync(reservation_fd)
                os.fsync(root_fd)
                _check_binding(root, root_fd)
                _verify_object(root_fd, name, reservation_fd)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError as exc:
            raise ValueError(f"capacity reservation failed: {exc}") from exc
        yield plan
    finally:
        try:
            if reservation_fd is not None and not (plan and plan.retained):
                # Use the pinned directory even if its public pathname changed.
                # Never unlink an object that replaced our created reservation.
                _verify_object(root_fd, name, reservation_fd)
                os.unlink(name, dir_fd=root_fd)
                os.fsync(root_fd)
        finally:
            if reservation_fd is not None:
                os.close(reservation_fd)
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(root_fd)
