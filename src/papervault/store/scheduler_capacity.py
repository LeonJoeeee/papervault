"""Durable capacity ownership and qualified runtime backend usage censuses.

PAPERVAULT_CATALOG_EXECUTION_BYTES is a positive operational output budget per
catalog job (default 1 GiB), including isolated rebuild scratch and final outputs.
It is an estimate, not an enforcement quota; operators must budget large jobs.
All new work requires PAPERVAULT_DERIVATIVE_USAGE_FILE, a mode-0600 JSON census
refreshed at least every 300 seconds using PostgresStore/Neo4jStore estimated_bytes
and catalog relation sizing. Its ``backend_bytes`` includes PostgreSQL, Neo4j
and retained snapshot derivatives outside ``derivative_roots``. Runtime also adds
measured catalog relation bytes (overlap is conservatively counted). An operator
measured zero is explicit evidence, never a missing-accounting fallback.
``scope`` identifies absolute ``vault_root``
and ``knowledge_root``. ``measured_at`` is a timezone-qualified ISO timestamp.
This is conservative admission accounting, not an exact on-disk guarantee.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
import fcntl
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

from .derivative_capacity import (
    DERIVATIVE_SOFT_CAP, _LOCK, _check_binding, _identity, _open_directory,
    _path, _regular, _sizes, _verify_object, reserve_capacity,
)


DEFAULT_EXECUTION_BYTES = 1_073_741_824
_intake_scope = ContextVar("scheduler_capacity_intake", default=None)
log = logging.getLogger("ks.capacity")


def measured_backend_usage(path, *, vault_root, knowledge_root, max_age_seconds=300) -> dict:
    """Read operator-owned fresh scoped measurements, never discover credentials."""
    if type(max_age_seconds) is not int or not 0 < max_age_seconds <= 300:
        raise ValueError("backend usage census maximum age must be <=300 seconds")
    if path is None or not str(path):
        raise ValueError("backend accounting requires a protected usage census")
    path = _path(Path(path))
    parent_fd = _open_directory(path.parent)
    if parent_fd is None:
        raise ValueError("backend accounting usage census is missing")
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        info = os.fstat(fd)
        _regular(info)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077 or info.st_size > 16384:
            raise ValueError("backend usage census must be protected and operator-owned")
        data = json.loads(os.read(fd, 16385))
        value = data["backend_bytes"]
        if type(value) is not int or value < 0:
            raise ValueError("backend usage census bytes must be a nonnegative integer")
        expected = {"vault_root": str(_path(vault_root)), "knowledge_root": str(_path(knowledge_root))}
        if data["scope"] != expected:
            raise ValueError("backend usage census scope differs from configured roots")
        stamp = datetime.fromisoformat(data["measured_at"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("backend usage census requires a timezone")
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
        if not 0 <= age <= max_age_seconds:
            raise ValueError("backend usage census is stale or from the future")
        if type(data.get("metadata_sample", False)) is not bool:
            raise ValueError("backend usage census scope is invalid")
        return {"backend_bytes": value, "measured_at": stamp.isoformat(),
                "metadata_sample": data.get("metadata_sample", False)}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("backend accounting usage census is unavailable or invalid") from exc
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def measured_backend_bytes(path, *, vault_root, knowledge_root, max_age_seconds=300) -> int:
    return measured_backend_usage(path, vault_root=vault_root, knowledge_root=knowledge_root,
                                  max_age_seconds=max_age_seconds)["backend_bytes"]


class CapacityAdmissionError(ValueError):
    """No new physical work may start, but recovery remains available."""


class SchedulerCapacity:
    """Persistent claim-bound reservations, including growth not yet in a census.

    Qualified callbacks return {backend_bytes, measured_at, metadata_sample?}.
    An explicitly injected integer callback means an actual synchronous fresh
    measurement at invocation; production always uses the scoped timestamped file.
    Unknown outcomes are never re-executed with the old reservation. Physical
    settled growth stays charged until a census begun after settlement includes it.
    """

    def __init__(self, root: Path, *, derivative_roots: tuple[Path, ...], estimated_bytes,
                 backend_usage, soft_cap: int = DERIVATIVE_SOFT_CAP):
        self.root = _path(root)
        self.derivative_roots = tuple(_path(path) for path in derivative_roots)
        self.estimated_bytes, self.backend_usage, self.soft_cap = estimated_bytes, backend_usage, soft_cap
        self._held = {}

    @staticmethod
    def _report(plan):
        if plan.get("alert"):
            log.warning("derivative capacity 80%% alert: projected=%s soft_cap=%s held=%s",
                        plan["projected_bytes"], plan["soft_cap_bytes"], plan["reserved_bytes"])

    @contextmanager
    def _accounting(self):
        """Serialize census + reservation inventory with settlement across workers.

        Always acquire this before the physical inventory lock. Keep it only for
        accounting, never across the caller's execution or asynchronous intake.
        """
        root_fd = _open_directory(self.root, create=True)
        lock_fd = None
        name = ".scheduler-accounting.lock"
        try:
            lock_fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                              0o600, dir_fd=root_fd)
            _regular(os.fstat(lock_fd))
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            _check_binding(self.root, root_fd)
            _verify_object(root_fd, name, lock_fd)
            yield
        except OSError as exc:
            raise ValueError("capacity accounting serialization unavailable") from exc
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(root_fd)

    def _usage(self, physical):
        try:
            started_at = datetime.now(timezone.utc)
            value = self.backend_usage(physical)
            if type(value) is int:
                value = {"backend_bytes": value, "measured_at": started_at.isoformat()}
            if not isinstance(value, dict):
                raise ValueError("qualified backend usage is required")
            _sizes(1, value["backend_bytes"], self.soft_cap)
            stamp = datetime.fromisoformat(value["measured_at"].replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - stamp).total_seconds()
            if stamp.tzinfo is None or not 0 <= age <= 300:
                raise ValueError("backend usage timestamp is unqualified")
            if physical and value.get("metadata_sample", False):
                raise ValueError("metadata-only census cannot fund physical execution")
            return value["backend_bytes"], stamp, value.get("metadata_sample", False)
        except Exception as exc:
            raise ValueError("backend accounting unavailable for capacity admission") from exc

    @staticmethod
    def _names(job_id):
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("capacity job identity is required")
        digest = hashlib.sha256(job_id.encode()).hexdigest()
        identity = "scheduler-" + digest
        return identity, f".capacity-reservation-{identity}.reserve", f".{identity}.json"

    @staticmethod
    def _binding(claim):
        keys = ("job_id", "attempt_id", "submission_id")
        if not isinstance(claim, dict) or any(not isinstance(claim.get(key), str) or not claim[key] for key in keys):
            raise ValueError("capacity requires an exact execution attempt identity")
        return {key: claim[key] for key in keys}

    @staticmethod
    def _write(fd, data):
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        raw = json.dumps(data, sort_keys=True).encode()
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(raw)
            output.flush()
            os.fsync(fd)

    @contextmanager
    def _record(self, job_id):
        _, filename, metadata = self._names(job_id)
        root_fd = _open_directory(self.root)
        if root_fd is None:
            yield None
            return
        lock_fd = metadata_fd = reservation_fd = None
        try:
            try:
                lock_fd = os.open(_LOCK, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
                _regular(os.fstat(lock_fd))
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                _verify_object(root_fd, _LOCK, lock_fd)
                _check_binding(self.root, root_fd)
                metadata_fd = os.open(metadata, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
            except FileNotFoundError:
                yield None
                return
            _regular(os.fstat(metadata_fd))
            if os.fstat(metadata_fd).st_size > 16384:
                raise ValueError("capacity reconciliation metadata is invalid")
            data = json.loads(os.read(metadata_fd, 16385))
            if (not isinstance(data, dict) or data.get("schema_version") != 2
                    or data.get("job_id") != job_id or data.get("reservation") != filename
                    or not isinstance(data.get("owner_id"), str)
                    or data.get("state") not in {"held", "unknown", "spent"}
                    or type(data.get("physical")) is not bool):
                raise ValueError("capacity reconciliation identity is unrecognized")
            if job_id in self._held:
                owned = self._held[job_id][1]
                if any(data.get(key) != owned.get(key) for key in ("device", "inode", "owner_id")):
                    raise ValueError("capacity reservation owner was replaced")
            reservation_fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
            info = os.fstat(reservation_fd)
            _regular(info)
            if (_identity(info) != (data.get("device"), data.get("inode"))
                    or info.st_size != data.get("estimated_bytes")):
                raise ValueError("capacity reconciliation reservation was replaced")
            _verify_object(root_fd, metadata, metadata_fd)
            _verify_object(root_fd, filename, reservation_fd)
            yield root_fd, data, metadata_fd
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("capacity reconciliation could not verify owned reservation") from exc
        finally:
            for fd in (reservation_fd, metadata_fd, lock_fd, root_fd):
                if fd is not None:
                    os.close(fd)

    def _delete(self, job_id, root_fd):
        _, filename, metadata = self._names(job_id)
        os.unlink(filename, dir_fd=root_fd)
        os.unlink(metadata, dir_fd=root_fd)
        os.fsync(root_fd)
        self._held.pop(job_id, None)

    def _reconcile_spent(self, stamp, *, metadata_sample=False):
        root_fd = _open_directory(self.root)
        if root_fd is None:
            return
        try:
            names = [name for name in os.listdir(root_fd) if name.startswith(".scheduler-") and name.endswith(".json")]
        finally:
            os.close(root_fd)
        for name in names:
            # Ownership is reverified under the shared lock before any unlink.
            root_fd = _open_directory(self.root)
            fd = None
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
                _regular(os.fstat(fd))
                if os.fstat(fd).st_size > 16384:
                    continue
                data = json.loads(os.read(fd, 16385))
                if not isinstance(data, dict) or not isinstance(data.get("job_id"), str):
                    continue
                job_id = data["job_id"]
                if self._names(job_id)[2] != name:
                    continue
                with self._record(job_id) as record:
                    if record is not None:
                        directory, owned, _ = record
                        if (owned["state"] == "spent" and owned.get("claim")
                                and not (owned["physical"] and metadata_sample)
                                and datetime.fromisoformat(owned["settled_at"]) <= stamp):
                            self._delete(job_id, directory)
            except (ValueError, OSError, KeyError, TypeError):
                continue  # Unrecognized/changed objects remain charged, never pruned.
            finally:
                if fd is not None:
                    os.close(fd)
                if root_fd is not None:
                    os.close(root_fd)

    @contextmanager
    def intake(self):
        active = _intake_scope.get()
        if active is not None and active[0] is self:
            yield active[1]
            return
        entered = False
        try:
            with ExitStack() as reservations:
                with self._accounting():
                    extra, stamp, metadata_sample = self._usage(False)
                    self._reconcile_spent(stamp, metadata_sample=metadata_sample)
                    estimate = self.estimated_bytes() if callable(self.estimated_bytes) else self.estimated_bytes
                    _sizes(estimate, extra, self.soft_cap)
                    plan = reservations.enter_context(reserve_capacity(self.root, "intake-" + uuid4().hex,
                        estimated_bytes=estimate, derivative_roots=self.derivative_roots,
                        extra_used_bytes=extra, soft_cap=self.soft_cap))
                entered = True
                self._report(plan)
                token = _intake_scope.set((self, plan))
                try:
                    yield plan
                except asyncio.CancelledError:
                    plan.retain()  # A remote/worker outcome can outlive cancellation.
                    raise
                finally:
                    _intake_scope.reset(token)
                    with self._accounting():
                        reservations.close()
        except Exception as exc:
            if entered:
                raise
            raise CapacityAdmissionError("capacity admission unavailable for new intake") from exc

    def acquire(self, job_id: str, *, physical: bool = True, reuse: bool = True) -> dict:
        with self._accounting():
            return self._acquire(job_id, physical=physical, reuse=reuse)

    def _acquire(self, job_id, *, physical, reuse):
        identity, filename, metadata = self._names(job_id)
        extra, stamp, metadata_sample = self._usage(physical)
        self._reconcile_spent(stamp, metadata_sample=metadata_sample)
        estimate = self.estimated_bytes() if callable(self.estimated_bytes) else self.estimated_bytes
        _sizes(estimate, extra, self.soft_cap)
        if job_id in self._held:
            with self._record(job_id) as record:
                if record is None:
                    # Another worker may have reconciled our spent charge. Only
                    # forget the cache; orphan/replacement files still block below.
                    self._held.pop(job_id)
                elif not reuse:
                    raise ValueError("unreconciled scheduler capacity reservation prevents new admission")
                elif record[1]["state"] != "held":
                    raise ValueError("unreconciled scheduler capacity reservation prevents execution")
                else:
                    return self._held[job_id][0]
        if (self.root / metadata).exists() or (self.root / filename).exists():
            raise ValueError("unreconciled scheduler capacity reservation requires receipt recovery")
        with reserve_capacity(self.root, identity, estimated_bytes=estimate,
                              derivative_roots=self.derivative_roots,
                              extra_used_bytes=extra, soft_cap=self.soft_cap) as plan:
            root_fd = _open_directory(self.root)
            fd = None
            try:
                info = os.stat(filename, dir_fd=root_fd, follow_symlinks=False)
                _regular(info)
                value = {"schema_version": 2, "job_id": job_id, "reservation": filename,
                         "device": info.st_dev, "inode": info.st_ino, "estimated_bytes": estimate,
                         "owner_id": uuid4().hex, "claim": None, "physical": physical, "state": "held"}
                fd = os.open(metadata, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=root_fd)
                self._write(fd, value)
                os.fsync(root_fd)
                _check_binding(self.root, root_fd)
                plan.retain()
                self._held[job_id] = (plan, value)
            finally:
                if fd is not None:
                    os.close(fd)
                os.close(root_fd)
        self._report(plan)
        return plan

    def bind(self, job_id: str, claim: dict) -> None:
        binding = self._binding(claim)
        if binding["job_id"] != job_id or job_id not in self._held:
            raise ValueError("capacity binding requires the admitted job owner")
        with self._record(job_id) as record:
            if record is None:
                raise ValueError("capacity reservation is absent")
            _, data, fd = record
            if data["state"] != "held" or data["claim"] not in (None, binding):
                raise ValueError("capacity attempt binding differs")
            data["claim"] = binding
            self._write(fd, data)
            self._held[job_id] = (self._held[job_id][0], data)

    def mark_unknown(self, job_id: str, claim: dict) -> None:
        binding = self._binding(claim)
        with self._record(job_id) as record:
            if record is not None:
                _, data, fd = record
                if data["claim"] != binding:
                    raise ValueError("unknown capacity attempt identity differs")
                data["state"] = "unknown"
                self._write(fd, data)

    def release(self, job_id: str, *, claim: dict | None = None, reconciled: bool = False) -> bool:
        """Abort unexecuted owned work, or settle an authoritatively verified claim.

        Physical settlement is retained until a qualified census begun after it;
        recovery never releases a replacement attempt's reservation by job alone.
        """
        with self._accounting():
            return self._release(job_id, claim=claim, reconciled=reconciled)

    def _release(self, job_id, *, claim, reconciled):
        if job_id not in self._held and not reconciled:
            return False
        with self._record(job_id) as record:
            if record is None:
                return False
            root_fd, data, fd = record
            if claim is None:
                if job_id not in self._held or data["claim"] is not None:
                    raise ValueError("capacity reconciliation requires exact attempt identity")
                self._delete(job_id, root_fd)
                return True
            if data["claim"] != self._binding(claim):
                raise ValueError("capacity reconciliation attempt identity differs")
            if not reconciled:
                raise ValueError("bound capacity requires authoritative reconciliation")
            if not data["physical"]:
                self._delete(job_id, root_fd)
                return True
            if data["state"] != "spent":
                data.update(state="spent", settled_at=datetime.now(timezone.utc).isoformat())
                self._write(fd, data)
            settled_at = datetime.fromisoformat(data["settled_at"])
            _, stamp, _ = self._usage(True)
            if stamp < settled_at:
                return False
            self._delete(job_id, root_fd)
            return True
