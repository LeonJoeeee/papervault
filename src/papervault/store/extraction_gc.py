"""Bounded, resumable GC for explicitly disposable superseded extract outputs."""

import hashlib
import os
import re
import stat
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from uuid import uuid4

from .extract_lifecycle import protections, validate_lease
from .knowledge_commands import require
from .recovery import checked_path, maintenance_deadline
from .repository import AuditEvent


def plan_gc(k, vault):
    """Return a read-only snapshot; apply must recheck every admission."""
    root = str(checked_path(Path(vault)))
    items = []
    with k.connection.transaction():
        for retained in k._rows("extract_retention"):
            if retained["vault"] != root:
                continue
            identity = retained["completion_id"]
            deletion = k._one("extract_deletions", completion_id=identity)
            reasons = protections(k, identity)
            items.append({"completion_id": identity, "eligible": not reasons,
                "reasons": reasons, "state": deletion["state"] if deletion else "retained",
                "disposable": retained["disposable"], "retention_revision": retained["revision"],
                "superseded_at": retained["superseded_at"].isoformat() if retained["superseded_at"] else None})
    return {"vault": root, "retention_days": 90, "extracts": items}


@contextmanager
def output_directory(vault, completion_id):
    if not isinstance(completion_id, str) or not re.fullmatch("[0-9a-f]{32}", completion_id):
        raise ValueError("invalid extraction completion identity")
    root = Path(vault).absolute()
    if ".." in root.parts:
        raise ValueError("invalid extract vault path")
    descriptors = []
    try:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        descriptors.append(fd)
        for part in (*root.parts[1:], "extracts", "versions", completion_id):
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            descriptors.append(fd)
        yield fd
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def unlink_output(vault, completion_id, output, *, before_unlink):
    expected = f"extracts/versions/{completion_id}/output.md"
    if output["path"] != expected or output["store"] != "filesystem" or output["role"] != "markdown":
        raise ValueError("GC output is not the exact immutable extract path")
    with output_directory(vault, completion_id) as directory:
        try:
            fd = os.open("output.md", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        except FileNotFoundError:
            os.fsync(directory)
            return
        try:
            original = os.fstat(fd)
            if not stat.S_ISREG(original.st_mode) or original.st_nlink != 1:
                raise ValueError("GC refuses shared or nonregular output")
            digest, size = hashlib.sha256(), 0
            while data := os.read(fd, 1024 * 1024):
                digest.update(data)
                size += len(data)
            if digest.hexdigest() != output["sha256"] or size != output["byte_length"]:
                raise ValueError("GC output hash/length differs from receipt")
            current = os.stat("output.md", dir_fd=directory, follow_symlinks=False)
            if (current.st_dev, current.st_ino, current.st_nlink) != (original.st_dev, original.st_ino, 1):
                raise ValueError("GC output changed during verification")
            before_unlink()
            os.unlink("output.md", dir_fd=directory)
            os.fsync(directory)
        finally:
            os.close(fd)


def collect_extract(k, vault, completion_id, lease):
    """Commit intent before IO; never erase receipts, inputs, or directories.

    The catalog writer transaction covers the final protection/lease check and
    unlink, so snapshot barriers and concurrent reference admissions cannot race
    the filesystem operation. A crash leaves a retryable permanent tombstone.
    """
    from .fencing import writer_lease
    from psycopg.pq import TransactionStatus

    if k.connection.info.transaction_status != TransactionStatus.IDLE:
        raise ValueError("GC requires an idle connection outside any caller transaction; intent must commit before unlink")
    root = checked_path(Path(vault))
    payload = {"completion_id": completion_id, "vault": str(root), "lease": lease}
    current = validate_lease(k, lease)
    seconds = (current["expires_at"] - k.connection.execute(
        "SELECT clock_timestamp() AS now").fetchone()["now"]).total_seconds()
    require(seconds > 0, "maintenance lease expired")
    # The standalone coordinator bounds preparation/readback too. Threaded
    # callers still recheck the DB clock immediately before every unlink.
    deadline = maintenance_deadline(seconds) if threading.current_thread() is threading.main_thread() else nullcontext()
    with deadline, writer_lease(root, None):
        intent = k.command("prepare_extract_deletion", "gc-prepare:" + uuid4().hex, payload)
        with k.repository._write(AuditEvent("maintenance", "extract_gc_unlink")):
            validate_lease(k, lease)
            reasons = protections(k, completion_id)
            require(not reasons, "extract cannot be deleted: " + ", ".join(reasons))
            unlink_output(root, completion_id, intent["output"], before_unlink=lambda: validate_lease(k, lease))
            validate_lease(k, lease)
        return k.command("finish_extract_deletion", "gc-finish:" + uuid4().hex,
            {**payload, "deletion_id": intent["deletion_id"]})
