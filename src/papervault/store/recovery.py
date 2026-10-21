"""Verified coordinated recovery packages, independent of their backend adapters.

Adapters hold an enforced writer barrier through export, verification and audited
registration. This module never treats a directory of independently taken dumps
as a recovery set. Failed work stays explicitly incomplete and cannot move retention.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .extraction import fsync_directory, write_once
from .repository import canonical_json


REQUIRED_STORES = frozenset({"catalog", "sources", "extracts", "inputs", "recipes", "postgres", "neo4j"})
KEEP_COMPLETE = 4
DERIVATIVE_SOFT_CAP = 2_000_000_000_000
ALERT_FRACTION = 0.8
MAX_WINDOW_SECONDS = 7200
_IDENTITY = re.compile(r"recovery-\d{8}T\d{12}Z-[0-9a-f]{32}")


def checked_path(path: Path) -> Path:
    """Reject aliases before resolving, including symlinks in parent components."""
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("recovery path contains a symlink")
    return path.resolve(strict=False)


def child(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or any(part in {".", ".."} for part in relative.split("/"))):
        raise ValueError("invalid recovery inventory path")
    path = checked_path(root / relative)
    if not path.is_relative_to(root):
        raise ValueError("recovery inventory escaped its package")
    return path


def digest_file(path: Path) -> dict:
    digest, length = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            length += len(block)
    return {"sha256": digest.hexdigest(), "byte_length": length}


def file_inventory(root: Path) -> list[dict]:
    inventory = []
    for path in sorted(root.rglob("*")):
        checked_path(path)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("recovery package contains a nonregular file")
        inventory.append({"path": path.relative_to(root).as_posix(), **digest_file(path)})
    return inventory


def _validate_structure(manifest: dict) -> None:
    if (manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or type(manifest.get("sample")) is not bool
            or not _IDENTITY.fullmatch(manifest.get("snapshot_id", ""))):
        raise ValueError("invalid or incomplete recovery manifest")
    required, components = manifest.get("required_stores"), manifest.get("components")
    if (not isinstance(required, list) or len(required) != len(set(required))
            or set(required) != REQUIRED_STORES or not isinstance(components, dict)
            or set(components) != set(required)):
        raise ValueError("recovery set omitted or duplicated a required store")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("backup_id") != manifest["snapshot_id"]:
        raise ValueError("invalid recovery checkpoint")
    paths = []
    for name, component in components.items():
        if component.get("checkpoint") != checkpoint:
            raise ValueError("store checkpoint differs from catalog checkpoint: " + name)
        if not component.get("format") or not component.get("paths"):
            raise ValueError("store requires schema identity and explicit package inventory: " + name)
        paths.extend(component["paths"])
    listed = [item["path"] for item in manifest["files"]]
    if len(set(paths)) != len(paths) or len(set(listed)) != len(listed) or set(paths) != set(listed):
        raise ValueError("store inventory differs from package file inventory")
    if not set(manifest.get("protected_paths", [])).issubset(listed):
        raise ValueError("protected source dependency is absent from package")
    if not manifest.get("tool_identities") or not isinstance(manifest.get("parent_inventory"), list):
        raise ValueError("recovery set requires tool identities and parent inventory")


def verify_recovery_set(path: Path) -> dict:
    """Verify the complete package without connecting, reserving or writing."""
    root = checked_path(path)
    try:
        if (root / "INCOMPLETE").exists():
            raise ValueError("incomplete recovery package")
        content = child(root, "manifest.json").read_bytes()
        expected = child(root, "manifest.sha256").read_text().strip()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("recovery manifest checksum failure")
        manifest = json.loads(content)
        _validate_structure(manifest)
        actual = file_inventory(root)
        data = [item for item in actual if item["path"] not in {"manifest.json", "manifest.sha256"}]
        if sorted(data, key=lambda row: row["path"]) != sorted(manifest["files"], key=lambda row: row["path"]):
            raise ValueError("recovery package checksum, length or inventory failure")
        for item in manifest["files"]:
            child(root, item["path"])
        if manifest["components"]["catalog"]["format"] == "papervault-catalog-v1":
            from .recovery_source import verify_file_dependencies
            from .recovery_postgres import validate_catalog_state
            catalog = json.loads(child(root, "catalog/data.json").read_bytes())
            validate_catalog_state(catalog)
            if catalog["checkpoint"] != manifest["checkpoint"]:
                raise ValueError("catalog export checkpoint differs from manifest")
            verify_file_dependencies(root, catalog)
            for name in ("postgres", "neo4j"):
                component = manifest["components"][name]
                data = json.loads(child(root, name + "/data.json").read_bytes())
                if component["format"] == "metadata-sample-not-applicable-v1":
                    if (not manifest["sample"] or data["catalog_sha256"] != digest_file(root / "catalog/data.json")["sha256"]
                            or not data["verified_no_referenced_backend_outputs"]
                            or any(row["manifest"]["tier"] != "metadata" for row in catalog["tables"]["input_manifests"])
                            or any(output["store"] in {"postgres", "neo4j"} for row in catalog["tables"]["knowledge_receipts"]
                                   for output in row["receipt"]["output_inventory"])):
                        raise ValueError("metadata sample omitted required backend material")
                elif name == "postgres":
                    from .recovery_backends import PG_ROLES, validate_postgres_rows
                    if set(data["roles"]) != PG_ROLES or not set(data["roles"].values()).issubset(data["tables"]):
                        raise ValueError("PostgreSQL package omitted required store or exact rows")
                    for table in data["tables"].values():
                        validate_postgres_rows(table)
                elif name == "neo4j":
                    ids = {row["id"] for row in data["nodes"]}
                    if len(ids) != len(data["nodes"]) or any(row["source"] not in ids or row["target"] not in ids
                                                            for row in data["relationships"]):
                        raise ValueError("Neo4j package contains inconsistent nodes/relationships")
            from .recovery_receipts import verify_receipt_closure
            verify_receipt_closure(root, catalog,
                json.loads(child(root, "postgres/data.json").read_bytes()),
                json.loads(child(root, "neo4j/data.json").read_bytes()))
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("unreadable or malformed recovery package") from exc


def _atomic_json(path: Path, value: dict) -> None:
    path = checked_path(path)
    temporary = path.parent / ("." + path.name + "." + uuid4().hex)
    write_once(temporary, canonical_json(value).encode())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def _retention(root: Path, manifest: dict) -> None:
    if manifest["sample"]:
        return
    index = child(root, "retention.json")
    previous = json.loads(index.read_bytes()) if index.exists() else {"retained": []}
    retained = list(previous["retained"])
    old_manifests = {}
    for identity in retained:
        if not isinstance(identity, str) or not _IDENTITY.fullmatch(identity):
            raise ValueError("invalid retained recovery identity")
        old = verify_recovery_set(child(root, identity))
        if old["snapshot_id"] != identity or old["sample"]:
            raise ValueError("retention contains an unqualified recovery set")
        old_manifests[identity] = old
    retained.append(manifest["snapshot_id"])
    # A source version absent from the new complete backup cannot lose its only
    # retained recovery copy. Pins and this closure can keep more than four sets.
    protected = {item["sha256"] for item in manifest["files"]
                 if item["path"] in manifest["protected_paths"]}
    retired = []
    for identity in retained[:-KEEP_COMPLETE]:
        old = old_manifests[identity]
        required = {item["sha256"] for item in old["files"] if item["path"] in old["protected_paths"]}
        if not old.get("pins") and required.issubset(protected):
            retired.append(identity)
    _atomic_json(index, {"schema_version": 1, "keep_complete": KEEP_COMPLETE,
                        "retained": [identity for identity in retained if identity not in retired]})
    for identity in retired:
        # These are verified owned package copies, never source roots or a glob.
        shutil.rmtree(child(root, identity))
    fsync_directory(root)


@contextmanager
def maintenance_deadline(seconds=MAX_WINDOW_SECONDS):
    """Bound blocking file/database exports in the standalone Linux coordinator."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("coordinated maintenance must run in its own main-thread process")
    def expired(signum, frame):
        raise TimeoutError("two-hour maintenance deadline reached; incomplete work retained")
    previous = signal.getsignal(signal.SIGALRM)
    timer = signal.getitimer(signal.ITIMER_REAL)
    if timer[0]:
        raise ValueError("maintenance process already has an active deadline")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def _space_reservation(root, identity, size):
    path = child(root, ".reservation-" + identity)
    try:
        with path.open("xb") as stream:
            os.posix_fallocate(stream.fileno(), 0, max(1, size))
            os.fsync(stream.fileno())
            yield
    finally:
        if path.exists():
            path.unlink()
            fsync_directory(root)


def snapshot(source, root: Path, *, apply: bool = False, sample: bool = False) -> dict:
    """Capture one recovery set using a source's enforced barrier and exporters.

    source.plan() must be read-only. barrier(id) yields an exact checkpoint;
    export(stage, checkpoint) inventories every required store, and register()
    appends an audited catalog event referring back to the captured checkpoint.
    """
    root = checked_path(root)
    plan = source.plan()
    if set(plan["required_stores"]) != REQUIRED_STORES:
        raise ValueError("snapshot plan must cover every required store")
    if plan.get("metadata_sample") and not sample:
        raise ValueError("metadata sample cannot become a complete production recovery set")
    for source_root in plan.get("source_roots", []):
        source_root = checked_path(source_root)
        if root.is_relative_to(source_root) or source_root.is_relative_to(root):
            raise ValueError("recovery output must be isolated from source roots")
    if not apply:
        return {**plan, "apply": False, "root": str(root), "sample": sample,
                "cadence": "weekly", "keep_complete": KEEP_COMPLETE}
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor)[2] < plan["estimated_bytes"] * 2:
        raise ValueError("insufficient peak snapshot capacity")
    root.mkdir(parents=True, exist_ok=True)
    with child(root, ".recovery.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = datetime.now(timezone.utc)
        identity = "recovery-" + stamp.strftime("%Y%m%dT%H%M%S%fZ-") + uuid4().hex
        existing = sum(checked_path(path).stat().st_size for path in root.rglob("*") if path.is_file())
        peak = plan.get("derivative_bytes", 0) + existing + plan["estimated_bytes"] * 2
        capacity = {"existing_recovery_bytes": existing, "reserved_bytes": plan["estimated_bytes"],
            "estimated_peak_bytes": peak, "derivative_soft_cap_bytes": DERIVATIVE_SOFT_CAP,
            "alert_at_bytes": int(DERIVATIVE_SOFT_CAP * ALERT_FRACTION),
            "alert": peak >= DERIVATIVE_SOFT_CAP * ALERT_FRACTION, "action": "alert_only"}
        stage = root / (".incomplete-" + identity)
        stage.mkdir()
        write_once(stage / "INCOMPLETE", b"export in progress\n")
        started = time.monotonic()
        with _space_reservation(root, identity, plan["estimated_bytes"]), maintenance_deadline(MAX_WINDOW_SECONDS), source.barrier(identity) as checkpoint:
            exported = source.export(stage, checkpoint)
            files = [row for row in file_inventory(stage) if row["path"] != "INCOMPLETE"]
            manifest = {**exported, "schema_version": 1, "snapshot_id": identity,
                        "created_at": stamp.isoformat(), "checkpoint": checkpoint,
                        "required_stores": sorted(REQUIRED_STORES), "files": files,
                        "complete": True, "sample": bool(sample),
                        "capacity": capacity,
                        "elapsed_seconds": time.monotonic() - started}
            _validate_structure(manifest)
            if manifest["elapsed_seconds"] >= MAX_WINDOW_SECONDS:
                raise ValueError("snapshot exceeded the two-hour maintenance limit")
            for row in files:
                with child(stage, row["path"]).open("rb") as stream:
                    os.fsync(stream.fileno())
            for directory in sorted((path for path in stage.rglob("*") if path.is_dir()), reverse=True):
                fsync_directory(directory)
            content = canonical_json(manifest).encode()
            write_once(stage / "manifest.json", content)
            write_once(stage / "manifest.sha256", hashlib.sha256(content).hexdigest().encode() + b"\n")
            (stage / "INCOMPLETE").unlink()
            fsync_directory(stage)
            verify_recovery_set(stage)
            destination = root / identity
            os.rename(stage, destination)
            fsync_directory(root)
            source.register(manifest, destination)
        _retention(root, manifest)
        return manifest
