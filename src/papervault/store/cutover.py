"""Explicit, restartable P2 rehearsal handoff; live activation stays owner-gated."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from psycopg import sql

from papervault.library.store import Library, _atomic_write
from .adapter import LibraryAdapter
from .compare import compare_catalog
from .exporter import export_catalog
from .fencing import exclusive_fence, handoff_state
from .importer import import_catalog
from .repository import AuditEvent


def set_authority(path: Path, authority: str):
    text = path.read_bytes().decode("utf-8") if path.exists() else ""
    lines = text.splitlines(keepends=True)
    retained = [line for line in lines if not re.match(r"^\s*(?:export\s+)?PAPERVAULT_CATALOG_AUTHORITY\s*=", line)]
    text = "".join(retained)
    if text and not text.endswith("\n"):
        text += "\n"
    # A config file may contain credentials; do not create a backup or log it.
    old_mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, old_mode)
    os.close(fd)
    _atomic_write(path, text + f"PAPERVAULT_CATALOG_AUTHORITY={authority}\n", durable=True)
    os.chmod(path, old_mode)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Cutover:
    def __init__(self, root: Path, env: Path, repository):
        self.root, self.env = Path(root).resolve(), Path(env).resolve()
        if self.root.is_relative_to(Path("/data/paper-vault").resolve()):
            raise ValueError("P2 rehearsal refuses the live vault")
        if not self.env.is_relative_to(self.root):
            raise ValueError("rehearsal env-file must be inside the vault copy")
        self.repository = repository
        self.index = self.root / "index.json"
        self.frozen = self.root / ".catalog-frozen-index.json"
        self.state_path = self.root / ".catalog-handoff.json"
        self.target = {"database": repository.connection.info.dbname, "schema": repository.schema,
                       "host": repository.connection.info.host, "port": repository.connection.info.port}

    def state(self):
        state = handoff_state(self.root)
        if state and state.get("target") != self.target:
            raise RuntimeError("handoff target differs from recorded database/schema")
        return state

    def _record(self, state, **updates):
        state.update(updates)
        _atomic_write(self.state_path, json.dumps(state, sort_keys=True, indent=2), durable=True)
        return state

    def _audit_head(self):
        return self.repository.connection.execute(sql.SQL("SELECT coalesce(max(id), 0) AS id FROM {}").format(
            self.repository._table("audit"))).fetchone()["id"]

    def _parity(self, index=None):
        report = compare_catalog(LibraryAdapter.from_snapshot(index or self.index), self.repository)
        if report["difference_rate"] != 0 or report["json_count"] != report["postgres_count"]:
            raise RuntimeError("catalog parity failed; no-go")
        return report

    def _only_probe_writes(self, state):
        rows = self.repository.connection.execute(sql.SQL(
            "SELECT actor, event, detail FROM {} WHERE id > %s ORDER BY id"
        ).format(self.repository._table("audit")), (state["audit_head"],))
        return all((r["actor"] == "cutover_probe" and r["event"] == "state_flush"
                    and r["detail"].get("handoff_probe") == state["fence_id"])
                   or (state["phase"] == "rolling_back" and r["actor"] == "cutover"
                       and r["event"] == "rollback") for r in rows)

    def preflight(self, *, apply=False):
        start = time.monotonic()
        with exclusive_fence(self.root):
            state = self.state()
            if state.get("writes_resumed") or state.get("phase") in {"flipping", "flipped_fenced", "rolling_back"}:
                raise RuntimeError("handoff already advanced; use rollback or forward recovery")
            adapter = LibraryAdapter.from_snapshot(self.index)
            if not apply:
                report = import_catalog(adapter)
                report.update(go=False, reason="dry run; --apply required for fenced delta import")
                return report
            if not state or state.get("phase") == "rolled_back":
                state = {"fence_id": uuid.uuid4().hex, "writes_resumed": False, "target": self.target}
            # Persist intent BEFORE acquiring the DB fence. An interrupted command
            # refuses fresh Libraries even if the process's flock is released.
            self._record(state, phase="preparing")
            self.repository.fence_id = state["fence_id"]
            self.repository.ensure_schema(audit=AuditEvent("cutover", "schema"))
            self.repository.set_fence(state["fence_id"], audit=AuditEvent("cutover", "writer_fence"))
            report = import_catalog(adapter, self.repository)
            if report["integrity_errors"] or report["difference_rate"] != 0:
                raise RuntimeError("delta import failed integrity/parity; writers remain fenced")
            self._parity()
            self._record(state, phase="preflighted", json_sha256=_sha(self.index),
                         audit_head=self._audit_head())
            return {**report, "go": True, "seconds": time.monotonic() - start}

    def _fenced(self, phase):
        state = self.state()
        if state.get("writes_resumed"):
            raise RuntimeError("writes resumed: frozen rollback forbidden; forward recovery required")
        if state.get("phase") not in phase:
            raise RuntimeError("handoff phase does not permit this operation")
        self.repository.fence_id = state["fence_id"]
        return state

    def flip(self):
        start = time.monotonic()
        with exclusive_fence(self.root):
            state = self._fenced({"preflighted"})
            if _sha(self.index) != state["json_sha256"] or self._audit_head() != state["audit_head"]:
                raise RuntimeError("source or catalog changed since preflight")
            self._parity()
            export_catalog(self.repository, self.frozen, verify=True)
            self._record(state, phase="flipping", frozen_sha256=_sha(self.frozen))
            set_authority(self.env, "postgres")
            # Probe the same facade used at boot/shutdown while all writers stay
            # fenced. Restore the field through a second guarded save; retain the
            # audit trail. No synthetic record or deletion path is introduced.
            library = Library(self.root, authority="postgres", _handoff_token=state["fence_id"])
            try:
                candidates = [r for r in library._catalog_backend.persisted.values() if "title" in r.payload]
                if not candidates:
                    raise RuntimeError("fenced first-save probe requires a record with an explicit title")
                paper = library.get(candidates[0].key)
                title = paper.title
                library._catalog_backend.actor = "cutover_probe"
                library._catalog_backend.audit_detail = {"handoff_probe": state["fence_id"]}
                paper.title = title + " [P2 fenced persistence probe]"
                library.save(force=True, durable=True)
                paper.title = title
                library.save(force=True, durable=True)
            finally:
                library.close()
            report = export_catalog(self.repository, self.index, verify=True)
            self._record(state, phase="flipped_fenced", audit_head=self._audit_head())
            return {**report, "phase": state["phase"], "seconds": time.monotonic() - start}

    def rollback(self):
        start = time.monotonic()
        with exclusive_fence(self.root):
            state = self._fenced({"preparing", "flipped_fenced", "flipping", "rolling_back"})
            if state["phase"] == "preparing":
                # Preparation can fail before schema creation or after partial
                # import. JSON authority is unchanged; no snapshot is restored.
                exists = self.repository.connection.execute("SELECT to_regclass(%s) AS name",
                    (self.repository.schema + ".writer_fence",)).fetchone()
                if exists["name"]:
                    self.repository.set_fence(None, audit=AuditEvent("cutover", "preflight_abort"))
                self._record(state, phase="rolled_back")
                return {"phase": "rolled_back", "seconds": time.monotonic() - start}
            if _sha(self.frozen) != state["frozen_sha256"] or not self._only_probe_writes(state):
                raise RuntimeError("unaccounted writes or changed snapshot: forward recovery required")
            self._record(state, phase="rolling_back")
            _atomic_write(self.index, self.frozen.read_text(), durable=True)
            set_authority(self.env, "json")
            self.repository.set_fence(None, audit=AuditEvent("cutover", "rollback"))
            self._record(state, phase="rolled_back")
        library = Library(self.root, authority="json")
        try:
            if len(library.all_papers(include_quarantined=True)) != len(list(self.repository.all())):
                raise RuntimeError("rollback load verification failed")
        finally:
            library.close()
        return {"phase": "rolled_back", "seconds": time.monotonic() - start}

    def resume(self, *, restart: list[str], health: list[str], reason: str):
        if not restart or not health or not reason.strip():
            raise ValueError("resume requires explicit restart/health commands and a reason")
        start = time.monotonic()
        with exclusive_fence(self.root):
            state = self._fenced({"flipped_fenced"})
            if self._audit_head() != state["audit_head"]:
                raise RuntimeError("unaccounted catalog writes; forward recovery required")
            self._parity()
            # This durable one-way boundary PRECEDES every possible writer boot.
            self._record(state, phase="resuming", writes_resumed=True, reason=reason)
            self.repository.set_fence(None, audit=AuditEvent("cutover", "writes_resumed", {"reason": reason}))
            self._record(state, phase="resumed")
        for label, command in (("restart", restart), ("health", health)):
            result = subprocess.run(command, capture_output=True, timeout=300)
            if result.returncode:
                raise RuntimeError(f"{label} failed after writes-resumed marker; forward recovery required")
        report = export_catalog(self.repository, self.index, verify=True)
        return {**report, "phase": "resumed", "seconds": time.monotonic() - start}
