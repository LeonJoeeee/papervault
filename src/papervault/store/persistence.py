"""PG persistence for the Library working set, with optimistic delta batches.

Only committed snapshots advance. A stale batch may merge disjoint bibliographic
metadata once; identity, policy, acquisition and claim changes require a fresh
operator decision. Rejected edits remain dirty and visible to the caller.
"""

from __future__ import annotations

import copy
import hashlib
import os
import threading
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace

import psycopg

from papervault.library.models import Paper
from .repository import (
    AuditEvent, DeletionRefused, PendingJob, PostgresCatalogRepository, Record,
    StaleCompletion, StaleWrite,
)

_REAL_PSYCOPG_CONNECT = psycopg.connect


def _paper_digest(paper: Paper) -> bytes:
    """Compact digest of a Paper's serialized form for fast dirty detection."""
    return hashlib.blake2b(paper.model_dump_json().encode("utf-8"), digest_size=16).digest()


def _close_pool_safe(pool) -> None:
    try:
        pool.close()
    except Exception:
        pass


class _NormalizedView(Mapping):
    """On-demand normalized Paper dict view backed by ``persisted`` records.

    Avoids holding a second 29.8k-paper dict tree in memory while keeping
    ``backend.normalized.get(key)`` / ``backend.normalized[key]`` compatible.
    """

    def __init__(self, persistence: "PostgresPersistence"):
        self._persistence = persistence

    def __getitem__(self, key: str) -> dict:
        record = self._persistence.persisted[key]
        return Paper.model_validate(record.payload).model_dump(mode="json")

    def __iter__(self):
        return iter(self._persistence.persisted)

    def __len__(self) -> int:
        return len(self._persistence.persisted)

    def __contains__(self, key: object) -> bool:
        return key in self._persistence.persisted


class CatalogConflict(RuntimeError):
    def __init__(self, keys):
        self.keys = tuple(sorted(set(keys)))
        super().__init__("catalog conflict: " + ", ".join(self.keys))


# Deliberately narrow: a concurrent authority/policy/identity change cannot be
# judged valid merely because another field happened not to overlap.
_MERGEABLE = {"title", "authors", "year", "venue", "volume", "pages", "issue", "abstract",
              "enriched_at", "citation_count"}


def _overlay(raw, before, after):
    result = copy.deepcopy(raw)
    for field in before.keys() - after.keys():
        result.pop(field, None)
    for field, value in after.items():
        if before.get(field) == value:
            continue
        if isinstance(value, dict) and isinstance(before.get(field), dict):
            result[field] = _overlay(raw.get(field, {}), before[field], value)
        else:
            result[field] = copy.deepcopy(value)
    return result


def merge_metadata(base: Record | None, local: Record, remote: Record | None) -> Record | None:
    if base is None or remote is None:
        return None
    before = Paper.model_validate(base.payload).model_dump(mode="json")
    mine = Paper.model_validate(local.payload).model_dump(mode="json")
    theirs = Paper.model_validate(remote.payload).model_dump(mode="json")
    changed = {k for k in mine.keys() | before.keys() if mine.get(k) != before.get(k)}
    concurrent = {k for k in theirs.keys() | before.keys() if theirs.get(k) != before.get(k)}
    # Unknown fields are authority too. Never silently discard an unknown change.
    if _overlay(base.payload, before, theirs) != remote.payload:
        return None
    if not (changed | concurrent) <= _MERGEABLE or changed & concurrent:
        return None
    return Record.from_payload(local.key, _overlay(remote.payload, before, mine))


class PostgresPersistence:
    def __init__(self, dsn: str, *, schema: str = "catalog", actor: str = "library"):
        if not dsn:
            raise ValueError("postgres authority requires PAPERVAULT_CATALOG_DSN")
        self.dsn, self.schema, self.actor = dsn, schema, actor
        self.persisted: dict[str, Record] = {}
        self._digests: dict[str, bytes] = {}
        self._normalized_override: dict[str, dict] | None = None
        self._normalized_view = _NormalizedView(self)
        self._pool = None
        self._pool_lock = threading.Lock()
        self._pool_finalizer = None
        self._active_repository = None
        self._deferred = None
        self._completion = None
        self.fence_id = None
        self.audit_detail = {}
        self._working = None
        self.operator_rulings = []
        self.last_changed_keys: set[str] = set()
        self._last_save_left_dirty: bool = False

    @property
    def normalized(self):
        return self._normalized_override if self._normalized_override is not None else self._normalized_view

    @normalized.setter
    def normalized(self, value: dict[str, dict]) -> None:
        self._normalized_override = value
        self._digests.clear()

    @staticmethod
    def from_environment():
        return PostgresPersistence(os.environ.get("PAPERVAULT_CATALOG_DSN", ""),
            schema=os.environ.get("PAPERVAULT_CATALOG_SCHEMA", "catalog"),
            actor=os.environ.get("PAPERVAULT_CATALOG_ACTOR", "library"))

    def _get_pool(self):
        with self._pool_lock:
            if self._pool is None:
                from psycopg_pool import ConnectionPool
                pool = ConnectionPool(
                    self.dsn,
                    min_size=1,
                    max_size=4,
                    open=True,
                    kwargs={"autocommit": True, "connect_timeout": 5},
                    check=ConnectionPool.check_connection,
                )
                self._pool = pool
                self._pool_finalizer = weakref.finalize(self, _close_pool_safe, pool)
            return self._pool

    def close(self) -> None:
        with self._pool_lock:
            pool, self._pool = self._pool, None
            finalizer, self._pool_finalizer = self._pool_finalizer, None
        if finalizer is not None:
            finalizer.detach()
        if pool is not None:
            _close_pool_safe(pool)

    @contextmanager
    def repository(self):
        # Respect test monkeypatches on psycopg.connect cleanly; otherwise reuse pooled connections.
        if psycopg.connect is not _REAL_PSYCOPG_CONNECT:
            with psycopg.connect(self.dsn, autocommit=True, connect_timeout=5) as connection:
                repository = PostgresCatalogRepository(connection, schema=self.schema)
                repository.fence_id = self.fence_id
                yield repository
            return
        pool = self._get_pool()
        with pool.connection() as connection:
            repository = PostgresCatalogRepository(connection, schema=self.schema)
            repository.fence_id = self.fence_id
            yield repository

    @contextmanager
    def _repository(self):
        if self._active_repository is not None:
            yield self._active_repository
        else:
            with self.repository() as repository:
                yield repository

    def load(self) -> dict[str, Paper]:
        # Schema changes belong to the explicitly fenced import/preflight job.
        with self.repository() as repository:
            records = list(repository.all())
        papers: dict[str, Paper] = {}
        persisted: dict[str, Record] = {}
        digests: dict[str, bytes] = {}
        for r in records:
            paper = Paper.model_validate(r.payload)
            papers[r.key] = paper
            persisted[r.key] = r
            digests[r.key] = _paper_digest(paper)
        self.persisted = persisted
        self._digests = digests
        self._normalized_override = None
        return papers

    def _jobs(self, records, revisions):
        jobs = []
        for record in records:
            jobs.append(PendingJob("catalog-export:" + record.key, "catalog_export", record.key,
                state="queued", attempts=revisions[record.key] + 1,
                detail={"payload_sha256": record.payload_sha256}))
            claim = record.payload.get("reopen_claim")
            if claim and claim["phase"] in {"queued", "dispatched"}:
                jobs.append(PendingJob("reopen:" + record.key, "reopen", record.key,
                    state=claim["phase"], attempts=claim["generation"], lease_id=claim["lease_id"],
                    detail={"campaign_id": claim["campaign_id"]}))
        return tuple(jobs)

    def _has_remaining_dirty(self, papers, persisted, digests, working_norm, *, exclude: set[str]) -> bool:
        """Check whether any non-excluded key in ``papers`` differs from ``persisted``."""
        if persisted.keys() != papers.keys():
            return True
        if working_norm is not None:
            return any(
                key not in exclude and working_norm.get(key) != paper.model_dump(mode="json")
                for key, paper in papers.items()
            )
        for key, paper in papers.items():
            if key in exclude:
                continue
            digest = _paper_digest(paper)
            if digests.get(key) == digest:
                continue
            before = Paper.model_validate(persisted[key].payload).model_dump(mode="json")
            if before != paper.model_dump(mode="json"):
                return True
            digests[key] = digest
        return False

    def _accept(self, records, revisions, papers, captured):
        left_dirty = False
        changed_keys: set[str] = set()
        for record in records:
            record = replace(record, revision=revisions[record.key])
            paper = Paper.model_validate(record.payload)
            # Keep live object identity: queues retain Paper references.
            current = papers[record.key]
            saved = paper.model_dump(mode="json")
            current_dump = current.model_dump(mode="json")
            prior_capture = captured[record.key]
            updated_live = False
            for name in Paper.model_fields:
                if (saved.get(name) != prior_capture.get(name)
                        and current_dump.get(name) == prior_capture.get(name)):
                    setattr(current, name, copy.deepcopy(getattr(paper, name)))
                    updated_live = True
            self.persisted[record.key] = record
            self._digests[record.key] = _paper_digest(paper)
            if self._normalized_override is not None:
                self._normalized_override[record.key] = saved
            changed_keys.add(record.key)
            if current_dump != prior_capture:
                final_dump = current.model_dump(mode="json") if updated_live else current_dump
                if final_dump != saved:
                    left_dirty = True
        self.last_changed_keys = changed_keys
        if not left_dirty:
            left_dirty = self._has_remaining_dirty(
                papers, self.persisted, self._digests, self._normalized_override, exclude=changed_keys
            )
        self._last_save_left_dirty = left_dirty

    def _flush_in_tx(self, repository, records, revisions, persisted, rulings):
        operator = rulings[-1] if rulings else None
        event = AuditEvent(operator["actor"] if operator else self.actor,
                           operator["event"] if operator else "state_flush",
                           {**self.audit_detail, **(operator or {})},
                           pending_jobs=self._jobs(records, revisions))
        with repository._write(event):
            # Rulings are decisions about this exact working snapshot,
            # including keys whose desired state is already present.
            for ruling in rulings:
                for key in ruling["keys"]:
                    expected = persisted.get(key)
                    current = repository.get(key)
                    if (expected.revision if expected else 0) != (current.revision if current else 0):
                        raise StaleWrite(key, expected.revision if expected else 0,
                                         current.revision if current else None)
            result = repository.flush(records, expected_revisions=revisions, audit=event)
            for ruling in rulings:
                repository.import_ruling(ruling, audit=AuditEvent(
                    ruling["actor"], ruling["event"], ruling))
            for record in records:
                previous = persisted.get(record.key)
                old = previous.payload.get("reopen_claim") if previous else None
                new = record.payload.get("reopen_claim")
                if old and old["phase"] in {"queued", "dispatched"} and not new:
                    raise StaleCompletion("cannot remove an active catalog claim")
                if old and old["phase"] in {"queued", "dispatched"} and new and new["phase"] in {"completed", "not_sent", "released"}:
                    generation = new["generation"] - int(new["phase"] == "released")
                    repository.complete_job("reopen:" + record.key,
                        attempt=generation, lease_id=new["lease_id"], outcome=new["phase"],
                        expected_attempt=old["generation"], expected_lease=old["lease_id"],
                        audit=AuditEvent(self.actor, "job_completion", {"lease_id": new["lease_id"]}))
        return result

    def save(self, papers: dict[str, Paper]) -> list[dict]:
        self.last_changed_keys = set()
        self._last_save_left_dirty = False
        if self._working is not None:
            persisted, digests, working_norm = self._working
        else:
            persisted, digests, working_norm = self.persisted, self._digests, self._normalized_override
        removed = persisted.keys() - papers.keys()
        if removed:
            raise DeletionRefused("PG record deletion is refused in P2: " + ", ".join(sorted(removed)))
        records, captured = [], {}
        if working_norm is not None:
            for key, paper in papers.items():
                current = paper.model_dump(mode="json")
                before = working_norm.get(key)
                if before == current:
                    continue
                captured[key] = current
                raw = _overlay(persisted[key].payload, before, current) if before is not None else current
                records.append(Record.from_payload(key, raw))
        else:
            for key, paper in papers.items():
                digest = _paper_digest(paper)
                if key in persisted and digests.get(key) == digest:
                    continue
                current = paper.model_dump(mode="json")
                before = (Paper.model_validate(persisted[key].payload).model_dump(mode="json")
                          if key in persisted else None)
                if before == current:
                    digests[key] = digest
                    continue
                captured[key] = current
                raw = _overlay(persisted[key].payload, before, current) if before is not None else current
                records.append(Record.from_payload(key, raw))
        rulings = list(self.operator_rulings)
        if not records and not rulings:
            return []
        revisions = {r.key: persisted[r.key].revision if r.key in persisted else 0
                     for r in records}
        with self._repository() as repository:
            for retry in range(2):
                try:
                    result = self._flush_in_tx(repository, records, revisions, persisted, rulings)
                    break
                except StaleWrite as exc:
                    if retry or self._completion is not None or rulings:
                        if rulings:
                            # Stale rulings cannot commit without a fresh operator
                            # decision; drop them so they cannot wedge unrelated saves.
                            del self.operator_rulings[:len(rulings)]
                        self._last_save_left_dirty = True
                        raise CatalogConflict([exc.key]) from exc
                    merged, conflicts = [], []
                    for record in records:
                        remote = repository.get(record.key)
                        if (remote.revision if remote else 0) == revisions[record.key]:
                            merged.append(record)
                            continue
                        candidate = merge_metadata(persisted.get(record.key), record, remote)
                        if candidate is None:
                            conflicts.append(record.key)
                        else:
                            merged.append(candidate)
                            revisions[record.key] = remote.revision
                    if conflicts:
                        # Flush non-conflicting records so one conflicted key cannot
                        # permanently wedge unrelated dirty records in the working set.
                        if merged:
                            try:
                                partial_revs = {r.key: revisions[r.key] for r in merged}
                                partial_result = self._flush_in_tx(
                                    repository, merged, partial_revs, persisted, ())
                                self._accept(merged, partial_result, papers, captured)
                            except StaleWrite as nested_exc:
                                conflicts.append(nested_exc.key)
                        self._last_save_left_dirty = True
                        raise CatalogConflict(conflicts) from exc
                    records = merged
        if self._completion is None:
            self._accept(records, result, papers, captured)
            del self.operator_rulings[:len(rulings)]
        else:
            changed_keys: set[str] = set()
            for record in records:
                persisted[record.key] = replace(record, revision=result[record.key])
                saved_paper = Paper.model_validate(record.payload)
                digests[record.key] = _paper_digest(saved_paper)
                if working_norm is not None:
                    working_norm[record.key] = saved_paper.model_dump(mode="json")
                changed_keys.add(record.key)
            self.last_changed_keys = changed_keys
            self._last_save_left_dirty = self._has_remaining_dirty(
                papers, persisted, digests, working_norm, exclude=changed_keys
            )
            deferred = {r.key: r for r in self._deferred[0]} if self._deferred else {}
            deferred.update({r.key: r for r in records})
            previous_capture = self._deferred[3] if self._deferred else {}
            previous_capture.update({r.key: captured[r.key] for r in records})
            self._deferred = (list(deferred.values()), {k: persisted[k].revision for k in deferred}, papers, previous_capture)
        return rulings

    def dirty(self, papers):
        if self.persisted.keys() != papers.keys():
            return True
        if self._normalized_override is not None:
            return any(
                self._normalized_override.get(key) != paper.model_dump(mode="json")
                for key, paper in papers.items()
            )
        for key, paper in papers.items():
            digest = _paper_digest(paper)
            if self._digests.get(key) == digest:
                continue
            before = Paper.model_validate(self.persisted[key].payload).model_dump(mode="json")
            if before != paper.model_dump(mode="json"):
                return True
            self._digests[key] = digest
        return False

    @contextmanager
    def completion(self, token):
        """Fence a staged publication BEFORE touching assets; commit batch + job together.

        The file ledger remains the pre-dispatch budget gate. PostgreSQL owns
        completion acceptance. A failed commit preserves stage files for forward
        recovery; it cannot advance the in-memory persisted snapshot.
        """
        audit = AuditEvent(self.actor, "job_completion", {"attempt": token.generation,
                                                      "lease_id": token.lease_id})
        with self.repository() as repository:
            try:
                with repository._write(audit):
                    repository.check_completion("reopen:" + token.key,
                        attempt=token.generation, lease_id=token.lease_id)
                    current = repository.get(token.key)
                    expected = self.persisted.get(token.key)
                    if current is None or expected is None or current.revision != expected.revision:
                        raise StaleCompletion(f"catalog changed before completion for {token.key}")
                    self._active_repository, self._completion = repository, token
                    repository.extraction_completion_token = token
                    # Copy-on-write shallow dicts instead of deepcopy over the full catalog.
                    self._working = (
                        dict(self.persisted),
                        dict(self._digests),
                        dict(self._normalized_override) if self._normalized_override is not None else None,
                    )
                    yield
                if self._deferred:
                    self._accept(*self._deferred)
            finally:
                repository.extraction_completion_token = None
                self._active_repository = self._completion = self._deferred = self._working = None


_ORIG_POSTGRES_PERSISTENCE_LOAD = PostgresPersistence.load
