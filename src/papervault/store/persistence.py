"""PG persistence for the Library working set, with optimistic delta batches.

Only committed snapshots advance. A stale batch may merge disjoint bibliographic
metadata once; identity, policy, acquisition and claim changes require a fresh
operator decision. Rejected edits remain dirty and visible to the caller.
"""

from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from dataclasses import replace

import psycopg

from papervault.library.models import Paper
from .repository import (
    AuditEvent, DeletionRefused, PendingJob, PostgresCatalogRepository, Record,
    StaleCompletion, StaleWrite,
)


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
        self.normalized: dict[str, dict] = {}
        self._active_repository = None
        self._deferred = None
        self._completion = None
        self.fence_id = None
        self.audit_detail = {}
        self._working = None
        self.operator_rulings = []

    @staticmethod
    def from_environment():
        return PostgresPersistence(os.environ.get("PAPERVAULT_CATALOG_DSN", ""),
            schema=os.environ.get("PAPERVAULT_CATALOG_SCHEMA", "catalog"),
            actor=os.environ.get("PAPERVAULT_CATALOG_ACTOR", "library"))

    @contextmanager
    def repository(self):
        with psycopg.connect(self.dsn, autocommit=True, connect_timeout=5) as connection:
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
        papers = {r.key: Paper.model_validate(r.payload) for r in records}
        self.persisted = {r.key: r for r in records}
        self.normalized = {key: paper.model_dump(mode="json") for key, paper in papers.items()}
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

    def _accept(self, records, revisions, papers, captured):
        for record in records:
            record = replace(record, revision=revisions[record.key])
            paper = Paper.model_validate(record.payload)
            # Keep live object identity: queues retain Paper references.
            current = papers[record.key]
            saved = paper.model_dump(mode="json")
            for name in Paper.model_fields:
                if (saved.get(name) != captured[record.key].get(name)
                        and current.model_dump(mode="json").get(name) == captured[record.key].get(name)):
                    setattr(current, name, copy.deepcopy(getattr(paper, name)))
            self.persisted[record.key] = record
            self.normalized[record.key] = paper.model_dump(mode="json")

    def save(self, papers: dict[str, Paper]) -> list[dict]:
        persisted, normalized = self._working if self._working is not None else (self.persisted, self.normalized)
        removed = persisted.keys() - papers.keys()
        if removed:
            raise DeletionRefused("PG record deletion is refused in P2: " + ", ".join(sorted(removed)))
        records, captured = [], {}
        for key, paper in papers.items():
            current = paper.model_dump(mode="json")
            before = normalized.get(key)
            if before == current:
                continue
            captured[key] = current
            raw = _overlay(persisted[key].payload, before, current) if before else current
            records.append(Record.from_payload(key, raw))
        rulings = list(self.operator_rulings)
        if not records and not rulings:
            return []
        revisions = {r.key: persisted[r.key].revision if r.key in persisted else 0
                     for r in records}
        with self._repository() as repository:
            for retry in range(2):
                try:
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
                    break
                except StaleWrite as exc:
                    if retry or self._completion is not None or rulings:
                        raise CatalogConflict([exc.key]) from exc
                    merged, conflicts = [], []
                    for record in records:
                        remote = repository.get(record.key)
                        if remote and remote.revision == revisions[record.key]:
                            merged.append(record)
                            continue
                        candidate = merge_metadata(self.persisted.get(record.key), record, remote)
                        if candidate is None:
                            conflicts.append(record.key)
                        else:
                            merged.append(candidate)
                            revisions[record.key] = remote.revision
                    if conflicts:
                        raise CatalogConflict(conflicts) from exc
                    records = merged
        if self._completion is None:
            self._accept(records, result, papers, captured)
            del self.operator_rulings[:len(rulings)]
        else:
            for record in records:
                persisted[record.key] = replace(record, revision=result[record.key])
                normalized[record.key] = Paper.model_validate(record.payload).model_dump(mode="json")
            deferred = {r.key: r for r in self._deferred[0]} if self._deferred else {}
            deferred.update({r.key: r for r in records})
            previous_capture = self._deferred[3] if self._deferred else {}
            previous_capture.update({r.key: captured[r.key] for r in records})
            self._deferred = (list(deferred.values()), {k: persisted[k].revision for k in deferred}, papers, previous_capture)
        return rulings

    def dirty(self, papers):
        return (self.persisted.keys() != papers.keys() or any(
            self.normalized.get(key) != paper.model_dump(mode="json") for key, paper in papers.items()))

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
                    self._working = (dict(self.persisted), copy.deepcopy(self.normalized))
                    yield
                if self._deferred:
                    self._accept(*self._deferred)
            finally:
                repository.extraction_completion_token = None
                self._active_repository = self._completion = self._deferred = self._working = None
