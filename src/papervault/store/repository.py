"""Lossless records and the transactional shadow-catalog interface."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from importlib.resources import files
from typing import Iterator, Protocol

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from papervault.library.models import Paper
from papervault.library.state_v2 import derive_legacy, project_domain_policy


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def payload_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def render_catalog_ddl(ddl: str, schema: str) -> str:
    """Qualify SQL objects while preserving authority/store value literals."""
    quoted = sql.Identifier(schema).as_string()
    ddl = re.sub(r"\bcatalog(?=\.)", lambda _: quoted, ddl)
    return re.sub(r"(CREATE\s+SCHEMA\s+IF\s+NOT\s+EXISTS\s+)catalog(?=\s*;)",
                  lambda match: match[1]+quoted, ddl, flags=re.IGNORECASE)


def normalized_identity(value: str | None) -> str | None:
    """F0 comparison normalization; retain arXiv versions pending identity adjudication."""
    return value.strip().lower() or None if value else None


@dataclass(frozen=True)
class Record:
    key: str
    doi: str | None
    arxiv_id: str | None
    download_status: str
    retrieval: str
    policy: str
    merge: str
    quarantine: bool
    source: str
    year: int | None
    title: str
    payload: dict
    payload_version: int = 1
    revision: int = 0

    @classmethod
    def from_payload(cls, key: str, payload: dict) -> Record:
        if not isinstance(payload, dict) or not isinstance(key, str) or not key.strip():
            raise ValueError("record requires a nonempty key and object payload")
        if payload.get("key") != key:
            raise ValueError(f"snapshot key {key!r} differs from payload key")
        canonical_json(payload)  # Reject values that cannot survive a JSONB round trip.
        paper = Paper.model_validate(payload)
        state = project_domain_policy(paper.terminal_state, paper) if paper.terminal_state else (
            derive_legacy(paper, include_domain=True)
        )
        return cls(
            key=key, doi=normalized_identity(paper.doi),
            arxiv_id=normalized_identity(paper.arxiv_id),
            download_status=paper.download_status, retrieval=state.retrieval.value,
            policy=state.policy.value, merge=state.merge.value,
            quarantine=state.quarantine or paper.merge_intent is not None,
            source=paper.source, year=paper.year, title=paper.title,
            payload=copy.deepcopy(payload),
        )

    @property
    def payload_sha256(self) -> str:
        return payload_hash(self.payload)


@dataclass(frozen=True)
class PendingJob:
    id: str
    kind: str
    key: str
    state: str = "pending"
    attempts: int = 0
    enqueued_at: str | datetime | None = None
    started_at: str | datetime | None = None
    finished_at: str | datetime | None = None
    detail: dict = field(default_factory=dict)
    lease_id: str | None = None

    def __post_init__(self):
        if (any(not isinstance(value, str) or not value.strip()
                for value in (self.id, self.kind, self.key, self.state))
                or type(self.attempts) is not int or self.attempts < 0):
            raise ValueError("job requires id, kind, key, state and nonnegative attempts")
        if self.lease_id is not None and (not isinstance(self.lease_id, str) or not self.lease_id.strip()):
            raise ValueError("lease_id must be a nonempty string when supplied")
        for name in ("enqueued_at", "started_at", "finished_at"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (str, datetime)):
                    raise ValueError("job timestamps must be ISO strings or datetimes")
                value = datetime.fromisoformat(value) if isinstance(value, str) else value
                if value.tzinfo is None:
                    raise ValueError("job timestamps must be timezone-aware")
                object.__setattr__(self, name, value)
        canonical_json(self.detail)


@dataclass(frozen=True)
class AuditEvent:
    actor: str
    event: str
    detail: dict = field(default_factory=dict)
    pending_jobs: tuple[PendingJob, ...] = ()

    def __post_init__(self):
        if not self.actor.strip() or not self.event.strip():
            raise ValueError("audit requires a nonempty actor and event")
        canonical_json(self.detail)


class CatalogRepository(Protocol):
    def get(self, key: str) -> Record | None: ...
    def all(self) -> Iterator[Record]: ...
    def upsert(self, record: Record, *, audit: AuditEvent, expected_revision: int) -> bool: ...
    def set_status(self, key: str, status: str, *, audit: AuditEvent, expected_revision: int) -> bool: ...
    def list_by_status(self, status: str) -> list[str]: ...
    def resolve_alias(self, key: str) -> str: ...
    def all_aliases(self) -> dict[str, str]: ...
    def add_alias(self, alias: str, canonical: str, *, audit: AuditEvent) -> bool: ...


_RECORD_COLUMNS = (
    "key", "doi", "arxiv_id", "download_status", "retrieval", "policy", "merge",
    "quarantine", "source", "year", "title", "payload", "payload_version",
)


class StaleWrite(RuntimeError):
    def __init__(self, key: str, expected: int, actual: int | None):
        self.key, self.expected, self.actual = key, expected, actual
        super().__init__(f"catalog conflict for {key}: expected revision {expected}, actual {actual}")


class StaleCompletion(RuntimeError):
    """An attempt no longer owns the active catalog job; no completion committed."""


class DeletionRefused(RuntimeError):
    """P2 PG authority retains records until retention/tombstones are specified."""


class PostgresCatalogRepository:
    """Synchronous, explicit-connection repository; caller owns connection lifetime.

    Supply an autocommit connection so each public write owns its commit. A
    schema-local transaction lock serializes writers, including alias cycle
    checks and imports. Revision predicates still reject stale observations.
    """

    def __init__(self, connection, *, schema: str = "catalog", read_only: bool = False):
        if not re.fullmatch(r"catalog(?:_[a-z0-9_]+)?", schema):
            raise ValueError("shadow namespace must be catalog or catalog_<suffix>")
        if not connection.autocommit:
            raise ValueError("repository requires an autocommit connection")
        self.connection = connection
        self.connection.row_factory = dict_row
        self.schema = schema
        self.read_only = read_only
        self.fence_id = None

    def _table(self, name: str):
        return sql.Identifier(self.schema, name)

    @contextmanager
    def _write(self, audit: AuditEvent):
        if not isinstance(audit, AuditEvent):
            raise TypeError("write requires an AuditEvent")
        if self.read_only:
            raise PermissionError("shadow comparator repository is read-only")
        with self.connection.transaction():
            self.connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                                    ("papervault:" + self.schema,))
            exists = self.connection.execute("SELECT to_regclass(%s) AS name",
                                             (self.schema + ".writer_fence",)).fetchone()
            if exists and exists["name"]:
                row = self.connection.execute(sql.SQL("SELECT fence_id FROM {}").format(
                    self._table("writer_fence"))).fetchone()
                if row and row["fence_id"] and row["fence_id"] != self.fence_id:
                    from .fencing import WriterFenced
                    raise WriterFenced("catalog writers durably fenced for handoff")
            yield

    def set_fence(self, fence_id: str | None, *, audit: AuditEvent):
        with self._write(audit):
            self.connection.execute(sql.SQL("UPDATE {} SET fence_id = %s").format(
                self._table("writer_fence")), (fence_id,))
            self._finish(None, audit)
        self.fence_id = fence_id

    def _finish(self, key: str | None, audit: AuditEvent):
        self.connection.execute(sql.SQL(
            "INSERT INTO {} (actor, event, key, detail) VALUES (%s, %s, %s, %s)"
        ).format(self._table("audit")),
            (audit.actor, audit.event, key, Jsonb(audit.detail)))
        for job in audit.pending_jobs:
            self._write_job(job)

    def ensure_schema(self, *, audit: AuditEvent) -> None:
        with self._write(audit):
            ddl = files("papervault.store").joinpath("schema.sql").read_text()
            ddl = render_catalog_ddl(ddl, self.schema)
            self.connection.execute(ddl)
            row = self.connection.execute(sql.SQL(
                "INSERT INTO {} (version, name) VALUES (1, 'P1 shadow catalog') "
                "ON CONFLICT (version) DO NOTHING RETURNING version"
            ).format(self._table("schema_version"))).fetchone()
            if row:
                self._finish(None, audit)
            row = self.connection.execute(sql.SQL(
                "INSERT INTO {} (version, name) VALUES (2, 'P2 guarded catalog persistence') "
                "ON CONFLICT (version) DO NOTHING RETURNING version"
            ).format(self._table("schema_version"))).fetchone()
            if row:
                self._finish(None, audit)

    def ensure_knowledge_schema(self, *, audit: AuditEvent) -> None:
        """Explicit administrative P4 migration; ordinary startup never calls it."""
        with self._write(audit):
            ddl = files("papervault.store").joinpath("knowledge_schema.sql").read_text()
            ddl = render_catalog_ddl(ddl, self.schema)
            self.connection.execute(ddl)
            # Existing registrations predate transactional P4 capture. Seed
            # unknown pending requirements at this explicit migration boundary.
            records = list(self.connection.execute(sql.SQL("SELECT key,revision,payload FROM {}").format(
                self._table("records"))))
            with self.connection.cursor() as cursor:
                cursor.executemany(sql.SQL("INSERT INTO {}(record_key,record_revision,payload_digest) "
                    "VALUES(%s,%s,%s) ON CONFLICT DO NOTHING").format(self._table("knowledge_pending_inputs")),
                    [(r["key"], r["revision"], payload_hash(r["payload"])) for r in records])
                if cursor.rowcount > 0:
                    self.connection.execute(sql.SQL("UPDATE {} SET requirements_revision=requirements_revision+1").format(
                        self._table("knowledge_runtime")))
            changed = self.connection.execute(sql.SQL(
                "INSERT INTO {}(version,name) VALUES(4,'P4 catalog knowledge commands'),"
                "(5,'P5a extraction derivation registration'),"
                "(6,'P5b isolated recovery generations'),"
                "(7,'P5c disposable extract lifecycle') "
                "ON CONFLICT(version) DO NOTHING RETURNING version"
            ).format(self._table("schema_version"))).fetchone()
            if changed:
                self._finish(None, audit)

    def _track_knowledge_requirement_change(self, record: Record, revision: int):
        """Capture and invalidate in the Library writer transaction, when installed."""
        exists = self.connection.execute("SELECT to_regclass(%s) AS name",
            (self.schema + ".knowledge_pending_inputs",)).fetchone()
        if not exists or not exists["name"]:
            return
        self.connection.execute(sql.SQL(
            "INSERT INTO {}(record_key,record_revision,payload_digest) VALUES(%s,%s,%s) "
            "ON CONFLICT DO NOTHING"
        ).format(self._table("knowledge_pending_inputs")), (record.key, revision, record.payload_sha256))
        affected = list(self.connection.execute(sql.SQL(
            "UPDATE {} r SET invalidated=TRUE,revision=r.revision+1 FROM {} u "
            "WHERE r.unit_id=u.unit_id AND u.record_key=%s RETURNING r.generation_id"
        ).format(self._table("knowledge_requirements"), self._table("units")), (record.key,)))
        affected.extend(self.connection.execute(sql.SQL(
            "UPDATE {} i SET invalidated=TRUE,revision=i.revision+1 FROM {} u "
            "WHERE i.parent_unit_id=u.unit_id AND u.record_key=%s RETURNING i.generation_id"
        ).format(self._table("current_inventories"), self._table("units")), (record.key,)))
        self.connection.execute(sql.SQL(
            "UPDATE {} SET requirements_revision=requirements_revision+1 WHERE singleton"
        ).format(self._table("knowledge_runtime")))
        from .knowledge import KnowledgeRepository
        from .knowledge_commands import outbox, snapshot
        knowledge = KnowledgeRepository(self)
        outbox(knowledge, "capture_input", {"record_key": record.key, "record_revision": revision},
               stable_id=f"capture:{record.key}:{revision}")
        for generation_id in sorted({row["generation_id"] for row in affected}):
            snapshot(knowledge, generation_id)

    def get(self, key: str) -> Record | None:
        row = self.connection.execute(sql.SQL("SELECT {} FROM {} WHERE key = %s").format(
            sql.SQL(", ").join(map(sql.Identifier, (*_RECORD_COLUMNS, "revision"))), self._table("records"),
        ), (key,)).fetchone()
        return Record(**row) if row else None

    def all(self) -> Iterator[Record]:
        rows = self.connection.execute(sql.SQL("SELECT {} FROM {} ORDER BY key").format(
            sql.SQL(", ").join(map(sql.Identifier, (*_RECORD_COLUMNS, "revision"))), self._table("records"),
        ))
        return (Record(**row) for row in rows)

    def _upsert(self, record: Record, audit: AuditEvent, expected_revision: int) -> bool:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        if replace(record, revision=0) != Record.from_payload(record.key, record.payload):
            raise ValueError("record columns do not match payload version 1")
        current = self.get(record.key)
        if ((current is None and expected_revision != 0)
                or (current is not None and current.revision != expected_revision)):
            raise StaleWrite(record.key, expected_revision, current.revision if current else None)
        if current and replace(current, revision=0) == replace(record, revision=0):
            return False
        if record.payload.get("extraction_completion_id") or any(
                record.payload.get(field) for field in ("pdf_path", "md_path", "txt_path")):
            exists = self.connection.execute("SELECT to_regclass(%s) AS name",
                (self.schema + ".extract_deletions",)).fetchone()
            if exists["name"]:
                from .extract_lifecycle import admit_record
                from .knowledge import KnowledgeRepository
                admit_record(KnowledgeRepository(self), record.payload)
        columns = sql.SQL(", ").join(map(sql.Identifier, _RECORD_COLUMNS))
        updates = sql.SQL(", ").join(sql.SQL("{} = EXCLUDED.{}").format(
            sql.Identifier(c), sql.Identifier(c),
        ) for c in _RECORD_COLUMNS if c != "key")
        values = [Jsonb(record.payload) if c == "payload" else getattr(record, c)
                  for c in _RECORD_COLUMNS]
        changed = self.connection.execute(sql.SQL(
            "INSERT INTO {} AS existing ({}, revision) VALUES ({}, 1) ON CONFLICT (key) DO UPDATE "
            "SET {}, revision = existing.revision + 1, updated_at = now() "
            "WHERE existing.revision = %s RETURNING key"
        ).format(self._table("records"), columns,
                 sql.SQL(", ").join(sql.Placeholder() for _ in values), updates),
            [*values, expected_revision]).fetchone()
        if not changed:
            actual = self.get(record.key)
            raise StaleWrite(record.key, expected_revision, actual.revision if actual else None)
        if changed:
            self._track_knowledge_requirement_change(record, expected_revision + 1)
            self._finish(record.key, audit)
        return bool(changed)

    def upsert(self, record: Record, *, audit: AuditEvent, expected_revision: int) -> bool:
        with self._write(audit):
            previous = self.get(record.key)
            changed = self._upsert(record, audit, expected_revision)
            if record.merge == "alias" or (previous and previous.merge == "alias"):
                before = self.all_aliases().get(record.key)
                self._sync_aliases([record])
                if not changed and before != self.all_aliases().get(record.key):
                    self._finish(record.key, audit)
            return changed

    def flush(self, records, *, expected_revisions: dict[str, int], audit: AuditEvent) -> dict[str, int]:
        """One atomic delta batch, with one state_flush event per changed key."""
        records = list(records)
        revisions = {}
        with self._write(audit):
            for record in records:
                event = replace(audit, detail={**audit.detail, "payload_sha256": record.payload_sha256},
                                pending_jobs=tuple(j for j in audit.pending_jobs if j.key == record.key))
                changed = self._upsert(record, event, expected_revisions[record.key])
                revisions[record.key] = expected_revisions[record.key] + int(changed)
            self._sync_aliases(records)
        return revisions

    def _sync_aliases(self, records):
        # Remove changed projections first so a valid final graph is not
        # rejected because the previous graph temporarily points the other way.
        for record in records:
            self.connection.execute(sql.SQL("DELETE FROM {} WHERE alias_key=%s").format(
                self._table("aliases")), (record.key,))
        for record in records:
            if record.merge == "alias":
                target = record.payload["terminal_state"]["alias_key"]
                if not target or target == record.key or self.resolve_alias(target) == record.key:
                    raise ValueError("invalid alias or alias cycle")
                self.connection.execute(sql.SQL(
                    "INSERT INTO {} (alias_key, canonical_key) VALUES (%s, %s)"
                ).format(self._table("aliases")), (record.key, target))

    def set_status(self, key: str, status: str, *, audit: AuditEvent, expected_revision: int) -> bool:
        with self._write(audit):
            record = self.get(key)
            if record is None:
                raise KeyError(key)
            if record.revision != expected_revision:
                raise StaleWrite(key, expected_revision, record.revision)
            if record.download_status == status:
                return False
            paper = Paper.model_validate(record.payload)
            # Capture legacy policy BEFORE changing retrieval. Otherwise a reset
            # to pending would derive clear policy from the new routing status.
            if paper.terminal_state is None:
                paper.terminal_state = derive_legacy(paper, include_domain=True)
            paper.download_status = status
            payload = copy.deepcopy(record.payload)
            payload["download_status"] = status
            state = paper.model_dump(mode="json")["terminal_state"]
            if isinstance(payload.get("terminal_state"), dict):
                # Only these axes changed; retain unknown nested payload fields.
                payload["terminal_state"].update({name: state[name] for name in ("retrieval", "cause")})
            else:
                payload["terminal_state"] = state
            return self._upsert(Record.from_payload(key, payload), audit, expected_revision)

    def check_completion(self, job_id: str, *, attempt: int, lease_id: str):
        row = self.connection.execute(sql.SQL("SELECT * FROM {} WHERE id = %s").format(
            self._table("jobs")), (job_id,)).fetchone()
        if (not row or row["attempts"] != attempt or row["lease_id"] != lease_id
                or row["state"] not in {"queued", "dispatched"}):
            raise StaleCompletion(f"stale completion for {job_id}")
        return row

    def complete_job(self, job_id: str, *, attempt: int, lease_id: str, outcome: str,
                     expected_attempt: int, expected_lease: str, audit: AuditEvent) -> bool:
        if outcome not in {"completed", "not_sent", "released", "failed"}:
            raise ValueError("completion requires a terminal outcome")
        with self._write(audit):
            if attempt != expected_attempt or lease_id != expected_lease:
                raise StaleCompletion(f"stale completion for {job_id}")
            row = self.connection.execute(sql.SQL(
                "UPDATE {} SET state = %s, finished_at = now() "
                "WHERE id = %s AND attempts = %s AND lease_id = %s "
                "AND kind <> 'distill' AND state IN ('queued', 'dispatched') RETURNING key"
            ).format(self._table("jobs")), (outcome, job_id, expected_attempt, expected_lease)).fetchone()
            if not row:
                raise StaleCompletion(f"stale completion for {job_id}")
            self._finish(row["key"], audit)
            return True

    def list_by_status(self, status: str) -> list[str]:
        return [r["key"] for r in self.connection.execute(sql.SQL(
            "SELECT key FROM {} WHERE download_status = %s ORDER BY key"
        ).format(self._table("records")), (status,))]

    def resolve_alias(self, key: str) -> str:
        seen = set()
        while key not in seen:
            seen.add(key)
            row = self.connection.execute(sql.SQL(
                "SELECT canonical_key FROM {} WHERE alias_key = %s"
            ).format(self._table("aliases")), (key,)).fetchone()
            if row is None:
                return key
            key = row["canonical_key"]
        raise ValueError("alias cycle")

    def all_aliases(self) -> dict[str, str]:
        return {row["alias_key"]: row["canonical_key"] for row in self.connection.execute(sql.SQL(
            "SELECT alias_key, canonical_key FROM {} ORDER BY alias_key"
        ).format(self._table("aliases")))}

    def identity_index_fields(self) -> set[str]:
        indexes = {row["indexname"] for row in self.connection.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = %s "
            "AND indexdef LIKE 'CREATE UNIQUE INDEX %%'", (self.schema,),
        )}
        return {field for field in ("doi", "arxiv_id") if "records_" + field + "_unique" in indexes}

    def add_alias(self, alias: str, canonical: str, *, audit: AuditEvent) -> bool:
        with self._write(audit):
            if not alias or not canonical or alias == canonical or self.resolve_alias(canonical) == alias:
                raise ValueError("invalid alias or alias cycle")
            changed = self.connection.execute(sql.SQL(
                "INSERT INTO {} AS existing (alias_key, canonical_key) VALUES (%s, %s) "
                "ON CONFLICT (alias_key) DO UPDATE SET canonical_key = EXCLUDED.canonical_key "
                "WHERE existing.canonical_key <> EXCLUDED.canonical_key RETURNING alias_key"
            ).format(self._table("aliases")), (alias, canonical)).fetchone()
            if changed:
                self._finish(alias, audit)
            return bool(changed)

    def _write_job(self, job: PendingJob) -> bool:
        from dataclasses import asdict

        values = asdict(job)
        old = self.connection.execute(sql.SQL("SELECT * FROM {} WHERE id = %s").format(
            self._table("jobs")), (job.id,)).fetchone()
        if job.kind == "distill" or (old and old["kind"] == "distill"):
            raise StaleCompletion("knowledge jobs require fenced KnowledgeRepository commands")
        if values["enqueued_at"] is None:
            values["enqueued_at"] = old["enqueued_at"] if old else datetime.now(timezone.utc)
        if old == values:
            return False
        if old and (job.attempts < old["attempts"] or (
                job.attempts == old["attempts"] and (
                    job.lease_id != old["lease_id"]
                    or (old["state"] not in {"pending", "queued", "dispatched"}
                        and job.state in {"queued", "dispatched"})))):
            raise StaleCompletion(f"job snapshot regresses active/settled attempt for {job.id}")
        columns = list(values)
        changed = self.connection.execute(sql.SQL(
            "INSERT INTO {} AS existing ({}) VALUES ({}) ON CONFLICT (id) DO UPDATE SET {} "
            "WHERE existing.attempts = %s AND existing.lease_id IS NOT DISTINCT FROM %s "
            "AND existing.state = %s RETURNING id"
        ).format(self._table("jobs"), sql.SQL(", ").join(map(sql.Identifier, columns)),
                 sql.SQL(", ").join(sql.Placeholder() for _ in columns),
                 sql.SQL(", ").join(sql.SQL("{} = EXCLUDED.{}").format(
                     sql.Identifier(c), sql.Identifier(c)) for c in columns if c != "id")),
            [*[Jsonb(value) if column == "detail" else value for column, value in values.items()],
             old["attempts"] if old else None, old["lease_id"] if old else None,
             old["state"] if old else None]).fetchone()
        if not changed:
            raise StaleCompletion(f"job changed during snapshot write for {job.id}")
        return True

    def import_job(self, job: PendingJob, *, audit: AuditEvent) -> bool:
        with self._write(audit):
            changed = self._write_job(job)
            if changed:
                self._finish(job.key, audit)
            return changed

    def import_ruling(self, ruling: dict, *, audit: AuditEvent) -> bool:
        with self._write(audit):
            changed = self.connection.execute(sql.SQL(
                "INSERT INTO {} (import_id, at, action, scope, reasons, keys, revision) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (import_id) DO NOTHING RETURNING id"
            ).format(self._table("rulings")), (
                payload_hash(ruling), ruling["timestamp"], ruling["action"],
                ruling.get("scope", ruling["action"]),
                Jsonb(ruling.get("reasons", [ruling.get("reason", "")])),
                ruling["keys"], ruling["revision"],
            )).fetchone()
            if changed:
                self._finish(None, audit)
            return bool(changed)

    def finalize_identity_indexes(self, *, audit: AuditEvent) -> dict:
        """Classify the ENTIRE target before creating each partial unique index.

        A conflicted field stays unindexed, with every key retained. Existing
        unique indexes are never removed to accommodate a later conflicting input.
        """
        result = {}
        with self._write(audit):
            for field_name in ("doi", "arxiv_id"):
                conflicts = list(self.connection.execute(sql.SQL(
                    "SELECT {field} AS identity, array_agg(key ORDER BY key) AS keys FROM {table} "
                    "WHERE {field} IS NOT NULL AND {field} <> '' GROUP BY {field} HAVING count(*) > 1"
                ).format(field=sql.Identifier(field_name), table=self._table("records"))))
                index_name = "records_" + field_name + "_unique"
                exists = self.connection.execute(
                    "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s",
                    (self.schema, index_name),
                ).fetchone()
                if not conflicts and not exists:
                    self.connection.execute(sql.SQL(
                        "CREATE UNIQUE INDEX {} ON {} ({}) WHERE {} IS NOT NULL AND {} <> ''"
                    ).format(sql.Identifier(index_name), self._table("records"),
                             *[sql.Identifier(field_name)] * 3))
                    self._finish(None, audit)
                result[field_name] = {"state": "deferred_conflicts" if conflicts else "created",
                                      "conflicts": conflicts}
        return result
