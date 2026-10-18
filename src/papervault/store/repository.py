"""Lossless records and the transactional shadow-catalog interface."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    enqueued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id or not self.kind or not self.key or not self.state or self.attempts < 0:
            raise ValueError("job requires id, kind, key, state and nonnegative attempts")
        for name in ("enqueued_at", "started_at", "finished_at"):
            value = getattr(self, name)
            if value is not None:
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
    def upsert(self, record: Record, *, audit: AuditEvent) -> bool: ...
    def set_status(self, key: str, status: str, *, audit: AuditEvent) -> bool: ...
    def list_by_status(self, status: str) -> list[str]: ...
    def resolve_alias(self, key: str) -> str: ...
    def add_alias(self, alias: str, canonical: str, *, audit: AuditEvent) -> bool: ...


_RECORD_COLUMNS = (
    "key", "doi", "arxiv_id", "download_status", "retrieval", "policy", "merge",
    "quarantine", "source", "year", "title", "payload", "payload_version",
)


class PostgresCatalogRepository:
    """Synchronous, explicit-connection repository; caller owns connection lifetime.

    Supply an autocommit connection so each public write owns its commit. A
    schema-local transaction lock serializes shadow writers, including alias
    cycle checks and imports. This deliberately favors integrity over throughput
    for P1; no live service imports or constructs this repository.
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
            yield

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
            ddl = re.sub(r"\bcatalog\b", lambda _: sql.Identifier(self.schema).as_string(), ddl)
            self.connection.execute(ddl)
            row = self.connection.execute(sql.SQL(
                "INSERT INTO {} (version, name) VALUES (1, 'P1 shadow catalog') "
                "ON CONFLICT (version) DO NOTHING RETURNING version"
            ).format(self._table("schema_version"))).fetchone()
            if row:
                self._finish(None, audit)

    def get(self, key: str) -> Record | None:
        row = self.connection.execute(sql.SQL("SELECT {} FROM {} WHERE key = %s").format(
            sql.SQL(", ").join(map(sql.Identifier, _RECORD_COLUMNS)), self._table("records"),
        ), (key,)).fetchone()
        return Record(**row) if row else None

    def all(self) -> Iterator[Record]:
        rows = self.connection.execute(sql.SQL("SELECT {} FROM {} ORDER BY key").format(
            sql.SQL(", ").join(map(sql.Identifier, _RECORD_COLUMNS)), self._table("records"),
        ))
        return (Record(**row) for row in rows)

    def _upsert(self, record: Record, audit: AuditEvent) -> bool:
        if record != Record.from_payload(record.key, record.payload):
            raise ValueError("record columns do not match payload version 1")
        columns = sql.SQL(", ").join(map(sql.Identifier, _RECORD_COLUMNS))
        updates = sql.SQL(", ").join(sql.SQL("{} = EXCLUDED.{}").format(
            sql.Identifier(c), sql.Identifier(c),
        ) for c in _RECORD_COLUMNS if c != "key")
        old = sql.SQL(", ").join(sql.SQL("existing.{}").format(sql.Identifier(c))
                                 for c in _RECORD_COLUMNS)
        new = sql.SQL(", ").join(sql.SQL("EXCLUDED.{}").format(sql.Identifier(c))
                                 for c in _RECORD_COLUMNS)
        values = [Jsonb(record.payload) if c == "payload" else getattr(record, c)
                  for c in _RECORD_COLUMNS]
        changed = self.connection.execute(sql.SQL(
            "INSERT INTO {} AS existing ({}) VALUES ({}) ON CONFLICT (key) DO UPDATE "
            "SET {}, updated_at = now() WHERE ({}) IS DISTINCT FROM ({}) RETURNING key"
        ).format(self._table("records"), columns,
                 sql.SQL(", ").join(sql.Placeholder() for _ in values), updates, old, new),
            values).fetchone()
        if changed:
            self._finish(record.key, audit)
        return bool(changed)

    def upsert(self, record: Record, *, audit: AuditEvent) -> bool:
        with self._write(audit):
            return self._upsert(record, audit)

    def set_status(self, key: str, status: str, *, audit: AuditEvent) -> bool:
        with self._write(audit):
            record = self.get(key)
            if record is None:
                raise KeyError(key)
            paper = Paper.model_validate(record.payload)
            paper.download_status = status
            payload = copy.deepcopy(record.payload)
            payload["download_status"] = status
            if "terminal_state" in payload:
                payload["terminal_state"] = paper.model_dump(mode="json")["terminal_state"]
            return self._upsert(Record.from_payload(key, payload), audit)

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
        if values["enqueued_at"] is None:
            values["enqueued_at"] = old["enqueued_at"] if old else datetime.now(timezone.utc)
        if old == values:
            return False
        columns = list(values)
        self.connection.execute(sql.SQL(
            "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT (id) DO UPDATE SET {}"
        ).format(self._table("jobs"), sql.SQL(", ").join(map(sql.Identifier, columns)),
                 sql.SQL(", ").join(sql.Placeholder() for _ in columns),
                 sql.SQL(", ").join(sql.SQL("{} = EXCLUDED.{}").format(
                     sql.Identifier(c), sql.Identifier(c)) for c in columns if c != "id")),
            [Jsonb(value) if column == "detail" else value for column, value in values.items()])
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
