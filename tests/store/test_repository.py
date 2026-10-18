"""Repository transaction boundaries without database credentials."""

from contextlib import contextmanager

import pytest


class Cursor:
    def __init__(self, row=None):
        self.row = row

    def fetchone(self):
        return self.row


class TransactionConnection:
    """Model commit/rollback only; SQL semantics are covered by opt-in PG tests."""

    autocommit = True

    def __init__(self, fail_table=None, changed=True):
        self.fail_table = fail_table
        self.changed = changed
        self.committed = []
        self.staged = []

    @contextmanager
    def transaction(self):
        try:
            yield
        except Exception:
            self.staged.clear()
            raise
        else:
            self.committed.extend(self.staged)
            self.staged.clear()

    def execute(self, query, params=None):
        query = query.as_string() if hasattr(query, "as_string") else query
        if query.lstrip().startswith("INSERT"):
            if f'"{self.fail_table}"' in query:
                raise RuntimeError("injected database failure")
            self.staged.append((query, params))
            return Cursor({"key": "A"} if self.changed else None)
        return Cursor()


def test_record_event_and_pending_job_commit_together():
    from papervault.store.repository import (
        AuditEvent, PendingJob, PostgresCatalogRepository, Record,
    )

    conn = TransactionConnection()
    repo = PostgresCatalogRepository(conn)
    repo.upsert(Record.from_payload("A", {"key": "A"}), audit=AuditEvent(
        actor="test", event="registered", pending_jobs=(PendingJob("job-A", "extract", "A"),),
    ))
    assert len(conn.committed) == 3
    assert '"records"' in conn.committed[0][0]
    assert '"audit"' in conn.committed[1][0]
    assert '"jobs"' in conn.committed[2][0]


@pytest.mark.parametrize("fail_table", ["audit", "jobs"])
def test_failed_audit_or_job_rolls_back_the_mutation(fail_table):
    from papervault.store.repository import (
        AuditEvent, PendingJob, PostgresCatalogRepository, Record,
    )

    conn = TransactionConnection(fail_table=fail_table)
    repo = PostgresCatalogRepository(conn)
    with pytest.raises(RuntimeError, match="injected database failure"):
        repo.upsert(Record.from_payload("A", {"key": "A"}), audit=AuditEvent(
            actor="test", event="registered", pending_jobs=(PendingJob("job-A", "extract", "A"),),
        ))
    assert conn.committed == []
    assert conn.staged == []


def test_unchanged_record_does_not_emit_events_or_jobs():
    from papervault.store.repository import AuditEvent, PostgresCatalogRepository, Record

    conn = TransactionConnection(changed=False)
    assert not PostgresCatalogRepository(conn).upsert(
        Record.from_payload("A", {"key": "A"}), audit=AuditEvent("test", "import"),
    )
    assert not any('"audit"' in query for query, _ in conn.committed)


def test_audit_is_required_and_read_only_repository_refuses_writes():
    from papervault.store.repository import AuditEvent, PostgresCatalogRepository, Record

    conn = TransactionConnection()
    repo = PostgresCatalogRepository(conn, read_only=True)
    record = Record.from_payload("A", {"key": "A"})
    with pytest.raises(TypeError):
        repo.upsert(record)
    with pytest.raises(PermissionError):
        repo.upsert(record, audit=AuditEvent("test", "import"))
    assert conn.committed == []


@pytest.mark.parametrize("schema", ["public", "pg_catalog", "lightrag", "bad.name"])
def test_repository_rejects_non_catalog_namespaces(schema):
    from papervault.store.repository import PostgresCatalogRepository

    with pytest.raises(ValueError):
        PostgresCatalogRepository(TransactionConnection(), schema=schema)
