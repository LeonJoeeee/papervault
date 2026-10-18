"""Opt-in real PostgreSQL tests for the shadow catalog.

Set PAPERVAULT_CATALOG_TEST_DSN explicitly to a dedicated test database, e.g.
papervault_catalog_test. No .env or production credential discovery is used.
Absent DSN or unavailable PostgreSQL skips cleanly. Every test owns a randomized
catalog_test_* schema; teardown drops only that schema, never LightRAG tables.
Run: .venv/bin/pytest tests/store/test_postgres.py -q
"""

import os
import uuid

import psycopg
from psycopg import sql
import pytest


@pytest.fixture
def repo():
    from papervault.store.repository import AuditEvent, PostgresCatalogRepository

    dsn = os.environ.get("PAPERVAULT_CATALOG_TEST_DSN")
    if not dsn:
        pytest.skip("PAPERVAULT_CATALOG_TEST_DSN not provided")
    try:
        conn = psycopg.connect(dsn, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("explicit catalog test database unavailable")
    schema = "catalog_test_" + uuid.uuid4().hex
    repository = PostgresCatalogRepository(conn, schema=schema)
    try:
        repository.ensure_schema(audit=AuditEvent("test", "schema"))
        yield repository
    finally:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        conn.close()


def count(repo, table):
    return repo.connection.execute(sql.SQL("SELECT count(*) AS n FROM {}.{}").format(
        sql.Identifier(repo.schema), sql.Identifier(table),
    )).fetchone()["n"]


def test_schema_and_record_replay_are_idempotent(repo):
    from papervault.store.repository import AuditEvent, Record

    event = AuditEvent("test", "import")
    before = count(repo, "audit")
    repo.ensure_schema(audit=event)
    assert count(repo, "audit") == before
    record = Record.from_payload("A", {"key": "A", "legacy": "retained"})
    assert repo.upsert(record, audit=event)
    assert not repo.upsert(record, audit=event)
    assert repo.get("A") == record
    assert [r.key for r in repo.all()] == ["A"]
    assert repo.list_by_status("pending") == ["A"]
    assert count(repo, "audit") == before + 1
    repo.set_status("A", "ok", audit=event)
    assert repo.get("A").retrieval == "acquired"
    assert repo.get("A").payload["legacy"] == "retained"
    assert not repo.set_status("A", "ok", audit=event)
    with pytest.raises(KeyError):
        repo.set_status("missing", "ok", audit=event)


def test_real_transaction_rolls_back_record_audit_and_pending_work(repo):
    from papervault.store.repository import AuditEvent, PendingJob, Record

    before = count(repo, "audit")
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        repo.upsert(Record.from_payload("A", {"key": "A"}), audit=AuditEvent(
            "test", "register", pending_jobs=(PendingJob("job", "extract", "missing"),),
        ))
    assert repo.get("A") is None
    assert count(repo, "audit") == before
    assert count(repo, "jobs") == 0


def test_alias_cycle_rejected_and_history_append_only(repo):
    from papervault.store.repository import AuditEvent, Record

    event = AuditEvent("test", "alias")
    for key in ("A", "B", "C"):
        repo.upsert(Record.from_payload(key, {"key": key}), audit=event)
    assert repo.add_alias("A", "B", audit=event)
    assert repo.add_alias("B", "C", audit=event)
    assert repo.resolve_alias("A") == "C"
    before = count(repo, "audit")
    assert not repo.add_alias("A", "B", audit=event)
    with pytest.raises(ValueError, match="cycle"):
        repo.add_alias("C", "A", audit=event)
    assert count(repo, "audit") == before
    for table in ("audit", "rulings"):
        with pytest.raises(psycopg.errors.RaiseException):
            repo.connection.execute(sql.SQL("TRUNCATE {}.{} CASCADE").format(
                sql.Identifier(repo.schema), sql.Identifier(table),
            ))


def test_real_import_conflicts_remain_separate_and_replay_has_no_new_events(repo, tmp_path):
    import json
    from papervault.store.adapter import LibraryAdapter
    from papervault.store.importer import import_catalog

    path = tmp_path / "index.json"
    path.write_text(json.dumps({"papers": {
        "A": {"key": "A", "doi": "10.1/SAME"},
        "B": {"key": "B", "doi": "10.1/same"},
    }}))
    adapter = LibraryAdapter.from_snapshot(path)
    first = import_catalog(adapter, repo)
    assert first["difference_rate"] == 0
    assert first["target_identity_indexes"]["doi"]["state"] == "deferred_conflicts"
    assert first["target_identity_indexes"]["doi"]["conflicts"] == [
        {"identity": "10.1/same", "keys": ["A", "B"]},
    ]
    assert count(repo, "records") == 2
    before = count(repo, "audit")
    assert import_catalog(adapter, repo)["changed"] == 0
    assert count(repo, "audit") == before


def test_unique_indexes_enforce_resolved_identities_and_allow_empty_values(repo):
    from papervault.store.repository import AuditEvent, Record

    event = AuditEvent("test", "identity")
    repo.upsert(Record.from_payload("A", {"key": "A", "doi": "10.1/a"}), audit=event)
    assert repo.finalize_identity_indexes(audit=event)["doi"]["state"] == "created"
    before = count(repo, "audit")
    with pytest.raises(psycopg.errors.UniqueViolation):
        repo.upsert(Record.from_payload("B", {"key": "B", "doi": " 10.1/A "}), audit=event)
    assert repo.get("B") is None
    assert count(repo, "audit") == before
    for key in ("C", "D"):
        repo.upsert(Record.from_payload(key, {"key": key, "doi": " "}), audit=event)
    assert count(repo, "records") == 3


def test_audit_insert_failure_rolls_back_record(repo):
    from papervault.store.repository import AuditEvent, Record

    repo.connection.execute(sql.SQL(
        "ALTER TABLE {} ADD CONSTRAINT test_actor CHECK (actor <> 'reject')"
    ).format(sql.Identifier(repo.schema, "audit")))
    before = count(repo, "audit")
    with pytest.raises(psycopg.errors.CheckViolation):
        repo.upsert(Record.from_payload("A", {"key": "A"}), audit=AuditEvent("reject", "import"))
    assert repo.get("A") is None
    assert count(repo, "audit") == before


def test_rulings_jobs_and_pending_work_are_audited_and_idempotent(repo):
    from papervault.store.repository import AuditEvent, PendingJob, Record

    job = PendingJob("job-A", "extract", "A")
    event = AuditEvent("test", "register", pending_jobs=(job,))
    repo.upsert(Record.from_payload("A", {"key": "A"}), audit=event)
    assert count(repo, "jobs") == 1
    before = count(repo, "audit")
    assert not repo.import_job(job, audit=AuditEvent("test", "job"))
    assert count(repo, "audit") == before
    changed = PendingJob("job-A", "extract", "A", state="done", attempts=1,
                         finished_at="2026-10-18T00:00:00Z")
    assert repo.import_job(changed, audit=AuditEvent("test", "job"))
    assert count(repo, "audit") == before + 1
    ruling = {"timestamp": "2026-10-18T00:00:00Z", "action": "acquire", "reason": "approved",
              "keys": ["A"], "revision": 1}
    assert repo.import_ruling(ruling, audit=AuditEvent("test", "ruling"))
    assert not repo.import_ruling(ruling, audit=AuditEvent("test", "ruling"))
    assert count(repo, "rulings") == 1
    assert count(repo, "audit") == before + 2
    for statement in ("UPDATE {} SET revision = 2", "DELETE FROM {}"):
        with pytest.raises(psycopg.errors.RaiseException):
            repo.connection.execute(sql.SQL(statement).format(sql.Identifier(repo.schema, "rulings")))
