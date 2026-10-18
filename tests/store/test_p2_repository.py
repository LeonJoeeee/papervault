"""P2 contracts exercised against an explicitly selected disposable PostgreSQL."""

import pytest
from psycopg import sql

from test_postgres import count
from papervault.store import repository as catalog


def record(key, **fields):
    return catalog.Record.from_payload(key, {"key": key, **fields})


def test_stale_revision_rejected_without_record_audit_or_pending_work(repo):
    event = catalog.AuditEvent("test", "write")
    assert repo.upsert(record("A"), expected_revision=0, audit=event)
    assert repo.get("A").revision == 1
    before = count(repo, "audit")
    with pytest.raises(catalog.StaleWrite) as failure:
        repo.upsert(record("A", title="stale"), expected_revision=0, audit=event)
    assert (failure.value.key, failure.value.expected, failure.value.actual) == ("A", 0, 1)
    assert repo.get("A").payload == {"key": "A"}
    assert count(repo, "audit") == before
    assert not repo.upsert(record("A"), expected_revision=1, audit=event)
    assert repo.get("A").revision == 1


def test_delta_batch_rolls_back_every_key_on_one_conflict(repo):
    event = catalog.AuditEvent("test", "state_flush")
    repo.upsert(record("A"), expected_revision=0, audit=event)
    before = count(repo, "audit")
    with pytest.raises(catalog.StaleWrite):
        repo.flush([record("B"), record("A", title="bad")],
                   expected_revisions={"A": 0, "B": 0}, audit=event)
    assert repo.get("B") is None
    assert count(repo, "audit") == before
    revisions = repo.flush([record("B"), record("A", title="good")],
                           expected_revisions={"A": 1, "B": 0}, audit=event)
    assert revisions == {"A": 2, "B": 1}
    events = list(repo.connection.execute(sql.SQL(
        "SELECT key, detail FROM {} WHERE event = 'state_flush' ORDER BY key"
    ).format(sql.Identifier(repo.schema, "audit"))))
    assert [row["key"] for row in events] == ["A", "A", "B"]
    assert all(len(row["detail"]["payload_sha256"]) == 64 for row in events[1:])


def test_job_completion_requires_current_attempt_lease_and_active_state(repo):
    event = catalog.AuditEvent("test", "queue")
    job = catalog.PendingJob("reopen:A", "reopen", "A", state="dispatched",
                             attempts=2, lease_id="current")
    repo.upsert(record("A"), expected_revision=0,
                audit=catalog.AuditEvent("test", "queue", pending_jobs=(job,)))
    before = count(repo, "audit")
    for attempt, lease in [(1, "current"), (2, "stale")]:
        with pytest.raises(catalog.StaleCompletion):
            repo.complete_job("reopen:A", attempt=attempt, lease_id=lease, outcome="completed",
                              expected_attempt=attempt, expected_lease=lease, audit=event)
    assert count(repo, "audit") == before
    assert repo.complete_job("reopen:A", attempt=2, lease_id="current", outcome="completed",
                             expected_attempt=2, expected_lease="current", audit=event)
    with pytest.raises(catalog.StaleCompletion):
        repo.complete_job("reopen:A", attempt=2, lease_id="current", outcome="completed",
                          expected_attempt=2, expected_lease="current", audit=event)
    assert count(repo, "audit") == before + 1


def test_durable_database_fence_refuses_direct_repository_writers(repo):
    from papervault.store.fencing import WriterFenced

    repo.set_fence("handoff", audit=catalog.AuditEvent("test", "fence"))
    repo.fence_id = None  # a separate, unprivileged service/CLI connection
    with pytest.raises(WriterFenced):
        repo.upsert(record("A"), expected_revision=0, audit=catalog.AuditEvent("test", "blocked"))
    repo.fence_id = "handoff"
    repo.upsert(record("A"), expected_revision=0, audit=catalog.AuditEvent("test", "probe"))
    repo.set_fence(None, audit=catalog.AuditEvent("test", "unfence"))


def test_guarded_single_upsert_updates_alias_projection(repo):
    event = catalog.AuditEvent("test", "merge")
    repo.upsert(record("Z"), expected_revision=0, audit=event)
    repo.upsert(record("A", terminal_state={"merge": "alias", "alias_key": "Z"}),
                expected_revision=0, audit=event)
    assert repo.all_aliases() == {"A": "Z"}
    repo.upsert(record("A"), expected_revision=1, audit=event)
    assert repo.all_aliases() == {}


def test_import_job_cannot_regress_a_newer_leased_attempt(repo):
    event = catalog.AuditEvent("test", "queue")
    repo.upsert(record("A"), expected_revision=0, audit=event)
    repo.import_job(catalog.PendingJob("reopen:A", "reopen", "A", state="dispatched", attempts=2,
                                      lease_id="new"), audit=event)
    before = count(repo, "audit")
    with pytest.raises(catalog.StaleCompletion):
        repo.import_job(catalog.PendingJob("reopen:A", "reopen", "A", state="completed", attempts=1,
                                          lease_id="old"), audit=event)
    assert count(repo, "audit") == before
