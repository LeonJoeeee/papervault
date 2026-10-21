"""The writer handoff requires a complete capture and durable source/backend fences."""
import copy
import os

from psycopg import sql
import pytest


def checkpoint():
    return {"source": "ledger-test", "capture_id": "capture1", "snapshot_sequence": 10,
            "last_sequence": 12, "handoff_eligible": True, "final_row_count": 0,
            "final_rows_digest": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"}


def evidence():
    return {"source": "ledger-test", "capture_id": "capture1", "final_sequence": 12,
        "source_fence": {"capture_id": "capture1", "final_sequence": 12, "fenced": True,
                         "legacy_dml_revoked": True, "revoked_roles": ["legacy"],
                         "permission_checks": [{"role": "legacy", "can_write": False}], "fence_id": "f1",
                         "final_row_count": 0, "final_rows_digest": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"},
        "drain": {"dispatch_stopped": True, "pipeline_admission_stopped": True, "inflight_drained": True,
                  "backend_writes_fenced": True, "internal_workers_stopped": True,
                  "receipt": {"mechanism": "isolated-namespace-fence", "id": "backend-f1"}},
        "validation": {"dispositions": True, "references": True, "quarantine": True,
                       "accounting": True, "external_attempts": True, "receipt": "validation1"}}


def test_cutover_rejects_flag_only_gap_unrevoked_roles_and_unfenced_backend():
    from papervault.store.knowledge_cutover import validate_handoff_evidence

    validate_handoff_evidence(checkpoint(), evidence())
    for mutate in (
        lambda e: e["drain"].update(backend_writes_fenced=False),
        lambda e: e["source_fence"].update(legacy_dml_revoked=False),
        lambda e: e["source_fence"].update(permission_checks=[{"role": "legacy", "can_write": True}]),
        lambda e: e.update(final_sequence=13),
        lambda e: e["source_fence"].update(final_rows_digest="wrong"),
        lambda e: e["validation"].update(accounting=False),
    ):
        bad = evidence()
        mutate(bad)
        with pytest.raises(ValueError):
            validate_handoff_evidence(checkpoint(), bad)
    with pytest.raises(ValueError, match="capture"):
        validate_handoff_evidence({**checkpoint(), "handoff_eligible": False}, evidence())


def test_capture_tool_refuses_live_database_before_any_connection():
    from papervault.store.knowledge_cutover import validate_rehearsal_dsn

    for name in ("papervault_catalog", "knowledge_system"):
        with pytest.raises(ValueError, match="live"):
            validate_rehearsal_dsn("postgresql://localhost/" + name)
    with pytest.raises(ValueError):
        validate_rehearsal_dsn("host=localhost")
    validate_rehearsal_dsn("dbname=papervault_p4_test")


def source_capture(repo):
    from papervault.store.knowledge_cutover import SourceCapture

    repo.connection.execute(sql.SQL("CREATE TABLE {} (workspace TEXT,ingest_source TEXT,source_id TEXT,"
        "status TEXT,attempts INTEGER,doc_id TEXT, PRIMARY KEY(workspace,ingest_source,source_id))").format(
        repo._table("ks_ledger")))
    capture = SourceCapture(repo.connection, ledger_schema=repo.schema, source="ledger-test", configuration={"workspace": "probe"})
    capture.install()
    return capture


def test_durable_capture_has_transactional_sequences_and_aligned_snapshot_final_delta(repo):
    capture = source_capture(repo)
    insert = sql.SQL("INSERT INTO {} VALUES ('probe','paper',%s,'done',0,%s)").format(repo._table("ks_ledger"))
    repo.connection.execute(insert, ("A", "doc:A"))
    snap = capture.snapshot()
    assert snap["checkpoint"] == 1
    assert snap["rows"][0]["source_id"] == "A"
    with pytest.raises(RuntimeError):
        with repo.connection.transaction():
            repo.connection.execute(insert, ("aborted", "aborted"))
            raise RuntimeError("crash before commit")
    repo.connection.execute(sql.SQL("UPDATE {} SET status='error',attempts=1 WHERE source_id='A'").format(repo._table("ks_ledger")))
    repo.connection.execute(sql.SQL("DELETE FROM {} WHERE source_id='A'").format(repo._table("ks_ledger")))
    delta = capture.delta(snap["checkpoint"])
    assert [change["sequence"] for change in delta["changes"]] == [2, 3]
    assert [change["operation"] for change in delta["changes"]] == ["UPDATE", "DELETE"]
    assert delta["final_checkpoint"] == 3
    with pytest.raises(Exception, match="TRUNCATE"):
        repo.connection.execute(sql.SQL("TRUNCATE {}").format(repo._table("ks_ledger")))


@pytest.mark.parametrize("operation,final_count", [("INSERT", 2), ("UPDATE", 1), ("DELETE", 0)])
@pytest.mark.parametrize("timestamp,canonical_time", [
    ("2026-10-19T17:30:00.123000+05:30", "2026-10-19T12:00:00.123Z"),
    ("2026-10-19T08:00:00.123456-04:00", "2026-10-19T12:00:00.123456Z"),
    ("2026-10-20T01:30:00+05:30", "2026-10-19T20:00:00Z"),
])
def test_timestamped_capture_delta_freeze_and_import_agree_across_timezones(
        repo, operation, final_count, timestamp, canonical_time):
    from papervault.store.knowledge import KnowledgeRepository
    from papervault.store.knowledge_cutover import CutoverRepository
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
    from papervault.store.repository import AuditEvent

    capture = source_capture(repo)
    table = repo._table("ks_ledger")
    repo.connection.execute(sql.SQL("ALTER TABLE {} ADD COLUMN last_processed_at timestamptz").format(table))
    repo.connection.execute("SET TIME ZONE 'Asia/Kolkata'")
    insert = sql.SQL("INSERT INTO {} VALUES ('probe','paper',%s,'done',0,%s,%s::timestamptz)").format(table)
    repo.connection.execute(insert, ("A", "doc:A", timestamp))
    snap = capture.snapshot()
    readonly = capture.readonly_snapshot()
    repo.connection.execute("SET TIME ZONE 'America/New_York'")
    if operation == "INSERT":
        repo.connection.execute(insert, ("B", "doc:B", timestamp))
    elif operation == "UPDATE":
        repo.connection.execute(sql.SQL("UPDATE {} SET status='error',attempts=1 WHERE source_id='A'").format(table))
    else:
        repo.connection.execute(sql.SQL("DELETE FROM {} WHERE source_id='A'").format(table))
    delta = capture.delta(snap["checkpoint"])
    assert [change["operation"] for change in delta["changes"]] == [operation]
    plan = plan_legacy_import(snap, changes=delta["changes"], final_checkpoint=delta["final_checkpoint"])
    assert snap["rows"][0]["last_processed_at"] == canonical_time
    assert readonly["rows"] == snap["rows"]
    assert all(event["raw"]["last_processed_at"] == canonical_time for event in plan["events"])
    assert all(row["last_processed_at"] == canonical_time
               for change in delta["changes"] for row in (change["old"], change["new"]) if row)

    repo.connection.execute("SET TIME ZONE 'UTC'")
    frozen = capture.freeze(legacy_roles=["pg_monitor"], drain=evidence()["drain"])
    assert plan["checkpoint"]["final_row_count"] == frozen["final_row_count"] == final_count
    assert plan["checkpoint"]["final_rows_digest"] == frozen["final_rows_digest"]
    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    imports = LegacyImportRepository(repo)
    imports.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    assert imports.apply(plan, operation_id="timestamp-import")["inserted"] == 2
    checkpoint = imports.knowledge._rows("knowledge_import_checkpoints")[0]["checkpoint"]
    assert checkpoint["final_row_count"] == frozen["final_row_count"]
    assert checkpoint["final_rows_digest"] == frozen["final_rows_digest"]
    assert all(event["evidence"]["raw"]["last_processed_at"] == canonical_time
               for event in imports.source_versions("ledger-test"))
    KnowledgeRepository(repo).command("create_generation", "timestamp-active", {
        "generation_id": "active1", "candidate": False})
    handoff = {**evidence(), "capture_id": snap["capture_id"],
               "final_sequence": delta["final_checkpoint"], "source_fence": frozen}
    assert CutoverRepository(repo).commit_handoff(handoff, expected_epoch=1,
                                                operation_id="timestamp-handoff")["authority"] == "catalog"


def pre_normalization_c0_plan(snapshot):
    """Reproduce the old planner's literal Python timestamp encoding and hashes."""
    from papervault.store.knowledge_import import plan_legacy_import, source_row_id
    from papervault.store.repository import payload_hash

    plan = plan_legacy_import(snapshot)
    original = {source_row_id(row): row for row in snapshot["rows"]}
    for event in plan["events"]:
        event["raw"] = copy.deepcopy(original[event["source_row_id"]])
        event["event_digest"] = payload_hash({key: event[key] for key in (
            "source", "capture_id", "source_row_id", "source_version", "operation", "raw", "deleted")})
    plan["checkpoint"]["final_rows_digest"] = payload_hash([
        original[key] for key in sorted(original)])
    return plan


@pytest.mark.parametrize("resume", ["replay", "enrichment", "INSERT", "UPDATE", "DELETE"])
def test_pre_normalization_c0_resumes_without_rewriting_history_and_handoff_keeps_unchanged_row(repo, resume):
    from papervault.store.knowledge import KnowledgeRepository, OperationConflict
    from papervault.store.knowledge_cutover import CutoverRepository
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
    from papervault.store.repository import AuditEvent, payload_hash

    capture = source_capture(repo)
    table = repo._table("ks_ledger")
    repo.connection.execute(sql.SQL("ALTER TABLE {} ADD COLUMN last_processed_at timestamptz").format(table))
    repo.connection.execute("SET TIME ZONE 'UTC'")
    insert = sql.SQL("INSERT INTO {} VALUES ('probe','paper',%s,'done',0,%s,%s::timestamptz)").format(table)
    for key in ("A", "B"):
        repo.connection.execute(insert, (key, "doc:" + key, "2026-10-19T12:00:00.123000+00:00"))
    snap = capture.snapshot()
    old_snapshot = copy.deepcopy(snap)
    for row in old_snapshot["rows"]:
        row["last_processed_at"] = "2026-10-19T12:00:00.123000+00:00"
    accepted = pre_normalization_c0_plan(old_snapshot)
    assert accepted["checkpoint"]["final_rows_digest"] != plan_legacy_import(snap)["checkpoint"]["final_rows_digest"]
    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    imports = LegacyImportRepository(repo)
    imports.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    imports.apply(accepted, operation_id="old-C0")
    k = imports.knowledge
    original_events = imports.source_versions("ledger-test")
    original_report = k._one("knowledge_import_reports", plan_digest=payload_hash(accepted))
    original_head = next(head for head in k._rows("knowledge_import_heads")
                         if head["source_row_id"] == accepted["events"][0]["source_row_id"])
    original_job = k._one("jobs", id=original_head["job_id"])
    assert original_job["detail"]["evidence"]["raw"]["last_processed_at"] == "2026-10-19T12:00:00.123000+00:00"

    repo.connection.execute("SET TIME ZONE 'America/New_York'")
    if resume == "INSERT":
        repo.connection.execute(insert, ("C", "doc:C", "2026-10-19T08:00:00.123456-04:00"))
    elif resume == "UPDATE":
        repo.connection.execute(sql.SQL("UPDATE {} SET status='error',attempts=1,"
            "last_processed_at='2026-10-19T08:00:00.123456-04:00' WHERE source_id='B'").format(table))
    elif resume == "DELETE":
        repo.connection.execute(sql.SQL("DELETE FROM {} WHERE source_id='B'").format(table))
    delta = capture.delta(snap["checkpoint"])
    statuses = [{"workspace": "probe", "doc_id": "doc:A", "status": "processed"}] if resume == "enrichment" else []
    proposed = plan_legacy_import(old_snapshot, changes=delta["changes"],
        final_checkpoint=delta["final_checkpoint"], doc_status_rows=statuses)
    before = copy.deepcopy(proposed)
    # Source capture changes its timezone; catalog projection hashes stay exact.
    repo.connection.execute("SET TIME ZONE 'UTC'")
    result = imports.apply(proposed, operation_id="resume")
    assert proposed == before
    assert result["inserted"] == (1 if resume in {"INSERT", "UPDATE", "DELETE"} else 0)
    assert imports.apply(proposed, operation_id="resume") == result
    assert [event for event in imports.source_versions("ledger-test")
            if event["source_version"] == snap["checkpoint"]] == original_events
    assert k._one("knowledge_import_reports", plan_digest=payload_hash(accepted)) == original_report
    assert k._one("knowledge_import_reports", plan_digest=payload_hash(proposed))["report"] == proposed
    unchanged = k._one("jobs", id=original_head["job_id"])
    assert unchanged["detail"]["evidence"]["raw"]["last_processed_at"] == "2026-10-19T12:00:00.123000+00:00"
    if resume != "enrichment":
        assert k._one("knowledge_import_heads", source_row_id=original_head["source_row_id"]) == original_head
        assert unchanged == original_job
    else:
        assert unchanged["detail"]["evidence"]["dispositions"] == ["matched", "unmatched_unit", "provenance_gap"]
    if resume in {"INSERT", "UPDATE", "DELETE"}:
        checkpoint_before = k._rows("knowledge_import_checkpoints")
        older = plan_legacy_import(old_snapshot, doc_status_rows=[{
            "workspace": "probe", "doc_id": "doc:A", "status": "processed"}])
        with pytest.raises(OperationConflict, match="older source checkpoint"):
            imports.apply(older, operation_id="older-enrichment")
        assert k._rows("knowledge_import_checkpoints") == checkpoint_before
        assert k._one("jobs", id=original_head["job_id"]) == unchanged

    repo.connection.execute("SET TIME ZONE 'Asia/Kolkata'")
    frozen = capture.freeze(legacy_roles=["pg_monitor"], drain=evidence()["drain"])
    checkpoint = k._rows("knowledge_import_checkpoints")[0]["checkpoint"]
    assert checkpoint["final_row_count"] == frozen["final_row_count"]
    assert checkpoint["final_rows_digest"] == frozen["final_rows_digest"]
    repo.connection.execute("SET TIME ZONE 'UTC'")
    KnowledgeRepository(repo).command("create_generation", "resumed-active", {
        "generation_id": "active1", "candidate": False})
    handoff = {**evidence(), "capture_id": snap["capture_id"],
               "final_sequence": delta["final_checkpoint"], "source_fence": frozen}
    assert CutoverRepository(repo).commit_handoff(handoff, expected_epoch=1,
                                                operation_id="resumed-handoff")["authority"] == "catalog"


@pytest.mark.parametrize("tampering", ["instant", "counter", "text", "invalid_digest", "counter_evidence"])
def test_old_timestamp_compatibility_rejects_conflicting_or_unhashed_source_evidence(repo, tampering):
    from papervault.store.knowledge import OperationConflict
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
    from papervault.store.repository import AuditEvent

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    imports = LegacyImportRepository(repo)
    imports.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    raw = {"source": "ledger-test", "capture_id": "capture1", "checkpoint": 1, "rows": [{
        "workspace": "probe", "ingest_source": "paper", "source_id": "A", "status": "done", "attempts": 0,
        "fingerprint": "2026-10-19T12:00:00.123000+00:00", "last_processed_at": "2026-10-19T12:00:00.123000+00:00"}]}
    accepted = pre_normalization_c0_plan(raw)
    imports.apply(accepted, operation_id="old-C0")
    changed = copy.deepcopy(raw)
    if tampering == "instant":
        changed["rows"][0]["last_processed_at"] = "2026-10-19T12:00:00.123001Z"
    elif tampering == "counter":
        changed["rows"][0]["attempts"] = 1
    elif tampering == "text":
        changed["rows"][0]["fingerprint"] = "2026-10-19T12:00:00.123Z"
    proposed = plan_legacy_import(changed)
    if tampering == "invalid_digest":
        proposed["events"][0]["event_digest"] = accepted["events"][0]["event_digest"]
    elif tampering == "counter_evidence":
        proposed["events"][0]["legacy_consecutive_failures"] = 1
    before_jobs = imports.knowledge._rows("jobs")
    before_reports = imports.knowledge._rows("knowledge_import_reports")
    with pytest.raises(OperationConflict, match="evidence|regression"):
        imports.apply(proposed, operation_id="tampering")
    assert imports.knowledge._rows("jobs") == before_jobs
    assert imports.knowledge._rows("knowledge_import_reports") == before_reports


@pytest.mark.parametrize("corruption", ["literal", "digest_column"])
def test_timestamp_compatibility_requires_valid_literal_hash_for_stored_events(repo, corruption):
    from papervault.store.knowledge import OperationConflict
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
    from papervault.store.repository import AuditEvent

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    imports = LegacyImportRepository(repo)
    imports.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    raw = {"source": "ledger-test", "capture_id": "capture1", "checkpoint": 1, "rows": [{
        "workspace": "probe", "ingest_source": "paper", "source_id": "A", "status": "done", "attempts": 0,
        "last_processed_at": "2026-10-19T12:00:00.123000+00:00"}]}
    old_event = pre_normalization_c0_plan(raw)["events"][0]
    stored = {field: old_event[field] for field in (
        "source", "capture_id", "source_row_id", "source_version", "event_digest")}
    stored["evidence"] = copy.deepcopy(old_event)
    if corruption == "literal":
        stored["evidence"]["raw"]["last_processed_at"] = "2026-10-19T12:00:00.123Z"
    else:
        stored["event_digest"] = "incorrect"
    audit = AuditEvent("test", "corrupt-source-fixture")
    with repo._write(audit):
        imports.knowledge._insert("knowledge_import_events", stored)
        repo._finish(None, audit)
    before = imports.source_versions("ledger-test")
    with pytest.raises(OperationConflict, match="evidence"):
        imports.apply(plan_legacy_import(raw), operation_id="invalid-history")
    assert imports.source_versions("ledger-test") == before
    assert imports.knowledge._rows("knowledge_import_reports") == []
    assert imports.knowledge._rows("jobs") == []


def test_timestamp_compatibility_preserves_exact_projection_hash_guard(repo):
    from papervault.store.knowledge import KnowledgeRepository, OperationConflict
    from papervault.store.knowledge_cutover import CutoverRepository
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
    from papervault.store.repository import AuditEvent

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    imports = LegacyImportRepository(repo)
    imports.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    raw = {"source": "ledger-test", "capture_id": "capture1", "checkpoint": 10, "rows": [{
        "workspace": "probe", "ingest_source": "paper", "source_id": "A", "status": "done", "attempts": 0,
        "last_processed_at": "2026-10-19T12:00:00.123000+00:00"}]}
    imports.apply(pre_normalization_c0_plan(raw), operation_id="old-C0")
    k = imports.knowledge
    head = k._rows("knowledge_import_heads")[0]
    job = k._one("jobs", id=head["job_id"])
    detail = copy.deepcopy(job["detail"])
    detail["evidence"]["raw"]["last_processed_at"] = "2026-10-19T12:00:00.123Z"
    audit = AuditEvent("test", "independent-projection-edit")
    with repo._write(audit):
        k._update("jobs", {"detail": detail}, id=job["id"])
        repo._finish(None, audit)
    changed_job = k._one("jobs", id=job["id"])
    proposed = plan_legacy_import(raw)
    with pytest.raises(OperationConflict, match="independently"):
        imports.apply(proposed, operation_id="overwrite")
    KnowledgeRepository(repo).command("create_generation", "guard-active", {
        "generation_id": "active1", "candidate": False})
    handoff = {**evidence(), "final_sequence": 10, "source_fence": {
        **evidence()["source_fence"], "final_sequence": 10,
        "final_row_count": 1, "final_rows_digest": proposed["checkpoint"]["final_rows_digest"]}}
    with pytest.raises(OperationConflict, match="independently"):
        CutoverRepository(repo).commit_handoff(handoff, expected_epoch=1, operation_id="tampered-handoff")
    assert k._one("jobs", id=job["id"]) == changed_job
    assert k._rows("knowledge_import_heads")[0] == head
    assert k.get_authority_state()["authority"] == "legacy"


def test_handoff_commits_epoch_once_and_rollback_cannot_restore_legacy(repo):
    from papervault.store.knowledge import KnowledgeRepository
    from papervault.store.knowledge_cutover import CutoverRepository
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
    from papervault.store.repository import AuditEvent

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    imports = LegacyImportRepository(repo)
    imports.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    snapshot = {"source": "ledger-test", "capture_id": "capture1", "checkpoint": 10,
                "rows": [], "external_attempts": {"reviewed": True}}
    changes = [{"sequence": 11, "operation": "INSERT", "old": None, "new": {
        "workspace": "probe", "ingest_source": "paper", "source_id": "A", "doc_id": "doc:A", "status": "done", "attempts": 0}},
        {"sequence": 12, "operation": "DELETE", "old": {"workspace": "probe", "ingest_source": "paper",
         "source_id": "A", "doc_id": "doc:A", "status": "done", "attempts": 0}, "new": None}]
    plan = plan_legacy_import(snapshot, changes=changes, final_checkpoint=12)
    imports.apply(plan, operation_id="import")
    cutover = CutoverRepository(repo)
    with pytest.raises(ValueError, match="active generation"):
        cutover.commit_handoff(evidence(), expected_epoch=1, operation_id="handoff")
    KnowledgeRepository(repo).command("create_generation", "initialize-active", {"generation_id": "active1", "candidate": False})
    result = cutover.commit_handoff(evidence(), expected_epoch=1, operation_id="handoff")
    assert result["authority"] == "catalog"
    assert result["authority_epoch"] == 2
    assert cutover.commit_handoff(copy.deepcopy(evidence()), expected_epoch=1, operation_id="handoff") == result
    assert KnowledgeRepository(repo).get_authority_state()["authority"] == "catalog"
    with pytest.raises(ValueError, match="retain catalog"):
        cutover.rollback(authority="legacy")
    assert cutover.rollback(authority="catalog")["authority_epoch"] == 2
    with pytest.raises(ValueError, match="fenced"):
        imports.apply(plan, operation_id="late-import")


def test_source_freeze_revokes_dml_and_triggers_reject_even_owner_writes(repo):
    import psycopg

    capture = source_capture(repo)
    repo.connection.execute(sql.SQL("GRANT INSERT,UPDATE,DELETE,TRUNCATE ON {} TO pg_monitor").format(repo._table("ks_ledger")))
    frozen = capture.freeze(legacy_roles=["pg_monitor"], drain=evidence()["drain"])
    assert frozen["fenced"] is True
    assert frozen["permission_checks"] == [{"role": "pg_monitor", "can_write": False}]
    with pytest.raises(psycopg.errors.RaiseException, match="fenced"):
        repo.connection.execute(sql.SQL("INSERT INTO {} VALUES ('probe','paper','A','done',0,'A')").format(repo._table("ks_ledger")))
    assert capture.freeze(legacy_roles=["pg_monitor"], drain=evidence()["drain"])["fence_id"] == frozen["fence_id"]


def test_capture_statement_lock_blocks_other_writer_before_row_changes_and_reserves_no_sequence(repo):
    import psycopg

    capture = source_capture(repo)
    insert = sql.SQL("INSERT INTO {} VALUES ('probe','paper',%s,'done',0,%s)").format(repo._table("ks_ledger"))
    with psycopg.connect(os.environ["PAPERVAULT_CATALOG_TEST_DSN"], autocommit=True) as other:
        other.execute("SET lock_timeout='50ms'")
        with repo.connection.transaction():
            repo.connection.execute(insert, ("A", "doc:A"))
            with pytest.raises(psycopg.errors.LockNotAvailable):
                other.execute(insert, ("B", "doc:B"))
        other.execute(insert, ("B", "doc:B"))
    delta = capture.delta(0)
    assert [change["sequence"] for change in delta["changes"]] == [1, 2]
    assert [change["new"]["source_id"] for change in delta["changes"]] == ["A", "B"]


def test_final_projection_validation_rejects_changed_raw_evidence_failure_counter_or_catalog_identity():
    from papervault.store.knowledge_cutover import validate_legacy_projection
    from papervault.store.knowledge_import import plan_legacy_import

    evidence = plan_legacy_import({"source": "ledger-test", "capture_id": "capture1", "checkpoint": 1,
        "rows": [{"workspace": "probe", "ingest_source": "paper", "source_id": "A",
                  "attempts": 3, "status": "error_parked"}]})["events"][0]
    event = {**{field: evidence[field] for field in (
        "source", "capture_id", "source_row_id", "source_version", "event_digest")}, "evidence": evidence}
    head = {"source_row_id": evidence["source_row_id"], "source_version": 1,
            "event_digest": evidence["event_digest"], "deleted": False}
    job = {"kind": "distill_legacy_unverified", "state": "quarantined", "quarantine": True,
        "attempts": 0, "achieved_tier": None, "consumed_input_revision": None, "recipe_digest": None,
        "target_generation": None, "detail": {"source_version": 1, "legacy_consecutive_failures": 3,
            "excluded": True, "deleted": False, "evidence": copy.deepcopy(event["evidence"])}}
    from papervault.store.repository import payload_hash
    head["projection_digest"] = payload_hash(job)
    validate_legacy_projection(head, job, event)
    for mutate in (lambda j: j["detail"].update(legacy_consecutive_failures=2),
                   lambda j: j["detail"]["evidence"]["raw"].update(status="done"),
                   lambda j: j.update(recipe_digest="invented")):
        changed = copy.deepcopy(job)
        mutate(changed)
        with pytest.raises(ValueError, match="projection"):
            validate_legacy_projection(head, changed, event)


@pytest.mark.parametrize("grantee", ["pg_monitor", "PUBLIC"])
def test_source_freeze_revokes_column_insert_and_update_grants(repo, grantee):
    capture = source_capture(repo)
    repo.connection.execute(sql.SQL("GRANT INSERT(workspace),UPDATE(status) ON {} TO {}").format(
        repo._table("ks_ledger"), sql.SQL("PUBLIC") if grantee == "PUBLIC" else sql.Identifier(grantee)))
    assert repo.connection.execute("SELECT has_any_column_privilege('pg_monitor',%s,'INSERT,UPDATE') AS allowed",
        (repo._table("ks_ledger").as_string(repo.connection),)).fetchone()["allowed"] is True
    frozen = capture.freeze(legacy_roles=["pg_monitor"], drain=evidence()["drain"])
    assert frozen["legacy_dml_revoked"] is True
    assert repo.connection.execute("SELECT has_any_column_privilege('pg_monitor',%s,'INSERT,UPDATE') AS allowed",
        (repo._table("ks_ledger").as_string(repo.connection),)).fetchone()["allowed"] is False


def test_source_freeze_detects_inherited_column_write_privilege_and_refuses_partial_fence(repo):
    capture = source_capture(repo)
    repo.connection.execute(sql.SQL("GRANT UPDATE(status) ON {} TO pg_read_all_stats").format(repo._table("ks_ledger")))
    with pytest.raises(ValueError, match="inherited"):
        capture.freeze(legacy_roles=["pg_monitor"], drain=evidence()["drain"])
    assert capture._state()["fenced"] is False
    frozen = capture.freeze(legacy_roles=["pg_monitor", "pg_read_all_stats"], drain=evidence()["drain"])
    assert frozen["legacy_dml_revoked"] is True
