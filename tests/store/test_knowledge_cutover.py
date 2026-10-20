"""The writer handoff requires a complete capture and durable source/backend fences."""
import copy

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
    with psycopg.connect(repo.connection.info.dsn, autocommit=True) as other:
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

    event = {"source_row_id": "a", "source_version": 1, "event_digest": "digest", "evidence": {
        "raw": {"attempts": 3, "status": "error_parked"}, "event_digest": "digest"}}
    head = {"source_row_id": "a", "source_version": 1, "event_digest": "digest", "deleted": False}
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
