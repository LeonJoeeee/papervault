"""Unverifiable legacy claims stay non-executable and retain source identities."""
import copy

import pytest


def source(source_id="A", **changes):
    return {"workspace": "probe", "ingest_source": "paper", "source_id": source_id,
            "doc_id": "doc:" + source_id, "status": "done", "attempts": 3,
            "fingerprint": "old-fingerprint", "last_processed_at": "2026-10-19T12:00:00Z", **changes}


def snapshot(rows=None):
    return {"source": "ledger-test", "capture_id": "capture1", "checkpoint": 10,
            "rows": [source()] if rows is None else rows, "configuration": {"workspace": "probe"},
            "schema": {"table": "public.ks_ledger"}}


def mapping(source_id="A", unit_id="paper:A", kind="paper"):
    return {"workspace": "probe", "ingest_source": "paper", "source_id": source_id,
            "unit_id": unit_id, "kind": kind, "record_key": "A"}


def test_done_claim_keeps_failure_streak_and_raw_evidence_without_executable_provenance():
    from papervault.store.knowledge_import import plan_legacy_import

    original = snapshot()
    before = copy.deepcopy(original)
    plan = plan_legacy_import(original, mappings=[mapping()], doc_status_rows=[
        {"workspace": "probe", "doc_id": "doc:A", "status": "processed"}])
    entry = plan["events"][0]
    assert entry["dispositions"] == ["matched", "provenance_gap"]
    assert entry["raw"]["attempts"] == 3
    assert entry["legacy_consecutive_failures"] == 3
    assert entry["claimed_tier"] == "fulltext"
    assert entry["achieved_tier"] is None
    assert entry["executable"] is False and entry["excluded"] is True
    assert entry["consumed_input_revision"] is None
    assert entry["recipe_digest"] is None
    assert entry["generation_id"] is None
    assert original == before


def test_every_mismatch_gets_keyed_disposition_including_docstatus_only_duplicate_conflict():
    from papervault.store.knowledge_import import plan_legacy_import

    plan = plan_legacy_import(snapshot([source(), source("B"), source("C")]), mappings=[
        mapping(), mapping("B", "paper:A"), mapping("C", "paper:C"), mapping("C", "section:C", "section")],
        doc_status_rows=[{"workspace": "probe", "doc_id": "doc:A", "status": "processed"},
                         {"workspace": "probe", "doc_id": "orphan", "status": "processed"}])
    assert len(plan["sets"]["docstatus_only"]) == 1
    assert len(plan["sets"]["ledger_only"]) == 2
    assert len(plan["sets"]["duplicates"]) == 2
    assert len(plan["sets"]["conflicts"]) == 1
    assert len(plan["sets"]["provenance_gaps"]) == 3
    assert all(event["quarantine"] for event in plan["events"])


def test_journal_updates_deletions_are_contiguous_and_distinct_source_versions():
    from papervault.store.knowledge_import import plan_legacy_import

    changes = [{"sequence": 11, "operation": "UPDATE", "old": source(), "new": source(status="error", attempts=4)},
               {"sequence": 12, "operation": "DELETE", "old": source(status="error", attempts=4), "new": None}]
    plan = plan_legacy_import(snapshot(), mappings=[mapping()], changes=changes, final_checkpoint=12)
    assert [event["source_version"] for event in plan["events"]] == [10, 11, 12]
    assert plan["events"][-1]["deleted"] is True
    assert plan["checkpoint"]["last_sequence"] == 12
    with pytest.raises(ValueError, match="contiguous"):
        plan_legacy_import(snapshot(), changes=changes[1:], final_checkpoint=12)
    with pytest.raises(ValueError, match="contiguous"):
        plan_legacy_import(snapshot(), changes=changes[:1], final_checkpoint=12)


def test_readonly_snapshot_without_capture_never_claims_a_handoff_checkpoint():
    from papervault.store.knowledge_import import plan_legacy_import

    raw = snapshot()
    raw.update(capture_id=None, checkpoint=None)
    plan = plan_legacy_import(raw)
    assert plan["checkpoint"]["handoff_eligible"] is False
    assert plan["checkpoint"]["last_sequence"] is None


def test_import_apply_is_resumable_and_does_not_touch_catalog_owned_job(repo):
    from papervault.store.knowledge import KnowledgeRepository
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
    from papervault.store.repository import AuditEvent, Record

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    importer = LegacyImportRepository(repo)
    importer.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    repo.upsert(Record.from_payload("A", {"key": "A", "title": "one"}), expected_revision=0,
                audit=AuditEvent("test", "record"))
    k = KnowledgeRepository(repo)
    k.command("register_unit", "unit", {"unit_id": "paper:A", "kind": "paper", "record_key": "A", "identity": "A"})
    plan = plan_legacy_import(snapshot(), mappings=[mapping()])
    first = importer.apply(plan, operation_id="import1")
    assert importer.apply(plan, operation_id="import1") == first
    job = repo.connection.execute("SELECT * FROM " + repo._table("jobs").as_string(repo.connection) +
                                  " WHERE kind='distill_legacy_unverified'").fetchone()
    assert job["state"] == "quarantined"
    assert job["achieved_tier"] is None
    assert job["consumed_input_revision"] is None
    assert job["attempts"] == 0
    assert job["detail"]["legacy_consecutive_failures"] == 3
    assert importer.apply(plan, operation_id="import2")["inserted"] == 0
    assert importer.source_versions("ledger-test")[0]["source_version"] == 10


def test_source_version_digest_does_not_change_when_mapping_or_observations_are_enriched():
    from papervault.store.knowledge_import import plan_legacy_import

    unmapped = plan_legacy_import(snapshot())
    mapped = plan_legacy_import(snapshot(), mappings=[mapping()], doc_status_rows=[
        {"workspace": "probe", "doc_id": "doc:A", "status": "processed"}])
    assert unmapped["events"][0]["event_digest"] == mapped["events"][0]["event_digest"]
    assert unmapped["events"][0]["dispositions"] != mapped["events"][0]["dispositions"]


def test_import_cli_dryrun_has_no_default_connection_and_reports_unverified_checkpoint(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    raw = snapshot()
    raw.update(capture_id=None, checkpoint=None)
    path = tmp_path / "source.json"
    path.write_text(json.dumps(raw))
    script = Path(__file__).resolve().parents[2] / "scripts/import_knowledge.py"
    completed = subprocess.run([sys.executable, str(script), "--snapshot", str(path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["checkpoint"]["handoff_eligible"] is False
    assert report["events"][0]["quarantine"] is True


def test_capture_rejects_journal_old_image_that_disagrees_with_aligned_snapshot():
    from papervault.store.knowledge_import import plan_legacy_import

    with pytest.raises(ValueError, match="old image"):
        plan_legacy_import(snapshot(), changes=[{"sequence": 11, "operation": "UPDATE",
            "old": source(status="error"), "new": source(status="done_meta")}])


def test_two_ledger_rows_claiming_same_backend_document_are_duplicate_even_without_mapping():
    from papervault.store.knowledge_import import plan_legacy_import

    plan = plan_legacy_import(snapshot([source(), source("B", doc_id="doc:A")]))
    assert len(plan["sets"]["duplicates"]) == 2


def test_duplicate_docstatus_only_identity_is_explicit_duplicate_not_just_two_orphans():
    from papervault.store.knowledge_import import plan_legacy_import

    raw = {"workspace": "probe", "doc_id": "orphan", "status": "processed"}
    plan = plan_legacy_import(snapshot([]), doc_status_rows=[raw, {**raw, "status": "failed"}])
    assert len(plan["sets"]["docstatus_only"]) == 2
    assert len(plan["sets"]["duplicates"]) == 1
    assert plan["sets"]["duplicates"][0]["identity"] == ["probe", "orphan"]


@pytest.mark.parametrize("same_source_version", [False, True])
def test_independent_audited_job_edit_is_preserved_and_reimport_refuses_new_delta(repo, same_source_version):
    from papervault.store.knowledge import OperationConflict
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
    from papervault.store.repository import AuditEvent, PendingJob, Record

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    importer = LegacyImportRepository(repo)
    importer.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    repo.upsert(Record.from_payload("A", {"key": "A", "title": "one"}), expected_revision=0, audit=AuditEvent("test", "record"))
    importer.knowledge.command("register_unit", "unit", {"unit_id": "paper:A", "kind": "paper", "record_key": "A", "identity": "A"})
    importer.apply(plan_legacy_import(snapshot(), mappings=[mapping()]), operation_id="initial")
    head = importer.knowledge._rows("knowledge_import_heads")[0]
    job = importer.knowledge._one("jobs", id=head["job_id"])
    edited_detail = {**job["detail"], "operator_note": "preserve independent catalog edit"}
    repo.import_job(PendingJob(id=job["id"], kind=job["kind"], key=job["key"], state=job["state"],
        attempts=job["attempts"], lease_id=job["lease_id"], enqueued_at=job["enqueued_at"], detail=edited_detail),
        audit=AuditEvent("operator", "reviewed-legacy-job"))
    plan = plan_legacy_import(snapshot(), changes=[{"sequence": 11, "operation": "UPDATE", "old": source(),
        "new": source(status="error", attempts=4)}], final_checkpoint=11)
    if same_source_version:
        plan = plan_legacy_import(snapshot(), mappings=[{**mapping(), "mapping_revision": 2}])
    with pytest.raises(OperationConflict, match="independent"):
        importer.apply(plan, operation_id="next-delta")
    assert importer.knowledge._one("jobs", id=head["job_id"])["detail"] == edited_detail


def handoff_evidence(plan):
    from papervault.store.knowledge_cutover import DRAIN_FIELDS, VALIDATION_FIELDS

    checkpoint = plan["checkpoint"]
    return {"source": plan["source"], "capture_id": plan["capture_id"],
        "final_sequence": checkpoint["last_sequence"],
        "source_fence": {"capture_id": plan["capture_id"], "final_sequence": checkpoint["last_sequence"],
            "fenced": True, "legacy_dml_revoked": True, "fence_id": "test-fence", "revoked_roles": ["legacy"],
            "permission_checks": [{"role": "legacy", "can_write": False}],
            "final_row_count": checkpoint["final_row_count"], "final_rows_digest": checkpoint["final_rows_digest"]},
        "drain": {**dict.fromkeys(DRAIN_FIELDS, True), "receipt": "test-backend-fence"},
        "validation": {**dict.fromkeys(VALIDATION_FIELDS, True), "receipt": "test-key-validation"}}


@pytest.mark.parametrize("mapping_precedes_unit", [False, True])
@pytest.mark.parametrize("quarantined", [False, True])
def test_same_source_mapping_resolution_preserves_parking_through_handoff(repo, mapping_precedes_unit, quarantined):
    from papervault.store.knowledge import ClaimRejected
    from papervault.store.knowledge_cutover import CutoverRepository
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
    from papervault.store.repository import AuditEvent, Record

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    importer = LegacyImportRepository(repo)
    importer.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    raw = snapshot([source(status="error_parked", attempts=3, quarantine=quarantined,
                           quarantine_reason="operator parking", quarantine_scope="unit")])
    initial = plan_legacy_import(raw, mappings=[mapping()] if mapping_precedes_unit else [])
    importer.apply(initial, operation_id="unresolved-import")
    k = importer.knowledge
    head = k._rows("knowledge_import_heads")[0]
    assert k._one("jobs", id=head["job_id"])["unit_id"] is None
    original_events = importer.source_versions("ledger-test")
    repo.upsert(Record.from_payload("A", {"key": "A", "title": "one"}), expected_revision=0,
                audit=AuditEvent("test", "record"))
    k.command("register_unit", "unit", {"unit_id": "paper:A", "kind": "paper", "record_key": "A", "identity": "A"})
    enriched = plan_legacy_import(raw, mappings=[mapping()])
    result = importer.apply(enriched, operation_id="resolved-import")
    assert result["inserted"] == 0 and result["projected"] == 1
    projected = k._one("jobs", id=head["job_id"])
    assert projected["unit_id"] == "paper:A" and projected["key"] == "A"
    assert projected["detail"]["mapping_resolution"] == "verified"
    assert projected["detail"]["evidence"] == enriched["events"][0]
    assert importer.source_versions("ledger-test") == original_events
    assert importer.apply(enriched, operation_id="resolved-replay")["projected"] == 0
    assert k._one("jobs", id=head["job_id"]) == projected
    k.command("create_generation", "generation", {"generation_id": "g", "candidate": False})
    k.command("register_input", "input", {"unit_id": "paper:A", "consumed_input_revision": "input1",
        "manifest": {"metadata_revision": "1", "artifacts": [], "upstream_derivation_ids": []}})
    assert k.get_reducer("paper:A", "input1") is None
    assert CutoverRepository(repo).commit_handoff(handoff_evidence(enriched), expected_epoch=1,
                                                operation_id="handoff")["authority_epoch"] == 2
    recipe = k.command("register_recipe", "recipe", {"recipe": {"schema_version": 1,
        "output_contract": {"fulltext": {"required": ["graph"], "not_applicable": []}}}})["recipe_digest"]
    desired = {"unit_id": "paper:A", "generation_id": "g", "consumed_input_revision": "input1", "recipe_digest": recipe}
    k.command("set_requirement", "requirement", {**desired, "expected_revision": 0,
                                                "expected_record_revision": repo.get("A").revision})
    job = k.command("enqueue", "enqueue", desired)
    claim = k.command("claim", "claim", {"job_id": job["job_id"], "expected_revision": job["revision"], "authority_epoch": 2})
    assert claim["claimed"] is False and claim["parked"] is True and claim["attempt_id"] is None
    assert claim["quarantined"] is quarantined
    assert k.get_reducer("paper:A", "input1")["failure_streak"] == 3
    assert k.get_unit("paper:A")["quarantine"] is quarantined
    if quarantined:
        assert k.get_unit("paper:A")["quarantine_reason"]["details"][0]["reason"] == "operator parking"
    with pytest.raises(ClaimRejected, match="parked|quarantine"):
        k.command("claim", "claim-again", {"job_id": job["job_id"], "expected_revision": claim["revision"], "authority_epoch": 2})
    assert k.get_job(job["job_id"])["attempts"] == 0
    assert k._rows("knowledge_attempt_events") == []
    assert len(k._rows("knowledge_legacy_checkpoint_applications")) == 1


def test_stale_import_cannot_remap_a_newer_source_projection(repo):
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
    from papervault.store.repository import AuditEvent

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    importer = LegacyImportRepository(repo)
    importer.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    newer = plan_legacy_import(snapshot(), changes=[{"sequence": 11, "operation": "UPDATE", "old": source(),
        "new": source(status="error_parked", attempts=4)}], final_checkpoint=11)
    importer.apply(newer, operation_id="newer")
    before = importer.knowledge._rows("jobs")
    assert importer.apply(plan_legacy_import(snapshot(), mappings=[mapping()]), operation_id="stale")["projected"] == 0
    assert importer.knowledge._rows("jobs") == before
    assert importer.knowledge._rows("knowledge_import_checkpoints")[0]["checkpoint"]["last_sequence"] == 11


@pytest.mark.parametrize("mapping_precedes_unit", [False, True])
def test_handoff_rejects_resolved_report_with_unprojected_mapping(repo, mapping_precedes_unit):
    from papervault.store.knowledge_cutover import CutoverRepository
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
    from papervault.store.repository import AuditEvent, Record, payload_hash

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    importer = LegacyImportRepository(repo)
    importer.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    raw = snapshot([source(status="error_parked")])
    importer.apply(plan_legacy_import(raw, mappings=[mapping()] if mapping_precedes_unit else []), operation_id="initial")
    k = importer.knowledge
    repo.upsert(Record.from_payload("A", {"key": "A", "title": "one"}), expected_revision=0,
                audit=AuditEvent("test", "record"))
    k.command("register_unit", "unit", {"unit_id": "paper:A", "kind": "paper", "record_key": "A", "identity": "A"})
    k.command("create_generation", "generation", {"generation_id": "g", "candidate": False})
    enriched = plan_legacy_import(raw, mappings=[mapping()])
    if not mapping_precedes_unit:
        # Reproduce a checkpoint left by the old importer: enriched report and
        # checkpoint committed, but the unchanged-source projection was skipped.
        audit = AuditEvent("test", "old-importer-checkpoint")
        with repo._write(audit):
            digest = payload_hash(enriched)
            k._insert("knowledge_import_reports", {"plan_digest": digest, "source": enriched["source"],
                "capture_id": enriched["capture_id"], "report": enriched})
            k._update("knowledge_import_checkpoints", {"plan_digest": digest}, source=enriched["source"], capture_id=enriched["capture_id"])
            repo._finish(None, audit)
    with pytest.raises(ValueError, match="projection"):
        CutoverRepository(repo).commit_handoff(handoff_evidence(enriched), expected_epoch=1, operation_id="handoff")
    assert k.get_authority_state()["authority"] == "legacy"


def test_projection_guard_checks_full_last_imported_row_not_only_owner_and_source_version():
    from papervault.store.knowledge_import import validate_projection_unchanged
    from papervault.store.repository import payload_hash

    job = {"kind": "distill_legacy_unverified", "detail": {"migration_owner": "owner", "source_version": 10},
           "state": "quarantined", "revision": 1}
    head = {"projection_digest": payload_hash(job)}
    validate_projection_unchanged(head, job)
    job["detail"]["independent_note"] = "reviewed"
    from papervault.store.knowledge import OperationConflict
    with pytest.raises(OperationConflict, match="independent"):
        validate_projection_unchanged(head, job)


def schema_rows(repo):
    """Compare every durable row, including audit and operation histories."""
    from psycopg import sql

    tables = repo.connection.execute("SELECT table_name FROM information_schema.tables "
        "WHERE table_schema=%s AND table_type='BASE TABLE' ORDER BY table_name", (repo.schema,)).fetchall()
    return {table["table_name"]: sorted((repr(row) for row in repo.connection.execute(
        sql.SQL("SELECT to_jsonb(t) AS row FROM {} t").format(repo._table(table["table_name"])))), key=str)
        for table in tables}


def parked_importer(repo):
    from papervault.store.knowledge_import import LegacyImportRepository
    from papervault.store.repository import AuditEvent, Record

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "knowledge-schema"))
    importer = LegacyImportRepository(repo)
    importer.ensure_schema(audit=AuditEvent("test", "migration-schema"))
    for key in ("A", "B"):
        repo.upsert(Record.from_payload(key, {"key": key, "title": key}), expected_revision=0,
                    audit=AuditEvent("test", "record"))
        importer.knowledge.command("register_unit", "unit:" + key,
            {"unit_id": "paper:" + key, "kind": "paper", "record_key": key, "identity": key})
    return importer


def test_interrupted_unmapped_plan_cannot_replace_enriched_projection_or_handoff_parking(repo, monkeypatch):
    from papervault.store.knowledge import OperationConflict
    from papervault.store.knowledge_cutover import CutoverRepository
    from papervault.store.knowledge_import import plan_legacy_import

    importer = parked_importer(repo)
    k = importer.knowledge
    raw = snapshot([source(status="error_parked", attempts=7, quarantine=True,
                           quarantine_reason="review required", quarantine_scope="unit")])
    old_plan = plan_legacy_import(raw)
    enriched = plan_legacy_import(raw, mappings=[mapping()], doc_status_rows=[
        {"workspace": "probe", "doc_id": "doc:A", "status": "failed"}])
    real_insert = k._insert

    def interrupted_insert(table, values):
        result = real_insert(table, values)
        if table == "jobs":
            raise RuntimeError("interrupted before import commit")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(k, "_insert", interrupted_insert)
        with pytest.raises(RuntimeError, match="interrupted"):
            importer.apply(old_plan, operation_id="interrupted-A")
    importer.apply(enriched, operation_id="committed-B")
    before = schema_rows(repo)
    with pytest.raises(OperationConflict, match="regress|enrich|evidence"):
        importer.apply(old_plan, operation_id="interrupted-A")
    assert schema_rows(repo) == before
    k.command("create_generation", "generation", {"generation_id": "g", "candidate": False})
    k.command("register_input", "input", {"unit_id": "paper:A", "consumed_input_revision": "input1",
        "manifest": {"metadata_revision": "1", "artifacts": [], "upstream_derivation_ids": []}})
    CutoverRepository(repo).commit_handoff(handoff_evidence(enriched), expected_epoch=1, operation_id="handoff")
    k.command("migrate_legacy_checkpoint", "checkpoint",
              {"unit_id": "paper:A", "consumed_input_revision": "input1"})
    reducer = k.get_reducer("paper:A", "input1")
    assert reducer["failure_streak"] == 7 and reducer["parked"] is True
    assert k.get_unit("paper:A")["quarantine"] is True
    assert k.get_unit("paper:A")["quarantine_reason"]["details"][0]["reason"] == "review required"
    projected = k._rows("jobs", kind="distill_legacy_unverified")[0]
    assert projected["unit_id"] == "paper:A"


def test_equal_source_version_verified_mapping_conflict_refuses_whole_plan_before_writes(repo, monkeypatch):
    from papervault.store.knowledge import OperationConflict
    from papervault.store.knowledge_import import plan_legacy_import

    importer = parked_importer(repo)
    raw = snapshot([source("B"), source(status="error_parked", attempts=7)])
    accepted = plan_legacy_import(raw, mappings=[mapping()])
    importer.apply(accepted, operation_id="accepted")
    conflicting = plan_legacy_import(raw, mappings=[mapping("B", "paper:B"),
        {**mapping(unit_id="paper:B"), "record_key": "B"}])
    before = schema_rows(repo)
    mutations = []
    k = importer.knowledge
    real_insert, real_update = k._insert, k._update

    def record_insert(*args, **kwargs):
        mutations.append(args[0])
        return real_insert(*args, **kwargs)

    def record_update(*args, **kwargs):
        mutations.append(args[0])
        return real_update(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(k, "_insert", record_insert)
        patch.setattr(k, "_update", record_update)
        with pytest.raises(OperationConflict, match="conflicting verified mapping"):
            importer.apply(conflicting, operation_id="conflict")
    assert mutations == []
    assert schema_rows(repo) == before
    projected = next(job for job in k._rows("jobs") if job["unit_id"] == "paper:A")
    assert projected["quarantine"] is True
    assert projected["detail"]["legacy_consecutive_failures"] == 7


def test_same_version_projection_lineage_enriches_and_replay_does_not_append(repo):
    from papervault.store.knowledge_import import plan_legacy_import
    from papervault.store.repository import payload_hash

    importer = parked_importer(repo)
    old = plan_legacy_import(snapshot())
    new = plan_legacy_import(snapshot(), mappings=[mapping()], doc_status_rows=[
        {"workspace": "probe", "doc_id": "doc:A", "status": "processed"}])
    importer.apply(old, operation_id="unmapped")
    importer.apply(new, operation_id="enriched")
    history = sorted(importer.knowledge._rows("knowledge_import_projections"), key=lambda row: row["revision"])
    assert len(history) == 2
    assert history[0]["plan_digest"] == payload_hash(old)
    assert history[1]["plan_digest"] == payload_hash(new)
    assert history[1]["previous_plan_digest"] == payload_hash(old)
    assert history[1]["evidence"] == new["events"][0]
    before = schema_rows(repo)
    importer.apply(new, operation_id="enriched")
    assert schema_rows(repo) == before
    importer.apply(new, operation_id="equivalent-replay")
    assert importer.knowledge._rows("knowledge_import_projections") == history
    checkpoint = importer.knowledge._rows("knowledge_import_checkpoints")[0]
    assert checkpoint["revision"] == 2


def test_newer_source_version_may_change_verified_mapping(repo):
    from papervault.store.knowledge_import import plan_legacy_import

    importer = parked_importer(repo)
    importer.apply(plan_legacy_import(snapshot(), mappings=[mapping()]), operation_id="initial")
    newer = plan_legacy_import(snapshot(), mappings=[{**mapping(unit_id="paper:B"), "record_key": "B"}],
        changes=[{"sequence": 11, "operation": "UPDATE", "old": source(),
                  "new": source(status="error_parked", attempts=4)}], final_checkpoint=11)
    importer.apply(newer, operation_id="new-version")
    job = importer.knowledge._rows("jobs", kind="distill_legacy_unverified")[0]
    assert job["unit_id"] == "paper:B" and job["key"] == "B"
    assert job["detail"]["source_version"] == 11
    assert job["detail"]["legacy_consecutive_failures"] == 4


@pytest.mark.parametrize("lost_evidence", ["orphan", "matched", "disposition", "counter"])
def test_same_version_plan_cannot_discard_accepted_evidence(repo, lost_evidence):
    from papervault.store.knowledge import OperationConflict
    from papervault.store.knowledge_import import plan_legacy_import

    importer = parked_importer(repo)
    statuses = [{"workspace": "probe", "doc_id": "doc:A", "status": "processed"},
                {"workspace": "probe", "doc_id": "orphan", "status": "failed"}]
    mappings = [mapping()]
    if lost_evidence == "disposition":
        statuses.append({"workspace": "probe", "doc_id": "doc:A", "status": "failed"})
    accepted = plan_legacy_import(snapshot(), mappings=mappings, doc_status_rows=statuses)
    importer.apply(accepted, operation_id="accepted")
    if lost_evidence == "orphan":
        statuses = statuses[:1]
    elif lost_evidence == "matched":
        statuses = statuses[1:]
    elif lost_evidence == "disposition":
        statuses = statuses[:2]
    losing = plan_legacy_import(snapshot(), mappings=mappings, doc_status_rows=statuses)
    if lost_evidence == "counter":
        losing["events"][0]["legacy_consecutive_failures"] = 0
    before = schema_rows(repo)
    with pytest.raises(OperationConflict, match="regress|enrich|evidence"):
        importer.apply(losing, operation_id="losing")
    assert schema_rows(repo) == before


def test_older_checkpoint_cannot_enrich_projection_outside_accepted_report(repo):
    from papervault.store.knowledge import OperationConflict
    from papervault.store.knowledge_cutover import CutoverRepository
    from papervault.store.knowledge_import import plan_legacy_import
    from papervault.store.repository import payload_hash

    importer = parked_importer(repo)
    k = importer.knowledge
    raw = snapshot([source(status="error_parked", attempts=7, quarantine=True), source("B")])
    changes = [{"sequence": 11, "operation": "UPDATE", "old": source("B"),
                "new": source("B", status="error", attempts=4)}]
    initial = plan_legacy_import(raw)
    initial_result = importer.apply(initial, operation_id="old-checkpoint")
    accepted = plan_legacy_import(raw, changes=changes, final_checkpoint=11)
    importer.apply(accepted, operation_id="current-checkpoint")
    older_enriched = plan_legacy_import(raw, mappings=[mapping()])
    before = schema_rows(repo)
    assert importer.apply(initial, operation_id="old-checkpoint") == initial_result
    assert schema_rows(repo) == before
    with pytest.raises(OperationConflict, match="older.*checkpoint|checkpoint.*older"):
        importer.apply(older_enriched, operation_id="older-enrichment")
    assert schema_rows(repo) == before
    current_enriched = plan_legacy_import(raw, mappings=[mapping()], changes=changes, final_checkpoint=11)
    assert importer.apply(current_enriched, operation_id="current-enrichment")["projected"] == 1
    checkpoint = k._rows("knowledge_import_checkpoints")[0]
    assert checkpoint["checkpoint"]["last_sequence"] == 11
    assert checkpoint["plan_digest"] == payload_hash(current_enriched)
    k.command("create_generation", "generation", {"generation_id": "g", "candidate": False})
    assert CutoverRepository(repo).commit_handoff(handoff_evidence(current_enriched), expected_epoch=1,
                                                operation_id="handoff")["authority_epoch"] == 2
