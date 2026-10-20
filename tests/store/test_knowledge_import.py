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


def test_independent_audited_job_edit_is_preserved_and_reimport_refuses_new_delta(repo):
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
    with pytest.raises(OperationConflict, match="independent"):
        importer.apply(plan, operation_id="next-delta")
    assert importer.knowledge._one("jobs", id=head["job_id"])["detail"] == edited_detail


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
