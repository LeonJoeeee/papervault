"""P5a atomic publication on the explicitly supplied disposable PostgreSQL."""

import copy

import psycopg
from psycopg import sql
import pytest

from papervault.store.extraction import ExtractionStore
from papervault.store.knowledge import ClaimRejected, KnowledgeRepository, OperationConflict
from papervault.store.repository import AuditEvent, Record


@pytest.fixture
def extraction_catalog(repo):
    repo.ensure_knowledge_schema(audit=AuditEvent("test", "schema"))
    repo.connection.execute(sql.SQL("UPDATE {} SET authority='catalog'").format(repo._table("knowledge_runtime")))
    repo.upsert(Record.from_payload("A", {"key": "A", "title": "Exact source",
                "terminal_state": {"retrieval": "eligible", "policy": "clear", "merge": "canonical"}}),
                expected_revision=0, audit=AuditEvent("test", "record"))
    return KnowledgeRepository(repo)


def frozen(k, tmp_path, *, previous=None, captured_epoch=1):
    store = ExtractionStore(tmp_path)
    request = store.capture(record_key="A", metadata={"key": "A", "title": "Exact source"},
        source=b"%PDF frozen source", source_role="pdf",
        recipe={"schema_version": 1, "engine": "test-ocr", "configured_model": "known-config",
                "output_contract": {"fulltext": {"required": ["markdown"], "not_applicable": []}}},
        guard={"previous_completion_id": previous, "identity": {"key": "A", "title": "Exact source"}},
        unit_evidence={"kind": "paper", "record_key": "A", "identity": {"key": "A", "title": "Exact source"}},
        source_catalog_revision=k.repository.get("A").revision if captured_epoch else None,
        authority_epoch=captured_epoch)
    return store.finalize(request, b"# Immutable output\nMeasured evidence.")


def payload(k, completion, vault):
    return {"kind": "extraction", "record_key": "A", "completion": completion,
            "authority_epoch": 1, "expected_record_revision": k.repository.get("A").revision,
            "expected_head_revision": 0, "vault": str(vault)}


def publish(k, completion, vault, **changes):
    data = {**payload(k, completion, vault), **changes}
    return k.command("publish_derivation", "publish-extraction:" + completion["completion_id"], data)


def count(k, table):
    return k.connection.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(
        k.repository._table(table))).fetchone()["n"]


def test_extraction_atomically_registers_without_distillation_attempts(extraction_catalog, tmp_path):
    k = extraction_catalog
    completion = frozen(k, tmp_path)
    result = publish(k, completion, tmp_path)
    assert result["registration"] == "registered"
    row = k.get_derivation(result["derivation_id"])
    assert row["artifact_inputs"] == completion["input_manifest"]
    assert row["recipe_digest"] == completion["recipe_digest"]
    assert row["extraction_generation_id"] == completion["generation_id"]
    assert row["achieved_tier"] == "fulltext"
    assert k.get_extraction_receipt(completion["completion_id"])["receipt"] == completion
    assert len(k.list_outbox(kind="extraction_completed")) == 1
    assert count(k, "knowledge_attempts") == 0
    assert count(k, "publications") == 0
    assert count(k, "selections") == 0
    audit = k.connection.execute(sql.SQL("SELECT detail FROM {} WHERE event='extraction_registration'").format(
        k.repository._table("audit"))).fetchall()
    assert len(audit) == 1
    assert audit[0]["detail"]["completion_id"] == completion["completion_id"]


def test_lost_response_replay_does_not_duplicate_any_history(extraction_catalog, tmp_path):
    k = extraction_catalog
    completion = frozen(k, tmp_path)
    first = publish(k, completion, tmp_path)
    counts = {table: count(k, table) for table in (
        "derivations", "extraction_receipts", "knowledge_operations", "audit", "knowledge_outbox")}
    assert publish(k, completion, tmp_path) == first
    assert {table: count(k, table) for table in counts} == counts
    changed = copy.deepcopy(completion)
    changed["achieved_tier"] = "abstract"
    with pytest.raises(OperationConflict):
        publish(k, changed, tmp_path)


def test_outbox_failure_rolls_back_receipt_derivation_and_audit(extraction_catalog, tmp_path):
    k = extraction_catalog
    completion = frozen(k, tmp_path)
    before = count(k, "audit")
    k.connection.execute(sql.SQL("CREATE TRIGGER injected_failure BEFORE INSERT ON {} "
        "FOR EACH STATEMENT EXECUTE FUNCTION {}.reject_history_change()").format(
            k.repository._table("knowledge_outbox"), sql.Identifier(k.repository.schema)))
    with pytest.raises(psycopg.Error, match="append-only"):
        publish(k, completion, tmp_path)
    for table in ("derivations", "extraction_receipts", "input_manifests", "recipes", "knowledge_operations"):
        assert count(k, table) == 0
    assert count(k, "audit") == before
    assert ExtractionStore(tmp_path).completion(completion["completion_id"]) == completion


@pytest.mark.parametrize("changed", ["record", "epoch", "authority"])
def test_stale_completion_is_refused_and_output_retained(extraction_catalog, tmp_path, changed):
    k = extraction_catalog
    completion = frozen(k, tmp_path)
    data = payload(k, completion, tmp_path)
    if changed == "record":
        k.repository.upsert(Record.from_payload("A", {"key": "A", "title": "Changed source"}),
            expected_revision=k.repository.get("A").revision, audit=AuditEvent("test", "change"))
    elif changed == "epoch":
        k.connection.execute(sql.SQL("UPDATE {} SET authority_epoch=2").format(k.repository._table("knowledge_runtime")))
    else:
        k.connection.execute(sql.SQL("UPDATE {} SET authority='legacy'").format(k.repository._table("knowledge_runtime")))
    with pytest.raises(ClaimRejected):
        k.command("publish_derivation", "publish-extraction:" + completion["completion_id"], data)
    assert count(k, "derivations") == 0
    assert ExtractionStore(tmp_path).completion(completion["completion_id"]) == completion


def test_concurrent_completion_cannot_replace_new_head(extraction_catalog, tmp_path):
    k = extraction_catalog
    first, stale = frozen(k, tmp_path), frozen(k, tmp_path)
    publish(k, first, tmp_path)
    with pytest.raises(ClaimRejected, match="head|stale"):
        publish(k, stale, tmp_path)
    assert count(k, "derivations") == 1


def test_legacy_pending_registers_without_fabricating_an_execution(extraction_catalog, tmp_path):
    k = extraction_catalog
    completion = frozen(k, tmp_path, captured_epoch=None)
    result = publish(k, completion, tmp_path)
    receipt = k.get_extraction_receipt(completion["completion_id"])["receipt"]
    assert receipt["source_catalog_revision"] is None
    assert receipt["authority_epoch"] is None
    assert receipt["provider_observations"]["actual_model"] == "unknown"
    assert count(k, "knowledge_attempt_events") == 0
    assert count(k, "knowledge_attempts") == 0
    assert publish(k, completion, tmp_path) == result


def test_extraction_receipt_and_derivation_are_append_only(extraction_catalog, tmp_path):
    k = extraction_catalog
    publish(k, frozen(k, tmp_path), tmp_path)
    for table in ("extraction_receipts", "derivations"):
        with pytest.raises(psycopg.Error, match="append-only"):
            k.connection.execute(sql.SQL("DELETE FROM {}").format(k.repository._table(table)))


async def test_catalog_staged_blank_fill_and_completion_share_transaction(
        extraction_catalog, tmp_path, monkeypatch, catalog_test_dsn):
    from papervault.library.claims_v2 import ClaimStore
    from papervault.library.reopen_v2 import StagedAttempt
    from papervault.library.services.extraction_registration import begin_extraction, finish_extraction

    k = extraction_catalog
    library, paper = runtime_library(k, tmp_path, monkeypatch, catalog_test_dsn)
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    try:
        paper.arxiv_id = "2601.12345"
        paper.download_status = "failed"
        library.save(force=True, durable=True)
        claims = ClaimStore(library)
        campaign = claims.ledger.start_campaign(cap=1, keys=[paper.key])
        token = await claims.claim(paper.key, expected_state=paper.acquisition_state,
                                   expected_generation=0, campaign_id=campaign)
        assert await claims.pre_dispatch(token)
        stage = StagedAttempt(library, token)
        candidate = stage.library.get(paper.key)
        candidate.doi = "10.1234/verified"
        request = begin_extraction(candidate, stage.library, b"provider response",
            engine="firecrawl", source_role="provider_markdown")
        completion = finish_extraction(candidate, stage.library, request, "# Verified staged output")
        assert await stage.writeback(claims)
        stage.cleanup()
        row = k.repository.get(paper.key)
        assert row.payload["doi"] == "10.1234/verified"
        assert row.payload["extraction_completion_id"] == completion["completion_id"]
        assert row.payload["reopen_claim"]["phase"] == "completed"
        assert count(k, "extraction_receipts") == count(k, "derivations") == 1
        assert len(k.list_outbox(kind="extraction_completed")) == 1
    finally:
        library.close()


@pytest.mark.parametrize("existing_head", [False, True])
def test_capture_fences_concurrent_catalog_head_change(
        extraction_catalog, tmp_path, monkeypatch, catalog_test_dsn, existing_head):
    import asyncio
    from papervault.library import extract
    from papervault.library.services.extraction_registration import begin_extraction
    from papervault.store.extraction import register_completion

    k = extraction_catalog
    library, paper = runtime_library(k, tmp_path, monkeypatch, catalog_test_dsn)
    try:
        if existing_head:
            asyncio.run(extract.extract_md(paper, library))
            library.save(force=True, durable=True)
        store = ExtractionStore(tmp_path)
        source = library.pdf_path(paper.key).read_bytes()
        winner_request = begin_extraction(paper, library, source)
        stale_request = begin_extraction(paper, library, source)
        winner = store.finalize(winner_request, b"# Winning concurrent output")
        stale = store.finalize(stale_request, b"# Stale concurrent output")
        register_completion(k, tmp_path, winner)
        with pytest.raises(ClaimRejected, match="head|stale"):
            register_completion(k, tmp_path, stale)
        unit = k.get_unit_for_record(paper.key)
        assert k.get_extraction_head(unit["unit_id"])["completion_id"] == winner["completion_id"]
        assert k.get_extraction_receipt(stale["completion_id"]) is None
        assert store.completion(stale["completion_id"]) == stale
        assert count(k, "extraction_receipts") == 1 + int(existing_head)
    finally:
        library.close()


def test_retained_staged_completion_requires_guarded_publication(extraction_catalog, tmp_path):
    k = extraction_catalog
    store = ExtractionStore(tmp_path)
    original = frozen(k, tmp_path)
    request = {key: original[key] for key in ("schema_version", "record_key", "unit_evidence",
        "source_catalog_revision", "authority_epoch", "guard", "input_manifest", "consumed_input_revision",
        "recipe", "recipe_digest")}
    from uuid import uuid4
    request["completion_id"] = uuid4().hex
    request["staged_claim"] = {"lease_id": "unaccepted", "generation": 1}
    staged = store.finalize(request, b"unaccepted staged output")
    with pytest.raises(ClaimRejected, match="staged|claim"):
        publish(k, staged, tmp_path)
    assert count(k, "extraction_receipts") == 0


def runtime_library(k, tmp_path, monkeypatch, dsn):
    from papervault.library import Library, extract

    monkeypatch.setenv("PAPERVAULT_CATALOG_DSN", dsn)
    monkeypatch.setenv("PAPERVAULT_CATALOG_SCHEMA", k.repository.schema)
    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")
    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "postgres")
    library = Library(tmp_path, authority="postgres")
    paper = library.get("A")
    library.pdf_path("A").write_bytes(b"%PDF exact runtime input")
    monkeypatch.setattr(extract, "pdf_probe", lambda *_: extract.PDFProbe(False, 1))
    monkeypatch.setattr(extract, "review_extract", lambda *_, **kw: {"ok": True})
    monkeypatch.setattr(extract, "confirmed_completeness_gate", lambda *_, **kw: {"complete": True})

    async def engine(*_, **kw):
        return "# Runtime output\nThe data are measured."

    monkeypatch.setattr(extract, "extract_mineru", engine)
    return library, paper


def test_catalog_runtime_commits_before_success(extraction_catalog, tmp_path, monkeypatch, catalog_test_dsn):
    import asyncio
    from papervault.library import extract

    k = extraction_catalog
    library, paper = runtime_library(k, tmp_path, monkeypatch, catalog_test_dsn)
    try:
        asyncio.run(extract.extract_md(paper, library))
        assert count(k, "extraction_receipts") == 1, "catalog extraction cannot report unregistered success"
        completion = ExtractionStore(tmp_path).completion(paper.extraction_completion_id)
        assert completion["authority_epoch"] == 1
        assert k.get_extraction_receipt(paper.extraction_completion_id)["receipt"] == completion
    finally:
        library.close()


def test_catalog_failure_withholds_success_and_recovers_lost_reply(
        extraction_catalog, tmp_path, monkeypatch, catalog_test_dsn):
    import asyncio
    from papervault.library import extract
    from papervault.library.services.extraction_registration import recover_completions

    k = extraction_catalog
    library, paper = runtime_library(k, tmp_path, monkeypatch, catalog_test_dsn)
    original = KnowledgeRepository.command
    lost = False

    def command(self, action, operation_id, data):
        nonlocal lost
        result = original(self, action, operation_id, data)
        if action == "publish_derivation" and data.get("kind") == "extraction" and not lost:
            lost = True
            raise ConnectionError("commit reply lost")
        return result

    monkeypatch.setattr(KnowledgeRepository, "command", command)
    try:
        with pytest.raises(ConnectionError, match="commit reply lost"):
            asyncio.run(extract.extract_md(paper, library))
        assert not library.has_extract("A", "md")
        assert paper.extraction_completion_id is None
        assert '"event": "extract_md_done"' not in library.manifest_path.read_text()
        assert count(k, "derivations") == 1
        assert recover_completions(library, key="A")["recovered"] == 1
        assert recover_completions(library, key="A")["recovered"] == 0
        assert count(k, "derivations") == count(k, "extraction_receipts") == 1
        assert len(k.list_outbox(kind="extraction_completed")) == 1
        assert paper.extract_attempts == 0
        assert '"event": "extract_md_done"' not in library.manifest_path.read_text()
    finally:
        library.close()


def test_legacy_runtime_transition_preserves_captured_unknown_revision(
        extraction_catalog, tmp_path, monkeypatch, catalog_test_dsn):
    import asyncio
    from papervault.library import Library, extract
    from papervault.library.services.extraction_registration import recover_completions

    k = extraction_catalog
    # Standalone JSON completion has no historical catalog revision or epoch.
    library, _ = runtime_library(k, tmp_path, monkeypatch, catalog_test_dsn)
    library.close()
    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "legacy")
    legacy = Library(tmp_path, authority="json")
    from papervault.library.models import Paper
    legacy._papers["A"] = Paper.model_validate(k.repository.get("A").payload)
    try:
        paper = legacy.get("A")
        asyncio.run(extract.extract_md(paper, legacy))
        legacy.save(force=True, durable=True)
        captured = ExtractionStore(tmp_path).completion(paper.extraction_completion_id)
        assert count(k, "derivations") == 0
        k.repository.upsert(Record.from_payload("A", paper.model_dump(mode="json")),
            expected_revision=k.repository.get("A").revision, audit=AuditEvent("test", "import_legacy"))
    finally:
        legacy.close()
    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "postgres")
    live = Library(tmp_path, authority="postgres")
    try:
        recover_completions(live, key="A")
        receipt = k.get_extraction_receipt(captured["completion_id"])
        assert receipt is not None
        assert receipt["receipt"] == captured
        assert captured["source_catalog_revision"] is None
        assert captured["authority_epoch"] is None
        assert count(k, "knowledge_attempts") == count(k, "knowledge_attempt_events") == 0
    finally:
        live.close()


def test_intake_registers_pending_and_delivers_exact_lineage_once(extraction_catalog, tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake
    from papervault.knowledge.scheduler.catalog_service import CatalogService

    k = extraction_catalog
    completion = frozen(k, tmp_path, captured_epoch=None)
    row = k.repository.get("A")
    k.repository.upsert(Record.from_payload("A", {**row.payload,
        "md_path": completion["output_inventory"][0]["path"],
        "extraction_completion_id": completion["completion_id"]}),
        expected_revision=row.revision, audit=AuditEvent("test", "legacy_completion"))
    k.command("create_generation", "active", {"generation_id": "active", "candidate": False})
    intake = CatalogIntake(k, ImmutableInputStore(tmp_path / "distill-inputs"), tmp_path,
        recipe={"schema_version": 1, "output_contract": {"fulltext": {"required": ["graph"]}}})
    assert intake.scan(only_keys=["A"])["captured"] == 1
    receipt = k.get_extraction_receipt(completion["completion_id"])
    assert receipt is not None
    requirement = k.get_requirement(receipt["unit_id"], "active")
    manifest = k.get_input(requirement["consumed_input_revision"])["manifest"]
    assert manifest["upstream_derivation_ids"] == ["extraction-derivation:" + completion["completion_id"]]
    # Conventional compatibility output changes cannot rewrite registered evidence.
    (tmp_path / "A.md").write_text("Unrelated replacement projection")
    service = CatalogService(k, intake, None)
    service.recover_intake()
    service.recover_intake()
    assert len(k.list_jobs()) == 1
    assert len(k.list_outbox(kind="extraction_completed")) == 0
    assert intake.input_store.read(manifest["artifacts"][0]) == b"# Immutable output\nMeasured evidence."
    assert count(k, "knowledge_attempts") == 0


@pytest.mark.parametrize("pending", [False, True])
def test_merged_extraction_keeps_origin_lineage(extraction_catalog, tmp_path, pending):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake
    from papervault.knowledge.scheduler.catalog_service import CatalogService

    k = extraction_catalog
    completion = frozen(k, tmp_path, captured_epoch=None if pending else 1)
    if not pending:
        publish(k, completion, tmp_path)
    k.repository.upsert(Record.from_payload("B", {"key": "B", "title": "Canonical paper",
        "md_path": completion["output_inventory"][0]["path"], "extraction_completion_id": completion["completion_id"],
        "terminal_state": {"retrieval": "acquired", "policy": "clear", "merge": "canonical"}}),
        expected_revision=0, audit=AuditEvent("test", "canonical"))
    row = k.repository.get("A")
    k.repository.upsert(Record.from_payload("A", {**row.payload, "terminal_state": {
        "retrieval": "acquired", "policy": "clear", "merge": "alias", "alias_key": "B", "quarantine": True}}),
        expected_revision=row.revision, audit=AuditEvent("test", "merge"))
    k.command("create_generation", "active", {"generation_id": "active", "candidate": False})
    intake = CatalogIntake(k, ImmutableInputStore(tmp_path / "distill-inputs"), tmp_path,
        recipe={"schema_version": 1, "output_contract": {"fulltext": {"required": ["graph"]}}})
    assert intake.scan(only_keys=["B"])["captured"] == 1
    unit = k.get_unit_for_record("B")
    requirement = k.get_requirement(unit["unit_id"], "active")
    manifest = k.get_input(requirement["consumed_input_revision"])["manifest"]
    receipt = k.get_extraction_receipt(completion["completion_id"])
    assert receipt["receipt"]["record_key"] == "A"
    assert receipt["unit_id"] != unit["unit_id"]
    assert manifest["upstream_derivation_ids"] == ["extraction-derivation:" + completion["completion_id"]]
    service = CatalogService(k, intake, None)
    service.recover_intake()
    assert not k.list_outbox(kind="extraction_completed")
    assert len(k.list_jobs()) == 1
    assert count(k, "knowledge_attempts") == 0


def test_catalog_cannot_shortcut_success_for_refused_pending_output(
        extraction_catalog, tmp_path, monkeypatch, catalog_test_dsn):
    import asyncio
    from papervault.library import extract

    k = extraction_catalog
    completion = frozen(k, tmp_path, captured_epoch=None)
    row = k.repository.get("A")
    k.repository.upsert(Record.from_payload("A", {**row.payload, "title": "Different identity",
        "md_path": completion["output_inventory"][0]["path"],
        "extraction_completion_id": completion["completion_id"]}),
        expected_revision=row.revision, audit=AuditEvent("test", "concurrent_identity"))
    library, paper = runtime_library(k, tmp_path, monkeypatch, catalog_test_dsn)
    library.md_path("A").write_bytes(ExtractionStore(tmp_path).read_artifact(completion["output_inventory"][0]))
    try:
        with pytest.raises(ClaimRejected, match="identity"):
            asyncio.run(extract.extract_md(paper, library))
        assert count(k, "extraction_receipts") == 0
    finally:
        library.close()


@pytest.mark.parametrize("recover_first", [False, True])
async def test_lost_reply_then_merge_into_existing_head_allows_recovery_and_fresh_extraction(
        extraction_catalog, tmp_path, monkeypatch, catalog_test_dsn, recover_first):
    from papervault.library import extract
    from papervault.library.merge_v2 import MergeStore
    from papervault.library.services.extraction_registration import recover_completions

    k = extraction_catalog
    k.repository.upsert(Record.from_payload("B", {"key": "B", "title": "Merge destination",
        "terminal_state": {"retrieval": "eligible", "policy": "clear", "merge": "canonical"}}),
        expected_revision=0, audit=AuditEvent("test", "destination"))
    library, origin = runtime_library(k, tmp_path, monkeypatch, catalog_test_dsn)
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    destination = library.get("B")
    library.pdf_path("B").write_bytes(b"%PDF destination source")
    original = KnowledgeRepository.command
    lost = False

    def command(self, action, operation_id, data):
        nonlocal lost
        result = original(self, action, operation_id, data)
        if (action == "publish_derivation" and data.get("kind") == "extraction"
                and data["record_key"] == "B" and not lost):
            lost = True
            raise ConnectionError("destination commit reply lost")
        return result

    try:
        await extract.extract_md(origin, library)
        library.save(force=True, durable=True)
        origin_id = origin.extraction_completion_id
        monkeypatch.setattr(KnowledgeRepository, "command", command)
        with pytest.raises(ConnectionError, match="destination commit reply lost"):
            await extract.extract_md(destination, library)
        unit = k.get_unit_for_record("B")
        committed_id = k.get_extraction_head(unit["unit_id"])["completion_id"]
        committed = ExtractionStore(tmp_path).completion(committed_id)
        assert not library.has_extract("B", "md")
        assert destination.extraction_completion_id is None

        assert await MergeStore(library).merge_records("B", "A") is destination
        assert destination.extraction_completion_id == origin_id
        assert k.get_extraction_head(unit["unit_id"])["completion_id"] == committed_id
        assert k.get_extraction_receipt(origin_id)["receipt"]["record_key"] == "A"
        before_recovery = library.manifest_path.read_text()
        if recover_first:
            report = recover_completions(library, key="B")
            assert report["recovered"] == 1, "the committed destination head must remain recoverable"
            assert report["blocked"] == []
            assert destination.extraction_completion_id == committed_id
            assert library.manifest_path.read_text() == before_recovery
            library.save(force=True, durable=True)

        await extract.extract_md(destination, library, force=True)
        fresh_id = destination.extraction_completion_id
        assert fresh_id not in {origin_id, committed_id}
        fresh = ExtractionStore(tmp_path).completion(fresh_id)
        assert fresh["guard"]["previous_completion_id"] == (committed_id if recover_first else origin_id)
        assert fresh["guard"]["catalog_previous_completion_id"] == committed_id
        assert k.get_extraction_head(unit["unit_id"])["completion_id"] == fresh_id
        before_recovery = library.manifest_path.read_text()
        assert recover_completions(library, key="B")["recovered"] == 0
        assert recover_completions(library, key="B")["recovered"] == 0
        assert destination.extraction_completion_id == fresh_id
        assert library.manifest_path.read_text() == before_recovery
        assert ExtractionStore(tmp_path).completion(committed_id) == committed
        assert count(k, "derivations") == count(k, "extraction_receipts") == 3
        assert len(k.list_outbox(kind="extraction_completed")) == 3
        assert count(k, "knowledge_attempts") == count(k, "knowledge_attempt_events") == 0
        assert destination.extract_attempts == 0
    finally:
        library.close()
