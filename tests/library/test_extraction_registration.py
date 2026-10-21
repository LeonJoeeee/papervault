"""Completion evidence survives execution, publication faults and later delivery."""

import asyncio
import hashlib
import json

import pytest

from papervault.library import Library, extract


@pytest.fixture
def extraction(tmp_path, monkeypatch):
    library = Library(tmp_path)
    paper, _ = library.upsert({"title": "Measured evidence", "authors": ["A. Author"], "year": 2026})
    library.pdf_path(paper.key).write_bytes(b"%PDF exact consumed source")
    paper.pdf_path = str(library.pdf_path(paper.key).relative_to(tmp_path))
    monkeypatch.setattr(extract, "pdf_probe", lambda *_: extract.PDFProbe(False, 2))
    monkeypatch.setattr(extract, "review_extract", lambda *_, **kw: {"ok": True})
    monkeypatch.setattr(extract, "confirmed_completeness_gate", lambda *_, **kw: {"complete": True})

    async def engine(*_, **kw):
        return "# Verified output\nThe measured value is 42."

    monkeypatch.setattr(extract, "extract_mineru", engine)
    yield library, paper
    library.close()


def completions(library):
    return [json.loads(p.read_text()) for p in sorted(
        library.root.glob("extracts/versions/*/completion.json"))]


def test_legacy_success_retains_exact_pending_evidence(extraction):
    library, paper = extraction
    output = asyncio.run(extract.extract_md(paper, library))
    records = completions(library)
    assert len(records) == 1, "fresh extraction must durably record pending registration"
    completion = records[0]
    assert completion["registration"] == "pending"
    assert completion["record_key"] == paper.key
    assert completion["unit_evidence"]["kind"] == "paper"
    assert completion["source_catalog_revision"] is None
    assert completion["recipe"]["engine"] == "mineru2.5-pro"
    assert completion["recipe"]["configured_model"] == "MinerU2.5-Pro-2605-1.2B"
    assert completion["provider_observations"]["actual_model"] == "unknown"
    source = completion["input_manifest"]["artifacts"][0]
    assert source["sha256"] == hashlib.sha256(b"%PDF exact consumed source").hexdigest()
    assert (library.root / source["path"]).read_bytes() == b"%PDF exact consumed source"
    produced = completion["output_inventory"][0]
    assert produced["sha256"] == hashlib.sha256(output.encode()).hexdigest()
    assert (library.root / produced["path"]).read_text() == output
    assert paper.md_path == produced["path"]
    assert completion["generation_id"]
    assert completion["achieved_tier"] == "fulltext"


def test_consumed_input_is_frozen_before_engine_call(extraction, monkeypatch):
    library, paper = extraction

    async def engine(data, *_args, **_kw):
        assert data == b"%PDF exact consumed source"
        library.pdf_path(paper.key).write_bytes(b"%PDF later source")
        return "The old source produced this exact output."

    monkeypatch.setattr(extract, "extract_mineru", engine)
    # A changed source refuses completion, retaining the completed old evidence.
    with pytest.raises(RuntimeError, match="source.*changed"):
        asyncio.run(extract.extract_md(paper, library))
    assert not library.has_extract(paper.key, "md")
    source = completions(library)[0]["input_manifest"]["artifacts"][0]
    assert (library.root / source["path"]).read_bytes() == b"%PDF exact consumed source"


def test_forced_extraction_preserves_prior_version(extraction, monkeypatch):
    library, paper = extraction
    first = asyncio.run(extract.extract_md(paper, library))
    original_path = library.root / paper.md_path

    async def changed_engine(*_, **kw):
        return "A different verified output."

    monkeypatch.setattr(extract, "extract_mineru", changed_engine)
    second = asyncio.run(extract.extract_md(paper, library, force=True))
    assert original_path.read_text() == first, "re-extraction cannot overwrite a derivative version"
    assert (library.root / paper.md_path).read_text() == second
    assert len(completions(library)) == 2
    assert library.md_path(paper.key).read_text() == second


def test_output_before_registration_recovers_without_reexecution(extraction, monkeypatch):
    from papervault.library.services import extraction_registration as registration

    library, paper = extraction
    original = registration.deliver_completion

    def crash(*args, **kwargs):
        raise RuntimeError("crash before registration")

    monkeypatch.setattr(registration, "deliver_completion", crash)
    with pytest.raises(RuntimeError, match="crash before registration"):
        asyncio.run(extract.extract_md(paper, library))
    assert len(completions(library)) == 1
    assert not library.has_extract(paper.key, "md")
    before = library.manifest_path.read_text()
    assert '"event": "extract_md_done"' not in before
    monkeypatch.setattr(registration, "deliver_completion", original)

    async def no_execution(*_, **kw):
        pytest.fail("registration recovery must not execute OCR")

    monkeypatch.setattr(extract, "extract_mineru", no_execution)
    assert registration.recover_completions(library, key=paper.key)["recovered"] == 1
    assert registration.recover_completions(library, key=paper.key)["recovered"] == 0
    assert paper.extract_attempts == 0
    assert library.manifest_path.read_text() == before
    assert len(completions(library)) == 1
    assert library.has_extract(paper.key, "md")


def test_recovery_refuses_corrupt_immutable_output(extraction):
    from papervault.library.services.extraction_registration import recover_completions

    library, paper = extraction
    asyncio.run(extract.extract_md(paper, library))
    (library.root / paper.md_path).write_text("corrupted")
    with pytest.raises(ValueError, match="hash|integrity"):
        recover_completions(library, key=paper.key)


def test_existing_legacy_output_does_not_invent_a_recipe(extraction):
    library, paper = extraction
    library.md_path(paper.key).write_text("Historical output with no known execution")
    asyncio.run(extract.extract_md(paper, library))
    assert completions(library) == []


def test_firecrawl_completion_captures_observed_response_and_unknown_provider(extraction, monkeypatch):
    from types import SimpleNamespace
    from papervault.library.download_sources.firecrawl import _try_firecrawl_text_fallback

    library, paper = extraction
    paper.url = "https://example.test/paper"
    body = "  # Measured page\n" + "The observed value is forty-two. " * 80 + "\n "
    monkeypatch.setenv("FIRECRAWL_API_URL", "http://127.0.0.1:3002")
    monkeypatch.setattr("papervault.library.download_sources.firecrawl.requests.post",
        lambda *_, **kw: SimpleNamespace(ok=True, json=lambda: {"success": True, "data": {"markdown": body}}))
    assert _try_firecrawl_text_fallback(paper, library)
    records = completions(library)
    assert len(records) == 1
    completion = records[0]
    assert completion["recipe"]["engine"] == "firecrawl"
    artifact = completion["input_manifest"]["artifacts"][0]
    assert artifact["role"] == "provider_markdown"
    assert (library.root / artifact["path"]).read_text() == body
    assert completion["provider_observations"]["source_bytes"] == "not_disclosed"
    assert completion["provider_observations"]["engine_version"] == "unknown"
    assert completion["recipe"]["request"]["url"] == paper.url


def test_firecrawl_registration_fault_does_not_leave_success_fields(extraction, monkeypatch):
    from types import SimpleNamespace
    from papervault.library.download_sources.firecrawl import _try_firecrawl_text_fallback
    from papervault.library.services import extraction_registration as registration

    library, paper = extraction
    paper.url = "https://example.test/paper"
    body = "# Paper\n" + "Measured evidence. " * 100
    monkeypatch.setenv("FIRECRAWL_API_URL", "http://127.0.0.1:3002")
    monkeypatch.setattr("papervault.library.download_sources.firecrawl.requests.post",
        lambda *_, **kw: SimpleNamespace(ok=True, json=lambda: {"success": True, "data": {"markdown": body}}))

    def crash(*_, **kw):
        raise ConnectionError("registration unavailable")

    monkeypatch.setattr(registration, "deliver_completion", crash)
    before = paper.download_status
    with pytest.raises(ConnectionError, match="registration unavailable"):
        _try_firecrawl_text_fallback(paper, library)
    assert paper.download_status == before
    assert not paper.firecrawl_pdf_hunt_exhausted
    assert not library.has_extract(paper.key, "md")
    assert len(completions(library)) == 1


def test_queue_recovers_output_before_early_success_shortcut(extraction, monkeypatch):
    from papervault.library.services import extraction_registration as registration
    from papervault.library.services.extract_queue import ExtractQueue

    library, paper = extraction
    original = registration.deliver_completion
    monkeypatch.setattr(registration, "deliver_completion",
        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("unavailable")))
    with pytest.raises(ConnectionError):
        asyncio.run(extract.extract_md(paper, library))
    monkeypatch.setattr(registration, "deliver_completion", original)

    async def forbidden(*_, **kw):
        pytest.fail("queue recovery must not spend another extraction")

    monkeypatch.setattr("papervault.library.services.extract_queue.extract_md", forbidden)
    seen = []
    queue = ExtractQueue(library, on_success=lambda key: seen.append(key))
    asyncio.run(queue._process_one_inner(paper.key, 0))
    assert seen == [paper.key]
    assert paper.extract_attempts == 0


def test_restart_after_projection_before_index_save_recovers_exact_output(extraction):
    from papervault.library.services.extraction_registration import recover_completions

    library, paper = extraction
    library.save(force=True, durable=True)
    output = asyncio.run(extract.extract_md(paper, library))
    completion_id = paper.extraction_completion_id
    restarted = Library(library.root)
    try:
        assert restarted.get(paper.key).extraction_completion_id is None
        assert recover_completions(restarted, key=paper.key)["recovered"] == 1
        assert restarted.get(paper.key).extraction_completion_id == completion_id
        assert restarted.md_path(paper.key).read_text() == output
        assert restarted.get(paper.key).extract_attempts == 0
    finally:
        restarted.close()


def test_stale_pending_does_not_block_new_source_or_other_papers(extraction, monkeypatch):
    from papervault.library.services import extraction_registration as registration

    library, paper = extraction
    request = registration.begin_extraction(paper, library, library.pdf_path(paper.key).read_bytes())
    library.pdf_path(paper.key).write_bytes(b"%PDF newer revision")
    with pytest.raises(RuntimeError, match="source.*changed"):
        registration.finish_extraction(paper, library, request, "Old source output")
    report = registration.recover_completions(library)
    assert report["recovered"] == 0
    assert len(report["blocked"]) == 1
    asyncio.run(extract.extract_md(paper, library))
    assert len(completions(library)) == 2
    assert paper.extraction_completion_id


@pytest.mark.parametrize("change", ["mutable_path", "generation", "unsealed", "recipe", "input"])
def test_receipt_cannot_be_resealed_with_false_evidence(extraction, change):
    from papervault.store.extraction import ExtractionStore
    from papervault.store.repository import payload_hash

    library, paper = extraction
    asyncio.run(extract.extract_md(paper, library))
    completion = completions(library)[0]
    if change == "mutable_path":
        completion["output_inventory"][0]["path"] = str(library.md_path(paper.key).relative_to(library.root))
    elif change == "generation":
        completion["generation_id"] = "extraction:another"
    elif change == "unsealed":
        completion["sealed"] = False
    elif change == "recipe":
        completion["recipe"]["configured_model"] = "invented"
    else:
        completion["consumed_input_revision"] = "input:invented"
    completion["evidence_digest"] = payload_hash({k: v for k, v in completion.items() if k != "evidence_digest"})
    with pytest.raises(ValueError):
        ExtractionStore(library.root).verify(completion)


def test_loaded_transport_settings_change_recipe_identity(monkeypatch):
    from papervault.library import mineru_client
    from papervault.library.services.extraction_registration import effective_recipe
    from papervault.store.knowledge import canonical_recipe

    before = canonical_recipe(effective_recipe("mineru2.5-pro"))[1]
    monkeypatch.setattr(mineru_client, "_MINERU_INLINE_NATIVE_RETRIES", 71)
    after = effective_recipe("mineru2.5-pro")
    assert canonical_recipe(after)[1] != before
    assert after["settings"]["native_retries"] == 71


def test_rejection_never_deletes_an_immutable_adopted_version(extraction, monkeypatch):
    from papervault.library.download_sources._shared import _gate_firecrawl_md
    from papervault.store.extraction import ExtractionStore

    library, paper = extraction
    asyncio.run(extract.extract_md(paper, library))
    completion = completions(library)[0]
    version = library.root / paper.md_path
    # A merge may serve the retained source version directly, without an alias.
    monkeypatch.setattr(library, "md_path", lambda _: version)
    monkeypatch.setattr(extract, "confirmed_completeness_gate", lambda *_: {"complete": False})
    assert not _gate_firecrawl_md(paper, library, "Rejected body")
    assert ExtractionStore(library.root).completion(completion["completion_id"]) == completion
    assert paper.md_path is None


def test_completion_syncs_storage_ancestors_before_delivery(extraction, monkeypatch):
    from papervault.store import extraction as evidence
    from papervault.library.services import extraction_registration as registration

    library, paper = extraction
    # These directories may have just been created by Library, without fsync.
    for child in ("inputs", "versions"):
        (library.root / "extracts" / child).mkdir(exist_ok=True)
    synced = []
    original_sync, original_deliver = evidence.fsync_directory, registration.deliver_completion

    def sync(path):
        original_sync(path)
        synced.append(path)

    def deliver(lib, completion):
        assert library.root in synced
        assert library.root / "extracts" in synced
        return original_deliver(lib, completion)

    monkeypatch.setattr(evidence, "fsync_directory", sync)
    monkeypatch.setattr(registration, "deliver_completion", deliver)
    asyncio.run(extract.extract_md(paper, library))


def test_extract_returns_newly_recovered_version(extraction, monkeypatch):
    from papervault.library.merge_v2 import MergeStore
    from papervault.library.services import extraction_registration as registration
    from papervault.store.extraction import ExtractionStore

    library, paper = extraction
    first = asyncio.run(extract.extract_md(paper, library))
    first_path = library.root / paper.md_path
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    destination, _ = library.upsert({"title": "Canonical destination", "authors": ["B. Author"], "year": 2026})
    paper = asyncio.run(MergeStore(library).merge_records(destination.key, paper.key))
    assert library.md_path(paper.key) == first_path
    request = registration.begin_extraction(paper, library, library.pdf_path(paper.key).read_bytes())
    recovered_text = "# Recovered revision\nA different measured result."
    completion = ExtractionStore(library.root).finalize(request, recovered_text.encode())
    before = library.manifest_path.read_text()

    async def no_execution(*_, **kw):
        pytest.fail("recovery must return the retained output without another OCR call")

    monkeypatch.setattr(extract, "extract_mineru", no_execution)
    assert asyncio.run(extract.extract_md(paper, library)) == recovered_text
    assert paper.extraction_completion_id == completion["completion_id"]
    assert library.md_path(paper.key).read_text() == recovered_text
    assert first_path.read_text() == first
    assert paper.extract_attempts == 0
    assert library.manifest_path.read_text() == before


@pytest.mark.parametrize("malformed", [b'{"record_key":', b'\xff', b'null', b'[]', b'{}'])
@pytest.mark.parametrize("per_key", [False, True])
def test_malformed_completion_does_not_block_healthy_recovery(extraction, caplog, malformed, per_key):
    from papervault.library.services.extraction_registration import begin_extraction, recover_completions
    from papervault.store.extraction import ExtractionStore

    library, paper = extraction
    store = ExtractionStore(library.root)
    request = begin_extraction(paper, library, library.pdf_path(paper.key).read_bytes())
    completion = store.finalize(request, b"# Healthy retained output")
    bad_path = store.versions / ("0" * 32) / "completion.json"
    bad_path.parent.mkdir()
    bad_path.write_bytes(malformed)
    assert not library.manifest_path.exists()

    report = recover_completions(library, key=paper.key if per_key else None)

    assert report == {"recovered": 1, "blocked": []}
    assert paper.extraction_completion_id == completion["completion_id"]
    assert library.md_path(paper.key).read_text() == "# Healthy retained output"
    assert bad_path.read_bytes() == malformed, "invalid evidence must remain available for diagnosis"
    assert str(bad_path) in caplog.text
    assert paper.extract_attempts == 0
    assert not library.manifest_path.exists()
