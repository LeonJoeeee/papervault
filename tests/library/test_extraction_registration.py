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


def missing_arxiv_firecrawl(library, paper, monkeypatch, *, persist_before_capture):
    """Real preflight/queue/gates; only the source responses are replaced."""
    from types import SimpleNamespace
    from papervault.library import download
    from papervault.library.download_sources import arxiv, firecrawl
    from papervault.store.extraction import ExtractionStore

    library.pdf_path(paper.key).unlink(missing_ok=True)
    paper.pdf_path = None
    paper.doi, paper.arxiv_id = "10.1234/corrected", "2601.99999"
    library.save(force=True, durable=True)
    library._save_min_interval = 3600  # Capture must bypass an ordinary queue save's debounce.
    # Exercise the real not_found branch and real PDF miss before Firecrawl.
    monkeypatch.setattr(download, "_STRATEGIES", [("arxiv", download._try_arxiv)])
    monkeypatch.setattr(arxiv, "_lookup_arxiv_record", lambda identifier: arxiv._ArxivRecord(
        requested_id=identifier, status="not_found", reason="base_record_not_found"))
    monkeypatch.setenv("FIRECRAWL_API_URL", "http://127.0.0.1:3002")
    body = "# Measured evidence\n" + "The measured value is forty-two. " * 80

    def response(url, **kwargs):
        assert paper.arxiv_id == "" and paper.doi == "10.1234/corrected"
        assert kwargs["json"]["url"] == "https://doi.org/10.1234/corrected"
        if persist_before_capture:
            library.save(force=True, durable=True)
        return SimpleNamespace(ok=True, json=lambda: {"success": True, "data": {"markdown": body}})

    monkeypatch.setattr(firecrawl.requests, "post", response)
    capture = ExtractionStore.capture
    observed = []

    def observe_capture(store, **kwargs):
        request = capture(store, **kwargs)
        if library._catalog_backend:
            with library._catalog_backend._repository() as repo:
                record = repo.get(paper.key)
            persisted, revision = record.payload, record.revision
        else:
            persisted = json.loads(library.index_path.read_bytes())["papers"][paper.key]
            revision = None
        observed.append((request, persisted, revision))
        return request

    monkeypatch.setattr(ExtractionStore, "capture", observe_capture)
    return observed


@pytest.mark.parametrize("persist_before_capture", [False, True])
@pytest.mark.parametrize("interrupt_delivery", [False, True])
async def test_corrected_download_identity_survives_json_capture_and_restart(
        extraction, monkeypatch, persist_before_capture, interrupt_delivery):
    from papervault.library.services import extraction_registration as registration
    from papervault.library.services.download_queue import DownloadQueue
    from papervault.store.extraction import ExtractionStore

    library, paper = extraction
    observed = missing_arxiv_firecrawl(library, paper, monkeypatch,
                                      persist_before_capture=persist_before_capture)
    delivered = []
    queue = DownloadQueue(library, on_success=delivered.append)
    original = registration.deliver_completion

    def crash(*args, **kwargs):
        raise ConnectionError("stopped after durable output")

    if interrupt_delivery:
        monkeypatch.setattr(registration, "deliver_completion", crash)
        with pytest.raises(ConnectionError, match="stopped after durable output"):
            await queue._process_one_inner(paper.key, 0)
        assert not delivered and paper.extraction_completion_id is None
    else:
        await queue._process_one_inner(paper.key, 0)
        assert delivered == [paper.key]
    completion, = list(ExtractionStore(library.root).completions())
    request, persisted, revision = observed[0]
    # The corrected state must already be durable even if no queue save runs.
    assert persisted["arxiv_id"] == ""
    assert request["source_catalog_revision"] == revision is None
    assert completion["unit_evidence"]["identity"]["arxiv_id"] == ""
    assert completion["authority_epoch"] is None
    library.save(force=True, durable=True)  # A later queue/flush save is harmless.
    library.close()
    monkeypatch.setattr(registration, "deliver_completion", original)
    restarted = Library(library.root, authority="json")
    try:
        before = restarted.manifest_path.read_text()
        assert registration.recover_completions(restarted) == {
            "recovered": int(interrupt_delivery), "blocked": []}
        assert registration.recover_completions(restarted) == {"recovered": 0, "blocked": []}
        assert restarted.get(paper.key).extraction_completion_id == completion["completion_id"]
        assert ExtractionStore(library.root).completion(completion["completion_id"]) == completion
        assert restarted.manifest_path.read_text() == before
        assert restarted.get(paper.key).extract_attempts == 0
    finally:
        restarted.close()


def test_json_capture_serializes_snapshot_with_index_rotation(extraction, monkeypatch):
    from pathlib import Path
    from threading import Event, Thread, current_thread
    from filelock import FileLock
    from papervault.library import store as library_store
    from papervault.library.services.extraction_registration import begin_extraction

    library, paper = extraction
    rotated, release_writer, reading_snapshot = Event(), Event(), Event()
    save, atomic_write = library.save, library_store._atomic_write
    read_bytes, acquire = Path.read_bytes, FileLock.acquire
    errors = []

    def concurrent_save():
        try:
            save(force=True, durable=True)
        except Exception as exc:
            errors.append(exc)

    writer = Thread(target=concurrent_save, daemon=True)

    def pause_replacement(path, *args, **kwargs):
        if current_thread() is writer and path == library.index_path:
            assert not path.exists(), "the real writer has rotated index.json to its backup"
            rotated.set()
            assert release_writer.wait(5)
        return atomic_write(path, *args, **kwargs)

    def save_before_rotation(**kwargs):
        save(**kwargs)
        writer.start()
        assert rotated.wait(5)
        reading_snapshot.set()

    def snapshot_lock(lock, *args, **kwargs):
        if current_thread() is not writer and reading_snapshot.is_set():
            release_writer.set()
        return acquire(lock, *args, **kwargs)

    def snapshot_read(path):
        try:
            return read_bytes(path)
        finally:
            if path == library.index_path and reading_snapshot.is_set():
                # Without the writer lock this read sees the actual missing index;
                # release the writer afterward so a failed regression leaves no thread.
                release_writer.set()

    monkeypatch.setattr(library_store, "_atomic_write", pause_replacement)
    monkeypatch.setattr(library, "save", save_before_rotation)
    monkeypatch.setattr(FileLock, "acquire", snapshot_lock)
    monkeypatch.setattr(Path, "read_bytes", snapshot_read)
    try:
        request = begin_extraction(paper, library, b"%PDF input")
        assert request["unit_evidence"]["identity"]["key"] == paper.key
        assert request["source_catalog_revision"] is None
    finally:
        release_writer.set()
        if writer.ident is not None:
            writer.join(timeout=5)
        assert not writer.is_alive() and not errors


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
    monkeypatch.setattr(mineru_client, "_INLINE_RETRIES", 71)
    after = effective_recipe("mineru2.5-pro")
    assert canonical_recipe(after)[1] != before
    assert after["settings"]["inline_retries"] == 71


def test_http_task_recipe_identifies_client_and_unknown_remote_engine():
    from papervault.library.services.extraction_registration import effective_recipe

    recipe = effective_recipe("mineru2.5-pro")
    assert recipe["transport"] == "mineru-api-tasks"
    assert recipe["settings"]["backend"] == "vlm-engine"
    assert recipe["remote_engine_version"] == "unknown"
    assert "httpx" in recipe["packages"]
    assert "mineru" not in recipe["packages"]
    assert "native_retries" not in recipe["settings"]
    assert "native_backoff_factor" not in recipe["settings"]


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


@pytest.mark.parametrize("malformed", [
    b'{"record_key":', b'\xff', b'null', b'[]', b'{}',
    b'{"record_key": ""}', b'{"record_key": null}', b'{"record_key": 42}',
])
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


def test_completion_key_filter_precedes_evidence_validation(extraction, monkeypatch):
    from papervault.library.services.extraction_registration import begin_extraction
    from papervault.store.extraction import ExtractionStore

    library, paper = extraction
    other, _ = library.upsert({"title": "Unrelated solar plasma observations",
                               "authors": ["B. Author"], "year": 2025})
    store = ExtractionStore(library.root)
    for row in (paper, other):
        request = begin_extraction(row, library, b"# Input", engine="firecrawl",
                                   source_role="provider_markdown")
        store.finalize(request, b"# Retained output")
    validate = store._validate_evidence
    validated = []

    def observe(completion):
        validated.append(completion["record_key"])
        return validate(completion)

    monkeypatch.setattr(store, "_validate_evidence", observe)
    assert [c["record_key"] for c in store.completions(key=paper.key)] == [paper.key]
    assert validated == [paper.key]
    validated.clear()
    assert list(store.completions(key="Absent")) == []
    assert validated == []


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_recovery_excludes_keys_in_one_pass_and_isolates_invalid_evidence(
        extraction, monkeypatch, damage):
    from papervault.library.services.extraction_registration import begin_extraction, recover_completions
    from papervault.store.extraction import ExtractionStore

    library, healthy = extraction
    excluded, _ = library.upsert({"title": "Scan awaiting post OCR identity verification",
                                  "authors": ["Scan"], "year": 2024})
    broken, _ = library.upsert({"title": "Coronal magnetic fields and reconnection",
                                "authors": ["Broken"], "year": 2023})
    store = ExtractionStore(library.root)
    retained = {}
    for paper in (healthy, excluded, broken):
        request = begin_extraction(paper, library, b"# Input", engine="firecrawl",
                                   source_role="provider_markdown")
        completion = store.finalize(request, b"# Retained output")
        retained[paper.key] = completion
        if paper is not healthy:
            output = store.path(completion["output_inventory"][0]["path"])
            if damage == "corrupt":
                output.write_bytes(b"Corrupt")
            else:
                output.unlink()
    scans = []
    completions = ExtractionStore.completions

    def observe_scan(self, *, key=None):
        scans.append(key)
        yield from completions(self, key=key)

    monkeypatch.setattr(ExtractionStore, "completions", observe_scan)
    report = recover_completions(library, exclude_keys={excluded.key})

    assert scans == [None]
    assert report == {"recovered": 1, "blocked": [{
        "completion_id": retained[broken.key]["completion_id"],
        "reason": "invalid retained evidence",
    }]}
    assert healthy.extraction_completion_id == retained[healthy.key]["completion_id"]
    assert library.md_path(healthy.key).read_bytes() == b"# Retained output"
    assert excluded.extraction_completion_id is None
    assert broken.extraction_completion_id is None
    assert not library.md_path(excluded.key).exists()
