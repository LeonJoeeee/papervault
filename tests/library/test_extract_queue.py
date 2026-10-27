"""Unit tests for ``papervault.library.services.extract_queue.ExtractQueue``.

Stubs the (now async) ``extract_md`` coroutine to avoid a real MinerU server
call. Verifies recovery / chaining / idempotence / failure paths. The pypdf
``extract_txt`` foreground pass is gone (2026-06-06 MinerU migration, SDD §3.4),
so it is no longer stubbed; the worker awaits ``extract_md`` directly under a
``concurrency.acquire_extract_slot`` permit (SDD §4.1).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from papervault.library.services import extract_queue as eq_mod
from papervault.library.services.extract_queue import ExtractQueue
from papervault.library.store import Library


class _Recorder:
    """Test double for downstream queue.add — captures calls."""

    def __init__(self):
        self.added: list[str] = []

    def __call__(self, key: str) -> None:
        self.added.append(key)


def _write_pdf(lib: Library, key: str) -> None:
    p = lib.pdf_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-1.4\nfake")


def _write_md(lib: Library, key: str) -> None:
    p = lib.md_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# fake\n", encoding="utf-8")


@pytest.fixture
def lib_four_papers(tmp_path: Path) -> tuple[Library, str, str, str, str]:
    """Four papers covering all recovery branches (D7: classify() routing):
       - paper_a: status=ok, has PDF, no md → classify EXTRACT → eligible
       - paper_b: status=ok, has PDF AND md → classify TERMINAL → skipped (done)
       - paper_c: status=pending → classify DOWNLOAD → skipped (not downloaded)
       - paper_d: status=extract_failed → classify TERMINAL → skipped (terminal)
    """
    lib = Library(tmp_path)
    pa, _ = lib.upsert({
        "title": "Paper A ok no-md ready for extract queue recovery test",
        "authors": ["AOne"], "year": 2020,
    })
    pb, _ = lib.upsert({
        "title": "Paper B ok and has md already skip recovery test",
        "authors": ["BTwo"], "year": 2021,
    })
    pc, _ = lib.upsert({
        "title": "Paper C still pending download skip recovery test",
        "authors": ["CThree"], "year": 2022,
    })
    pd, _ = lib.upsert({
        "title": "Paper D extract low quality terminal skip recovery test",
        "authors": ["DFour"], "year": 2023,
    })
    pa.download_status = "ok"
    pa.download_source = "fake"
    _write_pdf(lib, pa.key)
    pb.download_status = "ok"
    pb.download_source = "fake"
    _write_pdf(lib, pb.key)
    _write_md(lib, pb.key)
    # pc keeps default "pending"
    pd.download_status = "extract_failed"
    _write_pdf(lib, pd.key)
    lib.save()
    return lib, pa.key, pb.key, pc.key, pd.key


def test_start_recovery_enqueues_only_eligible(lib_four_papers, monkeypatch):
    """Only paper_a (ok + has PDF + no md) gets enqueued by recovery."""
    lib, key_a, *_ = lib_four_papers

    async def fake_md(*a, **kw):
        return None
    monkeypatch.setattr(eq_mod, "extract_md", fake_md)

    async def run():
        q = ExtractQueue(lib, num_workers=0)
        await q.start()
        try:
            return q.qsize()
        finally:
            await q.stop()

    n = asyncio.run(run())
    assert n == 1


@pytest.mark.parametrize("pending_identity", [False, True])
async def test_start_recovers_once_and_scans_off_loop(lib_four_papers, monkeypatch, pending_identity):
    import threading
    from papervault.library import extract
    from papervault.library.services import extraction_registration as registration

    lib, key_a, key_b, *_ = lib_four_papers
    lib.get(key_b).verification_deferred_to_ocr = pending_identity
    loop_thread = threading.get_ident()
    calls, scan_threads, identity_threads, classify_threads = [], [], [], []
    recover = registration.recover_completions
    all_papers = lib.all_papers
    identity = eq_mod.pending_ocr_identity
    classify = eq_mod.classify

    def observe_recovery(library, **kwargs):
        calls.append(kwargs)
        assert threading.get_ident() != loop_thread
        return recover(library, **kwargs)

    def observe_scan(*args, **kwargs):
        scan_threads.append(threading.get_ident())
        return all_papers(*args, **kwargs)

    def observe_identity(*args):
        identity_threads.append(threading.get_ident())
        return identity(*args)

    def observe_classify(*args):
        classify_threads.append(threading.get_ident())
        return classify(*args)

    monkeypatch.setattr(registration, "recover_completions", observe_recovery)
    monkeypatch.setattr(lib, "all_papers", observe_scan)
    monkeypatch.setattr(eq_mod, "pending_ocr_identity", observe_identity)
    monkeypatch.setattr(eq_mod, "classify", observe_classify)
    monkeypatch.setattr(extract, "pdf_probe", lambda *_: extract.PDFProbe(False, 2))
    queue = ExtractQueue(lib)
    monkeypatch.setattr(queue, "_target_worker_count", lambda: 0)
    try:
        await queue.start()
        await queue.start()  # Startup remains idempotent.
        assert calls == [{"exclude_keys": {key_b} if pending_identity else None}]
        for observed in (scan_threads, identity_threads, classify_threads):
            assert observed and loop_thread not in observed
        assert set(queue._pending) == ({key_a, key_b} if pending_identity else {key_a})
    finally:
        await queue.stop()


def test_add_processes_one_then_chains_insight(lib_four_papers, monkeypatch):
    """A successful extract should call on_success (insight chain)."""
    lib, key_a, *_ = lib_four_papers
    chain = _Recorder()

    async def fake_md(paper, library, **kw):
        _write_md(library, paper.key)
        paper.md_engine = "stub"
        return "# fake\n"

    monkeypatch.setattr(eq_mod, "extract_md", fake_md)

    async def run():
        q = ExtractQueue(lib, num_workers=0, on_success=chain)
        await q._process_one(key_a)

    asyncio.run(run())
    assert chain.added == [key_a]


def test_process_one_skips_when_has_md(lib_four_papers, monkeypatch):
    """A paper that already has md should bypass extract and just forward."""
    lib, _, key_b, _, _ = lib_four_papers
    called = []

    async def fake_md(paper, library, **kw):
        called.append(("md", paper.key))
        return None
    monkeypatch.setattr(eq_mod, "extract_md", fake_md)
    chain = _Recorder()

    async def run():
        q = ExtractQueue(lib, num_workers=0, on_success=chain)
        await q._process_one(key_b)

    asyncio.run(run())
    assert called == []           # extract_md never called (md already on disk)
    assert chain.added == [key_b]  # but downstream still notified


@pytest.mark.parametrize("marker", ["deferred", "source", "evidence"])
@pytest.mark.parametrize("match", [True, False])
def test_existing_md_pending_scan_reaches_ocr_verification(
        lib_four_papers, monkeypatch, marker, match):
    import hashlib
    from papervault.library import extract
    from papervault.library.models import ADSDocumentEvidence

    lib, _, key, *_ = lib_four_papers
    paper = lib.get(key)
    previous = lib.md_path(key).read_bytes()
    if marker == "deferred":
        paper.verification_deferred_to_ocr = True
    elif marker == "source":
        paper.download_source = "ads_scan"
    else:
        paper.ads_availability.documents = [ADSDocumentEvidence(
            outcome="pdf", document_kind="scanned_full_text",
            sha256=hashlib.sha256(lib.pdf_path(key).read_bytes()).hexdigest())]
    body = "# Verified research paper\n" + "Readable scientific prose. " * 40

    async def ocr(*args, **kwargs):
        return body

    class IdentityLLM:
        def call(self, messages):
            return '{"match": ' + str(match).lower() + ', "reason": "title and author"}'

    real_extract = extract.extract_md

    async def run_extract(paper, library):
        return await real_extract(paper, library, llm=IdentityLLM())

    monkeypatch.setattr(eq_mod, "extract_md", run_extract)
    monkeypatch.setattr(extract, "extract_mineru", ocr)
    monkeypatch.setattr(extract, "pdf_probe", lambda *args: extract.PDFProbe(False, 10, "ok"))
    monkeypatch.setattr(extract, "review_extract", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(extract, "confirmed_completeness_gate",
                        lambda *args, **kwargs: {"complete": True})
    chain = _Recorder()
    asyncio.run(ExtractQueue(lib, on_success=chain)._process_one(key))
    if match:
        assert paper.extraction_completion_id
        assert body in lib.md_path(key).read_text()
        assert chain.added == [key]
    else:
        assert paper.domain_status == "identity_mismatch"
        assert paper.extraction_completion_id is None
        assert lib.md_path(key).read_bytes() == previous
        assert chain.added == []


@pytest.mark.parametrize("corrupt_other_completion", [False, True])
def test_start_recovery_cannot_clear_pending_scan_verification(
        lib_four_papers, monkeypatch, corrupt_other_completion):
    from papervault.library.services.extraction_registration import begin_extraction
    from papervault.store.extraction import ExtractionStore

    lib, other_key, key, *_ = lib_four_papers
    paper = lib.get(key)
    paper.download_source = "ads_scan"
    previous = lib.md_path(key).read_bytes()
    request = begin_extraction(paper, lib, lib.pdf_path(key).read_bytes())
    ExtractionStore(lib.root).finalize(request, b"Unverified retained output")
    if corrupt_other_completion:
        other = lib.get(other_key)
        store = ExtractionStore(lib.root)
        other_request = begin_extraction(other, lib, lib.pdf_path(other_key).read_bytes())
        completion = store.finalize(other_request, b"Initially valid output")
        store.path(completion["output_inventory"][0]["path"]).write_bytes(b"Corrupted output")
    q = ExtractQueue(lib)
    monkeypatch.setattr(q, "_target_worker_count", lambda: 0)

    async def run():
        await q.start()
        try:
            assert key in q._pending
            assert paper.extraction_completion_id is None
            assert lib.md_path(key).read_bytes() == previous
        finally:
            await q.stop()

    asyncio.run(run())


def test_extract_failure_does_not_chain_insight(lib_four_papers, monkeypatch):
    """If extract_md doesn't produce md on disk (e.g., review_low_quality),
    we must NOT chain to insight."""
    lib, key_a, *_ = lib_four_papers
    # extract_md "runs" but produces no md (simulates a gate/clarity reject).
    async def fake_md(*a, **kw):
        return None
    monkeypatch.setattr(eq_mod, "extract_md", fake_md)
    chain = _Recorder()

    async def run():
        q = ExtractQueue(lib, num_workers=0, on_success=chain)
        await q._process_one(key_a)

    asyncio.run(run())
    assert chain.added == []  # no md on disk → no insight chain


def test_stop_cancels_workers_clean(lib_four_papers, monkeypatch):
    lib, *_ = lib_four_papers

    async def fake_md(*a, **kw):
        return None
    monkeypatch.setattr(eq_mod, "extract_md", fake_md)

    async def run():
        q = ExtractQueue(lib, num_workers=2)
        await q.start()
        await asyncio.sleep(0.05)
        await q.stop()
        return len(q._workers)

    n = asyncio.run(run())
    assert n == 0
