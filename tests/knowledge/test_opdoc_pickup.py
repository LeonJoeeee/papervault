"""Windowless in-service operator-doc pickup (#81) — scheduler.opdoc_pickup.drain_pending.

Pure/hermetic unit tests — NO LLM, NO graph, NO DB, NO GPU, NO network. `ingest_document`
is a fake recorder (the routing tests) or the REAL one over a fake `rag` + monkeypatched
ledger (the end-to-end test, mirroring tests/knowledge/test_operator_docs.py's FakeRag).
Covers: filename → provenance-key derivation, `.force` sidecar → force=True, and the outcome
routing (processed/ on success + AlreadyIngested, failed/ on any other error WITHOUT
propagating, leave-in-place-log-once on a disabled source).
"""
from __future__ import annotations

import asyncio
import logging
import time

import pytest

from papervault.knowledge.ingest.operator_docs import (
    AlreadyIngestedError,
    SourceDisabledError,
)
from papervault.knowledge.scheduler import opdoc_pickup as op
from papervault.knowledge.scheduler.opdoc_pickup import drain_pending, parse_drop_name
from papervault.library.mineru_client import (
    MineruExtractionError,
    MineruTransportError,
)


def _run(coro):
    return asyncio.run(coro)


class FakeIngest:
    """Stand-in for `ingest_document` — records each call (kind/key/path/force) and, per key,
    optionally raises a preconfigured exception OR RETURNS a preconfigured done/error split.
    Lets the routing tests drive every branch (full success / partial / done==0-nothing-built /
    AlreadyIngested / disabled / other-error) with no graph, ledger, or LLM. The default return
    is a clean full success (done=1/error=0); `return_for[key]` overrides the done/error split
    for a key (the #84 branches: done=0 real failure, done>0/error>0 partial)."""

    def __init__(self):
        self.calls: list[dict] = []
        self.raise_for: dict[str, Exception] = {}
        self.return_for: dict[str, dict] = {}

    async def __call__(self, rag, kind, key, path, *, force=False):
        self.calls.append({"kind": kind, "key": key, "path": path, "force": force})
        exc = self.raise_for.get(key)
        if exc is not None:
            raise exc
        override = self.return_for.get(key)
        if override is not None:
            return {"key": key, "kind": kind, "pending": 0, **override}
        return {"key": key, "kind": kind, "sections": 1, "done": 1, "error": 0, "pending": 0}


class FakeExtract:
    """Stand-in for `mineru_client.extract_mineru` — records each call (stem / pdf byte count)
    and optionally raises a preconfigured exception. Lets the PDF-front-end tests drive OCR
    success + the transport-vs-extraction error split with NO MinerU server / GPU / network."""

    def __init__(self, md: str = "# OCR A\nalpha beta gamma", raise_exc: Exception | None = None):
        self.md = md
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def __call__(self, pdf_bytes, endpoints, *, stem="doc", **kwargs):
        # Record wall_clock_deadline too (#89): the drain must thread a bounded deadline into
        # extract_mineru so a doomed file can never stall the serial drain indefinitely.
        self.calls.append({"stem": stem, "n_bytes": len(pdf_bytes), "endpoints": endpoints,
                           "wall_clock_deadline": kwargs.get("wall_clock_deadline")})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.md


@pytest.fixture
def pending(tmp_path, monkeypatch):
    """A tmp pending dir wired via KS_OPDOC_PENDING_DIR + a fresh FakeIngest patched in."""
    d = tmp_path / "pending"
    d.mkdir()
    monkeypatch.setenv("KS_OPDOC_PENDING_DIR", str(d))
    op._disabled_logged.clear()      # process-local dedup sets — isolate cross-test
    op._mineru_down_logged.clear()
    fake = FakeIngest()
    monkeypatch.setattr(op, "ingest_document", fake)
    return d, fake


def _patch_extract(monkeypatch, extract: FakeExtract):
    """Patch the OCR front-end so PDF tests need no MinerU / GPU / network. `endpoints_from_env`
    is stubbed to a sentinel (the faked extract ignores it)."""
    monkeypatch.setattr(op, "extract_mineru", extract)
    monkeypatch.setattr(op, "endpoints_from_env", lambda: ["ep0"])


# A minimal but STRUCTURALLY-VALID one-page PDF: pypdf opens it, so the pre-OCR `pdf_probe`
# (issue #89) passes it and the drop proceeds to OCR. Used as the default `_drop_pdf` body so
# every PDF test now runs through the REAL structural probe (not a stub) before the faked OCR.
_VALID_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000052 00000 n \n0000000101 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n164\n%%EOF\n"
)
# CORRUPT: a valid `%PDF` magic + a TRUNCATED body (no xref/trailer) — the incident's exact
# shape (valid header, broken internal structure). pypdf raises → `pdf_probe` verdict `not_pdf`.
_CORRUPT_PDF = _VALID_PDF[:120]


def _drop(d, name, body="# Ch1\nalpha beta gamma", *, force=False):
    (d / name).write_text(body, encoding="utf-8")
    if force:
        (d / (name + ".force")).write_text("", encoding="utf-8")


def _drop_pdf(d, name, body=_VALID_PDF, *, force=False):
    (d / name).write_bytes(body)
    if force:
        (d / (name + ".force")).write_text("", encoding="utf-8")


# --------------------------------------------------------------------------- #
#  filename → provenance-key derivation                                        #
# --------------------------------------------------------------------------- #

def test_parse_drop_name_both_prefixes():
    assert parse_drop_name("textbook-Schlickeiser2002.md") == (
        "textbook", "textbook:Schlickeiser2002")
    # notebook keys legitimately carry hyphens — only the FIRST hyphen splits kind from body.
    assert parse_drop_name("notebook-idea23-c12.md") == ("notebook", "notebook:idea23-c12")
    assert parse_drop_name("textbook-Foo2020.markdown") == ("textbook", "textbook:Foo2020")


def test_parse_drop_name_pdf_suffix():
    # (d) PDF drops derive the SAME key as their .md twin — only drain_pending branches on suffix.
    assert parse_drop_name("textbook-Foo2020.pdf") == ("textbook", "textbook:Foo2020")
    # notebook keys carry hyphens under .pdf exactly as under .md (only the first hyphen splits).
    assert parse_drop_name("notebook-idea23-c12.pdf") == ("notebook", "notebook:idea23-c12")
    # suffix match is case-insensitive.
    assert parse_drop_name("textbook-Schlickeiser2002.PDF") == (
        "textbook", "textbook:Schlickeiser2002")
    # 'paper' is still not an operator-doc kind, .pdf or not.
    assert parse_drop_name("paper-Reames2023.pdf") is None


@pytest.mark.parametrize("name", [
    "README.md",                 # no <kind>- prefix
    "paper-Reames2023.md",       # 'paper' is not an operator-doc kind
    "notes.txt",                 # not markdown
    "textbook-Foo2020.md.force", # a .force sidecar, not a doc
    "textbook-.md",              # empty body
])
def test_parse_drop_name_rejects_non_drops(name):
    assert parse_drop_name(name) is None


# --------------------------------------------------------------------------- #
#  (a) success → ingested + moved to processed/                                #
# --------------------------------------------------------------------------- #

def test_ingests_and_moves_to_processed(pending):
    d, fake = pending
    _drop(d, "textbook-Foo2020.md")
    counts = _run(drain_pending(rag=object()))

    assert counts["ingested"] == 1
    assert fake.calls == [{
        "kind": "textbook", "key": "textbook:Foo2020",
        "path": str(d / "textbook-Foo2020.md"), "force": False,
    }]
    assert not (d / "textbook-Foo2020.md").exists()          # left pending/
    assert (d / "processed" / "textbook-Foo2020.md").exists()  # landed in processed/


# --------------------------------------------------------------------------- #
#  (b) a <file>.force sidecar routes force=True (+ sidecar also moved)          #
# --------------------------------------------------------------------------- #

def test_force_sidecar_routes_force_true(pending):
    d, fake = pending
    _drop(d, "textbook-Foo2020.md", force=True)
    _run(drain_pending(rag=object()))

    assert fake.calls[0]["force"] is True
    # both the doc AND its .force sidecar move to processed/ (so neither is re-scanned).
    assert (d / "processed" / "textbook-Foo2020.md").exists()
    assert (d / "processed" / "textbook-Foo2020.md.force").exists()
    assert not (d / "textbook-Foo2020.md.force").exists()


# --------------------------------------------------------------------------- #
#  (d) key derivation for BOTH textbook-/notebook- prefixes, end to end        #
# --------------------------------------------------------------------------- #

def test_key_derivation_both_prefixes_end_to_end(pending):
    d, fake = pending
    _drop(d, "textbook-Schlickeiser2002.md")
    _drop(d, "notebook-idea23-c12.md")
    _run(drain_pending(rag=object()))

    by_key = {c["key"]: c["kind"] for c in fake.calls}
    assert by_key == {"textbook:Schlickeiser2002": "textbook", "notebook:idea23-c12": "notebook"}
    assert (d / "processed" / "textbook-Schlickeiser2002.md").exists()
    assert (d / "processed" / "notebook-idea23-c12.md").exists()


# --------------------------------------------------------------------------- #
#  (e) AlreadyIngestedError (push-once) → moved to processed/ (it's done)       #
# --------------------------------------------------------------------------- #

def test_already_ingested_moves_to_processed(pending):
    d, fake = pending
    fake.raise_for["textbook:Foo2020"] = AlreadyIngestedError("Foo2020 push-once")
    _drop(d, "textbook-Foo2020.md")
    counts = _run(drain_pending(rag=object()))

    assert counts["already"] == 1 and counts["failed"] == 0
    assert (d / "processed" / "textbook-Foo2020.md").exists()
    assert not (d / "failed" / "textbook-Foo2020.md").exists()


# --------------------------------------------------------------------------- #
#  (c) any OTHER error → moved to failed/ and does NOT propagate                #
# --------------------------------------------------------------------------- #

def test_ingest_error_moves_to_failed_without_propagating(pending):
    d, fake = pending
    fake.raise_for["textbook:Foo2020"] = RuntimeError("enqueue blip")
    _drop(d, "textbook-Foo2020.md", force=True)

    # MUST NOT raise — a bad file cannot be allowed to crash the scheduler loop.
    counts = _run(drain_pending(rag=object()))

    assert counts["failed"] == 1
    assert (d / "failed" / "textbook-Foo2020.md").exists()        # routed to failed/
    assert (d / "failed" / "textbook-Foo2020.md.force").exists()  # sidecar follows the doc
    assert not (d / "textbook-Foo2020.md").exists()


def test_one_bad_file_does_not_block_the_rest(pending):
    d, fake = pending
    fake.raise_for["textbook:Bad2020"] = RuntimeError("boom")
    _drop(d, "textbook-Bad2020.md")
    _drop(d, "textbook-Good2020.md")
    counts = _run(drain_pending(rag=object()))

    assert counts == {
        "ingested": 1, "already": 0, "failed": 1, "partial": 0, "disabled": 0, "deferred": 0}
    assert (d / "failed" / "textbook-Bad2020.md").exists()
    assert (d / "processed" / "textbook-Good2020.md").exists()


# --------------------------------------------------------------------------- #
#  (#84) ingest_document RETURNS a done/error split — a return is NOT success   #
#        done==0 (NOTHING built) → failed/ + .reason; done>0/error>0 → partial  #
# --------------------------------------------------------------------------- #

def test_done_zero_routes_to_failed_not_silently_processed(pending):
    """#84 regression: `ingest_document` RETURNS (no exception) with done=0 — EVERY section
    failed to build (a transient build outage; textbook:Bubeck2015 = done=0/error=51 during the
    #84 embedding break). The drain MUST NOT treat this as success: route to failed/ (NOT
    processed/), count it `failed` (NOT `ingested`), and drop a `.reason` note — so the book is
    visibly failed + re-droppable, never silently marked done and lost."""
    d, fake = pending
    fake.return_for["textbook:Bubeck2015"] = {"sections": 51, "done": 0, "error": 51}
    _drop(d, "textbook-Bubeck2015.md")

    counts = _run(drain_pending(rag=object()))

    assert counts["failed"] == 1
    assert counts["ingested"] == 0 and counts["partial"] == 0
    # routed to failed/, NOT silently marked done in processed/ (the actual #84 bug).
    assert (d / "failed" / "textbook-Bubeck2015.md").exists()
    assert not (d / "processed" / "textbook-Bubeck2015.md").exists()
    assert not (d / "textbook-Bubeck2015.md").exists()
    # a `.reason` note records the done/error split so it reads as re-droppable, not a defect.
    reason = d / "failed" / "textbook-Bubeck2015.md.reason"
    assert reason.exists()
    body = reason.read_text(encoding="utf-8")
    assert "done=0" in body and "error=51" in body


def test_done_zero_reason_and_sidecar_follow_to_failed(pending):
    """A done==0 failure carries its `.force` sidecar to failed/ (like any failed move) AND the
    `.reason` note lands beside it — so a re-drop is a clean, self-documenting retry."""
    d, fake = pending
    fake.return_for["textbook:Foo2020"] = {"sections": 3, "done": 0, "error": 3}
    _drop(d, "textbook-Foo2020.md", force=True)

    counts = _run(drain_pending(rag=object()))

    assert counts["failed"] == 1
    assert (d / "failed" / "textbook-Foo2020.md").exists()
    assert (d / "failed" / "textbook-Foo2020.md.force").exists()       # sidecar follows the doc
    assert (d / "failed" / "textbook-Foo2020.md.reason").exists()      # reason note beside it
    assert not (d / "processed" / "textbook-Foo2020.md").exists()


def test_partial_ingest_moves_to_processed_with_loud_warning(pending, caplog):
    """done>0 AND error>0 — some sections landed in the graph, some errored. The landed sections
    ARE ingested (re-scanning would hit push-once), so the file moves to processed/ — but a LOUD
    warning records the split so an operator can re-drop with `.force` to rebuild the errored
    sections. Counted `partial` (disjoint from `ingested` and `failed`)."""
    d, fake = pending
    fake.return_for["textbook:Foo2020"] = {"sections": 5, "done": 3, "error": 2}
    _drop(d, "textbook-Foo2020.md")

    with caplog.at_level(logging.WARNING, logger="ks.scheduler.opdoc_pickup"):
        counts = _run(drain_pending(rag=object()))

    assert counts["partial"] == 1
    assert counts["ingested"] == 0 and counts["failed"] == 0
    assert (d / "processed" / "textbook-Foo2020.md").exists()          # landed sections are in
    assert not (d / "failed" / "textbook-Foo2020.md").exists()
    # LOUD warning naming the done/error split (so a partial is never silent).
    partial_warnings = [r for r in caplog.records if "PARTIAL" in r.getMessage()]
    assert len(partial_warnings) == 1
    assert "done=3" in partial_warnings[0].getMessage()
    assert "error=2" in partial_warnings[0].getMessage()


# --------------------------------------------------------------------------- #
#  disabled source → left in place, logged ONCE (no spam), not moved            #
# --------------------------------------------------------------------------- #

def test_disabled_source_left_in_place_logged_once(pending, caplog):
    d, fake = pending
    fake.raise_for["notebook:idea23-c12"] = SourceDisabledError("PAPERVAULT_PRIVATE_SOURCES off")
    _drop(d, "notebook-idea23-c12.md")

    with caplog.at_level(logging.WARNING, logger="ks.scheduler.opdoc_pickup"):
        _run(drain_pending(rag=object()))   # round 1
        _run(drain_pending(rag=object()))   # round 2 (file still there → re-attempted)

    # left in pending across BOTH rounds (a re-enable will pick it up); never moved.
    assert (d / "notebook-idea23-c12.md").exists()
    assert not (d / "processed" / "notebook-idea23-c12.md").exists()
    assert not (d / "failed" / "notebook-idea23-c12.md").exists()
    # ingest was attempted both rounds, but the disabled warning was logged only ONCE.
    assert len(fake.calls) == 2
    disabled_warnings = [r for r in caplog.records if "source disabled" in r.getMessage()]
    assert len(disabled_warnings) == 1


# --------------------------------------------------------------------------- #
#  non-matching files + missing dir are safe no-ops                            #
# --------------------------------------------------------------------------- #

def test_non_matching_files_ignored(pending):
    d, fake = pending
    _drop(d, "README.md")
    _drop(d, "paper-Reames2023.md")
    (d / "notes.txt").write_text("hi", encoding="utf-8")
    counts = _run(drain_pending(rag=object()))

    assert fake.calls == []                       # nothing derivable as an operator doc
    assert counts == {
        "ingested": 0, "already": 0, "failed": 0, "partial": 0, "disabled": 0, "deferred": 0}
    assert (d / "README.md").exists()             # untouched


def test_missing_pending_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("KS_OPDOC_PENDING_DIR", str(tmp_path / "does-not-exist"))
    counts = _run(drain_pending(rag=object()))
    assert counts == {
        "ingested": 0, "already": 0, "failed": 0, "partial": 0, "disabled": 0, "deferred": 0}


# --------------------------------------------------------------------------- #
#  end to end — drain_pending over the REAL ingest_document (fake rag + ledger) #
# --------------------------------------------------------------------------- #

class FakeRag:
    """Mirror of tests/knowledge/test_operator_docs.py's FakeRag — the LightRAG insertion
    boundary for the real ingest_document, so the whole pickup path (incl. the #81 to_thread
    build_sections wrap) runs with no graph/LLM/GPU."""

    def __init__(self, doc_status: str = "processed"):
        self.enqueued: dict | None = None
        self._doc_status = doc_status

    async def apipeline_enqueue_documents(self, *, input, ids, file_paths):  # noqa: A002
        self.enqueued = {"input": list(input), "ids": list(ids), "file_paths": list(file_paths)}

    async def apipeline_process_enqueue_documents(self):
        pass

    async def aget_docs_by_ids(self, ids):
        return {i: {"status": self._doc_status} for i in ids}


def test_drain_end_to_end_real_ingest_document(tmp_path, monkeypatch):
    """drain_pending → the REAL ingest_document → fake rag: proves the windowless path works
    end to end (guard → key-validate → to_thread build_sections → enqueue → reconcile → move).
    """
    from papervault.knowledge.ingest import operator_docs as od

    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")  # enable textbook source
    d = tmp_path / "pending"
    d.mkdir()
    monkeypatch.setenv("KS_OPDOC_PENDING_DIR", str(d))
    op._disabled_logged.clear()

    # Stub the ledger (capture upserts, serve an empty load so push-once passes).
    writes: list[tuple] = []

    async def _fake_upsert(ingest_source, source_id, *, doc_id, status, fingerprint=None):
        writes.append((ingest_source, source_id, doc_id, status))

    async def _fake_load(ingest_source):
        return {}

    monkeypatch.setattr(od.ledger, "upsert", _fake_upsert)
    monkeypatch.setattr(od.ledger, "load", _fake_load)

    _drop(d, "textbook-Schlickeiser2002.md", body="# A\nalpha beta gamma\n\n# B\ndelta epsilon zeta")

    rag = FakeRag(doc_status="processed")
    counts = _run(drain_pending(rag))

    assert counts["ingested"] == 1
    assert (d / "processed" / "textbook-Schlickeiser2002.md").exists()
    # the real ingest ran the shared two-phase enqueue with colon-prefixed provenance ids…
    assert rag.enqueued is not None
    assert all(i.startswith("textbook:Schlickeiser2002") for i in rag.enqueued["ids"])
    # …and the ledger saw processing→done rows written under ingest_source=textbook.
    assert any(w[3] == "processing" for w in writes)
    assert any(w[3] == "done" for w in writes)
    assert all(w[0] == "textbook" for w in writes)


# --------------------------------------------------------------------------- #
#  PDF front-end (#81 follow-up): OCR via MinerU → md → SAME ingest path        #
# --------------------------------------------------------------------------- #

def test_pdf_drop_ocr_then_ingest_end_to_end(tmp_path, monkeypatch):
    """(a) A `textbook-Foo2020.pdf` drop → patched `extract_mineru` returns markdown → the REAL
    ingest_document runs over the PRODUCED md (heading-aware, colon-prefixed ids) → the pdf
    lands in processed/ with the OCR'd md kept alongside it. No MinerU / GPU / network."""
    from papervault.knowledge.ingest import operator_docs as od

    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")  # textbook source ON (pre-OCR gate)
    d = tmp_path / "pending"
    d.mkdir()
    monkeypatch.setenv("KS_OPDOC_PENDING_DIR", str(d))
    op._disabled_logged.clear()
    op._mineru_down_logged.clear()

    writes: list[tuple] = []

    async def _fake_upsert(ingest_source, source_id, *, doc_id, status, fingerprint=None):
        writes.append((ingest_source, source_id, doc_id, status))

    async def _fake_load(ingest_source):
        return {}

    monkeypatch.setattr(od.ledger, "upsert", _fake_upsert)
    monkeypatch.setattr(od.ledger, "load", _fake_load)

    extract = FakeExtract(md="# A\nalpha beta gamma\n\n# B\ndelta epsilon zeta")
    _patch_extract(monkeypatch, extract)
    _drop_pdf(d, "textbook-Foo2020.pdf")

    rag = FakeRag(doc_status="processed")
    counts = _run(drain_pending(rag))

    assert counts["ingested"] == 1
    # extract_mineru was called with the pdf's own stem (an inherently fs-safe name).
    assert len(extract.calls) == 1
    assert extract.calls[0]["stem"] == "textbook-Foo2020"
    # the REAL ingest ran over the OCR'd markdown → colon-prefixed textbook provenance ids.
    assert rag.enqueued is not None
    assert all(i.startswith("textbook:Foo2020") for i in rag.enqueued["ids"])
    assert any(w[3] == "done" for w in writes) and all(w[0] == "textbook" for w in writes)
    # the pdf moved to processed/, and the produced md is kept beside it (not re-scannable).
    assert not (d / "textbook-Foo2020.pdf").exists()
    assert (d / "processed" / "textbook-Foo2020.pdf").exists()
    assert (d / "processed" / "textbook-Foo2020.md").exists()
    # staging dir left clean — the md was moved out of .ocr/.
    assert not (d / ".ocr" / "textbook-Foo2020.md").exists()


def test_pdf_transport_error_left_in_pending(pending, monkeypatch):
    """(b) `MineruTransportError` (MinerU down/unreachable) → the pdf is LEFT in pending to
    retry next round; it is NEVER moved to failed/ (a server outage must not condemn a good
    pdf), and ingest_document is never reached."""
    d, fake = pending
    extract = FakeExtract(raise_exc=MineruTransportError("mineru unreachable"))
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    _patch_extract(monkeypatch, extract)
    _drop_pdf(d, "textbook-Foo2020.pdf")

    counts = _run(drain_pending(rag=object()))

    assert counts["deferred"] == 1 and counts["failed"] == 0
    assert extract.calls != []          # OCR was attempted…
    assert fake.calls == []             # …but ingest was never reached
    assert (d / "textbook-Foo2020.pdf").exists()                     # LEFT in pending
    assert not (d / "failed" / "textbook-Foo2020.pdf").exists()      # NOT condemned
    assert not (d / "processed" / "textbook-Foo2020.pdf").exists()


def test_pdf_transport_error_logs_once_across_rounds(pending, monkeypatch, caplog):
    """The MinerU-down warning is logged ONCE while the outage persists (SourceDisabled
    pattern), not once per round — so a persistent outage never spams the scheduler log."""
    d, fake = pending
    extract = FakeExtract(raise_exc=MineruTransportError("mineru unreachable"))
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    _patch_extract(monkeypatch, extract)
    _drop_pdf(d, "textbook-Foo2020.pdf")

    with caplog.at_level(logging.WARNING, logger="ks.scheduler.opdoc_pickup"):
        _run(drain_pending(rag=object()))   # round 1
        _run(drain_pending(rag=object()))   # round 2 (pdf still there → re-attempted)

    assert len(extract.calls) == 2          # OCR retried both rounds
    assert (d / "textbook-Foo2020.pdf").exists()
    down_warnings = [r for r in caplog.records if "MinerU unreachable" in r.getMessage()]
    assert len(down_warnings) == 1          # logged only ONCE


def test_pdf_extraction_error_moves_to_failed(pending, monkeypatch):
    """(c) `MineruExtractionError` (per-doc OCR defect) → the pdf is moved to failed/ (LOUD),
    and ingest_document is never reached."""
    d, fake = pending
    extract = FakeExtract(raise_exc=MineruExtractionError("thin_or_missing_md"))
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    _patch_extract(monkeypatch, extract)
    _drop_pdf(d, "textbook-Foo2020.pdf", force=True)

    counts = _run(drain_pending(rag=object()))

    assert counts["failed"] == 1
    assert fake.calls == []             # ingest never reached — OCR failed first
    assert (d / "failed" / "textbook-Foo2020.pdf").exists()          # routed to failed/
    assert (d / "failed" / "textbook-Foo2020.pdf.force").exists()    # sidecar follows the doc
    assert not (d / "textbook-Foo2020.pdf").exists()


def test_pdf_disabled_source_skips_ocr_and_left_in_place(pending, monkeypatch):
    """A PDF for a DISABLED source class must NOT burn a MinerU/GPU OCR pass every round — the
    source is gated BEFORE OCR, so extract_mineru is never called; the pdf is left in pending
    (a re-enable picks it up), exactly like a disabled .md drop."""
    d, fake = pending
    extract = FakeExtract()
    monkeypatch.delenv("PAPERVAULT_PRIVATE_SOURCES", raising=False)  # notebook source OFF
    _patch_extract(monkeypatch, extract)
    _drop_pdf(d, "notebook-idea23-c12.pdf")

    counts = _run(drain_pending(rag=object()))

    assert counts["disabled"] == 1
    assert extract.calls == []          # OCR NEVER ran for a disabled source
    assert fake.calls == []             # ingest never reached
    assert (d / "notebook-idea23-c12.pdf").exists()                  # left in pending
    assert not (d / "failed" / "notebook-idea23-c12.pdf").exists()


# --------------------------------------------------------------------------- #
#  (#89) corrupt-PDF drain-stall guard: pre-OCR structural probe + bounded OCR  #
# --------------------------------------------------------------------------- #

def test_pdf_corrupt_rejected_pre_ocr_to_failed(pending, monkeypatch):
    """(#89) A PDF with a valid `%PDF` magic but a truncated/corrupt body is REJECTED to failed/
    by the REAL pre-OCR structural probe (`library.extract.pdf_probe`) WITHOUT ever calling OCR —
    the exact stall the drain used to hit (pdfium retry-loop for ~30 min). A `.reason` note marks
    it corrupt (re-drop of the same bytes won't help), counted `failed` + logged LOUD."""
    d, fake = pending
    extract = FakeExtract()  # records calls; MUST stay empty — OCR is never reached
    # Source ON so it is the STRUCTURAL PROBE (not the source gate) that rejects the file.
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    _patch_extract(monkeypatch, extract)
    _drop_pdf(d, "textbook-Sullivan2015.pdf", body=_CORRUPT_PDF)

    counts = _run(drain_pending(rag=object()))

    assert counts["failed"] == 1
    assert extract.calls == []          # OCR NEVER called — the whole point of #89
    assert fake.calls == []             # ingest never reached
    assert (d / "failed" / "textbook-Sullivan2015.pdf").exists()          # quarantined
    assert not (d / "textbook-Sullivan2015.pdf").exists()
    assert not (d / "processed" / "textbook-Sullivan2015.pdf").exists()
    # a `.reason` note marks it corrupt (DISTINCT from the done==0 transient-outage reason note).
    reason = d / "failed" / "textbook-Sullivan2015.pdf.reason"
    assert reason.exists()
    body = reason.read_text(encoding="utf-8")
    assert "corrupt" in body.lower() and "pre-OCR" in body


def test_pdf_corrupt_force_sidecar_follows_to_failed(pending, monkeypatch):
    """The corrupt-PDF rejection carries any `.force` sidecar to failed/ (like every failed move),
    so a re-drop is clean once the operator supplies a good copy."""
    d, fake = pending
    extract = FakeExtract()
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    _patch_extract(monkeypatch, extract)
    _drop_pdf(d, "textbook-Sullivan2015.pdf", body=_CORRUPT_PDF, force=True)

    counts = _run(drain_pending(rag=object()))

    assert counts["failed"] == 1
    assert extract.calls == []          # OCR never called
    assert (d / "failed" / "textbook-Sullivan2015.pdf").exists()
    assert (d / "failed" / "textbook-Sullivan2015.pdf.force").exists()    # sidecar follows the doc
    assert (d / "failed" / "textbook-Sullivan2015.pdf.reason").exists()   # reason note beside it


def test_pdf_corrupt_disabled_source_gated_before_probe(pending, monkeypatch):
    """A corrupt PDF for a DISABLED source is still gated by the source flag FIRST (left in
    pending, counted `disabled`) — the pre-OCR probe only runs once the source is enabled, so a
    disabled source is never even probed and the file is never condemned."""
    d, fake = pending
    extract = FakeExtract()
    monkeypatch.delenv("PAPERVAULT_PRIVATE_SOURCES", raising=False)  # notebook source OFF
    _patch_extract(monkeypatch, extract)
    _drop_pdf(d, "notebook-idea23-c12.pdf", body=_CORRUPT_PDF)

    counts = _run(drain_pending(rag=object()))

    assert counts["disabled"] == 1 and counts["failed"] == 0
    assert extract.calls == []
    assert (d / "notebook-idea23-c12.pdf").exists()                 # left in pending, not failed/
    assert not (d / "failed" / "notebook-idea23-c12.pdf").exists()


def test_pdf_valid_passes_probe_and_reaches_ocr(tmp_path, monkeypatch):
    """A VALID PDF (pypdf opens it) PASSES the real pre-OCR probe and proceeds to OCR → ingest,
    so the #89 guard never false-rejects a good drop. Uses the REAL ingest_document over a fake
    rag (no MinerU / GPU / network)."""
    from papervault.knowledge.ingest import operator_docs as od

    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    d = tmp_path / "pending"
    d.mkdir()
    monkeypatch.setenv("KS_OPDOC_PENDING_DIR", str(d))
    op._disabled_logged.clear()
    op._mineru_down_logged.clear()

    async def _fake_upsert(ingest_source, source_id, *, doc_id, status, fingerprint=None):
        pass

    async def _fake_load(ingest_source):
        return {}

    monkeypatch.setattr(od.ledger, "upsert", _fake_upsert)
    monkeypatch.setattr(od.ledger, "load", _fake_load)

    extract = FakeExtract(md="# A\nalpha beta gamma\n\n# B\ndelta epsilon zeta")
    _patch_extract(monkeypatch, extract)
    _drop_pdf(d, "textbook-Good2020.pdf", body=_VALID_PDF)

    rag = FakeRag(doc_status="processed")
    counts = _run(drain_pending(rag))

    assert counts["ingested"] == 1
    assert len(extract.calls) == 1      # a VALID pdf DID reach OCR (the probe passed it)
    assert (d / "processed" / "textbook-Good2020.pdf").exists()


def test_pdf_ocr_call_is_bounded_by_wall_clock_deadline(tmp_path, monkeypatch):
    """(#89 backstop) The drain threads a `wall_clock_deadline` into `extract_mineru`, so even a
    corruption the probe misses (or a server-side hang) costs ONE bounded period, never an
    indefinite stall. A VALID pdf reaches OCR; assert the call received a deadline ~= now +
    KS_OPDOC_OCR_TIMEOUT_SEC. (Before #89 the drain passed NO deadline — the root of the stall.)"""
    from papervault.knowledge.ingest import operator_docs as od

    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    monkeypatch.setenv("KS_OPDOC_OCR_TIMEOUT_SEC", "123")
    d = tmp_path / "pending"
    d.mkdir()
    monkeypatch.setenv("KS_OPDOC_PENDING_DIR", str(d))
    op._disabled_logged.clear()
    op._mineru_down_logged.clear()

    async def _fake_upsert(ingest_source, source_id, *, doc_id, status, fingerprint=None):
        pass

    async def _fake_load(ingest_source):
        return {}

    monkeypatch.setattr(od.ledger, "upsert", _fake_upsert)
    monkeypatch.setattr(od.ledger, "load", _fake_load)

    extract = FakeExtract(md="# A\nalpha beta gamma")
    _patch_extract(monkeypatch, extract)
    _drop_pdf(d, "textbook-Good2020.pdf", body=_VALID_PDF)

    before = time.time()
    _run(drain_pending(FakeRag(doc_status="processed")))
    after = time.time()

    assert len(extract.calls) == 1
    dl = extract.calls[0]["wall_clock_deadline"]
    assert dl is not None                       # a bound WAS passed (the #89 fix)
    assert before + 123 <= dl <= after + 123    # ~= call-time + KS_OPDOC_OCR_TIMEOUT_SEC


def test_pdf_ocr_timeout_zero_disables_deadline(tmp_path, monkeypatch):
    """KS_OPDOC_OCR_TIMEOUT_SEC<=0 DISABLES the backstop cap (deadline=None) — the escape hatch
    for a document larger than the default cap allows; the call then relies on the mineru_client's
    own http_timeout × inline-retry budget (the pre-#89 bound)."""
    from papervault.knowledge.ingest import operator_docs as od

    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    monkeypatch.setenv("KS_OPDOC_OCR_TIMEOUT_SEC", "0")
    d = tmp_path / "pending"
    d.mkdir()
    monkeypatch.setenv("KS_OPDOC_PENDING_DIR", str(d))
    op._disabled_logged.clear()
    op._mineru_down_logged.clear()

    async def _fake_upsert(ingest_source, source_id, *, doc_id, status, fingerprint=None):
        pass

    async def _fake_load(ingest_source):
        return {}

    monkeypatch.setattr(od.ledger, "upsert", _fake_upsert)
    monkeypatch.setattr(od.ledger, "load", _fake_load)

    extract = FakeExtract(md="# A\nalpha beta gamma")
    _patch_extract(monkeypatch, extract)
    _drop_pdf(d, "textbook-Good2020.pdf", body=_VALID_PDF)

    _run(drain_pending(FakeRag(doc_status="processed")))

    assert len(extract.calls) == 1
    assert extract.calls[0]["wall_clock_deadline"] is None   # backstop disabled


def test_ocr_timeout_seconds_env(monkeypatch):
    """`_ocr_timeout_seconds` reads KS_OPDOC_OCR_TIMEOUT_SEC per-call: default 30 min, honors an
    override, allows a disable (<=0), and falls back to the default on a garbage value."""
    monkeypatch.delenv("KS_OPDOC_OCR_TIMEOUT_SEC", raising=False)
    assert op._ocr_timeout_seconds() == 30 * 60.0        # default = 30 min (paper ceiling parity)
    monkeypatch.setenv("KS_OPDOC_OCR_TIMEOUT_SEC", "600")
    assert op._ocr_timeout_seconds() == 600.0
    monkeypatch.setenv("KS_OPDOC_OCR_TIMEOUT_SEC", "0")
    assert op._ocr_timeout_seconds() == 0.0
    monkeypatch.setenv("KS_OPDOC_OCR_TIMEOUT_SEC", "not-a-number")
    assert op._ocr_timeout_seconds() == 30 * 60.0        # garbage → default (never crash)
