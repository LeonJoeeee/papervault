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
    optionally raises a preconfigured exception. Lets the routing tests drive every branch
    (success / AlreadyIngested / disabled / other-error) with no graph, ledger, or LLM."""

    def __init__(self):
        self.calls: list[dict] = []
        self.raise_for: dict[str, Exception] = {}

    async def __call__(self, rag, kind, key, path, *, force=False):
        self.calls.append({"kind": kind, "key": key, "path": path, "force": force})
        exc = self.raise_for.get(key)
        if exc is not None:
            raise exc
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
        self.calls.append({"stem": stem, "n_bytes": len(pdf_bytes), "endpoints": endpoints})
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


def _drop(d, name, body="# Ch1\nalpha beta gamma", *, force=False):
    (d / name).write_text(body, encoding="utf-8")
    if force:
        (d / (name + ".force")).write_text("", encoding="utf-8")


def _drop_pdf(d, name, body=b"%PDF-1.4 fake pdf bytes", *, force=False):
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

    assert counts == {"ingested": 1, "already": 0, "failed": 1, "disabled": 0, "deferred": 0}
    assert (d / "failed" / "textbook-Bad2020.md").exists()
    assert (d / "processed" / "textbook-Good2020.md").exists()


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
    assert counts == {"ingested": 0, "already": 0, "failed": 0, "disabled": 0, "deferred": 0}
    assert (d / "README.md").exists()             # untouched


def test_missing_pending_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("KS_OPDOC_PENDING_DIR", str(tmp_path / "does-not-exist"))
    counts = _run(drain_pending(rag=object()))
    assert counts == {"ingested": 0, "already": 0, "failed": 0, "disabled": 0, "deferred": 0}


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
