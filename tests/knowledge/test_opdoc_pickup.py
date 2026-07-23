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


@pytest.fixture
def pending(tmp_path, monkeypatch):
    """A tmp pending dir wired via KS_OPDOC_PENDING_DIR + a fresh FakeIngest patched in."""
    d = tmp_path / "pending"
    d.mkdir()
    monkeypatch.setenv("KS_OPDOC_PENDING_DIR", str(d))
    op._disabled_logged.clear()  # process-local dedup set — isolate cross-test
    fake = FakeIngest()
    monkeypatch.setattr(op, "ingest_document", fake)
    return d, fake


def _drop(d, name, body="# Ch1\nalpha beta gamma", *, force=False):
    (d / name).write_text(body, encoding="utf-8")
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

    assert counts == {"ingested": 1, "already": 0, "failed": 1, "disabled": 0}
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
    assert counts == {"ingested": 0, "already": 0, "failed": 0, "disabled": 0}
    assert (d / "README.md").exists()             # untouched


def test_missing_pending_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("KS_OPDOC_PENDING_DIR", str(tmp_path / "does-not-exist"))
    counts = _run(drain_pending(rag=object()))
    assert counts == {"ingested": 0, "already": 0, "failed": 0, "disabled": 0}


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
