"""Operator-supplied document ingest (issues #45 textbook / #47 notebook).

Pure/hermetic unit tests — NO LLM, NO graph, NO DB. The LightRAG insertion boundary is a
fake `rag`; the ledger is monkeypatched. Covers: provenance-key validation, the flag-gated
guards, heading-aware chunking (+ plain-text fallback), section/provenance assembly, and
the shared two-phase enqueue path.
"""
from __future__ import annotations

import asyncio

import pytest

from papervault.knowledge.ingest import operator_docs as od
from papervault.knowledge.ingest.operator_docs import (
    AlreadyIngestedError,
    DocSection,
    SourceDisabledError,
    build_sections,
    check_source_enabled,
    chunk_document,
    enqueue_sections,
    split_headed_markdown,
    validate_key,
)
from papervault.knowledge.ledger.store import LedgerRecord


class FakeTok:
    """Whitespace tokenizer (offline, no tiktoken) — satisfies both len(encode(x)) counting
    and the encode/decode contract chunking_by_sentence_boundary needs."""

    def encode(self, s: str) -> list[str]:
        return s.split()

    def decode(self, toks: list[str]) -> str:
        return " ".join(toks)


class FakeRag:
    def __init__(self, doc_status: str = "processed"):
        self.enqueued: dict | None = None
        self.processed = False
        self._doc_status = doc_status
        self.deleted_doc_ids: list[str] = []

    async def apipeline_enqueue_documents(self, *, input, ids, file_paths):  # noqa: A002
        self.enqueued = {"input": list(input), "ids": list(ids), "file_paths": list(file_paths)}

    async def apipeline_process_enqueue_documents(self):
        self.processed = True

    async def aget_docs_by_ids(self, ids):
        return {i: {"status": self._doc_status} for i in ids}

    async def adelete_by_doc_id(self, doc_id):
        # The #79 --force purge primitive (mirrors distill.remove_one's call). Record the id and
        # report success — a not_found doc would be fine too (purge treats both as deleted).
        self.deleted_doc_ids.append(doc_id)
        return {"status": "success"}


class DedupRag(FakeRag):
    """Fake LightRAG that REPRODUCES the 1.5 enqueue-time filename-dedup (#79 root cause).

    LightRAG's pipeline._add_content drops any 2nd+ doc_id in a batch that reuses an
    already-seen CANONICAL file_path (basename, `[hint]`-stripped) — BEFORE it ever gets a
    doc_status row. For slash-free / hint-free operator-doc keys the canonical basename == the
    raw file_path (Path(x).name == x), so we dedup on the raw file_path. A dropped doc_id
    returns NO terminal status, so the single reconcile pass sees it as never-PROCESSED and
    flips its ledger row to `error` — exactly the 38-section→1 silent loss the fix repairs."""

    async def apipeline_enqueue_documents(self, *, input, ids, file_paths):  # noqa: A002
        self.enqueued = {"input": list(input), "ids": list(ids), "file_paths": list(file_paths)}
        seen_fp: set[str] = set()
        self.landed_ids: list[str] = []
        for did, fp in zip(ids, file_paths):
            if fp in seen_fp:
                continue  # filename-dedup: dropped, no doc_status row ever written
            seen_fp.add(fp)
            self.landed_ids.append(did)

    async def aget_docs_by_ids(self, ids):
        # Only the docs that actually LANDED (survived filename-dedup) carry a terminal status.
        return {i: {"status": self._doc_status} for i in ids if i in self.landed_ids}


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
#  key validation                                                              #
# --------------------------------------------------------------------------- #

def test_validate_key_good():
    assert validate_key("textbook", "textbook:Schlickeiser2002") == ("textbook", "Schlickeiser2002")
    assert validate_key("textbook", "textbook:Xu2025e") == ("textbook", "Xu2025e")  # suffix letter
    assert validate_key("notebook", "notebook:idea23-c12") == ("notebook", "idea23-c12")
    assert validate_key("notebook", "notebook:idea-23-c12") == ("notebook", "idea-23-c12")


@pytest.mark.parametrize("kind,key", [
    ("textbook", "notebook:idea23-c12"),   # wrong prefix for kind
    ("textbook", "Schlickeiser2002"),       # missing prefix
    ("textbook", "textbook:"),              # empty body
    ("textbook", "textbook:NoYear"),        # no 4-digit year
    ("textbook", "textbook:2002"),          # no author token before year
    ("notebook", "notebook:ideaonly"),      # no <idea>-<scope> hyphen
    ("notebook", "notebook:"),              # empty body
    ("notebook", "textbook:Schlickeiser2002"),  # wrong prefix for kind
    ("bogus", "bogus:x"),                   # unknown kind
])
def test_validate_key_bad_raises(kind, key):
    with pytest.raises(ValueError):
        validate_key(kind, key)


# --------------------------------------------------------------------------- #
#  guards (flag-gated source classes)                                          #
# --------------------------------------------------------------------------- #

def test_notebook_guard_off_refuses_loudly(monkeypatch):
    monkeypatch.delenv("PAPERVAULT_PRIVATE_SOURCES", raising=False)
    with pytest.raises(SourceDisabledError) as e:
        check_source_enabled("notebook")
    msg = str(e.value)
    assert "PAPERVAULT_PRIVATE_SOURCES" in msg   # names the flag
    assert "NEVER" in msg and "UNPUBLISHED" in msg  # explains WHY (loud)


def test_notebook_guard_on_passes(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_PRIVATE_SOURCES", "1")
    check_source_enabled("notebook")  # no raise


def test_textbook_guard_off_refuses(monkeypatch):
    monkeypatch.delenv("PAPERVAULT_OPERATOR_SOURCES", raising=False)
    with pytest.raises(SourceDisabledError) as e:
        check_source_enabled("textbook")
    assert "PAPERVAULT_OPERATOR_SOURCES" in str(e.value)


def test_textbook_guard_on_passes(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "true")  # truthy variants accepted
    check_source_enabled("textbook")


def test_guard_default_off_when_flag_blank(monkeypatch):
    # empty / non-truthy value is treated as OFF (default-safe).
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "0")
    with pytest.raises(SourceDisabledError):
        check_source_enabled("textbook")


# --------------------------------------------------------------------------- #
#  heading-aware chunking                                                       #
# --------------------------------------------------------------------------- #

def test_headings_split_on_boundaries():
    md = "# A\nalpha beta gamma\n\n# B\ndelta epsilon zeta\n\n# C\neta theta iota"
    secs = split_headed_markdown(md, FakeTok(), max_tokens=6)  # each heading-block ~5 tokens
    assert len(secs) == 3
    assert secs[0].startswith("# A")
    assert secs[1].startswith("# B")
    assert secs[2].startswith("# C")


def test_small_doc_packs_into_one_section():
    md = "# A\nalpha\n\n# B\nbeta\n\n# C\ngamma"
    secs = split_headed_markdown(md, FakeTok(), max_tokens=1000)
    assert len(secs) == 1  # all heading-blocks fit under the cap → one packed section


def test_section_boundary_lands_on_a_heading():
    md = "# A\nx x x\n\n## A.1\ny y y\n\n# B\nz z z"
    secs = split_headed_markdown(md, FakeTok(), max_tokens=6)
    # every section must START at a heading line (packing never cuts inside a block).
    for s in secs:
        assert s.lstrip().startswith("#")


def test_oversized_heading_block_subdivides_via_fallback_chunker():
    big = "# H\n" + " ".join(f"word{i}." for i in range(40))  # one heading-block, ~41 tokens
    secs = split_headed_markdown(big, FakeTok(), max_tokens=8)
    assert len(secs) >= 3  # single oversized block hard-subdivided, nothing lost
    assert "".join(secs).count("word") == 40


def test_no_headings_returns_single_block_when_small():
    secs = split_headed_markdown("just some plain prose here", FakeTok(), max_tokens=1000)
    assert secs == ["just some plain prose here"]


def test_plain_text_fallback_uses_sentence_chunker():
    txt = " ".join(f"Sentence {i} sits here." for i in range(30))
    secs = chunk_document(txt, is_markdown=False, tokenizer=FakeTok(), max_tokens=8)
    assert len(secs) >= 2
    for s in secs:
        assert s == s.strip() and s


# --------------------------------------------------------------------------- #
#  section / provenance assembly                                               #
# --------------------------------------------------------------------------- #

def test_build_sections_single_uses_bare_key():
    secs = build_sections(
        "textbook", "textbook:Schlickeiser2002", "# Ch1\nsome text",
        is_markdown=True, tokenizer=FakeTok(), max_tokens=1000,
    )
    assert len(secs) == 1
    s = secs[0]
    assert s.doc_id == "textbook:Schlickeiser2002"       # no #s suffix when single
    assert s.file_path == "textbook:Schlickeiser2002"    # base key
    assert s.source_id == "Schlickeiser2002"             # ledger source_id (prefix stripped)


def test_build_sections_multi_suffixes_and_unique_file_path():
    md = "# A\nalpha beta gamma\n\n# B\ndelta epsilon zeta"
    secs = build_sections(
        "textbook", "textbook:Schlickeiser2002", md,
        is_markdown=True, tokenizer=FakeTok(), max_tokens=6,
    )
    assert len(secs) == 2
    assert [s.doc_id for s in secs] == [
        "textbook:Schlickeiser2002#s0", "textbook:Schlickeiser2002#s1",
    ]
    # file_path is UNIQUE per section (#79) — a shared file_path made LightRAG's filename-dedup
    # silently drop sections 2..N. Here file_path == doc_id (`<key>#s<N>`).
    assert [s.file_path for s in secs] == [
        "textbook:Schlickeiser2002#s0", "textbook:Schlickeiser2002#s1",
    ]
    assert len({s.file_path for s in secs}) == 2  # distinct → no dedup collision
    assert [s.source_id for s in secs] == ["Schlickeiser2002#s0", "Schlickeiser2002#s1"]
    # …but the book-level key is recoverable for query-time citation attribution.
    assert {od.strip_section_suffix(s.file_path) for s in secs} == {"textbook:Schlickeiser2002"}


def test_build_sections_notebook_provenance():
    secs = build_sections(
        "notebook", "notebook:idea23-c12", "# Cycle 12\nwe tried X and it failed",
        is_markdown=True, tokenizer=FakeTok(), max_tokens=1000,
    )
    assert secs[0].doc_id == "notebook:idea23-c12"
    assert secs[0].file_path == "notebook:idea23-c12"


def test_build_sections_bad_key_raises():
    with pytest.raises(ValueError):
        build_sections("textbook", "textbook:bad", "x", is_markdown=True, tokenizer=FakeTok())


# --------------------------------------------------------------------------- #
#  shared LightRAG insertion path                                              #
# --------------------------------------------------------------------------- #

def test_enqueue_sections_two_phase_with_provenance_keys():
    secs = [
        DocSection("textbook:X2020#s0", "textbook:X2020#s0", "X2020#s0", "body zero"),
        DocSection("textbook:X2020#s1", "textbook:X2020#s1", "X2020#s1", "body one"),
    ]
    rag = FakeRag()
    out = _run(enqueue_sections(rag, secs))
    assert out["queued"] == 2
    assert rag.processed is True                         # process phase ran
    assert rag.enqueued["ids"] == ["textbook:X2020#s0", "textbook:X2020#s1"]
    # UNIQUE per section (#79) — distinct file_paths so LightRAG's filename-dedup keeps both.
    assert rag.enqueued["file_paths"] == ["textbook:X2020#s0", "textbook:X2020#s1"]
    assert rag.enqueued["input"] == ["body zero", "body one"]


def test_enqueue_sections_empty_is_noop():
    rag = FakeRag()
    out = _run(enqueue_sections(rag, []))
    assert out == {"queued": 0}
    assert rag.enqueued is None and rag.processed is False


# --------------------------------------------------------------------------- #
#  ingest_document end-to-end (fake rag + monkeypatched ledger)                #
# --------------------------------------------------------------------------- #

def _patch_ledger(monkeypatch, existing=None):
    """Stub the ledger: capture upsert writes, serve `load` from a mutable `state` seeded from
    `existing` ({source_id: rec}, default empty = nothing previously ingested), and let `delete`
    mutate that state so `load` REFLECTS purges. The re-ingest push-once pre-check (#3) reads
    `load`, and the #79 --force purge deletes rows then re-checks `load`, so both must be stubbed
    for the hermetic path."""
    writes: list[tuple] = []
    state = dict(existing or {})  # source_id -> rec; delete() pops, load() returns a snapshot

    async def _fake_upsert(ingest_source, source_id, *, doc_id, status, fingerprint=None):
        writes.append((ingest_source, source_id, doc_id, status))

    async def _fake_load(ingest_source):
        return dict(state)

    async def _fake_delete(ingest_source, source_id):
        state.pop(source_id, None)

    monkeypatch.setattr(od.ledger, "upsert", _fake_upsert)
    monkeypatch.setattr(od.ledger, "load", _fake_load)
    monkeypatch.setattr(od.ledger, "delete", _fake_delete)
    return writes


def test_ingest_document_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    writes = _patch_ledger(monkeypatch)
    p = tmp_path / "book.md"
    p.write_text("# Ch1\nalpha beta gamma\n\n# Ch2\ndelta epsilon zeta", encoding="utf-8")

    rag = FakeRag(doc_status="processed")
    out = _run(od.ingest_document(
        rag, "textbook", "textbook:Schlickeiser2002", str(p),
        tokenizer=FakeTok(), max_tokens=6,
    ))
    assert out["kind"] == "textbook" and out["key"] == "textbook:Schlickeiser2002"
    assert out["sections"] >= 2
    assert out["done"] == out["sections"] and out["error"] == 0
    # ledger saw a processing row then a done row per section, all under ingest_source=textbook.
    procs = [w for w in writes if w[3] == "processing"]
    dones = [w for w in writes if w[3] == "done"]
    assert len(procs) == out["sections"]
    assert len(dones) == out["sections"]
    assert all(w[0] == "textbook" for w in writes)
    # sections went through the shared insertion path with colon-prefixed doc ids.
    assert all(i.startswith("textbook:Schlickeiser2002") for i in rag.enqueued["ids"])


def test_ingest_document_failed_doc_status_marks_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    writes = _patch_ledger(monkeypatch)
    p = tmp_path / "book.md"
    p.write_text("# Ch1\nsome content here", encoding="utf-8")

    rag = FakeRag(doc_status="failed")
    out = _run(od.ingest_document(
        rag, "textbook", "textbook:Schlickeiser2002", str(p), tokenizer=FakeTok(),
    ))
    assert out["error"] == out["sections"] and out["done"] == 0
    assert any(w[3] == "error" for w in writes)


def test_ingest_document_notebook_guard_off_refuses(monkeypatch, tmp_path):
    monkeypatch.delenv("PAPERVAULT_PRIVATE_SOURCES", raising=False)
    _patch_ledger(monkeypatch)
    p = tmp_path / "nb.md"
    p.write_text("# c12\nwe tried X", encoding="utf-8")
    with pytest.raises(SourceDisabledError):
        _run(od.ingest_document(
            FakeRag(), "notebook", "notebook:idea23-c12", str(p), tokenizer=FakeTok(),
        ))


def test_ingest_document_empty_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    _patch_ledger(monkeypatch)
    p = tmp_path / "empty.md"
    p.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(ValueError):
        _run(od.ingest_document(
            FakeRag(), "textbook", "textbook:Schlickeiser2002", str(p), tokenizer=FakeTok(),
        ))


# --------------------------------------------------------------------------- #
#  F17 invariant — enqueue/process failure rewrites processing rows → error    #
# --------------------------------------------------------------------------- #

class BoomRag(FakeRag):
    """enqueue succeeds, but the process phase raises — the class of blip (pipeline not
    init / PG-Neo4j hiccup) that usually fires AFTER the ledger `processing` write but
    BEFORE doc_status lands, orphaning the row. F17 must rewrite those rows to error."""

    async def apipeline_process_enqueue_documents(self):
        raise RuntimeError("pipeline not initialized")


def test_ingest_document_enqueue_failure_rewrites_processing_to_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    writes = _patch_ledger(monkeypatch)
    p = tmp_path / "book.md"
    p.write_text("# A\nalpha beta gamma\n\n# B\ndelta epsilon zeta", encoding="utf-8")

    out = _run(od.ingest_document(
        rag=BoomRag(), kind="textbook", key="textbook:Schlickeiser2002", path=str(p),
        tokenizer=FakeTok(), max_tokens=6,
    ))
    # every section reported as error, none done/pending; the enqueue failure did not
    # punch through to the caller (F17 batch backstop).
    assert out["error"] == out["sections"] and out["sections"] >= 2
    assert out["done"] == 0 and out["pending"] == 0
    # the just-written `processing` rows were rewritten to `error` (never orphaned).
    procs = [w for w in writes if w[3] == "processing"]
    errs = [w for w in writes if w[3] == "error"]
    assert len(procs) == out["sections"]
    assert len(errs) == out["sections"]
    assert {w[1] for w in procs} == {w[1] for w in errs}  # same source_ids


def test_reconcile_missing_doc_status_row_marks_error(monkeypatch, tmp_path):
    # stuck-guard: a section with NO terminal PROCESSED (here: no doc_status row at all —
    # an F16 enqueue-drop orphan) is flipped to error in the single reconcile pass, not
    # left processing.
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    writes = _patch_ledger(monkeypatch)

    class NoStatusRag(FakeRag):
        async def aget_docs_by_ids(self, ids):
            return {}  # nothing came back terminal

    p = tmp_path / "book.md"
    p.write_text("# A\nalpha beta gamma", encoding="utf-8")
    out = _run(od.ingest_document(
        NoStatusRag(), "textbook", "textbook:Schlickeiser2002", str(p),
        tokenizer=FakeTok(), max_tokens=1000,
    ))
    assert out["error"] == out["sections"] and out["done"] == 0 and out["pending"] == 0
    assert any(w[3] == "error" for w in writes)


# --------------------------------------------------------------------------- #
#  re-ingest safety — v1 push-once (#3)                                         #
# --------------------------------------------------------------------------- #

def test_ingest_document_already_ingested_refuses(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    # ledger already carries a section for this exact provenance key.
    _patch_ledger(monkeypatch, existing={"Schlickeiser2002": object()})
    p = tmp_path / "book.md"
    p.write_text("# A\nalpha beta gamma", encoding="utf-8")
    with pytest.raises(AlreadyIngestedError) as e:
        _run(od.ingest_document(
            FakeRag(), "textbook", "textbook:Schlickeiser2002", str(p),
            tokenizer=FakeTok(), max_tokens=1000,
        ))
    assert "push-once" in str(e.value) and "textbook:Schlickeiser2002" in str(e.value)


def test_ingest_document_already_ingested_multi_section_refuses(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    # prior push landed as multiple `#s<N>` section rows — still counts as ingested.
    _patch_ledger(monkeypatch, existing={
        "Schlickeiser2002#s0": object(), "Schlickeiser2002#s1": object(),
    })
    p = tmp_path / "book.md"
    p.write_text("# A\nalpha beta gamma", encoding="utf-8")
    with pytest.raises(AlreadyIngestedError):
        _run(od.ingest_document(
            FakeRag(), "textbook", "textbook:Schlickeiser2002", str(p), tokenizer=FakeTok(),
        ))


def test_ingest_document_sibling_key_not_treated_as_already_ingested(monkeypatch, tmp_path):
    # prefix-safety: a DIFFERENT key that merely shares a prefix must NOT trip push-once.
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    _patch_ledger(monkeypatch, existing={"Schlickeiser2002b": object()})  # sibling key
    p = tmp_path / "book.md"
    p.write_text("# A\nalpha beta gamma", encoding="utf-8")
    out = _run(od.ingest_document(  # no raise — Schlickeiser2002 != Schlickeiser2002b
        FakeRag(doc_status="processed"), "textbook", "textbook:Schlickeiser2002", str(p),
        tokenizer=FakeTok(), max_tokens=1000,
    ))
    assert out["done"] == out["sections"]


# --------------------------------------------------------------------------- #
#  #79 — multi-section survives LightRAG filename-dedup (the invisible bug)     #
# --------------------------------------------------------------------------- #

def test_multi_section_survives_lightrag_filename_dedup(monkeypatch, tmp_path):
    """#79 REGRESSION: a multi-section doc must not lose sections to LightRAG's filename-dedup.

    FAILS against the OLD shared-file_path code: sections 1..N reuse section 0's file_path, so
    LightRAG drops them at enqueue (no doc_status) → reconcile flips them to `error` → done=1,
    error=N-1 (the observed 38-section→1 silent loss). PASSES after the fix: each section gets a
    UNIQUE file_path → all N land → done=N, error=0. `DedupRag` simulates the exact enqueue-time
    filename-dedup (drops a doc_id reusing an already-seen file_path)."""
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    writes = _patch_ledger(monkeypatch)
    # 6 small heading-blocks, cap=6 forces each into its own section → 6 sections.
    md = "\n\n".join(f"# H{i}\nalpha beta" for i in range(6))
    p = tmp_path / "book.md"
    p.write_text(md, encoding="utf-8")

    rag = DedupRag(doc_status="processed")
    out = _run(od.ingest_document(
        rag, "textbook", "textbook:Schlickeiser2002", str(p),
        tokenizer=FakeTok(), max_tokens=6,
    ))
    assert out["sections"] == 6                       # the doc really did split into 6 sections
    assert out["done"] == out["sections"]             # ALL survived the dedup (was 1 pre-fix)
    assert out["error"] == 0
    # every enqueued section carried a DISTINCT file_path — the fix that defeats the dedup.
    assert len(set(rag.enqueued["file_paths"])) == out["sections"]
    # ledger recorded a done row per section (none wedged in error).
    dones = [w for w in writes if w[3] == "done"]
    assert len(dones) == out["sections"]


# --------------------------------------------------------------------------- #
#  #79 — --force purges a stuck/existing key then re-ingests                    #
# --------------------------------------------------------------------------- #

def _rec(doc_id: str, source_id: str) -> LedgerRecord:
    return LedgerRecord(
        workspace="test", ingest_source="textbook", source_id=source_id,
        fingerprint="fp", doc_id=doc_id, status="error",
    )


def test_ingest_document_force_purges_then_reingests(monkeypatch, tmp_path):
    """#79 --force: an already-ingested (here half-committed/errored) key is PURGED — its
    landed graph docs (adelete_by_doc_id) + ledger rows (ledger.delete) removed — then
    re-ingested clean, instead of being refused forever by the push-once guard."""
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    # a prior push left 2 section rows for this key (e.g. the pre-fix wedge: s0 done, s1 error).
    existing = {
        "Schlickeiser2002#s0": _rec("textbook:Schlickeiser2002#s0", "Schlickeiser2002#s0"),
        "Schlickeiser2002#s1": _rec("textbook:Schlickeiser2002#s1", "Schlickeiser2002#s1"),
    }
    writes = _patch_ledger(monkeypatch, existing=existing)
    p = tmp_path / "book.md"
    p.write_text("# A\nalpha beta\n\n# B\ndelta epsilon", encoding="utf-8")

    rag = FakeRag(doc_status="processed")
    out = _run(od.ingest_document(
        rag, "textbook", "textbook:Schlickeiser2002", str(p),
        tokenizer=FakeTok(), max_tokens=6, force=True,
    ))
    # purge deleted BOTH prior ledger rows and BOTH landed graph docs (by their stored doc_ids).
    assert out["purged"]["ledger_rows_deleted"] == 2
    assert out["purged"]["docs_deleted"] == 2
    assert out["purged"]["docs_failed"] == 0
    assert set(rag.deleted_doc_ids) == {
        "textbook:Schlickeiser2002#s0", "textbook:Schlickeiser2002#s1",
    }
    # …and the re-ingest then SUCCEEDED (push-once did not fire — purge cleared the rows).
    assert out["done"] == out["sections"] and out["sections"] >= 2 and out["error"] == 0
    # the re-ingest wrote fresh processing rows (post-purge).
    assert any(w[3] == "processing" for w in writes)


def test_force_purge_of_absent_key_is_noop_then_ingests(monkeypatch, tmp_path):
    """--force on a key that was never ingested is a clean no-op purge (0 rows), then a normal
    first ingest — --force must be safe to pass unconditionally."""
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    _patch_ledger(monkeypatch)  # empty ledger
    p = tmp_path / "book.md"
    p.write_text("# A\nalpha beta\n\n# B\ndelta epsilon", encoding="utf-8")

    rag = FakeRag(doc_status="processed")
    out = _run(od.ingest_document(
        rag, "textbook", "textbook:Schlickeiser2002", str(p),
        tokenizer=FakeTok(), max_tokens=6, force=True,
    ))
    assert out["purged"]["ledger_rows_deleted"] == 0
    assert rag.deleted_doc_ids == []
    assert out["done"] == out["sections"] and out["error"] == 0


# --------------------------------------------------------------------------- #
#  synth credibility label for operator-doc colon keys (#4)                    #
# --------------------------------------------------------------------------- #

def test_source_label_operator_colon_keys():
    from papervault.knowledge.query.synth import _CRED_BY_SOURCE, _source_label

    # notebook is banded (preliminary) and the WHOLE colon key is the label — not 'unknown'.
    assert "notebook" in _CRED_BY_SOURCE
    assert _source_label("notebook:idea23-c12") == ("notebook:idea23-c12", "preliminary")
    assert _source_label("textbook:Schlickeiser2002") == ("textbook:Schlickeiser2002", "established")
    assert _source_label("web:nasa-srag") == ("web:nasa-srag", "preliminary")
    # legacy paper slash-form still resolves to the bare key + empirical.
    assert _source_label("paper/Reames2023") == ("Reames2023", "empirical")
    # an unknown colon source is not banded → falls back to unknown/preliminary.
    assert _source_label("bogus:x") == ("unknown", "preliminary")
    # #79: a per-section file_path (`<key>#s<N>`) attributes to the WHOLE book (suffix stripped),
    # so all sections of one book cite as one key; the credibility band is unchanged.
    assert _source_label("textbook:Schlickeiser2002#s0") == ("textbook:Schlickeiser2002", "established")
    assert _source_label("textbook:Schlickeiser2002#s37") == ("textbook:Schlickeiser2002", "established")
    assert _source_label("notebook:idea23-c12#s2") == ("notebook:idea23-c12", "preliminary")


def test_strip_section_suffix():
    # multi-section file_path → book-level key; bare/paper keys unchanged; only a trailing #s<N>.
    assert od.strip_section_suffix("textbook:Schlickeiser2002#s0") == "textbook:Schlickeiser2002"
    assert od.strip_section_suffix("textbook:Schlickeiser2002#s37") == "textbook:Schlickeiser2002"
    assert od.strip_section_suffix("textbook:Schlickeiser2002") == "textbook:Schlickeiser2002"
    assert od.strip_section_suffix("paper/Reames2023") == "paper/Reames2023"
    assert od.strip_section_suffix("notebook:idea-c12") == "notebook:idea-c12"
    # not a section suffix — a mid-string #s or a non-numeric tail is left intact.
    assert od.strip_section_suffix("textbook:X2020#section") == "textbook:X2020#section"
