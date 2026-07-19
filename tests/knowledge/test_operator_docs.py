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
    DocSection,
    SourceDisabledError,
    build_sections,
    check_source_enabled,
    chunk_document,
    enqueue_sections,
    split_headed_markdown,
    validate_key,
)


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

    async def apipeline_enqueue_documents(self, *, input, ids, file_paths):  # noqa: A002
        self.enqueued = {"input": list(input), "ids": list(ids), "file_paths": list(file_paths)}

    async def apipeline_process_enqueue_documents(self):
        self.processed = True

    async def aget_docs_by_ids(self, ids):
        return {i: {"status": self._doc_status} for i in ids}


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


def test_build_sections_multi_suffixes_and_shares_file_path():
    md = "# A\nalpha beta gamma\n\n# B\ndelta epsilon zeta"
    secs = build_sections(
        "textbook", "textbook:Schlickeiser2002", md,
        is_markdown=True, tokenizer=FakeTok(), max_tokens=6,
    )
    assert len(secs) == 2
    assert [s.doc_id for s in secs] == [
        "textbook:Schlickeiser2002#s0", "textbook:Schlickeiser2002#s1",
    ]
    # file_path is the SHARED base key → the query path cites the whole book, not a section.
    assert all(s.file_path == "textbook:Schlickeiser2002" for s in secs)
    assert [s.source_id for s in secs] == ["Schlickeiser2002#s0", "Schlickeiser2002#s1"]


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
        DocSection("textbook:X2020#s0", "textbook:X2020", "X2020#s0", "body zero"),
        DocSection("textbook:X2020#s1", "textbook:X2020", "X2020#s1", "body one"),
    ]
    rag = FakeRag()
    out = _run(enqueue_sections(rag, secs))
    assert out["queued"] == 2
    assert rag.processed is True                         # process phase ran
    assert rag.enqueued["ids"] == ["textbook:X2020#s0", "textbook:X2020#s1"]
    assert rag.enqueued["file_paths"] == ["textbook:X2020", "textbook:X2020"]  # shared
    assert rag.enqueued["input"] == ["body zero", "body one"]


def test_enqueue_sections_empty_is_noop():
    rag = FakeRag()
    out = _run(enqueue_sections(rag, []))
    assert out == {"queued": 0}
    assert rag.enqueued is None and rag.processed is False


# --------------------------------------------------------------------------- #
#  ingest_document end-to-end (fake rag + monkeypatched ledger)                #
# --------------------------------------------------------------------------- #

def _patch_ledger(monkeypatch):
    writes: list[tuple] = []

    async def _fake_upsert(ingest_source, source_id, *, doc_id, status, fingerprint=None):
        writes.append((ingest_source, source_id, doc_id, status))

    monkeypatch.setattr(od.ledger, "upsert", _fake_upsert)
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
