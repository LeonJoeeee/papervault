"""Boundary-aware chunker tests (ingest.chunking).

Two layers: (1) fast synthetic unit tests for the contract + edge cases using a real
tiktoken tokenizer (the SAME one LightRAG uses, gpt-4o-mini); (2) a read-only check
against ~3 real vault extracts asserting the SIZE is unchanged (~<=1200) and NO chunk
ends mid-sentence (this is pure hygiene — only the cut location moves).
"""
from pathlib import Path

import pytest

from papervault.knowledge.ingest.chunking import (
    chunking_by_sentence_boundary,
    split_sentences,
)

CTS, OVL = 1200, 200
TERMINALS = '.!?"\')]'

_VAULT_MD = Path("/data/paper-vault/extracts/md")
_REAL_PAPERS = ["Parker1958.md", "Adriani2011.md", "Lagaris1997.md"]


@pytest.fixture(scope="module")
def tok():
    from lightrag.utils import TiktokenTokenizer

    return TiktokenTokenizer("gpt-4o-mini")


# --- contract / shape -------------------------------------------------------


def test_return_shape_and_order(tok):
    text = " ".join(f"Sentence number {i} has some words." for i in range(400))
    chunks = chunking_by_sentence_boundary(tok, text, None, False, OVL, CTS)
    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert set(c) >= {"tokens", "content", "chunk_order_index"}
        assert c["chunk_order_index"] == idx
        assert c["tokens"] == len(tok.encode(c["content"]))  # reported == real
        assert c["content"] == c["content"].strip()


def test_empty_and_whitespace_yield_no_chunks(tok):
    assert chunking_by_sentence_boundary(tok, "", None, False, OVL, CTS) == []
    assert chunking_by_sentence_boundary(tok, "   \n\n  \t ", None, False, OVL, CTS) == []


def test_size_never_exceeds_target(tok):
    text = " ".join(f"This is sentence {i} with several filler words here." for i in range(600))
    chunks = chunking_by_sentence_boundary(tok, text, None, False, OVL, CTS)
    for c in chunks:
        assert c["tokens"] <= CTS, f"chunk {c['chunk_order_index']} = {c['tokens']} > {CTS}"


def test_no_chunk_ends_mid_sentence(tok):
    text = " ".join(f"Alpha beta gamma delta sentence {i} ends now." for i in range(600))
    chunks = chunking_by_sentence_boundary(tok, text, None, False, OVL, CTS)
    for c in chunks:
        assert c["content"].rstrip()[-1:] in TERMINALS


def test_overlap_is_sentence_aligned(tok):
    # Distinct sentences so we can detect a carried whole sentence at the next chunk head.
    text = " ".join(f"Unique marker {i} appears in this distinct sentence." for i in range(600))
    chunks = chunking_by_sentence_boundary(tok, text, None, False, OVL, CTS)
    assert len(chunks) >= 2
    # The head of chunk 1 should be a whole sentence that also appears at the tail of chunk 0.
    head_sent = split_sentences(chunks[1]["content"])[0]
    assert head_sent in chunks[0]["content"], "overlap should carry a whole trailing sentence"


def test_oversized_single_sentence_is_hard_split_not_dropped(tok):
    # One sentence well over CTS tokens (no terminal punctuation inside it).
    giant = "word " * 4000  # ~4000 tokens, single "sentence"
    chunks = chunking_by_sentence_boundary(tok, giant.strip() + ".", None, False, OVL, CTS)
    assert len(chunks) >= 2  # had to be split
    for c in chunks:
        assert c["tokens"] <= CTS
    # nothing lost: total decoded tokens cover the input (overlap may add, never subtract)
    assert sum(c["tokens"] for c in chunks) >= len(tok.encode(giant))


def test_does_not_split_on_decimals_or_abbreviations(tok):
    text = "The value is 0.5 and the ratio is 3.14 here. See Eq. 2 and Fig. 1 et al. for details. Done."
    sents = split_sentences(text)
    # "0.5", "3.14", "Eq.", "Fig.", "et al." must NOT create extra sentences.
    assert len(sents) == 3, sents
    assert "0.5" in sents[0] and "3.14" in sents[0]
    assert "Eq. 2" in sents[1] and "Fig. 1" in sents[1] and "et al." in sents[1]


# --- real vault (read-only; skipped if vault absent) ------------------------

_present = [p for p in _REAL_PAPERS if (_VAULT_MD / p).exists()]
pytestmark_real = pytest.mark.skipif(not _present, reason="paper-vault extracts not present")


@pytest.mark.skipif(not _present, reason="paper-vault extracts not present")
@pytest.mark.parametrize("name", _present)
def test_real_paper_size_and_boundaries(tok, name):
    text = (_VAULT_MD / name).read_text(encoding="utf-8", errors="replace")
    chunks = chunking_by_sentence_boundary(tok, text, None, False, OVL, CTS)
    assert chunks, f"{name} produced no chunks"

    src_sents = split_sentences(text)
    for c in chunks:
        # size unchanged (<= target)
        assert len(tok.encode(c["content"])) <= CTS, f"{name} chunk over {CTS}"
        body = c["content"].rstrip()
        # clean boundary: terminal punctuation OR ends on a whole source sentence
        # (a sentence flowing into a display-math block has no terminal '.')
        clean = body[-1:] in TERMINALS or any(body.endswith(s) for s in src_sents)
        assert clean, f"{name} chunk {c['chunk_order_index']} cut mid-sentence: ...{body[-60:]!r}"


def test_preserves_paragraphs_headings_and_markdown_tables(tok):
    table = "\n".join([
        "| Mission | Energy | Flux |",
        "| --- | --- | --- |",
        "| Voyager 1 | 10 MeV | 1.2e-3 |",
        "| ACE | 20 MeV | 4.5e-4 |",
    ])
    doc = (
        "# Introduction\n\n"
        "First paragraph explains cosmic rays. It has two sentences.\n\n"
        "## Observations\n\n"
        f"{table}\n\n"
        "Concluding paragraph summarizes the measurements."
    )
    chunks = chunking_by_sentence_boundary(tok, doc, None, False, 30, 500)
    assert len(chunks) == 1
    content = chunks[0]["content"]
    assert "# Introduction\n\n" in content
    assert "## Observations\n\n" in content
    assert table in content

    # Oversized markdown table splits on row boundaries and repeats header+separator on every chunk.
    big_rows = ["| ColA | ColB | ColC |", "| --- | --- | --- |"] + [
        f"| Row{i} alpha beta | value_{i} gamma delta | note_{i} epsilon |" for i in range(60)
    ]
    big_table = "\n".join(big_rows)
    table_chunks = chunking_by_sentence_boundary(tok, big_table, None, False, 10, 120)
    assert len(table_chunks) >= 3
    seen_rows = set()
    for c in table_chunks:
        assert c["tokens"] <= 120
        assert c["tokens"] == len(tok.encode(c["content"]))
        lines = c["content"].splitlines()
        assert lines[0] == "| ColA | ColB | ColC |"
        assert lines[1] == "| --- | --- | --- |"
        assert len(lines) >= 3
        for line in lines:
            assert line.startswith("|") and line.endswith("|")
        seen_rows.update(lines[2:])
    assert len(seen_rows) == 60


def test_overlap_carry_trimmed_so_large_next_unit_never_exceeds_limit(tok):
    small = " ".join(f"Short sentence {i} ends cleanly." for i in range(12))
    large = " ".join(f"token{i}" for i in range(75)) + "."
    doc = small + " " + large
    chunks = chunking_by_sentence_boundary(tok, doc, None, False, 40, 90)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["tokens"] <= 90
        assert c["tokens"] == len(tok.encode(c["content"]))
    assert "token74." in chunks[-1]["content"]


def test_incremental_tokenization_stays_linear(tok):
    calls = 0

    class _CountingTok:
        def encode(self, text):
            nonlocal calls
            calls += 1
            return tok.encode(text)

        def decode(self, ids):
            return tok.decode(ids)

    text = " ".join(f"Sentence number {i} contains enough words to test scaling." for i in range(300))
    chunks = chunking_by_sentence_boundary(_CountingTok(), text, None, False, OVL, CTS)
    assert len(chunks) >= 2
    assert all(c["tokens"] <= CTS for c in chunks)
    # Linear in sentence count (~300 units + boundary verifications), nowhere near O(N^2) (~45,000).
    assert calls < 500, f"expected linear encode calls, got {calls}"
