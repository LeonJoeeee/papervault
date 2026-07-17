"""Boundary-aware chunker (ingest-side hygiene, drop-in for LightRAG's chunking_func).

WHY (pure hygiene, NOT a size change): LightRAG's default `chunking_by_token_size`
slices the token stream every `chunk_token_size - overlap` tokens, so it routinely
cuts in the MIDDLE of a sentence — the entity-extraction LLM then sees half a clause
at a chunk edge. This module keeps the SAME ~2400-token target but only ever ends a
chunk on a sentence boundary, so the cut is clean. The target size is unchanged; only
the boundaries move.

Contract (must match lightrag.operate.chunking_by_token_size EXACTLY so it can be
passed as LightRAG(chunking_func=...)):
  signature  (tokenizer, content, split_by_character, split_by_character_only,
              chunk_overlap_token_size, chunk_token_size)
  return     list[dict] with keys 'tokens' (int, len of the chunk's token ids),
             'content' (str, stripped), 'chunk_order_index' (int, 0-based order).
The LightRAG pipeline (lightrag.py:~1976) then layers on full_doc_id/file_path/etc.
It uses the SAME tokenizer that LightRAG passes in for counting (no new tokenizer).

Algorithm:
  1. Split into paragraphs on blank lines, then each paragraph into sentences by a
     scientific-text-tolerant regex (won't break on "0.5", "Eq.", "Fig.", "et al.").
  2. Greedily pack whole sentences into a chunk until the NEXT sentence would push it
     past chunk_token_size, then flush. A chunk never ends mid-sentence.
  3. Carry whole trailing sentences (~chunk_overlap_token_size tokens) into the next
     chunk so the overlap is sentence-aligned too.
  4. A lone sentence longer than chunk_token_size is hard token-split (that sentence
     only) so nothing is lost; empty/whitespace content yields no chunks.
"""
from __future__ import annotations

import re
from typing import Any

from lightrag.utils import Tokenizer

# ---------------------------------------------------------------------------
# Sentence splitting (scientific-text tolerant).
# ---------------------------------------------------------------------------

# Abbreviations whose trailing period must NOT be treated as a sentence end.
# Lower-cased compare; matched on the last whitespace-delimited word of a candidate.
_ABBREVIATIONS = frozenset(
    {
        "eq", "eqs", "fig", "figs", "ref", "refs", "sec", "secs", "ch", "chap",
        "no", "nos", "vol", "vols", "pp", "p", "al", "et", "etc", "cf", "vs",
        "approx", "resp", "i.e", "e.g", "viz",
        "dr", "prof", "mr", "mrs", "ms", "st", "ca", "c",
        # unit / scientific shorthands that often precede a digit or newline
        "km", "kpc", "pc", "au", "mev", "gev", "kev", "ev", "kg", "ms", "yr",
        "min", "sec", "hr", "deg",
    }
)

# A sentence terminator = one of . ! ? (optionally repeated / with trailing quote
# or bracket), followed by whitespace, OR one-or-more blank lines (paragraph break).
# We capture the boundary so we can re-attach the punctuation to the left sentence.
_SENT_BOUNDARY = re.compile(
    r"""
    (?<=[.!?])        # a terminal punctuation char just consumed
    [\"')\]]*         # optional closing quote/paren after it
    (?=\s)            # followed by whitespace (the split point sits before it)
    """,
    re.VERBOSE,
)

# True if a candidate boundary is a false positive (decimal number or abbreviation).
_DECIMAL_BEFORE = re.compile(r"\d\.\d*$")          # "0.5" / "3." mid-number
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.\-]*)\.$")


def _is_false_boundary(left: str) -> bool:
    """Reject a '.'-boundary that is really a decimal or a known abbreviation."""
    tail = left.rstrip()
    if not tail.endswith("."):
        return False  # '!'/'?' boundaries are always real
    if _DECIMAL_BEFORE.search(tail):
        return True
    m = _LAST_WORD.search(tail)
    if m:
        word = m.group(1).rstrip(".").lower()
        # strip internal dots ("i.e" -> "i.e"), keep as-is for multi-dot abbrevs
        if word in _ABBREVIATIONS:
            return True
        # single capital letter + '.' = an initial ("J. Smith")
        if len(word) == 1 and word.isalpha():
            return True
    return False


def split_sentences(text: str) -> list[str]:
    """Split into sentences, respecting paragraph (blank-line) boundaries first.

    Paragraph breaks are hard boundaries (never merged across a blank line). Within a
    paragraph, split on terminal punctuation, skipping decimals / abbreviations.
    Returns non-empty, stripped sentence strings in order.
    """
    sentences: list[str] = []
    # Hard-split on blank lines so a chunk boundary can also fall on a paragraph break.
    paragraphs = re.split(r"\n\s*\n", text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Candidate split points inside the paragraph.
        last = 0
        for m in _SENT_BOUNDARY.finditer(para):
            cut = m.start()
            left = para[last:cut]
            if _is_false_boundary(left):
                continue
            piece = para[last:cut].strip()
            if piece:
                sentences.append(piece)
            last = cut
        tail = para[last:].strip()
        if tail:
            sentences.append(tail)
    return sentences


# ---------------------------------------------------------------------------
# Boundary-aware packing.
# ---------------------------------------------------------------------------


def _hard_split_oversized(
    tokenizer: Tokenizer, sentence: str, chunk_token_size: int, overlap: int
) -> list[tuple[int, str]]:
    """Fallback for a single sentence longer than chunk_token_size: token-slice it
    (this sentence ONLY) so nothing is lost. Mirrors LightRAG's stride."""
    toks = tokenizer.encode(sentence)
    out: list[tuple[int, str]] = []
    stride = max(1, chunk_token_size - overlap)
    for start in range(0, len(toks), stride):
        window = toks[start : start + chunk_token_size]
        out.append((len(window), tokenizer.decode(window).strip()))
        if start + chunk_token_size >= len(toks):
            break
    return out


def chunking_by_sentence_boundary(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 200,   # LightRAG always passes the instance value (config default 200) positionally; this default only ever applies to direct unit tests — keep it equal to the real config default.
    chunk_token_size: int = 2400,
) -> list[dict[str, Any]]:
    """Drop-in for lightrag.operate.chunking_by_token_size — same signature + return
    shape, but chunks end on sentence boundaries (see module docstring).

    `split_by_character` / `split_by_character_only` are accepted for signature
    compatibility but ignored: KS never sets them (LightRAG passes None), and honoring
    them would reintroduce arbitrary cuts. The sentence splitter already covers
    paragraph (double-newline) boundaries, which is the only structural cut KS wants.
    """
    if not content or not content.strip():
        return []

    sentences = split_sentences(content)
    if not sentences:
        return []

    # Pre-tokenize each sentence once.
    sent_tokens = [len(tokenizer.encode(s)) for s in sentences]

    def _joined_tokens(seq: list[str]) -> int:
        """Token count of the chunk AS EMITTED (sentences joined by a space). This is
        the authoritative size: the sum of per-sentence counts can differ from the
        joined count because tiktoken merges differently at the seams, so the packing
        limit must be checked against this, not the per-sentence sum."""
        if not seq:
            return 0
        return len(tokenizer.encode(" ".join(seq).strip()))

    results: list[dict[str, Any]] = []
    cur: list[str] = []          # sentences in the current chunk

    def _flush() -> list[str]:
        """Emit the current chunk; return the trailing sentences to carry as overlap."""
        nonlocal results
        if not cur:
            return []
        text = " ".join(cur).strip()
        tok = len(tokenizer.encode(text))
        results.append(
            {
                "tokens": tok,
                "content": text,
                "chunk_order_index": len(results),
            }
        )
        # Sentence-aligned overlap: carry whole trailing sentences up to ~overlap.
        if chunk_overlap_token_size <= 0:
            return []
        carry: list[str] = []
        acc = 0
        for s in reversed(cur):
            t = len(tokenizer.encode(s))
            if acc + t > chunk_overlap_token_size and carry:
                break
            carry.insert(0, s)
            acc += t
            if acc >= chunk_overlap_token_size:
                break
        # Never carry the ENTIRE chunk (would stall forever on a 1-sentence chunk).
        if len(carry) >= len(cur):
            carry = carry[1:]
        return carry

    i = 0
    while i < len(sentences):
        s, t = sentences[i], sent_tokens[i]

        # Over-long single sentence: flush what we have, hard-split this one, continue.
        if t > chunk_token_size:
            if cur:
                _flush()  # overlap carry is irrelevant right before a hard split
                cur = []
            for sub_tok, sub_text in _hard_split_oversized(
                tokenizer, s, chunk_token_size, chunk_overlap_token_size
            ):
                results.append(
                    {
                        "tokens": sub_tok,
                        "content": sub_text,
                        "chunk_order_index": len(results),
                    }
                )
            i += 1
            continue

        # Would adding this sentence overflow the chunk (measured on the joined text,
        # which is what we emit)? Flush first so the cut lands on a clean boundary.
        if cur and _joined_tokens(cur + [s]) > chunk_token_size:
            carry = _flush()
            cur = list(carry)
            # `s` not yet added — fall through to add it to the fresh chunk.

        cur.append(s)
        i += 1

    if cur:
        _flush()

    return results
