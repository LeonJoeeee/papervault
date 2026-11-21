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


# Markdown structural patterns preserved by _split_units.
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}[ \t]+\S")
_TABLE_SEP_LINE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_TABLE_PIPE_LINE = re.compile(r"^\s*\|.*\|\s*$")


def _split_prose_sentences(para: str) -> list[str]:
    """Split a prose block into sentences on terminal punctuation, skipping decimals/abbreviations."""
    out: list[str] = []
    last = 0
    for m in _SENT_BOUNDARY.finditer(para):
        cut = m.start()
        left = para[last:cut]
        if _is_false_boundary(left):
            continue
        piece = para[last:cut].strip()
        if piece:
            out.append(piece)
        last = cut
    tail = para[last:].strip()
    if tail:
        out.append(tail)
    return out


def _is_table_start(lines: list[str], idx: int) -> bool:
    line = lines[idx]
    if _TABLE_PIPE_LINE.match(line):
        return True
    if "|" in line and idx + 1 < len(lines) and _TABLE_SEP_LINE.match(lines[idx + 1]):
        return True
    return False


def _is_table_continuation(line: str) -> bool:
    return bool(line.strip()) and (
        bool(_TABLE_PIPE_LINE.match(line))
        or bool(_TABLE_SEP_LINE.match(line))
        or "|" in line
    )


def _is_markdown_table(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    return any(_TABLE_SEP_LINE.match(ln) for ln in lines) or all(
        _TABLE_PIPE_LINE.match(ln) for ln in lines
    )


def _split_units(text: str) -> list[tuple[str, str]]:
    """Split text into ``(unit_text, sep_before)`` pairs preserving structure.

    - Paragraph breaks (blank lines) carry ``sep_before="\\n\\n"``.
    - Markdown ATX headings stay on their own line (``sep_before="\\n"`` to the following text).
    - Markdown pipe tables remain intact as single multi-line units rather than being flattened.
    - Sentences within the same prose paragraph carry ``sep_before=" "``.
    """
    units: list[tuple[str, str]] = []
    paragraphs = re.split(r"\n\s*\n", text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        lines = para.splitlines()
        first_in_para = True
        prose_buf: list[str] = []

        def _flush_prose() -> None:
            nonlocal first_in_para
            if not prose_buf:
                return
            block = "\n".join(prose_buf).strip()
            prose_buf.clear()
            if not block:
                return
            for s_idx, sent in enumerate(_split_prose_sentences(block)):
                if not units:
                    sep = ""
                elif s_idx == 0:
                    sep = "\n\n" if first_in_para else "\n"
                else:
                    sep = " "
                units.append((sent, sep))
                first_in_para = False

        idx = 0
        while idx < len(lines):
            ln = lines[idx]
            if not ln.strip():
                idx += 1
                continue
            if _HEADING_LINE.match(ln):
                _flush_prose()
                sep = "" if not units else ("\n\n" if first_in_para else "\n")
                units.append((ln.strip(), sep))
                first_in_para = False
                idx += 1
                continue
            if _is_table_start(lines, idx):
                _flush_prose()
                tlines = [lines[idx].rstrip()]
                idx += 1
                while idx < len(lines) and _is_table_continuation(lines[idx]):
                    tlines.append(lines[idx].rstrip())
                    idx += 1
                sep = "" if not units else ("\n\n" if first_in_para else "\n")
                units.append(("\n".join(tlines), sep))
                first_in_para = False
                continue
            prose_buf.append(ln)
            idx += 1
        _flush_prose()
    return units


def split_sentences(text: str) -> list[str]:
    """Split into structural units/sentences, preserving paragraph, heading, and table boundaries.

    Paragraph breaks, headings, and markdown tables are hard boundaries (never merged into
    flattened prose). Within a prose paragraph, split on terminal punctuation, skipping
    decimals / abbreviations. Returns non-empty, stripped strings in order.
    """
    return [unit for unit, _sep in _split_units(text)]


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


def _split_oversized_table(
    tokenizer: Tokenizer, table: str, chunk_token_size: int, overlap: int
) -> list[tuple[int, str]]:
    """Subdivide an oversized markdown table along row boundaries, repeating the header +
    separator rows on every continuation chunk so numeric data rows keep their column labels."""
    rows = [ln.rstrip() for ln in table.splitlines() if ln.strip()]
    if not rows:
        return []

    header_lines: list[str] = []
    data_rows = rows
    if len(rows) >= 2 and _TABLE_SEP_LINE.match(rows[1]):
        cand_header = rows[:2]
        cand_header_tok = len(tokenizer.encode("\n".join(cand_header)))
        if cand_header_tok < chunk_token_size:
            header_lines = cand_header
            data_rows = rows[2:]
            if not data_rows:
                joined = "\n".join(header_lines)
                return [(cand_header_tok, joined)]

    header_tok = len(tokenizer.encode("\n".join(header_lines))) if header_lines else 0
    out: list[tuple[int, str]] = []
    buf: list[str] = list(header_lines)
    buf_est = header_tok

    def _has_body() -> bool:
        return len(buf) > len(header_lines)

    def _flush_buf() -> None:
        nonlocal buf, buf_est
        if _has_body():
            joined = "\n".join(buf)
            out.append((len(tokenizer.encode(joined)), joined))
        buf = list(header_lines)
        buf_est = header_tok

    for row in data_rows:
        rtok = len(tokenizer.encode(row))
        if rtok > chunk_token_size:
            _flush_buf()
            out.extend(_hard_split_oversized(tokenizer, row, chunk_token_size, overlap))
            continue
        if _has_body() and buf_est + 1 + rtok > chunk_token_size:
            cand = "\n".join(buf + [row])
            if len(tokenizer.encode(cand)) > chunk_token_size:
                _flush_buf()
        if header_lines and not _has_body():
            cand_with_hdr = "\n".join(header_lines + [row])
            cand_tok = len(tokenizer.encode(cand_with_hdr))
            if cand_tok > chunk_token_size:
                # Header + single wide row exceeds chunk limit: emit row alone cleanly.
                out.append((rtok, row))
                continue
            buf.append(row)
            buf_est = cand_tok
            continue
        buf.append(row)
        buf_est = buf_est + 1 + rtok if len(buf) > 1 else rtok

    _flush_buf()
    return out


def _join_units(seq: list[tuple[str, str, int]]) -> str:
    if not seq:
        return ""
    parts = [seq[0][0]]
    for text, sep, _tok in seq[1:]:
        parts.append(sep)
        parts.append(text)
    return "".join(parts).strip()


def chunking_by_sentence_boundary(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 200,   # LightRAG always passes the instance value (config default 200) positionally; this default only ever applies to direct unit tests — keep it equal to the real config default.
    chunk_token_size: int = 2400,
) -> list[dict[str, Any]]:
    """Drop-in for lightrag.operate.chunking_by_token_size — same signature + return
    shape, but chunks end on sentence/paragraph/heading/table boundaries (see module docstring).

    `split_by_character` / `split_by_character_only` are accepted for signature
    compatibility but ignored: KS never sets them (LightRAG passes None), and honoring
    them would reintroduce arbitrary cuts. The structural splitter preserves paragraph
    (`\\n\\n`), heading, and markdown table boundaries.
    """
    if not content or not content.strip():
        return []

    units = _split_units(content)
    if not units:
        return []

    # Pre-tokenize each unit once (O(N) across the document).
    unit_tokens = [len(tokenizer.encode(u)) for u, _sep in units]
    sep_costs = {
        sep: max(1, len(tokenizer.encode(sep))) if sep else 0
        for sep in ("", " ", "\n", "\n\n")
    }

    def _estimate_tokens(seq: list[tuple[str, str, int]]) -> int:
        if not seq:
            return 0
        total = seq[0][2]
        for _u, sep, tok in seq[1:]:
            total += sep_costs.get(sep, 1) + tok
        return total

    def _joined_tokens(seq: list[tuple[str, str, int]]) -> int:
        if not seq:
            return 0
        return len(tokenizer.encode(_join_units(seq)))

    results: list[dict[str, Any]] = []
    cur: list[tuple[str, str, int]] = []  # (text, sep_before, unit_tokens)
    cur_est = 0

    def _flush() -> list[tuple[str, str, int]]:
        """Emit the current chunk; return trailing units to carry as overlap (using cached counts)."""
        if not cur:
            return []
        text = _join_units(cur)
        tok = len(tokenizer.encode(text))
        results.append(
            {
                "tokens": tok,
                "content": text,
                "chunk_order_index": len(results),
            }
        )
        # Sentence/unit-aligned overlap: carry trailing units up to ~overlap using precomputed counts.
        if chunk_overlap_token_size <= 0:
            return []
        carry: list[tuple[str, str, int]] = []
        acc = 0
        for item in reversed(cur):
            u_tok = item[2]
            if acc + u_tok > chunk_overlap_token_size and carry:
                break
            carry.insert(0, item)
            acc += u_tok
            if acc >= chunk_overlap_token_size:
                break
        # Never carry the ENTIRE chunk (would stall forever on a 1-unit chunk).
        if len(carry) >= len(cur):
            carry = carry[1:]
        return carry

    for (s, sep), t in zip(units, unit_tokens):
        # Over-long single unit: flush what we have, split this unit cleanly, continue.
        if t > chunk_token_size:
            if cur:
                _flush()  # overlap carry is irrelevant right before an oversized split
                cur = []
                cur_est = 0
            splitter = (
                _split_oversized_table
                if ("\n" in s and _is_markdown_table(s))
                else _hard_split_oversized
            )
            for sub_tok, sub_text in splitter(
                tokenizer, s, chunk_token_size, chunk_overlap_token_size
            ):
                results.append(
                    {
                        "tokens": sub_tok,
                        "content": sub_text,
                        "chunk_order_index": len(results),
                    }
                )
            continue

        item = (s, sep, t)
        sep_cost = sep_costs.get(sep, 1) if cur else 0
        # Incremental fast path: only exact-tokenize when the running upper bound crosses the cap.
        if cur and cur_est + sep_cost + t > chunk_token_size:
            if _joined_tokens(cur + [item]) > chunk_token_size:
                carry = _flush()
                cur = list(carry)
                cur_est = _estimate_tokens(cur)
                # Trim overlap carry from the front if carry + [item] would exceed chunk_token_size.
                while cur:
                    if cur_est + sep_costs.get(sep, 1) + t <= chunk_token_size:
                        break
                    if _joined_tokens(cur + [item]) <= chunk_token_size:
                        break
                    cur.pop(0)
                    cur_est = _estimate_tokens(cur)

        cur.append(item)
        cur_est = (cur_est + sep_costs.get(sep, 1) + t) if len(cur) > 1 else t

    if cur:
        _flush()

    return results
