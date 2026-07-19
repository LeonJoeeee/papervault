"""Operator-supplied document ingest (issues #45 textbook / #47 notebook).

The SHARED CORE of the two "multi-upstream" issues: documents the OPERATOR brings by
hand — canonical textbooks / major reviews (#45) and the lab's own executor notebooks
(#47) — entering the knowledge graph with TYPED PROVENANCE, OUTSIDE the paper-library
distill pipeline (which only syncs the auto-discovered paper vault).

Design (mirrors the paper distill insertion path — ingest/distill.py):
  - A paper is enqueued to LightRAG as ONE document (doc_id=`paper:<key>`,
    file_path=`paper/<key>`); LightRAG then re-chunks it with the sentence-boundary
    chunker. Operator docs reuse that SAME two-phase insertion path
    (apipeline_enqueue_documents → apipeline_process_enqueue_documents), but:

  - HEADING-AWARE pre-split for markdown (issue #45: a textbook is 100-1000 pages —
    chapter/section boundaries matter more than fixed windows). We split on markdown
    heading boundaries and greedily pack whole heading-blocks up to a max-token cap, so
    every section boundary lands on a heading. A single oversized heading-block, or a
    plain-text (.txt) doc with no headings, falls back to the existing
    `chunking_by_sentence_boundary` chunker. Each resulting section becomes its own
    LightRAG document.

  - PROVENANCE keys carry the source class in a COLON prefix that survives LightRAG 1.5's
    file_path basename normalization (no slash to strip):
        doc_id    = `<kind>:<source_id>`           (single section) or
                    `<kind>:<source_id>#s<N>`      (multi-section — unique per section)
        file_path = `<kind>:<source_id>`           (the base key — SHARED across all
                    sections so the query path cites the WHOLE book/notebook, not a
                    section, and synth reads the credibility band from the prefix)
    This is deliberately the COLON form (`textbook:Schlickeiser2002`), not the paper's
    SLASH form (`paper/<key>`): the query path's `_cited_papers` treats a slash-free
    file_path as a bare paper key, so a colon prefix is what keeps textbook/notebook keys
    OUT of cited_papers and routes them into a separate `cited_sources` list.

  - GUARDS (flag-gated, both default OFF — deploy-neutral):
      notebook → PAPERVAULT_PRIVATE_SOURCES=1  (unpublished research; shared/beta
                 instances must NEVER ingest notebooks)
      textbook → PAPERVAULT_OPERATOR_SOURCES=1 (operator-supplied canonical sources,
                 gated for beta safety)

Out of scope for v1 (see the PR): retrieval weighting / canonical-source priors (#46),
domain-pack textbook lists (#45), PDF OCR for books (operator supplies extracted md/txt),
and incremental re-ingest of appended notebooks (#47) — v1 is push-once.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from papervault.knowledge.ingest.chunking import chunking_by_sentence_boundary
from papervault.knowledge.ledger import store as ledger

log = logging.getLogger("ks.ingest.operator_docs")

KIND_TEXTBOOK = "textbook"
KIND_NOTEBOOK = "notebook"
KINDS = (KIND_TEXTBOOK, KIND_NOTEBOOK)

# Markdown suffixes → heading-aware split; anything else → plain-text fallback chunker.
_MARKDOWN_SUFFIXES = {".md", ".markdown"}

# Default per-section token cap (env-overridable). Aligned with the pipeline chunk size
# (ingest/chunking.py chunk_token_size default 2400) so a heading-section is roughly one
# LightRAG chunk — LightRAG re-chunks each section anyway, so this only bounds the doc /
# doc_status granularity, never the retrieval chunk size.
DEFAULT_MAX_TOKENS = int(os.getenv("PAPERVAULT_DOC_MAX_TOKENS", "2400"))


# --------------------------------------------------------------------------- #
#  Provenance-key validation                                                  #
# --------------------------------------------------------------------------- #

# textbook:AuthorYear — author token(s) then a 4-digit year, optional disambiguation
# letter (mirrors pl citation keys like `Xu2025e`). e.g. textbook:Schlickeiser2002.
_TEXTBOOK_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]*(?:19|20)\d{2}[a-z]?$")
# notebook:<idea>-<scope> — at least one hyphen separating a non-empty idea slug from a
# non-empty scope slug. e.g. notebook:idea23-c12 (idea23 / c12), notebook:idea-23-c12.
_NOTEBOOK_KEY = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+$")

_KEY_FORMAT_HINT = {
    KIND_TEXTBOOK: "textbook:AuthorYear (e.g. textbook:Schlickeiser2002)",
    KIND_NOTEBOOK: "notebook:<idea>-<scope> (e.g. notebook:idea23-c12)",
}
_KEY_RE = {KIND_TEXTBOOK: _TEXTBOOK_KEY, KIND_NOTEBOOK: _NOTEBOOK_KEY}


def validate_key(kind: str, key: str) -> tuple[str, str]:
    """Validate a provenance key against its kind. Returns (ingest_source, source_id).

    ingest_source == kind (`textbook` / `notebook`); source_id is the part after the
    `<kind>:` prefix (the citeable body of the key). Raises ValueError on any mismatch —
    wrong kind, missing/incorrect prefix, or a body that fails the kind's format.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r} (expected one of {list(KINDS)})")
    prefix = f"{kind}:"
    if not key.startswith(prefix):
        raise ValueError(
            f"key {key!r} must start with {prefix!r} for kind={kind} — "
            f"expected {_KEY_FORMAT_HINT[kind]}"
        )
    source_id = key[len(prefix):]
    if not source_id or not _KEY_RE[kind].match(source_id):
        raise ValueError(
            f"malformed {kind} key {key!r} — expected {_KEY_FORMAT_HINT[kind]}"
        )
    return kind, source_id


# --------------------------------------------------------------------------- #
#  Guards (flag-gated source classes)                                          #
# --------------------------------------------------------------------------- #

class SourceDisabledError(RuntimeError):
    """Raised when an operator-doc source class is ingested while its enabling flag is
    OFF. Carries a loud, WHY-explaining message (the CLI surfaces it verbatim)."""


_GUARD_ENV = {
    KIND_NOTEBOOK: "PAPERVAULT_PRIVATE_SOURCES",
    KIND_TEXTBOOK: "PAPERVAULT_OPERATOR_SOURCES",
}


def _flag_on(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def check_source_enabled(kind: str) -> None:
    """Enforce the per-kind enabling flag; raise SourceDisabledError (with the WHY) if OFF.

    Both flags default OFF, so a stock / shared / beta instance refuses operator-doc
    ingest until an operator explicitly opts in on an instance they control.
    """
    if kind not in _GUARD_ENV:
        raise ValueError(f"unknown kind {kind!r} (expected one of {list(KINDS)})")
    env = _GUARD_ENV[kind]
    if _flag_on(env):
        return
    if kind == KIND_NOTEBOOK:
        raise SourceDisabledError(
            f"notebook ingestion is DISABLED ({env} is not set). Notebooks are UNPUBLISHED "
            "research — shared/beta instances must NEVER ingest them, or the lab's private "
            "reasoning (failure modes, unreleased results, decisions) would leak into a "
            f"distributable knowledge base. This is PRIVATE-INSTANCE ONLY: set {env}=1 only "
            "on an instance you fully control and never redistribute."
        )
    raise SourceDisabledError(
        f"textbook ingestion is DISABLED ({env} is not set). Operator-supplied canonical "
        f"sources are gated OFF by default for beta safety; set {env}=1 to enable on an "
        "instance you operate."
    )


# --------------------------------------------------------------------------- #
#  Heading-aware chunking                                                       #
# --------------------------------------------------------------------------- #

# A markdown ATX heading line: up to 3 leading spaces, 1-6 '#', then whitespace + text.
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}[ \t]+\S")

_TOKENIZER: Any = None


def _default_tokenizer() -> Any:
    """Lazy LightRAG tiktoken tokenizer (same one the pipeline chunker uses). Mirrors the
    lazy pattern in query/multiquery.py so importing this module needs no tiktoken."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from lightrag.utils import TiktokenTokenizer

        _TOKENIZER = TiktokenTokenizer()
    return _TOKENIZER


def _split_heading_blocks(text: str) -> list[str]:
    """Split markdown into blocks that each START at a heading line (any preamble before
    the first heading is block 0). No content is dropped; blocks preserve their text."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if _HEADING_RE.match(ln)]
    if not starts:
        return [text] if text.strip() else []
    blocks: list[str] = []
    if starts[0] > 0:
        pre = "".join(lines[: starts[0]])
        if pre.strip():
            blocks.append(pre)
    for idx, s in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        blk = "".join(lines[s:end])
        if blk.strip():
            blocks.append(blk)
    return blocks


def split_headed_markdown(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    """Heading-aware split: pack whole heading-blocks up to `max_tokens` so every section
    boundary lands on a heading. A single heading-block larger than the cap is subdivided
    by the existing sentence-boundary chunker (nothing is lost)."""
    blocks = _split_heading_blocks(text)
    sections: list[str] = []
    cur: list[str] = []
    cur_tok = 0

    def _flush() -> None:
        nonlocal cur, cur_tok
        if cur:
            joined = "".join(cur).strip()
            if joined:
                sections.append(joined)
            cur = []
            cur_tok = 0

    for blk in blocks:
        btok = len(tokenizer.encode(blk))
        if btok > max_tokens:
            # Oversized single heading-block → reuse the sentence-boundary chunker
            # (overlap 0: these are provenance-doc boundaries, LightRAG adds its own
            # retrieval overlap when it re-chunks each section).
            _flush()
            for c in chunking_by_sentence_boundary(tokenizer, blk, None, False, 0, max_tokens):
                body = (c.get("content") or "").strip()
                if body:
                    sections.append(body)
            continue
        if cur and cur_tok + btok > max_tokens:
            _flush()
        cur.append(blk)
        cur_tok += btok
    _flush()
    return sections


def chunk_document(
    text: str,
    *,
    is_markdown: bool,
    tokenizer: Any = None,
    max_tokens: Optional[int] = None,
) -> list[str]:
    """Split a document into section texts. Markdown → heading-aware; else → the existing
    sentence-boundary chunker (plain-text fallback)."""
    tok = tokenizer or _default_tokenizer()
    cap = max_tokens or DEFAULT_MAX_TOKENS
    if is_markdown:
        return split_headed_markdown(text, tok, cap)
    return [
        (c.get("content") or "").strip()
        for c in chunking_by_sentence_boundary(tok, text, None, False, 0, cap)
        if (c.get("content") or "").strip()
    ]


# --------------------------------------------------------------------------- #
#  Section assembly + LightRAG insertion                                       #
# --------------------------------------------------------------------------- #

@dataclass
class DocSection:
    """One operator-doc section as it enters LightRAG.

    doc_id     unique per section (`<kind>:<source_id>` or `...#s<N>`).
    file_path  the base key, SHARED across all sections of one doc (book/notebook-level
               provenance in the query path's cited_sources + synth credibility band).
    source_id  the ledger source_id (doc_id without the `<kind>:` prefix).
    text       the section body handed to LightRAG (re-chunked there for retrieval).
    """

    doc_id: str
    file_path: str
    source_id: str
    text: str


def build_sections(
    kind: str,
    key: str,
    text: str,
    *,
    is_markdown: bool,
    tokenizer: Any = None,
    max_tokens: Optional[int] = None,
) -> list[DocSection]:
    """Validate the key, chunk the text, and assemble typed DocSections. Pure (no I/O)."""
    ingest_source, base_sid = validate_key(kind, key)
    parts = chunk_document(
        text, is_markdown=is_markdown, tokenizer=tokenizer, max_tokens=max_tokens
    )
    if not parts:
        return []
    single = len(parts) == 1
    sections: list[DocSection] = []
    for i, body in enumerate(parts):
        sid = base_sid if single else f"{base_sid}#s{i}"
        sections.append(
            DocSection(
                doc_id=f"{ingest_source}:{sid}",
                file_path=key,  # base key = `<kind>:<base_sid>`, shared across sections
                source_id=sid,
                text=body,
            )
        )
    return sections


async def enqueue_sections(rag, sections: list[DocSection]) -> dict:
    """The SHARED LightRAG insertion path (same two-phase batch enqueue distill_batch
    uses for papers): enqueue all sections by doc_id, then process once. No ledger here —
    kept pure over the LightRAG boundary so tests mock just `rag`."""
    if not sections:
        return {"queued": 0}
    inputs = [s.text for s in sections]
    ids = [s.doc_id for s in sections]
    fpaths = [s.file_path for s in sections]
    await rag.apipeline_enqueue_documents(input=inputs, ids=ids, file_paths=fpaths)
    await rag.apipeline_process_enqueue_documents()
    return {"queued": len(sections)}


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


async def _reconcile_sections(rag, ingest_source: str, sections: list[DocSection]) -> dict:
    """Read LightRAG doc_status terminal states for the just-enqueued sections and write
    the ledger back to done/error (the CLI runs enqueue+process synchronously, so terminal
    states are available inline — same dict-aware read as scheduler/round.reconcile_terminal).
    Non-terminal rows stay `processing` and are reported as pending."""
    from lightrag.base import DocStatus

    by_doc = {s.doc_id: s for s in sections}
    statuses = await rag.aget_docs_by_ids(list(by_doc))
    counts = {"done": 0, "error": 0, "pending": 0}
    for doc_id, s in by_doc.items():
        st = statuses.get(doc_id)
        ds = st.get("status") if isinstance(st, dict) else getattr(st, "status", None)
        if ds == DocStatus.PROCESSED:
            await ledger.upsert(ingest_source, s.source_id, doc_id=doc_id, status="done")
            counts["done"] += 1
        elif ds == DocStatus.FAILED:
            await ledger.upsert(ingest_source, s.source_id, doc_id=doc_id, status="error")
            counts["error"] += 1
            log.warning("operator-doc %s doc_status FAILED → ledger error", doc_id)
        else:
            counts["pending"] += 1
    return counts


async def ingest_document(
    rag,
    kind: str,
    key: str,
    path: str,
    *,
    tokenizer: Any = None,
    max_tokens: Optional[int] = None,
    read_text: Callable[[Path], str] | None = None,
) -> dict:
    """End-to-end operator-doc ingest against a (workspace-gated) LightRAG instance.

    guard → read → chunk → ledger(processing) → enqueue+process → reconcile(done/error).
    Returns a small counter summary. Raises SourceDisabledError (guard OFF) or ValueError
    (bad key / empty doc) before touching the graph.
    """
    check_source_enabled(kind)
    p = Path(path)
    text = (read_text or _read_text)(p)
    is_md = p.suffix.lower() in _MARKDOWN_SUFFIXES
    sections = build_sections(
        kind, key, text, is_markdown=is_md, tokenizer=tokenizer, max_tokens=max_tokens
    )
    if not sections:
        raise ValueError(f"{path}: no content to ingest (empty after chunking)")

    for s in sections:
        await ledger.upsert(
            kind, s.source_id, doc_id=s.doc_id, status="processing",
            fingerprint=_fingerprint(s.text),
        )
    await enqueue_sections(rag, sections)
    counts = await _reconcile_sections(rag, kind, sections)
    log.info("ingest_document %s: %d section(s) %s", key, len(sections), counts)
    return {"key": key, "kind": kind, "sections": len(sections), **counts}


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")
