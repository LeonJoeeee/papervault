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
        file_path = `<kind>:<source_id>`           (single section) or
                    `<kind>:<source_id>#s<N>`      (multi-section — UNIQUE per section, #79)
    file_path is UNIQUE per section (issue #79): LightRAG 1.5's enqueue de-dups documents by
    canonical file_path basename (pipeline._add_content: a 2nd doc_id reusing an already-seen
    file_path is dropped BEFORE it gets a doc_status row), so a SHARED file_path silently
    dropped sections 2..N of a multi-section doc — only section 0 landed, the rest wedged as
    ledger `error`. Each section now gets its own file_path (`<key>#s<N>`); the WHOLE-book
    citation key is recovered at QUERY time by stripping the trailing `#s<N>`
    (`strip_section_suffix`, consumed by query/synth._source_label + query/aquery._cited_sources),
    so citations still attribute to the book/notebook, not a section. synth still reads the
    credibility band off the source-class prefix (textbook→established, notebook→preliminary,
    via query/synth.py _CRED_BY_SOURCE), which the `#s<N>` suffix never touches.
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


# The per-section file_path suffix (#79). build_sections gives each section of a MULTI-section
# operator doc a UNIQUE file_path `<key>#s<N>` so LightRAG 1.5's enqueue-time filename-dedup
# (pipeline._add_content) can't drop sections 2..N for reusing section 0's file_path. Every
# QUERY-time reader that attributes a citation to the WHOLE book/notebook strips this suffix to
# recover the book-level provenance key.
_SECTION_SUFFIX_RE = re.compile(r"#s\d+$")


def strip_section_suffix(file_path: str) -> str:
    """Recover the book-level provenance key from a per-section operator-doc file_path (#79).

    build_sections tags each section of a multi-section doc with a UNIQUE file_path
    `<key>#s<N>` (defeats LightRAG's filename-dedup — see the module docstring). The query
    path attributes citations to the WHOLE book/notebook, so every reader of the chunk-level
    file_path (query/synth._source_label, query/aquery._cited_sources) strips the trailing
    `#s<N>` here so all sections collapse back to the base key. A bare key (single-section
    doc) or a paper file_path (`paper/<key>`, no `#s<N>`) is returned UNCHANGED. Mirrors how
    LightRAG's normalize_document_file_path strips a `[hint]` suffix (utils_pipeline.py)."""
    return _SECTION_SUFFIX_RE.sub("", file_path)


# --------------------------------------------------------------------------- #
#  Guards (flag-gated source classes)                                          #
# --------------------------------------------------------------------------- #

class SourceDisabledError(RuntimeError):
    """Raised when an operator-doc source class is ingested while its enabling flag is
    OFF. Carries a loud, WHY-explaining message (the CLI surfaces it verbatim)."""


class AlreadyIngestedError(RuntimeError):
    """Raised when a provenance key already has ledger sections (v1 push-once, #47). Carries
    a message naming the key + section count (the CLI surfaces it verbatim)."""


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
    file_path  UNIQUE per section (#79): the base key for a single-section doc, or
               `<key>#s<N>` for a multi-section one — a shared file_path made LightRAG's
               enqueue filename-dedup silently drop sections 2..N. The book/notebook-level
               provenance is recovered at query time via `strip_section_suffix` (drops the
               `#s<N>`) so cited_sources + the synth credibility band still see the whole book.
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
                # UNIQUE per section (#79): a single-section doc keeps the bare `key`; a
                # multi-section one gets `<key>#s{i}` so LightRAG's enqueue filename-dedup
                # can't drop sections 2..N. == doc_id in both cases; the book-level key is
                # recovered at query time via strip_section_suffix.
                file_path=key if single else f"{key}#s{i}",
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

    STUCK-GUARD (mirrors scheduler/round.reconcile_terminal, F17): this is the ONE and ONLY
    reconcile pass — the CLI runs synchronously with no daemon round to retry. So a section
    that did NOT reach PROCESSED — FAILED, a lingering mid-state, an unexpected phase, or a
    MISSING doc_status row (F16 enqueue-drop orphan) — is flipped to `error` here rather than
    left orphaned in `processing` forever (which would also silently block a future re-ingest).
    round.py needs a per-key counter because it re-scans every round; a single synchronous
    pass collapses that to 'not PROCESSED now → error'."""
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
        else:
            await ledger.upsert(ingest_source, s.source_id, doc_id=doc_id, status="error")
            counts["error"] += 1
            if ds == DocStatus.FAILED:
                log.warning("operator-doc %s doc_status FAILED → ledger error", doc_id)
            else:
                log.warning(
                    "operator-doc %s not PROCESSED in the single reconcile pass "
                    "(doc_status=%s) → ledger error (stuck-guard)", doc_id, ds,
                )
    return counts


async def _assert_not_already_ingested(ingest_source: str, base_sid: str, key: str) -> None:
    """v1 push-once (#47): REFUSE if this provenance key already has ANY ledger section.

    Matches the doc's single row (`base_sid`) or its multi-section rows (`base_sid#s<N>`),
    in ANY status — an errored or half-done prior push still counts, so a re-run can't
    silently double-insert / orphan graph docs. Clearing + re-ingest (`--force`/supersede)
    is deliberately future work; an operator must clear the old rows by hand for now.
    Prefix-safe: only an exact `base_sid` or a `base_sid#s...` section id matches (a sibling
    key like `<base_sid>b` does not)."""
    existing = await ledger.load(ingest_source)  # {source_id: LedgerRecord}, workspace-scoped
    hits = [sid for sid in existing if sid == base_sid or sid.startswith(f"{base_sid}#s")]
    if hits:
        raise AlreadyIngestedError(
            f"{key} is already ingested ({len(hits)} ledger section(s)) — v1 is push-once. "
            "Re-ingesting an existing key is REFUSED to avoid duplicate / orphaned graph docs; "
            "clearing the old sections + re-ingest (--force / supersede) is future work."
        )


async def _purge_sections(rag, ingest_source: str, base_sid: str, key: str) -> dict:
    """Delete every landed LightRAG doc + ledger row for one operator-doc provenance key (#79).

    The cleanup path that makes `--force` re-ingest possible. Before #79 a half-committed key
    (e.g. the filename-dedup bug landing only section 0, the rest wedged `error`) was stuck
    forever: re-ingest was REFUSED (_assert_not_already_ingested) with no cleanup path. This
    matches the base-key row and every `<base_sid>#s<N>` section row (workspace-scoped via
    ledger.load — same prefix-safe rule as the push-once check), deletes each row's landed
    LightRAG doc by its stored doc_id FIRST (rag.adelete_by_doc_id, the same primitive
    distill.remove_one uses; a not_found doc is benign — the ledger row is still cleared), then
    the ledger row (ledger.delete, the ingest_source-agnostic primitive). Idempotent: a key
    with no rows is a clean no-op. Best-effort on the graph side — a delete that raises / returns
    an unexpected status is logged LOUD and counted, but the ledger row is still cleared so the
    wedge is always broken (a fresh re-ingest reuses the SAME doc_ids, so a surviving orphan is
    then overwritten by the re-insert)."""
    existing = await ledger.load(ingest_source)  # {source_id: LedgerRecord}, workspace-scoped
    victims = [
        rec for sid, rec in existing.items()
        if sid == base_sid or sid.startswith(f"{base_sid}#s")
    ]
    counts = {"docs_deleted": 0, "docs_failed": 0, "ledger_rows_deleted": 0}
    for rec in victims:
        did = getattr(rec, "doc_id", None) or f"{ingest_source}:{getattr(rec, 'source_id', '')}"
        try:
            r = await rag.adelete_by_doc_id(did)
            status = getattr(r, "status", None)
            if status is None and isinstance(r, dict):
                status = r.get("status")
            if status in ("success", "not_found"):
                counts["docs_deleted"] += 1
            else:
                counts["docs_failed"] += 1
                log.warning("purge %s: adelete_by_doc_id(%s) unexpected status=%s", key, did, status)
        except Exception as e:  # noqa: BLE001 — never let one doc block clearing the wedge
            counts["docs_failed"] += 1
            log.warning("purge %s: adelete_by_doc_id(%s) raised: %r", key, did, e)
        await ledger.delete(ingest_source, getattr(rec, "source_id"))
        counts["ledger_rows_deleted"] += 1
    if victims:
        log.info("purge %s: %s", key, counts)
    return counts


async def purge_key(rag, kind: str, key: str) -> dict:
    """Validate a provenance key then purge its LightRAG docs + ledger rows (#79 --force)."""
    ingest_source, base_sid = validate_key(kind, key)
    return await _purge_sections(rag, ingest_source, base_sid, key)


async def ingest_document(
    rag,
    kind: str,
    key: str,
    path: str,
    *,
    tokenizer: Any = None,
    max_tokens: Optional[int] = None,
    read_text: Callable[[Path], str] | None = None,
    force: bool = False,
) -> dict:
    """End-to-end operator-doc ingest against a (workspace-gated) LightRAG instance.

    guard → key-validate → [force? purge] → push-once check → read → chunk → ledger(processing)
    → enqueue+process → reconcile(done/error). Returns a small counter summary. Raises
    SourceDisabledError (guard OFF), ValueError (bad key / empty doc), or
    AlreadyIngestedError (key already ingested and NOT --force) before/without touching the
    graph write path. `force=True` (#79) PURGES an already-ingested key (its landed graph docs
    + ledger rows) first so the push-once check passes and the key is re-ingested clean —
    the escape hatch for a half-committed / stale key; the purge counts ride back under
    `result["purged"]`.
    """
    check_source_enabled(kind)
    ingest_source, base_sid = validate_key(kind, key)  # fail-fast on a malformed key
    # #79 --force: clear a prior (possibly half-committed) ingest of this key BEFORE the
    # push-once check, so a wedged key can be re-ingested instead of being refused forever.
    purged = await _purge_sections(rag, ingest_source, base_sid, key) if force else None
    await _assert_not_already_ingested(ingest_source, base_sid, key)  # v1 push-once (#3)
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
    try:
        # F17 batch-level backstop (mirrors distill_batch, distill.py): an enqueue/process
        # setup / validation / connection blip usually raises BEFORE doc_status is written, so
        # the just-written `processing` rows would be orphaned. Rewrite them to `error` and
        # return — never punch a partial-write through to the caller.
        await enqueue_sections(rag, sections)
    except Exception as e:  # noqa: BLE001
        for s in sections:
            await ledger.upsert(
                kind, s.source_id, doc_id=s.doc_id, status="error",
                fingerprint=_fingerprint(s.text),
            )
        log.warning(
            "ingest_document %s enqueue/process failed (%d section(s) → error): %r",
            key, len(sections), e,
        )
        result = {"key": key, "kind": kind, "sections": len(sections),
                  "done": 0, "error": len(sections), "pending": 0}
        if purged is not None:
            result["purged"] = purged
        return result
    counts = await _reconcile_sections(rag, kind, sections)
    log.info("ingest_document %s: %d section(s) %s", key, len(sections), counts)
    result = {"key": key, "kind": kind, "sections": len(sections), **counts}
    if purged is not None:
        result["purged"] = purged
    return result


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")
