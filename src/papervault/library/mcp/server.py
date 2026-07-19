"""MCP server: 2 executor-facing tools (get_paper, search_papers) + 4 read-only resources.

Executor-facing consumer surface (the library↔executor interface). Writes / ingest are operator
ops in the ``papervault`` CLI, NOT on this MCP surface. (get_full_text / get_bibtex /
cite_check / get_book_chapter were moved off MCP 2026-05 — full text is read via each
record's ``text_path``; bibtex + \\cite validation are CLI commands.)

This file is the **frontend** layer (per the architecture in docs/architecture.md):
no business logic lives here, only protocol shaping. Each tool:

  - accepts any identifier form (citation key / DOI / arxiv id / fuzzy text)
  - is READ-ONLY for INGEST: a not-in-library DOI/arxiv is NEVER fetched +
    ingested — it returns ``not_found``. The library's single ingest entrance
    is ``search_papers`` (stop-gap, SDD I1). ``get_paper`` is not strictly
    side-effect-free, though: for an IN-library paper whose full text isn't
    ready (and is NOT terminal) it fires a fire-and-forget URGENT
    download/extract enqueue so a ``pending`` record progresses (SDD §5
    I-TERM-ENQ) — that is queue work, not ingest.
  - returns a normalized response shape with a discriminator ``status``
    field for partial-success / ambiguous / not_found cases

Concurrency primitives in ``papervault.library.services.concurrency`` are
shared between foreground tool calls and the background worker pool, so
multiple consumers asking for the same paper at the same time only run
the work once.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
from typing import Optional, Union

from mcp.server.fastmcp import Context, FastMCP

from papervault.library import Library
from papervault.library import fetch as fetcher
from papervault.library.llm import get_llm
from papervault.library.models import (
    DOWNLOAD_STATUS_EXTRACT_FAILED,
    DOWNLOAD_STATUS_FAILED,
    DOWNLOAD_STATUS_METADATA_ONLY,
    DOWNLOAD_STATUS_TERMINAL,
    MIN_TXT_SERVE_BYTES,
    Paper,
)
from papervault.library.search import search_all, search_external_async  # noqa: F401  # search_all kept for back-compat test monkeypatch
from papervault.library.services import ResolverService, SearchService
from papervault.library.services import concurrency
from papervault.library.services.intent_parser import parse_intent
from papervault.library.services.judge import judge_ingest, judge_return
from papervault.library.services.bm25_search import (
    paper_to_dict, bm25_per_term_ranklists, rrf_fuse, reorder_ranklists_by_rrf,
    round_robin, _node_key)
from papervault.library.services.download_queue import DownloadQueue
from papervault.library.services.extract_queue import ExtractQueue


# ----------------------------- helpers --------------------------------------

logger = logging.getLogger(__name__)


def _paper_dict(paper: Paper, library: Library) -> dict:
    """Minimal Executor-facing paper record — the **library↔executor interface** projection
    (2026-05; the canonical contract is ``docs/architecture.md``). 9 research fields
    + a text reference. Internal / ops / legacy fields are intentionally NOT exposed (the
    kernel keeps everything; the MCP view is the minimal projection — store-all,
    expose-minimal). This is the SINGLE caller-facing projection (SDD §5 I-PROJ): get_paper
    found AND its ambiguous candidates AND search results AND ``library://paper`` all flow
    through here — no raw second projection (e.g. ``resolver._candidate_dict``) reaches the
    caller.

    Text reference (D6 serve-safety, the ONE honest output chokepoint):
    exactly ONE of ``text_path`` / ``text_status`` is set, never both.

      * ``text_path`` (an absolute path the agent Reads — lossless verbatim,
        references kept) when a real extract is on disk.
      * ``text_status`` otherwise, telling the caller WHY there's no text:
          - ``"pending"``        — in the library, full text not ready yet
                                   (queued for download/OCR; ≠ not-found,
                                   the paper exists and will be readable).
          - ``"extract_failed"`` — terminal: a PDF is on disk but extraction
                                   gave up (D9 retry ceiling / gate reject).
          - ``"download_failed"``— terminal: every download tier missed and
                                   there's no abstract (a true zero).
          - ``"metadata_only"``  — terminal: no full text, but citable
                                   metadata (DOI/title/authors/year/abstract).

    The ``abstract`` KEY is ALWAYS present regardless of text_status (D6) — it
    lives in the metadata, no need to fake an extract to surface it. The VALUE
    is non-empty only when an abstract exists (always for ``metadata_only``,
    the one terminal status the caller is told to cite from; it may be ``""``
    for a ``pending`` / ``download_failed`` / ``extract_failed`` record that has
    no abstract on file).
    """
    rec: dict = {
        "key": paper.key,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "venue": paper.venue,
        "abstract": paper.abstract,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        # best-effort triage signal; may be 0 or stale — do NOT hard-filter on it.
        "citation_count": paper.citation_count,
    }
    _attach_text_reference(rec, paper, library)
    return rec


# Map terminal download_status → the serve-side text_status the caller sees.
# (``failed`` is surfaced as the more descriptive ``download_failed``.)
_TERMINAL_TEXT_STATUS = {
    DOWNLOAD_STATUS_EXTRACT_FAILED: "extract_failed",
    DOWNLOAD_STATUS_FAILED: "download_failed",
    DOWNLOAD_STATUS_METADATA_ONLY: "metadata_only",
}

# Minimum on-disk SIZE (BYTES — this is ``stat().st_size``, not a character
# count) a txt-ONLY extract must have before serve-safety hands it out as full
# text. The md path needs no such floor: md only lands on disk after the
# whole-document completeness_gate (D3) has passed. BOTH md write-paths enforce
# this at write time — extract.extract_md gates ``final_md`` before _save_md
# (no save on reject), and download._try_firecrawl_text_fallback gates the
# scraped body and deletes the md on reject. So a present md already certifies
# "real + complete"; serve-safety needs no re-check here. A txt is different —
# it is the dumb pypdf foreground pass, and a SCANNED PDF yields a near-empty
# txt (a few stray bytes of header / ligature noise) that is NOT the paper's
# body. Serving that as ``text_path`` would impersonate full text (exactly what
# D6 forbids). Below the floor we fall through to ``text_status`` so the caller
# learns the truth instead of opening a near-empty file. (md is the real
# extract; an md-bearing record never reaches this branch.)
#
# Unit note (SDD §4.3/§5): this floor is BYTES, deliberately distinct from the
# per-chunk review_extract floor (extract.py), which is CHARS (``len(text)``).
# The byte floor keeps the fast ``stat()`` path here — we only want to reject
# the few-byte scanned-PDF noise case, for which bytes vs chars is immaterial.
#
# SINGLE SOURCE OF TRUTH (SDD §5 I-SERVE): the floor lives in ``models`` so all
# four serve doors agree — this chokepoint, ``library://extract/{key}.txt``, and
# the bib ``file=`` txt gate (``Paper.to_bibtex``) all key off the same value.
_MIN_TXT_SERVE_BYTES = MIN_TXT_SERVE_BYTES


def _attach_text_reference(rec: dict, paper: Paper, library: Library) -> None:
    """Set exactly one of ``rec['text_path']`` / ``rec['text_status']`` per the
    D6 serve-safety contract — the ONE honest output chokepoint (SDD §4.3/§5
    ``_serve_full_text``).

    A record serves a ``text_path`` ONLY if a real, complete extract file is on
    disk; otherwise it gets an honest ``text_status`` (and the abstract still
    rides in the record, D6 — never a faked extract). Priority:

      1. md on disk → ``text_path`` (md). md only exists post-gate (D3), so its
         presence already certifies a real, complete extract — disk fact beats
         a stale status string.
      2. ``¬has_pdf ∧ ¬has_md ∧ has_txt`` ∧ non-terminal ∧ ≥ the serve floor →
         ``text_path`` (txt). This is NARROWED (2026-06-06 txt-drop, SDD §3.4)
         to the 2 no-pdf txt-only migration rows (Assaf2023, Sadykov2025): the
         pypdf ``extract_txt`` writer is gone, so a PDF paper never produces an
         interim txt. A PDF paper mid-extraction therefore has NEITHER md nor
         txt and correctly falls to ``text_status="pending"`` — the honest
         answer, not a dirty pypdf body.
      3. otherwise → ``text_status``: a terminal download_status maps to its
         honest value (extract_failed / download_failed / metadata_only),
         everything else (pending, has_pdf-not-yet-extracted) → ``pending`` /
         the terminal value.
    """
    status = paper.download_status or ""
    if library.has_extract(paper.key, "md"):
        rec["text_path"] = str(library.md_path(paper.key))
        return
    # txt-drop (SDD §3.4): serve a txt as ``text_path`` ONLY for a
    # ``¬has_pdf ∧ ¬has_md ∧ has_txt`` row — the 2 no-pdf migration artifacts
    # whose on-disk txt is their ONLY full text. The ``has_md`` branch above
    # already returned, so reaching here means ¬has_md; the explicit ¬has_pdf
    # guard is what restricts this to the no-pdf rows. A PDF paper produces no
    # interim txt anymore (the pypdf writer is gone), so it skips this clause and
    # serves ``pending`` mid-extraction. The non-terminal guard + byte floor are
    # kept for the no-pdf rows (a terminal status or a near-empty txt is not
    # served as full text).
    if (not library.has_pdf(paper.key)
            and status not in DOWNLOAD_STATUS_TERMINAL
            and library.has_extract(paper.key, "txt")):
        try:
            txt_bytes = library.txt_path(paper.key).stat().st_size
        except OSError:
            txt_bytes = 0
        if txt_bytes >= _MIN_TXT_SERVE_BYTES:
            rec["text_path"] = str(library.txt_path(paper.key))
            return
        # Near-empty txt — do NOT serve it as full text. Fall through.
    rec["text_status"] = _TERMINAL_TEXT_STATUS.get(status, "pending")


# ─────────────── search_papers tuning (block 2, 2026-05-31) ───────────────
# Thresholds/caps live in code (tunable), NOT baked into LLM prompts.
RETURN_THRESHOLD = 0.4          # LLM3 relevance score floor to make it into results
EXT_CAP = 60                    # external round-robin pool size (= 2×BATCH_SIZE → ingest = 2 batches)
LIB_CAP = 30                    # library round-robin pool size (= 1×BATCH_SIZE)
ALSO_INCLUDE_CAP = 30           # cap on the externally-vouched library recall tail (§2)
_YEAR_NEUTRAL = 10**9           # None-year sentinel: sorts as NEITHER freshest NOR oldest (also-include, R2-F7)

# ─── §8 source-health derivation (sources_degraded / sources_unconfigured) ───
# ONE health-layer dict is the truthiness authority — NO per-source
# ``is_configured()`` stub. A backend ABSENT from the dict is ALWAYS configured
# (the four keyless backends need no entry and no trivial ``return True``). The
# two lists are DISJOINT by construction (the degraded list excludes anything
# unconfigured). ``BACKENDS`` = the ``CAPS`` keys (``semantic_scholar``, never
# ``s2``) — the single backend-name source of truth.
from papervault.library.search import CAPS as _SEARCH_CAPS  # noqa: E402

BACKENDS = tuple(_SEARCH_CAPS.keys())
UNCONFIGURED_CHECK = {
    "ads":  lambda: bool(os.environ.get("ADS_API_TOKEN")),
    "core": lambda: bool(os.environ.get("CORE_API_KEY", "").strip()),  # .strip() load-bearing
}


def _derive_source_health(degraded_map: dict, n_terms: int) -> tuple[list[str], list[str]]:
    """Assemble ``(sources_unconfigured, sources_degraded)`` from the §3
    ``degraded_map`` (a Counter of backend → #DEGRADED (term,backend) pairs)
    and ``n_terms`` (T = len(search_terms)).

    A backend is UNCONFIGURED iff it has an entry in ``UNCONFIGURED_CHECK`` whose
    check is falsy. A backend is DEGRADED iff it is configured AND EVERY one of
    its T pairs degraded (``degraded_map[b] == T > 0``). Since the fan-out fires
    every (term, backend) pair unconditionally and §1 guarantees every term is
    non-empty, the per-backend pairs-fired count is exactly T by construction —
    so ``== T`` is the all-pairs-degraded test (``total_map`` deleted, R1-F2).
    The two lists are disjoint via the ``b not in sources_unconfigured`` guard.
    """
    sources_unconfigured = [
        b for b in BACKENDS
        if b in UNCONFIGURED_CHECK and not UNCONFIGURED_CHECK[b]()
    ]
    sources_degraded = [
        b for b in BACKENDS
        if b not in sources_unconfigured
        and n_terms > 0 and degraded_map.get(b, 0) == n_terms
    ]
    return sources_unconfigured, sources_degraded


def YEAR_DROP(paper: dict, year_min, year_max) -> bool:
    """The ONE real filter: drop a paper iff a window bound is set AND the
    paper's year is a non-None int OUTSIDE ``[year_min, year_max]``.

    A None-year (or any non-int year) paper is ALWAYS KEPT — recency is
    unverifiable, not a reason to cut (the return judge handles the soft
    recency nudge). No-op when both bounds are None.
    """
    if year_min is None and year_max is None:
        return False
    y = paper.get("year")                      # NEVER paper.year (AttributeError on a dict)
    if not isinstance(y, int):                 # None-year (and any non-int) → KEEP
        return False
    return (year_min is not None and y < year_min) or (year_max is not None and y > year_max)

# is_paper backstop (block 2): source ``publication_types`` is authoritative WHEN it carries a
# recognized marker; it's frequently empty (CORE never sets it; some Crossref/OpenAlex don't),
# so an empty/unrecognized value defers to the LLM's ``is_paper`` rather than guessing.
_PAPER_TYPES = {
    "journalarticle", "journal-article", "article", "review", "conference",
    "conferencepaper", "proceedings-article", "preprint", "posted-content",
    "letter", "lettersandcomments", "casereport", "clinicaltrial",
    "metaanalysis", "study", "dissertation", "thesis",
    "book-chapter", "booksection",
}
_NON_PAPER_TYPES = {
    "dataset", "software", "book", "monograph", "erratum", "retraction",
    "editorial", "news", "peer-review", "peerreview", "grant", "component",
    "paratext", "supplementary-materials", "supplementary-material",
}


def _metadata_is_paper(publication_types) -> Optional[bool]:
    """Decide "is this a research paper" from source metadata ALONE.

    Returns ``True`` / ``False`` only when the metadata is conclusive; ``None`` when it
    can't decide (empty, or only unrecognized markers) — the caller then falls back to
    the LLM's ``is_paper``. A non-paper marker (dataset/book/…) wins over a paper marker.
    """
    types = [str(t).strip().lower() for t in (publication_types or []) if str(t).strip()]
    if not types:
        return None
    if any(t in _NON_PAPER_TYPES for t in types):
        return False
    if any(t in _PAPER_TYPES for t in types):
        return True
    return None


async def _maybe_progress(ctx: Optional[Context], progress: float,
                          total: float, message: str) -> None:
    """ctx.report_progress is a no-op outside an MCP request, but it can
    still raise if the framework is misconfigured. Defensive guard."""
    if ctx is None:
        return
    try:
        await ctx.report_progress(progress, total, message)
    except Exception:
        pass


def _make_threadsafe_progress_cb(ctx: Optional[Context],
                                 *, step: float, total: float):
    """Build a sync ``on_progress(msg)`` callback safe to call from a worker
    thread (extract / download runs via ``asyncio.to_thread``).

    Schedules ``ctx.report_progress(step, total, msg)`` onto the running
    event loop via ``run_coroutine_threadsafe``; fire-and-forget so
    progress emits never block the extract. Returns None if ``ctx`` is
    None (no-op caller path).
    """
    if ctx is None:
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    def cb(msg: str) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(step, total, msg), loop
            )
        except Exception:
            pass

    return cb


# ----------------------------- resolution hints -----------------------------
#
# DELETED 2026-06-05 (SDD §5): an orphaned recovery-hint diagnostic subsystem
# (``_attach_diagnostics`` / ``_hints_for_paper`` / ``_synthesize_tried_sources``
# / ``_missing_for_download`` / ``_DOWNLOAD_TIERS`` / ``_has_field``, keyed off an
# ``in_library_state`` field that was NEVER written) used to live here. It was
# never called anywhere. Decision: a clear terminal ``text_status`` + the full
# metadata record is sufficient self-service for the caller — pl does NOT
# synthesize a best-effort per-tier "supply a DOI to re-arm" guess it cannot
# ground (the cascade's real per-tier results live in the manifest log, not in
# ``get_paper``'s reach). The only hint shaping that remains is for the
# ambiguous / not_found resolution shapes (below).


def _augment_ambiguous(resolution: dict) -> dict:
    """Add hints to an ambiguous resolution describing how to disambiguate.

    The lead hint is conditional on the candidate count (SDD §6a-bis): the
    ambiguous branch also fires for a SINGLE candidate that scored just below
    the auto-resolve threshold, where "Multiple library papers match" is wrong
    and the one-step recovery (call get_paper(<key>)) is the actionable advice.
    """
    out = dict(resolution)
    n = len(out.get("candidates") or [])
    if n <= 1:
        lead = ("One near-match below the auto-resolve threshold — call "
                "get_paper(<its key>) to pick it (the candidate's key is in the "
                "candidates list).")
    else:
        lead = ("Multiple library papers match this fuzzy text. "
                "Add author surname + year (e.g. 'Smith 2023') to disambiguate, "
                "or call get_paper(<one of the candidate keys>) to pick a "
                "specific match.")
    out["hints"] = [
        lead,
        "If you have a DOI or arxiv id for the intended paper, that's the "
        "most reliable identifier and will skip the fuzzy resolver entirely.",
    ]
    return out


def _augment_not_found(resolution: dict, identifier: str) -> dict:
    """Add hints to a not_found resolution suggesting how to refine.

    DOI/arxiv-aware (SDD §6a-bis): a well-formed-but-not-held DOI/arxiv id is a
    valid identifier that simply isn't in the library — the ONLY correct next
    step is search_papers (get_paper is read-only for ingest). The fuzzy hints
    ("did not match a DOI pattern" / "try Lastname Year") are FALSE for it and
    are dropped; they are kept only for a genuine fuzzy-text miss.
    """
    out = dict(resolution)
    # Empty / whitespace-only identifier: the resolver short-circuits at the
    # top of ``_resolve_identifier`` and the fuzzy step NEVER runs — so the
    # generic "fuzzy found no candidates" hint would describe a step that
    # didn't happen. Give the empty case its own honest hint.
    if not (identifier or "").strip():
        out["hints"] = [
            "Empty identifier — pass a citation key, a DOI, an arxiv id, or a "
            "paper title.",
            "Or use search_papers(query=<topic>) to discover related work first.",
        ]
        return out
    if fetcher.looks_like_doi(identifier) or fetcher.looks_like_arxiv(identifier):
        out["hints"] = [
            f"Identifier {identifier!r} is a well-formed DOI/arxiv id but is "
            "not in the library. get_paper is read-only — it does not fetch on "
            "lookup. Use search_papers(query=<topic>) to discover and ingest it.",
        ]
        return out
    out["hints"] = [
        f"Identifier {identifier!r} did not match a known citation key, "
        "DOI pattern, or arxiv id pattern, and fuzzy resolution found no "
        "library candidate above the auto-resolve floor (0.85 for the LLM "
        "reranker, 0.95 on the keyword fallback).",
        "Try a more specific identifier (DOI, arxiv id, or 'Lastname Year').",
        "Or use search_papers(query=<topic>) to discover related work first; "
        "candidates returned there can be passed back into get_paper.",
    ]
    return out


# Fuzzy auto-resolve thresholds (SDD §6a-bis). The absolute per-#1 floor PLUS a
# #1-vs-#2 score-gap so a dominant top hit (e.g. 1.0 vs a trivial 0.30 second)
# resolves directly to `found` instead of bouncing to `ambiguous` on count
# alone. Keyword-derived scores (resolver degraded path) clear a STRICTER floor
# so a token-saturated WRONG paper can't auto-resolve on overlap during an LLM
# outage (the keyword-overlap of a short query is trivially 1.0).
_AUTO_RESOLVE_FLOOR_LLM = 0.85
_AUTO_RESOLVE_FLOOR_KEYWORD = 0.95
_AUTO_RESOLVE_GAP = 0.2


def _should_auto_resolve(in_lib: list[dict]) -> bool:
    """True iff the top in-library candidate dominates enough to auto-resolve to
    `found` (rather than return `ambiguous`).

    Gate: top score ≥ its provenance floor (LLM 0.85 / keyword 0.95) AND
    (it is the sole candidate OR it leads #2 by ≥ the gap). The gap escape lets
    a clearly-dominant #1 resolve even when a weak #2 is also in-library — the
    old rule bounced those to `ambiguous` purely on count==2.
    """
    if not in_lib:
        return False
    top = in_lib[0]
    floor = (_AUTO_RESOLVE_FLOOR_KEYWORD
             if top.get("score_kind") == "keyword"
             else _AUTO_RESOLVE_FLOOR_LLM)
    if top.get("score", 0.0) < floor:
        return False
    if len(in_lib) == 1:
        return True
    return (top.get("score", 0.0) - in_lib[1].get("score", 0.0)) >= _AUTO_RESOLVE_GAP


def _resolve_identifier(identifier: str,
                        library: Library,
                        resolver: ResolverService) -> dict:
    """Map any identifier form → canonical state.

    Returns one of:
      {"status": "found", "key": "<canonical>", "paper": <Paper>}
      {"status": "fetchable_doi", "doi": "..."}
      {"status": "fetchable_arxiv", "arxiv_id": "..."}
      {"status": "ambiguous", "candidates": [...]}
      {"status": "not_found", "message": "..."}

    This is a pure-Python sync function (resolver.resolve calls the LLM
    via the LLM wrapper, which is sync). Callers wrap with to_thread to
    keep the asyncio event loop responsive.
    """
    ident = (identifier or "").strip()
    if not ident:
        return {"status": "not_found", "message": "empty identifier"}

    # 1. existing citation key
    paper = library.get(ident)
    if paper is not None:
        return {"status": "found", "key": ident, "paper": paper}

    # 2. DOI — normalize the common copy-paste wrappers (``doi:`` / the
    # ``https://doi.org/`` resolver URL) to the bare ``10.xxxx/...`` form
    # BEFORE both the recognizer and the ``_by_doi`` lookup, so a pasted
    # prefixed/URL DOI resolves to its held paper instead of falling to fuzzy.
    if fetcher.looks_like_doi(ident):
        doi = fetcher.normalize_doi(ident)
        existing = library.find(doi=doi)
        if existing:
            return {"status": "found", "key": existing.key, "paper": existing}
        return {"status": "fetchable_doi", "doi": doi}

    # 3. arxiv — ``library.find`` normalizes the ``arXiv:`` prefix + version
    # suffix via ``store._normalize_arxiv`` so the prefixed canonical form
    # hits the bare-keyed ``_by_arxiv`` index.
    if fetcher.looks_like_arxiv(ident):
        existing = library.find(arxiv_id=ident)
        if existing:
            return {"status": "found", "key": existing.key, "paper": existing}
        return {"status": "fetchable_arxiv", "arxiv_id": fetcher.normalize_arxiv(ident)}

    # 4. exact normalized-title hit (SDD §6a-bis). A pasted VERBATIM title must
    # resolve to ITS paper deterministically — consult the exact ``_by_title``
    # index (same normalize_title key used at index time) BEFORE the LLM fuzzy
    # path. Without this a verbatim title goes straight to the fuzzy resolver,
    # which can auto-resolve CONFIDENTLY to a DIFFERENT paper (the drill's
    # Raissi2019-title → Dong2025 mis-resolve). A typo'd / near title still
    # misses this exact index and falls through to fuzzy (acceptable).
    exact_title = library.find(title=ident)
    if exact_title is not None:
        return {"status": "found", "key": exact_title.key, "paper": exact_title}

    # 5. fuzzy text via resolver (uses LLM rerank)
    candidates = resolver.resolve(ident, top_k=5)
    in_lib = [c for c in candidates if c.get("in_library")]
    if in_lib and _should_auto_resolve(in_lib):
        p = library.get(in_lib[0]["key"])
        if p is not None:
            return {"status": "found", "key": p.key, "paper": p}
    if len(in_lib) >= 1:
        # I-PROJ (SDD §5): the resolver's _candidate_dict is INTERNAL triage
        # (carries score / has_pdf / has_extract_md / in_library / is_review and
        # omits venue / doi / arxiv_id / the text reference). RE-PROJECT every
        # candidate through the single caller-facing _paper_dict so the ambiguous
        # branch obeys the same minimal-record contract as the found / search /
        # library://paper surfaces — no score leak, plus the citable fields + an
        # honest text reference per candidate. (Mirror search_papers Stage 5.)
        projected = []
        for c in in_lib:
            p = library.get(c.get("key"))
            if p is not None:
                projected.append(_paper_dict(p, library))
        return {"status": "ambiguous", "candidates": projected}

    return {"status": "not_found",
            "message": f"could not resolve {ident!r} as DOI, arxiv id, "
                       "or library reference"}


async def _ensure_paper_in_lib(identifier: str,
                               library: Library,
                               resolver: ResolverService,
                               ctx: Optional[Context]) -> dict:
    """Resolve identifier → Paper that's at least metadata-in-library.

    If the identifier resolves to a paper already in library, return it.
    If it's a DOI/arxiv we don't yet have, fetch metadata and upsert.
    Returns the same shape as _resolve_identifier but with the new paper
    promoted to status='found' if metadata fetch succeeded.
    """
    await _maybe_progress(ctx, 1, 5, "resolving identifier")
    resolution = await asyncio.to_thread(
        _resolve_identifier, identifier, library, resolver)

    if resolution["status"] in ("ambiguous", "not_found"):
        return resolution
    if resolution["status"] == "found":
        return resolution

    # ``get_paper`` is READ-ONLY — it never writes to the library. The ingest gate in
    # ``search_papers`` is the library's SINGLE entrance (stop-gap). A well-formed DOI/arxiv we
    # don't yet hold resolves to not_found here — it is NOT fetched + upserted. BOTH
    # fetchable statuses (fetchable_doi / fetchable_arxiv) map to not_found so the get_paper
    # loop never falls through to its ``else:`` and leak a raw internal status.
    return {
        "status": "not_found",
        "message": (f"{identifier!r} is a well-formed DOI/arxiv id but is not in the "
                    f"library. get_paper is read-only — use search_papers to discover "
                    f"and ingest it."),
    }


# ----------------------------- server build ---------------------------------


def build_server(library_path: Optional[str] = None,
                 *,
                 mcp: Optional[FastMCP] = None,
                 library: Optional[Library] = None,
                 download_queue: Optional[DownloadQueue] = None,
                 extract_queue: Optional[ExtractQueue] = None,
                 num_workers: int = 2,
                 # Phase 28 (2026-05-24, route B): kept for back-compat
                 # with tests that pass ``enable_insight=False``. The
                 # write-side insight pipeline was deleted; this flag now
                 # has no effect. Will be removed once stale call sites
                 # are cleaned up.
                 enable_insight: bool = True,  # noqa: ARG001 (back-compat)
                 insight_queue=None) -> FastMCP:  # noqa: ARG001 (back-compat)
    """Construct a FastMCP server bound to a Library + 2 stage queues.

    For tests, pass any of ``library=`` / ``download_queue=`` /
    ``extract_queue=`` to inject pre-built dependencies. For production,
    only ``library_path`` is needed; both queues are constructed
    (chained via ``on_success`` callbacks) but not started — call
    ``await eq.start()`` then ``await dq.start()`` from the process
    entry point (``__main__.py``) before serving traffic. (Start
    downstream-first so each worker is ready by the time its upstream
    starts pushing work.)

    ✦ Phase 28 (2026-05-24, route B): the third stage (insight) was
    removed. ``insight_queue=`` and ``enable_insight=`` kwargs are
    accepted for back-compat with test fixtures but have no effect.
    All 5-Q digest intelligence moved to the research-side ``librarian/``
    curators; paper-library is now a pure fetch/extract/MCP service.
    """
    if library is None:
        library = Library(library_path)

    # Build queues bottom-up so each upstream can take the next's `.add`
    # as its `on_success` callback. Each stage uses its own worker count
    # tuned to its bottleneck — they are NOT all `num_workers` because
    # the stages are characterised differently:
    #
    #   - ExtractQueue: GPU-bound — Phase 23 auto-scales worker count to the
    #     number of active OCR GPUs (1 paper = 1 exclusive GPU lock). The
    #     engine pools keep 1 worker per GPU per engine.
    #   - DownloadQueue: network_sem(4) is the cap; 4 workers saturate.
    if extract_queue is None:
        # Worker count is dynamic (= active OCR GPU count); the num_workers
        # arg is ignored by ExtractQueue (kept for back-compat).
        extract_queue = ExtractQueue(library)
    if download_queue is None:
        download_queue = DownloadQueue(
            library, num_workers=max(num_workers, 4),
            on_success=extract_queue.add,
        )

    resolver_svc = ResolverService(library)
    search_svc = SearchService(library)

    # When composed into the unified papervault server, tools register onto the passed-in
    # instance (which owns the combined instructions); standalone, build our own.
    if mcp is None:
        mcp = FastMCP("paper-library", instructions="""
This is the literature-library plane of papervault — the raw papers themselves.
Find papers and read their full text. Adding papers to the library happens for you
(search ingests what it discovers) — not something you call. get_paper is read-only:
a named paper not already in the library returns not_found (it is NOT fetched on lookup).

How you use it across a cycle: search_papers to discover literature → get_paper to size up or
pull specific papers by name → open a paper's ``text_path`` to read its full text verbatim.

Tools (full params + good/bad examples are in each tool's own schema):
- search_papers(query): ⭐ LLM-to-LLM. DISCOVER literature — pass a natural-language
    description of what you're doing + what you want to find (NOT keywords); a backend LLM
    parses intent, searches multiple sources, drops off-domain + non-paper records, and returns
    papers ranked by relevance (each a minimal record: title / authors / year / venue / abstract
    / doi / arxiv_id / citation_count + a text reference).
- get_paper(identifiers): look up ONE paper — or a LIST in a single call — by citation key /
    DOI / arXiv id / fuzzy text. Returns the same minimal record per identifier (plus its
    resolution status). READ-ONLY: a DOI/arxiv not already in the library returns not_found
    (it is NOT fetched/ingested) — use search_papers to bring new papers in. (For an
    in-library paper whose text isn't ready yet, get_paper fire-and-forget URGENT-enqueues its
    missing download/extract so a ``pending`` record progresses — that is queue work, not
    ingest; a terminal paper is never re-enqueued.)

Reading full text: both tools give you, per paper, EXACTLY ONE of ``text_path`` (an absolute
file path — Read it for the verbatim full text, references intact) or ``text_status``. The
``text_status`` is one of four values: ``"pending"`` (still working — in the library, text still
downloading / OCR'ing; ≠ not-found, re-call later to pick it up) or three TERMINAL values that
will NOT progress — ``"extract_failed"`` (a PDF is on disk but extraction gave up),
``"download_failed"`` (no source found and no abstract — a true zero), ``"metadata_only"`` (no
full text, but citable DOI/title/authors/year/abstract). ``abstract`` rides in the record either
way — only ``pending`` is worth re-polling.

BibTeX rendering and \\cite validation are NOT MCP tools — run the ``papervault`` CLI for those.
""".strip())

    # Stash references on the server so __main__ / tests can manage lifecycle
    mcp._paper_library = library  # type: ignore[attr-defined]
    mcp._paper_download_queue = download_queue  # type: ignore[attr-defined]
    mcp._paper_extract_queue = extract_queue  # type: ignore[attr-defined]

    # ---------------- get_paper ----------------

    @mcp.tool()
    async def get_paper(identifiers: Union[str, list[str]],
                        ctx: Optional[Context] = None) -> dict:
        """Size up one or more papers you can name. Pass a single identifier or a
        **list** (batch — e.g. to check several papers a knowledge answer cited, or
        several ``\\cite`` keys at once).

        Each identifier accepts any form: a stable citation key (``Karniadakis2021``),
        a DOI, an arxiv id, or fuzzy text (``"Karniadakis 2021"``). **get_paper is
        read-only**: a DOI/arxiv not yet in the library returns ``not_found`` — it is NOT
        fetched + ingested. Use ``search_papers`` to discover and ingest new papers.

        Returns ``{"results": [...]}``, one entry per identifier. A resolved entry is
        the **minimal paper record** — ``key / title / authors / year / venue /
        abstract / doi / arxiv_id / citation_count`` — plus EXACTLY ONE full-text
        reference (``text_path`` XOR ``text_status``):

        - ``text_path`` — an absolute path you ``Read`` for the verbatim full text
          (references kept) when the extract is on disk; or
        - ``text_status`` — no full text yet/ever; one of FOUR values telling you why:
            * ``"pending"``        — still working: in the library, full text queued
              for download/OCR. ``pending`` ≠ not-found — the paper exists and will
              be readable later; re-call to pick it up.
            * ``"extract_failed"`` — terminal: a PDF is on disk but extraction gave
              up (retry ceiling / completeness-gate reject). No full text coming.
            * ``"download_failed"``— terminal: every download tier missed and there's
              no abstract either (a true zero).
            * ``"metadata_only"``  — terminal: no full text, but citable metadata
              (DOI / title / authors / year / abstract) is present.

          The three terminal values mean "stop polling"; only ``pending`` will
          progress. The ``abstract`` is ALWAYS in the record regardless (use it to
          cite a ``metadata_only`` paper).

        For an in-library paper whose text isn't ready (and is NOT terminal),
        get_paper fire-and-forget URGENT-enqueues its missing download/extract so
        the ``pending`` record progresses in the background — this is NOT ingest
        (no new paper is fetched), just queue work; a terminal paper is never
        re-enqueued (SDD §5).

        Unresolved identifiers come back as ``{"status": "ambiguous", "candidates": …}``
        (re-call with a specific key/DOI) or ``{"status": "not_found", …}``.
        """
        if isinstance(identifiers, str):
            identifiers = [identifiers]

        results: list[dict] = []
        # Cross-identifier dedup (SDD §6a-bis): two identifiers that alias the
        # SAME paper (e.g. a cite-key and its DOI) must not yield duplicate
        # records or duplicate URGENT enqueues. Keep the FIRST identifier echo;
        # a later alias is dropped from the batch (results still 1:1 with the
        # distinct papers the caller named, each carrying its first identifier).
        seen_keys: set[str] = set()
        for i, ident in enumerate(identifiers):
            await _maybe_progress(ctx, i, len(identifiers), f"resolving {ident}")
            resolution = await _ensure_paper_in_lib(ident, library, resolver_svc, ctx=None)
            status = resolution["status"]
            if status != "found":
                # ambiguous / not_found — pass through the (augmented) shape per item
                if status == "ambiguous":
                    item = _augment_ambiguous(resolution)
                elif status == "not_found":
                    item = _augment_not_found(resolution, ident)
                else:
                    item = dict(resolution)
                item["identifier"] = ident
                results.append(item)
                continue

            paper: Paper = resolution["paper"]
            if paper.key in seen_keys:
                continue  # alias of a paper already returned in this batch
            seen_keys.add(paper.key)
            # Kick off download/extract if the full text isn't ready, so a `pending`
            # paper progresses toward readable in the background (fire-and-forget; the
            # record carries text_status=pending until the queues finish).
            #
            # I-TERM-ENQ (SDD §5): NEVER re-arm a TERMINAL paper from this
            # foreground path. A metadata_only / download_failed / extract_failed
            # paper is honestly terminal — re-enqueuing fires a redundant URGENT
            # re-download or (expensive GPU) re-OCR that the recovery scan already
            # skips, and contradicts the "stop polling" contract. The gate is
            # download_status NOT in TERMINAL (same as the recovery scan).
            if (getattr(download_queue, "_started", False)
                    and (paper.download_status or "") not in DOWNLOAD_STATUS_TERMINAL):
                if not library.has_pdf(paper.key):
                    download_queue.add(paper.key, priority=concurrency.PRIORITY_URGENT)
                elif not library.has_extract(paper.key, "md"):
                    extract_queue.add(paper.key, priority=concurrency.PRIORITY_URGENT)
            rec = _paper_dict(paper, library)
            rec["identifier"] = ident
            rec["status"] = "found"
            results.append(rec)

        await _maybe_progress(ctx, len(identifiers), len(identifiers), "done")
        return {"status": "ok", "results": results}

    # ---------------- search_papers ----------------

    @mcp.tool()
    async def search_papers(query: str,
                            ctx: Optional[Context] = None) -> dict:
        """Find research papers matching a natural-language research intent.

        Use this when you need to discover or look up literature in
        **space physics + AI4Science**. **Do NOT pass keywords** — pass a
        full description of what you're doing and what you want to find.
        The backend parses your intent with an LLM, so being expressive
        helps; being terse hurts.

        A good ``query`` describes:
          - What problem you're working on (concrete enough to anchor)
          - Why you need these papers (context for the search)
          - What kind of literature you want (method paper? review?
            recent advance? specific subfield? cross-domain analogue?)
          - Any constraints (year, importance, specific venues, etc.)

        Good examples (rich intents):
          ✓ "I'm doing XPINN inversion of the Voyager 1/2 outer
             heliosphere; I want recent advances in PINN training
             stability for stiff PDEs — especially the adaptive-sampling
             line — to see what transfers to Parker transport-equation
             inversion."

          ✓ "I'm building a Neural Process model for 5-11 year GCR
             forecasting from sunspot history. Looking for: (1) recent
             long-horizon time series prediction papers using NP / GP
             baselines, (2) GCR-solar-cycle relationship studies — both
             modern ML and classical heliospheric physics."

          ✓ "Debugging a GNN-PINN for multi-spacecraft SEP transport in
             cycle 3. Looking for: physics-constrained GNN reviews (last
             2-3 years), and advances in multi-spacecraft SEP inversion —
             want to know where the field's SOTA is."

        Bad examples (too thin — LLM has too much interpretive latitude,
        results may drift):
          ✗ "PINN review"
          ✗ "SEP papers"
          ✗ "machine learning solar"

        Pipeline (3-LLM, 2026-05-31):
          1. LLM1 parses your intent → search terms + filters + result count
          2. Multi-source external search (arxiv + ads + S2 + openalex
             + inspire + core, in parallel) merged with the in-library snapshot
          3. LLM2 (ingest gate) judges NEW external candidates for domain +
             paper-ness; in-domain real papers are upserted into the library
             (PDFs queue for background download + extract). Off-domain /
             non-paper records are dropped here — this is what keeps the
             library clean.
          4. LLM3 (return gate) scores every in-library + just-ingested
             candidate for relevance to YOUR intent (absolute 0-1); below
             threshold is dropped, the rest sorted and truncated to your count.

        Returns:
          ``results``: papers ordered by relevance (the order IS the ranking).
            Each is a minimal record — title, authors, year, venue, abstract,
            doi, arxiv_id, citation_count — plus EXACTLY ONE text reference:
            ``text_path`` (an absolute file path you Read for the full verbatim
            text) when the extract is ready, else ``text_status`` — one of
            ``"pending"`` (still working: in the library, full text still
            downloading/OCR'ing — check back later; ``pending`` ≠ not found) OR
            one of three TERMINAL values that won't progress: ``"extract_failed"``
            (PDF on disk, extraction gave up), ``"download_failed"`` (no source
            found and no abstract), ``"metadata_only"`` (no full text but citable
            metadata). The ``abstract`` is always present — use it both to decide
            which to read deep and to cite a ``metadata_only`` hit.
          ``intent_parsed``: how the system interpreted you AND observability
            signals about the run. Read it to verify alignment (refine ``query``
            if off) and to catch silent recall loss. Fields:
              - ``search_terms`` / ``filters_applied`` (year_min, year_max,
                citation_pref, review_pref) / ``limit_resolved``: the parsed plan.
              - ``reasoning``: the parser's one-line read of your intent.
              - ``sources_degraded``: backends that were configured but failed on
                EVERY term this run (a transient outage signal) — fewer sources
                searched, so recall is reduced; worth a retry.
              - ``sources_unconfigured``: backends with no API key (a permanent
                config gap, not an outage) — not searched this run.
              - ``judge_batches_dropped``: ``{ingest, return}`` counts of LLM-gate
                batches that were silently dropped (parse-OK-zero-matched or
                all-retries-failed). Either value > 0 means recall was silently
                lost — RETRY the same query to recover those papers.

        The LLM infers filters (year window, review-only, min citations, result
        count, etc.) from your intent — just say them in words ("the last 2-3
        years", "5 papers", "reviews only"). After picking from ``results``, read the paper via its
        ``text_path``, or call ``get_paper(<key>)`` to re-pull its record.
        """

        # ──── Stage 0: fail CLOSED on an empty / whitespace-only query ────
        # A semantically-empty query has no intent to parse and no terms to
        # search. Short-circuit to a structured error BEFORE any LLM call,
        # external fan-out, or ingest — without this guard the intent
        # normalizer re-injected the raw (unstripped) query as the literal
        # anchor term and the pipeline ran, and on ``'   '`` it even INGESTED
        # a stray paper, mutating the library on a no-op input.
        query = (query or "").strip()
        if not query:
            return {
                "status": "error",
                "message": ("empty query — pass a natural-language description of "
                            "what you're working on and what literature you want to find"),
                "results": [],
            }

        # ONE shared LLM handle for the whole pipeline (parse_intent, judge_ingest,
        # judge_return, AND library.upsert) — §1. The llm_sem PERMIT is re-acquired
        # per LLM-call site and never held across a network wait.
        search_llm = get_llm()

        # ──── Stage 1: parse intent (LLM A) ────────────────────────────
        await _maybe_progress(ctx, 1, 5, "parsing intent")
        try:
            async with concurrency.llm_sem:  # scopes ONLY this call; released before the gather
                plan = await asyncio.to_thread(parse_intent, query, search_llm)
        except ValueError as e:  # §1: a fully-unparseable intent → structured error,
            # never a raw exception out of the tool. (parse_intent already degrades a
            # partial/missing-key plan to safe defaults; this only fires on no-JSON.)
            logger.warning("search_papers: intent parse failed for %r: %s", query, e)
            return {
                "status": "error",
                "message": ("could not parse the query into a search plan — try a "
                            "clearer natural-language description of what you want to find"),
                "results": [],
            }

        # Filters come entirely from the LLM's read of the intent (llm-in design).
        # year_min/year_max are the ONE real filter (the single end-to-end name);
        # citation_pref/review_pref are SOFT — routed to the return judge as text,
        # NEVER a hard cut (no orphaned scalar reads here).
        inferred = plan.get("filters") or {}
        year_min = inferred.get("year_min")
        year_max = inferred.get("year_max")

        # Result count: LLM-suggested (from intent, e.g. "5 papers") or default 10.
        # Default result count when the intent names no number. Bumped 10→15
        # (tokens are cheap — ~250/paper — so a researcher sees a wider slate);
        # env-overridable for quick tuning (e.g. 20).
        effective_limit = plan.get("limit_suggested") or int(
            os.environ.get("PAPER_SEARCH_DEFAULT_LIMIT", "15"))

        search_terms = plan["search_terms"]
        T = len(search_terms)

        # ──── Stage 2: parallel external fan-out + library snapshot ────
        # External arm fires T×6 tagged (term, backend) fetches and returns the
        # UN-deduped, UN-sorted concatenation ``external_raw`` (each node tagged
        # {_source_origin="external", term_idx, rank(native)}, year-cut applied
        # client-side AFTER native-rank stamping). The year bounds are THREADED
        # as params (no backend signature accepts a year bound — §3 reality).
        await _maybe_progress(ctx, 2, 5, "searching external sources")
        ranking_hint = plan.get("ranking_hint", "by_relevance")
        async with concurrency.network_sem:
            (external_raw, degraded_map), library_papers_obj = await asyncio.gather(
                search_external_async(search_terms, year_min=year_min,
                                      year_max=year_max, ranking_hint=ranking_hint),
                asyncio.to_thread(library.all_papers),
            )

        # ──── Stage 1 (library arm, ran co-gathered above) ────────────
        # Normalize the library snapshot to dicts. The FULL ``library_dicts``
        # (including year-INELIGIBLE nodes) is carried to the §4a fold for
        # IDENTITY (so a library twin wins identity over its external copy
        # regardless of year — closes the year-asymmetry manufactured-duplicate).
        # ``lib_corpus`` is the year-eligible subset — the eligibility set: a
        # year-excluded library paper is absent from lib_corpus (never a
        # ranklist / lib_pool / also-include) yet present in library_dicts.
        library_dicts = [paper_to_dict(p) for p in library_papers_obj]
        lib_corpus = [d for d in library_dicts if not YEAR_DROP(d, year_min, year_max)]
        # BM25 ONCE over lib_corpus, queried per-term → BM25>0 node-ref ranklists
        # sorted (-score, key). Empty-token term → []. Empty corpus → [[] …].
        lib_ranklists = bm25_per_term_ranklists(lib_corpus, search_terms)

        # ──── Stage 2.5: §4a fold → vote-union → fair-share (no LLM) ───
        # ONE fold over ``library_dicts + external_raw`` (library FIRST so it
        # wins identity). It (a) registers every library node's identity in the
        # cross-arm ``seen`` map (_node_key-keyed); (b) folds an external twin
        # into its library node when their _node_key collides, incrementing
        # ``folded_vouch`` (GUARDED to library-origin so an ext↔ext fold — which
        # carries no Paper.key — never KeyErrors); (c) UNIONs each external
        # node's per-term votes into ``term_ranks`` (min NATIVE rank, plain int);
        # (d) appends each external vote to ``buckets[term_idx]`` so the per-term
        # ext ranklists need NO second scan; (e) builds ``lib_by_key`` (Paper.key
        # → library node) for the also-include year-eligibility test.
        seen: dict[str, dict] = {}                                  # cross-arm fold key → node
        lib_by_key: dict[str, dict] = {}                            # Paper.key → library node
        folded_vouch: dict[str, int] = collections.defaultdict(int)  # lib Paper.key → vouch count
        buckets: dict[int, list] = collections.defaultdict(list)    # term_idx → ext node list
        # FULL-iteration invariant: register EVERY library node's identity first
        # (library-FIRST so a library twin wins identity over any external copy).
        for node in library_dicts:
            lib_by_key[node["key"]] = node                # library-positive partition
            k = _node_key(node)
            if k not in seen:        # two same-normalize_title library nodes: first registers (harmless)
                seen[k] = node
        # An external node appears ONCE per (term, backend) pair it surfaced
        # under — i.e. once per term it voted on (a distinct dict per fetch). The
        # fold UNIONs those votes onto the first-registered external
        # representative for that identity.
        for node in external_raw:
            k = _node_key(node)
            t = node.get("term_idx")
            rank = node.get("rank", 0)
            hit = seen.get(k)
            if hit is None:
                # First occurrence of this identity → it becomes the external
                # representative. Register + stamp its first per-term vote.
                seen[k] = node
                tr = node.setdefault("term_ranks", {})    # EXTERNAL nodes only
                tr[t] = rank
                buckets[t].append(node)
                continue
            if hit.get("_source_origin") == "library":
                # External twin folds into a LIBRARY node → vouch (guarded to
                # library-origin: hit["key"] is the stable Paper.key, real).
                folded_vouch[hit["key"]] += 1
                continue
            # ext↔ext fold: union this occurrence's vote onto the registered
            # external representative ``hit`` (no Paper.key, no vouch). A
            # representative is appended to buckets[t] exactly ONCE per term t —
            # the FIRST time it gets a vote for t (``t not in tr``); later votes
            # for the same t only tighten the min rank.
            tr = hit.setdefault("term_ranks", {})
            if t not in tr:
                tr[t] = rank
                buckets[t].append(hit)
            else:
                tr[t] = min(tr[t], rank)

        # Per-term external ranklists from ``buckets`` (no T-scan re-walk).
        # Sort each by (min native rank ASC, stable _node_key). LIST
        # COMPREHENSION (not a loop) so ``t`` binds per-iteration; defensive
        # ``.get`` on term_ranks crash-proofs a stray library-shaped node.
        ext_ranklists = [
            sorted(buckets.get(t, []),
                   key=lambda n: (n.get("term_ranks", {}).get(t, 0), _node_key(n)))
            for t in range(T)
        ]

        # ──── RRF consensus pass (§2.5): re-order EACH per-term ranklist by
        # cross-term/cross-backend agreement (Σ 1/(RRF_K+rank0)), so a paper that
        # MANY terms/backends rank high floats to its term's HEAD. This is the
        # WITHIN-term half of the locked RRF ⊕ fair-share composition; the
        # round-robin below stays the BETWEEN-term diversity FLOOR (every term
        # keeps a slot, term-0 leads each cycle), so RRF reorders but can NEVER
        # drop a niche-but-relevant per-term hit. The native-rank order that
        # ``ext_ranklists`` was just sorted into is the RRF tie's stable tiebreak.
        ext_rrf = rrf_fuse(ext_ranklists, _node_key)
        ext_ranklists = reorder_ranklists_by_rrf(ext_ranklists, ext_rrf, _node_key)
        _lib_key = lambda n: n["key"]
        lib_rrf = rrf_fuse(lib_ranklists, _lib_key)
        lib_ranklists = reorder_ranklists_by_rrf(lib_ranklists, lib_rrf, _lib_key)
        # Stamp the RRF consensus score onto each external node — INTERNAL only
        # (never leaked: _paper_dict is a 9-field whitelist; the projection drops
        # _rrf alongside _source_origin/term_idx/term_ranks/rank).
        for rl in ext_ranklists:
            for n in rl:
                n.setdefault("_rrf", ext_rrf.get(_node_key(n), 0.0))

        # ──── Fair-share round-robin (ONE function, per-arm key-fn) — the
        # BETWEEN-term diversity FLOOR over the RRF-reordered lists ────
        ext_pool = round_robin(ext_ranklists, EXT_CAP, _node_key)              # ≤60, external only
        lib_pool = round_robin(lib_ranklists, LIB_CAP, _lib_key)              # ≤30, BM25>0 heads

        # Externally-vouched library recall tail (also-include) — UNCONDITIONAL
        # (even when lib_pool==[]; its whole purpose is rescuing BM25-zero vouched
        # nodes). Ordered by DESC vouch count, then citation_count desc, then
        # None-NEUTRAL year desc, then the stable library key; year-eligible only;
        # not already in lib_pool; capped at ALSO_INCLUDE_CAP. Run AFTER
        # lib_pool_keys is computed from the round-robin head (load-bearing for
        # the return_pool key-uniqueness proof — a reorder reintroduces a dup).
        lib_pool_keys = {n.get("key") for n in lib_pool if n.get("key")}

        def _ai_year(k: str) -> int:
            y = lib_by_key[k].get("year")
            return y if isinstance(y, int) else _YEAR_NEUTRAL   # None-year → neutral, NOT 0

        also_include_keys = {
            k for k in folded_vouch
            if not YEAR_DROP(lib_by_key[k], year_min, year_max)
            and k not in lib_pool_keys
        }
        also_include = sorted(
            also_include_keys,
            key=lambda k: (-folded_vouch[k],
                           -(lib_by_key[k].get("citation_count") or 0),  # importance secondary
                           -_ai_year(k),                                 # recency tertiary, None-NEUTRAL
                           k))[:ALSO_INCLUDE_CAP]
        lib_pool = lib_pool + [lib_by_key[k] for k in also_include]

        # ──── Stage 3: LLM2 judge_ingest(ext) → gate → upsert ──────────
        # Double-filter (stop-gap): the INGEST gate decides what enters the library
        # (domain tier ∈ 1A-2C AND is_paper); a separate RETURN gate (Stage 4) decides
        # what comes back to the agent. Library cands skip the ingest judge entirely.
        await _maybe_progress(ctx, 3, 5, "judging new papers for ingest")
        ingest_judgments, ingest_dropped = await judge_ingest(ext_pool, llm=search_llm)

        ingested_dicts: list[dict] = []
        ingested_keys: set[str] = set()   # same-key collapse → keeps ingested_dicts key-unique
        async with concurrency.lib_write_lock:
            for idx, cand in enumerate(ext_pool):
                j = ingest_judgments.get(idx)
                if j is None:
                    continue  # judge dropped this batch (all retries failed) → NOT ingested (stop-gap)
                if not j["ingest_ok"]:
                    continue  # off-domain (Tier 3 / unjudged tier)
                # PLUG A (junk-ingress fix, 2026-06-03): EGU/Copernicus abstract-only
                # conference submissions are NOT papers (no full text exists). DOI-pattern
                # is the ONLY reliable key — venue strings are dirty (OpenAlex mislabels
                # real PoS/ICRC proceedings papers as "EGU … Conference Abstracts"). The
                # \d after egu scopes to egusphere-egu<NN>-* abstracts, NOT egusphere-<year>-*
                # preprints. Placed BEFORE the metadata wave-in so it also catches the
                # ['article']-tagged abstracts that _metadata_is_paper would otherwise pass.
                _doi_lc = (cand.get("doi") or "").lower()
                if _doi_lc.startswith("10.5194/egusphere-egu") and _doi_lc[21:22].isdigit():
                    continue  # EGU conference abstract (abstract-only, not a paper)
                # is_paper: trust source metadata when conclusive, else the LLM backstop.
                meta = _metadata_is_paper(cand.get("publication_types"))
                is_paper = meta if meta is not None else j["llm_is_paper"]
                if not is_paper:
                    continue  # non-paper record (dataset / book / errata / …)
                # PLUG B (junk-ingress fix, 2026-06-03): content floor — a candidate with
                # no DOI, no arxiv_id, AND no abstract has nothing to fetch and nothing to
                # index (a pure citation-graph ghost stub). Presence-only floor: every real
                # keep clears it (paywalled→DOI, preprint→arxiv, no-abstract-real→DOI/arxiv,
                # searchable stub→abstract); rejects ONLY the content-less pathology.
                if not ((cand.get("doi") or "").strip()
                        or (cand.get("arxiv_id") or "").strip()
                        or (cand.get("abstract") or "").strip()):
                    continue  # no DOI / arxiv / abstract → nothing fetchable or indexable
                # dict(cand) shallow-copy isolates upsert's TOP-LEVEL mutation
                # (it writes key/added_at + a None-coercion pass IN PLACE) from
                # the live ext_pool node still shared with the round-robin
                # emitted-set + return_pool. (authors list stays aliased — safe
                # only because no upsert path mutates it in place.)
                #
                # NO llm= is passed: upsert is purely heuristic. The old
                # borderline preprint↔journal LLM consult was a synchronous
                # llm.call on the event-loop thread held under lib_write_lock
                # (freezing the whole loop); dropping it removed that freeze AND
                # a confirmed dead path. Borderline (sub-0.85 title sim) twins
                # now conservatively DON'T merge — false-positive merges are
                # data loss; a residual duplicate is the safer outcome.
                paper, was_new = library.upsert(dict(cand))
                if paper is None:
                    continue  # phantom quality-gate reject (logged); do NOT touch ingested_keys
                # Persist the gate's judged domain tier onto the record (#57) so the
                # ingest decision is auditable after the fact ("what did the gate think
                # of this paper when it let it in"). j["tier"] is a keep tier here
                # (ingest_ok already gated it to INGEST_TIERS). Stamp ONLY on a fresh
                # ingest or a still-None record — never clobber a tier the audit path
                # (set_domain_status) assigned; domain_status is left untouched. In-place
                # mutation on the stored Paper; the batched library.save() below persists it.
                if was_new or paper.domain_tier is None:
                    paper.domain_tier = j["tier"]
                # Same-key collapse — GATE THE WHOLE TAIL (not just the append):
                # a second candidate resolving to an already-ingested key must
                # not re-enqueue a download.
                if paper.key in ingested_keys:
                    continue
                if was_new and not library.has_pdf(paper.key):
                    download_queue.add(paper.key)
                ingested_dicts.append(paper_to_dict(paper))  # canonical (key + "library" origin)
                ingested_keys.add(paper.key)                 # FINAL tail statement (lockstep)
            library.save()

        # ──── Stage 4: LLM3 judge_return(lib + ingested) → threshold ───
        # Everything here is already vetted (in-library, or just passed the ingest gate);
        # the return judge scores relevance to THIS query intent (absolute 0-1).
        await _maybe_progress(ctx, 4, 5, "scoring relevance")
        # DEDUP BY KEY before L3, keeping the RICHER ingested copy: a library
        # paper P can sit in lib_pool (BM25>0) AND be re-surfaced + ingest-judged
        # on a non-shared identity axis → upsert MERGES into P's key → P also in
        # ingested_dicts under the SAME key, but with a fresher post-merge
        # abstract. Drop the stale lib_pool copy so L3 scores each key ONCE on
        # its richest record (defensive .get → a None/missing key is a no-op).
        _ingested_keys = {d.get("key") for d in ingested_dicts if d.get("key")}
        return_pool = [c for c in lib_pool if c.get("key") not in _ingested_keys] + ingested_dicts
        # search_terms = the parser's distinct sub-topics, fed to the return judge as the
        # "intended sub-topics" it scores each paper's BEST-matching sub-part against (§7).
        # inferred carries the SOFT prefs (citation/review) + year window — nudges, not cuts.
        return_judgments, return_dropped = await judge_return(
            return_pool, query,
            search_terms=search_terms, filters=inferred, llm=search_llm)

        scored: list[tuple[float, dict]] = []
        for idx, cand in enumerate(return_pool):
            j = return_judgments.get(idx)
            if j is None:
                continue  # judge dropped this batch → not returned (fewer results, no contamination)
            if j["score"] < RETURN_THRESHOLD:
                continue
            scored.append((j["score"], cand))
        # Sort: PRIMARY = relevance score DESC always. The ranking_hint adds a
        # SECONDARY tiebreak among equal-score papers (V6 §7):
        #   by_recency    → year DESC (None year sorts LAST — recency-unverifiable)
        #   by_importance → citation_count DESC
        #   by_relevance  → score-only (no secondary)
        # Built as an ascending key over negated values (NO reverse=, which would
        # also flip the tiebreak). None-year is sent to -inf in the negated space
        # (== +inf year) so it lands last under the DESC ordering.
        def _sort_key(t: tuple[float, dict]):
            score, cand = t
            if ranking_hint == "by_recency":
                y = cand.get("year")
                year_rank = -y if isinstance(y, int) else float("inf")  # None → last
                return (-score, year_rank)
            if ranking_hint == "by_importance":
                return (-score, -(cand.get("citation_count") or 0))
            return (-score,)
        scored.sort(key=_sort_key)

        # ──── Stage 5: build minimal records, top-N (N = limit) ────────
        # Pure ``_paper_dict`` projection, ordered by relevance (the ORDER is the rank
        # signal — internal scores/tiers are not leaked into the agent's context).
        await _maybe_progress(ctx, 5, 5, "building response")
        results = []
        for _score, cand in scored[:effective_limit]:
            key = cand.get("key")
            paper_obj = library.get(key) if key else None
            if paper_obj is None:
                continue  # defensive: return_pool is all in-library by construction
            results.append(_paper_dict(paper_obj, library))

        logger.info(
            "search_papers: ext=%d lib=%d → ext_pool=%d lib_pool=%d → ingested=%d returned=%d",
            len(external_raw), len(library_dicts), len(ext_pool), len(lib_pool),
            len(ingested_dicts), len(results),
        )
        # §8 source-health: configured-but-fully-degraded vs unconfigured
        # (disjoint by construction). Derived from the §3 ``degraded_map``
        # channel + T=len(search_terms).
        sources_unconfigured, sources_degraded = _derive_source_health(
            degraded_map, T,
        )
        return {
            "status": "ok",
            "results": results,
            "intent_parsed": {
                "search_terms": plan["search_terms"],
                "filters_applied": {
                    "year_min": year_min,
                    "year_max": year_max,
                    "citation_pref": inferred.get("citation_pref"),
                    "review_pref": inferred.get("review_pref"),
                },
                "limit_resolved": effective_limit,
                "sources_degraded": sources_degraded,
                "sources_unconfigured": sources_unconfigured,
                # §8 observability: per-gate count of parse-OK-zero-matched (or
                # all-retries-failed) DROPPED judge batches — the ONE silent
                # recall-loss class given a caller signal (distinguish "searched,
                # found few" from "the judge silently ate a batch — retry").
                "judge_batches_dropped": {
                    "ingest": ingest_dropped,
                    "return": return_dropped,
                },
                "reasoning": plan.get("reasoning", ""),
            },
        }

    # ---------------- Distillation-tracking endpoints (curator queue) -------
    #
    # Semantic note (Phase 28, 2026-05-24, route B): these endpoints used
    # to track the Phase 11/13 distill-worker pipeline that fed the single
    # 2.8 MB ``distilled.md`` file. After Knowledge System v2 the consumer
    # changed: it's now the research-side ``librarian/`` paper-curator
    # sub-agent processing each paper into ``librarian/topics/*.md``
    # entries. Field ``Paper.distilled_at`` keeps the same name and the
    # same semantic ("this paper has been distilled into the knowledge
    # wiki") — only the write target changed (distilled.md → topics/).
    # No data migration needed; the ~800 historical ``distilled_at``
    # values stay valid.

    def _paper_distill_dict(paper: Paper) -> dict:
        """Slim per-paper dict for the curator queue endpoints. Used by
        list_undistilled / list_oldest_distilled.

        Deliberately excludes ``insight`` AND ``abstract`` so the
        response stays small enough for the Librarian main thread to
        read (a 100-paper batch runs ~20 KB instead of ~130 KB). The
        main thread does mechanical assignment (cluster into per-curator
        batches by title keywords / topic / venue) using these slim
        fields; a curator then calls ``get_paper(key)`` to pull the
        full content for its batch — the curator's context budget absorbs
        the per-paper load, not the main thread.
        """
        return {
            "key": paper.key,
            "title": paper.title or "",
            "year": paper.year,
            "venue": paper.venue or "",
            "authors": paper.authors or [],
            "authors_canonical": paper.canonical_authors,
            "distilled_at": paper.distilled_at,
        }


    # --------------------------- resources -------------------------------

    @mcp.resource(
        "library://bib",
        description=(
            "The whole-library BibTeX master — a large (~8MB) bulk dump of every "
            "(non-quarantined) entry. For a targeted lookup do NOT read this; "
            "prefer search_papers to discover, then library://paper/{key} for one "
            "record. Reading this raw can blow a context window."),
    )
    def _bib() -> str:
        """The whole-library BibTeX master — a large (~8MB) bulk dump of every
        (non-quarantined) entry. For a targeted lookup do NOT read this; prefer
        search_papers to discover, then library://paper/{key} for one record.
        Reading this raw can blow a context window."""
        return library.bib_path.read_text() if library.bib_path.exists() else ""

    @mcp.resource(
        "library://paper/{key}",
        description=(
            "One paper by citation key → the 9-field minimal record (title, "
            "authors, year, venue, abstract, doi, arxiv_id, citation_count) plus "
            "EXACTLY ONE text reference (text_path or text_status). An unknown key "
            "returns the structured not_found shape {\"error\":\"not_found\",\"key\":…}."),
    )
    def _paper(key: str) -> str:
        """One paper by citation key → the 9-field minimal record (title, authors,
        year, venue, abstract, doi, arxiv_id, citation_count) plus EXACTLY ONE
        text reference (text_path or text_status). An unknown key returns the
        structured not_found shape {"error":"not_found","key":…}."""
        paper = library.get(key)
        if paper is None:
            return json.dumps({"error": "not_found", "key": key})
        return json.dumps(_paper_dict(paper, library), ensure_ascii=False, indent=2)

    def _serve_safe_extract_text(key: str, fmt: str) -> str:
        """Return the {key}.{fmt} extract bytes ONLY if serve-safety's policy
        permits, else "" (F3 fix — the raw `path.read_text()` resources used to
        bypass BOTH serve-safety guards).

        These ``library://extract/{key}.{md,txt}`` resources are a parallel,
        MCP-listable serve surface that the §4.3 enumeration (get_paper /
        search / ``library://paper``) missed, so the ``_attach_text_reference``
        chokepoint never saw them. A terminal paper's leftover pypdf txt — the
        SAME PDF the completeness gate judged incomplete — was served raw as
        full text, the exact impersonation D6/§4.3 forbid. This applies the
        SAME disk-fact policy the chokepoint uses, per fmt:

          * ``md``  : md only lands on disk AFTER the completeness gate (D3) —
            both write-paths gate before write, and the historical-firecrawl /
            unlink-fail edges fail-closed by truncating to 0 bytes. So an md
            with content (``has_extract(md)``) is "real ∧ gated"; serve it.
          * ``txt`` : NARROWED (2026-06-06 txt-drop, SDD §3.4) to the 2 no-pdf
            txt-only migration rows. Serve ONLY when ``¬has_pdf ∧ ¬has_md`` AND
            the status is NON-terminal AND the file is ≥ ``_MIN_TXT_SERVE_BYTES``
            — exactly the txt-only chokepoint guards. The pypdf writer is gone,
            so a PDF paper never has an interim txt; a held txt belongs to a
            no-pdf migration artifact. A PDF paper / terminal status / near-empty
            txt falls through to "" rather than impersonating full text.

        Absence contract (SDD §5 I-ABSENCE): a MISSING key returns a structured
        ``{"error":"not_found","key":…}`` (matching ``library://paper/{key}``) so
        the caller can tell "wrong/mistyped key" from "this real paper has no
        servable full text" — the latter keeps the fail-closed ``""``.
        """
        paper = library.get(key)
        if paper is None:
            # Distinct from the fail-closed "" below: the KEY is absent, not a
            # held paper with no servable text. Mirror library://paper/{key}.
            return json.dumps({"error": "not_found", "key": key})
        if not library.has_extract(key, fmt):
            return ""
        if fmt == "md":
            return library.md_path(key).read_text()
        # txt: NARROWED to the no-pdf migration rows (¬has_pdf ∧ ¬has_md) plus
        # the chokepoint's non-terminal + byte-floor guards. A PDF paper (the
        # pypdf interim writer is gone) never reaches here usefully.
        if library.has_pdf(key) or library.has_extract(key, "md"):
            return ""
        status = paper.download_status or ""
        if status in DOWNLOAD_STATUS_TERMINAL:
            return ""
        txt_path = library.txt_path(key)
        try:
            if txt_path.stat().st_size < _MIN_TXT_SERVE_BYTES:
                return ""
            return txt_path.read_text()
        except OSError:
            return ""

    _EXTRACT_SERVE_DESC = (
        "The verbatim full text of a paper, when on disk and serve-safe. "
        "Returns the structured not_found shape {{\"error\":\"not_found\",\"key\":…}} "
        "for a MISSING key, and \"\" (fail-closed) for a HELD paper with no "
        "servable text — so the caller can tell a wrong/mistyped key from a real "
        "paper that simply has no full text. Serve-safety per format: .md is the "
        "gated full text, served only when present (md lands on disk only AFTER "
        "the completeness gate); .txt is the no-pdf migration-artifact full text "
        "ONLY — served only for a paper with no PDF and no md, under a "
        "NON-terminal status and above the byte floor (the pypdf interim writer "
        "is gone, so a PDF paper never serves a txt). (For routing: prefer "
        "text_path from search_papers / library://paper/{{key}} over reading "
        "this directly.)")

    @mcp.resource(
        "library://extract/{key}.md",
        description=_EXTRACT_SERVE_DESC.format() + " This is the .md (gated markdown) extract.",
    )
    def _extract_md(key: str) -> str:
        """Serve the verbatim .md (gated markdown) extract when on disk; structured
        not_found for a missing key; "" for a held-but-unservable paper. See
        _serve_safe_extract_text for the full serve-safety policy."""
        return _serve_safe_extract_text(key, "md")

    @mcp.resource(
        "library://extract/{key}.txt",
        description=_EXTRACT_SERVE_DESC.format()
        + " This is the .txt (no-pdf migration-artifact full text) extract.",
    )
    def _extract_txt(key: str) -> str:
        """Serve the verbatim .txt extract ONLY for a no-pdf migration-artifact
        row (¬has_pdf ∧ ¬has_md, non-terminal status, above the byte floor);
        structured not_found for a missing key; "" for any other held paper
        (a PDF paper never has a servable interim txt). See
        _serve_safe_extract_text for the full serve-safety policy."""
        return _serve_safe_extract_text(key, "txt")

    return mcp
