"""MCP server: 2 executor-facing tools (get_paper, search_papers) + 4 read-only resources.

Executor-facing consumer surface (the library↔executor interface). Writes / ingest are operator
ops in the ``papervault`` CLI, NOT on this MCP surface. (get_full_text / get_bibtex /
cite_check / get_book_chapter were moved off MCP 2026-05 — full text is read via each
record's ``text_path``; bibtex + \\cite validation are CLI commands.)

This file is the **frontend** layer (per the architecture in docs/architecture.md):
no business logic lives here, only protocol shaping. Each tool:

  - accepts any identifier form (citation key / DOI / arxiv id / fuzzy text)
  - is READ-ONLY for INGEST: a not-in-library DOI/arxiv is NEVER fetched +
    ingested — it returns ``not_found``. Search submits separate registration
    commands after constructing its response. ``get_paper`` is not strictly
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
import json
import logging
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
from papervault.library.services import ResolverService
from papervault.library.services import concurrency
from papervault.library.services.intent_parser import parse_intent
from papervault.library.services.judge import judge_ingest, judge_return
from papervault.library.services.download_queue import DownloadQueue
from papervault.library.services.extract_queue import ExtractQueue
from papervault.library.services.classify import pending_ocr_identity
from papervault.library.services.registration import (
    RegistrationQueue, RegistrationService, serving_excluded,
    metadata_is_paper as _metadata_is_paper, normalized_source_identities,  # noqa: F401 - compatibility import
    )


# ----------------------------- helpers --------------------------------------

logger = logging.getLogger(__name__)


def _degraded_fields(server) -> dict:
    """A ``{"service_degraded": <reason>}`` fragment when the library-plane boot
    FAILED, else ``{}`` (field omitted → healthy).

    ``library/mcp/boot.py`` stashes a short summary on ``server._paper_boot_failed``
    if the detached background boot RAISES — the stage queues / reconcile never came
    up, so a ``pending`` paper will NOT progress and ``get_paper``'s URGENT kick is a
    no-op. The heavy tools splice this fragment into their response so a caller is
    TOLD the plane is degraded (and why), instead of the failure living only in the
    server log. Distinct from ``intent_parsed.sources_degraded`` (a SEARCH-BACKEND
    outage signal); this is LIBRARY-PLANE (queue/reconcile) health. Defensive
    ``getattr`` → absent/None attribute (healthy or pre-boot) yields ``{}``."""
    reason = getattr(server, "_paper_boot_failed", None)
    return {"service_degraded": reason} if reason else {}


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

    Current quarantine/exclusion withholds all text, and pending OCR identity
    verification withholds older extracts. Otherwise a record serves a
    ``text_path`` ONLY if a real, complete extract file is on disk; otherwise
    it gets an honest ``text_status`` (and the abstract still rides in the
    record, D6 — never a faked extract). Priority:

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
    current = library.get(paper.key) or paper
    if serving_excluded(current):
        rec["text_status"] = "unavailable"
        return
    if pending_ocr_identity(current, library):
        rec["text_status"] = "pending"
        return
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


# Compatibility exports for callers of the former inline search implementation.
from papervault.library.services.search_pipeline import (  # noqa: E402, F401
    SearchPipeline, RETURN_THRESHOLD, EXT_CAP, LIB_CAP, ALSO_INCLUDE_CAP,
    SEARCH_AUTHORITY_PRIOR, SEARCH_AUTHORITY_LAMBDA, BACKENDS, UNCONFIGURED_CHECK,
    YEAR_DROP, _authority_reorder, _authority_value, _parse_authority_lambda,
    _derive_source_health,
)


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


# ─────────────── search_papers dead-end → reframe hint (issue #69) ───────────
# When search_papers dead-ends — the intent is UNPARSEABLE, or a clean run
# genuinely matched NOTHING — the caller (an LLM agent) used to get an OPAQUE
# result: a bare {status:error} with a one-line message, or {status:ok,
# results:[]} with no explanation. Abstract / methodology-only / CROSS-DOMAIN-
# METHOD queries are the class that dead-ends here ("model-discrepancy prior",
# "Laplace-approximation failure" — bare method terms that don't anchor to the
# corpus on their own). Those queries are exactly the pipeline's value axis
# (method transfer across fields), so a silent dead-end quietly steers the
# executor away from cross-domain borrowing.
#
# The cure is INSTRUCTIVE, not just diagnostic: hand back a STRUCTURED reframe
# hint the caller can act on — the documented usage is "say what you're DOING,
# then what you WANT to find", which grounds an abstract method in a concrete
# task the search can match. The example is drawn verbatim from the idea23
# field report (2026-07-20): the abstract phrasing failed, the "I'm doing X,
# hit problem Y, looking for Z" rewording worked normally.
def _reframe_hint() -> dict:
    """Structured, caller-actionable guidance for a search_papers dead-end.

    Attached (never raised) whenever the tool would otherwise return an opaque
    dead-end: an unparseable intent, or a clean run that matched nothing. Names
    the likely cause (abstract / methodology-only phrasing) and the documented
    cure (concrete task → problem → want), with a worked before/after example.
    """
    return {
        "reason": (
            "The query read as abstract / methodology-only, so it did not anchor "
            "to concrete literature — bare method terms rarely match on their own, "
            "especially for a cross-domain method transfer (using a method from "
            "field A to attack a problem in field B)."
        ),
        "how_to_reframe": (
            "Say what you're DOING, then what you WANT to find: a concrete task + "
            "the specific problem you hit + the kind of paper you're after. Anchor "
            "the abstract method to the concrete system/phenomenon you're applying "
            "it to, then let the method be one facet — not the whole query."
        ),
        "example": (
            "Instead of 'structural uncertainty / model-discrepancy prior / "
            "Laplace-approximation failure', try: 'I'm doing gradient inversion of "
            "a differentiable PDE solver and found discretisation error dominates "
            "statistical uncertainty — looking for how others quantify "
            "structural/model error in physics-based inverse problems.'"
        ),
    }


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
        from ..merge_v2 import resolve_alias
        # Absorbed aliases are quarantined as an execution fence. Only their
        # canonical destination can supply metadata or evidence.
        paper = resolve_alias(library, ident)
        if paper is None or serving_excluded(paper):
            return {"status": "not_found", "message": "paper is unavailable"}
        return {"status": "found", "key": paper.key, "paper": paper}

    # 2. DOI — normalize the common copy-paste wrappers (``doi:`` / the
    # ``https://doi.org/`` resolver URL) to the bare ``10.xxxx/...`` form
    # BEFORE both the recognizer and the ``_by_doi`` lookup, so a pasted
    # prefixed/URL DOI resolves to its held paper instead of falling to fuzzy.
    if fetcher.looks_like_doi(ident):
        doi = fetcher.normalize_doi(ident)
        existing = library.find(doi=doi)
        if existing:
            if serving_excluded(existing):
                return {"status": "not_found", "message": "paper is unavailable"}
            return {"status": "found", "key": existing.key, "paper": existing}
        return {"status": "fetchable_doi", "doi": doi}

    # 3. arxiv — ``library.find`` normalizes the ``arXiv:`` prefix + version
    # suffix via ``store._normalize_arxiv`` so the prefixed canonical form
    # hits the bare-keyed ``_by_arxiv`` index.
    if fetcher.looks_like_arxiv(ident):
        existing = library.find(arxiv_id=ident)
        if existing:
            if serving_excluded(existing):
                return {"status": "not_found", "message": "paper is unavailable"}
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
        if serving_excluded(exact_title):
            return {"status": "not_found", "message": "paper is unavailable"}
        return {"status": "found", "key": exact_title.key, "paper": exact_title}

    # 5. fuzzy text via resolver (uses LLM rerank)
    candidates = resolver.resolve(ident, top_k=5)
    in_lib = [c for c in candidates if c.get("in_library")
              and (p := library.get(c.get("key"))) is not None and not serving_excluded(p)]
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
    """Read an eligible held record; unknown DOI/arXiv identifiers stay not_found.

    Resolves canonical identities and aliases without fetching or registering
    metadata. Discovery and separately admitted registration belong to search.
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
    """Construct a FastMCP server bound to a Library + registration/download/extract queues.

    For tests, pass any of ``library=`` / ``download_queue=`` /
    ``extract_queue=`` to inject pre-built dependencies. For production,
    only ``library_path`` is needed; the queues are constructed
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

    registration_queue = RegistrationQueue(RegistrationService(library, download_queue))
    resolver_svc = ResolverService(library)

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
file path — Read it for the verbatim full text, references intact) or ``text_status``.
External search results carry stable source identities and provenance before registration.
``text_status`` is one of four values: ``"pending"`` (awaiting registration, download or OCR)
or three TERMINAL values that
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
    mcp._paper_registration_queue = registration_queue  # type: ignore[attr-defined]

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
                elif (not library.has_extract(paper.key, "md")
                      or pending_ocr_identity(paper, library)):
                    extract_queue.add(paper.key, priority=concurrency.PRIORITY_URGENT)
            rec = _paper_dict(paper, library)
            rec["identifier"] = ident
            rec["status"] = "found"
            results.append(rec)

        await _maybe_progress(ctx, len(identifiers), len(identifiers), "done")
        # Surface a library-plane boot failure to the caller (queues/reconcile
        # down → a pending paper won't progress + the URGENT kick above is a
        # no-op). Omitted entirely when the plane is healthy.
        return {"status": "ok", "results": results, **_degraded_fields(mcp)}

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
          3. LLM2 judges NEW external candidates for domain + paper-ness.
             Off-domain/non-paper records are dropped; eligible candidates
             remain external until an independent registration command commits.
          4. LLM3 scores library + eligible external candidates for relevance
             to YOUR intent (absolute 0-1); below threshold is dropped, the
             rest sorted and truncated to your count.
          5. After building the response, submit eligible discoveries for
             separately admitted registration, independently of the served list.
             Search never waits for registration or download/OCR success.

        Returns:
          ``results``: papers ordered by relevance (the order IS the ranking).
            Each is a minimal record — title, authors, year, venue, abstract,
            doi, arxiv_id, citation_count — plus EXACTLY ONE text reference:
            ``text_path`` (an absolute file path you Read for the full verbatim
            text) when the extract is ready, else ``text_status`` — one of
            ``"pending"`` (full text unavailable; external results may still
            await registration, download or OCR) OR
            one of three TERMINAL values that won't progress: ``"extract_failed"``
            (PDF on disk, extraction gave up), ``"download_failed"`` (no source
            found and no abstract), ``"metadata_only"`` (no full text but citable
            metadata). External records use a stable source identity in ``key``
            and include ``source_identities`` provenance before a catalog ID exists.
            The ``abstract`` is always present — use it both to decide
            which to read deep and to cite a ``metadata_only`` hit.
          ``intent_parsed``: how the system interpreted you AND observability
            signals about the run. Read it to verify alignment (refine ``query``
            if off) and to catch silent recall loss. Fields:
              - ``search_terms`` / ``filters_applied`` (year_min, year_max,
                citation_pref, review_pref) / ``limit_resolved``: the parsed plan.
              - ``reasoning``: the parser's one-line read of your intent.
              - ``sources_degraded``: backends that were configured but failed
                or were skipped on every term this run — fewer sources searched, so
                recall is reduced. Either they failed (a transient outage; worth a
                retry) or they are being SKIPPED because repeated failures opened
                their circuit breaker, in which case a retry changes nothing until
                the cooldown elapses (the server log names the open circuits).
              - ``sources_unconfigured``: backends with no API key (a permanent
                config gap, not an outage) — not searched this run.
              - ``judge_batches_dropped``: ``{ingest, return}`` counts of LLM-gate
                batches that were silently dropped (parse-OK-zero-matched or
                all-retries-failed). Either value > 0 means recall was silently
                lost — RETRY the same query to recover those papers.

        The LLM infers filters (year window, review-only, min citations, result
        count, etc.) from your intent — just say them in words ("the last 2-3
        years", "5 papers", "reviews only"). After picking from ``results``, read the paper via its
        ``text_path``. Once registration commits, ``get_paper`` can resolve its
        DOI/arXiv identity or internal citation key; admission may still be pending.

        Latency: minutes, not seconds — plan other work around the call. When the
        service is saturated the call returns a normal result
        {"status": "busy", "busy": true, "retry_after_s": N, ...} instead of
        ``results``: nothing is wrong — wait retry_after_s seconds, then retry,
        rather than treating papervault as down or retrying at once.
        """

        async def progress(step, message):
            await _maybe_progress(ctx, step, 5, message)

        pipeline = SearchPipeline(
            library=library, registration_queue=mcp._paper_registration_queue,
            llm_factory=get_llm, parse=parse_intent, retrieve=search_external_async,
            judge_ingest=judge_ingest, judge_return=judge_return, project=_paper_dict,
            reframe_hint=_reframe_hint, progress=progress,
            authority_enabled=SEARCH_AUTHORITY_PRIOR, authority_weight=SEARCH_AUTHORITY_LAMBDA,
        )
        return await pipeline.run(query, service_fields=_degraded_fields(mcp))

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
        if serving_excluded(paper) or pending_ocr_identity(paper, library):
            return ""
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
