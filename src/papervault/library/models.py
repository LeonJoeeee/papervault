"""Data models for the paper library."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field

# Forward ref: Paper.insight is Optional[Insight], retained as a
# read-only field after Phase 28 (route B). The schema lives in
# ``insight.schema`` and is the only surviving piece of the deprecated
# 5-Q digest pipeline — kept so the ~800 legacy ``Paper.insight``
# records on disk still deserialize cleanly. No new write path
# populates this field; the production curator pipeline is in the
# research-side ``librarian/`` curators.
from .insight.schema import Insight  # noqa: E402  (intentional pre-class import)

# Paper.download_status canonical routing enum (D7).
#
# This is the SINGLE field that decides "what happens next" for a paper.
# Provenance ("which strategy got the bytes") lives in the SEPARATE
# ``download_source`` field — status routes, source labels. The five
# values are TOTAL and mutually exclusive; ``services.classify.classify``
# is the only place that turns (status + disk facts) into an Action.
#
#   pending         — newly added / not yet downloaded; the download queue
#                     owns it (¬has_pdf → DOWNLOAD).
#   ok              — a real PDF was downloaded AND verified, OR a firecrawl
#                     markdown was obtained (no PDF, but text on disk). The
#                     extract queue / serve path take over from here.
#   extract_failed  — TERMINAL. The PDF is on disk but extraction gave up
#                     (bad block hit the retry ceiling, gate rejected the
#                     whole text, or the occupancy cap fired). Audit-only
#                     reset (D9).
#   failed          — TERMINAL. Every download tier missed AND there is no
#                     abstract to cite from (a "true zero").
#   metadata_only   — TERMINAL. Download missed but rich metadata
#                     (DOI/title/authors/year/abstract) is in hand; the
#                     paper is citable even without full text.
#
# Pre-D7 legacy values ("ok:<src>", "text-only:firecrawl",
# "extract_low_quality") are migrated once on disk by
# ``services.migrate_status.migrate_status`` and must never be written
# again. ``DownloadStatus`` is the canonical Literal; the stored field
# stays plain ``str`` so a not-yet-migrated index.json still deserializes
# (migration then normalizes it).
DOWNLOAD_STATUS_PENDING = "pending"
DOWNLOAD_STATUS_OK = "ok"
DOWNLOAD_STATUS_EXTRACT_FAILED = "extract_failed"
DOWNLOAD_STATUS_FAILED = "failed"
# Metadata-only victory: PDF unavailable AND firecrawl text fallback also
# missed, but we still have rich metadata (DOI/title/authors/year/abstract)
# from CrossRef / S2 / OpenAlex. The paper is "in the library" enough to
# cite from, even though we can't retrieve full text. Distinguishes a
# structurally-paywalled-but-known paper from an outright "failed" record.
DOWNLOAD_STATUS_METADATA_ONLY = "metadata_only"

DownloadStatus = Literal[
    "pending", "ok", "extract_failed", "failed", "metadata_only"
]

# The three terminal states — recovery / reconcile MUST NOT auto-re-enqueue
# these; only an operator ``audit --retry-*`` reset moves them back.
DOWNLOAD_STATUS_TERMINAL = frozenset({
    DOWNLOAD_STATUS_EXTRACT_FAILED,
    DOWNLOAD_STATUS_FAILED,
    DOWNLOAD_STATUS_METADATA_ONLY,
})

# Max real on-card extraction attempts before a paper is declared
# extract_failed (D9). Only true on-card failures count — a paper that
# only queued/waited for a GPU is never charged an attempt.
MAX_EXTRACT_ATTEMPTS = 3

# Minimum on-disk SIZE (BYTES — ``stat().st_size``) a txt-ONLY extract must
# have before any serve door hands it out as full text. The single source of
# truth for the serve-safety byte floor (SDD §5 I-SERVE); the MCP chokepoint
# (``mcp.server._MIN_TXT_SERVE_BYTES``) and the bib ``file=`` gate
# (``Paper.to_bibtex``) both key off it so all serve doors agree. A scanned PDF
# yields a near-empty pypdf txt (a few stray header/ligature bytes) that is NOT
# the paper's body — serving it would impersonate full text (D6 forbids).
MIN_TXT_SERVE_BYTES = 500

ExtractStatus = Literal["none", "txt", "md", "both"]


def slugify_lastname(name: str) -> str:
    """Strip a name down to ASCII letters for use in a citation key."""
    if not name:
        return "Anon"
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    parts = ascii_name.replace(",", " ").split()
    if not parts:
        return "Anon"
    if "," in name:
        return re.sub(r"[^A-Za-z]", "", parts[0]) or "Anon"
    return re.sub(r"[^A-Za-z]", "", parts[-1]) or "Anon"


def canonicalize_author(raw: str) -> str:
    """Normalize an author name to canonical "First [Middle] Last" form.

    ✦ Phase 12.x (#118): cross-source author matching needs consistent
    format. Different sources produce different conventions:

    Input variations → canonical output:
        "John Smith"        → "John Smith"
        "John M. Smith"     → "John M. Smith"
        "J Smith"           → "J. Smith"       (add dot to bare initial)
        "J. Smith"          → "J. Smith"
        "J.M. Smith"        → "J. M. Smith"    (split joined initials)
        "JM Smith"          → "J. M. Smith"    (split bare-letter initials)
        "Smith, John"       → "John Smith"     (BibTeX comma form → flip)
        "Smith, J."         → "J. Smith"
        "Smith, J.M."       → "J. M. Smith"
        "Smith J"           → "J. Smith"       (Crossref bare form)
        "Smith JM"          → "J. M. Smith"
        "Erdős, Paul"        → "Paul Erdős"
        ""                  → ""               (empty stays empty)
        "  John   Smith  "  → "John Smith"     (whitespace collapsed)

    Used by `Paper.canonical_authors` property → MCP get_paper response
    `authors_canonical` field. Raw `authors` list preserved for backward
    compat (don't lose original source format).
    """
    if not raw or not raw.strip():
        return ""
    s = " ".join(raw.strip().split())  # collapse internal whitespace

    # BibTeX comma form: "Last, First [Middle]" → flip
    if "," in s:
        last_part, _, first_part = s.partition(",")
        last_part = last_part.strip()
        first_part = first_part.strip()
        if first_part:
            s = f"{first_part} {last_part}"
        else:
            s = last_part  # "Smith,   " → "Smith"

    toks = s.split()
    if not toks:
        return ""
    if len(toks) == 1:
        return toks[0]  # single name as-is

    last = toks[-1]
    given_raw = toks[:-1]

    # If last looks like initials (≤2 chars all upper / dotted) and given is
    # a single word longer → swap convention: "Smith J" / "Smith JM" came in
    # as Crossref bare form (we read it as Last First-initial, no comma).
    # Heuristic: if last has all-upper letters (no lowercase) AND length ≤ 2,
    # treat last as initials and given as surname.
    last_strip = last.replace(".", "")
    if (len(last_strip) <= 2 and last_strip.isupper()
            and len(given_raw) == 1 and not given_raw[0].replace(".", "").isupper()):
        # "Smith J" → given=Smith, last=J → flip
        new_last = given_raw[0]
        new_given_raw = [last]
        last = new_last
        given_raw = new_given_raw

    # Normalize given names: each token → "F." form if initial, else as-is.
    given_canon = []
    for g in given_raw:
        # "J.M." or "JM" or "J.M" — multi-initial joined
        g_strip = g.replace(".", "")
        if g_strip.isupper() and 1 <= len(g_strip) <= 4:
            # Split: each letter → "X."
            for ch in g_strip:
                given_canon.append(f"{ch}.")
        elif len(g) == 1 and g.isalpha():
            # Single bare initial — "J" → "J."
            given_canon.append(f"{g}.")
        elif (len(g) == 2 and g.endswith(".") and g[0].isalpha()):
            # "J." — keep
            given_canon.append(g)
        else:
            # Full given name like "John" — keep
            given_canon.append(g)

    return " ".join(given_canon + [last])


def base_key(first_author: str, year: int | str | None) -> str:
    """Build the un-suffixed citation key {LastName}{Year}.

    BibTeX keys must be alphanumeric, so the year falls back to "nd" if it's
    missing or unparseable.
    """
    last = slugify_lastname(first_author)
    y = re.sub(r"[^A-Za-z0-9]", "", str(year or "")) or "nd"
    return f"{last}{y}"


def normalize_title(title: str) -> str:
    """Loose key for dedupe: lowercase, alpha-num only."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


class Paper(BaseModel):
    """One paper in the library. The citation key is the primary identifier."""

    # Existing /data/paper-vault/index.json rows carry legacy fields that no
    # longer exist on the model (e.g. the removed ``source_type`` /
    # ``chapters`` / ``publisher`` book-only columns). Tolerate any leftover
    # unknown key on load instead of raising — back-compat over the ~4000
    # historical records, no migration needed.
    model_config = ConfigDict(extra="ignore")

    key: str = Field(..., description="Citation key, unique within the library")
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    # Bibliographic locators. str (NOT int) — real values include "A12",
    # "1029-1041", "L7", "Suppl 1". Backfilled from Crossref / INSPIRE / ADS by
    # the enrich queue (gated by ``enriched_at``); without them BibTeX is
    # incomplete, which is the LLM bib-fabrication trigger.
    volume: str = ""
    pages: str = ""
    issue: str = ""
    abstract: str = ""

    doi: str = ""
    arxiv_id: str = ""
    paper_id: str = ""  # Semantic Scholar paperId
    url: str = ""

    citation_count: int = 0
    is_review: bool = False
    publication_types: list[str] = Field(default_factory=list)

    pdf_path: Optional[str] = None
    txt_path: Optional[str] = None
    md_path: Optional[str] = None

    # Provenance + version lock for idempotent extraction (see C-policy:
    # don't re-run extractors automatically when an upgrade is available).
    txt_engine: str = ""           # e.g., "pypdf"
    txt_engine_version: str = ""   # e.g., "4.2.0"
    md_engine: str = ""             # e.g., "marker"
    md_engine_version: str = ""     # e.g., "1.10.2"

    source: str = ""  # 'arxiv', 'semantic_scholar', 'unpaywall', 'manual', etc.
    # D7 routing enum: pending | ok | extract_failed | failed | metadata_only.
    # Stored as str (not the DownloadStatus Literal) so a not-yet-migrated
    # index.json with a legacy "ok:<src>" value still loads; the one-time
    # migrate_status() normalizes it before any worker runs.
    download_status: str = DOWNLOAD_STATUS_PENDING

    # Metadata-enrichment guard — WORKER-OWNED (never source-supplied). ISO-8601
    # UTC stamp the enrich queue writes by DIRECT mutation on EVERY Crossref
    # attempt (success OR clean-miss), so an un-completable DOI (book/preprint) is
    # probed exactly once, not re-enqueued every reconcile sweep. None = not yet
    # attempted. NOT routed through upsert/_merge (fill-blanks can't write it).
    enriched_at: Optional[str] = None

    # By-title DOI resolution guards (root-cause fix for no-DOI in-domain
    # stubs that can never be enriched or downloaded). Both WORKER/OPERATOR-
    # owned (never source-supplied), set by direct mutation, never routed
    # through upsert/_merge.
    #   ``resolved_doi_at`` — ISO-8601 UTC stamp written by
    #       ``Library.set_resolved_doi`` the moment a by-title-resolved DOI is
    #       claimed, so the action is auditable. None = the DOI (if any) was
    #       NOT resolved by title (it came in with the ingest record).
    #   ``resolve_attempted_at`` — ISO-8601 UTC termination guard (mirrors
    #       ``enriched_at``) the reconcile resolve sweep stamps on every
    #       NON-transient outcome (clean miss OR abstain), so an unresolvable
    #       stub is probed ONCE, not re-queried every 600s sweep. A transient
    #       failure leaves it None so a later sweep retries. None = not yet
    #       attempted.
    resolved_doi_at: Optional[str] = None
    resolve_attempted_at: Optional[str] = None

    # Zotero sync state. Populated by integrations.zotero.ZoteroSync after a
    # successful create_items call. When non-empty, the sync skips this paper
    # (already in user's Zotero library). Clear manually to force re-push.
    zotero_key: Optional[str] = None

    # Paper-insight layer — READ-ONLY field after Phase 28 (2026-05-24,
    # route B). Pre-route-B this was populated by the 5-Q LLM digest
    # worker (``insight.worker.ingest_insight`` running inside the
    # InsightQueue); that pipeline was deleted along with the worker
    # module. The field stays in the schema so the ~800 legacy
    # ``Paper.insight`` records on disk still load cleanly; downstream
    # consumers should treat ``insight is None`` as the normal case
    # (any paper ingested after 2026-05-24 will be None) and route
    # current knowledge queries to ``librarian/topics/`` via the
    # Librarian-side paper-curator instead.
    insight: Optional["Insight"] = None

    # Distillation tracking — kept after Phase 28 (route B) with
    # the SAME semantic ("this paper has been distilled into the
    # knowledge wiki") but a new write context. Pre-route-B this was
    # set by the Orchestrator's batch distill workflow when a paper's
    # Level 1 insight was incorporated into ``distilled.md``.
    # Post-route-B it is set by the Librarian's paper-curator (or by
    # the Librarian main thread via ``mark_distilled``) when the paper
    # has been integrated into ``librarian/topics/``. No data migration
    # was needed; existing ``distilled_at`` values stay valid (different
    # write target, same field semantic).
    # ISO 8601 UTC ("%Y-%m-%dT%H:%M:%SZ").
    # Set by ``mark_distilled`` MCP tool. Queried via ``list_undistilled``
    # (no distilled_at yet, extract on disk) or ``list_oldest_distilled``
    # (reprocess fallback when undistilled queue is short).
    distilled_at: Optional[str] = None

    # Legacy insight-content quality flag. Pre-route-B this was set
    # when the 5-Q worker (or audit scanner) detected the LLM output
    # was meaningless — ≥4/5 answers were the generic "not stated in paper"
    # fallback, which the LLM only emitted when the supposed paper
    # content was actually a publisher landing / paywall stub /
    # empty extract. Post-route-B (Phase 28) the worker is gone and
    # the auto-detection path with it; this field is now writable
    # only via the ``flag_insight_invalid`` MCP endpoint, which the
    # Librarian's paper-curator uses as a cleanup affordance to mark
    # legacy-data papers it doesn't want re-surfaced by
    # ``list_undistilled``. The field stays in the filter chain
    # there for back-compat with the ~hundreds of pre-Phase-12
    # populated-but-empty insights already flagged on disk.
    insight_invalid_reason: Optional[str] = None

    download_source: str = ""         # which strategy succeeded (provenance, D7)

    # D9: real on-card extraction attempts. Persisted (survives restart) so
    # "retry a bad block at most 3 times, then give up" holds across daemon
    # restarts instead of hot-looping forever. ONLY incremented when a paper
    # genuinely occupied a GPU and failed — never for a paper that merely
    # queued or waited for a free card. Reaching MAX_EXTRACT_ATTEMPTS flips
    # the paper to ``extract_failed`` (terminal).
    extract_attempts: int = 0

    # S3 (firecrawl terminal): one-shot resting marker for a firecrawl-md paper.
    # A firecrawl md only exists AFTER the full 18-tier PDF cascade missed, so
    # when ``_gate_firecrawl_md`` PASSes (the md is real full text) the real-PDF
    # hunt is, by construction, EXHAUSTED for this paper. We stamp this True at
    # that moment so ``classify`` stops re-routing it to DOWNLOAD on every
    # reconcile sweep (rule 3). Without it a status=ok firecrawl-md paper
    # (¬has_pdf) would be re-downloaded + re-gated every 600s forever — the
    # permanent hot-loop D9/terminal-never-revive exist to prevent, and repeated
    # re-gating would eventually let one spurious incomplete verdict DELETE a
    # genuinely-good historical md. So the PDF hunt + re-gate run AT MOST ONCE.
    # Persisted (survives restart) so the resting state is stable across daemon
    # sessions. A real PDF arriving later makes ``has_pdf`` True and classify's
    # PDF rules (2)/(4) take over regardless of this flag — it only gates the
    # firecrawl-md DOWNLOAD branch. This resting bit is PERMANENT: no command
    # re-arms the hunt and there is no auto re-scrape (decision: "firecrawl
    # tried and failed = failed, full stop"). ``cmd_audit`` (--fix/--queue/
    # --retry-failed/--retry-low-quality) does NOT touch it. The only way to
    # clear it back to False is a ``force_refresh`` re-add (AddService.add),
    # which is a manual library edit, not an audit action.
    firecrawl_pdf_hunt_exhausted: bool = False

    added_at: str = ""                 # ISO 8601 first-add timestamp

    # Phase 32 (2026-05-30): domain membership — single source of truth for
    # "is this paper in the lab's domain (space physics + AI4Science)". A
    # nullable soft-quarantine flag (mirrors ``insight_invalid_reason``): None =
    # in-domain (default for ALL back-compat data, no migration). A non-None
    # ``domain_status`` hides the paper from clean DEFAULT views (search-snapshot,
    # CLI, bib, KS graph build) WITHOUT deleting it — append-only invariant intact
    # (§11). Cleared back to None to un-quarantine. Stamped by ``search_papers``
    # Stage 4 going forward + the one-shot backfill over existing papers.
    #   "off_domain"  — Tier-3 per domain.md (medical/bio/finance/NLP/vision/...)
    #   "non_paper"   — abstract collection / journal index / data-book tome
    #   "bad_extract" — corrupted / oversized extract, not a usable single paper
    domain_status: Optional[str] = None
    domain_tier: Optional[str] = None   # domain.md tier "1A".."2C"|"3" — audit/explainability

    @computed_field
    @property
    def extract_quality_tier(self) -> str:
        """Quality tier of the on-disk markdown extract, A (best) → F (none).

        Ordering reflects how well the md serves a downstream LLM
        consumer (paper-pipeline RAG, summarization, citation analysis).
        Text fidelity dominates; image-awareness does not — even
        figure-rich papers' core information is in the prose.

        ::

          A  mineru2.5-pro    PDF→md via a professional OCR engine. The
                               current engine is mineru2.5-pro; any historical
                               OCR engine label (the retired dots / chandra /
                               marker / cascade-mixed) is also tier A, read
                               frozen for back-compat. Formulas as $...$, tables
                               as markdown tables, no hallucination. Best.
          B  firecrawl         HTML rendering of publisher page. Naked
                                LaTeX commands (LLM-readable but not $-wrapped),
                                tables preserved, no hallucination. No PDF
                                on disk; well suited when publisher returns
                                no PDF binary or all PDF tiers were CDN-blocked.
          F  metadata_only     No body text, but rich metadata is in hand
                                (DOI, title, authors, year, abstract). The
                                paper is citable; full text was paywalled
                                and all retrieval paths exhausted.
          G  none              No extracted markdown AND no metadata-only
                                fallback recorded (download pending, or
                                extraction fell through without an abstract).
        """
        if self.download_status == DOWNLOAD_STATUS_METADATA_ONLY:
            return "F"
        engine = self.md_engine or ""
        if engine == "firecrawl":
            return "B"
        # Any OCR engine label is tier A — the current mineru2.5-pro plus the
        # retired dots / chandra / marker / cascade-mixed labels (read-frozen).
        # Tiers C (txt-fallback) / D (mimo) / E (low_quality, the deleted ghost
        # state, D7) are gone — no write path stamps them.
        if engine:
            return "A"
        # Fallback for historical metadata gaps: md_engine wasn't recorded
        # but the download succeeded (status=ok). The vast majority of these
        # are OCR-extracted (primary path), so tier A is the honest
        # best-guess. md_path can be None even when an md file exists on disk
        # — callers should use Library.has_extract() for the authoritative
        # file-existence check.
        if self.download_status == DOWNLOAD_STATUS_OK:
            return "A"
        return "G"

    def first_author(self) -> str:
        return self.authors[0] if self.authors else ""

    @property
    def canonical_authors(self) -> list[str]:
        """Authors normalized to consistent "First [Middle] Last" form.

        ✦ Phase 12.x (#118): cross-source author matching via consistent
        format. Raw ``self.authors`` preserved as-is (don't lose source
        format); this returns a derived view via :func:`canonicalize_author`.

        Used by MCP server response shaping (``authors_canonical`` field)
        + cross-paper joins. Computed on access, not cached — author list
        is small (typical 1-10 names) so recomputation is cheap.
        """
        return [canonicalize_author(a) for a in self.authors]

    def display(self) -> str:
        a = self.first_author() or "?"
        if len(self.authors) > 1:
            a += " et al."
        return f"{a} ({self.year or '?'}) — {self.title[:80]}"

    def to_bibtex(self, library=None) -> str:
        """Render this paper as a BibTeX entry. Picks @article when DOI/venue
        present, @misc otherwise.

        If ``library`` is provided, emit a ``file = {...}`` field pointing to
        the on-disk PDF and/or extracts so Zotero (Better BibTeX) auto-attaches
        them on import. Multiple files are semicolon-separated, PDF first.
        """
        _journal_evidence = self.doi or self.venue or self.volume or self.pages
        entry_type = "article" if _journal_evidence else "misc"
        if self.arxiv_id and not _journal_evidence:
            entry_type = "misc"  # bare preprint: no DOI / venue / volume / pages

        fields = []
        if self.title:
            fields.append(("title", _bib_escape(self.title)))
        if self.authors:
            fields.append(("author", _bib_escape(" and ".join(self.authors))))
        if self.year:
            fields.append(("year", str(self.year)))
        if self.venue:
            fields.append(("journal", _bib_escape(self.venue)))
        if self.volume:
            fields.append(("volume", _bib_escape(self.volume)))
        if self.issue:
            fields.append(("number", _bib_escape(self.issue)))   # BibTeX calls issue "number"
        if self.pages:
            fields.append(("pages", _bib_escape(self.pages)))
        if self.doi:
            fields.append(("doi", _bib_escape(self.doi)))
        if self.arxiv_id:
            fields.append(("eprint", _bib_escape(self.arxiv_id)))
            fields.append(("archivePrefix", "arXiv"))
        if self.url:
            fields.append(("url", _bib_escape(self.url)))
        if self.abstract:
            fields.append(("abstract", _bib_escape(self.abstract)))

        # Local file attachments for Zotero. Prefer PDF; fall back to md/txt
        # when only an extract exists (text-only:firecrawl, and the 2 no-pdf
        # txt-only migration artifacts). Absolute paths so Zotero can resolve
        # them from any working directory.
        #
        # SERVE-SAFETY (SDD §5 I-SERVE, txt-drop §3.4): the bib ``file=`` is a
        # serve door like ``text_path`` / ``library://extract``. The pdf (a
        # binary, not impersonating extracted text) and md (gate-certified at
        # write time, D3) links are unconditional. The txt link is NARROWED to
        # the 2 no-pdf txt-only migration rows (Assaf2023, Sadykov2025): the
        # pypdf ``extract_txt`` writer is GONE, so no PDF paper produces an
        # interim txt anymore (a PDF paper mid-extraction serves a ``pending``
        # status, never a dirty pypdf body). The guard is ``¬has_pdf ∧ ¬has_md
        # (the ``elif`` after has_md) ∧ has_txt ∧ non-terminal ∧ ≥ the byte
        # floor`` — so ONLY the 2 no-pdf rows ever emit a txt ``file=`` link.
        if library is not None:
            file_paths: list[str] = []
            if library.has_pdf(self.key):
                file_paths.append(str(library.pdf_path(self.key).resolve()))
            if library.has_extract(self.key, "md"):
                file_paths.append(str(library.md_path(self.key).resolve()))
            elif (not library.has_pdf(self.key)
                  and library.has_extract(self.key, "txt")
                  and (self.download_status or "") not in DOWNLOAD_STATUS_TERMINAL
                  and _txt_size_ok(library, self.key)):
                file_paths.append(str(library.txt_path(self.key).resolve()))
            if file_paths:
                fields.append(("file", _bib_escape(";".join(file_paths))))

        body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields)
        return f"@{entry_type}{{{self.key},\n  {body}\n}}"


def _txt_size_ok(library, key: str) -> bool:
    """True iff {key}.txt is on disk and ≥ the serve byte floor — the same
    sub-floor guard the MCP serve chokepoint applies. TOCTOU-safe: a file
    deleted between the has_extract probe and stat() returns False, not raises.
    """
    try:
        return library.txt_path(key).stat().st_size >= MIN_TXT_SERVE_BYTES
    except OSError:
        return False


def _bib_escape(s: str) -> str:
    """Minimal BibTeX field-content escaping for braces and percent."""
    if s is None:
        return ""
    out = s.replace("\\", "\\\\")
    out = out.replace("{", "\\{").replace("}", "\\}")
    out = out.replace("%", r"\%")
    return out
