"""ETA estimation for MCP pending responses (D14).

When a paper isn't ready (no PDF, or no md extract), foreground MCP tools
must return within the 60s tool-call ceiling that every consumer client
imposes (Claude Desktop / Claude Code / Cursor / Anthropic API). They
return ``status="pending"`` with metadata + this ETA hint so the consumer
can decide whether to wait + re-call, or move on.

The estimate is *best-effort, not contract*:

- Extraction is now a SINGLE-TIER whole-PDF MinerU2.5-Pro call (2026-06-06
  migration) — there is NO second-engine fallback. The per-page time is a
  batched-server constant; a long / scanned PDF runs SLOWER than the average
  but never triggers a 2nd engine, so the worst case is a slow-PDF multiplier
  of the per-page constant, not a chandra fallback.
- In-flight wait is bounded by whatever the current worker is processing;
  we don't introspect that, so we use a conservative fixed estimate.
- URGENT priority means we jump past BG batch backlog at the priority-queue
  layer, so queue depth doesn't enter the formula — only in-flight matters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..models import Paper
    from ..store import Library
    from .download_queue import DownloadQueue
    from .extract_queue import ExtractQueue


# Per-page extraction time (seconds) for the single-tier whole-PDF MinerU2.5-Pro
# call (2026-06-06 migration). MinerU batches per-page sub-requests server-side
# and is 10–100× faster than the retired dots/chandra cascade; ~3 s/page is a
# conservative batched-throughput estimate. There is NO second-engine fallback.
MINERU_SECONDS_PER_PAGE = 3
# Worst-case slow-PDF multiplier: a long / scanned / formula-heavy PDF can run
# this many times slower than the per-page average — but never a 2nd engine.
SLOW_PDF_MULTIPLIER = 5

# Median paper size in this library (used when no PDF on disk to count).
TYPICAL_PAGES = 12

# Per-stage in-flight wait estimates. Workers can't be preempted by
# URGENT priority — once a worker starts a paper, we wait for it.
DOWNLOAD_INFLIGHT_SECONDS = 60     # most downloads finish in <2 min
EXTRACT_INFLIGHT_SECONDS = 120     # 2 min covers most MinerU whole-PDF runs

# Own-work estimates.
DOWNLOAD_AVG_SECONDS = 30          # one-shot download


def estimate_eta(
    paper: "Paper",
    library: "Library",
    download_queue: Optional["DownloadQueue"] = None,
    extract_queue: Optional["ExtractQueue"] = None,
) -> dict:
    """Estimate seconds-until-ready for an incomplete paper.

    Returns ``{"eta_seconds": int, "eta_note": str}``. ``eta_note`` is a
    human-readable explanation suitable for showing to a consumer LLM
    (or surfacing in a UI).
    """
    needs_download = not library.has_pdf(paper.key)
    needs_extract = not library.has_extract(paper.key, "md")

    pages = _estimate_pages(paper, library)

    own_processing = 0
    parts: list[str] = []

    if needs_download:
        own_processing += DOWNLOAD_AVG_SECONDS
        parts.append("download")
    if needs_extract:
        own_processing += pages * MINERU_SECONDS_PER_PAGE
        parts.append(
            f"extract (~{pages}p × {MINERU_SECONDS_PER_PAGE}s MinerU)")

    inflight_wait = 0
    if needs_download and download_queue is not None:
        inflight_wait += DOWNLOAD_INFLIGHT_SECONDS
    if needs_extract and extract_queue is not None:
        inflight_wait += EXTRACT_INFLIGHT_SECONDS

    eta = own_processing + inflight_wait
    eta_min = max(1, eta // 60)
    # Single-tier worst case: a slow PDF (long / scanned / formula-heavy) runs a
    # multiplier slower than the per-page average — there is no 2nd-engine
    # fallback anymore, so the worst case is just that multiplier on the work.
    worst_case_min = max(
        eta_min + 5,
        (pages * MINERU_SECONDS_PER_PAGE * SLOW_PDF_MULTIPLIER + 600) // 60)

    note = (
        f"URGENT priority (jumped past background batch). "
        f"Steps: {' + '.join(parts) or 'nothing pending'}. "
        f"Typical wait ~{eta_min} min; "
        f"a slow / scanned PDF could push to ~{worst_case_min} min."
    )

    return {"eta_seconds": int(eta), "eta_note": note}


def _estimate_pages(paper: "Paper", library: "Library") -> int:
    """Best-effort page count. Uses the on-disk PDF if available;
    otherwise falls back to the library median."""
    if library.has_pdf(paper.key):
        try:
            import pypdf

            reader = pypdf.PdfReader(str(library.pdf_path(paper.key)))
            return max(1, len(reader.pages))
        except Exception:
            pass
    return TYPICAL_PAGES


def available_fields(paper: "Paper", library: "Library") -> list[str]:
    """List the fields/artifacts a consumer can use *right now* — even
    while the rest is being produced. Drives the ``available_now`` field
    in a pending response."""
    fields: list[str] = ["title", "key", "authors", "year"]
    if paper.abstract:
        fields.append("abstract")
    if paper.doi:
        fields.append("doi")
    if paper.arxiv_id:
        fields.append("arxiv_id")
    if paper.venue:
        fields.append("venue")
    if library.has_pdf(paper.key):
        fields.append("pdf")
    if library.has_extract(paper.key, "txt"):
        fields.append("txt")
    if library.has_extract(paper.key, "md"):
        fields.append("md")
    return fields
