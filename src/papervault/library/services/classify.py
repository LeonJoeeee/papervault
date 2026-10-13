"""The post-ingest router (D7 / SDD §5).

``classify(paper, library)`` is the SINGLE place that turns a paper's
``download_status`` + on-disk facts into the one next Action. Every
consumer that used to sniff ``download_status.startswith("ok:")`` or
branch on the deleted ``text-only:firecrawl`` / ``extract_low_quality``
ghost states now asks this function instead — so the routing rules live
in exactly one place and stay total + mutually exclusive.

It is a PURE function: it reads ``paper.download_status`` /
``paper.extract_attempts`` and the disk facts
(``has_pdf`` / ``has_extract(md)``), and returns an :class:`Action`.
It mutates nothing — callers (queues / reconcile) do the enqueuing.

The rules are evaluated in PRIORITY ORDER and short-circuit on the first
hit, so each paper lands in exactly one Action (SDD §5 ``post``):

  1. TERMINAL  ⟸ status ∈ {extract_failed, failed, metadata_only}
                 (terminal first — never auto-revive; operator audit only)
  2. TERMINAL  ⟸ has_pdf ∧ has_md   (a real PDF already on disk alongside any
                 md = done; this is ALSO where a firecrawl-md paper that later
                 acquires a real PDF lands — it RESTS served as the firecrawl
                 md and is NEVER re-OCR'd. Per D5 "if firecrawl's fetch is no good, that's final":
                 a firecrawl text-only paper that passed the gate is terminal;
                 there is no PDF-upgrade re-OCR path.)
  3. ¬has_pdf → decide download vs rest:
       • ¬has_md                                     → DOWNLOAD (hunt the PDF)
       • has_md ∧ md_source=firecrawl ∧ ¬hunt_exhausted
                                                     → DOWNLOAD (re-run the
                 strategies once to hunt a real PDF — rescues the ~48 firecrawl
                 papers both queues used to forget, D8; ONLY while the hunt is
                 not yet exhausted, else it would re-download + re-gate forever.
                 NOTE: even if this download finds a PDF, the firecrawl md stays
                 on disk → the next classify hits rule 2 TERMINAL and serves the
                 firecrawl md; the real PDF is NOT re-OCR'd — D5 no upgrade.)
       • else (REAL-OCR md whose PDF is gone, OR an EXHAUSTED firecrawl-md)
                                                     → TERMINAL (REST; the full
                 text is still served via text_path — never a hot-loop)
  4. EXTRACT   ⟸ has_pdf ∧ ¬has_md ∧ status ∈ {ok, pending} ∧ attempts<MAX
                 (``pending`` accepted: a crash between the PDF write and the
                  status flip leaves a real PDF under a still-'pending' record;
                  it MUST extract, not black-hole — restart-safety §8 inv #4)
  5. TERMINAL  ⟸ otherwise (e.g. has_pdf ∧ attempts≥MAX with status not yet
                 flipped; reconcile treats it as terminal pending audit)

With PAPERVAULT_TERMINAL_V2 enabled, an existing DOWNLOAD decision must also
pass the single v2 admission predicate. download_paper asks this router again
immediately before dispatch. EXTRACT is a separate axis; neither routing nor
admission mutates policy. With the flag off, these rules are unchanged.
"""

from __future__ import annotations

from papervault.config import merge_v2_enabled, reopen_v2_enabled, terminal_v2_enabled

from ..models import (
    DOWNLOAD_STATUS_OK,
    DOWNLOAD_STATUS_PENDING,
    DOWNLOAD_STATUS_TERMINAL,
    MAX_EXTRACT_ATTEMPTS,
    Paper,
)
from ..state_v2 import admitted_budget, admission, resolved_identity, valid_lease

# Action labels. Plain strings (not an enum) to stay trivially comparable
# and JSON-loggable; callers branch on equality.
DOWNLOAD = "DOWNLOAD"
EXTRACT = "EXTRACT"
TERMINAL = "TERMINAL"

Action = str


def classify(paper: Paper, library, *, identity_provider=resolved_identity,
             lease_provider=valid_lease, budget_provider=admitted_budget) -> Action:
    """Route by the existing rules, with opt-in acquisition admission.

    Providers are evaluated again at the actual download dispatch boundary.
    The router never changes policy or any other persisted dimension.
    """
    from ..merge_v2 import blocked
    if blocked(paper):
        return TERMINAL
    action = _classify_legacy(paper, library)
    if action == DOWNLOAD and (terminal_v2_enabled() or reopen_v2_enabled() or merge_v2_enabled()
                               or paper.reopen_claim is not None or paper.merge_budget is not None):
        budget_ok = (admitted_budget(paper, library=library) if budget_provider is admitted_budget
                     else budget_provider(paper))
        allowed = admission(paper.acquisition_state, key=paper.key,
                            identity_resolved=identity_provider(paper),
                            lease_ok=lease_provider(paper), budget_ok=budget_ok)
        return DOWNLOAD if allowed else TERMINAL
    return action


def _classify_legacy(paper: Paper, library) -> Action:
    """Return the one next Action for ``paper`` (DOWNLOAD / EXTRACT / TERMINAL).

    Pure + total + mutually exclusive. See module docstring for the rules.

    Args:
        paper: the :class:`Paper` record (carries download_status +
            extract_attempts).
        library: a :class:`store.Library` — used only for the authoritative
            on-disk facts ``has_pdf`` / ``has_extract``.
    """
    status = paper.download_status or ""
    key = paper.key

    # (1) Terminal states never auto-revive. Highest priority so a
    #     metadata_only paper that happens to have a stray md file on disk
    #     still reads as terminal (audit-only reset).
    if status in DOWNLOAD_STATUS_TERMINAL:
        return TERMINAL

    has_pdf = library.has_pdf(key)
    has_md = library.has_extract(key, "md")

    # (2) Real PDF already on disk alongside an md = done. This also catches a
    #     firecrawl-md paper that later acquired a real PDF: it RESTS here,
    #     served as the firecrawl md, and is NEVER re-OCR'd (D5 "if firecrawl's
    #     fetch is no good, that's final" — no PDF-upgrade path; the dead force-upgrade branch in
    #     extract_md was removed). The persisted status may read cosmetically
    #     stale (e.g. a crash-lost 'pending'+pdf+md) but serve-safety keys off
    #     the on-disk md, so the user still gets the full text.
    if has_pdf and has_md:
        return TERMINAL

    # (3) No real PDF on disk → decide DOWNLOAD vs REST.
    if not has_pdf:
        if not has_md:
            return DOWNLOAD                      # nothing on disk → hunt the PDF
        # An md exists but no PDF. ONLY a firecrawl-md whose real-PDF hunt is
        # not yet exhausted re-downloads — to upgrade the scraped web text to a
        # real PDF (D8; this is what rescues the ~48 firecrawl papers both
        # queues used to forget). A REAL-OCR md (its PDF was had and is now
        # gone) OR an EXHAUSTED firecrawl-md instead RESTS (rule-5 TERMINAL,
        # served via text_path) — never a permanent re-download hot-loop.
        if (
            library.md_source(key) == "firecrawl"
            and not paper.firecrawl_pdf_hunt_exhausted
        ):
            return DOWNLOAD
        return TERMINAL

    # (4) PDF on disk, no md yet, retries left → extract. Accept status ∈
    #     {ok, pending}: a crash between the PDF write and the status flip
    #     leaves a real PDF under a still-'pending' record — it MUST extract,
    #     not black-hole to TERMINAL (restart-safety, SDD §8 inv #4).
    if (
        not has_md
        and status in (DOWNLOAD_STATUS_OK, DOWNLOAD_STATUS_PENDING)
        and paper.extract_attempts < MAX_EXTRACT_ATTEMPTS
    ):
        return EXTRACT

    # (5) Catch-all: has_pdf but not extractable under the above rules
    #     (e.g. attempts exhausted but status not yet flipped). Reconcile
    #     treats it as terminal pending operator audit, never hot-looping.
    return TERMINAL
