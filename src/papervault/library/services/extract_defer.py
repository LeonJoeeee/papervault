"""Transport-defer bookkeeping for the extract stage (issue #43).

Root cause of the reconcile busy-loop (issue #43): the C1 failure model
(``extract.py``/SDD §2.2) deliberately does NOT charge an ``extract_attempt``
for a ``MineruTransportError`` — the extraction backend being unreachable is a
transient condition, so the paper is left ``ok``/``pending`` and reconcile
re-routes it next sweep. That is correct for a brief blip, but when the backend
is down for a *prolonged* stretch (e.g. a broken deploy where ``/health`` is
green but every extract call returns ``mineru_import_failed``) EVERY reconcile
sweep re-enqueues the WHOLE pending-extract set (~1100 papers), each reading its
PDF off disk and re-failing on transport — indefinitely, with zero progress.

Fix (least-invasive, no new terminal ``download_status``): record, per paper, a
lightweight signature of the PDF artifact it was last transport-DEFERRED against.
While the artifact is unchanged AND the backend has produced no successful
extraction since (a process-global "success epoch"), reconcile SKIPS re-enqueuing
it. The deferral is NOT terminal and carries NO data loss — it auto-lifts when:

  * the PDF artifact changes (a re-download lands new bytes → new signature →
    the paper re-enters extraction: the "artifact-appearance retry" the issue
    requires), OR
  * the extraction backend recovers (any extract succeeds → the success epoch
    advances → every deferred paper's stored epoch goes stale → reconcile
    re-enqueues them → they drain).

Reconcile keeps probing recovery cheaply: while papers are deferred it still
promotes a small bounded canary slice each sweep (``DEFER_CANARY``), so the very
first post-recovery success advances the epoch and un-defers the rest. A process
restart resets the epoch, so a booted server always re-probes the full set once
(the persisted signature/epoch intentionally go stale on restart).

Everything here is process-local + duck-typed on ``library`` (``pdf_path`` only)
so this module imports nothing heavy and introduces no import cycle.
"""

from __future__ import annotations

import os
from typing import Optional

# How many still-deferred papers reconcile promotes as recovery canaries each
# sweep. Small enough that a prolonged outage costs O(DEFER_CANARY) extract
# attempts/sweep instead of O(pending-set); large enough that recovery is
# detected within a sweep or two. Env-overridable for ops.
try:
    DEFER_CANARY = max(0, int(os.environ.get("PAPERVAULT_EXTRACT_DEFER_CANARY", "8")))
except ValueError:
    DEFER_CANARY = 8


# ── process-global "extraction backend is producing output" signal ──────────
# Advanced on every successful md extraction. Read (indirectly, via the per-paper
# stored value) by reconcile to tell "no success since this paper was deferred"
# (backend still down → keep skipping) from "a success happened since" (backend
# recovered → re-enqueue). Intentionally NOT persisted: a fresh process starts at
# 0, so any paper carrying a non-zero stored epoch reads as stale → re-probed.
_SUCCESS_EPOCH = 0


def note_extract_success() -> None:
    """Record that an extraction just succeeded (advances the success epoch)."""
    global _SUCCESS_EPOCH
    _SUCCESS_EPOCH += 1


def success_epoch() -> int:
    return _SUCCESS_EPOCH


def reset_for_test() -> None:
    """Test hook: reset the process-global epoch to 0."""
    global _SUCCESS_EPOCH
    _SUCCESS_EPOCH = 0


# ── per-paper artifact signature ────────────────────────────────────────────
def pdf_sig(library, key: str) -> Optional[str]:
    """A cheap signature of the paper's PDF artifact (``size:mtime``), or None if
    there is no readable PDF. A re-download that lands different bytes changes the
    size and/or mtime → a different signature → the paper is no longer "deferred
    against the same artifact"."""
    try:
        st = os.stat(library.pdf_path(key))
    except OSError:
        return None
    return f"{st.st_size}:{int(st.st_mtime)}"


def mark_extract_deferred(paper, library) -> None:
    """Stamp ``paper`` as extraction-deferred against its CURRENT PDF artifact and
    the current success epoch (called from ``extract_md``'s transport arm)."""
    paper.extract_deferred_sig = pdf_sig(library, paper.key)
    paper.extract_deferred_epoch = success_epoch()


def clear_extract_deferred(paper) -> None:
    """Clear any deferral stamp (called on every NON-transport extract outcome:
    success, per-doc defect, probe-bad, gate/clarity reject)."""
    paper.extract_deferred_sig = None
    paper.extract_deferred_epoch = 0


def is_extract_deferred(paper, library) -> bool:
    """True iff ``paper`` should be SKIPPED by reconcile's extract re-enqueue:
    it carries a deferral stamp, the PDF artifact is unchanged since the defer,
    AND no extraction has succeeded since (backend still down). Any of those three
    ceasing to hold (artifact changed, or the backend recovered) makes this False
    → the paper re-enters extraction."""
    sig = getattr(paper, "extract_deferred_sig", None)
    if sig is None:
        return False
    if sig != pdf_sig(library, paper.key):
        return False  # artifact changed (re-download) → retry
    if getattr(paper, "extract_deferred_epoch", 0) != success_epoch():
        return False  # a success happened since → backend recovered → retry
    return True
