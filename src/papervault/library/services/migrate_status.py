"""One-time D7 ``download_status`` migration.

Pre-D7 the ``download_status`` field crammed two orthogonal things into
one cell: WHAT step the paper reached AND WHERE the bytes came from
(``"ok:arxiv"``, ``"ok:scihub"``, ``"ok:recovered"``, …), plus two ghost
states (``"text-only:firecrawl"``, ``"extract_low_quality"``). D7 splits
that into a clean routing enum (``download_status``) + a separate
provenance label (``download_source``).

``migrate_status(library)`` rewrites the in-memory records to the new
shape; the caller is responsible for ``library.save()``. It runs ONCE at
daemon startup, BEFORE any download / extract / reconcile worker touches
a paper, so every consumer downstream only ever sees the five canonical
values (see ``models.DownloadStatus``).

Mapping (legacy → new):

  | legacy download_status     | new download_status | new download_source     |
  |----------------------------|---------------------|-------------------------|
  | ``ok:<src>``               | ``ok``              | ``<src>``               |
  | ``ok:recovered``           | ``ok``              | ``recovered`` (kept)    |
  | ``ok:manual``              | ``ok``              | ``manual``              |
  | ``text-only:firecrawl``    | ``ok``              | ``firecrawl``           |
  | ``extract_low_quality``    | re-derived ↓        | unchanged               |
  | already-canonical (5 vals) | unchanged           | unchanged               |

``extract_low_quality`` is a GHOST state — it meant "the old single-engine
cascade produced suspect md". Under the new 3-engine cascade + completeness
gate (D3/D4) it has no meaning, so we clear it back into the live pipeline:

  * has a real PDF on disk → ``ok`` (classify will route it to EXTRACT to
    re-run under the current cascade; attempts reset to 0 — these papers
    never got a fair shot at the new engines).
  * no PDF but an md on disk (firecrawl-ish) → ``ok`` + source firecrawl-ish.
  * neither → ``metadata_only`` if there's an abstract, else ``failed``.

``text-only:firecrawl`` papers carry an md on disk but no PDF. For D7 they
become ``ok`` + ``source=firecrawl`` so ``classify`` routes them to
DOWNLOAD (¬has_pdf → go find the real PDF, rescuing the firecrawl backlog
D8) while the serve path keeps handing out their md text_path. The D5
completeness_gate now runs at WRITE time on freshly-scraped firecrawl text
(download._try_firecrawl_text_fallback): a stub/truncation is deleted +
demoted before it ever reaches disk. This migration only touches HISTORICAL
md that predate the gate, so it does NOT re-gate them inline — they migrate to
``ok`` and the FIRST reconcile sweep routes them DOWNLOAD → 18 tiers miss →
``_try_firecrawl_text_fallback`` idempotent re-entry → ``_gate_firecrawl_md``
re-gates the on-disk body exactly once (PASS → ok + stamp exhausted; FAIL →
deleted + terminal). No audit / re-scrape command is involved.

UN-GATED FIRECRAWL STUB + REAL PDF (state-machine blocker fix, Pass A). A small
historical population carries a firecrawl md (``md_source == "firecrawl"``,
predates the gate, never re-gated) AND a real PDF that was later downloaded,
with the one-shot ``firecrawl_pdf_hunt_exhausted`` stamp NOT set. The ¬has_pdf
re-gate path above never reaches them: classify rule (2) (``has_pdf ∧
has_md → TERMINAL``) fires first and rests the paper serving the never-gated
stub while the real PDF is never OCR'd. So BEFORE the status-value migration,
Pass A scans every paper and DELETES that stub (``_delete_ungated_firecrawl_stub``).
With the md gone, classify hits rule (4) EXTRACT and the real PDF is extracted
under the current cascade + completeness gate — the served full text becomes
the real OCR body, not the stub. Stamped (gate-PASS) firecrawl md that
legitimately coexists with a later PDF (D5 no-upgrade) is left untouched.

Idempotent: a record already in canonical shape is left untouched, so
running the migration twice (or on a half-migrated index) is safe.
"""

from __future__ import annotations

from ..models import (
    DOWNLOAD_STATUS_FAILED,
    DOWNLOAD_STATUS_METADATA_ONLY,
    DOWNLOAD_STATUS_OK,
    DOWNLOAD_STATUS_PENDING,
    DownloadStatus,
)

# The five canonical values — anything in this set is already migrated.
_CANONICAL: frozenset[str] = frozenset(DownloadStatus.__args__)  # type: ignore[attr-defined]

_LEGACY_TEXT_ONLY_FIRECRAWL = "text-only:firecrawl"
_LEGACY_LOW_QUALITY = "extract_low_quality"

# The by_rule key Pass A increments when it deletes an un-gated firecrawl stub.
# Named once here so the production save-guard (``should_persist``) and any test
# that mirrors it reference the SAME string — renaming it on one side only can no
# longer silently desync the guard (R2/redrill).
STUB_DELETION_RULE = "firecrawl_stub_over_pdf_deleted"


def should_persist(report: dict) -> bool:
    """The daemon save-guard predicate, factored out so it has exactly ONE
    definition (R2/redrill: the predicate must not be replicated inline).

    ``mcp/__main__._run_async`` calls this to decide whether to ``library.save()``
    after ``migrate_status``. It returns True when the migration produced ANY
    persistable mutation:

      * ``report["migrated"]`` > 0 — at least one ``download_status`` VALUE was
        normalized (``ok:<src>`` / ``text-only:firecrawl`` / ``extract_low_quality``
        / empty / unknown → canonical), OR
      * a Pass A stub deletion occurred (``by_rule[STUB_DELETION_RULE]`` > 0).

    Pass A (``_delete_ungated_firecrawl_stub``) clears ``md_path`` / ``md_engine`` /
    ``extract_attempts`` in memory and deletes the on-disk stub but does NOT bump
    ``migrated`` (that counter tracks download_status value migrations only). A boot
    whose SOLE mutation is a stub deletion — e.g. a row already canonical ``ok``
    carrying an un-gated firecrawl stub over a real PDF — must still save, or the
    in-memory ``md_path=None`` never persists and the reloaded ``index.json`` keeps
    pointing at the now-deleted stub (serve-safety would re-impersonate it).
    """
    if report.get("migrated"):
        return True
    return bool(report.get("by_rule", {}).get(STUB_DELETION_RULE, 0))


def _delete_ungated_firecrawl_stub(library, paper) -> bool:
    """If an UN-GATED firecrawl md coexists with a real PDF, delete the stub.

    The blocker (state-machine review): a handful of historical papers carry a
    firecrawl md (``md_source == "firecrawl"``) that predates the gate AND a
    real PDF that was later downloaded, but the one-shot static
    ``firecrawl_pdf_hunt_exhausted`` stamp is NOT set — so the firecrawl text was
    NEVER re-gated. After D7 migration ``classify`` rule (2) (``has_pdf ∧
    has_md → TERMINAL``) fires before any flag check and rests the paper
    serving the never-gated stub, while the real PDF is never OCR'd and no path
    re-gates them (reconcile skips TERMINAL; the download-side early re-gate
    only runs on the ¬has_pdf DOWNLOAD route, which this population never takes).

    Fix: delete the stub md here (the migration runs once at startup, before any
    worker). With the md gone, ``classify`` now hits rule (4) EXTRACT
    (``has_pdf ∧ ¬has_md ∧ status∈{ok,pending} ∧ attempts<MAX``) and the real
    PDF is extracted under the current cascade + completeness gate (D3), so the
    served full text is the real OCR body, not the stub. We do NOT re-gate the
    stub text (it is being replaced by the real PDF's extraction, which is
    strictly better); we simply remove the impersonation.

    Returns True if a stub was deleted (so the caller can count it).
    """
    key = paper.key
    if not library.has_pdf(key):
        return False
    if not library.has_extract(key, "md"):
        return False
    if library.md_source(key) != "firecrawl":
        return False
    if paper.firecrawl_pdf_hunt_exhausted:
        # A gate-PASS firecrawl md that legitimately coexists with a later PDF
        # (D5 no PDF-upgrade): the md is already gated, rule (2) TERMINAL is
        # correct, leave it. Only the UN-gated stub is the bug.
        return False

    try:
        library.md_path(key).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Cannot remove it — fail closed by truncating to an empty file so
        # ``has_extract(md)`` reads False (serve/classify treat md as absent),
        # matching the §4.3 unlink-fail fallback. If even that fails, leave it.
        try:
            library.md_path(key).write_text("")
        except OSError:
            return False
    # Clear the stale md provenance fields on the record so they don't lie.
    paper.md_path = None
    paper.md_engine = ""
    paper.md_engine_version = ""
    # Give the real PDF a fair extraction shot: a stale (≥MAX) attempts count
    # would make classify rule 4 fail its guard and rule 5 declare the paper
    # TERMINAL — the real PDF stuck un-extracted forever (the very black hole we
    # are escaping). Vacuous today (these stub papers have attempts=0) but keeps
    # the path trap-free, mirroring the lq→ok / unknown→ok branches.
    paper.extract_attempts = 0
    library.log({"event": "migrate_firecrawl_stub_over_pdf_deleted",
                 "key": key})
    return True


def migrate_status(library) -> dict:
    """Normalize every paper's ``download_status`` to the D7 enum in place.

    Does NOT call ``library.save()`` — the caller batches the write (and is
    expected to do so under the library write lock).

    Returns a small report ``{"total", "migrated", "by_rule": {...}}`` for
    logging / tests.
    """
    by_rule: dict[str, int] = {}
    migrated = 0
    papers = list(library._papers.values())

    # Pass A (blocker fix): an un-gated firecrawl md stub that coexists with a
    # real PDF (no exhausted stamp) impersonates the real PDF forever via
    # classify rule (2) TERMINAL. Delete the stub so the real PDF extracts. This
    # runs over EVERY paper (incl. already-canonical rows) because the stub is a
    # disk fact, not a download_status value — a row can be canonical ``ok`` yet
    # still carry the un-gated stub.
    for paper in papers:
        if _delete_ungated_firecrawl_stub(library, paper):
            by_rule[STUB_DELETION_RULE] = (
                by_rule.get(STUB_DELETION_RULE, 0) + 1)

    for paper in papers:
        status = paper.download_status or ""

        # Already canonical (incl. legacy empty → pending normalization).
        if status in _CANONICAL:
            continue
        if status == "":
            paper.download_status = DOWNLOAD_STATUS_PENDING
            by_rule["empty_to_pending"] = by_rule.get("empty_to_pending", 0) + 1
            migrated += 1
            continue

        # ok:<src> → ok + source=<src>
        if status.startswith("ok:"):
            src = status[len("ok:"):].strip()
            paper.download_status = DOWNLOAD_STATUS_OK
            # Don't clobber an already-set provenance label, but DO backfill
            # the source from the legacy suffix when it's missing (the old
            # download_source field was ~98.7% empty).
            if not (paper.download_source or "").strip() and src:
                paper.download_source = src
            by_rule["ok_split"] = by_rule.get("ok_split", 0) + 1
            migrated += 1
            continue

        # text-only:firecrawl → ok + source=firecrawl (md on disk, no PDF).
        if status == _LEGACY_TEXT_ONLY_FIRECRAWL:
            paper.download_status = DOWNLOAD_STATUS_OK
            if not (paper.download_source or "").strip():
                paper.download_source = "firecrawl"
            by_rule["firecrawl"] = by_rule.get("firecrawl", 0) + 1
            migrated += 1
            continue

        # extract_low_quality ghost → re-derive from disk facts.
        if status == _LEGACY_LOW_QUALITY:
            if library.has_pdf(paper.key):
                paper.download_status = DOWNLOAD_STATUS_OK
                # Give it a fresh shot at the new cascade.
                paper.extract_attempts = 0
                by_rule["lq_to_ok_repextract"] = by_rule.get("lq_to_ok_repextract", 0) + 1
            elif library.has_extract(paper.key, "md"):
                paper.download_status = DOWNLOAD_STATUS_OK
                if not (paper.download_source or "").strip():
                    paper.download_source = "firecrawl"
                by_rule["lq_to_ok_textonly"] = by_rule.get("lq_to_ok_textonly", 0) + 1
            elif (paper.abstract or "").strip():
                paper.download_status = DOWNLOAD_STATUS_METADATA_ONLY
                by_rule["lq_to_metadata_only"] = by_rule.get("lq_to_metadata_only", 0) + 1
            else:
                paper.download_status = DOWNLOAD_STATUS_FAILED
                by_rule["lq_to_failed"] = by_rule.get("lq_to_failed", 0) + 1
            migrated += 1
            continue

        # Unknown legacy value — be conservative, re-derive from disk so the
        # paper doesn't get stuck in a value no consumer understands.
        if library.has_pdf(paper.key):
            paper.download_status = DOWNLOAD_STATUS_OK
            # Mirror the lq→ok branch: a real PDF deserves a fair extraction
            # shot, so reset attempts. Otherwise a stale (≥MAX) attempts count
            # would make classify rule 4 fail its guard and rule 5 declare the
            # paper TERMINAL — an unextracted PDF stuck forever. Vacuous today
            # (extract_attempts is D9-new, default 0, no legacy writer) but
            # keeps the unknown→ok path symmetric and trap-free.
            paper.extract_attempts = 0
        elif library.has_extract(paper.key, "md"):
            paper.download_status = DOWNLOAD_STATUS_OK
        elif (paper.abstract or "").strip():
            paper.download_status = DOWNLOAD_STATUS_METADATA_ONLY
        else:
            paper.download_status = DOWNLOAD_STATUS_PENDING
        by_rule["unknown_rederived"] = by_rule.get("unknown_rederived", 0) + 1
        migrated += 1

    return {"total": len(papers), "migrated": migrated, "by_rule": by_rule}
