"""Tests for the D8 reconcile sweep (services.reconcile.reconcile_once).

The sweep re-routes every non-terminal paper via ``classify`` and is the
automatic safety net for the firecrawl-md papers (md on disk, no PDF) that the
download queue's pending-only recovery scan forgets.
"""

from __future__ import annotations

import asyncio

import pytest

from papervault.library import Library
from papervault.library.services import reconcile
from papervault.library.services.reconcile import reconcile_once


class _RecordingQueue:
    """Stand-in queue that just records (key, priority) of every add()."""

    def __init__(self):
        self.added: list[tuple[str, int]] = []

    def add(self, key, priority=5):
        self.added.append((key, priority))


def _mk(lib, key, status, *, pdf=False, md=False, md_source=None, attempts=0,
        firecrawl_exhausted=False):
    """Create a paper with the given on-disk facts + status."""
    p, _ = lib.upsert({"title": f"Paper {key} title for upsert",
                       "authors": ["A"], "year": 2024,
                       "doi": f"10.1/{key}"})
    p.download_status = status
    p.extract_attempts = attempts
    p.firecrawl_pdf_hunt_exhausted = firecrawl_exhausted
    if pdf:
        lib.pdf_path(p.key).write_bytes(b"%PDF-1.4 fake pdf bytes here")
    if md:
        body = "# body\n\nlots of text\n"
        if md_source:
            body = f"---\nsource: {md_source}\n---\n\n" + body
        lib.md_path(p.key).write_text(body)
    lib.save()
    return p


def _run(coro):
    return asyncio.run(coro)


def test_path_backfill_sets_md_path_when_on_disk_but_field_null(tmp_path):
    """The Ye2016-class lag: an md is on disk but md_path is null (torn write).
    The sweep sets the canonical relative path; serving was already fine."""
    lib = Library(tmp_path)
    p = _mk(lib, "Mdnoindexpath2024", "ok", pdf=True, md=True)   # md written, md_path NOT set
    assert lib.get(p.key).md_path is None
    assert lib.has_extract(p.key, "md")
    n = _run(reconcile._path_backfill_sweep(lib))
    assert n == 1
    assert lib.get(p.key).md_path == f"extracts/md/{p.key}.md"


def test_path_backfill_never_overwrites_an_existing_path(tmp_path):
    lib = Library(tmp_path)
    p = _mk(lib, "Hasindexpath2024", "ok", pdf=True, md=True)
    lib.get(p.key).md_path = f"extracts/md/{p.key}.md"
    lib.save()
    assert _run(reconcile._path_backfill_sweep(lib)) == 0        # already set → no-op


def test_path_backfill_noop_when_no_extract_on_disk(tmp_path):
    lib = Library(tmp_path)
    _mk(lib, "Nodisk2024", "pending", pdf=False, md=False)
    assert _run(reconcile._path_backfill_sweep(lib)) == 0


def test_reconcile_routes_pending_no_pdf_to_download(tmp_path):
    lib = Library(tmp_path)
    p = _mk(lib, "Pendingnopdf2024", "pending", pdf=False)
    dq, eq = _RecordingQueue(), _RecordingQueue()

    counts = _run(reconcile_once(lib, dq, eq))

    assert (p.key, 5) in dq.added           # PRIORITY_NORMAL
    assert eq.added == []
    assert counts["download"] == 1


def test_reconcile_rescues_firecrawl_md_to_download(tmp_path):
    """The D8 headline case: md on disk (firecrawl) but no PDF + status ok →
    classify routes DOWNLOAD to hunt the real PDF. The download queue's own
    recovery scan only re-enqueues ``pending``, so reconcile is what catches
    this — otherwise the paper is forgotten by both queues."""
    lib = Library(tmp_path)
    p = _mk(lib, "Firecrawlonly2024", "ok", pdf=False, md=True,
            md_source="firecrawl")
    dq, eq = _RecordingQueue(), _RecordingQueue()

    _run(reconcile_once(lib, dq, eq))

    assert (p.key, 5) in dq.added
    assert eq.added == []


def test_reconcile_skips_exhausted_firecrawl_md(tmp_path):
    """The other half of the firecrawl story (S3 issue #1/#2): a firecrawl-md
    paper whose real-PDF hunt is already exhausted
    (``firecrawl_pdf_hunt_exhausted`` stamped on a gate-PASS) must NOT be
    re-routed to DOWNLOAD on this or any later sweep — otherwise reconcile
    re-runs the whole 18-tier download + LLM gate on it every 600s forever (the
    permanent hot-loop, and the repeated re-gating eventually deletes a good
    md). It RESTS in classify's rule-5 TERMINAL/skip while serve-safety keeps
    handing out its md. So neither queue is touched, and it counts as terminal."""
    lib = Library(tmp_path)
    _mk(lib, "Firecrawldone2024", "ok", pdf=False, md=True,
        md_source="firecrawl", firecrawl_exhausted=True)
    dq, eq = _RecordingQueue(), _RecordingQueue()

    counts = _run(reconcile_once(lib, dq, eq))

    assert dq.added == []                    # NOT re-hunted (hot-loop is dead)
    assert eq.added == []
    assert counts["terminal"] == 1
    assert counts["download"] == 0


def test_reconcile_routes_ok_pdf_no_md_to_extract(tmp_path, monkeypatch):
    """PDF on disk, no md, status ok, attempts under ceiling → EXTRACT."""
    lib = Library(tmp_path)
    p = _mk(lib, "Pdfnoextract2024", "ok", pdf=True, md=False)
    # Short paper → PRIORITY_NORMAL. Stub probe so no real subprocess/parse.
    from papervault.library import extract
    monkeypatch.setattr(extract, "pdf_probe",
                        lambda *a, **kw: extract.PDFProbe(
                            bad=False, n_pages=10, reason="ok"))
    dq, eq = _RecordingQueue(), _RecordingQueue()

    _run(reconcile_once(lib, dq, eq))

    assert (p.key, 5) in eq.added           # PRIORITY_NORMAL
    assert dq.added == []


def test_reconcile_long_paper_goes_low_priority(tmp_path, monkeypatch):
    """>90-page paper routed to EXTRACT lands in the PRIORITY_LOW slow lane."""
    lib = Library(tmp_path)
    p = _mk(lib, "Longreview2024", "ok", pdf=True, md=False)
    from papervault.library import extract
    monkeypatch.setattr(extract, "pdf_probe",
                        lambda *a, **kw: extract.PDFProbe(
                            bad=False, n_pages=200, reason="ok"))
    dq, eq = _RecordingQueue(), _RecordingQueue()

    _run(reconcile_once(lib, dq, eq))

    assert (p.key, 10) in eq.added          # PRIORITY_LOW


@pytest.mark.parametrize("status", ["extract_failed", "failed", "metadata_only"])
def test_reconcile_skips_terminal(tmp_path, status):
    """Terminal papers are never auto-revived — neither queue is touched."""
    lib = Library(tmp_path)
    _mk(lib, f"Terminal{status}2024", status, pdf=True, md=False)
    dq, eq = _RecordingQueue(), _RecordingQueue()

    counts = _run(reconcile_once(lib, dq, eq))

    assert dq.added == []
    assert eq.added == []
    assert counts["terminal"] == 1


def test_reconcile_real_pdf_already_extracted_is_terminal(tmp_path):
    """has_pdf ∧ has_md (a real PDF already extracted) = done → skipped."""
    lib = Library(tmp_path)
    _mk(lib, "Donealready2024", "ok", pdf=True, md=True)  # md, no firecrawl src
    dq, eq = _RecordingQueue(), _RecordingQueue()

    counts = _run(reconcile_once(lib, dq, eq))

    assert dq.added == []
    assert eq.added == []
    assert counts["terminal"] == 1


def test_reconcile_is_idempotent(tmp_path):
    """Running twice is safe and produces the same routing (queue add() is
    itself idempotent at the worker, so re-adds are harmless)."""
    lib = Library(tmp_path)
    _mk(lib, "Pendingidem2024", "pending", pdf=False)
    dq, eq = _RecordingQueue(), _RecordingQueue()

    c1 = _run(reconcile_once(lib, dq, eq))
    c2 = _run(reconcile_once(lib, dq, eq))

    assert c1["download"] == c2["download"] == 1
    assert len(dq.added) == 2               # added each sweep, by design


def test_reconcile_bad_record_does_not_abort_sweep(tmp_path, monkeypatch):
    """A classify() blowup on ONE paper is logged + skipped; the rest of the
    sweep still routes (SDD §6.4 — one bad record never blocks the rest)."""
    lib = Library(tmp_path)
    good = _mk(lib, "Goodone2024", "pending", pdf=False)
    bad = _mk(lib, "Badone2024", "pending", pdf=False)
    dq, eq = _RecordingQueue(), _RecordingQueue()

    real_classify = reconcile.classify

    def flaky(paper, library):
        if paper.key == bad.key:
            raise RuntimeError("corrupt record")
        return real_classify(paper, library)

    monkeypatch.setattr(reconcile, "classify", flaky)

    counts = _run(reconcile_once(lib, dq, eq))

    assert counts["errors"] == 1
    assert (good.key, 5) in dq.added        # good paper still routed
    assert counts["scanned"] == 2


# ============ by-title DOI resolution sweep (root-cause PLUG) ============


def _mk_stub(lib, key, *, abstract="we model cosmic-ray transport here", doi="",
             arxiv_id="", attempted=None):
    """A no-DOI in-domain stub (has abstract). The title embeds the key so each
    stub has a distinct normalized title."""
    p, _ = lib.upsert({"title": f"Transport study {key} of heliospheric rays",
                       "authors": ["Strauss"], "year": 2012,
                       "abstract": abstract, "doi": doi, "arxiv_id": arxiv_id,
                       "source": "ads"})
    if attempted is not None:
        p.resolve_attempted_at = attempted
    lib.save()
    return p


def test_needs_doi_resolve_predicate(tmp_path):
    lib = Library(tmp_path)
    stub = _mk_stub(lib, "Stubone")
    assert reconcile._needs_doi_resolve(stub) is True

    # Has a DOI -> not a candidate.
    has_doi = _mk_stub(lib, "Hasdoi", doi="10.1/x")
    assert reconcile._needs_doi_resolve(has_doi) is False

    # Has a usable arxiv id -> not a candidate.
    has_arxiv = _mk_stub(lib, "Hasarxiv", arxiv_id="2401.12345")
    assert reconcile._needs_doi_resolve(has_arxiv) is False

    # No abstract -> not a searchable stub.
    no_abs = _mk_stub(lib, "Noabs", abstract="")
    assert reconcile._needs_doi_resolve(no_abs) is False

    # Already attempted -> termination guard skips it.
    attempted = _mk_stub(lib, "Attempted", attempted="2026-06-03T00:00:00Z")
    assert reconcile._needs_doi_resolve(attempted) is False


def test_resolve_sweep_sets_doi_and_stamps(tmp_path, monkeypatch):
    """A successful resolve sets the DOI (re-indexed) + stamps resolve_attempted_at."""
    lib = Library(tmp_path)
    stub = _mk_stub(lib, "Hitkey")

    monkeypatch.setattr(reconcile.fetch, "_crossref_title_candidates",
                        lambda *a, **k: [{"doi": "10.1234/hit"}])
    monkeypatch.setattr(reconcile.fetch, "resolve_doi_by_title",
                        lambda *a, **k: "10.1234/hit")

    resolved = _run(reconcile._resolve_doi_sweep(lib, cap=50))
    assert resolved == 1
    p = lib.get(stub.key)
    assert p.doi == "10.1234/hit"
    assert lib.find(doi="10.1234/hit").key == stub.key
    assert p.resolve_attempted_at is not None


def test_resolve_sweep_miss_stamps_no_doi(tmp_path, monkeypatch):
    """A clean miss (query succeeded, no safe match) stamps the guard but
    writes no DOI — so it isn't re-probed every sweep."""
    lib = Library(tmp_path)
    stub = _mk_stub(lib, "Misskey")
    monkeypatch.setattr(reconcile.fetch, "_crossref_title_candidates",
                        lambda *a, **k: [])  # query OK, no items
    monkeypatch.setattr(reconcile.fetch, "resolve_doi_by_title",
                        lambda *a, **k: None)

    resolved = _run(reconcile._resolve_doi_sweep(lib, cap=50))
    assert resolved == 0
    p = lib.get(stub.key)
    assert p.doi == ""
    assert p.resolve_attempted_at is not None  # stamped -> won't re-probe


def test_resolve_sweep_transient_does_not_stamp(tmp_path, monkeypatch):
    """A transient failure leaves resolve_attempted_at None -> retried later."""
    lib = Library(tmp_path)
    stub = _mk_stub(lib, "Transientkey")
    monkeypatch.setattr(reconcile.fetch, "_crossref_title_candidates",
                        lambda *a, **k: None)  # network/429/5xx
    monkeypatch.setattr(reconcile.fetch, "resolve_doi_by_title",
                        lambda *a, **k: None)

    resolved = _run(reconcile._resolve_doi_sweep(lib, cap=50))
    assert resolved == 0
    p = lib.get(stub.key)
    assert p.doi == ""
    assert p.resolve_attempted_at is None  # NOT stamped -> retried next sweep


# ─────────────── issue #43: transport-defer skip + bounded retry ────────────
#
# A paper with a PDF but no md, status ok (classify → EXTRACT), whose extraction
# keeps failing on TRANSPORT (backend down) is left non-terminal + uncharged by
# design (C1), so reconcile used to re-enqueue the whole pending-extract set
# every sweep forever. The fix: extract_md stamps a per-artifact deferral marker;
# reconcile SKIPS a still-deferred paper, but auto-re-enqueues it when the PDF
# artifact changes (re-download) OR the backend recovers (a sibling extract
# succeeds → the success epoch advances), and keeps a bounded recovery canary.


def _mk_deferred_extract(lib, key):
    """An EXTRACT-routed paper (pdf, no md, status ok) stamped transport-deferred
    against its current PDF artifact + the current success epoch."""
    from papervault.library.services import extract_defer
    p = _mk(lib, key, "ok", pdf=True, md=False)
    extract_defer.mark_extract_deferred(p, lib)
    lib.save()
    return p


def test_reconcile_skips_transport_deferred_extract(tmp_path, monkeypatch):
    """Terminal-state skip: a deferred EXTRACT paper is NOT re-enqueued. (Canary
    disabled here to isolate the skip; the canary bound is its own test below.)"""
    from papervault.library.services import extract_defer
    extract_defer.reset_for_test()
    monkeypatch.setattr(extract_defer, "DEFER_CANARY", 0)
    lib = Library(tmp_path)
    p = _mk_deferred_extract(lib, "Deferred2024")
    dq, eq = _RecordingQueue(), _RecordingQueue()

    counts = _run(reconcile_once(lib, dq, eq))

    assert eq.added == []                       # skipped, not re-enqueued
    assert counts["extract"] == 0
    assert counts["extract_deferred"] == 1


def test_reconcile_reenqueues_when_pdf_artifact_changes(tmp_path):
    """Artifact-appearance retry: if the PDF changes (re-download lands new
    bytes → new signature), the deferral no longer matches and the paper
    re-enters extraction."""
    from papervault.library.services import extract_defer
    extract_defer.reset_for_test()
    lib = Library(tmp_path)
    p = _mk_deferred_extract(lib, "Rezdl2024")
    # A re-download replaces the PDF with different bytes → different size/mtime.
    lib.pdf_path(p.key).write_bytes(b"%PDF-1.7 a DIFFERENT, larger real pdf body " * 4)
    dq, eq = _RecordingQueue(), _RecordingQueue()

    counts = _run(reconcile_once(lib, dq, eq))

    assert [k for k, _ in eq.added] == [p.key]  # re-enqueued (artifact changed)
    assert counts["extract"] == 1
    assert counts["extract_deferred"] == 0


def test_reconcile_reenqueues_deferred_after_backend_recovery(tmp_path):
    """Backend recovery: once ANY extraction succeeds (success epoch advances),
    a deferred paper goes stale and is re-enqueued (no data loss)."""
    from papervault.library.services import extract_defer
    extract_defer.reset_for_test()
    lib = Library(tmp_path)
    p = _mk_deferred_extract(lib, "Recov2024")
    dq, eq = _RecordingQueue(), _RecordingQueue()

    # Backend comes back: a sibling extraction succeeds somewhere.
    extract_defer.note_extract_success()

    counts = _run(reconcile_once(lib, dq, eq))
    assert [k for k, _ in eq.added] == [p.key]
    assert counts["extract"] == 1
    assert counts["extract_deferred"] == 0


def test_reconcile_deferred_canary_is_bounded(tmp_path):
    """During a stable outage reconcile still promotes a bounded canary slice
    (DEFER_CANARY) to detect recovery — not the whole pending-extract set."""
    from papervault.library.services import extract_defer
    extract_defer.reset_for_test()
    lib = Library(tmp_path)
    n = extract_defer.DEFER_CANARY + 5
    for i in range(n):
        _mk_deferred_extract(lib, f"Canary{i:03d}v2024")
    dq, eq = _RecordingQueue(), _RecordingQueue()

    counts = _run(reconcile_once(lib, dq, eq))

    # Only DEFER_CANARY promoted; the rest stay deferred (skipped).
    assert counts["extract_canary"] == extract_defer.DEFER_CANARY
    assert len(eq.added) == extract_defer.DEFER_CANARY
    assert counts["extract_deferred"] == n - extract_defer.DEFER_CANARY


def test_reconcile_extract_not_deferred_is_enqueued_normally(tmp_path):
    """Control: an EXTRACT paper with NO deferral marker is enqueued as before."""
    from papervault.library.services import extract_defer
    extract_defer.reset_for_test()
    lib = Library(tmp_path)
    p = _mk(lib, "Plainextract2024", "ok", pdf=True, md=False)
    dq, eq = _RecordingQueue(), _RecordingQueue()

    counts = _run(reconcile_once(lib, dq, eq))
    assert [k for k, _ in eq.added] == [p.key]
    assert counts["extract"] == 1
    assert counts["extract_deferred"] == 0
