"""Tests for services/add_service.py — D11 enqueue-only paper ingestion.

D11: ``AddService.add`` is the front door for ``paper-library add``. It
resolves the identifier, upserts the metadata card, persists it as
``pending``, and RETURNS ``status="queued"``. It does NOT download the PDF or
run OCR synchronously — the daemon's download queue (recovery + reconcile)
owns that, picking the card up off-disk by its ``pending ∧ ¬has_pdf`` state.
"""

from __future__ import annotations

import pytest

from papervault.library import Library
from papervault.library.models import DOWNLOAD_STATUS_PENDING
from papervault.library.services.add_service import AddService


def _seed_complete(lib: Library, **kw) -> str:
    """Seed a paper that already has both PDF and txt on disk (the 'complete' state)."""
    p, _ = lib.upsert(kw)
    lib.pdf_path(p.key).write_bytes(b"%PDF-1.0 fake")
    lib.txt_path(p.key).write_text("body")
    p.pdf_path = f"pdfs/{p.key}.pdf"
    p.txt_path = f"extracts/txt/{p.key}.txt"
    return p.key


@pytest.fixture
def lib(tmp_path):
    return Library(tmp_path)


@pytest.fixture(autouse=True)
def _no_synchronous_io(monkeypatch):
    """D11 guard: ``add`` must never call download/extract synchronously.

    The add_service module no longer imports them at all (enqueue-only). If a
    future regression re-introduces a synchronous call, these poisoned stubs
    on the *source* modules make the test fail loudly instead of silently
    re-acquiring the GPU in-process.
    """
    monkeypatch.setattr(
        "papervault.library.download.download_paper",
        lambda *a, **k: pytest.fail("download_paper must not run in add (D11 enqueue-only)"),
    )
    monkeypatch.setattr(
        "papervault.library.extract.extract_md",
        lambda *a, **k: pytest.fail("extract_md must not run in add (D11 enqueue-only)"),
    )


# ----------- top-of-funnel branches ----------------------------------------


def test_empty_identifier_returns_not_found(lib):
    out = AddService(lib).add("")
    assert out["status"] == "not_found"
    assert out["key"] is None


def test_garbage_identifier_returns_not_found(lib, monkeypatch):
    """Not a DOI, not an arxiv id, not in library, resolver gives nothing."""
    out = AddService(lib).add("garbage text xyzzy")
    assert out["status"] == "not_found"


# ----------- DOI path -------------------------------------------------------


def test_doi_already_complete_short_circuits(lib, monkeypatch):
    key = _seed_complete(lib, title="Padded test paper title for validation", authors=["X"], year=2020,
                         doi="10.5000/exists")
    # If short-circuit works, fetch_by_doi must NOT be called.
    monkeypatch.setattr(
        "papervault.library.services.add_service.fetcher.fetch_by_doi",
        lambda d: pytest.fail("fetch_by_doi should not be called when complete"),
    )
    out = AddService(lib).add("10.5000/exists")
    assert out["status"] == "exists"
    assert out["key"] == key


def test_doi_metadata_only_existing_is_queued_not_downloaded(lib, monkeypatch):
    """D11: a library entry without PDF/extract is ENQUEUED (status=queued),
    not downloaded in-process. Its on-disk status stays/lands ``pending`` so
    the daemon's download queue recovery picks it up."""
    p, _ = lib.upsert({"title": "Padded test paper title for validation", "authors": ["X"], "year": 2020,
                       "doi": "10.5000/stub"})
    # No PDF / extract on disk.
    monkeypatch.setattr(
        "papervault.library.services.add_service.fetcher.fetch_by_doi",
        lambda d: pytest.fail("fetch_by_doi should not run when paper is in lib"),
    )
    out = AddService(lib).add("10.5000/stub")
    assert out["status"] == "queued"
    assert out["key"] == p.key
    # Persisted as pending so the daemon's recovery scan re-enqueues it.
    reloaded = Library(lib.root)
    assert reloaded.get(p.key).download_status == DOWNLOAD_STATUS_PENDING


def test_doi_new_paper_queued_full_pipeline(lib, monkeypatch):
    monkeypatch.setattr(
        "papervault.library.services.add_service.fetcher.fetch_by_doi",
        lambda d: {"title": "Padded test paper title for validation", "authors": ["Eve"], "year": 2024,
                   "doi": d, "abstract": "abs"},
    )
    out = AddService(lib).add("10.7000/new")
    assert out["status"] == "queued"
    assert out["key"] is not None
    # The card was actually persisted to disk (fresh Library sees it, pending).
    reloaded = Library(lib.root)
    p = reloaded.get(out["key"])
    assert p is not None
    assert p.download_status == DOWNLOAD_STATUS_PENDING


def test_doi_fetch_returns_none_returns_not_found(lib, monkeypatch):
    monkeypatch.setattr(
        "papervault.library.services.add_service.fetcher.fetch_by_doi",
        lambda d: None,
    )
    out = AddService(lib).add("10.7000/missing")
    assert out["status"] == "not_found"


# ----------- arxiv path -----------------------------------------------------


def test_arxiv_already_complete_short_circuits(lib, monkeypatch):
    key = _seed_complete(lib, title="Padded test paper title for validation", authors=["X"], year=2020,
                         arxiv_id="2020.0001")
    monkeypatch.setattr(
        "papervault.library.services.add_service.fetcher.fetch_by_arxiv",
        lambda a: pytest.fail("fetch_by_arxiv should not be called when complete"),
    )
    out = AddService(lib).add("2020.0001")
    assert out["status"] == "exists"
    assert out["key"] == key


def test_arxiv_new_paper_queued(lib, monkeypatch):
    monkeypatch.setattr(
        "papervault.library.services.add_service.fetcher.fetch_by_arxiv",
        lambda a: {"title": "Padded test paper title for validation", "authors": ["Hank"], "year": 2024,
                   "arxiv_id": a},
    )
    out = AddService(lib).add("2401.0099")
    assert out["status"] == "queued"


# ----------- existing key path ----------------------------------------------


def test_pass_existing_key_short_circuits(lib):
    key = _seed_complete(lib, title="Padded test paper title for validation", authors=["X"], year=2020,
                         doi="10.5000/k")
    out = AddService(lib).add(key)
    assert out["status"] == "exists"
    assert out["key"] == key


# ----------- fuzzy / resolver path ------------------------------------------


def _stub_resolver(svc, candidates):
    class _Stub:
        def resolve(self, *a, **k):
            return candidates
    svc._resolver = _Stub()


def test_fuzzy_one_strong_library_match_returns_exists(lib):
    p, _ = lib.upsert({"title": "Solar wind modulation", "authors": ["Potgieter"],
                       "year": 2013, "doi": "10/sw"})
    svc = AddService(lib)
    _stub_resolver(svc, [
        {"key": p.key, "title": "Solar wind modulation", "score": 0.9,
         "in_library": True},
    ])
    out = svc.add("solar wind paper")
    assert out["status"] == "exists"
    assert out["key"] == p.key


def test_fuzzy_multiple_library_matches_returns_ambiguous(lib):
    p1, _ = lib.upsert({"title": "Padded test paper title for validation", "authors": ["A"], "year": 2020, "doi": "10/a"})
    p2, _ = lib.upsert({"title": "Padded test paper title for validation", "authors": ["B"], "year": 2021, "doi": "10/b"})
    svc = AddService(lib)
    _stub_resolver(svc, [
        {"key": p1.key, "score": 0.8, "in_library": True, "title": "Padded test paper title for validation"},
        {"key": p2.key, "score": 0.7, "in_library": True, "title": "Padded test paper title for validation"},
    ])
    out = svc.add("vague text")
    assert out["status"] == "ambiguous"
    assert out["candidates"] is not None
    assert len(out["candidates"]) == 2


def test_fuzzy_no_match_returns_not_found(lib):
    svc = AddService(lib)
    _stub_resolver(svc, [])
    out = svc.add("totally unknown thing")
    assert out["status"] == "not_found"


# ----------- force_refresh --------------------------------------------------


def test_force_refresh_requeues_complete_paper(lib, monkeypatch):
    """D11: force_refresh on an already-complete paper does NOT re-download
    in-process — it re-queues (status=queued) by resetting the card to pending
    + clearing the attempt budget, so the daemon re-runs the whole pipeline."""
    key = _seed_complete(lib, title="Padded test paper title for validation", authors=["X"], year=2020,
                         doi="10.5000/exists")
    # Give it a stale terminal-ish state + spent budget to prove the reset.
    p = lib.get(key)
    p.download_status = "ok"
    p.extract_attempts = 3
    p.firecrawl_pdf_hunt_exhausted = True
    lib.save()

    monkeypatch.setattr(
        "papervault.library.services.add_service.fetcher.fetch_by_doi",
        lambda d: {"title": "Padded test paper title for validation", "authors": ["X"], "year": 2020, "doi": d},
    )
    out = AddService(lib).add("10.5000/exists", force_refresh=True)
    assert out["status"] == "queued"
    assert out["key"] == key
    reloaded = Library(lib.root)
    rp = reloaded.get(key)
    assert rp.download_status == DOWNLOAD_STATUS_PENDING
    assert rp.extract_attempts == 0
    assert rp.firecrawl_pdf_hunt_exhausted is False


def test_plain_readd_does_not_clobber_terminal_status(lib, monkeypatch):
    """A non-force re-add of an existing (incomplete) card must NOT reset a
    terminal status to pending — only a blank status gets nudged onto the
    pending track. (Reviving terminals is an explicit ``audit`` action.)"""
    p, _ = lib.upsert({"title": "Padded test paper title for validation", "authors": ["X"], "year": 2020,
                       "doi": "10.5000/term"})
    p.download_status = "failed"  # terminal, no PDF
    lib.save()
    out = AddService(lib).add("10.5000/term")
    # Resolves to the existing card; the terminal status is left untouched
    # (no-clobber). The return is an HONEST "exists" (not "queued") because
    # classify() skips terminals — the daemon will never pick this up without
    # an explicit ``audit --retry-failed`` / ``--force-refresh``.
    assert out["status"] == "exists"
    assert "terminal" in out["message"].lower()
    reloaded = Library(lib.root)
    assert reloaded.get(p.key).download_status == "failed"
