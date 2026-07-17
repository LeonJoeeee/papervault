"""Tests for the metadata-completeness root-cause fix.

Covers: the new Paper locator fields + bibtex rendering; the 3-state
Crossref enrich lookup (ok / miss / transient); the targeted fill-blanks
Library.enrich (with the worker-owned enriched_at guard); and the reconcile
enrich sweep that backfills + self-terminates. See the metadata-rootcause workflow.
"""
from __future__ import annotations

import asyncio

import pytest

from papervault.library import fetch
from papervault.library.models import Paper
from papervault.library.services import reconcile
from papervault.library.store import Library


def _paper(**kw):
    d = {"key": "K1", "title": "T", "authors": ["A B"], "year": 2020, "doi": "10.1/x"}
    d.update(kw)
    return Paper(**d)


# ───────────── model + bibtex ─────────────

def test_paper_has_locator_fields():
    p = _paper(volume="961", pages="57", issue="2")
    assert (p.volume, p.pages, p.issue, p.enriched_at) == ("961", "57", "2", None)


def test_bibtex_emits_volume_issue_pages():
    bib = _paper(venue="The Astrophysical Journal", volume="961", issue="2", pages="L7").to_bibtex()
    assert bib.startswith("@article")
    assert "volume = {961}" in bib
    assert "number = {2}" in bib          # BibTeX names issue "number"
    assert "pages = {L7}" in bib


def test_bibtex_article_gate_on_volume_even_without_venue():
    assert _paper(venue="", volume="961", pages="57").to_bibtex().startswith("@article")


def test_bibtex_bare_preprint_is_misc():
    p = _paper(doi="", venue="", volume="", pages="", arxiv_id="2501.00001")
    assert p.to_bibtex().startswith("@misc")


# ───────────── bib file= serve-safety (fix #2, SDD §5 I-SERVE) ─────────────


@pytest.mark.parametrize("status", ["extract_failed", "failed", "metadata_only"])
def test_bibtex_file_field_omits_txt_under_terminal_status(tmp_path, status):
    """fix #2: the bib ``file=`` txt link must obey the SAME serve-safety as
    text_path — a TERMINAL paper's leftover fat pypdf txt (the SAME PDF the gate
    judged incomplete) must NOT be handed out as a Readable raw path. Before the
    fix to_bibtex gated on disk-only has_extract and leaked it."""
    lib = Library(tmp_path)
    lib.upsert({"title": "Terminal paper on solar wind transport",
                "authors": ["X Y"], "year": 2019, "doi": "10.1/term"})
    p = lib.all_papers()[0]
    # A fat (≥ floor) leftover txt on disk, NO md.
    lib.txt_path(p.key).write_text("paywall fragment " * 60)
    assert lib.txt_path(p.key).stat().st_size >= 500
    p.download_status = status

    bib = p.to_bibtex(library=lib)
    assert str(lib.txt_path(p.key).resolve()) not in bib, \
        "terminal leftover txt leaked into bib file= field"
    # No file= field at all here (no pdf, no md, txt withheld).
    assert "file = {" not in bib


def test_bibtex_file_field_omits_subfloor_txt(tmp_path):
    """fix #2: a near-empty (sub byte-floor, scanned-PDF noise) txt is withheld
    from file= even under a NON-terminal status — it would impersonate full text."""
    lib = Library(tmp_path)
    lib.upsert({"title": "Scanned paper on cosmic ray spectra",
                "authors": ["Y Z"], "year": 2021, "doi": "10.1/scan"})
    p = lib.all_papers()[0]
    lib.txt_path(p.key).write_text("\f \n")  # scan noise, < 500 bytes
    p.download_status = "pending"

    bib = p.to_bibtex(library=lib)
    assert str(lib.txt_path(p.key).resolve()) not in bib
    assert "file = {" not in bib


def test_bibtex_file_field_serves_real_txt_under_nonterminal(tmp_path):
    """fix #2 guard: a real (≥floor) txt under a NON-terminal status IS still a
    valid file= link — the gate fail-closes terminal/thin txt, not legitimate
    in-progress txt."""
    lib = Library(tmp_path)
    lib.upsert({"title": "Pending paper on heliospheric modulation",
                "authors": ["Z W"], "year": 2024, "doi": "10.1/pend"})
    p = lib.all_papers()[0]
    lib.txt_path(p.key).write_text("real extracted body " * 60)
    p.download_status = "pending"

    bib = p.to_bibtex(library=lib)
    assert str(lib.txt_path(p.key).resolve()) in bib


def test_bibtex_file_field_md_and_pdf_always_link(tmp_path):
    """fix #2 scope: the pdf (binary) and md (gate-certified at write time) file=
    links are unconditional — only the txt link gained the serve-safety gate.
    Even under a terminal status, a present md is served (it passed D3)."""
    lib = Library(tmp_path)
    lib.upsert({"title": "A complete paper with markdown extract",
                "authors": ["M N"], "year": 2022, "doi": "10.1/md"})
    p = lib.all_papers()[0]
    lib.pdf_path(p.key).write_bytes(b"%PDF-1.0 x")
    lib.md_path(p.key).write_text("# real complete markdown body")
    p.download_status = "extract_failed"  # terminal, but md is gate-certified

    bib = p.to_bibtex(library=lib)
    assert str(lib.pdf_path(p.key).resolve()) in bib
    assert str(lib.md_path(p.key).resolve()) in bib


# ───────────── Library.enrich (targeted, fill-blanks, stamps) ─────────────

def _lib_with_paper(tmp_path, **extra):
    lib = Library(tmp_path)
    data = {"title": "A sufficiently long real paper title for the gate",
            "authors": ["A B"], "year": 2020, "doi": "10.1/x"}
    data.update(extra)
    p, _ = lib.upsert(data)
    return lib, p.key


def test_enrich_fills_blanks_only_and_stamps(tmp_path):
    lib, key = _lib_with_paper(tmp_path, venue="Existing Journal")
    changed = lib.enrich(key, {"venue": "WRONG", "volume": "961", "pages": "57"},
                         enriched_at="2026-06-02T00:00:00Z")
    p = lib.get(key)
    assert changed is True
    assert p.venue == "Existing Journal"        # pre-set venue NOT overwritten
    assert p.volume == "961" and p.pages == "57"
    assert p.enriched_at == "2026-06-02T00:00:00Z"


def test_enrich_stamps_even_on_clean_miss(tmp_path):
    lib, key = _lib_with_paper(tmp_path)
    changed = lib.enrich(key, {}, enriched_at="t")   # Crossref had nothing
    assert changed is False
    assert lib.get(key).enriched_at == "t"           # stamped anyway → not re-probed


def test_enrich_never_touches_title_or_doi(tmp_path):
    lib, key = _lib_with_paper(tmp_path)
    lib.enrich(key, {"venue": "J", "title": "HACKED", "doi": "10.9/evil"}, enriched_at="t")
    p = lib.get(key)
    assert p.title.startswith("A sufficiently long") and p.doi == "10.1/x"   # non-locator ignored
    assert p.venue == "J"


# ───────────── 3-state Crossref lookup ─────────────

class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._p = payload or {}
        self.ok = 200 <= status < 300

    def json(self):
        return self._p


def test_crossref_lookup_ok_reads_article_number_as_pages(monkeypatch):
    payload = {"message": {"container-title": ["ApJ"], "volume": "961",
                           "article-number": "57", "issue": "2"}}   # ApJ: no 'page'
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Resp(200, payload))
    status, f = fetch.crossref_enrich_lookup("10.1/x")
    assert status == "ok"
    assert (f["venue"], f["volume"], f["pages"], f["issue"]) == ("ApJ", "961", "57", "2")


def test_crossref_lookup_404_is_miss(monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Resp(404))
    assert fetch.crossref_enrich_lookup("10.1/x")[0] == "miss"


def test_crossref_lookup_429_is_transient(monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Resp(429))
    assert fetch.crossref_enrich_lookup("10.1/x")[0] == "transient"


def test_crossref_lookup_connection_error_is_transient(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(fetch.requests, "get", boom)
    assert fetch.crossref_enrich_lookup("10.1/x")[0] == "transient"


# ───────────── reconcile enrich sweep ─────────────

def test_needs_enrich_predicate(tmp_path):
    assert reconcile._needs_enrich(_paper(doi="10.1/x"))                       # missing locators
    assert not reconcile._needs_enrich(_paper(doi=""))                         # no DOI
    assert not reconcile._needs_enrich(_paper(doi="10.1/x", venue="J", volume="1", pages="2"))  # complete
    assert not reconcile._needs_enrich(_paper(doi="10.1/x", enriched_at="t"))  # already probed


def test_enrich_sweep_fills_then_self_terminates(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    lib.upsert({"title": "A sufficiently long paper title number one for gate",
                "authors": ["A B"], "year": 2020, "doi": "10.1/a"})                      # needs
    lib.upsert({"title": "A sufficiently long paper title number two for gate",
                "authors": ["C D"], "year": 2021, "doi": "",
                "venue": "J", "volume": "1", "pages": "2"})                              # no DOI → skip
    monkeypatch.setattr(fetch, "crossref_enrich_lookup",
                        lambda doi: ("ok", {"venue": "ApJ", "volume": "961", "pages": "57", "issue": ""}))

    async def go():
        # bind fresh lock/sem to THIS loop (module-level ones may be bound elsewhere)
        monkeypatch.setattr(reconcile.concurrency, "lib_write_lock", asyncio.Lock())
        monkeypatch.setattr(reconcile, "_enrich_sem", asyncio.Semaphore(2))
        n1 = await reconcile._enrich_sweep(lib, cap=10)
        n2 = await reconcile._enrich_sweep(lib, cap=10)   # all stamped now
        return n1, n2

    n1, n2 = asyncio.run(go())
    assert n1 == 1 and n2 == 0
    enriched = next(p for p in lib.all_papers() if p.doi == "10.1/a")
    assert enriched.venue == "ApJ" and enriched.volume == "961" and enriched.enriched_at


def test_enrich_sweep_transient_does_not_stamp(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    lib.upsert({"title": "A sufficiently long paper title for transient test",
                "authors": ["A B"], "year": 2020, "doi": "10.1/a"})
    monkeypatch.setattr(fetch, "crossref_enrich_lookup", lambda doi: ("transient", {}))

    async def go():
        monkeypatch.setattr(reconcile.concurrency, "lib_write_lock", asyncio.Lock())
        monkeypatch.setattr(reconcile, "_enrich_sem", asyncio.Semaphore(2))
        return await reconcile._enrich_sweep(lib, cap=10)

    assert asyncio.run(go()) == 0
    p = next(iter(lib.all_papers()))
    assert p.enriched_at is None        # transient → NOT stamped → will retry
