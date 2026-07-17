"""Tests for Phase 32 domain-membership (quarantine) support.

Covers:
  - Paper.domain_status / domain_tier defaults (None = in-domain, back-compat)
  - Library.all_papers() clean-view filtering + include_quarantined escape hatch
  - Library.set_domain_status() set / clear / unknown-key
  - persistence roundtrip through index.json
  - library.bib excludes quarantined papers (clean legal pool)
"""
from __future__ import annotations

from papervault.library import Library, Paper


def _seed(lib: Library) -> tuple[str, str]:
    """Add one in-domain + one off-domain paper; return (in_key, off_key)."""
    in_p, _ = lib.upsert({"title": "PINN inversion of SEP transport",
                          "authors": ["A. Sun"], "year": 2024, "doi": "10.1/in"})
    off_p, _ = lib.upsert({"title": "Deep learning for breast cancer recurrence",
                           "authors": ["B. Med"], "year": 2023, "doi": "10.2/off"})
    return in_p.key, off_p.key


# ---- model defaults ----

def test_domain_status_defaults_none():
    p = Paper(key="X2024", title="t")
    assert p.domain_status is None
    assert p.domain_tier is None


def test_legacy_index_entry_loads_in_domain(tmp_path):
    """A pre-Phase-32 entry (no domain fields) loads as in-domain (None)."""
    p = Paper(**{"key": "Old2020", "title": "legacy"})
    assert p.domain_status is None


# ---- clean view ----

def test_all_papers_excludes_quarantined_by_default(tmp_path):
    lib = Library(tmp_path)
    in_key, off_key = _seed(lib)
    lib.set_domain_status(off_key, "off_domain", "3")

    clean = {p.key for p in lib.all_papers()}
    assert clean == {in_key}
    full = {p.key for p in lib.all_papers(include_quarantined=True)}
    assert full == {in_key, off_key}


# ---- mutator ----

def test_set_domain_status_set_and_clear(tmp_path):
    lib = Library(tmp_path)
    in_key, off_key = _seed(lib)

    assert lib.set_domain_status(off_key, "off_domain", "3") is True
    assert lib.get(off_key).domain_status == "off_domain"
    assert lib.get(off_key).domain_tier == "3"

    # clearing un-quarantines (back to in-domain)
    assert lib.set_domain_status(off_key, None) is True
    assert lib.get(off_key).domain_status is None
    assert len(lib.all_papers()) == 2


def test_set_domain_status_unknown_key(tmp_path):
    lib = Library(tmp_path)
    assert lib.set_domain_status("NoSuchKey", "off_domain") is False


# ---- persistence ----

def test_domain_status_persists_across_reload(tmp_path):
    lib = Library(tmp_path)
    _, off_key = _seed(lib)
    lib.set_domain_status(off_key, "off_domain", "3")
    lib.save()

    reloaded = Library(tmp_path)
    assert reloaded.get(off_key).domain_status == "off_domain"
    assert reloaded.get(off_key).domain_tier == "3"
    assert len(reloaded.all_papers()) == 1
    assert len(reloaded.all_papers(include_quarantined=True)) == 2


# ---- bib clean view ----

def test_bib_excludes_quarantined(tmp_path):
    lib = Library(tmp_path)
    in_key, off_key = _seed(lib)
    lib.set_domain_status(off_key, "off_domain", "3")
    lib.save()

    bib = (tmp_path / "library.bib").read_text()
    assert in_key in bib
    assert off_key not in bib
