"""Phase 12.x (#118): author name canonicalization tests.

`canonicalize_author(raw)` normalizes various source-specific formats to
consistent "First [Middle] Last" form for cross-paper matching.
"""
from __future__ import annotations

import pytest

from papervault.library import Paper
from papervault.library.models import canonicalize_author


# ─────────────────────────────────────────────────────────────────────────
# Basic well-formed input
# ─────────────────────────────────────────────────────────────────────────

def test_first_last_passes_through():
    assert canonicalize_author("John Smith") == "John Smith"


def test_first_middle_last_passes_through():
    assert canonicalize_author("John M. Smith") == "John M. Smith"


def test_full_first_middle_last_passes_through():
    assert canonicalize_author("John Michael Smith") == "John Michael Smith"


# ─────────────────────────────────────────────────────────────────────────
# Initial form variations → unify on "F." (dotted)
# ─────────────────────────────────────────────────────────────────────────

def test_bare_initial_gets_dot():
    """J Smith → J. Smith (add dot to bare initial)."""
    assert canonicalize_author("J Smith") == "J. Smith"


def test_dotted_initial_preserved():
    assert canonicalize_author("J. Smith") == "J. Smith"


def test_joined_dotted_initials_split():
    """J.M. Smith → J. M. Smith (split joined initials)."""
    assert canonicalize_author("J.M. Smith") == "J. M. Smith"


def test_joined_bare_initials_split():
    """JM Smith → J. M. Smith (split bare-letter initials)."""
    assert canonicalize_author("JM Smith") == "J. M. Smith"


def test_three_joined_initials_split():
    assert canonicalize_author("JMR Smith") == "J. M. R. Smith"


# ─────────────────────────────────────────────────────────────────────────
# BibTeX comma form: "Last, First" → flip
# ─────────────────────────────────────────────────────────────────────────

def test_bibtex_comma_full_name():
    """Smith, John → John Smith."""
    assert canonicalize_author("Smith, John") == "John Smith"


def test_bibtex_comma_with_initial():
    """Smith, J. → J. Smith."""
    assert canonicalize_author("Smith, J.") == "J. Smith"


def test_bibtex_comma_with_joined_initials():
    """Smith, J.M. → J. M. Smith."""
    assert canonicalize_author("Smith, J.M.") == "J. M. Smith"


def test_bibtex_comma_first_middle():
    """Smith, John M. → John M. Smith."""
    assert canonicalize_author("Smith, John M.") == "John M. Smith"


# ─────────────────────────────────────────────────────────────────────────
# Crossref bare form: "Smith J" / "Smith JM" → swap
# ─────────────────────────────────────────────────────────────────────────

def test_crossref_bare_single_initial():
    """Smith J → J. Smith."""
    assert canonicalize_author("Smith J") == "J. Smith"


def test_crossref_bare_two_initials():
    """Smith JM → J. M. Smith."""
    assert canonicalize_author("Smith JM") == "J. M. Smith"


# ─────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────

def test_empty_input():
    assert canonicalize_author("") == ""


def test_whitespace_only():
    assert canonicalize_author("   ") == ""


def test_single_word_name():
    """Mononymic / surname-only → preserved as-is."""
    assert canonicalize_author("Smith") == "Smith"


def test_internal_whitespace_collapsed():
    assert canonicalize_author("  John   Smith  ") == "John Smith"


def test_unicode_western_diacritic_preserved():
    """Erdős with comma-flip should preserve diacritic."""
    assert canonicalize_author("Erdős, Paul") == "Paul Erdős"


# ─────────────────────────────────────────────────────────────────────────
# Paper.canonical_authors property
# ─────────────────────────────────────────────────────────────────────────

def test_paper_canonical_authors_property():
    p = Paper(key="Smith2024",
              authors=["Smith, John", "J Doe", "JM Brown"])
    canonical = p.canonical_authors
    assert canonical == ["John Smith", "J. Doe", "J. M. Brown"]


def test_paper_canonical_authors_empty_list():
    p = Paper(key="X2024", authors=[])
    assert p.canonical_authors == []


def test_paper_canonical_authors_doesnt_modify_raw():
    """Raw authors preserved for backward compat."""
    raw = ["Smith, John", "Doe J"]
    p = Paper(key="X2024", authors=raw)
    _ = p.canonical_authors  # trigger
    assert p.authors == raw  # unchanged


# ─────────────────────────────────────────────────────────────────────────
# Cross-format match: same person, different forms → same canonical
# ─────────────────────────────────────────────────────────────────────────

def test_same_person_different_forms_unified():
    """The central use case: same author in different papers from different
    sources (Crossref vs SS vs bibtex) should canonicalize to same form."""
    variants = [
        "John Smith",       # natural
        "Smith, John",      # bibtex
        # NOT identical: "J Smith" / "Smith J" canonicalize to "J. Smith"
        # which is different from "John Smith" because we don't expand
        # initials. This is a limitation — but match heuristic at jaccard
        # / lastname layer handles the cross-form join.
    ]
    canonicals = {canonicalize_author(v) for v in variants}
    assert len(canonicals) == 1, f"got: {canonicals}"
    assert canonicals.pop() == "John Smith"


def test_initial_variants_unified():
    """All bare/dotted initial forms collapse to dotted form."""
    variants = [
        "J Smith",          # bare
        "J. Smith",         # dotted
        "Smith, J.",        # bibtex with dot
        "Smith, J",         # bibtex bare
        "Smith J",          # Crossref bare
    ]
    canonicals = {canonicalize_author(v) for v in variants}
    assert len(canonicals) == 1, f"got: {canonicals}"
    assert canonicals.pop() == "J. Smith"
