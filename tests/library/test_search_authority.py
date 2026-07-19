"""Tests for the authority-weighted return ordering (issue #46).

The prior is a SOFT RRF rank-blend layered on the return-gate relevance order in
``search_papers`` (server.py Stage 4→5), DEFAULT OFF and byte-identical when off.
These exercise the pure ``_authority_reorder`` / ``_authority_value`` helpers with
injected knobs (env-independent) so the age-normalization math is deterministic.
"""

from __future__ import annotations

from papervault.library.mcp.server import (
    SEARCH_AUTHORITY_PRIOR,
    SEARCH_AUTHORITY_LAMBDA,
    _authority_reorder,
    _authority_value,
)


def _cand(key, *, year, citation_count):
    return {"key": key, "year": year, "citation_count": citation_count}


# ----------- default is OFF -------------------------------------------------


def test_default_flag_is_off():
    """Ships DEFAULT OFF: no env set → prior disabled, λ at the documented default."""
    assert SEARCH_AUTHORITY_PRIOR is False
    assert SEARCH_AUTHORITY_LAMBDA == 0.5


# ----------- off = byte-identical -------------------------------------------


def test_off_is_byte_identical():
    """Disabled → the SAME list object is returned, order untouched."""
    landmark = _cand("Old2000", year=2000, citation_count=5000)
    preprint = _cand("New2025", year=2025, citation_count=0)
    scored = [(0.8, preprint), (0.8, landmark)]
    out = _authority_reorder(scored, enabled=False, now_year=2026)
    assert out is scored
    assert [c["key"] for _s, c in out] == ["New2025", "Old2000"]


def test_uniform_authority_is_identity_even_when_enabled():
    """Enabled but every candidate has 0 citations → no reordering (stable identity)."""
    a = _cand("A", year=2021, citation_count=0)
    b = _cand("B", year=2020, citation_count=0)
    c = _cand("C", year=2019, citation_count=0)
    scored = [(0.9, a), (0.8, b), (0.7, c)]
    out = _authority_reorder(scored, enabled=True, now_year=2026)
    assert [x["key"] for _s, x in out] == ["A", "B", "C"]


# ----------- on: landmark rises above an equal-relevance preprint -----------


def test_landmark_rises_above_preprint_on_equal_relevance():
    """Two equal-relevance papers: the high-citation older landmark must rise above
    the 0-citation preprint even though the preprint was listed first."""
    preprint = _cand("New2025", year=2025, citation_count=0)      # 0 / 1  = 0.0
    landmark = _cand("Old2000", year=2000, citation_count=5000)   # 5000/26 ≈ 192
    scored = [(0.8, preprint), (0.8, landmark)]  # equal relevance, preprint first
    out = _authority_reorder(scored, enabled=True, now_year=2026, lam=0.5)
    assert [c["key"] for _s, c in out] == ["Old2000", "New2025"]


# ----------- age-normalization math -----------------------------------------


def test_age_normalization_math():
    """Authority value = citations per year since publication (age = now - year,
    floored at 1)."""
    older = _cand("Old2016", year=2016, citation_count=1000)   # age 10 → 100.0
    recent = _cand("New2024", year=2024, citation_count=300)   # age  2 → 150.0
    assert _authority_value(older, 2026) == 100.0
    assert _authority_value(recent, 2026) == 150.0
    # same-year paper: age floored at 1 (no divide-by-zero) → raw count
    assert _authority_value(_cand("Now2026", year=2026, citation_count=40), 2026) == 40.0


def test_age_normalization_favors_rising_work_over_raw_count():
    """A newer paper with FEWER raw citations but higher citations/year outranks an
    older heavier-cited one at equal relevance — the point of age-normalization."""
    older = _cand("Old2016", year=2016, citation_count=1000)   # 100 / yr
    recent = _cand("New2024", year=2024, citation_count=300)   # 150 / yr
    scored = [(0.7, older), (0.7, recent)]  # equal relevance, older (more raw cites) first
    out = _authority_reorder(scored, enabled=True, now_year=2026, lam=0.5)
    assert [c["key"] for _s, c in out] == ["New2024", "Old2016"]


# ----------- missing-year fallback ------------------------------------------


def test_missing_year_falls_back_to_raw_count():
    """No year → the authority value is the RAW citation count (rank fallback)."""
    assert _authority_value(_cand("NoYear", year=None, citation_count=800), 2026) == 800.0
    # non-int / zero year also routes to the fallback
    assert _authority_value({"citation_count": 12}, 2026) == 12.0


def test_missing_year_paper_ranks_by_raw_count_in_blend():
    """A missing-year, high-citation paper still lifts above an equal-relevance,
    low-authority dated paper via its raw-count fallback rank."""
    no_year = _cand("NoYear", year=None, citation_count=800)     # raw 800
    dated = _cand("Dated2020", year=2020, citation_count=100)    # 100/6 ≈ 16.7
    scored = [(0.6, dated), (0.6, no_year)]  # equal relevance, dated first
    out = _authority_reorder(scored, enabled=True, now_year=2026, lam=0.5)
    assert [c["key"] for _s, c in out] == ["NoYear", "Dated2020"]


# ----------- relevance stays primary (soft prior) ---------------------------


def test_relevance_stays_primary_across_a_full_rank_gap():
    """A single authority-rank advantage cannot overtake a full relevance-rank lead:
    the top-relevance paper stays #1 even with the WORST authority in the pool."""
    top = _cand("TopRel", year=2025, citation_count=0)          # best relevance, worst authority
    mid = _cand("MidRel", year=2010, citation_count=200)
    low = _cand("LowRel", year=2000, citation_count=5000)       # worst relevance, best authority
    scored = [(0.9, top), (0.8, mid), (0.7, low)]  # strictly decreasing relevance
    out = _authority_reorder(scored, enabled=True, now_year=2026, lam=0.5)
    assert out[0][1]["key"] == "TopRel"
