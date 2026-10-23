"""Tests for the authority-weighted return ordering (issue #46).

The prior is a SOFT RRF rank-blend layered on the return-gate relevance order in
``search_papers`` (server.py Stage 4→5), DEFAULT ON at weight 0.25; byte-identical when explicitly disabled.
These exercise the pure ``_authority_reorder`` / ``_authority_value`` helpers with
injected knobs (env-independent) so the age-normalization math is deterministic.
"""

from __future__ import annotations

import pytest

from papervault.library.mcp.server import (
    SEARCH_AUTHORITY_PRIOR,
    SEARCH_AUTHORITY_LAMBDA,
    _authority_reorder,
    _authority_value,
)


def _cand(key, *, year, citation_count):
    return {"key": key, "year": year, "citation_count": citation_count}


# ----------- default is ON -------------------------------------------------


def test_default_flag_is_on():
    """Ships DEFAULT ON: no env set → prior enabled at gentle weight 0.25."""
    assert SEARCH_AUTHORITY_PRIOR is True
    assert SEARCH_AUTHORITY_LAMBDA == 0.25


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
    """A single authority-rank advantage cannot overtake a full relevance-rank lead
    WHEN the best-authority paper sits two relevance ranks down: the top-relevance
    paper stays #1 here. (This is NOT a general 'relevance is primary' guarantee —
    see ``test_adversarial_boundary_best_authority_overtakes_top_relevance`` for the
    pool-of-4 case where a rank-2 best-authority paper DOES overtake the #1.)"""
    top = _cand("TopRel", year=2025, citation_count=0)          # best relevance, worst authority
    mid = _cand("MidRel", year=2010, citation_count=200)
    low = _cand("LowRel", year=2000, citation_count=5000)       # worst relevance, best authority
    scored = [(0.9, top), (0.8, mid), (0.7, low)]  # strictly decreasing relevance
    out = _authority_reorder(scored, enabled=True, now_year=2026, lam=0.5)
    assert out[0][1]["key"] == "TopRel"


# ----------- KNOWN-BEHAVIOR PIN: authority is NOT strictly subordinate ----------
# These two pin what the k=60 / λ=0.5 math ACTUALLY does, so an arbitration reasons
# off real numbers rather than the reassuring "soft nudge" label. They assert the
# real outcome, not a wish — if the blend math changes, they must be re-derived.


def test_adversarial_boundary_best_authority_overtakes_top_relevance():
    """ADVERSARIAL BOUNDARY (pool of 4, λ=0.5): the #1-relevance paper carries the
    WORST authority, the #2-relevance paper the BEST. Per the flat k=60 RRF math the
    rank-2 best-authority paper WINS — a 1-rank relevance lead is not enough to
    survive the full authority spread in a 4-pool. This is the known behavior we are
    pinning, not a behavior we prefer.

    Blend (k=60): X(rel 0, auth 3)=1/60+0.5/63=0.0246032 vs
    Y(rel 1, auth 0)=1/61+0.5/60=0.0247268 → Y > X.
    """
    x_top_worst = _cand("X_toprel_worstauth", year=2025, citation_count=0)     # 0/1   = 0     (worst auth)
    y_rank2_best = _cand("Y_rank2_bestauth", year=2020, citation_count=6000)   # 6000/6= 1000  (best auth)
    f1 = _cand("F1", year=2015, citation_count=100)                            # 100/11≈ 9.1
    f2 = _cand("F2", year=2010, citation_count=200)                            # 200/16= 12.5
    # strictly-decreasing relevance so rel_rank = 0,1,2,3 (X leads Y by exactly one)
    scored = [(0.9, x_top_worst), (0.8, y_rank2_best), (0.7, f2), (0.6, f1)]
    out = _authority_reorder(scored, enabled=True, now_year=2026, lam=0.5)
    keys = [c["key"] for _s, c in out]
    # Known behavior: the rank-2 best-authority paper overtakes the #1.
    assert keys[0] == "Y_rank2_bestauth"
    assert keys.index("Y_rank2_bestauth") < keys.index("X_toprel_worstauth")


def test_lambda_zero_is_identity():
    """λ=0 zeroes the authority term → the blend is pure relevance → the input order
    is returned unchanged, even for a pool the authority prior WOULD reorder at
    λ=0.5 (same adversarial pool as above)."""
    x_top_worst = _cand("X_toprel_worstauth", year=2025, citation_count=0)
    y_rank2_best = _cand("Y_rank2_bestauth", year=2020, citation_count=6000)
    f1 = _cand("F1", year=2015, citation_count=100)
    f2 = _cand("F2", year=2010, citation_count=200)
    scored = [(0.9, x_top_worst), (0.8, y_rank2_best), (0.7, f2), (0.6, f1)]
    out = _authority_reorder(scored, enabled=True, now_year=2026, lam=0.0)
    assert [c["key"] for _s, c in out] == [c["key"] for _s, c in scored]


@pytest.mark.parametrize("venue", [
    "PRL", "PRX", "RMP", "SSR", "LRSP", "PNAS", "Physical Review Letters",
    "Physical Review X", "Reviews of Modern Physics", "Science", "Science Advances",
    "Nature", "Nature Physics", "Nature Reviews Physics", "Space Science Reviews",
    "Living Reviews in Solar Physics", "Reviews of Geophysics", "Physics Reports",
    "Reports on Progress in Physics", "Annual Review of Nuclear and Particle Science",
    "Annual Review of Astronomy and Astrophysics", "Annual Review of Fluid Mechanics",
    "SIAM Review", "Proceedings of the National Academy of Sciences",
    "  PHYSICAL   REVIEW, LETTERS. ", "Space Science Reviws",
])
def test_flagship_venue_authority(venue):
    from papervault.library.services.search_pipeline import _venue_tier_multiplier

    cand = dict(venue=venue, year=2016, citation_count=40)
    assert _venue_tier_multiplier(cand) == 1.5
    assert _authority_value(cand, 2026) == 6.0


@pytest.mark.parametrize("venue", [
    "ApJ", "ApJL", "ApJS", "JGR", "GRL", "A&A", "AA", "MNRAS", "PRD", "PRC", "PRE",
    "JCAP", "JCP", "SISC", "CMAME", "ICML", "NeurIPS", "NIPS", "ICLR",
    "Astrophysical Journal", "Astrophys. J.", "Journal of Geophysical Research",
    "Geophysical Research Letters", "Astronomy and Astrophysics", "Astron. Astrophys.",
    "Monthly Notices of the Royal Astronomical Society", "Mon. Not. Roy. Astron. Soc.",
    "Physical Review D", "Phys. Rev. D", "Physical Review C", "Physical Review E",
    "Journal of Cosmology and Astroparticle Physics", "Astroparticle Physics",
    "Nuclear Physics B", "Nuclear Instruments and Methods in Physics Research",
    "Solar Physics", "Planetary and Space Science", "Journal of Computational Physics",
    "SIAM Journal on Scientific Computing", "Computer Methods in Applied Mechanics and Engineering",
    "International Conference on Machine Learning", "Neural Information Processing Systems",
    "Advances in Neural Information Processing Systems",
    "International Conference on Learning Representations",
    "Communications in Mathematics and Statistics", "Physical Reviw D",
])
def test_core_domain_venue_authority(venue):
    from papervault.library.services.search_pipeline import _venue_tier_multiplier

    cand = dict(venue=venue, year=2016, citation_count=40)
    assert _venue_tier_multiplier(cand) == 1.2
    assert _authority_value(cand, 2026) == 4.8


@pytest.mark.parametrize("prefix", [
    "10.1103/physrevlett", "10.1103/physrevx", "10.1103/revmodphys", "10.1126/science",
    "10.1038/nature", "10.1038/s41567", "10.1038/s42254", "10.1007/s11214", "10.12942/lrsp",
    "10.1007/s41116", "10.1073/pnas", "10.1016/j.physrep", "10.1088/0034-4885",
    "10.1146/annurev", "10.1137/s00361445",
])
@pytest.mark.parametrize("venue", ["", "Phys. Rev. Lett.", "Unrecognized journal typo"])
def test_flagship_doi_fallback(prefix, venue):
    from papervault.library.services.search_pipeline import _venue_tier_multiplier

    assert _venue_tier_multiplier(dict(doi=prefix.upper() + ".123", venue=venue)) == 1.5


@pytest.mark.parametrize("prefix", [
    "10.1086/", "10.3847/", "10.1029/", "10.1051/0004-6361", "10.1093/mnras",
    "10.1103/physrevd", "10.1103/physreve", "10.1103/physrevc", "10.1088/1475-7516",
    "10.1016/j.astropartphys", "10.1016/j.nuclphysb", "10.1016/j.nima", "10.1007/s11207",
    "10.1016/j.pss", "10.1016/0032-0633", "10.1016/j.jcp", "10.1016/j.cma", "10.1007/s40304",
])
@pytest.mark.parametrize("venue", ["", "Astrophys. J.", "Unrecognized journal typo"])
def test_core_domain_doi_fallback(prefix, venue):
    from papervault.library.services.search_pipeline import _venue_tier_multiplier

    assert _venue_tier_multiplier(dict(doi=prefix.upper() + "123", venue=venue)) == 1.2


@pytest.mark.parametrize("venue", [
    "AGU Fall Meeting Abstracts", "AGU Spring Meeting", "Meeting Abstracts",
    "Conference Abstracts", "COSPAR Scientific Assembly", "EGU General Assembly",
    "AAS Meeting", "APS Division of Plasma Physics",
    "Bulletin of the American Astronomical Society", "Bulletin of the American Physical Society",
    "PoS", "Proceedings of Science", "International Cosmic Ray Conference",
    "AIP Conf. Proc.", "AIP Conference Proceedings", "HAL CCSD", "CCSD",
])
def test_abstract_and_short_proceedings_authority(venue):
    from papervault.library.services.search_pipeline import _venue_tier_multiplier

    cand = dict(venue=venue, doi="10.1234/abstract", year=2016, citation_count=40)
    assert _venue_tier_multiplier(cand) == 0.5
    assert _authority_value(cand, 2026) == 2.0


@pytest.mark.parametrize("cand,multiplier", [
    ({}, 1.0), ({"venue": "", "doi": ""}, 1.0),
    ({"venue": "arXiv", "arxiv_id": "2601.12345"}, 1.0),
    ({"arxiv_id": "2601.12345"}, 1.0),
    ({"venue": "Other Journal", "doi": "10.1234/paper"}, 1.0),
    ({"venue": "Other Journal", "arxiv_id": "2601.12345"}, 1.0),
    ({"venue": "Other Journal", "inspire_record_id": "123"}, 1.0),
    ({"venue": "Other Journal", "citation_count": 5}, 1.0),
    ({"venue": "Other Journal", "citation_count": 4}, 0.5),
    ({"venue": "Other Journal", "citation_count": None}, 0.5),
    ({"venue": "ApJ"}, 1.2), ({"venue": "PRL"}, 1.5),
    ({"doi": "10.5194/EGUSPHERE-EGU26-123"}, 0.5),
    ({"venue": "EGU General Assembly", "arxiv_id": "2601.12345"}, 0.5),
])
def test_tier_defaults_and_identifier_exemptions(cand, multiplier):
    from papervault.library.services.search_pipeline import _venue_tier_multiplier

    assert _venue_tier_multiplier(cand) == multiplier


@pytest.mark.parametrize("year", [None, 0, "2016", 2026, 2027])
def test_venue_multiplier_preserves_missing_or_future_year_fallback(year):
    assert _authority_value(dict(venue="PRL", citation_count=40, year=year), 2026) == 60.0
