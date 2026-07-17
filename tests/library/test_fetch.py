"""Tests for fetch.py — DOI / arxiv direct metadata lookup. No real HTTP."""

from __future__ import annotations

import pytest
import responses

from papervault.library import fetch


# ----------- regex helpers --------------------------------------------------


@pytest.mark.parametrize("s,expected", [
    ("10.1234/abc", True),
    ("10.99999/abc.def", True),
    ("arxiv:1234.5678", False),
    ("not a doi", False),
    ("10.1/x", False),  # too short prefix
    # Boundary fix #2: the common copy-paste DOI wrappers are recognized.
    ("doi:10.1016/j.jcp.2018.10.045", True),
    ("https://doi.org/10.1016/j.jcp.2018.10.045", True),
    ("http://dx.doi.org/10.1016/j.jcp.2018.10.045", True),
    ("DOI:10.1234/abc", True),  # case-insensitive scheme
])
def test_looks_like_doi(s, expected):
    assert fetch.looks_like_doi(s) is expected


@pytest.mark.parametrize("raw,expected", [
    ("10.1234/abc", "10.1234/abc"),
    ("  10.1234/abc  ", "10.1234/abc"),
    ("doi:10.1016/j.jcp.2018.10.045", "10.1016/j.jcp.2018.10.045"),
    ("https://doi.org/10.1016/j.jcp.2018.10.045", "10.1016/j.jcp.2018.10.045"),
    ("http://dx.doi.org/10.1016/j.jcp.2018.10.045", "10.1016/j.jcp.2018.10.045"),
    ("DOI:10.1234/abc", "10.1234/abc"),
    ("not a doi", "not a doi"),  # non-DOI passes through stripped, unchanged
])
def test_normalize_doi_strips_wrappers(raw, expected):
    """Boundary fix #2: normalize_doi strips the doi: scheme and the resolver
    URL (with/without the dx. host and the scheme) to the bare 10.xxxx/ form."""
    assert fetch.normalize_doi(raw) == expected


@pytest.mark.parametrize("s,expected", [
    ("1234.5678", True),
    ("1234.56789", True),
    ("arXiv:1234.5678", True),
    ("hep-th/0012345", True),
    ("1234.5678v3", True),
    ("10.1/x", False),
    ("not arxiv", False),
])
def test_looks_like_arxiv(s, expected):
    assert fetch.looks_like_arxiv(s) is expected


@pytest.mark.parametrize("raw,expected", [
    ("1234.5678", "1234.5678"),
    ("1234.5678v3", "1234.5678"),
    ("arXiv:1234.5678v1", "1234.5678"),
    ("hep-th/0012345v2", "hep-th/0012345"),
])
def test_normalize_arxiv_strips_version(raw, expected):
    assert fetch.normalize_arxiv(raw) == expected


# ----------- fetch_by_doi ---------------------------------------------------


@responses.activate
def test_fetch_by_doi_crossref_skeleton_with_ss_gapfill():
    """D6': Crossref's skeleton wins, SS fills the rest."""
    # Crossref response — authoritative for title/year/venue/authors/doi
    responses.add(
        responses.GET,
        "https://api.crossref.org/works/10.1234/abc",
        json={
            "message": {
                "DOI": "10.1234/abc",
                "title": ["Crossref Skeleton Title"],
                "author": [{"given": "A.", "family": "Alice"},
                           {"given": "B.", "family": "Bob"}],
                "issued": {"date-parts": [[2024]]},
                "container-title": ["Nature"],
                "type": "journal-article",
                "is-referenced-by-count": 50,
                "URL": "https://doi.org/10.1234/abc",
            }
        },
        status=200,
    )
    # SS response — supplies abstract + arxiv_id + full author names
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/abc",
        json={
            "title": "SS-rendered Title (should be ignored)",
            "authors": [{"name": "Alice Lastname Alice"},
                        {"name": "Bob Middle Bob"}],
            "year": 2024,
            "abstract": "An abstract from SS.",
            "citationCount": 42,
            "externalIds": {"ArXiv": "2401.0001"},
            "publicationTypes": ["JournalArticle"],
            "venue": "Nature",
            "url": "https://example.org/x",
            "paperId": "ssid-xyz",
        },
        status=200,
    )

    out = fetch.fetch_by_doi("10.1234/abc")
    assert out is not None
    # Crossref skeleton wins on overlapping fields
    assert out["title"] == "Crossref Skeleton Title"
    assert out["year"] == 2024
    assert out["venue"] == "Nature"
    assert out["doi"] == "10.1234/abc"
    # Authors: SS preferred IF last-name set matches (Alice / Bob both match)
    assert out["authors"] == ["Alice Lastname Alice", "Bob Middle Bob"]
    # SS-only fields
    assert out["abstract"] == "An abstract from SS."
    assert out["arxiv_id"] == "2401.0001"
    assert out["paper_id"] == "ssid-xyz"
    # Source label reflects merge
    assert out["source"] == "crossref+ss"


@responses.activate
def test_fetch_by_doi_crossref_only_when_ss_missing():
    """Crossref returns data, SS 404 → Crossref-only, source='crossref'."""
    responses.add(
        responses.GET,
        "https://api.crossref.org/works/10.999/x",
        json={
            "message": {
                "DOI": "10.999/x",
                "title": ["Crossref Title"],
                "author": [{"given": "Carol", "family": "Doe"}],
                "issued": {"date-parts": [[2022, 6]]},
                "container-title": ["J. Phys."],
                "type": "journal-article",
                "is-referenced-by-count": 7,
                "URL": "https://doi.org/10.999/x",
            }
        },
        status=200,
    )
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.999/x",
        status=404,
    )

    out = fetch.fetch_by_doi("10.999/x")
    assert out is not None
    assert out["title"] == "Crossref Title"
    assert out["authors"] == ["Carol Doe"]
    assert out["year"] == 2022
    assert out["venue"] == "J. Phys."
    assert out["citation_count"] == 7
    # No abstract (Crossref didn't ship it), no arxiv_id (SS would've had it)
    assert out["abstract"] == ""
    assert out["arxiv_id"] == ""
    assert out["source"] == "crossref"


@responses.activate
def test_fetch_by_doi_ss_only_when_crossref_404():
    """Crossref doesn't have the DOI (rare; pre-1996 papers, exotic preprints)
    → fall back to SS as primary, mark source='semantic_scholar'."""
    responses.add(
        responses.GET,
        "https://api.crossref.org/works/10.preprint/oldpaper",
        status=404,
    )
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.preprint/oldpaper",
        json={
            "title": "SS Primary Title",
            "authors": [{"name": "Eve Author"}],
            "year": 1995,
            "abstract": "abs",
            "citationCount": 1,
            "externalIds": {"ArXiv": ""},
            "publicationTypes": ["JournalArticle"],
            "venue": "Old Journal",
            "url": "https://x",
            "paperId": "ssid-old",
        },
        status=200,
    )

    out = fetch.fetch_by_doi("10.preprint/oldpaper")
    assert out is not None
    assert out["title"] == "SS Primary Title"
    assert out["authors"] == ["Eve Author"]
    assert out["source"] == "semantic_scholar"


@responses.activate
def test_fetch_by_doi_returns_none_when_all_fail():
    responses.add(responses.GET,
                  "https://api.crossref.org/works/10.bad/y",
                  status=404)
    responses.add(responses.GET,
                  "https://api.semanticscholar.org/graph/v1/paper/DOI:10.bad/y",
                  status=500)
    assert fetch.fetch_by_doi("10.bad/y") is None


@responses.activate
def test_fetch_by_doi_keeps_crossref_authors_when_ss_lastnames_dont_match():
    """If SS returns wildly different authors (data error), keep Crossref's
    truncated-but-correct names rather than over-trusting SS."""
    responses.add(
        responses.GET,
        "https://api.crossref.org/works/10.match/test",
        json={
            "message": {
                "DOI": "10.match/test",
                "title": ["Title"],
                "author": [{"given": "X.", "family": "Smith"},
                           {"given": "Y.", "family": "Jones"}],
                "issued": {"date-parts": [[2023]]},
                "container-title": ["Venue"],
                "type": "journal-article",
            }
        },
        status=200,
    )
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.match/test",
        json={
            "title": "Title",
            # SS returned someone ELSE's paper — last names don't match
            "authors": [{"name": "Different Person"},
                        {"name": "Another Wrong"}],
            "year": 2023,
            "abstract": "abs",
            "citationCount": 0,
            "externalIds": {},
            "publicationTypes": [],
            "venue": "Venue",
            "url": "",
            "paperId": "ssid-wrong",
        },
        status=200,
    )

    out = fetch.fetch_by_doi("10.match/test")
    assert out is not None
    # Crossref authors kept because SS authors fail the last-name overlap check
    assert out["authors"] == ["X. Smith", "Y. Jones"]
    # Still merge source — SS data was used for other fields
    assert out["source"] == "crossref+ss"


# ----------- fetch_by_arxiv -------------------------------------------------


def test_fetch_by_arxiv_uses_search_arxiv(monkeypatch):
    canned = [{
        "title": "ArXiv Paper",
        "authors": ["Eve"],
        "abstract": "Abstract here",
        "year": "2023",
        "arxiv_id": "2301.0001",
        "url": "https://arxiv.org/abs/2301.0001",
    }]
    monkeypatch.setattr("papervault.library.fetch.search_arxiv",
                        lambda q, max_results=1: canned)

    out = fetch.fetch_by_arxiv("2301.0001v2")
    assert out is not None
    assert out["title"] == "ArXiv Paper"
    assert out["year"] == 2023
    assert out["venue"] == "arXiv"
    assert out["source"] == "arxiv"


@responses.activate
def test_fetch_by_arxiv_falls_back_to_ss(monkeypatch):
    # search_arxiv returns nothing
    monkeypatch.setattr("papervault.library.fetch.search_arxiv",
                        lambda q, max_results=1: [])
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2301.0002",
        json={
            "title": "Fallback Paper",
            "authors": [{"name": "Frank"}],
            "year": 2023,
            "abstract": "abs",
            "citationCount": 0,
            "externalIds": {"DOI": "10.5/abc"},
            "venue": None,
            "url": "https://x",
        },
        status=200,
    )

    out = fetch.fetch_by_arxiv("2301.0002")
    assert out is not None
    assert out["title"] == "Fallback Paper"
    assert out["doi"] == "10.5/abc"
    assert out["arxiv_id"] == "2301.0002"
    assert out["source"] == "semantic_scholar"


def test_fetch_by_arxiv_returns_none_when_all_fail(monkeypatch):
    monkeypatch.setattr("papervault.library.fetch.search_arxiv",
                        lambda q, max_results=1: [])
    # No SS response registered → request will raise ConnectionError under
    # responses with assert_all_requests_are_fired=False (the default off-mode
    # is to error). Use a dedicated decorator with passthrough disabled.
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rmocks:
        rmocks.add(responses.GET,
                   "https://api.semanticscholar.org/graph/v1/paper/ARXIV:9999.99999",
                   status=500)
        assert fetch.fetch_by_arxiv("9999.99999") is None
