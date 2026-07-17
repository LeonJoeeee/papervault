"""Tests for search.py — multi-backend search, citation chase, OpenAlex
inverted-index reconstruction, dedupe.

Phase 29: ``expand_queries`` removed (LLM intent parser handles fan-out),
``europepmc`` search backend dropped (medical bias). ``search_all`` is
now ``search_external_async`` (async + parallel backends).
"""

from __future__ import annotations

import asyncio

import pytest
import responses

from papervault.library import search


# ----------- _looks_like_review --------------------------------------------


@pytest.mark.parametrize("title,types,expected", [
    ("A Comprehensive Review of X", [], True),
    ("Survey of Methods", [], True),
    ("Forecasting X", ["Review"], True),
    ("Forecasting X", ["JournalArticle"], False),
    ("Forecasting X", [], False),
])
def test_looks_like_review(title, types, expected):
    assert search._looks_like_review({"title": title, "publication_types": types}) is expected


# ----------- search_openalex -----------------------------------------------


@responses.activate
def test_openalex_inverted_index_reconstruction():
    responses.add(
        responses.GET, "https://api.openalex.org/works",
        json={"results": [{
            "id": "https://openalex.org/W123",
            "title": "Inverted index paper",
            "publication_year": 2024,
            "abstract_inverted_index": {"hello": [0, 3], "world": [1], "lovely": [2]},
            "cited_by_count": 5,
            "doi": "https://doi.org/10.1/abc",
            "type": "article",
            # Realistic OpenAlex shape: venue nested at
            # primary_location.source.display_name (host_venue is removed).
            "primary_location": {"source": {"display_name": "J. Test"}},
            "authorships": [{"author": {"display_name": "Alice"}}],
        }]},
        status=200,
    )
    out = search.search_openalex("anything")
    assert len(out) == 1
    p = out[0]
    assert p["title"] == "Inverted index paper"
    assert p["abstract"] == "hello world lovely hello"
    assert p["doi"] == "10.1/abc"
    assert p["citation_count"] == 5
    assert p["authors"] == ["Alice"]
    assert p["venue"] == "J. Test"          # nested primary_location.source.display_name
    assert p["source"] == "openalex"


@responses.activate
def test_openalex_legacy_host_venue_fallback():
    """A legacy payload carrying only the removed top-level host_venue dict still
    resolves the venue (back-compat fallback)."""
    responses.add(
        responses.GET, "https://api.openalex.org/works",
        json={"results": [{
            "id": "https://openalex.org/W9",
            "title": "Legacy host_venue paper",
            "publication_year": 2019,
            "host_venue": {"display_name": "Legacy Journal"},
            "type": "article",
        }]},
        status=200,
    )
    out = search.search_openalex("anything")
    assert len(out) == 1
    assert out[0]["venue"] == "Legacy Journal"


@responses.activate
def test_openalex_empty_query_returns_empty():
    assert search.search_openalex("") == []


@responses.activate
def test_openalex_http_error_returns_empty():
    responses.add(responses.GET, "https://api.openalex.org/works", status=500)
    assert search.search_openalex("x") == []


# ----------- fetch_references / fetch_citations ----------------------------


@responses.activate
def test_fetch_references_happy_path():
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/SSID-A/references",
        json={"data": [
            {"citedPaper": {"title": "Cited 1", "authors": [{"name": "X"}],
                            "year": 2010, "citationCount": 100,
                            "externalIds": {"DOI": "10.9/c1"}}},
            {"citedPaper": {"title": "", "authors": []}},   # skipped: no title
        ]},
        status=200,
    )
    out = search.fetch_references("SSID-A", limit=10)
    assert len(out) == 1
    assert out[0]["title"] == "Cited 1"
    assert out[0]["doi"] == "10.9/c1"
    assert out[0]["source"] == "ss_references"


@responses.activate
def test_fetch_citations_happy_path():
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/SSID-B/citations",
        json={"data": [
            {"citingPaper": {"title": "Newer Work", "authors": [{"name": "Y"}],
                             "year": 2024, "citationCount": 1,
                             "externalIds": {"ArXiv": "2401.0099"}}},
        ]},
        status=200,
    )
    out = search.fetch_citations("SSID-B")
    assert len(out) == 1
    assert out[0]["arxiv_id"] == "2401.0099"
    assert out[0]["source"] == "ss_citations"


def test_fetch_references_empty_id_returns_empty():
    assert search.fetch_references("") == []
    assert search.fetch_citations("") == []


@responses.activate
def test_fetch_references_http_failure_returns_empty():
    responses.add(responses.GET,
                  "https://api.semanticscholar.org/graph/v1/paper/SSID-X/references",
                  status=500)
    assert search.fetch_references("SSID-X") == []


# ----------- search_external_async fan-out (V6 §3) -------------------------
# V6 (Stage D): search_external_async returns the FLAT, TAGGED, UN-deduped,
# UN-sorted concatenation of every (term, backend) pair's results (``ext_raw``).
# Every surviving node carries {_source_origin="external", term_idx, rank(native)}
# stamped BEFORE the client-side year cut. Dedup + sort + cap moved DOWNSTREAM to
# the §4a fold / §4c round-robin — they are NOT done here anymore.


def test_search_external_async_does_not_dedupe_here(monkeypatch):
    """Three backends each return the same paper. V6: search_external_async
    does NOT dedupe (the §4a fold owns identity-dedup) — all three copies come
    back, each TAGGED with _source_origin/term_idx/rank."""
    monkeypatch.setattr(
        "papervault.library.search.search_arxiv",
        lambda q, max_results=30, **k: [{"title": "Same Paper", "authors": ["A"],
                                    "year": 2020, "arxiv_id": "2001.0001",
                                    "doi": "10.5/same", "citation_count": 0,
                                    "url": ""}],
    )
    monkeypatch.setattr(
        "papervault.library.search.search_semantic_scholar",
        lambda q, max_results=30: [{"title": "Same Paper", "authors": ["A"],
                                    "year": 2020, "doi": "10.5/SAME",
                                    "arxiv_id": "2001.0001v2",
                                    "citation_count": 100, "url": ""}],
    )
    monkeypatch.setattr(
        "papervault.library.search.search_openalex",
        lambda q, max_results=30: [{"title": "Same Paper", "authors": ["A"],
                                    "year": 2020, "doi": "10.5/same",
                                    "citation_count": 50, "url": ""}],
    )
    monkeypatch.setattr("papervault.library.search.search_inspire", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_ads", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_core", lambda *a, **k: [])

    out, _degraded = asyncio.run(search.search_external_async(["kw"]))
    titles = [p["title"] for p in out]
    assert titles.count("Same Paper") == 3            # NO dedup at this layer
    # Every node is tagged for the downstream fold.
    for p in out:
        assert p["_source_origin"] == "external"
        assert p["term_idx"] == 0
        assert p["rank"] == 0                         # each backend's single hit is native rank 0


def test_search_external_async_handles_backend_exceptions(monkeypatch):
    """If one backend raises, search keeps results from the others (one dead
    backend = one empty list, never a retry-to-abort)."""
    monkeypatch.setattr(
        "papervault.library.search.search_arxiv",
        lambda q, max_results=30, **k: [{"title": "A", "authors": ["X"], "year": 2020,
                                    "arxiv_id": "x", "citation_count": 0, "url": ""}],
    )

    def kaboom(*a, **k):
        raise RuntimeError("ss is down")
    monkeypatch.setattr("papervault.library.search.search_semantic_scholar", kaboom)
    monkeypatch.setattr("papervault.library.search.search_openalex", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_inspire", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_ads", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_core", lambda *a, **k: [])

    out, degraded = asyncio.run(search.search_external_async(["kw"]))
    assert any(p["title"] == "A" for p in out)
    # The raising backend is recorded as DEGRADED (the catch-all → []), the
    # healthy arXiv pair is not.
    assert degraded.get("semantic_scholar") == 1
    assert degraded.get("arxiv", 0) == 0


def test_search_external_async_dispatches_to_all_six_backends(monkeypatch):
    """All six backends should be called; a distinct paper from each shows up
    in the tagged fan-out with the correct source label."""
    monkeypatch.setattr(
        "papervault.library.search.search_arxiv",
        lambda q, max_results=30, **k: [{"title": "Arxiv One", "authors": ["A"],
                                    "year": 2020, "arxiv_id": "2001.0001",
                                    "doi": "10.1/arxiv", "citation_count": 1,
                                    "url": ""}],
    )
    monkeypatch.setattr(
        "papervault.library.search.search_semantic_scholar",
        lambda q, max_results=30: [{"title": "SS One", "authors": ["B"],
                                    "year": 2020, "arxiv_id": "2002.0002",
                                    "doi": "10.1/ss", "citation_count": 2,
                                    "url": ""}],
    )
    monkeypatch.setattr(
        "papervault.library.search.search_openalex",
        lambda q, max_results=30: [{"title": "OpenAlex One", "authors": ["C"],
                                    "year": 2020, "doi": "10.1/oa",
                                    "citation_count": 3, "url": ""}],
    )
    monkeypatch.setattr(
        "papervault.library.search.search_inspire",
        lambda q, max_results=30: [{"title": "Inspire One", "authors": ["D"],
                                    "year": 2020, "arxiv_id": "2003.0003",
                                    "doi": "10.1/insp", "citation_count": 4,
                                    "url": ""}],
    )
    monkeypatch.setattr(
        "papervault.library.search.search_ads",
        lambda q, max_results=30: [{"title": "ADS One", "authors": ["E"],
                                    "year": 2020, "doi": "10.1/ads",
                                    "citation_count": 5, "url": ""}],
    )
    monkeypatch.setattr(
        "papervault.library.search.search_core",
        lambda q, max_results=30: [{"title": "CORE One", "authors": ["F"],
                                    "year": 2020, "doi": "10.1/core",
                                    "citation_count": 6, "url": ""}],
    )

    out, _degraded = asyncio.run(search.search_external_async(["kw"]))
    titles = {p["title"] for p in out}
    assert {"Arxiv One", "SS One", "OpenAlex One", "Inspire One", "ADS One", "CORE One"} <= titles
    sources = {p["source"] for p in out}
    assert {"arxiv", "semantic_scholar", "openalex", "inspire", "ads", "core"} <= sources


def test_search_external_async_does_not_sort_preserves_native_rank(monkeypatch):
    """V6: search_external_async does NOT flatten-sort (no review-first / citation
    re-rank). It stamps the backend's TRUE NATIVE rank and returns nodes in fetch
    order — ranking is the §4c round-robin's job, on native rank."""
    monkeypatch.setattr(
        "papervault.library.search.search_arxiv",
        lambda q, max_results=30, **k: [
            {"title": "Plain Paper", "authors": ["A"], "year": 2020,
             "arxiv_id": "p1", "citation_count": 1000, "url": ""},
            {"title": "A Survey of X", "authors": ["B"], "year": 2020,
             "arxiv_id": "p2", "citation_count": 5, "url": ""},
        ],
    )
    monkeypatch.setattr("papervault.library.search.search_semantic_scholar", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_openalex", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_inspire", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_ads", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_core", lambda *a, **k: [])

    out, _degraded = asyncio.run(search.search_external_async(["kw"]))
    # Native fetch order preserved (NO review-first sort): Plain Paper (native
    # rank 0) stays ahead of A Survey (native rank 1).
    assert [p["title"] for p in out] == ["Plain Paper", "A Survey of X"]
    assert [p["rank"] for p in out] == [0, 1]


def test_search_external_async_threads_year_window(monkeypatch):
    """V6: year_min/year_max are THREADED as params and applied client-side per
    pair AFTER native-rank stamping. An out-of-window paper is dropped; a
    None-year paper is ALWAYS KEPT."""
    monkeypatch.setattr(
        "papervault.library.search.search_arxiv",
        lambda q, max_results=30, **k: [
            {"title": "In Window", "authors": ["A"], "year": 2024,
             "arxiv_id": "w1", "citation_count": 0, "url": ""},
            {"title": "Too Old", "authors": ["B"], "year": 2005,
             "arxiv_id": "w2", "citation_count": 0, "url": ""},
            {"title": "No Year", "authors": ["C"],
             "arxiv_id": "w3", "citation_count": 0, "url": ""},
        ],
    )
    monkeypatch.setattr("papervault.library.search.search_semantic_scholar", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_openalex", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_inspire", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_ads", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_core", lambda *a, **k: [])

    out, _degraded = asyncio.run(
        search.search_external_async(["kw"], year_min=2020, year_max=None))
    titles = {p["title"] for p in out}
    assert "In Window" in titles           # 2024 inside [2020, None]
    assert "No Year" in titles             # None-year ALWAYS kept
    assert "Too Old" not in titles         # 2005 < 2020 → dropped
    # Native rank stamped BEFORE the cut: "In Window" keeps rank 0.
    in_window = next(p for p in out if p["title"] == "In Window")
    assert in_window["rank"] == 0


def test_search_external_async_empty_queries_returns_2tuple():
    """V6 (§3, R3-F1): the empty-queries early-return MUST also emit the
    (ext_raw, degraded_map) 2-tuple — an unrewritten ``return []`` would
    ValueError when the server unpacks the tuple."""
    out, degraded = asyncio.run(search.search_external_async([]))
    assert out == []
    assert dict(degraded) == {}


def test_search_external_async_records_backend_degraded(monkeypatch):
    """V6 (§3): a source raising BackendDegraded is recorded into degraded_map
    (per (term,backend) pair) AND isolated to [] for that pair — it does NOT
    abort the whole fan-out. A genuine [] is NOT recorded."""
    from papervault.library.sources.exceptions import BackendDegraded

    def degrade(*a, **k):
        raise BackendDegraded("down")

    monkeypatch.setattr("papervault.library.search.search_arxiv",
                        lambda q, max_results=30, **k: [])      # genuine empty
    monkeypatch.setattr("papervault.library.search.search_semantic_scholar", degrade)
    monkeypatch.setattr("papervault.library.search.search_openalex", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_inspire", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_ads", lambda *a, **k: [])
    monkeypatch.setattr("papervault.library.search.search_core", degrade)

    # Two terms → each degrading backend degrades on BOTH (term,backend) pairs.
    out, degraded = asyncio.run(search.search_external_async(["a", "b"]))
    assert out == []
    assert degraded.get("semantic_scholar") == 2
    assert degraded.get("core") == 2
    assert degraded.get("arxiv", 0) == 0    # genuine empty is NOT a degrade
