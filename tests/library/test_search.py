"""Tests for search.py — multi-backend search, citation chase, OpenAlex
inverted-index reconstruction, dedupe.

Phase 29: ``expand_queries`` removed (LLM intent parser handles fan-out),
``europepmc`` search backend dropped (medical bias). ``search_all`` is
now ``search_external_async`` (async + parallel backends).
"""

from __future__ import annotations

import asyncio
import logging

import pytest
import responses

from papervault.library import search
from papervault.library.sources.exceptions import BackendDegraded


@pytest.fixture(autouse=True)
def _reset_search_breaker():
    """The #115 per-backend circuit breaker keeps PROCESS-LIFETIME state keyed by
    backend name. Without a reset it leaks across tests: three tests in a row
    degrading ``semantic_scholar`` would open its circuit, and the NEXT test's
    monkeypatched backend would silently never be called. Reset before AND after
    so neither this module's tests nor another module's inherit the state."""
    search._breaker_reset()
    yield
    search._breaker_reset()


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


# ----------- per-backend circuit breaker (#115) ------------------------------
# A backend that is rate-limiting this box (keyless S2 429s on the FIRST request)
# costs one full retry-backoff chain per (term, backend) pair on EVERY search.
# After N consecutive degraded outcomes its circuit opens: the fan-out skips it
# for a cooldown window WITHOUT calling it, still counting the pair into
# ``degraded_map`` so §8's ``sources_degraded`` keeps its shape. Time is injected
# through ``search._breaker_now`` — these tests never sleep.


def _stub_backends(monkeypatch, **overrides):
    """Point every ``CAPS`` backend at a stub. Default stub = a genuine empty
    list (NOT a degrade). ``overrides`` replaces individual backends by CAPS
    name, e.g. ``_stub_backends(monkeypatch, semantic_scholar=dead)``.

    Also zeroes the per-backend min-interval pacing: ``_LAST_CALL`` is
    process-global, so each of these fan-outs would otherwise really sleep up to
    6s (CORE) before dispatching. Pacing is a separate mechanism and is not
    under test here — these breaker tests must not sleep at all."""
    monkeypatch.setattr(search, "_interval_for", lambda backend: 0.0)
    for name in search.CAPS:
        fn = overrides.get(name) or (lambda *a, **k: [])
        monkeypatch.setattr(f"papervault.library.search.search_{name}", fn)


def _paper(title):
    return {"title": title, "authors": ["A"], "year": 2020, "doi": "",
            "arxiv_id": "", "citation_count": 0, "url": ""}


def _freeze_clock(monkeypatch, start=1000.0):
    """Install an injected breaker clock; returns the mutable time holder."""
    clock = {"t": start}
    monkeypatch.setattr(search, "_breaker_now", lambda: clock["t"])
    return clock


def test_breaker_defaults_match_the_documented_contract():
    """Defaults are 3 consecutive degraded outcomes and a 3600s cooldown
    (#115 Bounds) — an operator reads these off ``.env.example``."""
    assert search.BREAKER_TRIPS_DEFAULT == 3
    assert search.BREAKER_COOLDOWN_DEFAULT_S == 3600.0


@pytest.mark.parametrize("raw,expected", [
    (None,     3),      # unset            -> default
    ("",       3),      # blank            -> default
    ("   ",    3),      # whitespace       -> default
    ("5",      5),      # honest override
    ("0",      0),      # explicit off-switch (0 = breaker disabled)
    ("-2",     3),      # negative         -> default
    ("banana", 3),      # unparseable      -> default (never a silent 0)
])
def test_breaker_threshold_env_is_parsed_defensively(monkeypatch, raw, expected):
    """A typo in ``PAPERVAULT_SEARCH_BREAKER_TRIPS`` must fall back to the
    default, never to a value that silently disables or hair-triggers the
    breaker. Only an explicit ``0`` turns it off."""
    name = "PAPERVAULT_SEARCH_BREAKER_TRIPS"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert search._env_nonneg(name, search.BREAKER_TRIPS_DEFAULT, int) == expected


@pytest.mark.parametrize("raw,expected", [
    (None,     3600.0),
    ("90",       90.0),
    ("1.5",       1.5),
    ("-1",     3600.0),
    ("banana", 3600.0),
])
def test_breaker_cooldown_env_is_parsed_defensively(monkeypatch, raw, expected):
    name = "PAPERVAULT_SEARCH_BREAKER_COOLDOWN_S"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert search._env_nonneg(name, search.BREAKER_COOLDOWN_DEFAULT_S, float) == expected


@pytest.mark.parametrize("exc", [
    BackendDegraded("S2 429 exhausted"),   # the typed degrade (the #115 case)
    RuntimeError("boom"),                  # the catch-all degrade
])
def test_n_consecutive_degraded_outcomes_open_the_circuit_and_the_next_fanout_skips(
        monkeypatch, exc):
    """(#115 done-check a) N consecutive DEGRADED outcomes open the circuit, and
    the next fan-out skips that backend WITHOUT calling its ``search_*``."""
    monkeypatch.setattr(search, "BREAKER_TRIPS", 2)
    monkeypatch.setattr(search, "BREAKER_COOLDOWN_S", 3600.0)
    _freeze_clock(monkeypatch)

    calls = []

    def dead(query, **kwargs):
        calls.append(query)
        raise exc

    _stub_backends(monkeypatch, semantic_scholar=dead)

    asyncio.run(search.search_external_async(["a"]))       # degrade 1 of 2
    assert search._breaker_open_backends() == []           # one strike is not enough
    asyncio.run(search.search_external_async(["b"]))       # degrade 2 of 2 -> OPEN
    assert search._breaker_open_backends() == ["semantic_scholar"]
    assert calls == ["a", "b"]

    asyncio.run(search.search_external_async(["c"]))       # skipped, NOT called
    assert calls == ["a", "b"]


def test_open_circuit_still_counts_degraded_and_other_backends_run_normally(monkeypatch):
    """(#115 done-check b) While the circuit is open the skipped backend still
    increments ``degraded_map`` ONCE PER TERM — §8 derives ``sources_degraded``
    from ``degraded_map[b] == T``, so an under-count would silently drop the dead
    backend out of the report — and every other backend runs normally."""
    monkeypatch.setattr(search, "BREAKER_TRIPS", 1)
    monkeypatch.setattr(search, "BREAKER_COOLDOWN_S", 3600.0)
    _freeze_clock(monkeypatch)

    calls = []

    def dead(query, **kwargs):
        calls.append(query)
        raise BackendDegraded("S2 429 exhausted")

    _stub_backends(
        monkeypatch,
        semantic_scholar=dead,
        arxiv=lambda query, max_results=30, **k: [_paper(f"arxiv for {query}")],
    )

    asyncio.run(search.search_external_async(["open-it"]))   # one degrade -> OPEN
    assert calls == ["open-it"]

    out, degraded = asyncio.run(search.search_external_async(["a", "b", "c"]))

    assert calls == ["open-it"]                       # skipped on all three terms
    assert degraded["semantic_scholar"] == 3          # == T, so §8 still reports it
    assert sorted(p["title"] for p in out) == [
        "arxiv for a", "arxiv for b", "arxiv for c",  # the healthy backend is untouched
    ]
    assert degraded.get("arxiv", 0) == 0


def test_cooldown_elapse_retries_the_backend_and_a_success_closes_the_circuit(monkeypatch):
    """(#115 done-check c) The circuit stays open for the cooldown window; once
    it elapses the next fan-out calls the backend again, and a success closes the
    circuit (no real sleep — the clock is injected)."""
    monkeypatch.setattr(search, "BREAKER_TRIPS", 1)
    monkeypatch.setattr(search, "BREAKER_COOLDOWN_S", 3600.0)
    clock = _freeze_clock(monkeypatch, start=1000.0)

    calls = []
    healthy = {"yes": False}

    def flaky(query, **kwargs):
        calls.append(query)
        if healthy["yes"]:
            return [_paper("back from the dead")]
        raise BackendDegraded("S2 429 exhausted")

    _stub_backends(monkeypatch, semantic_scholar=flaky)

    asyncio.run(search.search_external_async(["open-it"]))       # -> OPEN until 4600
    assert search._breaker_open_backends() == ["semantic_scholar"]

    clock["t"] = 4599.0                                          # one second short
    asyncio.run(search.search_external_async(["too-soon"]))
    assert calls == ["open-it"]                                  # still skipped

    clock["t"] = 4600.0                                          # cooldown elapsed
    healthy["yes"] = True
    out, _degraded = asyncio.run(search.search_external_async(["retry"]))

    assert calls == ["open-it", "retry"]                         # called again
    assert [p["title"] for p in out] == ["back from the dead"]
    assert search._breaker_open_backends() == []                 # success closed it

    asyncio.run(search.search_external_async(["after"]))         # and it stays closed
    assert calls == ["open-it", "retry", "after"]


def test_a_failed_probe_after_the_cooldown_re_arms_the_circuit(monkeypatch):
    """A backend that is still dead when the cooldown elapses must not be
    retried on every subsequent fan-out: the failed probe re-arms the cooldown."""
    monkeypatch.setattr(search, "BREAKER_TRIPS", 1)
    monkeypatch.setattr(search, "BREAKER_COOLDOWN_S", 100.0)
    clock = _freeze_clock(monkeypatch, start=0.0)

    calls = []

    def dead(query, **kwargs):
        calls.append(query)
        raise BackendDegraded("S2 429 exhausted")

    _stub_backends(monkeypatch, semantic_scholar=dead)

    asyncio.run(search.search_external_async(["open-it"]))   # OPEN until t=100
    clock["t"] = 100.0
    asyncio.run(search.search_external_async(["probe"]))     # probe runs, fails
    assert calls == ["open-it", "probe"]
    assert search._breaker_open_backends() == ["semantic_scholar"]   # re-armed

    clock["t"] = 150.0
    asyncio.run(search.search_external_async(["skipped"]))   # inside the new window
    assert calls == ["open-it", "probe"]


def test_a_success_before_the_threshold_resets_the_counter(monkeypatch):
    """(#115 done-check d) The trip counter counts CONSECUTIVE degraded outcomes:
    a success in between resets it, so an occasionally-flaky backend never opens."""
    monkeypatch.setattr(search, "BREAKER_TRIPS", 3)
    monkeypatch.setattr(search, "BREAKER_COOLDOWN_S", 3600.0)
    _freeze_clock(monkeypatch)

    calls = []
    script = ["degrade", "degrade", "ok", "degrade", "degrade"]

    def flaky(query, **kwargs):
        calls.append(query)
        if script.pop(0) == "degrade":
            raise BackendDegraded("S2 429 exhausted")
        return [_paper("fine")]

    _stub_backends(monkeypatch, semantic_scholar=flaky)

    for term in ("t1", "t2", "t3", "t4", "t5"):
        asyncio.run(search.search_external_async([term]))

    # 2 degrades, a success (counter back to 0), then 2 more degrades -> never 3
    # in a row, so the circuit never opened and every fan-out called the backend.
    assert search._breaker_open_backends() == []
    assert calls == ["t1", "t2", "t3", "t4", "t5"]


def test_breaker_is_disabled_when_the_threshold_is_zero(monkeypatch):
    """``PAPERVAULT_SEARCH_BREAKER_TRIPS=0`` is the operator's off-switch: the
    backend is called on every fan-out no matter how many times it degrades."""
    monkeypatch.setattr(search, "BREAKER_TRIPS", 0)
    _freeze_clock(monkeypatch)

    calls = []

    def dead(query, **kwargs):
        calls.append(query)
        raise BackendDegraded("S2 429 exhausted")

    _stub_backends(monkeypatch, semantic_scholar=dead)

    for term in ("t1", "t2", "t3", "t4"):
        asyncio.run(search.search_external_async([term]))

    assert search._breaker_open_backends() == []
    assert calls == ["t1", "t2", "t3", "t4"]


def test_breaker_logs_one_open_and_one_close_line_not_one_per_skipped_term(
        monkeypatch, caplog):
    """(#115 Bounds) ONE INFO line when the circuit opens and ONE when it closes —
    never one per skipped (term, backend) pair, which is exactly the log noise the
    breaker exists to remove. The per-fan-out summary line names the open circuits
    so an operator can tell 'circuit open' from 'tried and failed'."""
    monkeypatch.setattr(search, "BREAKER_TRIPS", 1)
    monkeypatch.setattr(search, "BREAKER_COOLDOWN_S", 100.0)
    clock = _freeze_clock(monkeypatch, start=0.0)

    healthy = {"yes": False}

    def flaky(query, **kwargs):
        if healthy["yes"]:
            return [_paper("ok")]
        raise BackendDegraded("S2 429 exhausted")

    _stub_backends(monkeypatch, semantic_scholar=flaky)

    caplog.set_level(logging.INFO, logger="papervault.library.search")

    # Three terms degrade in ONE fan-out -> exactly one OPEN line.
    asyncio.run(search.search_external_async(["a", "b", "c"]))
    opens = [r for r in caplog.records if "circuit OPEN" in r.getMessage()]
    assert len(opens) == 1
    assert "semantic_scholar" in opens[0].getMessage()

    # A fan-out that SKIPS three pairs adds no new OPEN lines, and its one
    # summary line reports the open circuit.
    caplog.clear()
    asyncio.run(search.search_external_async(["d", "e", "f"]))
    assert [r for r in caplog.records if "circuit OPEN" in r.getMessage()] == []
    fanout = [r.getMessage() for r in caplog.records
              if "concurrent fetches" in r.getMessage()]
    assert len(fanout) == 1
    assert "semantic_scholar" in fanout[0]

    # Cooldown elapses and three pairs succeed -> exactly one CLOSED line.
    caplog.clear()
    clock["t"] = 100.0
    healthy["yes"] = True
    asyncio.run(search.search_external_async(["g", "h", "i"]))
    closes = [r for r in caplog.records if "circuit CLOSED" in r.getMessage()]
    assert len(closes) == 1
    assert "semantic_scholar" in closes[0].getMessage()
