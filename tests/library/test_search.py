"""Tests for search.py — multi-backend search, citation chase, OpenAlex
inverted-index reconstruction, dedupe.

Phase 29: ``expand_queries`` removed (LLM intent parser handles fan-out),
``europepmc`` search backend dropped (medical bias). ``search_all`` is
now ``search_external_async`` (async + parallel backends).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest
import requests
import responses

from papervault.library import polite_http, search
from papervault.library.politeness_v2 import AdmissionDenied, Ledger, Scope
from papervault.library.sources.exceptions import BackendDegraded


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
def test_openalex_http_error_raises_degraded():
    responses.add(responses.GET, "https://api.openalex.org/works", status=500)
    with pytest.raises(BackendDegraded) as caught:
        search.search_openalex("x")
    assert caught.value.code == "provider_unavailable"


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


# ----------- polite-v2 fan-out pacing (#306) --------------------------------


@pytest.fixture
def polite_search(monkeypatch, fetch_pool):
    """Real adapters and configured accounting, with controlled HTTP and time."""
    fetch_pool(8)
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    monkeypatch.setenv("PAPERVAULT_POLITE_ENVELOPES", json.dumps({
        "openalex": {"analogue": "arxiv", "initial": [[1, 6]],
                     "hard_cap": [[1, 3]], "authorization": "unit test"},
    }))
    clock = SimpleNamespace(now=100.0, sleeps=[], denials=[], delay=0.01, sleep_delay=0.0,
                            sleep_fraction=1.0, blocked=None,
                            started=threading.Event(), release=threading.Event(),
                            other_finished=threading.Event())
    monkeypatch.setattr(polite_http, "_ledger", None)
    monkeypatch.setattr(polite_http, "_configuration", None)
    monkeypatch.setattr(polite_http, "_unavailable", set())
    monkeypatch.setattr(polite_http, "Ledger",
                        lambda policies: Ledger(policies, clock=lambda: clock.now))
    clock.ledger = polite_http.get_ledger()
    note_sent = clock.ledger.note_sent

    def record_send(scope):
        try:
            return note_sent(scope)
        except AdmissionDenied as error:
            clock.denials.append(error.decision.reason)
            raise

    monkeypatch.setattr(clock.ledger, "note_sent", record_send)
    # Isolate ledger windows from legacy arXiv/CORE pacing.
    monkeypatch.setattr(search, "_interval_for", lambda backend: 0.0)
    real_sleep = asyncio.sleep

    async def sleep(delay):
        clock.sleeps.append(delay)
        clock.now += delay * clock.sleep_fraction
        clock.sleep_fraction = 1.0
        await real_sleep(clock.sleep_delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    # A broken fan-out must fail promptly rather than spend seconds on retries.
    for module in ("arxiv", "semantic_scholar"):
        monkeypatch.setattr(f"papervault.library.sources.{module}.time",
                            SimpleNamespace(sleep=lambda _: None))

    def transport(url, *, params, **kwargs):
        response = requests.Response()
        response.status_code = 200
        if "export.arxiv.org" in url:
            backend, query = "arxiv", params["search_query"].removeprefix("all:")
            body = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
                    f'<id>http://arxiv.org/abs/1710.09668</id><title>{query}</title>'
                    '<summary>Test</summary><published>2017-10-26T00:00:00Z</published>'
                    '</entry></feed>')
        elif "api.semanticscholar.org" in url:
            backend, query = "semantic_scholar", params["query"]
            body = json.dumps({"data": [{"title": query, "year": 2017}]})
        else:
            assert url == "https://api.openalex.org/works"
            backend, query = "openalex", params["search"]
            body = json.dumps({"results": [{"title": query, "publication_year": 2017}]})
        if backend == clock.blocked:
            clock.started.set()
            assert clock.release.wait(5)
        time.sleep(clock.delay)  # Hold the real in-flight ledger ticket.
        if backend == "openalex" and query == "b":
            clock.other_finished.set()
        response._content = body.encode()
        return response

    monkeypatch.setattr(requests, "get", transport)
    return clock


@pytest.mark.parametrize("backend,interval", [
    ("arxiv", 3.0), ("semantic_scholar", 2.0), ("openalex", 6.0),
])
def test_polite_v2_fanout_serializes_across_searches_and_event_loops(
        monkeypatch, polite_search, backend, interval):
    monkeypatch.setattr(search, "CAPS", {backend: 5})
    observations = []

    async def run():
        with search.observe_source_gates(observations.append):
            return await asyncio.gather(search.search_external_async(["a", "b"]),
                                        search.search_external_async(["c", "d"]))

    for _ in range(2):
        for (papers, degraded), titles in zip(asyncio.run(run()), (["a", "b"], ["c", "d"])):
            assert [p["title"] for p in papers] == titles
            assert [p["term_idx"] for p in papers] == [0, 1]
            assert not degraded
        assert not search._BREAKER_OPEN_UNTIL
    assert polite_search.sleeps == [interval] * 7
    assert polite_search.denials == []
    snapshot = polite_search.ledger.snapshot(Scope(backend))
    assert (snapshot.sent, snapshot.in_flight, snapshot.positive_completions) == (8, 0, 8)
    assert [o.status for o in observations] == ["success"] * 8


def test_polite_v2_rechecks_window_after_early_wakeup(monkeypatch, polite_search):
    monkeypatch.setattr(search, "CAPS", {"semantic_scholar": 5})
    polite_search.sleep_fraction = 0.5
    papers, degraded = asyncio.run(search.search_external_async(["a", "b"]))
    assert [p["title"] for p in papers] == ["a", "b"]
    assert not degraded
    assert polite_search.sleeps == [2.0, 1.0]
    assert polite_search.denials == []


def test_polite_v2_lock_and_window_waits_do_not_use_fetch_timeout(polite_search):
    polite_search.sleep_delay = 0.25  # Each window wait exceeds the fetch timeout.

    async def run():
        return await asyncio.gather(*(
            search._fetch_one_backend(term, "semantic_scholar", 5, term_idx=i, timeout=0.2)
            for i, term in enumerate(["a", "b", "c", "d"])
        ))

    assert [batch[0]["title"] for batch in asyncio.run(run())] == ["a", "b", "c", "d"]
    assert polite_search.sleeps == [2.0] * 3
    assert polite_search.denials == []
    assert not search._BREAKER_OPEN_UNTIL


def test_polite_v2_backends_progress_independently(monkeypatch, polite_search):
    monkeypatch.setattr(search, "CAPS", {"arxiv": 5, "openalex": 5})
    polite_search.blocked = "arxiv"

    async def run():
        task = asyncio.create_task(search.search_external_async(["a", "b"]))
        try:
            assert await asyncio.to_thread(polite_search.started.wait, 5)
            assert await asyncio.to_thread(polite_search.other_finished.wait, 5)
        finally:
            polite_search.release.set()
            papers, degraded = await task
        assert len(papers) == 4
        assert not degraded

    asyncio.run(run())
    assert polite_search.denials == []


@pytest.mark.parametrize("backend", ["arxiv", "semantic_scholar", "openalex"])
@pytest.mark.parametrize("flag,stub", [("0", False), ("1", True), ("1", "same_module")])
def test_polite_v2_adds_no_wait_for_flag_off_or_injected_backends(
        monkeypatch, polite_search, backend, flag, stub):
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", flag)
    monkeypatch.setattr(search, "CAPS", {backend: 5})
    if stub:
        def injected(q, **kw):
            return [_paper(q)]
        if stub == "same_module":
            injected.__module__ = getattr(search, f"search_{backend}").__module__
        monkeypatch.setattr(search, f"search_{backend}", injected)

    def forbidden(*args):
        pytest.fail("legacy/test backend consulted the politeness ledger")

    monkeypatch.setattr(polite_search.ledger, "gate", forbidden)
    papers, degraded = asyncio.run(search.search_external_async(["a", "b"]))
    assert [p["title"] for p in papers] == ["a", "b"]
    assert not degraded
    assert polite_search.sleeps == []


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


@pytest.mark.parametrize("retryability", ["permanent", "transient_no_delay"])
def test_item_scoped_failures_keep_receipts_without_opening_service_breaker(monkeypatch, retryability):
    monkeypatch.setattr(search, "BREAKER_TRIPS", 2)
    calls = []

    def backend(query, **kwargs):
        calls.append(query)
        if query.startswith("bad"):
            raise BackendDegraded("query rejected", code="invalid_request", scope="item",
                                  retryability=retryability)
        return [_paper("Valid later query")]

    _stub_backends(monkeypatch, semantic_scholar=backend)
    degraded = collections.Counter()
    observations = []

    async def run():
        with search.observe_source_gates(observations.append):
            for i, term in enumerate(("bad one", "bad two")):
                assert await search._fetch_one_backend(
                    term, "semantic_scholar", 5, term_idx=i, degraded_map=degraded) == []
            return await search._fetch_one_backend("valid", "semantic_scholar", 5, term_idx=2)

    results = asyncio.run(run())
    assert [paper["title"] for paper in results] == ["Valid later query"]
    assert calls == ["bad one", "bad two", "valid"]
    assert degraded == {"semantic_scholar": 2}
    assert [(item.status, item.scope, item.code) for item in observations[:2]] == [
        ("degraded", "item", "invalid_request"), ("degraded", "item", "invalid_request")]
    assert search._breaker_open_backends() == []


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


def test_fanout_log_has_no_parenthetical_when_no_circuits_are_open(monkeypatch, caplog):
    _stub_backends(monkeypatch)
    caplog.set_level(logging.INFO, logger="papervault.library.search")

    asyncio.run(search.search_external_async(["a", "b", "c"]))

    fanout = [r.getMessage() for r in caplog.records
              if "concurrent fetches" in r.getMessage()]
    assert fanout == [
        "search_external_async: 3 terms x 6 backends = 18 concurrent fetches",
    ]


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
    assert fanout == [
        "search_external_async: 3 terms x 6 backends = 15 concurrent fetches "
        "(circuit open, skipped: ['semantic_scholar'])",
    ]

    # Cooldown elapses and three pairs succeed -> exactly one CLOSED line.
    caplog.clear()
    clock["t"] = 100.0
    healthy["yes"] = True
    asyncio.run(search.search_external_async(["g", "h", "i"]))
    closes = [r for r in caplog.records if "circuit CLOSED" in r.getMessage()]
    assert len(closes) == 1
    assert "semantic_scholar" in closes[0].getMessage()


# ----------- bounded search-fetch executor (#99) ------------------------------
# The (term, backend) fan-out used to run on the loop's DEFAULT executor, shared
# with rerank / embedding / get_paper — one search_papers burst of 30-48 blocking
# fetches could starve them. The fan-out now runs on its own bounded pool.


@pytest.fixture
def fetch_pool(monkeypatch):
    """A fresh search-fetch pool of a chosen size, shut down after the test."""
    def _make(workers):
        monkeypatch.setattr(search, "FETCH_WORKERS", workers)
        monkeypatch.setattr(search, "_FETCH_EXECUTOR", None)
    yield _make
    if search._FETCH_EXECUTOR is not None:
        search._FETCH_EXECUTOR.shutdown(wait=True)


def test_search_fetches_run_on_the_bounded_search_executor(monkeypatch, fetch_pool):
    import threading

    fetch_pool(4)
    threads = []

    def record(*a, **k):
        threads.append(threading.current_thread().name)
        return []

    _stub_backends(monkeypatch, **{name: record for name in search.CAPS})
    asyncio.run(search.search_external_async(["a", "b"]))
    assert len(threads) == 2 * len(search.CAPS)
    assert all(n.startswith(search.FETCH_THREAD_PREFIX) for n in threads), threads


def test_search_fetch_executor_bounds_concurrency(monkeypatch, fetch_pool):
    import threading
    import time as _time

    fetch_pool(2)
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def slow(*a, **k):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        _time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return []

    _stub_backends(monkeypatch, **{name: slow for name in search.CAPS})
    out, degraded = asyncio.run(search.search_external_async(["a", "b"]))
    assert out == [] and sum(degraded.values()) == 0
    assert state["max"] <= 2
    assert search._FETCH_EXECUTOR._max_workers == 2


def test_executor_queue_wait_is_not_charged_to_the_fetch_timeout(monkeypatch, fetch_pool):
    """Like the pacing wait, time spent QUEUED for a pool thread must not count
    against the per-fetch timeout: a healthy backend that merely waited for a
    free thread is not DEGRADED (and must not feed the #115 breaker)."""
    import time as _time

    fetch_pool(1)
    monkeypatch.setattr(search, "_interval_for", lambda backend: 0.0)

    def takes_300ms(q, **k):
        _time.sleep(0.3)
        return [_paper(q)]

    monkeypatch.setattr("papervault.library.search.search_inspire", takes_300ms)

    async def two_fetches():
        degraded = collections.Counter()
        res = await asyncio.gather(*[
            search._fetch_one_backend(q, "inspire", 5, term_idx=i,
                                      degraded_map=degraded, timeout=0.5)
            for i, q in enumerate(["first", "second"])
        ])
        return res, degraded

    res, degraded = asyncio.run(two_fetches())
    # The second fetch queued ~0.3 s then ran 0.3 s: 0.6 s since submission.
    assert [r[0]["title"] for r in res] == ["first", "second"]
    assert sum(degraded.values()) == 0


def test_cancelled_fetch_still_queued_never_runs(monkeypatch, fetch_pool):
    import threading

    fetch_pool(1)
    monkeypatch.setattr(search, "_interval_for", lambda backend: 0.0)
    release = threading.Event()
    ran = []

    def blocker(q, **k):
        ran.append(q)
        release.wait(5)
        return []

    monkeypatch.setattr("papervault.library.search.search_inspire", blocker)

    async def scenario():
        first = asyncio.create_task(search._fetch_one_backend("first", "inspire", 5, term_idx=0))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(search._fetch_one_backend("second", "inspire", 5, term_idx=1))
        await asyncio.sleep(0.05)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        return await first

    assert asyncio.run(scenario()) == []
    search._FETCH_EXECUTOR.shutdown(wait=True)
    assert ran == ["first"]                    # the cancelled, queued fetch never ran


def test_fetch_workers_default_matches_the_documented_contract():
    assert search.FETCH_WORKERS_DEFAULT == 16


@pytest.mark.parametrize("raw,expected", [
    (None,     16),     # unset       -> default
    ("",       16),     # blank       -> default
    ("8",       8),     # honest override
    ("0",      16),     # a pool cannot have zero workers -> default
    ("-3",     16),     # negative    -> default
    ("banana", 16),     # unparseable -> default
])
def test_fetch_workers_env_is_parsed_defensively(monkeypatch, raw, expected):
    name = "PAPERVAULT_SEARCH_FETCH_WORKERS"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert search._fetch_workers_from_env() == expected


# ----------- outer deadline on the wait for a fetch-pool thread (#137) -----------
# ``requests``' timeout=30 is per read, not total: hung backends can hold every
# pool thread, and a pair still QUEUED for a thread used to wait forever, so
# ``search_papers`` stalled instead of degrading.


def test_fetch_queued_past_the_deadline_degrades_without_calling_the_backend(
        monkeypatch, fetch_pool, caplog):
    import threading

    fetch_pool(1)
    monkeypatch.setattr(search, "_interval_for", lambda backend: 0.0)
    holding = threading.Event()
    release = threading.Event()
    called = []

    def hung(q, **k):                     # a backend whose socket never gives up
        holding.set()
        release.wait()
        return []

    def healthy(q, **k):
        called.append(q)
        return [_paper(q)]

    monkeypatch.setattr("papervault.library.search.search_ads", hung)
    monkeypatch.setattr("papervault.library.search.search_inspire", healthy)

    async def scenario():
        degraded = collections.Counter()
        occupier = asyncio.create_task(
            search._fetch_one_backend("h", "ads", 5, term_idx=0, timeout=60))
        try:
            assert await asyncio.to_thread(holding.wait, 5)
            # The short deadline applies to the queued fetch, not occupier startup.
            monkeypatch.setattr(search, "FETCH_QUEUE_TIMEOUT_S", 0.05)
            out = await search._fetch_one_backend("q", "inspire", 5, term_idx=1,
                                                  degraded_map=degraded)
            # fut.cancel() forwards cancellation to the executor on the next loop
            # turn. Let that callback withdraw the job before freeing the worker.
            await asyncio.sleep(0)
        finally:
            release.set()
            await occupier
            # Drain queued work while the loop is alive, so a missing cancellation
            # would call healthy rather than fail against a closed event loop.
            await asyncio.get_running_loop().run_in_executor(
                search._fetch_executor(), lambda: None)
        return out, degraded

    with caplog.at_level(logging.WARNING, logger="papervault.library.search"):
        out, degraded = asyncio.run(scenario())
    search._FETCH_EXECUTOR.shutdown(wait=True)
    assert out == []                                   # degraded like a timeout
    assert degraded["inspire"] == 1
    assert called == []                                # withdrawn, never called
    # It was never called, so it is not evidence against the backend: the #115
    # breaker is fed by the backend's own failures, not by pool starvation.
    assert "inspire" not in search._BREAKER_FAILS
    msgs = [r.getMessage() for r in caplog.records]
    assert any("inspire" in m and "no search-fetch thread" in m for m in msgs), msgs


def test_fetch_queue_timeout_default_matches_the_documented_contract():
    assert search.FETCH_QUEUE_TIMEOUT_DEFAULT_S == 120.0


@pytest.mark.parametrize("raw,expected", [
    (None,     120.0),    # unset       -> default
    ("",       120.0),    # blank       -> default
    ("45",      45.0),    # honest override
    ("0",      120.0),    # a zero deadline would refuse every fetch -> default
    ("-3",     120.0),    # negative    -> default
    ("banana", 120.0),    # unparseable -> default
])
def test_fetch_queue_timeout_env_is_parsed_defensively(monkeypatch, raw, expected):
    name = "PAPERVAULT_SEARCH_FETCH_QUEUE_TIMEOUT_S"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert search._fetch_queue_timeout_from_env() == expected


# ----------- polite-v2 denial isolation and upstream failure preservation (#308) ---


@pytest.mark.parametrize("wrapped", [False, True])
def test_local_politeness_denial_does_not_feed_breaker(monkeypatch, wrapped):
    def denied(*args, **kwargs):
        error = polite_http.PolitenessDenied("local admission denied")
        if wrapped:
            from papervault.library.sources.exceptions import backend_degraded
            raise backend_degraded(error) from error
        raise error

    monkeypatch.setattr(search, "search_semantic_scholar", denied)
    monkeypatch.setattr(search, "_interval_for", lambda _: 0.0)
    monkeypatch.setattr(search, "BREAKER_TRIPS", 3)
    search._BREAKER_FAILS["semantic_scholar"] = 2
    degraded, observations = collections.Counter(), []

    async def run():
        with search.observe_source_gates(observations.append):
            for i in range(4):
                assert await search._fetch_one_backend(
                    "solar wind", "semantic_scholar", 5, term_idx=i,
                    degraded_map=degraded) == []

    asyncio.run(run())
    assert degraded == {"semantic_scholar": 4}
    assert [o.status for o in observations] == ["degraded"] * 4
    assert [o.code for o in observations] == ["transport_error"] * 4
    assert search._BREAKER_FAILS == {"semantic_scholar": 2}
    assert search._BREAKER_OPEN_UNTIL == {}


@pytest.mark.parametrize("backend", ["arxiv", "semantic_scholar", "openalex"])
@pytest.mark.parametrize("reason", ["cooldown", "disabled"])
def test_polite_v2_terminal_gate_denial_never_submits_fetch(
        monkeypatch, polite_search, backend, reason):
    if reason == "cooldown":
        polite_search.ledger.enter_cooldown(Scope(backend), "unit test")
    else:
        monkeypatch.setattr(search, "get_ledger", lambda: Ledger({}))
    calls = []
    original = search._run_fetch

    async def record(*args, **kwargs):
        calls.append(args)
        return await original(*args, **kwargs)

    monkeypatch.setattr(search, "_run_fetch", record)
    monkeypatch.setattr(search, "BREAKER_TRIPS", 1)
    degraded, observations = collections.Counter(), []

    async def run():
        with search.observe_source_gates(observations.append):
            return await search._fetch_one_backend(
                "solar wind", backend, 5, term_idx=0, degraded_map=degraded)

    assert asyncio.run(run()) == []
    assert calls == []
    assert degraded == {backend: 1}
    assert [(o.status, o.code) for o in observations] == [("degraded", "transport_error")]
    assert search._BREAKER_FAILS == {}
    assert search._BREAKER_OPEN_UNTIL == {}


@pytest.mark.parametrize("backend", ["arxiv", "semantic_scholar", "openalex"])
def test_polite_v2_concurrency_waits_for_admission(monkeypatch, polite_search, backend):
    scope = Scope(backend)
    ticket = polite_search.ledger.note_sent(scope)
    polite_search.now += 10
    original_sleep = asyncio.sleep

    async def release_after_wait(delay):
        await original_sleep(delay)
        polite_search.ledger.note_result(ticket, status_code=200)

    monkeypatch.setattr(asyncio, "sleep", release_after_wait)
    degraded = collections.Counter()
    result = asyncio.run(search._fetch_one_backend(
        "solar wind", backend, 5, term_idx=0, degraded_map=degraded))
    assert [p["title"] for p in result] == ["solar wind"]
    assert polite_search.sleeps == [0.05]
    assert polite_search.denials == []
    assert degraded == {}
    assert search._BREAKER_OPEN_UNTIL == {}


@pytest.mark.parametrize("backend", ["arxiv", "semantic_scholar", "openalex"])
@pytest.mark.parametrize("late_release", [False, True])
def test_polite_v2_concurrency_deadline_never_submits_fetch(
        monkeypatch, polite_search, backend, late_release):
    # Give the occupied scope no window wait after its active ticket is released.
    ticket = polite_search.ledger.note_sent(Scope(backend))
    polite_search.now += 10
    monkeypatch.setattr(search, "time", SimpleNamespace(monotonic=lambda: polite_search.now))
    monkeypatch.setattr(search, "FETCH_QUEUE_TIMEOUT_S", 0.09)
    monkeypatch.setattr(search, "BREAKER_TRIPS", 1)
    original_sleep = asyncio.sleep
    released = False

    async def release_after_deadline(delay):
        nonlocal released
        await original_sleep(delay)
        if late_release and len(polite_search.sleeps) == 2:
            polite_search.ledger.note_result(ticket, status_code=200)
            released = True

    monkeypatch.setattr(asyncio, "sleep", release_after_deadline)
    calls = []
    original = search._run_fetch

    async def record(*args, **kwargs):
        calls.append(args)
        return await original(*args, **kwargs)

    monkeypatch.setattr(search, "_run_fetch", record)
    degraded, observations = collections.Counter(), []

    async def run():
        with search.observe_source_gates(observations.append):
            return await search._fetch_one_backend(
                "solar wind", backend, 5, term_idx=0, degraded_map=degraded)

    try:
        assert asyncio.run(run()) == []
    finally:
        if not released:
            polite_search.ledger.note_result(ticket, status_code=200)
    assert calls == []
    assert polite_search.sleeps == [0.05, 0.05]
    assert degraded == {backend: 1}
    assert [(o.status, o.code) for o in observations] == [("skipped", "fetch_queue_timeout")]
    assert search._BREAKER_FAILS == {}
    assert search._BREAKER_OPEN_UNTIL == {}


@pytest.mark.parametrize("backend", ["arxiv", "semantic_scholar", "openalex"])
def test_polite_v2_429_stays_rate_limited_and_cooldown_does_not_trip_breaker(
        monkeypatch, polite_search, backend):
    sends, sleeps = [], []

    def rate_limited(url, **kwargs):
        sends.append(url)
        response = requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = "17"
        return response

    monkeypatch.setattr(requests, "get", rate_limited)
    monkeypatch.setattr(search, "BREAKER_TRIPS", 2)
    if backend == "openalex":
        monkeypatch.setattr(search, "time", SimpleNamespace(
            monotonic=time.monotonic, sleep=sleeps.append))
    else:
        monkeypatch.setattr(f"papervault.library.sources.{backend}.time",
                            SimpleNamespace(sleep=sleeps.append))
    degraded, observations = collections.Counter(), []

    async def run():
        with search.observe_source_gates(observations.append):
            for i in range(4):
                assert await search._fetch_one_backend(
                    "solar wind", backend, 5, term_idx=i, degraded_map=degraded) == []

    asyncio.run(run())
    assert len(sends) == 1
    assert sleeps == []
    assert degraded == {backend: 4}
    assert observations[0].code == "rate_limited"
    assert observations[0].retry_after_s == 17.0
    assert [o.status for o in observations] == ["degraded"] * 4
    assert search._BREAKER_FAILS == {backend: 1}
    assert search._BREAKER_OPEN_UNTIL == {}


@pytest.mark.parametrize("backend", ["arxiv", "semantic_scholar"])
@pytest.mark.parametrize("prior_failure", [False, True])
def test_adapter_politeness_denial_stops_retries_and_preserves_prior_failure(
        monkeypatch, backend, prior_failure):
    import importlib

    adapter = importlib.import_module(f"papervault.library.sources.{backend}")
    attempts, sleeps = [], []

    def get(*args, **kwargs):
        attempts.append(None)
        if prior_failure and len(attempts) == 1:
            response = requests.Response()
            response.status_code = 429
            response.headers["Retry-After"] = "23"
            return response
        raise polite_http.PolitenessDenied("local admission denied")

    monkeypatch.setattr(adapter, "polite_get", get)
    monkeypatch.setattr(adapter, "time", SimpleNamespace(sleep=sleeps.append))
    with pytest.raises(BackendDegraded) as caught:
        getattr(adapter, f"search_{backend}")("solar wind")
    assert len(attempts) == (2 if prior_failure else 1)
    assert len(sleeps) == (1 if prior_failure else 0)
    assert caught.value.code == ("rate_limited" if prior_failure else "transport_error")
    assert isinstance(caught.value.__cause__, polite_http.PolitenessDenied)
    if prior_failure:
        assert caught.value.retry_after_s == 23.0


@pytest.mark.parametrize("raw,doi,arxiv", [
    ({"doi": "DOI: 10.48550/arXiv.2401.12345v2"}, "10.48550/arXiv.2401.12345v2", "2401.12345"),
    ({"url": "https://arxiv.org/abs/math.GT/0309136v2"}, "", "math.GT/0309136"),
    ({"doi": "https://doi.org/10.1000/xyz 10.1000/xyz"}, "10.1000/xyz", ""),
    ({"arxiv_id": "arXiv:2401.12345v2"}, "", "2401.12345"),
    ({"arxiv_id": "2025.35287", "doi": "10.1109/TNS.2025.3528764"},
     "10.1109/TNS.2025.3528764", ""),
    ({"arxiv_id": "2012.05554", "doi": "10.1111/j.1365-246X.2012.05554.x"},
     "10.1111/j.1365-246X.2012.05554.x", "2012.05554"),
    ({"arxiv_id": "1234.5678v2", "doi": "10.1000/unrelated"}, "10.1000/unrelated", "1234.5678"),
    ({"arxiv_id": "2020.0001"}, "", "2020.0001"),
    ({"doi": "10.48550/arXiv.2413.12345"}, "10.48550/arXiv.2413.12345", ""),
    ({"url": "https://example.org/abs/2401.12345"}, "", ""),
])
def test_normalize_paper_identifier_boundaries(raw, doi, arxiv):
    paper = search._normalize_paper({"title": "Identifier boundary", **raw}, "ads")
    assert paper["doi"] == doi
    assert paper["arxiv_id"] == arxiv
    assert paper["source_identities"][0]["arxiv_ids"] == ([arxiv] if arxiv else [])


@pytest.mark.parametrize("fields,doi,arxiv", [
    ({"doi": "DOI: 10.48550/arXiv.2401.12345v2"}, "10.48550/arXiv.2401.12345v2", "2401.12345"),
    ({"doi": "https://doi.org/10.1000/xyz 10.1000/xyz"}, "10.1000/xyz", ""),
    ({"primary_location": {"landing_page_url": "https://arxiv.org/abs/math.GT/0309136"}},
     "", "math.GT/0309136"),
    ({"primary_location": {"pdf_url": "https://arxiv.org/pdf/astro-ph/0009009v1.pdf"}},
     "", "astro-ph/0009009"),
    ({"doi": "10.1000/published", "open_access": {"oa_url": "https://arxiv.org/abs/2401.12345"}},
     "10.1000/published", "2401.12345"),
    ({"doi": "10.48550/arXiv.2413.12345"}, "10.48550/arXiv.2413.12345", ""),
])
@responses.activate
def test_openalex_derives_arxiv_identity_from_metadata(fields, doi, arxiv):
    responses.get("https://api.openalex.org/works", json={"results": [{
        "id": "https://openalex.org/W123", "title": "Preprint metadata", **fields,
    }]})
    paper = search.search_openalex("test")[0]
    assert paper["doi"] == doi
    assert paper["arxiv_id"] == arxiv
    assert paper["source_identities"][0]["arxiv_ids"] == ([arxiv] if arxiv else [])


@pytest.mark.parametrize("reason", ["cooldown", "disabled"])
def test_polite_v2_denied_arxiv_fanout_skips_pacing(monkeypatch, polite_search, reason):
    monkeypatch.setattr(search, "_interval_for", lambda backend: 3.0)
    monkeypatch.setattr(search, "time", SimpleNamespace(monotonic=lambda: polite_search.now))
    monkeypatch.setattr(search, "_LAST_CALL", {"arxiv": polite_search.now})
    monkeypatch.setattr(search, "_PACE_LOCK", {})
    if reason == "cooldown":
        polite_search.ledger.enter_cooldown(Scope("arxiv"), "test")
    else:
        monkeypatch.setattr(search, "get_ledger", lambda: Ledger({}))
    paced = []
    original = search._pace

    async def pace(*args):
        paced.append(args)
        await original(*args)

    monkeypatch.setattr(search, "_pace", pace)
    degraded, observations = collections.Counter(), []

    async def run():
        with search.observe_source_gates(observations.append):
            return await asyncio.gather(*[
                search._fetch_one_backend(term, "arxiv", 5, term_idx=i, degraded_map=degraded)
                for i, term in enumerate(["solar", "wind", "plasma"])
            ])

    assert asyncio.run(run()) == [[], [], []]
    assert paced == []
    assert polite_search.sleeps == []
    assert search._LAST_CALL == {"arxiv": 100.0}
    assert degraded == {"arxiv": 3}
    assert [o.latency_ms for o in observations] == [0.0, 0.0, 0.0]
    assert search._BREAKER_FAILS == {}


@pytest.mark.parametrize("reason", ["cooldown", "disabled"])
def test_polite_v2_rechecks_gate_after_arxiv_pacing(monkeypatch, polite_search, reason):
    monkeypatch.setattr(search, "_interval_for", lambda backend: 3.0)
    monkeypatch.setattr(search, "time", SimpleNamespace(monotonic=lambda: polite_search.now))
    monkeypatch.setattr(search, "_LAST_CALL", {"arxiv": polite_search.now})
    monkeypatch.setattr(search, "_PACE_LOCK", {})
    original_sleep = asyncio.sleep

    async def sleep(delay):
        await original_sleep(delay)
        if reason == "cooldown":
            polite_search.ledger.enter_cooldown(Scope("arxiv"), "during pacing")
        else:
            monkeypatch.setattr(search, "get_ledger", lambda: Ledger({}))

    async def forbidden(*args):
        pytest.fail("gate changed during pacing but search still submitted a fetch")

    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(search, "_run_fetch", forbidden)
    assert asyncio.run(search._fetch_one_backend("solar", "arxiv", 5, term_idx=0)) == []
    assert polite_search.sleeps == [3.0]
    assert search._BREAKER_FAILS == {}
