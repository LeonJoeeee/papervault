"""Source failures stay distinguishable from genuine zero-hit observations."""

import asyncio
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import importlib

import pytest
import requests
import responses

from papervault.library import search
from papervault.library.sources.exceptions import BackendDegraded


ADAPTERS = [
    ("arxiv", "http://export.arxiv.org/api/query"),
    ("semantic_scholar", "https://api.semanticscholar.org/graph/v1/paper/search"),
    ("inspire", "https://inspirehep.net/api/literature"),
    ("ads", "https://api.adsabs.harvard.edu/v1/search/query"),
    ("core", "https://api.core.ac.uk/v3/search/works/"),
    ("openalex", "https://api.openalex.org/works"),
]


def _adapter(name, monkeypatch):
    monkeypatch.setenv("ADS_API_TOKEN", "test-token")
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    module = (search if name == "openalex" else
              importlib.import_module(f"papervault.library.sources.{name}"))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return getattr(module, f"search_{name}")


@pytest.mark.parametrize("name,url", ADAPTERS)
@pytest.mark.parametrize("status,code,retryability,scope", [
    (429, "rate_limited", "transient_with_delay", "service"),
    (503, "provider_unavailable", "transient_with_delay", "service"),
    (401, "authentication_failed", "permanent", "service"),
    (400, "invalid_request", "permanent", "item"),
])
@responses.activate
def test_http_failure_taxonomy_is_not_a_zero_hit(name, url, status, code, retryability,
                                                scope, monkeypatch):
    adapter = _adapter(name, monkeypatch)
    responses.add(responses.GET, url, status=status, headers={"Retry-After": "7"})
    with pytest.raises(BackendDegraded) as caught:
        adapter("test")
    error = caught.value
    assert (error.code, error.scope, error.retryability) == (code, scope, retryability)
    assert error.retry_after_s == (7.0 if retryability == "transient_with_delay" else None)


@pytest.mark.parametrize("name,url", ADAPTERS)
@pytest.mark.parametrize("error,code", [
    (requests.Timeout("secret transport details"), "timeout"),
    (requests.ConnectionError("secret transport details"), "transport_error"),
])
@responses.activate
def test_transport_failure_is_structured(name, url, error, code, monkeypatch):
    adapter = _adapter(name, monkeypatch)
    responses.add(responses.GET, url, body=error)
    with pytest.raises(BackendDegraded) as caught:
        adapter("test")
    assert (caught.value.code, caught.value.scope, caught.value.retryability,
            caught.value.retry_after_s) == (code, "service", "transient_no_delay", None)
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("name,url", ADAPTERS)
@pytest.mark.parametrize("body", ['not json or XML', '{"unexpected": "envelope"}'])
@responses.activate
def test_invalid_provider_payload_is_not_a_zero_hit(name, url, body, monkeypatch):
    adapter = _adapter(name, monkeypatch)
    responses.add(responses.GET, url, body=body, status=200)
    with pytest.raises(BackendDegraded) as caught:
        adapter("test")
    assert (caught.value.code, caught.value.scope, caught.value.retryability) == (
        "provider_invalid_payload", "service", "permanent")


@pytest.mark.parametrize("header,want", [
    (None, 5.0), ("garbage", 5.0), ("-1", 5.0), ("nan", 5.0),
    ("2.5", 2.5), ("Wed, 21 Oct 2026 10:00:04 GMT", 4.0),
    ("Wed, 21 Oct 2026 09:59:59 GMT", 0.0),
])
def test_retry_after_numeric_date_and_absent_semantics(header, want):
    from papervault.library.sources.exceptions import backend_degraded
    response = requests.Response()
    response.status_code = 429
    if header is not None:
        response.headers["Retry-After"] = header
    error = backend_degraded(response=response,
                             now=datetime(2026, 10, 21, 10, tzinfo=timezone.utc))
    assert error.retry_after_s == want


def test_legacy_message_defaults_and_explicit_retry_delay_semantics():
    error = BackendDegraded("old caller")
    assert str(error) == "old caller"
    assert (error.code, error.scope, error.retryability, error.retry_after_s) == (
        "backend_degraded", "service", "transient_with_delay", 5.0)
    assert BackendDegraded("permanent", retryability="permanent").retry_after_s is None


def test_source_observer_records_native_hits_zero_hits_and_typed_degradation(monkeypatch):
    async def no_pace(*args):
        pass
    monkeypatch.setattr(search, "_pace", no_pace)
    monkeypatch.setattr(search, "BREAKER_TRIPS", 0)
    search._breaker_reset()
    monkeypatch.setattr(search, "search_openalex", lambda *a, **k: [{"title": "Old", "year": 1990}])
    monkeypatch.setattr(search, "search_arxiv", lambda *a, **k: [])
    def fail(*args, **kwargs):
        raise BackendDegraded("token=secret", code="rate_limited", retry_after_s=8)
    monkeypatch.setattr(search, "search_core", fail)
    events = []
    async def run():
        with search.observe_source_gates(events.append):
            await search._fetch_one_backend("term", "openalex", 5, term_idx=2, year_min=2020)
            await search._fetch_one_backend("term", "arxiv", 5, term_idx=2)
            await search._fetch_one_backend("term", "core", 5, term_idx=2)
    asyncio.run(run())
    assert [(e.backend, e.term_idx, e.status, e.hit_count) for e in events] == [
        ("openalex", 2, "success", 1), ("arxiv", 2, "success", 0),
        ("core", 2, "degraded", 0),
    ]
    assert all(e.latency_ms >= 0 and e.term == "term" for e in events)
    assert (events[2].code, events[2].scope, events[2].retryability,
            events[2].retry_after_s) == ("rate_limited", "service", "transient_with_delay", 8)
    assert "secret" not in repr(events)
    with pytest.raises(FrozenInstanceError):
        events[0].hit_count = 99


@pytest.mark.parametrize("failure,status,code,retryability", [
    (search.FetchQueueTimeout(), "skipped", "fetch_queue_timeout", "transient_with_delay"),
    (TimeoutError(), "degraded", "timeout", "transient_no_delay"),
    (requests.ConnectionError("secret"), "degraded", "transport_error", "transient_no_delay"),
    (ValueError("secret"), "degraded", "provider_invalid_payload", "permanent"),
])
def test_observer_records_boundary_failures(monkeypatch, failure, status, code, retryability):
    async def no_pace(*args):
        pass
    async def fail(*args):
        raise failure
    monkeypatch.setattr(search, "_pace", no_pace)
    monkeypatch.setattr(search, "_run_fetch", fail)
    monkeypatch.setattr(search, "BREAKER_TRIPS", 0)
    search._breaker_reset()
    events = []
    with search.observe_source_gates(events.append):
        assert asyncio.run(search._fetch_one_backend("term", "arxiv", 5, term_idx=3)) == []
    assert len(events) == 1
    assert (events[0].status, events[0].code, events[0].retryability) == (
        status, code, retryability)
    assert "secret" not in repr(events)


def test_observer_context_is_nested_request_scoped_and_breaker_skip_is_explicit(monkeypatch):
    search._breaker_reset()
    monkeypatch.setitem(search._BREAKER_OPEN_UNTIL, "arxiv", float("inf"))
    outer, inner = [], []
    def fetch():
        return asyncio.run(search._fetch_one_backend("term", "arxiv", 5, term_idx=0))
    try:
        with search.observe_source_gates(outer.append):
            fetch()
            with search.observe_source_gates(inner.append):
                fetch()
            fetch()
        fetch()
        assert len(outer) == 2 and len(inner) == 1
        assert (inner[0].status, inner[0].code, inner[0].scope) == (
            "skipped", "circuit_open", "service")
        assert inner[0].retry_after_s > 0
    finally:
        search._breaker_reset()


@pytest.mark.parametrize("backend", ["ads", "core"])
def test_unconfigured_adapter_is_observed_as_skipped_without_degradation(monkeypatch, backend):
    async def no_pace(*args):
        pass
    monkeypatch.setattr(search, "_pace", no_pace)
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    search._breaker_reset()
    events, degraded = [], {}
    with search.observe_source_gates(events.append):
        assert asyncio.run(search._fetch_one_backend(
            "term", backend, 5, term_idx=0, degraded_map=degraded)) == []
    assert degraded == {}
    assert (events[0].status, events[0].code, events[0].retryability) == (
        "skipped", "unconfigured", "permanent")


def test_observers_do_not_cross_concurrent_requests_and_failures_do_not_change_results(monkeypatch):
    async def no_pace(*args):
        pass
    monkeypatch.setattr(search, "_pace", no_pace)
    monkeypatch.setattr(search, "search_arxiv", lambda *a, **k: [{"title": "Match"}])
    search._breaker_reset()
    events_a, events_b = [], []
    async def request(term, callback):
        with search.observe_source_gates(callback):
            return await search._fetch_one_backend(term, "arxiv", 5, term_idx=0)
    def fail_callback(_):
        raise ValueError("token=secret")
    async def run():
        return await asyncio.gather(request("a", events_a.append), request("b", events_b.append),
                                    request("c", fail_callback))
    out = asyncio.run(run())
    assert [[p["title"] for p in result] for result in out] == [["Match"], ["Match"], ["Match"]]
    assert [e.term for e in events_a] == ["a"]
    assert [e.term for e in events_b] == ["b"]


@pytest.mark.parametrize("name,url", ADAPTERS)
@responses.activate
def test_exhaustion_classifies_final_transport_failure_not_stale_http_response(name, url, monkeypatch):
    adapter = _adapter(name, monkeypatch)
    responses.add(responses.GET, url, status=429)
    responses.add(responses.GET, url, body=requests.Timeout("secret"))
    with pytest.raises(BackendDegraded) as caught:
        adapter("test")
    assert (caught.value.code, caught.value.retryability) == ("timeout", "transient_no_delay")


@responses.activate
def test_ads_receipt_reports_actual_empty_http_result_even_for_whitespace_token(monkeypatch):
    async def no_pace(*args):
        pass
    monkeypatch.setattr(search, "_pace", no_pace)
    monkeypatch.setenv("ADS_API_TOKEN", " ")
    search._breaker_reset()
    responses.add(responses.GET, "https://api.adsabs.harvard.edu/v1/search/query",
                  status=200, json={"response": {"docs": []}})
    events = []
    with search.observe_source_gates(events.append):
        asyncio.run(search._fetch_one_backend("term", "ads", 5, term_idx=0))
    assert (events[0].status, events[0].hit_count, events[0].code) == ("success", 0, None)
