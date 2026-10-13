"""Real transport wrapper and API send paths with no network."""

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
import responses
from urllib3.response import HTTPResponse

from papervault.library import polite_http
from papervault.library.politeness_v2 import Ledger, Policy, Scope, Window
from test_politeness_v2 import Clock


API = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/test"


@pytest.fixture(autouse=True)
def isolate_wrapper(monkeypatch):
    monkeypatch.delenv("PAPERVAULT_POLITE_V2", raising=False)
    monkeypatch.delenv("PAPERVAULT_POLITE_ENVELOPES", raising=False)
    monkeypatch.setattr(polite_http, "_ledger", None)
    monkeypatch.setattr(polite_http, "_configuration", None)
    monkeypatch.setattr(polite_http, "_unavailable", set())


def response(status=200):
    result = requests.Response()
    result.status_code = status
    result._content = b"%PDF-test"
    result._content_consumed = True
    return result


def stream_response():
    result = response()
    result.raw = HTTPResponse(body=io.BytesIO(b"{}"), preload_content=False,
                              headers={"Content-Length": "2"})
    result._content = False
    result._content_consumed = False
    return result


@pytest.mark.parametrize("flag", [None, "0", "false", "", "no"])
def test_flag_off_calls_original_transport_with_identical_arguments(monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv("PAPERVAULT_POLITE_V2", flag)
    monkeypatch.setenv("PAPERVAULT_POLITE_ENVELOPES", "invalid JSON ignored while off")
    expected = response()
    def transport(url, **kwargs):
        assert url == API
        assert kwargs == {"timeout": 60, "headers": {"token": "test"}, "allow_redirects": True}
        return expected
    assert polite_http.polite_get(transport, API, timeout=60, headers={"token": "test"},
                                  allow_redirects=True) is expected
    assert polite_http._ledger is None


def test_enabled_accounts_at_send_and_denies_shared_tier_without_transport(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "true")
    clock = Clock()
    ledger = Ledger({Scope("wiley"): Policy((Window(1, 1),))}, clock=clock)
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    def transport(url, **kwargs):
        assert ledger.snapshot(Scope("wiley")).in_flight == 1
        assert ledger.snapshot(Scope("wiley")).sent == 1
        return response()
    polite_http.polite_get(transport, API)
    assert ledger.snapshot(Scope("wiley")).positive_completions == 1
    def forbidden(*args, **kwargs):
        pytest.fail("blocked request reached the transport")
    with pytest.raises(polite_http.PolitenessDenied):
        polite_http.polite_get(forbidden, API.replace("test", "another-tier"))
    clock.now = 1
    polite_http.polite_get(lambda *args, **kwargs: response(), API)
    assert ledger.snapshot(Scope("wiley")).sent == 2


def test_accounting_unavailable_emits_nothing(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    ledger = Ledger({Scope("wiley"): Policy((Window(1, 1),))})
    def broken(*args):
        raise OSError("ledger unavailable")
    monkeypatch.setattr(ledger, "note_sent", broken)
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    with pytest.raises(polite_http.PolitenessUnavailable):
        polite_http.polite_get(lambda *args, **kwargs: pytest.fail("request emitted"), API)


def test_completion_accounting_failure_poison_scope_before_next_send(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    ledger = Ledger({Scope("wiley"): Policy((Window(10, 1),), concurrency=2)})
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    def broken(*args, **kwargs):
        raise OSError("result accounting failed")
    monkeypatch.setattr(ledger, "note_result", broken)
    with pytest.raises(polite_http.PolitenessUnavailable):
        polite_http.polite_get(lambda *args, **kwargs: response(), API)
    with pytest.raises(polite_http.PolitenessUnavailable):
        polite_http.polite_get(lambda *args, **kwargs: pytest.fail("request after accounting failure"), API)


def test_completion_failure_disables_scope_atomically_with_concurrent_admission(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    ledger = Ledger({Scope("wiley"): Policy((Window(10, 1),), concurrency=2)}, clock=Clock())
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    # Hold result accounting at the failure boundary while another caller
    # tries admission. Completion+poison must exclude that competing send.
    recording = threading.Event()
    release = threading.Event()
    competing = threading.Event()
    emitted = threading.Event()
    def broken(*args, **kwargs):
        recording.set()
        assert release.wait(2)
        raise OSError("result accounting failed")
    monkeypatch.setattr(ledger, "note_result", broken)
    def second():
        competing.set()
        def transport(*args, **kwargs):
            emitted.set()
            return response()
        return polite_http.polite_get(transport, API)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(polite_http.polite_get, lambda *args, **kwargs: response(), API)
        assert recording.wait(2)
        later = pool.submit(second)
        try:
            assert competing.wait(2)
            assert not emitted.wait(0.05)
        finally:
            release.set()
        for future in (first, later):
            with pytest.raises(polite_http.PolitenessUnavailable):
                future.result(timeout=2)
    assert not emitted.is_set()


@pytest.mark.parametrize("flag", ["0", "1"])
def test_stream_holds_concurrency_until_context_closes(monkeypatch, flag):
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", flag)
    ledger = Ledger({Scope("openalex"): Policy((Window(10, 1),))}, clock=Clock())
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    result = stream_response()
    with polite_http.polite_stream(lambda *args, **kwargs: result,
                                  "https://api.openalex.org/works/W1", stream=True) as actual:
        assert actual is result
        if flag == "1":
            assert ledger.gate(Scope("openalex")).reason == "concurrency"
            assert ledger.snapshot(Scope("openalex")).positive_completions == 0
        assert actual.raw.read() == b"{}"
    assert result.raw.closed
    if flag == "1":
        assert ledger.snapshot(Scope("openalex")).in_flight == 0
        assert ledger.snapshot(Scope("openalex")).positive_completions == 1


def test_stream_body_timeout_counts_failure_and_releases_permit(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    scope = Scope("openalex")
    ledger = Ledger({scope: Policy((Window(10, 1),), timeout_cluster=1)}, clock=Clock())
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    with pytest.raises(TimeoutError):
        with polite_http.polite_stream(lambda *args, **kwargs: stream_response(),
                                      "https://api.openalex.org/works/W1"):
            raise TimeoutError("body stalled")
    assert ledger.snapshot(scope).in_flight == 0
    assert ledger.snapshot(scope).positive_completions == 0
    assert ledger.gate(scope).reason == "cooldown"


@pytest.mark.parametrize("close_early", [False, True])
def test_unread_or_early_closed_stream_never_earns_ramp_evidence(monkeypatch, close_early):
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    ledger = Ledger({Scope("openalex"): Policy((Window(10, 1),))}, clock=Clock())
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    with polite_http.polite_stream(lambda *args, **kwargs: stream_response(),
                                  "https://api.openalex.org/works/W1") as result:
        assert result.raw.read(1) == b"{"
        if close_early:
            result.close()
    assert ledger.snapshot(Scope("openalex")).in_flight == 0
    assert ledger.snapshot(Scope("openalex")).positive_completions == 0


@responses.activate
def test_crossref_download_and_metadata_caller_share_one_scope(monkeypatch):
    from papervault.library import fetch
    from papervault.library.download_sources.crossref import _try_crossref_link
    from papervault.library.models import Paper
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    ledger = Ledger({Scope("crossref"): Policy((Window(1, 60),))}, clock=Clock())
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    responses.get("https://api.crossref.org/works/10.1002/test", json={"message": {}})
    assert _try_crossref_link(Paper(key="test", title="Test", doi="10.1002/test")) is None
    assert fetch._fetch_crossref_by_doi("10.1002/test") is None
    assert len(responses.calls) == 1
    assert ledger.snapshot(Scope("crossref")).sent == 1


@responses.activate
def test_api_retry_after_429_cannot_send_during_scope_cooldown(monkeypatch):
    from papervault.library.sources import semantic_scholar
    from papervault.library.sources.exceptions import BackendDegraded
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    ledger = Ledger({Scope("semantic_scholar"): Policy((Window(10, 1),))}, clock=Clock())
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    monkeypatch.setattr(semantic_scholar.time, "sleep", lambda seconds: None)
    responses.get(semantic_scholar.API_URL, status=429)
    with pytest.raises(BackendDegraded):
        semantic_scholar.search_semantic_scholar("test")
    assert len(responses.calls) == 1
    assert ledger.gate(Scope("semantic_scholar")).reason == "cooldown"


def test_429_headers_stop_scope_even_when_eager_response_body_times_out(monkeypatch):
    from requests.adapters import BaseAdapter
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    ledger = Ledger({Scope("wiley"): Policy((Window(10, 1),))}, clock=Clock())
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    class StalledBody(io.BytesIO):
        def read(self, *args, **kwargs):
            raise TimeoutError("429 body stalled")
    class Adapter(BaseAdapter):
        def send(self, request, **kwargs):
            result = response(429)
            result._content = False
            result._content_consumed = False
            result.raw = HTTPResponse(body=StalledBody(b"{}"), preload_content=False,
                                      headers={"Content-Length": "2"})
            result.request = request
            return result
        def close(self):
            pass
    with requests.Session() as session:
        session.mount("https://", Adapter())
        with pytest.raises(requests.ConnectionError):
            polite_http.polite_get(session.get, API)
    assert ledger.gate(Scope("wiley")).reason == "cooldown"
    assert ledger.snapshot(Scope("wiley")).ramp_stopped
    assert ledger.snapshot(Scope("wiley")).positive_completions == 0


def test_enabled_response_hook_keeps_existing_hook_and_original_hook_container(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    ledger = Ledger({Scope("wiley"): Policy((Window(10, 1),))}, clock=Clock())
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    observed = []
    def hook(result, **kwargs):
        observed.append(result.status_code)
        return result
    hooks = {"response": hook}
    with responses.RequestsMock() as mock:
        mock.get(API, status=429)
        assert polite_http.polite_get(requests.get, API, hooks=hooks).status_code == 429
    assert hooks == {"response": hook}
    assert observed == [429]
    assert ledger.gate(Scope("wiley")).reason == "cooldown"


def test_unknown_scope_and_invalid_envelope_config_fail_closed(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    url = "https://api.unpaywall.org/v2/10.1/test"
    with pytest.raises(polite_http.PolitenessDenied):
        polite_http.polite_get(lambda *args, **kwargs: pytest.fail("unconfigured request"), url)
    monkeypatch.setenv("PAPERVAULT_POLITE_ENVELOPES", "{broken")
    with pytest.raises(polite_http.PolitenessUnavailable):
        polite_http.polite_get(lambda *args, **kwargs: pytest.fail("invalid configuration request"), url)


def test_explicit_envelope_is_validated_and_cannot_reset_live_counters(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    configuration = {"crossref": {"analogue": "wiley", "initial": [[1, 1], [30, 600]],
                                  "hard_cap": [[1, 1], [60, 600]], "authorization": "issue:221 fixture"}}
    monkeypatch.setenv("PAPERVAULT_POLITE_ENVELOPES", json.dumps(configuration))
    ledger = polite_http.get_ledger()
    url = "https://api.crossref.org/works/10.1/test"
    polite_http.polite_get(lambda *args, **kwargs: response(), url)
    assert ledger.snapshot(Scope("crossref")).sent == 1
    configuration["crossref"]["initial"][1][0] = 31
    monkeypatch.setenv("PAPERVAULT_POLITE_ENVELOPES", json.dumps(configuration))
    with pytest.raises(polite_http.PolitenessUnavailable):
        polite_http.polite_get(lambda *args, **kwargs: pytest.fail("configuration reset counters"), url)


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("source", ["wiley", "crossref", "unpaywall", "semantic_scholar"])
@responses.activate
def test_download_api_flag_parity_preserves_pdf_and_requests(monkeypatch, flag, source):
    from papervault.library.download_sources import crossref, semantic_scholar, unpaywall, wiley
    from papervault.library.models import Paper
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", flag)
    monkeypatch.setenv("WILEY_TDM_TOKEN", "fixture-token")
    ledger = Ledger({Scope(source): Policy((Window(10, 1),))}, clock=Clock())
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    pdf_url = "https://publisher.test/paper.pdf"
    pdf = b"%PDF-fixture"
    api_urls = {
        "wiley": "https://api.wiley.com/onlinelibrary/tdm/v1/articles/10.1002%2Ftest",
        "crossref": "https://api.crossref.org/works/10.1002/test",
        "unpaywall": "https://api.unpaywall.org/v2/10.1002/test?email=research@example.invalid",
        "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1002/test?fields=openAccessPdf",
    }
    payloads = {"crossref": {"message": {"link": [{"URL": pdf_url, "content-type": "application/pdf"}]}},
                "unpaywall": {"best_oa_location": {"url_for_pdf": pdf_url}},
                "semantic_scholar": {"openAccessPdf": {"url": pdf_url}}}
    if source == "wiley":
        responses.get(api_urls[source], body=pdf)
    else:
        responses.get(api_urls[source], json=payloads[source])
        responses.get(pdf_url, body=pdf)
    methods = {"wiley": wiley._try_wiley_tdm, "crossref": crossref._try_crossref_link,
               "unpaywall": unpaywall._try_unpaywall, "semantic_scholar": semantic_scholar._try_semantic_scholar_oa}
    assert methods[source](Paper(key="test", title="Test", doi="10.1002/test")) == pdf
    assert len(responses.calls) == (1 if source == "wiley" else 2)
    assert ledger.snapshot(Scope(source)).sent == (1 if flag == "1" else 0)


@pytest.mark.parametrize("flag", ["0", "1"])
@responses.activate
def test_arxiv_api_flag_parity(monkeypatch, flag):
    from papervault.library.sources.arxiv import search_arxiv
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", flag)
    ledger = Ledger({Scope("arxiv"): Policy((Window(1, 3),))}, clock=Clock())
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    responses.get("http://export.arxiv.org/api/query", body='<feed xmlns="http://www.w3.org/2005/Atom"/>')
    assert search_arxiv("test") == []
    assert len(responses.calls) == 1
    assert ledger.snapshot(Scope("arxiv")).sent == (1 if flag == "1" else 0)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_openalex_adapter_api_flag_parity(monkeypatch, flag):
    from papervault.library.download_sources import openalex
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", flag)
    ledger = Ledger({Scope("openalex"): Policy((Window(10, 1),))})
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    result = stream_response()
    def transport(self, request, **kwargs):
        assert request.url == "https://api.openalex.org/works/W1"
        assert kwargs["stream"] is True
        return result
    monkeypatch.setattr(openalex._DeadlineAdapter, "send", transport)
    with openalex._request("https://api.openalex.org/works/W1", deadline=openalex.time.monotonic() + 10) as actual:
        assert actual.raw.read() == b"{}"
    assert ledger.snapshot(Scope("openalex")).sent == (1 if flag == "1" else 0)
