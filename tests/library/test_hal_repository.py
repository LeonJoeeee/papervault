"""HAL lookup and normal cascade identity checks, without HTTP or vault writes."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
import responses

from papervault.library import download
from papervault.library.models import Paper
from tests.library.test_download import _text_pdf

API = "https://api.archives-ouvertes.fr/search/"
DOCUMENT = "https://insu.hal.science/insu-01269841/document"
PDF = b"%PDF-1.4\nHAL control"
TITLE = "Formation of downstream high-speed jets by a rippled nonstationary quasi-parallel shock: 2-D hybrid simulations"


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    class Clock:
        now = 100.0

        def monotonic(self):
            return self.now

        def time(self):
            return 1700000000 + self.now - 100

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    if hasattr(download, "_try_hal_repository"):
        from papervault.library.download_sources import hal_repository

        monkeypatch.setattr(hal_repository, "time", clock)
        monkeypatch.setattr(hal_repository, "_next_request_at", 0.0)
        monkeypatch.setattr(hal_repository, "_cooldown_until", 0.0)
    return clock


@pytest.fixture
def paper():
    return Paper(key="Hao2016", doi="10.1002/2015JA021419", title=TITLE)


def _try(paper):
    # The absent tier is a behavioral miss during the initial red run.
    return dict(download._STRATEGIES).get("hal_repository", lambda _: None)(paper)


def _record(**updates):
    return {"halId_s": "insu-01269841", "doiId_s": "10.1002/2015JA021419",
            "title_s": [TITLE], "fileMain_s": DOCUMENT,
            "uri_s": "https://insu.hal.science/insu-01269841v1",
            "submitType_s": "file", **updates}


def _docs(*records):
    responses.get(API, json={"responseHeader": {"status": 0},
                             "response": {"numFound": len(records), "start": 0,
                                          "docs": list(records)}})


@pytest.mark.parametrize("returned_doi", ["10.1002/2015ja021419", ["10.1002/2015JA021419"]])
@responses.activate
def test_hal_parses_exact_doi_and_document_redirect(paper, returned_doi, clock):
    _docs(_record(doiId_s=returned_doi))
    responses.get(DOCUMENT, status=302, headers={"Location": "/insu-01269841/file/paper.pdf"})
    responses.get("https://insu.hal.science/insu-01269841/file/paper.pdf", body=PDF)

    assert _try(paper) == PDF
    params = parse_qs(urlsplit(responses.calls[0].request.url).query)
    assert params["q"] == ['doiId_s:("10.1002/2015JA021419")']
    assert params["wt"] == ["json"]
    assert params["fl"] == ["halId_s,doiId_s,title_s,fileMain_s,uri_s,submitType_s"]
    assert len(responses.calls) == 3
    assert clock.now >= 102  # API, /document and redirect each consume a slot.
    assert paper.url == ""  # HAL discovery must not replace stored metadata.


@responses.activate
def test_hal_escapes_query_quotes_and_backslashes(paper):
    paper.doi = '10.1/a\\b" OR *:*'
    _docs()
    assert _try(paper) is None
    params = parse_qs(urlsplit(responses.calls[0].request.url).query)
    assert params["q"] == ['doiId_s:("10.1/a\\\\b\\" OR *:*")']


@responses.activate
def test_hal_document_uses_library_headers_to_avoid_browser_challenge(paper):
    """Live HAL serves challenge HTML to the shared browser header profile."""
    _docs(_record())

    def document(request):
        is_library = request.headers.get("User-Agent", "").startswith("paper-pipeline/")
        if is_library and "Sec-Fetch-Mode" not in request.headers:
            return 200, {"Content-Type": "application/pdf"}, PDF
        return 200, {"Content-Type": "text/html"}, b"<html>Making sure you're not a bot!</html>"

    responses.add_callback(responses.GET, DOCUMENT, callback=document)
    assert _try(paper) == PDF


@pytest.mark.parametrize("updates", [
    {"doiId_s": "10.1002/2015JA021419.extra"},
    {"doiId_s": "10.1002/2015JA02141"},
    {"doiId_s": None}, {"fileMain_s": None}, {"fileMain_s": ""},
    {"fileMain_s": "file:///tmp/paper.pdf"}, {"fileMain_s": [DOCUMENT]},
])
@responses.activate
def test_hal_ignores_nonexact_dois_and_missing_or_invalid_files(paper, updates):
    _docs(_record(**updates))
    assert _try(paper) is None
    assert len(responses.calls) == 1  # Never follow uri_s or an unrelated record.


@responses.activate
def test_hal_tries_later_file_record_after_notice_or_non_pdf(paper):
    other = "https://hal.science/hal-12345678/document"
    _docs(_record(fileMain_s=None, submitType_s="notice"),
          _record(), _record(fileMain_s=other))
    responses.get(DOCUMENT, body="<html>embargo</html>")
    responses.get(other, body=PDF)
    assert _try(paper) == PDF
    assert len(responses.calls) == 3


@pytest.mark.parametrize("status,headers,body", [
    (403, {}, PDF), (206, {}, PDF), (200, {"Content-Range": "bytes 0-10/100"}, PDF),
    (200, {}, b"<html>landing page</html>"),
])
@responses.activate
def test_hal_requires_complete_successful_pdf(paper, status, headers, body):
    _docs(_record())
    responses.get(DOCUMENT, status=status, headers=headers, body=body)
    assert _try(paper) is None
    assert len(responses.calls) == 2


@responses.activate
def test_hal_without_doi_skips_network(paper):
    paper.doi = ""
    assert _try(paper) is None
    assert download._download_skip_reason("hal_repository", paper) == "missing_doi"
    assert not responses.calls


@pytest.mark.parametrize("body,status", [
    ('{"response":{"docs":[]}}', 503), ("not json", 200),
    ('{"response":null}', 200), ('{"response":{"docs":null}}', 200),
])
@responses.activate
def test_hal_api_failures_are_misses(paper, body, status):
    responses.get(API, body=body, status=status)
    assert _try(paper) is None
    assert len(responses.calls) == 1


@responses.activate
def test_hal_transport_error_is_a_miss(paper):
    responses.get(API, body=requests.Timeout("bounded timeout"))
    assert _try(paper) is None
    assert len(responses.calls) == 1


@pytest.mark.parametrize("retry_after,delay", [
    ("120", 120), ("Tue, 14 Nov 2023 22:15:20 GMT", 120),
    (None, 60), ("invalid", 60), ("-10", 1), ("nan", 60),
])
@responses.activate
def test_hal_429_honors_cooldown_without_retry(paper, clock, retry_after, delay):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    responses.get(API, status=429, headers=headers)
    _docs()
    assert _try(paper) is None
    assert _try(paper) is None
    assert len(responses.calls) == 1
    clock.now += delay - 0.01
    assert _try(paper) is None
    assert len(responses.calls) == 1
    clock.now += 0.01
    assert _try(paper) is None
    assert len(responses.calls) == 2


@responses.activate
def test_hal_pdf_429_stops_remaining_candidates_and_next_lookup(paper, clock):
    _docs(_record(), _record(fileMain_s="https://hal.science/hal-12345678/document"))
    responses.get(DOCUMENT, status=429, headers={"Retry-After": "120"})
    assert _try(paper) is None
    assert _try(paper) is None
    assert len(responses.calls) == 2


@responses.activate
def test_hal_concurrent_lookups_are_paced(paper, clock):
    starts = []

    def reply(request):
        starts.append(clock.now)
        return 200, {}, '{"response":{"docs":[]}}'

    responses.add_callback(responses.GET, API, callback=reply)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(_try, [paper] * 3)) == [None, None, None]
    assert len(starts) == 3
    assert all(b - a >= 1 for a, b in zip(starts, starts[1:]))


@pytest.mark.parametrize("wrong_paper", [False, True])
@responses.activate
def test_hal_normal_cascade_uses_identity_gate_in_memory(paper, monkeypatch, wrong_paper):
    if wrong_paper:
        paper.key = "He2013c"
        paper.doi = "10.1088/1367-2630"
        paper.title = ("Generation of quasi-monoenergetic protons from thin multi-ion foils by a "
                       "combination of laser radiation pressure acceleration and shielded Coulomb repulsion")
        candidate_title = "Structure and dynamics of multicellular assemblies measured by coherent light scattering"
        url = "https://hal.science/hal-01525521/document"
        _docs(_record(halId_s="hal-01525521", doiId_s=paper.doi, title_s=[candidate_title],
                      fileMain_s=url, uri_s="https://hal.science/hal-01525521v1"))
    else:
        candidate_title, url = paper.title, DOCUMENT
        _docs(_record())
    data = _text_pdf(candidate_title + ". Authors. Abstract. We present the detailed results.")
    responses.get(url, body=data)
    saved, events, prompts = {}, [], []
    root = Path("/in-memory-only")
    library = SimpleNamespace(root=root, pdf_path=lambda _: root / "paper.pdf",
                              has_pdf=lambda _: False, has_extract=lambda *_: False, log=events.append)
    monkeypatch.setenv("PAPERVAULT_VAULT", str(root))
    monkeypatch.setattr(download, "_atomic_save", lambda dest, data: saved.update(pdf=data))
    monkeypatch.setattr(download, "_STRATEGIES", [
        (tag, fn if tag == "hal_repository" else lambda _: None)
        for tag, fn in download._STRATEGIES
    ])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)

    def judge(messages):
        head = messages[1]["content"].split("--- first pages of the PDF ---\n", 1)[1]
        prompts.append(head)
        match = paper.title in head
        return json.dumps({"match": match, "reason": "same work" if match else "different work"})

    monkeypatch.setattr("papervault.library.llm.get_llm", lambda **_: SimpleNamespace(call=judge))
    assert download.download_paper(paper, library) is (not wrong_paper)
    assert len(prompts) == 1
    assert candidate_title in prompts[0]
    if wrong_paper:
        assert not saved
        assert any(e["event"] == "download_pdf_mismatch" and e["source"] == "hal_repository"
                   and e["reason"] == "llm_mismatch: different work" for e in events)
        assert paper.download_status == "failed"
    else:
        assert saved["pdf"] == data
        assert paper.download_source == "hal_repository"
        assert paper.download_status == "ok"
        assert next(e for e in events if e["event"] == "downloaded")["verify"] == "llm_match: same work"
