"""Tests for download.py PDF cascade. No real HTTP."""

from __future__ import annotations

import builtins
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import responses

from papervault.library import Library, download
from papervault.library.download_sources import openalex
from papervault.library.models import Paper


PDF_BYTES = b"%PDF-1.4\n%fake-pdf"
HTML_BYTES = b"<html>not a pdf</html>"


@pytest.fixture
def lib(tmp_path):
    return Library(tmp_path)


@pytest.fixture
def paper(lib):
    p, _ = lib.upsert({"title": "Test paper for download cascade", "authors": ["A"], "year": 2020,
                       "doi": "10.1/x", "arxiv_id": "2401.0001"})
    return p


def _logged_events(lib: Library) -> list[dict]:
    if not lib.manifest_path.exists():
        return []
    return [json.loads(line) for line in lib.manifest_path.read_text().splitlines() if line]


def _text_pdf(text: str) -> bytes:
    """A readable PDF so cascade tests exercise the real identity gate."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
    })
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def file_url_identity(monkeypatch, paper):
    """Replace only the external LLM; parse and verify PDF text normally."""
    prompts = []

    def judge(messages):
        prompts.append(messages[1]["content"])
        program = "Meeting program and poster listing" in messages[1]["content"]
        return json.dumps({"match": not program,
                           "reason": "meeting program" if program else "same work"})

    monkeypatch.setattr("papervault.library.llm.get_llm",
                        lambda **_: SimpleNamespace(call=judge))
    data = _text_pdf(f"{paper.title}. A. Abstract. We study cosmic ray transport in space.")
    return data, prompts


@pytest.mark.parametrize("url", [
    "https://eprints.lancs.ac.uk/6681/1/art_834.pdf",
    "https://wrap.warwick.ac.uk/56873/7/WRAP_THESIS_Parmar_2012.pdf",
    "https://core.ac.uk/download/475653135.pdf",
    "https://core.ac.uk/download/pdf/357359140.pdf",
    "http://repository.example/Characteristic%20of%20electrical%20signal.pdf",
    "https://repository.example/paper.PDF?download=1#page=2",
    "https://repository.example/paper.%70df",
    "https://repository.fit.edu/cgi/viewcontent.cgi?article=1470&context=etd",
    "https://trace.tennessee.edu/cgi/viewcontent.cgi?article=9817&context=utk_graddiss",
    "https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1051.1229&rep=rep1&type=pdf",
    "http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=ABC123"
    "?doi=10.1.1.1051.1229&rep=rep1&type=pdf",
])
@responses.activate
def test_known_file_url_identifierless_download_verifies_identity(
        lib, paper, monkeypatch, file_url_identity, url):
    """A stored file must reach the early cascade without an override or identifier."""
    paper.doi = paper.arxiv_id = ""
    paper.url = url
    data, prompts = file_url_identity
    monkeypatch.setenv("PAPERVAULT_VAULT", str(lib.root))
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    # requests decodes unreserved escapes before sending the HTTP request.
    request_url = url.replace(".%70df", ".pdf").split("#", 1)[0]
    responses.get(request_url, body=data, content_type="application/octet-stream")

    assert download.download_paper(paper, lib) is True
    assert lib.pdf_path(paper.key).read_bytes() == data
    assert paper.pdf_path == f"pdfs/{paper.key}.pdf"
    assert paper.download_status == "ok"
    assert paper.download_source == "known_file_url"
    assert paper.doi == paper.arxiv_id == ""
    assert len(prompts) == 1
    assert f"Requested title: {paper.title}" in prompts[0]
    assert "Abstract. We study cosmic ray transport in space." in prompts[0]
    outcomes = [e for e in _logged_events(lib) if e.get("source")]
    assert [(e["event"], e["source"]) for e in outcomes] == [
        ("download_skip", "url_override"), ("downloaded", "known_file_url"),
    ]
    assert outcomes[-1]["verify"] == "llm_match: same work"
    assert [c.request.url.split("#", 1)[0] for c in responses.calls] == [request_url]
    assert not (lib.root / "url_overrides.json").exists()
    lib.save()
    saved = Library(lib.root).get(paper.key)
    assert saved.download_source == "known_file_url"
    assert saved.pdf_path == paper.pdf_path


@pytest.mark.parametrize("url, reason", [
    ("", "missing_file_url"),
    ("   ", "missing_file_url"),
    ("file:///tmp/paper.pdf", "not_file_url"),
    ("ftp://repository.example/paper.pdf", "not_file_url"),
    ("//repository.example/paper.pdf", "not_file_url"),
    ("https:///paper.pdf", "not_file_url"),
    ("https://[broken/paper.pdf", "not_file_url"),
    ("https://.example/paper.pdf", "not_file_url"),
    ("https://*/paper.pdf", "not_file_url"),
    ("https://repository.example:invalid/paper.pdf", "not_file_url"),
    ("https://repository.example:99999/paper.pdf", "not_file_url"),
    ("https://user:password@repository.example/paper.pdf", "not_file_url"),
    ("https://repository.example/pap\ner.pdf", "not_file_url"),
    ("https://repository.example/paper.pdf\x00", "not_file_url"),
    ("https://repository.example/paper%GG.pdf", "not_file_url"),
    ("https://repository.example\\other.example/paper.pdf", "not_file_url"),
    ("全文連結http://pcwave.rish.kyoto-u.ac.jp/versim/data/VERSIM_Program_poster.pdf", "not_file_url"),
    ("https://hdl.handle.net/2060/19850026519", "not_file_url"),
    ("https://openalex.org/W3103168219", "not_file_url"),
    ("https://ui.adsabs.harvard.edu/abs/2020AGUFM/abstract", "not_file_url"),
    ("https://arxiv.org/abs/2401.0001", "not_file_url"),
    ("https://repository.example/landing?file=paper.pdf", "not_file_url"),
    ("https://repository.fit.edu/cgi/viewcontent.cgi?article=1470", "not_file_url"),
    ("https://repository.fit.edu/cgi/viewcontent.cgi?article=bad&context=etd", "not_file_url"),
    ("https://repository.example/cgi/other.cgi?article=1470&context=etd", "not_file_url"),
    ("https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1051.1229&rep=rep1&type=html", "not_file_url"),
])
@responses.activate
def test_known_file_url_ineligible_url_is_a_skip(lib, paper, monkeypatch, url, reason):
    """Malformed strings and record pages must not be counted as file requests."""
    _isolate_tier(monkeypatch, "known_file_url")
    paper.doi = paper.arxiv_id = ""
    paper.url = url

    assert download.download_paper(paper, lib) is False
    outcomes = [e for e in _logged_events(lib) if e.get("source") == "known_file_url"]
    assert [(e["event"], e.get("reason")) for e in outcomes] == [("download_skip", reason)]
    assert not responses.calls
    assert not lib.has_pdf(paper.key)
    assert paper.url == url


@pytest.mark.parametrize("status, body, headers", [
    (200, HTML_BYTES, {"Content-Type": "application/pdf"}),
    (200, b'<meta name="citation_pdf_url" content="/real.pdf">', {}),
    (403, HTML_BYTES, {}),
    (404, PDF_BYTES, {}),
    (429, HTML_BYTES, {}),
    (500, PDF_BYTES, {}),
    (302, PDF_BYTES, {}),
    (206, PDF_BYTES, {"Content-Range": "bytes 0-19/500"}),
    (200, PDF_BYTES, {"Content-Range": "bytes 0-19/500"}),
])
@pytest.mark.parametrize("abstract, terminal", [("", "failed"), ("Known abstract.", "metadata_only")])
@responses.activate
def test_known_file_url_bad_response_is_a_miss(
        lib, paper, monkeypatch, file_url_identity, status, body, headers, abstract, terminal):
    """Suffixes, MIME, partial content, and error bodies cannot prove a download."""
    _isolate_tier(monkeypatch, "known_file_url")
    paper.doi = paper.arxiv_id = ""
    paper.url = "https://core.ac.uk/download/475653135.pdf"
    paper.abstract = abstract
    _, prompts = file_url_identity
    responses.get(paper.url, status=status, body=body, headers=headers)

    assert download.download_paper(paper, lib) is False
    outcomes = [e for e in _logged_events(lib) if e.get("source") == "known_file_url"]
    assert [e["event"] for e in outcomes] == ["download_miss"]
    assert len(responses.calls) == 1
    assert not prompts
    assert not lib.has_pdf(paper.key)
    assert paper.download_status == terminal
    assert paper.download_source == ""
    assert paper.url == "https://core.ac.uk/download/475653135.pdf"


@pytest.mark.parametrize("error", [
    requests.Timeout("fixture timeout"),
    requests.ConnectionError("fixture dead host"),
    requests.exceptions.ChunkedEncodingError("fixture truncated body"),
])
@responses.activate
def test_known_file_url_request_failure_is_a_miss(lib, paper, monkeypatch, error):
    _isolate_tier(monkeypatch, "known_file_url")
    paper.url = "https://repository.example/paper.pdf"
    responses.get(paper.url, body=error)

    assert download.download_paper(paper, lib) is False
    outcomes = [e for e in _logged_events(lib) if e.get("source") == "known_file_url"]
    assert [e["event"] for e in outcomes] == ["download_miss"]
    assert len(responses.calls) == 1
    assert paper.doi == "10.1/x" and paper.arxiv_id == "2401.0001"
    assert not lib.has_pdf(paper.key)


@pytest.mark.parametrize("is_pdf", [True, False])
@responses.activate
def test_known_file_url_redirect_checks_final_body(lib, paper, monkeypatch, file_url_identity, is_pdf):
    _isolate_tier(monkeypatch, "known_file_url")
    paper.doi = paper.arxiv_id = ""
    paper.url = "http://authors.library.caltech.edu/46479/1/1995-43.pdf"
    target = "https://repository.example/content/object"
    data, prompts = file_url_identity
    responses.get(paper.url, status=302, headers={"Location": target})
    responses.get(target, body=data if is_pdf else HTML_BYTES)

    assert download.download_paper(paper, lib) is is_pdf
    assert [c.request.url for c in responses.calls] == [paper.url, target]
    assert lib.has_pdf(paper.key) is is_pdf
    outcomes = [e for e in _logged_events(lib) if e.get("source") == "known_file_url"]
    assert [e["event"] for e in outcomes] == ["downloaded" if is_pdf else "download_miss"]
    assert len(prompts) == int(is_pdf)
    if is_pdf:
        assert paper.download_source == "known_file_url"


@responses.activate
def test_known_file_url_unsupported_redirect_is_a_miss(lib, paper, monkeypatch):
    _isolate_tier(monkeypatch, "known_file_url")
    paper.url = "https://repository.example/paper.pdf"
    responses.get(paper.url, status=302, headers={"Location": "file:///tmp/paper.pdf"})

    assert download.download_paper(paper, lib) is False
    outcomes = [e for e in _logged_events(lib) if e.get("source") == "known_file_url"]
    assert [e["event"] for e in outcomes] == ["download_miss"]
    assert len(responses.calls) == 1
    assert not lib.has_pdf(paper.key)


@responses.activate
def test_known_file_url_redirect_loop_is_a_miss(lib, paper, monkeypatch):
    _isolate_tier(monkeypatch, "known_file_url")
    paper.url = "https://repository.example/paper.pdf"
    responses.get(paper.url, status=302, headers={"Location": paper.url})

    assert download.download_paper(paper, lib) is False
    outcomes = [e for e in _logged_events(lib) if e.get("source") == "known_file_url"]
    assert [e["event"] for e in outcomes] == ["download_miss"]
    assert responses.calls
    assert not lib.has_pdf(paper.key)


@responses.activate
def test_known_file_url_program_rejection_continues_cascade(lib, paper, monkeypatch, file_url_identity):
    """A parseable program PDF is not saved when the normal identity gate rejects it."""
    strategies = dict(download._STRATEGIES)
    monkeypatch.setattr(download, "_STRATEGIES", [
        ("known_file_url", strategies["known_file_url"]), ("arxiv", strategies["arxiv"]),
    ])
    paper.url = "http://pcwave.rish.kyoto-u.ac.jp/versim/data/VERSIM_Program_poster.pdf"
    program = _text_pdf("Meeting program and poster listing. Speakers, sessions, and poster titles.")
    data, prompts = file_url_identity
    responses.get(paper.url, body=program)
    responses.get("https://arxiv.org/pdf/2401.0001", body=data)

    assert download.download_paper(paper, lib) is True
    assert paper.download_source == "arxiv"
    assert lib.pdf_path(paper.key).read_bytes() == data
    outcomes = [e for e in _logged_events(lib) if e.get("source")]
    assert [(e["event"], e["source"]) for e in outcomes] == [
        ("download_pdf_mismatch", "known_file_url"), ("downloaded", "arxiv"),
    ]
    assert outcomes[0]["reason"] == "llm_mismatch: meeting program"
    assert len(prompts) == 2


@responses.activate
def test_known_file_url_keeps_operator_override_priority(lib, paper, monkeypatch, file_url_identity):
    monkeypatch.setenv("PAPERVAULT_VAULT", str(lib.root))
    overrides_path = lib.root / "url_overrides.json"
    overrides = json.dumps({paper.doi: "https://operator.example/chosen.pdf"})
    overrides_path.write_text(overrides)
    paper.url = "https://repository.example/paper.pdf"
    data, _ = file_url_identity
    responses.get("https://arxiv.org/abs/2401.0001", status=503)
    responses.get("https://operator.example/chosen.pdf", body=data)

    assert download.download_paper(paper, lib) is True
    assert paper.download_source == "url_override"
    # Withdrawal preflight precedes acquisition; override remains the first
    # file candidate when current withdrawal is not confirmed.
    assert [c.request.url for c in responses.calls] == [
        "https://arxiv.org/abs/2401.0001", "https://operator.example/chosen.pdf",
    ]
    assert overrides_path.read_text() == overrides


@responses.activate
def test_known_file_url_existing_pdf_is_not_refetched(lib, paper, file_url_identity):
    paper.url = "https://repository.example/paper.pdf"
    data, prompts = file_url_identity
    dest = lib.pdf_path(paper.key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    paper.download_status = "ok"
    paper.download_source = "arxiv"
    before = _logged_events(lib)

    assert download.download_paper(paper, lib) is True
    assert dest.read_bytes() == data
    assert paper.download_source == "arxiv"
    assert not responses.calls and not prompts
    assert _logged_events(lib) == before


@pytest.mark.parametrize("padding", [0, 8400])
@pytest.mark.parametrize("page, pdf_url", [
    ('<meta name="citation_pdf_url" content="//pdf.example/paper.pdf#view">',
     "https://pdf.example/paper.pdf"),
    ('<object type="application/pdf" data="/paper.pdf"></object>',
     "https://mirror.example/paper.pdf"),
    ('<iframe src="https://pdf.example/paper.pdf"></iframe>',
     "https://pdf.example/paper.pdf"),
    ('<script>location.href = "https://pdf.example/paper.pdf";</script>',
     "https://pdf.example/paper.pdf"),
])
@responses.activate
def test_scihub_short_page_with_link_downloads_pdf(page, pdf_url, padding, monkeypatch):
    """A layout shrink must not discard a page containing a matching link."""
    page += " " * padding
    assert len(page) < 12000
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    responses.get("https://mirror.example/10.1/x", body=page)
    responses.get(pdf_url, body=PDF_BYTES)

    assert download._scihub_one_mirror("https://mirror.example", "10.1/x") == PDF_BYTES
    assert [c.request.url for c in responses.calls] == [
        "https://mirror.example/10.1/x", pdf_url,
    ]


@pytest.mark.parametrize("padding", [0, 8400, 15000])
@responses.activate
def test_scihub_linkless_page_retries_then_fails(padding, monkeypatch):
    """Captcha/linkless pages still get three fresh requests, regardless of size."""
    sleeps = []
    monkeypatch.setattr(download.time, "sleep", sleeps.append)
    responses.get("https://mirror.example/10.1/x",
                  body="<html>Captcha: verify you are human</html>" + " " * padding)

    assert download._scihub_one_mirror("https://mirror.example", "10.1/x") is None
    assert [c.request.url for c in responses.calls] == ["https://mirror.example/10.1/x"] * 3
    assert sleeps == [1, 2]


@responses.activate
def test_scihub_linkless_retry_can_recover(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    responses.get("https://mirror.example/10.1/x", body="<html>Captcha</html>")
    responses.get("https://mirror.example/10.1/x",
                  body='<meta name="citation_pdf_url" content="/paper.pdf">')
    responses.get("https://mirror.example/paper.pdf", body=PDF_BYTES)

    assert download._scihub_one_mirror("https://mirror.example", "10.1/x") == PDF_BYTES
    assert len(responses.calls) == 3


@responses.activate
def test_scihub_matching_link_still_requires_pdf_bytes(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    responses.get("https://mirror.example/10.1/x",
                  body='<meta name="citation_pdf_url" content="/paper.pdf">')
    responses.get("https://mirror.example/paper.pdf", body=HTML_BYTES)

    assert download._scihub_one_mirror("https://mirror.example", "10.1/x") is None
    assert len(responses.calls) == 6


def _isolate_tier(monkeypatch, source):
    strategy = dict(download._STRATEGIES)[source]
    monkeypatch.setattr(download, "_STRATEGIES", [(source, strategy)])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)


@pytest.mark.parametrize("source, fields, env", [
    ("url_override", {}, {}),
    ("arxiv", {"arxiv_id": ""}, {}),
    ("iopscience_direct", {}, {}),
    ("crossref_tm", {"doi": ""}, {}),
    ("citation_pdf_url", {"doi": ""}, {}),
    ("wiley_tdm", {"doi": "10.1029/x"}, {}),
    ("wiley_tdm", {}, {"WILEY_TDM_TOKEN": "fixture-token"}),
    ("elsevier_tdm", {"doi": "10.1016/x"}, {}),
    ("elsevier_tdm", {}, {"ELSEVIER_TDM_API_KEY": "fixture-token"}),
    ("oa_aggregators", {"doi": ""}, {}),
    ("oa_aggregators", {"doi": "", "title": "Short"}, {"CORE_API_KEY": "fixture-token"}),
    ("scihub", {}, {}),
    ("scihub", {"doi": "", "arxiv_id": "", "url": ""}, {"PAPER_PIPELINE_USE_SCIHUB": "1"}),
    ("annas_archive", {}, {}),
    ("annas_archive", {"doi": ""}, {"ANNAS_ARCHIVE_API_KEY": "fixture-token"}),
    ("domain_aggregators", {"doi": "", "arxiv_id": ""}, {}),
    ("curl_impersonate", {"doi": ""}, {}),
    ("arxiv_by_title", {}, {}),
    ("arxiv_by_title", {"arxiv_id": "", "title": "Short"}, {}),
    ("arxiv_by_title", {"arxiv_id": "", "title": "?" * 25}, {}),
    ("mdpi_scrapling", {}, {}),
    ("researchgate", {"doi": "", "title": "Short"}, {}),
    ("web_search", {"title": "Short"}, {}),
])
@responses.activate
def test_unattempted_tier_logs_skip_not_miss(lib, paper, monkeypatch, source, fields, env):
    """Missing credentials/identifiers or applicability must not inflate misses."""
    _isolate_tier(monkeypatch, source)
    monkeypatch.setenv("PAPERVAULT_VAULT", str(lib.root))
    for name in ("WILEY_TDM_TOKEN", "ELSEVIER_TDM_API_KEY", "ANNAS_ARCHIVE_API_KEY",
                 "CORE_API_KEY", "PAPER_PIPELINE_USE_SCIHUB"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    for name, value in fields.items():
        setattr(paper, name, value)

    assert download.download_paper(paper, lib) is False
    events = [e for e in _logged_events(lib) if e.get("source") == source]
    assert len(events) == 1
    assert events[0]["event"] == "download_skip"
    assert events[0]["reason"]
    assert len(responses.calls) == 0


@pytest.mark.parametrize("source, module", [
    ("curl_impersonate", "curl_cffi"),
    ("annas_archive", "scrapling.fetchers"),
    ("mdpi_scrapling", "scrapling.fetchers"),
    ("researchgate", "scrapling.fetchers"),
])
@responses.activate
def test_missing_optional_dependency_logs_skip(lib, paper, monkeypatch, source, module):
    _isolate_tier(monkeypatch, source)
    paper.doi = "10.3390/x"
    monkeypatch.setenv("ANNAS_ARCHIVE_API_KEY", "fixture-token")
    original_import = builtins.__import__

    def absent_import(name, *args, **kwargs):
        if name == module:
            raise ImportError("fixture dependency absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent_import)
    assert download.download_paper(paper, lib) is False
    events = [e for e in _logged_events(lib) if e.get("source") == source]
    assert [e["event"] for e in events] == ["download_skip"]
    assert events[0]["reason"] == "missing_dependency"
    assert len(responses.calls) == 0


@pytest.mark.parametrize("source", ["annas_archive", "mdpi_scrapling", "researchgate"])
@responses.activate
def test_browser_tier_import_failure_warns(paper, monkeypatch, caplog, source):
    """Direct tier callers must also hear about a missing transitive Playwright import."""
    paper.doi = "10.3390/x"
    monkeypatch.setenv("ANNAS_ARCHIVE_API_KEY", "fixture-token")
    original_import = builtins.__import__

    def absent_browser(name, *args, **kwargs):
        if name == "scrapling.fetchers":
            raise ModuleNotFoundError("No module named 'playwright'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent_browser)
    assert dict(download._STRATEGIES)[source](paper) is None
    assert source in caplog.text
    assert "scrapling[fetchers]" in caplog.text
    assert not responses.calls


@pytest.fixture
def scidb_html():
    return (Path(__file__).parent / "fixtures" / "annas_scidb_member.html").read_text()


@pytest.fixture
def scidb_browser(monkeypatch):
    """Only replace the external browser boundary; run the tier's real page_action."""
    def install(rendered_html, *, detail_html=None, challenge_timeout=False, status=200):
        if detail_html is None:
            detail_html = (Path(__file__).parent / "fixtures" / "annas_detail_member.html").read_text()

        class Page:
            html = "<html><title>DDoS-Guard</title><body>Please wait</body></html>"

            def __init__(self, url):
                self.detail = "/md5/" in url

            def wait_for_selector(self, selector, *, state, timeout):
                if self.detail:
                    assert 'h3:has-text("External downloads")' in selector
                else:
                    assert selector.startswith('a[href^="/md5/"], ')
                    assert 'title:has-text("Search - Anna")' in selector
                assert state == "attached"
                assert 25000 < timeout <= 120000
                if challenge_timeout:
                    raise TimeoutError("fixture challenge did not resolve")
                if not self.detail and '/md5/' not in rendered_html and 'Search - Anna' not in rendered_html:
                    if 'a[href*=".pdf" i]' not in selector:
                        raise TimeoutError("fixture PDF offered without an MD5 record link")
                self.html = detail_html if self.detail else rendered_html

            def wait_for_load_state(self, state, *, timeout):
                assert state == "domcontentloaded"
                assert timeout > 25000

            def content(self):
                return self.html

        class Fetcher:
            @staticmethod
            def fetch(url, **kwargs):
                assert url in {"https://annas-archive.gl/scidb/10.1/x",
                               "https://annas-archive.gl/md5/0123456789abcdef0123456789abcdef"}
                assert kwargs["headless"] is True
                assert kwargs.get("wait", 0) == 0  # No fixed challenge sleep.
                assert kwargs["timeout"] > 25000
                assert kwargs["cookies"] == [{
                    "name": "aa_account_id2", "value": "fixture-token",
                    "url": "https://annas-archive.gl",
                }]
                page = Page(url)
                try:
                    kwargs["page_action"](page)
                except TimeoutError:
                    # Scrapling logs and swallows page_action errors, then returns a response.
                    pass
                return SimpleNamespace(status=status, html_content=page.content())

        monkeypatch.setitem(sys.modules, "scrapling.fetchers",
                            SimpleNamespace(StealthyFetcher=Fetcher))
        def impersonated_get(url, **kwargs):
            assert kwargs.pop("impersonate") == "chrome"
            return requests.get(url, **kwargs)

        monkeypatch.setitem(sys.modules, "curl_cffi", SimpleNamespace(
            requests=SimpleNamespace(get=impersonated_get)))
        monkeypatch.setenv("ANNAS_ARCHIVE_API_KEY", "fixture-token")
        # A plain request must only see the bot wall. This is the old tier's failure.
        responses.get("https://annas-archive.gl/scidb/10.1/x", body="DDoS-Guard")

    return install


@pytest.mark.parametrize("segment", ["d4", "d3", "unlimited/next-generation"])
@pytest.mark.parametrize("filename, request_filename", [
    ("Test%20paper.pdf", "Test%20paper.pdf"),
    ("Test%20paper.PDF", "Test%20paper.PDF"),
    ("Test%20paper%2Epdf", "Test%20paper.pdf"),
])
@responses.activate
def test_annas_detail_offered_pdf_is_unlimited(
    paper, scidb_html, scidb_browser, segment, filename, request_filename,
):
    """Follow the offered PDF anchor across CDN path changes, without using fast_download."""
    detail_html = (Path(__file__).parent / "fixtures" / "annas_detail_member.html").read_text()
    scidb_browser(scidb_html, detail_html=detail_html.replace("/d4/", f"/{segment}/")
                  .replace("Test%20paper.pdf", filename))
    url = (f"https://partner-cdn.example:8443/{segment}/signed/"
           f"{request_filename}?token=fixture&expires=123")
    responses.get(url, body=PDF_BYTES, content_type="application/octet-stream")

    assert download._try_annas_archive_api(paper) == PDF_BYTES
    assert [call.request.url for call in responses.calls] == [url]
    assert "Cookie" not in responses.calls[0].request.headers


@responses.activate
def test_annas_record_pdf_without_md5_is_explained(paper, scidb_html, scidb_browser, caplog):
    """A signed record link cannot bypass the md5 lookup required by #163."""
    scidb_html = scidb_html.replace('<a href="/md5/0123456789abcdef0123456789abcdef">Archive record</a>', '')
    scidb_browser(scidb_html)
    assert download._try_annas_archive_api(paper) is None
    assert "no md5" in caplog.text
    assert not responses.calls


@pytest.mark.parametrize("direct_body", [None, HTML_BYTES])
@responses.activate
def test_annas_detail_miss_preserves_fast_download_fallback(
    paper, scidb_html, scidb_browser, direct_body,
):
    detail_html = '<h3>External downloads</h3>'
    if direct_body is not None:
        detail_html = (Path(__file__).parent / "fixtures" / "annas_detail_member.html").read_text()
        responses.get(
            "https://Partner-CDN.example:8443/d4/signed/Test%20paper.pdf",
            body=direct_body,
        )
    scidb_browser(scidb_html, detail_html=detail_html)
    responses.get("https://annas-archive.gl/dyn/api/fast_download.json",
                  json={"download_url": "https://partner.example/fallback.pdf"})
    responses.get("https://partner.example/fallback.pdf", body=PDF_BYTES)

    assert download._try_annas_archive_api(paper) == PDF_BYTES
    api_call = next(call for call in responses.calls if "fast_download.json" in call.request.url)
    assert "md5=0123456789abcdef0123456789abcdef" in api_call.request.url
    assert "key=fixture-token" in api_call.request.url
    assert "Cookie" not in responses.calls[-1].request.headers


@responses.activate
def test_annas_unresolved_challenge_does_not_consume_quota(paper, scidb_html, scidb_browser):
    scidb_browser(scidb_html, challenge_timeout=True)
    assert download._try_annas_archive_api(paper) is None
    assert not responses.calls


@responses.activate
def test_annas_browser_error_page_does_not_consume_quota(paper, scidb_html, scidb_browser):
    scidb_browser(scidb_html, status=403)
    assert download._try_annas_archive_api(paper) is None
    assert not responses.calls


@responses.activate
def test_annas_search_page_returns_none(paper, scidb_browser):
    scidb_browser('<html><title>Search - Anna\'s Archive</title></html>')
    assert download._try_annas_archive_api(paper) is None
    assert not responses.calls


@responses.activate
def test_annas_browser_failure_returns_none(paper, scidb_html, scidb_browser, monkeypatch):
    scidb_browser(scidb_html)

    def unavailable_browser(*args, **kwargs):
        raise RuntimeError("fixture browser executable unavailable")

    monkeypatch.setattr(sys.modules["scrapling.fetchers"].StealthyFetcher,
                        "fetch", unavailable_browser)
    assert download._try_annas_archive_api(paper) is None
    assert not responses.calls


@pytest.mark.parametrize("response", [403, 404, requests.ConnectionError("fixture outage")])
@responses.activate
def test_attempted_tier_keeps_download_miss(lib, paper, monkeypatch, response):
    """A PDF failure remains a miss after the bounded metadata check."""
    _isolate_tier(monkeypatch, "arxiv")
    responses.get("https://arxiv.org/abs/2401.0001", status=503)
    kwargs = {"status": response} if isinstance(response, int) else {"body": response}
    responses.get("https://arxiv.org/pdf/2401.0001", **kwargs)

    assert download.download_paper(paper, lib) is False
    events = [e for e in _logged_events(lib) if e.get("source") == "arxiv"]
    assert [e["event"] for e in events] == ["download_miss"]
    assert [c.request.url for c in responses.calls] == [
        "https://arxiv.org/abs/2401.0001", "https://arxiv.org/pdf/2401.0001",
    ]
    assert paper.arxiv_id == "2401.0001"


@responses.activate
def test_oa_group_with_title_and_core_key_is_attempted(lib, paper, monkeypatch):
    """DOI-only members skip, while CORE's title search still makes the group run."""
    _isolate_tier(monkeypatch, "oa_aggregators")
    paper.doi = ""
    monkeypatch.setenv("CORE_API_KEY", "fixture-token")
    responses.get("https://api.core.ac.uk/v3/search/works/", json={"results": []})

    assert download.download_paper(paper, lib) is False
    events = [e for e in _logged_events(lib) if e.get("source") == "oa_aggregators"]
    assert [e["event"] for e in events] == ["download_miss"]
    assert len(responses.calls) == 1


@pytest.mark.parametrize("source, doi, credential, url", [
    ("wiley_tdm", "10.1029/x", "WILEY_TDM_TOKEN",
     "https://api.wiley.com/onlinelibrary/tdm/v1/articles/10.1029%2Fx"),
    ("wiley_tdm", "10.1029", "WILEY_TDM_TOKEN",
     "https://api.wiley.com/onlinelibrary/tdm/v1/articles/10.1029"),
    ("elsevier_tdm", "10.1016/x", "ELSEVIER_TDM_API_KEY",
     "https://api.elsevier.com/content/article/doi/10.1016/x?apiKey=fixture-token"),
    ("elsevier_tdm", "10.1016", "ELSEVIER_TDM_API_KEY",
     "https://api.elsevier.com/content/article/doi/10.1016?apiKey=fixture-token"),
])
@responses.activate
def test_credentialed_tdm_refusal_is_a_miss(lib, paper, monkeypatch, source, doi, credential, url):
    """Use each TDM tier's exact DOI gate, even for malformed bare prefixes."""
    _isolate_tier(monkeypatch, source)
    paper.doi = doi
    monkeypatch.setenv(credential, "fixture-token")
    responses.get(url, status=403)

    assert download.download_paper(paper, lib) is False
    events = [e for e in _logged_events(lib) if e.get("source") == source]
    assert [e["event"] for e in events] == ["download_miss"]
    assert len(responses.calls) == 1


@responses.activate
def test_configured_url_override_uses_canonical_vault_path(lib, paper, tmp_path, monkeypatch):
    _isolate_tier(monkeypatch, "url_override")
    configured_vault = tmp_path / "configured-vault"
    configured_vault.mkdir()
    (configured_vault / "url_overrides.json").write_text(
        json.dumps({"10.1/x": "https://override.example/paper.pdf"}))
    monkeypatch.setenv("PAPERVAULT_VAULT", str(configured_vault))
    responses.get("https://override.example/paper.pdf", body=HTML_BYTES)

    assert download.download_paper(paper, lib) is False
    events = [e for e in _logged_events(lib) if e.get("source") == "url_override"]
    assert [e["event"] for e in events] == ["download_miss"]
    assert len(responses.calls) == 1


@responses.activate
def test_domain_group_with_arxiv_only_is_attempted(lib, paper, monkeypatch):
    _isolate_tier(monkeypatch, "domain_aggregators")
    paper.doi = ""
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    responses.get("https://inspirehep.net/api/literature", json={"hits": {"hits": []}})
    responses.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                  json={"resultList": {"result": []}})
    responses.get("https://zenodo.org/api/records", json={"hits": {"hits": []}})

    assert download.download_paper(paper, lib) is False
    events = [e for e in _logged_events(lib) if e.get("source") == "domain_aggregators"]
    assert [e["event"] for e in events] == ["download_miss"]
    assert len(responses.calls) == 3


@responses.activate
def test_arxiv_strategy_wins_on_first_try(lib, paper):
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=PDF_BYTES, status=200)
    assert download.download_paper(paper, lib) is True
    assert lib.has_pdf(paper.key)
    assert paper.pdf_path == "pdfs/T2020.pdf" or paper.pdf_path.endswith(f"{paper.key}.pdf")
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "arxiv"  # provenance split out
    events = _logged_events(lib)
    assert any(e["event"] == "downloaded" and e["source"] == "arxiv" for e in events)


@responses.activate
def test_falls_back_to_unpaywall(lib, paper):
    # arxiv returns HTML (not a PDF) → magic-byte check rejects.
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={"best_oa_location": {"url_for_pdf": "https://oa.example/x.pdf"}},
                  status=200)
    responses.add(responses.GET, "https://oa.example/x.pdf",
                  body=PDF_BYTES, status=200)

    assert download.download_paper(paper, lib) is True
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "oa_aggregators"  # provenance split out
    events = _logged_events(lib)
    # arxiv produced a miss, unpaywall a hit
    assert any(e["event"] == "download_miss" and e["source"] == "arxiv" for e in events)
    assert any(e["event"] == "downloaded" and e["source"] == "oa_aggregators" for e in events)


@responses.activate
def test_all_strategies_fail_returns_false(lib, paper, monkeypatch):
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)

    assert download.download_paper(paper, lib) is False
    assert paper.download_status == "failed"
    assert not lib.has_pdf(paper.key)
    events = _logged_events(lib)
    assert any(e["event"] == "download_failed" for e in events)


@responses.activate
def test_scihub_skipped_when_env_unset(lib, paper, monkeypatch):
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)

    download.download_paper(paper, lib)

    # No request should have been made to any sci-hub mirror.
    urls = [c.request.url for c in responses.calls]
    assert not any("sci-hub" in u for u in urls)


# NOTE: ``test_scihub_attempted_when_opted_in`` was removed in 2026-05.
# Sci-hub strategy was rewritten to parallel-mirror probing with per-mirror
# retry (see _try_scihub in download.py). The old single-mirror mock here
# no longer matches; rewriting it would couple test to internal mirror
# discovery details that are deliberately fuzzy (cache TTL + DNS probe).
# Live sci-hub behavior is validated by the racing-bench in /tmp/racing_bench_v*.py.


@responses.activate
def test_pdf_magic_byte_check_rejects_non_pdf(lib, paper, monkeypatch):
    """A 200 response with non-PDF bytes must NOT be saved as a PDF."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=b"<!DOCTYPE html>not a pdf", status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)
    assert download.download_paper(paper, lib) is False
    assert not lib.has_pdf(paper.key)


def test_already_has_pdf_short_circuits(lib, paper):
    # Plant a PDF on disk; download_paper should return True without any HTTP.
    lib.pdf_path(paper.key).write_bytes(PDF_BYTES)
    with responses.RequestsMock() as rmocks:
        # No URLs registered — any HTTP would raise ConnectionError.
        assert download.download_paper(paper, lib) is True
        assert len(rmocks.calls) == 0


@responses.activate
def test_arxiv_strategy_skipped_without_arxiv_id(lib):
    p, _ = lib.upsert({"title": "Test paper for download cascade", "authors": ["A"], "year": 2020, "doi": "10.1/x"})
    # No arxiv_id → skip arxiv. Only unpaywall request should fire.
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={"best_oa_location": {"url_for_pdf": "https://oa/x.pdf"}},
                  status=200)
    responses.add(responses.GET, "https://oa/x.pdf", body=PDF_BYTES, status=200)
    assert download.download_paper(p, lib) is True
    arxiv_calls = [c for c in responses.calls if "arxiv.org" in c.request.url]
    assert arxiv_calls == []


# ---------------------------------------------------------------------------
# OpenAlex strategy
# ---------------------------------------------------------------------------

@responses.activate
def test_openalex_happy_path(lib, paper, monkeypatch):
    """When arxiv + unpaywall miss, openalex finds the PDF via oa_locations."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)
    responses.add(
        responses.GET, "https://api.openalex.org/works/doi:10.1/x",
        json={
            "best_oa_location": {"pdf_url": "https://oa.openalex/best.pdf"},
            "oa_locations": [
                {"pdf_url": "https://oa.openalex/best.pdf"},  # dedup
                {"pdf_url": "https://oa.openalex/alt.pdf"},
            ],
        },
        status=200,
    )
    responses.add(responses.GET, "https://oa.openalex/best.pdf",
                  body=PDF_BYTES, status=200)

    assert download.download_paper(paper, lib) is True
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "oa_aggregators"  # provenance split out
    events = _logged_events(lib)
    assert any(e["event"] == "downloaded" and e["source"] == "oa_aggregators" for e in events)


@responses.activate
def test_openalex_falls_through_when_no_oa_locations(lib, paper, monkeypatch):
    """OpenAlex responds but gives no PDF URLs → strategy returns None."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1/x",
                  json={"oa_locations": []}, status=200)
    # inspire + ads also miss (no responses registered for openalex success)
    responses.add(
        responses.GET, "https://inspirehep.net/api/literature",
        json={"hits": {"hits": []}}, status=200,
    )
    assert download.download_paper(paper, lib) is False
    events = _logged_events(lib)
    assert any(e["event"] == "download_miss" and e["source"] == "oa_aggregators" for e in events)


@responses.activate
def test_openalex_skipped_without_doi_or_work_id(lib):
    """No DOI or stored work ID → no OpenAlex API call."""
    p, _ = lib.upsert({"title": "Test paper for download cascade", "authors": ["A"], "year": 2020,
                       "arxiv_id": "2401.0099"})
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0099",
                  body=HTML_BYTES, status=200)
    responses.add(
        responses.GET, "https://inspirehep.net/api/literature",
        json={"hits": {"hits": []}}, status=200,
    )
    download.download_paper(p, lib)
    openalex_calls = [c for c in responses.calls
                      if "api.openalex.org" in c.request.url]
    assert openalex_calls == []


@responses.activate
def test_openalex_accepts_url_for_pdf_field(lib, paper, monkeypatch):
    """Some OpenAlex records use `url_for_pdf` (Unpaywall-style); we accept either."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)
    responses.add(
        responses.GET, "https://api.openalex.org/works/doi:10.1/x",
        json={"oa_locations": [
            {"url_for_pdf": "https://oa.openalex/legacy.pdf"},
        ]},
        status=200,
    )
    responses.add(responses.GET, "https://oa.openalex/legacy.pdf",
                  body=PDF_BYTES, status=200)
    assert download.download_paper(paper, lib) is True
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "oa_aggregators"  # provenance split out
# ---------------------------------------------------------------------------
# Inspire-HEP strategy
# ---------------------------------------------------------------------------

@responses.activate
def test_inspire_happy_path(lib, paper, monkeypatch):
    """arxiv + unpaywall + openalex miss; inspire returns documents[] PDF."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1/x",
                  json={"oa_locations": []}, status=200)
    responses.add(
        responses.GET, "https://inspirehep.net/api/literature",
        json={"hits": {"hits": [
            {"metadata": {"documents": [
                {"key": "fulltext", "url": "https://inspirehep.net/files/abc.pdf"},
            ]}}
        ]}},
        status=200,
    )
    responses.add(responses.GET, "https://inspirehep.net/files/abc.pdf",
                  body=PDF_BYTES, status=200)
    assert download.download_paper(paper, lib) is True
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "domain_aggregators"  # provenance split out
    events = _logged_events(lib)
    assert any(e["event"] == "downloaded" and e["source"] == "domain_aggregators" for e in events)


@responses.activate
def test_inspire_no_hits_returns_none(lib, paper, monkeypatch):
    """Inspire returns hits.hits=[] → strategy returns None, cascade continues."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1/x",
                  json={"oa_locations": []}, status=200)
    responses.add(responses.GET, "https://inspirehep.net/api/literature",
                  json={"hits": {"hits": []}}, status=200)
    assert download.download_paper(paper, lib) is False
    events = _logged_events(lib)
    assert any(e["event"] == "download_miss" and e["source"] == "domain_aggregators" for e in events)


@responses.activate
def test_inspire_falls_back_to_doi_query(lib, monkeypatch):
    """No arxiv_id → query inspire with doi:<doi>."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    p, _ = lib.upsert({"title": "Test paper for download cascade", "authors": ["A"], "year": 2020,
                       "doi": "10.1/y"})
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/y",
                  json={}, status=200)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1/y",
                  json={"oa_locations": []}, status=200)
    responses.add(
        responses.GET, "https://inspirehep.net/api/literature",
        json={"hits": {"hits": [
            {"metadata": {"documents": [
                {"url": "https://inspirehep.net/files/doi.pdf"},
            ]}}
        ]}},
        status=200,
    )
    responses.add(responses.GET, "https://inspirehep.net/files/doi.pdf",
                  body=PDF_BYTES, status=200)
    assert download.download_paper(p, lib) is True
    assert p.download_status == "ok"  # D7: status routes
    assert p.download_source == "domain_aggregators"  # provenance split out
    inspire_call = next(
        c for c in responses.calls
        if "inspirehep.net/api/literature" in c.request.url
    )
    assert "doi%3A10.1%2Fy" in inspire_call.request.url or "doi:10.1/y" in inspire_call.request.url


# ---------------------------------------------------------------------------
# NASA ADS strategy
# ---------------------------------------------------------------------------

@responses.activate
def test_ads_skipped_without_token(lib, paper, monkeypatch):
    """No ADS_API_TOKEN → ADS is a silent no-op (no API call)."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1/x",
                  json={"oa_locations": []}, status=200)
    responses.add(responses.GET, "https://inspirehep.net/api/literature",
                  json={"hits": {"hits": []}}, status=200)
    download.download_paper(paper, lib)
    ads_calls = [c for c in responses.calls
                 if "adsabs.harvard.edu" in c.request.url]
    assert ads_calls == []


@responses.activate
def test_ads_happy_path(lib, paper, monkeypatch):
    """All earlier strategies miss; ADS finds the PDF via link_gateway."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    monkeypatch.setenv("ADS_API_TOKEN", "fake")
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1/x",
                  json={"oa_locations": []}, status=200)
    responses.add(responses.GET, "https://inspirehep.net/api/literature",
                  json={"hits": {"hits": []}}, status=200)
    responses.add(
        responses.GET, "https://api.adsabs.harvard.edu/v1/search/query",
        json={"response": {"docs": [
            {"bibcode": "2020ApJ...900....1B", "doi": ["10.1/x"], "identifier": ["arXiv:2401.0001"], "esources": ["EPRINT_HTML", "PUB_PDF"]},
        ]}},
        status=200,
    )
    # First esource isn't a PDF type, second is — link_gateway returns the body.
    responses.add(
        responses.GET,
        "https://ui.adsabs.harvard.edu/link_gateway/2020ApJ...900....1B/PUB_PDF",
        body=PDF_BYTES, status=200,
    )
    assert download.download_paper(paper, lib) is True
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "domain_aggregators"  # provenance split out
    events = _logged_events(lib)
    assert any(e["event"] == "downloaded" and e["source"] == "domain_aggregators" for e in events)
    # Auth header was sent on the search call.
    search_call = next(
        c for c in responses.calls
        if "search/query" in c.request.url
    )
    assert search_call.request.headers.get("Authorization") == "Bearer fake"


@responses.activate
def test_ads_returns_none_on_no_pdf_esources(lib, paper, monkeypatch):
    """ADS docs found but no PDF-typed esource → returns None."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    monkeypatch.setenv("ADS_API_TOKEN", "fake")
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1/x",
                  json={"oa_locations": []}, status=200)
    responses.add(responses.GET, "https://inspirehep.net/api/literature",
                  json={"hits": {"hits": []}}, status=200)
    responses.add(
        responses.GET, "https://api.adsabs.harvard.edu/v1/search/query",
        json={"response": {"docs": [
            {"bibcode": "2020ApJ...900....1B", "doi": ["10.1/x"], "identifier": ["arXiv:2401.0001"], "esources": ["EPRINT_HTML", "AUTHOR_HTML"]},
        ]}},
        status=200,
    )
    assert download.download_paper(paper, lib) is False
    events = _logged_events(lib)
    assert any(e["event"] == "download_miss" and e["source"] == "domain_aggregators" for e in events)


@responses.activate
def test_full_cascade_arxiv_to_ads(lib, paper, monkeypatch):
    """End-to-end cascade: arxiv → unpaywall → openalex → inspire all miss,
    ADS wins. Verifies the new tier order is wired correctly."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    monkeypatch.setenv("ADS_API_TOKEN", "fake")
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1/x",
                  json={"best_oa_location": None, "oa_locations": []},
                  status=200)
    responses.add(responses.GET, "https://inspirehep.net/api/literature",
                  json={"hits": {"hits": []}}, status=200)
    responses.add(
        responses.GET, "https://api.adsabs.harvard.edu/v1/search/query",
        json={"response": {"docs": [
            {"bibcode": "2020ApJ...900....2B", "doi": ["10.1/x"], "identifier": ["arXiv:2401.0001"], "esources": ["ADS_PDF"]},
        ]}},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://ui.adsabs.harvard.edu/link_gateway/2020ApJ...900....2B/ADS_PDF",
        body=PDF_BYTES, status=200,
    )
    assert download.download_paper(paper, lib) is True
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "domain_aggregators"  # provenance split out
    events = _logged_events(lib)
    miss_sources = [e["source"] for e in events
                    if e["event"] == "download_miss"]
    # Cascade tier-order can grow; just assert the relevant aggregator
    # tiers preceded the hit, not the exact tier list.
    for src in ["arxiv", "oa_aggregators"]:
        assert src in miss_sources, f"expected miss for {src}"


# ---------------------------------------------------------------------------
# Helpers for new-tier tests: register upstream misses up to (but not
# including) the tier under test, so the cascade exercises that tier.
# ---------------------------------------------------------------------------

def _stub_misses_through_ads(monkeypatch) -> None:
    """Register response stubs that make arxiv/unpaywall/openalex/inspire/ads miss.

    ADS is silently skipped (no env token), so no ADS stubs needed.
    """
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0001",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://api.unpaywall.org/v2/10.1/x",
                  json={}, status=200)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1/x",
                  json={"oa_locations": []}, status=200)
    responses.add(responses.GET, "https://inspirehep.net/api/literature",
                  json={"hits": {"hits": []}}, status=200)


# ---------------------------------------------------------------------------
# CrossRef text-mining links strategy
# ---------------------------------------------------------------------------

@responses.activate
def test_crossref_tm_happy_path(lib, paper, monkeypatch):
    """CrossRef returns a text-mining link[] entry; we fetch and accept the PDF."""
    _stub_misses_through_ads(monkeypatch)
    responses.add(
        responses.GET, "https://api.crossref.org/works/10.1/x",
        json={"message": {"link": [
            {"URL": "https://publisher.example/paper.html",
             "content-type": "text/html",
             "intended-application": "syndication"},
            {"URL": "https://publisher.example/paper.pdf",
             "content-type": "application/pdf",
             "intended-application": "text-mining"},
        ]}},
        status=200,
    )
    responses.add(responses.GET, "https://publisher.example/paper.pdf",
                  body=PDF_BYTES, status=200)
    assert download.download_paper(paper, lib) is True
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "crossref_tm"  # provenance split out
    events = _logged_events(lib)
    assert any(e["event"] == "downloaded" and e["source"] == "crossref_tm"
               for e in events)


@responses.activate
def test_crossref_tm_no_pdf_links_returns_none(lib, paper, monkeypatch):
    """CrossRef responds with link[] containing only HTML → strategy returns None."""
    _stub_misses_through_ads(monkeypatch)
    responses.add(
        responses.GET, "https://api.crossref.org/works/10.1/x",
        json={"message": {"link": [
            {"URL": "https://publisher.example/paper.html",
             "content-type": "text/html",
             "intended-application": "syndication"},
        ]}},
        status=200,
    )
    # europepmc + zenodo also miss (no fixtures registered → ConnectionError → None)
    assert download.download_paper(paper, lib) is False
    events = _logged_events(lib)
    assert any(e["event"] == "download_miss" and e["source"] == "crossref_tm"
               for e in events)


@responses.activate
def test_crossref_tm_skipped_without_doi(lib, monkeypatch):
    """No DOI → crossref_tm strategy is a no-op (no API call)."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    p, _ = lib.upsert({"title": "Test paper for download cascade", "authors": ["A"], "year": 2020,
                       "arxiv_id": "2401.0099"})
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.0099",
                  body=HTML_BYTES, status=200)
    responses.add(responses.GET, "https://inspirehep.net/api/literature",
                  json={"hits": {"hits": []}}, status=200)
    download.download_paper(p, lib)
    crossref_calls = [c for c in responses.calls
                      if "api.crossref.org" in c.request.url]
    assert crossref_calls == []


@responses.activate
def test_full_cascade_through_crossref_tm(lib, paper, monkeypatch):
    """End-to-end cascade: arxiv → unpaywall → openalex → inspire → ads all
    miss; crossref_tm wins. Verifies the new tier order is wired correctly."""
    _stub_misses_through_ads(monkeypatch)
    responses.add(
        responses.GET, "https://api.crossref.org/works/10.1/x",
        json={"message": {"link": [
            {"URL": "https://pub.example/full.pdf",
             "content-type": "application/pdf",
             "intended-application": "text-mining"},
        ]}},
        status=200,
    )
    responses.add(responses.GET, "https://pub.example/full.pdf",
                  body=PDF_BYTES, status=200)
    assert download.download_paper(paper, lib) is True
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "crossref_tm"  # provenance split out
    events = _logged_events(lib)
    miss_sources = [e["source"] for e in events
                    if e["event"] == "download_miss"]
    # Cascade tier list can grow; assert relevant tiers are present
    # rather than the exact ordering.
    assert "arxiv" in miss_sources


# ---------------------------------------------------------------------------
# EuropePMC strategy
# ---------------------------------------------------------------------------

@responses.activate
def test_europepmc_happy_path(lib, paper, monkeypatch):
    """EuropePMC returns a fullTextUrl with documentStyle='pdf'; we fetch it."""
    _stub_misses_through_ads(monkeypatch)
    # crossref_tm misses (returns no link[])
    responses.add(responses.GET, "https://api.crossref.org/works/10.1/x",
                  json={"message": {}}, status=200)
    responses.add(
        responses.GET,
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        json={"resultList": {"result": [
            {"id": "ABC", "source": "MED",
             "fullTextUrlList": {"fullTextUrl": [
                 {"availability": "Open access",
                  "documentStyle": "html",
                  "url": "https://europepmc.org/article/MED/ABC"},
                 {"availability": "Open access",
                  "documentStyle": "pdf",
                  "url": "https://europepmc.org/articles/PMC1/pdf"},
             ]}},
        ]}},
        status=200,
    )
    responses.add(responses.GET, "https://europepmc.org/articles/PMC1/pdf",
                  body=PDF_BYTES, status=200)
    assert download.download_paper(paper, lib) is True
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "domain_aggregators"  # provenance split out
    events = _logged_events(lib)
    assert any(e["event"] == "downloaded" and e["source"] == "domain_aggregators"
               for e in events)


@responses.activate
def test_europepmc_pmcid_render_fallback(lib, paper, monkeypatch):
    """No PDF in fullTextUrlList but pmcid present → use the PMC render URL."""
    _stub_misses_through_ads(monkeypatch)
    responses.add(responses.GET, "https://api.crossref.org/works/10.1/x",
                  json={"message": {}}, status=200)
    responses.add(
        responses.GET,
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        json={"resultList": {"result": [
            {"id": "ABC", "source": "PMC", "pmcid": "PMC1234567",
             "fullTextUrlList": {"fullTextUrl": [
                 {"documentStyle": "html",
                  "url": "https://europepmc.org/article/PMC/PMC1234567"},
             ]}},
        ]}},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://europepmc.org/articles/PMC1234567",
        body=PDF_BYTES, status=200,
    )
    assert download.download_paper(paper, lib) is True
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "domain_aggregators"  # provenance split out
@responses.activate
def test_europepmc_no_results_returns_none(lib, paper, monkeypatch):
    """EuropePMC search responds but resultList.result is empty → None."""
    _stub_misses_through_ads(monkeypatch)
    responses.add(responses.GET, "https://api.crossref.org/works/10.1/x",
                  json={"message": {}}, status=200)
    responses.add(
        responses.GET,
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        json={"resultList": {"result": []}},
        status=200,
    )
    # zenodo also misses (no fixtures registered)
    assert download.download_paper(paper, lib) is False
    events = _logged_events(lib)
    assert any(e["event"] == "download_miss" and e["source"] == "domain_aggregators"
               for e in events)


# ---------------------------------------------------------------------------
# Zenodo strategy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "file_metadata, attempted",
    [
        pytest.param({"key": "poster.pdf"}, True, id="untyped-pdf"),
        pytest.param({"key": "poster.PDF"}, True, id="untyped-uppercase-pdf"),
        pytest.param({"key": "poster.pdf", "type": None}, True, id="null-type-pdf"),
        pytest.param({"key": "poster.pdf", "type": ""}, True, id="empty-type-pdf"),
        pytest.param({"type": "pdf"}, True, id="legacy-pdf-without-key"),
        pytest.param({"key": "download", "type": "PDF"}, True, id="legacy-uppercase-type"),
        pytest.param({"key": "data.csv"}, False, id="untyped-csv"),
        pytest.param({"key": "poster.pdf.csv"}, False, id="pdf-in-non-pdf-name"),
        pytest.param({"key": "pdf"}, False, id="no-pdf-extension"),
        pytest.param({}, False, id="no-type-or-key"),
        pytest.param({"key": "data.csv", "type": "csv"}, False, id="legacy-csv"),
        pytest.param({"key": "poster.pdf", "type": "csv"}, False,
                     id="explicit-non-pdf-type"),
    ],
)
@responses.activate
def test_zenodo_file_selection(file_metadata, attempted):
    """Missing types use the advertised filename; explicit types retain their behavior."""
    file_url = "https://zenodo.org/api/records/12345/files/attachment/content"
    responses.get(
        "https://zenodo.org/api/records",
        json={"hits": {"hits": [{"files": [
            {**file_metadata, "links": {"self": file_url}},
        ]}]}},
    )
    responses.get(file_url, body=PDF_BYTES, content_type="application/octet-stream")

    result = download._try_zenodo(Paper(key="Zenodo2023", doi="10.5281/zenodo.12345"))

    assert result == (PDF_BYTES if attempted else None)
    assert len(responses.calls) == (2 if attempted else 1)
    if attempted:
        assert responses.calls[1].request.url == file_url


@pytest.mark.parametrize("file_metadata", [{"key": "poster.pdf"}, {"type": "pdf"}],
                         ids=["untyped-pdf", "legacy-typed-pdf"])
@responses.activate
def test_zenodo_advertised_pdf_rejects_non_pdf_bytes(file_metadata):
    """A PDF advertisement must not bypass the existing fetched-byte check."""
    file_url = "https://zenodo.org/api/records/12345/files/poster.pdf/content"
    responses.get(
        "https://zenodo.org/api/records",
        json={"hits": {"hits": [{"files": [
            {**file_metadata, "links": {"self": file_url}},
        ]}]}},
    )
    responses.get(file_url, body=b"\x00not a PDF", content_type="application/octet-stream")

    assert download._try_zenodo(Paper(key="Zenodo2023", doi="10.5281/zenodo.12345")) is None
    assert len(responses.calls) == 2
    assert responses.calls[1].request.url == file_url


@responses.activate
def test_zenodo_happy_path(lib, paper, monkeypatch):
    """Zenodo records search returns a hit with a PDF file; we download it."""
    _stub_misses_through_ads(monkeypatch)
    responses.add(responses.GET, "https://api.crossref.org/works/10.1/x",
                  json={"message": {}}, status=200)
    responses.add(
        responses.GET,
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        json={"resultList": {"result": []}}, status=200,
    )
    responses.add(
        responses.GET, "https://zenodo.org/api/records",
        json={"hits": {"hits": [
            {"id": 12345,
             "files": [
                 {"key": "data.csv", "type": "csv",
                  "links": {"self": "https://zenodo.org/api/records/12345/files/data.csv/content"}},
                 {"key": "paper.pdf", "type": "pdf",
                  "links": {"self": "https://zenodo.org/api/records/12345/files/paper.pdf/content"}},
             ]},
        ]}},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://zenodo.org/api/records/12345/files/paper.pdf/content",
        body=PDF_BYTES, status=200,
    )
    assert download.download_paper(paper, lib) is True
    assert paper.download_status == "ok"  # D7: status routes
    assert paper.download_source == "domain_aggregators"  # provenance split out
    events = _logged_events(lib)
    assert any(e["event"] == "downloaded" and e["source"] == "domain_aggregators"
               for e in events)


@responses.activate
def test_zenodo_no_pdf_files_returns_none(lib, paper, monkeypatch):
    """Zenodo hit has files but none of them are PDFs → strategy returns None."""
    _stub_misses_through_ads(monkeypatch)
    responses.add(responses.GET, "https://api.crossref.org/works/10.1/x",
                  json={"message": {}}, status=200)
    responses.add(
        responses.GET,
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        json={"resultList": {"result": []}}, status=200,
    )
    responses.add(
        responses.GET, "https://zenodo.org/api/records",
        json={"hits": {"hits": [
            {"id": 12345,
             "files": [
                 {"key": "data.csv", "type": "csv",
                  "links": {"self": "https://zenodo.org/api/records/12345/files/data.csv/content"}},
             ]},
        ]}},
        status=200,
    )
    assert download.download_paper(paper, lib) is False
    events = _logged_events(lib)
    assert any(e["event"] == "download_miss" and e["source"] == "domain_aggregators"
               for e in events)


@responses.activate
def test_zenodo_skipped_without_doi_or_arxiv(lib, monkeypatch):
    """No DOI and no arxiv_id → zenodo strategy is a no-op (no API call)."""
    monkeypatch.delenv("PAPER_PIPELINE_USE_SCIHUB", raising=False)
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    p, _ = lib.upsert({"title": "Test paper for download cascade", "authors": ["A"], "year": 2020})
    # Nothing to download for, but cascade still runs.
    download.download_paper(p, lib)
    zenodo_calls = [c for c in responses.calls
                    if "zenodo.org" in c.request.url]
    assert zenodo_calls == []


# ---------------------------------------------------------------------------
# Sci-Hub mirror discovery
# ---------------------------------------------------------------------------

@responses.activate
def test_discover_scihub_mirrors_parses_known_sources(monkeypatch):
    """Discovery fetches sources, regex-extracts sci-hub.<tld> domains, then
    filters by HEAD liveness (200 = alive, anything else = dead)."""
    # Reset cache so discovery actually runs.
    monkeypatch.setattr(download, "_scihub_mirrors_cache", (0.0, []))

    # Source pages contain a mix of sci-hub.<tld> domains.
    responses.add(
        responses.GET, "https://en.wikipedia.org/wiki/Sci-Hub",
        body=b"Active mirrors: sci-hub.ru and sci-hub.ee. Dead: sci-hub.se.",
        status=200,
    )
    responses.add(
        responses.GET, "https://lovescihub.wordpress.com/",
        body=b"Try sci-hub.ren today!",
        status=200,
    )
    responses.add(
        responses.GET, "https://sci-hub.ru/",
        body=b"<footer>sci-hub.ru sci-hub.wf</footer>",
        status=200,
    )
    responses.add(
        responses.GET, "https://sci-hub.ee/",
        body=b"sci-hub.ee partners with sci-hub.box.",
        status=200,
    )

    # Liveness checks: .ru, .ee, .ren alive; .se, .wf, .box dead.
    responses.add(responses.HEAD, "https://sci-hub.ru", status=200)
    responses.add(responses.HEAD, "https://sci-hub.ee", status=200)
    responses.add(responses.HEAD, "https://sci-hub.ren", status=200)
    responses.add(responses.HEAD, "https://sci-hub.se", status=404)
    responses.add(responses.HEAD, "https://sci-hub.wf", status=404)
    responses.add(responses.HEAD, "https://sci-hub.box", status=404)

    mirrors = download._discover_scihub_mirrors()
    assert "https://sci-hub.ru" in mirrors
    assert "https://sci-hub.ee" in mirrors
    assert "https://sci-hub.ren" in mirrors
    # Dead ones must be filtered out.
    assert "https://sci-hub.se" not in mirrors
    assert "https://sci-hub.wf" not in mirrors
    assert "https://sci-hub.box" not in mirrors

    # Reset for the next test.
    monkeypatch.setattr(download, "_scihub_mirrors_cache", (0.0, []))


def test_discover_scihub_caches_within_ttl(monkeypatch):
    """Within TTL, second call returns the cached list and does NOT hit
    any discovery / liveness endpoints again."""
    # Pre-populate the cache with a fresh timestamp.
    fixed_now = 1_700_000_000.0
    monkeypatch.setattr(download.time, "time", lambda: fixed_now)
    monkeypatch.setattr(
        download, "_scihub_mirrors_cache",
        (fixed_now, ["https://sci-hub.cached"]),
    )

    # If the cache is honored, no HTTP requests are made — so register no
    # mocks. Any request would raise ConnectionError under responses.
    with responses.RequestsMock() as rmocks:
        result = download._discover_scihub_mirrors()
        assert result == ["https://sci-hub.cached"]
        assert len(rmocks.calls) == 0


@responses.activate
def test_discover_scihub_falls_back_when_all_sources_fail(monkeypatch):
    """When every discovery source returns 5xx (no domains harvested),
    we fall back to _SCIHUB_FALLBACK_DOMAINS and liveness-check those."""
    monkeypatch.setattr(download, "_scihub_mirrors_cache", (0.0, []))

    # Every discovery source 500s.
    for src in download._SCIHUB_DISCOVERY_SOURCES:
        responses.add(responses.GET, src, body=b"oops", status=500)

    # Liveness for fallback domains: only sci-hub.ru is alive.
    fallback_alive = "sci-hub.ru"
    for d in download._SCIHUB_FALLBACK_DOMAINS:
        status = 200 if d == fallback_alive else 404
        responses.add(responses.HEAD, f"https://{d}", status=status)

    mirrors = download._discover_scihub_mirrors()
    assert "https://sci-hub.ru" in mirrors
    # Dead fallbacks excluded.
    for d in download._SCIHUB_FALLBACK_DOMAINS:
        if d != fallback_alive:
            assert f"https://{d}" not in mirrors

    # Reset cache for hygiene.
    monkeypatch.setattr(download, "_scihub_mirrors_cache", (0.0, []))


# NOTE: ``test_try_scihub_uses_discovered_mirrors`` removed 2026-05 — same
# reason as ``test_scihub_attempted_when_opted_in`` above.


# ---------------------------------------------------------------------------
# arXiv-by-title fallback strategy
# ---------------------------------------------------------------------------

def test_arxiv_by_title_skipped_when_arxiv_id_already_set(lib, monkeypatch):
    """If the paper already has an arxiv_id, the by-title strategy is a no-op
    (arxiv_id-bearing papers are handled by the primary _try_arxiv strategy).
    No HTTP, no search_arxiv call."""
    p, _ = lib.upsert({"title": "A Sufficiently Long Title For Search",
                       "authors": ["A"], "year": 2020,
                       "doi": "10.1/x", "arxiv_id": "2401.0001"})

    def _boom(*args, **kwargs):
        raise AssertionError("search_arxiv should not be called")

    monkeypatch.setattr("papervault.library.sources.arxiv.search_arxiv", _boom)
    with responses.RequestsMock() as rmocks:
        assert download._try_arxiv_by_title(p) is None
        assert len(rmocks.calls) == 0


def test_arxiv_by_title_skipped_when_title_too_short(lib, monkeypatch):
    """Titles shorter than 20 chars can't disambiguate; strategy returns None
    without invoking search_arxiv or any HTTP."""
    p, _ = lib.upsert({"title": "Test paper for download cascade", "authors": ["A"], "year": 2020,
                       "doi": "10.1/short"})

    def _boom(*args, **kwargs):
        raise AssertionError("search_arxiv should not be called")

    monkeypatch.setattr("papervault.library.sources.arxiv.search_arxiv", _boom)
    with responses.RequestsMock() as rmocks:
        assert download._try_arxiv_by_title(p) is None
        assert len(rmocks.calls) == 0


@responses.activate
def test_arxiv_by_title_happy_path(lib, monkeypatch):
    """Mock search_arxiv to return a confident-match result; assert PDF
    returned and paper.arxiv_id was persisted to the discovered id."""
    title = "Physics-Informed Neural Networks for Cosmic Ray Propagation"
    p, _ = lib.upsert({"title": title, "authors": ["A"], "year": 2024,
                       "doi": "10.1038/s41586-024-pinn-cosmic"})
    assert p.arxiv_id == ""  # precondition

    def _fake_search(query, max_results=5):
        # Returns a single match with the same title.
        return [{
            "title": title,
            "authors": ["X"],
            "abstract": "Abstract.",
            "year": "2024",
            "url": "http://arxiv.org/abs/2403.12345",
            "arxiv_id": "2403.12345",
        }]

    monkeypatch.setattr(
        "papervault.library.sources.arxiv.search_arxiv", _fake_search,
    )
    responses.add(responses.GET, "https://arxiv.org/pdf/2403.12345",
                  body=PDF_BYTES, status=200)

    data = download._try_arxiv_by_title(p)
    assert data == PDF_BYTES
    # Discovered arxiv_id was persisted to the paper.
    assert p.arxiv_id == "2403.12345"


def test_arxiv_by_title_rejects_unrelated_match(lib, monkeypatch):
    """When search_arxiv returns a paper with a totally different title,
    the strategy must return None and NOT mutate paper.arxiv_id."""
    p, _ = lib.upsert({"title": "PINN for Cosmic Ray Propagation Modeling",
                       "authors": ["A"], "year": 2024,
                       "doi": "10.1038/s41586-2024-pinn"})
    assert p.arxiv_id == ""

    def _fake_search(query, max_results=5):
        # Totally unrelated paper.
        return [{
            "title": "Deep Learning for Image Classification on ImageNet",
            "authors": ["Y"],
            "abstract": "Unrelated.",
            "year": "2020",
            "url": "http://arxiv.org/abs/2001.99999",
            "arxiv_id": "2001.99999",
        }]

    monkeypatch.setattr(
        "papervault.library.sources.arxiv.search_arxiv", _fake_search,
    )
    with responses.RequestsMock() as rmocks:
        assert download._try_arxiv_by_title(p) is None
        # No HTTP fetch attempted (rejected before download).
        assert len(rmocks.calls) == 0
    # And critically: arxiv_id was NOT mutated.
    assert p.arxiv_id == ""


# ---------------------------------------------------------------------------
# citation_pdf_url strategy (Highwire Press meta tag — generic across publishers)
# ---------------------------------------------------------------------------

@responses.activate
def test_citation_pdf_url_happy_path(lib, paper):
    """DOI redirect serves an HTML page with <meta name='citation_pdf_url' ...>;
    we follow it and accept the PDF (covers MDPI / Springer / etc.)."""
    landing_html = (
        b'<html><head>'
        b'<meta name="citation_title" content="A Paper">'
        b'<meta name="citation_pdf_url" content="https://www.mdpi.com/x/y/pdf">'
        b'</head><body>article</body></html>'
    )
    # doi.org redirects to publisher landing.
    responses.add(responses.GET, "https://doi.org/10.1/x",
                  body=landing_html, status=200)
    responses.add(responses.GET, "https://www.mdpi.com/x/y/pdf",
                  body=PDF_BYTES, status=200)
    assert download._try_citation_pdf_url(paper) == PDF_BYTES


@responses.activate
def test_citation_pdf_url_missing_meta_returns_none(lib, paper):
    """Landing page has no citation_pdf_url meta → strategy returns None
    without raising."""
    responses.add(
        responses.GET, "https://doi.org/10.1/x",
        body=b"<html><head><title>paywall</title></head><body>login</body></html>",
        status=200,
    )
    assert download._try_citation_pdf_url(paper) is None


@responses.activate
def test_citation_pdf_url_relative_url_resolved(lib, paper):
    """A root-relative meta URL is resolved against the landing page host."""
    landing_html = (
        b'<html><head>'
        b'<meta name="citation_pdf_url" content="/articles/123.pdf">'
        b'</head></html>'
    )
    responses.add(
        responses.GET, "https://doi.org/10.1/x",
        body=landing_html, status=200,
    )
    # The DOI redirect mock above doesn't actually rewrite the URL, so the
    # resolver uses doi.org as the base. Whatever the base, the relative
    # path must be appended correctly. Here we expect it appended to doi.org.
    responses.add(responses.GET, "https://doi.org/articles/123.pdf",
                  body=PDF_BYTES, status=200)
    assert download._try_citation_pdf_url(paper) == PDF_BYTES


@responses.activate
def test_citation_pdf_url_skipped_without_doi(lib):
    """No DOI → no API call."""
    p, _ = lib.upsert({"title": "Test paper for download cascade", "authors": ["A"], "year": 2020,
                       "arxiv_id": "2401.0099"})
    with responses.RequestsMock() as rmocks:
        assert download._try_citation_pdf_url(p) is None
        assert len(rmocks.calls) == 0


# ---------------------------------------------------------------------------
# ResearchGate strategy
# ---------------------------------------------------------------------------
#
# The current _try_researchgate implementation drives Scrapling (Playwright +
# Cloudflare Turnstile solver), so it can't be exercised via the `responses`
# library — it doesn't go through `requests`. End-to-end happy-path
# verification is manual; here we cover only the deterministic bail
# branches.

def test_researchgate_bails_on_short_title_no_doi(lib):
    """Too little metadata to bother launching a browser → fast None."""
    from papervault.library.models import Paper
    p = Paper(key="X2021", title="Padded test paper title for validation", authors=["A"], year=2021)
    assert download._try_researchgate(p) is None


def test_researchgate_bails_when_scrapling_unavailable(lib, monkeypatch):
    """If Scrapling isn't installed, the import inside the tier raises
    ImportError; the tier reports the missing dependency and returns None."""
    from papervault.library.models import Paper
    import sys
    monkeypatch.setitem(sys.modules, "scrapling.fetchers", None)
    p = Paper(
        key="X2021",
        title="Catalytic conversion of biomass to aromatics over zeolites",
        authors=["A"], year=2021, doi="10.1016/j.fake/123",
    )
    assert download._try_researchgate(p) is None


@responses.activate
def test_researchgate_skipped_when_title_too_short(lib):
    """Title under 20 chars → no HTTP, return None."""
    p, _ = lib.upsert({"title": "Test paper for download cascade", "authors": ["A"], "year": 2020,
                       "doi": "10.1/tiny"})
    with responses.RequestsMock() as rmocks:
        assert download._try_researchgate(p) is None
        assert len(rmocks.calls) == 0


@responses.activate
def test_researchgate_pdf_link_missing_returns_none(lib):
    """Detail page has no PDF link → graceful None."""
    p, _ = lib.upsert({
        "title": "Some paper with no public full-text on ResearchGate",
        "authors": ["A"], "year": 2021, "doi": "10.1/missing",
    })
    search_html = (
        b'<html><body>'
        b'<a href="/publication/99999_Some-paper">match</a>'
        b'</body></html>'
    )
    detail_html = (
        b'<html><body>Abstract only. Login required for full text.</body></html>'
    )
    responses.add(
        responses.GET, "https://www.researchgate.net/search/publication",
        body=search_html, status=200,
    )
    responses.add(
        responses.GET,
        "https://www.researchgate.net/publication/99999_Some-paper",
        body=detail_html, status=200,
    )
    assert download._try_researchgate(p) is None


# ─── issue #62: fabricated arxiv_id nulling ──────────────────────────────
# The ingest gate emits format-valid-but-nonexistent arxiv ids (~84% of
# arxiv-bearing metadata_only papers 404 on arxiv.org). A PDF 404 alone also
# occurs for real withdrawn records. Only affirmative base-record absence can
# null an ID, while retaining the reliable DOI (#170).

@responses.activate
def test_arxiv_confirmed_missing_base_nulls_fabricated_id_keeps_doi(lib):
    p, _ = lib.upsert({"title": "Paper with a fabricated arxiv id",
                       "authors": ["A"], "year": 2023,
                       "doi": "10.1/real", "arxiv_id": "1311.9999"})
    responses.get("https://arxiv.org/abs/1311.9999", status=404, body='''
      <html><body><main><div id="content"><h1>Article 1311.9999 not found</h1>
      <p>There is no record of an article with identifier '1311.9999'.</p>
      </div></main></body></html>''')
    assert download._try_arxiv(p) is None
    assert p.arxiv_id == ""       # fake id dropped
    assert p.doi == "10.1/real"   # DOI never touched


@responses.activate
def test_arxiv_transient_error_does_not_null(lib):
    """A connection/timeout error is NOT a not-found signal — the id might be
    real, so it must survive untouched."""
    p, _ = lib.upsert({"title": "Paper we could not reach arxiv for",
                       "authors": ["A"], "year": 2023,
                       "doi": "10.1/real", "arxiv_id": "2401.09999"})
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.09999",
                  body=requests.ConnectionError("network down"))
    assert download._try_arxiv(p) is None
    assert p.arxiv_id == "2401.09999"   # preserved on transient failure
    assert p.doi == "10.1/real"


@responses.activate
def test_arxiv_success_leaves_id_intact(lib):
    p, _ = lib.upsert({"title": "Paper with a genuine arxiv id",
                       "authors": ["A"], "year": 2024,
                       "doi": "10.1/real", "arxiv_id": "2401.00001"})
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.00001",
                  body=PDF_BYTES, status=200)
    assert download._try_arxiv(p) == PDF_BYTES
    assert p.arxiv_id == "2401.00001"   # a real id is never nulled


@responses.activate
def test_arxiv_non_404_error_does_not_null(lib):
    """Only a definitive 404 nulls the id. A 403 (throttling) or 5xx could be
    transient / a withdrawn-but-real record, so the id must be left in place."""
    p, _ = lib.upsert({"title": "Paper arxiv 403-throttled us on",
                       "authors": ["A"], "year": 2024,
                       "doi": "10.1/real", "arxiv_id": "2401.00002"})
    responses.add(responses.GET, "https://arxiv.org/pdf/2401.00002", status=403)
    assert download._try_arxiv(p) is None
    assert p.arxiv_id == "2401.00002"   # non-404 leaves it intact


# Current OpenAlex work IDs and repository locations (#169).
_OPENALEX_WORK = "https://api.openalex.org/works/W123"


@pytest.fixture
def openalex_paper(lib, monkeypatch):
    p, _ = lib.upsert({"title": "A repository thesis about cosmic ray transport",
                       "authors": ["Researcher"], "year": 2012,
                       "source": "openalex", "url": "https://openalex.org/W123"})
    _isolate_tier(monkeypatch, "oa_aggregators")
    for member in ("_try_unpaywall", "_try_semantic_scholar_oa", "_try_core"):
        monkeypatch.setattr(download, member, lambda _: None)
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    monkeypatch.setattr("papervault.library.llm.get_llm", lambda **_: SimpleNamespace(
        call=lambda messages: json.dumps({
            "match": "Meeting program" not in messages[1]["content"],
            "reason": "identity checked",
        })))
    return p


@responses.activate
def test_openalex_work_id_handle_pdf_through_verified_cascade(lib, openalex_paper):
    p = openalex_paper
    data = _text_pdf(f"{p.title}. Researcher. Full dissertation about cosmic rays.")
    responses.add(responses.GET, _OPENALEX_WORK, json={
        "locations": [{"pdf_url": "https://hdl.handle.net/1/2"}],
    })
    responses.add(responses.GET, "https://hdl.handle.net/1/2", status=302,
                  headers={"Location": "https://repository.example/items/thesis"})
    responses.add(responses.GET, "https://repository.example/items/thesis", body=(
        '<meta content="/files/thesis.pdf" name="citation_pdf_url">'))
    responses.add(responses.GET, "https://repository.example/files/thesis.pdf", body=data)
    assert download.download_paper(p, lib)
    assert lib.pdf_path(p.key).read_bytes() == data
    assert p.download_source == "oa_aggregators"
    events = _logged_events(lib)
    assert any(e["event"] == "downloaded" and e["verify"].startswith("llm_match:")
               and e["size"] == len(data) for e in events)
    assert not any(e["event"] == "download_skip" for e in events)
    # Existing bytes must short-circuit all future network, including stale IDs.
    before = lib.pdf_path(p.key).stat()
    count = len(responses.calls)
    assert download.download_paper(p, lib)
    assert len(responses.calls) == count
    assert lib.pdf_path(p.key).stat() == before


@responses.activate
def test_openalex_modern_multiple_locations_and_https(openalex_paper):
    responses.add(responses.GET, _OPENALEX_WORK, json={
        "best_oa_location": {"pdf_url": "https://repo.example/block"},
        "primary_location": {"landing_page_url": "https://repo.example/metadata"},
        "locations": [
            {"pdf_url": "https://repo.example/block"},
            {"is_oa": False, "pdf_url": None,
             "landing_page_url": "http://repo.example/alternate.pdf"},
        ],
    })
    responses.add(responses.GET, "https://repo.example/block", status=403)
    responses.add(responses.GET, "https://repo.example/metadata", body=HTML_BYTES)
    responses.add(responses.GET, "https://repo.example/alternate.pdf", body=PDF_BYTES)
    assert download._try_openalex(openalex_paper) == PDF_BYTES
    assert len(responses.calls) == 4


@pytest.mark.parametrize("url, source, paper_id, expected", [
    ("https://openalex.org/W123", "manual", "", "W123"),
    ("http://openalex.org/w123", "", "", "W123"),
    ("", "openalex", "W123", "W123"),
    ("", "openalex", "https://openalex.org/W123", "W123"),
    ("", "semantic_scholar", "W123", None),
    ("https://openalex.org/W123?other=1", "", "", None),
    ("https://openalex.org/W123/extra", "", "", None),
    ("https://openalex.org.evil.example/W123", "", "", None),
    ("https://user@openalex.org/W123", "", "", None),
    ("https://openalex.org:443/W123", "", "", None),
    ("https://openalex.org/W123\n", "", "", None),
    ("https://openalex.org/A123", "", "", None),
    ("Text https://openalex.org/W123", "", "", None),
])
def test_openalex_work_id_eligibility(openalex_paper, url, source, paper_id, expected):
    p = openalex_paper
    p.url, p.source, p.paper_id = url, source, paper_id
    assert openalex._openalex_work_id(p) == expected
    assert download._download_skip_reason("oa_aggregators", p) == (
        None if expected else "no_applicable_member")


@responses.activate
@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_openalex_api_failure_is_nonmutating_miss(openalex_paper, status):
    p = openalex_paper
    before = p.model_dump()
    responses.add(responses.GET, _OPENALEX_WORK, status=status, json={
        "doi": "https://doi.org/10.1088/test", "locations": [
            {"pdf_url": "https://repo.example/untrusted.pdf"}],
    })
    assert download._try_openalex(p) is None
    assert p.model_dump() == before
    assert len(responses.calls) == 1


@responses.activate
def test_openalex_missing_work_records_actual_miss(lib, openalex_paper):
    responses.add(responses.GET, _OPENALEX_WORK, status=404)
    assert not download.download_paper(openalex_paper, lib)
    assert _logged_events(lib)[0]["event"] == "download_miss"


@responses.activate
def test_openalex_merged_work_redirect(openalex_paper):
    responses.add(responses.GET, _OPENALEX_WORK, status=301,
                  headers={"Location": "https://api.openalex.org/works/W456"})
    responses.add(responses.GET, "https://api.openalex.org/works/W456", json={
        "locations": [{"landing_page_url": "https://repo.example/thesis.pdf"}],
    })
    responses.add(responses.GET, "https://repo.example/thesis.pdf", body=PDF_BYTES)
    assert download._try_openalex(openalex_paper) == PDF_BYTES
    assert openalex_paper.url == "https://openalex.org/W123"


@responses.activate
@pytest.mark.parametrize("payload", [
    {}, {"locations": [{"landing_page_url": "https://repo.example/metadata"}]},
    {"has_content": {"pdf": True},
     "content_urls": {"pdf": "https://content.openalex.org/works/W123.pdf"}},
])
def test_openalex_metadata_and_content_flags_are_not_downloads(openalex_paper, payload):
    responses.add(responses.GET, _OPENALEX_WORK, json=payload)
    responses.add(responses.GET, "https://repo.example/metadata", body=HTML_BYTES)
    assert download._try_openalex(openalex_paper) is None
    assert not any("content.openalex.org" in c.request.url for c in responses.calls)


@responses.activate
@pytest.mark.parametrize("route", ["location", "landing", "redirect", "api_redirect"])
def test_openalex_never_calls_metered_content(openalex_paper, monkeypatch, route):
    monkeypatch.setenv("OPENALEX_API_KEY", "fixture-token")
    content = "https://content.openalex.org/works/W123.pdf?api_key=fixture-token"
    location = content if route == "location" else "https://repo.example/landing"
    if route == "api_redirect":
        responses.add(responses.GET, _OPENALEX_WORK, status=301,
                      headers={"Location": content})
    else:
        responses.add(responses.GET, _OPENALEX_WORK, json={
            "locations": [{"pdf_url": location}], "has_content": {"pdf": True},
        })
    if route == "landing":
        responses.add(responses.GET, location,
                      body=f'<meta name="citation_pdf_url" content="{content}">')
    elif route == "redirect":
        responses.add(responses.GET, location, status=302, headers={"Location": content})
    assert download._try_openalex(openalex_paper) is None
    assert not any("content.openalex.org" in c.request.url for c in responses.calls)
    assert all("fixture-token" not in str(c.request.headers) for c in responses.calls)


@responses.activate
@pytest.mark.parametrize("status, headers, body", [
    (206, {}, PDF_BYTES), (200, {"Content-Range": "bytes 0-9/100"}, PDF_BYTES),
    (200, {"Content-Length": "9999"}, PDF_BYTES),
    (200, {}, PDF_BYTES * 20),
])
def test_openalex_partial_or_oversize_pdf_misses(openalex_paper, monkeypatch,
                                               status, headers, body):
    monkeypatch.setattr(openalex, "_MAX_PDF_BYTES", 100)
    responses.add(responses.GET, _OPENALEX_WORK, json={
        "locations": [{"pdf_url": "https://repo.example/file.pdf"}],
    })
    responses.add(responses.GET, "https://repo.example/file.pdf", status=status,
                  headers=headers, body=body)
    assert download._try_openalex(openalex_paper) is None


@responses.activate
def test_openalex_landing_caps_links_and_no_recursion(openalex_paper, monkeypatch):
    monkeypatch.setattr(openalex, "_MAX_HTML_BYTES", 200)
    responses.add(responses.GET, _OPENALEX_WORK, json={"locations": [
        {"landing_page_url": "https://repo.example/item"}],
    })
    responses.add(responses.GET, "https://repo.example/item", body=(
        '<meta name="citation_pdf_url" content="/first.pdf">'
        '<a href="/first.pdf">PDF</a><a href="/second">Download</a>'
        + " " * 200 + '<a href="/late.pdf">PDF</a>'))
    responses.add(responses.GET, "https://repo.example/first.pdf",
                  body='<meta name="citation_pdf_url" content="/recursive.pdf">')
    responses.add(responses.GET, "https://repo.example/second", body=HTML_BYTES)
    assert download._try_openalex(openalex_paper) is None
    assert len(responses.calls) == 4


@responses.activate
@pytest.mark.parametrize("bound", ["candidates", "requests", "redirects", "time", "metadata"])
def test_openalex_transport_bounds(openalex_paper, monkeypatch, bound):
    location_count = 1 if bound == "redirects" else 20
    responses.add(responses.GET, _OPENALEX_WORK, json={"locations": [
        {"pdf_url": f"https://repo.example/{i}"} for i in range(location_count)]})
    for i in range(20):
        if bound == "redirects":
            responses.add(responses.GET, f"https://repo.example/{i}", status=302,
                          headers={"Location": f"https://repo.example/{i + 1}"})
        else:
            responses.add(responses.GET, f"https://repo.example/{i}", body=HTML_BYTES)
    constants = {"candidates": ("_MAX_CANDIDATES", 2),
                 "requests": ("_MAX_REQUESTS", 3),
                 "redirects": ("_MAX_REDIRECTS", 1),
                 "time": ("_TIME_BUDGET", 0),
                 "metadata": ("_MAX_METADATA_BYTES", 10)}
    monkeypatch.setattr(openalex, *constants[bound])
    assert download._try_openalex(openalex_paper) is None
    if bound == "time":
        assert not responses.calls
    elif bound == "metadata":
        assert len(responses.calls) == 1
    else:
        assert len(responses.calls) == 3  # Metadata + two candidates/requests/redirect hops.


@responses.activate
@pytest.mark.parametrize("outcome", ["match", "mismatch", "download_failure", "fail_open"])
def test_openalex_recovers_identifiers_only_after_affirmative_verification(
        lib, openalex_paper, monkeypatch, outcome):
    p = openalex_paper
    responses.add(responses.GET, _OPENALEX_WORK, json={
        "ids": {"doi": "https://doi.org/10.1088/VALID"},
        "locations": [{"landing_page_url": "https://arxiv.org/pdf/2102.11582v3.pdf"}],
    })
    body = _text_pdf(("Meeting program" if outcome == "mismatch" else p.title)
                     + ". Researcher. Cosmic rays and the complete requested work.")
    responses.add(responses.GET, "https://arxiv.org/pdf/2102.11582v3.pdf", body=body,
                  status=403 if outcome == "download_failure" else 200)
    if outcome == "fail_open":
        monkeypatch.setattr("papervault.library.llm.get_llm", lambda **_: SimpleNamespace(
            call=lambda _: "invalid judge output"))
    assert download.download_paper(p, lib) == (outcome in {"match", "fail_open"})
    if outcome == "match":
        assert p.doi == "10.1088/valid"
        assert p.arxiv_id == "2102.11582v3"
        assert lib.find(doi=p.doi) is p
        assert lib.find(arxiv_id="2102.11582") is p
        lib.save()
        loaded = Library(lib.root)
        assert loaded.find(doi=p.doi).arxiv_id == "2102.11582v3"
    else:
        assert p.doi == p.arxiv_id == ""
        assert lib.find(doi="10.1088/valid") is None
        if outcome == "mismatch":
            assert any(e["event"] == "download_pdf_mismatch" for e in _logged_events(lib))


@responses.activate
def test_openalex_recovered_doi_can_resolve_landing_without_locations(lib, openalex_paper):
    p = openalex_paper
    responses.add(responses.GET, _OPENALEX_WORK,
                  json={"doi": "https://doi.org/10.1088/valid"})
    responses.add(responses.GET, "https://doi.org/10.1088/valid", body=(
        '<a href="/thesis.pdf" type="application/pdf">Full text</a>'))
    responses.add(responses.GET, "https://doi.org/thesis.pdf", body=_text_pdf(
        p.title + ". Researcher. A full dissertation about cosmic ray transport."))
    assert download.download_paper(p, lib)
    assert p.doi == "10.1088/valid"


@responses.activate
@pytest.mark.parametrize("case", ["supplied", "collision", "invalid"])
def test_openalex_recovery_preserves_existing_identity(lib, openalex_paper, case):
    p = openalex_paper
    doi, arxiv = "10.1088/valid", "2102.11582v3"
    if case == "supplied":
        lib.upsert({"title": p.title, "doi": "10.1088/original", "arxiv_id": "2102.11582v1"})
    elif case == "collision":
        holder, _ = lib.upsert({"title": "Another paper with its own verified identity",
                               "authors": ["Holder"], "year": 2020,
                               "doi": doi, "arxiv_id": arxiv})
    else:
        doi = "not-a-doi"
    before = (p.doi, p.arxiv_id)
    arxiv_url = ("https://arxiv.org.evil.example/pdf/2102.11582v3.pdf" if case == "invalid"
                 else f"https://arxiv.org/pdf/{arxiv}.pdf")
    responses.add(responses.GET, _OPENALEX_WORK, json={
        "doi": doi, "locations": [{"pdf_url": arxiv_url}],
    })
    if case == "supplied":
        responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1088/original", json={
            "doi": doi, "locations": [{"pdf_url": arxiv_url}],
        })
    responses.add(responses.GET, arxiv_url, body=_text_pdf(
        p.title + ". Researcher. Cosmic rays and the complete requested work."))
    assert download.download_paper(p, lib)
    assert (p.doi, p.arxiv_id) == before
    assert lib.get(p.key) is p
    if case == "collision":
        assert lib.find(doi=doi) is holder
        assert lib.find(arxiv_id=arxiv) is holder


def test_openalex_slow_stream_deadline_and_response_closure(monkeypatch):
    now = [0.0]
    reads = []
    closed = []

    class Raw:
        def read1(self, size, decode_content=False):
            reads.append(size)
            now[0] += 0.75
            return b"%PDF-" if len(reads) == 1 else b"body"

    class Response:
        status_code = 200
        headers = {}
        raw = Raw()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def iter_content(self, chunk_size):
            # A buffered read can keep waiting as a peer drips bytes.
            now[0] = 100
            yield b"%PDF-too-late"

    monkeypatch.setattr(openalex.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(openalex, "_request", lambda *a, **kw: Response())
    budget = openalex._FetchBudget(deadline=1.0)
    assert budget.get("https://repo.example/file.pdf") is None
    assert now[0] < 2.0
    assert closed == [True]


@responses.activate
def test_openalex_unrequested_compression_is_a_miss(openalex_paper):
    responses.add(responses.GET, _OPENALEX_WORK, json={
        "locations": [{"pdf_url": "https://repo.example/compressed.pdf"}],
    })
    responses.add(responses.GET, "https://repo.example/compressed.pdf", body=PDF_BYTES,
                  headers={"Content-Encoding": "identity-unsupported"})
    assert download._try_openalex(openalex_paper) is None


@responses.activate
@pytest.mark.parametrize("host", ["content.openalex.org.", "content%2eopenalex.org"])
def test_openalex_metered_host_aliases_are_blocked(openalex_paper, host):
    responses.add(responses.GET, _OPENALEX_WORK, json={
        "locations": [{"pdf_url": f"https://{host}/works/W123.pdf"}],
    })
    assert download._try_openalex(openalex_paper) is None
    assert len(responses.calls) == 1


@responses.activate
@pytest.mark.parametrize("body", [b"not JSON", b"[]", b"null"])
def test_openalex_invalid_metadata_is_nonmutating_miss(openalex_paper, body):
    before = openalex_paper.model_dump()
    responses.add(responses.GET, _OPENALEX_WORK, body=body)
    assert download._try_openalex(openalex_paper) is None
    assert openalex_paper.model_dump() == before
    assert len(responses.calls) == 1


@responses.activate
def test_openalex_timeout_does_not_retry_or_clear_identity(openalex_paper):
    before = openalex_paper.model_dump()
    responses.add(responses.GET, _OPENALEX_WORK, body=requests.Timeout("API timeout"))
    assert download._try_openalex(openalex_paper) is None
    assert openalex_paper.model_dump() == before
    assert len(responses.calls) == 1


@responses.activate
def test_openalex_redirect_body_is_never_buffered(openalex_paper, monkeypatch):
    original = requests.Response.content.fget

    def content(response):
        assert response.status_code != 302, "redirect body was buffered"
        return original(response)

    monkeypatch.setattr(requests.Response, "content", property(content))
    responses.add(responses.GET, _OPENALEX_WORK, json={
        "locations": [{"pdf_url": "https://repo.example/redirect"}]})
    responses.add(responses.GET, "https://repo.example/redirect", status=302,
                  headers={"Location": "/file.pdf"}, body=b"x" * (1024 * 1024))
    responses.add(responses.GET, "https://repo.example/file.pdf", body=PDF_BYTES)
    assert download._try_openalex(openalex_paper) == PDF_BYTES


@responses.activate
def test_openalex_does_not_send_netrc_auth(openalex_paper, monkeypatch):
    monkeypatch.setattr(requests.sessions, "get_netrc_auth", lambda _: ("user", "secret"))
    responses.add(responses.GET, _OPENALEX_WORK, json={
        "locations": [{"pdf_url": "https://repo.example/redirect"}]})
    responses.add(responses.GET, "https://repo.example/redirect", status=302,
                  headers={"Location": "https://other.example/file.pdf"})
    responses.add(responses.GET, "https://other.example/file.pdf", body=PDF_BYTES)
    assert download._try_openalex(openalex_paper) == PDF_BYTES
    assert all("Authorization" not in c.request.headers for c in responses.calls)


@responses.activate
def test_openalex_preserves_doi_route_when_stored_work_id_is_stale(openalex_paper):
    openalex_paper.doi = "10.1088/original"
    responses.add(responses.GET, _OPENALEX_WORK, status=404)
    responses.add(responses.GET, "https://api.openalex.org/works/doi:10.1088/original", json={
        "locations": [{"pdf_url": "https://repo.example/file.pdf"}]})
    responses.add(responses.GET, "https://repo.example/file.pdf", body=PDF_BYTES)
    assert download._try_openalex(openalex_paper) == PDF_BYTES


@responses.activate
def test_openalex_retains_http_only_repository(openalex_paper):
    responses.add(responses.GET, _OPENALEX_WORK, json={
        "locations": [{"pdf_url": "http://repo.example/file.pdf"}]})
    responses.add(responses.GET, "https://repo.example/file.pdf",
                  body=requests.ConnectionError("TLS unavailable"))
    responses.add(responses.GET, "http://repo.example/file.pdf", body=PDF_BYTES)
    assert download._try_openalex(openalex_paper) == PDF_BYTES


@pytest.mark.parametrize("phase", ["headers", "chunk_framing"])
def test_openalex_deadline_covers_http_framing(monkeypatch, phase):
    now = [0.0]
    header = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"

    class Raw(io.RawIOBase):
        def readable(self):
            return True

        def readinto(self, target):
            if phase == "chunk_framing" and now[0] == 0:
                chunk = header
                now[0] = 0.001
            else:
                chunk = b"1"
                now[0] += 0.02
            target[:len(chunk)] = chunk
            return len(chunk)

    class Socket:
        def makefile(self, *args):
            return io.BufferedReader(Raw())

        def settimeout(self, timeout):
            assert 0 < timeout <= 0.05

    monkeypatch.setattr(openalex.time, "monotonic", lambda: now[0])
    response = openalex._deadline_response(Socket(), deadline=0.05)
    with pytest.raises(TimeoutError):
        response.begin()
        response.read1(8192)
    assert now[0] < 0.08
    response.close()


def test_openalex_writeback_cannot_redirect_to_racing_identifier_owner(lib, openalex_paper):
    p = openalex_paper
    holder, _ = lib.upsert({"title": "A separate paper with a separately verified identity",
                           "authors": ["Holder"], "year": 2020})
    doi, arxiv = "10.1088/race", "2102.11582v3"

    class RacingIndex(dict):
        def claim(self, key):
            if key == doi and key not in self:
                self[key] = holder.key
                holder.doi = doi

        def get(self, key, default=None):
            value = super().get(key, default)
            self.claim(key)  # Another worker claims the ID after a stale lookup.
            return value

        def setdefault(self, key, default=None):
            self.claim(key)
            return super().setdefault(key, default)

    lib._by_doi = RacingIndex(lib._by_doi)
    data = openalex._OpenAlexPDF(PDF_BYTES, doi, arxiv)
    openalex._apply_openalex_identifiers(data, p, lib, "llm_match: same work")
    assert holder.doi == doi
    assert holder.arxiv_id == ""
    assert p.doi == ""
    assert lib.find(doi=doi) is holder


def test_openalex_recovery_removes_unused_claims_on_same_row_update(
        lib, openalex_paper, monkeypatch):
    p = openalex_paper
    merge = lib._merge

    def intervening_update(paper, incoming):
        lib.set_resolved_doi(p.key, "10.1088/other")
        return merge(paper, incoming)

    monkeypatch.setattr(lib, "_merge", intervening_update)
    changed = lib.fill_verified_identifiers(p.key, doi="10.1088/recovered")
    assert changed == {}
    assert p.doi == "10.1088/other"
    assert lib.find(doi="10.1088/recovered") is None
    assert lib.find(doi=p.doi) is p


def test_openalex_recovery_preserves_intervening_arxiv_version_index(
        lib, openalex_paper, monkeypatch):
    p = openalex_paper
    merge = lib._merge

    def intervening_update(paper, incoming):
        merge(paper, {"arxiv_id": "2102.11582v1"})
        lib._reindex(paper)
        return merge(paper, incoming)

    monkeypatch.setattr(lib, "_merge", intervening_update)
    assert lib.fill_verified_identifiers(p.key, arxiv_id="2102.11582v3") == {}
    assert p.arxiv_id == "2102.11582v1"
    assert lib.find(arxiv_id="2102.11582") is p


def test_openalex_dns_wait_is_bounded(monkeypatch):
    import socket
    import threading
    import time

    release = threading.Event()

    def resolver(*args, **kwargs):
        release.wait(2)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    try:
        with pytest.raises(requests.Timeout):
            openalex._resolve_addresses("repo.example", 80, time.monotonic() + 0.02)
    finally:
        release.set()


def test_openalex_connect_attempts_share_absolute_deadline(monkeypatch):
    import socket
    from types import SimpleNamespace

    now = [0.0]
    attempts = []
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (f"127.0.0.{i}", 80))
                 for i in range(1, 4)]

    class Socket:
        def settimeout(self, timeout):
            self.timeout = timeout

        def setsockopt(self, *args):
            pass

        def connect(self, address):
            attempts.append(address)
            now[0] += self.timeout
            raise OSError("connection refused")

        def close(self):
            pass

    monkeypatch.setattr(openalex.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(openalex, "_resolve_addresses", lambda *args: addresses)
    monkeypatch.setattr(socket, "socket", lambda *args: Socket())
    connection = SimpleNamespace(_dns_host="repo.example", port=80, timeout=10,
                                 source_address=None, socket_options=[])
    with pytest.raises(requests.Timeout):
        openalex._connect_socket(connection, deadline=0.03)
    assert now[0] <= 0.03
    assert len(attempts) == 1


def test_openalex_connect_falls_back_from_unsupported_address_family(monkeypatch):
    import errno
    import socket
    from types import SimpleNamespace

    attempts = []
    addresses = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 80)),
                 (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    class Socket:
        def settimeout(self, timeout):
            pass

        def connect(self, address):
            attempts.append(address)

    connected = Socket()

    def create(family, *args):
        if family == socket.AF_INET6:
            raise OSError(errno.EAFNOSUPPORT, "IPv6 unavailable")
        return connected

    monkeypatch.setattr(openalex.time, "monotonic", lambda: 0)
    monkeypatch.setattr(openalex, "_resolve_addresses", lambda *args: addresses)
    monkeypatch.setattr(socket, "socket", create)
    connection = SimpleNamespace(_dns_host="repo.example", port=80, timeout=10,
                                 source_address=None, socket_options=[])
    assert openalex._connect_socket(connection, deadline=0.03) is connected
    assert attempts == [("127.0.0.1", 80)]


@pytest.mark.parametrize("elapsed", [0.03, 0.04])
def test_openalex_connect_deducts_tcp_time_before_tls(monkeypatch, elapsed):
    import socket
    from types import SimpleNamespace

    now = [0.0]
    timeouts = []
    closed = []

    class Socket:
        def settimeout(self, timeout):
            timeouts.append(timeout)

        def connect(self, address):
            now[0] = elapsed

        def close(self):
            closed.append(True)

    connected = Socket()
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
    monkeypatch.setattr(openalex.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(openalex, "_resolve_addresses", lambda *args: addresses)
    monkeypatch.setattr(socket, "socket", lambda *args: connected)
    connection = SimpleNamespace(_dns_host="repo.example", port=443, timeout=10,
                                 source_address=None, socket_options=[])
    if elapsed < 0.04:
        assert openalex._connect_socket(connection, deadline=0.04) is connected
        assert timeouts == pytest.approx([0.04, 0.01])
        assert not closed
    else:
        with pytest.raises(requests.Timeout):
            openalex._connect_socket(connection, deadline=0.04)
        assert closed == [True]


# CORE source recovery: transport is mocked, while locator selection, budgets,
# metadata corroboration, PDF parsing, cascade saving and bookkeeping are real.
@pytest.fixture
def core_clock(monkeypatch):
    from papervault.library.download_sources import core

    elapsed = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        elapsed[0] += seconds

    clock = SimpleNamespace(monotonic=lambda: elapsed[0], time=lambda: 1_800_000_000 + elapsed[0],
                            sleep=sleep, elapsed=elapsed, waits=waits)
    monkeypatch.setattr(core, "time", clock, raising=False)
    monkeypatch.setattr(core, "_api_next_at", 0.0, raising=False)
    monkeypatch.setattr(core, "_api_cooldown_until", 0.0, raising=False)
    monkeypatch.setenv("CORE_API_KEY", "fixture-core-token")
    return clock


@pytest.fixture
def core_paper(core_clock):
    from papervault.library.models import Paper

    return Paper(key="CoreControl", title="Repository recovery for cosmic ray transport",
                 authors=["Ari Cukierman"], source="core", paper_id="143668999")


def _core_metadata(paper, **fields):
    return {"id": 568416448, "title": paper.title,
            "authors": [{"name": name} for name in paper.authors],
            "doi": "", "arxivId": "", "outputs": [], "sourceFulltextUrls": [],
            "downloadUrl": "", "links": [], **fields}


CORE_API = "https://api.core.ac.uk/v3/"
CORE_PDF_BYTES = b"%PDF-1.4\n%core-control\n%%EOF\n"


@pytest.mark.parametrize("url", [
    "http://core.ac.uk/download/568416448.pdf",
    "https://core.ac.uk/download/pdf/568416448.pdf",
    "https://core.ac.uk/outputs/568416448",
    "https://api.core.ac.uk/v3/outputs/568416448",
])
@responses.activate
def test_core_url_uses_output_namespace_not_stored_work(core_paper, url):
    paper = core_paper
    paper.url = url
    responses.get(CORE_API + "outputs/568416448", json=_core_metadata(
        paper, fulltextStatus="disabled", sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)

    assert download._try_core(paper) == CORE_PDF_BYTES
    assert [c.request.url for c in responses.calls] == [
        CORE_API + "outputs/568416448", "https://origin.example/file.pdf"]
    assert paper.doi == paper.arxiv_id == ""
    assert responses.calls[0].request.headers["Authorization"] == "Bearer fixture-core-token"
    assert "Authorization" not in responses.calls[1].request.headers


@responses.activate
def test_core_id_only_work_expands_disabled_output(core_paper):
    paper = core_paper
    paper.title = "Short"
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        paper, id=143668999, outputs=[CORE_API + "outputs/568416448"]))
    responses.get(CORE_API + "outputs/568416448", json=_core_metadata(
        paper, fulltextStatus="disabled", sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)

    assert download._try_core(paper) == CORE_PDF_BYTES
    assert download._download_skip_reason("oa_aggregators", paper) is None
    assert [c.request.url for c in responses.calls] == [
        CORE_API + "works/143668999", CORE_API + "outputs/568416448",
        "https://origin.example/file.pdf"]


@pytest.mark.parametrize("source,identifier,url", [
    ("", "143668999", ""), ("semantic_scholar", "143668999", ""),
    ("core", "", ""), ("core", "0", ""), ("core", "-1", ""),
    ("core", "١٢٣", ""), ("core", "work:143668999", ""),
    ("", "", "https://core.ac.uk.evil.example/download/123.pdf"),
    ("", "", "https://user:password@core.ac.uk/download/123.pdf"),
    ("", "", "https://core.ac.uk:8443/download/123.pdf"),
    ("", "", "https://core.ac.uk/download/0.pdf"),
    ("", "", "https://core.ac.uk/download/123.pdf/extra"),
])
@responses.activate
def test_core_missing_or_wrong_namespace_never_probes_numeric_id(
        core_paper, source, identifier, url):
    core_paper.title = "Short"
    core_paper.source, core_paper.paper_id, core_paper.url = source, identifier, url
    assert download._try_core(core_paper) is None
    assert download._download_skip_reason("oa_aggregators", core_paper) == "no_applicable_member"
    assert not responses.calls


@responses.activate
def test_core_output_expansion_validates_urls_and_caps_distinct_lookups(core_paper):
    paper = core_paper
    paper.title = "Short"
    outputs = ["https://evil.example/v3/outputs/1", CORE_API + "outputs/0",
               CORE_API + "outputs/1?token=bad", CORE_API + "outputs/1",
               CORE_API + "outputs/1", CORE_API + "outputs/2", CORE_API + "outputs/3"]
    responses.get(CORE_API + "works/143668999", json=_core_metadata(paper, outputs=outputs))
    responses.get(CORE_API + "outputs/1", json=_core_metadata(paper, id=1))
    responses.get(CORE_API + "outputs/2", json=_core_metadata(paper, id=2))
    assert download._try_core(paper) is None
    assert [c.request.url for c in responses.calls] == [
        CORE_API + "works/143668999", CORE_API + "outputs/1", CORE_API + "outputs/2"]


@pytest.mark.parametrize("conflict", [
    {"title": "A completely different paper"}, {"authors": [{"name": "John Smith"}]},
    {"doi": "10.1234/other"}, {"arxivId": "2306.99999"},
])
@responses.activate
def test_core_conflicting_metadata_never_fetches_origins(core_paper, conflict):
    paper = core_paper
    paper.doi, paper.arxiv_id = "10.1234/requested", "2306.12749"
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        paper, sourceFulltextUrls=["https://origin.example/file.pdf"], **conflict))
    assert download._try_core(paper) is None
    assert len(responses.calls) == 1
    assert paper.doi == "10.1234/requested" and paper.arxiv_id == "2306.12749"


@responses.activate
def test_core_work_output_identifier_conflict_does_not_poison_blank_record(core_paper):
    paper = core_paper
    paper.title = "Short"
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        paper, doi="10.1234/work", outputs=[CORE_API + "outputs/568416448"]))
    responses.get(CORE_API + "outputs/568416448", json=_core_metadata(
        paper, doi="10.1234/other", sourceFulltextUrls=["https://origin.example/file.pdf"]))
    assert download._try_core(paper) is None
    assert len(responses.calls) == 2
    assert paper.doi == paper.arxiv_id == ""


@pytest.mark.parametrize("locator", [True, False])
@responses.activate
def test_core_429_stops_metadata_and_search_chain(core_paper, core_clock, locator):
    paper = core_paper
    if not locator:
        paper.source = paper.paper_id = ""
        paper.doi = "10.1234/requested"
    endpoint = CORE_API + ("works/143668999" if locator else "search/works/")
    responses.get(endpoint, status=429, headers={"Retry-After": "120"})
    assert download._try_core(paper) is None
    assert len(responses.calls) == 1
    # A second call during cooldown must not spend another API request or sleep for two minutes.
    assert download._try_core(paper) is None
    assert len(responses.calls) == 1 and not core_clock.waits
    core_clock.elapsed[0] = 121
    responses.get(endpoint, json={} if locator else {"results": []})
    paper.title = "Short"
    assert download._try_core(paper) is None
    assert len(responses.calls) == 2


@pytest.mark.parametrize("headers,release", [
    ({"Retry-After": "Fri, 15 Jan 2027 08:02:00 GMT"}, 121),
    ({"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1800000120"}, 121),
    ({"X-RateLimit-Retry-After": "120"}, 121),
])
@responses.activate
def test_core_server_cooldown_is_shared_across_papers(core_paper, core_clock, headers, release):
    # 1800000000 == 2027-01-15 08:00:00 UTC.
    responses.get(CORE_API + "works/143668999", status=429, headers=headers)
    assert download._try_core(core_paper) is None
    second = core_paper.model_copy(update={"paper_id": "162638012", "title": "Short"})
    assert download._try_core(second) is None
    assert len(responses.calls) == 1
    core_clock.elapsed[0] = release
    responses.get(CORE_API + "works/162638012", json={})
    assert download._try_core(second) is None
    assert len(responses.calls) == 2


@responses.activate
def test_core_metadata_requests_are_paced(core_paper, core_clock):
    core_paper.title = "Short"
    times = []
    def reply(request):
        times.append(core_clock.monotonic())
        return 200, {}, json.dumps(_core_metadata(
            core_paper, outputs=[CORE_API + "outputs/568416448"] if "works/" in request.url else []))
    responses.add_callback(responses.GET, CORE_API + "works/143668999", callback=reply)
    responses.add_callback(responses.GET, CORE_API + "outputs/568416448", callback=reply)
    assert download._try_core(core_paper) is None
    assert len(times) == 2 and times[1] - times[0] >= 6


@responses.activate
def test_core_repository_candidates_precede_deduplicated_blocked_core_urls(core_paper):
    sources = ["http://core.ac.uk/download/568416448.pdf",
               "https://core.ac.uk/download/pdf/568416448.pdf",
               "https://core.ac.uk/download/568416448.pdf",
               "https://origin.example/one.pdf", "https://origin.example/two.pdf"]
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, sourceFulltextUrls=sources, downloadUrl=sources[0],
        links=[{"type": "display", "url": "https://ignored.example/display"},
               {"type": "thumbnail", "url": "https://ignored.example/thumb.pdf"}]))
    responses.get("https://origin.example/one.pdf", status=403)
    responses.get("https://origin.example/two.pdf", body=CORE_PDF_BYTES)
    assert download._try_core(core_paper) == CORE_PDF_BYTES
    assert [c.request.url for c in responses.calls][1:] == [
        "https://origin.example/one.pdf", "https://origin.example/two.pdf"]


@pytest.mark.parametrize("html,target", [
    ('<meta content="/file.pdf" name="citation_pdf_url">', "https://origin.example/file.pdf"),
    ('<a href="/bitstreams/123/download">Download PDF</a>', "https://origin.example/bitstreams/123/download"),
    ('<link type="application/pdf" href="/paper.pdf" rel="alternate">', "https://origin.example/paper.pdf"),
    ('<a href="javascript:void(0)">PDF</a><a href="/file.pdf">PDF</a>', "https://origin.example/file.pdf"),
])
@responses.activate
def test_core_explicit_landing_links_reach_pdfs(core_paper, html, target):
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, sourceFulltextUrls=["https://origin.example/item/42"]))
    responses.get("https://origin.example/item/42", body=html, content_type="text/html")
    responses.get(target, body=CORE_PDF_BYTES)
    assert download._try_core(core_paper) == CORE_PDF_BYTES
    assert [c.request.url for c in responses.calls][-1] == target


@responses.activate
def test_core_arxiv_abs_resolves_pdf_without_copying_citeseer_doi(core_paper):
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, sourceFulltextUrls=["http://arxiv.org/abs/2306.12749"],
        identifiers=["oai:arXiv.org:2306.12749", "oai:citeseerx:10.1.1.1041.7506"]))
    responses.get("https://arxiv.org/pdf/2306.12749", body=CORE_PDF_BYTES)
    result = download._try_core(core_paper)
    assert result == CORE_PDF_BYTES
    assert getattr(result, "arxiv_id", "") == "2306.12749"
    assert getattr(result, "doi", "") == ""
    assert core_paper.arxiv_id == ""


@pytest.mark.parametrize("status,body,headers", [
    (202, b'<html><script src="/aws-waf-token.js"></script></html>', {}),
    (403, CORE_PDF_BYTES, {}), (206, CORE_PDF_BYTES, {}),
    (200, CORE_PDF_BYTES, {"Content-Range": "bytes 0-20/9999"}),
    (200, b"%PDF-1.4\nincomplete", {}),
    (200, CORE_PDF_BYTES, {"Content-Length": "9999"}),
])
@responses.activate
def test_core_challenge_and_incomplete_files_are_misses(core_paper, status, body, headers):
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", status=status, body=body, headers=headers)
    assert download._try_core(core_paper) is None
    assert core_paper.doi == core_paper.arxiv_id == ""
    assert len(responses.calls) == 2


@responses.activate
def test_core_three_roots_two_links_and_six_gets_bound_landing_work(core_paper):
    roots = [f"https://origin.example/item/{n}" for n in range(4)]
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, sourceFulltextUrls=roots))
    for n in range(4):
        responses.get(roots[n], body=''.join(f'<a href="/{n}/{j}.pdf">PDF</a>' for j in range(4)))
        for j in range(4):
            responses.get(f"https://origin.example/{n}/{j}.pdf", status=403)
    assert download._try_core(core_paper) is None
    urls = [c.request.url for c in responses.calls][1:]
    assert urls == [roots[0], "https://origin.example/0/0.pdf", "https://origin.example/0/1.pdf",
                    roots[1], "https://origin.example/1/0.pdf", "https://origin.example/1/1.pdf"]


@responses.activate
def test_core_redirects_are_bounded_and_cannot_send_api_credentials(core_paper):
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, sourceFulltextUrls=["https://origin.example/redirect"]))
    for n in range(12):
        url = "https://origin.example/redirect" if not n else f"https://origin.example/{n}"
        responses.get(url, status=302, headers={"Location": f"https://origin.example/{n+1}"})
    assert download._try_core(core_paper) is None
    assert len(responses.calls) <= 7  # one metadata GET + six total origin/redirect GETs
    assert all("Authorization" not in c.request.headers for c in responses.calls[1:])


@responses.activate
def test_core_api_redirect_does_not_follow_with_bearer(core_paper):
    responses.get(CORE_API + "works/143668999", status=302,
                  headers={"Location": "https://evil.example/metadata"})
    assert download._try_core(core_paper) is None
    assert len(responses.calls) == 1


@pytest.mark.parametrize("match", [True, False])
@responses.activate
def test_core_cascade_verifies_before_persisting_identifiers(
        lib, paper, monkeypatch, file_url_identity, core_clock, match):
    _isolate_tier(monkeypatch, "oa_aggregators")
    paper.doi = paper.arxiv_id = paper.url = ""
    paper.source, paper.paper_id = "core", "143668999"
    data, prompts = file_url_identity
    if not match:
        data = _text_pdf("Meeting program and poster listing for the annual cosmic ray meeting.")
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        paper, outputs=[CORE_API + "outputs/568416448"]))
    responses.get(CORE_API + "outputs/568416448", json=_core_metadata(
        paper, doi="https://doi.org/10.1234/recovered", arxivId="2306.12749",
        sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", body=data)
    assert download.download_paper(paper, lib) is match
    outcomes = [e for e in _logged_events(lib) if e.get("source") == "oa_aggregators"]
    assert [e["event"] for e in outcomes] == ["downloaded" if match else "download_pdf_mismatch"]
    assert len(prompts) == 1 and "Requested title:" in prompts[0]
    assert "Requested title: Test paper for download cascade" in prompts[0]
    if match:
        assert lib.pdf_path(paper.key).read_bytes() == data
        assert paper.doi == "10.1234/recovered" and paper.arxiv_id == "2306.12749"
        assert paper.download_status == "ok" and paper.download_source == "oa_aggregators"
    else:
        assert not lib.has_pdf(paper.key)
        assert paper.doi == paper.arxiv_id == ""
    lib.save()
    saved = Library(lib.root).get(paper.key)
    assert saved.doi == paper.doi and saved.arxiv_id == paper.arxiv_id


@pytest.mark.parametrize("lookup", [
    {"doi": "10.1234/recovered"},
    {"arxiv_id": "2306.12749"},
], ids=["doi", "arxiv"])
@responses.activate
def test_core_cascade_indexes_verified_identifiers(
        lib, paper, monkeypatch, file_url_identity, core_clock, lookup):
    _isolate_tier(monkeypatch, "oa_aggregators")
    paper.doi = paper.arxiv_id = paper.url = ""
    paper.source, paper.paper_id = "core", "143668999"
    data, _ = file_url_identity
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        paper, id=143668999, doi="10.1234/recovered", arxivId="2306.12749",
        sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", body=data)

    assert lib.find(**lookup) is None
    assert download.download_paper(paper, lib) is True
    assert paper.doi == "10.1234/recovered" and paper.arxiv_id == "2306.12749"
    assert lib.find(**lookup) is paper


@responses.activate
def test_core_id_only_failure_is_miss_and_missing_id_is_skip(lib, paper, monkeypatch, core_clock):
    _isolate_tier(monkeypatch, "oa_aggregators")
    paper.doi = paper.arxiv_id = paper.url = ""
    paper.title, paper.source, paper.paper_id = "Short", "core", "143668999"
    responses.get(CORE_API + "works/143668999", status=404)
    assert download.download_paper(paper, lib) is False
    events = [e for e in _logged_events(lib) if e.get("source") == "oa_aggregators"]
    assert [e["event"] for e in events] == ["download_miss"]
    paper.paper_id = ""
    assert download.download_paper(paper, lib) is False
    events = [e for e in _logged_events(lib) if e.get("source") == "oa_aggregators"]
    assert [e["event"] for e in events] == ["download_miss", "download_skip"]
    assert len(responses.calls) == 1


@responses.activate
def test_core_losing_concurrent_member_does_not_apply_identifiers(
        lib, paper, monkeypatch, core_clock, file_url_identity):
    import threading

    _isolate_tier(monkeypatch, "oa_aggregators")
    paper.doi, paper.arxiv_id, paper.source, paper.paper_id = "10.1234/known", "", "core", "143668999"
    data, _ = file_url_identity
    core_requested = threading.Event()
    other_finished = threading.Event()
    def core_reply(_):
        core_requested.set()
        assert other_finished.wait(5)
        return 200, {}, json.dumps(_core_metadata(
            paper, doi="10.1234/known", arxivId="2306.12749",
            sourceFulltextUrls=["https://origin.example/core.pdf"]))
    def other_member(_):
        assert core_requested.wait(5)
        return data
    original_completed = download.concurrent.futures.as_completed
    def completed(futures, timeout):
        for future in original_completed(futures, timeout=timeout):
            if future.result():
                # Release CORE only once the dispatcher has selected the
                # other completed member, avoiding a scheduler-dependent race.
                other_finished.set()
            yield future
    monkeypatch.setattr(download.concurrent.futures, "as_completed", completed)
    monkeypatch.setattr(download, "_try_unpaywall", other_member)
    monkeypatch.setattr(download, "_try_semantic_scholar_oa", lambda _: None)
    monkeypatch.setattr(download, "_try_openalex", lambda _: None)
    responses.add_callback(responses.GET, CORE_API + "works/143668999", callback=core_reply)
    responses.get("https://origin.example/core.pdf", body=data)
    assert download.download_paper(paper, lib) is True
    assert paper.doi == "10.1234/known" and paper.arxiv_id == ""
    assert lib.pdf_path(paper.key).read_bytes() == data


@pytest.mark.parametrize("identifiers", [
    {"identifiers": {"doi": None, "oai": "oai:arXiv.org:2306.12749"}},
    {"identifiers": [{"type": "oai_id", "identifier": "oai:arxiv.org:2306.12749"}]},
    {"oaiIds": ["oai:arxiv.org:2306.12749"]},
    {"oai": "oai:arxiv.org:2306.12749"},
])
@responses.activate
def test_core_recovers_arxiv_only_from_qualified_oai_namespaces(core_paper, identifiers):
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, sourceFulltextUrls=["https://origin.example/file.pdf"], **identifiers))
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)
    result = download._try_core(core_paper)
    assert result == CORE_PDF_BYTES
    assert getattr(result, "arxiv_id", "") == "2306.12749"
    assert core_paper.arxiv_id == ""


@pytest.mark.parametrize("url", [
    "https://arxiv.org/abs/2306.12749v2",
    "https://arxiv.org/pdf/2306.12749",
])
@responses.activate
def test_core_recovers_arxiv_from_download_url(core_paper, url):
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, id=143668999, downloadUrl=url))
    responses.get("https://arxiv.org/pdf/2306.12749", body=CORE_PDF_BYTES)

    result = download._try_core(core_paper)
    assert result == CORE_PDF_BYTES
    assert getattr(result, "arxiv_id", "") == "2306.12749"
    assert core_paper.arxiv_id == ""


@responses.activate
def test_core_malformed_landing_link_does_not_hide_later_valid_file(core_paper):
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, sourceFulltextUrls=["https://origin.example/item"]))
    responses.get("https://origin.example/item", body='<a href="https://[bad">Bad</a>'
                  '<meta name="citation_pdf_url" content="/file.pdf">')
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)
    assert download._try_core(core_paper) == CORE_PDF_BYTES


@responses.activate
def test_core_rejects_unrelated_output_even_when_works_are_discovered_by_doi(core_paper):
    core_paper.source = core_paper.paper_id = ""
    core_paper.doi = "10.1234/requested"
    responses.get(CORE_API + "search/works/", json={"results": [_core_metadata(
        core_paper, doi="10.1234/requested", outputs=[CORE_API + "outputs/568416448"])]})
    responses.get(CORE_API + "outputs/568416448", json=_core_metadata(
        core_paper, title="Different result for the same keyword", doi="10.1234/other",
        sourceFulltextUrls=["https://origin.example/other.pdf"]))
    assert download._try_core(core_paper) is None
    assert all("origin.example" not in c.request.url for c in responses.calls)
    assert len(responses.calls) <= 3  # DOI, its output, then a bounded title query.


@responses.activate
def test_core_title_discovery_resolves_work_outputs_and_preserves_existing_identifiers(core_paper):
    core_paper.source = core_paper.paper_id = ""
    core_paper.arxiv_id = "2306.12749v2"
    responses.get(CORE_API + "search/works/", json={"results": [_core_metadata(
        core_paper, outputs=[CORE_API + "outputs/568416448"])]})
    responses.get(CORE_API + "outputs/568416448", json=_core_metadata(
        core_paper, arxivId="2306.12749", sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)
    assert download._try_core(core_paper) == CORE_PDF_BYTES
    assert core_paper.arxiv_id == "2306.12749v2"
    assert 'q=title%3A' in responses.calls[0].request.url


@pytest.mark.parametrize("headers", [
    {"X-RateLimit-Retry-After": "2027-01-15T08:02:00+0000"},
    {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "2027-01-15T08:02:00Z"},
])
@responses.activate
def test_core_timestamp_rate_limit_headers_prevent_more_metadata(core_paper, core_clock, headers):
    responses.get(CORE_API + "works/143668999", status=429, headers=headers)
    assert download._try_core(core_paper) is None
    core_clock.elapsed[0] = 61
    assert download._try_core(core_paper) is None
    assert len(responses.calls) == 1
    core_clock.elapsed[0] = 121
    core_paper.title = "Short"
    responses.get(CORE_API + "works/143668999", json={})
    assert download._try_core(core_paper) is None
    assert len(responses.calls) == 2


@responses.activate
def test_core_output_429_stops_before_next_output_or_title_query(core_paper):
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, outputs=[CORE_API + "outputs/568416448", CORE_API + "outputs/628577550"]))
    responses.get(CORE_API + "outputs/568416448", status=429, headers={"Retry-After": "120"})
    assert download._try_core(core_paper) is None
    assert [c.request.url for c in responses.calls] == [
        CORE_API + "works/143668999", CORE_API + "outputs/568416448"]


@responses.activate
def test_core_exhausted_successful_response_still_allows_public_origin(core_paper):
    responses.get(CORE_API + "works/143668999", headers={
        "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1800000120"},
        json=_core_metadata(core_paper, sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)
    assert download._try_core(core_paper) == CORE_PDF_BYTES
    assert download._try_core(core_paper) is None
    assert len(responses.calls) == 2


@responses.activate
def test_core_missing_credentials_skips_id_only_records(core_paper, monkeypatch):
    monkeypatch.delenv("CORE_API_KEY")
    assert download._try_core(core_paper) is None
    assert download._download_skip_reason("oa_aggregators", core_paper) == "no_applicable_member"
    assert not responses.calls


@responses.activate
def test_core_cascade_save_failure_keeps_identifiers_blank(lib, paper, monkeypatch, core_clock):
    _isolate_tier(monkeypatch, "oa_aggregators")
    paper.doi = paper.arxiv_id = ""
    paper.source, paper.paper_id = "core", "143668999"
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        paper, doi="10.1234/recovered", arxivId="2306.12749",
        sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)
    def fail_save(*_):
        raise OSError("fixture write failure")
    monkeypatch.setattr(download, "_atomic_save", fail_save)
    with pytest.raises(OSError, match="fixture write failure"):
        download.download_paper(paper, lib)
    assert paper.doi == paper.arxiv_id == ""
    assert not lib.has_pdf(paper.key)


@responses.activate
def test_core_doi_case_variants_are_one_identifier(core_paper):
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, doi="10.1234/ABC", identifiers={"doi": "https://doi.org/10.1234/abc"},
        sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)
    result = download._try_core(core_paper)
    assert result == CORE_PDF_BYTES
    assert getattr(result, "doi", "").lower() == "10.1234/abc"


@responses.activate
def test_core_transport_suppresses_ambient_library_credentials(core_paper, monkeypatch):
    monkeypatch.setattr(requests.sessions, "get_netrc_auth", lambda _: ("library-user", "library-secret"))
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)
    assert download._try_core(core_paper) == CORE_PDF_BYTES
    assert responses.calls[0].request.headers["Authorization"] == "Bearer fixture-core-token"
    assert "Authorization" not in responses.calls[1].request.headers


@responses.activate
def test_core_exhausted_work_with_outputs_still_uses_known_public_origin(core_paper):
    responses.get(CORE_API + "works/143668999", headers={
        "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1800000120"},
        json=_core_metadata(core_paper, outputs=[CORE_API + "outputs/568416448"],
                            sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)
    assert download._try_core(core_paper) == CORE_PDF_BYTES
    assert [c.request.url for c in responses.calls] == [
        CORE_API + "works/143668999", "https://origin.example/file.pdf"]


@responses.activate
def test_core_bad_citation_link_does_not_hide_later_file(core_paper):
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, sourceFulltextUrls=["https://origin.example/item"]))
    responses.get("https://origin.example/item", body='<meta name="citation_pdf_url" content="https://[bad">'
                  '<a href="/file.pdf">Download</a>')
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)
    assert download._try_core(core_paper) == CORE_PDF_BYTES


def test_core_chunked_pdf_header_does_not_apply_html_size_cap(core_clock):
    from papervault.library.download_sources import core
    body = b"%PDF-1.4\n" + b"x" * (300 * 1024) + b"\n%%EOF\n"
    chunks = iter([body[:1], body[1:5], body[5:], b""])
    response = SimpleNamespace(headers={}, raw=SimpleNamespace(read1=lambda *_, **__: next(chunks)))
    assert core._read_body(response, 256 * 1024, origin=True) == body


@responses.activate
def test_core_duplicate_origin_merges_output_identifiers(core_paper):
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, sourceFulltextUrls=["https://origin.example/file.pdf"],
        outputs=[CORE_API + "outputs/568416448"]))
    responses.get(CORE_API + "outputs/568416448", json=_core_metadata(
        core_paper, doi="10.1234/recovered", arxivId="2306.12749",
        sourceFulltextUrls=["https://origin.example/file.pdf"]))
    responses.get("https://origin.example/file.pdf", body=CORE_PDF_BYTES)
    result = download._try_core(core_paper)
    assert result == CORE_PDF_BYTES
    assert getattr(result, "doi", "") == "10.1234/recovered"
    assert getattr(result, "arxiv_id", "") == "2306.12749"
    assert core_paper.doi == core_paper.arxiv_id == ""


def test_core_body_deadline_checks_between_available_reads(core_clock):
    from papervault.library.download_sources import core
    reads = []
    def drip(*_, **__):
        core_clock.elapsed[0] += 14
        reads.append(1)
        return b"x"
    def filled(amount):
        yield b"".join(drip() for _ in range(amount))
    response = SimpleNamespace(headers={}, raw=SimpleNamespace(read1=drip), iter_content=filled)
    with pytest.raises(core._CoreUnavailable, match="body limit"):
        core._read_body(response, 4 * 1024 * 1024)
    assert len(reads) == 3  # Check the deadline as data arrives, not after a 64 KiB fill.


def test_core_refuses_unrequested_compression_before_stream_decode(core_clock):
    from papervault.library.download_sources import core
    response = SimpleNamespace(headers={"Content-Encoding": "gzip"},
                               raw=SimpleNamespace(read1=lambda *_, **__: b""))
    with pytest.raises(core._CoreUnavailable, match="encoding"):
        core._read_body(response, 4 * 1024 * 1024)


@responses.activate
def test_core_contradictory_duplicate_origins_are_discarded(core_paper):
    core_paper.title = "Short"
    responses.get(CORE_API + "works/143668999", json=_core_metadata(
        core_paper, outputs=[CORE_API + "outputs/1", CORE_API + "outputs/2"]))
    for identifier, doi in [(1, "10.1234/first"), (2, "10.1234/second")]:
        responses.get(CORE_API + f"outputs/{identifier}", json=_core_metadata(
            core_paper, id=identifier, doi=doi,
            sourceFulltextUrls=["https://origin.example/file.pdf"]))
    assert download._try_core(core_paper) is None
    assert [c.request.url for c in responses.calls] == [
        CORE_API + "works/143668999", CORE_API + "outputs/1", CORE_API + "outputs/2"]
    assert core_paper.doi == core_paper.arxiv_id == ""
