"""Anna's md5 detail-page downloads, using fixture HTTP/browser boundaries only."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import responses

from papervault.library import download
from papervault.library.models import Paper


BASE = "https://annas-archive.gl"
MD5 = "0123456789abcdef0123456789abcdef"
DETAIL = f"{BASE}/md5/{MD5}"
EXPIRED = "https://partner.example/expired.pdf?token=expired&expires=1"
LIVE = "https://partner.example/live.pdf?token=live&expires=2"
PDF = b"%PDF-1.4\nfixture paper"
FIXTURES = Path(__file__).parent / "fixtures"
RECORD = f'<html><a href="/md5/{MD5}">Archive record</a></html>'


def detail_option(href, label="Other mirror"):
    return f'<h3>External downloads</h3><ul><li><a href="{href}">{label}</a></li></ul>'


@pytest.fixture
def paper():
    return Paper(key="Fixture2026", title="Fixture paper", doi="10.1/x")


@pytest.fixture
def browser(monkeypatch):
    """Run the real page_action and transport headers against fixture responses."""
    def install(detail, record=RECORD, *, detail_timeout=False):
        pages = list(detail) if isinstance(detail, list) else [detail]
        visited = []

        class Page:
            def __init__(self, html, url):
                self.html, self.url = html, url

            def wait_for_selector(self, selector, *, state, timeout):
                assert state == "attached" and timeout <= 60000
                if self.url == DETAIL and detail_timeout:
                    raise TimeoutError("fixture detail challenge")

            def wait_for_load_state(self, state, *, timeout):
                assert state == "domcontentloaded"

            def content(self):
                return self.html

        class Fetcher:
            @staticmethod
            def fetch(url, **kwargs):
                assert kwargs["cookies"] == [{
                    "name": "aa_account_id2", "value": "fixture-token", "url": BASE,
                }]
                visited.append(url)
                if url == f"{BASE}/scidb/10.1/x":
                    html = record
                elif url == DETAIL:
                    html = pages.pop(0) if len(pages) > 1 else pages[0]
                else:
                    raise AssertionError(f"Unexpected browser URL: {url}")
                try:
                    kwargs["page_action"](Page(html, url))
                except TimeoutError:
                    pass  # Scrapling swallows callback errors.
                return SimpleNamespace(status=200, html_content=html)

        def impersonated_get(url, **kwargs):
            assert kwargs.pop("impersonate") == "chrome"
            return requests.get(url, **kwargs)

        monkeypatch.setenv("ANNAS_ARCHIVE_API_KEY", "fixture-token")
        monkeypatch.setitem(sys.modules, "scrapling.fetchers", SimpleNamespace(StealthyFetcher=Fetcher))
        monkeypatch.setitem(sys.modules, "curl_cffi", SimpleNamespace(
            requests=SimpleNamespace(get=impersonated_get)))
        return visited

    return install


@responses.activate
def test_expired_detail_option_refreshes_and_yields_live_pdf(paper, browser, caplog):
    """An expired signed link must refresh the detail page and try remaining mirrors."""
    visited = browser((FIXTURES / "annas_detail_options.html").read_text())
    responses.get(EXPIRED, status=403, body="Link expired or invalid. Get a new link.")
    responses.get(LIVE, body=PDF, content_type="application/octet-stream")

    assert download._try_annas_archive_api(paper) == PDF
    assert visited == [f"{BASE}/scidb/10.1/x", DETAIL, DETAIL]
    assert "expired" in caplog.text
    assert all("Cookie" not in c.request.headers for c in responses.calls)
    assert all(c.request.headers["Referer"] == c.request.url for c in responses.calls)
    assert not any("fast_download" in c.request.url for c in responses.calls)


@responses.activate
def test_expiry_re_resolves_from_fresh_detail_page(paper, browser):
    browser([detail_option(EXPIRED), detail_option(LIVE)])
    responses.get(EXPIRED, status=403, body="Link expired or invalid.")
    responses.get(LIVE, body=PDF, content_type="application/pdf")
    assert download._try_annas_archive_api(paper) == PDF


@responses.activate
def test_refresh_keeps_refused_options_from_exhausting_budget(paper, browser):
    refused = [f"https://partner.example/refused{i}.pdf" for i in range(3)]
    browser("".join(detail_option(url) for url in [*refused, EXPIRED, LIVE]))
    for url in refused:
        responses.get(url, status=403, body="Download refused")
    responses.get(EXPIRED, status=403, body="Link expired or invalid.")
    responses.get(LIVE, body=PDF, content_type="application/pdf")
    responses.get(f"{BASE}/dyn/api/fast_download.json", json={"error": "daily_limit"})

    assert download._try_annas_archive_api(paper) == PDF
    assert all(sum(c.request.url == url for c in responses.calls) == 1 for url in refused)
    assert not any("fast_download" in c.request.url for c in responses.calls)


@responses.activate
def test_detail_document_relative_option_uses_document_url(paper, browser):
    browser(detail_option("files/paper.pdf"))
    responses.get(f"{BASE}/md5/files/paper.pdf", body=PDF, content_type="application/pdf")
    assert download._try_annas_archive_api(paper) == PDF


@responses.activate
def test_last_primary_expiry_leaves_unused_api_refresh_available(paper, browser):
    refused = [f"https://partner.example/refused{i}.pdf" for i in range(7)]
    visited = browser("".join(detail_option(url) for url in [*refused, EXPIRED]))
    for url in refused:
        responses.get(url, status=403, body="Download refused")
    responses.get(EXPIRED, status=403, body="Link expired or invalid.")
    responses.get(f"{BASE}/dyn/api/fast_download.json", json={"download_url": EXPIRED})
    responses.get(f"{BASE}/dyn/api/fast_download.json", json={"download_url": LIVE})
    responses.get(LIVE, body=PDF)

    assert download._try_annas_archive_api(paper) == PDF
    assert visited.count(DETAIL) == 1
    assert sum("fast_download.json" in c.request.url for c in responses.calls) == 2


@responses.activate
def test_record_signed_link_is_not_a_download_option(paper, browser):
    browser(detail_option(LIVE), record=RECORD + '<a href="https://old.example/stale.pdf">Download</a>')
    responses.get("https://old.example/stale.pdf", body=b"%PDF-old wrong paper")
    responses.get(LIVE, body=PDF, content_type="application/pdf")
    assert download._try_annas_archive_api(paper) == PDF
    assert [c.request.url for c in responses.calls] == [LIVE]


@responses.activate
def test_no_viable_option_records_reason_before_api_fallback(paper, browser, caplog):
    browser((FIXTURES / "annas_detail_no_viable.html").read_text())
    responses.get(f"{BASE}/dyn/api/fast_download.json", json={"error": "daily_limit"})
    assert download._try_annas_archive_api(paper) is None
    assert "no viable option" in caplog.text
    assert "quota" in caplog.text
    assert len(responses.calls) == 1


@responses.activate
def test_expired_and_no_retry_left_is_distinct_from_refusal(paper, browser, caplog):
    visited = browser(detail_option(EXPIRED))
    responses.get(EXPIRED, status=403, body="Link expired or invalid.")
    responses.get(f"{BASE}/dyn/api/fast_download.json", json={"error": "daily_limit"})
    assert download._try_annas_archive_api(paper) is None
    assert "expired and no retry left" in caplog.text
    assert visited.count(DETAIL) == 2
    assert sum(c.request.url == EXPIRED for c in responses.calls) == 2


@responses.activate
def test_api_fallback_expiry_resolves_fresh_option(paper, browser, caplog):
    browser((FIXTURES / "annas_detail_no_viable.html").read_text())
    responses.get(f"{BASE}/dyn/api/fast_download.json", json={"download_url": EXPIRED})
    responses.get(f"{BASE}/dyn/api/fast_download.json", json={"download_url": LIVE})
    responses.get(EXPIRED, status=403, body="Link expired or invalid.")
    responses.get(LIVE, body=PDF)
    assert download._try_annas_archive_api(paper) == PDF
    assert "expired" in caplog.text
    assert sum("fast_download.json" in c.request.url for c in responses.calls) == 2


@responses.activate
def test_api_fallback_expiry_stops_with_specific_reason(paper, browser, caplog):
    browser((FIXTURES / "annas_detail_no_viable.html").read_text())
    responses.get(f"{BASE}/dyn/api/fast_download.json", json={"download_url": EXPIRED})
    responses.get(EXPIRED, status=403, body="Link expired or invalid.")
    assert download._try_annas_archive_api(paper) is None
    assert "expired and no retry left" in caplog.text
    assert sum("fast_download.json" in c.request.url for c in responses.calls) == 2


@responses.activate
def test_lookup_miss_is_explained_without_spending_quota(paper, browser, caplog):
    browser("", record="<html><title>Search - Anna's Archive</title></html>")
    assert download._try_annas_archive_api(paper) is None
    assert "not in Anna's index" in caplog.text
    assert not responses.calls


@responses.activate
def test_search_page_with_unrelated_md5_is_an_index_miss(paper, browser, caplog):
    """A search result's available PDF must never stand in for the requested DOI."""
    record = (FIXTURES / "annas_search_unrelated_md5.html").read_text()
    visited = browser(detail_option(LIVE), record=record)
    responses.get(LIVE, body=b"%PDF-1.4\nunrelated search result", content_type="application/pdf")
    caplog.set_level("INFO", logger=download.log.name)

    assert download._try_annas_archive_api(paper) is None
    assert "not in Anna's index" in caplog.text
    assert "download refused" not in caplog.text
    assert "retrieved" not in caplog.text
    assert visited == [f"{BASE}/scidb/10.1/x"]
    assert not responses.calls


@responses.activate
def test_detail_challenge_is_not_reported_as_index_miss(paper, browser, caplog):
    browser(detail_option(LIVE), detail_timeout=True)
    responses.get(f"{BASE}/dyn/api/fast_download.json", json={"error": "unavailable"})
    assert download._try_annas_archive_api(paper) is None
    assert "download refused" in caplog.text
    assert "not in Anna's index" not in caplog.text


@responses.activate
@pytest.mark.parametrize("source, landing, resolved", [
    ("Sci-Hub", '<object type="application/pdf" data="/paper.pdf#view"></object>',
     "https://sci-hub.example/paper.pdf"),
    ("Libgen", '<a href="get.php?md5=fixture">GET</a>',
     "https://libgen.example/get.php?md5=fixture"),
])
def test_detail_source_resolvers_return_pdf(paper, browser, source, landing, resolved):
    url = f"https://{'sci-hub' if source == 'Sci-Hub' else 'libgen'}.example/ads.php"
    browser(detail_option(url, source))
    responses.get(url, body=landing, content_type="text/html")
    responses.get(resolved, body=PDF, content_type="application/pdf")
    assert download._try_annas_archive_api(paper) == PDF


@responses.activate
def test_generic_file_response_without_pdf_suffix(paper, browser):
    browser(detail_option("https://partner.example/download?id=1"))
    responses.get("https://partner.example/download?id=1", body=PDF,
                  content_type="application/octet-stream")
    assert download._try_annas_archive_api(paper) == PDF


@responses.activate
def test_malformed_option_does_not_hide_live_mirror(paper, browser):
    browser(detail_option("https://[") + detail_option(LIVE))
    responses.get(LIVE, body=PDF, content_type="application/pdf")
    assert download._try_annas_archive_api(paper) == PDF


@responses.activate
def test_scihub_metadata_entities_are_decoded(paper, browser):
    url = "https://sci-hub.example/doi"
    browser(detail_option(url, "Sci-Hub"))
    responses.get(url, body='<meta name="citation_pdf_url" content="/paper.pdf?a=1&amp;b=2">',
                  content_type="text/html")
    responses.get("https://sci-hub.example/paper.pdf?a=1&b=2", body=PDF,
                  content_type="application/pdf")
    assert download._try_annas_archive_api(paper) == PDF


@responses.activate
def test_offered_scihub_robot_page_uses_existing_sibling_mirror(paper, browser):
    """An offered mirror's robot wall must not discard a live Sci-Hub option."""
    browser(detail_option("https://sci-hub.ru/10.1/x", "Sci-Hub"))
    responses.get("https://sci-hub.ru/10.1/x", body="<title>Sci-Hub: are you a robot?</title>",
                  content_type="text/html")
    responses.get("https://sci-hub.ee/10.1/x", body='<object type="application/pdf" data="/paper.pdf"></object>',
                  content_type="text/html")
    responses.get("https://sci-hub.ee/paper.pdf", body=PDF, content_type="application/pdf")
    assert download._try_annas_archive_api(paper) == PDF


@responses.activate
def test_html_response_is_not_accepted_as_pdf(paper, browser, caplog):
    browser(detail_option(LIVE))
    responses.get(LIVE, body=PDF, content_type="text/html")
    responses.get(f"{BASE}/dyn/api/fast_download.json", json={"error": "unavailable"})
    assert download._try_annas_archive_api(paper) is None
    assert "download refused" in caplog.text


def test_browser_and_waitlist_requirements_outrank_source_priority():
    html = '''
      <h3>Slow downloads</h3><ul>
        <li><a href="/slow_download/hash/0/0">Server with waitlist</a> (browser verification, waitlist)</li>
        <li><a href="/slow_download/hash/0/1">Server without waitlist</a> (browser verification, no waitlist)</li>
        <li><a href="/slow_download/hash/0/2">Direct server</a> (no browser verification, no waitlist)</li>
      </ul><h3>External downloads</h3><ul>
        <li><a href="https://unknown.example/file">Unknown</a></li>
        <li><a href="https://sci-hub.example/doi">Sci-Hub</a></li>
        <li><a href="https://libgen.example/ads.php">Libgen</a><a href="?viewer=1">Viewer</a></li>
      </ul>
    '''
    options = download._annas_download_options(html, BASE)
    assert [o.url for o in options] == [
        "https://libgen.example/ads.php", "https://sci-hub.example/doi",
        f"{BASE}/slow_download/hash/0/2", "https://unknown.example/file",
        f"{BASE}/slow_download/hash/0/1", f"{BASE}/slow_download/hash/0/0",
    ]
