"""Tests for the thin source wrappers — sources/arxiv.py,
sources/semantic_scholar.py, sources/inspire.py, sources/ads.py,
sources/core.py."""

from __future__ import annotations

import pytest
import responses

from papervault.library.sources.arxiv import search_arxiv
from papervault.library.sources.semantic_scholar import search_semantic_scholar
from papervault.library.sources.inspire import search_inspire
from papervault.library.sources import ads as ads_mod
from papervault.library.sources.ads import search_ads
from papervault.library.sources.core import search_core
from papervault.library.sources.exceptions import BackendDegraded


# ----------- arxiv ----------------------------------------------------------


_ATOM_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.0001v1</id>
    <title>A Test Paper Title</title>
    <summary>A paper about testing</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Alice Tester</name></author>
    <author><name>Bob Mocker</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.0002v1</id>
    <title>Second Paper</title>
    <summary>About something else</summary>
    <published>2024-02-20T00:00:00Z</published>
    <author><name>Carol Stub</name></author>
  </entry>
</feed>
"""


@responses.activate
def test_search_arxiv_parses_atom_feed():
    responses.add(responses.GET, "http://export.arxiv.org/api/query",
                  body=_ATOM_RESPONSE, status=200)
    out = search_arxiv("test query", max_results=2)
    assert len(out) == 2
    p = out[0]
    assert p["title"] == "A Test Paper Title"
    assert p["authors"] == ["Alice Tester", "Bob Mocker"]
    assert p["year"] == "2024"
    assert p["arxiv_id"] == "2401.0001v1"


def test_search_arxiv_empty_query_returns_empty():
    assert search_arxiv("") == []


@responses.activate
def test_search_arxiv_handles_empty_feed():
    empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"/>'
    responses.add(responses.GET, "http://export.arxiv.org/api/query",
                  body=empty, status=200)
    assert search_arxiv("nothing") == []


@responses.activate
def test_search_arxiv_retry_exhausted_raises_degraded(monkeypatch):
    """V6: a transient empty/429/5xx that exhausts the bounded retry → raise
    BackendDegraded (NOT [] — that would masquerade as an authoritative 0-hit)."""
    monkeypatch.setattr("papervault.library.sources.arxiv.time.sleep", lambda *a, **k: None)
    for _ in range(5):
        responses.add(responses.GET, "http://export.arxiv.org/api/query", status=503)
    with pytest.raises(BackendDegraded):
        search_arxiv("anything")


@responses.activate
def test_search_arxiv_default_sort_is_relevance():
    """V6: default sort is relevance (recency only on by_recency)."""
    from urllib.parse import urlparse, parse_qs
    captured: dict = {}

    def callback(request):
        captured["qs"] = parse_qs(urlparse(request.url).query)
        return (200, {}, '<?xml version="1.0"?>'
                '<feed xmlns="http://www.w3.org/2005/Atom"/>')

    responses.add_callback(responses.GET, "http://export.arxiv.org/api/query",
                           callback=callback)
    search_arxiv("kw")
    assert captured["qs"].get("sortBy") == ["relevance"]


@responses.activate
def test_search_arxiv_recency_sort_threaded():
    """V6: sort_by_recency=True → submittedDate descending."""
    from urllib.parse import urlparse, parse_qs
    captured: dict = {}

    def callback(request):
        captured["qs"] = parse_qs(urlparse(request.url).query)
        return (200, {}, '<?xml version="1.0"?>'
                '<feed xmlns="http://www.w3.org/2005/Atom"/>')

    responses.add_callback(responses.GET, "http://export.arxiv.org/api/query",
                           callback=callback)
    search_arxiv("kw", sort_by_recency=True)
    assert captured["qs"].get("sortBy") == ["submittedDate"]
    assert captured["qs"].get("sortOrder") == ["descending"]


# ----------- semantic_scholar ----------------------------------------------


@responses.activate
def test_search_semantic_scholar_happy_path():
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/search",
        json={"data": [{
            "title": "SS Paper",
            "authors": [{"name": "Eve"}],
            "year": 2023,
            "abstract": "Abstract",
            "citationCount": 12,
            "externalIds": {"DOI": "10.1/x", "ArXiv": "2301.0099"},
            "publicationTypes": ["Review"],
            "venue": "J. Test",
            "publicationVenue": {},
            "url": "https://example.com",
            "paperId": "ssid-zzz",
        }]},
        status=200,
    )
    out = search_semantic_scholar("anything")
    assert len(out) == 1
    p = out[0]
    assert p["title"] == "SS Paper"
    assert p["doi"] == "10.1/x"
    assert p["arxiv_id"] == "2301.0099"
    assert p["citation_count"] == 12
    assert p["publication_types"] == ["Review"]
    assert p["paper_id"] == "ssid-zzz"


def test_search_semantic_scholar_empty_query_returns_empty():
    assert search_semantic_scholar("") == []


@responses.activate
def test_search_semantic_scholar_429_then_success(monkeypatch):
    """429 → retry with backoff (we patch sleep) → success on second try."""
    monkeypatch.setattr("papervault.library.sources.semantic_scholar.time.sleep",
                        lambda *a, **k: None)
    # Two responses queued: first 429, then 200.
    responses.add(responses.GET,
                  "https://api.semanticscholar.org/graph/v1/paper/search",
                  status=429)
    responses.add(responses.GET,
                  "https://api.semanticscholar.org/graph/v1/paper/search",
                  json={"data": [{"title": "After Retry", "authors": [],
                                  "year": 2024, "externalIds": {}, "venue": ""}]},
                  status=200)
    out = search_semantic_scholar("foo")
    assert len(out) == 1
    assert out[0]["title"] == "After Retry"


@responses.activate
def test_search_semantic_scholar_429_exhausted_raises_degraded(monkeypatch):
    """V6: keyless S2 429s are retried with BOUNDED backoff+jitter; on exhaustion
    raise BackendDegraded (so S2 alone never aborts — _fetch_one_backend → [])."""
    monkeypatch.setattr("papervault.library.sources.semantic_scholar.time.sleep",
                        lambda *a, **k: None)
    for _ in range(5):
        responses.add(responses.GET,
                      "https://api.semanticscholar.org/graph/v1/paper/search",
                      status=429)
    with pytest.raises(BackendDegraded):
        search_semantic_scholar("foo")


@responses.activate
def test_search_semantic_scholar_non_429_http_raises_degraded(monkeypatch):
    """V6: a non-429 HTTPError (e.g. 500) converts to BackendDegraded (so the
    typed give-up claim is accurate, not a bare HTTPError out of the loop)."""
    monkeypatch.setattr("papervault.library.sources.semantic_scholar.time.sleep",
                        lambda *a, **k: None)
    responses.add(responses.GET,
                  "https://api.semanticscholar.org/graph/v1/paper/search",
                  status=500)
    with pytest.raises(BackendDegraded):
        search_semantic_scholar("foo")


@responses.activate
def test_search_semantic_scholar_handles_publication_venue_fallback():
    """venue='' but publicationVenue.name is set → fall back to that."""
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/search",
        json={"data": [{"title": "X", "authors": [], "year": 2020,
                        "externalIds": {},
                        "venue": "",
                        "publicationVenue": {"name": "Backup Venue"}}]},
        status=200,
    )
    out = search_semantic_scholar("x")
    assert out[0]["venue"] == "Backup Venue"


# ----------- inspire -------------------------------------------------------


_INSPIRE_RESPONSE = {
    "hits": {
        "hits": [
            {
                "metadata": {
                    "titles": [{"title": "A High-Energy Test Paper"}],
                    "authors": [{"full_name": "Doe, Jane"},
                                {"full_name": "Roe, Richard"}],
                    "publication_info": [{
                        "year": 2022,
                        "journal_title": "Phys. Rev. D",
                    }],
                    "arxiv_eprints": [{"value": "2201.0001"}],
                    "dois": [{"value": "10.1103/PhysRevD.0.000000"}],
                    "abstracts": [{"value": "An abstract about HEP."}],
                    "citation_count": 7,
                },
            },
            {
                "metadata": {
                    "titles": [{"title": "Another Inspire Paper"}],
                    "authors": [{"full_name": "Singleton, Sam"}],
                    "preprint_date": "2019-03-15",
                    "arxiv_eprints": [{"value": "1903.0099"}],
                    "dois": [],
                    "abstracts": [],
                },
            },
        ],
    },
}


@responses.activate
def test_search_inspire_happy_path():
    responses.add(responses.GET, "https://inspirehep.net/api/literature",
                  json=_INSPIRE_RESPONSE, status=200)
    out = search_inspire("hep test", max_results=2)
    assert len(out) == 2
    p = out[0]
    assert p["title"] == "A High-Energy Test Paper"
    assert p["authors"] == ["Doe, Jane", "Roe, Richard"]
    assert p["year"] == 2022
    assert p["doi"] == "10.1103/PhysRevD.0.000000"
    assert p["arxiv_id"] == "2201.0001"
    assert p["abstract"] == "An abstract about HEP."
    assert p["venue"] == "Phys. Rev. D"
    assert p["citation_count"] == 7
    assert p["url"] == "https://arxiv.org/abs/2201.0001"

    # Second hit: preprint_date fallback for year, no DOI / abstract.
    p2 = out[1]
    assert p2["title"] == "Another Inspire Paper"
    assert p2["year"] == 2019
    assert p2["doi"] == ""
    assert p2["abstract"] == ""
    assert p2["arxiv_id"] == "1903.0099"


def test_search_inspire_empty_query_returns_empty():
    assert search_inspire("") == []


@responses.activate
def test_search_inspire_http_500_returns_empty():
    responses.add(responses.GET, "https://inspirehep.net/api/literature", status=500)
    assert search_inspire("anything") == []


@responses.activate
def test_search_inspire_429_then_success(monkeypatch):
    """429 → retry with backoff (sleep patched out) → success on next try."""
    monkeypatch.setattr("papervault.library.sources.inspire.time.sleep",
                        lambda *a, **k: None)
    responses.add(responses.GET, "https://inspirehep.net/api/literature",
                  status=429)
    responses.add(responses.GET, "https://inspirehep.net/api/literature",
                  json={"hits": {"hits": [{
                      "metadata": {
                          "titles": [{"title": "Retried Hit"}],
                          "authors": [],
                          "publication_info": [{"year": 2023}],
                          "arxiv_eprints": [], "dois": [],
                          "abstracts": [],
                      }
                  }]}},
                  status=200)
    out = search_inspire("foo")
    assert len(out) == 1
    assert out[0]["title"] == "Retried Hit"
    assert out[0]["year"] == 2023


# ----------- ads -----------------------------------------------------------


@responses.activate
def test_search_ads_happy_path_with_token(monkeypatch):
    monkeypatch.setenv("ADS_API_TOKEN", "fake-token")
    # Reset any prior auth-warning state so other tests don't interfere.
    monkeypatch.setattr(ads_mod, "_AUTH_WARNED", False)

    captured: dict = {}

    def callback(request):
        captured["headers"] = dict(request.headers)
        body = {
            "response": {
                "docs": [{
                    "title": ["An ADS Paper"],
                    "author": ["Adams, A.", "Brown, B.", "Carter, C."],
                    "year": "2020",
                    "abstract": "An astrophysics abstract.",
                    "doi": ["10.1000/ads.1"],
                    "identifier": ["arXiv:2401.0001", "2024arXiv240100001K"],
                    "bibcode": "2020ApJ...900....1A",
                    "pub": "Astrophysical Journal",
                    "citation_count": 42,
                }]
            }
        }
        import json as _json
        return (200, {}, _json.dumps(body))

    responses.add_callback(
        responses.GET,
        "https://api.adsabs.harvard.edu/v1/search/query",
        callback=callback,
        content_type="application/json",
    )

    out = search_ads("astro test")
    assert len(out) == 1
    p = out[0]
    assert p["title"] == "An ADS Paper"
    assert p["authors"] == ["Adams, A.", "Brown, B.", "Carter, C."]
    assert p["year"] == 2020
    assert p["doi"] == "10.1000/ads.1"
    assert p["arxiv_id"] == "2401.0001"
    assert p["paper_id"] == "2020ApJ...900....1A"
    assert p["venue"] == "Astrophysical Journal"
    assert p["citation_count"] == 42
    assert "abs/2020ApJ...900....1A" in p["url"]
    # Bearer auth header was sent.
    assert captured["headers"].get("Authorization") == "Bearer fake-token"


def test_search_ads_no_token_returns_empty(monkeypatch):
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    # Clear stale responses.calls accounting between tests is not necessary
    # since responses isn't activated here — confirm no HTTP attempted by
    # virtue of returning [] before requests.get runs.
    assert search_ads("anything") == []


@responses.activate
def test_search_ads_401_raises_degraded(monkeypatch):
    """V6: a PRESENT-but-rejected token (401/403) is a DEGRADE, not a 0-hit —
    raise BackendDegraded so _fetch_one_backend records it (the §8 unconfigured
    signal is the separate no-token-at-all case)."""
    monkeypatch.setenv("ADS_API_TOKEN", "bad-token")
    monkeypatch.setattr(ads_mod, "_AUTH_WARNED", False)
    responses.add(responses.GET,
                  "https://api.adsabs.harvard.edu/v1/search/query",
                  status=401)
    with pytest.raises(BackendDegraded):
        search_ads("x")


@responses.activate
def test_search_ads_waf_405_raises_degraded(monkeypatch):
    """V6: a WAF 405 (or x-amzn-waf-action:captcha) → BackendDegraded. 405 is
    NOT 5xx so it would otherwise fall through to raise_for_status → HTTPError →
    swallowed-to-[]; the standalone WAF guard catches it as a degrade."""
    monkeypatch.setenv("ADS_API_TOKEN", "good-token")
    monkeypatch.setattr(ads_mod, "_AUTH_WARNED", False)
    responses.add(responses.GET,
                  "https://api.adsabs.harvard.edu/v1/search/query",
                  status=405)
    with pytest.raises(BackendDegraded):
        search_ads("x")


@responses.activate
def test_search_ads_waf_captcha_header_raises_degraded(monkeypatch):
    """V6: a 200 carrying x-amzn-waf-action:captcha (the WAF challenge form) is
    also a degrade, not a 0-hit."""
    monkeypatch.setenv("ADS_API_TOKEN", "good-token")
    monkeypatch.setattr(ads_mod, "_AUTH_WARNED", False)
    responses.add(responses.GET,
                  "https://api.adsabs.harvard.edu/v1/search/query",
                  status=200, body="{}",
                  headers={"x-amzn-waf-action": "captcha"})
    with pytest.raises(BackendDegraded):
        search_ads("x")


@responses.activate
def test_search_ads_captcha_403_labeled_waf_not_auth(monkeypatch):
    """V6: AWS WAF serves its captcha challenge WITH a 403 status. The captcha
    header check runs BEFORE the 401/403 auth block, so a captcha-403 is labeled a
    WAF degrade ('ADS WAF'), not mis-attributed as an auth error ('ADS auth 403')."""
    monkeypatch.setenv("ADS_API_TOKEN", "good-token")
    monkeypatch.setattr(ads_mod, "_AUTH_WARNED", False)
    responses.add(responses.GET,
                  "https://api.adsabs.harvard.edu/v1/search/query",
                  status=403, body="{}",
                  headers={"x-amzn-waf-action": "captcha"})
    with pytest.raises(BackendDegraded) as exc:
        search_ads("x")
    assert "WAF" in str(exc.value)        # WAF-labeled, not "auth"
    assert "auth" not in str(exc.value).lower()


@responses.activate
def test_search_ads_plain_403_still_labeled_auth(monkeypatch):
    """A 403 WITHOUT the WAF captcha header still falls through to the auth block
    (a genuinely-rejected token), labeled 'ADS auth 403'."""
    monkeypatch.setenv("ADS_API_TOKEN", "bad-token")
    monkeypatch.setattr(ads_mod, "_AUTH_WARNED", False)
    responses.add(responses.GET,
                  "https://api.adsabs.harvard.edu/v1/search/query", status=403)
    with pytest.raises(BackendDegraded) as exc:
        search_ads("x")
    assert "auth" in str(exc.value).lower()


@responses.activate
def test_search_ads_sort_is_relevance(monkeypatch):
    """V6: ADS sorts by relevance (fix the prior hardcoded ``date desc``)."""
    monkeypatch.setenv("ADS_API_TOKEN", "good-token")
    monkeypatch.setattr(ads_mod, "_AUTH_WARNED", False)
    captured: dict = {}

    def callback(request):
        from urllib.parse import urlparse, parse_qs
        captured["qs"] = parse_qs(urlparse(request.url).query)
        return (200, {}, '{"response": {"docs": []}}')

    responses.add_callback(
        responses.GET,
        "https://api.adsabs.harvard.edu/v1/search/query",
        callback=callback,
        content_type="application/json",
    )
    search_ads("astro")
    assert captured["qs"].get("sort") == ["relevance"]


@responses.activate
def test_search_ads_429_exhausted_raises_degraded(monkeypatch):
    """V6: retry-exhausted (repeated 429) → BackendDegraded, NOT [] — consistent
    with arxiv / semantic_scholar / core (a transient failure must not masquerade
    as an authoritative 0-hit)."""
    monkeypatch.setenv("ADS_API_TOKEN", "good-token")
    monkeypatch.setattr("papervault.library.sources.ads.time.sleep", lambda *a, **k: None)
    for _ in range(ads_mod.MAX_RETRIES + 2):
        responses.add(responses.GET,
                      "https://api.adsabs.harvard.edu/v1/search/query", status=429)
    with pytest.raises(BackendDegraded):
        search_ads("query")


@responses.activate
def test_search_ads_empty_results_returns_empty(monkeypatch):
    """Canary: a genuine 0-hit (200 + zero docs) returns [], NOT BackendDegraded —
    only transient failures degrade."""
    monkeypatch.setenv("ADS_API_TOKEN", "good-token")
    responses.add(responses.GET,
                  "https://api.adsabs.harvard.edu/v1/search/query",
                  json={"response": {"docs": []}}, status=200)
    assert search_ads("query") == []


@responses.activate
def test_search_ads_extracts_arxiv_id_from_identifier(monkeypatch):
    monkeypatch.setenv("ADS_API_TOKEN", "fake")
    monkeypatch.setattr(ads_mod, "_AUTH_WARNED", False)
    responses.add(
        responses.GET,
        "https://api.adsabs.harvard.edu/v1/search/query",
        json={"response": {"docs": [{
            "title": ["Identifier Test"],
            "author": [],
            "year": "2024",
            "identifier": ["arXiv:2401.0001", "2024arXiv240100001K"],
            "bibcode": "2024arXiv240100001K",
        }]}},
        status=200,
    )
    out = search_ads("anything")
    assert len(out) == 1
    assert out[0]["arxiv_id"] == "2401.0001"


# ----------- core (D4) ------------------------------------------------------


@responses.activate
def test_search_core_happy_path(monkeypatch):
    monkeypatch.setenv("CORE_API_KEY", "fake-core-key")
    captured: dict = {}

    def callback(request):
        captured["headers"] = dict(request.headers)
        body = {
            "results": [{
                "id": 123456,
                "title": "A CORE paper on something",
                "authors": [{"name": "Lastname, F."},
                             {"name": "Other, S."}],
                "abstract": "A CORE abstract about open-access papers.",
                "yearPublished": 2024,
                "doi": "10.5000/core.1",
                "arxivId": "",
                "publisher": "Some Publisher",
                "journals": [{"title": "Journal of CORE Stuff"}],
                "downloadUrl": "https://core.ac.uk/download/123456.pdf",
                "citationCount": 7,
            }]
        }
        import json as _json
        return (200, {}, _json.dumps(body))

    responses.add_callback(
        responses.GET,
        "https://api.core.ac.uk/v3/search/works/",
        callback=callback,
        content_type="application/json",
    )

    out = search_core("open access metadata")
    assert len(out) == 1
    p = out[0]
    assert p["title"] == "A CORE paper on something"
    assert p["authors"] == ["Lastname, F.", "Other, S."]
    assert p["year"] == 2024
    assert p["doi"] == "10.5000/core.1"
    assert p["venue"] == "Journal of CORE Stuff"
    assert p["url"] == "https://core.ac.uk/download/123456.pdf"
    assert p["citation_count"] == 7
    assert p["paper_id"] == "123456"
    # Bearer auth header was sent.
    assert captured["headers"].get("Authorization") == "Bearer fake-core-key"


def test_search_core_no_api_key_returns_empty(monkeypatch):
    """Without CORE_API_KEY env var, the search is a silent no-op."""
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    assert search_core("anything") == []


def test_search_core_empty_query_returns_empty(monkeypatch):
    monkeypatch.setenv("CORE_API_KEY", "x")
    assert search_core("") == []


@responses.activate
def test_search_core_http_500_raises_degraded(monkeypatch):
    """V6: a non-ok CORE response is a DEGRADE, not an authoritative 0-hit —
    raise BackendDegraded (a genuine empty result is an empty data['results'])."""
    monkeypatch.setenv("CORE_API_KEY", "fake")
    responses.add(responses.GET,
                  "https://api.core.ac.uk/v3/search/works/",
                  status=500)
    with pytest.raises(BackendDegraded):
        search_core("query")


@responses.activate
def test_search_core_429_exhausted_raises_degraded(monkeypatch):
    """V6: retry-exhausted (repeated 429) → BackendDegraded, not []."""
    monkeypatch.setenv("CORE_API_KEY", "fake")
    monkeypatch.setattr("papervault.library.sources.core.time.sleep", lambda *a, **k: None)
    # Every attempt 429s → loop exhausts → degraded.
    for _ in range(5):
        responses.add(responses.GET,
                      "https://api.core.ac.uk/v3/search/works/", status=429)
    with pytest.raises(BackendDegraded):
        search_core("query")


@responses.activate
def test_search_core_empty_results_returns_empty(monkeypatch):
    """V6 canary: a genuine 0-hit (empty results array, ok status) returns [],
    NOT BackendDegraded — only transient failures degrade."""
    monkeypatch.setenv("CORE_API_KEY", "fake")
    responses.add(responses.GET,
                  "https://api.core.ac.uk/v3/search/works/",
                  json={"results": []}, status=200)
    assert search_core("query") == []


@responses.activate
def test_search_core_falls_back_publisher_when_no_journal(monkeypatch):
    """When journals[] is empty, venue falls back to the publisher field."""
    monkeypatch.setenv("CORE_API_KEY", "fake")
    responses.add(
        responses.GET,
        "https://api.core.ac.uk/v3/search/works/",
        json={"results": [{
            "id": 1,
            "title": "Paper without journal listed",
            "authors": [{"name": "A."}],
            "yearPublished": 2023,
            "publisher": "Springer",
            "journals": [],
        }]},
        status=200,
    )
    out = search_core("x")
    assert len(out) == 1
    assert out[0]["venue"] == "Springer"

