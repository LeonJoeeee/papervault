"""Tests for download.py PDF cascade. No real HTTP."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import responses

from papervault.library import Library, Paper, download


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
def test_openalex_skipped_without_doi(lib):
    """No DOI → openalex strategy is a no-op (no API call)."""
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
            {"bibcode": "2020ApJ...1B", "esources": ["EPRINT_HTML", "PUB_PDF"]},
        ]}},
        status=200,
    )
    # First esource isn't a PDF type, second is — link_gateway returns the body.
    responses.add(
        responses.GET,
        "https://ui.adsabs.harvard.edu/link_gateway/2020ApJ...1B/PUB_PDF",
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
            {"bibcode": "2020ApJ...1B", "esources": ["EPRINT_HTML", "AUTHOR_HTML"]},
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
            {"bibcode": "2020Bib", "esources": ["ADS_PDF"]},
        ]}},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://ui.adsabs.harvard.edu/link_gateway/2020Bib/ADS_PDF",
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
# SSRN strategy
# ---------------------------------------------------------------------------

@responses.activate
def test_ssrn_happy_path(lib):
    """Paper with SSRN DOI → fetch landing page, follow Delivery.cfm link."""
    p, _ = lib.upsert({"title": "An SSRN paper title that is long enough",
                       "authors": ["A"], "year": 2022,
                       "doi": "10.2139/ssrn.4000235"})
    landing_html = (
        b'<html><body>'
        b'<a class="download" href="/sol3/Delivery.cfm/'
        b'SSRN_ID4000235_code1234.pdf?abstractid=4000235">Download</a>'
        b'</body></html>'
    )
    responses.add(
        responses.GET,
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4000235",
        body=landing_html, status=200,
    )
    responses.add(
        responses.GET,
        "https://papers.ssrn.com/sol3/Delivery.cfm/"
        "SSRN_ID4000235_code1234.pdf",
        body=PDF_BYTES, status=200,
    )
    assert download._try_ssrn(p) == PDF_BYTES


@responses.activate
def test_ssrn_skipped_for_non_ssrn_doi(lib, paper):
    """Paper with non-SSRN DOI → no HTTP, return None."""
    # The fixture `paper` has doi 10.1/x — not SSRN.
    with responses.RequestsMock() as rmocks:
        assert download._try_ssrn(paper) is None
        assert len(rmocks.calls) == 0


@responses.activate
def test_ssrn_landing_without_delivery_link_returns_none(lib):
    """SSRN landing page renders but has no Delivery.cfm link (paywalled or
    the layout has changed) → strategy returns None gracefully."""
    p, _ = lib.upsert({"title": "Another SSRN paper without a delivery link",
                       "authors": ["A"], "year": 2022,
                       "doi": "10.2139/ssrn.5000999"})
    responses.add(
        responses.GET,
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5000999",
        body=b"<html><body>Subscription required</body></html>",
        status=200,
    )
    assert download._try_ssrn(p) is None


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
    ImportError; the tier should swallow it and return None rather than
    propagate."""
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
