"""Search-response identifier capture is lossless through admission and JSON storage."""

from __future__ import annotations

import pytest
import responses

from papervault.library import Library
from papervault.library import search
from papervault.library.sources.ads import search_ads
from papervault.library.sources.arxiv import search_arxiv
from papervault.library.sources.core import search_core
from papervault.library.sources.inspire import search_inspire
from papervault.library.sources.semantic_scholar import search_semantic_scholar


TITLE = "Identifier capture for cosmic ray transport"
DOI = "10.1234/capture"
ARXIV = "2401.00001v2"
PDF = "https://repository.example/paper.pdf"
LANDING = "https://repository.example/paper"

# Literal expectations are independent of the adapters and capture helpers.
CASES = [
    ("arxiv", {"arxiv:paper": [ARXIV]}, [DOI], [ARXIV],
     [f"https://arxiv.org/abs/{ARXIV}", f"https://arxiv.org/pdf/{ARXIV}"]),
    ("ads", {"ads:bibcode": ["2024ApJ...123..456A"],
             "ads:identifier": [f"arXiv:{ARXIV}", "oai:ads:record"]},
     [DOI, "10.1234/second"], [ARXIV],
     ["https://ui.adsabs.harvard.edu/abs/2024ApJ...123..456A/abstract"]),
    ("semantic_scholar", {"semantic_scholar:paper": ["ss-paper"],
                          "semantic_scholar:external:CorpusId": ["123"],
                          "semantic_scholar:external:PubMed": ["456"]},
     [DOI], [ARXIV], ["https://www.semanticscholar.org/paper/ss-paper"]),
    ("openalex", {"openalex:work": ["W123"]}, [DOI], [],
     [f"https://doi.org/{DOI}", "https://openalex.org/W123", LANDING, PDF]),
    ("inspire", {"inspire:literature": ["123"]},
     [DOI, "10.1234/second"], [ARXIV, "hep-ph/9901001"],
     [f"https://arxiv.org/abs/{ARXIV}", "https://inspirehep.net/literature/123", PDF]),
    ("core", {"core:work": ["123"], "core:output": ["123", "456"],
              "core:identifier:oai": ["oai:repo:123"],
              "core:oai": ["oai:repo:456"]},
     [DOI], [ARXIV], [PDF, LANDING]),
]


def _search_result(source, monkeypatch):
    if source == "arxiv":
        body = f'''<feed xmlns="http://www.w3.org/2005/Atom"
                        xmlns:arxiv="http://arxiv.org/schemas/atom"><entry>
          <id>https://arxiv.org/abs/{ARXIV}</id><title>{TITLE}</title>
          <summary>A cosmic ray transport study.</summary>
          <published>2024-01-01T00:00:00Z</published>
          <author><name>Alice Tester</name></author><arxiv:doi>{DOI}</arxiv:doi>
          <link href="https://arxiv.org/pdf/{ARXIV}" type="application/pdf"/>
        </entry></feed>'''
        responses.get("http://export.arxiv.org/api/query", body=body)
        return search_arxiv("cosmic rays")[0]
    if source == "ads":
        monkeypatch.setenv("ADS_API_TOKEN", "test-token")
        responses.get("https://api.adsabs.harvard.edu/v1/search/query", json={
            "response": {"docs": [{"title": [TITLE], "author": ["Alice Tester"],
                "year": "2024", "abstract": "A transport study.",
                "bibcode": "2024ApJ...123..456A", "doi": [DOI, "10.1234/second"],
                "identifier": [f"arXiv:{ARXIV}", "oai:ads:record"]}]}})
        return search_ads("cosmic rays")[0]
    if source == "semantic_scholar":
        responses.get("https://api.semanticscholar.org/graph/v1/paper/search", json={
            "data": [{"title": TITLE, "authors": [{"name": "Alice Tester"}],
                "year": 2024, "paperId": "ss-paper", "abstract": "A transport study.",
                "externalIds": {"DOI": DOI, "ArXiv": ARXIV, "CorpusId": 123,
                                "PubMed": "456"},
                "url": "https://www.semanticscholar.org/paper/ss-paper"}]})
        return search_semantic_scholar("cosmic rays")[0]
    if source == "openalex":
        responses.get("https://api.openalex.org/works", json={"results": [{
            "title": TITLE, "id": "https://openalex.org/W123",
            "doi": f"https://doi.org/{DOI}", "publication_year": 2024,
            "authorships": [{"author": {"display_name": "Alice Tester"}}],
            "primary_location": {"landing_page_url": LANDING, "pdf_url": PDF},
            "open_access": {"oa_url": PDF}}]})
        return search.search_openalex("cosmic rays")[0]
    if source == "inspire":
        responses.get("https://inspirehep.net/api/literature", json={"hits": {"hits": [{
            "id": "123", "metadata": {"control_number": 123, "titles": [{"title": TITLE}],
                "authors": [{"full_name": "Alice Tester"}], "preprint_date": "2024-01-01",
                "dois": [{"value": DOI}, {"value": "10.1234/second"}],
                "arxiv_eprints": [{"value": ARXIV}, {"value": "hep-ph/9901001"}],
                "documents": [{"url": PDF}, {"url": PDF}]}}]}})
        return search_inspire("cosmic rays")[0]
    monkeypatch.setenv("CORE_API_KEY", "test-token")
    responses.get("https://api.core.ac.uk/v3/search/works/", json={"results": [{
        "id": 123, "outputs": [123, 456], "title": TITLE, "yearPublished": 2024,
        "authors": [{"name": "Alice Tester"}], "doi": DOI, "arxivId": ARXIV,
        "identifiers": [{"type": "oai", "identifier": "oai:repo:123"}],
        "oaiIds": ["oai:repo:456"], "downloadUrl": PDF,
        "sourceFulltextUrls": [LANDING, PDF]}]})
    return search_core("cosmic rays")[0]


@pytest.mark.parametrize("source,ids,dois,arxivs,urls", CASES)
@responses.activate
def test_backend_identifier_surface_roundtrips(source, ids, dois, arxivs, urls,
                                               monkeypatch, tmp_path):
    raw = _search_result(source, monkeypatch)
    normalized = search._normalize_paper(raw, source)
    lib = Library(tmp_path)
    paper, fresh = lib.upsert(normalized)
    assert fresh
    lib.save(force=True)
    stored = Library(tmp_path).get(paper.key)
    observations = getattr(stored, "source_identities", [])
    assert [o.model_dump() for o in observations] == [{
        "source": source, "source_ids": ids, "dois": dois,
        "arxiv_ids": arxivs, "urls": urls}]
    assert stored.doi == DOI
    assert len(responses.calls) == 1  # Only the original search request, no admission fetch.


@pytest.mark.parametrize("source,namespace", [
    ("arxiv", "arxiv:paper"), ("ads", "ads:bibcode"),
    ("semantic_scholar", "semantic_scholar:paper"), ("openalex", "openalex:work"),
    ("inspire", "inspire:literature"), ("core", "core:work"),
])
def test_missing_source_id_is_explicit_and_legacy_unknown_stays_unknown(
        tmp_path, source, namespace):
    lib = Library(tmp_path)
    legacy, _ = lib.upsert({"title": "Historical cosmic ray transport record",
                            "source": source, "paper_id": "unqualified-legacy-id"})
    raw = {"title": TITLE, "abstract": "Enough content for admission.",
           "url": LANDING, "paper_id": None, "doi": None, "arxiv_id": None}
    observed, _ = lib.upsert(search._normalize_paper(raw, source))
    lib.save(force=True)
    reloaded = Library(tmp_path)
    assert getattr(reloaded.get(legacy.key), "source_identities", None) == []
    observations = getattr(reloaded.get(observed.key), "source_identities", [])
    assert len(observations) == 1
    assert observations[0].source == source
    assert observations[0].source_ids[namespace] == []
    assert observations[0].dois == observations[0].arxiv_ids == []
    assert observations[0].urls == [LANDING]


def test_upsert_preserves_source_observations_and_canonical_provenance(tmp_path):
    lib = Library(tmp_path)
    first = {"source": "core", "source_ids": {"core:work": ["123"]},
             "dois": [DOI], "arxiv_ids": [], "urls": [PDF]}
    second = {"source": "ads", "source_ids": {"ads:bibcode": ["123"]},
              "dois": ["10.1234/conflicting"], "arxiv_ids": [ARXIV], "urls": [LANDING]}
    paper, _ = lib.upsert({"title": TITLE, "doi": DOI, "source": "core",
                           "paper_id": "123", "url": PDF, "source_identities": [first]})
    paper.download_status = "metadata_only"
    for _ in range(2):
        merged, fresh = lib.upsert({"title": TITLE, "doi": "10.1234/conflicting",
            "source": "ads", "paper_id": "other", "url": LANDING,
            "source_identities": [second]})
        assert merged is paper and not fresh
    lib.save(force=True)
    stored = Library(tmp_path).get(paper.key)
    assert [o.model_dump() for o in getattr(stored, "source_identities", [])] == [first, second]
    assert (stored.doi, stored.source, stored.paper_id, stored.url,
            stored.download_status) == (DOI, "core", "123", PDF, "metadata_only")


@responses.activate
def test_core_output_api_urls_and_typed_identifiers_are_captured(monkeypatch, tmp_path):
    monkeypatch.setenv("CORE_API_KEY", "test-token")
    output_url = "https://api.core.ac.uk/v3/outputs/123"
    responses.get("https://api.core.ac.uk/v3/search/works/", json={"results": [{
        "id": 123, "outputs": [output_url], "title": TITLE,
        "abstract": "A transport study.", "sourceFulltextUrls": [LANDING],
        "identifiers": {"doi": DOI, "arxiv": ARXIV, "oai": "oai:arxiv.org:2401.00002"},
    }]})
    lib = Library(tmp_path)
    paper, _ = lib.upsert(search._normalize_paper(search_core("cosmic rays")[0], "core"))
    lib.save(force=True)
    observation = Library(tmp_path).get(paper.key).source_identities[0]
    assert observation.source_ids == {
        "core:work": ["123"], "core:output": ["123"],
        "core:identifier:doi": [DOI], "core:identifier:arxiv": [ARXIV],
        "core:identifier:oai": ["oai:arxiv.org:2401.00002"]}
    assert observation.dois == [DOI]
    assert observation.arxiv_ids == [ARXIV, "2401.00002"]
    assert observation.urls == [LANDING, output_url]
    assert len(responses.calls) == 1


@pytest.mark.parametrize("source,namespace,url,payload", [
    ("ads", "ads:bibcode", "https://api.adsabs.harvard.edu/v1/search/query",
     {"response": {"docs": [{"title": [TITLE], "abstract": "A study."}]}}),
    ("semantic_scholar", "semantic_scholar:paper",
     "https://api.semanticscholar.org/graph/v1/paper/search",
     {"data": [{"title": TITLE, "abstract": "A study."}]}),
    ("openalex", "openalex:work", "https://api.openalex.org/works",
     {"results": [{"title": TITLE}]}),
    ("inspire", "inspire:literature", "https://inspirehep.net/api/literature",
     {"hits": {"hits": [{"metadata": {"titles": [{"title": TITLE}]}}]}}),
    ("core", "core:work", "https://api.core.ac.uk/v3/search/works/",
     {"results": [{"title": TITLE, "abstract": "A study."}]}),
])
@responses.activate
def test_backend_response_without_id_records_absence(source, namespace, url, payload,
                                                     monkeypatch, tmp_path):
    monkeypatch.setenv("CORE_API_KEY", "test-token")
    monkeypatch.setenv("ADS_API_TOKEN", "test-token")
    responses.get(url, json=payload)
    raw = getattr(search, f"search_{source}")("cosmic rays")[0]
    lib = Library(tmp_path)
    paper, _ = lib.upsert(search._normalize_paper(raw, source))
    lib.save(force=True)
    observed = Library(tmp_path).get(paper.key).source_identities[0]
    assert observed.source_ids[namespace] == []
    assert observed.dois == observed.arxiv_ids == []
    assert len(responses.calls) == 1


@responses.activate
def test_core_flat_oai_identifiers_also_retain_the_arxiv_rung(monkeypatch, tmp_path):
    """CORE flat identifier arrays carry arXiv IDs even without arxivId."""
    monkeypatch.setenv("CORE_API_KEY", "test-token")
    responses.get("https://api.core.ac.uk/v3/search/works/", json={"results": [{
        "id": 123, "title": TITLE, "abstract": "A transport study.",
        "identifiers": ["oai:arXiv.org:2306.12749", "oai:citeseerx:10.1.1.1041.7506"],
    }]})
    lib = Library(tmp_path)
    paper, _ = lib.upsert(search._normalize_paper(search_core("cosmic rays")[0], "core"))
    lib.save(force=True)
    observed = Library(tmp_path).get(paper.key).source_identities[0]
    assert observed.source_ids["core:identifier"] == [
        "oai:arXiv.org:2306.12749", "oai:citeseerx:10.1.1.1041.7506"]
    assert observed.arxiv_ids == ["2306.12749"]
    assert len(responses.calls) == 1


@responses.activate
def test_identifier_add_preserves_arxiv_adapter_surface(monkeypatch, tmp_path):
    from papervault.library.services.add_service import AddService

    raw = _search_result("arxiv", monkeypatch)
    # Reuse the existing HTTP fixture for the real identifier lookup.
    assert raw["title"] == TITLE
    lib = Library(tmp_path)
    result = AddService(lib).add(ARXIV)
    assert result["status"] == "queued"
    stored = Library(tmp_path).get(result["key"])
    assert stored.doi == DOI
    assert [o.model_dump() for o in stored.source_identities] == [{
        "source": "arxiv", "source_ids": {"arxiv:paper": [ARXIV]}, "dois": [DOI],
        "arxiv_ids": [ARXIV],
        "urls": [f"https://arxiv.org/abs/{ARXIV}", f"https://arxiv.org/pdf/{ARXIV}"]}]
    assert len(responses.calls) == 2  # Test setup search plus the original add lookup.


@responses.activate
def test_identifier_add_preserves_crossref_and_ss_surfaces(tmp_path):
    from papervault.library.services.add_service import AddService

    responses.get(f"https://api.crossref.org/works/{DOI}", json={"message": {
        "DOI": DOI, "title": [TITLE], "author": [{"given": "Alice", "family": "Tester"}],
        "issued": {"date-parts": [[2024]]}, "type": "journal-article",
        "alternative-id": ["publisher-id"], "URL": LANDING,
        "link": [{"URL": PDF}], "resource": {"primary": {"URL": LANDING}}}})
    responses.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{DOI}", json={
        "paperId": "ss-paper", "title": TITLE, "authors": [{"name": "Alice Tester"}],
        "externalIds": {"DOI": DOI, "ArXiv": ARXIV, "CorpusId": 123},
        "url": "https://www.semanticscholar.org/paper/ss-paper"})
    lib = Library(tmp_path)
    result = AddService(lib).add(DOI)
    assert result["status"] == "queued"
    stored = Library(tmp_path).get(result["key"])
    assert stored.source == "crossref+ss" and stored.doi == DOI
    assert [o.model_dump() for o in stored.source_identities] == [
        {"source": "crossref", "source_ids": {"crossref:doi": [DOI],
                                              "crossref:alternative": ["publisher-id"]},
         "dois": [DOI], "arxiv_ids": [], "urls": [LANDING, PDF]},
        {"source": "semantic_scholar", "source_ids": {"semantic_scholar:paper": ["ss-paper"],
                                                      "semantic_scholar:external:CorpusId": ["123"]},
         "dois": [DOI], "arxiv_ids": [ARXIV],
         "urls": ["https://www.semanticscholar.org/paper/ss-paper"]}]
    assert len(responses.calls) == 2  # The existing Crossref and SS metadata calls only.


@responses.activate
def test_identifier_add_preserves_ss_arxiv_fallback_surface(monkeypatch, tmp_path):
    from papervault.library.services.add_service import AddService

    monkeypatch.setattr("papervault.library.fetch.search_arxiv", lambda *_, **__: [])
    responses.get("https://api.semanticscholar.org/graph/v1/paper/ARXIV:2401.00001", json={
        "paperId": "ss-paper", "title": TITLE, "externalIds": {"DOI": DOI, "ArXiv": ARXIV},
        "url": LANDING})
    lib = Library(tmp_path)
    result = AddService(lib).add(ARXIV)
    assert result["status"] == "queued"
    stored = Library(tmp_path).get(result["key"])
    assert [o.model_dump() for o in stored.source_identities] == [{
        "source": "semantic_scholar", "source_ids": {"semantic_scholar:paper": ["ss-paper"]},
        "dois": [DOI], "arxiv_ids": [ARXIV], "urls": [LANDING]}]
    assert len(responses.calls) == 1


@responses.activate
def test_core_search_links_and_secondary_id_fields_roundtrip(monkeypatch, tmp_path):
    """CORE work search exposes links and secondary IDs outside identifiers too."""
    monkeypatch.setenv("CORE_API_KEY", "test-token")
    responses.get("https://api.core.ac.uk/v3/search/works/", json={"results": [{
        "id": 98460589, "title": TITLE, "abstract": "A transport study.",
        "magId": "2801602492", "pubmedId": None,
        "links": [{"type": "download", "url": "http://arxiv.org/abs/2003.13865"},
                  {"type": "display", "url": "https://core.ac.uk/works/98460589"}],
    }]})
    lib = Library(tmp_path)
    paper, _ = lib.upsert(search._normalize_paper(search_core("cosmic rays")[0], "core"))
    lib.save(force=True)
    observed = Library(tmp_path).get(paper.key).source_identities[0]
    assert observed.source_ids == {"core:work": ["98460589"], "core:output": [],
                                   "core:mag": ["2801602492"], "core:pubmed": []}
    assert observed.urls == ["http://arxiv.org/abs/2003.13865", "https://core.ac.uk/works/98460589"]
    assert len(responses.calls) == 1
