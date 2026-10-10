"""INSPIRE source identity survives ingestion and the existing PDF cascade."""

from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import responses

from papervault.library import Library, download
from papervault.library.models import Paper
from papervault.library.search import _normalize_paper
from papervault.library.sources.inspire import API_URL, search_inspire


TITLE = "Alignment of the AMS-02 silicon Tracker"
URL = "https://inspirehep.net/files/8a9e9f449f462b89f67c55b73fbdfcdd"


def _card(**updates):
    return dict(title=TITLE, authors=["Ambrosi, G."], year=2013,
                source="inspire", abstract="A silicon tracker alignment study.", **updates)


def _hit(recid="1412451", **updates):
    meta = {
        "control_number": 1412451,
        "titles": [{"title": TITLE}],
        "authors": [{"full_name": "Ambrosi, G."}],
        "publication_info": [{"year": 2013}],
        "documents": [{"url": URL}],
    }
    meta.update(updates)
    return {"id": recid, "metadata": meta}


def _response(*hits, total=None):
    return {"hits": {"total": len(hits) if total is None else total, "hits": list(hits)}}


@responses.activate
def test_search_source_pair_roundtrips_without_other_identifiers(tmp_path):
    hit = _hit(documents=[{"url": URL, "attachment": {"content": "embedded body"}},
                          {"url": URL}, {"url": "/relative"}, {"url": "javascript:x"}])
    responses.get(API_URL, json=_response(hit))
    raw = search_inspire("tracker")[0]
    normalized = _normalize_paper(raw, "inspire")
    lib = Library(tmp_path)
    paper, fresh = lib.upsert(normalized)
    assert fresh
    lib.save(force=True)
    reloaded = Library(tmp_path).get(paper.key)
    assert getattr(reloaded, "inspire_record_id", "") == "1412451"
    assert reloaded.inspire_document_urls == [URL]
    assert reloaded.doi == reloaded.arxiv_id == reloaded.paper_id == ""
    assert "embedded body" not in lib.index_path.read_text()
    fields = parse_qs(urlsplit(responses.calls[0].request.url).query)["fields"][0]
    assert "documents.url" in fields and "control_number" in fields
    assert "preprint_date" in fields


@pytest.mark.parametrize("recid,control,want", [
    ("1412451", 1412451, "1412451"), (1412451, None, "1412451"),
    (None, 1412451, "1412451"), (None, None, ""),
    ("1412451", 99, ""), ("../1412451", 1412451, ""),
    ("0", 0, ""), (True, 1, ""), ("SS-record", None, ""),
])
@responses.activate
def test_search_requires_valid_consistent_source_id(recid, control, want):
    responses.get(API_URL, json=_response(_hit(recid, control_number=control)))
    result = _normalize_paper(search_inspire("tracker")[0], "inspire")
    assert result.get("inspire_record_id") == want
    assert result.get("inspire_document_urls") == ([URL] if want else [])


@pytest.mark.parametrize("documents", [None, [], [{}], [{"url": "ftp://example.org/x"}],
    [{"url": "https://"}], [{"url": "https://a b/x"}],
    [{"url": "https://user:pass@example.org/x"}], [{"url": "https://example.org:bad/x"}],
    [{"url": 5}], [None], [{"url": "https://example.org/line\nbreak"}],
    [{"url": "https://example.org/%zz"}], [{"url": "https://example.org:0/x"}],
    [{"url": "https://example.org/\x7f"}], [{"url": " https://example.org/file"}],
])
@responses.activate
def test_absent_or_malformed_documents_retain_identity_only(documents):
    responses.get(API_URL, json=_response(_hit(documents=documents)))
    result = _normalize_paper(search_inspire("tracker")[0], "inspire")
    assert result.get("inspire_record_id") == "1412451"
    assert result.get("inspire_document_urls") == []


def test_legacy_cards_have_empty_source_fields():
    paper = Paper(key="Old", title=TITLE)
    assert getattr(paper, "inspire_record_id", None) == ""
    assert getattr(paper, "inspire_document_urls", None) == []


@pytest.mark.parametrize("incoming,want_id,want_urls", [
    ({"inspire_record_id": "1412451", "inspire_document_urls": [URL]}, "1412451", [URL]),
    ({"inspire_document_urls": [URL]}, "", []),
    ({"inspire_record_id": "invalid", "inspire_document_urls": [URL]}, "", []),
])
def test_ingress_rejects_orphan_links(tmp_path, incoming, want_id, want_urls):
    paper, _ = Library(tmp_path).upsert(_card(**incoming))
    assert getattr(paper, "inspire_record_id", None) == want_id
    assert paper.inspire_document_urls == want_urls


@pytest.mark.parametrize("conflict", ["recid", "author", "initial", "year", "title", "doi"])
def test_source_pair_merge_abstains_on_identity_conflict(tmp_path, conflict):
    lib = Library(tmp_path)
    existing, _ = lib.upsert(_card(doi="10.1/x", inspire_record_id="1412451"))
    incoming = _card(inspire_record_id="1412451", inspire_document_urls=[URL])
    if conflict == "recid":
        incoming["inspire_record_id"] = "99"
    elif conflict == "author":
        incoming["authors"] = ["Smith, G."]
    elif conflict == "initial":
        incoming["authors"] = ["Ambrosi, X."]
    elif conflict == "year":
        incoming["year"] = 2014
    elif conflict == "title":
        incoming["title"] = "Alignment of another silicon Tracker"
        incoming["doi"] = "10.1/x"
    elif conflict == "doi":
        incoming["doi"] = "10.1/other"
    merged, _ = lib.upsert(incoming)
    assert merged is existing
    assert getattr(merged, "inspire_record_id", None) == "1412451"
    assert merged.inspire_document_urls == []


def test_verified_merge_fills_pair_in_place_and_unions_only_same_record(tmp_path):
    lib = Library(tmp_path)
    existing, _ = lib.upsert(_card())
    existing.download_status = "metadata_only"
    merged, _ = lib.upsert(_card(inspire_record_id="1412451", inspire_document_urls=[URL]))
    assert merged is existing
    second = "https://inspirehep.net/files/second"
    lib.upsert(_card(inspire_record_id="1412451", inspire_document_urls=[second, URL]))
    lib.save(force=True)
    saved = Library(tmp_path).get(existing.key)
    assert getattr(saved, "inspire_record_id", "") == "1412451"
    assert saved.inspire_document_urls == [URL, second]
    assert saved.download_status == "metadata_only"


@pytest.mark.parametrize("left,right,want", [
    ("John Smith", "Jane Smith", ""), ("Smith, John Paul", "Smith, John Peter", ""),
    ("Smith, J.", "John Smith", "1412451"), ("John Smith", "Smith, J.", "1412451"),
    ("Smith, John Paul", "Smith, J. P.", "1412451"),
])
def test_full_given_names_must_agree_beyond_shared_initial(tmp_path, left, right, want):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(dict(_card(), authors=[left]))
    lib.upsert(dict(_card(inspire_record_id="1412451", inspire_document_urls=[URL]), authors=[right]))
    assert paper.inspire_record_id == want
    assert paper.inspire_document_urls == ([URL] if want else [])


@pytest.mark.parametrize("left,right,want", [
    (["Smith, John", "Smith, James", "Smith, K."],
     ["Smith, J.", "Smith, Kate", "Smith, Karl"], ""),
    (["Smith, John", "Smith, Kate"], ["Smith, K.", "Smith, J."], "1412451"),
])
def test_repeated_surnames_require_distinct_matching_authors(tmp_path, left, right, want):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(dict(_card(), authors=left))
    lib.upsert(dict(_card(inspire_record_id="1412451", inspire_document_urls=[URL]), authors=right))
    assert paper.inspire_record_id == want
    assert paper.inspire_document_urls == ([URL] if want else [])


@pytest.fixture
def cascade(tmp_path, monkeypatch):
    from tests.library.test_download import _text_pdf

    lib = Library(tmp_path)
    paper, _ = lib.upsert(_card())
    monkeypatch.setenv("PAPERVAULT_VAULT", str(tmp_path))
    for env in ("PAPER_PIPELINE_USE_SCIHUB", "CORE_API_KEY", "ADS_API_TOKEN",
                "ANNAS_ARCHIVE_API_KEY", "WILEY_TDM_TOKEN", "ELSEVIER_TDM_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    responses.get("http://export.arxiv.org/api/query",
                  body='<feed xmlns="http://www.w3.org/2005/Atom"/>')
    prompts = []

    def judge(messages):
        prompts.append(messages[1]["content"])
        wrong = "Unrelated document about dinosaurs" in messages[1]["content"]
        return json.dumps({"match": not wrong, "reason": "different work" if wrong else "same work"})

    monkeypatch.setattr("papervault.library.llm.get_llm",
                        lambda **_: SimpleNamespace(call=judge))
    data = _text_pdf(TITLE + ". G. Ambrosi. We study tracker alignment in cosmic ray physics.")
    return lib, paper, data, prompts


def _domain_events(lib):
    return [event for line in lib.manifest_path.read_text().splitlines()
            if (event := json.loads(line)).get("source") == "domain_aggregators"
            and event["event"] != "download_telemetry"]


@pytest.mark.parametrize("route", ["links", "recid", "title"])
@responses.activate
def test_identifierless_inspire_reaches_verified_normal_cascade(cascade, route):
    lib, paper, data, prompts = cascade
    if route in {"links", "recid"}:
        paper.inspire_record_id = "1412451"
    if route == "links":
        paper.inspire_document_urls = [URL]
    elif route == "recid":
        responses.get(API_URL + "/1412451", json=_hit())
    else:
        # A non-matching first hit must not preempt the unique matching second hit.
        other = _hit("99", control_number=99, titles=[{"title": "Another work"}])
        responses.get(API_URL, json=_response(other, _hit()))
    responses.get(URL, body=data)
    assert download.download_paper(paper, lib)
    assert lib.pdf_path(paper.key).read_bytes() == data
    assert paper.download_status == "ok"
    assert paper.download_source == "domain_aggregators"
    assert paper.inspire_record_id == "1412451"
    assert paper.inspire_document_urls == [URL]
    assert paper.doi == paper.arxiv_id == paper.paper_id == ""
    assert len(prompts) == 1
    assert _domain_events(lib)[0]["event"] == "downloaded"
    assert _domain_events(lib)[0]["verify"].startswith("llm_match")
    api_calls = [c for c in responses.calls if API_URL in c.request.url]
    if route == "links":
        assert api_calls == []
    elif route == "title":
        query = parse_qs(urlsplit(api_calls[0].request.url).query)
        assert query["q"] == ['title:"Alignment of the AMS-02 silicon Tracker"']
        assert query["size"] == ["3"]


@pytest.mark.parametrize("case", ["ambiguous", "truncated", "missing_total", "inexact_total",
    "no_hit", "no_id", "conflicting_id", "author", "initial", "year", "title", "no_authors",
    "no_documents", "malformed_documents", "http_error", "bad_json", "incomplete_twin"])
@responses.activate
def test_historical_lookup_abstains_honestly(cascade, case):
    lib, paper, _, prompts = cascade
    hit = _hit()
    body = _response(hit)
    if case == "ambiguous":
        body = _response(hit, _hit("99", control_number=99))
    elif case == "truncated":
        body = _response(hit, total=4)
    elif case == "missing_total":
        del body["hits"]["total"]
    elif case == "inexact_total":
        body["hits"]["total"] = {"value": 1, "relation": "gte"}
    elif case == "no_hit":
        body = _response()
    elif case == "no_id":
        hit.pop("id")
        hit["metadata"].pop("control_number")
    elif case == "conflicting_id":
        hit["metadata"]["control_number"] = 99
    elif case == "author":
        hit["metadata"]["authors"] = [{"full_name": "Smith, G."}]
    elif case == "initial":
        hit["metadata"]["authors"] = [{"full_name": "Ambrosi, X."}]
    elif case == "year":
        hit["metadata"]["publication_info"] = [{"year": 2014}]
    elif case == "title":
        hit["metadata"]["titles"] = [{"title": TITLE + " revisited"}]
    elif case == "no_authors":
        hit["metadata"].pop("authors")
    elif case == "incomplete_twin":
        body = _response(hit, _hit("99", control_number=99, authors=[]))
    elif case == "no_documents":
        hit["metadata"].pop("documents")
    elif case == "malformed_documents":
        hit["metadata"]["documents"] = [{"url": "/relative"}]
    if case == "http_error":
        responses.get(API_URL, status=503)
    elif case == "bad_json":
        responses.get(API_URL, body="not JSON")
    else:
        responses.get(API_URL, json=body)
    assert not download.download_paper(paper, lib)
    assert not lib.has_pdf(paper.key)
    assert paper.download_status == "metadata_only"
    assert paper.download_source == ""
    assert prompts == []
    assert [e["event"] for e in _domain_events(lib)] == ["download_miss"]
    assert len([c for c in responses.calls if API_URL in c.request.url]) == 1
    assert not any(URL == c.request.url for c in responses.calls)
    if case in {"no_documents", "malformed_documents"}:
        assert paper.inspire_record_id == "1412451"
        assert paper.inspire_document_urls == []
    else:
        assert paper.inspire_record_id == ""


@pytest.mark.parametrize("change", [{"source": "core"}, {"source": "notinspire"},
    {"authors": []}, {"title": "Short"}, {"inspire_document_urls": [URL], "source": "manual"},
    {"inspire_record_id": "invalid"}])
@responses.activate
def test_missing_inspire_prerequisites_skip_without_source_lookup(cascade, change):
    lib, paper, _, _ = cascade
    for key, value in change.items():
        setattr(paper, key, value)
    assert not download.download_paper(paper, lib)
    assert [e["event"] for e in _domain_events(lib)] == ["download_skip"]
    assert not any(API_URL in c.request.url or URL == c.request.url for c in responses.calls)


@pytest.mark.parametrize("case", ["different_recid", "different_work", "missing_recid"])
@responses.activate
def test_direct_record_lookup_verifies_identity_before_taking_links(cascade, case):
    lib, paper, _, _ = cascade
    paper.inspire_record_id = "1412451"
    hit = _hit()
    if case == "different_recid":
        hit = _hit("99", control_number=99)
    elif case == "different_work":
        hit["metadata"]["authors"] = [{"full_name": "Someone, Else"}]
    else:
        hit.pop("id")
        hit["metadata"].pop("control_number")
    responses.get(API_URL + "/1412451", json=hit)
    assert not download.download_paper(paper, lib)
    assert paper.inspire_record_id == "1412451"
    assert paper.inspire_document_urls == []
    assert [e["event"] for e in _domain_events(lib)] == ["download_miss"]
    assert not any(c.request.url == URL for c in responses.calls)


@responses.activate
def test_direct_lookup_rechecks_current_card_before_using_document(cascade):
    lib, paper, _, _ = cascade
    paper.inspire_record_id = "1412451"

    def changed_during_request(_):
        paper.year = 2014
        return 200, {}, json.dumps(_hit())

    responses.add_callback(responses.GET, API_URL + "/1412451", callback=changed_during_request)
    assert not download.download_paper(paper, lib)
    assert paper.inspire_document_urls == []
    assert not any(c.request.url == URL for c in responses.calls)


@responses.activate
def test_composite_source_and_unicode_exact_title_lookup(cascade):
    lib, paper, data, _ = cascade
    paper.source = "semantic_scholar+inspire"
    paper.title = "Mesure de l'amplitude d'une onde de plasma créée"
    paper.authors = ["Franck Wojda"]
    paper.year = None
    hit = _hit(titles=[{"title": "  MESURE de l'amplitude d'une onde de plasma cre\u0301e\u0301e  "}],
               authors=[{"full_name": "Wojda, Franck"}], publication_info=[])
    responses.get(API_URL, json=_response(hit))
    responses.get(URL, body=data)
    assert download.download_paper(paper, lib)
    assert paper.inspire_record_id == "1412451"
    assert paper.year is None


@responses.activate
def test_source_document_failure_is_miss_and_tries_next_link(cascade):
    lib, paper, data, _ = cascade
    paper.inspire_record_id = "1412451"
    second = "https://inspirehep.net/files/second"
    paper.inspire_document_urls = [URL, second]
    responses.get(URL, status=404)
    responses.get(second, body=data)
    assert download.download_paper(paper, lib)
    assert paper.download_source == "domain_aggregators"


@responses.activate
def test_unrelated_pdf_rejected_before_write_and_existing_extracts_preserved(cascade):
    from tests.library.test_download import _text_pdf

    lib, paper, _, prompts = cascade
    paper.inspire_record_id = "1412451"
    paper.inspire_document_urls = [URL]
    lib.txt_path(paper.key).write_text("Existing extraction. " * 50)
    lib.md_path(paper.key).write_text("# Existing markdown\n" * 20)
    before = [p.read_bytes() for p in (lib.txt_path(paper.key), lib.md_path(paper.key))]
    responses.get(URL, body=_text_pdf("Unrelated document about dinosaurs. A different author. " * 4))
    assert not download.download_paper(paper, lib)
    assert not lib.pdf_path(paper.key).exists()
    assert not lib.pdf_path(paper.key).with_suffix(".pdf.tmp").exists()
    assert paper.download_source == ""
    assert len(prompts) == 1
    assert [e["event"] for e in _domain_events(lib)] == ["download_pdf_mismatch"]
    assert [p.read_bytes() for p in (lib.txt_path(paper.key), lib.md_path(paper.key))] == before


@responses.activate
def test_existing_pdf_preserved_without_any_probe(cascade):
    lib, paper, _, prompts = cascade
    lib.pdf_path(paper.key).write_bytes(b"%PDF- existing asset")
    assert download.download_paper(paper, lib)
    assert lib.pdf_path(paper.key).read_bytes() == b"%PDF- existing asset"
    assert len(responses.calls) == 0
    assert prompts == []


@responses.activate
async def test_historical_pair_persists_at_existing_download_queue_save(cascade):
    from papervault.library.services.download_queue import DownloadQueue

    lib, paper, data, _ = cascade
    responses.get(API_URL, json=_response(_hit()))
    responses.get(URL, body=data)
    await DownloadQueue(lib)._process_one(paper.key)
    saved = Library(lib.root).get(paper.key)
    assert saved.download_status == "ok"
    assert saved.download_source == "domain_aggregators"
    assert saved.inspire_record_id == "1412451"
    assert saved.inspire_document_urls == [URL]
