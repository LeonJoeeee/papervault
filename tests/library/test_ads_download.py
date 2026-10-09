"""ADS record qualification, document evidence and existing cascade integration."""
from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
import responses

from papervault.library import Library, cli, download, models
from papervault.library.download_sources import ads
from tests.library.test_download import _text_pdf

BIBCODE = "2019amos.confE...8M"
OTHER = "2019mlhp.confR..59T"
API = "https://api.adsabs.harvard.edu/v1/search/query"
GATEWAY = f"https://ui.adsabs.harvard.edu/link_gateway/{BIBCODE}/PUB_PDF"
LEGACY = f"https://articles.adsabs.harvard.edu/pdf/{BIBCODE}"
TITLE = "Physics Based Density Estimation Using Orbital Debris Tracking Data"


@pytest.fixture
def lib(tmp_path):
    return Library(tmp_path)


@pytest.fixture
def paper(lib, monkeypatch):
    p, _ = lib.upsert({"title": TITLE, "authors": ["Mutschler, Shaylah"], "year": 2019,
                       "source": "ads", "paper_id": BIBCODE,
                       "url": f"https://ui.adsabs.harvard.edu/abs/{BIBCODE}/abstract"})
    monkeypatch.setenv("ADS_API_TOKEN", "inert-test-token")
    monkeypatch.setattr(download, "_STRATEGIES", [
        ("domain_aggregators", download._try_domain_aggregators),
    ])
    for name in ("_try_inspire", "_try_europepmc", "_try_zenodo"):
        monkeypatch.setattr(download, name, lambda _: None)
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    return p


def metadata(**changes):
    doc = {"bibcode": BIBCODE, "title": [TITLE], "author": ["Mutschler, Shaylah"],
           "doctype": "inproceedings", "esources": ["PUB_PDF"], "identifier": [BIBCODE]}
    return doc | changes


def stub_metadata(**changes):
    responses.get(API, json={"response": {"docs": [metadata(**changes)]}})


def paper_pdf():
    return _text_pdf(f"{TITLE}\nShaylah Mutschler\n1. Introduction\nWe study orbital debris.\n"
                     "2. Results\nThe model produces density estimates.")


def abstract_pdf(marker="Geophysical Research Abstracts. EGU General Assembly"):
    return _text_pdf(f"{marker}\n{TITLE}\nShaylah Mutschler\n{ABSTRACT_BODY}")


ABSTRACT_BODY = ('We study how orbital tracking data constrain atmospheric density estimates. '
                 'The measurements cover several objects and include changes in the space environment. '
                 'A numerical model combines these observations to estimate density. '
                 'The contribution describes the experiment and summarizes the resulting estimates.')


@pytest.mark.parametrize("url", [
    f"https://ui.adsabs.harvard.edu/abs/{BIBCODE}/abstract",
    f"https://ui.adsabs.harvard.edu/abs/{BIBCODE}",
    "https://ui.adsabs.harvard.edu/abs/1996A%26A...316..538H/abstract",
])
def test_ads_bibcode_from_recognized_url(paper, url):
    paper.url = url
    paper.source = "manual"
    paper.paper_id = "unrelated-semantic-scholar-id"
    assert ads._ads_bibcode(paper) == urlsplit(url).path.split('/')[2].replace('%26', '&')


@pytest.mark.parametrize("url", [
    f"https://evil.example/abs/{BIBCODE}/abstract",
    f"https://ui.adsabs.harvard.edu.evil.example/abs/{BIBCODE}/abstract",
    f"https://user:password@ui.adsabs.harvard.edu/abs/{BIBCODE}/abstract",
    f"https://ui.adsabs.harvard.edu:8443/abs/{BIBCODE}/abstract",
    f"https://ui.adsabs.harvard.edu/abs/{BIBCODE}/unexpected",
    "https://ui.adsabs.harvard.edu/abs/1996A%2526A...316..538H/abstract",
    "https://ui.adsabs.harvard.edu/abs/invalid/abstract",
    f"https://ui.adsabs.harvard.edu/abs/{BIBCODE}/abstract\n",
    "https://[broken/abs/record",
])
@responses.activate
def test_unqualified_ads_identity_never_requests(paper, url):
    paper.url = url
    paper.source = "manual"
    assert ads._ads_bibcode(paper) is None
    assert ads._try_ads(paper) is None
    assert not responses.calls


@responses.activate
def test_conflicting_ads_ids_are_not_eligible(paper):
    paper.paper_id = OTHER
    assert ads._ads_bibcode(paper) is None
    assert ads._try_ads(paper) is None
    assert not responses.calls


def test_standalone_bibcode_requires_ads_source(paper):
    paper.url = ""
    assert ads._ads_bibcode(paper) == BIBCODE
    paper.source = "semantic_scholar"
    assert ads._ads_bibcode(paper) is None


@responses.activate
def test_idless_ads_pdf_uses_normal_identity_save_and_bookkeeping(lib, paper, monkeypatch):
    stub_metadata()
    data = paper_pdf()
    responses.get(GATEWAY, body=data)
    prompts = []

    def judge(messages):
        prompts.append(messages[1]["content"])
        return json.dumps({"match": True, "reason": "same work"})

    monkeypatch.setattr("papervault.library.llm.get_llm",
                        lambda **_: SimpleNamespace(call=judge))
    assert download.download_paper(paper, lib) is True
    assert lib.pdf_path(paper.key).read_bytes() == data
    assert paper.pdf_path == f"pdfs/{paper.key}.pdf"
    assert paper.download_status == "ok"
    assert paper.download_source == "domain_aggregators"
    assert paper.doi == paper.arxiv_id == ""
    assert len(prompts) == 1 and TITLE in prompts[0]
    params = parse_qs(urlsplit(responses.calls[0].request.url).query)
    assert params['q'] == [f'bibcode:"{BIBCODE}"']
    assert {'doctype', 'esources', 'identifier', 'doi', 'title', 'author'} <= set(params['fl'][0].split(','))
    assert responses.calls[0].request.headers['Authorization'] == 'Bearer inert-test-token'
    assert 'Authorization' not in responses.calls[1].request.headers
    assert paper.ads_availability.availability == 'retrievable'
    events = [json.loads(l) for l in lib.manifest_path.read_text().splitlines()]
    assert not any(e['event'] == 'download_skip' and e.get('source') == 'domain_aggregators' for e in events)
    assert any(e['event'] == 'downloaded' and e['verify'] == 'llm_match: same work' for e in events)
    assert any(e['event'] == 'ads_availability' for e in events)
    lib.save()
    assert Library(lib.root).get(paper.key).ads_availability == paper.ads_availability


@responses.activate
def test_bibcode_query_encodes_ampersand(paper):
    paper.paper_id = '1996A&A...316..538H'
    paper.url = 'https://ui.adsabs.harvard.edu/abs/1996A%26A...316..538H/abstract'
    stub_metadata(bibcode=paper.paper_id, esources=[])
    responses.get('https://articles.adsabs.harvard.edu/pdf/1996A%26A...316..538H', status=404)
    assert ads._try_ads(paper) is None
    params = parse_qs(urlsplit(responses.calls[0].request.url).query)
    assert params['q'] == ['bibcode:"1996A&A...316..538H"']


@pytest.mark.parametrize('changes', [{'bibcode': OTHER}, {'bibcode': '../bad'}])
@responses.activate
def test_ads_metadata_identity_conflict_does_not_fetch_pdf(paper, changes):
    stub_metadata(**changes)
    assert ads._try_ads(paper) is None
    assert len(responses.calls) == 1
    assert paper.ads_availability.availability == 'unknown'
    assert paper.ads_availability.reason == 'metadata_identity_mismatch'


@pytest.mark.parametrize('status', [401, 403, 429, 500])
@responses.activate
def test_ads_api_errors_never_confirm_absence(paper, status):
    responses.get(API, status=status)
    assert ads._try_ads(paper) is None
    assert paper.ads_availability.availability == 'unknown'
    assert paper.ads_availability.reason == f'api_http_{status}'
    assert len(responses.calls) == 1


@pytest.mark.parametrize('body', ['not JSON', '{}', '{"response":{"docs":[null]}}'])
@responses.activate
def test_ads_malformed_metadata_stays_unknown(paper, body):
    responses.get(API, body=body)
    assert ads._try_ads(paper) is None
    assert paper.ads_availability.availability == 'unknown'


@responses.activate
def test_no_token_is_skip_and_keeps_unknown(paper, monkeypatch):
    monkeypatch.delenv('ADS_API_TOKEN')
    assert ads._try_ads(paper) is None
    assert paper.ads_availability.availability == 'unknown'
    assert paper.ads_availability.reason == 'missing_credentials'
    assert download._download_skip_reason('domain_aggregators', paper) == 'missing_credentials'
    assert not responses.calls


@pytest.mark.parametrize('status, expected', [(403, 'blocked'), (429, 'blocked'),
                                            (404, 'unknown'), (500, 'unknown')])
@responses.activate
def test_failed_document_routes_distinguish_access_and_stale(paper, status, expected):
    stub_metadata()
    responses.get(GATEWAY, status=status)
    responses.get(LEGACY, status=404)
    assert ads._try_ads(paper) is None
    assert paper.ads_availability.availability == expected
    assert len(paper.ads_availability.documents) == (1 if status == 429 else 2)
    assert paper.ads_availability.availability != 'confirmed_abstract_only'


@responses.activate
def test_gateway_failure_kept_when_legacy_succeeds(paper):
    stub_metadata(doctype='article')
    responses.get(GATEWAY, body=requests.Timeout('gateway failed'))
    responses.get(LEGACY, body=paper_pdf())
    assert ads._try_ads(paper) == paper_pdf()
    assert [d.outcome for d in paper.ads_availability.documents] == ['transport_error', 'pdf']
    assert paper.ads_availability.availability == 'retrievable'


@pytest.mark.parametrize('doctype, esources', [('abstract', []), ('article', []),
                                            ('phdthesis', []), ('inproceedings', ['PUB_HTML'])])
@responses.activate
def test_metadata_and_no_esource_or_html_are_unknown(paper, doctype, esources):
    stub_metadata(doctype=doctype, esources=esources)
    responses.get(LEGACY, status=404)
    assert ads._try_ads(paper) is None
    assert paper.ads_availability.publication_kind == doctype
    assert paper.ads_availability.availability == 'unknown'


@pytest.mark.parametrize('marker', ['Geophysical Research Abstracts. EGU General Assembly',
                                    'COSPAR Scientific Assembly. Consider as poster only.'])
@responses.activate
def test_confirmed_abstract_stops_hunt_and_persists_reason(lib, paper, monkeypatch, marker):
    stub_metadata(doctype='abstract', property=['ARTICLE'])
    responses.get(GATEWAY, body=abstract_pdf(marker))
    responses.get(LEGACY, status=404)
    later = []
    monkeypatch.setattr(download, '_STRATEGIES', download._STRATEGIES + [
        ('later', lambda _: later.append('called')),
    ])
    monkeypatch.setattr(download, '_try_firecrawl_text_fallback', lambda *_: later.append('firecrawl'))
    assert download.download_paper(paper, lib) is False
    assert paper.ads_availability.availability == 'confirmed_abstract_only'
    assert paper.ads_availability.reason == 'published_meeting_abstract'
    assert paper.download_status == 'metadata_only'
    assert not later and not lib.has_pdf(paper.key)
    lib.save()
    loaded = Library(lib.root).get(paper.key)
    assert loaded.ads_availability == paper.ads_availability
    events = [json.loads(l) for l in lib.manifest_path.read_text().splitlines()]
    assert any(e['event'] == 'download_abstract_only' and e['reason'] == 'published_meeting_abstract' for e in events)


def test_abstract_anthology_inspects_target_after_first_pages(paper):
    import io
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    writer.add_page(PdfReader(io.BytesIO(_text_pdf('Machine Learning in Heliophysics\nAbstract book'))).pages[0])
    for _ in range(57):
        writer.add_page(PdfReader(io.BytesIO(_text_pdf('Other abstracts'))).pages[0])
    writer.add_page(PdfReader(io.BytesIO(_text_pdf(f'Shaylah Mutschler\n{TITLE}\n' +
        'We discuss how orbital observations can constrain atmospheric density models. '
        'The experiment combines measurements from several objects with different trajectories. '
        'A numerical estimator uses these measurements to infer changes in density. '
        'We summarize the method and report the resulting estimates for the observed conditions.'))).pages[0])
    out = io.BytesIO()
    writer.write(out)
    evidence = ads._inspect_ads_document(out.getvalue(), paper)
    result = ads._classify_ads_availability(metadata(), [evidence])
    assert result.availability == 'confirmed_abstract_only'
    assert result.reason == 'published_abstract_in_anthology'
    assert evidence.target_page == 59
    assert result.publication_kind == 'inproceedings'


@pytest.mark.parametrize('data', [b'%PDF-1.4 invalid', _text_pdf('Unrelated author and meeting abstract'),
                                 _text_pdf(f'{TITLE}\nShaylah Mutschler\nAbstract. Brief text.')])
def test_short_or_unmatched_pdf_never_confirms_abstract(paper, data):
    evidence = ads._inspect_ads_document(data, paper)
    result = ads._classify_ads_availability(metadata(doctype='abstract'), [evidence])
    assert result.availability == 'unknown'


@responses.activate
def test_pdf_identity_rejection_clears_retrievable(lib, paper, monkeypatch):
    stub_metadata()
    responses.get(GATEWAY, body=paper_pdf())
    monkeypatch.setattr(download, '_verify_pdf_matches_metadata', lambda *_: (False, 'llm_mismatch: other work'))
    assert download.download_paper(paper, lib) is False
    assert not lib.has_pdf(paper.key)
    assert paper.ads_availability.availability == 'unknown'
    assert paper.ads_availability.reason == 'document_identity_mismatch'


@pytest.mark.parametrize('outcome', ['incomplete', 'html', 'invalid_pdf', 'transport_error'])
def test_incomplete_non_pdf_and_request_errors_are_unknown(outcome):
    evidence = models.ADSDocumentEvidence(outcome=outcome)
    result = ads._classify_ads_availability(metadata(doctype='abstract'), [evidence])
    assert result.availability == 'unknown'


@responses.activate
def test_capped_document_does_not_retire(paper, monkeypatch):
    stub_metadata(doctype='phdthesis')
    monkeypatch.setattr(ads, '_MAX_PDF_BYTES', 12)
    responses.get(GATEWAY, body=paper_pdf())
    responses.get(LEGACY, status=404)
    assert ads._try_ads(paper) is None
    assert paper.ads_availability.availability == 'unknown'
    assert paper.ads_availability.documents[0].outcome == 'incomplete'


@pytest.mark.parametrize('asset', ['pdf', 'md', 'txt'])
def test_preconfirmed_retirement_preserves_existing_assets(lib, paper, monkeypatch, asset):
    evidence = ads._inspect_ads_document(abstract_pdf(), paper)
    paper.ads_availability = ads._classify_ads_availability(metadata(doctype='abstract'), [evidence])
    path = getattr(lib, f'{asset}_path')(paper.key)
    original = b'Existing asset must remain byte-for-byte unchanged.'
    path.write_bytes(original)
    setattr(paper, f'{asset}_path', str(path.relative_to(lib.root)))
    paper.download_status = 'ok'
    monkeypatch.setattr(download, '_STRATEGIES', [('unexpected', lambda _: pytest.fail('hunt restarted'))])
    monkeypatch.setattr(download, '_gate_firecrawl_md', lambda *_: pytest.fail('asset gate invoked'))
    assert download.download_paper(paper, lib) is (asset == 'pdf')
    assert path.read_bytes() == original
    assert getattr(paper, f'{asset}_path') == str(path.relative_to(lib.root))
    assert paper.download_status == ('metadata_only' if asset == 'txt' else 'ok')
    lib.save()
    assert Library(lib.root).get(paper.key).ads_availability.reason == 'published_meeting_abstract'


def test_document_identity_allows_pdf_word_spacing(paper):
    data = _text_pdf('Geophysical Research Abstracts. EGU General Assembly\n'
                     + TITLE.replace('Physics', 'Phy sics') + f'\nShaylah Mutschler\n{ABSTRACT_BODY}')
    evidence = ads._inspect_ads_document(data, paper)
    assert evidence.identity_match
    assert ads._classify_ads_availability(metadata(doctype='abstract'), [evidence]).availability == 'confirmed_abstract_only'


def test_legacy_records_default_unknown():
    paper = models.Paper(key='Legacy', title=TITLE)
    assert paper.ads_availability.availability == 'unknown'
    assert not paper.ads_availability.checked_at


def test_ads_audit_reports_separate_states_without_writes(lib, paper, monkeypatch, capsys):
    for state in ('confirmed_abstract_only', 'retrievable', 'blocked', 'unknown'):
        p, _ = lib.upsert({'title': f'{state} independent document', 'authors': [state], 'year': 2020,
                           'source': 'ads', 'paper_id': BIBCODE})
        p.ads_availability = models.ADSAvailability(bibcode=BIBCODE, availability=state,
                                                  publication_kind='article', reason='test_evidence')
    lib.save()
    before = lib.index_path.read_bytes()
    monkeypatch.setenv('PAPER_LIBRARY_PATH', str(lib.root))
    monkeypatch.setattr(requests, 'get', lambda *_, **__: pytest.fail('audit made request'))
    assert cli.main(['audit', '--ads-availability', '--json']) == 0
    result = json.loads(capsys.readouterr().out)['ads_availability']
    assert result['counts'] == {'confirmed_abstract_only': 1, 'retrievable': 1, 'blocked': 1, 'unknown': 2}
    assert all('publication_kind' in row and 'reason' in row for row in result['records'])
    assert lib.index_path.read_bytes() == before
    assert lib.get(paper.key).download_status == 'pending'


@responses.activate
def test_conflicting_stored_ads_identity_rejects_even_doi_lookup(paper):
    paper.paper_id = OTHER
    paper.doi = '10.1/exact'
    stub_metadata(doi=[paper.doi])
    assert ads._try_ads(paper) is None
    assert not responses.calls
    assert paper.ads_availability.reason == 'invalid_ads_identity'


@pytest.mark.parametrize('field, value, returned', [
    ('doi', '10.1/exact', {'doi': ['10.1/other']}),
    ('arxiv_id', '2401.0001v2', {'identifier': ['arXiv:2401.9999']}),
])
@responses.activate
def test_doi_and_arxiv_search_require_returned_identifier(paper, field, value, returned):
    paper.url = paper.paper_id = ''
    setattr(paper, field, value)
    stub_metadata(**returned)
    assert ads._try_ads(paper) is None
    assert paper.ads_availability.reason == 'metadata_identity_mismatch'
    assert len(responses.calls) == 1


@responses.activate
def test_api_transport_failure_does_not_retire(paper):
    responses.get(API, body=requests.Timeout('API failed'))
    assert ads._try_ads(paper) is None
    assert paper.ads_availability.availability == 'unknown'
    assert paper.ads_availability.reason == 'api_transport_error'


@responses.activate
def test_full_text_route_overrides_abstract_document(paper):
    stub_metadata(doctype='abstract', esources=['PUB_PDF', 'AUTHOR_PDF'])
    responses.get(GATEWAY, body=abstract_pdf())
    responses.get(f'https://ui.adsabs.harvard.edu/link_gateway/{BIBCODE}/AUTHOR_PDF', body=paper_pdf())
    assert ads._try_ads(paper) == paper_pdf()
    assert paper.ads_availability.availability == 'retrievable'
    assert len(paper.ads_availability.documents) == 2


@responses.activate
def test_content_length_mismatch_does_not_retire(paper):
    stub_metadata(doctype='abstract')
    responses.get(GATEWAY, body=abstract_pdf(), headers={'Content-Length': '1000000'})
    responses.get(LEGACY, status=404)
    assert ads._try_ads(paper) is None
    assert paper.ads_availability.availability == 'unknown'


def test_abstract_book_phrase_in_article_is_not_anthology(paper):
    data = _text_pdf(f'{TITLE}\nShaylah Mutschler\nThis report cites an abstract book.\n'
                     '1. Introduction\nWe study particles.\n2. Results\nDensity estimates follow.')
    evidence = ads._inspect_ads_document(data, paper)
    assert evidence.document_kind == 'full_text'


@responses.activate
def test_fresh_abstract_retirement_keeps_existing_extract(lib, paper):
    path = lib.md_path(paper.key)
    original = b'Existing extract to preserve.'
    path.write_bytes(original)
    paper.md_path = str(path.relative_to(lib.root))
    paper.download_status = 'ok'
    stub_metadata(doctype='abstract')
    responses.get(GATEWAY, body=abstract_pdf())
    responses.get(LEGACY, status=404)
    assert download.download_paper(paper, lib) is False
    assert path.read_bytes() == original
    assert paper.download_status == 'ok'
    assert paper.ads_availability.existing_assets == ['md']
    lib.save()
    assert Library(lib.root).get(paper.key).md_path == paper.md_path


def test_abstract_book_with_full_paper_body_does_not_retire(paper):
    import io
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    for text in ('Abstract book', f'{TITLE}\nShaylah Mutschler\n{ABSTRACT_BODY}',
                 '1. Introduction\nA full paper follows.\n2. Results\nMany experiments.'):
        writer.add_page(PdfReader(io.BytesIO(_text_pdf(text))).pages[0])
    out = io.BytesIO()
    writer.write(out)
    evidence = ads._inspect_ads_document(out.getvalue(), paper)
    assert ads._classify_ads_availability(metadata(), [evidence]).availability == 'unknown'


@pytest.mark.parametrize('status', [200, 206])
@responses.activate
def test_explicit_partial_response_is_incomplete(paper, status):
    stub_metadata(doctype='abstract')
    data = abstract_pdf()
    responses.get(GATEWAY, status=status, body=data,
                  headers={'Content-Range': f'bytes 0-{len(data)-1}/{len(data)+10000}'})
    responses.get(LEGACY, status=404)
    assert ads._try_ads(paper) is None
    assert paper.ads_availability.availability == 'unknown'
    assert paper.ads_availability.documents[0].outcome == 'incomplete'
    assert len(paper.ads_availability.documents) == 2


def _book_pdf(*entries):
    import io
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    for text in entries:
        writer.add_page(PdfReader(io.BytesIO(_text_pdf(text))).pages[0])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


@pytest.mark.parametrize('label', ['Contents\n', ''])
def test_anthology_listing_without_contribution_stays_unknown(paper, label):
    data = _book_pdf('Abstract book', f'{label}{TITLE}\nShaylah Mutschler\n59',
                     'Other contribution by another author')
    evidence = ads._inspect_ads_document(data, paper)
    assert ads._classify_ads_availability(metadata(), [evidence]).availability == 'unknown'


def test_anthology_skips_listing_and_finds_real_contribution(paper):
    body = ('We study how orbital tracking data constrain atmospheric density estimates. '
            'The measurements cover several objects and include changes in the space environment. '
            'A numerical model combines these observations to estimate density. '
            'The contribution describes the experiment and summarizes the resulting estimates.')
    data = _book_pdf('Abstract book', f'Contents\n{TITLE}\nShaylah Mutschler\n59',
                     f'Shaylah Mutschler\n{TITLE}\n{body}')
    evidence = ads._inspect_ads_document(data, paper)
    assert ads._classify_ads_availability(metadata(), [evidence]).availability == 'confirmed_abstract_only'
    assert evidence.target_page == 3


@pytest.mark.parametrize('label', ['', 'Contents (continued)\n'])
@pytest.mark.parametrize('author_reference', ['Shaylah Mutschler\n59', 'Shaylah Mutschler ... 59'])
def test_anthology_other_listings_do_not_supply_target_body(paper, label, author_reference):
    other_entries = '\n'.join(
        f'An unrelated study of solar particles and their atmospheric density effects {i}\n'
        f'Other Author ... {60 + i}' for i in range(5))
    data = _book_pdf('Abstract book', f'{label}{TITLE}\n{author_reference}\n{other_entries}')
    evidence = ads._inspect_ads_document(data, paper)
    assert ads._classify_ads_availability(metadata(), [evidence]).availability == 'unknown'


@pytest.mark.parametrize('citation', [f'Shaylah Mutschler, {TITLE}. 2019.',
                                    f'Shaylah Mutschler\n{TITLE}\n{ABSTRACT_BODY}'])
def test_meeting_citation_does_not_confirm_requested_contribution(paper, citation):
    data = _text_pdf('Geophysical Research Abstracts\nEGU General Assembly\n'
                     f'A different study of solar particles\nOther Author\n{ABSTRACT_BODY}\n'
                     f'References\n{citation}')
    evidence = ads._inspect_ads_document(data, paper)
    assert ads._classify_ads_availability(metadata(doctype='abstract'), [evidence]).availability == 'unknown'


def test_meeting_title_mentioned_in_prose_does_not_confirm(paper):
    data = _text_pdf('Geophysical Research Abstracts\nEGU General Assembly\n'
                     'A different study\nOther Author\n'
                     f'We compare with {TITLE} by Shaylah Mutschler.\n{ABSTRACT_BODY}')
    evidence = ads._inspect_ads_document(data, paper)
    assert ads._classify_ads_availability(metadata(doctype='abstract'), [evidence]).availability == 'unknown'


def test_meeting_heading_without_substantive_body_stays_unknown(paper):
    data = _text_pdf(f'Geophysical Research Abstracts\nEGU General Assembly\n'
                     f'{TITLE}\nShaylah Mutschler\nBrief listing only.')
    evidence = ads._inspect_ads_document(data, paper)
    assert ads._classify_ads_availability(metadata(doctype='abstract'), [evidence]).availability == 'unknown'


@pytest.mark.parametrize('punctuation', ['', '.'])
def test_anthology_numberless_titles_are_not_contribution_prose(paper, punctuation):
    listings = '\n'.join(
        f'An unrelated study of solar particles and their atmospheric density effects{punctuation}\n'
        'Other Author' for _ in range(5))
    data = _book_pdf('Abstract book', f'{TITLE}\nShaylah Mutschler\n{listings}')
    evidence = ads._inspect_ads_document(data, paper)
    assert ads._classify_ads_availability(metadata(), [evidence]).availability == 'unknown'


@responses.activate
def test_anthology_listing_is_not_returned_as_full_paper(paper):
    stub_metadata()
    responses.get(GATEWAY, body=_book_pdf('Abstract book', f'{TITLE}\nShaylah Mutschler\n59'))
    responses.get(LEGACY, status=404)
    assert ads._try_ads(paper) is None
    assert paper.ads_availability.availability == 'unknown'


@responses.activate
def test_rejected_other_domain_member_still_settles_ads_abstract(lib, paper, monkeypatch):
    stub_metadata(doctype='abstract')
    responses.get(GATEWAY, body=abstract_pdf())
    responses.get(LEGACY, status=404)
    monkeypatch.setattr(download, '_try_inspire', lambda _: _text_pdf('A different paper'))
    monkeypatch.setattr(download, '_verify_pdf_matches_metadata', lambda *_: (False, 'llm_mismatch: other work'))
    later = []
    monkeypatch.setattr(download, '_STRATEGIES', download._STRATEGIES + [
        ('later', lambda _: later.append('later')),
    ])
    monkeypatch.setattr(download, '_try_firecrawl_text_fallback', lambda *_: later.append('firecrawl'))
    assert download.download_paper(paper, lib) is False
    assert paper.ads_availability.availability == 'confirmed_abstract_only'
    assert paper.download_status == 'metadata_only'
    assert not later


@responses.activate
def test_fresh_abstract_preserves_firecrawl_asset_before_gate(lib, paper, monkeypatch):
    original = b'---\nsource: firecrawl\n---\nExisting text must be preserved.'
    path = lib.md_path(paper.key)
    path.write_bytes(original)
    paper.md_path = str(path.relative_to(lib.root))
    paper.download_status = 'ok'
    paper.download_source = 'firecrawl'
    stub_metadata(doctype='abstract')
    responses.get(GATEWAY, body=abstract_pdf())
    responses.get(LEGACY, status=404)
    monkeypatch.setattr(download, '_gate_firecrawl_md', lambda *_: pytest.fail('asset gate preceded ADS'))
    assert download.download_paper(paper, lib) is False
    assert path.read_bytes() == original
    assert paper.download_status == 'ok'
    assert paper.firecrawl_pdf_hunt_exhausted
    assert paper.ads_availability.availability == 'confirmed_abstract_only'
    assert paper.ads_availability.existing_assets == ['md']
    lib.save()
    assert Library(lib.root).get(paper.key).md_path == paper.md_path


@responses.activate
def test_firecrawl_preflight_pdf_reuses_source_attempt_and_verifies(lib, paper, monkeypatch):
    original = b'---\nsource: firecrawl\n---\nExisting full text.'
    lib.md_path(paper.key).write_bytes(original)
    paper.md_path = f'extracts/md/{paper.key}.md'
    stub_metadata()
    data = paper_pdf()
    responses.get(GATEWAY, body=data)
    gates, verified = [], []
    monkeypatch.setattr(download, '_gate_firecrawl_md', lambda *args: gates.append(args[2]) or True)
    monkeypatch.setattr(download, '_verify_pdf_matches_metadata', lambda payload, _: verified.append(payload) or (True, 'test match'))
    assert download.download_paper(paper, lib) is True
    assert len([c for c in responses.calls if 'search/query' in c.request.url]) == 1
    assert len(responses.calls) == 2
    assert len(gates) == 1 and verified == [data]
    assert lib.md_path(paper.key).read_bytes() == original
    assert lib.pdf_path(paper.key).read_bytes() == data
    assert paper.download_source == 'domain_aggregators'


@responses.activate
def test_firecrawl_preflight_unknown_is_reused(lib, paper, monkeypatch):
    lib.md_path(paper.key).write_text('---\nsource: firecrawl\n---\nExisting text.')
    responses.get(API, status=403)
    monkeypatch.setattr(download, '_gate_firecrawl_md', lambda *_: True)
    assert download.download_paper(paper, lib) is False
    assert len(responses.calls) == 1
    assert paper.ads_availability.availability == 'unknown'
    assert paper.ads_availability.reason == 'api_http_403'


@responses.activate
def test_firecrawl_preflight_refreshes_when_arxiv_identity_changes(lib, paper, monkeypatch):
    lib.md_path(paper.key).write_text('---\nsource: firecrawl\n---\nExisting text.')
    paper.arxiv_id = '2401.0001'
    responses.get(API, json={'response': {'docs': []}})
    stub_metadata()
    responses.get(GATEWAY, body=paper_pdf())
    monkeypatch.setattr(download, '_gate_firecrawl_md', lambda *_: True)
    monkeypatch.setattr(download, '_verify_pdf_matches_metadata', lambda *_: (True, 'test match'))

    def arxiv_miss(p):
        p.arxiv_id = ''
        return None

    monkeypatch.setattr(download, '_STRATEGIES', [('arxiv', arxiv_miss)] + download._STRATEGIES)
    assert download.download_paper(paper, lib) is True
    queries = [parse_qs(urlsplit(c.request.url).query)['q'][0]
               for c in responses.calls if 'search/query' in c.request.url]
    assert queries == ['identifier:"2401.0001"', f'bibcode:"{BIBCODE}"']
