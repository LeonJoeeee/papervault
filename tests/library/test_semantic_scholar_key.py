"""Optional credentials reach every S2 API call, never an external PDF origin."""
from types import SimpleNamespace

import pytest

from papervault.library import fetch, search
from papervault.library.download_sources import semantic_scholar as download
from papervault.library.models import Paper
from papervault.library.sources import semantic_scholar as source


@pytest.mark.parametrize('key', [None, '', '   ', '  fixture-key  '])
@pytest.mark.parametrize('path', ['search', 'references', 'citations', 'doi', 'arxiv', 'download'])
def test_optional_api_key_at_every_http_boundary(monkeypatch, key, path):
    if key is None:
        monkeypatch.delenv('SEMANTIC_SCHOLAR_API_KEY', raising=False)
    else:
        monkeypatch.setenv('SEMANTIC_SCHOLAR_API_KEY', key)
    api_calls, pdf_calls = [], []
    module = {'search': source, 'references': search, 'citations': search,
              'doi': fetch, 'arxiv': fetch, 'download': download}[path]
    retried = path in {'search', 'references', 'citations'}

    def get(transport, url, **kwargs):
        assert url.startswith('https://api.semanticscholar.org/')
        api_calls.append(kwargs)
        payload = {'data': [], 'title': 'Fixture paper',
                   'openAccessPdf': {'url': 'https://publisher.test/paper.pdf'}}
        return SimpleNamespace(status_code=429 if retried and len(api_calls) == 1 else 200,
                               ok=True, headers={}, raise_for_status=lambda: None,
                               json=lambda: payload)

    def pdf_get(url, **kwargs):
        assert url == 'https://publisher.test/paper.pdf'
        pdf_calls.append(kwargs)
        return SimpleNamespace(ok=True, content=b'%PDF-fixture')

    monkeypatch.setattr(module, 'polite_get', get)
    monkeypatch.setattr(search.time, 'sleep', lambda _: None)
    monkeypatch.setattr(fetch, 'search_arxiv', lambda *args, **kwargs: [])
    monkeypatch.setattr(download.requests, 'get', pdf_get)
    if path == 'search':
        assert source.search_semantic_scholar('term') == []
    elif path in {'references', 'citations'}:
        assert getattr(search, 'fetch_' + path)('paper-id') == []
    elif path == 'doi':
        assert fetch._fetch_ss_by_doi('10.1/test')['title'] == 'Fixture paper'
    elif path == 'arxiv':
        assert fetch.fetch_by_arxiv('2301.0002')['title'] == 'Fixture paper'
    else:
        assert download._try_semantic_scholar_oa(Paper(key='K', title='Title', doi='10.1/test')) == b'%PDF-fixture'
        assert len(pdf_calls) == 1
        assert 'x-api-key' not in pdf_calls[0]['headers']
    assert len(api_calls) == (2 if retried else 1)
    for kwargs in api_calls:
        headers = {'User-Agent': download.USER_AGENT} if path == 'download' else {}
        if key and key.strip():
            headers['x-api-key'] = 'fixture-key'
        if headers:
            assert kwargs['headers'] == headers
        else:
            assert 'headers' not in kwargs
