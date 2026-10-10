"""Current-only arXiv acquisition; all HTTP is controlled, no live vault."""

from datetime import datetime
import json
from types import SimpleNamespace

import pytest
import requests
import responses

from papervault.library import Library, download
from papervault.library.download_sources import arxiv
from papervault.library.models import Paper


PDF = b"%PDF-1.4\n%fixture"
BASE = "2603.20546"
ABS = "https://arxiv.org/abs/2603.20546"


def _page(base=BASE, latest=2, *, withdrawn=True, old_withdrawn=False):
    # Reduced layout from the read-only abstract controls on 2026-10-09.
    banner = '<span class="error">This paper has been withdrawn by Peter Catt</span>'
    older = (f'<strong><a href="/abs/{base}v1">[v1]</a></strong> '
             f'Mon, 1 Jan 2024 00:00:00 UTC (12 KB) {"(withdrawn)" if old_withdrawn else ""}<br>'
             if latest > 1 else '')
    return f'''<!doctype html><html><head>
      <link rel="canonical" href="https://arxiv.org/abs/{base}">
      <meta name="citation_arxiv_id" content="{base}">
      <meta name="citation_title" content="A sufficiently long scientific title">
      <meta name="citation_author" content="Catt, Peter">
      <meta property="og:url" content="https://arxiv.org/abs/{base}v{latest}">
      </head><body><div id="abs">{banner if withdrawn else ''}
      <td class="tablecell comments mathjax"><em>Results need further work.
      See <a href="https://arxiv.org/abs/2603.27074">arXiv:2603.27074</a>.</em></td>
      </div><div class="submission-history"><h2>Submission history</h2>
      {older}
      <strong>[v{latest}]</strong>Tue, 2 Jan 2024 00:00:00 UTC (1 KB)
      {'<em>(withdrawn)</em>' if withdrawn else ''}<br></div></body></html>'''


def _missing(base="1311.9999"):
    return f'''<html><body><main><div id="content">
      <h1>Article {base} not found</h1>
      <p>There is no record of an article with identifier '{base}'.
      You might instead try to <a href="/search">search for articles</a>.</p>
      </div></main></body></html>'''


def _paper(**fields):
    return Paper(key="Catt2026", title="A sufficiently long scientific title",
                 authors=["Peter Catt"], doi="10.1234/original", **fields)


def _events(lib):
    # These tests assert acquisition outcomes; span records have their own contract tests.
    return [event for line in lib.manifest_path.read_text().splitlines()
            if (event := json.loads(line))["event"] != "download_telemetry"]


@responses.activate
def test_pdf_404_does_not_establish_fabricated_identity():
    paper = _paper(arxiv_id=BASE + "v2")
    responses.get(ABS, status=404, body="generic error")
    responses.get("https://arxiv.org/pdf/2603.20546", status=404)
    assert arxiv._try_arxiv(paper) is None
    assert paper.arxiv_id == "2603.20546v2"
    assert paper.doi == "10.1234/original"


@responses.activate
def test_affirmative_missing_base_clears_only_arxiv_identity():
    paper = _paper(arxiv_id="1311.9999v7")
    responses.get("https://arxiv.org/abs/1311.9999", status=404, body=_missing())
    assert arxiv._try_arxiv(paper) is None
    assert paper.arxiv_id == ""
    assert paper.doi == "10.1234/original"
    assert [c.request.url for c in responses.calls] == ["https://arxiv.org/abs/1311.9999"]


@pytest.mark.parametrize("requested", [BASE, BASE + "v1", BASE + "v2", BASE + "v999"])
@responses.activate
def test_withdrawn_latest_preserves_input_and_never_fetches_historical(requested):
    paper = _paper(arxiv_id=requested)
    responses.get(ABS, body=_page())
    responses.get("https://arxiv.org/pdf/2603.20546v1", body=PDF)
    responses.get("https://arxiv.org/pdf/2603.20546", status=404)
    assert arxiv._try_arxiv(paper) is None
    assert paper.arxiv_id == requested
    assert paper.doi == "10.1234/original"
    evidence = paper.arxiv_withdrawal
    assert evidence.requested_id == requested
    assert evidence.latest_id == "2603.20546v2"
    assert evidence.reason == "Results need further work. See arXiv:2603.27074."
    assert "This paper has been withdrawn" in evidence.evidence
    assert "[v2]" in evidence.evidence and "(withdrawn)" in evidence.evidence
    assert evidence.evidence_url == ABS
    assert datetime.fromisoformat(evidence.observed_at).utcoffset().total_seconds() == 0
    assert [c.request.url for c in responses.calls] == [ABS]


@pytest.mark.parametrize("requested", ["0704.0001v1", "0704.0001v999"])
@responses.activate
def test_historical_or_missing_requested_version_uses_only_confirmed_current(requested):
    paper = _paper(arxiv_id=requested)
    responses.get("https://arxiv.org/abs/0704.0001", body=_page("0704.0001", withdrawn=False))
    responses.get("https://arxiv.org/pdf/0704.0001v2", body=PDF)
    responses.get("https://arxiv.org/pdf/0704.0001v1", body=b"%PDF-old")
    assert arxiv._try_arxiv(paper) == PDF
    assert paper.arxiv_id == requested
    assert paper.arxiv_withdrawal is None
    assert [c.request.url for c in responses.calls] == [
        "https://arxiv.org/abs/0704.0001", "https://arxiv.org/pdf/0704.0001v2",
    ]


@responses.activate
def test_real_active_record_with_missing_pdf_keeps_identity():
    paper = _paper(arxiv_id=BASE + "v999")
    responses.get(ABS, body=_page(withdrawn=False))
    responses.get("https://arxiv.org/pdf/2603.20546v2", status=404)
    assert arxiv._try_arxiv(paper) is None
    assert paper.arxiv_id == "2603.20546v999"
    assert paper.arxiv_withdrawal is None


@pytest.mark.parametrize("status, body", [
    (403, _missing(BASE)), (429, _missing(BASE)), (503, _missing(BASE)),
    (404, "<html><body>Not found</body></html>"), (404, _missing("1311.9999")),
    (200, _missing(BASE)), (200, "<html><body>challenge</body></html>"),
    (200, _page().replace('content="2603.20546"', 'content="0704.0001"')),
    (200, _page().replace("2603.20546v2", "2603.20546v1")),
    (200, _page().replace("</html>", "")),
    (200, _page().replace('<meta property="og:url"', '<meta property="unrecognized"')),
    (200, _page().replace('<em>(withdrawn)</em>', '')),
    (200, _page().replace('This paper has been withdrawn', 'Newer versions of this paper were withdrawn')),
    (200, '<html><body class>temporary challenge</body></html>'),
], ids=["403", "429", "503", "generic_404", "wrong_missing_id", "missing_200", "challenge",
        "wrong_identity", "wrong_version", "truncated", "missing_version", "missing_history_marker",
        "ambiguous_banner", "valueless_class"])
@responses.activate
def test_ambiguous_lookup_preserves_identity(status, body):
    responses.get(ABS, status=status, body=body)
    record = arxiv._lookup_arxiv_record(BASE + "v2")
    assert record.status == "unknown"
    paper = _paper(arxiv_id=BASE + "v2")
    responses.get("https://arxiv.org/pdf/2603.20546", status=404)
    assert arxiv._try_arxiv(paper) is None
    assert paper.arxiv_id == "2603.20546v2"
    assert paper.arxiv_withdrawal is None


@pytest.mark.parametrize("error", [requests.Timeout("timeout"), requests.ConnectionError("down"),
                                      requests.exceptions.ChunkedEncodingError("incomplete")])
@responses.activate
def test_transient_lookup_cannot_clear_identity(error):
    responses.get(ABS, body=error)
    assert arxiv._lookup_arxiv_record(BASE).status == "unknown"
    paper = _paper(arxiv_id=BASE)
    responses.get("https://arxiv.org/pdf/2603.20546", status=404)
    assert arxiv._try_arxiv(paper) is None
    assert paper.arxiv_id == BASE and paper.doi == "10.1234/original"


@pytest.mark.parametrize("headers, body", [
    ({"Content-Range": "bytes 0-100/90000"}, _page()),
    ({"Content-Length": "90000"}, _page()),
    ({}, _page() + "x" * (256 * 1024)),
], ids=["partial", "wrong_length", "oversized"])
@responses.activate
def test_partial_or_oversized_metadata_is_unknown(headers, body):
    responses.get(ABS, body=body, headers=headers)
    assert arxiv._lookup_arxiv_record(BASE).status == "unknown"


@responses.activate
def test_prior_withdrawal_does_not_mark_active_latest_withdrawn():
    responses.get(ABS, body=_page(withdrawn=False, old_withdrawn=True))
    record = arxiv._lookup_arxiv_record(BASE)
    assert record.status == "found"
    assert record.latest_id == "2603.20546v2" and not record.withdrawn


@responses.activate
def test_author_comment_keyword_alone_does_not_confirm_withdrawal():
    body = _page(withdrawn=False).replace("Results need further work.", "A previous draft was withdrawn.")
    responses.get(ABS, body=body)
    record = arxiv._lookup_arxiv_record(BASE)
    assert record.status == "found" and not record.withdrawn


@pytest.mark.parametrize("url", ["https://arxiv.org/abs/2603.20546v2",
                                   "http://arxiv.org/abs/2603.20546v2"])
@responses.activate
def test_canonical_url_qualifies_without_bulk_backfill(url):
    paper = _paper(url=url)
    responses.get(ABS, body=_page())
    assert download._download_skip_reason("arxiv", paper) is None
    assert arxiv._try_arxiv(paper) is None
    assert paper.arxiv_id == "2603.20546v2"
    assert paper.arxiv_withdrawal.requested_id == "2603.20546v2"
    assert download._download_skip_reason("arxiv_by_title", paper) == "already_has_arxiv_id"


@pytest.mark.parametrize("url", [
    "https://arxiv.org.evil.example/abs/2603.20546v2", "https://user@arxiv.org/abs/2603.20546",
    "https://arxiv.org:444/abs/2603.20546", "https://arxiv.org/abs/2603.20546v0",
    "https://arxiv.org/abs/2603.20546/extra", "https://arxiv.org/abs/2603.20546?x=1",
    "https://arxiv.org/abs/2603.20546#x", "https://arxiv.org/abs/2603.20546\n",
    "https://arxiv.org/abs/%32%36%30%33.20546", "https://arxiv.org\\evil/abs/2603.20546",
    "https://arxiv.org/abs/arXiv:2603.20546",
])
@responses.activate
def test_noncanonical_urls_are_skips(url):
    paper = _paper(url=url)
    assert download._download_skip_reason("arxiv", paper) == "missing_arxiv_id"
    assert arxiv._try_arxiv(paper) is None
    assert not responses.calls and paper.arxiv_id == ""


@pytest.mark.parametrize("abstract, status", [("", "failed"), ("Known abstract.", "metadata_only")])
@responses.activate
def test_withdrawal_stops_all_copies_and_round_trips(tmp_path, monkeypatch, abstract, status):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(arxiv_id=BASE + "v1", abstract=abstract).model_dump())
    called = []

    def historical_copy(p):
        called.append("copy")
        return PDF

    monkeypatch.setattr(download, "_STRATEGIES", [
        ("url_override", historical_copy), ("known_file_url", historical_copy),
        ("arxiv", download._try_arxiv), ("oa_aggregators", historical_copy),
        ("web_search", historical_copy),
    ])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: called.append("text"))
    responses.get(ABS, body=_page())
    assert download.download_paper(paper, lib) is False
    assert not called and not lib.has_pdf(paper.key)
    assert paper.download_status == status and paper.download_source == ""
    outcomes = [e for e in _events(lib) if e.get("source")]
    assert [(e["event"], e["source"]) for e in outcomes] == [("download_miss", "arxiv")]
    assert outcomes[0]["reason"] == "current_version_withdrawn"
    assert outcomes[0]["withdrawal"]["latest_id"] == "2603.20546v2"
    lib.save()
    loaded_lib = Library(tmp_path)
    loaded = loaded_lib.get(paper.key)
    assert loaded.arxiv_withdrawal.model_dump() == paper.arxiv_withdrawal.model_dump()
    assert loaded.arxiv_id == "2603.20546v1" and loaded.doi == "10.1234/original"
    responses.calls.reset()
    assert download.download_paper(loaded, loaded_lib) is False
    assert not responses.calls and not called
    assert [e["event"] for e in _events(lib) if e.get("source")] == ["download_miss", "download_skip"]


@responses.activate
def test_current_lookup_reused_at_arxiv_tier_and_counted_once(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(arxiv_id="0704.0001v1").model_dump())
    monkeypatch.setattr(download, "_STRATEGIES", [("arxiv", download._try_arxiv)])
    responses.get("https://arxiv.org/abs/0704.0001", body=_page("0704.0001", withdrawn=False))
    responses.get("https://arxiv.org/pdf/0704.0001v2", body=PDF)
    assert download.download_paper(paper, lib) is True
    assert lib.pdf_path(paper.key).read_bytes() == PDF
    assert paper.download_source == "arxiv"
    assert [c.request.url for c in responses.calls] == [
        "https://arxiv.org/abs/0704.0001", "https://arxiv.org/pdf/0704.0001v2",
    ]
    assert [e["event"] for e in _events(lib) if e.get("source")] == ["downloaded"]


@responses.activate
def test_missing_base_lookup_remains_miss_after_id_clear(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(arxiv_id="1311.9999").model_dump())
    monkeypatch.setattr(download, "_STRATEGIES", [("arxiv", download._try_arxiv)])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    responses.get("https://arxiv.org/abs/1311.9999", status=404, body=_missing())
    assert download.download_paper(paper, lib) is False
    assert paper.arxiv_id == "" and paper.doi == "10.1234/original"
    assert [e["event"] for e in _events(lib) if e.get("source")] == ["download_miss"]
    assert len(responses.calls) == 1


@responses.activate
def test_unknown_provenance_does_not_audit_existing_asset_pair(tmp_path):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(arxiv_id=BASE + "v2").model_dump())
    lib.pdf_path(paper.key).write_bytes(PDF)
    lib.md_path(paper.key).write_text("Existing text with unproven acquired version.")
    before = paper.model_dump()
    assert download.download_paper(paper, lib) is True
    assert paper.model_dump() == before
    assert lib.pdf_path(paper.key).read_bytes() == PDF
    assert lib.md_path(paper.key).read_text() == "Existing text with unproven acquired version."
    assert not responses.calls


@responses.activate
def test_withdrawal_precedes_historical_firecrawl_gate(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(arxiv_id=BASE).model_dump())
    paper.md_path = f"extracts/md/{paper.key}.md"
    paper.md_engine = "firecrawl"
    lib.md_path(paper.key).write_text("---\nsource: firecrawl\n---\nExisting asset.")
    before = lib.md_path(paper.key).read_bytes()
    responses.get(ABS, body=_page())
    assert download.download_paper(paper, lib) is False
    assert lib.md_path(paper.key).read_bytes() == before
    assert paper.md_engine == "firecrawl" and not paper.firecrawl_pdf_hunt_exhausted
    assert paper.arxiv_withdrawal.latest_id == "2603.20546v2"
    assert [c.request.url for c in responses.calls] == [ABS]


@responses.activate
def test_title_discovery_obeys_current_policy_without_poisoning_a_failed_candidate(monkeypatch):
    paper = _paper()
    monkeypatch.setattr("papervault.library.sources.arxiv.search_arxiv", lambda *_args, **_kw: [
        {"arxiv_id": BASE + "v1", "title": paper.title, "authors": ["Peter Catt"]},
    ])
    responses.get(ABS, body=_page())
    assert arxiv._try_arxiv_by_title(paper) is None
    assert paper.arxiv_id == "2603.20546v1"
    assert paper.arxiv_withdrawal.latest_id == "2603.20546v2"
    assert [c.request.url for c in responses.calls] == [ABS]


@responses.activate
def test_loose_title_candidate_withdrawal_is_not_attributed_to_original(monkeypatch):
    paper = _paper()
    monkeypatch.setattr("papervault.library.sources.arxiv.search_arxiv", lambda *_args, **_kw: [
        {"arxiv_id": BASE, "title": paper.title, "authors": ["Someone Else"]},
    ])
    responses.get(ABS, body=_page().replace("Catt, Peter", "Else, Someone"))
    assert arxiv._try_arxiv_by_title(paper) is None
    assert paper.arxiv_id == "" and paper.arxiv_withdrawal is None
    assert [c.request.url for c in responses.calls] == [ABS]


@pytest.mark.parametrize("source", ["known_file_url", "url_override"])
@pytest.mark.parametrize("has_id", [True, False])
@responses.activate
def test_explicit_historical_arxiv_url_cannot_bypass_current_resolution(tmp_path, monkeypatch, source, has_id):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(arxiv_id="0704.0001v1" if has_id else "").model_dump())
    historical = "https://arxiv.org/pdf/0704.0001v1.pdf"
    if source == "known_file_url":
        paper.url = historical
    else:
        (lib.root / "url_overrides.json").write_text(json.dumps({paper.doi: historical}))
    monkeypatch.setenv("PAPERVAULT_VAULT", str(lib.root))
    strategy = dict(download._STRATEGIES)[source]
    monkeypatch.setattr(download, "_STRATEGIES", [(source, strategy), ("arxiv", download._try_arxiv)])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    responses.get("https://arxiv.org/abs/0704.0001", body=_page("0704.0001", withdrawn=False))
    responses.get(historical, body=b"%PDF-historical")
    responses.get("https://arxiv.org/pdf/0704.0001v2", body=PDF)
    assert download.download_paper(paper, lib) is True
    assert lib.pdf_path(paper.key).read_bytes() == PDF
    assert paper.download_source == "arxiv"
    assert [c.request.url for c in responses.calls] == [
        "https://arxiv.org/abs/0704.0001", "https://arxiv.org/pdf/0704.0001v2",
    ]
    outcomes = [e for e in _events(lib) if e.get("source")]
    assert [(e["event"], e["source"]) for e in outcomes] == [("download_skip", source), ("downloaded", "arxiv")]
    assert outcomes[0]["reason"] == "current_arxiv_version_required"


@responses.activate
def test_pdf_redirect_cannot_select_a_historical_version():
    paper = _paper(arxiv_id="0704.0001v1")
    responses.get("https://arxiv.org/abs/0704.0001", body=_page("0704.0001", withdrawn=False))
    responses.get("https://arxiv.org/pdf/0704.0001v2", status=302,
                  headers={"Location": "https://arxiv.org/pdf/0704.0001v1"})
    responses.get("https://arxiv.org/pdf/0704.0001v1", body=PDF)
    assert arxiv._try_arxiv(paper) is None
    assert paper.arxiv_id == "0704.0001v1"
    assert [c.request.url for c in responses.calls] == [
        "https://arxiv.org/abs/0704.0001", "https://arxiv.org/pdf/0704.0001v2",
    ]


@responses.activate
def test_serialized_evidence_merge_retains_terminal_stop(tmp_path):
    paper = _paper(arxiv_id=BASE)
    responses.get(ABS, body=_page())
    assert arxiv._try_arxiv(paper) is None
    serialized = paper.model_dump()
    lib = Library(tmp_path)
    merged, _ = lib.upsert(_paper(arxiv_id=BASE).model_dump())
    lib.upsert(serialized)
    responses.calls.reset()
    assert download.download_paper(merged, lib) is False
    assert not responses.calls
    assert merged.arxiv_withdrawal.model_dump() == serialized["arxiv_withdrawal"]
    lib.save()
    assert Library(tmp_path).get(merged.key).arxiv_withdrawal.model_dump() == serialized["arxiv_withdrawal"]


@responses.activate
def test_by_title_never_overwrites_a_nonempty_unrecognized_id(monkeypatch):
    paper = _paper(arxiv_id="arXiv:0704.0001v1")
    monkeypatch.setattr("papervault.library.sources.arxiv.search_arxiv", lambda *_args, **_kw: [
        {"arxiv_id": "0704.0001", "title": paper.title},
    ])
    responses.get("https://arxiv.org/abs/0704.0001", status=503)
    responses.get("https://arxiv.org/pdf/0704.0001", body=PDF)
    assert arxiv._try_arxiv_by_title(paper) is None
    assert paper.arxiv_id == "arXiv:0704.0001v1"
    assert download._download_skip_reason("arxiv_by_title", paper) == "already_has_arxiv_id"
    assert not responses.calls


@responses.activate
def test_persisted_withdrawal_is_terminal_until_explicitly_cleared_even_if_version_changes():
    paper = _paper(arxiv_id=BASE)
    responses.get(ABS, body=_page())
    assert arxiv._try_arxiv(paper) is None
    paper.arxiv_id = BASE + "v3"
    responses.calls.reset()
    assert arxiv._try_arxiv(paper) is None
    assert not responses.calls
    assert paper.arxiv_withdrawal.latest_id == "2603.20546v2"


@responses.activate
def test_withdrawal_evidence_cannot_bind_a_different_base_identity():
    paper = _paper(arxiv_id=BASE)
    responses.get(ABS, body=_page())
    assert arxiv._try_arxiv(paper) is None
    paper.arxiv_id = "0704.0001"
    responses.get("https://arxiv.org/abs/0704.0001", body=_page("0704.0001", withdrawn=False))
    responses.get("https://arxiv.org/pdf/0704.0001v2", body=PDF)
    assert arxiv._try_arxiv(paper) == PDF
    assert paper.arxiv_id == "0704.0001" and paper.doi == "10.1234/original"


@responses.activate
def test_arxiv_override_uses_the_configured_override_location(tmp_path, monkeypatch):
    lib = Library(tmp_path / "library")
    configured = tmp_path / "configured-vault"
    configured.mkdir()
    (configured / "url_overrides.json").write_text(json.dumps({
        "10.1234/original": "https://arxiv.org/pdf/0704.0001v1.pdf",
    }))
    monkeypatch.setenv("PAPERVAULT_VAULT", str(configured))
    monkeypatch.setattr(download, "_STRATEGIES", [
        ("url_override", download._try_url_overrides), ("arxiv", download._try_arxiv),
    ])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    paper, _ = lib.upsert(_paper().model_dump())
    responses.get("https://arxiv.org/abs/0704.0001", body=_page("0704.0001", withdrawn=False))
    responses.get("https://arxiv.org/pdf/0704.0001v2", body=PDF)
    assert download.download_paper(paper, lib) is True
    assert lib.pdf_path(paper.key).read_bytes() == PDF
    assert paper.download_source == "arxiv"
    assert len(responses.calls) == 2


@responses.activate
def test_prefixed_identity_is_preserved_and_its_withdrawal_still_stops_acquisition():
    paper = _paper(arxiv_id="arXiv:2603.20546v1")
    responses.get(ABS, body=_page())
    assert arxiv._try_arxiv(paper) is None
    assert paper.arxiv_id == "arXiv:2603.20546v1"
    assert paper.arxiv_withdrawal.latest_id == "2603.20546v2"
    responses.calls.reset()
    assert arxiv._try_arxiv(paper) is None
    assert not responses.calls


@pytest.mark.parametrize("base", ["2603.27074", "hep-th/9901001"])
@responses.activate
def test_single_current_version_modern_and_legacy_ids(base):
    paper = _paper(arxiv_id=base)
    responses.get(f"https://arxiv.org/abs/{base}", body=_page(base, latest=1, withdrawn=False))
    responses.get(f"https://arxiv.org/pdf/{base}v1", body=PDF)
    assert arxiv._try_arxiv(paper) == PDF
    assert paper.arxiv_id == base and paper.arxiv_withdrawal is None
    assert [c.request.url for c in responses.calls] == [
        f"https://arxiv.org/abs/{base}", f"https://arxiv.org/pdf/{base}v1",
    ]


@responses.activate
def test_metadata_deadline_is_unknown_without_retry(monkeypatch):
    calls = []

    def monotonic():
        calls.append(True)
        return 0 if len(calls) <= 2 else 21

    monkeypatch.setattr(arxiv, "time", SimpleNamespace(monotonic=monotonic))
    responses.get(ABS, body=_page())
    record = arxiv._lookup_arxiv_record(BASE)
    assert record.status == "unknown" and record.reason == "metadata_time_limit"
    assert len(responses.calls) == 1


@pytest.mark.parametrize("has_id", [True, False])
@responses.activate
def test_missing_canonical_url_does_not_block_title_recovery(tmp_path, monkeypatch, has_id):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(arxiv_id="1311.9999" if has_id else "",
                                url="https://arxiv.org/abs/1311.9999").model_dump())
    monkeypatch.setattr(download, "_STRATEGIES", [
        ("arxiv", download._try_arxiv), ("arxiv_by_title", download._try_arxiv_by_title),
    ])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    monkeypatch.setattr("papervault.library.sources.arxiv.search_arxiv", lambda *_args, **_kw: [
        {"arxiv_id": "2603.27074", "title": paper.title, "authors": paper.authors},
    ])
    responses.get("https://arxiv.org/abs/1311.9999", status=404, body=_missing())
    responses.get("https://arxiv.org/abs/2603.27074", body=_page("2603.27074", latest=1, withdrawn=False))
    responses.get("https://arxiv.org/pdf/2603.27074v1", body=PDF)
    assert download.download_paper(paper, lib) is True
    assert paper.arxiv_id == "2603.27074" and paper.doi == "10.1234/original"
    assert paper.url == "https://arxiv.org/abs/1311.9999"
    assert paper.download_source == "arxiv_by_title"
    assert [(e["event"], e["source"]) for e in _events(lib) if e.get("source")] == [
        ("download_miss", "arxiv"), ("downloaded", "arxiv_by_title"),
    ]
    assert len(responses.calls) == 3
