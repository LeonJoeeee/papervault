"""Verified URL identity recovery, scoped acquisition, and zero-write inspection."""

import json
from datetime import datetime

import pytest
import requests
import responses

from papervault.library import Library, cli
from papervault.library.models import Paper

TITLE = "The transport of cosmic rays in the heliosheath"
DOI = "10.1234/correct"


def _paper(**fields):
    return Paper(key="Strauss2012", title=TITLE, authors=["R. D. Strauss"],
                 year=2012, **fields)


def _registry(doi=DOI, *, title=TITLE, family="Strauss", canonical=None, year=2012):
    responses.get(f"https://api.crossref.org/works/{doi}", json={"message": {
        "DOI": canonical or doi, "title": [title],
        "author": [{"given": "R. D.", "family": family}],
        "issued": {"date-parts": [[year]]}}})


def _arxiv_page(*, withdrawn=False):
    return f'''<html><head>
    <link rel="canonical" href="https://arxiv.org/abs/1201.12345">
    <meta name="citation_arxiv_id" content="1201.12345">
    <meta name="citation_title" content="{TITLE}">
    <meta name="citation_author" content="R. D. Strauss">
    <meta property="og:url" content="https://arxiv.org/abs/1201.12345v2">
    </head><body><div id="abs">
    {'<span class="error">This paper has been withdrawn by the author</span>' if withdrawn else ''}
    </div><div class="submission-history"><strong>[v1]</strong> Jan 2012<br>
    <strong>[v2]</strong> Feb 2012 {'(withdrawn)' if withdrawn else ''}<br>
    </div></body></html>'''


@pytest.mark.parametrize("url", [
    "https://doi.org/10.1234/correct",
    "http://dx.doi.org/10.1234%2Fcorrect?tracking=1",
    "https://onlinelibrary.wiley.com/doi/full/10.1234/correct",
    "https://link.springer.com/article/10.1234/correct",
    "https://iopscience.iop.org/article/10.1234/correct/pdf",
    "https://journals.aps.org/prl/abstract/10.1234/correct",
])
@responses.activate
def test_url_patterns_need_only_registry_verification(url):
    from papervault.library.services.identity_backfill import recover_identity
    _registry()
    paper = _paper(url=url)
    recovery = recover_identity(paper)
    assert recovery.doi == DOI
    assert recovery.route == "url_pattern"
    assert recovery.source_url == url
    assert paper.doi == ""  # discovery never mutates the input
    assert [c.request.url for c in responses.calls] == [f"https://api.crossref.org/works/{DOI}"]


@pytest.mark.parametrize("bad", ["wrong_title", "wrong_author", "wrong_doi", "unavailable"])
@responses.activate
def test_url_doi_rejects_wrong_paper_and_unverified_metadata(bad):
    from papervault.library.services.identity_backfill import recover_identity
    url = f"https://doi.org/{DOI}"
    if bad == "unavailable":
        responses.get(f"https://api.crossref.org/works/{DOI}", status=429)
    else:
        _registry(title="Galactic Cosmic Rays in the Dynamic Heliosphere" if bad == "wrong_title" else TITLE,
                  family="Potgieter" if bad == "wrong_author" else "Strauss",
                  canonical="10.1234/another" if bad == "wrong_doi" else DOI)
    responses.get(url, body="<html></html>")
    assert recover_identity(_paper(url=url)) is None


@responses.activate
def test_meta_route_reuses_landing_parser_and_verifies_registry():
    from papervault.library.services.identity_backfill import recover_identity
    url = "https://publisher.example/article/42"
    responses.get(url, body=f'<META CONTENT="https://doi.org/{DOI}" NAME="citation_doi">')
    _registry()
    result = recover_identity(_paper(url=url))
    assert result.doi == DOI
    assert result.route == "citation_meta"
    assert len([c for c in responses.calls if c.request.url == url]) == 1


@pytest.mark.parametrize("meta", [
    '<meta name="citation_doi" content="10.1234/a"><meta name="citation_doi" content="10.1234/b">',
    '<meta name="citation_doi" content="garbage">',
])
@responses.activate
def test_ambiguous_or_malformed_meta_abstains(meta):
    from papervault.library.services.identity_backfill import recover_identity
    url = "https://publisher.example/item"
    responses.get(url, body=meta)
    assert recover_identity(_paper(url=url)) is None
    assert len(responses.calls) == 1


@pytest.mark.parametrize("route", ["url_pattern", "citation_meta"])
@responses.activate
def test_arxiv_recovery_uses_current_identity_page(route):
    from papervault.library.services.identity_backfill import recover_identity
    url = "https://arxiv.org/pdf/1201.12345v1.pdf"
    if route == "citation_meta":
        url = "https://publisher.example/item"
        responses.get(url, body='<meta name="citation_arxiv" content="arXiv:1201.12345v1">')
    responses.get("https://arxiv.org/abs/1201.12345", body=_arxiv_page())
    result = recover_identity(_paper(url=url))
    assert result.arxiv_id == "1201.12345"
    assert result.route == route
    assert all("/pdf/" not in c.request.url for c in responses.calls)


@pytest.mark.parametrize("source", ["ads", "core", "openalex", "inspire"])
@responses.activate
def test_exact_source_metadata_recovers_doi(source, monkeypatch):
    from papervault.library.download_sources import core
    from papervault.library.services.identity_backfill import recover_identity
    monkeypatch.setenv("ADS_API_TOKEN", "test-ads")
    monkeypatch.setenv("CORE_API_KEY", "test-core")
    monkeypatch.setattr(core, "_api_next_at", 0)
    monkeypatch.setattr(core, "_api_cooldown_until", 0)
    urls = {"ads": "https://ui.adsabs.harvard.edu/abs/2012ApJ...750....1S/abstract",
            "core": "https://core.ac.uk/outputs/123",
            "openalex": "https://openalex.org/W123",
            "inspire": "https://inspirehep.net/literature/123"}
    url = urls[source]
    responses.get(url, body="<html></html>")
    if source == "ads":
        responses.get("https://api.adsabs.harvard.edu/v1/search/query", json={"response": {"docs": [{
            "bibcode": "2012ApJ...750....1S", "title": [TITLE], "author": ["Strauss, R. D."],
            "doi": [DOI], "identifier": [], "doctype": "article", "pub": "ApJ", "esources": []}]}})
    elif source == "core":
        responses.get("https://api.core.ac.uk/v3/outputs/123", json={
            "id": 123, "title": TITLE, "authors": ["R. D. Strauss"], "doi": DOI})
    elif source == "openalex":
        responses.get("https://api.openalex.org/works/W123", json={
            "id": "https://openalex.org/W123", "title": TITLE, "doi": "https://doi.org/" + DOI,
            "publication_year": 2012, "authorships": [{"author": {"display_name": "R. D. Strauss"}}]})
    else:
        responses.get("https://inspirehep.net/api/literature/123", json={"id": "123", "metadata": {
            "titles": [{"title": TITLE}], "authors": [{"full_name": "Strauss, R. D."}],
            "publication_info": [{"year": 2012}], "dois": [{"value": DOI}]}})
    _registry()
    result = recover_identity(_paper(url=url, source=source))
    assert result.doi == DOI
    assert result.route == source + "_api"


@responses.activate
def test_source_wrong_title_is_rejected_before_claiming_its_doi():
    from papervault.library.services.identity_backfill import recover_identity
    url = "https://openalex.org/W123"
    responses.get(url, body="")
    responses.get("https://api.openalex.org/works/W123", json={
        "id": url, "title": "Galactic Cosmic Rays in the Dynamic Heliosphere", "doi": DOI})
    assert recover_identity(_paper(url=url)) is None


@responses.activate
def test_landing_429_does_not_retry_alternate_user_agent():
    from papervault.library.download_sources.publisher import _try_citation_pdf_url
    responses.get(f"https://doi.org/{DOI}", status=429, headers={"Retry-After": "60"})
    assert _try_citation_pdf_url(_paper(doi=DOI)) is None
    assert len(responses.calls) == 1


def _recovery():
    from papervault.library.models import IdentityRecovery
    return IdentityRecovery(doi=DOI, route="url_pattern", source_url=f"https://doi.org/{DOI}",
                            verified_title=TITLE, observed_at="2026-10-09T15:00:00+00:00")


def test_provenance_roundtrip_and_identifier_index(tmp_path):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=f"https://doi.org/{DOI}").model_dump())
    assert lib.set_recovered_identity(paper.key, _recovery()) == "set"
    lib.save(force=True)
    reloaded = Library(tmp_path)
    found = reloaded.find(doi=DOI)
    assert found.key == paper.key
    assert found.identity_recovery.route == "url_pattern"
    assert found.identity_recovery.source_url == f"https://doi.org/{DOI}"
    assert found.identity_recovery.verified_title == TITLE
    assert datetime.fromisoformat(found.identity_recovery.observed_at).utcoffset().total_seconds() == 0


@pytest.mark.parametrize("old", [{"doi": "10.9999/original"}, {"arxiv_id": "1201.54321"}])
def test_no_overwrite_of_either_existing_identifier(tmp_path, old):
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(**old).model_dump())
    assert lib.set_recovered_identity(paper.key, _recovery()) == "has_identifier"
    assert paper.identity_recovery is None
    for name, value in old.items():
        assert getattr(paper, name) == value


def test_collision_does_not_merge_purge_or_acquire_another_row(tmp_path):
    lib = Library(tmp_path)
    stub, _ = lib.upsert(_paper().model_dump())
    holder, _ = lib.upsert({"title": "An independently stored scientific paper", "authors": ["Other"],
                           "doi": DOI})
    assert lib.set_recovered_identity(stub.key, _recovery()) == "collision"
    assert lib.get(stub.key) is stub
    assert lib.find(doi=DOI) is holder
    assert stub.doi == ""


@responses.activate
def test_capped_pass_acquires_only_newly_recovered_records_and_records_source(tmp_path, monkeypatch):
    from papervault.library import download
    from papervault.library.services.identity_backfill import backfill_identities
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=f"https://doi.org/{DOI}", download_status="metadata_only").model_dump())
    other, _ = lib.upsert({"title": "An unrelated terminal scientific record", "authors": ["Other"],
                          "url": "https://example.org/unrelated", "download_status": "failed"})
    _registry()
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", lambda p: b"%PDF-fixture")])
    result = backfill_identities(lib, cap=1, keys=[paper.key, other.key], acquire=True)
    assert result["scanned"] == 1
    assert result["recovered_by_route"] == {"url_pattern": 1}
    assert result["acquisition"]["attempted"] == 1
    assert result["acquisition"]["by_source"] == {"fixture": 1}
    assert result["acquisition"]["outcomes"] == {"pdf": 1}
    assert lib.has_pdf(paper.key)
    assert other.download_status == "failed" and other.doi == ""
    events = [json.loads(line) for line in lib.manifest_path.read_text().splitlines()]
    assert any(e["event"] == "identity_backfill_pass" and e["cap"] == 1 for e in events)
    assert Library(tmp_path).get(paper.key).identity_recovery is not None


@responses.activate
def test_recovered_withdrawal_remains_terminal_without_substitution(tmp_path, monkeypatch):
    from papervault.library.services.identity_backfill import backfill_identities
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url="https://arxiv.org/abs/1201.12345v1",
                               abstract="A citable abstract.", download_status="failed").model_dump())
    responses.get("https://arxiv.org/abs/1201.12345", body=_arxiv_page(withdrawn=True))
    result = backfill_identities(lib, cap=1, acquire=True)
    assert paper.arxiv_id == "1201.12345"
    assert paper.arxiv_withdrawal.latest_id == "1201.12345v2"
    assert paper.download_status == "metadata_only"
    assert result["acquisition"]["outcomes"] == {"withdrawn": 1}
    assert result["acquisition"]["by_source"] == {}
    assert not lib.has_pdf(paper.key)
    assert all(c.request.url == "https://arxiv.org/abs/1201.12345" for c in responses.calls)


@responses.activate
def test_inspection_reads_index_without_creating_library_files(tmp_path, monkeypatch, capsys):
    root = tmp_path / "readonly"
    root.mkdir()
    index = root / "index.json"
    index.write_text(json.dumps({"papers": {"Strauss2012": _paper(url=f"https://doi.org/{DOI}").model_dump()}}))
    before = index.read_bytes()
    monkeypatch.setenv("PAPERVAULT_VAULT", str(root))
    _registry()
    assert cli.main(["identity-backfill", "--cap", "1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["recovered_by_route"] == {"url_pattern": 1}
    assert index.read_bytes() == before
    assert sorted(p.name for p in root.iterdir()) == ["index.json"]


@responses.activate
def test_inspection_uses_store_snapshot_during_index_rotation(tmp_path, monkeypatch, capsys):
    backup = tmp_path / "index.json.bak"
    backup.write_text(json.dumps({"papers": {"Strauss2012": _paper(url=f"https://doi.org/{DOI}").model_dump()}}))
    before = backup.read_bytes()
    monkeypatch.setenv("PAPERVAULT_VAULT", str(tmp_path))
    _registry()
    assert cli.main(["identity-backfill", "--cap", "1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["recovered"] == 1
    assert backup.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.json.bak"]


@pytest.mark.parametrize("args", [[], ["--cap", "0"], ["--cap", "51"], ["--cap", "1", "--acquire"]])
def test_cli_requires_scoped_cap_and_apply_for_acquisition(args, tmp_path, monkeypatch):
    root = tmp_path / "absent"
    monkeypatch.setenv("PAPERVAULT_VAULT", str(root))
    with pytest.raises(SystemExit) as exc:
        cli.main(["identity-backfill", *args])
    assert exc.value.code == 2
    assert not root.exists()


@responses.activate
def test_redirect_target_doi_can_recover_without_citation_meta():
    from papervault.library.services.identity_backfill import recover_identity
    url = "https://publisher.example/item"
    responses.get(url, status=302, headers={"Location": f"https://doi.org/{DOI}"})
    responses.get(f"https://doi.org/{DOI}", body="<html></html>")
    _registry()
    result = recover_identity(_paper(url=url))
    assert result.doi == DOI
    assert result.route == "url_redirect"


@responses.activate
def test_core_work_url_uses_the_work_namespace(monkeypatch):
    from papervault.library.download_sources import core
    from papervault.library.services.identity_backfill import recover_identity
    monkeypatch.setenv("CORE_API_KEY", "test-core")
    monkeypatch.setattr(core, "_api_next_at", 0)
    monkeypatch.setattr(core, "_api_cooldown_until", 0)
    url = "https://core.ac.uk/works/123"
    responses.get(url, body="")
    responses.get("https://api.core.ac.uk/v3/works/123", json={
        "id": 123, "title": TITLE, "authors": ["R. D. Strauss"], "doi": DOI})
    _registry()
    result = recover_identity(_paper(url=url))
    assert result.doi == DOI and result.route == "core_api"


@responses.activate
def test_acquisition_start_cap_and_keys_survive_an_interrupted_pass(tmp_path, monkeypatch):
    from papervault.library import download
    from papervault.library.services.identity_backfill import backfill_identities
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=f"https://doi.org/{DOI}").model_dump())
    _registry()
    def interrupted(*args):
        raise RuntimeError("interrupted")
    monkeypatch.setattr(download, "download_paper", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        backfill_identities(lib, cap=1, acquire=True)
    events = [json.loads(line) for line in lib.manifest_path.read_text().splitlines()]
    start = next(e for e in events if e["event"] == "identity_backfill_start")
    assert start["cap"] == 1 and start["keys"] == [paper.key] and start["acquire"] is True


@responses.activate
def test_group_winner_yield_names_the_actual_source(tmp_path, monkeypatch):
    from papervault.library import download
    from papervault.library.services.identity_backfill import backfill_identities
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=f"https://doi.org/{DOI}").model_dump())
    _registry()
    def group(p):
        return download._try_concurrent_first_hit(p, [("openalex", lambda _: b"%PDF-fixture")])
    monkeypatch.setattr(download, "_STRATEGIES", [("oa_aggregators", group)])
    result = backfill_identities(lib, cap=1, acquire=True)
    assert result["acquisition"]["by_source"] == {"openalex": 1}
    assert paper.download_source == "oa_aggregators"  # preserve existing tier labels


@responses.activate
def test_landing_inspection_does_not_read_a_pdf_body(monkeypatch):
    from papervault.library.download_sources import publisher
    from papervault.library.services.identity_backfill import recover_identity
    def body_read(*args, **kwargs):
        pytest.fail("advertised PDF body must stay unread")
    monkeypatch.setattr(publisher, "_read_body", body_read)
    responses.get("https://publisher.example/file.pdf", body=b"%PDF-" + b"x" * 300000,
                  content_type="application/pdf")
    assert recover_identity(_paper(url="https://publisher.example/file.pdf")) is None


@responses.activate
def test_landing_redirect_bodies_are_never_buffered(monkeypatch):
    from papervault.library.services.identity_backfill import recover_identity
    original = requests.Response.content.fget
    def content(response):
        assert response.status_code != 302, "redirect body was buffered"
        return original(response)
    monkeypatch.setattr(requests.Response, "content", property(content))
    responses.get("https://publisher.example/item", status=302, body=b"x" * 1048576,
                  headers={"Location": "/article"})
    responses.get("https://publisher.example/article", body=f'<meta name="citation_doi" content="{DOI}">')
    _registry()
    assert recover_identity(_paper(url="https://publisher.example/item")).doi == DOI


@responses.activate
def test_source_api_rate_limit_abstains_without_search_or_retry(monkeypatch):
    from papervault.library.download_sources import core
    from papervault.library.services.identity_backfill import recover_identity
    monkeypatch.setenv("CORE_API_KEY", "test-core")
    monkeypatch.setattr(core, "_api_next_at", 0)
    monkeypatch.setattr(core, "_api_cooldown_until", 0)
    responses.get("https://core.ac.uk/outputs/123", body="")
    responses.get("https://api.core.ac.uk/v3/outputs/123", status=429, headers={"Retry-After": "60"})
    assert recover_identity(_paper(url="https://core.ac.uk/outputs/123")) is None
    assert len(responses.calls) == 2


@pytest.mark.parametrize("namespace", ["outputs", "works"])
@responses.activate
def test_core_rejects_metadata_from_a_different_locator(namespace, monkeypatch):
    from papervault.library.download_sources import core
    from papervault.library.services.identity_backfill import recover_identity
    monkeypatch.setenv("CORE_API_KEY", "test-core")
    monkeypatch.setattr(core, "_api_next_at", 0)
    monkeypatch.setattr(core, "_api_cooldown_until", 0)
    url = f"https://core.ac.uk/{namespace}/123"
    responses.get(url, body="")
    responses.get(f"https://api.core.ac.uk/v3/{namespace}/123", json={
        "id": 456, "title": TITLE, "authors": ["R. D. Strauss"], "doi": DOI})
    _registry()
    assert recover_identity(_paper(url=url)) is None


@responses.activate
def test_unavailable_doi_falls_through_to_independently_verified_arxiv():
    from papervault.library.services.identity_backfill import recover_identity
    responses.get(f"https://api.crossref.org/works/{DOI}", status=404)
    responses.get(f"https://doi.org/{DOI}", body=f'<meta name="citation_doi" content="{DOI}">'
                  '<meta name="citation_arxiv" content="1201.12345">')
    responses.get("https://arxiv.org/abs/1201.12345", body=_arxiv_page())
    result = recover_identity(_paper(url=f"https://doi.org/{DOI}"))
    assert result.arxiv_id == "1201.12345" and result.doi == ""
    assert len([c for c in responses.calls if "api.crossref.org" in c.request.url]) == 1


@responses.activate
def test_inspection_ignores_existing_identifiers_without_fetch():
    from papervault.library.services.identity_backfill import inspect_identities
    result = inspect_identities([_paper(doi=DOI, url=f"https://doi.org/{DOI}")], cap=1)
    assert result["scanned"] == 0 and result["recovered"] == 0
    assert len(responses.calls) == 0


@responses.activate
def test_failed_acquisition_is_counted_without_success_source(tmp_path, monkeypatch):
    from papervault.library import download
    from papervault.library.services.identity_backfill import backfill_identities
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=f"https://doi.org/{DOI}").model_dump())
    _registry()
    monkeypatch.setattr(download, "_STRATEGIES", [])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    result = backfill_identities(lib, cap=1, acquire=True)
    assert result["acquisition"]["outcomes"] == {"miss": 1}
    assert result["acquisition"]["by_source"] == {}
    assert paper.download_status == "failed"


@responses.activate
def test_text_yield_ignores_a_stale_aggregator_member(tmp_path, monkeypatch):
    from papervault.library import download
    from papervault.library.services.identity_backfill import backfill_identities
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=f"https://doi.org/{DOI}",
                               download_source="oa_aggregators", download_source_member="core").model_dump())
    _registry()
    monkeypatch.setattr(download, "_STRATEGIES", [])
    def text_fallback(p, library):
        library.md_path(p.key).write_text("Complete requested article")
        p.download_status, p.download_source = "ok", "firecrawl"
        return True
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", text_fallback)
    result = backfill_identities(lib, cap=1, acquire=True)
    assert result["acquisition"]["outcomes"] == {"text": 1}
    assert result["acquisition"]["by_source"] == {"firecrawl": 1}


def test_apply_refuses_a_running_co_writer_before_loading_vault(tmp_path, monkeypatch, capsys):
    from papervault import ops_guards
    root = tmp_path / "absent"
    monkeypatch.setenv("PAPERVAULT_VAULT", str(root))
    monkeypatch.setattr(ops_guards, "active_service_units", lambda *a, **kw: ["papervault.service"])
    assert cli.main(["identity-backfill", "--cap", "1", "--apply"]) == 2
    assert "overwrite" in capsys.readouterr().err
    assert not root.exists()


@pytest.mark.parametrize("flag", [None, "0"])
@responses.activate
def test_identity_v2_off_preserves_legacy_recovery_snapshot(flag, monkeypatch):
    from papervault.library.services.identity_backfill import inspect_identities
    if flag is None:
        monkeypatch.delenv("PAPERVAULT_IDENTITY_V2", raising=False)
    else:
        monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", flag)
    _registry()
    result = inspect_identities([_paper(url=f"https://doi.org/{DOI}")], cap=1)
    record = result["records"][0]
    recovery = record["recovery"]
    assert isinstance(recovery.pop("observed_at"), str)
    assert result == {"dry_run": True, "cap": 1, "scanned": 1, "recovered": 1,
        "recovered_by_route": {"url_pattern": 1}, "records": [{"key": "Strauss2012",
            "outcome": "verified", "recovery": {"doi": DOI, "arxiv_id": "",
                "route": "url_pattern", "source_url": f"https://doi.org/{DOI}", "verified_title": TITLE}}],
        "acquisition": {"attempted": 0, "by_source": {}, "outcomes": {}}}


@pytest.mark.parametrize("failure,outcome,reason", [
    ("title", "contradicted", "title_mismatch"),
    ("author", "contradicted", "author_mismatch"),
    ("doi", "contradicted", "doi_mismatch"),
    ("unavailable", "unresolved", "insufficient_evidence"),
])
@responses.activate
def test_v2_recovery_preserves_correspondence_failures_for_application(failure, outcome, reason, monkeypatch):
    from papervault.library.services.identity_backfill import inspect_identities
    monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", "1")
    url = f"https://doi.org/{DOI}"
    if failure == "unavailable":
        responses.get(f"https://api.crossref.org/works/{DOI}", status=429)
    else:
        _registry(title="Galactic Cosmic Rays in the Dynamic Heliosphere" if failure == "title" else TITLE,
                  family="Potgieter" if failure == "author" else "Strauss",
                  canonical="10.1234/other" if failure == "doi" else DOI)
    responses.get(url, body="<html></html>")
    result = inspect_identities([_paper(url=url)], cap=1)
    assert result["recovered"] == 0
    assert result["records"][0]["outcome"] == outcome
    decision = result["records"][0]["recovery"]["correspondence"]
    assert decision["outcome"] == outcome and decision["reason"] == reason


def _report(key, recovery):
    return {"dry_run": True, "records": [{"key": key, "outcome": "verified",
            "recovery": recovery.model_dump()}], "recovered": 1,
            "acquisition": {"attempted": 0, "by_source": {}, "outcomes": {}}}


@pytest.mark.parametrize("case,outcome", [("matched", "set"), ("missing", "unresolved"),
                                             ("contradicted", "contradicted")])
def test_v2_adoption_rechecks_evidence_and_only_resolved_can_set(tmp_path, monkeypatch, case, outcome):
    from papervault.library import download
    from papervault.library.identity_v2 import CorrespondenceDecision, Outcome, Reason
    from papervault.library.services.identity_backfill import apply_identity_report, eligible
    monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", "1")
    monkeypatch.setattr(download, "download_paper", lambda *_: True)
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url="https://publisher.example/item",
                               download_status="metadata_only").model_dump())
    before = paper.model_dump()
    recovery = _recovery().model_copy(update={
        "source_url": "https://doi.org/10.1234/wrong" if case == "contradicted" else paper.url,
        "observed_metadata": {"title": TITLE, "doi": DOI, "year": 2012,
                              "authors": [] if case == "missing" else ["R. D. Strauss"]},
        # A stale/precomputed verdict cannot smuggle an adoption through P3.
        "correspondence": CorrespondenceDecision(Outcome.RESOLVED, Reason.DOI_EQUIVALENT)})
    result = apply_identity_report(lib, _report(paper.key, recovery), acquire=case != "matched")
    assert result["records"][0]["outcome"] == outcome
    assert result["acquisition"]["attempted"] == 0
    if case != "matched":
        assert paper.doi == "" and paper.identity_recovery is None
        assert paper.download_status == before["download_status"]
        assert paper.terminal_state == before.get("terminal_state")
        assert paper.resolve_attempted_at is None and eligible(paper)
        assert lib.get(paper.key) is paper
    else:
        assert paper.doi == DOI
        assert lib.find(doi=DOI) is paper


def test_rejected_route_persists_and_second_route_can_still_adopt(tmp_path, monkeypatch):
    from papervault.library import download
    from papervault.library.services.identity_backfill import apply_identity_report
    monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", "1")
    monkeypatch.setattr(download, "download_paper", lambda *_: True)
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url="https://publisher.example/item").model_dump())
    recovery = _recovery().model_copy(update={"observed_metadata": {
        "title": TITLE, "authors": ["Strauss"], "year": 2012, "doi": DOI},
        "source_url": "https://doi.org/10.1234/wrong"})
    result = apply_identity_report(lib, _report(paper.key, recovery), acquire=True)
    assert result["records"][0]["outcome"] == "contradicted"
    reloaded = Library(tmp_path)
    retained = reloaded.get(paper.key)
    assert retained.download_status == "pending" and retained.doi == ""
    assert retained.identity_route_rejections[0].reason.value == "doi_mismatch"
    assert retained.identity_route_rejections[0].kind.value == "route_rejected"
    second = recovery.model_copy(update={"source_url": f"https://doi.org/{DOI}"})
    result = apply_identity_report(reloaded, _report(paper.key, second))
    assert result["records"][0]["outcome"] == "set"
    assert retained.doi == DOI and retained.download_status == "pending"
    assert len(retained.identity_route_rejections) == 1


@responses.activate
def test_v2_recovery_keeps_rejected_route_while_trying_another_identity(monkeypatch):
    from papervault.library.services.identity_backfill import recover_identity
    monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", "1")
    _registry(title="Galactic Cosmic Rays in the Dynamic Heliosphere")
    responses.get(f"https://doi.org/{DOI}", body='<meta name="citation_arxiv" content="1201.12345">')
    responses.get("https://arxiv.org/abs/1201.12345", body=_arxiv_page())
    recovery = recover_identity(_paper(url=f"https://doi.org/{DOI}"))
    assert recovery.arxiv_id == "1201.12345" and recovery.doi == ""
    assert recovery.route_rejections[0].reason.value == "title_mismatch"


def test_old_format_models_omit_new_evidence_fields():
    recovery = _recovery()
    assert set(recovery.model_dump()) == {"doi", "arxiv_id", "route", "source_url",
                                          "verified_title", "observed_at"}
    assert "identity_route_rejections" not in _paper().model_dump()
    from papervault.library.models import IdentityRecovery
    assert IdentityRecovery.model_validate_json(recovery.model_dump_json()) == recovery


@responses.activate
def test_v2_redirect_keeps_valid_final_route_after_rejecting_original(tmp_path, monkeypatch):
    from papervault.library.services.identity_backfill import backfill_identities
    monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", "1")
    old = "https://doi.org/10.1234/wrong"
    final = f"https://doi.org/{DOI}"
    _registry("10.1234/wrong", title="Galactic Cosmic Rays in the Dynamic Heliosphere")
    _registry()
    responses.get(old, status=302, headers={"Location": final})
    responses.get(final, body="<html></html>")
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=old).model_dump())
    result = backfill_identities(lib, cap=1)
    assert result["recovered"] == 1 and paper.doi == DOI
    assert paper.identity_recovery.source_url == final
    assert paper.identity_route_rejections[0].url == old


@responses.activate
def test_v2_opaque_redirect_cannot_hide_final_url_contradiction(tmp_path, monkeypatch):
    from papervault.library.services.identity_backfill import backfill_identities
    monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", "1")
    old = "https://publisher.example/item"
    final = "https://doi.org/10.1234/wrong"
    responses.get(old, status=302, headers={"Location": final})
    responses.get(final, body=f'<meta name="citation_doi" content="{DOI}">')
    _registry("10.1234/wrong", title="Galactic Cosmic Rays in the Dynamic Heliosphere")
    _registry()
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=old).model_dump())
    result = backfill_identities(lib, cap=1)
    assert result["recovered"] == 0 and paper.doi == ""
    assert paper.download_status == "pending"
    assert any(r.url == final and r.reason.value == "doi_mismatch"
               for r in paper.identity_route_rejections)


@pytest.mark.parametrize("title,outcome", [(TITLE, "set"),
    ("Galactic Cosmic Rays in the Dynamic Heliosphere", "contradicted")])
@responses.activate
def test_v2_checks_available_landing_bibliography(tmp_path, monkeypatch, title, outcome):
    from papervault.library.services.identity_backfill import backfill_identities
    monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", "1")
    url = "https://publisher.example/item"
    responses.get(url, body=f'<meta name="citation_doi" content="{DOI}">'
                  f'<meta name="citation_title" content="{title}">')
    _registry()
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=url).model_dump())
    result = backfill_identities(lib, cap=1)
    assert result["records"][0]["outcome"] == outcome
    assert paper.doi == (DOI if outcome == "set" else "")
    if outcome == "contradicted":
        assert paper.identity_route_rejections[0].reason.value == "title_mismatch"


@pytest.mark.parametrize("year", ["2012", "not-a-year", True, {"value": 2012}])
@responses.activate
def test_v2_malformed_year_stays_unresolved_through_apply(tmp_path, monkeypatch, year):
    from papervault.library.services.identity_backfill import backfill_identities
    monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", "1")
    _registry(year=year)
    responses.get(f"https://doi.org/{DOI}", body="<html></html>")
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=f"https://doi.org/{DOI}").model_dump())
    result = backfill_identities(lib, cap=1)
    assert result["records"][0]["outcome"] == "unresolved"
    assert paper.doi == "" and paper.resolve_attempted_at is None


@responses.activate
def test_v2_doi_only_deferral_still_records_rejected_routes(tmp_path, monkeypatch):
    from papervault.library.services.identity_backfill import apply_identity_report, inspect_identities
    monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", "1")
    _registry(title="Galactic Cosmic Rays in the Dynamic Heliosphere")
    responses.get(f"https://doi.org/{DOI}", body='<meta name="citation_arxiv" content="1201.12345">')
    responses.get("https://arxiv.org/abs/1201.12345", body=_arxiv_page())
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=f"https://doi.org/{DOI}").model_dump())
    result = apply_identity_report(lib, inspect_identities([paper], cap=1), doi_only=True)
    assert result["records"][0]["outcome"] == "deferred_non_doi"
    assert paper.doi == paper.arxiv_id == ""
    assert Library(tmp_path).get(paper.key).identity_route_rejections[0].reason.value == "title_mismatch"


@pytest.mark.parametrize("source", ["ads", "core", "openalex", "inspire"])
@responses.activate
def test_v2_source_bibliography_contradiction_is_recorded(source, tmp_path, monkeypatch):
    from papervault.library.download_sources import core
    from papervault.library.services.identity_backfill import backfill_identities
    monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", "1")
    monkeypatch.setenv("CORE_API_KEY", "fixture")
    monkeypatch.setenv("ADS_API_TOKEN", "fixture")
    monkeypatch.setattr(core, "_api_next_at", 0)
    monkeypatch.setattr(core, "_api_cooldown_until", 0)
    wrong_title = "Galactic Cosmic Rays in the Dynamic Heliosphere"
    urls = {"ads": "https://ui.adsabs.harvard.edu/abs/2012ApJ...750....1S/abstract",
            "core": "https://core.ac.uk/works/123", "openalex": "https://openalex.org/W123",
            "inspire": "https://inspirehep.net/literature/123"}
    responses.get(urls[source], body="<html></html>")
    if source == "ads":
        responses.get("https://api.adsabs.harvard.edu/v1/search/query", json={"response": {"docs": [{
            "bibcode": "2012ApJ...750....1S", "title": [wrong_title], "author": ["Strauss, R. D."],
            "doi": [DOI], "identifier": [], "doctype": "article", "pub": "ApJ", "esources": []}]}})
    elif source == "core":
        responses.get("https://api.core.ac.uk/v3/works/123", json={"id": 123, "doi": DOI,
            "title": wrong_title, "authors": ["Strauss"]})
    elif source == "openalex":
        responses.get("https://api.openalex.org/works/W123", json={"id": urls[source], "doi": DOI,
            "title": wrong_title, "publication_year": 2012,
            "authorships": [{"author": {"display_name": "Strauss"}}]})
    else:
        responses.get("https://inspirehep.net/api/literature/123", json={"id": "123", "metadata": {
            "titles": [{"title": wrong_title}], "authors": [{"full_name": "Strauss, R. D."}],
            "publication_info": [{"year": 2012}], "dois": [{"value": DOI}]}})
    _registry()
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=urls[source], source=source).model_dump())
    result = backfill_identities(lib, cap=1)
    assert result["records"][0]["outcome"] == "contradicted"
    assert paper.doi == "" and paper.download_status == "pending"
    assert paper.identity_route_rejections[0].reason.value == "title_mismatch"


@pytest.mark.parametrize("source", ["core", "openalex", "inspire"])
@responses.activate
def test_v2_valid_source_route_survives_rejected_original_url(source, tmp_path, monkeypatch):
    from papervault.library.download_sources import core
    from papervault.library.services.identity_backfill import backfill_identities
    monkeypatch.setenv("PAPERVAULT_IDENTITY_V2", "1")
    monkeypatch.setenv("CORE_API_KEY", "fixture")
    monkeypatch.setattr(core, "_api_next_at", 0)
    monkeypatch.setattr(core, "_api_cooldown_until", 0)
    original = "https://doi.org/10.1234/wrong"
    responses.get(original, body="<html></html>")
    _registry("10.1234/wrong", title="Galactic Cosmic Rays in the Dynamic Heliosphere")
    _registry()
    urls = {"core": "https://api.core.ac.uk/v3/works/123",
            "openalex": "https://api.openalex.org/works/W123",
            "inspire": "https://inspirehep.net/api/literature/123"}
    if source == "core":
        responses.get(urls[source], json={"id": 123, "doi": DOI, "title": TITLE, "authors": ["Strauss"]})
        fields = {"paper_id": "123"}
    elif source == "openalex":
        responses.get(urls[source], json={"id": "https://openalex.org/W123", "doi": DOI,
            "title": TITLE, "publication_year": 2012,
            "authorships": [{"author": {"display_name": "Strauss"}}]})
        fields = {"paper_id": "W123"}
    else:
        responses.get(urls[source], json={"id": "123", "metadata": {
            "titles": [{"title": TITLE}], "authors": [{"full_name": "Strauss, R. D."}],
            "publication_info": [{"year": 2012}], "dois": [{"value": DOI}]}})
        fields = {"inspire_record_id": "123"}
    lib = Library(tmp_path)
    paper, _ = lib.upsert(_paper(url=original, source=source, **fields).model_dump())
    result = backfill_identities(lib, cap=1)
    assert result["recovered"] == 1 and paper.doi == DOI
    assert paper.identity_recovery.source_url == urls[source]
    assert paper.identity_route_rejections[0].url == original
