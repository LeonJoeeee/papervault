"""Fixture-only correction events, monthly limits, and the actual dispatch packet."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
import responses

from papervault.library import Library
from papervault.library.models import ADSAvailability, ADSDocumentEvidence

NOW = datetime(2026, 10, 11, tzinfo=timezone.utc)


def stub(lib, n, source="ads", **extra):
    paper, _ = lib.upsert({"title": f"Transport of cosmic rays in heliospheric region {n}",
                          "authors": [f"Scientist{n}"], "year": 2020, "source": source,
                          "url": f"https://doi.org/10.1234/paper{n}",
                          "download_status": "failed", **extra})
    return paper


def registry(paper, n):
    responses.get(f"https://api.crossref.org/works/10.1234/paper{n}", json={"message": {
        "DOI": f"10.1234/paper{n}", "title": [paper.title],
        "author": [{"family": f"Scientist{n}"}], "issued": {"date-parts": [[2020]]}}})


@responses.activate
@pytest.mark.parametrize("trigger", ["credential_arrived", "source_repaired", "identifiers_arrived"])
def test_events_scope_keys_source_cap_and_never_acquire(tmp_path, monkeypatch, trigger):
    from papervault.library.services.correction_pipeline import run_event
    from papervault.library.services import identity_backfill
    pauses = []
    monkeypatch.setattr(identity_backfill.time, "sleep", pauses.append)
    lib = Library(tmp_path)
    papers = [stub(lib, n) for n in range(3)]
    unrelated = stub(lib, 9, "core")
    for n, paper in enumerate(papers[:2]):
        registry(paper, n)
    result = asyncio.run(run_event(lib, trigger=trigger, source="ads", cap=2,
                                  keys=[p.key for p in papers] + [unrelated.key], now=NOW))
    assert result["keys"] == [p.key for p in papers[:2]]
    assert result["scanned"] == result["recovered"] == 2
    assert result["acquisition"]["attempted"] == 0
    assert pauses == [1.0]
    assert papers[2].doi == unrelated.doi == ""
    assert all(p.download_status == "failed" for p in papers + [unrelated])
    assert len(responses.calls) == 2
    assert result["trigger"] == trigger and result["source"] == "ads"
    events = [json.loads(line) for line in lib.manifest_path.read_text().splitlines()]
    assert any(e["event"] == "correction_pass" and e["cap"] == 2 for e in events)


@pytest.mark.parametrize("kwargs", [
    {"cap": 0}, {"cap": 51}, {"cap": True}, {"source": "all"}, {"trigger": "poll"},
    {"trigger": "identifiers_arrived", "keys": []}, {"pace_seconds": 0},
])
def test_invalid_event_refuses_before_any_write(tmp_path, kwargs):
    from papervault.library.services.correction_pipeline import run_event
    lib = Library(tmp_path)
    args = {"trigger": "source_repaired", "source": "ads", "cap": 1, **kwargs}
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(ValueError):
        asyncio.run(run_event(lib, **args))
    assert sorted(p.name for p in tmp_path.iterdir()) == before


@responses.activate
def test_monthly_sweep_caps_rotation_cadence_and_reports_needs_human(tmp_path, monkeypatch):
    from papervault.library.services.correction_pipeline import monthly_sweep
    from papervault.library.services import identity_backfill
    monkeypatch.setattr(identity_backfill.time, "sleep", lambda _: None)
    lib = Library(tmp_path)
    cohorts = {source: [stub(lib, offset + n, source) for n in range(3)]
               for source, offset in (("ads", 0), ("core", 10), ("openalex", 20))}
    # Registry misses abstain; the next month must progress beyond these keys.
    for papers in cohorts.values():
        for paper in papers:
            responses.get(paper.url.replace("https://doi.org/", "https://api.crossref.org/works/"), status=404)
            responses.get(paper.url, body="<html></html>")
    first = asyncio.run(monthly_sweep(lib, per_cohort_cap=2, total_cap=3, now=NOW))
    assert first["selected"] == 3
    assert all(p["selected"] <= 2 for p in first["passes"])
    seen = {key for p in first["passes"] for key in p["keys"]}
    calls = len(responses.calls)
    second = asyncio.run(monthly_sweep(Library(tmp_path), per_cohort_cap=2, total_cap=3,
                                      now=NOW + timedelta(days=1)))
    assert second["skipped"] == "not_due" and len(responses.calls) == calls
    third = asyncio.run(monthly_sweep(lib, per_cohort_cap=2, total_cap=3,
                                     now=NOW + timedelta(days=30)))
    assert third["selected"] == 3
    assert any(k not in seen for p in third["passes"] for k in p["keys"])
    assert any(p["source"] == "core" for p in third["passes"])
    assert lib.corrections.read()["months"]["2026-10"]["sweeps"] == 1
    assert lib.corrections.read()["months"]["2026-11"]["sweeps"] == 1


@pytest.mark.parametrize("kwargs", [{"per_cohort_cap": 51}, {"total_cap": 101}, {"total_cap": 0}])
def test_monthly_hard_caps_cannot_be_overridden(tmp_path, kwargs):
    from papervault.library.services.correction_pipeline import monthly_sweep
    with pytest.raises(ValueError):
        asyncio.run(monthly_sweep(Library(tmp_path), now=NOW, **kwargs))


@responses.activate
def test_simulated_ten_paper_batch_exposes_luna_manifest_and_retained_monthly_report(tmp_path, monkeypatch):
    from papervault.library.services.correction_pipeline import run_event, monthly_report
    from papervault.library.services import identity_backfill
    monkeypatch.setattr(identity_backfill.time, "sleep", lambda _: None)
    lib = Library(tmp_path)
    papers = []
    for n in range(10):
        paper = stub(lib, n)
        papers.append(paper)
        lib.upsert({"title": f"Independently stored journal identity holder number {n}",
                    "authors": [f"Holder{n}"], "doi": f"10.1234/paper{n}"})
        registry(paper, n)
    result = asyncio.run(run_event(lib, trigger="source_repaired", source="ads", cap=10,
                                  keys=[p.key for p in papers], now=NOW))
    assert result["scanned"] == 10 and result["recovered"] == 0
    assert len(result["manifest_ids"]) == 1
    manifest = lib.corrections.read()["manifests"][result["manifest_ids"][0]]
    assert manifest["executor"]["model"] == "gpt-6-luna"
    assert manifest["executor"]["effort"] == "max"
    assert len(manifest["conflicts"]) == 10
    assert all(c["provenance"]["recovery"]["route"] == "url_pattern" for c in manifest["conflicts"])
    lib.corrections.adjudicate({"manifest_id": manifest["id"], "results": []})
    report = monthly_report(Library(tmp_path), "2026-10")
    assert report["conflicts"]["needs_human"] == 10
    assert report["aggregate"]["scanned"] == 10
    assert all(p.download_status == "failed" and p.doi == "" for p in papers)


def test_stored_ads_route_evidence_is_retained_for_reopen_without_network(tmp_path):
    from papervault.library.services.correction_pipeline import run_event
    lib = Library(tmp_path)
    paper = stub(lib, 0, doi="10.1234/paper0")
    paper.ads_availability = ADSAvailability(
        availability="retrievable", bibcode="2020ApJ...900....1S", checked_at=NOW.isoformat(),
        documents=[ADSDocumentEvidence(url="https://articles.adsabs.harvard.edu/pdf/test",
            outcome="pdf", document_kind="full_text", identity_match=True, sha256="a" * 64),
            ADSDocumentEvidence(url="https://example.org/blocked", outcome="blocked")])
    result = asyncio.run(run_event(lib, trigger="source_repaired", source="ads", cap=1, now=NOW))
    assert result["scanned"] == 0 and result["route_findings"] == 1
    finding = next(iter(lib.corrections.read()["routes"].values()))
    assert finding["key"] == paper.key and finding["target_issue"] == 174
    assert finding["url"] == "https://articles.adsabs.harvard.edu/pdf/test"
    assert finding["provenance"]["checked_at"] == NOW.isoformat()
    assert paper.download_status == "failed"


@pytest.mark.asyncio
async def test_overlapping_sweeps_share_durable_due_claim(tmp_path):
    from papervault.library.services.correction_pipeline import monthly_sweep
    lib = Library(tmp_path)
    results = await asyncio.gather(monthly_sweep(lib, now=NOW), monthly_sweep(lib, now=NOW))
    assert sum(r.get("skipped") == "not_due" for r in results) == 1


def test_monthly_schedule_survives_restart(tmp_path):
    from papervault.library.services.correction_pipeline import seconds_until_sweep
    lib = Library(tmp_path)
    assert seconds_until_sweep(lib, now=NOW) == 30 * 86400
    assert seconds_until_sweep(Library(tmp_path), now=NOW + timedelta(days=29)) == 86400
    assert seconds_until_sweep(lib, now=NOW + timedelta(days=31)) == 0


@responses.activate
def test_source_event_uses_cap_for_actionable_rows_instead_of_already_anchored_rows(tmp_path):
    from papervault.library.services.correction_pipeline import run_event
    lib = Library(tmp_path)
    stub(lib, 0, doi="10.1234/paper0")
    target = stub(lib, 1)
    registry(target, 1)
    result = asyncio.run(run_event(lib, trigger="credential_arrived", source="ads", cap=1, now=NOW))
    assert result["keys"] == [target.key] and target.doi == "10.1234/paper1"


@responses.activate
def test_verified_arxiv_only_identity_is_retained_as_deferred_evidence(tmp_path):
    from papervault.library.services.correction_pipeline import run_event, monthly_report
    lib = Library(tmp_path)
    paper = stub(lib, 0, "arxiv", url="https://arxiv.org/abs/1201.12345")
    responses.get(paper.url, body=f'''<html><head>
        <link rel="canonical" href="https://arxiv.org/abs/1201.12345">
        <meta name="citation_arxiv_id" content="1201.12345">
        <meta name="citation_title" content="{paper.title}">
        <meta name="citation_author" content="Scientist0">
        <meta property="og:url" content="https://arxiv.org/abs/1201.12345v1">
        </head><body><div id="abs"></div><div class="submission-history">
        <strong>[v1]</strong> Jan 2012<br></div></body></html>''')
    result = asyncio.run(run_event(lib, trigger="identifiers_arrived", source="arxiv",
                                  keys=[paper.key], cap=1, now=NOW))
    assert result["recovered"] == 0 and result["deferred_non_doi"] == 1
    assert result["records"][0]["recovery"]["arxiv_id"] == "1201.12345"
    assert paper.doi == paper.arxiv_id == "" and paper.download_status == "failed"
    assert monthly_report(lib, "2026-10")["aggregate"]["deferred_non_doi"] == 1
    events = [json.loads(line) for line in lib.manifest_path.read_text().splitlines()]
    assert any(e["event"] == "correction_pass" and e["records"][0]["outcome"] == "deferred_non_doi"
               for e in events)


def test_ads_survey_route_import_is_scoped_capped_and_keeps_provenance(tmp_path):
    from papervault.library.services.correction_pipeline import run_event
    lib = Library(tmp_path)
    code = "2020ApJ...900....1S"
    paper = stub(lib, 0, doi="10.1234/paper0", paper_id=code,
                 url=f"https://ui.adsabs.harvard.edu/abs/{code}/abstract")
    finding = {"key": paper.key, "url": f"https://articles.adsabs.harvard.edu/pdf/{code}",
               "source_identifier": code, "provenance": {"issue": 205,
               "checked_at": NOW.isoformat(), "verification": "remote_pdf", "status_code": 200}}
    result = asyncio.run(run_event(lib, trigger="source_repaired", source="ads", cap=1,
                                  keys=[paper.key], route_findings=[finding], now=NOW))
    assert result["route_findings"] == 1
    saved = next(iter(lib.corrections.read()["routes"].values()))
    assert saved["provenance"] == finding["provenance"] and saved["target_issue"] == 174
    assert paper.download_status == "failed"
    before = lib.corrections.path.read_bytes()
    with pytest.raises(ValueError):
        asyncio.run(run_event(lib, trigger="source_repaired", source="core", cap=1,
                              route_findings=[finding], now=NOW))
    with pytest.raises(ValueError):
        asyncio.run(run_event(lib, trigger="source_repaired", source="ads", cap=1,
                              keys=[paper.key], route_findings=[finding] * 4, now=NOW))
    assert lib.corrections.path.read_bytes() == before


@pytest.mark.asyncio
async def test_cancelled_discovery_is_joined_and_cannot_apply_after_shutdown(tmp_path, monkeypatch):
    import threading
    from papervault.library.services.correction_pipeline import run_event
    from papervault.library.services import identity_backfill
    lib = Library(tmp_path)
    paper = stub(lib, 0)
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = identity_backfill.inspect_identities
    def paused(*args, **kwargs):
        started.set()
        release.wait(2)
        # No network here: return the ordinary empty inspection shape.
        result = original([], cap=1, keys=[])
        finished.set()
        return result
    monkeypatch.setattr(identity_backfill, "inspect_identities", paused)
    task = asyncio.create_task(run_event(lib, trigger="source_repaired", source="ads", cap=1))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set() and paper.doi == "" and not lib.corrections.path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["quarantine", "title", "key_reuse", "doi"])
async def test_changed_snapshot_cannot_receive_a_previously_verified_identity(tmp_path, monkeypatch, change):
    from papervault.library.services.correction_pipeline import run_event
    from papervault.library.services import identity_backfill
    from papervault.library.models import IdentityRecovery
    lib = Library(tmp_path)
    paper = stub(lib, 0)
    original = identity_backfill.inspect_identities
    def changed(papers, **kwargs):
        result = original([], cap=1)
        candidate = IdentityRecovery(doi="10.1234/paper0", route="url_pattern", source_url=paper.url,
            verified_title=paper.title, observed_at=NOW.isoformat())
        result.update(scanned=1, recovered=1, records=[{"key": paper.key, "outcome": "verified",
                                                     "recovery": candidate.model_dump()}])
        if change == "quarantine":
            paper.domain_status = "out_of_domain"
        elif change == "title":
            paper.title = "An entirely different scientific publication"
        elif change == "doi":
            lib.set_resolved_doi(paper.key, "10.1234/concurrent")
        else:
            lib.purge(paper.key, reason="fixture_replaced")
            replacement = stub(lib, 0, title="Replacement scientific publication with a reused citation key")
            assert replacement.key == paper.key
        return result
    monkeypatch.setattr(identity_backfill, "inspect_identities", changed)
    result = await run_event(lib, trigger="source_repaired", source="ads", cap=1, keys=[paper.key])
    assert result["recovered"] == 0 and result["records"][0]["outcome"] == "stale_snapshot"
    assert lib.get(paper.key).doi == ("10.1234/concurrent" if change == "doi" else "")
    if change == "doi":
        record = next(iter(lib.corrections.read()["conflicts"].values()))
        assert record["fields"]["doi"] == {"existing": "10.1234/concurrent", "incoming": "10.1234/paper0"}
        assert record["incoming"]["doi"] == ""
        assert record["provenance"]["recovery"]["doi"] == "10.1234/paper0"


@pytest.mark.asyncio
async def test_overlapping_events_serialize_discovery_and_do_not_repeat_identity_work(tmp_path, monkeypatch):
    import threading
    from papervault.library.services.correction_pipeline import run_event
    from papervault.library.services import identity_backfill
    from papervault.library.models import IdentityRecovery
    lib = Library(tmp_path)
    paper = stub(lib, 0)
    entered, release = threading.Event(), threading.Event()
    calls = 0
    active = 0
    maximum = 0
    mutex = threading.Lock()
    def inspect(papers, **kwargs):
        nonlocal calls, active, maximum
        selected = identity_backfill._selection(papers, kwargs["cap"], kwargs["keys"])
        result = {"dry_run": True, "cap": 1, "scanned": len(selected), "recovered": len(selected),
                  "recovered_by_route": {}, "records": [],
                  "acquisition": {"attempted": 0, "by_source": {}, "outcomes": {}}}
        if selected:
            with mutex:
                calls += 1
                active += 1
                maximum = max(maximum, active)
            entered.set()
            release.wait(2)
            recovery = IdentityRecovery(doi="10.1234/paper0", route="url_pattern", source_url=paper.url,
                verified_title=paper.title, observed_at=NOW.isoformat())
            result["records"] = [{"key": paper.key, "outcome": "verified", "recovery": recovery.model_dump()}]
            with mutex:
                active -= 1
        return result
    monkeypatch.setattr(identity_backfill, "inspect_identities", inspect)
    first = asyncio.create_task(run_event(lib, trigger="source_repaired", source="ads", cap=1, keys=[paper.key]))
    await asyncio.to_thread(entered.wait, 2)
    second = asyncio.create_task(run_event(lib, trigger="identifiers_arrived", source="ads", cap=1, keys=[paper.key]))
    await asyncio.sleep(0.02)
    release.set()
    results = await asyncio.gather(first, second)
    assert maximum == calls == 1
    assert sum(r["recovered"] for r in results) == 1


def test_positive_ads_route_after_failed_gateways_is_selected(tmp_path):
    from papervault.library.services.correction_pipeline import run_event
    lib = Library(tmp_path)
    paper = stub(lib, 0, doi="10.1234/paper0")
    paper.ads_availability = ADSAvailability(bibcode="2020ApJ...900....1S", checked_at=NOW.isoformat(),
        documents=[ADSDocumentEvidence(url=f"https://example.org/blocked{n}", outcome="blocked")
                   for n in range(3)] + [ADSDocumentEvidence(url="https://example.org/verified",
                   outcome="pdf", document_kind="full_text", identity_match=True)])
    result = asyncio.run(run_event(lib, trigger="source_repaired", source="ads", cap=1))
    assert result["route_findings"] == 1
    assert next(iter(lib.corrections.read()["routes"].values()))["url"] == "https://example.org/verified"


def test_older_survey_packet_cannot_replace_newer_stored_route_evidence(tmp_path):
    from papervault.library.services.correction_pipeline import run_event
    lib = Library(tmp_path)
    code = "2020ApJ...900....1S"
    paper = stub(lib, 0, doi="10.1234/paper0", paper_id=code,
                 url=f"https://ui.adsabs.harvard.edu/abs/{code}/abstract")
    url = f"https://articles.adsabs.harvard.edu/pdf/{code}"
    paper.ads_availability = ADSAvailability(bibcode=code, checked_at=NOW.isoformat(),
        documents=[ADSDocumentEvidence(url=url, outcome="pdf", document_kind="full_text",
                                      identity_match=True, sha256="newer-digest")])
    old = {"key": paper.key, "url": url, "source_identifier": code, "provenance": {
        "issue": 205, "checked_at": (NOW - timedelta(days=1)).isoformat(), "verification": "remote_pdf"}}
    asyncio.run(run_event(lib, trigger="source_repaired", source="ads", cap=1,
                          keys=[paper.key], route_findings=[old]))
    record = next(iter(lib.corrections.read()["routes"].values()))
    assert record["provenance"]["document"]["sha256"] == "newer-digest"
    assert len(record["observations"]) == 2


@responses.activate
def test_conflicts_from_a_download_thread_during_save_are_not_attributed_to_the_pass(tmp_path, monkeypatch):
    import threading
    from papervault.library.services.correction_pipeline import run_event
    lib = Library(tmp_path)
    paper = stub(lib, 0)
    registry(paper, 0)
    save = lib.save
    def save_with_concurrent_conflict(**kwargs):
        worker = threading.Thread(target=lambda: lib.corrections.stage(
            key="OtherPaper", fields={"doi": {"existing": "10.1234/old", "incoming": "10.1234/new"}},
            existing={"doi": "10.1234/old"}, incoming={"doi": "10.1234/new"},
            provenance={"operation": "verified_identifiers", "incoming_source": "core"}))
        worker.start()
        worker.join()
        save(**kwargs)
    monkeypatch.setattr(lib, "save", save_with_concurrent_conflict)
    result = asyncio.run(run_event(lib, trigger="source_repaired", source="ads", cap=1))
    assert result["recovered"] == 1
    assert result["conflicts_staged"] == 0 and result["manifest_ids"] == []
    assert len(lib.corrections.read()["conflicts"]) == 1
