"""Additive A5 measurement, preserving old JSONL and explicit censoring."""

import json

import pytest

from papervault.library.claims_v2 import ReopenClaim
from papervault.library.download_telemetry import emit, observation, span
from papervault.library.store import Library


def records(library):
    return [json.loads(line) for line in library.manifest_path.read_text().splitlines()]


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_STOPRULES_V2", "1")
    lib = Library(tmp_path)
    paper, _ = lib.upsert({"title": "A correlated measurement fixture", "doi": "10.1234/telemetry"})
    paper.reopen_claim = ReopenClaim(key=paper.key, generation=7, lease_id="fixture-lease",
                                    campaign_id="fixture-campaign", lease_expiry=200.0)
    return lib, paper


def test_enrichment_keeps_legacy_fields_and_snapshots_original_attempt(vault):
    from papervault.library.telemetry_v2 import attempt, request_scope, scope_id
    from papervault.library.politeness_v2 import Scope
    lib, paper = vault
    old = {"event": "download_miss", "key": paper.key, "source": "fixture", "size": 12}
    with attempt(paper.reopen_claim), observation(lib, paper.key), request_scope(Scope("fixture")):
        paper.reopen_claim = paper.reopen_claim.model_copy(update={"generation": 8, "lease_id": "new-lease"})
        lib.log(old)
        emit("request", "start")
    row, telemetry = records(lib)[-2:]
    assert {key: row[key] for key in old} == old
    for event in (row, telemetry):
        assert event["identity_version"] == 2
        assert event["attempt_generation"] == 7 and event["lease_id"] == "fixture-lease"
        assert event["request_scope_id"] == scope_id(Scope("fixture"))
    assert json.loads(json.dumps(old)) == old
    assert "attempt_generation" not in old


def test_off_manifest_is_byte_identical_without_new_fields(vault, monkeypatch):
    lib, paper = vault
    monkeypatch.delenv("PAPERVAULT_STOPRULES_V2")
    event = {"event": "old_record", "key": paper.key, "source": "fixture"}
    before = lib.manifest_path.read_bytes() if lib.manifest_path.exists() else b""
    lib.log(event)
    assert lib.manifest_path.read_bytes()[len(before):] == (json.dumps(event) + "\n").encode()


def test_censored_helper_preserves_unknown_durations_and_never_backfills(vault):
    from papervault.library.telemetry_v2 import record_censored
    lib, paper = vault
    clock = [10.0]
    record_censored(lib, paper.key, reason="executor outcome unknown", wait_s=2.0,
                    clock=lambda: clock[0])
    censored = records(lib)[-1]
    assert censored["phase"] == "attempt" and censored["mark"] == "censored"
    assert censored["censored"] is True
    assert censored["wait_s"] == 2.0
    assert censored["service_s"] is None and censored["drain_s"] is None
    assert "duration_s" not in censored and "status" not in censored
    original = lib.manifest_path.read_bytes()
    clock[0] = 1000.0
    with observation(lib, paper.key):
        emit("attempt", "end", censored=False, service_s=3.0)
    assert lib.manifest_path.read_bytes().startswith(original)
    assert records(lib)[-2] == censored


def test_wait_service_and_drain_are_separate_observed_spans(vault, monkeypatch):
    lib, paper = vault
    from papervault.library import download_telemetry
    clock = [10.0]
    monkeypatch.setattr(download_telemetry.time, "monotonic", lambda: clock[0])
    with observation(lib, paper.key):
        with span("network_wait"):
            clock[0] += 2
        with span("executor"):
            clock[0] += 3
        with span("executor_drain"):
            clock[0] += 5
    ends = [e for e in records(lib) if e.get("mark") == "end"]
    wait, service, drain = ends
    assert wait["wait_s"] == 2 and "service_s" not in wait and "drain_s" not in wait
    assert service["service_s"] == 3 and "wait_s" not in service and "drain_s" not in service
    assert drain["drain_s"] == 5 and "wait_s" not in drain and "service_s" not in drain


def test_cancelled_span_is_censored_not_success(vault):
    lib, paper = vault
    with pytest.raises(RuntimeError), observation(lib, paper.key), span("executor"):
        raise RuntimeError("unfinished")
    row = records(lib)[-1]
    assert row["censored"] is True
    assert row["service_s"] is None
    assert row["observed_elapsed_s"] >= 0


def test_positive_verification_emits_success_only_for_live_publication(vault, monkeypatch):
    from papervault.library import download
    lib, paper = vault
    # Ordinary acquisition fixture; campaign publication is tested separately.
    paper.reopen_claim = None
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", lambda _: b"%PDF-fixture")])
    monkeypatch.setattr(download, "_verify_pdf_matches_metadata", lambda *_: (True, "llm_match: fixture"))
    assert download.download_paper(paper, lib)
    assert len([r for r in records(lib) if r.get("phase") == "verified-success"]) == 1


def test_fail_open_download_never_reports_verified_success(vault, monkeypatch):
    from papervault.library import download
    lib, paper = vault
    paper.reopen_claim = None
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", lambda _: b"%PDF-fixture")])
    monkeypatch.setattr(download, "_verify_pdf_matches_metadata", lambda *_: (True, "verify_fail_open_after_3_tries"))
    assert download.download_paper(paper, lib)
    assert not any(r.get("phase") == "verified-success" for r in records(lib))


@pytest.mark.parametrize("field,value", [
    ("arxiv_id", ""), ("arxiv_id", "2601.00002v3"), ("paper_id", "repaired-source-id"),
    ("inspire_record_id", "456"), ("url", "https://example.org/repaired-route"),
    ("source", "inspire"), ("doi", "https://doi.org/10.1234/TELEMETRY"),
])
def test_canonical_doi_publication_ignores_secondary_identifier_repairs(vault, field, value):
    from papervault.library.telemetry_v2 import identity_snapshot, publication_fields
    _, paper = vault
    paper.arxiv_id = "2601.00001v1"
    paper.paper_id = "original-source-id"
    paper.inspire_record_id = "123"
    paper.url = "https://example.org/original-route"
    before = identity_snapshot(paper)
    setattr(paper, field, value)
    assert publication_fields(before, paper, verified=True) == {
        "identity_matches": True, "identity_verified": True}


def test_canonical_arxiv_publication_normalizes_version_and_ignores_secondary_route(vault):
    from papervault.library.telemetry_v2 import identity_snapshot, publication_fields
    _, paper = vault
    paper.doi = ""
    paper.arxiv_id = "2601.00001v1"
    paper.url = "https://example.org/original-route"
    before = identity_snapshot(paper)
    paper.arxiv_id = "2601.00001v3"
    paper.url = "https://example.org/repaired-route"
    assert publication_fields(before, paper, verified=True)["identity_matches"] is True
    paper.arxiv_id = "2601.00002"
    assert publication_fields(before, paper, verified=True)["identity_matches"] is False


def test_publication_identity_resolves_persisted_merge_aliases(vault):
    from dataclasses import replace
    from papervault.library.state_v2 import Merge
    from papervault.library.telemetry_v2 import identity_snapshot, publication_fields
    lib, canonical = vault
    alias, _ = lib.upsert({"title": "An absorbed record with stale secondary identity",
                           "doi": "10.1234/absorbed"})
    alias.terminal_state = replace(alias.acquisition_state, merge=Merge.ALIAS, alias_key=canonical.key)
    lib.save()
    lib = Library(lib.root)
    alias, canonical = lib.get(alias.key), lib.get(canonical.key)
    before = identity_snapshot(alias, library=lib)
    assert before == identity_snapshot(canonical)
    alias.doi = "10.1234/stale"
    assert publication_fields(before, alias, library=lib, verified=True)["identity_matches"] is True
    canonical.doi = "10.1234/different"
    assert publication_fields(before, alias, library=lib, verified=True)["identity_matches"] is False


def test_request_denial_records_scope_without_emitting_a_send(vault, monkeypatch):
    from papervault.library import polite_http
    from papervault.library.politeness_v2 import Ledger, Policy, Scope, Window
    from papervault.library.telemetry_v2 import scope_id
    lib, paper = vault
    monkeypatch.setenv("PAPERVAULT_POLITE_V2", "1")
    scope = Scope("wiley")
    ledger = Ledger({scope: Policy((Window(1, 60),))}, clock=lambda: 10.0)
    ticket = ledger.note_sent(scope)
    ledger.note_result(ticket, status_code=200)
    monkeypatch.setattr(polite_http, "get_ledger", lambda: ledger)
    def transport(*args, **kwargs):
        raise AssertionError("denied send reached transport")
    with observation(lib, paper.key), pytest.raises(polite_http.PolitenessDenied):
        polite_http.polite_get(transport, "https://api.wiley.com/onlinelibrary/tdm/v1/articles/test")
    denied = [r for r in records(lib) if r.get("phase") == "admission-denied"]
    assert denied[0]["request_scope_id"] == scope_id(scope)
    assert ledger.snapshot(scope).sent == 1


def test_group_winner_service_and_losing_executor_drain_are_distinct(vault):
    import threading
    from papervault.library import download
    lib, paper = vault
    started = threading.Event()
    winner_seen = threading.Event()
    original_log = lib.log
    def record(event):
        original_log(event)
        if event.get("phase") == "winner":
            winner_seen.set()
    lib.log = record
    def winner(_):
        assert started.wait(2)
        return b"%PDF-fixture"
    def loser(_):
        started.set()
        assert winner_seen.wait(2)
        return None
    with observation(lib, paper.key):
        assert download._try_concurrent_first_hit(paper, [("winner", winner), ("loser", loser)])
    events = records(lib)
    winner_event = next(e for e in events if e.get("phase") == "winner")
    drain = next(e for e in events if e.get("phase") == "group_drain" and e["mark"] == "end")
    assert winner_event["at"] <= drain["started_at"] <= drain["ended_at"]
    assert drain["drain_s"] >= 0 and "service_s" not in drain
    assert all("service_s" in e for e in events if e.get("phase") == "member" and e["mark"] == "end")


async def test_staged_success_is_reported_after_live_fenced_publication(vault, monkeypatch):
    from papervault.library import download
    from papervault.library.reopen_v2 import run_campaign
    from papervault.library.state_v2 import Policy, Retrieval, TerminalState
    lib, paper = vault
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    paper.reopen_claim = None
    paper.download_status = "failed"
    paper.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
    lib.save()
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", lambda _: b"%PDF-fixture")])
    monkeypatch.setattr(download, "_verify_pdf_matches_metadata", lambda *_: (True, "llm_match: fixture"))
    await run_campaign(lib, cap=1)
    events = records(lib)
    success, = [e for e in events if e.get("phase") == "verified-success"]
    complete = next(e for e in events if e["event"] == "reopen_complete")
    assert success["attempt_generation"] == complete["attempt_generation"] == 1
    assert success["lease_id"] == paper.reopen_claim.lease_id
    assert lib.has_pdf(paper.key)


@pytest.mark.parametrize("source", ["openalex", "core"])
@pytest.mark.parametrize("staged", [False, True])
async def test_verified_blank_fill_promotes_canonical_identity_without_stop(vault, monkeypatch, source, staged):
    from papervault.library import download
    from papervault.library.claims_v2 import BudgetLedger
    from papervault.library.download_sources.core import _CorePDF
    from papervault.library.download_sources.openalex import _OpenAlexPDF
    from papervault.library.reopen_v2 import run_campaign
    from papervault.library.state_v2 import Policy, Retrieval, TerminalState
    from papervault.library.stop_rules import evaluate_round
    lib, paper = vault
    paper.reopen_claim = None
    paper.doi = ""
    paper.arxiv_id = "2601.00001"
    ledger = BudgetLedger(lib.root)
    if staged:
        monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
        paper.download_status = "failed"
        paper.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
    else:
        ledger.start_campaign(cap=1, keys=[paper.key])
    lib.save()
    pdf_type = _OpenAlexPDF if source == "openalex" else _CorePDF
    data = pdf_type(b"%PDF-fixture", "10.1234/adopted", paper.arxiv_id)
    monkeypatch.setattr(download, "_STRATEGIES", [(source, lambda _: data)])
    monkeypatch.setattr(download, "_verify_pdf_matches_metadata", lambda *_: (True, "llm_match: fixture"))
    if staged:
        await run_campaign(lib, cap=1)
    else:
        assert download.download_paper(paper, lib)
    publications = [event for event in records(lib) if event["event"] in {"downloaded", "reopen_complete"}]
    assert publications and all(event["identity_matches"] for event in publications)
    assert paper.doi == "10.1234/adopted"
    assert len([event for event in records(lib) if event.get("phase") == "verified-success"]) == 1
    assert await evaluate_round(lib, ledger=ledger) == ()
    assert not any(event["event"] == "campaign_halt" for event in ledger.events())
    assert not paper.acquisition_state.quarantine


@pytest.mark.parametrize("route", ["preflight", "title", "title_thread"])
def test_arxiv_adoption_from_url_identity_is_a_clean_publication(vault, monkeypatch, route):
    from papervault.library import download
    from papervault.library.download_sources import arxiv
    from papervault.library.sources import arxiv as arxiv_search
    lib, paper = vault
    paper.reopen_claim = None
    paper.doi = ""
    paper.url = "https://arxiv.org/abs/2601.00001" if route == "preflight" else "https://example.org/paper"
    monkeypatch.setattr(arxiv, "_lookup_arxiv_record", lambda identifier: arxiv._ArxivRecord(
        requested_id=identifier, status="found", latest_id="2601.00001v2"))
    monkeypatch.setattr(arxiv, "_try_arxiv_with_record", lambda *_: b"%PDF-fixture")
    monkeypatch.setattr(download, "_try_arxiv_with_record", arxiv._try_arxiv_with_record)
    monkeypatch.setattr(arxiv_search, "search_arxiv", lambda *_args, **_kwargs: [
        {"title": paper.title, "arxiv_id": "2601.00001"}])
    strategy = download._try_arxiv if route == "preflight" else download._try_arxiv_by_title
    if route == "title_thread":
        def strategy(candidate):
            return download._try_concurrent_first_hit(candidate, [("arxiv_title", download._try_arxiv_by_title)])
    monkeypatch.setattr(download, "_STRATEGIES", [("arxiv", strategy)])
    monkeypatch.setattr(download, "_verify_pdf_matches_metadata", lambda *_: (True, "llm_match: fixture"))
    assert download.download_paper(paper, lib)
    assert paper.arxiv_id == "2601.00001"
    publication = next(event for event in records(lib) if event["event"] == "downloaded")
    assert publication["identity_matches"] is True
    assert any(event.get("phase") == "verified-success" for event in records(lib))


def test_acknowledged_fill_can_change_selected_source_namespace(vault, monkeypatch):
    from papervault.library import download
    from papervault.library.download_sources.openalex import _OpenAlexPDF
    lib, paper = vault
    paper.reopen_claim = None
    paper.doi = ""
    paper.inspire_record_id = "123"
    data = _OpenAlexPDF(b"%PDF-fixture", "", "2601.00001")
    monkeypatch.setattr(download, "_STRATEGIES", [("openalex", lambda _: data)])
    monkeypatch.setattr(download, "_verify_pdf_matches_metadata", lambda *_: (True, "llm_match: fixture"))
    assert download.download_paper(paper, lib)
    publication = next(event for event in records(lib) if event["event"] == "downloaded")
    assert paper.arxiv_id == "2601.00001" and paper.inspire_record_id == "123"
    assert publication["identity_matches"] is True


def test_acknowledged_secondary_fill_never_masks_competing_canonical_doi_change(vault, monkeypatch):
    from papervault.library import download
    from papervault.library.download_sources.openalex import _OpenAlexPDF
    lib, paper = vault
    paper.reopen_claim = None
    merge = lib._merge
    def competing_writer(current, incoming, **kwargs):
        result = merge(current, incoming, **kwargs)
        current.doi = "10.1234/different"
        return result
    monkeypatch.setattr(lib, "_merge", competing_writer)
    data = _OpenAlexPDF(b"%PDF-fixture", "", "2601.00001")
    monkeypatch.setattr(download, "_STRATEGIES", [("openalex", lambda _: data)])
    monkeypatch.setattr(download, "_verify_pdf_matches_metadata", lambda *_: (True, "llm_match: fixture"))
    assert download.download_paper(paper, lib)
    publication = next(event for event in records(lib) if event["event"] == "downloaded")
    assert publication["identity_matches"] is False
    assert not any(event.get("phase") == "verified-success" for event in records(lib))


def test_cancelled_queue_remains_censored_after_real_executor_finishes(vault, monkeypatch):
    import asyncio
    import threading
    from papervault.library.services import concurrency, download_queue
    lib, paper = vault
    paper.reopen_claim = None
    started = threading.Event()
    release = threading.Event()
    def unfinished(*args):
        started.set()
        assert release.wait(2)
        return False
    monkeypatch.setattr(download_queue, "download_paper", unfinished)
    async def run():
        monkeypatch.setattr(concurrency, "network_sem", asyncio.Semaphore(1))
        queue = download_queue.DownloadQueue(lib, num_workers=1)
        await queue.start()
        try:
            for _ in range(1000):
                if started.is_set():
                    break
                await asyncio.sleep(0.001)
            assert started.is_set()
            await queue.stop()
            censored, = [e for e in records(lib) if e.get("mark") == "censored"]
            assert censored["service_s"] is None and censored["drain_s"] is None
            assert not any(e.get("phase") == "verified-success" for e in records(lib))
            original = lib.manifest_path.read_bytes()
        finally:
            release.set()
        return original
    original = asyncio.run(run())
    assert lib.manifest_path.read_bytes().startswith(original)
    assert [e for e in records(lib) if e.get("mark") == "censored"][0]["service_s"] is None


async def test_reopen_expiry_records_censored_evidence(vault, monkeypatch):
    import asyncio
    import threading
    from papervault.library.claims_v2 import BudgetLedger, ClaimStore
    from papervault.library.reopen_v2 import process_attempt
    from papervault.library.services import concurrency, download_queue
    from papervault.library.state_v2 import Policy, Retrieval, TerminalState
    lib, paper = vault
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    paper.reopen_claim = None
    paper.download_status = "failed"
    paper.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
    now = [100.0]
    ledger = BudgetLedger(lib.root, clock=lambda: now[0])
    store = ClaimStore(lib, ledger=ledger, clock=lambda: now[0])
    campaign = ledger.start_campaign(cap=1, keys=[paper.key])
    token = await store.claim(paper.key, expected_state=paper.acquisition_state,
                             expected_generation=0, campaign_id=campaign, ttl=0.02)
    started = threading.Event()
    release = threading.Event()
    def unfinished(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return False
    monkeypatch.setattr(download_queue, "download_paper", unfinished)
    monkeypatch.setattr(concurrency, "network_sem", asyncio.Semaphore(1))
    queue = download_queue.DownloadQueue(lib)
    queue._reopen_clock = lambda: now[0]
    task = asyncio.create_task(process_attempt(queue, token, 5))
    try:
        for _ in range(1000):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        now[0] += 1
        for _ in range(1000):
            if paper.acquisition_state.quarantine:
                break
            await asyncio.sleep(0.001)
        assert paper.acquisition_state.quarantine
        assert any(e.get("mark") == "censored" for e in records(lib))
    finally:
        release.set()
        await task
    assert ledger.spent(campaign) == 1


async def test_stale_queue_tuple_keeps_its_lease_correlation(vault, monkeypatch):
    from papervault.library.services import download_queue
    lib, paper = vault
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    old = paper.reopen_claim
    queue = download_queue.DownloadQueue(lib, num_workers=1)
    queue.add_reopen(old)
    paper.reopen_claim = old.model_copy(update={"generation": 8, "lease_id": "new-lease"})
    queue.add_reopen(paper.reopen_claim)
    # Drop processing at the I/O boundary; queue correlation remains real.
    async def no_network(*args, **kwargs):
        pass
    monkeypatch.setattr(queue, "_process_one", no_network)
    await queue.start(recover=False)
    await queue._queue.join()
    await queue.stop()
    drop, = [e for e in records(lib) if e.get("phase") == "queue_drop"]
    assert drop["attempt_generation"] == 7 and drop["lease_id"] == "fixture-lease"


def test_censored_recording_failure_does_not_change_control_flow(vault, monkeypatch):
    from papervault.library.telemetry_v2 import record_censored
    lib, paper = vault
    def unavailable(*args):
        raise OSError("manifest unavailable")
    monkeypatch.setattr(lib, "log", unavailable)
    record_censored(lib, paper.key, reason="uncertain")


def test_mismatched_publication_never_reports_verified_success(vault):
    lib, paper = vault
    lib.log({"event": "reopen_complete", "key": paper.key,
             "identity_matches": False, "identity_verified": True})
    assert not any(e.get("phase") == "verified-success" for e in records(lib))


async def test_campaign_service_and_drain_telemetry_survives_stage_cleanup(vault, monkeypatch):
    from papervault.library import download
    from papervault.library.reopen_v2 import run_campaign
    from papervault.library.state_v2 import Policy, Retrieval, TerminalState
    lib, paper = vault
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    paper.reopen_claim = None
    paper.download_status = "failed"
    paper.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
    lib.save()
    def grouped(candidate):
        return download._try_concurrent_first_hit(candidate, [("fixture", lambda _: b"%PDF-fixture")])
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", grouped)])
    monkeypatch.setattr(download, "_verify_pdf_matches_metadata", lambda *_: (True, "llm_match: fixture"))
    await run_campaign(lib, cap=1)
    assert not list((lib.root / ".reopen-staging").glob("*"))
    events = records(lib)
    member, = [e for e in events if e.get("phase") == "member" and e["mark"] == "end"]
    drain, = [e for e in events if e.get("phase") == "group_drain" and e["mark"] == "end"]
    assert member["service_s"] >= 0 and drain["drain_s"] >= 0
    assert member["lease_id"] == drain["lease_id"] == paper.reopen_claim.lease_id
    assert len([e for e in events if e.get("phase") == "verified-success"]) == 1


async def test_campaign_policy_refusal_emits_admission_denied(vault, monkeypatch):
    from papervault.library.claims_v2 import BudgetLedger, ClaimStore
    from papervault.library.state_v2 import Policy, Retrieval, TerminalState
    lib, paper = vault
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    paper.reopen_claim = None
    paper.download_status = "failed"
    paper.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
    ledger = BudgetLedger(lib.root, clock=lambda: 100.0)
    store = ClaimStore(lib, ledger=ledger, clock=lambda: 100.0)
    campaign = ledger.start_campaign(cap=1, keys=[paper.key])
    claim = await store.claim(paper.key, expected_state=paper.acquisition_state,
                              expected_generation=0, campaign_id=campaign)
    paper.terminal_state = TerminalState(retrieval=Retrieval.ELIGIBLE, policy=Policy.EXCLUDED,
                                         policy_reasons=("fixture exclusion",))
    assert not await store.pre_dispatch(claim)
    denied, = [e for e in records(lib) if e.get("phase") == "admission-denied"]
    assert denied["lease_id"] == claim.lease_id and denied["attempt_generation"] == 1
    assert paper.reopen_claim.phase == "not_sent" and ledger.spent(campaign) == 0


def test_verified_success_recording_failure_is_observational(vault, monkeypatch):
    from papervault.library import download
    lib, paper = vault
    paper.reopen_claim = None
    original = lib.log
    def broken_success_record(event):
        if event.get("phase") == "verified-success":
            raise OSError("telemetry disk failure")
        original(event)
    monkeypatch.setattr(lib, "log", broken_success_record)
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", lambda _: b"%PDF-fixture")])
    monkeypatch.setattr(download, "_verify_pdf_matches_metadata", lambda *_: (True, "llm_match: fixture"))
    assert download.download_paper(paper, lib)
    assert lib.has_pdf(paper.key)


def test_cross_key_event_uses_its_own_claim_not_queue_actor(vault):
    from papervault.library.telemetry_v2 import attempt
    lib, paper = vault
    other, _ = lib.upsert({"title": "Another campaign record for correlation", "doi": "10.1234/other"})
    other.reopen_claim = paper.reopen_claim.model_copy(update={"key": other.key,
                                        "generation": 4, "lease_id": "other-lease"})
    with attempt(paper.reopen_claim):
        lib.log({"event": "reopen_quarantine", "key": other.key})
    event = records(lib)[-1]
    assert event["attempt_generation"] == 4 and event["lease_id"] == "other-lease"
