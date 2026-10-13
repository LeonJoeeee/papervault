"""A5 detectors and real actuators, using durable fixture vaults only."""

import json

import pytest
import responses

from papervault.library.claims_v2 import BudgetLedger, ClaimStore
from papervault.library.politeness_v2 import Ledger, Policy, Scope, Window
from papervault.library.state_v2 import Policy as AcquisitionPolicy
from papervault.library.state_v2 import Retrieval, TerminalState
from papervault.library.store import Library


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    monkeypatch.setenv("PAPERVAULT_STOPRULES_V2", "1")
    now = [100.0]
    library = Library(tmp_path)
    paper, _ = library.upsert({"title": "Executable stop rule fixture", "doi": "10.1234/stops"})
    paper.download_status = "failed"
    paper.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=AcquisitionPolicy.CLEAR)
    library.save()
    ledger = BudgetLedger(tmp_path, clock=lambda: now[0])
    store = ClaimStore(library, ledger=ledger, clock=lambda: now[0])
    return library, paper, ledger, store, now


async def reserve(campaign):
    _, paper, ledger, store, _ = campaign
    identifier = ledger.start_campaign(cap=2, keys=[paper.key])
    return await store.claim(paper.key, expected_state=paper.acquisition_state,
                             expected_generation=0, campaign_id=identifier)


async def test_duplicate_identity_halts_durably_and_clean_round_is_noop(campaign):
    from papervault.library.stop_rules import DUPLICATE_INFLIGHT_THRESHOLD, InFlight, evaluate_round
    library, paper, ledger, _, _ = campaign
    token = await reserve(campaign)
    one = InFlight(paper.key, "doi:10.1234/stops", token.campaign_id, token.lease_id)
    before = ledger.path.read_bytes()
    assert await evaluate_round(library, ledger=ledger, inflight=[one], accounting=None) == ()
    assert ledger.path.read_bytes() == before
    duplicate = InFlight("other-key", one.identity, token.campaign_id, "other-lease")
    assert len([duplicate]) > DUPLICATE_INFLIGHT_THRESHOLD
    findings = await evaluate_round(library, ledger=ledger, inflight=[one, duplicate], accounting=None)
    assert findings[0].rule == "duplicate_inflight_identity"
    assert BudgetLedger(library.root).campaign_halted(token.campaign_id)
    assert ledger.spent(token.campaign_id) == 1
    before = ledger.path.read_bytes()
    await evaluate_round(library, ledger=ledger, inflight=[one, duplicate], accounting=None)
    assert ledger.path.read_bytes() == before


async def test_quota_detector_uses_actual_sends_not_denials_and_cools_scope(campaign):
    from papervault.library.stop_rules import QUOTA_WINDOW_VIOLATION_THRESHOLD, evaluate_round
    library, _, ledger, _, now = campaign
    token = await reserve(campaign)
    scope = Scope("fixture")
    accounting = Ledger({scope: Policy((Window(2, 1), Window(3, 600)), quiet_seconds=30)},
                        clock=lambda: now[0])
    for offset in (0, 2, 4):
        now[0] = 100.0 + offset
        ticket = accounting.note_sent(scope)
        accounting.note_result(ticket, status_code=200)
    assert accounting.gate(scope).reason == "window"
    assert await evaluate_round(library, ledger=ledger, accounting=accounting) == ()
    # Fault injection: an unaccounted send is present in the locked accounting state.
    accounting._states[scope].sends.append(now[0])
    assert 1 > QUOTA_WINDOW_VIOLATION_THRESHOLD
    findings = await evaluate_round(library, ledger=ledger, accounting=accounting)
    assert findings[0].rule == "quota_window_violation"
    assert accounting.gate(scope).reason == "cooldown"
    assert accounting.snapshot(scope).cooldown_until == 134.0
    assert ledger.campaign_halted(token.campaign_id)
    now[0] += 1
    await evaluate_round(library, ledger=ledger, accounting=accounting)
    assert accounting.snapshot(scope).cooldown_until == 134.0  # repeat does not extend quiet window


async def test_published_identity_mismatch_halts_and_quarantines_current_claim(campaign):
    from papervault.library.stop_rules import IDENTITY_MISMATCH_PUBLICATION_THRESHOLD, Publication, evaluate_round
    library, paper, ledger, store, _ = campaign
    token = await reserve(campaign)
    assert await store.pre_dispatch(token)
    clean = Publication(token, identity_matches=True)
    assert await evaluate_round(library, ledger=ledger, publications=[clean], accounting=None) == ()
    mismatch = Publication(token, identity_matches=False)
    assert 1 > IDENTITY_MISMATCH_PUBLICATION_THRESHOLD
    await evaluate_round(library, ledger=ledger, publications=[mismatch], accounting=None)
    assert ledger.campaign_halted(token.campaign_id)
    assert Library(library.root).get(paper.key).acquisition_state.quarantine
    assert ledger.spent(token.campaign_id) == 1 and ledger.phase(token) == "dispatch"
    assert not ledger.refund_if_not_sent(token, proven_not_sent=False)


async def test_round_reads_ledger_claims_and_merge_aliases(campaign):
    from papervault.library.claims_v2 import ReopenClaim
    from papervault.library.stop_rules import evaluate_round
    library, paper, ledger, _, now = campaign
    other, _ = library.upsert({"title": "A separately keyed identity collision", "arxiv_id": "2601.00001"})
    other.doi = paper.doi  # injected duplicate adopted identity
    identifier = ledger.start_campaign(cap=2, keys=[paper.key, other.key])
    for key in (paper.key, other.key):
        token = ReopenClaim(key=key, generation=1, campaign_id=identifier,
                            lease_id=key, lease_expiry=now[0] + 60)
        ledger.reserve(token)
    findings = await evaluate_round(library, ledger=ledger, accounting=None)
    assert findings[0].rule == "duplicate_inflight_identity"
    assert ledger.campaign_halted(identifier)


async def test_disabled_round_never_reads_inputs_or_writes(campaign, monkeypatch):
    from papervault.library.stop_rules import SIGNIFICANT_REGRESSION_FRACTION, evaluate_round
    library, _, ledger, _, _ = campaign
    await reserve(campaign)
    monkeypatch.delenv("PAPERVAULT_STOPRULES_V2")
    before = {p.name: p.read_bytes() for p in library.root.glob("*") if p.is_file()}
    class Unreadable:
        def events(self):
            raise AssertionError("disabled evaluation read ledger")
    assert await evaluate_round(library, ledger=Unreadable(), accounting=Unreadable()) == ()
    assert 0 < SIGNIFICANT_REGRESSION_FRACTION < 1
    assert {p.name: p.read_bytes() for p in library.root.glob("*") if p.is_file()} == before
    assert not any(r["event"] == "campaign_halt" for r in map(json.loads, ledger.path.read_text().splitlines()))


async def test_manifest_publication_evidence_is_evaluated_but_tier_rejections_are_not(campaign):
    from papervault.library.stop_rules import evaluate_round
    library, paper, ledger, store, _ = campaign
    token = await reserve(campaign)
    assert await store.pre_dispatch(token)
    library.log({"event": "download_pdf_mismatch", "key": paper.key, "reason": "rejected route"})
    assert await evaluate_round(library, ledger=ledger, accounting=None) == ()
    library.log({"event": "reopen_complete", "key": paper.key, "identity_matches": False})
    await evaluate_round(library, ledger=ledger, accounting=None)
    assert ledger.campaign_halted(token.campaign_id)
    assert paper.acquisition_state.quarantine


async def test_reconcile_evaluates_before_routing_campaign(campaign, monkeypatch):
    from papervault.library.claims_v2 import ReopenClaim
    from papervault.library.services import reconcile
    library, paper, ledger, _, now = campaign
    other, _ = library.upsert({"title": "Another colliding adopted record", "arxiv_id": "2601.00002"})
    other.doi = paper.doi
    identifier = ledger.start_campaign(cap=2, keys=[paper.key, other.key])
    for row in (paper, other):
        token = ReopenClaim(key=row.key, generation=1, campaign_id=identifier,
                            lease_id=row.key, lease_expiry=now[0] + 60)
        ledger.reserve(token)
        row.reopen_claim = token
    async def no_network(*args, **kwargs):
        return 0
    monkeypatch.setattr(reconcile, "_enrich_sweep", no_network)
    monkeypatch.setattr(reconcile, "_abstract_sweep", no_network)
    class Queue:
        def add(self, *args):
            pass
    await reconcile.reconcile_once(library, Queue(), Queue())
    assert ledger.campaign_halted(identifier)


async def test_redundant_same_lease_queue_capability_records_denial_without_halt(campaign):
    from papervault.library.services import concurrency
    from papervault.library.services.download_queue import DownloadQueue
    library, paper, ledger, _, _ = campaign
    token = await reserve(campaign)
    queue = DownloadQueue(library)
    # with_dedup registers the existing executor before invoking its body.
    async def existing():
        await queue._process_one(paper.key, claim=token)
    await concurrency.with_dedup(f"dl:{paper.key}", existing)
    assert not ledger.campaign_halted(token.campaign_id)
    assert ledger.phase(token) == "reserve"
    assert any(e.get("phase") == "admission-denied"
               for e in map(json.loads, library.manifest_path.read_text().splitlines()))


async def test_ordinary_published_mismatch_halts_active_campaign_and_quarantines(campaign):
    from papervault.library.stop_rules import evaluate_round
    library, _, ledger, _, _ = campaign
    token = await reserve(campaign)
    ordinary, _ = library.upsert({"title": "Ordinary publication with contradictory identity",
                                 "doi": "10.1234/ordinary"})
    library.log({"event": "downloaded", "key": ordinary.key,
                 "identity_matches": False, "identity_verified": False})
    await evaluate_round(library, ledger=ledger, accounting=None)
    assert ledger.campaign_halted(token.campaign_id)
    assert Library(library.root).get(ordinary.key).acquisition_state.quarantine


@pytest.mark.parametrize("scenario", ["clean", "secondary_missing", "canonical_changed"])
@responses.activate
async def test_verified_doi_pipeline_stops_only_for_canonical_identity_change(
        campaign, monkeypatch, caplog, scenario):
    from papervault.library import download
    from papervault.library.services import concurrency
    from papervault.library.stop_rules import evaluate_round
    library, _, ledger, _, now = campaign
    token = await reserve(campaign)
    ordinary, _ = library.upsert({"title": "Canonical DOI publication after secondary route repair",
                                 "doi": "10.1234/ordinary",
                                 "arxiv_id": "1311.9999v7" if scenario == "secondary_missing" else ""})
    scope = Scope("doi-fixture")
    accounting = Ledger({scope: Policy((Window(2, 1), Window(3, 600)), quiet_seconds=30)},
                        clock=lambda: now[0])
    monkeypatch.setattr(concurrency, "_vault_path", lambda: library.root)
    if scenario == "secondary_missing":
        responses.get("https://arxiv.org/abs/1311.9999", status=404, body='''
            <html><body><main><div id="content"><h1>Article 1311.9999 not found</h1>
            <p>There is no record of an article with identifier '1311.9999'.</p>
            </div></main></body></html>''')

    def doi_download(paper):
        # The real preflight must clear the secondary ID before the DOI tier.
        assert paper.arxiv_id == ""
        ticket = accounting.note_sent(scope)
        accounting.note_result(ticket, status_code=200)
        if scenario == "canonical_changed":
            paper.doi = "10.1234/different"
        return b"%PDF-fixture"

    monkeypatch.setattr(download, "_STRATEGIES", [("arxiv", download._try_arxiv),
                                                   ("doi-fixture", doi_download)])
    monkeypatch.setattr(download, "_verify_pdf_matches_metadata", lambda *_: (True, "llm_match: fixture"))
    assert download.download_paper(ordinary, library)
    if scenario == "secondary_missing":
        assert [call.request.url for call in responses.calls] == ["https://arxiv.org/abs/1311.9999"]
    events = [json.loads(line) for line in library.manifest_path.read_text().splitlines()]
    publication = next(event for event in events if event["event"] == "downloaded")
    mismatch = scenario == "canonical_changed"
    assert publication["identity_matches"] is not mismatch
    assert publication["identity_verified"] is True
    assert any(event.get("phase") == "verified-success" for event in events) is not mismatch

    before = {path.name: path.read_bytes() for path in library.root.glob("*") if path.is_file()}
    scope_events = accounting.snapshot(scope).events
    findings = await evaluate_round(library, ledger=ledger, accounting=accounting)
    if mismatch:
        assert [finding.rule for finding in findings] == ["identity_mismatched_publication"]
        assert ledger.campaign_halted(token.campaign_id)
        assert Library(library.root).get(ordinary.key).acquisition_state.quarantine
    else:
        assert findings == ()  # All three detectors run against clean fixture state.
        assert not ledger.campaign_halted(token.campaign_id)
        assert not ordinary.acquisition_state.quarantine
        assert ordinary.doi == "10.1234/ordinary"
        assert {path.name: path.read_bytes() for path in library.root.glob("*") if path.is_file()} == before
        assert not any(record.name == "papervault.library.stop_rules" for record in caplog.records)
    assert accounting.snapshot(scope).events == scope_events
    assert accounting.snapshot(scope).cooldown_until == 0


async def test_ordinary_executor_collision_is_detected_without_merge_flag(campaign, monkeypatch):
    import asyncio
    import threading
    from papervault.library import download
    from papervault.library.stop_rules import evaluate_round
    library, paper, ledger, _, _ = campaign
    monkeypatch.delenv("PAPERVAULT_MERGE_V2", raising=False)
    token = await reserve(campaign)
    other, _ = library.upsert({"title": "An ordinary active duplicate identity",
                               "arxiv_id": "2601.00003"})
    other.doi = paper.doi
    other.terminal_state = TerminalState(retrieval=Retrieval.ELIGIBLE, policy=AcquisitionPolicy.CLEAR)
    started = threading.Event()
    release = threading.Event()
    def blocked(candidate):
        started.set()
        assert release.wait(2)
        return None
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", blocked)])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    task = asyncio.create_task(asyncio.to_thread(download.download_paper, other, library))
    try:
        for _ in range(1000):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()
        findings = await evaluate_round(library, ledger=ledger, accounting=None)
        assert findings[0].rule == "duplicate_inflight_identity"
        assert ledger.campaign_halted(token.campaign_id)
    finally:
        release.set()
        await task


async def test_publication_arriving_during_evaluation_is_retained(campaign, monkeypatch):
    from papervault.library.stop_rules import evaluate_round
    library, paper, ledger, _, _ = campaign
    token = await reserve(campaign)
    other, _ = library.upsert({"title": "Publication observed while an actuator commits",
                               "doi": "10.1234/late"})
    library.log({"event": "reopen_complete", "key": paper.key, "identity_matches": False})
    halt = ledger.halt_campaign
    added = []
    def with_concurrent_observation(campaign_id, reason):
        halt(campaign_id, reason)
        if not added:
            added.append(True)
            library.log({"event": "downloaded", "key": other.key, "identity_matches": False})
    monkeypatch.setattr(ledger, "halt_campaign", with_concurrent_observation)
    await evaluate_round(library, ledger=ledger, accounting=None)
    assert paper.acquisition_state.quarantine
    await evaluate_round(library, ledger=ledger, accounting=None)
    assert other.acquisition_state.quarantine
    assert ledger.campaign_halted(token.campaign_id)


async def test_short_window_violation_survives_delayed_round_and_send_pruning(campaign):
    from papervault.library.stop_rules import evaluate_round
    library, _, ledger, _, now = campaign
    token = await reserve(campaign)
    scope = Scope("fixture")
    accounting = Ledger({scope: Policy((Window(1, 1),), quiet_seconds=30)}, clock=lambda: now[0])
    ticket = accounting.note_sent(scope)
    accounting._states[scope].sends.append(now[0])  # injected accounting violation
    now[0] += 2
    accounting.note_result(ticket, status_code=200)  # normal refresh prunes the short window
    findings = await evaluate_round(library, ledger=ledger, accounting=accounting)
    assert findings[0].rule == "quota_window_violation"
    assert ledger.campaign_halted(token.campaign_id)
    assert accounting.gate(scope).reason == "cooldown"
    now[0] += 30
    assert await evaluate_round(library, ledger=ledger, accounting=accounting) == ()
    assert accounting.gate(scope).allowed


async def test_quota_violation_during_existing_cooldown_is_acknowledged(campaign):
    from papervault.library.stop_rules import evaluate_round
    library, _, ledger, _, now = campaign
    token = await reserve(campaign)
    scope = Scope("fixture")
    accounting = Ledger({scope: Policy((Window(1, 1),), quiet_seconds=30)}, clock=lambda: now[0])
    ticket = accounting.note_sent(scope)
    accounting.note_response(ticket, status_code=429)
    deadline = accounting.snapshot(scope).cooldown_until
    accounting._states[scope].sends.append(now[0])
    now[0] += 1
    await evaluate_round(library, ledger=ledger, accounting=accounting)
    assert ledger.campaign_halted(token.campaign_id)
    assert accounting.snapshot(scope).cooldown_until == deadline
    assert await evaluate_round(library, ledger=ledger, accounting=accounting) == ()
    assert accounting.snapshot(scope).events[-1].kind == "quota_window_violation"


async def test_existing_ordinary_executor_remains_visible_after_claim_adoption(campaign):
    from papervault.library.stop_rules import evaluate_round
    library, paper, ledger, _, _ = campaign
    token = await reserve(campaign)
    # Tracking was acquired before claim adoption; current row ownership cannot
    # erase the still-running ordinary executor's resource count.
    library._stoprule_active = {paper.key: 1}
    findings = await evaluate_round(library, ledger=ledger, accounting=None)
    assert findings[0].rule == "duplicate_inflight_identity"
    assert ledger.campaign_halted(token.campaign_id)


async def test_round_accounting_failure_denies_work_without_stranding_queue(campaign, monkeypatch):
    import asyncio
    from papervault.library.services import download_queue
    library, paper, ledger, _, _ = campaign
    other, _ = library.upsert({"title": "A second queued acquisition after accounting failure",
                               "doi": "10.1234/unavailable"})
    paper.reopen_claim = None
    paper.download_status = "pending"
    ledger.path.write_bytes(b"{")
    def forbidden_send(*args, **kwargs):
        raise AssertionError("accounting failure emitted an upstream request")
    monkeypatch.setattr(download_queue, "download_paper", forbidden_send)
    queue = download_queue.DownloadQueue(library, num_workers=1)
    queue.add(paper.key)
    queue.add(other.key)
    await queue.start(recover=False)
    try:
        await asyncio.wait_for(queue._queue.join(), 1)
        assert not queue._workers[0].done()
    finally:
        await queue.stop()
