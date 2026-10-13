"""Durable A3 contracts, using fixture vaults and a frozen wall clock."""

import asyncio
import json
from dataclasses import replace

import pytest

from papervault.library.models import Paper
from papervault.library.state_v2 import Policy, Retrieval, TerminalState, admission
from papervault.library.store import Library


@pytest.fixture
def machinery(tmp_path, monkeypatch):
    from papervault.library.claims_v2 import BudgetLedger, ClaimStore
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    now = [1791849600.0]  # 2026-10-13 UTC
    lib = Library(tmp_path)
    paper, _ = lib.upsert({"title": "Durable re-open correctness fixture paper",
                           "authors": ["Curie"], "doi": "10.1234/claims"})
    paper.download_status = "failed"
    paper.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
    lib.save()
    ledger = BudgetLedger(tmp_path, clock=lambda: now[0])
    store = ClaimStore(lib, ledger=ledger, clock=lambda: now[0])
    return lib, paper, ledger, store, now


async def claim_one(machinery, ttl=60):
    _, paper, ledger, store, _ = machinery
    campaign = ledger.start_campaign(cap=1, keys=[paper.key])
    token = await store.claim(paper.key, expected_state=paper.acquisition_state,
                              expected_generation=0, campaign_id=campaign, ttl=ttl)
    assert token is not None
    return token


async def test_claim_cas_single_winner_and_durable_reservation(machinery):
    lib, paper, ledger, store, _ = machinery
    state = paper.acquisition_state
    campaign = ledger.start_campaign(cap=1, keys=[paper.key])
    tokens = await asyncio.gather(*[
        store.claim(paper.key, expected_state=state, expected_generation=0,
                    campaign_id=campaign, ttl=60) for _ in range(8)])
    winners = [token for token in tokens if token is not None]
    assert len(winners) == 1
    token = winners[0]
    assert token.generation == 1
    assert paper.acquisition_state.retrieval == Retrieval.ELIGIBLE
    assert Library(lib.root).get(paper.key).reopen_claim == token
    assert ledger.admitted(token)
    assert ledger.spent(campaign) == 1


async def test_expiry_quarantines_without_changing_dimensions_or_refunding(machinery):
    lib, paper, ledger, store, now = machinery
    token = await claim_one(machinery)
    assert await store.pre_dispatch(token)
    before = paper.acquisition_state
    now[0] += 60
    assert await store.expire(paper.key)
    assert paper.acquisition_state == replace(before, quarantine=True)
    assert not admission(paper.acquisition_state, key=paper.key, identity_resolved=True)
    assert not await store.release_quarantine(paper.key)
    assert ledger.spent(token.campaign_id) == 1
    assert await store.release_quarantine(paper.key, drained_lease_id=token.lease_id)
    persisted = Library(lib.root).get(paper.key)
    assert not persisted.acquisition_state.quarantine
    assert persisted.acquisition_state.policy == Policy.CLEAR
    assert persisted.acquisition_state.retrieval == Retrieval.EXHAUSTED
    assert persisted.reopen_claim.generation == 2
    assert not await store.pre_dispatch(token)


async def test_human_release_is_recorded_and_does_not_refund(machinery):
    lib, paper, ledger, store, _ = machinery
    token = await claim_one(machinery)
    await store.quarantine(token, reason="executor outcome unknown")
    assert await store.release_quarantine(paper.key, human_reason="owner fenced old executor")
    assert ledger.spent(token.campaign_id) == 1
    assert "owner fenced old executor" in ledger.path.read_text()
    assert not Library(lib.root).get(paper.key).acquisition_state.quarantine


async def test_refund_only_proven_not_sent_and_idempotent(machinery):
    _, _, ledger, store, _ = machinery
    token = await claim_one(machinery)
    assert not ledger.refund_if_not_sent(token, proven_not_sent=False)
    assert ledger.spent(token.campaign_id) == 1
    assert await store.not_sent(token)
    original = ledger.path.read_bytes()
    assert ledger.spent(token.campaign_id) == 0
    assert not await store.not_sent(token)
    assert ledger.path.read_bytes() == original
    assert not ledger.admitted(token)


async def test_dispatched_restart_never_resends_or_refunds(machinery):
    from papervault.library.claims_v2 import BudgetLedger, ClaimStore
    lib, paper, ledger, store, now = machinery
    token = await claim_one(machinery)
    assert await store.pre_dispatch(token)
    replay = ClaimStore(Library(lib.root), ledger=BudgetLedger(lib.root, clock=lambda: now[0]),
                        clock=lambda: now[0])
    assert not await replay.pre_dispatch(token)
    assert replay.library.get(paper.key).acquisition_state.quarantine
    assert ledger.spent(token.campaign_id) == 1
    assert not ledger.refund_if_not_sent(token, proven_not_sent=False)


async def test_index_rollback_cannot_reuse_generation_or_send(machinery):
    lib, paper, ledger, store, _ = machinery
    index_before = lib.index_path.read_bytes()
    token = await claim_one(machinery)
    lib.index_path.write_bytes(index_before)  # injected restoration of old snapshot
    from papervault.library.claims_v2 import ClaimStore
    rolled_back = ClaimStore(Library(lib.root), ledger=ledger, clock=store.clock)
    assert await rolled_back.claim(paper.key, expected_state=paper.acquisition_state,
                                   expected_generation=0, campaign_id=token.campaign_id) is None
    assert ledger.spent(token.campaign_id) == 1


def test_standing_paper_cap_and_week_cap_survive_restart(tmp_path):
    from papervault.library.claims_v2 import BudgetLedger, LedgerError, ReopenClaim
    now = [1791849600.0]
    ledger = BudgetLedger(tmp_path, clock=lambda: now[0])
    with pytest.raises(ValueError):
        ledger.start_campaign(cap=201, keys=[])
    campaign = ledger.start_campaign(cap=200, keys=[str(n) for n in range(201)])
    for n in range(200):
        token = ReopenClaim(key=str(n), generation=1, lease_id=f"lease-{n}",
                            lease_expiry=now[0] + 60, campaign_id=campaign)
        ledger.reserve(token)
    with pytest.raises(LedgerError):
        ledger.reserve(ReopenClaim(key="200", generation=1, lease_id="lease-200",
                                   lease_expiry=now[0] + 60, campaign_id=campaign))
    ledger.finish_campaign(campaign, drained=True)
    for _ in range(2):
        campaign = ledger.start_campaign(cap=1, keys=["K"])
        ledger.finish_campaign(campaign, drained=True)
    restarted = BudgetLedger(tmp_path, clock=lambda: now[0])
    with pytest.raises(LedgerError, match="three"):
        restarted.start_campaign(cap=1, keys=["K"])
    now[0] += 7 * 86400
    assert restarted.start_campaign(cap=1, keys=["K"])


def test_active_campaign_blocks_even_dry_run(machinery):
    _, paper, ledger, _, _ = machinery
    ledger.start_campaign(cap=1, keys=[paper.key])
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="active"):
        ledger.check_campaign(cap=1)
    assert ledger.path.read_bytes() == before


@pytest.mark.parametrize("corrupt", [b"{", b"{}\n", b"", b'{"version":99}\n'])
async def test_unreadable_ledger_denies_dispatch_without_refund(machinery, corrupt):
    _, paper, ledger, store, _ = machinery
    token = await claim_one(machinery)
    ledger.path.write_bytes(corrupt)
    assert not ledger.admitted(token)
    with pytest.raises(ValueError):
        await store.pre_dispatch(token)
    assert paper.acquisition_state.retrieval == Retrieval.ELIGIBLE


async def test_claim_requires_strict_durable_index_before_dispatch(machinery, monkeypatch):
    lib, paper, ledger, store, _ = machinery
    campaign = ledger.start_campaign(cap=1, keys=[paper.key])
    import papervault.library.store as store_module
    original_fsync = store_module.os.fsync
    def fsync(fd):
        if store_module.os.readlink(f"/proc/self/fd/{fd}") == str(lib.index_path) + ".tmp":
            raise OSError("disk failure")
        original_fsync(fd)
    monkeypatch.setattr(store_module.os, "fsync", fsync)
    with pytest.raises(OSError, match="disk failure"):
        await store.claim(paper.key, expected_state=paper.acquisition_state,
                          expected_generation=0, campaign_id=campaign)
    assert not lib.pdf_path(paper.key).exists()
    assert ledger.spent(campaign) == 1  # reservation survived the uncertain index write


async def test_replaying_queued_index_after_dispatch_is_quarantined(machinery):
    from papervault.library.claims_v2 import ClaimStore
    lib, paper, ledger, store, _ = machinery
    token = await claim_one(machinery)
    queued_index = lib.index_path.read_bytes()
    assert await store.pre_dispatch(token)
    lib.index_path.write_bytes(queued_index)
    restarted = ClaimStore(Library(lib.root), ledger=ledger, clock=store.clock)
    assert not await restarted.pre_dispatch(token)
    assert restarted.library.get(paper.key).acquisition_state.quarantine
    assert ledger.spent(token.campaign_id) == 1


def test_claim_fields_are_optional_and_never_source_writable(machinery):
    lib, paper, _, _, _ = machinery
    assert "reopen_claim" not in Paper(key="old").model_dump()
    incoming = {"doi": paper.doi, "reopen_claim": {"key": paper.key, "generation": 900,
                "lease_id": "forged", "lease_expiry": 9999999999, "campaign_id": "forged"}}
    lib.upsert(incoming)
    assert paper.reopen_claim is None
    new, _ = lib.upsert({"title": "An independent source cannot inject worker claims",
                          "doi": "10.1234/new", **{k: v for k, v in incoming.items() if k == "reopen_claim"}})
    assert new.reopen_claim is None
    assert "reopen_claim" not in json.loads(paper.model_dump_json())


async def test_missing_ledger_with_persisted_claims_fails_closed(machinery):
    _, paper, ledger, store, _ = machinery
    token = await claim_one(machinery)
    ledger.path.unlink()
    assert not ledger.admitted(token)
    with pytest.raises(ValueError, match="missing"):
        ledger.check_campaign(cap=1)
    with pytest.raises(ValueError, match="missing"):
        await store.pre_dispatch(token)
    assert paper.reopen_claim.phase == "queued"


@pytest.mark.parametrize("retrieval,status", [(Retrieval.ACQUIRED, "ok"), (Retrieval.INELIGIBLE, "pending")])
async def test_quarantine_release_preserves_independent_retrieval(machinery, retrieval, status):
    _, paper, _, store, _ = machinery
    token = await claim_one(machinery)
    await store.quarantine(token, reason="publication fenced")
    paper.download_status = status
    paper.terminal_state = replace(paper.acquisition_state, retrieval=retrieval)
    before = paper.acquisition_state
    assert await store.release_quarantine(paper.key, human_reason="owner fenced executor")
    assert paper.acquisition_state == replace(before, quarantine=False)
    assert paper.download_status == status
