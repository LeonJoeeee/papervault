"""Operator recovery exercises real fixture claims and append-only accounting."""

from dataclasses import replace

import pytest

from papervault.library.claims_v2 import BudgetLedger, ClaimStore, LedgerError
from papervault.library.state_v2 import Policy, Retrieval, TerminalState
from papervault.library.store import Library


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    from papervault.library.services import concurrency
    # Contended asyncio locks bind to their test's event loop. Do not leak
    # that loop into other modules exercising the process-global writer lock.
    monkeypatch.setattr(concurrency, "lib_write_lock", concurrency.LibraryWriteLock())
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    library = Library(tmp_path)
    paper, _ = library.upsert({"title": "Operator recovery preserves reservation evidence",
                               "authors": ["Curie"], "doi": "10.1234/operator-recovery"})
    paper.download_status = "failed"
    paper.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
    library.save(force=True, durable=True)
    ledger = BudgetLedger(tmp_path)
    store = ClaimStore(library, ledger=ledger)
    return library, paper, ledger, store


async def reserve(recovery):
    _, paper, ledger, store = recovery
    campaign = ledger.start_campaign(cap=1, keys=[paper.key])
    token = await store.claim(paper.key, expected_state=paper.acquisition_state,
                             expected_generation=0, campaign_id=campaign)
    assert token is not None
    return token


async def operate(library, **kwargs):
    from papervault.library.services.lifecycle_recovery import operate as recover

    return await recover(library, reason="owner verified executor disposition", actor="leon", **kwargs)


def operator_event(library):
    return Library(library.root).corrections.read()["operator_events"][-1]


async def test_queued_recovery_refunds_only_current_unsent_attempt_and_audits(recovery):
    library, paper, ledger, _ = recovery
    token = await reserve(recovery)
    result = await operate(library, action="claim-recover", key=paper.key, lease_id=token.lease_id)
    assert result["changed"] is True
    assert Library(library.root).get(paper.key).reopen_claim.phase == "not_sent"
    assert ledger.spent(token.campaign_id) == 0
    row = operator_event(library)
    assert row["action"] == "claim-recover"
    assert row["event"] == "claim_recover"
    assert row["actor"] == "leon"
    assert row["reason"] == "owner verified executor disposition"
    assert row["keys"] == [paper.key]
    assert not ledger.admitted(token)


async def test_dispatched_recovery_without_executor_proof_never_refunds(recovery):
    library, paper, ledger, store = recovery
    token = await reserve(recovery)
    assert await store.pre_dispatch(token)
    result = await operate(library, action="claim-recover", key=paper.key)
    assert result["changed"] is False
    assert ledger.spent(token.campaign_id) == 1
    assert paper.reopen_claim.phase == "dispatched"
    assert operator_event(library)["event"] == "claim_recover"


async def test_dispatched_recovery_accepts_explicit_executor_proof(recovery):
    library, paper, ledger, store = recovery
    token = await reserve(recovery)
    assert await store.pre_dispatch(token)
    result = await operate(library, action="claim-recover", key=paper.key,
                           executor_proof=True, lease_id=token.lease_id)
    assert result["changed"] is True
    assert ledger.spent(token.campaign_id) == 0
    assert paper.reopen_claim.phase == "not_sent"


@pytest.mark.parametrize("action", ["claim-recover", "release-quarantine"])
async def test_mismatched_lease_cannot_recover_current_attempt(recovery, action):
    library, paper, ledger, store = recovery
    token = await reserve(recovery)
    await store.quarantine(token, reason="unknown outcome")
    before = paper.model_dump()
    with pytest.raises(ValueError, match="lease"):
        await operate(library, action=action, key=paper.key, lease_id="stale-lease", executor_proof=True)
    assert paper.model_dump() == before
    assert ledger.spent(token.campaign_id) == 1


async def test_quarantine_human_disposition_fences_without_refund_or_policy_grant(recovery):
    library, paper, ledger, store = recovery
    token = await reserve(recovery)
    await store.quarantine(token, reason="executor uncertain")
    paper.terminal_state = replace(paper.acquisition_state, policy=Policy.UNKNOWN)
    result = await operate(library, action="release-quarantine", key=paper.key, lease_id=token.lease_id)
    assert result["changed"] is True
    persisted = Library(library.root).get(paper.key)
    assert persisted.acquisition_state.quarantine is False
    assert persisted.acquisition_state.policy == Policy.UNKNOWN
    assert persisted.reopen_claim.phase == "released"
    assert persisted.reopen_claim.generation == 2
    assert ledger.spent(token.campaign_id) == 1
    assert ledger.events()[-1]["event"] == "fence"
    assert ledger.events()[-1]["evidence"] == "owner verified executor disposition"
    assert operator_event(library)["event"] == "release_quarantine"


async def test_campaign_halt_denies_new_work_without_refund_or_finish(recovery):
    library, paper, ledger, _ = recovery
    token = await reserve(recovery)
    result = await operate(library, action="campaign-halt", campaign_id=token.campaign_id)
    assert result["changed"] is True
    assert ledger.campaign_halted(token.campaign_id)
    assert ledger.spent(token.campaign_id) == 1
    assert not ledger.admitted(token)
    assert not any(row["event"] == "campaign_finish" for row in ledger.events())
    row = operator_event(library)
    assert row["keys"] == [paper.key]
    assert row["event"] == "campaign_halt"


async def test_drain_flag_cannot_finish_campaign_with_outstanding_reservation(recovery):
    library, _, ledger, _ = recovery
    token = await reserve(recovery)
    with pytest.raises(LedgerError, match="outstanding"):
        await operate(library, action="campaign-halt", campaign_id=token.campaign_id, drained=True)
    assert ledger.campaign_halted(token.campaign_id)
    assert ledger.spent(token.campaign_id) == 1
    assert not any(row["event"] == "campaign_finish" for row in ledger.events())


async def test_drain_flag_cannot_finish_with_unsettled_record_claim(recovery):
    library, _, ledger, _ = recovery
    token = await reserve(recovery)
    assert ledger.refund_if_not_sent(token, proven_not_sent=True)
    assert ledger.outstanding([token.key]) == []
    with pytest.raises(LedgerError, match="claim"):
        await operate(library, action="campaign-halt", campaign_id=token.campaign_id, drained=True)
    assert not any(row["event"] == "campaign_finish" for row in ledger.events())


async def test_settled_campaign_finishes_only_after_explicit_drain(recovery):
    library, paper, ledger, store = recovery
    token = await reserve(recovery)
    assert await store.not_sent(token)
    first = await operate(library, action="campaign-halt", campaign_id=token.campaign_id)
    assert first["finished"] is False
    second = await operate(library, action="campaign-halt", campaign_id=token.campaign_id, drained=True)
    assert second["finished"] is True
    assert ledger.events()[-1]["event"] == "campaign_finish"
    assert ledger.spent(token.campaign_id) == 0
    assert operator_event(library)["keys"] == [paper.key]


@pytest.mark.parametrize("field", ["reason", "actor"])
async def test_recovery_requires_nonempty_operator_identity_and_reason(recovery, field):
    from papervault.library.services.lifecycle_recovery import operate as recover

    library, paper, ledger, _ = recovery
    token = await reserve(recovery)
    values = {"reason": "reviewed", "actor": "leon", field: " "}
    with pytest.raises(ValueError, match="actor|reason"):
        await recover(library, action="claim-recover", key=paper.key, **values)
    assert ledger.spent(token.campaign_id) == 1
    assert paper.reopen_claim.phase == "queued"


async def test_recovery_rejects_missing_record_before_mutation(recovery):
    library, _, ledger, _ = recovery
    await reserve(recovery)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="record|key"):
        await operate(library, action="claim-recover", key="missing")
    assert ledger.path.read_bytes() == before


async def test_ledger_only_campaign_halt_flushes_staged_evidence_durably(recovery, monkeypatch):
    library, _, _, _ = recovery
    token = await reserve(recovery)
    saves = []
    save = library.save

    def record_save(*args, **kwargs):
        saves.append(kwargs)
        return save(*args, **kwargs)

    monkeypatch.setattr(library, "save", record_save)
    await operate(library, action="campaign-halt", campaign_id=token.campaign_id)
    assert saves == [{"force": True, "durable": True}]


async def test_quarantine_release_rechecks_lease_after_waiting_for_writer_lock(recovery):
    import asyncio

    library, paper, ledger, store = recovery
    token = await reserve(recovery)
    await store.quarantine(token, reason="unknown executor")
    async with store.lock:
        task = asyncio.create_task(operate(library, action="release-quarantine", key=paper.key,
                                            lease_id=token.lease_id))
        await asyncio.sleep(0)  # The operator now waits on ClaimStore's writer lock.
        paper.reopen_claim = token.model_copy(update={"lease_id": "replacement-lease", "generation": 2})
    result = await task
    assert result["changed"] is False
    assert paper.acquisition_state.quarantine is True
    assert paper.reopen_claim.lease_id == "replacement-lease"
    assert ledger.spent(token.campaign_id) == 1
    assert not any(row["event"] == "fence" for row in ledger.events())


async def test_recovery_reconciles_refund_committed_before_interrupted_record_save(recovery, monkeypatch):
    library, paper, ledger, _ = recovery
    token = await reserve(recovery)
    with monkeypatch.context() as patch:
        def fail(**kwargs):
            raise OSError("record save interrupted")
        patch.setattr(library, "save", fail)
        with pytest.raises(OSError, match="interrupted"):
            await operate(library, action="claim-recover", key=paper.key)
    assert ledger.phase(token) == "refund" and ledger.spent(token.campaign_id) == 0
    fresh = Library(library.root)
    assert fresh.get(paper.key).reopen_claim.phase == "queued"
    result = await operate(fresh, action="claim-recover", key=paper.key)
    assert result["changed"] is True
    assert Library(library.root).get(paper.key).reopen_claim.phase == "not_sent"
    assert sum(row["event"] == "refund" for row in ledger.events()) == 1
    result = await operate(fresh, action="campaign-halt", campaign_id=token.campaign_id, drained=True)
    assert result["finished"] is True


async def orphan_reservation(recovery, *, dispatched=False):
    """Restore the durable record from immediately before claim persistence."""
    library, paper, ledger, store = recovery
    before = library.index_path.read_bytes()
    token = await reserve(recovery)
    if dispatched:
        assert await store.pre_dispatch(token)
    library.index_path.write_bytes(before)
    return Library(library.root), token


async def test_claim_recover_reconciles_reservation_without_record_claim(recovery):
    _, paper, ledger, _ = recovery
    fresh, token = await orphan_reservation(recovery)
    assert fresh.get(paper.key).reopen_claim is None
    result = await operate(fresh, action="claim-recover", key=paper.key, lease_id=token.lease_id)
    assert result["changed"] is True
    persisted = Library(fresh.root).get(paper.key)
    assert persisted.reopen_claim.lease_id == token.lease_id
    assert persisted.reopen_claim.phase == "not_sent"
    assert ledger.spent(token.campaign_id) == 0
    assert ledger.outstanding([paper.key]) == []
    again = await operate(fresh, action="claim-recover", key=paper.key)
    assert again["changed"] is False
    assert sum(row["event"] == "refund" for row in ledger.events()) == 1


async def test_orphan_dispatch_requires_executor_proof_before_refund(recovery):
    _, paper, ledger, _ = recovery
    fresh, token = await orphan_reservation(recovery, dispatched=True)
    result = await operate(fresh, action="claim-recover", key=paper.key, lease_id=token.lease_id)
    assert result["changed"] is False
    assert ledger.spent(token.campaign_id) == 1
    assert fresh.get(paper.key).reopen_claim is None
    result = await operate(fresh, action="claim-recover", key=paper.key, executor_proof=True)
    assert result["changed"] is True
    assert ledger.spent(token.campaign_id) == 0
    assert Library(fresh.root).get(paper.key).reopen_claim.phase == "not_sent"


async def test_claim_recover_reconciles_completed_record_without_refund(recovery):
    library, paper, ledger, store = recovery
    token = await reserve(recovery)
    assert await store.pre_dispatch(token)
    paper.reopen_claim = paper.reopen_claim.model_copy(update={"phase": "completed"})
    paper.download_status = "ok"
    library.save(force=True, durable=True)
    fresh = Library(library.root)
    result = await operate(fresh, action="claim-recover", key=paper.key, executor_proof=True)
    assert result["changed"] is True
    assert ledger.phase(token) == "complete"
    assert ledger.spent(token.campaign_id) == 1
    assert fresh.get(paper.key).download_status == "ok"
    assert fresh.get(paper.key).reopen_claim.phase == "completed"
    assert not any(row["event"] == "refund" for row in ledger.events())
    assert (await operate(fresh, action="claim-recover", key=paper.key))["changed"] is False


@pytest.mark.parametrize("completed", [False, True])
async def test_campaign_drain_reconciles_only_orphaned_or_completed_work(recovery, completed):
    library, paper, ledger, store = recovery
    if completed:
        token = await reserve(recovery)
        assert await store.pre_dispatch(token)
        paper.reopen_claim = paper.reopen_claim.model_copy(update={"phase": "completed"})
        library.save(force=True, durable=True)
        fresh = Library(library.root)
    else:
        fresh, token = await orphan_reservation(recovery)
    result = await operate(fresh, action="campaign-halt", campaign_id=token.campaign_id, drained=True)
    assert result["finished"] is True
    assert ledger.outstanding([paper.key]) == []
    assert ledger.spent(token.campaign_id) == (1 if completed else 0)


async def test_campaign_drain_cannot_refund_orphaned_dispatched_work(recovery):
    _, paper, ledger, _ = recovery
    fresh, token = await orphan_reservation(recovery, dispatched=True)
    with pytest.raises(LedgerError, match="outstanding"):
        await operate(fresh, action="campaign-halt", campaign_id=token.campaign_id, drained=True)
    assert ledger.spent(token.campaign_id) == 1
    assert ledger.phase(token) == "dispatch"
    assert fresh.get(paper.key).reopen_claim is None


async def test_orphan_quarantine_release_fences_without_refund(recovery):
    _, paper, ledger, _ = recovery
    fresh, token = await orphan_reservation(recovery, dispatched=True)
    orphan = fresh.get(paper.key)
    orphan.terminal_state = replace(orphan.acquisition_state, quarantine=True, policy=Policy.UNKNOWN)
    fresh.save(force=True, durable=True)
    result = await operate(fresh, action="release-quarantine", key=paper.key, lease_id=token.lease_id)
    assert result["changed"] is True
    persisted = Library(fresh.root).get(paper.key)
    assert persisted.reopen_claim.generation == token.generation + 1
    assert persisted.reopen_claim.phase == "released"
    assert persisted.acquisition_state.quarantine is False
    assert persisted.acquisition_state.policy == Policy.UNKNOWN
    assert ledger.spent(token.campaign_id) == 1
    assert ledger.outstanding([paper.key]) == []
    assert not any(row["event"] == "refund" for row in ledger.events())


async def test_recovery_rejects_orphan_lease_or_campaign_mismatch(recovery):
    _, paper, ledger, _ = recovery
    fresh, token = await orphan_reservation(recovery)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="lease"):
        await operate(fresh, action="claim-recover", key=paper.key, lease_id="wrong")
    with pytest.raises(ValueError, match="campaign"):
        await operate(fresh, action="claim-recover", key=paper.key, campaign_id="wrong")
    assert ledger.path.read_bytes() == before
    assert fresh.get(paper.key).reopen_claim is None
    assert ledger.spent(token.campaign_id) == 1


async def test_recovery_prefers_new_orphan_reservation_over_old_settled_claim(recovery):
    library, paper, ledger, store = recovery
    first = await reserve(recovery)
    assert await store.not_sent(first)
    ledger.finish_campaign(first.campaign_id, drained=True)
    before = library.index_path.read_bytes()
    campaign = ledger.start_campaign(cap=1, keys=[paper.key])
    token = await store.claim(paper.key, expected_state=paper.acquisition_state,
                             expected_generation=1, campaign_id=campaign)
    assert token is not None
    library.index_path.write_bytes(before)
    fresh = Library(library.root)
    result = await operate(fresh, action="claim-recover", key=paper.key, lease_id=token.lease_id)
    assert result["changed"] is True
    assert ledger.spent(campaign) == 0
    assert Library(library.root).get(paper.key).reopen_claim.generation == 2


async def test_orphan_recovery_replays_refund_after_interrupted_record_save(recovery, monkeypatch):
    _, paper, ledger, _ = recovery
    fresh, token = await orphan_reservation(recovery)
    with monkeypatch.context() as patch:
        def fail(**kwargs):
            raise OSError("record save interrupted")
        patch.setattr(fresh, "save", fail)
        with pytest.raises(OSError, match="interrupted"):
            await operate(fresh, action="claim-recover", key=paper.key)
    assert ledger.phase(token) == "refund"
    replay = Library(fresh.root)
    assert replay.get(paper.key).reopen_claim is None
    assert (await operate(replay, action="claim-recover", key=paper.key))["changed"] is True
    assert Library(fresh.root).get(paper.key).reopen_claim.phase == "not_sent"
    assert sum(row["event"] == "refund" for row in ledger.events()) == 1


async def test_release_quarantine_settles_completed_ledger_gap_without_refund(recovery):
    library, paper, ledger, store = recovery
    token = await reserve(recovery)
    assert await store.pre_dispatch(token)
    paper.reopen_claim = paper.reopen_claim.model_copy(update={"phase": "completed"})
    paper.download_status = "ok"
    paper.terminal_state = replace(paper.acquisition_state, quarantine=True)
    library.save(force=True, durable=True)
    result = await operate(library, action="release-quarantine", key=paper.key)
    assert result["changed"] is True
    assert any(row["event"] == "complete" for row in ledger.events())
    assert not any(row["event"] == "refund" for row in ledger.events())
    assert ledger.spent(token.campaign_id) == 1
    assert ledger.outstanding([paper.key]) == []
    assert paper.reopen_claim.phase == "released"
    assert paper.download_status == "ok"


async def test_completed_recovery_requires_full_reservation_identity(recovery):
    library, paper, ledger, store = recovery
    token = await reserve(recovery)
    assert await store.pre_dispatch(token)
    paper.reopen_claim = paper.reopen_claim.model_copy(update={"phase": "completed", "generation": 99})
    library.save(force=True, durable=True)
    result = await operate(library, action="claim-recover", key=paper.key)
    assert result["changed"] is False
    assert ledger.phase(token) == "dispatch"
    assert ledger.spent(token.campaign_id) == 1


async def test_orphan_recovery_preserves_independent_acquired_record_state(recovery):
    _, paper, ledger, _ = recovery
    fresh, token = await orphan_reservation(recovery)
    current = fresh.get(paper.key)
    current.download_status = "ok"
    current.terminal_state = replace(current.acquisition_state, retrieval=Retrieval.ACQUIRED,
                                     policy=Policy.UNKNOWN)
    before = current.acquisition_state
    fresh.save(force=True, durable=True)
    result = await operate(fresh, action="claim-recover", key=paper.key)
    assert result["changed"] is True
    assert current.acquisition_state == before
    assert current.download_status == "ok"
    assert ledger.spent(token.campaign_id) == 0


async def test_orphan_release_replays_fence_after_interrupted_record_save(recovery, monkeypatch):
    _, paper, ledger, _ = recovery
    fresh, token = await orphan_reservation(recovery, dispatched=True)
    current = fresh.get(paper.key)
    current.terminal_state = replace(current.acquisition_state, quarantine=True)
    fresh.save(force=True, durable=True)
    with monkeypatch.context() as patch:
        def fail(**kwargs):
            raise OSError("release record save interrupted")
        patch.setattr(fresh, "save", fail)
        with pytest.raises(OSError, match="interrupted"):
            await operate(fresh, action="release-quarantine", key=paper.key)
    replay = Library(fresh.root)
    assert replay.get(paper.key).reopen_claim is None
    assert ledger.latest_generation(paper.key) == token.generation + 1
    result = await operate(replay, action="release-quarantine", key=paper.key, lease_id=token.lease_id)
    assert result["changed"] is True
    persisted = Library(fresh.root).get(paper.key)
    assert persisted.reopen_claim.phase == "released"
    assert persisted.reopen_claim.generation == token.generation + 1
    assert persisted.acquisition_state.quarantine is False
    assert ledger.spent(token.campaign_id) == 1
    assert sum(row["event"] == "fence" for row in ledger.events()) == 1


async def test_completed_quarantine_release_requires_full_reservation_identity(recovery):
    library, paper, ledger, store = recovery
    token = await reserve(recovery)
    assert await store.pre_dispatch(token)
    paper.reopen_claim = paper.reopen_claim.model_copy(update={"phase": "completed", "generation": 99})
    paper.terminal_state = replace(paper.acquisition_state, quarantine=True)
    library.save(force=True, durable=True)
    before = ledger.path.read_bytes()
    result = await operate(library, action="release-quarantine", key=paper.key)
    assert result["changed"] is False
    assert ledger.path.read_bytes() == before
    assert paper.acquisition_state.quarantine is True


async def test_campaign_drain_replays_orphan_refund_after_interrupted_record_save(recovery, monkeypatch):
    _, paper, ledger, _ = recovery
    fresh, token = await orphan_reservation(recovery)
    with monkeypatch.context() as patch:
        def fail(**kwargs):
            raise OSError("drain record save interrupted")
        patch.setattr(fresh, "save", fail)
        with pytest.raises(OSError, match="interrupted"):
            await operate(fresh, action="campaign-halt", campaign_id=token.campaign_id, drained=True)
    assert ledger.phase(token) == "refund"
    replay = Library(fresh.root)
    assert replay.get(paper.key).reopen_claim is None
    result = await operate(replay, action="campaign-halt", campaign_id=token.campaign_id, drained=True)
    assert result["finished"] is True
    persisted = Library(fresh.root).get(paper.key)
    assert persisted.reopen_claim is not None
    assert persisted.reopen_claim.generation == token.generation
    assert persisted.reopen_claim.phase == "not_sent"
    assert sum(row["event"] == "refund" for row in ledger.events()) == 1
    assert ledger.spent(token.campaign_id) == 0
