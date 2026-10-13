"""A3's four rollback drills: real APIs, injected failures, disposable vaults.

Rollback disables the v2 paths and restarts a compatible reader. It never rolls
back a ledger or index snapshot: after an irreversible point, A5 supports a
forward fix, and these fixtures prove the retained data stays fenced meanwhile.
"""

import asyncio

import pytest

from papervault.library.claims_v2 import BudgetLedger, ClaimStore, record_generation
from papervault.library.drain_v2 import verify_drained
from papervault.library.merge_v2 import MergeStore, guard_execution, resolve_alias
from papervault.library.reopen_v2 import StagedAttempt
from papervault.library.services import concurrency
from papervault.library.state_v2 import Merge, Policy, Retrieval
from papervault.library.store import Library


async def sealed_intake():
    return True  # fixtures have no external producers or other writer processes


def tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def enable(monkeypatch, enabled):
    for name in ("TERMINAL", "IDENTITY", "REOPEN", "MERGE", "STOPRULES"):
        monkeypatch.setenv(f"PAPERVAULT_{name}_V2", "1" if enabled else "0")


@pytest.fixture
def machinery(tmp_path, monkeypatch):
    enable(monkeypatch, True)
    now = [1791849600.0]
    lib = Library(tmp_path)
    left, _ = lib.upsert({"title": "Rollback canonical identity and durable accounting fixture",
                          "authors": ["Curie"], "doi": "10.1234/rollback",
                          "download_status": "failed",
                          "terminal_state": {"retrieval": "exhausted", "policy": "clear"}})
    right, _ = lib.upsert({"title": "Independent rollback preprint identity fixture",
                           "authors": ["Einstein"], "arxiv_id": "2101.12345",
                           "download_status": "failed",
                           "terminal_state": {"retrieval": "exhausted", "policy": "clear"}})
    lib.save()
    ledger = BudgetLedger(lib.root, clock=lambda: now[0])
    store = ClaimStore(lib, ledger=ledger, clock=lambda: now[0])
    return lib, left, right, ledger, store, now


def restart(lib, now, monkeypatch):
    enable(monkeypatch, False)
    replay = Library(lib.root)
    ledger = BudgetLedger(lib.root, clock=lambda: now[0])
    return replay, ledger, ClaimStore(replay, ledger=ledger, clock=lambda: now[0])


async def reserve(store, paper, campaign, ttl=60):
    token = await store.claim(paper.key, expected_state=paper.acquisition_state,
                              expected_generation=record_generation(paper),
                              campaign_id=campaign, ttl=ttl)
    assert token is not None
    return token


async def test_rollback_after_reopen_retains_identity_and_replays_eligible_transition(machinery, monkeypatch):
    lib, paper, _, ledger, store, now = machinery
    original_state = paper.acquisition_state
    campaign = ledger.start_campaign(cap=2, keys=[paper.key])
    save = store.save
    def crash_after_save():
        save()
        raise OSError("crash after reopen save, before enqueue")
    with monkeypatch.context() as fault:
        fault.setattr(store, "save", crash_after_save)
        with pytest.raises(OSError, match="before enqueue"):
            await reserve(store, paper, campaign)

    replay, accounting, recovered = restart(lib, now, monkeypatch)
    current = replay.get(paper.key)
    token = current.reopen_claim
    assert current.doi == "10.1234/rollback" and current.key == paper.key
    assert current.acquisition_state.retrieval == Retrieval.ELIGIBLE
    assert current.acquisition_state.policy == Policy.CLEAR
    assert token.generation == 1 and accounting.spent(campaign) == 1
    assert accounting.events()[1]["transition"] == {"from": "exhausted", "to": "eligible"}
    assert not (await verify_drained(replay, stop_intake=sealed_intake, execution_tracking_verified=True, ledger=accounting)).drained
    before = tree_bytes(lib.root)
    with pytest.raises(ValueError, match="disabled"):
        await recovered.claim(paper.key, expected_state=original_state,
                               expected_generation=0, campaign_id=campaign)
    assert tree_bytes(lib.root) == before

    enable(monkeypatch, True)
    # Re-reading the persisted queued claim replays the exhausted->eligible
    # transition once; CAS cannot create another reservation for the old state.
    assert await recovered.claim(paper.key, expected_state=original_state,
                                  expected_generation=0, campaign_id=campaign) is None
    from papervault.library.services import download_queue
    # Inject an offline retrieval miss at the worker boundary. The real queue,
    # dispatch, detached staging, publication and completion paths still run.
    def retrieval_miss(candidate, staging, **_):
        assert candidate.key == paper.key and staging.root != replay.root
        candidate.download_status = "failed"
        return False
    monkeypatch.setattr(download_queue, "download_paper", retrieval_miss)
    queue = download_queue.DownloadQueue(replay, num_workers=1)
    queue._reopen_clock = lambda: now[0]
    queue.add_reopen(token)
    queued = await verify_drained(replay, stop_intake=sealed_intake,
                                  execution_tracking_verified=True, ledger=accounting)
    assert not queued.drained and queued.queued == queued.unfinished == 1
    try:
        await queue.start(recover=False)
        await asyncio.wait_for(queue._queue.join(), timeout=5)
    finally:
        await queue.stop()
    evidence = await verify_drained(replay, stop_intake=sealed_intake, execution_tracking_verified=True, ledger=accounting)
    assert evidence.drained and evidence.queued == evidence.unfinished == evidence.pending == 0
    assert accounting.spent(campaign) == 1
    assert len([r for r in accounting.events() if r["event"] == "reserve"]) == 1
    persisted = Library(lib.root).get(paper.key)
    assert persisted.doi == current.doi and record_generation(persisted) == 1
    assert persisted.acquisition_state.retrieval == Retrieval.EXHAUSTED
    assert not replay.pdf_path(paper.key).exists()


async def test_rollback_after_merge_preserves_canonical_aliases_restrictions_and_budget(machinery, monkeypatch):
    lib, left, right, ledger, store, now = machinery
    campaign = ledger.start_campaign(cap=4, keys=[left.key, right.key])
    for paper in (left, right):
        token = await reserve(store, paper, campaign)
        assert await store.pre_dispatch(token)
        stage = StagedAttempt(lib, token)
        try:
            result = stage.library.get(paper.key)
            if paper is right:
                stage.path("pdf").write_bytes(b"%PDF retained preprint artifact")
                result.download_status = "ok"
            else:
                result.download_status = "failed"
            assert await stage.writeback(store)
        finally:
            stage.cleanup()
    # Restrictions enter through the metadata writer, not direct state surgery.
    lib.upsert({"doi": left.doi, "terminal_state": {
        "policy": "excluded", "policy_reasons": ["license"]}})
    lib.upsert({"arxiv_id": right.arxiv_id, "terminal_state": {
        "policy": "excluded", "policy_reasons": ["publisher"]}})
    lib.save()
    before_generations = [record_generation(p) for p in (left, right)]
    assert before_generations == [1, 1]
    assert (await verify_drained(lib, stop_intake=sealed_intake, execution_tracking_verified=True, ledger=ledger)).drained
    ledger.finish_campaign(campaign, drained=True)
    merger = MergeStore(lib, ledger=ledger, clock=lambda: now[0])
    save = lib.save
    def crash_after_merge(**kwargs):
        save(**kwargs)
        if left.merge_budget is not None:
            raise OSError("crash after canonical merge publication")
    with monkeypatch.context() as fault:
        fault.setattr(lib, "save", crash_after_merge)
        with pytest.raises(OSError, match="merge publication"):
            await merger.merge_records(left.key, right.key)

    replay, accounting, _ = restart(lib, now, monkeypatch)
    canonical, alias = replay.get(left.key), replay.get(right.key)
    assert canonical.doi == "10.1234/rollback" and canonical.arxiv_id == "2101.12345"
    assert canonical.acquisition_state.merge == Merge.CANONICAL
    assert alias.acquisition_state.merge == Merge.ALIAS and alias.acquisition_state.alias_key == left.key
    assert resolve_alias(replay, right.key) is canonical
    assert replay.find(arxiv_id=right.arxiv_id) is canonical
    assert canonical.acquisition_state.policy == Policy.EXCLUDED
    assert set(canonical.acquisition_state.policy_reasons) == {"license", "publisher"}
    assert record_generation(canonical) == record_generation(alias) == 3
    assert canonical.merge_budget.spent == 2 and canonical.merge_budget.budget == 4
    assert accounting.key_budget(left.key) == {"spent": 2, "budget": 4, "generation": 3}
    assert accounting.spent(campaign) == 2
    assert len([r for r in accounting.events() if r["event"] == "merge"]) == 1
    assert replay.pdf_path(left.key).read_bytes() == b"%PDF retained preprint artifact"
    assert replay.pdf_path(right.key).read_bytes() == b"%PDF retained preprint artifact"
    assert (await verify_drained(replay, stop_intake=sealed_intake, execution_tracking_verified=True, ledger=accounting)).drained


async def test_rollback_after_success_keeps_artifacts_and_fences_stale_generation(machinery, monkeypatch):
    lib, paper, _, ledger, store, now = machinery
    campaign = ledger.start_campaign(cap=2, keys=[paper.key])
    token = await reserve(store, paper, campaign)
    assert await store.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    try:
        result = stage.library.get(paper.key)
        stage.path("pdf").write_bytes(b"%PDF current verified artifact")
        stage.path("md").write_bytes(b"# Current verified extract")
        result.md_engine = "fixture"
        result.download_status = "ok"
        def crash_before_complete(_token):
            raise OSError("crash after artifacts/index, before accounting completion")
        with monkeypatch.context() as fault:
            fault.setattr(ledger, "complete", crash_before_complete)
            with pytest.raises(OSError, match="before accounting completion"):
                await stage.writeback(store)

        replay, accounting, recovered = restart(lib, now, monkeypatch)
        current = replay.get(paper.key)
        assert current.doi == "10.1234/rollback" and current.acquisition_state.policy == Policy.CLEAR
        assert current.acquisition_state.retrieval == Retrieval.VERIFIED
        assert current.reopen_claim.phase == "completed" and record_generation(current) == 1
        assert accounting.phase(token) == "dispatch" and accounting.spent(campaign) == 1
        assert not (await verify_drained(replay, stop_intake=sealed_intake, execution_tracking_verified=True, ledger=accounting)).drained
        # The local executor has exited. The existing recovery API supplies
        # drain evidence and persists a higher generation without a refund.
        await recovered.quarantine(token, reason="completion accounting uncertain")
        assert await recovered.release_quarantine(paper.key, drained_lease_id=token.lease_id)
        assert record_generation(Library(lib.root).get(paper.key)) == 2
        enable(monkeypatch, True)  # show the generation fence, independently of the kill switch
        stage.path("pdf").write_bytes(b"stale executor replacement")
        stage.path("md").write_bytes(b"stale executor extract")
        result.download_status = "failed"
        before = tree_bytes(lib.root)
        assert not await stage.writeback(recovered)
        assert tree_bytes(lib.root) == before
        assert replay.pdf_path(paper.key).read_bytes() == b"%PDF current verified artifact"
        assert replay.md_path(paper.key).read_bytes() == b"# Current verified extract"
        assert current.acquisition_state.retrieval == Retrieval.VERIFIED
        assert accounting.spent(campaign) == 1
        assert (await verify_drained(replay, stop_intake=sealed_intake, execution_tracking_verified=True, ledger=accounting)).drained
    finally:
        stage.cleanup()


async def test_rollback_midflight_timeout_keeps_quarantine_and_charged_uncertainty(machinery, monkeypatch):
    lib, paper, _, ledger, store, now = machinery
    campaign = ledger.start_campaign(cap=2, keys=[paper.key])
    token = await reserve(store, paper, campaign, ttl=60)
    assert await store.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    entered, release = asyncio.Event(), asyncio.Event()
    @guard_execution
    async def executor(candidate, library):
        entered.set()
        await release.wait()
        stage.path("pdf").write_bytes(b"expired executor artifact")
        stage.library.get(candidate.key).download_status = "ok"
    task = asyncio.create_task(concurrency.with_dedup("dl:" + paper.key, lambda: executor(paper, lib)))
    try:
        await entered.wait()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.01)
        now[0] += 60
        assert await store.expire(paper.key, token=token)
        replay, accounting, recovered = restart(lib, now, monkeypatch)
        current = replay.get(paper.key)
        assert current.doi == "10.1234/rollback" and current.acquisition_state.policy == Policy.CLEAR
        assert current.acquisition_state.quarantine and current.acquisition_state.retrieval == Retrieval.IN_PROGRESS
        assert current.reopen_claim.phase == "dispatched" and record_generation(current) == 1
        assert accounting.spent(campaign) == 1 and accounting.outstanding([paper.key]) == [token]
        evidence = await verify_drained(replay, stop_intake=sealed_intake, execution_tracking_verified=True, ledger=accounting)
        assert not evidence.drained and token.lease_id in evidence.claims
        assert "dl:" + paper.key in evidence.in_flight and not task.done()
        assert not await recovered.release_quarantine(paper.key)
        assert not await recovered.not_sent(token)
        release.set()
        await task
        enable(monkeypatch, True)
        assert not await stage.writeback(recovered)
        assert not replay.pdf_path(paper.key).exists()
        assert accounting.spent(campaign) == 1
        assert not (await verify_drained(replay, stop_intake=sealed_intake, execution_tracking_verified=True, ledger=accounting)).drained
        assert await recovered.release_quarantine(paper.key, drained_lease_id=token.lease_id)
        persisted = Library(lib.root).get(paper.key)
        assert not persisted.acquisition_state.quarantine
        assert persisted.acquisition_state.retrieval == Retrieval.EXHAUSTED
        assert record_generation(persisted) == 2 and accounting.spent(campaign) == 1
        assert (await verify_drained(replay, stop_intake=sealed_intake, execution_tracking_verified=True, ledger=accounting)).drained
    finally:
        release.set()
        await task
        stage.cleanup()
