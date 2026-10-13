"""A3 deploy evidence checks real registries, queues and durable claims."""

import asyncio
import threading

import pytest

from papervault.library.claims_v2 import BudgetLedger, ClaimStore
from papervault.library.services import concurrency
from papervault.library.services.download_queue import DownloadQueue
from papervault.library.services.extract_queue import ExtractQueue
from papervault.library.store import Library


async def stopped():
    return True


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    lib = Library(tmp_path)
    paper, _ = lib.upsert({"title": "Drain barrier fixture with a durable identity", "doi": "10.1234/drain",
                           "download_status": "failed",
                           "terminal_state": {"retrieval": "exhausted", "policy": "clear"}})
    lib.save()
    return lib, paper


async def test_drained_snapshot_seals_intake_first_and_writes_nothing(vault):
    from papervault.library.drain_v2 import verify_drained
    lib, paper = vault
    ledger = BudgetLedger(lib.root)
    ledger.start_campaign(cap=1, keys=[paper.key])
    before = {p: p.read_bytes() for p in lib.root.rglob("*") if p.is_file()}
    called = []
    async def seal():
        assert not concurrency.lib_write_lock.locked()
        called.append("sealed")
        return True
    evidence = await verify_drained(lib, stop_intake=seal, execution_tracking_verified=True, ledger=ledger, clock=lambda: 123.0)
    assert called == ["sealed"]
    assert evidence.intake_stopped and evidence.drained and evidence.checked_at == 123.0
    assert evidence.execution_tracking_verified
    assert evidence.queued == evidence.unfinished == 0
    assert evidence.claims == evidence.in_flight == evidence.executors == evidence.merge_intents == ()
    assert {p: p.read_bytes() for p in lib.root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("ack", [False, None, 1, "yes"])
async def test_unconfirmed_stop_intake_never_certifies_barrier(vault, ack):
    from papervault.library.drain_v2 import verify_drained
    async def seal():
        return ack
    evidence = await verify_drained(vault[0], stop_intake=seal, execution_tracking_verified=True)
    assert not evidence.intake_stopped and not evidence.drained


async def test_in_flight_executor_blocks_until_actual_exit(vault):
    from papervault.library.drain_v2 import verify_drained
    lib, paper = vault
    entered, release = asyncio.Event(), asyncio.Event()
    async def work():
        entered.set()
        await release.wait()
    task = asyncio.create_task(concurrency.with_dedup("dl:" + paper.key, work))
    await entered.wait()
    try:
        evidence = await verify_drained(lib, stop_intake=stopped, execution_tracking_verified=True)
        assert not evidence.drained and "dl:" + paper.key in evidence.in_flight
    finally:
        release.set()
        await task
    assert (await verify_drained(lib, stop_intake=stopped, execution_tracking_verified=True)).drained


@pytest.mark.parametrize("queue_type", [DownloadQueue, ExtractQueue])
async def test_registered_queued_work_blocks(vault, queue_type):
    from papervault.library.drain_v2 import verify_drained
    lib, paper = vault
    queue = queue_type(lib)
    # Failed download rows route terminal; explicitly queue through the same
    # consumer API used by a recovered campaign (extract accepts any key).
    if queue_type is DownloadQueue:
        ledger = BudgetLedger(lib.root)
        campaign = ledger.start_campaign(cap=1, keys=[paper.key])
        store = ClaimStore(lib, ledger=ledger)
        token = await store.claim(paper.key, expected_state=paper.acquisition_state,
                                  expected_generation=0, campaign_id=campaign)
        queue.add_reopen(token)
    else:
        queue.add(paper.key)
    evidence = await verify_drained(lib, stop_intake=stopped, execution_tracking_verified=True)
    assert not evidence.drained and evidence.queued == 1 and evidence.unfinished == 1


async def test_dequeued_work_without_task_done_cannot_appear_drained(vault):
    from papervault.library.drain_v2 import verify_drained
    lib, paper = vault
    queue = ExtractQueue(lib)
    queue.add(paper.key)
    await queue._queue.get()  # real consumer boundary, before registry entry
    evidence = await verify_drained(lib, stop_intake=stopped, execution_tracking_verified=True)
    assert evidence.queued == 0 and evidence.unfinished == 1 and not evidence.drained
    queue._queue.task_done()


async def test_ledger_only_reservation_blocks_after_uncertain_index_write(vault, monkeypatch):
    from papervault.library.drain_v2 import verify_drained
    lib, paper = vault
    ledger = BudgetLedger(lib.root)
    store = ClaimStore(lib, ledger=ledger)
    campaign = ledger.start_campaign(cap=1, keys=[paper.key])
    def crash():
        raise OSError("crash before index publication")
    monkeypatch.setattr(store, "save", crash)
    with pytest.raises(OSError, match="crash"):
        await store.claim(paper.key, expected_state=paper.acquisition_state,
                          expected_generation=0, campaign_id=campaign)
    restarted = Library(lib.root)
    assert restarted.get(paper.key).reopen_claim is None
    evidence = await verify_drained(restarted, stop_intake=stopped, execution_tracking_verified=True, ledger=ledger)
    assert not evidence.drained and len(evidence.claims) == 1
    assert ledger.spent(campaign) == 1


async def test_unreadable_accounting_cannot_return_success(vault):
    from papervault.library.claims_v2 import LedgerError
    from papervault.library.drain_v2 import verify_drained
    lib, _ = vault
    (lib.root / "reopen-budget.jsonl").write_bytes(b"{")
    with pytest.raises(LedgerError):
        await verify_drained(lib, stop_intake=stopped, execution_tracking_verified=True)


async def test_ordinary_thread_lifetime_blocks_without_merge_tracking(vault, monkeypatch):
    from papervault.library.drain_v2 import verify_drained
    from papervault.library.telemetry_v2 import track_execution
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "0")
    monkeypatch.setenv("PAPERVAULT_STOPRULES_V2", "1")
    lib, paper = vault
    entered, release = threading.Event(), threading.Event()
    @track_execution
    def work(candidate, library):
        entered.set()
        assert release.wait(5)
    executor = asyncio.get_running_loop().run_in_executor(None, work, paper, lib)
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert not concurrency.in_flight_keys() and not lib._merge_active
        evidence = await verify_drained(lib, stop_intake=stopped, execution_tracking_verified=True)
        assert not evidence.drained and paper.key in evidence.executors
    finally:
        release.set()
        await executor


async def test_untracked_execution_cannot_be_certified_by_empty_registries(vault, monkeypatch):
    from papervault.library.drain_v2 import verify_drained
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "0")
    monkeypatch.setenv("PAPERVAULT_STOPRULES_V2", "0")
    lib, _ = vault
    entered, release = threading.Event(), threading.Event()
    def untracked_work():
        entered.set()
        assert release.wait(5)
    executor = asyncio.get_running_loop().run_in_executor(None, untracked_work)
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        evidence = await verify_drained(lib, stop_intake=stopped)
        assert not evidence.execution_tracking_verified and not evidence.drained
    finally:
        release.set()
        await executor


@pytest.mark.parametrize("proof", [False, None, 1, "yes"])
async def test_tracking_history_requires_explicit_confirmation(vault, monkeypatch, proof):
    from papervault.library.drain_v2 import verify_drained
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")  # current flag says nothing about earlier admissions
    evidence = await verify_drained(vault[0], stop_intake=stopped, execution_tracking_verified=proof)
    assert evidence.intake_stopped
    assert not evidence.execution_tracking_verified and not evidence.drained
