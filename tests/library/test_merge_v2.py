"""Spec A1/A3: fixture-only drained merges and durable audit evidence."""

import asyncio
import json
from dataclasses import replace

import pytest

from papervault.library.claims_v2 import BudgetLedger, ClaimStore, LedgerError, ReopenClaim
from papervault.library.state_v2 import Merge, Policy, PolicyOverride, Retrieval, TerminalState, admission, effective_policy
from papervault.library.store import Library


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    now = [1791849600.0]
    lib = Library(tmp_path)
    left, _ = lib.upsert({"title": "Canonical journal identity merge fixture", "authors": ["Curie"],
                          "doi": "10.1234/canonical", "source_identities": [
                              {"source": "core", "source_ids": {"core:work": ["123"]}}]})
    right, _ = lib.upsert({"title": "Independent preprint identity merge fixture", "authors": ["Einstein"],
                           "arxiv_id": "2101.12345", "source_identities": [
                               {"source": "ads", "source_ids": {"ads:bibcode": ["abc"]}}]})
    for paper in (left, right):
        paper.download_status = "failed"
        paper.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
    lib.save()
    ledger = BudgetLedger(tmp_path, clock=lambda: now[0])
    return lib, left, right, ledger, now


def coordinator(pair):
    from papervault.library.merge_v2 import MergeStore
    lib, _, _, ledger, now = pair
    return MergeStore(lib, ledger=ledger, clock=lambda: now[0])


async def claims(pair):
    lib, left, right, ledger, now = pair
    store = ClaimStore(lib, ledger=ledger, clock=lambda: now[0])
    campaign = ledger.start_campaign(cap=5, keys=[left.key, right.key])
    tokens = [await store.claim(p.key, expected_state=p.acquisition_state,
                                expected_generation=0, campaign_id=campaign) for p in (left, right)]
    return store, tokens


async def test_quiesce_persists_intent_and_blocks_new_claims(pair):
    lib, left, right, ledger, now = pair
    merge = coordinator(pair)
    campaign = ledger.start_campaign(cap=2, keys=[left.key, right.key])
    intent = await merge.quiesce(left.key, right.key, wait_seconds=30)
    restarted = Library(lib.root)
    for paper in (left, right):
        row = restarted.get(paper.key)
        assert row.merge_intent == intent
        assert row.acquisition_state.quarantine
        assert not admission(row.acquisition_state, identity_resolved=True, key=row.key)
        store = ClaimStore(lib, ledger=ledger, clock=lambda: now[0])
        assert await store.claim(paper.key, expected_state=paper.acquisition_state,
                                 expected_generation=0, campaign_id=campaign) is None
    assert ledger.spent(campaign) == 0


@pytest.mark.parametrize("result_kind", ["failed", "pdf", "verified_fill"])
async def test_queued_and_dispatched_claims_block_until_drained(pair, result_kind):
    from papervault.library.reopen_v2 import StagedAttempt
    lib, left, right, ledger, _ = pair
    store, (queued, dispatched) = await claims(pair)
    assert await store.pre_dispatch(dispatched)
    stage = StagedAttempt(lib, dispatched)
    merge = coordinator(pair)
    intent = await merge.quiesce(left.key, right.key)
    assert not await merge.drain_ready(intent)
    assert await merge.merge_records(left.key, right.key) is None
    assert left.arxiv_id == ""
    assert await store.not_sent(queued)
    assert not await merge.drain_ready(intent)
    result = stage.library.get(right.key)
    if result_kind == "failed":
        result.download_status = "failed"
    else:
        stage.path("pdf").write_bytes(b"settled staged PDF")
        result.download_status = "ok"
        if result_kind == "verified_fill":
            result.doi = "10.1234/verified-preprint"
    try:
        assert await stage.writeback(store)
    finally:
        stage.cleanup()
    assert right.reopen_claim.phase == "completed"
    assert ledger.phase(dispatched) == "complete"
    assert ledger.outstanding([left.key, right.key]) == []
    assert right.acquisition_state.quarantine  # New claims remain blocked until merge.
    assert not right.terminal_state.quarantine  # Settlement is not execution uncertainty.
    if result_kind == "verified_fill":
        assert right.doi == "10.1234/verified-preprint"
        assert lib.find(doi=right.doi) is right
    assert await merge.drain_ready(intent)
    assert await merge.merge_records(left.key, right.key) is left
    assert left.doi == "10.1234/canonical" and left.arxiv_id == "2101.12345"
    assert [o.source for o in left.source_identities] == ["core", "ads"]
    assert lib.get(right.key).acquisition_state.merge == Merge.ALIAS
    assert not left.acquisition_state.quarantine
    if result_kind != "failed":
        assert left.acquisition_state.retrieval == Retrieval.ACQUIRED
        assert lib.pdf_path(left.key).read_bytes() == b"settled staged PDF"
    assert ledger.spent(dispatched.campaign_id) == 1


async def test_expiry_remains_blocked_after_later_drain_and_restart(pair):
    from papervault.library.reopen_v2 import StagedAttempt
    lib, left, right, ledger, now = pair
    store, (queued, dispatched) = await claims(pair)
    assert await store.pre_dispatch(dispatched)
    stage = StagedAttempt(lib, dispatched)
    merge = coordinator(pair)
    intent = await merge.quiesce(left.key, right.key, wait_seconds=10)
    now[0] += 10
    assert not await merge.drain_ready(intent)
    assert await merge.merge_records(left.key, right.key) is None
    assert ledger.spent(dispatched.campaign_id) == 2
    assert await store.not_sent(queued)
    stage.library.get(right.key).download_status = "failed"
    try:
        assert await stage.writeback(store)
    finally:
        stage.cleanup()
    assert right.reopen_claim.phase == "completed"
    assert ledger.phase(dispatched) == "complete"
    assert ledger.outstanding([left.key, right.key]) == []
    assert not await merge.drain_ready(intent)
    from papervault.library.merge_v2 import MergeStore
    replay = MergeStore(Library(lib.root), ledger=ledger, clock=lambda: now[0])
    assert await replay.merge_records(left.key, right.key) is None
    for paper in replay.library.all_papers(include_quarantined=True):
        assert paper.merge_intent.status == "expired"
        assert paper.acquisition_state.quarantine
    assert not any(r["event"] == "merge" for r in ledger._read())


async def test_lease_expiry_during_quiescence_drops_staged_result(pair):
    from papervault.library.reopen_v2 import StagedAttempt
    lib, left, right, ledger, now = pair
    store, (queued, dispatched) = await claims(pair)
    assert await store.pre_dispatch(dispatched)
    stage = StagedAttempt(lib, dispatched)
    merge = coordinator(pair)
    intent = await merge.quiesce(left.key, right.key, wait_seconds=7200)
    assert await store.not_sent(queued)
    now[0] = dispatched.lease_expiry
    assert await store.expire(right.key, token=dispatched)
    stage.path("pdf").write_bytes(b"stale lease result")
    stage.library.get(right.key).download_status = "ok"
    restarted = ClaimStore(Library(lib.root), ledger=ledger, clock=lambda: now[0])
    try:
        assert not await stage.writeback(restarted)
    finally:
        stage.cleanup()
    row = restarted.library.get(right.key)
    assert row.terminal_state.quarantine and row.merge_intent.status == "waiting"
    assert row.reopen_claim.phase == "dispatched" and row.download_status == "pending"
    assert not lib.has_pdf(right.key)
    assert ledger.phase(dispatched) == "dispatch"
    assert ledger.spent(dispatched.campaign_id) == 1
    assert not await merge.drain_ready(intent)
    assert await merge.merge_records(left.key, right.key) is None


async def test_policy_observation_during_quiescence_still_fences_writeback(pair):
    from papervault.library.reopen_v2 import StagedAttempt
    from papervault.library.services import concurrency
    lib, left, right, ledger, _ = pair
    store, (dispatched, queued) = await claims(pair)
    assert await store.pre_dispatch(dispatched)
    stage = StagedAttempt(lib, dispatched)
    merge = coordinator(pair)
    intent = await merge.quiesce(left.key, right.key)
    assert await store.not_sent(queued)
    async with concurrency.lib_write_lock:
        assert merge._merge_records(left.key, right.key, metadata={"doi": left.doi, "arxiv_id": right.arxiv_id,
                    "terminal_state": {"policy": "excluded", "policy_reasons": ["non_paper"]}}) is None
    stage.path("pdf").write_bytes(b"policy fenced staged PDF")
    stage.library.get(left.key).download_status = "ok"
    try:
        assert not await stage.writeback(store)
    finally:
        stage.cleanup()
    assert left.acquisition_state.policy == Policy.EXCLUDED
    assert left.terminal_state.quarantine
    assert left.reopen_claim.phase == "dispatched" and ledger.phase(dispatched) == "dispatch"
    assert not lib.has_pdf(left.key) and not await merge.drain_ready(intent)


async def test_metadata_observation_during_quiescence_allows_valid_writeback(pair):
    from papervault.library.reopen_v2 import StagedAttempt
    from papervault.library.services import concurrency
    lib, left, right, _, _ = pair
    store, (dispatched, queued) = await claims(pair)
    assert await store.pre_dispatch(dispatched)
    stage = StagedAttempt(lib, dispatched)
    merge = coordinator(pair)
    intent = await merge.quiesce(left.key, right.key)
    assert await store.not_sent(queued)
    async with concurrency.lib_write_lock:
        assert merge._merge_records(left.key, right.key, metadata={"doi": left.doi, "arxiv_id": right.arxiv_id,
                    "terminal_state": {"policy": "clear"},
                    "abstract": "New metadata retained at settlement"}) is None
    stage.library.get(left.key).download_status = "failed"
    try:
        assert await stage.writeback(store)
    finally:
        stage.cleanup()
    assert left.abstract == "New metadata retained at settlement"
    assert await merge.drain_ready(intent)
    assert await merge.merge_records(left.key, right.key) is left
    assert left.abstract == "New metadata retained at settlement" and not left.acquisition_state.quarantine


@pytest.mark.parametrize("result_kind", ["pdf", "cascade"])
async def test_real_campaign_finishes_both_dispatched_results_after_quiescence(pair, monkeypatch, result_kind):
    import threading
    from papervault.library.reopen_v2 import run_campaign
    from papervault.library.services import download_queue
    lib, left, right, ledger, now = pair
    started = {paper.key: threading.Event() for paper in (left, right)}
    finish = threading.Event()
    def download(paper, stage, **_):
        started[paper.key].set()
        assert finish.wait(5), "fixture executor was not released"
        stage.pdf_path(paper.key).write_bytes(b"campaign settled PDF")
        paper.download_status = "ok"
        return True
    if result_kind == "pdf":
        monkeypatch.setattr(download_queue, "download_paper", download)
    else:
        from papervault.library import download as cascade
        def tier(paper):
            started[paper.key].set()
            assert finish.wait(5), "fixture executor was not released"
            return None
        monkeypatch.setattr(cascade, "_STRATEGIES", [("fixture", tier)])
        monkeypatch.setattr(cascade, "_try_firecrawl_text_fallback", lambda *_: False)
    campaign = asyncio.create_task(run_campaign(lib, cap=2, clock=lambda: now[0]))
    merge = coordinator(pair)
    try:
        for event in started.values():
            assert await asyncio.to_thread(event.wait, 5)
        intent = await merge.quiesce(left.key, right.key)
        assert not await merge.drain_ready(intent)
        assert await merge.merge_records(left.key, right.key) is None
    finally:
        finish.set()
    report = await asyncio.wait_for(campaign, 5)
    assert report["completed"] == 2
    assert all(p.reopen_claim.phase == "completed" for p in (left, right))
    assert ledger.outstanding([left.key, right.key]) == []
    assert await merge.drain_ready(intent)
    assert await merge.merge_records(left.key, right.key) is left
    if result_kind == "pdf":
        assert lib.pdf_path(left.key).read_bytes() == b"campaign settled PDF"
    else:
        assert left.download_status == "failed" and not lib.has_pdf(left.key)
    assert left.merge_budget.generation == 3 and left.merge_budget.spent == 2
    assert lib.get(right.key).acquisition_state.alias_key == left.key


@pytest.mark.parametrize("restore_old_index,snapshot_phase", [
    (False, "completed"), (True, "queued"), (True, "dispatched"), (True, "completed"),
], ids=["current", "stale-queued", "stale-dispatched", "stale-completed"])
async def test_extraction_checks_persisted_merge_generation_in_every_claim_phase(pair, monkeypatch, restore_old_index, snapshot_phase):
    from papervault.library import extract
    from papervault.library.claims_v2 import record_generation
    from papervault.library.reopen_v2 import StagedAttempt
    lib, left, right, ledger, _ = pair
    store, tokens = await claims(pair)
    snapshots = {"queued": lib.index_path.read_bytes()}
    for token in tokens:
        assert await store.pre_dispatch(token)
    snapshots["dispatched"] = lib.index_path.read_bytes()
    for token in tokens:
        stage = StagedAttempt(lib, token)
        stage.path("pdf").write_bytes(b"settled PDF for extraction")
        stage.library.get(token.key).download_status = "ok"
        try:
            assert await stage.writeback(store)
        finally:
            stage.cleanup()
    ledger.finish_campaign(tokens[0].campaign_id, drained=True)
    snapshots["completed"] = lib.index_path.read_bytes()
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    assert ledger.latest_generation(left.key) == 3
    if restore_old_index:
        lib.index_path.write_bytes(snapshots[snapshot_phase])
    extraction_lib = Library(lib.root)
    paper = extraction_lib.get(left.key)
    assert paper.reopen_claim.phase == (snapshot_phase if restore_old_index else "completed")
    assert record_generation(paper) == (1 if restore_old_index else 3)
    body = "Complete fixture paper markdown. " * 30
    monkeypatch.setattr(extract, "pdf_probe", lambda _: extract.PDFProbe(bad=False, n_pages=1, reason="ok"))
    async def ocr(*_args, **_kwargs):
        return body
    class Controller:
        async def ensure_ready(self):
            pass
    monkeypatch.setattr(extract, "get_server_controller", Controller)
    monkeypatch.setattr(extract, "extract_mineru", ocr)
    monkeypatch.setattr(extract, "review_extract", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(extract, "completeness_gate", lambda *_args, **_kwargs: {"complete": True})
    result = await extract.extract_md(paper, extraction_lib, llm=object())
    if restore_old_index:
        assert result is None
        assert not extraction_lib.has_extract(paper.key, "md")
        assert paper.md_engine == ""
    else:
        assert result == body
        assert extraction_lib.md_path(paper.key).read_text() == body
        assert paper.md_engine == "mineru2.5-pro"


async def test_alias_restart_and_serialized_second_writer_refusal(pair):
    lib, left, right, ledger, _ = pair
    results = await asyncio.gather(*[coordinator(pair).merge_records(left.key, right.key) for _ in range(6)])
    assert sum(result is not None for result in results) == 1
    from papervault.library.merge_v2 import resolve_alias
    restarted = Library(lib.root)
    assert resolve_alias(restarted, right.key).key == left.key
    assert restarted.find(arxiv_id="2101.12345").key == left.key
    assert restarted.get(right.key).acquisition_state.alias_key == left.key
    assert len([r for r in ledger._read() if r["event"] == "merge"]) == 1


@pytest.mark.parametrize("left_policy,right_policy,want", [
    (Policy.CLEAR, Policy.EXCLUDED, Policy.EXCLUDED),
    (Policy.EXCLUDED, Policy.UNKNOWN, Policy.UNKNOWN),
    (Policy.CLEAR, Policy.OVERRIDE, Policy.EXCLUDED),
])
async def test_restrictions_union_and_no_one_sided_grant(pair, left_policy, right_policy, want):
    _, left, right, _, _ = pair
    left.terminal_state = replace(left.acquisition_state, policy=left_policy,
                                  policy_reasons=("off_domain",), required_revision=4)
    right.terminal_state = replace(right.acquisition_state, policy=right_policy,
        policy_reasons=("non_paper",), required_revision=7,
        override=PolicyOverride(scope=("acquire",), reasons=("non_paper",), revision=9, key=right.key))
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    assert left.acquisition_state.policy == want
    assert left.acquisition_state.policy_reasons == ("off_domain", "non_paper")
    assert left.acquisition_state.required_revision == 7
    assert left.acquisition_state.override is None
    assert not effective_policy(left.acquisition_state, key=left.key)


async def test_grants_intersect_scope_reasons_revision_and_key(pair):
    _, left, right, _, _ = pair
    for paper, scope, reasons, revision in (
        (left, ("acquire", "route"), ("off_domain", "non_paper"), 9),
        (right, ("acquire",), ("off_domain",), 6),
    ):
        paper.terminal_state = replace(paper.acquisition_state, policy=Policy.OVERRIDE,
            policy_reasons=("off_domain",), required_revision=5,
            override=PolicyOverride(scope=scope, reasons=reasons, revision=revision, key=left.key))
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    assert left.acquisition_state.override == PolicyOverride(
        scope=("acquire",), reasons=("off_domain",), revision=6, key=left.key)
    assert effective_policy(left.acquisition_state, key=left.key)
    assert not effective_policy(left.acquisition_state, key=left.key, action="route")


async def test_sum_min_budget_generation_persisted_and_enforced(pair):
    lib, left, right, ledger, now = pair
    for paper, cap, count in ((left, 5, 2), (right, 3, 1)):
        campaign = ledger.start_campaign(cap=cap, keys=[paper.key])
        for generation in range(1, count + 1):
            token = ReopenClaim(key=paper.key, generation=generation,
                lease_id=f"{paper.key}-{generation}", lease_expiry=now[0] + 60, campaign_id=campaign)
            ledger.reserve(token)
            ledger.dispatch(token)
            ledger.complete(token)
            paper.reopen_claim = token.model_copy(update={"phase": "completed"})
        ledger.finish_campaign(campaign, drained=True)
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    assert (left.merge_budget.spent, left.merge_budget.budget, left.merge_budget.generation) == (3, 3, 4)
    event = ledger._read()[-1]
    assert event["event"] == "merge"
    assert event["canonical_key"] == left.key and event["absorbed_key"] == right.key
    assert event["generations_before"] == [2, 1] and event["generation_after"] == 4
    restarted = Library(lib.root)
    replay = BudgetLedger(lib.root, clock=lambda: now[0])
    assert restarted.get(left.key).merge_budget == left.merge_budget
    assert replay.latest_generation(left.key) == replay.latest_generation(right.key) == 4
    assert replay.key_budget(left.key) == {"spent": 3, "budget": 3, "generation": 4}
    campaign = replay.start_campaign(cap=5, keys=[left.key])
    store = ClaimStore(restarted, ledger=replay, clock=lambda: now[0])
    current = restarted.get(left.key)
    with pytest.raises(LedgerError):
        await store.claim(left.key, expected_state=current.acquisition_state,
                          expected_generation=4, campaign_id=campaign)


async def test_index_rollback_cannot_reuse_generations_or_merge(pair):
    lib, left, right, ledger, now = pair
    before = lib.index_path.read_bytes()
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    lib.index_path.write_bytes(before)
    rolled = Library(lib.root)
    campaign = ledger.start_campaign(cap=2, keys=[right.key])
    store = ClaimStore(rolled, ledger=ledger, clock=lambda: now[0])
    paper = rolled.get(right.key)
    assert await store.claim(right.key, expected_state=paper.acquisition_state,
                              expected_generation=0, campaign_id=campaign) is None
    from papervault.library.merge_v2 import MergeStore
    assert await MergeStore(rolled, ledger=ledger, clock=lambda: now[0]).merge_records(left.key, right.key) is None


def test_off_byte_identical_and_plain_fill_preserves_object(tmp_path, monkeypatch):
    monkeypatch.delenv("PAPERVAULT_MERGE_V2", raising=False)
    lib = Library(tmp_path)
    paper, _ = lib.upsert({"title": "Legacy merge serialization fixture", "doi": "10.1234/legacy"})
    lib.save()
    before = lib.index_path.read_bytes()
    assert "merge_intent" not in paper.model_dump()
    assert "merge_budget" not in json.loads(paper.model_dump_json())
    lib.upsert({"doi": paper.doi, "source_identities": []})
    lib.save()
    assert lib.index_path.read_bytes() == before
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    merged, new = lib.upsert({"doi": paper.doi, "venue": "Journal"})
    assert merged is paper and not new and merged.venue == "Journal"
    assert merged.merge_intent is None
    assert not (tmp_path / "reopen-budget.jsonl").exists()


async def test_doi_collision_routes_to_alias_without_purging_artifacts(pair):
    from papervault.library.services import concurrency
    lib, left, right, _, _ = pair
    artifact = lib.pdf_path(right.key)
    artifact.write_bytes(b"retained preprint artifact")
    async with concurrency.lib_write_lock:
        outcome, detail = lib.set_resolved_doi(right.key, left.doi)
    assert (outcome, detail) == ("collision", left.key)
    assert artifact.read_bytes() == b"retained preprint artifact"
    assert lib.get(right.key).acquisition_state.merge == Merge.ALIAS


async def test_doi_collision_does_not_mutate_identity_while_active(pair):
    from papervault.library.services import concurrency
    lib, left, right, _, _ = pair
    store, (_, active) = await claims(pair)
    assert await store.pre_dispatch(active)
    async with concurrency.lib_write_lock:
        outcome, detail = lib.set_resolved_doi(right.key, left.doi)
    assert (outcome, detail) == ("merge_waiting", left.key)
    assert right.doi == "" and left.arxiv_id == ""
    assert right.merge_intent is not None


def test_fuzzy_identity_promotion_uses_durable_barrier(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    lib = Library(tmp_path)
    preprint, _ = lib.upsert({"title": "Cosmic ray transport in turbulent magnetic fields",
                             "authors": ["Jane Curie"], "year": 2020, "arxiv_id": "2001.12345"})
    merged, new = lib.upsert({"title": "Cosmic ray transport in turbulent magnetic fields study",
                            "authors": ["Jane Curie"], "year": 2021, "doi": "10.1234/journal"})
    assert merged is preprint and not new
    assert merged.doi == "10.1234/journal"
    assert merged.merge_budget.generation == 1
    assert len(lib.all_papers()) == 1
    assert len(lib.all_papers(include_quarantined=True)) == 2


async def test_exact_upsert_of_two_persisted_identities_merges_both_keys(pair):
    from papervault.library.services import concurrency
    lib, left, right, _, _ = pair
    async with concurrency.lib_write_lock:
        result, new = lib.upsert({"doi": left.doi, "arxiv_id": right.arxiv_id})
    assert result is left and not new
    assert lib.get(right.key).acquisition_state.alias_key == left.key


async def test_ordinary_queued_work_must_drain_and_new_work_is_refused(pair):
    from papervault.library.services.download_queue import DownloadQueue
    from papervault.library.services.extract_queue import ExtractQueue
    from papervault.library.services.classify import TERMINAL, classify
    lib, left, right, _, _ = pair
    left.download_status = right.download_status = "pending"
    download = DownloadQueue(lib, num_workers=0)
    extract = ExtractQueue(lib)
    download.add(left.key)
    extract.add(right.key)
    merge = coordinator(pair)
    intent = await merge.quiesce(left.key, right.key)
    assert not await merge.drain_ready(intent)
    download.add(right.key)
    extract.add(left.key)
    assert right.key not in download._pending and left.key not in extract._pending
    assert classify(left, lib) == classify(right, lib) == TERMINAL
    download._pending.pop(left.key)
    extract._pending.pop(right.key)
    assert await merge.drain_ready(intent)


async def test_canceled_awaiter_does_not_hide_live_download_thread(pair):
    import threading
    from papervault.library.merge_v2 import guard_execution
    lib, left, right, _, _ = pair
    started, finish = threading.Event(), threading.Event()
    @guard_execution
    def execute(paper, library):
        started.set()
        finish.wait(5)
        return True
    worker = asyncio.create_task(asyncio.to_thread(execute, left, lib))
    await asyncio.to_thread(started.wait, 5)
    merge = coordinator(pair)
    intent = await merge.quiesce(left.key, right.key)
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker
    try:
        assert not await merge.drain_ready(intent)
        assert execute(right, lib) is False
    finally:
        finish.set()
    for _ in range(100):
        if not lib._merge_active.get(left.key):
            break
        await asyncio.sleep(0.01)
    assert await merge.drain_ready(intent)


async def test_another_task_cannot_borrow_the_writer_lock(pair):
    from papervault.library.services import concurrency
    lib, left, right, _, _ = pair
    async with concurrency.lib_write_lock:
        async def other():
            with pytest.raises(RuntimeError, match="owner"):
                lib.set_resolved_doi(right.key, left.doi)
        await asyncio.create_task(other())
    assert left.arxiv_id == "" and right.merge_intent is None


async def test_ledger_claims_outlive_rolled_back_record_phases(pair):
    _, left, right, _, _ = pair
    await claims(pair)
    left.reopen_claim = right.reopen_claim = None
    assert await coordinator(pair).merge_records(left.key, right.key) is None


async def test_alias_chains_preserve_budget_and_reject_cycles(pair):
    from papervault.library.merge_v2 import resolve_alias
    lib, left, right, ledger, _ = pair
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    third, _ = lib.upsert({"title": "Third independent canonical scientific identity", "authors": ["Feynman"],
                           "doi": "10.1234/third"})
    assert await coordinator(pair).merge_records(third.key, left.key) is third
    assert resolve_alias(Library(lib.root), right.key).key == third.key
    assert ledger.key_budget(third.key)["generation"] == 2
    third.terminal_state = replace(third.acquisition_state, merge=Merge.ALIAS, alias_key=right.key)
    assert resolve_alias(lib, right.key) is None


async def test_uncertain_post_ledger_save_cannot_repeat(pair, monkeypatch):
    lib, left, right, ledger, _ = pair
    merge = coordinator(pair)
    await merge.quiesce(left.key, right.key)
    original = lib.save
    def fail(**kwargs):
        if left.merge_budget is not None:
            raise OSError("injected post-ledger index failure")
        return original(**kwargs)
    monkeypatch.setattr(lib, "save", fail)
    with pytest.raises(OSError, match="post-ledger"):
        await merge.merge_records(left.key, right.key)
    assert ledger.latest_generation(left.key) == 1
    assert not await merge.drain_ready(left.merge_intent)
    assert await merge.merge_records(left.key, right.key) is None


async def test_merge_flag_alone_enforces_domain_restrictions_and_admission(pair, monkeypatch):
    from papervault.library.services.classify import TERMINAL, classify
    _, left, right, _, _ = pair
    monkeypatch.delenv("PAPERVAULT_REOPEN_V2", raising=False)
    monkeypatch.delenv("PAPERVAULT_TERMINAL_V2", raising=False)
    monkeypatch.delenv("PAPERVAULT_IDENTITY_V2", raising=False)
    left.download_status = right.download_status = "pending"
    right.domain_status = "off_domain"
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    assert left.acquisition_state.policy == Policy.EXCLUDED
    assert left.acquisition_state.policy_reasons == ("off_domain",)
    assert classify(left, pair[0]) == TERMINAL


async def test_corrupt_or_missing_post_merge_ledger_refuses_operations(pair):
    _, left, right, ledger, _ = pair
    merge = coordinator(pair)
    await merge.quiesce(left.key, right.key)
    ledger.path.write_bytes(b"{\n")
    with pytest.raises(LedgerError):
        await merge.merge_records(left.key, right.key)
    assert left.arxiv_id == ""
    ledger.path.unlink()  # no claim or completed merge in this fixture
    assert await merge.merge_records(left.key, right.key) is left
    ledger.path.unlink()
    with pytest.raises(LedgerError, match="missing"):
        ledger.latest_generation(left.key)


def test_null_metadata_promotion_and_worker_fields_cannot_be_injected(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    lib = Library(tmp_path)
    paper, _ = lib.upsert({"title": "Null metadata identity promotion fixture", "arxiv_id": "2101.12345"})
    merged, _ = lib.upsert({"title": paper.title, "doi": "10.1234/null", "venue": None,
                            "merge_budget": {"merge_id": "forged", "generation": 999, "spent": 0, "budget": 999}})
    assert merged is paper and merged.doi == "10.1234/null"
    assert merged.merge_budget.generation == 1 and merged.merge_budget.budget == 200


async def test_remaining_merged_budget_can_fund_next_generation(pair):
    lib, left, right, ledger, now = pair
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    campaign = ledger.start_campaign(cap=2, keys=[left.key])
    store = ClaimStore(lib, ledger=ledger, clock=lambda: now[0])
    token = await store.claim(left.key, expected_state=left.acquisition_state,
                              expected_generation=1, campaign_id=campaign)
    assert token is not None and token.generation == 2
    assert ledger.admitted(token)


async def test_async_cancellation_persists_uncertain_merge_barrier(pair):
    from papervault.library.merge_v2 import guard_execution, MergeStore
    lib, left, right, ledger, now = pair
    started = asyncio.Event()
    @guard_execution
    async def execute(paper, library):
        started.set()
        await asyncio.Event().wait()
    worker = asyncio.create_task(execute(left, lib))
    await started.wait()
    merge = coordinator(pair)
    intent = await merge.quiesce(left.key, right.key)
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker
    assert not await merge.drain_ready(intent)
    restarted = Library(lib.root)
    assert restarted.get(left.key).merge_intent.status == "uncertain"
    assert await MergeStore(restarted, ledger=ledger, clock=lambda: now[0]).merge_records(left.key, right.key) is None


async def test_public_campaign_uses_persisted_merge_generation(pair, monkeypatch):
    from papervault.library.reopen_v2 import inspect_campaign, run_campaign
    from papervault.library.services import download_queue
    lib, left, right, ledger, now = pair
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    assert inspect_campaign(lib.root, cap=2, clock=lambda: now[0])["keys"] == [left.key]
    def exhausted(paper, library, **kwargs):
        paper.download_status = "failed"
        return False
    monkeypatch.setattr(download_queue, "download_paper", exhausted)
    result = await run_campaign(lib, cap=2, clock=lambda: now[0])
    assert result["keys"] == [left.key] and result["completed"] == 1
    assert left.reopen_claim.generation == 2 and ledger.latest_generation(left.key) == 2


async def test_stored_twin_upsert_retains_new_restrictions_and_observations(pair):
    from papervault.library.services import concurrency
    lib, left, right, _, _ = pair
    async with concurrency.lib_write_lock:
        merged, _ = lib.upsert({"doi": left.doi, "arxiv_id": right.arxiv_id, "venue": "New Journal",
            "domain_status": "off_domain", "source_identities": [{"source": "openalex", "source_ids": {"openalex:work": ["W1"]}}],
            "terminal_state": {"policy": "excluded", "policy_reasons": ["non_paper"]}})
    assert merged is left and merged.venue == "New Journal"
    assert set(left.acquisition_state.policy_reasons) == {"non_paper", "off_domain"}
    assert not effective_policy(left.acquisition_state, key=left.key)
    assert {o.source for o in left.source_identities} == {"core", "ads", "openalex"}


def test_quarantined_promotion_cannot_create_schedulable_twin(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    lib = Library(tmp_path)
    paper, _ = lib.upsert({"title": "Quarantined identity promotion scientific fixture", "arxiv_id": "2101.12345"})
    paper.terminal_state = replace(paper.acquisition_state, quarantine=True)
    before = lib.keys()
    result, _ = lib.upsert({"title": paper.title, "doi": "10.1234/quarantined"})
    assert result is paper and paper.doi == ""
    assert lib.keys() == before


async def test_disabled_flag_preserves_persisted_barrier_and_alias(pair, monkeypatch):
    lib, left, right, ledger, _ = pair
    intent = await coordinator(pair).quiesce(left.key, right.key)
    monkeypatch.delenv("PAPERVAULT_MERGE_V2", raising=False)
    result, _ = lib.upsert({"doi": left.doi, "arxiv_id": right.arxiv_id})
    assert result is left and left.arxiv_id == ""
    assert left.merge_intent == intent
    assert lib.set_resolved_doi(right.key, left.doi)[0] == "merge_waiting"
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    artifact = lib.pdf_path(right.key)
    artifact.write_bytes(b"alias artifact")
    monkeypatch.delenv("PAPERVAULT_MERGE_V2", raising=False)
    assert lib.set_resolved_doi(right.key, left.doi) == ("alias", left.key)
    assert artifact.read_bytes() == b"alias artifact"
    assert lib.find(arxiv_id=right.arxiv_id) is left
    assert len([r for r in ledger._read() if r["event"] == "merge"]) == 1


async def test_merge_only_rollback_refuses_ordinary_download_and_extract(pair, monkeypatch):
    from papervault.library.services.classify import TERMINAL, classify
    from papervault.library.merge_v2 import guard_execution
    lib, left, right, _, _ = pair
    monkeypatch.delenv("PAPERVAULT_REOPEN_V2", raising=False)
    left.download_status = right.download_status = "pending"
    lib.save()
    before = lib.index_path.read_bytes()
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    lib.index_path.write_bytes(before)
    rolled = Library(lib.root)
    @guard_execution
    def should_not_execute(paper, library):
        raise AssertionError("rolled-back identity reached an executor")
    for key in (left.key, right.key):
        row = rolled.get(key)
        assert classify(row, rolled) == TERMINAL
        assert should_not_execute(row, rolled) is False


async def test_repeated_cancellation_cannot_release_merge_resources(pair):
    from papervault.library.merge_v2 import guard_execution
    from papervault.library.services import concurrency
    lib, left, right, _, _ = pair
    started = asyncio.Event()
    @guard_execution
    async def execute(paper, library):
        started.set()
        await asyncio.Event().wait()
    worker = asyncio.create_task(execute(left, lib))
    await started.wait()
    intent = await coordinator(pair).quiesce(left.key, right.key)
    async with concurrency.lib_write_lock:
        worker.cancel()
        await asyncio.sleep(0)
        worker.cancel()
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await worker
    assert lib._merge_active.get(left.key) == 1
    assert not await coordinator(pair).drain_ready(intent)
    assert Library(lib.root).get(left.key).merge_intent.status == "uncertain"


async def test_canonical_adopts_only_absorbed_artifacts_without_deleting_them(pair):
    lib, left, right, _, _ = pair
    lib.pdf_path(right.key).write_bytes(b"verified PDF")
    lib.md_path(right.key).write_text("verified markdown")
    right.pdf_path = f"pdfs/{right.key}.pdf"
    right.md_path = f"extracts/md/{right.key}.md"
    right.md_engine = "mineru2.5-pro"
    right.download_status = "ok"
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    restarted = Library(lib.root)
    assert restarted.has_pdf(left.key) and restarted.has_extract(left.key, "md")
    assert restarted.md_path(left.key).read_text() == "verified markdown"
    assert restarted.pdf_path(right.key).read_bytes() == b"verified PDF"


async def test_new_doi_cannot_bypass_an_existing_intent(pair):
    lib, left, right, _, _ = pair
    await coordinator(pair).quiesce(left.key, right.key)
    assert lib.set_resolved_doi(right.key, "10.1234/unrelated")[0] == "merge_waiting"
    assert right.doi == ""


async def test_unfenced_stub_cannot_purge_into_quiesced_holder(pair, monkeypatch):
    lib, left, right, _, _ = pair
    third, _ = lib.upsert({"title": "Unfenced third scientific stub", "authors": ["Feynman"]})
    await coordinator(pair).quiesce(left.key, right.key)
    monkeypatch.delenv("PAPERVAULT_MERGE_V2", raising=False)
    assert lib.set_resolved_doi(third.key, left.doi)[0] == "merge_waiting"
    assert lib.get(third.key) is third


async def test_disable_flag_does_not_amplify_merged_policy(pair, monkeypatch):
    from papervault.library.services.classify import TERMINAL, classify
    lib, left, right, _, _ = pair
    left.download_status = right.download_status = "pending"
    right.terminal_state = replace(right.acquisition_state, policy=Policy.EXCLUDED, policy_reasons=("off_domain",))
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    for name in ("PAPERVAULT_MERGE_V2", "PAPERVAULT_REOPEN_V2", "PAPERVAULT_TERMINAL_V2"):
        monkeypatch.delenv(name, raising=False)
    assert classify(left, lib) == TERMINAL


async def test_verified_blank_fill_refuses_quiesced_identity(pair):
    lib, left, right, _, _ = pair
    await coordinator(pair).quiesce(left.key, right.key)
    assert lib.fill_verified_identifiers(right.key, doi="10.1234/new") == {}
    assert right.doi == ""


async def test_exhausted_merged_budget_is_skipped_before_campaign_start(pair):
    from papervault.library.reopen_v2 import inspect_campaign, run_campaign
    lib, left, right, ledger, now = pair
    campaign = ledger.start_campaign(cap=1, keys=[right.key])
    token = ReopenClaim(key=right.key, generation=1, lease_id="spent", lease_expiry=now[0] + 60, campaign_id=campaign)
    ledger.reserve(token)
    ledger.dispatch(token)
    ledger.complete(token)
    right.reopen_claim = token.model_copy(update={"phase": "completed"})
    ledger.finish_campaign(campaign, drained=True)
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    before = ledger.path.read_bytes()
    assert inspect_campaign(lib.root, cap=1, clock=lambda: now[0])["keys"] == []
    assert (await run_campaign(lib, cap=1, clock=lambda: now[0]))["selected"] == 0
    assert ledger.path.read_bytes() == before


async def test_absorbed_pdf_only_success_can_continue_extraction(pair):
    from papervault.library.services.classify import EXTRACT, classify
    lib, left, right, _, _ = pair
    lib.pdf_path(right.key).write_bytes(b"verified PDF")
    right.download_status = "ok"
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    assert left.download_status == "ok"
    assert classify(left, lib) == EXTRACT


async def test_alias_index_selection_quiesces_the_actual_canonical(pair):
    from papervault.library.services import concurrency
    lib, left, right, _, _ = pair
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    third, _ = lib.upsert({"title": "Third unrelated canonical identity scientific record", "authors": ["Feynman"],
                           "arxiv_id": "2102.12345"})
    assert await coordinator(pair).merge_records(third.key, left.key) is third
    fourth, _ = lib.upsert({"title": "Fourth separate canonical identity scientific record", "authors": ["Bohr"],
                            "doi": "10.1234/fourth"})
    lib.save()
    lib = Library(lib.root)
    async with concurrency.lib_write_lock:
        merged, _ = lib.upsert({"doi": fourth.doi, "arxiv_id": right.arxiv_id})
    assert merged.key == fourth.key
    assert lib.get(third.key).acquisition_state.alias_key == fourth.key
    assert len(lib.all_papers()) == 1


async def test_force_add_cannot_reset_or_queue_a_merge_intent(pair):
    from papervault.library.services.add_service import AddService
    lib, left, right, _, _ = pair
    await coordinator(pair).quiesce(left.key, right.key)
    before = left.download_status
    result = AddService(lib).add(left.key, force_refresh=True)
    assert result["status"] == "blocked"
    assert left.download_status == before


async def test_rolled_back_canonical_cannot_start_another_merge(pair):
    lib, left, right, ledger, now = pair
    third, _ = lib.upsert({"title": "Another independent scientific identity for stale merge", "authors": ["Bohr"],
                           "doi": "10.1234/stale"})
    lib.save()
    before = lib.index_path.read_bytes()
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    lib.index_path.write_bytes(before)
    from papervault.library.merge_v2 import MergeStore
    stale = MergeStore(Library(lib.root), ledger=ledger, clock=lambda: now[0])
    assert await stale.merge_records(left.key, third.key) is None


async def test_acquired_pdf_can_extract_after_merged_download_budget_is_spent(pair):
    from papervault.library.merge_v2 import guard_execution
    from papervault.library.services.classify import EXTRACT, classify
    lib, left, right, ledger, now = pair
    campaign = ledger.start_campaign(cap=1, keys=[right.key])
    token = ReopenClaim(key=right.key, generation=1, lease_id="last-download", lease_expiry=now[0] + 60,
                        campaign_id=campaign)
    ledger.reserve(token)
    ledger.dispatch(token)
    ledger.complete(token)
    right.reopen_claim = token.model_copy(update={"phase": "completed"})
    ledger.finish_campaign(campaign, drained=True)
    lib.pdf_path(right.key).write_bytes(b"acquired on last admitted download")
    right.download_status = "ok"
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    assert ledger.key_budget(left.key)["spent"] == ledger.key_budget(left.key)["budget"] == 1
    assert classify(left, lib) == EXTRACT
    @guard_execution
    async def extraction(paper, library):
        return library.pdf_path(paper.key).read_bytes()
    assert await extraction(left, lib) == b"acquired on last admitted download"


async def test_stop_rule_identity_follows_durable_merge_redirect_for_late_executor(pair, monkeypatch):
    from papervault.library.claims_v2 import record_generation
    from papervault.library.stop_rules import evaluate_round
    monkeypatch.setenv("PAPERVAULT_STOPRULES_V2", "1")
    lib, left, right, ledger, now = pair
    assert await coordinator(pair).merge_records(left.key, right.key) is left
    assert ledger.merged_into(right.key) == left.key
    campaign = ledger.start_campaign(cap=2, keys=[left.key])
    store = ClaimStore(lib, ledger=ledger, clock=lambda: now[0])
    token = await store.claim(left.key, expected_state=left.acquisition_state,
                              expected_generation=record_generation(left), campaign_id=campaign)
    assert token is not None
    # Inject a late executor on the absorbed key; compare its redirected identity,
    # not the old arXiv-only identity still stored on the alias record.
    lib._stoprule_active = {right.key: 1}
    findings = await evaluate_round(lib, ledger=ledger, accounting=None)
    assert findings[0].rule == "duplicate_inflight_identity"
    assert findings[0].keys == tuple(sorted((left.key, right.key)))
    assert ledger.campaign_halted(campaign)
