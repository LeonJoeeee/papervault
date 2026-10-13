"""A5 format fences stop v2 writers before touching unsupported durable data."""

import json

import pytest

from papervault.library.claims_v2 import BudgetLedger, ClaimStore, LedgerError
from papervault.library.merge_v2 import MergeStore
from papervault.library.store import Library


def tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("current,version", [(1, 1), (1, 0), (2, 2), (2, 1), (9, 8)])
def test_matrix_reads_current_and_predecessor(current, version):
    from papervault.library.compat_v2 import check_readable
    assert check_readable(version, current_version=current) is None


@pytest.mark.parametrize("current,version", [(1, 2), (2, 3), (9, 7), (1, -1),
                                             (1, True), (1, "1"), (1, None), (1, 1.0)])
def test_matrix_refuses_newer_too_old_and_malformed(current, version):
    from papervault.library.compat_v2 import CompatibilityError, check_readable
    with pytest.raises(CompatibilityError):
        check_readable(version, current_version=current)


@pytest.fixture
def vault(tmp_path):
    lib = Library(tmp_path)
    lib.upsert({"title": "Current durable format compatibility fixture", "doi": "10.1234/compat"})
    lib.save()
    return lib


@pytest.mark.parametrize("surface", ["index", "identity", "budget", "claims", "lease", "queue"])
@pytest.mark.parametrize("entry", [ClaimStore, MergeStore])
def test_future_index_surface_refuses_v2_entry_without_writes(vault, surface, entry):
    from papervault.library.compat_v2 import CompatibilityError, FORMAT_VERSIONS
    data = json.loads(vault.index_path.read_text())
    data["acquisition_versions"] = {surface: FORMAT_VERSIONS[surface] + 1}
    vault.index_path.write_text(json.dumps(data))  # explicitly injected future-format fixture
    before = tree_bytes(vault.root)
    with pytest.raises(CompatibilityError, match=surface):
        entry(vault)
    assert tree_bytes(vault.root) == before


@pytest.mark.parametrize("version", [0, 1])
def test_index_container_current_and_predecessor_read_without_mutation(vault, version):
    from papervault.library.state_v2 import check_index_readable
    data = json.loads(vault.index_path.read_text())
    data["version"] = version
    vault.index_path.write_text(json.dumps(data))
    before = tree_bytes(vault.root)
    check_index_readable(vault.root)
    assert Library(vault.root).get(vault.keys()[0]).doi == "10.1234/compat"
    assert tree_bytes(vault.root) == before


def test_newer_container_is_checked_by_v2_without_retrofitting_legacy_loader(vault):
    from papervault.library.compat_v2 import CompatibilityError
    data = json.loads(vault.index_path.read_text())
    data["version"] = 2
    vault.index_path.write_text(json.dumps(data))
    legacy = Library(vault.root)  # legacy reader retains its own existing semantics
    assert legacy.keys() == vault.keys()
    before = tree_bytes(vault.root)
    with pytest.raises(CompatibilityError, match="index"):
        ClaimStore(legacy)
    assert tree_bytes(vault.root) == before


@pytest.mark.parametrize("flags", [False, True])
def test_current_formats_are_byte_identical_with_checks_always_on(vault, monkeypatch, flags):
    from papervault.library.compat_v2 import FORMAT_VERSIONS, check_readable
    from papervault.library.state_v2 import check_index_readable
    for name in ("TERMINAL", "IDENTITY", "REOPEN", "MERGE", "STOPRULES"):
        monkeypatch.setenv(f"PAPERVAULT_{name}_V2", "1" if flags else "0")
    ledger = BudgetLedger(vault.root, clock=lambda: 100.0)
    campaign = ledger.start_campaign(cap=1, keys=vault.keys())
    before = tree_bytes(vault.root)
    for _ in range(2):
        for version in FORMAT_VERSIONS.values():
            check_readable(version, current_version=version)
        check_index_readable(vault.root)
        assert ClaimStore(vault).ledger.spent(campaign) == 0
        MergeStore(vault)
        assert ledger.events()[0]["version"] == 1
    assert tree_bytes(vault.root) == before
    # The guard itself adds no markers or fields to an ordinary save.
    ClaimStore(vault).save()
    assert vault.index_path.read_bytes() == before["index.json"]


@pytest.mark.parametrize("version", [0, 1, 2])
def test_budget_matrix_and_append_refusal(vault, version):
    ledger = BudgetLedger(vault.root, clock=lambda: 100.0)
    campaign = ledger.start_campaign(cap=1, keys=vault.keys())
    row = json.loads(ledger.path.read_text())
    row["version"] = version
    ledger.path.write_text(json.dumps(row) + "\n")
    before = tree_bytes(vault.root)
    if version <= 1:
        assert ledger.spent(campaign) == 0
        assert tree_bytes(vault.root) == before
    else:
        with pytest.raises(LedgerError, match="budget"):
            ledger.spent(campaign)
        with pytest.raises(LedgerError):
            ledger.finish_campaign(campaign, drained=True)
        assert tree_bytes(vault.root) == before


def test_future_claim_event_format_is_not_downgraded_by_replay(vault):
    ledger = BudgetLedger(vault.root, clock=lambda: 100.0)
    ledger.start_campaign(cap=1, keys=vault.keys())
    row = json.loads(ledger.path.read_text())
    row["acquisition_versions"] = {"claims": 2}
    ledger.path.write_text(json.dumps(row) + "\n")
    before = tree_bytes(vault.root)
    with pytest.raises(LedgerError, match="claims"):
        ledger.events()
    assert tree_bytes(vault.root) == before


def test_backup_versions_are_checked_when_primary_is_unreadable(vault):
    from papervault.library.compat_v2 import CompatibilityError
    from papervault.library.state_v2 import check_index_readable
    data = json.loads(vault.index_path.read_text())
    data["version"] = 2
    vault.index_bak_path.write_text(json.dumps(data))
    vault.index_path.write_bytes(b"{")
    before = tree_bytes(vault.root)
    with pytest.raises(CompatibilityError, match="index"):
        check_index_readable(vault.root)
    assert tree_bytes(vault.root) == before


@pytest.mark.parametrize("versions", [{"future_surface": 1}, None, [], {"lease": True}])
def test_unknown_or_malformed_version_header_refuses(vault, versions):
    from papervault.library.compat_v2 import CompatibilityError
    data = json.loads(vault.index_path.read_text())
    data["acquisition_versions"] = versions
    vault.index_path.write_text(json.dumps(data))
    with pytest.raises(CompatibilityError):
        ClaimStore(vault)


async def test_future_format_at_publication_refuses_before_artifact_or_accounting_changes(vault, monkeypatch):
    from papervault.library.compat_v2 import CompatibilityError
    from papervault.library.reopen_v2 import StagedAttempt
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    paper, _ = vault.upsert({"title": "Publication version fence fixture", "doi": "10.1234/publication",
                             "download_status": "failed",
                             "terminal_state": {"retrieval": "exhausted", "policy": "clear"}})
    vault.save()
    ledger = BudgetLedger(vault.root, clock=lambda: 100.0)
    store = ClaimStore(vault, ledger=ledger, clock=lambda: 100.0)
    campaign = ledger.start_campaign(cap=1, keys=[paper.key])
    token = await store.claim(paper.key, expected_state=paper.acquisition_state,
                              expected_generation=0, campaign_id=campaign)
    assert await store.pre_dispatch(token)
    stage = StagedAttempt(vault, token)
    try:
        stage.path("pdf").write_bytes(b"must not publish across a format fence")
        stage.library.get(paper.key).download_status = "ok"
        data = json.loads(vault.index_path.read_text())
        data["version"] = 2
        vault.index_path.write_text(json.dumps(data))
        before = tree_bytes(vault.root)
        with pytest.raises(CompatibilityError):
            await stage.writeback(store)
        assert tree_bytes(vault.root) == before
        assert not vault.pdf_path(paper.key).exists()
    finally:
        stage.cleanup()


async def test_future_format_at_claim_refuses_before_reservation(vault, monkeypatch):
    from papervault.library.compat_v2 import CompatibilityError
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    paper, _ = vault.upsert({"title": "Claim version fence fixture", "doi": "10.1234/reservation",
                             "download_status": "failed",
                             "terminal_state": {"retrieval": "exhausted", "policy": "clear"}})
    vault.save()
    ledger = BudgetLedger(vault.root, clock=lambda: 100.0)
    store = ClaimStore(vault, ledger=ledger, clock=lambda: 100.0)
    campaign = ledger.start_campaign(cap=1, keys=[paper.key])
    data = json.loads(vault.index_path.read_text())
    data["version"] = 2
    vault.index_path.write_text(json.dumps(data))
    before = tree_bytes(vault.root)
    with pytest.raises(CompatibilityError):
        await store.claim(paper.key, expected_state=paper.acquisition_state,
                          expected_generation=0, campaign_id=campaign)
    assert tree_bytes(vault.root) == before


async def test_future_format_recovery_refund_refuses_before_accounting_changes(vault, monkeypatch):
    from papervault.library.compat_v2 import CompatibilityError
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    paper, _ = vault.upsert({"title": "Refund recovery format fence fixture", "doi": "10.1234/refund",
                             "download_status": "failed",
                             "terminal_state": {"retrieval": "exhausted", "policy": "clear"}})
    vault.save()
    ledger = BudgetLedger(vault.root, clock=lambda: 100.0)
    store = ClaimStore(vault, ledger=ledger, clock=lambda: 100.0)
    campaign = ledger.start_campaign(cap=1, keys=[paper.key])
    token = await store.claim(paper.key, expected_state=paper.acquisition_state,
                              expected_generation=0, campaign_id=campaign)
    data = json.loads(vault.index_path.read_text())
    data["version"] = 2
    vault.index_path.write_text(json.dumps(data))
    before = tree_bytes(vault.root)
    with pytest.raises(CompatibilityError):
        await store.not_sent(token)
    assert tree_bytes(vault.root) == before
    assert ledger.spent(campaign) == 1


async def test_future_format_cancellation_refuses_before_quarantine_save(vault, monkeypatch):
    import asyncio
    from papervault.library.compat_v2 import CompatibilityError
    from papervault.library.merge_v2 import guard_execution
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    entered = asyncio.Event()
    @guard_execution
    async def work(paper, library):
        entered.set()
        await asyncio.Event().wait()
    task = asyncio.create_task(work(vault.get(vault.keys()[0]), vault))
    await entered.wait()
    data = json.loads(vault.index_path.read_text())
    data["version"] = 2
    vault.index_path.write_text(json.dumps(data))
    before = tree_bytes(vault.root)
    task.cancel()
    outcome, = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(outcome, CompatibilityError)
    assert tree_bytes(vault.root) == before


def test_future_index_refuses_alias_resolution(vault):
    from papervault.library.compat_v2 import CompatibilityError
    from papervault.library.merge_v2 import resolve_alias
    data = json.loads(vault.index_path.read_text())
    data["version"] = 2
    vault.index_path.write_text(json.dumps(data))
    with pytest.raises(CompatibilityError):
        resolve_alias(vault, vault.keys()[0])


async def test_future_index_refuses_tracked_executor_before_entry(vault, monkeypatch):
    from papervault.library.compat_v2 import CompatibilityError
    from papervault.library.merge_v2 import guard_execution
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    data = json.loads(vault.index_path.read_text())
    data["version"] = 2
    vault.index_path.write_text(json.dumps(data))
    entered = []
    @guard_execution
    async def work(paper, library):
        entered.append(paper.key)
    with pytest.raises(CompatibilityError):
        await work(vault.get(vault.keys()[0]), vault)
    assert not entered and not vault._merge_active
