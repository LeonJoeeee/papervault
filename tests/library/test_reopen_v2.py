"""Campaign and queue fixtures: no upstream network and no live vault."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from dataclasses import replace

import pytest

from papervault.library import state_v2
from papervault.library.models import Paper
from papervault.library.services import concurrency, download_queue
from papervault.library.store import Library


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    lib = Library(tmp_path)
    paper, _ = lib.upsert({"title": "A fixture paper for bounded campaign execution",
                           "authors": ["Curie"], "year": 2026, "doi": "10.1234/reopen"})
    paper.download_status = "failed"
    paper.terminal_state = state_v2.TerminalState(retrieval=state_v2.Retrieval.EXHAUSTED,
                                                 policy=state_v2.Policy.CLEAR)
    lib.save()
    return lib, paper


def tree_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


async def test_staged_extraction_evidence_survives_authorized_writeback_and_cleanup(vault):
    from papervault.library.reopen_v2 import StagedAttempt
    from papervault.library.services.extraction_registration import begin_extraction, finish_extraction
    from papervault.store.extraction import ExtractionStore

    lib, paper = vault
    claims, token = await prepare(vault)
    assert await claims.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    staged_paper = stage.library.get(paper.key)
    request = begin_extraction(staged_paper, stage.library, b"observed provider markdown",
        engine="firecrawl", source_role="provider_markdown")
    completion = finish_extraction(staged_paper, stage.library, request, "# Immutable staged output")
    assert await stage.writeback(claims)
    stage.cleanup()
    retained = ExtractionStore(lib.root).completion(completion["completion_id"])
    assert retained == completion
    assert lib.get(paper.key).extraction_completion_id == completion["completion_id"]
    assert (lib.root / lib.get(paper.key).md_path).read_text() == "# Immutable staged output"
    assert ExtractionStore(lib.root).read_artifact(retained["input_manifest"]["artifacts"][0]) == b"observed provider markdown"


async def test_malformed_staged_evidence_blocks_cleanup(vault):
    from papervault.library.reopen_v2 import StagedAttempt
    from papervault.library.services.extraction_registration import begin_extraction, finish_extraction

    lib, paper = vault
    claims, token = await prepare(vault)
    assert await claims.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    try:
        candidate = stage.library.get(paper.key)
        request = begin_extraction(candidate, stage.library, b"observed provider markdown",
            engine="firecrawl", source_role="provider_markdown")
        completion = finish_extraction(candidate, stage.library, request, "# Retain this staged output")
        output = stage.root / completion["output_inventory"][0]["path"]
        receipt = output.parent / "completion.json"
        receipt.write_bytes(b'{"incomplete":')

        with pytest.raises(ValueError):
            stage.cleanup()

        assert output.read_text() == "# Retain this staged output"
        assert receipt.read_bytes() == b'{"incomplete":'
    finally:
        stage.library.close()


@pytest.mark.parametrize("disposition", ["expired", "identical", "uncommitted"])
async def test_staged_evidence_survives_refusal_without_bypassing_claim(vault, disposition):
    from papervault.library.reopen_v2 import StagedAttempt
    from papervault.library.services.extraction_registration import begin_extraction, finish_extraction, recover_completions
    from papervault.store.extraction import ExtractionStore

    lib, paper = vault
    body = "# Exact completed output"
    if disposition == "identical":
        lib.md_path(paper.key).write_text(body)
        paper.md_path = str(lib.md_path(paper.key).relative_to(lib.root))
        lib.save(force=True, durable=True)
    claims, token = await prepare(vault)
    assert await claims.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    candidate = stage.library.get(paper.key)
    request = begin_extraction(candidate, stage.library, b"provider response",
        engine="firecrawl", source_role="provider_markdown")
    completion = finish_extraction(candidate, stage.library, request, body)
    if disposition == "expired":
        claims.clock = lambda: token.lease_expiry + 1
        assert not await stage.writeback(claims)
    elif disposition == "identical":
        assert await stage.writeback(claims)
    else:
        ExtractionStore(lib.root).retain_from(ExtractionStore(stage.root), completion)
    stage.cleanup()
    assert ExtractionStore(lib.root).completion(completion["completion_id"]) == completion
    report = recover_completions(lib)
    if disposition == "identical":
        assert paper.extraction_completion_id == completion["completion_id"]
    else:
        assert paper.extraction_completion_id is None
        assert not lib.has_extract(paper.key, "md")
        assert report["blocked"]


def test_cli_dry_run_end_to_end_writes_nothing(vault):
    lib, paper = vault
    before = tree_bytes(lib.root)
    env = {**os.environ, "PAPERVAULT_REOPEN_V2": "1"}
    env.pop("PAPERVAULT_VAULT", None)
    run = subprocess.run([sys.executable, "-m", "papervault.library.cli", "--library-path", str(lib.root),
                          "reopen-campaign", "--cap", "10", "--dry-run", "--json"],
                         capture_output=True, text=True, env=env, timeout=15)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report["keys"] == [paper.key] and report["dry_run"] is True
    assert tree_bytes(lib.root) == before


@pytest.mark.parametrize("cap,expected,deferred", [
    (1, ["ZListed"], ["AListed"]), (2, ["ZListed", "AListed"], []),
])
def test_key_file_preserves_order_cap_and_reports_every_refusal(vault, monkeypatch, capsys, cap, expected, deferred):
    from papervault import ops_guards
    from papervault.library.cli import main
    lib, unrelated = vault
    for key in ("ZListed", "AListed", "Blocked"):
        lib._papers[key] = Paper(key=key, doi=f"10.1234/{key}", download_status="metadata_only",
            terminal_state=state_v2.TerminalState(retrieval=state_v2.Retrieval.EXHAUSTED,
                policy=state_v2.Policy.UNKNOWN if key == "Blocked" else state_v2.Policy.CLEAR))
    lib.save(force=True)
    keys = lib.root.parent / "campaign.keys"
    keys.write_text("# file order differs from alphabetical order\nZListed\nBlocked\nMissing\nAListed # second\nZListed\n")
    monkeypatch.setenv("PAPERVAULT_VAULT", str(lib.root))
    monkeypatch.setattr(ops_guards, "active_service_units", lambda *a, **kw: [])
    attempted = []
    def download(paper, stage, **providers):
        attempted.append(paper.key)
        paper.download_status = "metadata_only"
        return False
    monkeypatch.setattr(download_queue, "download_paper", download)
    args = ["reopen-campaign", "--keys", str(keys), "--cap", str(cap), "--json"]
    before = tree_bytes(lib.root)
    assert main([*args, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["keys"] == expected and preview["selected"] == cap
    assert preview["missing_keys"] == ["Missing"] and preview["missing_count"] == 1
    assert preview["rejected"] == [{"key": "Blocked", "reason": "not_reopenable"}]
    assert preview["rejected_count"] == 1 and preview["deferred_keys"] == deferred
    assert tree_bytes(lib.root) == before
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["keys"] == expected and result["completed"] == cap
    assert result["missing_keys"] == preview["missing_keys"]
    assert result["rejected"] == preview["rejected"]
    assert result["deferred_keys"] == deferred
    assert sorted(attempted) == sorted(expected)
    assert Library(lib.root).get(unrelated.key).reopen_claim is None


def test_empty_key_file_never_falls_back_to_whole_library(vault, monkeypatch, capsys):
    from papervault.library.cli import main
    lib, _ = vault
    monkeypatch.setenv("PAPERVAULT_VAULT", str(lib.root))
    keys = lib.root.parent / "empty.keys"
    keys.write_text("# select nobody\n")
    before = tree_bytes(lib.root)
    assert main(["reopen-campaign", "--keys", str(keys), "--cap", "1", "--dry-run", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["keys"] == [] and result["selected"] == 0
    assert tree_bytes(lib.root) == before


def test_empty_key_filename_refuses_instead_of_selecting_whole_library(vault, monkeypatch, capsys):
    from papervault.library.cli import main
    lib, _ = vault
    monkeypatch.setenv("PAPERVAULT_VAULT", str(lib.root))
    before = tree_bytes(lib.root)
    assert main(["reopen-campaign", "--keys", "", "--cap", "1", "--dry-run", "--json"]) == 2
    assert capsys.readouterr().err
    assert tree_bytes(lib.root) == before


def test_human_campaign_output_reports_missing_and_rejected_keys(vault, monkeypatch, capsys):
    from papervault.library.cli import main
    lib, paper = vault
    paper.terminal_state = replace(paper.acquisition_state, policy=state_v2.Policy.UNKNOWN)
    lib.save(force=True)
    keys = lib.root.parent / "refused.keys"
    keys.write_text(f"Missing\n{paper.key}\n")
    monkeypatch.setenv("PAPERVAULT_VAULT", str(lib.root))
    assert main(["reopen-campaign", "--keys", str(keys), "--cap", "1", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Missing" in output and paper.key in output and "not_reopenable" in output


@pytest.mark.parametrize("override", [False, True], ids=["excluded", "scoped_override"])
def test_cli_dry_run_and_apply_agree_on_off_domain_policy(vault, monkeypatch, capsys, override):
    from papervault import ops_guards
    from papervault.library import download
    from papervault.library.cli import main
    lib, paper = vault
    paper.domain_status = "off_domain"
    if override:
        paper.terminal_state = replace(paper.acquisition_state, policy=state_v2.Policy.OVERRIDE,
            required_revision=4, override=state_v2.PolicyOverride(
                scope=("acquire",), reasons=("off_domain",), revision=4, key=paper.key))
    lib.save()
    expected = [paper.key] if override else []
    attempted = []
    def tier(candidate):
        attempted.append(candidate.key)
        assert Library(lib.root).get(candidate.key).reopen_claim.phase == "dispatched"
        return None
    monkeypatch.setattr(ops_guards, "active_service_units", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", tier)])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    args = ["--library-path", str(lib.root), "reopen-campaign", "--cap", "1", "--json"]
    before = tree_bytes(lib.root)
    assert main([*args, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["keys"] == expected and preview["selected"] == len(expected)
    assert tree_bytes(lib.root) == before and attempted == []
    assert main(args) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["keys"] == preview["keys"]
    assert applied["selected"] == preview["selected"]
    assert applied["completed"] == len(expected) and attempted == expected
    persisted = Library(lib.root).get(paper.key)
    if override:
        assert persisted.reopen_claim.phase == "completed"
        assert persisted.reopen_claim.generation == 1
    else:
        assert persisted.reopen_claim is None
        assert not (lib.root / "reopen-budget.jsonl").exists()
    assert persisted.acquisition_state.retrieval == state_v2.Retrieval.EXHAUSTED
    assert persisted.acquisition_state.policy == (state_v2.Policy.OVERRIDE if override else state_v2.Policy.EXCLUDED)
    assert persisted.acquisition_state.override == paper.acquisition_state.override
    assert persisted.domain_status == "off_domain"


def test_dry_run_missing_index_never_creates_vault(tmp_path, monkeypatch):
    from papervault.library.cli import main
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    root = tmp_path / "absent"
    assert main(["--library-path", str(root), "reopen-campaign", "--cap", "1", "--dry-run", "--json"]) == 2
    assert not root.exists()


@pytest.mark.parametrize("block", ["unknown", "excluded", "alias", "quarantine", "unresolved", "acquired"])
def test_selector_respects_all_dimensions_without_mutation(vault, block):
    from papervault.library.reopen_v2 import inspect_campaign
    lib, paper = vault
    changes = {"unknown": {"policy": state_v2.Policy.UNKNOWN},
               "excluded": {"policy": state_v2.Policy.EXCLUDED},
               "alias": {"merge": state_v2.Merge.ALIAS, "alias_key": "another"},
               "quarantine": {"quarantine": True},
               "acquired": {"retrieval": state_v2.Retrieval.ACQUIRED}}
    if block == "unresolved":
        paper.doi = ""
    else:
        paper.terminal_state = replace(paper.acquisition_state, **changes[block])
    lib.save()
    before = tree_bytes(lib.root)
    assert inspect_campaign(lib.root, cap=10)["keys"] == []
    assert tree_bytes(lib.root) == before


async def test_real_campaign_reserves_before_dispatch_and_does_not_recover_unrelated(vault, monkeypatch):
    from papervault.library.claims_v2 import BudgetLedger
    from papervault.library.reopen_v2 import run_campaign
    lib, paper = vault
    unrelated, _ = lib.upsert({"title": "An unrelated pending acquisition must stay queued elsewhere",
                              "doi": "10.1234/unrelated"})
    attempted = []
    def download(candidate, stage, **providers):
        live = lib.get(candidate.key)
        assert BudgetLedger(lib.root).spent(live.reopen_claim.campaign_id) == 1
        assert Library(lib.root).get(candidate.key).reopen_claim.phase == "dispatched"
        assert stage.root != lib.root
        attempted.append(candidate.key)
        stage.pdf_path(candidate.key).write_bytes(b"%PDF-1.4\nfixture")
        candidate.pdf_path = f"pdfs/{candidate.key}.pdf"
        candidate.download_status = "ok"
        return True
    monkeypatch.setattr(download_queue, "download_paper", download)
    report = await run_campaign(lib, cap=1)
    assert report["completed"] == 1 and attempted == [paper.key]
    assert lib.has_pdf(paper.key) and paper.acquisition_state.retrieval == state_v2.Retrieval.ACQUIRED
    assert unrelated.download_status == "pending" and not lib.has_pdf(unrelated.key)
    assert not list((lib.root / ".reopen-staging").glob("*"))


async def prepare(vault):
    from papervault.library.claims_v2 import ClaimStore
    lib, paper = vault
    store = ClaimStore(lib)
    campaign = store.ledger.start_campaign(cap=1, keys=[paper.key])
    token = await store.claim(paper.key, expected_state=paper.acquisition_state,
                              expected_generation=0, campaign_id=campaign)
    return store, token


@pytest.mark.parametrize("artifact", ["pdf", "md", "txt"])
@pytest.mark.parametrize("status", ["ok", "failed"])
async def test_stale_generation_drops_state_and_artifact(vault, artifact, status):
    from papervault.library.reopen_v2 import StagedAttempt
    lib, paper = vault
    store, token = await prepare(vault)
    assert await store.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    result = stage.library.get(paper.key)
    stage.path(artifact).write_bytes(b"stale artifact")
    result.download_status = status
    await store.quarantine(token, reason="fenced execution")
    assert await store.release_quarantine(paper.key, human_reason="owner release")
    before = tree_bytes(lib.root)
    assert not await stage.writeback(store)
    assert tree_bytes(lib.root) == before
    assert not getattr(lib, f"{artifact}_path")(paper.key).exists()
    stage.cleanup()


async def test_completion_replay_after_restart_applies_once(vault):
    from papervault.library.claims_v2 import ClaimStore
    from papervault.library.reopen_v2 import StagedAttempt
    lib, paper = vault
    store, token = await prepare(vault)
    assert await store.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    stage.path("pdf").write_bytes(b"%PDF replay fixture")
    result = stage.library.get(paper.key)
    result.download_status, result.pdf_path = "ok", f"pdfs/{paper.key}.pdf"
    assert await stage.writeback(store)
    before = tree_bytes(lib.root)
    restarted = ClaimStore(Library(lib.root))
    assert not await stage.writeback(restarted)
    assert tree_bytes(lib.root) == before
    assert lib.manifest_path.read_text().count('"event": "reopen_complete"') == 1
    assert restarted.ledger.spent(token.campaign_id) == 1
    stage.cleanup()


@pytest.mark.parametrize("mutation", ["policy", "identity", "artifact"])
async def test_publication_rechecks_live_policy_identity_and_artifact(vault, mutation):
    from papervault.library.reopen_v2 import StagedAttempt
    lib, paper = vault
    store, token = await prepare(vault)
    await store.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    stage.path("pdf").write_bytes(b"staged PDF")
    stage.library.get(paper.key).download_status = "ok"
    if mutation == "policy":
        paper.terminal_state = replace(paper.acquisition_state, policy=state_v2.Policy.EXCLUDED)
    elif mutation == "identity":
        paper.doi = "10.1234/changed"
    else:
        lib.pdf_path(paper.key).write_bytes(b"newer live PDF")
    assert not await stage.writeback(store)
    assert paper.download_status == "pending"
    assert (lib.pdf_path(paper.key).read_bytes() if lib.has_pdf(paper.key) else None) == (
        b"newer live PDF" if mutation == "artifact" else None)
    stage.cleanup()


async def test_publication_propagates_md_deletion_and_preserves_metadata(vault):
    from papervault.library.reopen_v2 import StagedAttempt
    lib, paper = vault
    lib.md_path(paper.key).write_text("rejected text")
    store, token = await prepare(vault)
    await store.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    stage.path("md").unlink()
    stage.library.get(paper.key).download_status = "metadata_only"
    paper.abstract = "New concurrent metadata"
    assert await stage.writeback(store)
    assert not lib.has_extract(paper.key, "md")
    assert paper.abstract == "New concurrent metadata"
    assert paper.acquisition_state.policy == state_v2.Policy.CLEAR
    stage.cleanup()


async def test_real_providers_deny_ordinary_callers_and_unreadable_budget(vault):
    from papervault.library.download import download_paper
    lib, paper = vault
    store, token = await prepare(vault)
    assert not state_v2.valid_lease(paper)
    assert state_v2.valid_lease(paper, lease_id=token.lease_id)
    assert state_v2.admitted_budget(paper, library=lib)
    assert download_paper(paper, lib) is False
    store.ledger.path.write_bytes(b"truncated")
    assert not state_v2.admitted_budget(paper, library=lib)
    assert paper.download_status == "pending"


async def test_policy_change_during_network_slot_wait_refunds_proven_not_sent(vault, monkeypatch):
    lib, paper = vault
    store, token = await prepare(vault)
    waiting = asyncio.Event()
    class Slot(asyncio.Semaphore):
        async def acquire(self):
            waiting.set()
            return await super().acquire()
    slot = Slot(0)
    monkeypatch.setattr(concurrency, "network_sem", slot)
    queue = download_queue.DownloadQueue(lib, num_workers=0)
    task = asyncio.create_task(queue._process_one(paper.key, claim=token))
    await asyncio.wait_for(waiting.wait(), 5)
    paper.terminal_state = replace(paper.acquisition_state, policy=state_v2.Policy.EXCLUDED)
    slot.release()
    await task
    assert paper.reopen_claim.phase == "not_sent"
    assert paper.acquisition_state.policy == state_v2.Policy.EXCLUDED
    assert store.ledger.spent(token.campaign_id) == 0


async def test_cancellation_quarantines_thread_result_without_refund(vault, monkeypatch):
    from papervault.library.claims_v2 import ClaimStore
    lib, paper = vault
    store, token = await prepare(vault)
    slot = asyncio.Semaphore(1)
    monkeypatch.setattr(concurrency, "network_sem", slot)
    quarantined = asyncio.Event()
    original_quarantine = ClaimStore._quarantine
    def quarantine(claim_store, candidate, reason):
        original_quarantine(claim_store, candidate, reason)
        quarantined.set()
    monkeypatch.setattr(ClaimStore, "_quarantine", quarantine)
    started, release, drained = threading.Event(), threading.Event(), threading.Event()
    def slow(candidate, stage, **_):
        started.set()
        release.wait(5)
        stage.pdf_path(candidate.key).write_bytes(b"late result")
        candidate.download_status = "ok"
        drained.set()
        return True
    monkeypatch.setattr(download_queue, "download_paper", slow)
    queue = download_queue.DownloadQueue(lib, num_workers=0)
    task = asyncio.create_task(queue._process_one(paper.key, claim=token))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.wait_for(quarantined.wait(), 5)
    assert paper.acquisition_state.quarantine and store.ledger.spent(token.campaign_id) == 1
    slots_during_uncertainty = slot._value
    assert list((lib.root / ".reopen-staging").glob("*"))
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(drained.wait, 5)
    assert not lib.has_pdf(paper.key)
    assert slots_during_uncertainty == 0
    assert slot._value == 1


async def test_flag_off_blocks_campaign_without_any_writes(vault, monkeypatch):
    from papervault.library.reopen_v2 import run_campaign
    lib, _ = vault
    monkeypatch.delenv("PAPERVAULT_REOPEN_V2")
    before = tree_bytes(lib.root)
    with pytest.raises(ValueError, match="disabled"):
        await run_campaign(lib, cap=1)
    assert tree_bytes(lib.root) == before


async def test_kill_switch_never_turns_claim_into_ordinary_work(vault, monkeypatch):
    from papervault.library.download import download_paper
    lib, paper = vault
    _, _token = await prepare(vault)
    monkeypatch.delenv("PAPERVAULT_REOPEN_V2")
    before = tree_bytes(lib.root)
    assert download_paper(paper, lib) is False
    queue = download_queue.DownloadQueue(lib, num_workers=0)
    await queue._process_one(paper.key)
    # The existing download tracer still records the refused invocation.
    assert {k: v for k, v in tree_bytes(lib.root).items() if k != "manifest.log"} == {
        k: v for k, v in before.items() if k != "manifest.log"}


def test_flag_off_paper_bytes_match_original_shape(monkeypatch):
    monkeypatch.delenv("PAPERVAULT_REOPEN_V2", raising=False)
    original = Paper(key="old", download_status="failed")
    assert "reopen_claim" not in json.loads(original.model_dump_json())


@pytest.mark.parametrize("answer", ["active", "probe_error"])
def test_cli_apply_refuses_cowriter_or_unavailable_probe(vault, monkeypatch, answer):
    from papervault.library.cli import main
    from papervault import ops_guards
    lib, _ = vault
    before = tree_bytes(lib.root)
    def probe(*_, **__):
        if answer == "probe_error":
            raise OSError("systemd unavailable")
        return subprocess.CompletedProcess([], 0, stdout="active\n", stderr="")
    monkeypatch.setattr(ops_guards.subprocess, "run", probe)
    assert main(["--library-path", str(lib.root), "reopen-campaign", "--cap", "1", "--json"]) == 2
    assert tree_bytes(lib.root) == before


async def test_stale_queue_token_never_adopts_new_generation(vault, monkeypatch):
    lib, paper = vault
    store, token = await prepare(vault)
    await store.quarantine(token, reason="owner fencing")
    await store.release_quarantine(paper.key, human_reason="old executor fenced")
    store.ledger.finish_campaign(token.campaign_id, drained=True)
    next_campaign = store.ledger.start_campaign(cap=1, keys=[paper.key])
    current = await store.claim(paper.key, expected_state=paper.acquisition_state,
                                expected_generation=2, campaign_id=next_campaign)
    assert current.generation == 3
    before = tree_bytes(lib.root)
    queue = download_queue.DownloadQueue(lib, num_workers=0)
    await queue._process_one(paper.key, claim=token)
    assert tree_bytes(lib.root) == before
    assert paper.reopen_claim.phase == "queued"


async def test_live_identifier_collision_never_publishes_staged_asset(vault):
    from papervault.library.reopen_v2 import StagedAttempt
    lib, paper = vault
    paper.doi, paper.url = "", "https://example.invalid/paper"
    store, token = await prepare(vault)
    await store.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    candidate = stage.library.get(paper.key)
    candidate.doi = "10.1234/new-owner"
    candidate.download_status = "ok"
    stage.path("pdf").write_bytes(b"staged collision")
    twin, _ = lib.upsert({"title": "A distinct paper acquired the same identity meanwhile",
                         "authors": ["Einstein"], "doi": "10.1234/new-owner"})
    assert not await stage.writeback(store)
    assert not lib.has_pdf(paper.key) and paper.doi == ""
    assert lib.find(doi="10.1234/new-owner") is twin
    stage.cleanup()


async def test_running_executor_is_quarantined_at_lease_expiry(vault, monkeypatch):
    from papervault.library import reopen_v2
    from papervault.library.claims_v2 import ClaimStore
    from papervault.library.reopen_v2 import run_campaign
    lib, paper = vault
    now = [1791849600.0]
    expiry_waiting, expire, quarantined = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def sleep_until_expiry(seconds):
        assert seconds == 60
        expiry_waiting.set()
        await expire.wait()
    original_quarantine = ClaimStore._quarantine
    def quarantine(store, candidate, reason):
        original_quarantine(store, candidate, reason)
        quarantined.set()
    monkeypatch.setattr(reopen_v2.asyncio, "sleep", sleep_until_expiry)
    monkeypatch.setattr(ClaimStore, "_quarantine", quarantine)
    started, release, drained = threading.Event(), threading.Event(), threading.Event()
    def blocked(candidate, stage, **_):
        started.set()
        release.wait(5)
        stage.pdf_path(candidate.key).write_bytes(b"expired artifact")
        candidate.download_status = "ok"
        drained.set()
        return True
    monkeypatch.setattr(download_queue, "download_paper", blocked)
    task = asyncio.create_task(run_campaign(lib, cap=1, lease_seconds=60, clock=lambda: now[0]))
    assert await asyncio.to_thread(started.wait, 5)
    await asyncio.wait_for(expiry_waiting.wait(), 5)
    now[0] += 60
    expire.set()
    await asyncio.wait_for(quarantined.wait(), 5)
    try:
        assert paper.acquisition_state.quarantine
        assert not drained.is_set()
    finally:
        release.set()
    with pytest.raises(ValueError, match="uncertain"):
        await task
    assert not lib.has_pdf(paper.key)


async def test_publication_cannot_overwrite_current_retrieval_decision(vault):
    from papervault.library.reopen_v2 import StagedAttempt
    lib, paper = vault
    store, token = await prepare(vault)
    await store.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    stage.path("pdf").write_bytes(b"obsolete retrieval result")
    stage.library.get(paper.key).download_status = "ok"
    paper.terminal_state = replace(paper.acquisition_state, retrieval=state_v2.Retrieval.INELIGIBLE)
    assert not await stage.writeback(store)
    assert paper.acquisition_state.retrieval == state_v2.Retrieval.INELIGIBLE
    assert paper.download_status == "pending" and not lib.has_pdf(paper.key)
    stage.cleanup()


async def test_campaign_uses_real_cascade_admission_and_terminal_write(vault, monkeypatch):
    from papervault.library import download
    from papervault.library.reopen_v2 import run_campaign
    lib, paper = vault
    attempted = []
    def tier(candidate):
        attempted.append(candidate.key)
        assert Library(lib.root).get(candidate.key).reopen_claim.phase == "dispatched"
        return None
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", tier)])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    result = await run_campaign(lib, cap=1)
    assert attempted == [paper.key]
    assert result["completed"] == 1
    assert paper.download_status == "failed"
    assert Library(lib.root).get(paper.key).acquisition_state.retrieval == state_v2.Retrieval.EXHAUSTED


async def test_predispatch_refusal_preserves_another_writers_retrieval(vault):
    lib, paper = vault
    store, token = await prepare(vault)
    paper.download_status = "ok"
    paper.terminal_state = replace(paper.acquisition_state, retrieval=state_v2.Retrieval.VERIFIED)
    state = paper.acquisition_state
    assert not await store.pre_dispatch(token)
    assert paper.acquisition_state == state and paper.download_status == "ok"
    assert paper.reopen_claim.phase == "not_sent"
    assert store.ledger.spent(token.campaign_id) == 0


async def test_campaign_queue_entry_supersedes_stale_ordinary_entry(vault, monkeypatch):
    lib, paper = vault
    paper.download_status = "pending"
    queue = download_queue.DownloadQueue(lib, num_workers=1)
    queue.add(paper.key)
    paper.download_status = "failed"
    store, token = await prepare(vault)
    calls = []
    def download(candidate, stage, **_):
        calls.append(candidate.key)
        candidate.download_status = "failed"
        return False
    monkeypatch.setattr(download_queue, "download_paper", download)
    queue.add_reopen(token)
    await queue.start(recover=False)
    try:
        await asyncio.wait_for(queue._queue.join(), 5)
    finally:
        await queue.stop()
    assert calls == [paper.key]
    assert paper.reopen_claim.phase == "completed"
    assert store.ledger.spent(token.campaign_id) == 1


async def test_identity_mismatch_halts_remaining_campaign_dispatch(vault, monkeypatch):
    from papervault.library.claims_v2 import BudgetLedger
    from papervault.library.reopen_v2 import run_campaign
    lib, first = vault
    second, _ = lib.upsert({"title": "A second paper in the halted campaign fixture",
                           "authors": ["Einstein"], "doi": "10.1234/second"})
    second.download_status = "failed"
    lib.save()
    calls = []
    def mismatched(candidate, stage, **_):
        calls.append(candidate.key)
        candidate.doi = "10.1234/wrong"
        stage.pdf_path(candidate.key).write_bytes(b"mismatched asset")
        candidate.download_status = "ok"
        return True
    monkeypatch.setattr(download_queue, "download_paper", mismatched)
    queue = download_queue.DownloadQueue(lib, num_workers=1)
    try:
        with pytest.raises(ValueError, match="uncertain"):
            await run_campaign(lib, cap=2, queue=queue)
    finally:
        await queue.stop()
    assert len(calls) == 1
    stopped = next(p for p in (first, second) if p.key not in calls)
    assert stopped.reopen_claim.phase == "not_sent"
    assert not lib.has_pdf(first.key) and not lib.has_pdf(second.key)
    assert BudgetLedger(lib.root).spent(first.reopen_claim.campaign_id) == 1


def test_cli_apply_fixture_uses_real_queue_without_network(vault, monkeypatch, capsys):
    from papervault.library.cli import main
    from papervault.library import download
    from papervault import ops_guards
    lib, paper = vault
    monkeypatch.setattr(ops_guards.subprocess, "run", lambda *_args, **_kwargs:
                        subprocess.CompletedProcess([], 3, stdout="inactive\n", stderr=""))
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", lambda _: None)])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    assert main(["--library-path", str(lib.root), "reopen-campaign", "--cap", "1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["completed"] == 1
    assert Library(lib.root).get(paper.key).reopen_claim.phase == "completed"


def test_standalone_campaign_refuses_another_process_writer_lock(vault, monkeypatch):
    from filelock import FileLock
    from papervault.library.cli import main
    from papervault.library import download
    from papervault import ops_guards
    lib, _ = vault
    monkeypatch.setattr(ops_guards.subprocess, "run", lambda *_args, **_kwargs:
                        subprocess.CompletedProcess([], 3, stdout="inactive\n", stderr=""))
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", lambda _: None)])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    with FileLock(str(lib.root / ".reopen-campaign.lock")):
        before = tree_bytes(lib.root)
        assert main(["--library-path", str(lib.root), "reopen-campaign", "--cap", "1", "--json"]) == 2
        assert tree_bytes(lib.root) == before


async def test_halted_campaign_replay_keeps_uncertain_dispatch_charged(vault):
    lib, paper = vault
    store, token = await prepare(vault)
    assert await store.pre_dispatch(token)
    await store.quarantine(token, reason="dispatched outcome unknown")
    queue = download_queue.DownloadQueue(lib, num_workers=0)
    queue._reopen_halted.add(token.campaign_id)
    await queue._process_one(paper.key, claim=token)
    assert store.ledger.spent(token.campaign_id) == 1
    assert paper.reopen_claim.phase == "dispatched"
    assert paper.acquisition_state.quarantine


async def test_restart_preserves_halt_and_never_dispatches_remaining_queued_work(vault, monkeypatch):
    lib, first = vault
    second, _ = lib.upsert({"title": "A queued restart must retain the campaign halt",
                           "authors": ["Einstein"], "doi": "10.1234/restart"})
    second.download_status = "failed"
    from papervault.library.claims_v2 import ClaimStore
    store = ClaimStore(lib)
    campaign = store.ledger.start_campaign(cap=2, keys=[first.key, second.key])
    tokens = [await store.claim(p.key, expected_state=p.acquisition_state,
                                expected_generation=0, campaign_id=campaign) for p in (first, second)]
    await store.pre_dispatch(tokens[0])
    await store.quarantine(tokens[0], reason="identity mismatch halted campaign")
    calls = []
    monkeypatch.setattr(download_queue, "download_paper", lambda *_args, **_kwargs: calls.append("sent"))
    restarted = Library(lib.root)
    queue = download_queue.DownloadQueue(restarted, num_workers=1)
    await queue.start()
    try:
        await asyncio.wait_for(queue._queue.join(), 5)
    finally:
        await queue.stop()
    assert calls == []
    assert restarted.get(second.key).reopen_claim.phase == "not_sent"
    assert store.ledger.spent(campaign) == 1


async def test_cancellation_keeps_permit_even_when_quarantine_persistence_fails(vault, monkeypatch):
    from papervault.library.claims_v2 import ClaimStore
    from papervault.library.reopen_v2 import process_attempt
    lib, paper = vault
    _, token = await prepare(vault)
    slot = asyncio.Semaphore(1)
    monkeypatch.setattr(concurrency, "network_sem", slot)
    started, release = threading.Event(), threading.Event()
    def blocked(*_args, **_kwargs):
        started.set()
        release.wait(5)
        return False
    monkeypatch.setattr(download_queue, "download_paper", blocked)
    queue = download_queue.DownloadQueue(lib, num_workers=0)
    task = asyncio.create_task(process_attempt(queue, token, concurrency.PRIORITY_NORMAL))
    assert await asyncio.to_thread(started.wait, 5)
    failed_save = asyncio.Event()
    def fail_save(_store):
        failed_save.set()
        raise OSError("injected quarantine fsync failure")
    monkeypatch.setattr(ClaimStore, "save", fail_save)
    task.cancel()
    await asyncio.wait_for(failed_save.wait(), 5)
    # A failed diagnostic must not complete the worker before the controlled
    # executor drains. Shield keeps this observation from cancelling the worker.
    try:
        await asyncio.wait_for(asyncio.shield(task), 0.05)
    except (OSError, asyncio.TimeoutError):
        pass
    free_slots_before_drain = slot._value
    release.set()
    with pytest.raises(OSError, match="fsync failure"):
        await task
    assert free_slots_before_drain == 0
    assert slot._value == 1 and not lib.has_pdf(paper.key)


async def test_restored_claimless_index_cannot_bypass_campaign_ownership(vault, monkeypatch):
    from papervault.library import download
    lib, paper = vault
    before_claim = lib.index_path.read_bytes()
    store, token = await prepare(vault)
    await store.pre_dispatch(token)
    lib.index_path.write_bytes(before_claim)
    restored = Library(lib.root)
    candidate = restored.get(paper.key)
    assert candidate.reopen_claim is None
    candidate.download_status = "pending"  # ordinary audit reset after index restoration
    calls = []
    monkeypatch.setattr(download, "_STRATEGIES", [("fixture", lambda _: calls.append("sent"))])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    assert download.download_paper(candidate, restored) is False
    assert calls == [] and candidate.download_status == "pending"
    assert store.ledger.spent(token.campaign_id) == 1


async def test_staged_trusted_scan_preserves_deferred_identity_through_writeback(vault):
    from papervault.library.reopen_v2 import StagedAttempt
    lib, paper = vault
    claims, token = await prepare(vault)
    assert await claims.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    result = stage.library.get(paper.key)
    stage.path('pdf').write_bytes(b'%PDF trusted scanned artifact')
    result.download_status = 'ok'
    result.download_source = 'ads_scan'
    result.verification_deferred_to_ocr = True
    assert await stage.writeback(claims)
    stage.cleanup()
    persisted = Library(lib.root).get(paper.key)
    assert persisted.download_source == 'ads_scan'
    assert persisted.verification_deferred_to_ocr
    assert lib.pdf_path(paper.key).read_bytes() == b'%PDF trusted scanned artifact'


async def test_staged_verified_inspire_blank_fill_is_adopted(vault):
    from papervault.library.reopen_v2 import StagedAttempt

    lib, paper = vault
    claims, token = await prepare(vault)
    assert await claims.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    result = stage.library.get(paper.key)
    result.inspire_record_id = "123456"
    result.inspire_document_urls = ["https://inspirehep.net/files/verified.pdf"]
    result.download_status = "ok"
    stage.path("pdf").write_bytes(b"verified INSPIRE PDF")
    try:
        assert await stage.writeback(claims)
        persisted = Library(lib.root).get(paper.key)
        assert persisted.inspire_record_id == "123456"
        assert persisted.inspire_document_urls == ["https://inspirehep.net/files/verified.pdf"]
        assert lib.pdf_path(paper.key).read_bytes() == b"verified INSPIRE PDF"
    finally:
        stage.cleanup()


@pytest.mark.parametrize("change", ["adopted", "collision"])
async def test_staged_inspire_fill_never_replaces_or_collides_with_adopted_identity(vault, change):
    from papervault.library.reopen_v2 import StagedAttempt

    lib, paper = vault
    if change == "adopted":
        paper.inspire_record_id = "111111"
    claims, token = await prepare(vault)
    assert await claims.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    result = stage.library.get(paper.key)
    result.inspire_record_id = "222222"
    result.download_status = "ok"
    stage.path("pdf").write_bytes(b"refused INSPIRE PDF")
    if change == "collision":
        twin = Paper(key="Other2026", title="Another adopted INSPIRE publication", inspire_record_id="222222")
        lib._papers[twin.key] = twin
        lib._reindex(twin)
        lib.save(force=True, durable=True)
    try:
        assert not await stage.writeback(claims)
        assert paper.inspire_record_id == ("111111" if change == "adopted" else "")
        assert not lib.has_pdf(paper.key)
    finally:
        stage.cleanup()


async def test_staged_reopen_preserves_insight_invalidation_result(vault):
    from papervault.library.reopen_v2 import StagedAttempt

    lib, paper = vault
    claims, token = await prepare(vault)
    assert await claims.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    stage.library.get(paper.key).insight_invalid_reason = "reacquired extract invalidates old insight"
    try:
        assert await stage.writeback(claims)
        assert Library(lib.root).get(paper.key).insight_invalid_reason == "reacquired extract invalidates old insight"
    finally:
        stage.cleanup()
