"""Operator authority, scope and recovery contracts on isolated fixture vaults."""

import json

import pytest

from papervault.library.models import Paper
from papervault.library.store import Library
from papervault.library.state_v2 import Policy, Retrieval, TerminalState


def test_read_only_library_never_initializes_or_logs_a_vault(tmp_path):
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match="readable"):
        Library(root, read_only=True)
    assert not root.exists()
    root.mkdir()
    paper = Paper(key="A", title="retained")
    (root / "index.json").write_text(json.dumps({"papers": {"A": paper.model_dump()}}))
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    lib = Library(root, read_only=True)
    assert lib.get("A").title == "retained"
    with pytest.raises(PermissionError):
        lib.save()
    with pytest.raises(PermissionError):
        lib.write_topic("readonly", {})
    with pytest.raises(PermissionError):
        lib.enrich("A", {"venue": "changed"}, enriched_at="now")
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_pg_identity_reader_ignores_stale_export_and_creates_no_files(tmp_path, monkeypatch):
    from papervault.library.services.identity_backfill import read_identity_records
    from papervault.store.persistence import PostgresPersistence

    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")
    monkeypatch.setenv("PAPERVAULT_CATALOG_DSN", "unused")
    monkeypatch.setattr(PostgresPersistence, "load", lambda self: {"PG": Paper(key="PG")})
    root = tmp_path / "no-export"
    assert [p.key for p in read_identity_records(root)] == ["PG"]
    assert not root.exists()


def test_missing_budget_ledger_cannot_reset_catalog_claim_accounting(tmp_path, monkeypatch):
    from papervault.library.claims_v2 import BudgetLedger, LedgerError, ReopenClaim
    from papervault.store.persistence import PostgresPersistence

    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")
    monkeypatch.setenv("PAPERVAULT_CATALOG_DSN", "unused")
    paper = Paper(key="PG", reopen_claim=ReopenClaim(key="PG", generation=1,
        lease_id="lease", lease_expiry=1e20, campaign_id="campaign"))
    monkeypatch.setattr(PostgresPersistence, "load", lambda self: {"PG": paper})
    with pytest.raises(LedgerError, match="persisted claims"):
        BudgetLedger(tmp_path)._read()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("flag,status", [("--retry-failed", "failed"),
                                         ("--retry-metadata-only", "metadata_only")])
def test_download_retry_reopens_retrieval_with_operator_evidence(tmp_path, monkeypatch, capsys, flag, status):
    from papervault.library import cli

    monkeypatch.setenv("PAPERVAULT_VAULT", str(tmp_path))
    lib = Library(tmp_path)
    lib._papers["A"] = Paper(key="A", download_status=status,
        terminal_state=TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.UNKNOWN))
    lib.save()
    args = ["audit", flag, "--json", "--reason", "verified retry", "--actor", "operator"]
    if flag == "--retry-metadata-only":
        args += ["--all"]
    assert cli.main(args) == 0
    capsys.readouterr()
    fresh = Library(tmp_path)
    state = fresh.get("A").acquisition_state
    assert state.retrieval == Retrieval.ELIGIBLE and state.policy == Policy.UNKNOWN
    rows = fresh.corrections.read()["operator_events"]
    assert rows[-1]["event"] == "operator_reopen"
    assert rows[-1]["reason"] == "verified retry" and rows[-1]["actor"] == "operator"
    assert rows[-1]["keys"] == ["A"]


def test_extraction_retry_clears_defer_without_inventing_a_quality_axis(tmp_path, monkeypatch, capsys):
    from papervault.library import cli

    monkeypatch.setenv("PAPERVAULT_VAULT", str(tmp_path))
    lib = Library(tmp_path)
    lib._papers["A"] = Paper(key="A", download_status="extract_failed", extract_attempts=9,
        terminal_state=TerminalState(retrieval=Retrieval.ACQUIRED, policy=Policy.CLEAR),
        extract_deferred_sig="100:1", extract_deferred_epoch=7)
    lib.pdf_path("A").write_bytes(b"%PDF retained")
    lib.save()
    assert cli.main(["audit", "--retry-low-quality", "--json"]) == 0
    capsys.readouterr()
    fresh = Library(tmp_path)
    paper = fresh.get("A")
    assert paper.download_status == "ok" and paper.extract_attempts == 0
    assert paper.extract_deferred_sig is None and paper.extract_deferred_epoch == 0
    assert paper.acquisition_state.retrieval == Retrieval.ACQUIRED
    assert fresh.corrections.read()["operator_events"][-1]["keys"] == ["A"]


def test_policy_ruling_can_enter_a_scoped_campaign(tmp_path, monkeypatch):
    from papervault.library.reopen_v2 import inspect_campaign
    from papervault.library.services.policy_ruling import apply_ruling

    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    lib = Library(tmp_path)
    lib._papers["A"] = Paper(key="A", doi="10.1234/example", download_status="failed")
    lib.save()
    apply_ruling(tmp_path, keys=["A"], action="acquire", reason="owner authorizes retry")
    report = inspect_campaign(tmp_path, cap=1, keys=["A"])
    assert report["keys"] == ["A"] and report["selected"] == 1


def test_merge_cli_retains_alias_and_records_operator_reason(tmp_path, monkeypatch, capsys):
    from papervault import ops_guards
    from papervault.library import cli

    monkeypatch.setenv("PAPERVAULT_VAULT", str(tmp_path))
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    monkeypatch.setattr(ops_guards, "active_service_units", lambda *a, **kw: [])
    lib = Library(tmp_path)
    lib._papers["A"] = Paper(key="A", doi="10.1234/canonical", title="canonical")
    lib._papers["B"] = Paper(key="B", title="observation")
    lib.save()
    assert cli.main(["merge-consolidate", "A", "B", "--reason", "confirmed same work",
                     "--actor", "owner"]) == 0
    assert json.loads(capsys.readouterr().out)["changed"]
    fresh = Library(tmp_path)
    assert fresh.get("B").acquisition_state.alias_key == "A"
    event = fresh.corrections.read()["operator_events"][-1]
    assert event["event"] == "identity_merge" and event["actor"] == "owner"
    assert "confirmed same work" in event["reason"]


def test_recovery_cli_refuses_active_service_before_loading_store(tmp_path, monkeypatch, capsys):
    from papervault import ops_guards
    from papervault.library import cli

    monkeypatch.setenv("PAPERVAULT_VAULT", str(tmp_path / "absent"))
    monkeypatch.setattr(ops_guards, "active_service_units", lambda *a, **kw: ["papervault.service"])
    assert cli.main(["claim-recover", "A", "--reason", "investigated", "--actor", "owner"]) == 2
    assert "running" in capsys.readouterr().err
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("command", ["claim-recover", "release-quarantine", "campaign-halt", "merge-consolidate", "merge-restart"])
def test_standalone_operators_share_campaign_writer_exclusion(tmp_path, monkeypatch, capsys, command):
    from filelock import FileLock
    from papervault import ops_guards
    from papervault.library import cli

    monkeypatch.setenv("PAPERVAULT_VAULT", str(tmp_path))
    monkeypatch.setattr(ops_guards, "active_service_units", lambda *a, **kw: [])
    args = [command, "A", *( ["B"] if command == "merge-consolidate" else []),
            "--reason", "investigated", "--actor", "owner"]
    with FileLock(str(tmp_path / ".reopen-campaign.lock")):
        assert cli.main(args) == 2
    assert "lock" in capsys.readouterr().err.lower()
    assert not (tmp_path / "index.json").exists()


async def test_retry_preserves_temporary_merge_fence_without_making_it_permanent(tmp_path, monkeypatch):
    from papervault.library.merge_v2 import MergeStore
    from papervault.library.services.operator_actions import reopen_records

    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    lib = Library(tmp_path)
    for key in ("A", "B"):
        lib._papers[key] = Paper(key=key, download_status="failed",
            terminal_state=TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR))
    lib.save()
    merge = MergeStore(lib)
    await merge.quiesce("A", "B")
    reopen_records(lib, [lib.get("A")], reason="reviewed retry")
    assert lib.get("A").acquisition_state.quarantine
    assert not lib.get("A").terminal_state.quarantine
    assert await merge.merge_records("A", "B") is lib.get("A")
    assert not lib.get("A").acquisition_state.quarantine
