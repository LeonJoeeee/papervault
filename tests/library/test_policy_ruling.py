"""Recorded, scoped policy grants through the maintenance CLI; no live writes."""

import json
from dataclasses import replace
from datetime import datetime

import pytest

from papervault import ops_guards
from papervault.library import Library, cli, state_v2
from papervault.library.models import Paper


def tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def ruling_vault(tmp_path, monkeypatch):
    root = tmp_path / "vault"
    root.mkdir()
    papers = {}
    for key in ("A", "B", "Unlisted"):
        paper = Paper(key=key, title=f"ADS fixture {key}", source="ads",
                      paper_id="2020ApJ...900....1A",
                      url="https://ui.adsabs.harvard.edu/abs/2020ApJ...900....1A/abstract",
                      download_status="metadata_only")
        papers[key] = paper.model_dump()
    (root / "index.json").write_text(json.dumps({"version": 1, "papers": papers}))
    keys = tmp_path / "keys.txt"
    keys.write_text("# explicit order\nB\nA # approved\nB\n")
    monkeypatch.setenv("PAPERVAULT_VAULT", str(root))
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    monkeypatch.setattr(ops_guards, "active_service_units", lambda *a, **kw: [])
    return root, keys


def command(keys, *extra):
    return ["policy-ruling", "--keys", str(keys), "--action", "acquire",
            "--reason", "Owner approved ADS recovery #245", *extra]


def test_dry_run_reports_current_policies_without_creating_files(ruling_vault, capsys):
    root, keys = ruling_vault
    before = tree_bytes(root)
    assert cli.main(command(keys)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True and report["matched_keys"] == ["B", "A"]
    assert report["missing_count"] == report["mismatch_count"] == 0
    assert {r["key"]: r["policy"] for r in report["records"]} == {"B": "unknown", "A": "unknown"}
    assert tree_bytes(root) == before


def test_apply_appends_one_ruling_and_scoped_revision_aware_grants(ruling_vault, capsys):
    root, keys = ruling_vault
    lib = Library(root)
    paper = lib.get("A")
    paper.terminal_state = replace(paper.acquisition_state, required_revision=4)
    lib.save()
    unrelated = lib.get("Unlisted").model_dump()
    assert cli.main(command(keys, "--apply")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["applied"] == 2 and report["rulings_added"] == 1
    lib = Library(root)
    rows = lib.corrections.read()["policy_rulings"]
    assert len(rows) == 1
    assert rows[0]["keys"] == ["B", "A"] and rows[0]["action"] == "acquire"
    assert rows[0]["reason"] == "Owner approved ADS recovery #245" and rows[0]["revision"] == 4
    assert datetime.fromisoformat(rows[0]["timestamp"]).tzinfo is not None
    for key in ("A", "B"):
        paper = lib.get(key)
        state = paper.acquisition_state
        assert state.policy == state_v2.Policy.OVERRIDE
        assert state.retrieval == state_v2.Retrieval.EXHAUSTED and paper.download_status == "metadata_only"
        assert state.override == state_v2.PolicyOverride(scope=("acquire",), revision=4, key=key)
        assert state_v2.effective_policy(state, key=key)
        assert not state_v2.effective_policy(state, key="Other")
        assert not state_v2.effective_policy(state, key=key, action="serve")
    assert lib.get("Unlisted").model_dump() == unrelated
    before = tree_bytes(root)
    assert cli.main(command(keys, "--apply")) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["applied"] == replay["rulings_added"] == 0
    assert tree_bytes(root) == before
    keys.write_text("A\nUnlisted\n")
    assert cli.main(command(keys, "--apply")) == 0
    capsys.readouterr()
    later = Library(root).corrections.read()["policy_rulings"]
    assert later[0] == rows[0] and len(later) == 2
    assert later[1]["keys"] == ["Unlisted"]


@pytest.mark.parametrize("apply", [False, True])
def test_unknown_key_refuses_the_whole_request_with_accounting(ruling_vault, capsys, apply):
    root, keys = ruling_vault
    keys.write_text("A\nMissing\n")
    before = tree_bytes(root)
    assert cli.main(command(keys, *(["--apply"] if apply else []))) == 2
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["matched_keys"] == ["A"]
    assert report["missing_keys"] == ["Missing"] and report["missing_count"] == 1
    assert "Missing" in output.err
    assert tree_bytes(root) == before


@pytest.mark.parametrize("mutation", ["source", "doi", "arxiv", "identity", "landing", "malformed_url", "status",
                                      "excluded", "reason", "domain", "alias", "quarantine"])
def test_outside_cohort_or_protected_state_is_reported_and_never_granted(ruling_vault, capsys, mutation):
    root, keys = ruling_vault
    lib = Library(root)
    paper = lib.get("A")
    if mutation == "source":
        paper.source = "other"
    elif mutation in {"doi", "arxiv"}:
        setattr(paper, "doi" if mutation == "doi" else "arxiv_id", "existing")
    elif mutation == "identity":
        paper.paper_id = "2021ApJ...900....1A"
    elif mutation == "landing":
        paper.url = "https://example.org/2020ApJ...900....1A/abstract"
    elif mutation == "malformed_url":
        paper.url = "https://["
    elif mutation == "status":
        paper.download_status = "ok"
    elif mutation == "domain":
        paper.domain_status = "off_domain"
    else:
        change = {"excluded": {"policy": state_v2.Policy.EXCLUDED},
                  "reason": {"policy_reasons": ("owner_exclusion",)},
                  "alias": {"merge": state_v2.Merge.ALIAS, "alias_key": "Unlisted"},
                  "quarantine": {"quarantine": True}}[mutation]
        paper.terminal_state = replace(paper.acquisition_state, **change)
    lib.save()
    before = tree_bytes(root)
    assert cli.main(command(keys, "--apply")) == 2
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["mismatch_count"] == 1 and report["mismatches"][0]["key"] == "A"
    assert report["mismatches"][0]["reason"] and "A" in output.err
    assert tree_bytes(root) == before


@pytest.mark.parametrize("invalid", ["empty_keys", "blank_reason"])
def test_empty_request_refuses_before_writes(ruling_vault, capsys, invalid):
    root, keys = ruling_vault
    args = command(keys, "--apply")
    if invalid == "empty_keys":
        keys.write_text("# nobody\n")
    else:
        args[args.index("--reason") + 1] = "   "
    before = tree_bytes(root)
    assert cli.main(args) == 2
    assert capsys.readouterr().err
    assert tree_bytes(root) == before


def test_service_guard_precedes_writable_store(ruling_vault, monkeypatch, capsys):
    root, keys = ruling_vault
    monkeypatch.setattr(ops_guards, "active_service_units", lambda *a, **kw: ["papervault.service"])
    before = tree_bytes(root)
    assert cli.main(command(keys, "--apply")) == 2
    assert "running" in capsys.readouterr().err
    assert tree_bytes(root) == before
    assert cli.main(command(keys)) == 0  # read-only inspection remains available


def test_interrupted_index_save_reuses_durable_ruling_on_retry(ruling_vault, monkeypatch, capsys):
    root, keys = ruling_vault
    original = Library.save
    with monkeypatch.context() as patch:
        def fail(*a, **kw):
            raise OSError("injected index save interruption")
        patch.setattr(Library, "save", fail)
        assert cli.main(command(keys, "--apply")) == 2
    assert "interruption" in capsys.readouterr().err
    assert Library(root).get("A").acquisition_state.policy == state_v2.Policy.UNKNOWN
    rows = Library(root).corrections.read()["policy_rulings"]
    assert len(rows) == 1
    assert Library.save is original
    assert cli.main(command(keys, "--apply")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["applied"] == 2 and report["rulings_added"] == 0
    assert Library(root).corrections.read()["policy_rulings"] == rows
    assert state_v2.effective_policy(Library(root).get("A").acquisition_state, key="A")
