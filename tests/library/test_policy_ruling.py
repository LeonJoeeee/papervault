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


def test_apply_records_independent_key_revisions_and_scoped_grants(ruling_vault, capsys):
    root, keys = ruling_vault
    lib = Library(root)
    paper = lib.get("A")
    paper.terminal_state = replace(paper.acquisition_state, required_revision=4)
    lib.save()
    unrelated = lib.get("Unlisted").model_dump()
    assert cli.main(command(keys, "--apply")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["applied"] == 2 and report["rulings_added"] == 2
    assert report["revisions"] == {"B": 1, "A": 5}
    lib = Library(root)
    rows = lib.corrections.read()["policy_rulings"]
    assert len(rows) == 2
    assert [(row["keys"], row["revision"]) for row in rows] == [(["B"], 1), (["A"], 5)]
    assert all(row["action"] == "acquire" and row["reason"] == "Owner approved ADS recovery #245"
               and datetime.fromisoformat(row["timestamp"]).tzinfo is not None for row in rows)
    for key in ("A", "B"):
        paper = lib.get(key)
        state = paper.acquisition_state
        assert state.policy == state_v2.Policy.OVERRIDE
        assert state.retrieval == state_v2.Retrieval.ELIGIBLE and paper.download_status == "metadata_only"
        assert state.override == state_v2.PolicyOverride(scope=("acquire",), revision={"A": 5, "B": 1}[key], key=key)
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
    assert later[:-1] == rows and len(later) == 3
    assert later[-1]["keys"] == ["Unlisted"]


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


@pytest.mark.parametrize("mutation", ["alias", "quarantine"])
def test_protected_state_is_reported_and_never_granted(ruling_vault, capsys, mutation):
    root, keys = ruling_vault
    lib = Library(root)
    paper = lib.get("A")
    change = {"alias": {"merge": state_v2.Merge.ALIAS, "alias_key": "Unlisted"},
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


@pytest.mark.parametrize("required_revision", [0, 4])
def test_interrupted_index_save_reuses_durable_ruling_on_retry(ruling_vault, monkeypatch, capsys, required_revision):
    root, keys = ruling_vault
    lib = Library(root)
    lib.get("A").terminal_state = replace(lib.get("A").acquisition_state, required_revision=required_revision)
    lib.save()
    original = Library.save
    with monkeypatch.context() as patch:
        def fail(*a, **kw):
            raise OSError("injected index save interruption")
        patch.setattr(Library, "save", fail)
        assert cli.main(command(keys, "--apply")) == 2
    assert "interruption" in capsys.readouterr().err
    assert Library(root).get("A").acquisition_state.policy == state_v2.Policy.UNKNOWN
    rows = Library(root).corrections.read()["policy_rulings"]
    assert [(row["keys"], row["revision"]) for row in rows] == (
        [(["B", "A"], 1)] if required_revision == 0 else [(["B"], 1), (["A"], 5)])
    assert Library.save is original
    assert cli.main(command(keys, "--apply")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["applied"] == 2 and report["rulings_added"] == 0
    assert Library(root).corrections.read()["policy_rulings"] == rows
    assert state_v2.effective_policy(Library(root).get("A").acquisition_state, key="A")


@pytest.mark.parametrize("action,policy", [("acquire", "override"), ("override", "override"),
    ("exclude", "excluded"), ("restrict", "excluded"), ("clear", "clear")])
def test_rulings_cover_any_cohort_and_are_revisioned(ruling_vault, capsys, action, policy):
    root, keys = ruling_vault
    lib = Library(root)
    paper = lib.get("A")
    paper.source = "inspire"
    paper.doi = "10.1234/arbitrary"
    paper.domain_status = "off_domain"
    paper.terminal_state = replace(paper.acquisition_state, policy_reasons=("owner_exclusion",))
    lib.save()
    args = command(keys, "--apply")
    args[args.index("--action") + 1] = action
    assert cli.main(args) == 0
    capsys.readouterr()
    lib = Library(root)
    paper = lib.get("A")
    state = paper.acquisition_state
    assert paper.terminal_state.policy.value == policy
    if action == "clear":
        # Clearing a ruling does not silently remove a separate domain exclusion.
        assert not state_v2.effective_policy(state, key="A")
    if policy == "override":
        assert state_v2.effective_policy(state, key="A")
        assert not state_v2.effective_policy(state, key="B")
    if policy == "excluded":
        assert lib.get("A").download_status == "metadata_only"
    old = lib.corrections.read()["policy_rulings"]
    assert cli.main(args) == 0
    capsys.readouterr()
    assert Library(root).corrections.read()["policy_rulings"] == old
    args[args.index("--action") + 1] = "clear" if policy != "clear" else "exclude"
    assert cli.main(args) == 0
    capsys.readouterr()
    later = Library(root).corrections.read()["policy_rulings"]
    assert later[:-1] == old and later[-1]["revision"] > old[-1]["revision"]


def test_replaying_acquire_does_not_reopen_a_later_failed_attempt(ruling_vault, capsys):
    root, keys = ruling_vault
    assert cli.main(command(keys, "--apply")) == 0
    capsys.readouterr()
    lib = Library(root)
    lib.get("A").download_status = "failed"
    lib.save()
    before = tree_bytes(root)
    assert cli.main(command(keys, "--apply")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["applied"] == report["rulings_added"] == 0
    assert tree_bytes(root) == before
    assert Library(root).get("A").acquisition_state.retrieval == state_v2.Retrieval.EXHAUSTED


@pytest.mark.parametrize("action", ["acquire", "clear"])
def test_replay_cannot_authorize_a_later_policy_restriction(ruling_vault, capsys, action):
    root, keys = ruling_vault
    args = command(keys, "--apply")
    args[args.index("--action") + 1] = action
    assert cli.main(args) == 0
    capsys.readouterr()
    lib = Library(root)
    current = lib.get("A").acquisition_state
    lib.get("A").terminal_state = replace(current, policy_reasons=("later exclusion",))
    lib.save()
    before = tree_bytes(root)
    assert not state_v2.effective_policy(lib.get("A").acquisition_state, key="A")
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["applied"] == report["rulings_added"] == 0
    assert not state_v2.effective_policy(Library(root).get("A").acquisition_state, key="A")
    assert tree_bytes(root) == before


@pytest.mark.parametrize("action", ["acquire", "clear"])
@pytest.mark.parametrize("restriction", ["observation", "ruling"])
def test_replay_preserves_scope_when_required_revision_advances(ruling_vault, capsys, action, restriction):
    root, keys = ruling_vault
    keys.write_text("A\n")
    args = command(keys, "--apply")
    args[args.index("--action") + 1] = action
    assert cli.main(args) == 0
    capsys.readouterr()
    lib = Library(root)
    assert lib.get("A").acquisition_state.required_revision == 1
    if restriction == "observation":
        lib.get("A").terminal_state = replace(lib.get("A").acquisition_state,
            policy_reasons=("later exclusion",), required_revision=2)
        lib.save()
    else:
        exclusion = command(keys, "--apply")
        exclusion[exclusion.index("--action") + 1] = "exclude"
        exclusion[exclusion.index("--reason") + 1] = "later exclusion"
        assert cli.main(exclusion) == 0
        capsys.readouterr()
    lib = Library(root)
    assert lib.get("A").acquisition_state.required_revision == 2
    assert not state_v2.effective_policy(lib.get("A").acquisition_state, key="A")
    before = tree_bytes(root)

    assert cli.main(args) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["applied"] == replay["rulings_added"] == 0
    assert tree_bytes(root) == before
    assert not state_v2.effective_policy(Library(root).get("A").acquisition_state, key="A")

    args[args.index("--reason") + 1] = "Owner approved later exclusion"
    assert cli.main(args) == 0
    renewed = json.loads(capsys.readouterr().out)
    current = Library(root).get("A").acquisition_state
    assert renewed["applied"] == renewed["rulings_added"] == 1
    assert current.required_revision == 3
    assert state_v2.effective_policy(current, key="A")


@pytest.mark.parametrize("action", ["acquire", "clear"])
def test_higher_revision_in_batch_does_not_renew_a_settled_key(ruling_vault, capsys, action):
    root, keys = ruling_vault
    keys.write_text("A\n")
    args = command(keys, "--apply")
    args[args.index("--action") + 1] = action
    assert cli.main(args) == 0
    capsys.readouterr()
    lib = Library(root)
    lib.get("A").download_status = "failed"
    lib.get("B").terminal_state = replace(lib.get("B").acquisition_state, required_revision=4)
    lib.save()
    settled = lib.get("A").model_dump()
    rows = lib.corrections.read()["policy_rulings"]
    keys.write_text("A\nB\n")

    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    fresh = Library(root)
    assert fresh.get("A").model_dump() == settled
    assert fresh.get("A").acquisition_state.retrieval == state_v2.Retrieval.EXHAUSTED
    assert fresh.get("B").acquisition_state.required_revision == 5
    assert report["revisions"] == {"A": 1, "B": 5}
    assert report["applied"] == report["rulings_added"] == 1
    later = fresh.corrections.read()["policy_rulings"]
    assert later[:-1] == rows
    assert later[-1]["keys"] == ["B"] and later[-1]["revision"] == 5


def test_interrupted_clear_replay_refuses_restrictions_outside_original_scope(ruling_vault, monkeypatch, capsys):
    root, keys = ruling_vault
    args = command(keys, "--apply")
    args[args.index("--action") + 1] = "clear"
    with monkeypatch.context() as patch:
        def fail(*a, **kw):
            raise OSError("interrupted clear")
        patch.setattr(Library, "save", fail)
        assert cli.main(args) == 2
    capsys.readouterr()
    lib = Library(root)
    lib.get("A").terminal_state = replace(lib.get("A").acquisition_state,
                                          policy_reasons=("later exclusion",))
    lib.save()
    before = tree_bytes(root)
    assert cli.main(args) == 2
    assert "restrictions changed" in capsys.readouterr().err
    assert tree_bytes(root) == before
