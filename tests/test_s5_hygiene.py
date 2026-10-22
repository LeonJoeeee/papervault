"""S5 protects source trees, retained work, credentials, and disposable PDF scratch."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_sweeper():
    spec = importlib.util.spec_from_file_location("s5_sweeper", REPO / "scripts/sweep_worktrees.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_paths_default_and_override(monkeypatch, tmp_path):
    from papervault.eval._paths import eval_path

    monkeypatch.delenv("PAPERVAULT_EVAL_DIR", raising=False)
    assert eval_path("fixture.jsonl") == Path("/data/paper-vault/eval-results/fixture.jsonl")
    monkeypatch.setenv("PAPERVAULT_EVAL_DIR", str(tmp_path / "runs"))
    assert eval_path("judge", "fixture") == tmp_path / "runs/judge/fixture"
    assert not (tmp_path / "runs").exists()  # resolving alone never creates artifacts
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAPERVAULT_EVAL_DIR", "runs")
    assert eval_path("fixture.jsonl") == tmp_path / "runs/fixture.jsonl"


@pytest.mark.parametrize("relative", ["src", "src/papervault/eval", "src/other/results"])
def test_eval_refuses_source_tree(monkeypatch, tmp_path, relative):
    from papervault.eval._paths import eval_path

    monkeypatch.setenv("PAPERVAULT_EVAL_DIR", str(tmp_path / relative))
    with pytest.raises(ValueError, match="src"):
        eval_path()


def test_eval_refuses_symlink_and_tag_escape(monkeypatch, tmp_path):
    from papervault.eval._paths import eval_path

    source = tmp_path / "src/papervault/eval"
    source.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    monkeypatch.setenv("PAPERVAULT_EVAL_DIR", str(alias))
    with pytest.raises(ValueError, match="src"):
        eval_path()
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "judge").symlink_to(source, target_is_directory=True)
    monkeypatch.setenv("PAPERVAULT_EVAL_DIR", str(runs))
    with pytest.raises(ValueError):
        eval_path("judge", "fixture")
    with pytest.raises(ValueError):
        eval_path("../src/fixture.jsonl")


@pytest.mark.parametrize("writer", ["gold", "precision", "pairwise"])
def test_prompt_writers_refuse_final_file_symlink_into_source(monkeypatch, tmp_path, writer):
    from papervault.eval import build_judge_prompts as judge
    from papervault.eval import build_reffree_prompts as reffree

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setenv("PAPERVAULT_EVAL_DIR", str(root))
    record = {"qid": "q", "data": {"chunks": [{"paper_key": "P", "content": "fixture"}]}}
    for name in ("fixture.jsonl", "baseline.jsonl"):
        (root / name).write_text(json.dumps(record) + "\n")
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({"qid": "q"}) + "\n")
    protected = tmp_path / "src/protected.txt"
    protected.parent.mkdir()
    protected.write_text("retained source")
    tag = "fixture__vs__baseline" if writer == "pairwise" else "fixture"
    outdir = root / "judge" / tag
    outdir.mkdir(parents=True)
    stem = f"q.{reffree.safe_stem(reffree.chunk_id('P', 0))}" if writer == "precision" else "q"
    (outdir / f"{stem}.prompt.txt").symlink_to(protected)
    monkeypatch.setattr("sys.argv", ["build_judge_prompts", "fixture", str(gold)])
    with pytest.raises(ValueError, match="src"):
        if writer == "gold":
            judge.main()
        elif writer == "precision":
            reffree.build_precision("fixture")
        else:
            reffree.build_pairwise("baseline", "fixture")
    assert protected.read_text() == "retained source"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "s5@example.invalid")
    git(repo, "config", "user.name", "S5 test")
    (repo / ".gitignore").write_text(".agents/\n.claude/\n*.ignored\n")
    (repo / "tracked").write_text("retained\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def add_worktree(repo, name, parent=".agents/worktrees"):
    path = repo / parent / name
    git(repo, "worktree", "add", "-b", name, str(path), "main")
    return path


def test_sweeper_dry_run_and_execute_preserve_all_branch_refs(git_repo, capsys):
    sweep = load_sweeper()
    a = add_worktree(git_repo, "merged-agent")
    b = add_worktree(git_repo, "merged-claude", ".claude/worktrees")
    outside = add_worktree(git_repo, "outside", "elsewhere")
    git(git_repo, "update-ref", "refs/remotes/origin/merged-agent", "HEAD")
    before = git(git_repo, "for-each-ref", "--format=%(refname) %(objectname)")
    assert sweep.main(["--repo", str(git_repo)]) == 0
    assert "WOULD REMOVE" in capsys.readouterr().out
    assert a.exists() and b.exists()
    assert sweep.main(["--repo", str(git_repo), "--execute"]) == 0
    assert not a.exists() and not b.exists()
    assert outside.exists() and (git_repo / "tracked").exists()
    assert git(git_repo, "for-each-ref", "--format=%(refname) %(objectname)") == before


@pytest.mark.parametrize("dirty", ["tracked", "untracked", "cache.ignored"])
def test_sweeper_keeps_tracked_untracked_and_ignored_work(git_repo, dirty):
    sweep = load_sweeper()
    path = add_worktree(git_repo, "dirty")
    (path / dirty).write_text("must survive")
    assert sweep.main(["--repo", str(git_repo), "--execute"]) == 0
    assert (path / dirty).read_text() == "must survive"


def test_sweeper_keeps_unmerged_locked_current_detached_and_unregistered(git_repo, monkeypatch):
    sweep = load_sweeper()
    unmerged = add_worktree(git_repo, "unmerged")
    (unmerged / "tracked").write_text("new commit\n")
    git(unmerged, "commit", "-am", "unmerged")
    locked = add_worktree(git_repo, "locked")
    git(git_repo, "worktree", "lock", str(locked))
    current = add_worktree(git_repo, "current")
    detached = git_repo / ".agents/worktrees/detached"
    git(git_repo, "worktree", "add", "--detach", str(detached), "main")
    unknown = git_repo / ".agents/worktrees/unknown"
    unknown.mkdir()
    (unknown / "important").write_text("retain")
    monkeypatch.chdir(current)
    assert sweep.main(["--repo", str(git_repo), "--execute"]) == 0
    assert all(p.exists() for p in (unmerged, locked, current, detached, unknown))


def test_sweeper_accepts_only_merged_pr_for_exact_head(git_repo, monkeypatch):
    sweep = load_sweeper()
    path = add_worktree(git_repo, "squashed")
    (path / "tracked").write_text("squashed change\n")
    git(path, "commit", "-am", "squashed")
    git(git_repo, "remote", "add", "origin", "https://github.com/example/repo.git")
    run = subprocess.run
    pr = {"state": "MERGED", "headRefOid": git(path, "rev-parse", "HEAD"),
          "baseRefName": "main", "headRepository": {"id": "R_test", "name": "repo", "nameWithOwner": ""},
          "headRepositoryOwner": {"login": "example"}}

    def fake_run(args, **kwargs):
        if args[0] == "gh":
            return subprocess.CompletedProcess(args, 0, json.dumps([pr]), "")
        return run(args, **kwargs)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    original_head = pr["headRefOid"]
    pr["headRefOid"] = "0" * 40
    assert sweep.main(["--repo", str(git_repo), "--execute"]) == 0
    assert path.exists()
    pr["headRefOid"] = original_head
    pr["headRepositoryOwner"]["login"] = "fork-owner"
    assert sweep.main(["--repo", str(git_repo), "--execute"]) == 0
    assert path.exists()
    pr["headRepositoryOwner"]["login"] = "example"
    assert sweep.main(["--repo", str(git_repo), "--execute"]) == 0
    assert not path.exists()
    assert git(git_repo, "rev-parse", "squashed") == original_head


def test_doctor_warns_about_repo_env_backups_without_reading_them(monkeypatch, tmp_path, capsys):
    from papervault import cli

    # Resolve a synthetic source checkout; working directory is intentionally elsewhere.
    source = tmp_path / "repo/src/papervault/cli.py"
    source.parent.mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(source))
    monkeypatch.chdir(tmp_path)
    backup = tmp_path / "repo/.env.bak-cutover"
    backup.write_text("DO_NOT_PRINT_SECRET")
    backup.chmod(0)
    report = cli._Report()
    cli._check_config(report)
    output = capsys.readouterr().out
    assert "[WARN]" in output and ".env.bak" in output
    assert "/home/leon/backups/papervault-env/" in output
    assert "DO_NOT_PRINT_SECRET" not in output
    assert report.warnings > 0


@pytest.mark.parametrize("outcome", ["browser-error", "no-link", "save-error", "invalid", "success"])
def test_mdpi_always_removes_temporary_pdf(monkeypatch, tmp_path, outcome):
    from papervault.library.download_sources import mdpi
    from papervault.library.models import Paper

    created = []
    original = tempfile.NamedTemporaryFile

    def temporary(*args, **kwargs):
        kwargs["dir"] = tmp_path
        handle = original(*args, **kwargs)
        created.append(Path(handle.name))
        return handle

    class Download:
        def save_as(self, filename):
            Path(filename).write_bytes(b"%PDF-1.7\n" if outcome == "success" else b"not a PDF")
            if outcome == "save-error":
                raise RuntimeError("browser failed during save")

    class DownloadInfo:
        value = Download()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Page:
        def content(self):
            return "" if outcome == "no-link" else '<meta name="citation_pdf_url" content="https://mdpi.test/x.pdf">'

        def expect_download(self, **kwargs):
            return DownloadInfo()

        def evaluate(self, script):
            pass

    def fetch(fetcher, url, **kwargs):
        if outcome == "browser-error":
            raise RuntimeError("browser unavailable")
        kwargs["page_action"](Page())

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", temporary)
    monkeypatch.setattr(mdpi, "_load_stealthy_fetcher", lambda tier: object())
    monkeypatch.setattr(mdpi, "browser_fetch", fetch)
    data = mdpi._try_mdpi_scrapling(Paper(key="MDPI", title="fixture", doi="10.3390/fixture"))
    assert data == (b"%PDF-1.7\n" if outcome == "success" else None)
    assert created and all(not p.exists() for p in created)
