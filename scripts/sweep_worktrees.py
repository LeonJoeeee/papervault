#!/usr/bin/env python3
"""Reclaim clean, integrated agent worktrees; retain every Git branch ref.

Dry-run by default. --execute removes eligible registered worktrees without force.
Run only when the candidate workers have stopped; the caller owns their lifecycle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                            capture_output=True, text=True, timeout=30)
    return result.stdout


def _worktrees(repo: Path) -> list[dict[str, str]]:
    records = []
    current = {}
    for field in _git(repo, "worktree", "list", "--porcelain", "-z").split("\0"):
        if not field:
            if current:
                records.append(current)
                current = {}
        else:
            key, _, value = field.partition(" ")
            current[key] = value
    return records


def _repo_slug(url: str) -> str:
    return url.strip().removesuffix("/").removesuffix(".git").removeprefix(
        "https://github.com/").removeprefix("git@github.com:").removeprefix(
        "ssh://git@github.com/")


def _merged_pr(repo: Path, branch: str, head: str) -> bool:
    """A reused branch name is insufficient: the merged PR must contain this exact head."""
    try:
        remote = _git(repo, "remote", "get-url", "origin").strip()
        slug = _repo_slug(remote)
        result = subprocess.run(
            ["gh", "pr", "list", "--repo", remote, "--head", branch, "--base", "main",
             "--state", "merged", "--limit", "100", "--json",
             "state,headRefOid,baseRefName,headRepository,headRepositoryOwner"],
            check=True, capture_output=True, text=True, timeout=30)
        return any(pr.get("state") == "MERGED" and pr.get("headRefOid") == head
                   and pr.get("baseRefName") == "main"
                   and f"{(pr.get('headRepositoryOwner') or {}).get('login', '')}/"
                       f"{(pr.get('headRepository') or {}).get('name', '')}" == slug
                   for pr in json.loads(result.stdout))
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError):
        return False  # Unavailable or malformed evidence never authorizes removal.


def _eligible(repo: Path, worktree: dict[str, str], active: Path) -> tuple[bool, str]:
    path = Path(worktree["worktree"])
    if "locked" in worktree:
        return False, "locked"
    if active.is_relative_to(path):
        return False, "current worktree"
    branch_ref = worktree.get("branch", "")
    if not branch_ref.startswith("refs/heads/"):
        return False, "detached or missing branch"
    if _git(path, "rev-parse", "--show-toplevel").strip() != str(path):
        return False, "worktree identity changed"
    if (_git(path, "symbolic-ref", "HEAD").strip() != branch_ref
            or _git(path, "rev-parse", "HEAD").strip() != worktree["HEAD"]):
        return False, "branch or head changed"
    if _git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all",
            "--ignored=matching", "--ignore-submodules=none"):
        return False, "tracked, untracked, or ignored files present"
    merged = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", worktree["HEAD"], "origin/main"],
        capture_output=True, text=True, timeout=30)
    if merged.returncode == 0:
        return True, "merged into origin/main"
    if merged.returncode != 1:
        return False, "cannot verify origin/main ancestry"
    if _merged_pr(repo, branch_ref.removeprefix("refs/heads/"), worktree["HEAD"]):
        return True, "MERGED PR for exact branch head"
    return False, "no integration evidence for branch head"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="any checkout of the repository")
    parser.add_argument("--execute", action="store_true", help="remove eligible worktrees (never branches)")
    args = parser.parse_args(argv)
    try:
        records = _worktrees(args.repo.resolve())
        repo = Path(records[0]["worktree"]).resolve()  # Git lists the primary checkout first.
        _git(repo, "rev-parse", "--verify", "origin/main^{commit}")
        active = Path.cwd().resolve()
        registered = {Path(w["worktree"]): w for w in records}
        candidates = set()
        for prefix in (".claude", ".agents"):
            root = repo / prefix / "worktrees"
            if root.is_dir() and root.resolve() == root:
                candidates.update(root.iterdir())
        for path in sorted(candidates):
            worktree = registered.get(path)
            if path.is_symlink() or path.resolve() != path or not path.is_dir():
                print(f"KEEP {path}: not a canonical directory")
                continue
            if worktree is None:
                print(f"KEEP {path}: not a registered worktree")
                continue
            if any(other != path and other.is_relative_to(path) for other in registered):
                print(f"KEEP {path}: contains another worktree")
                continue
            eligible, reason = _eligible(repo, worktree, active)
            if not eligible:
                print(f"KEEP {path}: {reason}")
                continue
            evidence = f"{worktree['branch']} {worktree['HEAD']}: {reason}"
            if not args.execute:
                print(f"WOULD REMOVE {path}: {evidence}")
                continue
            # Re-read registration and dirt immediately before each non-forced removal.
            fresh = next((w for w in _worktrees(repo) if w["worktree"] == str(path)), None)
            if fresh != worktree or not _eligible(repo, fresh, active)[0]:
                print(f"KEEP {path}: eligibility changed")
                continue
            _git(repo, "worktree", "remove", str(path))
            _git(repo, "worktree", "prune")
            print(f"REMOVED {path}: {evidence}; branch refs retained")
        return 0
    except (OSError, subprocess.SubprocessError, KeyError, IndexError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
