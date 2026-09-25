"""Release discipline guard: the repo carries one version (README "Releases").

The plugin manifest, its marketplace entry, and the Python package version (pyproject.toml
plus its uv.lock entry) move together in every change PR; this test fails the moment any of
them drifts.
"""
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _versions() -> dict[str, str]:
    plugin = json.loads((ROOT / "plugin/.claude-plugin/plugin.json").read_text())
    market = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    entry = next(p for p in market["plugins"] if p["name"] == "papervault")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = next(p for p in lock["package"] if p["name"] == "papervault")
    return {
        "plugin/.claude-plugin/plugin.json": plugin["version"],
        ".claude-plugin/marketplace.json": entry["version"],
        "pyproject.toml": project["version"],
        "uv.lock": locked["version"],
    }


def test_plugin_and_marketplace_versions_match():
    v = _versions()
    assert v["plugin/.claude-plugin/plugin.json"] == v[".claude-plugin/marketplace.json"]


def test_plugin_manifests_follow_package_version():
    v = _versions()
    assert len(set(v.values())) == 1, f"version fields disagree: {v}"
