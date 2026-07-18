"""Release discipline guard (walkthrough finding 4): the two plugin manifests agree."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_plugin_and_marketplace_versions_match():
    plugin = json.loads((ROOT / "plugin/.claude-plugin/plugin.json").read_text())
    market = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    entry = next(p for p in market["plugins"] if p["name"] == "papervault")
    assert plugin["version"] == entry["version"]
