"""Synthetic shadow workload, independent of PostgreSQL."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys


class Records:
    def __init__(self, *records):
        self.records = {r.key: r for r in records}

    def all(self):
        return iter(self.records.values())

    def get(self, key):
        return self.records.get(key)


def test_comparator_checks_columns_and_payload_and_missing_keys():
    from papervault.store.compare import compare_catalog
    from papervault.store.repository import Record

    a = Record.from_payload("A", {"key": "A", "title": "Original", "legacy": 1})
    b = Record.from_payload("B", {"key": "B"})
    extra = Record.from_payload("extra", {"key": "extra"})
    report = compare_catalog(Records(a, b), Records(
        replace(a, title="Drift", payload={**a.payload, "legacy": 2}), extra,
    ))
    assert report["compared"] == 3
    assert report["different"] == 3
    assert report["difference_rate"] == 1.0
    fields = report["differences"][0]["fields"]
    assert fields["title"] == {"json": "Original", "postgres": "Drift"}
    assert "payload_sha256" in fields
    assert report["differences"][1]["fields"]["presence"] == {"json": True, "postgres": False}
    assert report["differences"][2]["fields"]["presence"] == {"json": False, "postgres": True}
    assert compare_catalog(Records(a), Records(a))["difference_rate"] == 0


def test_payload_only_difference_is_not_hidden_by_matching_metadata():
    from papervault.store.compare import compare_catalog
    from papervault.store.repository import Record

    left = Record.from_payload("A", {"key": "A", "legacy": "old"})
    right = Record.from_payload("A", {"key": "A", "legacy": "new"})
    report = compare_catalog(Records(left), Records(right))
    assert set(report["differences"][0]["fields"]) == {"payload_sha256"}


def test_limit_always_includes_explicit_and_f0_pool_keys(tmp_path):
    from papervault.store.compare import compare_catalog
    from papervault.store.repository import Record

    records = [Record.from_payload(k, {"key": k, "doi": doi, "title": title}) for k, doi, title in (
        ("A", "", "Other"), ("Z", "10.1/canon", "Canon"), ("Y", "", "Title MATCH!"),
    )]
    pool = tmp_path / "pools.jsonl"
    pool.write_text(json.dumps({"qid": "r2", "pool": [
        {"doi": "10.1/CANON"}, {"title": "title match"}, {"title": "Missing landmark"},
    ]}) + "\n")
    report = compare_catalog(Records(*records), Records(*records), limit=1,
                             pool_path=pool, pool_keys=["absent"])
    assert report["compared"] == 4
    assert report["pool_keys"] == ["Y", "Z", "absent"]
    assert report["different"] == 1
    assert report["pool_unmatched"] == [{"qid": "r2", "entry": {"title": "Missing landmark"}}]


def test_comparator_command_requires_explicit_shadow_dsn(tmp_path):
    path = tmp_path / "index.json"
    path.write_text('{"papers": {}}')
    env = {k: v for k, v in os.environ.items() if k != "PAPERVAULT_CATALOG_DSN"}
    script = Path(__file__).resolve().parents[2] / "scripts" / "shadow_read_compare.py"
    result = subprocess.run([sys.executable, str(script), "--index", str(path)],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 2
    assert "PAPERVAULT_CATALOG_DSN" in result.stderr
    assert list(tmp_path.iterdir()) == [path]
