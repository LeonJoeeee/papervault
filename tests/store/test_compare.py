"""Synthetic shadow workload, independent of PostgreSQL."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


class Records:
    def __init__(self, *records):
        self.records = {r.key: r for r in records}

    def all(self):
        return iter(self.records.values())

    def get(self, key):
        return self.records.get(key)

    def all_aliases(self):
        return {r.key: r.payload["terminal_state"]["alias_key"] for r in self.records.values()
                if r.merge == "alias"}


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


def test_limit_keeps_landmark_keys_that_exist_only_in_postgres(tmp_path):
    from papervault.store.compare import compare_catalog
    from papervault.store.repository import Record

    a = Record.from_payload("A", {"key": "A"})
    z = Record.from_payload("Z", {"key": "Z", "doi": "10.1/canon"})
    pool = tmp_path / "pools.jsonl"
    pool.write_text(json.dumps({"qid": "r2", "pool": [{"doi": "10.1/canon"}]}) + "\n")
    report = compare_catalog(Records(a), Records(a, z), limit=1, pool_path=pool)
    assert report["pool_keys"] == ["Z"]
    assert report["compared"] == 2
    assert report["different"] == 1
    assert report["difference_rate"] == 0.5


def test_comparator_reports_relational_alias_drift():
    from papervault.store.compare import compare_catalog
    from papervault.store.repository import Record

    a, b = (Record.from_payload(key, {"key": key}) for key in "AB")
    left, right = Records(a, b), Records(a, b)
    right.all_aliases = lambda: {"A": "B"}
    report = compare_catalog(left, right)
    assert report["different"] == 1
    assert report["differences"][0]["fields"]["alias"] == {"json": None, "postgres": "B"}


@pytest.mark.parametrize(("identifiers", "columns"), [
    ({"doi": " 10.1234/Retained "}, {"doi": "10.1234/retained"}),
    ({"arxiv_id": " 2101.12345V2 "}, {"arxiv_id": "2101.12345v2"}),
    ({"doi": " 10.1234/Retained ", "arxiv_id": " 2101.12345V2 "},
     {"doi": "10.1234/retained", "arxiv_id": "2101.12345v2"}),
])
def test_export_snapshot_comparison_accepts_retained_legacy_alias_columns(tmp_path, identifiers, columns):
    from papervault.store.adapter import LibraryAdapter
    from papervault.store.compare import compare_catalog
    from papervault.store.repository import Record

    canonical = Record.from_payload("A", {"key": "A"})
    alias = Record.from_payload("B", {"key": "B", **identifiers,
        "terminal_state": {"merge": "alias", "alias_key": "A"}})
    snapshot = tmp_path / "index.json"
    snapshot.write_text(json.dumps({"version": 1, "papers": {"A": canonical.payload, "B": alias.payload}}))
    report = compare_catalog(LibraryAdapter.from_snapshot(snapshot),
                             Records(canonical, replace(alias, **columns)))
    assert report["compared"] == 2
    assert report["differences"] == []
    assert report["difference_rate"] == 0


@pytest.mark.parametrize(("drift", "expected_fields"), [
    ("different_doi", {"doi"}), ("different_arxiv_version", {"arxiv_id"}),
    ("unnormalized_doi", {"doi"}), ("unnormalized_arxiv", {"arxiv_id"}),
    ("different_payload", {"doi", "arxiv_id", "payload_sha256"}),
    ("canonical", {"doi", "arxiv_id"}),
    ("one_canonical", {"doi", "arxiv_id", "merge"}),
    ("left_identifier", {"doi"}), ("reverse", {"doi", "arxiv_id"}),
    ("title", {"title"}),
])
def test_retained_alias_exception_does_not_hide_catalog_drift(drift, expected_fields):
    from papervault.store.compare import record_differences
    from papervault.store.repository import Record

    left = Record.from_payload("B", {"key": "B", "doi": "10.1234/Retained",
        "arxiv_id": "2101.12345V2", "terminal_state": {"merge": "alias", "alias_key": "A"}})
    right = replace(left, doi="10.1234/retained", arxiv_id="2101.12345v2")
    if drift == "different_doi":
        right = replace(right, doi="10.1234/other")
    elif drift == "different_arxiv_version":
        right = replace(right, arxiv_id="2101.12345v3")
    elif drift == "unnormalized_doi":
        right = replace(right, doi="10.1234/Retained")
    elif drift == "unnormalized_arxiv":
        right = replace(right, arxiv_id="2101.12345V2")
    elif drift == "different_payload":
        right = replace(right, payload={**right.payload, "legacy": "changed"})
    elif drift == "canonical":
        left, right = replace(left, merge="canonical"), replace(right, merge="canonical")
    elif drift == "one_canonical":
        right = replace(right, merge="canonical")
    elif drift == "left_identifier":
        left = replace(left, doi="10.1234/other")
    elif drift == "reverse":
        left, right = right, left
    elif drift == "title":
        right = replace(right, title="Column drift")
    assert set(record_differences(left, right)) == expected_fields
