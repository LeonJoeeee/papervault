"""Import mapping, restart, identity conflict and command-line contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


class MemoryCatalog:
    """Test-only repository double; real SQL constraints live in test_postgres.py."""

    def __init__(self):
        self.records = {}
        self.audit = []
        self.aliases = {}
        self.rulings = {}
        self.jobs = {}
        self.unique_fields = set()

    def ensure_schema(self, *, audit):
        pass

    def all(self):
        return iter(self.records.values())

    def get(self, key):
        return self.records.get(key)

    def upsert(self, record, *, audit, expected_revision):
        if self.records.get(record.key) == record:
            return False
        self.records[record.key] = record
        self.audit.append(audit)
        return True

    def add_alias(self, alias, canonical, *, audit):
        if self.aliases.get(alias) == canonical:
            return False
        self.aliases[alias] = canonical
        self.audit.append(audit)
        return True

    def import_ruling(self, ruling, *, audit):
        from papervault.store.repository import payload_hash
        identifier = payload_hash(ruling)
        if identifier in self.rulings:
            return False
        self.rulings[identifier] = ruling
        self.audit.append(audit)
        return True

    def import_job(self, job, *, audit):
        if self.jobs.get(job.id) == job:
            return False
        self.jobs[job.id] = job
        self.audit.append(audit)
        return True

    def finalize_identity_indexes(self, *, audit):
        return {}

    def identity_index_fields(self):
        return self.unique_fields

    def all_aliases(self):
        return dict(self.aliases)


def snapshot(tmp_path, papers):
    from papervault.store.adapter import LibraryAdapter

    path = tmp_path / "index.json"
    path.write_text(json.dumps({"version": 1, "papers": {p["key"]: p for p in papers}}))
    return LibraryAdapter.from_snapshot(path)


def test_duplicate_identity_is_reported_without_merging_and_import_replays(tmp_path):
    from papervault.store.importer import import_catalog

    adapter = snapshot(tmp_path, [
        {"key": "A", "doi": "10.1/SAME", "download_status": "ok", "legacy": 42},
        {"key": "B", "doi": " 10.1/same "},
    ])
    repo = MemoryCatalog()
    first = import_catalog(adapter, repo)
    assert first["identity_conflicts"] == [{"field": "doi", "value": "10.1/same", "keys": ["A", "B"]}]
    assert first["counts"]["download_status"] == {"ok": 1, "pending": 1}
    assert first["per_key_match"] == {"A": True, "B": True}
    assert first["difference_rate"] == 0
    assert first["changed"] == 2
    assert repo.records["A"].payload["legacy"] == 42
    second = import_catalog(adapter, repo)
    assert second["changed"] == 0
    assert len(repo.records) == 2
    assert len(repo.audit) == 2
    dry = import_catalog(adapter)
    assert dry["dry_run"] is True
    assert dry["per_key_match"] == {"A": None, "B": None}
    assert dry["difference_rate"] is None


def test_import_resumes_after_a_committed_prefix(tmp_path):
    from papervault.store.importer import import_catalog

    adapter = snapshot(tmp_path, [{"key": key} for key in "ABC"])
    repo = MemoryCatalog()
    upsert = repo.upsert

    def interrupted(record, *, audit, expected_revision):
        if record.key == "B":
            raise RuntimeError("interrupted")
        return upsert(record, audit=audit, expected_revision=expected_revision)

    repo.upsert = interrupted
    with pytest.raises(RuntimeError, match="interrupted"):
        import_catalog(adapter, repo)
    assert list(repo.records) == ["A"]
    repo.upsert = upsert
    result = import_catalog(adapter, repo)
    assert result["changed"] == 2
    assert list(repo.records) == ["A", "B", "C"]
    assert len(repo.audit) == 3


def test_import_aliases_rulings_and_job_snapshots_idempotently(tmp_path):
    from papervault.store.importer import import_catalog

    adapter = snapshot(tmp_path, [{"key": "A"}, {
        "key": "B", "terminal_state": {"merge": "alias", "alias_key": "A"},
    }])
    ledger = tmp_path / "correction-ledger.json"
    ledger.write_text(json.dumps({"policy_rulings": [{
        "timestamp": "2026-10-18T00:00:00+00:00", "action": "acquire", "reason": "human ruling",
        "keys": ["A"], "revision": 1,
    }]}))
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"id": "job-A", "kind": "extract", "key": "A"}]}))
    repo = MemoryCatalog()
    import_catalog(adapter, repo, jobs_path=jobs)
    before = len(repo.audit)
    report = import_catalog(adapter, repo, jobs_path=jobs)
    assert report["supporting_changed"] == {"aliases": 0, "rulings": 0, "jobs": 0}
    assert repo.aliases == {"B": "A"}
    assert len(repo.rulings) == len(repo.jobs) == 1
    assert len(repo.audit) == before


def test_bad_alias_references_are_reported_before_writes(tmp_path):
    from papervault.store.importer import import_catalog

    adapter = snapshot(tmp_path, [{
        "key": "A", "terminal_state": {"merge": "alias", "alias_key": "missing"},
    }])
    repo = MemoryCatalog()
    report = import_catalog(adapter, repo)
    assert report["integrity_errors"]
    assert repo.records == {}
    assert repo.audit == []


def test_dry_run_command_does_not_require_dsn_or_write_the_snapshot(tmp_path):
    adapter = snapshot(tmp_path, [{"key": "A", "legacy": "retained"}])
    path = Path(adapter.source["path"])
    before = path.read_bytes()
    env = {k: v for k, v in os.environ.items() if k != "PAPERVAULT_CATALOG_DSN"}
    script = Path(__file__).resolve().parents[2] / "scripts" / "import_catalog.py"
    result = subprocess.run([sys.executable, str(script), "--dry-run", "--index", str(path)],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["total"] == 1
    assert len(report["payload_sha256"]["A"]) == 64
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    result = subprocess.run([sys.executable, str(script), "--apply", "--index", str(path)],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 2
    assert "PAPERVAULT_CATALOG_DSN" in result.stderr


def test_existing_target_identity_collision_is_reported_before_record_writes(tmp_path):
    from papervault.store.importer import import_catalog
    from papervault.store.repository import Record

    repo = MemoryCatalog()
    repo.records["Z"] = Record.from_payload("Z", {"key": "Z", "doi": "10.1/same"})
    repo.unique_fields = {"doi"}
    adapter = snapshot(tmp_path, [{"key": "A"}, {"key": "B", "doi": "10.1/SAME"}])
    report = import_catalog(adapter, repo)
    assert report["identity_conflicts"] == [{"field": "doi", "value": "10.1/same", "keys": ["B", "Z"]}]
    assert report["integrity_errors"]
    assert report["changed"] == 0
    assert list(repo.records) == ["Z"]
    assert repo.audit == []


def test_stale_target_alias_prevents_a_false_zero_difference_report(tmp_path):
    from papervault.store.importer import import_catalog

    repo = MemoryCatalog()
    before = snapshot(tmp_path, [{"key": "A", "terminal_state": {"merge": "alias", "alias_key": "B"}},
                                {"key": "B"}])
    import_catalog(before, repo)
    after = snapshot(tmp_path, [{"key": "A"}, {"key": "B"}])
    report = import_catalog(after, repo)
    assert report["per_key_match"] == {"A": False, "B": True}
    assert report["difference_rate"] == 0.5
    assert report["supporting_differences"]["aliases"] == [
        {"key": "A", "json": None, "postgres": "B"},
    ]


def test_interrupted_import_reports_the_committed_progress(tmp_path):
    from papervault.store.importer import import_catalog

    adapter = snapshot(tmp_path, [{"key": "A"}, {"key": "B"}])
    repo = MemoryCatalog()
    upsert = repo.upsert

    def interrupted(record, *, audit, expected_revision):
        if record.key == "B":
            raise RuntimeError("interrupted")
        return upsert(record, audit=audit, expected_revision=expected_revision)

    repo.upsert = interrupted
    with pytest.raises(RuntimeError) as failure:
        import_catalog(adapter, repo)
    assert failure.value.catalog_report["changed"] == 1
    assert failure.value.catalog_report["difference_rate"] is None


def test_import_materializes_jobs_for_active_claims_before_handoff(tmp_path):
    from papervault.store.importer import import_catalog

    adapter = snapshot(tmp_path, [{"key": "A", "reopen_claim": {
        "key": "A", "generation": 2, "lease_id": "current", "lease_expiry": 9999999999.,
        "campaign_id": "c", "phase": "dispatched",
    }}])
    repo = MemoryCatalog()
    report = import_catalog(adapter, repo)
    assert report["supporting_counts"]["jobs"] == 1
    assert repo.jobs["reopen:A"].attempts == 2
    assert repo.jobs["reopen:A"].lease_id == "current"
