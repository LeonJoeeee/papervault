"""Recovery packages must be complete before publication or retention can move."""
import hashlib
import json
from contextlib import contextmanager

import pytest


STORES = ("catalog", "sources", "extracts", "inputs", "recipes", "postgres", "neo4j")


class FrozenSource:
    def __init__(self):
        self.fenced = False
        self.registered = []
        self.omit = None
        self.mismatch = None
        self.fail = False

    def plan(self):
        return {"required_stores": list(STORES), "estimated_bytes": 4096}

    @contextmanager
    def barrier(self, snapshot_id):
        self.fenced = True
        try:
            yield {"backup_id": snapshot_id, "authority_epoch": 4,
                   "catalog_snapshot": "104:104:", "audit_head": 97}
        finally:
            self.fenced = False

    def export(self, destination, checkpoint):
        components = {}
        for name in STORES:
            if name == self.omit:
                continue
            path = destination / name / "data.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"store": name, "rows": []}))
            components[name] = {"checkpoint": dict(checkpoint), "format": "fixture-json-v1",
                                "paths": [str(path.relative_to(destination))]}
            if name == self.mismatch:
                components[name]["checkpoint"]["audit_head"] = 98
        if self.fail:
            raise OSError("injected export failure")
        return {"components": components, "parent_inventory": [], "protected_paths": ["sources/data.json"],
                "tool_identities": {"exporter": "fixture-v1"}, "limitations": ["fixture source"]}

    def register(self, manifest, path):
        assert self.fenced, "barrier must cover verification and registration"
        assert path.is_dir()
        self.registered.append(manifest["snapshot_id"])


def recovery():
    # A missing implementation is an assertion failure in the initial red run.
    import importlib.util
    assert importlib.util.find_spec("papervault.store.recovery") is not None
    from papervault.store import recovery
    return recovery


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_snapshot_dry_run_has_no_files_barrier_or_catalog_effect(tmp_path):
    module = recovery()
    source = FrozenSource()
    root = tmp_path / "absent"
    result = module.snapshot(source, root)
    assert result["apply"] is False
    assert set(result["required_stores"]) == set(STORES)
    assert not root.exists()
    assert not source.fenced and not source.registered


def test_source_isolation_canonicalizes_dotdot_before_comparison(tmp_path):
    module = recovery()
    source = FrozenSource()
    vault = tmp_path / "vault"
    vault.mkdir()
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    source.plan = lambda: {"required_stores": list(STORES), "estimated_bytes": 4096,
                           "source_roots": [str(vault)]}
    with pytest.raises(ValueError, match="isolated"):
        module.snapshot(source, sibling / ".." / "vault" / "packages", apply=True)
    assert not (vault / "packages").exists()


def test_capacity_alert_does_not_delete_assets_or_relax_free_space_admission(tmp_path, monkeypatch):
    module = recovery()
    monkeypatch.setattr(module, "DERIVATIVE_SOFT_CAP", 4000)
    source = FrozenSource()
    result = module.snapshot(source, tmp_path, apply=True)
    assert result["capacity"]["alert"] is True
    assert result["capacity"]["action"] == "alert_only"
    assert (tmp_path / result["snapshot_id"] / "sources/data.json").exists()


def test_snapshot_stops_at_maintenance_deadline_and_cannot_rotate_retention(tmp_path, monkeypatch):
    import time
    module = recovery()
    monkeypatch.setattr(module, "MAX_WINDOW_SECONDS", 0.02)
    source = FrozenSource()
    original = source.export
    export_finished = False

    def slow(destination, checkpoint):
        nonlocal export_finished
        time.sleep(0.1)
        export_finished = True
        return original(destination, checkpoint)

    source.export = slow
    with pytest.raises((TimeoutError, ValueError), match="maintenance|two-hour"):
        module.snapshot(source, tmp_path, apply=True)
    # Assert that the deadline interrupts export, rather than measuring the
    # unrelated setup/fsync/cleanup cost against a host scheduling threshold.
    assert not export_finished
    assert not (tmp_path / "retention.json").exists()
    assert not source.registered


def test_same_day_snapshots_have_distinct_verified_identities(tmp_path):
    module = recovery()
    source = FrozenSource()
    first = module.snapshot(source, tmp_path, apply=True)
    second = module.snapshot(source, tmp_path, apply=True)
    assert first["snapshot_id"] != second["snapshot_id"]
    for manifest in (first, second):
        package = tmp_path / manifest["snapshot_id"]
        verified = module.verify_recovery_set(package)
        assert verified == manifest
        assert verified["complete"] is True
        assert set(verified["components"]) == set(STORES)
        for item in verified["files"]:
            data = (package / item["path"]).read_bytes()
            assert item["sha256"] == hashlib.sha256(data).hexdigest()
            assert item["byte_length"] == len(data)
    assert source.registered == [first["snapshot_id"], second["snapshot_id"]]
    assert not source.fenced


@pytest.mark.parametrize("store", STORES)
def test_omitted_store_cannot_publish_or_register(tmp_path, store):
    module = recovery()
    source = FrozenSource()
    source.omit = store
    with pytest.raises(ValueError, match="store"):
        module.snapshot(source, tmp_path, apply=True)
    assert not source.registered
    assert not (tmp_path / "retention.json").exists()


def test_mixed_checkpoint_cannot_publish(tmp_path):
    module = recovery()
    source = FrozenSource()
    source.mismatch = "postgres"
    with pytest.raises(ValueError, match="checkpoint"):
        module.snapshot(source, tmp_path, apply=True)
    assert not source.registered


@pytest.mark.parametrize("fault", ["corrupt", "missing", "extra", "symlink", "incomplete", "manifest"])
def test_verifier_refuses_damaged_or_incomplete_packages(tmp_path, fault):
    module = recovery()
    result = module.snapshot(FrozenSource(), tmp_path, apply=True)
    package = tmp_path / result["snapshot_id"]
    file = package / "postgres/data.json"
    if fault == "corrupt":
        file.write_bytes(b"corrupt")
    elif fault == "missing":
        file.unlink()
    elif fault == "extra":
        (package / "unmanifested").write_bytes(b"surprise")
    elif fault == "symlink":
        file.unlink()
        file.symlink_to(package / "neo4j/data.json")
    elif fault == "incomplete":
        (package / "INCOMPLETE").write_text("interrupted")
    else:
        manifest = json.loads((package / "manifest.json").read_bytes())
        manifest["checkpoint"]["audit_head"] += 1
        (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.verify_recovery_set(package)


def test_failed_and_sample_exports_do_not_displace_four_complete_sets(tmp_path):
    module = recovery()
    source = FrozenSource()
    retained = [module.snapshot(source, tmp_path, apply=True)["snapshot_id"] for _ in range(4)]
    original = (tmp_path / "retention.json").read_bytes()
    source.fail = True
    with pytest.raises(OSError, match="injected"):
        module.snapshot(source, tmp_path, apply=True)
    source.fail = False
    sample = module.snapshot(source, tmp_path, apply=True, sample=True)
    assert sample["sample"] is True
    assert (tmp_path / "retention.json").read_bytes() == original
    assert all((tmp_path / identity).is_dir() for identity in retained)
    newest = module.snapshot(source, tmp_path, apply=True)
    index = json.loads((tmp_path / "retention.json").read_bytes())
    assert index["retained"] == retained[1:] + [newest["snapshot_id"]]
    assert not (tmp_path / retained[0]).exists()
    assert (tmp_path / sample["snapshot_id"]).is_dir()


def test_capacity_exhaustion_preserves_existing_packages(tmp_path, monkeypatch):
    module = recovery()
    module.snapshot(FrozenSource(), tmp_path, apply=True)
    before = tree(tmp_path)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda root: (1000, 999, 1))
    with pytest.raises(ValueError, match="capacity"):
        module.snapshot(FrozenSource(), tmp_path, apply=True)
    assert tree(tmp_path) == before


def test_snapshot_refuses_symlink_root_before_writing(tmp_path):
    module = recovery()
    owned = tmp_path / "owned"
    owned.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(owned, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.snapshot(FrozenSource(), alias, apply=True)
    assert not list(owned.iterdir())
