"""Replay uses original immutable inputs with new normal admitted executions."""
import importlib.util
import json

import pytest

from test_recovery_restore import package as package, target_catalog as target_catalog


def replay_module():
    assert importlib.util.find_spec("papervault.store.recovery_replay") is not None
    from papervault.store import recovery_replay
    return recovery_replay


async def test_replay_dry_run_does_not_allocate_targets_or_jobs(package, target_catalog, tmp_path):
    module = replay_module()
    path, source = package
    connection, schema = target_catalog
    before = source.get_authority_state()
    target = tmp_path / "replay"
    result = await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
        catalog_connection=connection, catalog_schema=schema)
    assert result["apply"] is False and len(result["units"]) == 2
    assert not target.exists()
    assert connection.execute("SELECT to_regnamespace(%s)", (schema,)).fetchone()[0] is None
    assert source.get_authority_state() == before


async def test_replay_uses_real_metadata_pipeline_and_new_fences_without_serving(package, target_catalog, tmp_path):
    module = replay_module()
    path, source = package
    connection, schema = target_catalog
    before = source.get_authority_state()
    result = await module.replay_recovery_set(path, tmp_path / "replay", source_generation="source",
        target_generation="replayed", catalog_connection=connection, catalog_schema=schema, apply=True)
    assert result["kind"] == "replay_materialization"
    assert result["sample"] is True
    assert len(result["receipts"]) == 2
    assert {row["achieved_tier"] for row in result["receipts"]} == {"metadata"}
    assert len({row["namespace"] for row in result["receipts"]}) == 2
    assert all(row["authority_epoch"] > before["authority_epoch"] for row in result["receipts"])
    assert all(row["sealed"] for row in result["receipts"])
    assert result["serving_generation_id"] is None
    assert source.get_authority_state() == before
    from papervault.store.knowledge import KnowledgeRepository, ClaimRejected
    from papervault.store.repository import PostgresCatalogRepository
    repo = PostgresCatalogRepository(connection, schema=schema)
    repo.fence_id = result["writer_fence"]
    with pytest.raises(ClaimRejected, match="sample"):
        KnowledgeRepository(repo).command("promote_generation", "unsafe", {"generation_id": "replayed"})
    with pytest.raises(ClaimRejected):
        source.command("publish_derivation", "refuse-isolated-receipt", {
            **result["receipts"][0], "receipt": result["receipts"][0], "expected_revision": 1,
            "expected_selection_revision": 0, "select": True})


async def test_replay_resumes_completed_admissions_without_new_attempts(package, target_catalog, tmp_path, monkeypatch):
    module = replay_module()
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "replay"
    original = CatalogScheduler.run_round
    calls = 0

    async def interrupt(scheduler, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected between admitted units")
        return await original(scheduler, *args, **kwargs)

    monkeypatch.setattr(CatalogScheduler, "run_round", interrupt)
    with pytest.raises(OSError, match="injected"):
        await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
            catalog_connection=connection, catalog_schema=schema, apply=True)
    state = json.loads((target / ".replay.json").read_bytes())
    assert len(state["rounds"]) == 1
    monkeypatch.setattr(CatalogScheduler, "run_round", original)
    receipt = await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
        catalog_connection=connection, catalog_schema=schema, apply=True)
    assert len(receipt["receipts"]) == 2
    assert receipt["attempt_count"] == 2
    assert receipt["failures"] == [{"error_type": "OSError"}]


async def test_interrupted_replay_can_resume_in_next_monthly_window(package, target_catalog, tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from psycopg import sql
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    module = replay_module()
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "replay"
    original = CatalogScheduler.run_round
    previous = datetime.now(timezone.utc).replace(day=1) - timedelta(days=1)

    class PreviousMonth(datetime):
        @classmethod
        def now(cls, tz=None):
            return previous

    async def interrupt(*args, **kwargs):
        raise OSError("window interrupted before first admission")

    monkeypatch.setattr(CatalogScheduler, "run_round", interrupt)
    monkeypatch.setattr(module, "datetime", PreviousMonth)
    with pytest.raises(OSError):
        await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
            catalog_connection=connection, catalog_schema=schema, apply=True)
    # Real database time governs admission in the new window.
    monkeypatch.setattr(module, "datetime", datetime)
    monkeypatch.setattr(CatalogScheduler, "run_round", original)
    result = await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
        catalog_connection=connection, catalog_schema=schema, apply=True)
    assert result["attempt_count"] == 2
    windows = connection.execute(sql.SQL("SELECT * FROM {}").format(
        sql.Identifier(schema, "knowledge_maintenance_windows"))).fetchall()
    assert len(windows) == 2
    assert all((row["ends_at"] - row["starts_at"]).total_seconds() <= 7200 for row in windows)


async def test_completed_replay_requires_fresh_output_readback(package, target_catalog, tmp_path, monkeypatch):
    module = replay_module()
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "replay"
    await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
        catalog_connection=connection, catalog_schema=schema, apply=True)
    original = CatalogLightRAGAdapter.collect_observations

    async def changed(adapter, receipt):
        observations = await original(adapter, receipt)
        observations[0]["state"] = "missing"
        return observations

    monkeypatch.setattr(CatalogLightRAGAdapter, "collect_observations", changed)
    with pytest.raises(ValueError, match="readback"):
        await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
            catalog_connection=connection, catalog_schema=schema, apply=True)


@pytest.mark.parametrize("damage", ["fence", "epoch", "authority", "receipt"])
async def test_completed_replay_revalidates_fence_epoch_and_receipt(package, target_catalog, tmp_path, damage):
    from psycopg import sql
    module = replay_module()
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "replay"
    await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
        catalog_connection=connection, catalog_schema=schema, apply=True)
    if damage == "fence":
        connection.execute(sql.SQL("UPDATE {} SET fence_id=NULL").format(sql.Identifier(schema, "writer_fence")))
    elif damage == "epoch":
        connection.execute(sql.SQL("UPDATE {} SET authority_epoch=authority_epoch+1").format(
            sql.Identifier(schema, "knowledge_runtime")))
    elif damage == "authority":
        connection.execute(sql.SQL("UPDATE {} SET authority='legacy'").format(sql.Identifier(schema, "knowledge_runtime")))
    else:
        receipt = json.loads((target / "replay-receipt.json").read_bytes())
        receipt.update(sample=False, acceptance_scope="live", source_snapshot_id="corrupt")
        (target / "replay-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="fence|epoch|authority|receipt"):
        await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
            catalog_connection=connection, catalog_schema=schema, apply=True)


async def test_replay_recovers_receipt_after_lost_completion_response(package, target_catalog, tmp_path, monkeypatch):
    module = replay_module()
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "replay"
    original = module.write_once

    def interrupt(path, data):
        original(path, data)
        if path.name == "replay-receipt.json":
            raise OSError("response lost after durable receipt")

    monkeypatch.setattr(module, "write_once", interrupt)
    with pytest.raises(OSError, match="response lost"):
        await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
            catalog_connection=connection, catalog_schema=schema, apply=True)
    frozen = (target / "replay-receipt.json").read_bytes()
    monkeypatch.setattr(module, "write_once", original)
    result = await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
        catalog_connection=connection, catalog_schema=schema, apply=True)
    assert result == json.loads(frozen)
    assert result["attempt_count"] == 2


async def test_isolated_replay_records_unit_quarantine_reset(package, target_catalog, tmp_path):
    from papervault.store.recovery import snapshot
    from papervault.store.recovery_source import CoordinatedSource
    from papervault.store.knowledge import KnowledgeRepository
    from papervault.store.repository import PostgresCatalogRepository

    _, source = package
    unit = source._rows("units")[0]
    reason = {"reason": "source execution uncertain", "scope": "unit", "origin": "test"}
    source.command("set_quarantine", "quarantine-before-snapshot", {"unit_id": unit["unit_id"],
        "expected_revision": unit["revision"], "quarantine": True, "reason": reason})
    captured = snapshot(CoordinatedSource(source.repository, vault=tmp_path / "vault",
        knowledge_root=tmp_path / "knowledge", metadata_sample=True), tmp_path / "quarantined-sets",
        apply=True, sample=True)
    path = tmp_path / "quarantined-sets" / captured["snapshot_id"]
    connection, schema = target_catalog
    result = await replay_module().replay_recovery_set(path, tmp_path / "replay", source_generation="source",
        target_generation="replayed", catalog_connection=connection, catalog_schema=schema, apply=True)
    assert len(result["receipts"]) == 2
    disposition = result.get("unit_quarantine_disposition")
    assert disposition == {"policy": "reset_in_isolated_replay", "reset_units": [
        {"unit_id": unit["unit_id"], "source_revision": unit["revision"] + 1, "source_reason": reason}]}
    assert source.get_unit(unit["unit_id"])["quarantine"] is True
    assert KnowledgeRepository(PostgresCatalogRepository(connection, schema=schema)).get_unit(unit["unit_id"])["quarantine"] is False


def test_replay_default_capacity_counts_catalog_and_requires_physical_census(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from papervault.store import recovery_replay
    from papervault.store.scheduler_capacity import CapacityAdmissionError

    monkeypatch.delenv("PAPERVAULT_DERIVATIVE_USAGE_FILE", raising=False)
    monkeypatch.setenv("PAPERVAULT_CATALOG_EXECUTION_BYTES", "1024")
    connection = SimpleNamespace(execute=lambda *args: SimpleNamespace(fetchone=lambda: {"bytes": 91}))
    gate = recovery_replay.replay_capacity(tmp_path / "target", connection, "catalog_test", physical=False)
    with gate.intake() as plan:
        assert plan["used_bytes"] >= 91
    physical = recovery_replay.replay_capacity(tmp_path / "physical", connection, "catalog_test", physical=True)
    with pytest.raises(CapacityAdmissionError):
        with physical.intake():
            pytest.fail("physical replay must have a fresh scoped backend census")


@pytest.mark.asyncio
async def test_replay_capacity_denial_precedes_copies_catalog_import_and_backend_factory(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from papervault.store import recovery_replay
    from papervault.store.scheduler_capacity import CapacityAdmissionError

    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text("{}")
    manifest = {"source_roots": []}
    catalog = {"tables": {"generations": []}}
    plan = {"physical": True, "inputs": [{"manifest": {"artifacts": [{"sha256": "abc", "byte_length": 1}]}}]}
    monkeypatch.setattr(recovery_replay, "replay_plan", lambda *args, **kwargs: (manifest, catalog, plan))
    monkeypatch.setattr(recovery_replay, "validate_catalog_target", lambda *args: None)
    monkeypatch.setattr(recovery_replay, "copy_verified_file", lambda *args: pytest.fail("unfunded input copy"))
    monkeypatch.setattr(recovery_replay, "_initialize", lambda *args: pytest.fail("unfunded catalog initialize"))
    monkeypatch.setattr(recovery_replay, "_import_inputs", lambda *args: pytest.fail("unfunded import"))

    class Denied:
        @contextmanager
        def intake(self):
            raise CapacityAdmissionError("capacity exhausted")
            yield

    connection = SimpleNamespace(info=SimpleNamespace(dsn="dbname=isolated_test user=test host=127.0.0.1"),
        execute=lambda *args: SimpleNamespace(fetchone=lambda: (None,)))
    with pytest.raises(CapacityAdmissionError):
        await recovery_replay.replay_recovery_set(package, tmp_path / "target", source_generation="old",
            target_generation="new", catalog_connection=connection, catalog_schema="catalog_test", apply=True,
            adapter_factory=lambda *args, **kwargs: pytest.fail("unfunded backend preparation"),
            backend_identity={"isolated": True}, capacity_gate=Denied())


@pytest.mark.asyncio
async def test_replay_shares_intake_gate_with_scheduler_and_backend_preparation(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from papervault.store import recovery_replay

    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text("{}")
    manifest = {"source_roots": []}
    catalog = {"tables": {"generations": []}}
    plan = {"physical": True, "inputs": [{"manifest": {"artifacts": [{"sha256": "abc", "byte_length": 1}]}}],
        "requirements": [{"unit_id": "unit", "excluded": False}]}
    monkeypatch.setattr(recovery_replay, "replay_plan", lambda *args, **kwargs: (manifest, catalog, plan))
    monkeypatch.setattr(recovery_replay, "validate_catalog_target", lambda *args: None)
    events = []

    class Gate:
        active = False

        @contextmanager
        def intake(self):
            self.active = True
            try:
                yield {}
            finally:
                self.active = False

    gate = Gate()

    def funded(name):
        assert gate.active, name + " escaped capacity reservation"
        events.append(name)

    def initialize(*args):
        funded("catalog")
        return SimpleNamespace(get_selection=lambda *args: {"derivation_id": "selected"})

    def adapter(*args, **kwargs):
        funded("backend")
        return object()

    class StopAtScheduler(RuntimeError):
        pass

    class Scheduler:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("capacity_gate") is gate
            assert events == ["copy", "catalog", "import", "backend"]
            raise StopAtScheduler

    monkeypatch.setattr(recovery_replay, "copy_verified_file", lambda *args: funded("copy"))
    monkeypatch.setattr(recovery_replay, "_initialize", initialize)
    monkeypatch.setattr(recovery_replay, "_import_inputs", lambda *args: funded("import"))
    monkeypatch.setattr("papervault.knowledge.scheduler.catalog.CatalogScheduler", Scheduler)
    connection = SimpleNamespace(info=SimpleNamespace(dsn="dbname=isolated_test user=test host=127.0.0.1"),
        execute=lambda *args: SimpleNamespace(fetchone=lambda: (None,)))
    with pytest.raises(StopAtScheduler):
        await recovery_replay.replay_recovery_set(package, tmp_path / "target", source_generation="old",
            target_generation="new", catalog_connection=connection, catalog_schema="catalog_test", apply=True,
            adapter_factory=adapter, backend_identity={"isolated": True}, capacity_gate=gate)
