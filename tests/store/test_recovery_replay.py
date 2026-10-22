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
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("checkpoint", ["sealed", "selected", "complete"])
async def test_exhausted_replay_recovers_existing_work_without_new_admission(
        package, target_catalog, tmp_path, monkeypatch, checkpoint):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler

    module = replay_module()
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "replay"
    options = dict(source_generation="source", target_generation="replayed",
        catalog_connection=connection, catalog_schema=schema, apply=True)

    class Interrupted(BaseException):
        pass

    finish = CatalogScheduler._finish
    write = module.write_once

    async def before_completion(scheduler, receipt, summary):
        if scheduler.repository.get_unit(receipt["unit_id"])["record_key"] == "B":
            raise Interrupted("second receipt sealed before completion")
        return await finish(scheduler, receipt, summary)

    def before_receipt(destination, data):
        if destination.name == "replay-receipt.json":
            raise Interrupted("all units selected before final receipt")
        return write(destination, data)

    if checkpoint == "sealed":
        monkeypatch.setattr(CatalogScheduler, "_finish", before_completion)
    elif checkpoint == "selected":
        monkeypatch.setattr(module, "write_once", before_receipt)
    if checkpoint == "complete":
        completed = await module.replay_recovery_set(path, target, **options)
    else:
        with pytest.raises(Interrupted):
            await module.replay_recovery_set(path, target, **options)
    monkeypatch.setattr(CatalogScheduler, "_finish", finish)
    monkeypatch.setattr(module, "write_once", write)

    gate = module.replay_capacity(target, connection, schema, physical=False)
    gate.soft_cap = 1
    result = await module.replay_recovery_set(path, target, **options, capacity_gate=gate)
    assert len(result["receipts"]) == 2
    assert result["attempt_count"] == 2
    assert result["readback"]["outputs"] > 0
    if checkpoint == "sealed":
        progress = json.loads((target / ".replay.json").read_bytes())
        assert progress["rounds"][-1]["completed"] == 1
        assert progress["rounds"][-1]["published"] == 1
        assert progress["rounds"][-1]["claimed"] == 0
    elif checkpoint == "complete":
        assert result == completed


async def test_exhausted_completed_replay_still_checks_current_output_bytes(
        package, target_catalog, tmp_path):
    module = replay_module()
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "replay"
    options = dict(source_generation="source", target_generation="replayed",
        catalog_connection=connection, catalog_schema=schema, apply=True)
    await module.replay_recovery_set(path, target, **options)
    next((target / "knowledge/catalog_inputs").iterdir()).write_bytes(b"corrupt retained input")
    gate = module.replay_capacity(target, connection, schema, physical=False)
    gate.soft_cap = 1
    with pytest.raises(ValueError, match="checksum|readback"):
        await module.replay_recovery_set(path, target, **options, capacity_gate=gate)


async def test_denied_initial_replay_can_retry_its_capacity_bookkeeping(package, target_catalog, tmp_path):
    from papervault.store.scheduler_capacity import CapacityAdmissionError

    module = replay_module()
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "replay"
    gate = module.replay_capacity(target, connection, schema, physical=False)
    gate.soft_cap = 1
    options = dict(source_generation="source", target_generation="replayed",
        catalog_connection=connection, catalog_schema=schema, apply=True, capacity_gate=gate)
    with pytest.raises(CapacityAdmissionError):
        await module.replay_recovery_set(path, target, **options)
    assert not (target / ".replay.json").exists()
    assert not (target / "knowledge/catalog_inputs").exists()
    gate.soft_cap = 2_000_000_000_000
    assert (await module.replay_recovery_set(path, target, **options))["attempt_count"] == 2


@pytest.mark.parametrize("entry", ["unowned", "knowledge/maintenance/capacity/.scheduler-accounting.lock"])
async def test_unowned_target_content_is_not_adopted_as_capacity_bookkeeping(
        package, target_catalog, tmp_path, entry):
    module = replay_module()
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "replay"
    artifact = target / entry
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"retain unowned content")
    with pytest.raises(ValueError, match="empty isolated target"):
        await module.replay_recovery_set(path, target, source_generation="source", target_generation="replayed",
            catalog_connection=connection, catalog_schema=schema, apply=True)
    assert artifact.read_bytes() == b"retain unowned content"


@pytest.mark.parametrize("boundary", ["copy_verified_file", "_initialize", "_import_inputs"])
async def test_exhausted_partial_bootstrap_cannot_allocate_missing_replay_state(
        package, target_catalog, tmp_path, monkeypatch, boundary):
    from papervault.store.scheduler_capacity import CapacityAdmissionError

    module = replay_module()
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "replay"
    options = dict(source_generation="source", target_generation="replayed",
        catalog_connection=connection, catalog_schema=schema, apply=True)
    original = getattr(module, boundary)

    def interrupt(*args, **kwargs):
        raise OSError("bootstrap interrupted")

    monkeypatch.setattr(module, boundary, interrupt)
    with pytest.raises(OSError, match="bootstrap interrupted"):
        await module.replay_recovery_set(path, target, **options)
    monkeypatch.setattr(module, boundary, original)
    before = sorted(str(item.relative_to(target)) for item in target.rglob("*"))
    gate = module.replay_capacity(target, connection, schema, physical=False)
    gate.soft_cap = 1
    with pytest.raises(CapacityAdmissionError):
        await module.replay_recovery_set(path, target, **options, capacity_gate=gate)
    assert sorted(str(item.relative_to(target)) for item in target.rglob("*")) == before


async def test_exhausted_resumed_replay_does_not_admit_pending_jobs(
        package, target_catalog, tmp_path, monkeypatch):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.store.knowledge import KnowledgeRepository
    from papervault.store.repository import PostgresCatalogRepository

    module = replay_module()
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "replay"
    options = dict(source_generation="source", target_generation="replayed",
        catalog_connection=connection, catalog_schema=schema, apply=True)
    original = CatalogScheduler.run_round
    rounds = 0

    async def interrupt(scheduler, *args, **kwargs):
        nonlocal rounds
        rounds += 1
        if rounds == 2:
            raise OSError("before second admission")
        return await original(scheduler, *args, **kwargs)

    monkeypatch.setattr(CatalogScheduler, "run_round", interrupt)
    with pytest.raises(OSError, match="second admission"):
        await module.replay_recovery_set(path, target, **options)
    monkeypatch.setattr(CatalogScheduler, "run_round", original)
    gate = module.replay_capacity(target, connection, schema, physical=False)
    gate.soft_cap = 1
    with pytest.raises(RuntimeError, match="stopped without complete"):
        await module.replay_recovery_set(path, target, **options, capacity_gate=gate)
    k = KnowledgeRepository(PostgresCatalogRepository(connection, schema=schema))
    assert len(k._rows("knowledge_attempts")) == 1
    assert sorted(job["state"] for job in k.list_jobs()) == ["queued", "succeeded"]
    assert not (target / "replay-receipt.json").exists()


@pytest.mark.parametrize("missing", [None, "postgres", "graph"])
def test_physical_replay_factory_recovers_owned_backends_without_unfunded_creation(
        target_catalog, tmp_path, monkeypatch, missing):
    from contextlib import contextmanager, nullcontext
    from pathlib import Path
    from types import SimpleNamespace
    import os
    from psycopg import sql
    from psycopg.types.json import Jsonb
    from papervault.store import recovery, recovery_backends
    from papervault.store.repository import payload_hash
    from papervault.store.scheduler_capacity import CapacityAdmissionError

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import recovery_runtime
    import prepare_knowledge_backend

    # This standalone factory replaces process environment; keep the test local.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(recovery, "verify_recovery_set", lambda *args: {})
    monkeypatch.setattr(recovery_backends, "validate_postgres_target", lambda *args: None)
    monkeypatch.setattr(prepare_knowledge_backend, "preparation_plan", lambda **kwargs: {
        "tables": ["replay_docs"], "postgres": ["CREATE TABLE replay_docs(id text)"],
        "neo4j": ["CREATE INDEX IF NOT EXISTS FOR (n:PaperVaultCatalogEntity) ON (n.entity_id)"]})
    package = tmp_path / "physical-package"
    for relative, value in {"manifest.json": {}, "postgres/data.json": {}, "neo4j/data.json": {
            "source_uri": "bolt://127.0.0.2:7687", "database_identity": {"id": "source"}}}.items():
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value))

    binding = None

    class Result(list):
        def single(self):
            return self[0]

    class Graph:
        def session(self, **kwargs):
            return nullcontext(self)

        def begin_transaction(self):
            return nullcontext(self)

        def commit(self):
            pass

        def run(self, query, **kwargs):
            if query.startswith("CALL db.info"):
                return Result([{"id": "isolated-target"}])
            if query.startswith("MATCH (n:_PapervaultReplay)"):
                return Result([] if missing == "graph" else [{"binding": payload_hash(binding)}])
            if query.startswith("MATCH(n) RETURN count"):
                return Result([{"count": 0 if missing == "graph" else 1}])
            pytest.fail("unfunded graph allocation: " + query)

    class Denied:
        @contextmanager
        def intake(self):
            raise CapacityAdmissionError("capacity exhausted")
            yield

    connection, schema = target_catalog
    factory, identity = recovery_runtime.physical_replay_factory({}, {"postgres": {
        "connection": connection, "schema": schema}, "neo4j": {"driver": Graph(),
        "uri": "bolt://127.0.0.1:7687", "database": "isolated", "disposable": True}}, package, tmp_path / "target")
    binding = {"operation": "recovery-replay:test", "target": identity}
    if missing != "postgres":
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        connection.execute(sql.SQL("CREATE TABLE {}(id text)").format(sql.Identifier(schema, "replay_docs")))
        connection.execute(sql.SQL("CREATE TABLE {}(binding jsonb)").format(sql.Identifier(schema, "_papervault_replay")))
        connection.execute(sql.SQL("INSERT INTO {} VALUES(%s)").format(
            sql.Identifier(schema, "_papervault_replay")), (Jsonb(binding),))
    k = SimpleNamespace(repository=SimpleNamespace(fence_id=binding["operation"]))
    if missing is None:
        assert factory(k, capacity_gate=Denied()).repository is k
    else:
        with pytest.raises(CapacityAdmissionError):
            factory(k, capacity_gate=Denied())
    if missing == "postgres":
        assert connection.execute("SELECT to_regnamespace(%s) AS name", (schema,)).fetchone()["name"] is None


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
            previous = self.active
            self.active = True
            try:
                yield {}
            finally:
                self.active = previous

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
