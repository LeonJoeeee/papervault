"""Catalog work requires real, conservative filesystem capacity admission."""

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from papervault.knowledge.scheduler.catalog import CatalogScheduler
from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
from papervault.store.derivative_capacity import capacity_plan
from papervault.store.knowledge import ClaimRejected
from tests.knowledge.test_catalog_round import EmptyCatalog
from tests.store.test_knowledge_postgres import receipt as sealed_receipt


def make_gate(tmp_path, *, backend_bytes=0, soft_cap=10000, estimated_bytes=1000):
    from papervault.store.scheduler_capacity import SchedulerCapacity

    work = tmp_path / "knowledge"
    return SchedulerCapacity(
        work / "maintenance" / "capacity", derivative_roots=(work,),
        estimated_bytes=estimated_bytes, backend_usage=lambda physical: backend_bytes,
        soft_cap=soft_cap,
    )


def reserved(gate):
    return capacity_plan(gate.root, estimated_bytes=1, derivative_roots=gate.derivative_roots,
                         soft_cap=100000)["reserved_bytes"]


def test_scheduler_reservation_survives_owner_restart_until_reconciled(tmp_path):
    gate = make_gate(tmp_path)
    gate.acquire("job-one")
    claim = {"job_id": "job-one", "attempt_id": "attempt", "submission_id": "submission"}
    gate.bind("job-one", claim)
    assert reserved(gate) == 1000
    restarted = make_gate(tmp_path)
    with pytest.raises(ValueError, match="reconcil"):
        restarted.acquire("job-one")
    assert reserved(restarted) == 1000
    assert restarted.release("job-one", claim=claim, reconciled=True) is True
    assert reserved(restarted) == 0


def test_unowned_reservation_is_never_deleted_by_reconciliation(tmp_path):
    gate = make_gate(tmp_path)
    gate.root.mkdir(parents=True)
    foreign = gate.root / ".capacity-reservation-scheduler-foreign.reserve"
    foreign.write_bytes(b"foreign")
    assert gate.release("foreign", reconciled=True) is False
    assert foreign.read_bytes() == b"foreign"


def test_backend_usage_is_counted_before_any_reservation(tmp_path):
    gate = make_gate(tmp_path, backend_bytes=9500)
    with pytest.raises(ValueError, match="capacity"):
        gate.acquire("blocked")
    assert reserved(gate) == 0


def race_gate(tmp_path, usage):
    from papervault.store.scheduler_capacity import SchedulerCapacity

    return SchedulerCapacity(tmp_path / "capacity", derivative_roots=(tmp_path / "outputs",),
                             estimated_bytes=40, backend_usage=usage, soft_cap=100)


def attempt_admission(gate, kind):
    try:
        if kind == "acquire":
            gate.acquire("new-job")
        else:
            with gate.intake():
                pass
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("kind", ["acquire", "intake"])
def test_admission_census_cannot_race_newer_settlement(tmp_path, monkeypatch, kind):
    """A samples 40; B materializes 40 and settles; A must not fund another 40."""
    import fcntl
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, get_ident

    sampled, resume, settlement_progress = Event(), Event(), Event()
    settlement_thread = []
    backend = [40]
    owner = race_gate(tmp_path, lambda physical: backend[0])
    claim = {"job_id": "old-job", "attempt_id": "old-attempt", "submission_id": "old-submission"}
    owner.acquire("old-job")
    owner.bind("old-job", claim)
    protected = tmp_path / "protected-source"
    protected.write_bytes(b"source")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    current = outputs / "current-version"
    current.write_bytes(b"current")

    def read_old_census(physical):
        value = backend[0]
        sampled.set()
        assert resume.wait(5)
        return value

    # Observe a real blocking lock without timing sleeps. The old implementation
    # lets B finish; the fix must block B until A has accounted for its reservation.
    flock = fcntl.flock

    def observed_lock(fd, operation):
        if settlement_thread == [get_ident()] and operation == fcntl.LOCK_EX:
            try:
                return flock(fd, operation | fcntl.LOCK_NB)
            except BlockingIOError:
                settlement_progress.set()
        return flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", observed_lock)
    contender = race_gate(tmp_path, read_old_census)

    def settle():
        settlement_thread.append(get_ident())
        backend[0] = 80
        try:
            return owner.release("old-job", claim=claim, reconciled=True)
        finally:
            settlement_progress.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        admission = pool.submit(attempt_admission, contender, kind)
        try:
            assert sampled.wait(5)
            settlement = pool.submit(settle)
            assert settlement_progress.wait(5)
        finally:
            resume.set()
        admitted = admission.result(timeout=5)
        assert settlement.result(timeout=5) is True
    assert admitted is False, "stale census admitted backend 80 + new budget 40 over cap 100"
    restarted = race_gate(tmp_path, lambda physical: backend[0])
    assert attempt_admission(restarted, kind) is False
    assert reserved(restarted) == 0
    assert protected.read_bytes() == b"source"
    assert current.read_bytes() == b"current"


@pytest.mark.parametrize("kind", ["acquire", "intake"])
@pytest.mark.parametrize("settlement_first", [False, True])
def test_ordered_settlement_and_admission_remain_funded(tmp_path, kind, settlement_first):
    backend = [40]
    gate = race_gate(tmp_path, lambda physical: backend[0])
    claim = {"job_id": "old-job", "attempt_id": "attempt", "submission_id": "submission"}
    gate.acquire("old-job")
    gate.bind("old-job", claim)
    if not settlement_first:
        assert attempt_admission(race_gate(tmp_path, lambda physical: backend[0]), kind) is False
        assert reserved(gate) == 40
    backend[0] = 80
    assert gate.release("old-job", claim=claim, reconciled=True)
    restarted = race_gate(tmp_path, lambda physical: backend[0])
    assert attempt_admission(restarted, kind) is False
    assert reserved(restarted) == 0


class OpenBreaker:
    def allows(self):
        return True


class WorkCatalog(EmptyCatalog):
    def __init__(self, store, *, candidate=False, window=True, claimed=False):
        super().__init__()
        self.artifact = store.capture(b"physical fulltext", role="fulltext")
        self.job = {"job_id": "job", "unit_id": "unit", "generation_id": "generation",
                    "recipe_digest": "recipe", "consumed_input_revision": "input",
                    "state": "queued", "revision": 1, "current_attempt_id": None}
        self.generation = {"state": "candidate" if candidate else "active", "manifest_revision": 1}
        self.window = window
        self.claim = None
        self.receipt = None
        self.operations = {}
        self.commands = []
        if claimed:
            self.command("claim", "initial-claim", {})

    def get_generation(self, generation_id):
        return self.generation

    def get_active_maintenance_window(self):
        return {"window_id": "monthly"} if self.window else None

    def list_pending_repairs(self):
        return []

    def get_active_generation(self):
        return None

    def list_admission_jobs(self, scope=None):
        return [self.job] if self.job["state"] == "queued" else []

    def list_outbox(self, limit=100, kind=None):
        if kind == "submission" and self.claim and not self.operations.get("ack:submission"):
            return [{"outbox_id": "submission", "kind": "submission", "payload": self.claim}]
        return []

    def get_job(self, job_id):
        return self.job

    def get_input(self, revision):
        return {"manifest": {"tier": "fulltext", "artifacts": [self.artifact]}}

    def get_attempt(self, attempt_id):
        return {**self.claim, "outcome": None, "sealed": self.receipt is not None,
                "receipt_id": self.receipt["receipt_id"] if self.receipt else None,
                "lease_expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()}

    def get_submission(self, submission_id):
        return {"claim": self.claim, "receipt": self.receipt}

    def get_unit(self, unit_id):
        return {"unit_id": "unit", "kind": "paper", "record_key": "paper"}

    def get_section(self, *args):
        return None

    def get_recipe(self, digest):
        return {"output_contract": {"fulltext": {"required": ["graph"], "not_applicable": []}}}

    def get_receipt(self, receipt_id):
        return self.receipt if self.receipt and self.receipt["receipt_id"] == receipt_id else None

    def get_selection(self, *args):
        return None

    def get_operation(self, operation_id):
        return self.operations.get(operation_id)

    def command(self, action, operation_id, payload):
        self.commands.append(action)
        if action == "claim":
            self.claim = {key: self.job[key] for key in (
                "job_id", "unit_id", "generation_id", "recipe_digest", "consumed_input_revision")}
            self.claim.update(attempt_id="attempt", submission_id="submission", namespace="delta",
                authority_epoch=1, fencing_token=1, lease_id="lease", aggregate_base_manifest_revision=1,
                base_aggregate_layers=[], base_selected_unit_receipts={}, retired_receipt_ids=[])
            self.job.update(state="running", current_attempt_id="attempt", revision=2)
            result = self.claim
        elif action == "complete":
            self.job.update(state="succeeded", revision=3)
            result = {"revision": 3}
        else:
            result = {"ok": True}
        self.operations[operation_id] = result
        return result


class WorkAdapter:
    def __init__(self, repository, gate, *, unknown=False):
        self.repository, self.gate, self.unknown = repository, gate, unknown
        self.executed = 0
        self.found = None

    async def lookup_submission(self, submission_id):
        return self.found

    async def submit(self, claim, *args, **kwargs):
        assert reserved(self.gate) == 1000
        self.executed += 1
        if self.unknown:
            raise TimeoutError("external outcome unknown")
        self.found = sealed_receipt(claim)
        self.repository.receipt = self.found
        return self.found

    async def collect_observations(self, receipt):
        return []


def scheduler_fixture(tmp_path, *, backend_bytes=0, candidate=False, window=True, claimed=False,
                      unknown=False):
    gate = make_gate(tmp_path, backend_bytes=backend_bytes)
    store = ImmutableInputStore(tmp_path / "knowledge" / "catalog_inputs")
    repository = WorkCatalog(store, candidate=candidate, window=window, claimed=claimed)
    adapter = WorkAdapter(repository, gate, unknown=unknown)
    scheduler = CatalogScheduler(repository, adapter, store, breaker=OpenBreaker(), capacity_gate=gate)
    return scheduler, repository, adapter, gate


@pytest.mark.asyncio
async def test_capacity_exhaustion_prevents_claim_and_execution(tmp_path):
    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path, backend_bytes=10000)
    result = await scheduler.run_round()
    assert result["claimed"] == result["delivered"] == 0
    assert result["capacity_blocked"] >= 1
    assert "claim" not in repository.commands
    assert adapter.executed == 0
    assert reserved(gate) == 0


@pytest.mark.asyncio
async def test_execution_holds_reservation_until_verified_receipt(tmp_path):
    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path)
    result = await scheduler.run_round()
    assert result["claimed"] == result["delivered"] == result["published"] == 1
    assert adapter.executed == 1
    assert reserved(gate) == 0


@pytest.mark.asyncio
async def test_unknown_execution_outcome_retains_reservation_until_receipt_recovery(tmp_path):
    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path, unknown=True)
    first = await scheduler.run_round()
    assert first["discrepancies"] == 1
    assert reserved(gate) == 1000
    # Recovery after restart must not require spare capacity or another execution.
    recovered_gate = make_gate(tmp_path, backend_bytes=10000)
    repository.receipt = sealed_receipt(repository.claim)
    recovered = CatalogScheduler(repository, adapter, scheduler.input_store,
                                 breaker=OpenBreaker(), capacity_gate=recovered_gate)
    second = await recovered.run_round()
    assert second["published"] == 1
    assert adapter.executed == 1
    assert reserved(recovered_gate) == 0


@pytest.mark.asyncio
async def test_existing_unfunded_submission_cannot_execute_at_exhaustion(tmp_path):
    scheduler, repository, adapter, gate = scheduler_fixture(
        tmp_path, backend_bytes=10000, claimed=True,
    )
    result = await scheduler.run_round()
    assert adapter.executed == 0
    assert result["capacity_blocked"] >= 1
    assert reserved(gate) == 0


@pytest.mark.asyncio
async def test_candidate_submission_does_not_execute_outside_monthly_window(tmp_path):
    scheduler, repository, adapter, gate = scheduler_fixture(
        tmp_path, candidate=True, window=False, claimed=True,
    )
    await scheduler.run_round()
    assert adapter.executed == 0
    assert reserved(gate) == 0


def write_census(tmp_path, *, age=0, mode=0o600, scope=None, backend_bytes=23):
    path = tmp_path / "usage.json"
    path.write_text(json.dumps({"backend_bytes": backend_bytes,
        "measured_at": (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat(),
        "scope": scope or {"vault_root": str(tmp_path / "vault"),
                           "knowledge_root": str(tmp_path / "knowledge")}}))
    path.chmod(mode)
    return path


def test_scoped_fresh_protected_backend_census_is_usable(tmp_path):
    from papervault.store.scheduler_capacity import measured_backend_bytes

    path = write_census(tmp_path)
    assert measured_backend_bytes(path, vault_root=tmp_path / "vault",
                                  knowledge_root=tmp_path / "knowledge") == 23


@pytest.mark.parametrize("fault", ["missing", "stale", "future", "public", "scope", "negative", "symlink"])
def test_unqualified_backend_census_fails_closed(tmp_path, fault):
    from papervault.store.scheduler_capacity import measured_backend_bytes

    path = write_census(tmp_path, age=301 if fault == "stale" else -10 if fault == "future" else 0,
        mode=0o644 if fault == "public" else 0o600,
        scope={"vault_root": "/wrong", "knowledge_root": "/wrong"} if fault == "scope" else None,
        backend_bytes=-1 if fault == "negative" else 23)
    if fault == "missing":
        path.unlink()
    elif fault == "symlink":
        alias = tmp_path / "alias.json"
        alias.symlink_to(path)
        path = alias
    with pytest.raises(ValueError, match="account|census|usage"):
        measured_backend_bytes(path, vault_root=tmp_path / "vault", knowledge_root=tmp_path / "knowledge")


def runtime_service(tmp_path, monkeypatch, repository=None):
    from papervault import config
    from papervault.knowledge import config as knowledge_config
    from papervault.knowledge.scheduler import catalog_service
    from papervault.knowledge.store import catalog_adapter
    from papervault.store import knowledge_runtime

    repository = repository or SimpleNamespace()
    monkeypatch.setattr(knowledge_runtime, "assert_knowledge_authority", lambda mode: None)
    monkeypatch.setattr(knowledge_runtime, "get_knowledge_repository", lambda: repository)
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path / "vault")
    monkeypatch.setattr(knowledge_config, "CONFIG", replace(knowledge_config.CONFIG,
        lightrag=replace(knowledge_config.CONFIG.lightrag, working_dir=str(tmp_path / "knowledge"))))
    monkeypatch.setattr(catalog_adapter, "CatalogLightRAGAdapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(catalog_service, "_service", None)
    monkeypatch.delenv("PAPERVAULT_DERIVATIVE_USAGE_FILE", raising=False)
    return catalog_service.get_catalog_service()


def test_runtime_wires_real_gate_and_missing_physical_census_fails_closed(tmp_path, monkeypatch):
    service = runtime_service(tmp_path, monkeypatch)
    gate = service.scheduler.capacity_gate
    assert gate.root == tmp_path / "knowledge" / "maintenance" / "capacity"
    assert gate.derivative_roots == (tmp_path / "vault" / "extracts", tmp_path / "knowledge")
    with pytest.raises(ValueError, match="account|census|usage"):
        gate.acquire("job", physical=True)
    assert reserved(gate) == 0


def test_held_reservation_cannot_be_reused_after_foreign_replacement(tmp_path):
    gate = make_gate(tmp_path)
    gate.acquire("job-one")
    reservation = next(gate.root.glob(".capacity-reservation-*.reserve"))
    reservation.unlink()
    reservation.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="replaced|identity|capacity"):
        gate.acquire("job-one")
    assert reservation.read_bytes() == b"foreign"


@pytest.mark.asyncio
async def test_rejected_claim_releases_unexecuted_reservation(tmp_path):
    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path)
    original = repository.command

    def reject(action, *args):
        if action == "claim":
            raise ClaimRejected("concurrent claimant won")
        return original(action, *args)

    repository.command = reject
    result = await scheduler.run_round()
    assert result["claimed"] == result["delivered"] == 0
    assert adapter.executed == 0
    assert reserved(gate) == 0


@pytest.mark.asyncio
async def test_unrecognized_capacity_metadata_preserves_charge_without_blocking_publication(tmp_path):
    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path, unknown=True)
    await scheduler.run_round()
    next(gate.root.glob(".scheduler-*.json")).write_text("[]")
    repository.receipt = sealed_receipt(repository.claim)
    result = await scheduler.run_round()
    assert result["published"] == 1
    assert reserved(gate) == 1000
    assert "capacity_reconciliation_blocked" in result


def test_runtime_metadata_counts_fresh_global_census_and_catalog_relations(tmp_path, monkeypatch):
    class Connection:
        def execute(self, query, parameters):
            assert parameters == ("catalog_fixture",)
            return SimpleNamespace(fetchone=lambda: {"bytes": 17})

    repository = SimpleNamespace(connection=Connection(), repository=SimpleNamespace(schema="catalog_fixture"))
    service = runtime_service(tmp_path, monkeypatch, repository)
    monkeypatch.setenv("PAPERVAULT_DERIVATIVE_USAGE_FILE", str(write_census(tmp_path)))
    monkeypatch.setenv("PAPERVAULT_CATALOG_EXECUTION_BYTES", "1000")
    plan = service.scheduler.capacity_gate.acquire("metadata", physical=False)
    assert plan["used_bytes"] == 40
    assert service.scheduler.capacity_gate.release("metadata") is True


@pytest.mark.asyncio
async def test_intake_exhaustion_skips_capture_but_recovers_sealed_receipt(tmp_path):
    from papervault.knowledge.scheduler.catalog_service import CatalogService

    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path, claimed=True)
    gate.acquire("job")
    gate.bind("job", repository.claim)
    gate.backend_usage = lambda physical: 10000
    repository.receipt = sealed_receipt(repository.claim)
    intake = SimpleNamespace(scan=lambda **kwargs: pytest.fail("unfunded input capture executed"))
    result = await CatalogService(repository, intake, scheduler).run_round()
    assert result["intake"]["capacity_blocked"] is True
    assert result["published"] == 1
    assert adapter.executed == 0
    assert reserved(gate) == 0


@pytest.mark.asyncio
async def test_funded_intake_holds_real_capacity_through_capture_and_recovery(tmp_path):
    from papervault.knowledge.scheduler.catalog_service import CatalogService

    gate = make_gate(tmp_path)
    observed = []

    def capture(**kwargs):
        observed.append(reserved(gate))
        return {"captured": 1}

    async def run_round(**kwargs):
        assert reserved(gate) == 0
        return {"published": 0}

    scheduler = SimpleNamespace(capacity_gate=gate, run_round=run_round)
    service = CatalogService(SimpleNamespace(list_outbox=lambda **kwargs: []),
                             SimpleNamespace(scan=capture), scheduler)
    result = await service.run_round()
    assert observed == [1000]
    assert result["intake"]["captured"] == 1


@pytest.mark.asyncio
async def test_direct_operator_capture_pauses_before_unfunded_write(tmp_path, monkeypatch):
    from papervault.knowledge.scheduler import catalog_service
    from papervault.knowledge.store.build_breaker import BuildPausedError

    gate = make_gate(tmp_path, backend_bytes=10000)
    service = SimpleNamespace(scheduler=SimpleNamespace(capacity_gate=gate), intake=SimpleNamespace(
        capture_document=lambda *args, **kwargs: pytest.fail("unfunded operator capture")))
    monkeypatch.setattr(catalog_service, "get_catalog_service", lambda: service)
    with pytest.raises(BuildPausedError, match="capacity"):
        await catalog_service.capture_operator_document("textbook", "textbook:Book", "book.md")
    assert reserved(gate) == 0


@pytest.mark.asyncio
async def test_operator_capture_reuses_outer_drain_reservation(tmp_path, monkeypatch):
    from papervault.knowledge.scheduler import catalog_service

    gate = make_gate(tmp_path)

    def capture(*args, **kwargs):
        assert reserved(gate) == 1000
        return {"catalog_queued": True}

    service = SimpleNamespace(scheduler=SimpleNamespace(capacity_gate=gate),
                              intake=SimpleNamespace(capture_document=capture))
    monkeypatch.setattr(catalog_service, "get_catalog_service", lambda: service)
    with gate.intake():
        result = await catalog_service.capture_operator_document("textbook", "textbook:Book", "book.md")
    assert result["catalog_queued"] is True
    assert reserved(gate) == 0


@pytest.mark.asyncio
async def test_loop_exhaustion_keeps_pending_documents_and_continues_recovery(tmp_path, monkeypatch):
    import asyncio
    from papervault.knowledge.scheduler import catalog_service, opdoc_pickup
    from papervault.knowledge.store import build_breaker

    gate = make_gate(tmp_path, backend_bytes=10000)
    pending = tmp_path / "pending.pdf"
    pending.write_bytes(b"protected source")
    recovered = []

    async def recover():
        recovered.append(True)

    async def unfunded(*args):
        pytest.fail("unfunded operator drain/OCR")

    async def stop_after_round(*args):
        raise asyncio.CancelledError

    service = SimpleNamespace(scheduler=SimpleNamespace(capacity_gate=gate), run_round=recover)
    monkeypatch.setattr(catalog_service, "get_catalog_service", lambda: service)
    monkeypatch.setattr(opdoc_pickup, "drain_pending", unfunded)
    monkeypatch.setattr(build_breaker, "get_breaker", lambda: OpenBreaker())
    monkeypatch.setattr(catalog_service.asyncio, "sleep", stop_after_round)
    with pytest.raises(asyncio.CancelledError):
        await catalog_service.run_catalog_loop()
    assert recovered == [True]
    assert pending.read_bytes() == b"protected source"


def test_old_owner_cannot_release_replacement_attempt_reservation(tmp_path):
    gate = make_gate(tmp_path)
    old = {"job_id": "job", "attempt_id": "attempt-old", "submission_id": "submission-old"}
    new = {"job_id": "job", "attempt_id": "attempt-new", "submission_id": "submission-new"}
    gate.acquire("job")
    gate.bind("job", old)
    restarted = make_gate(tmp_path)
    assert restarted.release("job", claim=old, reconciled=True) is True
    restarted.acquire("job", reuse=False)
    restarted.bind("job", new)
    with pytest.raises(ValueError, match="owner|replaced|identity|attempt"):
        gate.release("job", claim=old, reconciled=True)
    assert reserved(restarted) == 1000


def test_completed_external_growth_stays_charged_until_new_census(tmp_path):
    gate = make_gate(tmp_path, estimated_bytes=4000, soft_cap=10000)
    census = {"backend_bytes": 4000, "measured_at": datetime.now(timezone.utc).isoformat()}
    gate.backend_usage = lambda physical: dict(census)
    first = {"job_id": "first", "attempt_id": "attempt", "submission_id": "submission"}
    gate.acquire("first")
    gate.bind("first", first)
    assert gate.release("first", claim=first, reconciled=True) is False
    assert reserved(gate) == 4000
    with pytest.raises(ValueError, match="capacity"):
        gate.acquire("second")
    # An updated measurement now includes the materialized 4000-byte output.
    census.update(backend_bytes=8000, measured_at=datetime.now(timezone.utc).isoformat())
    with pytest.raises(ValueError, match="capacity"):
        gate.acquire("second")
    assert reserved(gate) == 0


@pytest.mark.parametrize("sweeper", ["intake", "acquire"])
def test_owner_can_readmit_after_another_worker_reconciles_spent_charge(tmp_path, sweeper):
    owner = make_gate(tmp_path)
    census = {"backend_bytes": 1000, "measured_at": datetime.now(timezone.utc).isoformat()}
    owner.backend_usage = lambda physical: dict(census)
    old = {"job_id": "job", "attempt_id": "old", "submission_id": "old-submission"}
    owner.acquire("job")
    owner.bind("job", old)
    assert owner.release("job", claim=old, reconciled=True) is False
    assert reserved(owner) == 1000
    census.update(backend_bytes=2000, measured_at=datetime.now(timezone.utc).isoformat())
    other = make_gate(tmp_path)
    other.backend_usage = lambda physical: dict(census)
    assert attempt_admission(other, sweeper) is True
    if sweeper == "acquire":
        assert other.release("new-job") is True
    assert reserved(owner) == 0
    assert owner.acquire("job", reuse=False)["admitted"] is True
    new = {"job_id": "job", "attempt_id": "new", "submission_id": "new-submission"}
    owner.bind("job", new)
    with pytest.raises(ValueError, match="identity"):
        owner.release("job", claim=old, reconciled=True)
    assert reserved(owner) == 1000


@pytest.mark.asyncio
async def test_no_attempt_claim_result_releases_unexecuted_capacity(tmp_path):
    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path)
    original = repository.command

    def no_attempt(action, *args):
        return {"attempt_id": None, "claimed": False} if action == "claim" else original(action, *args)

    repository.command = no_attempt
    result = await scheduler.run_round()
    assert result["claimed"] == result["delivered"] == 0
    assert reserved(gate) == 0


@pytest.mark.asyncio
async def test_cancellation_keeps_intake_worker_funded_until_it_stops(tmp_path):
    import asyncio
    import threading
    from papervault.knowledge.scheduler.catalog_service import CatalogService

    gate = make_gate(tmp_path)
    started, release = threading.Event(), threading.Event()
    observed = []

    def capture(**kwargs):
        started.set()
        assert release.wait(5)
        observed.append(reserved(gate))
        return {"captured": 1}

    async def run_round(**kwargs):
        return {}

    service = CatalogService(SimpleNamespace(list_outbox=lambda **kwargs: []),
        SimpleNamespace(scan=capture), SimpleNamespace(capacity_gate=gate, run_round=run_round))
    task = asyncio.create_task(service.run_round())
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert reserved(gate) == 1000
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert observed == [1000]


@pytest.mark.asyncio
async def test_unknown_attempt_is_not_reexecuted_with_its_held_reservation(tmp_path):
    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path, unknown=True)
    await scheduler.run_round()
    result = await scheduler.run_round()
    assert adapter.executed == 1
    assert result["capacity_blocked"] >= 1
    assert reserved(gate) == 1000


@pytest.mark.asyncio
async def test_receipt_rejected_by_catalog_does_not_release_capacity(tmp_path):
    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path)
    original = repository.command

    def refuse(action, *args):
        if action == "complete":
            raise ClaimRejected("namespace was never sealed by catalog")
        return original(action, *args)

    repository.command = refuse
    repository.get_receipt = lambda receipt_id: None  # Adapter's structure was never registered/sealed.
    result = await scheduler.run_round()
    assert result["published"] == 0
    assert reserved(gate) == 1000


@pytest.mark.asyncio
async def test_candidate_execution_is_cancelled_at_window_deadline(tmp_path):
    import asyncio

    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path, candidate=True)
    deadline = datetime.now(timezone.utc) + timedelta(milliseconds=50)
    repository.get_active_maintenance_window = lambda: {"window_id": "monthly", "ends_at": deadline.isoformat()}
    cancelled = []

    async def slow(claim, *args, **kwargs):
        adapter.executed += 1
        try:
            await asyncio.sleep(10)
            pytest.fail("candidate executed beyond window")
        finally:
            cancelled.append(True)

    adapter.submit = slow
    result = await asyncio.wait_for(scheduler.run_round(), 1)
    await asyncio.sleep(0)
    assert adapter.executed == 1
    assert cancelled == [True]
    assert result["published"] == 0
    assert reserved(gate) == 1000
    await scheduler.run_round()
    assert adapter.executed == 1


def test_metadata_sample_census_cannot_fund_new_physical_job(tmp_path):
    gate = make_gate(tmp_path)
    gate.backend_usage = lambda physical: {"backend_bytes": 0,
        "measured_at": datetime.now(timezone.utc).isoformat(), "metadata_sample": True}
    with pytest.raises(ValueError, match="account"):
        gate.acquire("physical", physical=True)
    assert reserved(gate) == 0


@pytest.mark.asyncio
async def test_eighty_percent_alert_is_visible_in_round_and_logs(tmp_path, caplog):
    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path, backend_bytes=7000)
    result = await scheduler.run_round()
    assert result["capacity_alert"] is True
    assert result["capacity"]["projected_bytes"] >= 8000
    assert any("capacity" in record.message and "80%" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_authoritative_completion_settles_capacity_even_when_publication_needs_rebase(tmp_path):
    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path)
    original = repository.command

    def refuse_publication(action, *args):
        if action == "publish_derivation":
            raise ClaimRejected("stale aggregate base requires rebase")
        return original(action, *args)

    repository.command = refuse_publication
    result = await scheduler.run_round()
    assert result["completed"] == 1
    assert result["published"] == 0
    assert reserved(gate) == 0


@pytest.mark.asyncio
async def test_cancellation_resistant_active_submit_cannot_be_reexecuted(tmp_path):
    import asyncio

    scheduler, repository, adapter, gate = scheduler_fixture(tmp_path)
    started, finish = asyncio.Event(), asyncio.Event()

    async def resistant(claim, *args, **kwargs):
        adapter.executed += 1
        started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            await finish.wait()
        return sealed_receipt(claim)

    adapter.submit = resistant
    operation = asyncio.create_task(scheduler.run_round())
    await started.wait()
    operation.cancel()
    try:
        for _ in range(5):
            await asyncio.sleep(0)
        assert operation.done()
        await scheduler.run_round()
        assert adapter.executed == 1
        assert reserved(gate) == 1000
    finally:
        finish.set()
        try:
            await operation
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)


def test_integer_measurement_keeps_unsettled_output_charged(tmp_path):
    import threading

    gate = make_gate(tmp_path)
    claim = {"job_id": "job", "attempt_id": "attempt", "submission_id": "submission"}
    gate.acquire("job")
    gate.bind("job", claim)
    before = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    measuring, finish = threading.Event(), threading.Event()
    observed, errors = [], []

    def measurement(physical):
        if threading.current_thread() is threading.main_thread():
            return {"backend_bytes": 0, "measured_at": before}
        measuring.set()
        assert finish.wait(5)
        return 0

    gate.backend_usage = measurement

    def capture():
        try:
            with gate.intake():
                observed.append(reserved(gate))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=capture)
    worker.start()
    try:
        assert measuring.wait(5)
    finally:
        # Admission now serializes sampling with settlement. Complete this census
        # before settling; an older later census still cannot clear the charge.
        finish.set()
        worker.join(5)
    assert not errors
    assert observed == [2000]
    assert gate.release("job", claim=claim, reconciled=True) is False
    assert reserved(gate) == 1000


def test_metadata_only_census_cannot_reconcile_physical_spent_charge(tmp_path):
    gate = make_gate(tmp_path)
    stamp = datetime.now(timezone.utc).isoformat()
    gate.backend_usage = lambda physical: {"backend_bytes": 0, "measured_at": stamp}
    claim = {"job_id": "job", "attempt_id": "attempt", "submission_id": "submission"}
    gate.acquire("job")
    gate.bind("job", claim)
    assert gate.release("job", claim=claim, reconciled=True) is False
    gate.backend_usage = lambda physical: {"backend_bytes": 0,
        "measured_at": datetime.now(timezone.utc).isoformat(), "metadata_sample": True}
    with gate.intake():
        assert reserved(gate) == 2000
    assert reserved(gate) == 1000
