"""Controlled build pauses settle claims without spending the failure budget."""
from datetime import datetime, timezone

import psycopg
from psycopg import sql
import pytest

from tests.store.test_knowledge_postgres import (
    _NoBackend,
    _OpenBuildGate,
    claim,
    knowledge as knowledge,
    prepare,
    prepare_fulltext,
    seal,
)


def fail_once(k, job):
    admitted = claim(k, job, "first-failure-claim")
    k.command("fail", "first-failure", {**admitted, "expected_revision": admitted["revision"],
        "error": {"kind": "verified_failure"}})
    k.command("acknowledge_outbox", "first-failure-ack", {"outbox_id": "submit:" + admitted["submission_id"]})
    return k.get_job(job["job_id"])


def defer(k, admitted, operation="defer"):
    return k.command("defer_attempt", operation, {**admitted,
        "expected_revision": k.get_job(admitted["job_id"])["revision"],
        "reason": {"kind": "BuildPausedError"}})


def test_defer_atomically_closes_delivery_preserves_streak_and_fences_namespace(knowledge, monkeypatch):
    from papervault.store.knowledge import ClaimRejected

    monkeypatch.setenv("KS_REDISTILL_MAX_ATTEMPTS", "2")
    k = knowledge
    job = fail_once(k, prepare(k))
    admitted = claim(k, job, "pause-claim")
    k.command("accept_submission", "accepted", {"claim": admitted})
    before = k.get_reducer("paper:A", "input1")
    payload = {**admitted, "expected_revision": admitted["revision"],
               "reason": {"kind": "BuildPausedError"}}
    result = k.command("defer_attempt", "defer", payload)
    assert k.command("defer_attempt", "defer", payload) == result
    queued = k.get_job(job["job_id"])
    assert queued["state"] == "queued" and queued["lease_id"] is None
    assert k.get_reducer("paper:A", "input1") == before
    attempt = k.get_attempt(admitted["attempt_id"])
    assert attempt["outcome"] == "deferred" and attempt["finished_at"] is not None
    assert attempt["accepted"] and attempt["cost_state"] == "unknown"
    assert attempt["actual_cost"] is None
    assert k.get_authority_state()["budget_reserved"] == 2
    assert k.get_pin("attempt-pin:" + admitted["attempt_id"])["kind"] == "recovery"
    assert admitted["namespace"] in k.protected_namespaces()
    event = k._rows("knowledge_attempt_events", attempt_id=admitted["attempt_id"])[-1]
    assert event["event"] == "deferred" and "failure_streak_event" not in event["detail"]
    assert not k.list_outbox(kind="submission")
    assert [row["job_id"] for row in k.list_admission_jobs()] == [job["job_id"]]
    with pytest.raises(ClaimRejected):
        k.command("record_tombstone", "stale-write", {"claim": admitted, "role": "graph", "output_id": "stale"})
    next_claim = claim(k, queued, "after-pause")
    assert next_claim["fencing_token"] > admitted["fencing_token"]
    assert next_claim["namespace"] != admitted["namespace"]
    assert k.get_reducer("paper:A", "input1") == before


@pytest.mark.parametrize("guard", ["sealed", "stale_revision", "stale_fence"])
def test_defer_rejects_sealed_or_stale_claim_without_partial_change(knowledge, guard):
    from papervault.store.knowledge import ClaimRejected

    k = knowledge
    admitted = claim(k, prepare(k))
    if guard == "sealed":
        seal(k, admitted)
    changed = dict(admitted)
    if guard == "stale_revision":
        changed["revision"] -= 1
    if guard == "stale_fence":
        changed["fencing_token"] += 1
    with pytest.raises(ClaimRejected):
        k.command("defer_attempt", "rejected-defer", {**changed,
            "expected_revision": changed["revision"], "reason": {"kind": "BuildPausedError"}})
    assert k.get_job(admitted["job_id"])["state"] == "running"
    assert k.get_attempt(admitted["attempt_id"])["outcome"] is None
    assert len(k.list_outbox(kind="submission")) == 1


def test_defer_transaction_rolls_back_if_delivery_acknowledgement_fails(knowledge):
    k = knowledge
    admitted = claim(k, prepare(k))
    r = k.repository
    r.connection.execute(sql.SQL("""CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'injected defer acknowledgement failure'; END $$""").format(
            sql.Identifier(r.schema, "fail_defer_ack")))
    r.connection.execute(sql.SQL("CREATE TRIGGER fail_defer_ack BEFORE UPDATE ON {} "
        "FOR EACH STATEMENT EXECUTE FUNCTION {}()").format(r._table("knowledge_outbox"),
            sql.Identifier(r.schema, "fail_defer_ack")))
    with pytest.raises(psycopg.errors.RaiseException, match="injected defer acknowledgement"):
        defer(k, admitted)
    assert k.get_job(admitted["job_id"])["state"] == "running"
    assert k.get_attempt(admitted["attempt_id"])["outcome"] is None
    assert k.get_pin("attempt-pin:" + admitted["attempt_id"])["kind"] == "staging"
    assert len(k.list_outbox(kind="submission")) == 1
    assert k.get_operation("defer") is None
    assert [event["event"] for event in k._rows("knowledge_attempt_events")] == ["claimed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("pause", ["peak", "cooldown"])
async def test_delivery_pause_does_not_park_penultimate_failure_and_can_resume(knowledge, tmp_path, monkeypatch, pause):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.store.build_breaker import BuildBreaker, BuildSchedule
    from papervault.store.knowledge import KnownExecutionFailure

    monkeypatch.setenv("KS_REDISTILL_MAX_ATTEMPTS", "2")
    k = knowledge
    store, initial = prepare_fulltext(k, tmp_path)
    job = fail_once(k, initial)
    admitted = claim(k, job, "pause-claim")
    before = k.get_reducer(job["unit_id"], job["consumed_input_revision"])
    schedule = BuildSchedule(enabled=pause == "peak", tz=timezone.utc,
        now=lambda: datetime(2026, 10, 21, 10, tzinfo=timezone.utc))
    breaker = BuildBreaker(threshold=1, cooldown_s=600, clock=lambda: 0, schedule=schedule)
    if pause == "cooldown":
        breaker.record_failure()

    class PausingBackend(_NoBackend):
        async def submit(self, claim, text, file_path, *, achieved_tier):
            # The real adapter accepts before the build gate can pause a call.
            k.command("accept_submission", "accept:" + claim["submission_id"], {"claim": claim})
            breaker.before_call()

    scheduler = CatalogScheduler(k, PausingBackend(), store, breaker=_OpenBuildGate())
    summary = {"observations": 0, "discrepancies": 0, "delivered": 0, "completed": 0,
               "published": 0, "repairs": 0}
    row = next(row for row in k.list_outbox(kind="submission") if row["submission_id"] == admitted["submission_id"])
    await scheduler._deliver(row, summary)
    paused_job = k.get_job(job["job_id"])
    assert paused_job["state"] == "queued"
    assert paused_job["attempts"] == 2
    assert k.get_attempt(admitted["attempt_id"])["outcome"] == "deferred"
    assert k.get_reducer(job["unit_id"], job["consumed_input_revision"]) == before
    assert summary["paused"] is True and summary["discrepancies"] == 0
    assert not k.list_outbox(kind="submission")
    assert not k._rows("knowledge_discrepancies")
    assert not any(event["detail"].get("failure_streak_event") for event in
                   k._rows("knowledge_attempt_events", attempt_id=admitted["attempt_id"]))
    # A stale copy of the delivery cannot create work or spend the failure budget.
    await scheduler._deliver(row, summary)
    assert k.get_job(job["job_id"])["attempts"] == 2
    assert k.get_reducer(job["unit_id"], job["consumed_input_revision"]) == before

    class FailingBackend(_NoBackend):
        async def submit(self, claim, text, file_path, *, achieved_tier):
            raise KnownExecutionFailure("actual failure after resumption")

    scheduler.adapter = FailingBackend()
    resumed = await scheduler.run_round(only_keys={"A"})
    assert resumed["claimed"] == 1
    reducer = k.get_reducer(job["unit_id"], job["consumed_input_revision"])
    assert reducer["failure_streak"] == 2 and reducer["parked"] is True
    assert k.get_job(job["job_id"])["attempts"] == 3
    assert k.get_job(job["job_id"])["state"] == "failed"


@pytest.mark.asyncio
async def test_unknown_delivery_retries_stay_one_attempt_and_real_expiry_still_parks(knowledge, tmp_path, monkeypatch):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler

    monkeypatch.setenv("KS_REDISTILL_MAX_ATTEMPTS", "2")
    k = knowledge
    store, initial = prepare_fulltext(k, tmp_path)
    job = fail_once(k, initial)
    admitted = claim(k, job, "ambiguous-claim")

    class AmbiguousBackend(_NoBackend):
        async def submit(self, claim, text, file_path, *, achieved_tier):
            raise OSError("external outcome unknown")

    scheduler = CatalogScheduler(k, AmbiguousBackend(), store, breaker=_OpenBuildGate())
    for _ in range(2):
        await scheduler.run_round(only_keys={"A"})
    assert k.get_job(job["job_id"])["attempts"] == 2
    assert k.get_reducer(job["unit_id"], job["consumed_input_revision"])["failure_streak"] == 1
    k.connection.execute(sql.SQL("UPDATE {} SET lease_expires_at=clock_timestamp()-interval '1 second' "
        "WHERE attempt_id=%s").format(k.repository._table("knowledge_attempts")), (admitted["attempt_id"],))
    await scheduler.run_round(only_keys={"A"})
    assert k.get_attempt(admitted["attempt_id"])["outcome"] == "abandoned"
    reducer = k.get_reducer(job["unit_id"], job["consumed_input_revision"])
    assert reducer["failure_streak"] == 2 and reducer["parked"] is True
    event = k._rows("knowledge_attempt_events", attempt_id=admitted["attempt_id"])[-1]
    assert event["event"] == "abandoned" and event["detail"]["failure_streak_event"] is True


@pytest.mark.asyncio
async def test_retirement_failure_keeps_delivery_recoverable_and_requests_published_graph_repair(knowledge, tmp_path):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.store.knowledge import RetirementFailure

    k = knowledge
    store, job = prepare_fulltext(k, tmp_path)
    prior = claim(k, job, "prior-publication-claim")
    rec = seal(k, prior)
    published = k.command("publish_derivation", "prior-publication", {**prior, "receipt": rec,
        "expected_revision": prior["revision"], "expected_selection_revision": 0, "select": True})
    # A new recipe requires retiring the selected contribution in a fresh private delta.
    recipe = k.command("register_recipe", "replacement-recipe", {"recipe": {"schema_version": 1,
        "code_version": "replacement", "output_contract": {"fulltext": {"required": ["graph"]}}}})["recipe_digest"]
    k.command("set_requirement", "replacement-requirement", {"unit_id": job["unit_id"], "generation_id": "g",
        "consumed_input_revision": job["consumed_input_revision"], "recipe_digest": recipe,
        "expected_revision": k.get_requirement(job["unit_id"], "g")["revision"], "expected_record_revision": 1})
    replacement = k.command("enqueue", "replacement-job", {"unit_id": job["unit_id"], "generation_id": "g",
        "consumed_input_revision": job["consumed_input_revision"], "recipe_digest": recipe})
    admitted = claim(k, replacement, "replacement-claim")
    before = k.get_reducer(job["unit_id"], job["consumed_input_revision"])
    pin = k.get_pin("attempt-pin:" + admitted["attempt_id"])

    class UnverifiedRetirementBackend(_NoBackend):
        async def submit(self, claim, text, file_path, *, achieved_tier):
            k.command("accept_submission", "accept:" + claim["submission_id"], {"claim": claim})
            raise RetirementFailure(rec["receipt_id"], "old-doc", "success", ("entity rebuild failed",),
                {"pipeline_finished": False})

    scheduler = CatalogScheduler(k, UnverifiedRetirementBackend(), store, breaker=_OpenBuildGate())
    summary = {"observations": 0, "discrepancies": 0, "delivered": 0, "completed": 0,
               "published": 0, "repairs": 0}
    row = next(row for row in k.list_outbox(kind="submission") if row["submission_id"] == admitted["submission_id"])
    for _ in range(2):
        await scheduler._deliver(row, summary)
    discrepancies = k._rows("knowledge_discrepancies")
    assert len(discrepancies) == 1
    assert discrepancies[0]["reason"] == "retirement_rebuild_unverified"
    assert discrepancies[0]["evidence"]["retired_receipt_id"] == rec["receipt_id"]
    assert discrepancies[0]["evidence"]["messages"] == ["entity rebuild failed"]
    repairs = k._rows("knowledge_repairs")
    assert len(repairs) == 1
    assert repairs[0]["publication_id"] == published["publication_id"]
    assert repairs[0]["role"] == "graph" and repairs[0]["reason"] == "retirement_rebuild_unverified"
    assert k.get_job(replacement["job_id"])["state"] == "running"
    assert k.get_job(replacement["job_id"])["attempts"] == 1
    assert k.get_attempt(admitted["attempt_id"])["outcome"] is None
    assert k.get_submission(admitted["submission_id"])["receipt"] is None
    assert k.get_pin(pin["pin_id"]) == pin
    assert k.get_reducer(job["unit_id"], job["consumed_input_revision"]) == before
    assert len(k.list_outbox(kind="submission")) >= 1
    assert len(k._rows("publications")) == 1


@pytest.mark.asyncio
async def test_pause_stops_round_before_readmitting_just_deferred_work(knowledge, tmp_path):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.store.build_breaker import BuildPausedError

    k = knowledge
    store, job = prepare_fulltext(k, tmp_path)
    admitted = claim(k, job, "pause-before-round")

    class PausingBackend(_NoBackend):
        async def submit(self, claim, text, file_path, *, achieved_tier):
            raise BuildPausedError("pause began after the initial round gate")

    scheduler = CatalogScheduler(k, PausingBackend(), store, breaker=_OpenBuildGate(), max_admissions=2)
    summary = await scheduler.run_round(only_keys={"A"})
    assert summary["paused"] is True
    assert summary["claimed"] == 0
    assert k.get_job(job["job_id"])["state"] == "queued"
    assert k.get_job(job["job_id"])["attempts"] == 1
    assert k.get_attempt(admitted["attempt_id"])["outcome"] == "deferred"


@pytest.mark.asyncio
async def test_pause_after_lease_loss_preserves_unknown_attempt_for_expiry(knowledge, tmp_path):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.store.build_breaker import BuildPausedError

    k = knowledge
    store, job = prepare_fulltext(k, tmp_path)
    admitted = claim(k, job, "pause-lost-lease")

    class ExpiredPauseBackend(_NoBackend):
        async def submit(self, claim, text, file_path, *, achieved_tier):
            k.connection.execute(sql.SQL("UPDATE {} SET lease_expires_at=clock_timestamp()-interval '1 second' "
                "WHERE attempt_id=%s").format(k.repository._table("knowledge_attempts")), (claim["attempt_id"],))
            raise BuildPausedError("pause with an already lost execution lease")

    scheduler = CatalogScheduler(k, ExpiredPauseBackend(), store, breaker=_OpenBuildGate())
    summary = {"observations": 0, "discrepancies": 0, "delivered": 0, "completed": 0,
               "published": 0, "repairs": 0}
    row = next(row for row in k.list_outbox(kind="submission") if row["submission_id"] == admitted["submission_id"])
    await scheduler._deliver(row, summary)
    assert summary["paused"] is True
    assert k.get_attempt(admitted["attempt_id"])["outcome"] is None
    assert k.get_job(job["job_id"])["state"] == "running"
    assert k.get_reducer(job["unit_id"], job["consumed_input_revision"]) is None
    assert len(k.list_outbox(kind="submission")) == 1
    assert [event["event"] for event in k._rows("knowledge_attempt_events")] == ["claimed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("pause", ["peak", "cooldown"])
async def test_metadata_retirement_llm_pause_preserves_failure_budget(knowledge, tmp_path, monkeypatch, pause):
    """The pinned rebuild swallows its LLM pause before deletion returns success."""
    from copy import deepcopy

    from lightrag.prompt import PROMPTS
    from papervault.knowledge.scheduler import catalog_intake
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.scheduler.catalog_inputs import make_input_manifest
    from papervault.knowledge.store.build_breaker import BuildBreaker, BuildPausedError, BuildSchedule
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter, _graph_inventory, _inventory
    from papervault.store.knowledge import ClaimRejected
    from tests.knowledge.test_catalog_retirement import memory_rag_for_retirement
    from tests.store.test_knowledge_postgres import receipt

    monkeypatch.setenv("KS_REDISTILL_MAX_ATTEMPTS", "2")
    k = knowledge
    store, old_job = prepare_fulltext(k, tmp_path)
    old_claim = claim(k, old_job, "published-fulltext-claim")
    old_receipt = receipt(old_claim)
    old_receipt["provider_observations"]["doc_id"] = "old-doc"
    # The prior publication must carry verified discovery evidence so this test
    # reaches the reducer's pause instead of failing retirement admission.
    stamp = old_receipt["output_inventory"][0]["verified_at"]
    graph = _graph_inventory(old_claim["namespace"], [{"entity_id": "Shared", "source_id": "old:0"}], [], stamp)
    graph["verification"]["evidence"]["document_contribution"] = {
        "doc_id": "old-doc", "chunk_ids": ["old:0"], "entity_ids": ["Shared"], "relation_pairs": []}
    old_receipt["output_inventory"] = [graph, _inventory("full_entities", "postgres", old_claim["namespace"],
        [{"id": "old-doc", "entity_names": ["Shared"]}], ["old-doc"], stamp),
        _inventory("full_relations", "postgres", old_claim["namespace"], [], [], stamp)]
    old_receipt["document_index_verification"] = {"version": 1, "retired_receipt_ids": []}
    k.command("seal_namespace", "published-fulltext-seal", {"claim": old_claim, "receipt": old_receipt})
    k.command("publish_derivation", "published-fulltext", {**old_claim, "receipt": old_receipt,
        "expected_revision": old_claim["revision"], "expected_selection_revision": 0, "select": True})
    k.command("acknowledge_outbox", "published-fulltext-ack", {"outbox_id": "submit:" + old_claim["submission_id"]})
    artifact = store.capture(b'{"title":"one"}', role="metadata")
    manifest, input_revision = make_input_manifest("paper:A", {"title": "one"}, [artifact], tier="metadata")
    k.command("register_input", "downgrade-input", {"unit_id": "paper:A", "consumed_input_revision": input_revision,
                                                  "manifest": manifest})
    recipe = {"schema_version": 1, "prompts": deepcopy(dict(PROMPTS)), "output_contract": {
        "metadata": {"required": ["metadata", "tombstones"], "not_applicable": ["graph", "vectors"]}}}
    digest = k.command("register_recipe", "downgrade-recipe", {"recipe": recipe})["recipe_digest"]
    monkeypatch.setattr(catalog_intake, "effective_distill_recipe", lambda: deepcopy(recipe))
    k.command("set_requirement", "downgrade-requirement", {"unit_id": "paper:A", "generation_id": "g",
        "consumed_input_revision": input_revision, "recipe_digest": digest,
        "expected_revision": k.get_requirement("paper:A", "g")["revision"], "expected_record_revision": 1})
    job = k.command("enqueue", "downgrade-job", {"unit_id": "paper:A", "generation_id": "g",
        "consumed_input_revision": input_revision, "recipe_digest": digest})
    job = fail_once(k, job)
    before = k.get_reducer("paper:A", input_revision)
    assert before["failure_streak"] == 1 and not before["parked"]
    admitted = claim(k, job, "downgrade-pause-claim")
    schedule = BuildSchedule(enabled=pause == "peak", tz=timezone.utc,
        now=lambda: datetime(2026, 10, 21, 10, tzinfo=timezone.utc))
    breaker = BuildBreaker(threshold=1, cooldown_s=600, clock=lambda: 0, schedule=schedule)
    if pause == "cooldown":
        breaker.record_failure()
    with pytest.raises(BuildPausedError) as paused:
        breaker.before_call()
    rag = await memory_rag_for_retirement(admitted, pause=paused.value)

    async def factory(*args, **kwargs):
        return rag

    adapter = CatalogLightRAGAdapter(k, rag_factory=factory, input_store=store)
    scheduler = CatalogScheduler(k, adapter, store, breaker=_OpenBuildGate())
    summary = {"observations": 0, "discrepancies": 0, "delivered": 0, "completed": 0,
               "published": 0, "repairs": 0}
    row = next(row for row in k.list_outbox(kind="submission") if row["submission_id"] == admitted["submission_id"])
    await scheduler._deliver(row, summary)
    assert len(rag.llm_calls) == 1 and rag.finalized
    assert k.get_job(job["job_id"])["state"] == "queued"
    attempt = k.get_attempt(admitted["attempt_id"])
    assert attempt["outcome"] == "deferred" and attempt["accepted"] and not attempt["sealed"]
    assert attempt["cost_state"] == "unknown"
    assert k.get_reducer("paper:A", input_revision) == before
    assert summary["paused"] is True and summary["discrepancies"] == summary["repairs"] == 0
    assert not k._rows("knowledge_discrepancies") and not k._rows("knowledge_repairs")
    assert not k.list_outbox(kind="submission")
    events = k._rows("knowledge_attempt_events", attempt_id=admitted["attempt_id"])
    assert events[-1]["event"] == "deferred"
    assert not any(event["detail"].get("failure_streak_event") for event in events)
    assert len(k._rows("publications")) == 1
    assert k.get_pin("attempt-pin:" + admitted["attempt_id"])["kind"] == "recovery"
    with pytest.raises(ClaimRejected):
        k.command("record_tombstone", "paused-retirement-stale-write", {
            "claim": admitted, "role": "graph", "output_id": "stale"})
    resumed = claim(k, k.get_job(job["job_id"]), "resume-retirement")
    assert resumed["fencing_token"] > admitted["fencing_token"]
    assert resumed["namespace"] != admitted["namespace"]
    assert k.get_reducer("paper:A", input_revision) == before
    # An actual failed execution still spends the next failure and parks.
    k.command("fail", "real-failure-after-retirement-pause", {**resumed,
        "expected_revision": resumed["revision"], "error": {"kind": "verified_failure"}})
    reducer = k.get_reducer("paper:A", input_revision)
    assert reducer["failure_streak"] == 2 and reducer["parked"] is True
