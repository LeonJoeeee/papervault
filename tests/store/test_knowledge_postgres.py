"""P4 transactions on an explicitly configured disposable PostgreSQL only."""
from concurrent.futures import ThreadPoolExecutor

import psycopg
from psycopg import sql
import pytest

from papervault.store.repository import AuditEvent, Record


@pytest.fixture
def knowledge(repo):
    from papervault.store.knowledge import KnowledgeRepository

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "p4_schema"))
    repo.connection.execute(sql.SQL("UPDATE {} SET authority='catalog'").format(repo._table("knowledge_runtime")))
    return KnowledgeRepository(repo)


def prepare(k, suffix="", candidate=False):
    r = k.repository
    if not r.get("A"):
        r.upsert(Record.from_payload("A", {"key": "A", "title": "one"}),
                 expected_revision=0, audit=AuditEvent("test", "record"))
    k.command("register_unit", "unit" + suffix,
              {"unit_id": "paper:A", "kind": "paper", "record_key": "A", "identity": "A"})
    k.command("register_input", "input" + suffix, {"unit_id": "paper:A",
              "consumed_input_revision": "input1", "manifest": {"metadata_revision": "1",
              "artifacts": [{"artifact_id": "a", "sha256": "a" * 64, "byte_length": 3}],
              "upstream_derivation_ids": []}})
    recipe = k.command("register_recipe", "recipe" + suffix, {"recipe": {"schema_version": 1,
             "code_version": "test", "output_contract": {"fulltext": {
                 "required": ["graph"], "not_applicable": []}}}})["recipe_digest"]
    generation_id = "g" + suffix
    k.command("create_generation", "gen" + suffix,
              {"generation_id": generation_id, "candidate": candidate})
    k.command("set_requirement", "req" + suffix, {"unit_id": "paper:A",
              "consumed_input_revision": "input1", "recipe_digest": recipe,
              "generation_id": generation_id, "expected_revision": 0,
              "expected_record_revision": k.repository.get("A").revision})
    job = k.command("enqueue", "job" + suffix, {"unit_id": "paper:A",
              "consumed_input_revision": "input1", "recipe_digest": recipe,
              "generation_id": generation_id})
    return job


def claim(k, job, op="claim"):
    return k.command("claim", op, {"job_id": job["job_id"], "expected_revision": job["revision"],
                                   "authority_epoch": 1, "lease_seconds": 300, "reserved_cost": 1})


def receipt(c):
    from papervault.store.repository import payload_hash
    selected = {**c["base_selected_unit_receipts"], c["unit_id"]: "receipt:" + c["attempt_id"]}
    return {key: value for key, value in c.items() if key not in {"revision", "lease_expires_at"}} | {
        "receipt_id": "receipt:" + c["attempt_id"], "sealed": True, "durable": True,
        "aggregate_layers": c["base_aggregate_layers"] + [c["namespace"]],
        "selected_unit_receipts": selected, "selected_unit_receipts_digest": payload_hash(selected),
        "achieved_tier": "fulltext", "provider_observations": {"model": "unknown"},
        "output_inventory": [{"role": "graph", "output_id": "graph:" + c["attempt_id"],
           "store": "neo4j", "namespace": c["namespace"], "backend_ids": [],
           "sha256": None, "byte_length": None, "verified_at": "2026-10-20T00:00:00Z",
           "verification": {"state": "verified_empty", "method": "count",
                            "evidence": {"count": 0, "verified": True}}}]}


def seal(k, c):
    rec = receipt(c)
    k.command("seal_namespace", "seal:" + c["attempt_id"], {"claim": c, "receipt": rec})
    return rec


def test_command_replay_conflict_and_logical_job_uniqueness(knowledge):
    from papervault.store.knowledge import OperationConflict

    job = prepare(knowledge)
    same = knowledge.command("enqueue", "another-enqueue", {
        key: job[key] for key in ("unit_id", "consumed_input_revision", "recipe_digest", "generation_id")})
    assert same["job_id"] == job["job_id"]
    payload = {"job_id": job["job_id"], "expected_revision": job["revision"], "authority_epoch": 1}
    first = knowledge.command("claim", "claim", payload)
    assert knowledge.command("claim", "claim", payload) == first
    with pytest.raises(OperationConflict):
        knowledge.command("claim", "claim", {**payload, "lease_seconds": 4})
    assert knowledge.get_job(job["job_id"])["attempts"] == 1
    assert len([r for r in knowledge.list_outbox() if r["kind"] == "submission"]) == 1


def test_success_is_not_publication_and_candidate_does_not_change_serving(knowledge):
    job = prepare(knowledge)
    c = claim(knowledge, job)
    rec = seal(knowledge, c)
    complete = knowledge.command("complete", "complete", {**c, "receipt": rec,
                                                          "expected_revision": c["revision"]})
    assert knowledge.get_job(job["job_id"])["state"] == "succeeded"
    assert knowledge.get_selection("paper:A", "g") is None
    result = knowledge.command("publish_derivation", "publish", {**c, "receipt": rec,
              "expected_revision": complete["revision"], "expected_selection_revision": 0, "select": True})
    assert result["selected"] is True
    assert knowledge.get_serving_manifest()["generation_id"] == "g"
    before = knowledge.get_serving_manifest()
    candidate = prepare(knowledge, "candidate", candidate=True)
    from datetime import datetime, timedelta, timezone
    start = datetime.now(timezone.utc) - timedelta(seconds=1)
    knowledge.command("begin_maintenance_window", "window", {"window_id": "window", "kind": "monthly_reboot",
        "starts_at": start.isoformat(), "ends_at": (start+timedelta(hours=1)).isoformat(),
        "operator_nonce": "test", "expected_revision": 0})
    c2 = claim(knowledge, candidate, "claimcandidate")
    rec2 = seal(knowledge, c2)
    knowledge.command("publish_derivation", "publishcandidate", {**c2, "receipt": rec2,
              "expected_revision": c2["revision"], "expected_selection_revision": 0, "select": True})
    assert knowledge.get_serving_manifest() == before


def test_input_change_during_execution_allows_history_but_never_selection(knowledge):
    job = prepare(knowledge)
    c = claim(knowledge, job)
    rec = seal(knowledge, c)
    r = knowledge.repository
    r.upsert(Record.from_payload("A", {"key": "A", "title": "changed"}),
             expected_revision=1, audit=AuditEvent("test", "changed"))
    result = knowledge.command("publish_derivation", "publish", {**c, "receipt": rec,
              "expected_revision": c["revision"], "expected_selection_revision": 0, "select": True})
    assert result["selected"] is False
    assert result["historical"] is True
    assert knowledge.get_selection("paper:A", "g") is None
    assert knowledge.get_job(job["job_id"])["state"] == "succeeded"
    assert knowledge.get_requirement("paper:A", "g")["invalidated"] is True


def test_expiry_records_abandoned_attempt_unknown_charge_and_rejects_old_fence(knowledge):
    from papervault.store.knowledge import ClaimRejected

    job = prepare(knowledge)
    c1 = claim(knowledge, job)
    knowledge.repository.connection.execute(sql.SQL(
        "UPDATE {} SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE attempt_id=%s"
    ).format(knowledge.repository._table("knowledge_attempts")), (c1["attempt_id"],))
    job = knowledge.get_job(job["job_id"])
    c2 = claim(knowledge, job, "retry")
    assert c2["fencing_token"] > c1["fencing_token"]
    assert c2["submission_id"] != c1["submission_id"]
    abandoned = knowledge.get_attempt(c1["attempt_id"])
    assert abandoned["outcome"] == "abandoned"
    assert abandoned["cost_state"] == "unknown"
    with pytest.raises(ClaimRejected):
        knowledge.command("complete", "stale", {**c1, "receipt": receipt(c1),
                                                 "expected_revision": c1["revision"]})


def test_wrong_recipe_receipt_and_publication_transaction_failure_leave_no_partial_state(knowledge):
    from papervault.store.knowledge import ClaimRejected

    job = prepare(knowledge)
    c = claim(knowledge, job)
    bad = receipt(c) | {"recipe_digest": "wrong"}
    with pytest.raises((ValueError, ClaimRejected)):
        knowledge.command("seal_namespace", "bad-seal", {"claim": c, "receipt": bad})
    assert knowledge.get_receipt(bad["receipt_id"]) is None
    rec = seal(knowledge, c)
    r = knowledge.repository
    # An actual failing database trigger proves all publication writes roll back together.
    r.connection.execute(sql.SQL("""CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'injected publication failure'; END $$""").format(
        sql.Identifier(r.schema, "fail_publish")))
    r.connection.execute(sql.SQL("CREATE TRIGGER fail_publish BEFORE INSERT ON {} "
        "FOR EACH STATEMENT EXECUTE FUNCTION {}()").format(r._table("publications"),
            sql.Identifier(r.schema, "fail_publish")))
    with pytest.raises(psycopg.errors.RaiseException, match="injected"):
        knowledge.command("publish_derivation", "publish", {**c, "receipt": rec,
            "expected_revision": c["revision"], "expected_selection_revision": 0, "select": True})
    assert knowledge.get_selection("paper:A", "g") is None
    assert knowledge.get_job(job["job_id"])["state"] == "running"


def test_history_tables_reject_update_and_delete(knowledge):
    job = prepare(knowledge)
    c = claim(knowledge, job)
    seal(knowledge, c)
    r = knowledge.repository
    for table in ("input_manifests", "recipes", "knowledge_receipts", "generation_manifests"):
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            r.connection.execute(sql.SQL("DELETE FROM {}").format(r._table(table)))


def test_two_connections_can_claim_one_job_only_once(knowledge):
    from papervault.store.knowledge import ClaimRejected, KnowledgeRepository
    from papervault.store.repository import PostgresCatalogRepository

    job = prepare(knowledge)
    r = knowledge.repository
    def run(number):
        with psycopg.connect(r.connection.info.dsn, autocommit=True) as conn:
            other = KnowledgeRepository(PostgresCatalogRepository(conn, schema=r.schema))
            try:
                return claim(other, job, "concurrent" + str(number))
            except ClaimRejected:
                return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [1, 2]))
    assert sum(value is not None for value in results) == 1
    assert knowledge.get_job(job["job_id"])["attempts"] == 1


def test_observations_do_not_heal_and_repair_requests_deduplicate(knowledge):
    job = prepare(knowledge)
    payload = {"observation_id": "o", "unit_id": "paper:A", "generation_id": "g",
               "consumed_input_revision": "input1", "recipe_digest": job["recipe_digest"],
               "role": "graph", "store": "neo4j", "namespace": "unverified", "output_id": "x",
               "state": "present", "observed_at": "2026-10-20T00:00:00Z"}
    knowledge.command("observe", "observe", payload)
    assert knowledge.get_job(job["job_id"])["state"] == "queued"
    c = claim(knowledge, job)
    rec = seal(knowledge, c)
    published = knowledge.command("publish_derivation", "pub", {**c, "receipt": rec,
        "expected_revision": c["revision"], "expected_selection_revision": 0, "select": True})
    repair = {"publication_id": published["publication_id"], "role": "graph", "reason": "missing"}
    first = knowledge.command("request_repair", "repair1", repair)
    second = knowledge.command("request_repair", "repair2", repair)
    assert first["repair_id"] == second["repair_id"]
    assert knowledge.get_job(job["job_id"])["state"] == "succeeded"


def test_section_mapping_and_inventory_revisions_preserve_old_definition(knowledge):
    prepare(knowledge)
    knowledge.command("register_unit", "section", {"unit_id": "section:s", "kind": "section",
         "record_key": "A", "identity": "s"})
    for revision, locator in [(1, "chapter 1"), (2, "chapter 2")]:
        knowledge.command("register_section", "section" + str(revision), {
            "unit_id": "section:s", "mapping_revision": revision, "parent_unit_id": "paper:A",
            "parent_evidence_revision": "e1", "definition": {"locator": locator}})
    knowledge.command("register_inventory", "inventory", {"parent_unit_id": "paper:A",
        "inventory_revision": "v1", "parent_evidence_revision": "e1", "mode": "sections",
        "required_units": ["section:s"], "optional_units": [],
        "section_mapping_revisions": {"section:s": 1}})
    assert knowledge.get_section("section:s", 1)["definition"]["locator"] == "chapter 1"
    assert knowledge.get_inventory("paper:A", "v1")["required_units"] == ["section:s"]


def test_candidate_claim_requires_explicit_bounded_monthly_window(knowledge):
    from datetime import datetime, timedelta, timezone
    from papervault.store.knowledge import ClaimRejected

    job = prepare(knowledge, candidate=True)
    with pytest.raises(ClaimRejected, match="monthly"):
        claim(knowledge, job)
    start = datetime.now(timezone.utc) - timedelta(seconds=1)
    payload = {"window_id": "window", "kind": "monthly_reboot", "starts_at": start.isoformat(),
               "ends_at": (start + timedelta(hours=3)).isoformat(), "operator_nonce": "operator-explicit",
               "expected_revision": 0}
    with pytest.raises(ClaimRejected, match="two hours"):
        knowledge.command("begin_maintenance_window", "too-long", payload)
    payload["ends_at"] = (start + timedelta(hours=1)).isoformat()
    knowledge.command("begin_maintenance_window", "window", payload)
    c = claim(knowledge, job)
    assert c["attempt_id"]
    with pytest.raises(ClaimRejected, match="already recorded"):
        knowledge.command("begin_maintenance_window", "second", {**payload, "window_id": "second",
           "operator_nonce": "second", "expected_revision": 1})


def test_requirement_cannot_credit_inputs_captured_before_record_changed(knowledge):
    from papervault.store.knowledge import ClaimRejected

    job = prepare(knowledge)
    knowledge.repository.upsert(Record.from_payload("A", {"key": "A", "title": "changed"}),
        expected_revision=1, audit=AuditEvent("test", "changed"))
    with pytest.raises(ClaimRejected, match="record revision"):
        knowledge.command("set_requirement", "stale-capture", {"unit_id": "paper:A", "generation_id": "g",
            "consumed_input_revision": "input1", "recipe_digest": job["recipe_digest"],
            "expected_revision": 2, "expected_record_revision": 1})
    assert knowledge.get_requirement("paper:A", "g")["invalidated"]


def test_concurrent_aggregate_bases_cannot_publish_independently(knowledge):
    from papervault.store.knowledge import ClaimRejected

    first = prepare(knowledge)
    k = knowledge
    k.command("register_unit", "unitB", {"unit_id": "paper:B", "kind": "paper", "identity": "B"})
    k.command("register_input", "inputB", {"unit_id": "paper:B", "consumed_input_revision": "inputB",
        "manifest": {"metadata_revision": "b", "artifacts": [], "upstream_derivation_ids": []}})
    k.command("set_requirement", "reqB", {"unit_id": "paper:B", "generation_id": "g",
        "consumed_input_revision": "inputB", "recipe_digest": first["recipe_digest"], "expected_revision": 0})
    second = k.command("enqueue", "jobB", {"unit_id": "paper:B", "generation_id": "g",
        "consumed_input_revision": "inputB", "recipe_digest": first["recipe_digest"]})
    c1, c2 = claim(k, first), claim(k, second, "claimB")
    rec1, rec2 = seal(k, c1), seal(k, c2)
    for c, rec, op in [(c1, rec1, "publishA"), (c2, rec2, "publishB")]:
        payload = {**c, "receipt": rec, "expected_revision": c["revision"],
                   "expected_selection_revision": 0, "select": True}
        if op == "publishA":
            k.command("publish_derivation", op, payload)
        else:
            with pytest.raises(ClaimRejected, match="aggregate base"):
                k.command("publish_derivation", op, payload)
    assert k.get_selection("paper:B", "g") is None
    assert k.get_job(second["job_id"])["state"] == "running"


def test_fork_freezes_all_requirements_and_inventories_without_copying_selected_outputs(knowledge):
    job = prepare(knowledge)
    generation = knowledge.get_generation("g")
    runtime = knowledge.get_authority_state()
    fork = knowledge.command("fork_generation", "fork", {"generation_id": "rebuild", "source_generation_id": "g",
        "expected_manifest_revision": generation["manifest_revision"],
        "expected_requirements_revision": runtime["requirements_revision"]})
    assert fork["state"] == "candidate"
    assert fork["queued_jobs"] == 1
    manifest = knowledge.get_manifest("rebuild")["manifest"]
    assert manifest["aggregate_layers"] == []
    assert manifest["entries"][0]["consumed_input_revision"] == job["consumed_input_revision"]
    assert manifest["entries"][0]["missing"] is True


def test_failure_streak_parks_once_and_delivery_replay_does_not_increment(knowledge, monkeypatch):
    from papervault.store.knowledge import ClaimRejected

    monkeypatch.setenv("KS_REDISTILL_MAX_ATTEMPTS", "2")
    job = prepare(knowledge)
    for number in (1, 2):
        current = knowledge.get_job(job["job_id"])
        c = claim(knowledge, current, "claim" + str(number))
        payload = {**c, "expected_revision": c["revision"], "error": {"kind": "verified_failure"}}
        result = knowledge.command("fail", "fail" + str(number), payload)
        assert knowledge.command("fail", "fail" + str(number), payload) == result
    checkpoint = knowledge.get_reducer("paper:A", "input1")
    assert checkpoint["failure_streak"] == 2
    assert checkpoint["parked"] is True
    with pytest.raises(ClaimRejected, match="parked"):
        claim(knowledge, knowledge.get_job(job["job_id"]), "parked")


class _OpenBuildGate:
    def allows(self):
        return True


class _NoBackend:
    async def lookup_submission(self, submission_id):
        return None


def prepare_metadata(k, tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore, make_input_manifest

    prepare(k)
    store = ImmutableInputStore(tmp_path)
    artifact = store.capture(b'{"title":"one"}', role="metadata")
    manifest, input_revision = make_input_manifest("paper:A", {"title": "one"}, [artifact], tier="metadata")
    k.command("register_input", "metadata-input", {"unit_id": "paper:A", "consumed_input_revision": input_revision,
                                                "manifest": manifest})
    recipe = k.command("register_recipe", "metadata-recipe", {"recipe": {"schema_version": 1,
        "output_contract": {"metadata": {"required": ["metadata", "tombstones"], "not_applicable": ["graph", "vectors"]}}}})["recipe_digest"]
    k.command("set_requirement", "metadata-requirement", {"unit_id": "paper:A", "generation_id": "g",
        "consumed_input_revision": input_revision, "recipe_digest": recipe, "expected_revision": 1,
        "expected_record_revision": 1})
    job = k.command("enqueue", "metadata-job", {"unit_id": "paper:A", "generation_id": "g",
        "consumed_input_revision": input_revision, "recipe_digest": recipe})
    return store, job


@pytest.mark.asyncio
async def test_catalog_round_metadata_publication_replay_does_not_create_another_attempt(knowledge, tmp_path):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler

    store, job = prepare_metadata(knowledge, tmp_path)
    scheduler = CatalogScheduler(knowledge, _NoBackend(), store, breaker=_OpenBuildGate())
    first = await scheduler.run_round(only_keys={"A"})
    assert first["claimed"] == first["completed"] == first["published"] == 1
    assert knowledge.get_job(job["job_id"])["state"] == "succeeded"
    second = await scheduler.run_round(only_keys={"A"})
    assert second["claimed"] == second["completed"] == second["published"] == 0
    assert knowledge.get_job(job["job_id"])["attempts"] == 1
    assert knowledge.get_serving_manifest()["entries"][0]["eligible"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("fault_action", ["accept_submission", "seal_namespace", "complete", "publish_derivation", "acknowledge_outbox"])
@pytest.mark.parametrize("fault_stage", ["before", "after"])
async def test_round_recovers_catalog_commit_before_response_without_reexecuting(knowledge, tmp_path, monkeypatch, fault_action, fault_stage):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler

    store, job = prepare_metadata(knowledge, tmp_path)
    scheduler = CatalogScheduler(knowledge, _NoBackend(), store, breaker=_OpenBuildGate())
    original = knowledge.command
    injected = False
    def lose_response(action, operation_id, payload):
        nonlocal injected
        if action == fault_action and fault_stage == "before" and not injected:
            injected = True
            raise OSError("injected before catalog commit")
        result = original(action, operation_id, payload)
        if action == fault_action and fault_stage == "after" and not injected:
            injected = True
            raise OSError("injected lost catalog response")
        return result
    monkeypatch.setattr(knowledge, "command", lose_response)
    await scheduler.run_round(only_keys={"A"})
    assert injected
    await scheduler.run_round(only_keys={"A"})
    assert knowledge.get_job(job["job_id"])["attempts"] == 1
    assert knowledge.get_job(job["job_id"])["state"] == "succeeded"
    assert knowledge.get_selection("paper:A", "g") is not None
    assert not [r for r in knowledge.list_outbox() if r["kind"] == "submission"]


def test_mapped_legacy_parking_checkpoint_is_consumed_once_without_execution_events(knowledge):
    from papervault.store.knowledge import ClaimRejected

    job = prepare(knowledge)
    knowledge._insert("jobs", {"id": "legacy-parked", "kind": "distill_legacy_unverified", "key": "A",
        "unit_id": "paper:A", "state": "quarantined", "detail": {"source_version": 1,
        "legacy_consecutive_failures": 3, "evidence": {"raw": {"status": "error_parked", "attempts": 3}}}})
    payload = {"unit_id": "paper:A", "consumed_input_revision": "input1"}
    applied = knowledge.command("migrate_legacy_checkpoint", "migrate-parked", payload)
    assert applied["evidence"]["historical_events_replayed"] is False
    assert knowledge.get_reducer("paper:A", "input1")["parked"] is True
    assert knowledge.command("migrate_legacy_checkpoint", "migrate-parked", payload) == applied
    with pytest.raises(ClaimRejected, match="parked"):
        claim(knowledge, job)
    assert knowledge.get_job(job["job_id"])["attempts"] == 0
    assert knowledge._rows("knowledge_attempt_events") == []


def test_staging_recovery_and_reader_pins_protect_outputs_without_publication_gap(knowledge):
    from papervault.store.knowledge import ClaimRejected

    job = prepare(knowledge)
    c = claim(knowledge, job)
    assert c["namespace"] in knowledge.protected_namespaces()
    staged = knowledge.get_pin("attempt-pin:" + c["attempt_id"])
    assert staged["detail"]["unfinished_output_exists"] is False
    rec = seal(knowledge, c)
    result = knowledge.command("publish_derivation", "pub", {**c, "receipt": rec,
        "expected_revision": c["revision"], "expected_selection_revision": 0, "select": True})
    assert result["selected"]
    assert knowledge.get_pin(staged["pin_id"])["released_at"] is not None
    assert c["namespace"] in knowledge.protected_namespaces()
    current = knowledge.get_serving_manifest()
    pin = knowledge.command("pin_manifest", "reader", {"generation_id": "g",
        "manifest_revision": current["manifest_revision"], "expected_serving_revision": current["revision"],
        "owner": "reader1", "ttl_seconds": 300})
    assert pin["kind"] == "reader"
    assert c["namespace"] in pin["namespaces"]
    renewed = knowledge.command("renew_pin", "renew-reader", {"pin_id": pin["pin_id"], "owner": "reader1",
        "expected_revision": pin["revision"], "ttl_seconds": 300})
    with pytest.raises(ClaimRejected, match="revision"):
        knowledge.command("release_pin", "stale-release", {"pin_id": pin["pin_id"], "owner": "reader1",
            "expected_revision": pin["revision"]})
    knowledge.command("release_pin", "release", {"pin_id": pin["pin_id"], "owner": "reader1",
        "expected_revision": renewed["revision"]})
    assert knowledge.get_pin(pin["pin_id"])["released_at"] is not None


def test_force_inventory_intent_cannot_execute_in_active_generation_or_be_reset_by_normal_retry(knowledge):
    from papervault.store.knowledge import ClaimRejected

    job = prepare(knowledge)
    knowledge.command("register_inventory", "whole-inventory", {"parent_unit_id": "paper:A", "inventory_revision": "whole",
        "parent_evidence_revision": "e1", "parent_consumed_input_revision": "input1", "mode": "whole",
        "required_units": ["paper:A"], "optional_units": [], "section_mapping_revisions": {}})
    knowledge.command("set_inventory", "initial-whole", {"parent_unit_id": "paper:A", "generation_id": "g",
        "inventory_revision": "whole", "expected_revision": 0, "expected_record_revision": 1})
    knowledge.command("set_requirement", "require-known-whole", {"unit_id": "paper:A", "generation_id": "g",
        "inventory_revision": "whole", "consumed_input_revision": "input1", "recipe_digest": job["recipe_digest"],
        "expected_revision": knowledge.get_requirement("paper:A", "g")["revision"], "expected_record_revision": 1})
    knowledge.command("set_inventory", "force", {"parent_unit_id": "paper:A", "generation_id": "g",
        "inventory_revision": "whole", "expected_revision": 1, "expected_record_revision": 1,
        "enqueue_mode": "candidate", "request_id": "operator-force-1", "desired_recipe_digest": job["recipe_digest"]})
    with pytest.raises(ClaimRejected, match="candidate rebuild"):
        claim(knowledge, job)
    assert knowledge.get_requirement("paper:A", "g")["invalidated"] is False
    knowledge.command("set_inventory", "normal-retry", {"parent_unit_id": "paper:A", "generation_id": "g",
        "inventory_revision": "whole", "expected_revision": 2, "expected_record_revision": 1,
        "enqueue_mode": "active", "request_id": "normal-retry"})
    current = knowledge.get_current_inventory("paper:A", "g")
    assert current["enqueue_mode"] == "candidate"
    assert current["request_id"] == "operator-force-1"
    assert knowledge.get_requirement("paper:A", "g")["enqueue_mode"] == "candidate"
    with pytest.raises(ClaimRejected, match="candidate rebuild"):
        claim(knowledge, job, "still-forced")


def prepare_fulltext(k, tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore, make_input_manifest

    old = prepare(k)
    store = ImmutableInputStore(tmp_path)
    artifact = store.capture(b"Real full text consumed by the test.\n", role="fulltext_evidence")
    manifest, input_revision = make_input_manifest("paper:A", {"title": "one"}, [artifact], tier="fulltext")
    k.command("register_input", "text-input", {"unit_id": "paper:A", "consumed_input_revision": input_revision,
                                             "manifest": manifest})
    k.command("set_requirement", "text-requirement", {"unit_id": "paper:A", "generation_id": "g",
        "consumed_input_revision": input_revision, "recipe_digest": old["recipe_digest"],
        "expected_revision": 1, "expected_record_revision": 1})
    job = k.command("enqueue", "text-job", {"unit_id": "paper:A", "generation_id": "g",
        "consumed_input_revision": input_revision, "recipe_digest": old["recipe_digest"]})
    return store, job


@pytest.mark.asyncio
async def test_scheduler_definitive_execution_failure_trace_parks_without_counting_delivery_again(knowledge, tmp_path, monkeypatch):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.store.knowledge import KnownExecutionFailure

    monkeypatch.setenv("KS_REDISTILL_MAX_ATTEMPTS", "2")
    store, job = prepare_fulltext(knowledge, tmp_path)
    class FailingBackend(_NoBackend):
        async def submit(self, claim, text, file_path, *, achieved_tier):
            raise KnownExecutionFailure("confirmed output verification failure")
    scheduler = CatalogScheduler(knowledge, FailingBackend(), store, breaker=_OpenBuildGate())
    for _ in range(4):
        await scheduler.run_round(only_keys={"A"})
    settled = knowledge.get_job(job["job_id"])
    reducer = knowledge.get_reducer(job["unit_id"], job["consumed_input_revision"])
    assert settled["attempts"] == 2
    assert settled["state"] == "failed"
    assert reducer["failure_streak"] == 2 and reducer["parked"] is True
    assert [r["event"] for r in knowledge._rows("knowledge_attempt_events")].count("failed") == 2
    assert not knowledge._rows("publications")
    assert not knowledge.list_outbox(kind="submission")


@pytest.mark.asyncio
async def test_scheduler_ambiguous_acceptance_retries_one_submission_until_expiry_then_parks_conservatively(knowledge, tmp_path, monkeypatch):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler

    monkeypatch.setenv("KS_REDISTILL_MAX_ATTEMPTS", "2")
    store, job = prepare_fulltext(knowledge, tmp_path)
    class AmbiguousBackend(_NoBackend):
        async def submit(self, claim, text, file_path, *, achieved_tier):
            raise OSError("external acceptance response lost")
    scheduler = CatalogScheduler(knowledge, AmbiguousBackend(), store, breaker=_OpenBuildGate())
    await scheduler.run_round(only_keys={"A"})
    await scheduler.run_round(only_keys={"A"})
    assert knowledge.get_job(job["job_id"])["attempts"] == 1
    assert knowledge.get_reducer(job["unit_id"], job["consumed_input_revision"]) is None
    for _ in range(2):
        current = knowledge.get_job(job["job_id"])
        knowledge.repository.connection.execute(sql.SQL("UPDATE {} SET lease_expires_at=clock_timestamp()-interval '1 second' "
            "WHERE attempt_id=%s").format(knowledge.repository._table("knowledge_attempts")), (current["current_attempt_id"],))
        await scheduler.run_round(only_keys={"A"})
    await scheduler.run_round(only_keys={"A"})
    attempts = knowledge._rows("knowledge_attempts", job_id=job["job_id"])
    assert len(attempts) == 2
    assert all(a["outcome"] == "abandoned" and a["cost_state"] == "unknown" for a in attempts)
    assert knowledge.get_reducer(job["unit_id"], job["consumed_input_revision"])["parked"] is True
    assert knowledge.get_authority_state()["budget_reserved"] == 2


@pytest.mark.asyncio
async def test_scheduler_recovers_external_acceptance_and_preserves_public_citation_path(knowledge, tmp_path):
    from datetime import datetime, timezone
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.store.knowledge import CLAIM_FIELDS

    store, job = prepare_fulltext(knowledge, tmp_path)
    accepted = {}
    class RecoverableBackend(_NoBackend):
        async def submit(self, claim, text, file_path, *, achieved_tier):
            knowledge.command("accept_submission", "accept:"+claim["submission_id"], {"claim": claim})
            if claim["submission_id"] not in accepted:
                accepted[claim["submission_id"]] = {"text": text, "file_path": file_path}
                raise OSError("external output durable before receipt registration")
            rec = receipt(claim)
            knowledge.command("seal_namespace", "seal:"+claim["submission_id"], {"claim": claim, "receipt": rec})
            return rec
        async def collect_observations(self, rec):
            output = rec["output_inventory"][0]
            return [{**{key: rec[key] for key in CLAIM_FIELDS}, "receipt_id": rec["receipt_id"],
                **{key: output[key] for key in ("role", "store", "namespace", "output_id")},
                "state": "present", "observed_at": datetime.now(timezone.utc).isoformat(), "detail": {}}]
    scheduler = CatalogScheduler(knowledge, RecoverableBackend(), store, breaker=_OpenBuildGate())
    await scheduler.run_round(only_keys={"A"})
    assert knowledge.get_job(job["job_id"])["state"] == "running"
    await scheduler.run_round(only_keys={"A"})
    assert knowledge.get_job(job["job_id"])["state"] == "succeeded"
    assert knowledge.get_job(job["job_id"])["attempts"] == 1
    assert knowledge.get_selection("paper:A", "g") is not None
    assert len(accepted) == 1
    assert next(iter(accepted.values()))["file_path"] == "paper/A"
    assert not knowledge.list_outbox(kind="submission")


def test_promotion_refuses_interrupted_inventory_with_missing_required_section(knowledge):
    from datetime import datetime, timedelta, timezone
    from papervault.store.knowledge import ClaimRejected

    k = knowledge
    for uid, kind in (("O", "operator_document"), ("S1", "section"), ("S2", "section")):
        k.command("register_unit", "unit:"+uid, {"unit_id": uid, "kind": kind, "identity": uid})
    parent = {"metadata_revision": "m", "parent_evidence_revision": "e", "upstream_derivation_ids": [],
              "artifacts": [{"artifact_id": "parent-text", "sha256": "a"*64, "byte_length": 12, "role": "fulltext_evidence"}]}
    k.command("register_input", "parent-input", {"unit_id": "O", "consumed_input_revision": "parent-input", "manifest": parent})
    for uid in ("S1", "S2"):
        k.command("register_section", "map:"+uid, {"unit_id": uid, "mapping_revision": 1,
            "parent_unit_id": "O", "parent_evidence_revision": "e", "definition": {"locator": uid}})
    section = {**parent, "section_definition_revision": 1, "parent_consumed_input_revision": "parent-input",
               "artifacts": [{"artifact_id": "section-text", "sha256": "b"*64, "byte_length": 12, "role": "fulltext_evidence"}]}
    k.command("register_input", "section-input", {"unit_id": "S1", "consumed_input_revision": "section-input", "manifest": section})
    recipe = k.command("register_recipe", "recipe", {"recipe": {"schema_version": 1,
        "output_contract": {"fulltext": {"required": ["graph"], "not_applicable": []}}}})["recipe_digest"]
    k.command("create_generation", "active", {"generation_id": "active", "candidate": False})
    k.command("register_inventory", "inventory", {"parent_unit_id": "O", "inventory_revision": "v1",
        "parent_evidence_revision": "e", "parent_consumed_input_revision": "parent-input", "mode": "sections",
        "required_units": ["S1", "S2"], "optional_units": [], "section_mapping_revisions": {"S1": 1, "S2": 1}})
    k.command("set_inventory", "inventory-current", {"parent_unit_id": "O", "inventory_revision": "v1",
        "generation_id": "active", "expected_revision": 0})
    k.command("set_requirement", "S1-only", {"unit_id": "S1", "generation_id": "active", "inventory_revision": "v1",
        "consumed_input_revision": "section-input", "recipe_digest": recipe, "expected_revision": 0})
    k.command("fork_generation", "fork", {"generation_id": "candidate", "source_generation_id": "active",
        "expected_manifest_revision": k.get_generation("active")["manifest_revision"],
        "expected_requirements_revision": k.get_authority_state()["requirements_revision"]})
    stamp = datetime.now(timezone.utc)
    k.command("begin_maintenance_window", "window", {"window_id": "window", "kind": "monthly_reboot",
        "starts_at": (stamp-timedelta(seconds=1)).isoformat(), "ends_at": (stamp+timedelta(hours=1)).isoformat(),
        "operator_nonce": "test", "expected_revision": 0})
    job = k.get_logical_job("S1", "section-input", recipe, "candidate")
    c = claim(k, job)
    rec = seal(k, c)
    k.command("publish_derivation", "publish", {**c, "receipt": rec, "expected_revision": c["revision"],
        "expected_selection_revision": 0, "select": True})
    # Supply actual version-qualified fixture evidence for S1 and the parent;
    # S2 has a declared mapping but no desired row, job, receipt or observation.
    for uid, rev, artifact in (("O", "parent-input", parent["artifacts"][0]), ("S1", "section-input", section["artifacts"][0])):
        k.command("observe", "input:"+uid, {"observation_id": "input:"+uid, "unit_id": uid,
            "consumed_input_revision": rev, "role": "input_evidence", "store": "library",
            "output_id": artifact["artifact_id"], "state": "present", "observed_at": stamp.isoformat(), "detail": artifact})
    output = rec["output_inventory"][0]
    k.command("observe", "graph", {**c, "receipt_id": rec["receipt_id"], "observation_id": "graph",
        "role": "graph", "store": "neo4j", "output_id": output["output_id"], "state": "present", "observed_at": stamp.isoformat()})
    with pytest.raises(ClaimRejected, match="required inventory unit"):
        k.command("promote_generation", "promote", {"generation_id": "candidate",
            "expected_manifest_revision": k.get_generation("candidate")["manifest_revision"],
            "expected_requirements_revision": k.get_authority_state()["requirements_revision"],
            "expected_serving_revision": k.get_authority_state()["serving_revision"]})
    assert k.get_active_generation()["generation_id"] == "active"


@pytest.mark.parametrize("age", [-1, 301, 3600, "300", True])
def test_promotion_cannot_relax_resolved_observation_age_policy(knowledge, age):
    from papervault.store.knowledge import ClaimRejected

    with pytest.raises(ClaimRejected, match="300 seconds"):
        knowledge.command("promote_generation", "bad-age", {"generation_id": "uninitialized",
            "max_observation_age": age})


def test_generation_embedding_contract_binds_vector_claims_only_and_fences_same_dimension_model_changes(knowledge):
    from datetime import datetime, timedelta, timezone
    from papervault.store.knowledge import ClaimRejected

    k = knowledge
    k.command("create_generation", "embedding-gen", {"generation_id": "embedding-g", "candidate": False})
    def vector_recipe(model):
        return {"schema_version": 1, "embedding_contract": {
            "schema_version": 1, "model": model, "dimension": 1024, "max_length": 8192,
            "use_fp16": True, "representation": "dense:cosine", "implementation_sha256": "a" * 64,
            "packages": {"test-embedding": "1"}},
            "output_contract": {"fulltext": {"required": ["entities_vdb"]},
                                "metadata": {"required": ["metadata"], "not_applicable": ["entities_vdb"]}}}

    def job(unit, model, tier="fulltext", generation="embedding-g"):
        k.command("register_unit", "embedding-unit:"+unit, {
            "unit_id": unit, "kind": "operator_document", "identity": {"source": unit}})
        k.command("register_input", "embedding-input:"+unit, {"unit_id": unit,
            "consumed_input_revision": "input:"+unit, "manifest": {
                "metadata_revision": "1", "tier": tier, "artifacts": []}})
        recipe = k.command("register_recipe", "embedding-recipe:"+model, {"recipe": vector_recipe(model)})
        k.command("set_requirement", "embedding-required:"+unit, {"unit_id": unit,
            "consumed_input_revision": "input:"+unit, "recipe_digest": recipe["recipe_digest"],
            "generation_id": generation, "expected_revision": 0})
        return k.command("enqueue", "embedding-job:"+unit, {"unit_id": unit,
            "consumed_input_revision": "input:"+unit, "recipe_digest": recipe["recipe_digest"],
            "generation_id": generation})

    metadata = job("metadata", "model-b", "metadata")
    claim(k, metadata, "embedding-metadata-claim")
    assert k.get_generation("embedding-g")["retrieval_contract"] is None
    first = job("vector-a", "model-a")
    c = claim(k, first, "embedding-first-claim")
    expected = vector_recipe("model-a")["embedding_contract"]
    assert k.get_generation("embedding-g")["retrieval_contract"] == expected
    assert k.get_manifest("embedding-g")["manifest"]["retrieval_contract"] == expected
    assert k.get_serving_manifest()["retrieval_contract"] == expected
    rec = receipt(c)
    rec["output_inventory"][0]["role"] = "entities_vdb"
    k.command("seal_namespace", "embedding-seal", {"claim": c, "receipt": rec})
    incompatible = job("vector-b", "model-b")
    with pytest.raises(ClaimRejected, match="retrieval contract"):
        claim(k, incompatible, "embedding-incompatible-claim")
    assert k.get_job(incompatible["job_id"])["attempts"] == 0
    assert k.get_generation("embedding-g")["retrieval_contract"] == expected
    with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
        with k.connection.transaction():
            k.connection.execute(sql.SQL("UPDATE {} SET retrieval_contract=NULL WHERE generation_id=%s").format(
                k.repository._table("generations")), ("embedding-g",))

    k.command("create_generation", "embedding-candidate", {"generation_id": "embedding-new", "candidate": True})
    start = datetime.now(timezone.utc)-timedelta(seconds=1)
    k.command("begin_maintenance_window", "embedding-window", {"window_id": "embedding-window",
        "kind": "monthly_reboot", "starts_at": start.isoformat(),
        "ends_at": (start+timedelta(hours=1)).isoformat(), "operator_nonce": "test", "expected_revision": 0})
    claim(k, job("candidate-vector-b", "model-b", generation="embedding-new"), "embedding-new-claim")
    assert k.get_generation("embedding-new")["retrieval_contract"] == vector_recipe("model-b")["embedding_contract"]
    assert k.get_generation("embedding-g")["retrieval_contract"] == expected


@pytest.mark.parametrize("counter", [-1, None, 2**31])
def test_invalid_imported_failure_counter_does_not_block_first_or_unrelated_intake(repo, counter):
    from papervault.store.knowledge import ClaimRejected, KnowledgeRepository
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "p4-schema"))
    importer = LegacyImportRepository(repo)
    importer.ensure_schema(audit=AuditEvent("test", "legacy-schema"))
    for key in ("A", "B"):
        repo.upsert(Record.from_payload(key, {"key": key, "title": key}), expected_revision=0,
                    audit=AuditEvent("test", "record"))
    sources = [{"workspace": "probe", "ingest_source": "paper", "source_id": key,
                "doc_id": "doc:"+key, "status": "error_parked" if key == "A" else "error",
                "attempts": counter if key == "A" else 1} for key in ("A", "B")]
    mappings = [{"workspace": "probe", "ingest_source": "paper", "source_id": key,
                 "unit_id": "paper:"+key, "kind": "paper", "record_key": key} for key in ("A", "B")]
    plan = plan_legacy_import({"source": "ledger-test", "capture_id": "capture1", "checkpoint": 1,
                               "rows": sources}, mappings=mappings)
    importer.apply(plan, operation_id="invalid-counter-import")
    repo.connection.execute(sql.SQL("UPDATE {} SET authority='catalog'").format(repo._table("knowledge_runtime")))
    k = KnowledgeRepository(repo)
    imported_before = k._rows("jobs", kind="distill_legacy_unverified")
    # First registration occurs after import: mapping evidence, rather than a
    # synthesized executable legacy identity, links the newly registered unit.
    for key in ("A", "B"):
        k.command("register_unit", "intake-unit:"+key, {"unit_id": "paper:"+key,
            "kind": "paper", "record_key": key, "identity": {"source": key}})
        payload = {"unit_id": "paper:"+key, "consumed_input_revision": "input:"+key,
                   "manifest": {"metadata_revision": "1", "tier": "fulltext", "artifacts": []}}
        assert k.command("register_input", "intake-input:"+key, payload)["manifest"] == payload["manifest"]
    bad_unit = k.get_unit("paper:A")
    assert bad_unit["quarantine"] is True
    assert bad_unit["quarantine_reason"]["reason"] == "invalid legacy failure checkpoint"
    assert k.get_reducer("paper:A", "input:A") is None
    application = k._one("knowledge_legacy_checkpoint_applications", unit_id="paper:A")
    assert application["evidence"]["counter_state"] == "unknown"
    assert application["evidence"]["reducer_created"] is False
    assert application["evidence"]["legacy_jobs"][0]["detail"]["legacy_consecutive_failures"] == counter
    assert k.get_unit("paper:B")["quarantine"] is False
    assert k.get_reducer("paper:B", "input:B")["failure_streak"] == 1
    assert k._rows("knowledge_attempt_events") == []
    assert k._rows("jobs", kind="distill_legacy_unverified") == imported_before
    recipe = k.command("register_recipe", "intake-recipe", {"recipe": {"schema_version": 1,
        "output_contract": {"fulltext": {"required": ["graph"]}}}})["recipe_digest"]
    k.command("create_generation", "intake-generation", {"generation_id": "g", "candidate": False})
    repo.connection.execute(sql.SQL("UPDATE {} SET authority='catalog'").format(repo._table("knowledge_runtime")))
    for key in ("A", "B"):
        desired = {"unit_id": "paper:"+key, "consumed_input_revision": "input:"+key,
                   "generation_id": "g", "recipe_digest": recipe}
        k.command("set_requirement", "intake-requirement:"+key, {**desired, "expected_revision": 0,
                  "expected_record_revision": repo.get(key).revision})
        job = k.command("enqueue", "intake-job:"+key, desired)
        if key == "A":
            with pytest.raises(ClaimRejected, match="quarantined"):
                claim(k, job, "intake-claim:"+key)
        else:
            assert claim(k, job, "intake-claim:"+key)["attempt_id"]


def prepare_published_section(k, *, paper_parent=False, previous_whole=False):
    from datetime import datetime, timezone

    parent = "paper:A" if paper_parent else "operator:O"
    if paper_parent:
        k.repository.upsert(Record.from_payload("A", {"key": "A", "title": "source"}),
            expected_revision=0, audit=AuditEvent("test", "parent-record"))
    for unit, kind in ((parent, "paper" if paper_parent else "operator_document"), ("section:S", "section")):
        k.command("register_unit", "source-unit:"+unit, {"unit_id": unit, "kind": kind,
            "identity": {"source": unit}, "record_key": "A" if kind == "paper" else None})
    k.command("register_section", "mapping-v1", {"unit_id": "section:S", "parent_unit_id": parent,
        "mapping_revision": 1, "parent_evidence_revision": "parent-bytes", "definition": {"locator": "chapter1"}})
    for unit, input_revision in ((parent, "parent-input"), ("section:S", "section-input")):
        manifest = {"metadata_revision": "m", "tier": "fulltext", "parent_evidence_revision": "parent-bytes",
            "artifacts": [{"artifact_id": "text:"+unit, "sha256": "a"*64, "byte_length": 12,
                           "role": "fulltext_evidence"}]}
        if unit == "section:S":
            manifest.update(section_definition_revision=1, parent_consumed_input_revision="parent-input")
        k.command("register_input", "source-input:"+unit, {"unit_id": unit,
            "consumed_input_revision": input_revision, "manifest": manifest})
    recipe = k.command("register_recipe", "source-recipe", {"recipe": {"schema_version": 1,
        "output_contract": {"fulltext": {"required": ["graph"]}}}})["recipe_digest"]
    k.command("create_generation", "source-generation", {"generation_id": "g", "candidate": False})
    if previous_whole:
        k.command("register_input", "old-whole-input", {"unit_id": parent, "consumed_input_revision": "old-parent-input",
            "manifest": {"metadata_revision": "old", "tier": "fulltext", "artifacts": []}})
        k.command("register_inventory", "old-whole-inventory", {"parent_unit_id": parent, "inventory_revision": "old-whole",
            "parent_evidence_revision": "old-parent-bytes", "parent_consumed_input_revision": "old-parent-input",
            "mode": "whole", "required_units": [parent], "optional_units": [], "section_mapping_revisions": {}})
        old_selector = {"parent_unit_id": parent, "generation_id": "g", "inventory_revision": "old-whole", "expected_revision": 0}
        old_required = {"unit_id": parent, "generation_id": "g", "inventory_revision": "old-whole",
            "consumed_input_revision": "old-parent-input", "recipe_digest": recipe, "expected_revision": 0}
        if paper_parent:
            old_selector["expected_record_revision"] = old_required["expected_record_revision"] = k.repository.get("A").revision
        k.command("set_inventory", "old-whole-selector", old_selector)
        k.command("set_requirement", "old-whole-desired", old_required)
    inventory = {"parent_unit_id": parent, "inventory_revision": "v1", "parent_evidence_revision": "parent-bytes",
        "parent_consumed_input_revision": "parent-input", "mode": "sections", "required_units": ["section:S"],
        "optional_units": [], "section_mapping_revisions": {"section:S": 1}}
    k.command("register_inventory", "source-inventory", inventory)
    selector = {"parent_unit_id": parent, "generation_id": "g", "inventory_revision": "v1", "expected_revision": 1 if previous_whole else 0}
    if paper_parent:
        selector["expected_record_revision"] = k.repository.get("A").revision
    k.command("set_inventory", "source-current-inventory", selector)
    identity = {"unit_id": "section:S", "generation_id": "g", "consumed_input_revision": "section-input", "recipe_digest": recipe}
    k.command("set_requirement", "source-required", {**identity, "inventory_revision": "v1", "expected_revision": 0})
    c = claim(k, k.command("enqueue", "source-job", identity), "source-claim")
    rec = seal(k, c)
    publication = k.command("publish_derivation", "source-publish", {**c, "receipt": rec,
        "expected_revision": c["revision"], "expected_selection_revision": 0, "select": True})
    stamp = datetime.now(timezone.utc)
    for unit, input_revision in ((parent, "parent-input"), ("section:S", "section-input")):
        artifact = k.get_input(input_revision)["manifest"]["artifacts"][0]
        k.command("observe", "source-input-observation:"+unit, {"observation_id": "source-input:"+unit,
            "unit_id": unit, "consumed_input_revision": input_revision, "role": "input_evidence", "store": "library",
            "output_id": artifact["artifact_id"], "state": "present", "observed_at": stamp.isoformat(), "detail": artifact})
    output = rec["output_inventory"][0]
    k.command("observe", "source-output-observation", {**c, "receipt_id": rec["receipt_id"],
        "observation_id": "source-output", "role": "graph", "store": output["store"], "output_id": output["output_id"],
        "state": "present", "observed_at": stamp.isoformat()})
    return parent, inventory, publication


def test_changed_inventory_atomically_invalidates_published_section_before_requirement_recapture(knowledge):
    from papervault.store.coverage import query_coverage

    k = knowledge
    parent, inventory, publication = prepare_published_section(k)
    assert next(e for e in k.get_serving_manifest()["entries"] if e["unit_id"] == "section:S")["eligible"]
    old_manifest_revision = k.get_generation("g")["manifest_revision"]
    old_manifest = k.get_manifest("g", old_manifest_revision)
    old_receipt = k.get_receipt(k.get_derivation(publication["derivation_id"])["receipt_id"])
    k.command("register_section", "mapping-v2", {"unit_id": "section:S", "parent_unit_id": parent,
        "mapping_revision": 2, "parent_evidence_revision": "parent-bytes", "definition": {"locator": "chapter1 revised"}})
    k.command("register_inventory", "inventory-v2", {**inventory, "inventory_revision": "v2",
        "section_mapping_revisions": {"section:S": 2}})
    k.command("set_inventory", "select-v2-crash-before-input", {"parent_unit_id": parent, "generation_id": "g",
        "inventory_revision": "v2", "expected_revision": 1})
    requirement = k.get_requirement("section:S", "g")
    assert requirement["invalidated"] is True and requirement["inventory_revision"] == "v1"
    assert not next(e for e in k.get_serving_manifest()["entries"] if e["unit_id"] == "section:S")["eligible"]
    assert not next(u for u in query_coverage(k)["units"] if u["unit_id"] == "section:S")["source_eligible"]
    assert k.get_selection("section:S", "g")["derivation_id"] == publication["derivation_id"]
    assert k.get_manifest("g", old_manifest_revision) == old_manifest
    assert k.get_receipt(old_receipt["receipt_id"]) == old_receipt


@pytest.mark.parametrize("parent_change", ["operator_quarantine", "paper_policy", "paper_quarantine"])
def test_section_serving_rechecks_parent_quarantine_and_paper_policy(knowledge, parent_change):
    from papervault.store.coverage import query_coverage

    k = knowledge
    parent, _, publication = prepare_published_section(k, paper_parent=parent_change.startswith("paper"))
    assert next(e for e in k.get_serving_manifest()["entries"] if e["unit_id"] == "section:S")["eligible"]
    if parent_change == "operator_quarantine":
        k.command("set_quarantine", "parent-quarantine", {"unit_id": parent, "quarantine": True,
            "reason": {"reason": "source withheld", "scope": "unit", "origin": "operator"},
            "expected_revision": k.get_unit(parent)["revision"]})
    else:
        record = k.repository.get("A")
        payload = {"key": "A", "title": "source", "policy": "excluded" if parent_change == "paper_policy" else "clear",
                   "quarantine": parent_change == "paper_quarantine"}
        k.repository.upsert(Record.from_payload("A", payload), expected_revision=record.revision,
                            audit=AuditEvent("test", "parent-policy-change"))
    covered = next(u for u in query_coverage(k)["units"] if u["unit_id"] == "section:S")
    assert covered["source_eligible"] is False
    assert not next(e for e in k.get_serving_manifest()["entries"] if e["unit_id"] == "section:S")["eligible"]
    assert k.get_selection("section:S", "g")["derivation_id"] == publication["derivation_id"]


def test_shadow_capture_does_not_seed_before_final_parked_checkpoint_handoff(repo):
    from papervault.store.knowledge import ClaimRejected, KnowledgeRepository
    from papervault.store.knowledge_cutover import CutoverRepository, DRAIN_FIELDS, VALIDATION_FIELDS
    from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "p4-schema"))
    imports = LegacyImportRepository(repo)
    imports.ensure_schema(audit=AuditEvent("test", "legacy-schema"))
    repo.upsert(Record.from_payload("A", {"key": "A", "title": "source"}), expected_revision=0,
                audit=AuditEvent("test", "source-record"))
    k = KnowledgeRepository(repo)
    mapping = {"workspace": "probe", "ingest_source": "paper", "source_id": "A",
               "unit_id": "paper:A", "kind": "paper", "record_key": "A"}
    original = {"workspace": "probe", "ingest_source": "paper", "source_id": "A",
                "doc_id": "doc:A", "status": "error", "attempts": 0}
    snapshot = {"source": "ledger", "capture_id": "capture", "checkpoint": 0, "rows": [original]}
    imports.apply(plan_legacy_import(snapshot, mappings=[mapping]), operation_id="import-C0")
    k.command("register_unit", "unit", {"unit_id": "paper:A", "kind": "paper", "record_key": "A", "identity": "A"})
    k.command("register_input", "shadow-input-C0", {"unit_id": "paper:A", "consumed_input_revision": "input",
        "manifest": {"metadata_revision": "m", "tier": "fulltext", "artifacts": []}})
    assert k.get_reducer("paper:A", "input") is None
    assert k._one("knowledge_legacy_checkpoint_applications", unit_id="paper:A") is None
    assert k.command("migrate_legacy_checkpoint", "shadow-explicit-migrate", {
        "unit_id": "paper:A", "consumed_input_revision": "input"}) == {"applied": False, "deferred": "legacy_authority"}
    parked = {**original, "status": "error_parked", "attempts": 3}
    final_plan = plan_legacy_import(snapshot, mappings=[mapping], changes=[{
        "sequence": 1, "operation": "UPDATE", "old": original, "new": parked}], final_checkpoint=1)
    imports.apply(final_plan, operation_id="import-C1")
    k.command("create_generation", "active", {"generation_id": "g", "candidate": False})
    checkpoint = final_plan["checkpoint"]
    handoff = {"source": "ledger", "capture_id": "capture", "final_sequence": 1,
        "source_fence": {"capture_id": "capture", "final_sequence": 1, "fenced": True, "legacy_dml_revoked": True,
            "fence_id": "test-source-fence", "revoked_roles": ["legacy"], "permission_checks": [{"role": "legacy", "can_write": False}],
            "final_row_count": checkpoint["final_row_count"], "final_rows_digest": checkpoint["final_rows_digest"]},
        "drain": {**{field: True for field in DRAIN_FIELDS}, "receipt": "test-isolated-backend-fence"},
        "validation": {**{field: True for field in VALIDATION_FIELDS}, "receipt": "test-validation"}}
    assert CutoverRepository(repo).commit_handoff(handoff, expected_epoch=1, operation_id="commit-C1")["authority_epoch"] == 2
    digest = k.command("register_recipe", "recipe", {"recipe": {"schema_version": 1,
        "output_contract": {"fulltext": {"required": ["graph"]}}}})["recipe_digest"]
    desired = {"unit_id": "paper:A", "generation_id": "g", "consumed_input_revision": "input", "recipe_digest": digest}
    k.command("set_requirement", "desired", {**desired, "expected_revision": 0, "expected_record_revision": repo.get("A").revision})
    job = k.command("enqueue", "enqueue", desired)
    payload = {"job_id": job["job_id"], "expected_revision": job["revision"], "authority_epoch": 2}
    blocked = k.command("claim", "claim-final", payload)
    assert blocked["claimed"] is False and blocked["parked"] is True and blocked["attempt_id"] is None
    assert blocked["revision"] == job["revision"]+1
    assert k.command("claim", "claim-final", payload) == blocked
    applied = k._one("knowledge_legacy_checkpoint_applications", unit_id="paper:A")
    assert applied["evidence"]["legacy_jobs"][0]["detail"]["source_version"] == 1
    reducer = k.get_reducer("paper:A", "input")
    assert reducer["parked"] and reducer["failure_streak"] == 3
    with pytest.raises(ClaimRejected, match="parked"):
        k.command("claim", "claim-final-retry", {**payload, "expected_revision": blocked["revision"]})
    assert k.get_job(job["job_id"])["attempts"] == 0
    assert k._rows("knowledge_attempt_events") == []
    assert k.get_reducer("paper:A", "input") == reducer
    assert k.list_admission_jobs() == []


@pytest.mark.parametrize("paper_parent", [False, True])
def test_section_inventory_parent_evidence_replaces_excluded_whole_requirement_in_read_view(knowledge, paper_parent):
    from papervault.store.coverage import query_coverage

    k = knowledge
    parent, _, _ = prepare_published_section(k, paper_parent=paper_parent, previous_whole=True)
    historical = k.get_requirement(parent, "g")
    assert historical["excluded"] and historical["consumed_input_revision"] == "old-parent-input"
    row = next(row for row in k.coverage_rows(parent) if row["unit"]["unit_id"] == parent)
    assert row["requirement"]["inventory_parent_selector"] is True
    assert row["requirement"]["consumed_input_revision"] == "parent-input"
    parent_coverage = next(unit for unit in query_coverage(k)["units"] if unit["unit_id"] == parent)
    assert parent_coverage["readable"] is True
    serving = {entry["unit_id"]: entry for entry in k.get_serving_manifest()["entries"]}
    assert serving[parent]["eligible"] is False
    assert serving["section:S"]["eligible"] is True
    assert k.get_requirement(parent, "g") == historical


def test_admission_limit_does_not_starve_active_job_behind_closed_candidate_jobs(knowledge):
    k = knowledge
    digest = k.command("register_recipe", "candidate-recipe", {"recipe": {"schema_version": 1,
        "output_contract": {"fulltext": {"required": ["graph"]}}}})["recipe_digest"]
    k.command("create_generation", "closed-candidate", {"generation_id": "closed", "candidate": True})
    # Fixture population is intentionally larger than the scheduler's first page.
    for number in range(101):
        unit = "old-candidate:"+str(number)
        input_revision = "input:"+unit
        k._insert("units", {"unit_id": unit, "kind": "operator_document", "identity": {"source": unit}})
        k._insert("input_manifests", {"unit_id": unit, "consumed_input_revision": input_revision,
            "manifest": {"metadata_revision": "1", "tier": "fulltext", "artifacts": []}, "manifest_digest": input_revision})
        k._insert("knowledge_requirements", {"unit_id": unit, "generation_id": "closed",
            "consumed_input_revision": input_revision, "recipe_digest": digest})
        k._insert("jobs", {"id": "job:"+unit, "kind": "distill", "state": "queued", "detail": {},
            "unit_id": unit, "consumed_input_revision": input_revision, "recipe_digest": digest, "target_generation": "closed"})
    active = prepare(k)
    rows = k.list_admission_jobs(limit=100)
    assert [row["job_id"] for row in rows] == [active["job_id"]]
    assert claim(k, rows[0], "claim-active-after-closed-page")["attempt_id"]
