"""Transactional reducers for the P4 knowledge repository.

All functions execute inside the repository's schema writer transaction. The
advisory writer lock protects multi-row decisions; exact row revisions still
reject stale clients. Outputs are always private until an accepted publication.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from uuid import uuid4

from psycopg import sql
from psycopg.types.json import Jsonb

from .knowledge import CLAIM_FIELDS, OBSERVATION_IDENTITY_FIELDS, ClaimRejected, canonical_recipe, requires_vectors, retrieval_contract, validate_input_manifest, validate_receipt
from .repository import payload_hash


def identifier(prefix):
    return prefix + ":" + uuid4().hex


def require(value, reason):
    if not value:
        raise ClaimRejected(reason)
    return value


def now(k):
    return k.connection.execute("SELECT clock_timestamp() AS now").fetchone()["now"]


def state(k):
    return k._one("knowledge_runtime", singleton=True)


def revision(row, payload, field="expected_revision"):
    require(payload.get(field) == (row["revision"] if row else 0), "optimistic revision mismatch: " + field)


def bump_requirements(k):
    return k._update("knowledge_runtime", {"requirements_revision": state(k)["requirements_revision"] + 1},
                     singleton=True)["requirements_revision"]


def immutable(k, table, keys, values):
    old = k._one(table, **keys)
    if old:
        require(all(old[key] == value for key, value in values.items()), "immutable identity already has different contents")
        return old
    return k._insert(table, {**keys, **values})


def outbox(k, kind, payload, submission_id=None, stable_id=None):
    return k._insert("knowledge_outbox", {"outbox_id": stable_id or identifier("outbox"),
                 "kind": kind, "submission_id": submission_id, "payload": payload})


def supersession(k, derivation_id, event_name):
    previous = k.connection.execute(sql.SQL("SELECT event FROM {} WHERE derivation_id=%s ORDER BY id DESC LIMIT 1").format(
        k.repository._table("supersession_events")), (derivation_id,)).fetchone()
    if not previous or previous["event"] != event_name:
        k._insert("supersession_events", {"derivation_id": derivation_id, "event": event_name})


def snapshot(k, generation_id, aggregate_receipt=None):
    generation = k.get_generation(generation_id)
    rows = k.connection.execute(sql.SQL(
        "SELECT q.*,s.derivation_id AS selected_derivation,s.revision AS selection_revision,"
        "d.consumed_input_revision AS selected_input,d.recipe_digest AS selected_recipe,d.generation_id AS selected_generation "
        "FROM {} q LEFT JOIN {} s ON s.unit_id=q.unit_id AND s.generation_id=q.generation_id "
        "LEFT JOIN {} d USING(derivation_id) WHERE q.generation_id=%s ORDER BY q.unit_id"
    ).format(k.repository._table("knowledge_requirements"), k.repository._table("selections"),
        k.repository._table("derivations")), (generation_id,))
    entries = []
    auxiliary = {"selected_derivation", "selection_revision", "selected_input", "selected_recipe", "selected_generation"}
    for row in rows:
        requirement = {key: value for key, value in row.items() if key not in auxiliary}
        compatible = bool(row["selected_derivation"] and not row["invalidated"] and not row["excluded"] and
                          row["selected_input"] == row["consumed_input_revision"] and
                          row["selected_recipe"] == row["recipe_digest"] and row["selected_generation"] == generation_id)
        entries.append({**requirement, "selected_derivation_id": row["selected_derivation"] if compatible else None,
                        "selection_revision": row["selection_revision"] or 0,
                        "missing": not compatible and not row["excluded"]})
    manifest_revision = generation["manifest_revision"] + 1
    requirements_revision = state(k)["requirements_revision"]
    previous = k.get_manifest(generation_id) if generation["manifest_revision"] else None
    aggregate = aggregate_receipt or (previous["manifest"] if previous else {})
    selected_receipts = aggregate.get("selected_unit_receipts", {})
    inventories = k.connection.execute(sql.SQL(
        "SELECT i.*,c.revision AS selection_revision,c.invalidated,c.record_revision,c.enqueue_mode,c.request_id,c.desired_recipe_digest "
        "FROM {} c JOIN {} i USING(parent_unit_id,inventory_revision) WHERE c.generation_id=%s"
    ).format(k.repository._table("current_inventories"), k.repository._table("unit_inventories")), (generation_id,))
    expected_inventories = {row["parent_unit_id"]: row for row in inventories}
    previous_selections = {e["selected_derivation_id"] for e in previous["manifest"]["entries"]
                           if e.get("selected_derivation_id")} if previous else set()
    current_selections = {e["selected_derivation_id"] for e in entries if e["selected_derivation_id"]}
    for derivation_id in sorted(previous_selections-current_selections):
        supersession(k, derivation_id, "superseded")
    k._insert("generation_manifests", {"generation_id": generation_id, "manifest_revision": manifest_revision,
        "requirements_revision": requirements_revision, "manifest": {"entries": entries,
        "aggregate_layers": aggregate.get("aggregate_layers", []), "expected_inventories": expected_inventories,
        "retrieval_contract": generation["retrieval_contract"],
        "selected_unit_receipts": selected_receipts, "selected_unit_receipts_digest": payload_hash(selected_receipts)}})
    k._update("generations", {"manifest_revision": manifest_revision,
        "requirements_revision": requirements_revision}, generation_id=generation_id)
    if generation["state"] == "active":
        runtime = state(k)
        k._update("knowledge_runtime", {"serving_generation_id": generation_id,
            "serving_manifest_revision": manifest_revision, "serving_revision": runtime["serving_revision"] + 1},
            singleton=True)
    return manifest_revision


def job_identity(job):
    return {"job_id": job["id"], "unit_id": job["unit_id"],
            "consumed_input_revision": job["consumed_input_revision"], "recipe_digest": job["recipe_digest"],
            "generation_id": job["target_generation"]}


def claim_result(job, attempt):
    return {**job_identity(job), **{key: attempt[key] for key in
        ("attempt_id", "submission_id", "authority_epoch", "fencing_token", "lease_id", "namespace",
         "lease_expires_at", "aggregate_base_manifest_revision", "base_aggregate_layers",
         "base_selected_unit_receipts", "retired_receipt_ids")}, "revision": job["revision"]}


def eligible(k, job, *, current=True):
    from .extract_lifecycle import admit_input
    admit_input(k, k.get_input(job["consumed_input_revision"])["manifest"])
    unit = require(k.get_unit(job["unit_id"]), "unit absent")
    require(not unit["quarantine"] and not job["quarantine"], "unit/job quarantined")
    if unit["record_key"]:
        record = require(k.repository.get(unit["record_key"]), "record absent")
        require(not record.quarantine and record.policy in {"clear", "override"} and record.merge == "canonical",
                "record policy prohibits execution/publication")
    generation = require(k.get_generation(job["target_generation"]), "generation absent")
    require(generation["state"] in {"active", "candidate"}, "generation closed")
    desired = k.get_requirement(job["unit_id"], job["target_generation"])
    if unit["kind"] == "section":
        source = k.get_input(job["consumed_input_revision"])["manifest"]
        mapping = k.get_section(unit["unit_id"], source.get("section_definition_revision", source.get("section_mapping_revision")))
        parent = require(k.get_unit(mapping["parent_unit_id"]), "section parent absent")
        require(not parent["quarantine"], "section parent quarantined")
        if current:
            inventory = k.get_current_inventory(parent["unit_id"], job["target_generation"])
            require(inventory and not inventory["invalidated"] and desired and
                    desired["inventory_revision"] == inventory["inventory_revision"], "section inventory no longer desired")
    if current:
        require(desired and not desired["invalidated"] and not desired["excluded"], "desired input invalid or excluded")
        require(generation["state"] == "candidate" or desired["enqueue_mode"] == "active", "desired work requires candidate rebuild")
        require(all(desired[key] == job[key] for key in ("consumed_input_revision", "recipe_digest")),
                "job no longer desired")
    return desired


def generation_contract(k, job, *, tier=None, bind=False):
    recipe = require(k.get_recipe(job["recipe_digest"]), "recipe absent")
    if tier is None:
        tier = k.get_input(job["consumed_input_revision"])["manifest"].get("tier", "fulltext")
    if not requires_vectors(recipe, tier):
        return False
    contract = retrieval_contract(recipe)
    generation = require(k.get_generation(job["target_generation"]), "generation absent")
    if generation["retrieval_contract"] is None and bind:
        k._update("generations", {"retrieval_contract": contract}, generation_id=job["target_generation"])
        return True
    require(generation["retrieval_contract"] == contract,
            "generation retrieval contract differs; a fresh generation is required")
    return False


def valid_claim(k, payload, *, allow_sealed=True, publication=False):
    job = require(k.get_job(payload.get("job_id")), "job absent")
    attempt = require(k.get_attempt(payload.get("attempt_id")), "attempt absent")
    runtime = state(k)
    require(runtime["authority"] == "catalog", "catalog authority not committed")
    require(job["kind"] == "distill" and job["current_attempt_id"] == attempt["attempt_id"], "superseded attempt")
    expected = claim_result(job, attempt)
    for key in CLAIM_FIELDS:
        if key in payload:
            require(payload[key] == expected[key], "claim identity mismatch: " + key)
    require(payload.get("authority_epoch") == runtime["authority_epoch"] == attempt["authority_epoch"], "authority epoch changed")
    require(payload.get("fencing_token") == attempt["fencing_token"] and
            payload.get("lease_id") == attempt["lease_id"], "invalid fence or lease")
    clock = now(k)
    generation = k.get_generation(job["target_generation"])
    if generation["state"] == "candidate":
        window = k.connection.execute(sql.SQL("SELECT window_id FROM {} WHERE starts_at<=clock_timestamp() "
            "AND ends_at>clock_timestamp()").format(k.repository._table("knowledge_maintenance_windows"))).fetchone()
        require(window, "candidate claim outside monthly maintenance window")
    if publication and job["state"] == "succeeded":
        valid_execution = attempt["lease_expires_at"] > clock
        valid_publication = (payload.get("publication_claim_id") == attempt["publication_claim_id"]
                             and attempt["publication_claim_id"] is not None
                             and attempt["publication_lease_expires_at"] > clock)
        require(valid_execution or valid_publication, "publication claim expired")
    else:
        require(job["state"] == "running" and attempt["outcome"] is None and
                attempt["lease_expires_at"] > clock, "execution claim expired or settled")
    require(allow_sealed or not attempt["sealed"], "namespace sealed")
    generation_contract(k, job)
    return job, attempt, expected


def event(k, attempt_id, name, detail):
    k._insert("knowledge_attempt_events", {"attempt_id": attempt_id, "event": name, "detail": detail})


def reduce_outcome(k, job, outcome):
    """Reuse the legacy failure-streak reducer; never replay imported events."""
    from papervault.knowledge.ledger.store import _max_attempts, _next_attempts_status
    old = k.get_reducer(job["unit_id"], job["consumed_input_revision"])
    if not old:
        old = k._insert("knowledge_unit_reducers", {"unit_id": job["unit_id"],
            "consumed_input_revision": job["consumed_input_revision"]})
    status = "done" if outcome == "succeeded" else "error"
    streak, status = _next_attempts_status(old["failure_streak"], job["consumed_input_revision"],
        status, job["consumed_input_revision"], _max_attempts())
    return k._update("knowledge_unit_reducers", {"failure_streak": streak,
        "parked": status == "error_parked", "terminal_outcome": outcome,
        "revision": old["revision"] + 1}, unit_id=job["unit_id"], consumed_input_revision=job["consumed_input_revision"])


def migrate_legacy_checkpoint(k, unit_id, input_revision):
    """Carry final handed-off legacy control effects once, without inventing attempts."""
    # A shadow capture at C0 is not the final C1 handoff checkpoint. Seeding
    # before the legacy writer is fenced would permanently miss later parking.
    if state(k)["authority"] != "catalog":
        return None
    previous = k._one("knowledge_legacy_checkpoint_applications", unit_id=unit_id)
    if previous:
        return previous
    unit = require(k.get_unit(unit_id), "unit absent")
    rows = list(k.connection.execute(sql.SQL(
        "SELECT * FROM {} WHERE kind='distill_legacy_unverified' AND "
        "(unit_id=%s OR detail->'evidence'->'mapping'->>'unit_id'=%s)"
    ).format(k.repository._table("jobs")), (unit_id, unit_id)))
    rows = [r for r in rows if not r["detail"].get("deleted")]
    if not rows:
        return None
    checkpoint_id = "legacy-checkpoint:" + payload_hash([{ "id": r["id"], "detail": r["detail"]}
                                                          for r in sorted(rows, key=lambda r: r["id"])])
    # Imported values are raw evidence, never implicitly cast or defaulted to
    # zero. PostgreSQL's reducer counter is a signed INTEGER; an unrepresentable
    # or ambiguous value cannot safely initialize executable catalog state.
    invalid_counters = [{"legacy_job_id": r["id"], "raw_counter": r["detail"].get("legacy_consecutive_failures"),
        "reason": "invalid legacy failure checkpoint", "scope": "unit", "origin": "legacy-import"}
        for r in rows if type(r["detail"].get("legacy_consecutive_failures")) is not int
        or not 0 <= r["detail"]["legacy_consecutive_failures"] <= 2147483647]
    streak = None if invalid_counters else max(r["detail"]["legacy_consecutive_failures"] for r in rows)
    parked = any(r["detail"].get("evidence", {}).get("raw", {}).get("status") == "error_parked" for r in rows)
    from papervault.knowledge.ledger.store import _max_attempts
    limit = _max_attempts()
    parked = parked or (limit > 0 and streak is not None and streak >= limit)
    quarantine = list(invalid_counters)
    for row in rows:
        evidence = row["detail"].get("evidence", {})
        raw = evidence.get("raw", {})
        if raw.get("quarantine") or raw.get("status") == "quarantined":
            quarantine.append({"legacy_job_id": row["id"], "raw_quarantine": raw.get("quarantine"),
                "reason": raw.get("quarantine_reason", raw.get("reason", "legacy quarantine")),
                "scope": raw.get("quarantine_scope", "unit"), "origin": "legacy-import"})
    if len(rows) > 1:
        quarantine.append({"reason": "multiple legacy rows map to one unit", "scope": "unit", "origin": "legacy-import"})
    existing = k.get_reducer(unit_id, input_revision)
    require(not existing, "legacy checkpoint must be applied before catalog execution state")
    if not invalid_counters:
        k._insert("knowledge_unit_reducers", {"unit_id": unit_id, "consumed_input_revision": input_revision,
            "failure_streak": streak, "parked": parked, "legacy_checkpoint_id": checkpoint_id})
    if quarantine:
        k._update("units", {"quarantine": True,
            "quarantine_reason": {"reason": "invalid legacy failure checkpoint" if invalid_counters else "preserved legacy quarantine",
                                  "scope": "unit", "origin": "legacy-import", "details": quarantine,
                                  "prior_quarantine_reason": unit.get("quarantine_reason") if unit.get("quarantine") else None}, "revision": unit["revision"]+1}, unit_id=unit_id)
    return k._insert("knowledge_legacy_checkpoint_applications", {"unit_id": unit_id,
        "checkpoint_id": checkpoint_id, "consumed_input_revision": input_revision,
        "evidence": {"legacy_jobs": [{"job_id": r["id"], "detail": r["detail"]} for r in rows],
                     "counter_semantics": "consecutive_build_failures", "historical_events_replayed": False,
                     "counter_state": "unknown" if invalid_counters else "known", "reducer_created": not invalid_counters}})


def abandon_attempt(k, job, attempt):
    require(job["state"] == "running" and job["current_attempt_id"] == attempt["attempt_id"] and
            attempt["outcome"] is None and attempt["lease_expires_at"] <= now(k), "attempt not expired/current")
    k._update("knowledge_attempts", {"outcome": "abandoned", "finished_at": now(k), "cost_state": "unknown"},
              attempt_id=attempt["attempt_id"])
    event(k, attempt["attempt_id"], "abandoned", {"external_outcome": "unknown", "failure_streak_event": True})
    pin = k._one("knowledge_pins", pin_id="attempt-pin:"+attempt["attempt_id"])
    if pin:
        k._update("knowledge_pins", {"kind": "recovery", "revision": pin["revision"]+1}, pin_id=pin["pin_id"])
    checkpoint = reduce_outcome(k, job, "failed")
    job = k._update("jobs", {"state": "failed", "finished_at": now(k), "revision": job["revision"]+1}, id=job["id"])
    return job, checkpoint


def complete(k, payload, *, publish=False):
    job, attempt, claim = valid_claim(k, payload, publication=publish)
    revision(job, payload)
    rec = payload["receipt"]
    recipe = k._one("recipes", recipe_digest=job["recipe_digest"])["spec"]
    validate_receipt(rec, claim, recipe)
    stored = require(k.get_receipt(rec["receipt_id"]), "receipt must first be durably sealed")
    require(stored == rec and attempt["receipt_id"] == rec["receipt_id"], "receipt differs from sealed outputs")
    eligible(k, job, current=False)
    if job["state"] != "succeeded":
        actual_cost = payload.get("actual_cost")
        if actual_cost is not None:
            require(type(actual_cost) in {int, float} and actual_cost >= 0, "invalid actual cost")
            actual_cost = Decimal(str(actual_cost))
        k._update("knowledge_attempts", {"outcome": "succeeded", "finished_at": now(k),
                  "actual_cost": actual_cost, "cost_state": "known" if actual_cost is not None else "unknown"},
                  attempt_id=attempt["attempt_id"])
        event(k, attempt["attempt_id"], "succeeded", {"receipt_id": rec["receipt_id"],
              "actual_cost": str(actual_cost) if actual_cost is not None else None})
        pin = k._one("knowledge_pins", pin_id="attempt-pin:"+attempt["attempt_id"])
        if pin:
            k._update("knowledge_pins", {"kind": "recovery", "revision": pin["revision"]+1}, pin_id=pin["pin_id"])
        if actual_cost is not None:
            runtime = state(k)
            retained = max(actual_cost, 0)
            k._update("knowledge_runtime", {"budget_reserved": runtime["budget_reserved"] - attempt["reserved_cost"] + retained},
                      singleton=True)
        job = k._update("jobs", {"state": "succeeded", "achieved_tier": rec["achieved_tier"],
                "finished_at": now(k), "revision": job["revision"] + 1}, id=job["id"])
        reduce_outcome(k, job, "succeeded")
    return job, attempt, claim


def dispatch(k, action, operation_id, p):
    from .extract_lifecycle import ACTIONS, dispatch_lifecycle, admit_input, admit_manifest
    if action in ACTIONS:
        return dispatch_lifecycle(k, action, p)
    if action == "register_unit":
        kind = p["kind"]
        require(kind in {"paper", "operator_document", "section"}, "unknown unit kind")
        require(bool(p["unit_id"]) and p.get("identity") is not None, "stable typed identity required")
        unit = immutable(k, "units", {"unit_id": p["unit_id"]},
              {"kind": kind, "record_key": p.get("record_key"), "identity": p["identity"]})
        for mapping in p.get("legacy_keys", []):
            mapping = {"source": "legacy", "legacy_key": mapping} if isinstance(mapping, str) else mapping
            immutable(k, "unit_mappings", {"source": mapping["source"], "legacy_key": mapping["legacy_key"]},
                      {"unit_id": unit["unit_id"]})
        return unit

    if action == "register_section":
        require(k.get_unit(p["unit_id"])["kind"] == "section", "section requires typed section unit")
        require(k.get_unit(p["parent_unit_id"])["kind"] in {"paper", "operator_document"}, "invalid section parent")
        require(bool(p["parent_evidence_revision"]) and bool(p["definition"]), "section requires precise parent and definition")
        return immutable(k, "section_mappings", {"unit_id": p["unit_id"], "mapping_revision": p["mapping_revision"]},
            {key: p[key] for key in ("parent_unit_id", "parent_evidence_revision", "definition")})

    if action == "register_inventory":
        required, optional = p["required_units"], p.get("optional_units", [])
        require(len(set(required + optional)) == len(required + optional), "duplicate inventory units")
        require(p["mode"] in {"whole", "sections"}, "inventory mode invalid")
        if p.get("parent_consumed_input_revision"):
            parent_input = k.get_input(p["parent_consumed_input_revision"])
            require(parent_input and parent_input["unit_id"] == p["parent_unit_id"], "inventory parent input mismatch")
            admit_input(k, parent_input["manifest"])
        if p["mode"] == "whole":
            require(required == [p["parent_unit_id"]] and not optional, "whole inventory must contain parent exactly once")
        else:
            require(p["parent_unit_id"] not in required + optional, "whole and sections cannot count together")
            for unit_id in required + optional:
                mapping = k.get_section(unit_id, p["section_mapping_revisions"].get(unit_id))
                require(mapping and mapping["parent_unit_id"] == p["parent_unit_id"] and
                        mapping["parent_evidence_revision"] == p["parent_evidence_revision"], "section inventory mismatch")
        return immutable(k, "unit_inventories", {"parent_unit_id": p["parent_unit_id"], "inventory_revision": p["inventory_revision"]},
            {"parent_evidence_revision": p["parent_evidence_revision"], "mode": p["mode"], "required_units": required,
             "parent_consumed_input_revision": p.get("parent_consumed_input_revision"),
             "optional_units": optional, "section_mapping_revisions": p.get("section_mapping_revisions", {})})

    if action == "set_inventory":
        inventory = require(k.get_inventory(p["parent_unit_id"], p["inventory_revision"]), "inventory missing")
        if inventory.get("parent_consumed_input_revision"):
            admit_input(k, k.get_input(inventory["parent_consumed_input_revision"])["manifest"])
        old = k._one("current_inventories", parent_unit_id=p["parent_unit_id"], generation_id=p["generation_id"])
        revision(old, p)
        require(k.get_generation(p["generation_id"])["state"] in {"active", "candidate"}, "generation closed")
        parent = k.get_unit(p["parent_unit_id"])
        record = k.repository.get(parent["record_key"]) if parent["record_key"] else None
        if record:
            require(p.get("expected_record_revision") == record.revision, "captured parent record revision changed")
        mode = p.get("enqueue_mode", "active")
        require(mode in {"active", "candidate"}, "unknown inventory enqueue mode")
        request_id = p.get("request_id")
        recipe_digest = p.get("desired_recipe_digest") or (old.get("desired_recipe_digest") if old else None)
        if old and old["enqueue_mode"] == "candidate" and old["inventory_revision"] == p["inventory_revision"] and mode == "active":
            mode, request_id, recipe_digest = "candidate", old["request_id"], old["desired_recipe_digest"]
        if mode == "candidate":
            require(request_id and recipe_digest and k.get_recipe(recipe_digest), "candidate intent requires stable request_id and desired_recipe_digest")
        request_id = request_id or "inventory-request:"+payload_hash([p["parent_unit_id"], p["generation_id"], p["inventory_revision"]])
        values = {"inventory_revision": p["inventory_revision"], "revision": old["revision"]+1 if old else 1,
                  "record_revision": record.revision if record else None, "invalidated": False,
                  "enqueue_mode": mode, "request_id": request_id, "desired_recipe_digest": recipe_digest}
        if old:
            row = k._update("current_inventories", values, parent_unit_id=p["parent_unit_id"], generation_id=p["generation_id"])
        else:
            row = k._insert("current_inventories", {"parent_unit_id": p["parent_unit_id"], "generation_id": p["generation_id"], **values})
        required = set(inventory["required_units"])
        children = {r["unit_id"] for r in k._rows("section_mappings", parent_unit_id=p["parent_unit_id"])}
        children.add(p["parent_unit_id"])
        for desired in k._rows("knowledge_requirements", generation_id=p["generation_id"]):
            if desired["unit_id"] in children:
                if desired["unit_id"] in required:
                    admit_input(k, k.get_input(desired["consumed_input_revision"])["manifest"])
                k._update("knowledge_requirements", {"excluded": desired["unit_id"] not in required,
                          "invalidated": desired["invalidated"] or (desired["unit_id"] in required and
                              desired["inventory_revision"] != inventory["inventory_revision"]),
                          "enqueue_mode": mode, "revision": desired["revision"]+1},
                          unit_id=desired["unit_id"], generation_id=p["generation_id"])
        bump_requirements(k)
        snapshot(k, p["generation_id"])
        outbox(k, "inventory_changed", {"parent_unit_id": p["parent_unit_id"], "generation_id": p["generation_id"],
               "inventory_revision": p["inventory_revision"], "enqueue_mode": mode, "request_id": request_id,
               "desired_recipe_digest": recipe_digest})
        return row

    if action == "register_input":
        validate_input_manifest(p["manifest"])
        admit_input(k, p["manifest"])
        unit = require(k.get_unit(p["unit_id"]), "unit absent")
        if unit["kind"] == "section":
            mapping_revision = p["manifest"].get("section_definition_revision", p["manifest"].get("section_mapping_revision"))
            require(mapping_revision is not None, "section input requires exact section definition revision")
            mapping = k.get_section(unit["unit_id"], mapping_revision)
            require(mapping and mapping["parent_evidence_revision"] == p["manifest"].get("parent_evidence_revision"),
                    "section input requires exact mapping and parent evidence")
        row = immutable(k, "input_manifests", {"consumed_input_revision": p["consumed_input_revision"]},
            {"unit_id": p["unit_id"], "manifest": p["manifest"], "manifest_digest": payload_hash(p["manifest"])})
        migrate_legacy_checkpoint(k, p["unit_id"], p["consumed_input_revision"])
        return row

    if action == "migrate_legacy_checkpoint":
        require(k.get_input(p["consumed_input_revision"])["unit_id"] == p["unit_id"], "checkpoint input mismatch")
        if state(k)["authority"] != "catalog":
            return {"applied": False, "deferred": "legacy_authority"}
        return migrate_legacy_checkpoint(k, p["unit_id"], p["consumed_input_revision"]) or {"applied": False}

    if action == "register_recipe":
        spec, digest = canonical_recipe(p["recipe"])
        retrieval_contract(spec)
        require(p.get("recipe_digest", digest) == digest, "recipe digest differs from effective specification")
        immutable(k, "recipes", {"recipe_digest": digest}, {"spec": spec})
        return {"recipe_digest": digest, "recipe": spec}

    if action == "create_generation":
        candidate = p.get("candidate", True)
        require(type(p.get("sample", False)) is bool, "generation sample marker must be boolean")
        if not candidate:
            require(not k._rows("generations", state="active"), "active generation already exists; use promotion")
        immutable(k, "generations", {"generation_id": p["generation_id"]},
            {"state": "candidate" if candidate else "active", "requirements_revision": state(k)["requirements_revision"],
             "sample": p.get("sample", False)})
        if not k.get_generation(p["generation_id"])["manifest_revision"]:
            snapshot(k, p["generation_id"])
        return k.get_generation(p["generation_id"])

    if action == "fork_generation":
        source = require(k.get_generation(p["source_generation_id"]), "source generation absent")
        runtime = state(k)
        require(source["state"] == "active" and p["expected_manifest_revision"] == source["manifest_revision"] and
                p["expected_requirements_revision"] == runtime["requirements_revision"], "fork source requirements changed")
        require(not k.get_generation(p["generation_id"]), "fork requires a fresh generation")
        admit_manifest(k, k.get_manifest(source["generation_id"])["manifest"])
        k._insert("generations", {"generation_id": p["generation_id"], "state": "candidate",
            "requirements_revision": runtime["requirements_revision"], "source_generation_id": source["generation_id"],
            "source_manifest_revision": source["manifest_revision"], "sample": source.get("sample", False)})
        k.connection.execute(sql.SQL("INSERT INTO {}(unit_id,generation_id,consumed_input_revision,recipe_digest,"
            "record_revision,inventory_revision,invalidated,excluded,enqueue_mode) SELECT unit_id,%s,consumed_input_revision,recipe_digest,"
            "record_revision,inventory_revision,invalidated,excluded,enqueue_mode FROM {} WHERE generation_id=%s").format(
            k.repository._table("knowledge_requirements"), k.repository._table("knowledge_requirements")),
            (p["generation_id"], source["generation_id"]))
        k.connection.execute(sql.SQL("INSERT INTO {}(parent_unit_id,generation_id,inventory_revision,record_revision,invalidated,"
            "enqueue_mode,request_id,desired_recipe_digest) SELECT parent_unit_id,%s,inventory_revision,record_revision,invalidated,"
            "enqueue_mode,request_id,desired_recipe_digest FROM {} WHERE generation_id=%s").format(
            k.repository._table("current_inventories"), k.repository._table("current_inventories")),
            (p["generation_id"], source["generation_id"]))
        snapshot(k, p["generation_id"])
        cursor = k.connection.execute(sql.SQL("INSERT INTO {}(id,kind,key,state,detail,unit_id,consumed_input_revision,"
            "recipe_digest,target_generation) SELECT 'distill-fork:'||md5(jsonb_build_array(q.unit_id,q.consumed_input_revision,"
            "q.recipe_digest,q.generation_id)::text),'distill',u.record_key,'queued','{{}}'::jsonb,q.unit_id,"
            "q.consumed_input_revision,q.recipe_digest,q.generation_id FROM {} q JOIN {} u USING(unit_id) "
            "WHERE q.generation_id=%s AND NOT q.excluded").format(k.repository._table("jobs"),
                k.repository._table("knowledge_requirements"), k.repository._table("units")), (p["generation_id"],))
        outbox(k, "generation_forked", {"generation_id": p["generation_id"], "source_generation_id": source["generation_id"]})
        return {**k.get_generation(p["generation_id"]), "queued_jobs": cursor.rowcount}

    if action == "set_requirement":
        admit_input(k, k.get_input(p["consumed_input_revision"])["manifest"])
        unit = require(k.get_unit(p["unit_id"]), "unit absent")
        generation = require(k.get_generation(p["generation_id"]), "generation absent")
        require(generation["state"] in {"active", "candidate"}, "generation closed")
        require(k._one("input_manifests", consumed_input_revision=p["consumed_input_revision"])["unit_id"] == p["unit_id"],
                "input belongs to another unit")
        require(k._one("recipes", recipe_digest=p["recipe_digest"]), "recipe absent")
        inventory_selector = k.get_current_inventory(unit["unit_id"], p["generation_id"])
        if unit["kind"] == "section":
            source = k.get_input(p["consumed_input_revision"])["manifest"]
            mapping = k.get_section(unit["unit_id"], source.get("section_definition_revision", source.get("section_mapping_revision")))
            current_inventory = k.get_current_inventory(mapping["parent_unit_id"], p["generation_id"])
            inventory_selector = current_inventory
            require(current_inventory and current_inventory["inventory_revision"] == p.get("inventory_revision") and
                    unit["unit_id"] in current_inventory["required_units"] + current_inventory["optional_units"] and
                    current_inventory["section_mapping_revisions"].get(unit["unit_id"]) == mapping["mapping_revision"] and
                    current_inventory["parent_evidence_revision"] == mapping["parent_evidence_revision"],
                    "section requirement differs from generation current inventory")
        elif inventory_selector and inventory_selector["mode"] == "sections":
            require(p.get("excluded") is True, "section-built parent cannot also be an executable whole unit")
        old = k.get_requirement(p["unit_id"], p["generation_id"])
        revision(old, p)
        record = k.repository.get(unit["record_key"]) if unit["record_key"] else None
        if record:
            require(p.get("expected_record_revision") == record.revision,
                    "captured record revision changed; recapture exact inputs")
        values = {"consumed_input_revision": p["consumed_input_revision"], "recipe_digest": p["recipe_digest"],
                  "record_revision": record.revision if record else None, "inventory_revision": p.get("inventory_revision"),
                  "revision": old["revision"] + 1 if old else 1, "invalidated": False, "excluded": p.get("excluded", False),
                  "enqueue_mode": inventory_selector["enqueue_mode"] if inventory_selector else p.get("enqueue_mode", "active")}
        row = k._update("knowledge_requirements", values, unit_id=p["unit_id"], generation_id=p["generation_id"]) if old else k._insert(
            "knowledge_requirements", {"unit_id": p["unit_id"], "generation_id": p["generation_id"], **values})
        bump_requirements(k)
        snapshot(k, p["generation_id"])
        outbox(k, "requirement_changed", {"unit_id": p["unit_id"], "generation_id": p["generation_id"],
            "enqueue_mode": values["enqueue_mode"], "request_id": inventory_selector["request_id"] if inventory_selector else None})
        return row

    if action == "enqueue":
        admit_input(k, k.get_input(p["consumed_input_revision"])["manifest"])
        job = k._one("jobs", kind="distill", unit_id=p["unit_id"], consumed_input_revision=p["consumed_input_revision"],
                     recipe_digest=p["recipe_digest"], target_generation=p["generation_id"])
        if job:
            return {**job_identity(job), "revision": job["revision"]}
        unit = require(k.get_unit(p["unit_id"]), "unit absent")
        require(k.get_input(p["consumed_input_revision"])["unit_id"] == unit["unit_id"], "job input belongs to another unit")
        job = k._insert("jobs", {"id": p.get("job_id") or identifier("distill"), "kind": "distill", "key": unit["record_key"],
              "state": "queued", "detail": {}, "unit_id": p["unit_id"], "consumed_input_revision": p["consumed_input_revision"],
              "recipe_digest": p["recipe_digest"], "target_generation": p["generation_id"],
              "predecessor_job_id": p.get("predecessor_job_id")})
        return {**job_identity(job), "revision": job["revision"]}

    if action == "claim":
        job = require(k.get_job(p["job_id"]), "job absent")
        revision(job, p)
        require(job["kind"] == "distill", "legacy/unverified jobs cannot execute")
        require(state(k)["authority"] == "catalog", "catalog authority not committed")
        require(p.get("authority_epoch") == state(k)["authority_epoch"], "authority epoch mismatch")
        previous_checkpoint = k._one("knowledge_legacy_checkpoint_applications", unit_id=job["unit_id"])
        applied_checkpoint = migrate_legacy_checkpoint(k, job["unit_id"], job["consumed_input_revision"])
        if applied_checkpoint and not previous_checkpoint:
            checkpoint = k.get_reducer(job["unit_id"], job["consumed_input_revision"])
            parked = bool(checkpoint and checkpoint["parked"])
            quarantined = bool(k.get_unit(job["unit_id"])["quarantine"])
            if parked or quarantined:
                # Commit the learned handoff control effect without creating an
                # execution. A revision advance lets an explicit later reset
                # use a fresh operation ID instead of replaying this refusal.
                job = k._update("jobs", {"revision": job["revision"]+1}, id=job["id"])
                return {"job_id": job["id"], "revision": job["revision"], "claimed": False,
                    "parked": parked, "quarantined": quarantined, "attempt_id": None, "state": job["state"]}
        eligible(k, job)
        maintenance_end = None
        if k.get_generation(job["target_generation"])["state"] == "candidate":
            window = k.connection.execute(sql.SQL("SELECT ends_at FROM {} WHERE starts_at<=clock_timestamp() "
                "AND ends_at>clock_timestamp()").format(k.repository._table("knowledge_maintenance_windows"))).fetchone()
            require(window, "candidate rebuild requires an explicit monthly reboot maintenance window")
            maintenance_end = window["ends_at"]
        checkpoint = k.get_reducer(job["unit_id"], job["consumed_input_revision"])
        require(not checkpoint or not checkpoint["parked"], "unit input parked after consecutive failures")
        old = k.get_attempt(job["current_attempt_id"]) if job["current_attempt_id"] else None
        if old and job["state"] == "running":
            require(old["lease_expires_at"] <= now(k), "job already claimed")
            job, checkpoint = abandon_attempt(k, job, old)
            if checkpoint["parked"]:
                return {"job_id": job["id"], "revision": job["revision"], "claimed": False,
                        "parked": True, "attempt_id": None, "state": "failed"}
        else:
            require(job["state"] in {"queued", "failed"} or (job["state"] == "succeeded" and p.get("rebase") is True),
                    "job not claimable")
        reserved = p.get("reserved_cost", 0)
        require(type(reserved) in {int, float} and reserved >= 0, "invalid reserved cost")
        reserved = Decimal(str(reserved))
        runtime = state(k)
        require(runtime["budget_reserved"] + reserved <= runtime["budget_limit"], "budget exhausted")
        duration = p.get("lease_seconds", 300)
        require(type(duration) is int and 0 < duration <= 86400, "invalid lease duration")
        token = runtime["fencing_counter"] + 1
        k._update("knowledge_runtime", {"fencing_counter": token,
                  "budget_reserved": runtime["budget_reserved"] + reserved}, singleton=True)
        attempt_id = identifier("attempt")
        if generation_contract(k, job, bind=True):
            snapshot(k, job["target_generation"])
        base = k.get_manifest(job["target_generation"])
        old_selection = k.get_selection(job["unit_id"], job["target_generation"])
        old_derivation = k._one("derivations", derivation_id=old_selection["derivation_id"]) if old_selection else None
        attempt = k._insert("knowledge_attempts", {"attempt_id": attempt_id, "job_id": job["id"],
            "sequence": job["attempts"] + 1, "submission_id": identifier("submission"),
            "authority_epoch": runtime["authority_epoch"], "fencing_token": token, "lease_id": identifier("lease"),
            "lease_expires_at": min(now(k) + timedelta(seconds=duration), maintenance_end) if maintenance_end else now(k) + timedelta(seconds=duration),
            "namespace": "pv_" + payload_hash({"epoch": runtime["authority_epoch"], "attempt": attempt_id})[:48],
            "reserved_cost": reserved, "aggregate_base_manifest_revision": base["manifest_revision"],
            "base_aggregate_layers": base["manifest"]["aggregate_layers"],
            "base_selected_unit_receipts": base["manifest"]["selected_unit_receipts"],
            "retired_receipt_ids": [old_derivation["receipt_id"]] if old_derivation else []})
        job = k._update("jobs", {"state": "running", "current_attempt_id": attempt_id,
            "attempts": job["attempts"] + 1, "started_at": now(k), "lease_id": attempt["lease_id"],
            "revision": job["revision"] + 1}, id=job["id"])
        k._insert("knowledge_pins", {"pin_id": "attempt-pin:"+attempt_id, "kind": "staging",
            "owner": attempt_id, "authority_epoch": runtime["authority_epoch"],
            "namespaces": attempt["base_aggregate_layers"]+[attempt["namespace"]],
            "generation_id": job["target_generation"], "manifest_revision": attempt["aggregate_base_manifest_revision"],
            "detail": {"reservation_only": True, "unfinished_output_exists": False}})
        result = claim_result(job, attempt)
        outbox(k, "submission", k._json(result), attempt["submission_id"], "submit:" + attempt["submission_id"])
        event(k, attempt_id, "claimed", k._json(result))
        return result

    if action == "expire_attempt":
        job = require(k.get_job(p["job_id"]), "job absent")
        revision(job, p)
        require(state(k)["authority"] == "catalog" and p["authority_epoch"] == state(k)["authority_epoch"],
                "catalog authority epoch changed")
        attempt = require(k.get_attempt(p["attempt_id"]), "attempt absent")
        job, checkpoint = abandon_attempt(k, job, attempt)
        return {"job_id": job["id"], "state": job["state"], "revision": job["revision"], "reducer": checkpoint}

    if action in {"renew", "accept_submission"}:
        payload = p.get("claim", p)
        job, attempt, claim = valid_claim(k, payload, allow_sealed=action == "renew")
        if action == "accept_submission":
            if not attempt["accepted"]:
                k._update("knowledge_attempts", {"accepted": True}, attempt_id=attempt["attempt_id"])
                event(k, attempt["attempt_id"], "submission_accepted", {"submission_id": attempt["submission_id"]})
            return {"submission_id": attempt["submission_id"], "accepted": True, "claim": claim}
        revision(job, p)
        duration = p.get("lease_seconds", 300)
        require(type(duration) is int and 0 < duration <= 86400, "invalid lease duration")
        attempt = k._update("knowledge_attempts", {"lease_expires_at": now(k) + timedelta(seconds=duration)},
                            attempt_id=attempt["attempt_id"])
        job = k._update("jobs", {"revision": job["revision"] + 1}, id=job["id"])
        return claim_result(job, attempt)

    if action == "begin_retirement":
        job, attempt, _ = valid_claim(k, p["claim"], allow_sealed=False)
        require(isinstance(p.get("delivery_id"), str) and p["delivery_id"], "retirement delivery identity required")
        require(not k._rows("knowledge_attempt_events", attempt_id=attempt["attempt_id"], event="retirement_started"),
                "retirement already started; isolated rebuild required")
        event(k, attempt["attempt_id"], "retirement_started", {"delivery_id": p["delivery_id"]})
        return {"delivery_id": p["delivery_id"], "attempt_id": attempt["attempt_id"]}

    if action == "seal_namespace":
        payload = p.get("claim", p)
        namespace = payload["namespace"]
        lock_key = "papervault:" + k.repository.schema + ":namespace:" + namespace
        # Session lock matches namespace_access and is reentrant on its bound connection.
        k.connection.execute("SELECT pg_advisory_lock(hashtextextended(%s,0))", (lock_key,))
        try:
            job, attempt, claim = valid_claim(k, payload, allow_sealed=True)
            rec = p["receipt"]
            recipe = k._one("recipes", recipe_digest=job["recipe_digest"])["spec"]
            validate_receipt(rec, claim, recipe)
            generation_contract(k, job, tier=rec["achieved_tier"])
            for output in rec["output_inventory"]:
                if output["role"] == "tombstones":
                    actual = k.tombstone_inventory(namespace)
                    require(all(output.get(key) == actual[key] for key in ("store", "output_id", "sha256", "byte_length", "backend_ids")),
                            "tombstone receipt differs from actual catalog readback")
            immutable(k, "knowledge_receipts", {"receipt_id": rec["receipt_id"]},
                {"submission_id": attempt["submission_id"], "attempt_id": attempt["attempt_id"],
                 "receipt": rec, "digest": payload_hash(rec)})
            require(not attempt["receipt_id"] or attempt["receipt_id"] == rec["receipt_id"], "namespace sealed with another receipt")
            k._update("knowledge_attempts", {"sealed": True, "receipt_id": rec["receipt_id"]}, attempt_id=attempt["attempt_id"])
            event(k, attempt["attempt_id"], "sealed", {"receipt_id": rec["receipt_id"]})
            return {"receipt_id": rec["receipt_id"], "sealed": True}
        finally:
            k.connection.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))", (lock_key,))

    if action == "complete":
        job, attempt, _ = complete(k, p)
        return {"job_id": job["id"], "state": job["state"], "revision": job["revision"], "receipt_id": attempt["receipt_id"]}

    if action == "defer_attempt":
        job, attempt, _ = valid_claim(k, p, allow_sealed=False)
        revision(job, p)
        # A controlled pause is settled, but may follow partial external work.
        # Retain its reservation and private output protection until accounting
        # can be established; never invent a zero charge or a build failure.
        k._update("knowledge_attempts", {"outcome": "deferred", "finished_at": now(k),
            "cost_state": "unknown"}, attempt_id=attempt["attempt_id"])
        event(k, attempt["attempt_id"], "deferred", {"reason": p.get("reason", {})})
        pin = k._one("knowledge_pins", pin_id="attempt-pin:" + attempt["attempt_id"])
        if pin:
            k._update("knowledge_pins", {"kind": "recovery", "revision": pin["revision"]+1}, pin_id=pin["pin_id"])
        k._update("knowledge_outbox", {"acknowledged_at": now(k)}, outbox_id="submit:" + attempt["submission_id"])
        job = k._update("jobs", {"state": "queued", "lease_id": None, "started_at": None,
            "finished_at": None, "revision": job["revision"]+1}, id=job["id"])
        return {"job_id": job["id"], "state": job["state"], "revision": job["revision"]}

    if action in {"fail", "cancel", "fail_retirement"}:
        job, attempt, _ = valid_claim(k, p, allow_sealed=action != "fail_retirement")
        revision(job, p)
        outcome = "cancelled" if action == "cancel" else "failed"
        k._update("knowledge_attempts", {"outcome": outcome, "finished_at": now(k),
            "error": p.get("error", {}), "cost_state": "unknown"}, attempt_id=attempt["attempt_id"])
        event(k, attempt["attempt_id"], outcome, {"error": p.get("error", {})})
        pin = k._one("knowledge_pins", pin_id="attempt-pin:"+attempt["attempt_id"])
        if pin:
            k._update("knowledge_pins", {"kind": "recovery", "revision": pin["revision"]+1}, pin_id=pin["pin_id"])
        job = k._update("jobs", {"state": outcome, "finished_at": now(k), "revision": job["revision"] + 1}, id=job["id"])
        checkpoint = reduce_outcome(k, job, outcome) if outcome == "failed" else None
        if action == "fail_retirement":
            k._update("knowledge_outbox", {"acknowledged_at": now(k)}, outbox_id="submit:" + attempt["submission_id"])
        return {"job_id": job["id"], "state": job["state"], "revision": job["revision"], "reducer": checkpoint}

    if action == "claim_publication":
        job = require(k.get_job(p["job_id"]), "job absent")
        revision(job, p)
        require(job["state"] == "succeeded", "publication claim requires accepted completion")
        require(state(k)["authority"] == "catalog", "catalog authority not committed")
        require(p["authority_epoch"] == state(k)["authority_epoch"], "authority epoch mismatch")
        eligible(k, job)
        if k.get_generation(job["target_generation"])["state"] == "candidate":
            require(k.get_active_maintenance_window(), "candidate publication requires monthly maintenance window")
        attempt = require(k.get_attempt(job["current_attempt_id"]), "attempt absent")
        require(attempt["receipt_id"] and attempt["sealed"], "publication requires sealed receipt")
        require(not attempt["publication_lease_expires_at"] or attempt["publication_lease_expires_at"] <= now(k),
                "publication already claimed")
        token = identifier("publication-claim")
        attempt = k._update("knowledge_attempts", {"publication_claim_id": token,
            "publication_lease_expires_at": now(k) + timedelta(seconds=p.get("lease_seconds", 300))},
            attempt_id=attempt["attempt_id"])
        return {**claim_result(job, attempt), "publication_claim_id": token}

    if action == "publish_derivation":
        if p.get("kind") == "extraction":
            from .extraction_commands import publish_extraction
            return publish_extraction(k, operation_id, p)
        job, attempt, claim = complete(k, p, publish=True)
        rec = p["receipt"]
        generation_contract(k, job, tier=rec["achieved_tier"])
        desired = k.get_requirement(job["unit_id"], job["target_generation"])
        compatible = bool(desired and not desired["invalidated"] and not desired["excluded"] and
                          (desired["enqueue_mode"] == "active" or k.get_generation(job["target_generation"])["state"] == "candidate") and
                          all(desired[key] == job[key] for key in ("consumed_input_revision", "recipe_digest")))
        selection = k.get_selection(job["unit_id"], job["target_generation"])
        revision(selection, p, "expected_selection_revision")
        select = bool(p.get("select", False) and compatible)
        if select:
            require(k.get_generation(job["target_generation"])["manifest_revision"] == attempt["aggregate_base_manifest_revision"],
                    "aggregate base manifest changed; rebase attempt required")
        restoration = p.get("restoration_recipe_digest")
        lineage = rec.get("origin_lineage", {})
        if restoration:
            require(k._one("recipes", recipe_digest=restoration), "restoration recipe absent")
            require(lineage.get("original_recipe_digest") and lineage.get("origin_derivation_id"), "restoration lineage missing")
            require(lineage["original_recipe_digest"] == job["recipe_digest"], "restoration content recipe differs from original content recipe")
        derivation_id = p.get("derivation_id") or "derivation:" + rec["receipt_id"]
        existing = k._one("derivations", derivation_id=derivation_id)
        if existing:
            require(existing["receipt_id"] == rec["receipt_id"], "derivation receipt differs")
        else:
            audit = k.connection.execute(sql.SQL("INSERT INTO {}(actor,event,key,detail) "
                "VALUES('knowledge','derivation_registration',%s,%s) RETURNING id").format(k.repository._table("audit")),
                (job["key"], Jsonb({"operation_id": operation_id}))).fetchone()
            k._insert("derivations", {"derivation_id": derivation_id, "key": job["key"], "unit_id": job["unit_id"],
                "artifact_inputs": k._one("input_manifests", consumed_input_revision=job["consumed_input_revision"])["manifest"],
                "catalog_revision": audit["id"], "recipe": k._one("recipes", recipe_digest=job["recipe_digest"])["spec"],
                "published_at": now(k), "consumed_input_revision": job["consumed_input_revision"],
                "recipe_digest": job["recipe_digest"], "generation_id": job["target_generation"],
                "achieved_tier": rec["achieved_tier"], "receipt_id": rec["receipt_id"],
                "origin_lineage": lineage, "restoration_recipe_digest": restoration})
        publication_id = identifier("publication")
        k._insert("publications", {"publication_id": publication_id, "derivation_id": derivation_id,
            "unit_id": job["unit_id"], "generation_id": job["target_generation"], "selected": select})
        if select:
            if selection:
                k._update("selections", {"derivation_id": derivation_id, "revision": selection["revision"] + 1},
                          unit_id=job["unit_id"], generation_id=job["target_generation"])
                if selection["derivation_id"] != derivation_id:
                    supersession(k, selection["derivation_id"], "superseded")
            else:
                k._insert("selections", {"unit_id": job["unit_id"], "generation_id": job["target_generation"],
                    "derivation_id": derivation_id})
            supersession(k, derivation_id, "selected")
            manifest_revision = snapshot(k, job["target_generation"], aggregate_receipt=rec)
        else:
            manifest_revision = k.get_generation(job["target_generation"])["manifest_revision"]
        outbox(k, "publication", {"publication_id": publication_id, "derivation_id": derivation_id,
               "receipt_id": rec["receipt_id"], "generation_id": job["target_generation"], "manifest_revision": manifest_revision})
        pin = k._one("knowledge_pins", pin_id="attempt-pin:"+attempt["attempt_id"])
        if pin and not pin["released_at"]:
            k._update("knowledge_pins", {"released_at": now(k), "revision": pin["revision"]+1,
                "detail": {"protection_transferred_to_derivation": derivation_id}}, pin_id=pin["pin_id"])
        return {"job_id": job["id"], "revision": job["revision"], "derivation_id": derivation_id,
                "publication_id": publication_id, "selected": select, "historical": not compatible,
                "manifest_revision": manifest_revision}

    if action == "acknowledge_outbox":
        row = require(k._one("knowledge_outbox", outbox_id=p["outbox_id"]), "outbox missing")
        if not row["acknowledged_at"]:
            k._update("knowledge_outbox", {"acknowledged_at": now(k)}, outbox_id=p["outbox_id"])
        return {"outbox_id": p["outbox_id"], "acknowledged": True}

    if action == "record_tombstone":
        claim = p["claim"]
        valid_claim(k, claim, allow_sealed=False)
        require(bool(p["role"]) and bool(p["output_id"]), "tombstone requires role and output identity")
        return immutable(k, "knowledge_output_tombstones", {"namespace": claim["namespace"],
            "role": p["role"], "output_id": p["output_id"]}, {})

    if action == "observe":
        require(p["state"] in {"present", "missing", "unknown", "not_applicable"}, "invalid observation state")
        stamp = p["observed_at"]
        from datetime import datetime
        stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00")) if isinstance(stamp, str) else stamp
        require(stamp.tzinfo is not None and stamp <= now(k), "observation time invalid or future")
        values = {key: p.get(key) for key in ("unit_id", "consumed_input_revision", "recipe_digest", "generation_id",
            "attempt_id", "submission_id", "authority_epoch", "fencing_token", "receipt_id", "namespace", "output_id")}
        values.update({"role": p["role"], "store": p["store"], "state": p["state"], "observed_at": stamp,
                       "detail": p.get("detail", {})})
        row = immutable(k, "observations", {"observation_id": p["observation_id"]}, values)
        k.connection.execute(sql.SQL("INSERT INTO {} AS h(identity,observation_id,unit_id,namespace,receipt_id,observed_at) "
            "VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(identity) DO UPDATE SET observation_id=EXCLUDED.observation_id,"
            "unit_id=EXCLUDED.unit_id,namespace=EXCLUDED.namespace,receipt_id=EXCLUDED.receipt_id,observed_at=EXCLUDED.observed_at "
            "WHERE EXCLUDED.observed_at>=h.observed_at").format(k.repository._table("observation_heads")),
            (Jsonb([row.get(key) for key in OBSERVATION_IDENTITY_FIELDS]), row["observation_id"], row["unit_id"],
             row["namespace"], row["receipt_id"], row["observed_at"]))
        return row

    if action == "request_repair":
        key = {"publication_id": p["publication_id"], "role": p["role"], "reason": p["reason"]}
        old = k._one("knowledge_repairs", **key)
        if old:
            return {"repair_id": old["repair_id"]}
        row = k._insert("knowledge_repairs", {"repair_id": identifier("repair"), **key})
        requirement = {"isolated_rebuild_required": True} if key["reason"] == "retirement_rebuild_unverified" else {}
        outbox(k, "repair", {"repair_id": row["repair_id"], **key, **requirement})
        return {"repair_id": row["repair_id"]}

    if action == "record_discrepancy":
        return immutable(k, "knowledge_discrepancies", {"discrepancy_id": p["discrepancy_id"]},
            {"unit_id": p.get("unit_id"), "reason": p["reason"], "evidence": p.get("evidence", {})})

    if action == "link_repair":
        repair = require(k.get_repair(p["repair_id"]), "repair missing")
        revision(repair, p)
        generation = require(k.get_generation(p["generation_id"]), "repair candidate missing")
        require(generation["state"] == "candidate", "repair must target candidate generation")
        return k._update("knowledge_repairs", {"target_generation": generation["generation_id"],
            "linked_at": now(k), "revision": repair["revision"]+1}, repair_id=repair["repair_id"])

    if action == "set_quarantine":
        if p.get("quarantine"):
            require(isinstance(p.get("reason"), dict) and all(p["reason"].get(key) for key in
                ("reason", "scope", "origin")), "quarantine requires reason, scope and origin")
        unit = require(k.get_unit(p["unit_id"]), "unit absent")
        revision(unit, p)
        row = k._update("units", {"quarantine": bool(p["quarantine"]), "quarantine_reason": p.get("reason", {}),
                        "revision": unit["revision"] + 1}, unit_id=p["unit_id"])
        bump_requirements(k)
        for requirement in k._rows("knowledge_requirements", unit_id=p["unit_id"]):
            snapshot(k, requirement["generation_id"])
        return row

    if action == "begin_maintenance_window":
        from datetime import datetime
        from zoneinfo import ZoneInfo
        runtime = state(k)
        require(runtime["authority"] == "catalog", "catalog authority not committed")
        require(p.get("expected_revision") == runtime["maintenance_revision"], "maintenance revision changed")
        start, end = (datetime.fromisoformat(p[key].replace("Z", "+00:00")) for key in ("starts_at", "ends_at"))
        require(start.tzinfo is not None and end.tzinfo is not None and 0 < (end-start).total_seconds() <= 7200,
                "monthly reboot maintenance must be positive and at most two hours")
        zone = p.get("timezone", "Asia/Shanghai")
        month = start.astimezone(ZoneInfo(zone)).strftime("%Y-%m")
        require(not k._one("knowledge_maintenance_windows", calendar_month=month), "monthly maintenance window already recorded")
        require(p.get("kind") == "monthly_reboot" and bool(p.get("operator_nonce")), "explicit monthly reboot authorization required")
        row = k._insert("knowledge_maintenance_windows", {"window_id": p["window_id"], "calendar_month": month,
            "timezone": zone, "kind": p["kind"], "starts_at": start, "ends_at": end,
            "operator_nonce": p["operator_nonce"]})
        k._update("knowledge_runtime", {"maintenance_revision": runtime["maintenance_revision"]+1}, singleton=True)
        return {**row, "revision": runtime["maintenance_revision"]+1}

    if action == "pin_manifest":
        runtime = state(k)
        require(runtime["authority"] == "catalog", "catalog authority not committed")
        require(runtime["serving_generation_id"] == p["generation_id"] and
                runtime["serving_manifest_revision"] == p["manifest_revision"] and
                runtime["serving_revision"] == p["expected_serving_revision"], "serving manifest changed before reader pin")
        ttl = p.get("ttl_seconds", 300)
        require(type(ttl) is int and 0 < ttl <= 7200 and bool(p.get("owner")), "reader pin requires owner and bounded TTL")
        manifest = k.get_manifest(p["generation_id"], p["manifest_revision"])
        admit_manifest(k, manifest["manifest"])
        return k._insert("knowledge_pins", {"pin_id": identifier("reader-pin"), "kind": "reader", "owner": p["owner"],
            "authority_epoch": runtime["authority_epoch"], "namespaces": manifest["manifest"]["aggregate_layers"],
            "generation_id": p["generation_id"], "manifest_revision": p["manifest_revision"],
            "expires_at": now(k)+timedelta(seconds=ttl)})

    if action in {"renew_pin", "release_pin"}:
        pin = require(k._one("knowledge_pins", pin_id=p["pin_id"]), "pin absent")
        revision(pin, p)
        require(pin["owner"] == p["owner"] and not pin["released_at"], "pin owner changed or pin released")
        require(pin["authority_epoch"] == state(k)["authority_epoch"], "pin authority epoch changed")
        if action == "renew_pin":
            ttl = p.get("ttl_seconds", 300)
            require(pin["kind"] == "reader" and pin["expires_at"] > now(k) and type(ttl) is int and 0 < ttl <= 7200,
                    "reader pin expired or TTL invalid")
            return k._update("knowledge_pins", {"expires_at": now(k)+timedelta(seconds=ttl), "revision": pin["revision"]+1},
                             pin_id=pin["pin_id"])
        require(pin["kind"] == "reader" or (p.get("disposition") == "reviewed_recovery_disposition" and
                p.get("reason") and p.get("operator_nonce")), "unfinished/recovery output requires reviewed explicit disposition")
        return k._update("knowledge_pins", {"released_at": now(k), "revision": pin["revision"]+1}, pin_id=pin["pin_id"])

    if action == "promote_generation":
        import math
        observation_age = p.get("max_observation_age", 300)
        require(type(observation_age) in {int, float} and math.isfinite(observation_age) and 0 <= observation_age <= 300,
                "promotion observation age must be finite and at most 300 seconds")
        generation = require(k.get_generation(p["generation_id"]), "generation missing")
        require(not generation.get("sample"), "sample generations cannot be promoted")
        runtime = state(k)
        require(generation["state"] == "candidate", "only candidate can promote")
        require(p["expected_manifest_revision"] == generation["manifest_revision"] and
                p["expected_requirements_revision"] == runtime["requirements_revision"] == generation["requirements_revision"] and
                p["expected_serving_revision"] == runtime["serving_revision"], "promotion revision changed")
        manifest = k.get_manifest(generation["generation_id"])["manifest"]
        require(manifest["entries"] and all(not entry["missing"] and not entry["invalidated"]
            for entry in manifest["entries"] if not entry["excluded"]), "candidate incomplete")
        require(all(not inventory["invalidated"] for inventory in manifest.get("expected_inventories", {}).values()),
                "candidate expected inventory changed")
        candidate_requirements = {r["unit_id"]: r for r in k._rows("knowledge_requirements", generation_id=generation["generation_id"])}
        for inventory in manifest.get("expected_inventories", {}).values():
            for unit_id in inventory["required_units"]:
                desired = candidate_requirements.get(unit_id)
                require(desired and desired["inventory_revision"] == inventory["inventory_revision"],
                        "required inventory unit has no compatible desired requirement")
                if inventory["mode"] == "sections" and not desired["excluded"]:
                    source = k.get_input(desired["consumed_input_revision"])["manifest"]
                    mapping_revision = source.get("section_definition_revision", source.get("section_mapping_revision"))
                    require(mapping_revision == inventory["section_mapping_revisions"].get(unit_id) and
                            source.get("parent_evidence_revision") == inventory["parent_evidence_revision"],
                            "required inventory mapping or parent evidence differs")
        uncaptured = k.connection.execute(sql.SQL(
            "SELECT r.key FROM {} r LEFT JOIN {} u ON u.record_key=r.key AND u.kind<>'section' "
            "LEFT JOIN {} q ON q.unit_id=u.unit_id AND q.generation_id=%s "
            "LEFT JOIN {} ci ON ci.parent_unit_id=u.unit_id AND ci.generation_id=%s "
            "LEFT JOIN {} inv ON inv.parent_unit_id=ci.parent_unit_id AND inv.inventory_revision=ci.inventory_revision "
            "WHERE r.policy IN ('clear','override') AND NOT r.quarantine AND r.merge='canonical' "
            "AND (q.unit_id IS NULL OR q.invalidated OR q.record_revision<>r.revision) "
            "AND NOT COALESCE(inv.mode='sections' AND NOT ci.invalidated AND ci.record_revision=r.revision,FALSE) LIMIT 1"
        ).format(k.repository._table("records"), k.repository._table("units"), k.repository._table("knowledge_requirements"),
                 k.repository._table("current_inventories"), k.repository._table("unit_inventories")),
            (generation["generation_id"], generation["generation_id"])).fetchone()
        require(not uncaptured, "candidate omits newly registered or changed source requirements")
        from .coverage import query_coverage
        covered = query_coverage(k, generation_id=generation["generation_id"], max_observation_age=observation_age)
        required_ids = {unit_id for unit_id, r in candidate_requirements.items() if not r["excluded"]}
        require(all(u["tier_covered"] and (u["achieved_tier"] != "fulltext" or u["readable"])
                    for u in covered["units"] if u["unit_id"] in required_ids), "candidate outputs not freshly verified")
        require(all(any(record["tier_coverage"].values()) for record in covered["records"]
                    if required_ids.intersection(record["unit_ids"])), "candidate expected inventory completeness is missing or unknown")
        active = k._rows("generations", state="active")
        active_required = {r["unit_id"]: r for g in active for r in k._rows("knowledge_requirements", generation_id=g["generation_id"])
                           if not r["excluded"]}
        candidate_required = {r["unit_id"]: r for r in k._rows("knowledge_requirements", generation_id=generation["generation_id"])
                              if not r["excluded"]}
        require(set(active_required) == set(candidate_required) if active else True, "candidate omits current requirements")
        active_inventories = {r["parent_unit_id"]: r["inventory_revision"] for g in active
                             for r in k._rows("current_inventories", generation_id=g["generation_id"])}
        candidate_inventories = {r["parent_unit_id"]: r["inventory_revision"]
                                for r in k._rows("current_inventories", generation_id=generation["generation_id"])}
        require(active_inventories == candidate_inventories if active else True, "candidate expected inventories differ from current requirements")
        require(manifest.get("retrieval_contract") == generation["retrieval_contract"], "candidate retrieval contract differs from manifest")
        for unit_id, requirement in candidate_required.items():
            generation_contract(k, {"target_generation": generation["generation_id"], **requirement})
            eligible(k, {"unit_id": unit_id, "quarantine": False,
                "target_generation": generation["generation_id"], **requirement})
            current = active_required.get(unit_id)
            if current:
                require(all(current[key] == requirement[key] for key in ("consumed_input_revision", "recipe_digest", "record_revision")),
                        "candidate differs from current requirements")
        for old in active:
            for selected in k._rows("selections", generation_id=old["generation_id"]):
                supersession(k, selected["derivation_id"], "superseded")
            k._update("generations", {"state": "retired"}, generation_id=old["generation_id"])
        k._update("generations", {"state": "active"}, generation_id=generation["generation_id"])
        k._update("knowledge_runtime", {"serving_generation_id": generation["generation_id"],
            "serving_manifest_revision": generation["manifest_revision"], "serving_revision": runtime["serving_revision"] + 1}, singleton=True)
        outbox(k, "generation_promoted", {"generation_id": generation["generation_id"]})
        return k.get_serving_manifest()

    raise ValueError("unsupported knowledge command: " + action)
