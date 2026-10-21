"""Re-execute retained inputs through normal admission in a fresh isolated catalog.

Historical jobs/outboxes are never imported into this executor. Exact immutable
inputs and content recipes are; normal claims, verification, sealing and candidate
publication create the new lineage. No serving generation is installed here.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from .extraction import write_once
from .knowledge import KnowledgeRepository
from .recovery import MAX_WINDOW_SECONDS, _atomic_json, checked_path, child, digest_file, verify_recovery_set
from .recovery_postgres import validate_catalog_state, validate_catalog_target
from .recovery_restore import copy_verified_file
from .recovery_source import verify_file_dependencies
from .repository import AuditEvent, PostgresCatalogRepository, Record, canonical_json, payload_hash


def replay_plan(package, *, source_generation, parent_ids=None):
    manifest = verify_recovery_set(package)
    catalog = json.loads(child(package, "catalog/data.json").read_bytes())
    validate_catalog_state(catalog)
    verify_file_dependencies(package, catalog)
    tables = catalog["tables"]
    selectors = [row for row in tables["current_inventories"] if row["generation_id"] == source_generation]
    if not selectors:
        raise ValueError("replay requires an explicit source generation with recorded parent inventories")
    known = {row["parent_unit_id"] for row in selectors}
    chosen = set(parent_ids) if parent_ids is not None else known
    if not chosen or not chosen.issubset(known):
        raise ValueError("replay parent selection is absent from source generation")
    selectors = [row for row in selectors if row["parent_unit_id"] in chosen]
    if any(row["invalidated"] for row in selectors):
        raise ValueError("replay source inventory is invalidated")
    inventories = [row for row in tables["unit_inventories"] if any(
        row["parent_unit_id"] == item["parent_unit_id"] and row["inventory_revision"] == item["inventory_revision"]
        for item in selectors)]
    needed = chosen | {unit for row in inventories for unit in row["required_units"] + row["optional_units"]}
    requirements = [row for row in tables["knowledge_requirements"]
                    if row["generation_id"] == source_generation and row["unit_id"] in needed]
    executable = [row for row in requirements if not row["excluded"]]
    if not executable or any(row["invalidated"] for row in requirements):
        raise ValueError("replay requires current nonempty source requirements")
    input_ids = {row["consumed_input_revision"] for row in requirements}
    input_ids.update(row["parent_consumed_input_revision"] for row in inventories if row["parent_consumed_input_revision"])
    inputs = [row for row in tables["input_manifests"] if row["consumed_input_revision"] in input_ids]
    recipe_ids = {row["recipe_digest"] for row in requirements}
    recipes = [row for row in tables["recipes"] if row["recipe_digest"] in recipe_ids]
    if {row["consumed_input_revision"] for row in inputs} != input_ids or len(recipes) != len(recipe_ids):
        raise ValueError("replay input/recipe closure incomplete")
    return manifest, catalog, {"source_generation": source_generation, "parent_ids": sorted(chosen),
        "units": [row for row in tables["units"] if row["unit_id"] in needed],
        "requirements": requirements, "inventories": inventories, "inputs": inputs, "recipes": recipes,
        "unit_quarantine_disposition": {"policy": "reset_in_isolated_replay", "reset_units": [
            {"unit_id": row["unit_id"], "source_revision": row["revision"], "source_reason": row["quarantine_reason"]}
            for row in sorted(tables["units"], key=lambda row: row["unit_id"])
            if row["unit_id"] in needed and row["quarantine"]]},
        "sections": [row for row in tables["section_mappings"] if row["unit_id"] in needed],
        "physical": any(row["manifest"]["tier"] != "metadata" for row in inputs)}


def _initialize(connection, schema, state, catalog):
    from psycopg.rows import dict_row
    connection.row_factory = dict_row
    repository = PostgresCatalogRepository(connection, schema=schema)
    exists = connection.execute("SELECT to_regnamespace(%s) AS name", (schema,)).fetchone()["name"]
    fence = "recovery-replay:" + state["operation_id"]
    if exists:
        existing = connection.execute(sql.SQL("SELECT detail FROM {} WHERE event='recovery_replay_initialized'").format(
            repository._table("audit"))).fetchone()
        if not existing or existing["detail"] != state["binding"]:
            raise ValueError("replay catalog belongs to another operation")
    else:
        with connection.transaction():
            repository.ensure_schema(audit=AuditEvent("recovery", "replay_schema"))
            repository.ensure_knowledge_schema(audit=AuditEvent("recovery", "replay_schema"))
            runtime = catalog["tables"]["knowledge_runtime"][0]
            # Isolated harness bootstrap, never the production cutover protocol.
            connection.execute(sql.SQL("UPDATE {} SET authority='catalog',authority_epoch=%s,fencing_counter=%s").format(
                repository._table("knowledge_runtime")),
                (runtime["authority_epoch"] + 1, runtime["fencing_counter"]))
            repository.set_fence(fence, audit=AuditEvent("recovery", "recovery_replay_initialized", state["binding"]))
    persisted_fence = connection.execute(sql.SQL("SELECT fence_id FROM {}").format(repository._table("writer_fence"))).fetchone()
    runtime = connection.execute(sql.SQL("SELECT authority,authority_epoch,serving_generation_id FROM {}").format(
        repository._table("knowledge_runtime"))).fetchone()
    if (not persisted_fence or persisted_fence["fence_id"] != fence or not runtime
            or runtime["authority"] != "catalog" or runtime["serving_generation_id"] is not None
            or runtime["authority_epoch"] != catalog["tables"]["knowledge_runtime"][0]["authority_epoch"] + 1):
        raise ValueError("replay durable fence, authority or epoch changed")
    repository.fence_id = fence
    return KnowledgeRepository(repository)


def _import_inputs(k, plan, catalog, generation, operation):
    def command(action, payload):
        return k.command(action, operation + ":" + action + ":" + payload_hash(payload), payload)

    keys = {row["record_key"] for row in plan["units"] if row["record_key"]}
    for row in catalog["tables"]["records"]:
        if row["key"] not in keys:
            continue
        old = k.repository.get(row["key"])
        if old is None:
            k.repository.upsert(Record.from_payload(row["key"], row["payload"]), expected_revision=0,
                                audit=AuditEvent("recovery", "replay_source_record"))
        elif old.payload != row["payload"]:
            raise ValueError("replay target record changed")
    command("create_generation", {"generation_id": generation, "candidate": True, "sample": True})
    for row in plan["units"]:
        command("register_unit", {key: row[key] for key in ("unit_id", "kind", "record_key", "identity")})
    for row in plan["sections"]:
        command("register_section", {key: row[key] for key in (
            "unit_id", "parent_unit_id", "mapping_revision", "parent_evidence_revision", "definition")})
    for row in plan["inputs"]:
        command("register_input", {key: row[key] for key in ("unit_id", "consumed_input_revision", "manifest")})
    for row in plan["recipes"]:
        command("register_recipe", {"recipe": row["spec"], "recipe_digest": row["recipe_digest"]})
    for row in plan["inventories"]:
        command("register_inventory", {key: row[key] for key in ("parent_unit_id", "inventory_revision",
            "parent_evidence_revision", "parent_consumed_input_revision", "mode", "required_units", "optional_units",
            "section_mapping_revisions")})
        unit = k.get_unit(row["parent_unit_id"])
        record = k.repository.get(unit["record_key"]) if unit["record_key"] else None
        command("set_inventory", {"parent_unit_id": row["parent_unit_id"], "generation_id": generation,
            "inventory_revision": row["inventory_revision"], "expected_revision": 0,
            "expected_record_revision": record.revision if record else None})
    for row in plan["requirements"]:
        unit = k.get_unit(row["unit_id"])
        record = k.repository.get(unit["record_key"]) if unit["record_key"] else None
        desired = {key: row[key] for key in ("unit_id", "consumed_input_revision", "recipe_digest", "inventory_revision", "excluded")}
        desired["generation_id"] = generation
        command("set_requirement", {**desired, "expected_revision": 0,
            "expected_record_revision": record.revision if record else None})
        if not row["excluded"]:
            command("enqueue", {key: desired[key] for key in (
                "unit_id", "consumed_input_revision", "recipe_digest", "generation_id")})


async def replay_recovery_set(package, target, *, source_generation, target_generation, parent_ids=None,
                              catalog_connection=None, catalog_schema=None, apply=False,
                              adapter_factory=None, backend_identity=None):
    execution_deadline = time.monotonic() + MAX_WINDOW_SECONDS
    package, target = checked_path(package), checked_path(target)
    manifest, catalog, plan = replay_plan(package, source_generation=source_generation, parent_ids=parent_ids)
    if (not target_generation or any(row["generation_id"] == target_generation for row in catalog["tables"]["generations"])):
        raise ValueError("replay requires a new isolated generation")
    for root in [package, *(checked_path(root) for root in manifest.get("source_roots", []))]:
        if target.is_relative_to(root) or root.is_relative_to(target):
            raise ValueError("replay target must be isolated from source roots")
    if not apply:
        return {"apply": False, "mode": "replay", "target": str(target), "target_generation": target_generation,
                "sample": True, "parent_ids": plan["parent_ids"], "units": [row["unit_id"] for row in plan["units"]],
                "physical_backends_required": plan["physical"]}
    if not catalog_connection or not catalog_schema:
        raise ValueError("replay requires an explicit isolated catalog target")
    validate_catalog_target(catalog_connection, catalog_schema)
    if plan["physical"] and (adapter_factory is None or not backend_identity):
        raise ValueError("nonmetadata replay requires explicit verified isolated backends and the exact installed recipe")
    identity = {key: value for key, value in conninfo_to_dict(catalog_connection.info.dsn).items()
                if key in {"host", "port", "dbname", "user"}}
    binding = {"manifest_sha256": digest_file(package / "manifest.json")["sha256"], "catalog_target": identity,
        "catalog_schema": catalog_schema, "target_generation": target_generation, "plan_sha256": payload_hash(plan),
        "backends": backend_identity}
    marker = child(target, ".replay.json")
    if target.exists() and any(target.iterdir()) and not marker.exists():
        raise ValueError("replay requires an empty isolated target")
    target.mkdir(parents=True, exist_ok=True)
    with child(target, ".replay.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists():
            state = json.loads(marker.read_bytes())
            if state["binding"] != binding:
                raise ValueError("replay target/checkpoint belongs to another operation")
        else:
            present = catalog_connection.execute("SELECT to_regnamespace(%s)", (catalog_schema,)).fetchone()
            present = next(iter(present.values())) if isinstance(present, dict) else present[0]
            if present:
                raise ValueError("replay requires an absent catalog schema")
            start = datetime.now(timezone.utc)
            state = {"binding": binding, "operation_id": "replay-" + uuid4().hex, "rounds": [], "failures": [],
                "started_at": start.isoformat(), "ends_at": (start + timedelta(seconds=MAX_WINDOW_SECONDS)).isoformat()}
            _atomic_json(marker, state)
        try:
            from papervault.knowledge.scheduler.catalog import CatalogScheduler
            from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
            from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
            from papervault.knowledge.store.build_breaker import BuildBreaker, BuildSchedule

            for row in plan["inputs"]:
                for artifact in row["manifest"]["artifacts"]:
                    relative = "catalog_inputs/" + artifact["sha256"]
                    copy_verified_file(child(package, "inputs/" + relative), child(target, "knowledge/" + relative),
                        {key: artifact[key] for key in ("sha256", "byte_length")})
            k = _initialize(catalog_connection, catalog_schema, state, catalog)
            _import_inputs(k, plan, catalog, target_generation, state["operation_id"])
            op = state["operation_id"]
            expected = [row for row in plan["requirements"] if not row["excluded"]]
            sessions = state.setdefault("sessions", [{"window_id": op, "starts_at": state["started_at"],
                "ends_at": state["ends_at"], "expected_revision": 0}])
            session = sessions[-1]
            if not all(k.get_selection(row["unit_id"], target_generation) for row in expected):
                if datetime.now(timezone.utc) >= datetime.fromisoformat(session["ends_at"]):
                    start = datetime.now(timezone.utc)
                    session = {"window_id": op + ":" + uuid4().hex, "starts_at": start.isoformat(),
                        "ends_at": (start + timedelta(seconds=MAX_WINDOW_SECONDS)).isoformat(),
                        "expected_revision": k.get_authority_state()["maintenance_revision"]}
                    sessions.append(session)
                # Preserve the normal once-per-calendar-month admission rule.
                # A later --apply resumes this operation in the next permitted
                # window; it cannot extend an expired window in the same month.
                _atomic_json(marker, state)
                k.command("begin_maintenance_window", session["window_id"] + ":window", {
                    **session, "kind": "monthly_reboot", "timezone": "Asia/Shanghai",
                    "operator_nonce": session["window_id"]})
            inputs = ImmutableInputStore(target / "knowledge/catalog_inputs")
            factory = adapter_factory or CatalogLightRAGAdapter
            adapter = factory(k, working_dir=str(target / "knowledge"), input_store=inputs)
            scheduler = CatalogScheduler(k, adapter, inputs, max_admissions=1, reserved_cost=0 if not plan["physical"] else 1,
                breaker=BuildBreaker(threshold=0, schedule=BuildSchedule(enabled=False)) if not plan["physical"] else None)
            started = time.monotonic()
            while True:
                selected = [k.get_selection(row["unit_id"], target_generation) for row in expected]
                if all(selected):
                    break
                remaining = min(execution_deadline - time.monotonic(),
                    (datetime.fromisoformat(session["ends_at"]) - datetime.now(timezone.utc)).total_seconds())
                if remaining <= 0:
                    raise RuntimeError("replay maintenance window expired; pending work retained")
                async with asyncio.timeout(remaining):
                    report = await scheduler.run_round(only_keys=[row["unit_id"] for row in expected])
                state["rounds"].append(report)
                _atomic_json(marker, state)
                if report["discrepancies"] or report["paused"] or not (report["claimed"] or report["published"] or report["completed"]):
                    raise RuntimeError("replay stopped without complete verified publication; inspect isolated catalog evidence")
            receipts = [k.get_receipt(k.get_derivation(item["derivation_id"])["receipt_id"]) for item in selected]
            # Re-read the entire transitive aggregate even on a completed resume.
            # Historical selections/receipts cannot certify current target bytes.
            head = max(receipts, key=lambda item: len(item["aggregate_layers"]))
            inherited = k.get_namespace_receipts(head["aggregate_layers"])
            expected_outputs = {(row["receipt_id"], output["role"], output["store"], output["output_id"])
                for row in inherited for output in row["output_inventory"]}
            async with asyncio.timeout(max(0, execution_deadline - time.monotonic())):
                observations = await adapter.collect_observations(head)
            seen = set()
            for observation in observations:
                scheduler._observe(observation)
                seen.add((observation["receipt_id"], observation["role"], observation["store"], observation["output_id"]))
            if seen != expected_outputs or any(row["state"] != "present" for row in observations):
                raise ValueError("replay target readback differs from retained output receipts")
            state["last_readback"] = {"observed_at": datetime.now(timezone.utc).isoformat(),
                "outputs": len(observations), "observations_sha256": payload_hash(observations)}
            _atomic_json(marker, state)
            runtime = k.get_authority_state()
            if runtime["serving_generation_id"] is not None:
                raise ValueError("replay unexpectedly changed serving selection")
            receipt = {"schema_version": 1, "kind": "replay_materialization", "operation_id": op,
                "acceptance_scope": "isolated_recovery", "sample": True, "target_generation": target_generation,
                "target_root": str(target), "catalog_schema": catalog_schema, "writer_fence": k.repository.fence_id,
                "source_snapshot_id": manifest["snapshot_id"], "source_manifest_sha256": binding["manifest_sha256"],
                "source_generation": source_generation, "parent_ids": plan["parent_ids"],
                "source_content_recipes": [row["recipe_digest"] for row in plan["recipes"]],
                "unit_quarantine_disposition": plan["unit_quarantine_disposition"],
                "original_receipt_ids": [row["receipt_id"] for row in catalog["tables"]["knowledge_receipts"]],
                "operation_recipe": {"name": "papervault-replay", "implementation_sha256": digest_file(Path(__file__))["sha256"]},
                "receipts": receipts, "serving_generation_id": None,
                "readback": state["last_readback"],
                "maintenance_sessions": sessions,
                "attempt_count": len(k._rows("knowledge_attempts")), "failures": state["failures"],
                "last_session_elapsed_seconds": time.monotonic() - started}
            output = child(target, "replay-receipt.json")
            if output.exists():
                existing = json.loads(output.read_bytes())
                if state.get("receipt_sha256") and digest_file(output)["sha256"] != state["receipt_sha256"]:
                    raise ValueError("completed replay receipt checksum differs")
                stable = set(receipt) - {"last_session_elapsed_seconds", "failures", "readback"}
                if any(existing.get(key) != receipt[key] for key in stable):
                    raise ValueError("completed replay receipt differs from verified selections")
                # A receipt fsync can succeed before the final progress write.
                # Adopt only the fully checked immutable binding and lineage.
                state["receipt_sha256"] = digest_file(output)["sha256"]
                _atomic_json(marker, state)
                return existing
            write_once(output, canonical_json(receipt).encode())
            state["receipt_sha256"] = digest_file(output)["sha256"]
            _atomic_json(marker, state)
            return receipt
        except BaseException as exc:
            state["failures"].append({"error_type": type(exc).__name__})
            _atomic_json(marker, state)
            raise
