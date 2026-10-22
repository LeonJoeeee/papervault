"""Runtime wiring for catalog admission, durable intake recovery and delivery."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path

from papervault.store.repository import Record, payload_hash

log = logging.getLogger("ks.scheduler.catalog")


async def _funded_thread(function, *args, **kwargs):
    """Keep the enclosing reservation until an uncancellable thread has stopped."""
    operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    interrupted = False
    while True:
        try:
            result = await asyncio.shield(operation)
            break
        except asyncio.CancelledError:
            if operation.done():
                raise
            interrupted = True
        except Exception:
            if interrupted:
                raise asyncio.CancelledError from None
            raise
    if interrupted:
        raise asyncio.CancelledError
    return result


class CatalogService:
    def __init__(self, repository, intake, scheduler):
        self.repository, self.intake, self.scheduler = repository, intake, scheduler

    def _command(self, action, payload, *, operation_id=None):
        operation_id = operation_id or "intake-recovery:" + action + ":" + payload_hash(payload)
        previous = self.repository.get_operation(operation_id)
        return previous if previous is not None else self.repository.command(action, operation_id, payload)

    def recover_inventory(self, event):
        """Finish a captured section inventory without re-reading mutable sources."""
        repo = self.repository
        inventory = repo.get_current_inventory(event["parent_unit_id"], event["generation_id"])
        if inventory is None or inventory.get("invalidated"):
            raise ValueError("inventory recovery requires a current validated inventory")
        if any(event.get(key) is not None and event[key] != inventory.get(key)
               for key in ("inventory_revision", "request_id")):
            return  # This work was durably superseded by another inventory selection.
        if inventory.get("mode") == "whole":
            return
        recipe = inventory.get("desired_recipe_digest")
        if not recipe:
            raise ValueError("inventory has no admitted recipe for intake recovery")
        generation_id = event["generation_id"]
        for unit in inventory["required_units"] + inventory.get("optional_units", []):
            matches = [row for row in repo.list_inputs_for_unit(unit)
                if row["manifest"].get("parent_evidence_revision") == inventory["parent_evidence_revision"]
                and row["manifest"].get("section_definition_revision") == inventory["section_mapping_revisions"][unit]
                and row["manifest"].get("parent_consumed_input_revision") == inventory.get("parent_consumed_input_revision")]
            if len(matches) != 1:
                raise ValueError(f"missing or ambiguous captured section input: {unit}")
            input_revision = matches[0]["consumed_input_revision"]
            previous = repo.get_requirement(unit, generation_id)
            desired = {"unit_id": unit, "generation_id": generation_id,
                       "consumed_input_revision": input_revision, "recipe_digest": recipe,
                       "inventory_revision": inventory["inventory_revision"]}
            if previous is None or previous.get("invalidated") or previous.get("excluded") or any(
                    previous.get(key) != value for key, value in desired.items()):
                self._command("set_requirement", {**desired, "expected_revision": previous["revision"] if previous else 0})
            if inventory.get("enqueue_mode", "active") == "active":
                self._command("enqueue", {key: desired[key] for key in
                    ("unit_id", "generation_id", "consumed_input_revision", "recipe_digest")})
        if inventory.get("enqueue_mode") == "candidate":
            request_id = inventory.get("request_id")
            if not request_id:
                raise ValueError("candidate recovery requires a durable rebuild request")
            operation_id = "operator-fork:" + request_id
            result = repo.get_operation(operation_id)
            if result is None:
                generation = repo.get_generation(generation_id)
                if generation["state"] != "active":
                    raise ValueError("operator rebuild source must be active")
                self._command("fork_generation", {
                    "generation_id": "generation:" + payload_hash({"request_id": request_id}),
                    "source_generation_id": generation_id,
                    "expected_manifest_revision": generation["manifest_revision"],
                    "expected_requirements_revision": repo.get_authority_state()["requirements_revision"]},
                    operation_id=operation_id)

    def recover_intake(self, only_keys=None):
        repo = self.repository
        scope = set(only_keys) if only_keys is not None else None
        recovered = 0
        for kind in ("extraction_completed", "capture_input", "inventory_changed", "requirement_changed"):
            for event in repo.list_outbox(kind=kind):
                payload = event["payload"]
                if kind in {"capture_input", "extraction_completed"}:
                    record_key = payload["record_key"]
                    if kind == "extraction_completed":
                        record_key = repo.resolve_record_alias(record_key)
                    if scope is not None and record_key not in scope:
                        continue
                    row = repo.get_record(record_key)
                    if row is None:
                        continue
                    if kind == "extraction_completed" and row["payload"].get("extraction_completion_id") != payload["completion_id"]:
                        head = repo.get_extraction_head(payload["unit_id"])
                        origin = repo.get_unit(payload["unit_id"])
                        if (head and head["completion_id"] == payload["completion_id"]
                                and origin["record_key"] == record_key):
                            continue  # Registration committed; Library projection/save has not.
                        # An aliased origin's head is historical. Once its canonical
                        # destination selects another output, no origin projection is due.
                        self._command("acknowledge_outbox", {"outbox_id": event["outbox_id"]},
                                      operation_id="ack:" + event["outbox_id"])
                        recovered += 1  # A newer completion superseded this input; history stays.
                        continue
                    record = replace(Record.from_payload(row["key"], row["payload"]), revision=row["revision"])
                    if self.intake._capture_rows([record])["errors"]:
                        continue
                elif kind == "inventory_changed":
                    if scope is not None and payload["parent_unit_id"] not in scope:
                        continue
                    self.recover_inventory(payload)
                else:
                    if scope is not None and payload["unit_id"] not in scope:
                        continue
                    section = repo.get_section(payload["unit_id"])
                    if section:
                        self.recover_inventory({**payload, "parent_unit_id": section["parent_unit_id"]})
                    else:
                        unit = repo.get_unit(payload["unit_id"])
                        if not unit or not unit.get("record_key"):
                            continue
                        row = repo.get_record(unit["record_key"])
                        if row is None:
                            continue
                        self.intake.capture_record(replace(
                            Record.from_payload(row["key"], row["payload"]), revision=row["revision"]))
                self._command("acknowledge_outbox", {"outbox_id": event["outbox_id"]},
                              operation_id="ack:" + event["outbox_id"])
                recovered += 1
        return recovered

    async def run_round(self, only_keys=None):
        from papervault.store.scheduler_capacity import CapacityAdmissionError

        gate = getattr(self.scheduler, "capacity_gate", None)
        intake_plan = None
        if gate is None:
            intake = await _funded_thread(self.intake.scan, only_keys=only_keys)
            recovered = await _funded_thread(self.recover_intake, only_keys)
        else:
            try:
                with gate.intake() as intake_plan:
                    intake = await _funded_thread(self.intake.scan, only_keys=only_keys)
                    recovered = await _funded_thread(self.recover_intake, only_keys)
            except CapacityAdmissionError as exc:
                intake = {"captured": 0, "capacity_blocked": True, "reason": str(exc)}
                recovered = 0
        result = await self.scheduler.run_round(only_keys=only_keys)
        if intake_plan and intake_plan.get("alert"):
            result.update(capacity_alert=True, capacity=dict(intake_plan))
        return {**result, "intake": intake, "recovered_intake": recovered}


_service = None


def get_catalog_service():
    global _service
    from papervault.store.knowledge_runtime import assert_knowledge_authority, get_knowledge_repository

    assert_knowledge_authority("catalog")
    if _service is None:
        from papervault import config
        from papervault.knowledge.config import CONFIG
        from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

        from .catalog import CatalogScheduler
        from .catalog_inputs import ImmutableInputStore
        from .catalog_intake import CatalogIntake

        repository = get_knowledge_repository()
        working_dir = Path(CONFIG.lightrag.working_dir)
        inputs = ImmutableInputStore(working_dir / "catalog_inputs")
        intake = CatalogIntake(repository, inputs, config.VAULT_PATH)
        adapter = CatalogLightRAGAdapter(repository, working_dir=str(working_dir), input_store=inputs)
        from papervault.store.scheduler_capacity import (
            DEFAULT_EXECUTION_BYTES, SchedulerCapacity, measured_backend_usage,
        )

        def backend_usage(physical):
            external = measured_backend_usage(os.getenv("PAPERVAULT_DERIVATIVE_USAGE_FILE"),
                vault_root=config.VAULT_PATH, knowledge_root=working_dir)
            # Metadata writes share the same global derivative budget. Fresh
            # backend accounting is required even when this job creates no graph.
            row = repository.connection.execute(
                "SELECT COALESCE(sum(pg_total_relation_size(c.oid)),0) AS bytes FROM pg_class c "
                "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND c.relkind='r'",
                (repository.repository.schema,)).fetchone()
            return {**external, "backend_bytes": external["backend_bytes"] + int(row["bytes"])}

        capacity = SchedulerCapacity(working_dir / "maintenance" / "capacity",
            derivative_roots=(Path(config.VAULT_PATH) / "extracts", working_dir),
            estimated_bytes=lambda: int(os.getenv("PAPERVAULT_CATALOG_EXECUTION_BYTES", str(DEFAULT_EXECUTION_BYTES))),
            backend_usage=backend_usage)
        _service = CatalogService(repository, intake,
            CatalogScheduler(repository, adapter, inputs, capacity_gate=capacity))
    return _service


async def run_catalog_round(only_keys=None):
    service = await asyncio.to_thread(get_catalog_service)
    return await service.run_round(only_keys=only_keys)


async def capture_operator_document(kind, key, path, *, force=False, tokenizer=None,
                                    max_tokens=None, read_text=None):
    service = await asyncio.to_thread(get_catalog_service)
    from papervault.store.scheduler_capacity import CapacityAdmissionError
    from papervault.knowledge.store.build_breaker import BuildPausedError

    gate = getattr(service.scheduler, "capacity_gate", None)
    try:
        with gate.intake() if gate is not None else nullcontext():
            return await _funded_thread(service.intake.capture_document, kind, key, path,
                force=force, tokenizer=tokenizer, max_tokens=max_tokens, read_text=read_text)
    except CapacityAdmissionError as exc:
        # Pickup already treats BuildPausedError as retryable and leaves the
        # operator source in pending rather than marking an unfunded job failed.
        raise BuildPausedError("operator input capacity admission unavailable") from exc


async def run_catalog_loop(*, interval=60):
    service = await asyncio.to_thread(get_catalog_service)
    from papervault.knowledge.scheduler.opdoc_pickup import drain_pending
    from papervault.knowledge.store.build_breaker import get_breaker
    from papervault.store.scheduler_capacity import CapacityAdmissionError

    while True:
        try:
            await service.run_round()
            if get_breaker().allows():
                gate = getattr(service.scheduler, "capacity_gate", None)
                with gate.intake() if gate is not None else nullcontext():
                    await drain_pending(None)
        except CapacityAdmissionError:
            log.info("operator pickup paused for capacity; pending sources retained")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("catalog scheduler round failed; durable work retained for recovery")
        await asyncio.sleep(interval)
