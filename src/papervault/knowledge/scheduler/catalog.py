"""Catalog scheduler: observations never imply execution or publication.

Submission delivery is at least once and uses one stable attempt identity. An
ambiguous external outcome remains recoverable; only a verified sealed receipt
can settle an execution. Publication remains an independent catalog command.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from datetime import datetime, timezone

from papervault.knowledge.store.build_breaker import BuildPausedError
from papervault.store.knowledge import CLAIM_FIELDS, ClaimRejected, KnownExecutionFailure, RetirementFailure, validate_receipt
from papervault.store.repository import payload_hash


def wire(value):
    return json.loads(json.dumps(value, default=lambda v: v.isoformat() if isinstance(v, datetime) else str(v)))


def citation_file_path(unit, manifest):
    """Preserve the public source key independently of immutable byte location."""
    if unit["kind"] == "paper" and unit.get("record_key"):
        from papervault.knowledge.ingest.distill import file_path
        return file_path(unit["record_key"])
    source = manifest.get("metadata", {}).get("source_key")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("operator/section input requires its original public source key")
    return source


def execution_text(unit, tier, data):
    text = data.decode("utf-8", errors="replace")
    if tier == "fulltext" and unit.get("kind") == "paper":
        from papervault.knowledge.ingest.distill import clean

        return clean(text)
    return text


def metadata_receipt(claim, manifest, input_store):
    """Verify already captured metadata bytes; no graph presence is invented."""
    artifacts = manifest["artifacts"]
    if manifest.get("tier") != "metadata" or len(artifacts) != 1:
        raise ValueError("metadata publication requires one exact metadata artifact")
    artifact = artifacts[0]
    data = input_store.read(artifact)
    stamp = datetime.now(timezone.utc).isoformat()
    receipt_id = "metadata-receipt:" + payload_hash({"submission_id": claim["submission_id"], "artifact_id": artifact["artifact_id"]})
    selected = {**claim["base_selected_unit_receipts"], claim["unit_id"]: receipt_id}
    receipt = {key: claim[key] for key in CLAIM_FIELDS}
    receipt.update({"receipt_id": receipt_id, "achieved_tier": "metadata", "sealed": True, "durable": True,
        "provider_observations": {"provider": "not_applicable", "model": "not_applicable"}, "origin_lineage": {},
        **{key: claim[key] for key in ("aggregate_base_manifest_revision", "base_aggregate_layers",
            "base_selected_unit_receipts", "retired_receipt_ids")},
        "aggregate_layers": claim["base_aggregate_layers"] + [claim["namespace"]],
        "selected_unit_receipts": selected, "selected_unit_receipts_digest": payload_hash(selected),
        "output_inventory": [{"role": "metadata", "store": "catalog_inputs", "namespace": claim["namespace"],
            "output_id": artifact["artifact_id"], "backend_ids": [artifact["artifact_id"]],
            "sha256": hashlib.sha256(data).hexdigest(), "byte_length": len(data), "verified_at": stamp,
            "verification": {"state": "present", "method": "immutable_file_readback_sha256_v1",
                             "evidence": {"artifact_id": artifact["artifact_id"], "verified": True}}}]})
    return receipt


class CatalogScheduler:
    def __init__(self, repository, adapter, input_store, *, max_admissions=1,
                 lease_seconds=300, reserved_cost=1, breaker=None, capacity_gate=None):
        self.repository = repository
        self.adapter = adapter
        self.input_store = input_store
        self.max_admissions = max_admissions
        self.lease_seconds = lease_seconds
        self.reserved_cost = reserved_cost
        self.breaker = breaker
        self.capacity_gate = capacity_gate
        self._input_observation_cache = {}
        self._collected_outputs = {}

    def _candidate_allowed(self, job):
        generation = self.repository.get_generation(job["generation_id"])
        return generation["state"] != "candidate" or bool(self.repository.get_active_maintenance_window())

    def _reserve_job(self, job, summary, *, claim=None, admission=False):
        if self.capacity_gate is None:
            return True  # Explicit injection seam; production always wires a gate.
        manifest = self.repository.get_input(job["consumed_input_revision"])["manifest"]
        physical = manifest.get("tier", "fulltext") != "metadata"
        if not physical:
            retired = claim.get("retired_receipt_ids", []) if claim else []
            if not claim:
                selection = self.repository.get_selection(job["unit_id"], job["generation_id"])
                derivation = self.repository.get_derivation(selection["derivation_id"]) if selection else None
                retired = [derivation["receipt_id"]] if derivation else []
            physical = any(self.repository.get_receipt(identity)["achieved_tier"] != "metadata"
                           for identity in retired)
        try:
            plan = self.capacity_gate.acquire(job["job_id"], physical=physical, reuse=not admission)
            if claim is not None:
                self.capacity_gate.bind(job["job_id"], claim)
            if plan.get("alert"):
                summary["capacity_alert"] = True
                summary["capacity"] = dict(plan)
        except ValueError as exc:
            summary["capacity_blocked"] = summary.get("capacity_blocked", 0) + 1
            summary["capacity_reason"] = str(exc)
            return False
        return True

    def _release_capacity(self, job_id, summary, *, claim=None, reconciled=False):
        if self.capacity_gate is None:
            return
        try:
            self.capacity_gate.release(job_id, claim=claim, reconciled=reconciled)
        except ValueError as exc:
            # Bad ownership metadata must preserve bytes, but cannot prevent a
            # valid sealed receipt from completing/publication at exhaustion.
            summary["capacity_reconciliation_blocked"] = str(exc)

    def _unknown_capacity(self, claim, summary):
        if self.capacity_gate is None:
            return
        try:
            self.capacity_gate.mark_unknown(claim["job_id"], claim)
        except ValueError as exc:
            summary["capacity_reconciliation_blocked"] = str(exc)

    async def _await_execution(self, awaitable, claim):
        generation = self.repository.get_generation(claim["generation_id"])
        execution = asyncio.create_task(awaitable)

        def consume(task):
            if not task.cancelled():
                task.exception()

        try:
            if generation["state"] != "candidate":
                return await asyncio.shield(execution)
            while True:
                window = self.repository.get_active_maintenance_window()
                if not window:
                    raise TimeoutError("candidate execution maintenance window expired")
                attempt = self.repository.get_attempt(claim["attempt_id"])
                deadlines = [datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                             for value in (window["ends_at"], attempt["lease_expires_at"])]
                remaining = (min(deadlines) - datetime.now(timezone.utc)).total_seconds()
                if remaining <= 0:
                    raise TimeoutError("candidate execution claim/window deadline expired")
                done, _ = await asyncio.wait({execution}, timeout=remaining)
                if done:
                    return execution.result()
                # A heartbeat can extend the execution lease, never the window.
        except BaseException:
            if not execution.done():
                execution.cancel()
                execution.add_done_callback(consume)
            raise

    def _build_allowed(self):
        if self.breaker is None:
            from papervault.knowledge.store.build_breaker import get_breaker
            self.breaker = get_breaker()
        return self.breaker.allows()

    def _discrepancy(self, reason, identity, evidence=None):
        detail = {"identity": {key: identity.get(key) for key in CLAIM_FIELDS}, **(evidence or {})}
        discrepancy_id = "discrepancy:" + payload_hash(wire({"reason": reason, "evidence": detail}))
        self.repository.command("record_discrepancy", discrepancy_id, {"discrepancy_id": discrepancy_id,
            "unit_id": identity.get("unit_id"), "reason": reason, "evidence": wire(detail)})

    def _observe(self, observation):
        value = wire(observation)
        if not value.get("observed_at"):
            raise ValueError("collector must supply actual backend observation time")
        value.setdefault("observation_id", "observation:" + payload_hash(value))
        self.repository.command("observe", value["observation_id"], value)

    def _verify_input(self, artifact):
        key = (artifact["artifact_id"], artifact["sha256"], artifact["byte_length"])
        if key not in self._input_observation_cache:
            present = True
            try:
                self.input_store.read(artifact)
            except (OSError, RuntimeError, ValueError):
                present = False
            self._input_observation_cache[key] = (present, datetime.now(timezone.utc).isoformat())
        return self._input_observation_cache[key]

    async def _collect(self, receipt, summary, *, backend=True):
        manifest_row = self.repository.get_input(receipt["consumed_input_revision"])
        manifest = manifest_row["manifest"] if manifest_row else {}
        for artifact in manifest.get("artifacts", []):
            present, stamp = self._verify_input(artifact)
            observation = {key: receipt.get(key) for key in CLAIM_FIELDS}
            observation.update(role="input_evidence", store="library", output_id=artifact["artifact_id"],
                               observed_at=stamp, state="present", receipt_id=receipt["receipt_id"],
                               detail={key: artifact[key] for key in ("artifact_id", "sha256", "byte_length")})
            if not present:
                observation.update(state="missing", detail={**observation["detail"], "verification_failed": True})
            self._observe(observation)
            summary["observations"] += 1
        section = self.repository.get_section(receipt["unit_id"], manifest.get("section_definition_revision"))
        if section:
            inventory = self.repository.get_current_inventory(section["parent_unit_id"], receipt["generation_id"])
            parent_input = self.repository.get_input(inventory.get("parent_consumed_input_revision")) if inventory else None
            if parent_input:
                for artifact in parent_input["manifest"].get("artifacts", []):
                    present, stamp = self._verify_input(artifact)
                    self._observe({"unit_id": section["parent_unit_id"],
                        "consumed_input_revision": parent_input["consumed_input_revision"],
                        "role": "input_evidence", "store": "library", "output_id": artifact["artifact_id"],
                        "state": "present" if present else "missing", "observed_at": stamp,
                        "detail": {key: artifact[key] for key in ("artifact_id", "sha256", "byte_length")}})
                    summary["observations"] += 1
        if not backend:
            return
        if (receipt["achieved_tier"] == "metadata" and len(receipt.get("aggregate_layers", [])) == 1
                and all(o["role"] in {"metadata", "tombstones"} for o in receipt["output_inventory"])):
            artifacts = {a["artifact_id"]: a for a in manifest.get("artifacts", [])}
            outputs = []
            for output in receipt["output_inventory"]:
                if output["role"] == "tombstones":
                    continue
                state = "present"
                try:
                    self.input_store.read(artifacts[output["output_id"]])
                except (KeyError, OSError, RuntimeError, ValueError):
                    state = "missing"
                outputs.append({**{key: receipt[key] for key in CLAIM_FIELDS}, "receipt_id": receipt["receipt_id"],
                    **{key: output[key] for key in ("role", "store", "namespace", "output_id")},
                    "observed_at": datetime.now(timezone.utc).isoformat(), "state": state,
                    "detail": {"sha256": output["sha256"], "byte_length": output["byte_length"]}})
        else:
            collector = getattr(self.adapter, "collect_observations", None)
            if collector is None:
                self._discrepancy("collector_unavailable", receipt)
                summary["discrepancies"] += 1
                return
            try:
                outputs = collector(receipt)
                if inspect.isawaitable(outputs):
                    outputs = await outputs
            except Exception as exc:
                self._discrepancy("collector_unavailable", receipt, {"error_type": type(exc).__name__})
                summary["discrepancies"] += 1
                return
        for output in outputs:
            try:
                self._observe(output)
            except (ValueError, ClaimRejected):
                self._discrepancy("unqualified_backend_observation", receipt, {"observation": wire(output)})
                summary["discrepancies"] += 1
            else:
                summary["observations"] += 1
                if output.get("receipt_id"):
                    self._collected_outputs.setdefault(output["receipt_id"], set()).add(
                        (output["role"], output["store"], output["output_id"]))
        for output in receipt["output_inventory"]:
            if output["role"] != "tombstones":
                continue
            current = self.repository.tombstone_inventory(receipt["namespace"])
            identical = all(current[key] == output[key] for key in ("output_id", "sha256", "byte_length"))
            self._observe({**{key: receipt[key] for key in CLAIM_FIELDS}, "receipt_id": receipt["receipt_id"],
                **{key: output[key] for key in ("role", "store", "namespace", "output_id")},
                "observed_at": current["verified_at"], "state": "present" if identical else "missing",
                "detail": {"sha256": current["sha256"], "byte_length": current["byte_length"],
                           "verification": current["verification"]}})
            summary["observations"] += 1
            self._collected_outputs.setdefault(receipt["receipt_id"], set()).add(
                (output["role"], output["store"], output["output_id"]))

    async def _collect_publication_events(self, summary):
        for event in self.repository.list_outbox(kind="publication"):
            receipt_id = event["payload"].get("receipt_id")
            if not receipt_id:
                derivation = self.repository.get_derivation(event["payload"]["derivation_id"])
                receipt_id = derivation["receipt_id"] if derivation else None
            receipt = self.repository.get_receipt(receipt_id) if receipt_id else None
            if not receipt:
                continue
            expected = {(o["role"], o["store"], o["output_id"]) for o in receipt["output_inventory"]}
            if not expected <= self._collected_outputs.get(receipt_id, set()):
                await self._collect(receipt, summary)
            if expected <= self._collected_outputs.get(receipt_id, set()):
                self._command_once("acknowledge_outbox", "ack:"+event["outbox_id"], {"outbox_id": event["outbox_id"]})

    def _current_claim(self, job):
        if not job.get("current_attempt_id"):
            return None
        attempt = self.repository.get_attempt(job["current_attempt_id"])
        submission = self.repository.get_submission(attempt["submission_id"])
        return wire(submission["claim"]) if submission else None

    def _command_once(self, action, operation_id, payload):
        prior = self.repository.get_operation(operation_id)
        if prior is not None:
            return prior
        try:
            return self.repository.command(action, operation_id, wire(payload))
        except Exception:
            committed = self.repository.get_operation(operation_id)
            if committed is not None:
                return committed
            raise

    async def _finish(self, receipt, summary):
        job = self.repository.get_job(receipt["job_id"])
        claim = self._current_claim(job) if job else None
        if not claim:
            self._discrepancy("unmatched_receipt", receipt)
            summary["discrepancies"] += 1
            return False
        recipe = self.repository.get_recipe(job["recipe_digest"])
        try:
            validate_receipt(receipt, claim, recipe)
            if self.capacity_gate is not None:
                sealed_attempt = self.repository.get_attempt(claim["attempt_id"])
                registered = self.repository.get_receipt(receipt["receipt_id"])
                if (sealed_attempt.get("sealed") is True
                        and sealed_attempt.get("receipt_id") == receipt["receipt_id"]
                        and registered == receipt):
                    # Catalog-sealed exact outputs settle physical ownership even
                    # when expired completion leases or a stale base need rebase.
                    self._release_capacity(job["job_id"], summary, claim=claim, reconciled=True)
            if job["state"] == "running":
                self._command_once("complete", "complete:" + receipt["receipt_id"],
                    {**claim, "expected_revision": job["revision"], "receipt": receipt})
                summary["completed"] += 1
                job = self.repository.get_job(job["job_id"])
            if job["state"] != "succeeded":
                return False
            publication_operation = "publish:" + receipt["receipt_id"]
            if self.repository.get_operation(publication_operation):
                return True
            attempt = self.repository.get_attempt(job["current_attempt_id"])
            clock = datetime.now(timezone.utc)
            expiry = datetime.fromisoformat(str(attempt["lease_expires_at"]).replace("Z", "+00:00"))
            if expiry <= clock:
                publication_expiry = attempt.get("publication_lease_expires_at")
                if publication_expiry:
                    publication_expiry = datetime.fromisoformat(str(publication_expiry).replace("Z", "+00:00"))
                if publication_expiry and publication_expiry > clock:
                    claim["publication_claim_id"] = attempt["publication_claim_id"]
                else:
                    claim = self.repository.command("claim_publication",
                        "publication-claim:" + receipt["receipt_id"] + ":" + str(attempt.get("publication_claim_id")),
                        {"job_id": job["job_id"], "expected_revision": job["revision"],
                         "authority_epoch": self.repository.get_authority_state()["authority_epoch"],
                         "lease_seconds": self.lease_seconds})
            selection = self.repository.get_selection(job["unit_id"], job["generation_id"])
            result = self._command_once("publish_derivation", publication_operation,
                {**claim, "expected_revision": job["revision"], "receipt": receipt,
                 "expected_selection_revision": selection["revision"] if selection else 0, "select": True})
            summary["published"] += 1
            return bool(result)
        except (ValueError, ClaimRejected) as exc:
            self._discrepancy("receipt_completion_or_publication_rejected", receipt, {"error_type": type(exc).__name__})
            summary["discrepancies"] += 1
            return False

    async def _heartbeat(self, job_id, stop, errors):
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(0.05, self.lease_seconds / 3))
                return
            except asyncio.TimeoutError:
                pass
            job = self.repository.get_job(job_id)
            claim = self._current_claim(job)
            if not claim or job["state"] != "running":
                return
            try:
                self.repository.command("renew", "renew:" + claim["attempt_id"] + ":" + str(job["revision"]),
                    {**claim, "expected_revision": job["revision"], "lease_seconds": self.lease_seconds})
            except Exception as exc:
                errors.append(exc)
                return

    async def _deliver(self, row, summary):
        claim = wire(row["payload"])
        job = self.repository.get_job(claim["job_id"])
        if not job or job.get("current_attempt_id") != claim["attempt_id"]:
            self._command_once("acknowledge_outbox", "ack:" + row["outbox_id"], {"outbox_id": row["outbox_id"]})
            return False
        try:
            # Recover catalog registration before asking the external adapter.
            submission = self.repository.get_submission(claim["submission_id"])
            receipt = submission.get("receipt") if submission else None
            if not receipt:
                receipt = await self.adapter.lookup_submission(claim["submission_id"])
            recovered = bool(receipt)
            attempt = self.repository.get_attempt(claim["attempt_id"])
            if attempt["outcome"] == "deferred" or (job["state"] in {"failed", "cancelled"} and attempt["outcome"] is not None):
                self._command_once("acknowledge_outbox", "ack:"+row["outbox_id"], {"outbox_id": row["outbox_id"]})
                return False
            expires = datetime.fromisoformat(str(attempt["lease_expires_at"]).replace("Z", "+00:00"))
            if job["state"] == "running" and expires <= datetime.now(timezone.utc):
                self._command_once("expire_attempt", "expire:"+attempt["attempt_id"], {
                    "job_id": job["job_id"], "attempt_id": attempt["attempt_id"], "expected_revision": job["revision"],
                    "authority_epoch": self.repository.get_authority_state()["authority_epoch"]})
                return False
            if not receipt:
                if self.capacity_gate is not None and not self._candidate_allowed(job):
                    return False
                if not self._reserve_job(job, summary, claim=claim):
                    return False
                manifest = self.repository.get_input(claim["consumed_input_revision"])["manifest"]
                data = [self.input_store.read(a) for a in manifest["artifacts"]]
                evidence = [{"path": str(self.input_store.root / artifact["sha256"]),
                             "sha256": artifact["sha256"], "byte_length": artifact["byte_length"]}
                            for artifact in manifest["artifacts"]]
                claim["input_evidence"] = evidence
                stop, errors = asyncio.Event(), []
                heartbeat = asyncio.create_task(self._heartbeat(job["job_id"], stop, errors))
                try:
                    tier = manifest["tier"]
                    if tier == "metadata":
                        old = [self.repository.get_receipt(rid) for rid in claim["retired_receipt_ids"]]
                        requires_removal = any(rec and rec["achieved_tier"] != "metadata" for rec in old)
                        if requires_removal:
                            submit_metadata = getattr(self.adapter, "submit_metadata", None)
                            if submit_metadata is None:
                                self._discrepancy("metadata_downgrade_requires_cow_removal", claim)
                                summary["discrepancies"] += 1
                                return
                            receipt = await self._await_execution(
                                submit_metadata(claim, manifest, self.input_store), claim)
                        else:
                            self._command_once("accept_submission", "accept:" + claim["submission_id"], {"claim": claim})
                            receipt = metadata_receipt(claim, manifest, self.input_store)
                            receipt["output_inventory"].append(self.repository.tombstone_inventory(claim["namespace"]))
                            self._command_once("seal_namespace", "seal:" + claim["submission_id"], {"claim": claim, "receipt": receipt})
                    else:
                        unit = self.repository.get_unit(claim["unit_id"])
                        text = execution_text(unit, tier, data[0])
                        source = citation_file_path(unit, manifest)
                        receipt = await self._await_execution(
                            self.adapter.submit(claim, text, source, achieved_tier=tier), claim)
                    if errors:
                        raise ClaimRejected("heartbeat could not retain completion claim")
                finally:
                    stop.set()
                    await heartbeat
                summary["delivered"] += 1
            await self._collect(receipt, summary)
            if await self._finish(receipt, summary):
                self._command_once("acknowledge_outbox", "ack:" + row["outbox_id"], {"outbox_id": row["outbox_id"]})
            # Recovering a sealed receipt does not execute new work. In
            # particular, a stale aggregate base must leave room to admit its
            # rebased attempt instead of consuming the only delivery slot.
            return not recovered
        except asyncio.CancelledError:
            self._unknown_capacity(claim, summary)
            raise
        except BuildPausedError:
            latest = self.repository.get_job(claim["job_id"])
            current = self._current_claim(latest)
            if current and current["attempt_id"] == claim["attempt_id"]:
                try:
                    self._command_once("defer_attempt", "defer:" + claim["attempt_id"], {**current,
                        "expected_revision": latest["revision"], "reason": {"kind": "BuildPausedError"}})
                    self._release_capacity(claim["job_id"], summary, claim=current, reconciled=True)
                except ClaimRejected:
                    self._discrepancy("pause_deferral_claim_rejected", claim)
                    summary["discrepancies"] += 1
            summary["paused"] = True
            return True
        except RetirementFailure as exc:
            self._discrepancy("retirement_rebuild_unverified", claim, {
                "retired_receipt_id": exc.retired_receipt_id, "doc_id": exc.doc_id,
                "status": exc.status, "messages": list(exc.messages), "retirement_evidence": exc.evidence})
            summary["discrepancies"] += 1
            for publication in self.repository.get_receipt_publications(exc.retired_receipt_id):
                repair = {"publication_id": publication["publication_id"], "role": "graph",
                          "reason": "retirement_rebuild_unverified"}
                self._command_once("request_repair", "repair:" + payload_hash(repair), repair)
                summary["repairs"] += 1
            latest = self.repository.get_job(claim["job_id"])
            current = self._current_claim(latest)
            if current and current["attempt_id"] == claim["attempt_id"]:
                try:
                    self._command_once("fail_retirement", "fail-retirement:" + claim["attempt_id"], {
                        **current, "expected_revision": latest["revision"], "error": {
                            "kind": "RetirementFailure", "isolated_rebuild_required": True,
                            "retired_receipt_id": exc.retired_receipt_id}})
                except ClaimRejected:
                    self._discrepancy("retirement_failure_claim_rejected", claim)
                    summary["discrepancies"] += 1
            # Preserve unknown charges and recovery pins, but fence the private
            # delta. Only a fresh admitted attempt can execute the replacement.
            return True
        except KnownExecutionFailure as exc:
            latest = self.repository.get_job(claim["job_id"])
            current = self._current_claim(latest)
            if current and current["attempt_id"] == claim["attempt_id"]:
                try:
                    self._command_once("fail", "fail:"+claim["attempt_id"], {**current,
                        "expected_revision": latest["revision"], "error": {"kind": type(exc).__name__, "outcome": "known_failure"}})
                    self._release_capacity(job["job_id"], summary, claim=current, reconciled=True)
                    self._command_once("acknowledge_outbox", "ack:"+row["outbox_id"], {"outbox_id": row["outbox_id"]})
                except ClaimRejected:
                    pass
            self._discrepancy("known_execution_failure", claim, {"error_type": type(exc).__name__})
            summary["discrepancies"] += 1
            return True
        except Exception as exc:
            # Submission may already have been accepted. Preserve its attempt,
            # reserved charge and outbox, then look it up on the next delivery.
            self._unknown_capacity(claim, summary)
            self._discrepancy("submission_outcome_unknown", claim, {"error_type": type(exc).__name__})
            summary["discrepancies"] += 1
            return True

    def _repair_missing(self, scope, summary):
        from papervault.store.coverage import query_coverage
        coverage = query_coverage(self.repository, scope)
        for unit in coverage["units"]:
            if not unit.get("derivation_id"):
                continue
            publications = [p for p in unit["publication_history"] if p["derivation_id"] == unit["derivation_id"]]
            if not publications:
                continue
            publication = max(publications, key=lambda p: str(p["published_at"]))
            for role in [*unit["roles"], *unit.get("aggregate_roles", []), *unit.get("input_roles", [])]:
                if role["state"] != "missing":
                    continue
                role_name = role.get("role") or "aggregate:"+str(role.get("namespace", "unknown"))
                repair = {"publication_id": publication["publication_id"], "role": role_name, "reason": "missing_materialization"}
                self._command_once("request_repair", "repair:" + payload_hash(repair), repair)
                summary["repairs"] += 1

    def _fork_repairs_in_maintenance(self, summary):
        if not self.repository.get_active_maintenance_window():
            return
        source = self.repository.get_active_generation()
        if not source:
            return
        identity = {"source_generation_id": source["generation_id"], "source_manifest_revision": source["manifest_revision"]}
        candidate_id = "repair-generation:" + payload_hash(identity)[:32]
        linked = set()
        for repair in self.repository.list_pending_repairs():
            try:
                self._command_once("fork_generation", "fork:"+candidate_id, {
                    "generation_id": candidate_id, "source_generation_id": source["generation_id"],
                    "expected_manifest_revision": source["manifest_revision"],
                    "expected_requirements_revision": self.repository.get_authority_state()["requirements_revision"]})
                latest = self.repository.get_repair(repair["repair_id"])
                if latest["target_generation"] != candidate_id:
                    self._command_once("link_repair", "link:"+repair["repair_id"]+":"+candidate_id+":"+str(latest["revision"]),
                        {"repair_id": repair["repair_id"], "generation_id": candidate_id,
                         "expected_revision": latest["revision"]})
                linked.add(repair["repair_id"])
            except ClaimRejected as exc:
                self._discrepancy("repair_candidate_fork_rejected", {}, {"repair_id": repair["repair_id"],
                    "error_type": type(exc).__name__})
                summary["discrepancies"] += 1
        for event in self.repository.list_outbox(kind="repair"):
            repair = self.repository.get_repair(event["payload"]["repair_id"])
            if repair and (repair["repair_id"] in linked or repair["target_generation"] == candidate_id):
                self._command_once("acknowledge_outbox", "ack:"+event["outbox_id"], {"outbox_id": event["outbox_id"]})

    async def run_round(self, only_keys=None):
        summary = dict.fromkeys(("observations", "discrepancies", "completed", "published", "claimed", "delivered", "repairs"), 0)
        summary["paused"] = False
        summary["capacity_blocked"] = 0
        if self.repository.get_authority_state()["authority"] != "catalog":
            return {**summary, "paused": True, "reason": "catalog_authority_not_committed"}
        scope = None if only_keys is None else set(only_keys)
        self._input_observation_cache = {}
        self._collected_outputs = {}
        def included(job):
            if scope is None or job.get("key") in scope or job.get("unit_id") in scope:
                return True
            actual = self.repository.get_job(job["job_id"]) if job.get("job_id") else None
            return bool(actual and actual.get("key") in scope)
        seen = set()
        for receipt in self.repository.list_collection_heads():
            seen.add(receipt["receipt_id"])
            await self._collect(receipt, summary)
        for receipt in self.repository.list_selected_receipts(scope):
            if receipt["receipt_id"] not in seen:
                seen.add(receipt["receipt_id"])
                await self._collect(receipt, summary, backend=False)
        for receipt in self.repository.list_reconcilable_receipts(scope):
            if receipt["receipt_id"] not in seen:
                seen.add(receipt["receipt_id"])
                await self._collect(receipt, summary)
            await self._finish(receipt, summary)
        await self._collect_publication_events(summary)
        self._repair_missing(scope, summary)
        if not self._build_allowed():
            summary["paused"] = True
            return summary
        self._fork_repairs_in_maintenance(summary)
        delivered = 0
        handled = set()
        for row in self.repository.list_outbox(kind="submission"):
            if row["kind"] == "submission" and included(row["payload"]) and delivered < self.max_admissions:
                handled.add(row["outbox_id"])
                delivered += bool(await self._deliver(row, summary))
                if summary.get("paused"):
                    return summary
        for job in self.repository.list_admission_jobs(scope):
            if summary["claimed"] >= max(0, self.max_admissions - delivered):
                break
            attempt = self.repository.get_attempt(job["current_attempt_id"]) if job.get("current_attempt_id") else None
            expired = bool(attempt and datetime.fromisoformat(str(attempt["lease_expires_at"]).replace("Z", "+00:00")) <= datetime.now(timezone.utc))
            rebase = bool(job["state"] == "succeeded" and attempt and
                self.repository.get_generation(job["generation_id"])["manifest_revision"] != attempt["aggregate_base_manifest_revision"] and
                not self.repository.get_operation("publish:" + str(attempt.get("receipt_id"))))
            if job["state"] not in {"queued", "failed"} and not (job["state"] == "running" and expired) and not rebase:
                continue
            if self.capacity_gate is not None and not self._candidate_allowed(job):
                continue
            if job["state"] == "queued" and attempt and attempt["outcome"] == "deferred":
                # Recover a crash between catalog deferral and capacity settlement.
                # Partial bytes stay charged until a census begun after settlement;
                # the exact old claim cannot release another attempt's reservation.
                deferred = self._current_claim(job)
                if deferred:
                    self._release_capacity(job["job_id"], summary, claim=deferred, reconciled=True)
            if not self._reserve_job(job, summary, admission=True):
                continue
            try:
                admitted = self.repository.command("claim", "claim:" + job["job_id"] + ":" + str(job["revision"]),
                    {"job_id": job["job_id"], "expected_revision": job["revision"],
                     "authority_epoch": self.repository.get_authority_state()["authority_epoch"],
                     "lease_seconds": self.lease_seconds, "reserved_cost": self.reserved_cost, "rebase": rebase})
            except ClaimRejected:
                self._release_capacity(job["job_id"], summary)
                continue
            if admitted.get("attempt_id"):
                if self.capacity_gate is not None:
                    self.capacity_gate.bind(job["job_id"], admitted)
                summary["claimed"] += 1
            else:
                self._release_capacity(job["job_id"], summary)
        for row in self.repository.list_outbox(kind="submission"):
            if row["outbox_id"] not in handled and row["kind"] == "submission" and included(row["payload"]) and delivered < self.max_admissions:
                delivered += bool(await self._deliver(row, summary))
                if summary.get("paused"):
                    return summary
        await self._collect_publication_events(summary)
        return summary
