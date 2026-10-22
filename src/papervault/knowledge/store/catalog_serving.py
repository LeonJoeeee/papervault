"""Catalog-selected namespace reads, pinned for one complete synthesis request."""
from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from uuid import uuid4


def _selection(snapshot):
    if not snapshot:
        raise RuntimeError("catalog serving manifest absent")
    entries = snapshot.get("entries")
    if entries is None:
        raise RuntimeError("serving manifest lacks eligibility verification")
    admitted = []
    selected_receipts = snapshot.get("manifest", {}).get("selected_unit_receipts", {})
    for entry in entries:
        if "eligible" not in entry:
            raise RuntimeError("serving manifest lacks current policy/evidence verification")
        if not entry["eligible"]:
            if entry["unit_id"] in selected_receipts:
                raise RuntimeError("selected aggregate contains an ineligible contribution")
            continue
        receipt = entry.get("receipt") or {}
        if not receipt.get("sealed") or not receipt.get("durable") or not receipt.get("namespace"):
            raise RuntimeError("serving manifest references an unsealed publication")
        admitted.append((entry["unit_id"], receipt["namespace"], receipt.get("receipt_id")))
        if selected_receipts.get(entry["unit_id"]) != receipt.get("receipt_id"):
            raise RuntimeError("selected aggregate receipt set differs from catalog selection")
    if set(selected_receipts) != {unit_id for unit_id, _, _ in admitted}:
        raise RuntimeError("selected aggregate contains a receipt without a current unit entry")
    return tuple(sorted(admitted))


class _PinnedCoverageRows:
    """Use the pinned aggregate while retaining fresh catalog authority/evidence rows."""

    def __init__(self, repository, snapshot):
        self.repository = repository
        self.generation_id = snapshot["generation_id"]
        self.manifest = {"generation_id": self.generation_id,
                         "manifest_revision": snapshot["manifest_revision"],
                         "manifest": snapshot["manifest"]}

    def coverage_rows(self, scope=None):
        # One shared manifest object preserves coverage's aggregate-verification cache.
        return [dict(row, generation_manifest=self.manifest)
                if (row.get("requirement") or {}).get("generation_id") == self.generation_id else row
                for row in self.repository.coverage_rows(scope)]


class CatalogServingFacade:
    def __init__(self, repository, snapshot, *, rag_factory=None, working_dir=None, pin_ttl_seconds=300):
        from papervault.knowledge.store.graph import build_catalog_graph

        self.repository = repository
        self.snapshot = copy.deepcopy(snapshot)
        self.selection = _selection(snapshot)
        self.layers = tuple(snapshot.get("manifest", {}).get("aggregate_layers", []))
        if self.selection and not self.layers:
            raise RuntimeError("catalog serving manifest lacks aggregate layers")
        if any(namespace not in self.layers for _, namespace, _ in self.selection):
            raise RuntimeError("selected receipt namespace absent from aggregate layers")
        self.rag_factory = rag_factory or build_catalog_graph
        self.working_dir = working_dir
        self._graph_instance = None
        self._open_lock = asyncio.Lock()
        self._pin_ttl = pin_ttl_seconds
        self._pin_owner = "query:" + uuid4().hex
        self._pin = None
        self._pin_error = None
        self._heartbeat = None
        self._close_task = None

    @classmethod
    async def pin(cls, repository, **kwargs):
        # Repository applies current source/policy/quarantine and receipt-backed
        # materialization freshness checks before any storage object is opened.
        snapshot = await asyncio.to_thread(repository.get_serving_manifest, max_observation_age=300)
        instance = cls(repository, snapshot, **kwargs)
        instance._pin = await asyncio.to_thread(instance._pin_command, "pin_manifest", "pin:" + instance._pin_owner,
            {"generation_id": instance.snapshot["generation_id"],
             "manifest_revision": instance.snapshot["manifest_revision"],
             "expected_serving_revision": instance.snapshot["revision"],
             "owner": instance._pin_owner, "ttl_seconds": instance._pin_ttl})
        instance._heartbeat = asyncio.create_task(instance._renew_pin())
        return instance

    def _pin_command(self, action, operation_id, payload):
        try:
            return self.repository.command(action, operation_id, payload)
        except Exception:
            # A committed command with a lost response is still one operation.
            committed = self.repository.get_operation(operation_id)
            if committed is not None:
                return committed
            raise

    async def _renew_pin(self):
        try:
            while True:
                await asyncio.sleep(self._pin_ttl / 3)
                renewal = asyncio.create_task(asyncio.to_thread(self._pin_command, "renew_pin",
                    f"renew-pin:{self._pin['pin_id']}:{self._pin['revision']}",
                    {"pin_id": self._pin["pin_id"], "expected_revision": self._pin["revision"],
                     "owner": self._pin_owner, "ttl_seconds": self._pin_ttl}))
                try:
                    self._pin = await asyncio.shield(renewal)
                except asyncio.CancelledError:
                    # Cancelling to_thread cannot stop its database command. Drain it
                    # and retain the committed revision before close releases the pin.
                    self._pin = await renewal
                    raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._pin_error = exc

    def validate(self):
        contract = self.snapshot.get("manifest", {}).get("retrieval_contract")
        if contract is not None:
            from papervault.knowledge.scheduler.catalog_intake import effective_embedding_contract

            if contract != effective_embedding_contract():
                raise RuntimeError("runtime embedding contract differs from the selected generation; rebuild required")
        if self._pin_error is not None:
            raise RuntimeError("catalog reader pin renewal failed") from self._pin_error
        if self._pin is None:
            raise RuntimeError("catalog reader manifest is not durably pinned")
        expiry = self._pin["expires_at"]
        expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00")) if isinstance(expiry, str) else expiry
        if expiry <= datetime.now(timezone.utc):
            raise RuntimeError("catalog reader pin expired")
        if (_selection(self.snapshot) != self.selection or
                tuple(self.snapshot.get("manifest", {}).get("aggregate_layers", [])) != self.layers):
            raise RuntimeError("pinned catalog serving manifest invariants changed")
        if self.selection:
            from papervault.store.coverage import query_coverage

            # Promotion cannot change the view midway through a read. Recheck policy,
            # input compatibility and fresh evidence for the exact pinned aggregate,
            # rather than selecting a different active generation or newer layer set.
            coverage = query_coverage(
                _PinnedCoverageRows(self.repository, self.snapshot),
                scope=[unit_id for unit_id, _, _ in self.selection],
                generation_id=self.snapshot["generation_id"], max_observation_age=300,
            )
            by_unit = {unit["unit_id"]: unit for unit in coverage["units"]}
            for unit_id, namespace, receipt_id in self.selection:
                unit = by_unit.get(unit_id, {})
                receipt = unit.get("receipt") or {}
                if (not unit.get("source_eligible") or not unit.get("tier_covered")
                        or (unit.get("achieved_tier") == "fulltext" and not unit.get("readable"))
                        or receipt.get("receipt_id") != receipt_id
                        or receipt.get("namespace") != namespace):
                    raise RuntimeError("pinned catalog selection is ineligible or evidence unavailable")

    async def _graph(self):
        async with self._open_lock:
            if self._graph_instance is None:
                self._graph_instance = await self.rag_factory(
                    self.repository, {"namespace": self.layers[-1], "aggregate_layers": list(self.layers)},
                    read_only=True,
                    working_dir=self.working_dir)
            return self._graph_instance

    async def aquery_data(self, query, param):
        await asyncio.to_thread(self.validate)
        if not self.layers:
            return {"status": "success", "data": {}}
        graph = await self._graph()
        result = await graph.aquery_data(query, param)
        empty = (result.get("status") == "failure"
                 and (result.get("metadata") or {}).get("failure_reason") == "no_results")
        if result.get("status") != "success" and not empty:
            raise RuntimeError("selected aggregate retrieval failed")
        await asyncio.to_thread(self.validate)
        return result

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        cancelled = False
        while True:
            try:
                await asyncio.shield(self._close_task)
                break
            except asyncio.CancelledError:
                if self._close_task.cancelled():
                    raise
                # Caller cancellation must not cancel a database command or its
                # revision bookkeeping. Finish cleanup, then preserve cancellation.
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self):
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            try:
                await self._heartbeat
            except asyncio.CancelledError:
                pass
            self._heartbeat = None
        graph, self._graph_instance = self._graph_instance, None
        try:
            if graph is not None:
                await graph.finalize_storages()
        finally:
            if self._pin is not None:
                await asyncio.to_thread(self._pin_command, "release_pin", f"release-pin:{self._pin['pin_id']}",
                    {"pin_id": self._pin["pin_id"], "expected_revision": self._pin["revision"],
                     "owner": self._pin_owner})
                self._pin = None


async def get_query_graph():
    from papervault.store.knowledge_runtime import (
        assert_knowledge_authority, get_knowledge_repository, knowledge_jobs_mode,
    )

    if knowledge_jobs_mode() == "postgres":
        await asyncio.to_thread(assert_knowledge_authority, "catalog")
        return await CatalogServingFacade.pin(get_knowledge_repository())
    from papervault.knowledge.store.graph import get_graph

    return await get_graph()
