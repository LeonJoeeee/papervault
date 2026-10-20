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
                self._pin = await asyncio.to_thread(self._pin_command, "renew_pin",
                    f"renew-pin:{self._pin['pin_id']}:{self._pin['revision']}",
                    {"pin_id": self._pin["pin_id"], "expected_revision": self._pin["revision"],
                     "owner": self._pin_owner, "ttl_seconds": self._pin_ttl})
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
        current = self.repository.get_serving_manifest(max_observation_age=300)
        identity = lambda row: (row.get("generation_id"), row.get("manifest_revision"))
        if (identity(current) != identity(self.snapshot) or _selection(current) != self.selection or
                tuple(current.get("manifest", {}).get("aggregate_layers", [])) != self.layers):
            raise RuntimeError("catalog serving manifest or eligibility changed during query")

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
        if result.get("status") != "success":
            raise RuntimeError("selected aggregate retrieval failed")
        await asyncio.to_thread(self.validate)
        return result

    async def close(self):
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
