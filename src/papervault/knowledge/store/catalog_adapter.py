"""Recoverable, isolated LightRAG deliveries and backend-verified receipts."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from functools import wraps

from lightrag.kg.postgres_impl import namespace_to_table_name
from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

_IDENTITY = ("job_id", "attempt_id", "submission_id", "authority_epoch", "fencing_token",
             "unit_id", "consumed_input_revision", "recipe_digest", "generation_id", "namespace")
_COW_IDENTITY = ("aggregate_base_manifest_revision", "base_aggregate_layers",
                 "base_selected_unit_receipts", "retired_receipt_ids")
_PG_STORES = ("full_docs", "text_chunks", "full_entities", "full_relations",
              "entity_chunks", "relation_chunks", "llm_response_cache", "doc_status",
              "chunks_vdb", "entities_vdb", "relationships_vdb")
_recipe = ContextVar("catalog_execution_recipe", default=None)


def current_execution_recipe():
    return _recipe.get()


class _RecipePrompts(Mapping):
    """Keep pinned LightRAG's global prompt reads local to each build task."""
    def __init__(self, defaults):
        self.defaults = defaults

    def _values(self):
        bound = _recipe.get()
        return bound["prompts"] if bound is not None else self.defaults

    def __getitem__(self, key):
        return self._values()[key]

    def __iter__(self):
        return iter(self._values())

    def __len__(self):
        return len(self._values())


@contextmanager
def execution_recipe(recipe):
    import lightrag.operate as operate

    if not isinstance(operate.PROMPTS, _RecipePrompts):
        operate.PROMPTS = _RecipePrompts(operate.PROMPTS)
    token = _recipe.set(deepcopy(recipe))
    try:
        yield
    finally:
        _recipe.reset(token)


def _admitted_recipe(method):
    @wraps(method)
    async def guarded(self, claim, *args, **kwargs):
        # Already sealed accepted work recovers under its historical recipe.
        if await self.lookup_submission(claim["submission_id"]) is not None:
            return await method(self, claim, *args, **kwargs)
        snapshot = self._verify_recipe(claim)
        with execution_recipe(snapshot):
            return await method(self, claim, *args, **kwargs)
    return guarded


def _bytes(value):
    def encode(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "to_list"):
            return item.to_list()
        if isinstance(item, (datetime, date)):
            return item.isoformat()
        if isinstance(item, Decimal):
            return str(item)
        raise TypeError(f"unsupported backend proof value: {type(item).__name__}")

    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=encode, ensure_ascii=False, allow_nan=False).encode("utf-8")


class CatalogLightRAGAdapter:
    def __init__(self, repository, *, working_dir=None, rag_factory=None, input_store=None):
        from papervault.knowledge.store.graph import build_catalog_graph

        self.repository = repository
        self.working_dir = working_dir
        self.rag_factory = rag_factory or build_catalog_graph
        self.input_store = input_store

    async def lookup_submission(self, submission_id):
        submission = self.repository.get_submission(submission_id)
        if not submission:
            return None
        if submission.get("receipt"):
            return submission["receipt"]
        if submission.get("receipt_id"):
            return self.repository.get_receipt(submission["receipt_id"])
        return None

    @_admitted_recipe
    async def submit(self, claim, text, file_path, *, achieved_tier="fulltext"):
        """Redelivery uses the same durable submission and document ID.

        A lost execution claim needs a newly admitted attempt; delivery retries
        never allocate attempts. Output sealing and receipt persistence happen
        under the same durable gate used by every backend mutator.
        """
        receipt = await self._recover_or_accept(claim)
        if receipt is not None:
            return receipt
        self._verify_inputs(claim)
        rag = await self.rag_factory(self.repository, claim, working_dir=self.working_dir)
        doc_id = "doc-" + hashlib.sha256(claim["submission_id"].encode()).hexdigest()
        try:
            await self._retire(rag, claim)
            await rag.ainsert(text, ids=doc_id, file_paths=file_path,
                              track_id=claim["submission_id"])
            # Finish every buffered vector operation before entering the sealing
            # section. No pipeline tasks remain after awaited ainsert.
            for storage in rag._index_storages():
                await storage.index_done_callback()
            async with self.repository.namespace_access(claim, seal=True):
                self._verify_inputs(claim)
                inventory, observations = await verify_outputs(rag, claim, doc_id, text)
                receipt = self._seal(claim, achieved_tier, inventory, observations)
            return receipt
        finally:
            await rag.finalize_storages()

    async def _recover_or_accept(self, claim):
        for field in _IDENTITY + _COW_IDENTITY:
            if field not in claim:
                raise ValueError(f"missing admitted claim field: {field}")
        receipt = await self.lookup_submission(claim["submission_id"])
        if receipt is not None:
            if any(receipt.get(field) != claim[field] for field in _IDENTITY + _COW_IDENTITY):
                raise RuntimeError("recovered receipt identity differs from admitted claim")
            if not receipt.get("sealed") or not receipt.get("durable"):
                raise RuntimeError("recovered receipt is not durably sealed")
            return receipt
        # Delivery identity survives lease renewal and locator changes. A
        # stable operation ID must never include mutable lease timestamps or
        # optimistic job revisions in its idempotency payload.
        stable = {key: claim[key] for key in _IDENTITY + _COW_IDENTITY + ("lease_id",)}
        self.repository.command("accept_submission", f"accept:{claim['submission_id']}",
                                {"claim": stable})
        return None

    async def _retire(self, rag, claim):
        for retired_id in claim["retired_receipt_ids"]:
            retired = self.repository.get_receipt(retired_id)
            old_doc = (retired or {}).get("provider_observations", {}).get("doc_id")
            if not old_doc:
                if retired and retired.get("achieved_tier") == "metadata":
                    continue
                raise RuntimeError("retired contribution lacks verified document identity")
            result = await rag.adelete_by_doc_id(old_doc)
            if result.status not in {"success", "not_found"}:
                raise RuntimeError("original LightRAG deletion/rebuild did not complete")

    @_admitted_recipe
    async def submit_metadata(self, claim, manifest, input_store):
        """A downgrade removes the prior contribution using LightRAG's reducer."""
        if manifest.get("tier") != "metadata" or len(manifest.get("artifacts", [])) != 1:
            raise ValueError("metadata publication requires one exact metadata artifact")
        artifact = manifest["artifacts"][0]
        admitted = self.repository.get_input(claim["consumed_input_revision"])
        if not admitted or admitted["manifest"] != manifest:
            raise RuntimeError("metadata manifest differs from exact admitted inputs")
        input_store.read(artifact)
        self.input_store = input_store
        recovered = await self._recover_or_accept(claim)
        if recovered is not None:
            return recovered
        rag = await self.rag_factory(self.repository, claim, working_dir=self.working_dir)
        try:
            await self._retire(rag, claim)
            for storage in rag._index_storages():
                await storage.index_done_callback()
            async with self.repository.namespace_access(claim, seal=True):
                inventory, stamp = await verify_delta_outputs(rag, claim)
                data = input_store.read(artifact)
                inventory.append({"role": "metadata", "store": "catalog_inputs",
                    "namespace": claim["namespace"], "output_id": artifact["artifact_id"],
                    "backend_ids": [artifact["artifact_id"]], "sha256": hashlib.sha256(data).hexdigest(),
                    "byte_length": len(data), "verified_at": stamp,
                    "verification": {"state": "present", "method": "immutable_file_readback_sha256_v1",
                                     "evidence": {"artifact_id": artifact["artifact_id"], "verified": True}}})
                return self._seal(claim, "metadata", inventory,
                                  {"metadata_only": True, "observed_at": stamp,
                                   "provider": "unknown", "model": "unknown", "version": "unknown"})
        finally:
            await rag.finalize_storages()

    def _verify_inputs(self, claim):
        admitted = self.repository.get_input(claim["consumed_input_revision"])
        artifacts = (admitted or {}).get("manifest", {}).get("artifacts", [])
        if not artifacts:
            raise RuntimeError("exact captured input inventory is required")
        evidence = claim.get("input_evidence", [])
        by_digest = {row["sha256"]: row for row in evidence}
        for artifact in artifacts:
            try:
                if self.input_store is not None:
                    data = self.input_store.read(artifact)
                else:
                    source = by_digest[artifact["sha256"]]
                    if source["byte_length"] != artifact["byte_length"]:
                        raise ValueError("input length differs from admitted manifest")
                    path = Path(source["path"])
                    if path.is_symlink():
                        raise ValueError("captured input path is a symlink")
                    data = path.read_bytes()
                if len(data) != artifact["byte_length"] or hashlib.sha256(data).hexdigest() != artifact["sha256"]:
                    raise ValueError("captured input hash/length mismatch")
            except (KeyError, OSError, ValueError, RuntimeError) as exc:
                raise RuntimeError("captured input verification failed") from exc

    def _verify_recipe(self, claim):
        from papervault.knowledge.scheduler.catalog_intake import effective_distill_recipe
        from papervault.store.knowledge import canonical_recipe

        admitted = self.repository.get_recipe(claim["recipe_digest"])
        if admitted is None:
            raise RuntimeError("admitted distillation recipe unavailable")
        snapshot, admitted_digest = canonical_recipe(admitted)
        _, effective_digest = canonical_recipe(effective_distill_recipe())
        if admitted_digest != claim["recipe_digest"] or effective_digest != admitted_digest:
            raise RuntimeError("effective distillation recipe differs from admitted recipe")
        return snapshot

    def _seal(self, claim, achieved_tier, inventory, observations):
        from papervault.store.repository import payload_hash

        self._verify_recipe(claim)
        tombstones = _tombstones(self.repository.list_tombstones(claim["namespace"]))
        inventory.append(_inventory("tombstones", "catalog", claim["namespace"], tombstones,
                                    [row["role"] + ":" + row["output_id"] for row in tombstones],
                                    datetime.now(timezone.utc).isoformat()))
        receipt = {key: claim[key] for key in _IDENTITY + _COW_IDENTITY}
        receipt.update(receipt_id="receipt-" + hashlib.sha256(_bytes(receipt)).hexdigest(),
                       achieved_tier=achieved_tier, output_inventory=inventory,
                       sealed=True, durable=True, provider_observations=observations,
                       origin_lineage={})
        selected = {**claim["base_selected_unit_receipts"], claim["unit_id"]: receipt["receipt_id"]}
        receipt.update(aggregate_layers=claim["base_aggregate_layers"] + [claim["namespace"]],
                       selected_unit_receipts=selected,
                       selected_unit_receipts_digest=payload_hash(selected))
        self.repository.command("seal_namespace", f"seal:{claim['submission_id']}",
                                {"claim": claim, "receipt": receipt})
        return receipt

    async def collect_observations(self, receipt):
        """Batch actual physical readback for every transitive aggregate layer.

        Backend outages produce unknown observations, never fresh positives.
        One LightRAG resource set serves the collection, independent of the
        number of immutable publication layers being checked.
        """
        layers = receipt["aggregate_layers"]
        inherited = self.repository.get_namespace_receipts(layers, compact=True)
        receipts = {row["namespace"]: row for row in inherited}
        if set(receipts) != set(layers):
            raise RuntimeError("aggregate ancestor receipt unavailable")
        for index, namespace in enumerate(layers):
            ancestor = receipts[namespace]
            if not ancestor.get("sealed") or not ancestor.get("durable"):
                raise RuntimeError("aggregate ancestor is not durably sealed")
            # The catalog projects these scalars from its immutable, validated
            # full receipt. Downloading every cumulative prefix here is quadratic.
            if (ancestor.get("compact_projection") is not True or
                    not ancestor.get("catalog_validation_digest") or
                    ancestor.get("aggregate_parent_namespace") != (layers[index - 1] if index else None) or
                    ancestor.get("aggregate_layer_count") != index + 1 or
                    ancestor.get("selected_unit_receipt_id") != ancestor.get("receipt_id")):
                raise RuntimeError("aggregate ancestor chain differs from sealed manifest")
        physical = any(output["store"] in {"postgres", "neo4j"}
                       for row in receipts.values() for output in row["output_inventory"])
        rag = None
        try:
            if physical:
                rag = await self.rag_factory(self.repository,
                    {"namespace": layers[-1], "aggregate_layers": layers},
                    working_dir=self.working_dir, read_only=True)
            actual, errors = await self._batch_inventory(rag, receipts)
            stamp = datetime.now(timezone.utc).isoformat()
            return [observation for namespace in layers
                    for observation in qualify_observations(receipts[namespace],
                        actual[namespace], errors[namespace], stamp)]
        finally:
            if rag is not None:
                await rag.finalize_storages()

    async def _batch_inventory(self, rag, receipts):
        from papervault.knowledge.store.catalog_storage import LayeredGraphView

        actual = {namespace: {} for namespace in receipts}
        errors = {namespace: {} for namespace in receipts}
        try:
            all_tombstones = self.repository.list_tombstones(list(receipts))
            tombstones_error = None
        except Exception as exc:
            all_tombstones, tombstones_error = [], type(exc).__name__
        metadata_revisions = {row["consumed_input_revision"] for row in receipts.values()
                              if any(output["store"] == "catalog_inputs" for output in row["output_inventory"])}
        try:
            input_manifests = {row["consumed_input_revision"]: row["manifest"]
                               for row in self.repository.get_inputs(metadata_revisions)} if metadata_revisions else {}
        except Exception:
            input_manifests = {}
        groups = {}
        for namespace, receipt in receipts.items():
            for output in receipt["output_inventory"]:
                if output["store"] == "postgres":
                    table = output["verification"]["evidence"].get("backend_table")
                    if not table or not table.replace("_", "").isalnum():
                        errors[namespace][output["role"]] = "backend table identity unavailable"
                    else:
                        groups.setdefault((output["role"], table), []).append(namespace)
        for (role, table), namespaces in groups.items():
            try:
                store = getattr(rag, role)
                db = getattr(store.db, "raw", store.db)
                rows = await db.query(f'SELECT * FROM "{table}" WHERE workspace=ANY($1::varchar[]) '
                                      'ORDER BY workspace,id', [namespaces], multirows=True)
                if rows is None:
                    raise RuntimeError("backend returned no inventory response")
                grouped = {namespace: [] for namespace in namespaces}
                for row in rows:
                    grouped[row["workspace"]].append(dict(row))
                stamp = datetime.now(timezone.utc).isoformat()
                for namespace, values in grouped.items():
                    actual[namespace][role] = _inventory(role, "postgres", namespace, values,
                                                        [str(row["id"]) for row in values], stamp)
            except Exception as exc:
                for namespace in namespaces:
                    errors[namespace][role] = type(exc).__name__
        graph_namespaces = [namespace for namespace, receipt in receipts.items()
                            if any(output["store"] == "neo4j" for output in receipt["output_inventory"])]
        if graph_namespaces:
            try:
                view = LayeredGraphView(rag.chunk_entity_relation_graph, graph_namespaces, [])
                node_rows = await view._records(
                    "MATCH (n:PaperVaultCatalogEntity) WHERE n._pv_namespace IN $namespaces "
                    "RETURN properties(n) AS properties")
                edge_rows = await view._records(
                    "MATCH (a:PaperVaultCatalogEntity)-[r:DIRECTED]->(b:PaperVaultCatalogEntity) "
                    "WHERE r._pv_namespace IN $namespaces AND a._pv_namespace=r._pv_namespace "
                    "AND b._pv_namespace=r._pv_namespace "
                    "RETURN a.entity_id AS source,b.entity_id AS target,properties(r) AS properties")
                nodes = {namespace: [] for namespace in graph_namespaces}
                edges = {namespace: [] for namespace in graph_namespaces}
                for row in node_rows:
                    value = dict(row["properties"])
                    nodes[value["_pv_namespace"]].append({**value, "id": value["entity_id"]})
                for row in edge_rows:
                    value = dict(row["properties"])
                    edges[value["_pv_namespace"]].append({**value, "source": row["source"], "target": row["target"]})
                stamp = datetime.now(timezone.utc).isoformat()
                for namespace in graph_namespaces:
                    actual[namespace]["graph"] = _graph_inventory(namespace, nodes[namespace], edges[namespace], stamp)
            except Exception as exc:
                for namespace in graph_namespaces:
                    errors[namespace]["graph"] = type(exc).__name__
        for namespace, receipt in receipts.items():
            for output in receipt["output_inventory"]:
                role, store = output["role"], output["store"]
                if store not in {"catalog", "catalog_inputs"}:
                    continue
                try:
                    stamp = datetime.now(timezone.utc).isoformat()
                    if store == "catalog" and role == "tombstones":
                        if tombstones_error:
                            raise RuntimeError(tombstones_error)
                        rows = _tombstones(row for row in all_tombstones if row["namespace"] == namespace)
                        measured = _inventory(role, store, namespace, rows,
                                              [row["role"] + ":" + row["output_id"] for row in rows], stamp)
                    elif store == "catalog_inputs" and self.input_store is not None:
                        manifest = input_manifests[receipt["consumed_input_revision"]]
                        artifact = next(row for row in manifest["artifacts"] if row["artifact_id"] == output["output_id"])
                        data = self.input_store.read(artifact)
                        measured = {"sha256": hashlib.sha256(data).hexdigest(), "byte_length": len(data),
                                    "verified_at": stamp}
                    else:
                        raise RuntimeError("output collector unavailable")
                    actual[namespace][role] = measured
                except Exception as exc:
                    errors[namespace][role] = type(exc).__name__
        return actual, errors


async def verify_outputs(rag, claim, doc_id, text):
    """Read all persistent stores back; processed status alone proves nothing."""
    from papervault.store.knowledge import KnownExecutionFailure

    status = await rag.doc_status.get_by_id(doc_id)
    if status and status.get("status") == "failed" and status.get("track_id") == claim["submission_id"]:
        raise KnownExecutionFailure("LightRAG settled the admitted document in failed state")
    if not status or str(status.get("status")) != "processed":
        raise RuntimeError("document output is not processed")
    if status.get("track_id") != claim["submission_id"]:
        raise RuntimeError("document output belongs to another submission")
    document = await rag.full_docs.get_by_id(doc_id)
    expected = sanitize_text_for_encoding(text)
    if not document or document.get("content") != expected:
        raise KnownExecutionFailure("full document output content failed readback verification")
    chunk_ids = status.get("chunks_list") or []
    if not chunk_ids:
        raise RuntimeError("processed document has no verified chunk outputs")
    chunks = await rag.text_chunks.get_by_ids(chunk_ids)
    if len(chunks) != len(chunk_ids) or any(not row for row in chunks):
        raise KnownExecutionFailure("document chunk output readback is incomplete")
    vectors = await rag.chunks_vdb.get_vectors_by_ids(chunk_ids)
    if set(vectors) != set(chunk_ids) or any(not value for value in vectors.values()):
        raise RuntimeError("chunk vector output readback is incomplete")
    inventory, observed_at = await verify_delta_outputs(rag, claim)
    doc_output = next(item for item in inventory if item["role"] == "full_docs")
    content = expected.encode("utf-8")
    doc_output["verification"]["evidence"].update(
        content_sha256=hashlib.sha256(content).hexdigest(), content_byte_length=len(content))
    return inventory, {"doc_id": doc_id, "doc_status": "processed", "observed_at": observed_at,
                       "chunk_count": len(chunk_ids), "provider": "unknown",
                       "model": "unknown", "version": "unknown"}


async def verify_delta_outputs(rag, claim):
    """Verify private physical rows plus effective vectors for rebuilt graph rows."""
    observed_at = datetime.now(timezone.utc).isoformat()
    inventory = []
    for name in _PG_STORES:
        store = getattr(rag, name)
        if store.workspace != claim["namespace"]:
            raise RuntimeError("backend output escaped admitted namespace")
        table = getattr(store, "table_name", None) or namespace_to_table_name(store.namespace)
        if not table or not table.replace("_", "").isalnum():
            raise RuntimeError("invalid backend output table")
        raw_db = getattr(store.db, "raw", store.db)
        rows = await raw_db.query(f'SELECT * FROM "{table.lower()}" WHERE workspace=$1 ORDER BY id',
                                    [claim["namespace"]], multirows=True)
        if rows is None:
            raise RuntimeError(f"{name} output inventory unavailable")
        rows = [dict(row) for row in rows]
        inventory.append(_inventory(name, "postgres", claim["namespace"], rows,
                                    [str(row["id"]) for row in rows], observed_at))
        inventory[-1]["verification"]["evidence"]["backend_table"] = table.lower()
    graph = rag.chunk_entity_relation_graph
    if graph.workspace != claim["namespace"]:
        raise RuntimeError("graph output escaped admitted namespace")
    raw_reads = getattr(graph, "_catalog_raw_reads", {})
    nodes = await raw_reads.get("get_all_nodes", graph.get_all_nodes)()
    edges = await raw_reads.get("get_all_edges", graph.get_all_edges)()
    entry = _graph_inventory(claim["namespace"], nodes, edges, observed_at)
    inventory.append(entry)
    nodes, edges = _canonical_graph(nodes, edges)
    # Extraction can legitimately yield no entities, but nonempty graphs must
    # have all three kinds of vector materialization inventoried explicitly.
    expected_entities = {compute_mdhash_id(str(node.get("entity_id", node.get("id"))), prefix="ent-")
                         for node in nodes}
    entity_vectors = await rag.entities_vdb.get_vectors_by_ids(list(expected_entities))
    expected_relations = {value for edge in edges for value in (
        compute_mdhash_id(edge["source"] + edge["target"], prefix="rel-"),
        compute_mdhash_id(edge["target"] + edge["source"], prefix="rel-"))}
    relation_vectors = await rag.relationships_vdb.get_vectors_by_ids(list(expected_relations))
    actual_relations = {key for key, value in relation_vectors.items() if value}
    if (not expected_entities <= {key for key, value in entity_vectors.items() if value} or
            any(not ({compute_mdhash_id(edge["source"] + edge["target"], prefix="rel-"),
                      compute_mdhash_id(edge["target"] + edge["source"], prefix="rel-")}
                     & actual_relations) for edge in edges)):
        raise RuntimeError("graph vector output inventory incomplete")
    return inventory, observed_at


def _canonical_graph(nodes, edges):
    nodes = sorted(nodes, key=lambda row: str(row.get("entity_id", row.get("id"))))
    # Neo4j's undirected all-edge query returns both orientations. Bind each
    # logical relationship once and require its corresponding stored vector.
    unique_edges = {}
    for edge in edges:
        source, target = sorted((edge["source"], edge["target"]))
        unique_edges[(source, target)] = {**edge, "source": source, "target": target}
    edges = [unique_edges[key] for key in sorted(unique_edges)]
    return nodes, edges


def _graph_inventory(namespace, nodes, edges, observed_at):
    nodes, edges = _canonical_graph(nodes, edges)
    entry = _inventory("graph", "neo4j", namespace, {"nodes": nodes, "edges": edges},
                       [str(row.get("entity_id", row.get("id", ""))) for row in nodes],
                       observed_at)
    entry["verification"]["evidence"].update(node_count=len(nodes), edge_count=len(edges))
    if not nodes and not edges:
        entry["verification"]["state"] = "verified_empty"
    return entry


def qualify_observations(receipt, actual, errors, observed_at):
    result = []
    for output in receipt["output_inventory"]:
        role = output["role"]
        measured = actual.get(role)
        if role in errors or measured is None:
            state, detail = "unknown", {"reason": errors.get(role, "collector unavailable")}
        else:
            detail = {key: measured[key] for key in ("sha256", "byte_length")}
            valid = all(measured[key] == output[key] for key in ("sha256", "byte_length"))
            state = "present" if valid else "missing"
            detail["reason"] = "verified_readback" if valid else "checksum_or_length_mismatch"
        result.append({**{key: receipt[key] for key in _IDENTITY},
                       **{key: output[key] for key in ("role", "store", "namespace", "output_id")},
                       "receipt_id": receipt["receipt_id"], "state": state, "detail": detail,
                       "observed_at": measured.get("verified_at", observed_at) if measured else observed_at})
    return result


def _tombstones(rows):
    return sorted(({"role": row["role"], "output_id": row["output_id"]} for row in rows),
                  key=lambda row: (row["role"], row["output_id"]))


def _inventory(role, store, namespace, value, backend_ids, verified_at):
    encoded = _bytes(value)
    digest = hashlib.sha256(encoded).hexdigest()
    return {"role": role, "output_id": f"{namespace}:{role}:{digest}", "store": store,
            "namespace": namespace, "backend_ids": backend_ids, "sha256": digest,
            "byte_length": len(encoded), "verified_at": verified_at,
            "verification": {"state": "present" if backend_ids else "verified_empty",
                             "method": "namespace_backend_readback_sha256_v1",
                             "evidence": {"record_count": len(backend_ids)}}}
