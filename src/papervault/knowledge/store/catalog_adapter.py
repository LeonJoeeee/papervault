"""Recoverable, isolated LightRAG deliveries and backend-verified receipts."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from functools import wraps
from uuid import uuid4

from lightrag.kg.postgres_impl import namespace_to_table_name
from lightrag.utils import (
    compute_mdhash_id, make_relation_chunk_key, parse_relation_chunk_key, sanitize_text_for_encoding,
)
from papervault.store.knowledge import RetirementFailure

_IDENTITY = ("job_id", "attempt_id", "submission_id", "authority_epoch", "fencing_token",
             "unit_id", "consumed_input_revision", "recipe_digest", "generation_id", "namespace")
_COW_IDENTITY = ("aggregate_base_manifest_revision", "base_aggregate_layers",
                 "base_selected_unit_receipts", "retired_receipt_ids")
_PG_STORES = ("full_docs", "text_chunks", "full_entities", "full_relations",
              "entity_chunks", "relation_chunks", "llm_response_cache", "doc_status",
              "chunks_vdb", "entities_vdb", "relationships_vdb")
_recipe = ContextVar("catalog_execution_recipe", default=None)
_retirement = ContextVar("catalog_retirement_rebuild", default=None)


def current_execution_recipe():
    return _recipe.get()


class _RetirementRebuild:
    """Validate pinned deletion inputs without changing LightRAG's reducers."""
    def __init__(self, original):
        self.original = original

    async def __call__(self, *args, **kwargs):
        bound = _retirement.get()
        if bound is None or kwargs.get("knowledge_graph_inst") is not bound["graph"]:
            return await self.original(*args, **kwargs)
        try:
            await self._validate(kwargs)
        except Exception as exc:
            bound["error"] = exc
            raise
        config = dict(kwargs["global_config"])
        roles = dict(config.get("role_llm_funcs", {}))
        callback = roles.get("extract")
        if callback is not None:
            async def observed(*callback_args, **callback_kwargs):
                from papervault.knowledge.store.build_breaker import BuildPausedError

                try:
                    return await callback(*callback_args, **callback_kwargs)
                except Exception as exc:
                    # The pinned LLM queue/cache can wrap the original pause.
                    # Retain its typed root before the rebuild worker swallows it.
                    cause, seen = exc, set()
                    while cause is not None and id(cause) not in seen:
                        seen.add(id(cause))
                        if isinstance(cause, BuildPausedError):
                            bound["pause"] = cause
                            break
                        cause = cause.__cause__
                    raise

            roles["extract"] = observed
            config["role_llm_funcs"] = roles
        bound["config"] = config
        await self.original(*args, **{**kwargs, "global_config": config})
        if bound["pause"] is not None:
            # Also stop deletion before it removes the document's retry state.
            raise bound["pause"]
        if bound["error"] is not None:
            raise bound["error"]

    @staticmethod
    async def _validate(kwargs):
        import lightrag.operate as operate

        entities, relations = kwargs["entities_to_rebuild"], kwargs["relationships_to_rebuild"]
        required = {chunk for targets in (entities, relations) for chunks in targets.values() for chunk in chunks}
        if not required:
            return
        cached = await operate._get_cached_extraction_results(kwargs["llm_response_cache"], required,
            text_chunks_storage=kwargs["text_chunks_storage"])
        if set(cached) != required:
            raise RuntimeError("retirement rebuild requires cached extraction for every remaining chunk; "
                               "isolated rebuild required")
        parsed = {}
        for chunk, results in cached.items():
            parsed[chunk] = [await operate._rebuild_from_extraction_result(
                text_chunks_storage=kwargs["text_chunks_storage"], extraction_result=result,
                chunk_id=chunk, timestamp=stamp) for result, stamp in results]
        config = kwargs["global_config"]
        for kind, targets, limit in ((0, entities, "max_source_ids_per_entity"),
                                      (1, relations, "max_source_ids_per_relation")):
            for target, chunks in targets.items():
                limited = operate.apply_source_ids_limit(chunks, config[limit],
                    config.get("source_ids_limit_method") or "KEEP", identifier=str(target))
                keys = [target] if kind == 0 else [target, tuple(reversed(target))]
                descriptions = [row.get("description") for chunk in limited
                    for extraction in parsed[chunk] for key in keys for row in extraction[kind].get(key, [])]
                # Entity-from-relationship and current-description fallbacks can
                # retain retired evidence. They need an isolated rebuild instead.
                if not any(isinstance(value, str) and value.strip() for value in descriptions):
                    raise RuntimeError(f"retirement rebuild lacks attributable description for {target}; "
                                       "isolated rebuild required")


class _RetirementSummary:
    """Reject empty original summaries, including cache hits and map groups."""
    def __init__(self, original, config_position, *, tuple_result=False):
        self.original, self.config_position, self.tuple_result = original, config_position, tuple_result

    async def __call__(self, *args, **kwargs):
        from lightrag.utils import remove_think_tags

        bound = _retirement.get()
        config = kwargs.get("global_config", args[self.config_position] if len(args) > self.config_position else None)
        result = await self.original(*args, **kwargs)
        if bound is not None and config is bound.get("config"):
            value = result[0] if self.tuple_result else result
            if not isinstance(value, str) or not sanitize_text_for_encoding(remove_think_tags(value)).strip():
                error = RuntimeError("retirement rebuild produced an empty summary; isolated rebuild required")
                bound["error"] = error
                raise error
        return result


@contextmanager
def _retirement_rebuild(rag):
    import lightrag.lightrag as dependency
    import lightrag.operate as operate

    if not isinstance(dependency.rebuild_knowledge_from_chunks, _RetirementRebuild):
        dependency.rebuild_knowledge_from_chunks = _RetirementRebuild(dependency.rebuild_knowledge_from_chunks)
    for name, position, tuple_result in (("_summarize_descriptions", 3, False),
                                         ("_handle_entity_relation_summary", 4, True)):
        original = getattr(operate, name)
        if not isinstance(original, _RetirementSummary):
            setattr(operate, name, _RetirementSummary(original, position, tuple_result=tuple_result))
    state = {"graph": rag.chunk_entity_relation_graph, "pause": None, "error": None}
    token = _retirement.set(state)
    try:
        yield state
    finally:
        _retirement.reset(token)


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
        except Exception as exc:
            self._replacement_failure(claim, exc)
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
        from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
        from papervault.store.knowledge import ClaimRejected, OperationConflict

        if claim["retired_receipt_ids"]:
            # Commit before the first external mutation. A crash may leave any
            # subset of graph/tracker/vector writes durable. A new invocation
            # must never ask the deletion reducer to resume that partial view.
            stable = {key: claim[key] for key in _IDENTITY + _COW_IDENTITY + ("lease_id",)}
            try:
                self.repository.command("begin_retirement", "begin-retirement:" + claim["submission_id"],
                                        {"claim": stable, "delivery_id": uuid4().hex})
            except (OperationConflict, ClaimRejected) as exc:
                retired_id = claim["retired_receipt_ids"][0]
                retired = self.repository.get_receipt(retired_id) or {}
                raise RetirementFailure(retired_id, retired.get("provider_observations", {}).get("doc_id"),
                    "unverified", evidence={"recovery_error":
                        "retirement attempt cannot be resumed; isolated rebuild required"}) from exc

        for retired_id in claim["retired_receipt_ids"]:
            retired = self.repository.get_receipt(retired_id)
            old_doc = (retired or {}).get("provider_observations", {}).get("doc_id")
            if not old_doc:
                if retired and retired.get("achieved_tier") == "metadata":
                    continue
                raise RetirementFailure(retired_id, old_doc, "unverified", evidence={
                    "index_validation_error": "retired contribution lacks verified document identity; isolated rebuild required"})
            try:
                graph_output = next(output for output in retired.get("output_inventory", [])
                                    if output["role"] == "graph")
                contribution = graph_output["verification"]["evidence"]["document_contribution"]
                if contribution["doc_id"] != old_doc:
                    raise RuntimeError("document index contribution identity differs from receipt")
                chunks = contribution["chunk_ids"]
                if (not isinstance(chunks, list) or not chunks
                        or any(not isinstance(chunk, str) or not chunk for chunk in chunks)):
                    raise RuntimeError("document index contribution has invalid chunk identities")
                status = await rag.doc_status.get_by_id(old_doc)
                if not status or set(status.get("chunks_list") or []) != set(chunks):
                    raise RuntimeError("document status index does not cover recorded contribution chunks")
                # Discovery is part of verification, not evidence of absence.
                # The upstream deletion returns success if both indexes are
                # missing, without ever invoking its graph rebuild reducer.
                _verify_document_indexes(contribution,
                    await rag.full_entities.get_by_id(old_doc),
                    await rag.full_relations.get_by_id(old_doc))
            except Exception as exc:
                raise RetirementFailure(retired_id, old_doc, "unverified", evidence={
                    "index_validation_error": "document index contribution unverified; isolated rebuild required: "
                                              + str(exc)}) from exc
            with _retirement_rebuild(rag) as rebuild:
                result = await rag.adelete_by_doc_id(old_doc)
            if rebuild["pause"] is not None:
                raise rebuild["pause"]
            messages = [result.message]
            if rebuild["error"] is not None:
                detail = str(rebuild["error"])
                raise RetirementFailure(retired_id, old_doc, result.status, messages + [detail],
                                        {"rebuild_validation_error": detail})
            if result.status != "success":
                # Missing document status does not prove the inherited graph,
                # vectors or shared summaries no longer contain this document.
                raise RetirementFailure(retired_id, old_doc, result.status, messages)
            try:
                if rag.workspace != claim["namespace"]:
                    raise RuntimeError("deletion workspace differs from admitted namespace")
                status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
                # Never retain this lock across the deletion: its reducers take
                # the same lock when appending their operation evidence.
                async with get_namespace_lock("pipeline_status", workspace=rag.workspace):
                    history = list(status["history_messages"])
                if not history or history[0] != f"Starting deletion for document: {old_doc}":
                    raise RuntimeError("deletion history is not attributed to retired document")
                if any(message.startswith("Starting deletion for document:") for message in history[1:]):
                    raise RuntimeError("deletion history contains another operation")
                messages.extend(history)
            except Exception as exc:
                raise RetirementFailure(retired_id, old_doc, result.status, messages,
                                        {"history_error": str(exc)}) from exc
            # LightRAG 1.5.4 catches individual rebuild/parse failures and can
            # return success with a clean final result message. Its operation
            # history is the reporting route for failed counts and no-cache exits.
            if any(re.search(r"^Failed to\b|\bFailed:\s*\d+\s+entities\b|"
                             r"^No cached extraction results found, cannot rebuild", message, re.IGNORECASE)
                   for message in messages):
                raise RetirementFailure(retired_id, old_doc, result.status, messages)
            try:
                for storage in rag._index_storages():
                    await storage.index_done_callback()
                await _verify_retired_vectors(rag, contribution)
            except Exception as exc:
                raise RetirementFailure(retired_id, old_doc, "unverified", messages, {
                    "vector_validation_error": str(exc) + "; isolated rebuild required"}) from exc

    def _replacement_failure(self, claim, exc):
        from papervault.knowledge.store.build_breaker import BuildPausedError

        if claim["retired_receipt_ids"] and not isinstance(exc, (BuildPausedError, RetirementFailure)):
            # A lost seal response is an ambiguous delivery, not a failed
            # retirement. Never discard accepted work or its completion path.
            try:
                submission = self.repository.get_submission(claim["submission_id"])
            except Exception:
                raise exc
            if submission and (submission.get("receipt") or submission.get("receipt_id")):
                raise exc
            raise RetirementFailure(claim["retired_receipt_ids"][0], None, "unverified", evidence={
                "replacement_validation_error": str(exc) + "; isolated rebuild required"}) from exc
        raise exc

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
        except Exception as exc:
            self._replacement_failure(claim, exc)
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
                       document_index_verification={"version": 1, "retired_receipt_ids": claim["retired_receipt_ids"]},
                       origin_lineage={})
        if claim["retired_receipt_ids"]:
            graph = next(output for output in inventory if output["role"] == "graph")
            receipt["retirement_verification"] = {"version": 1,
                "retired_receipt_ids": claim["retired_receipt_ids"], "graph_sha256": graph["sha256"]}
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
    inventory, observed_at = await verify_delta_outputs(rag, claim, document=(doc_id, chunk_ids))
    doc_output = next(item for item in inventory if item["role"] == "full_docs")
    content = expected.encode("utf-8")
    doc_output["verification"]["evidence"].update(
        content_sha256=hashlib.sha256(content).hexdigest(), content_byte_length=len(content))
    return inventory, {"doc_id": doc_id, "doc_status": "processed", "observed_at": observed_at,
                       "chunk_count": len(chunk_ids), "provider": "unknown",
                       "model": "unknown", "version": "unknown"}


async def verify_delta_outputs(rag, claim, *, document=None):
    """Verify private physical rows plus effective vectors for rebuilt graph rows."""
    observed_at = datetime.now(timezone.utc).isoformat()
    inventory = []
    index_rows = {}
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
        if name in {"full_entities", "full_relations", "entity_chunks", "relation_chunks"}:
            index_rows[name] = {str(row["id"]): row for row in rows}
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
    if document is not None:
        doc_id, chunk_ids = document
        contribution = await _document_graph_contribution(rag, doc_id, chunk_ids, nodes, edges, index_rows)
        _verify_document_indexes(contribution, index_rows["full_entities"].get(doc_id),
                                 index_rows["full_relations"].get(doc_id))
        entry["verification"]["evidence"]["document_contribution"] = contribution
    await _verify_graph_vectors(rag, nodes, edges)
    return inventory, observed_at


async def _verify_graph_vectors(rag, nodes, edges):
    """Bind effective vector payloads to graph content, even when inherited.

    The pinned reducers embed these canonical strings. Physical PostgreSQL rows
    retain endpoint identities and source chunks alongside the vector; presence
    of a vector with the right key alone cannot prove a rebuild completed.
    """
    from lightrag.operate import _truncate_vdb_content
    from lightrag.utils import GRAPH_FIELD_SEP

    config = rag._build_global_config()

    async def verify(store, key, row, content, identity, label):
        vector = await store.get_by_id(key)
        embedding = await store.get_vectors_by_ids([key])
        # Pinned ingestion reducers can truncate the string, while entity
        # source-appends and relationship rebuilds write it in full.
        expected_content = {content, _truncate_vdb_content(content, config, label)}
        sources = set(filter(None, str(row.get("source_id") or "").split(GRAPH_FIELD_SEP)))
        if (not vector or not embedding.get(key) or vector.get("id") != key
                or vector.get("content") not in expected_content
                or any(vector.get(field) != value for field, value in identity.items())
                or set(_index_list(vector.get("chunk_ids", []))) != sources):
            raise RuntimeError(f"graph vector content/identity mismatch or absent: {label}")

    for node in nodes:
        name = str(node.get("entity_id", node.get("id")))
        if not isinstance(node.get("description"), str):
            raise RuntimeError(f"graph vector description unavailable: {name}")
        await verify(rag.entities_vdb, compute_mdhash_id(name, prefix="ent-"), node,
                     f"{name}\n{node['description']}", {"entity_name": name}, f"entity:{name}")
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if not all(isinstance(edge.get(field), str) for field in ("description", "keywords")):
            raise RuntimeError(f"graph vector description/keywords unavailable: {source}-{target}")
        found = False
        for src, tgt in {(source, target), (target, source)}:
            key = compute_mdhash_id(src + tgt, prefix="rel-")
            if await rag.relationships_vdb.get_by_id(key) is None:
                continue
            found = True
            await verify(rag.relationships_vdb, key, edge,
                         f"{edge['keywords']}\t{src}\n{tgt}\n{edge['description']}",
                         {"source_id": src, "target_id": tgt}, f"relationship:{src}-{tgt}")
        if not found:
            raise RuntimeError(f"graph vector output absent: relationship:{source}-{target}")


async def _verify_retired_vectors(rag, contribution):
    """Check all affected identities, including deletions and untouched rows."""
    graph = rag.chunk_entity_relation_graph
    nodes, edges = [], []
    for name in contribution["entity_ids"]:
        node = await graph.get_node(name)
        if node is not None:
            nodes.append({**node, "entity_id": name})
        elif await rag.entities_vdb.get_by_id(compute_mdhash_id(name, prefix="ent-")) is not None:
            raise RuntimeError(f"retired graph vector survives deleted entity: {name}")
    for source, target in contribution["relation_pairs"]:
        edge = await graph.get_edge(source, target)
        if edge is not None:
            edges.append({**edge, "source": source, "target": target})
        else:
            for src, tgt in {(source, target), (target, source)}:
                if await rag.relationships_vdb.get_by_id(compute_mdhash_id(src + tgt, prefix="rel-")) is not None:
                    raise RuntimeError(f"retired graph vector survives deleted relationship: {src}-{tgt}")
    await _verify_graph_vectors(rag, nodes, edges)


async def _document_graph_contribution(rag, doc_id, chunk_ids, nodes, edges, private_rows):
    """Derive ownership independently of the indexes used by deletion.

    A replacement delta can also contain rows rebuilt while retiring the old
    version. Only this document's chunks establish its contribution. Trackers
    retain uncapped source IDs when the graph's display source list is limited.
    """
    from lightrag.utils import GRAPH_FIELD_SEP

    chunks = set(chunk_ids)
    entities, relations = set(), set()
    for rows, store, key, selected in (
        (nodes, rag.entity_chunks, lambda row: str(row.get("entity_id", row.get("id"))), entities),
        (edges, rag.relation_chunks,
         lambda row: make_relation_chunk_key(row["source"], row["target"]), relations),
    ):
        for row in rows:
            sources = set(filter(None, str(row.get("source_id") or "").split(GRAPH_FIELD_SEP)))
            tracker = await store.get_by_id(key(row))
            if tracker:
                sources.update(_index_list(tracker.get("chunk_ids", [])))
            if not sources:
                raise RuntimeError("document index contribution has no attributable graph sources")
            if sources & chunks:
                selected.add(key(row) if selected is entities else (row["source"], row["target"]))
    # KEEP source limits can append uncapped memberships and return before any
    # private graph write. Those contributions can become visible in a later
    # rebuild, so a zero-row graph delta cannot prove an empty contribution.
    for role, selected in (("entity_chunks", entities), ("relation_chunks", relations)):
        for key, row in private_rows[role].items():
            if not chunks.intersection(_index_list(row.get("chunk_ids", []))):
                continue
            identity = key if role == "entity_chunks" else parse_relation_chunk_key(key)
            graph = rag.chunk_entity_relation_graph
            material = await graph.get_node(identity) if role == "entity_chunks" else await graph.get_edge(*identity)
            if not material:
                raise RuntimeError("document index membership has no verified graph material")
            selected.add(identity)
    return {"doc_id": doc_id, "chunk_ids": sorted(chunks),
            "entity_ids": sorted(entities), "relation_pairs": [list(pair) for pair in sorted(relations)]}


def _index_list(values):
    if isinstance(values, str):
        values = json.loads(values)
    if not isinstance(values, list):
        raise RuntimeError("document index membership is malformed")
    return values


def _verify_document_indexes(contribution, entities, relations):
    """Require target discovery to cover immutable, independently derived IDs."""
    for role, row, field, expected_field in (
        ("full_entities", entities, "entity_names", "entity_ids"),
        ("full_relations", relations, "relation_pairs", "relation_pairs"),
    ):
        expected = contribution[expected_field]
        values = row.get(field) if row else []
        values = _index_list(values)
        if not isinstance(values, list) or not isinstance(expected, list):
            raise RuntimeError(f"{role} document index is malformed")
        def identities(items):
            if role == "full_entities":
                if any(not isinstance(item, str) or not item for item in items):
                    raise RuntimeError(f"{role} document index is malformed")
                return set(items)
            if any(not isinstance(item, (tuple, list)) or len(item) != 2
                   or any(not isinstance(part, str) or not part for part in item) for item in items):
                raise RuntimeError(f"{role} document index is malformed")
            return {tuple(sorted(item)) for item in items}
        if not identities(expected) <= identities(values):
            raise RuntimeError(f"{role} document index does not cover recorded graph contribution")


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
