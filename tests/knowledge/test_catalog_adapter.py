"""Submission recovery and immutable receipt verification."""
from contextlib import asynccontextmanager
from copy import deepcopy

import pytest
from papervault.store.knowledge import canonical_recipe

RECIPE = {"schema_version": 1, "name": "fixture",
          "configured_models": {"build": "admitted-model", "keyword": "admitted-keyword"},
          "prompts": {"summarize_entity_descriptions": "frozen admitted prompt"}}


@pytest.fixture(autouse=True)
def effective_recipe(monkeypatch):
    from papervault.knowledge.scheduler import catalog_intake
    monkeypatch.setattr(catalog_intake, "effective_distill_recipe", lambda: deepcopy(RECIPE))


CLAIM = {"job_id": "job", "attempt_id": "attempt", "submission_id": "submission",
         "authority_epoch": 2, "fencing_token": 3, "unit_id": "paper:one",
         "consumed_input_revision": "input", "recipe_digest": "recipe",
         "generation_id": "gen", "namespace": "pv_private", "lease_id": "lease"}
CLAIM.update(aggregate_base_manifest_revision=0, base_aggregate_layers=[],
             base_selected_unit_receipts={}, retired_receipt_ids=[])
CLAIM["recipe_digest"] = canonical_recipe(RECIPE)[1]


class Repository:
    def __init__(self):
        self.receipt = None
        self.accepted = False
        self.sealed = False

    def get_submission(self, submission):
        return {"receipt": self.receipt} if self.accepted else None

    def get_receipt(self, receipt_id):
        return {"provider_observations": {"doc_id": "old-doc"}}

    def list_tombstones(self, namespaces, role=None):
        return []

    def get_input(self, revision):
        return {"manifest": getattr(self, "manifest", {"artifacts": []})}

    def get_inputs(self, revisions):
        return [{"consumed_input_revision": revision, **self.get_input(revision)} for revision in revisions]

    def get_recipe(self, recipe_digest):
        return deepcopy(RECIPE)

    def command(self, action, operation_id, payload):
        if action == "accept_submission":
            self.accepted = True
            return {"accepted": True}
        assert action == "seal_namespace"
        self.receipt = payload["receipt"]
        self.sealed = True
        return self.receipt

    @asynccontextmanager
    async def namespace_access(self, claim, **kwargs):
        if self.sealed:
            raise RuntimeError("sealed")
        yield


def with_evidence(repo, claim, tmp_path, text="actual content"):
    import hashlib

    path = tmp_path / "captured"
    data = text.encode()
    path.write_bytes(data)
    evidence = {"sha256": hashlib.sha256(data).hexdigest(), "byte_length": len(data), "path": str(path)}
    repo.manifest = {"artifacts": [{"artifact_id": "artifact", **{key: evidence[key] for key in ("sha256", "byte_length")}}]}
    return {**claim, "input_evidence": [evidence]}


@pytest.mark.asyncio
async def test_recoverable_receipt_prevents_ambiguous_submission_redelivery(tmp_path):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    repo = Repository()
    repo.accepted = True
    repo.receipt = {**CLAIM, "receipt_id": "durable", "sealed": True, "durable": True}

    async def should_not_build(*args, **kwargs):
        raise AssertionError("receipt lookup must precede external delivery")

    adapter = CatalogLightRAGAdapter(repo, working_dir=tmp_path, rag_factory=should_not_build)
    assert await adapter.submit(CLAIM, "content", "paper/one") == repo.receipt


@pytest.mark.asyncio
async def test_recovered_receipt_must_match_every_admitted_identity(tmp_path):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    repo = Repository()
    repo.accepted = True
    repo.receipt = {**CLAIM, "receipt_id": "durable", "sealed": True, "durable": True,
                    "recipe_digest": "wrong"}
    adapter = CatalogLightRAGAdapter(repo, working_dir=tmp_path)
    with pytest.raises(RuntimeError, match="receipt.*identity"):
        await adapter.submit(CLAIM, "content", "paper/one")


@pytest.mark.asyncio
async def test_receipt_cannot_be_sealed_from_processed_status_without_output_readback(tmp_path):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    class Store:
        async def get_by_id(self, key):
            return {"status": "processed", "track_id": "submission", "chunks_list": []}

    class Rag:
        doc_status = Store()
        full_docs = Store()

        async def ainsert(self, *args, **kwargs):
            return "submission"

        def _index_storages(self):
            return []

        async def finalize_storages(self):
            pass

    async def factory(*args, **kwargs):
        return Rag()

    repo = Repository()
    adapter = CatalogLightRAGAdapter(repo, working_dir=tmp_path, rag_factory=factory)
    with pytest.raises(RuntimeError, match="content|output"):
        await adapter.submit(with_evidence(repo, CLAIM, tmp_path), "actual content", "paper/one")
    assert repo.receipt is None
    assert not repo.sealed


def verified_rag(*, wrong_entity_vector=False):
    from lightrag.utils import compute_mdhash_id
    from types import SimpleNamespace

    class DB:
        async def query(self, sql, params, **kwargs):
            if '"entity_vectors"' in sql:
                return [{"id": "other" if wrong_entity_vector else compute_mdhash_id("Earth", prefix="ent-")}]
            if '"chunk_vectors"' in sql:
                return [{"id": "chunk"}]
            return []

    class Store:
        workspace = "pv_private"
        namespace = "full_docs"
        table_name = "records"
        db = DB()

        async def get_by_id(self, key):
            return {"content": "actual content", "status": "processed", "track_id": "submission",
                    "chunks_list": ["chunk"]}

        async def get_by_ids(self, keys):
            return [{"id": "chunk", "content": "actual content"}]

        async def get_vectors_by_ids(self, keys):
            return {key: [0.5, 0.2] for key in keys
                    if not wrong_entity_vector or key == "chunk"}

    class Graph:
        workspace = "pv_private"

        async def get_all_nodes(self):
            return [{"entity_id": "Earth", "id": "Earth"}]

        async def get_all_edges(self):
            return []

    stores = {name: Store() for name in (
        "full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks",
        "relation_chunks", "llm_response_cache", "doc_status", "chunks_vdb", "entities_vdb",
        "relationships_vdb")}
    stores["entities_vdb"].table_name = "entity_vectors"
    stores["chunks_vdb"].table_name = "chunk_vectors"
    return SimpleNamespace(**stores, chunk_entity_relation_graph=Graph())


@pytest.mark.asyncio
async def test_graph_vector_verification_requires_matching_ids_not_only_counts():
    from papervault.knowledge.store.catalog_adapter import verify_outputs

    with pytest.raises(RuntimeError, match="vector"):
        await verify_outputs(verified_rag(wrong_entity_vector=True), CLAIM, "doc", "actual content")


@pytest.mark.asyncio
async def test_receipt_records_exact_document_content_hash():
    from papervault.knowledge.store.catalog_adapter import verify_outputs

    inventory, _ = await verify_outputs(verified_rag(), CLAIM, "doc", "actual content")
    document = next(row for row in inventory if row["role"] == "full_docs")
    assert document["verification"]["evidence"]["content_sha256"] == (
        "97e9fe2a827cb02d4ba78d17d76cb0089feb95ed5ae8ab61c7af97d9d5e354c8")


@pytest.mark.asyncio
async def test_replacement_runs_original_deletion_rebuild_then_insert_and_binds_aggregate(tmp_path):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.store.repository import payload_hash
    from types import SimpleNamespace

    calls = []
    rag = verified_rag()

    async def delete(doc_id):
        calls.append(("delete", doc_id))
        return SimpleNamespace(status="success")

    async def insert(text, **kwargs):
        calls.append(("insert", text))

    async def finalize():
        pass

    rag.adelete_by_doc_id = delete
    rag.ainsert = insert
    rag.finalize_storages = finalize
    rag._index_storages = lambda: []

    async def factory(*args, **kwargs):
        return rag

    claim = {**CLAIM, "base_aggregate_layers": ["base"], "retired_receipt_ids": ["old"],
             "base_selected_unit_receipts": {"paper:one": "old", "paper:two": "other"}}
    repo = Repository()
    claim = with_evidence(repo, claim, tmp_path)
    adapter = CatalogLightRAGAdapter(repo, working_dir=tmp_path, rag_factory=factory)
    receipt = await adapter.submit(claim, "actual content", "paper/one")
    assert calls == [("delete", "old-doc"), ("insert", "actual content")]
    assert receipt["aggregate_layers"] == ["base", "pv_private"]
    expected = {"paper:one": receipt["receipt_id"], "paper:two": "other"}
    assert receipt["selected_unit_receipts"] == expected
    assert receipt["selected_unit_receipts_digest"] == payload_hash(expected)
    tombstones = next(row for row in receipt["output_inventory"] if row["role"] == "tombstones")
    assert tombstones["verification"]["evidence"]["record_count"] == 0


@pytest.mark.asyncio
async def test_metadata_downgrade_removes_retired_graph_using_original_reducer(tmp_path):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from types import SimpleNamespace

    input_store = ImmutableInputStore(tmp_path / "inputs")
    artifact = input_store.capture(b'{"title":"metadata only"}', role="metadata")
    manifest = {"tier": "metadata", "artifacts": [artifact]}
    claim = {**CLAIM, "retired_receipt_ids": ["old"], "base_aggregate_layers": ["base"],
             "base_selected_unit_receipts": {"paper:one": "old"}}
    rag = verified_rag()
    deleted = []

    async def delete(doc_id):
        deleted.append(doc_id)
        return SimpleNamespace(status="success")

    async def finalize():
        pass

    rag.adelete_by_doc_id = delete
    rag.finalize_storages = finalize
    rag._index_storages = lambda: []

    async def factory(*args, **kwargs):
        return rag

    repo = Repository()
    repo.manifest = manifest
    adapter = CatalogLightRAGAdapter(repo, working_dir=tmp_path, rag_factory=factory)
    receipt = await adapter.submit_metadata(claim, manifest, input_store)
    assert deleted == ["old-doc"]
    assert receipt["achieved_tier"] == "metadata"
    output = next(row for row in receipt["output_inventory"] if row["role"] == "metadata")
    assert output["sha256"] == artifact["sha256"]
    assert output["byte_length"] == artifact["byte_length"]
    assert receipt["aggregate_layers"] == ["base", "pv_private"]


def test_observation_uses_actual_hash_and_never_turns_corruption_or_outage_into_present():
    from papervault.knowledge.store.catalog_adapter import qualify_observations

    output = {"role": "graph", "store": "neo4j", "namespace": "pv_private", "output_id": "graph-id",
              "sha256": "expected", "byte_length": 10}
    receipt = {**CLAIM, "receipt_id": "receipt", "output_inventory": [output]}
    actual = {"graph": {**output, "sha256": "corrupt", "byte_length": 9}}
    result = qualify_observations(receipt, actual, {}, "2026-10-20T00:00:00+00:00")
    assert result[0]["state"] == "missing"
    assert result[0]["detail"]["sha256"] == "corrupt"
    outage = qualify_observations(receipt, {}, {"graph": "unavailable"}, "2026-10-20T00:00:00+00:00")
    assert outage[0]["state"] == "unknown"
    assert "sha256" not in outage[0]["detail"]


@pytest.mark.asyncio
async def test_lost_captured_input_during_execution_cannot_receive_sealed_receipt(tmp_path):
    import hashlib
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    data = b"actual content"
    path = tmp_path / "input"
    path.write_bytes(data)
    evidence = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "byte_length": len(data)}
    repo = Repository()
    repo.manifest = {"artifacts": [{"artifact_id": "artifact", **{key: evidence[key] for key in ("sha256", "byte_length")}}]}
    claim = {**CLAIM, "input_evidence": [evidence]}
    rag = verified_rag()

    async def insert(*args, **kwargs):
        path.unlink()

    async def finalize():
        pass

    rag.ainsert = insert
    rag.finalize_storages = finalize
    rag._index_storages = lambda: []

    async def factory(*args, **kwargs):
        return rag

    adapter = CatalogLightRAGAdapter(repo, working_dir=tmp_path, rag_factory=factory)
    with pytest.raises(RuntimeError, match="input"):
        await adapter.submit(claim, "actual content", "paper/one")
    assert repo.receipt is None


@pytest.mark.asyncio
async def test_collector_reads_real_metadata_file_and_refuses_unsealed_ancestor(tmp_path):
    from papervault.knowledge.scheduler.catalog import metadata_receipt
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    store = ImmutableInputStore(tmp_path / "inputs")
    artifact = store.capture(b'{"title":"metadata only"}', role="metadata")
    manifest = {"tier": "metadata", "artifacts": [artifact]}
    receipt = metadata_receipt(CLAIM, manifest, store)
    repo = Repository()
    repo.manifest = manifest
    projection = {key: value for key, value in receipt.items()
                  if key not in {"aggregate_layers", "base_aggregate_layers", "base_selected_unit_receipts",
                                 "selected_unit_receipts"}}
    projection.update(compact_projection=True, aggregate_parent_namespace=None, aggregate_layer_count=1,
                      selected_unit_receipt_id=receipt["receipt_id"], catalog_validation_digest="a" * 64)

    def compact_receipts(namespaces, *, compact):
        assert compact is True
        assert namespaces == receipt["aggregate_layers"]
        return [projection]

    repo.get_namespace_receipts = compact_receipts
    adapter = CatalogLightRAGAdapter(repo, working_dir=tmp_path, input_store=store)
    observed = await adapter.collect_observations(receipt)
    metadata = next(row for row in observed if row["role"] == "metadata")
    assert metadata["state"] == "present"
    assert metadata["detail"]["sha256"] == artifact["sha256"]
    projection["aggregate_parent_namespace"] = "unrelated"
    with pytest.raises(RuntimeError, match="chain"):
        await adapter.collect_observations(receipt)
    projection["aggregate_parent_namespace"] = None
    projection["sealed"] = False
    with pytest.raises(RuntimeError, match="sealed"):
        await adapter.collect_observations(receipt)


@pytest.mark.asyncio
async def test_ambiguous_delivery_retry_after_renewal_keeps_accept_payload_stable(tmp_path):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    class DeduplicatingRepository(Repository):
        payload = None

        def command(self, action, operation_id, payload):
            if self.payload is not None and self.payload != payload:
                raise ValueError("operation ID reused with different payload")
            self.payload = payload
            return super().command(action, operation_id, payload)

    async def ambiguous(*args, **kwargs):
        raise TimeoutError("ambiguous backend acceptance")

    repo = DeduplicatingRepository()
    claim = with_evidence(repo, CLAIM, tmp_path)
    adapter = CatalogLightRAGAdapter(repo, working_dir=tmp_path, rag_factory=ambiguous)
    with pytest.raises(TimeoutError):
        await adapter.submit({**claim, "revision": 1, "lease_expires_at": "earlier"}, "actual content", "one")
    with pytest.raises(TimeoutError):
        await adapter.submit({**claim, "revision": 2, "lease_expires_at": "later"}, "actual content", "one")


def test_vector_inventory_digest_uses_numeric_values_independent_of_numpy_print_options():
    import numpy as np
    from papervault.knowledge.store.catalog_adapter import _inventory

    values = [{"id": "vector", "content_vector": np.array([0.123456789, 0.987654321], dtype=np.float32)}]
    with np.printoptions(precision=2):
        first = _inventory("vectors", "postgres", "private", values, ["vector"], "stamp")
    with np.printoptions(precision=10):
        second = _inventory("vectors", "postgres", "private", values, ["vector"], "stamp")
    assert first["sha256"] == second["sha256"]
    assert first["byte_length"] == second["byte_length"]


@pytest.mark.asyncio
async def test_changed_effective_recipe_fails_before_backend_execution(monkeypatch, tmp_path):
    from papervault.knowledge.scheduler import catalog_intake
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    changed = {**RECIPE, "configured_models": {"build": "changed"}}
    monkeypatch.setattr(catalog_intake, "effective_distill_recipe", lambda: changed)

    async def forbidden(*args, **kwargs):
        raise AssertionError("mismatched recipe must never reach backend initialization")

    repo = Repository()
    claim = with_evidence(repo, CLAIM, tmp_path)
    adapter = CatalogLightRAGAdapter(repo, rag_factory=forbidden)
    with pytest.raises(RuntimeError, match="recipe"):
        await adapter.submit(claim, "actual content", "one")


@pytest.mark.asyncio
async def test_build_uses_frozen_recipe_models_and_prompts_if_globals_change_mid_attempt(monkeypatch, tmp_path):
    import lightrag.operate as operate
    from lightrag.prompt import PROMPTS
    from papervault.knowledge.store import graph
    from papervault.knowledge.store.build_breaker import BuildBreaker
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    observed = []

    async def complete(*args, **kwargs):
        observed.append((kwargs["model"], operate.PROMPTS["summarize_entity_descriptions"]))
        return "summary"

    monkeypatch.setattr(graph, "mimo_complete", complete)
    monkeypatch.setattr(graph, "get_breaker", lambda: BuildBreaker())
    rag = verified_rag()

    async def insert(*args, **kwargs):
        monkeypatch.setattr(graph, "route", lambda role: "drifted-model")
        monkeypatch.setitem(PROMPTS, "summarize_entity_descriptions", "drifted prompt")
        await graph.build_llm("input")

    async def finalize():
        pass

    rag.ainsert, rag.finalize_storages = insert, finalize
    rag._index_storages = lambda: []

    async def factory(*args, **kwargs):
        return rag

    repo = Repository()
    claim = with_evidence(repo, CLAIM, tmp_path)
    adapter = CatalogLightRAGAdapter(repo, rag_factory=factory)
    await adapter.submit(claim, "actual content", "one")
    assert observed == [("admitted-model", "frozen admitted prompt")]
    assert operate.PROMPTS["summarize_entity_descriptions"] == "drifted prompt"


@pytest.mark.asyncio
async def test_verified_failed_pipeline_result_is_known_failure_after_resource_cleanup(tmp_path):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.store.knowledge import KnownExecutionFailure

    rag = verified_rag()
    finalized = []

    async def failed(doc_id):
        return {"status": "failed", "track_id": "submission"}

    async def insert(*args, **kwargs):
        pass

    async def finalize():
        finalized.append(True)

    rag.doc_status.get_by_id = failed
    rag.ainsert, rag.finalize_storages = insert, finalize
    rag._index_storages = lambda: []

    async def factory(*args, **kwargs):
        return rag

    repo = Repository()
    claim = with_evidence(repo, CLAIM, tmp_path)
    adapter = CatalogLightRAGAdapter(repo, rag_factory=factory)
    with pytest.raises(KnownExecutionFailure):
        await adapter.submit(claim, "actual content", "one")
    assert finalized == [True]
    assert repo.receipt is None


@pytest.mark.asyncio
async def test_pending_delivery_with_missing_document_remains_ambiguous():
    from types import SimpleNamespace
    from papervault.knowledge.store.catalog_adapter import verify_outputs
    from papervault.store.knowledge import KnownExecutionFailure

    class Pending:
        async def get_by_id(self, key):
            return {"status": "pending", "track_id": "submission"}

    class Absent:
        async def get_by_id(self, key):
            return None

    with pytest.raises(RuntimeError) as caught:
        await verify_outputs(SimpleNamespace(doc_status=Pending(), full_docs=Absent()),
                             CLAIM, "doc", "actual content")
    assert not isinstance(caught.value, KnownExecutionFailure)


@pytest.mark.asyncio
async def test_recipe_drift_during_execution_prevents_sealing(monkeypatch, tmp_path):
    from papervault.knowledge.scheduler import catalog_intake
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    rag = verified_rag()

    async def insert(*args, **kwargs):
        monkeypatch.setattr(catalog_intake, "effective_distill_recipe",
                            lambda: {**RECIPE, "chunking": {"token_size": 99}})

    async def finalize():
        pass

    rag.ainsert, rag.finalize_storages = insert, finalize
    rag._index_storages = lambda: []

    async def factory(*args, **kwargs):
        return rag

    repo = Repository()
    claim = with_evidence(repo, CLAIM, tmp_path)
    adapter = CatalogLightRAGAdapter(repo, rag_factory=factory)
    with pytest.raises(RuntimeError, match="recipe"):
        await adapter.submit(claim, "actual content", "one")
    assert repo.receipt is None
