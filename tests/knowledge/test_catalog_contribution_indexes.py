"""Document discovery indexes must cover independently recorded graph material."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.knowledge.test_catalog_adapter import CLAIM, RECIPE, Repository, verified_rag, with_evidence
from tests.knowledge.test_catalog_retirement import memory_rag_for_retirement


@pytest.fixture(autouse=True)
def effective_recipe(monkeypatch):
    from papervault.knowledge.scheduler import catalog_intake

    monkeypatch.setattr(catalog_intake, "effective_distill_recipe", lambda: deepcopy(RECIPE))


def indexed_rag(doc_id, *, empty=False):
    rag = verified_rag()
    nodes = [] if empty else [
        {"entity_id": name, "id": name, "source_id": "chunk"} for name in ("Earth", "Moon")]
    edges = [] if empty else [{"source": "Earth", "target": "Moon", "source_id": "chunk"}]
    rag.chunk_entity_relation_graph.get_all_nodes = AsyncMock(return_value=nodes)
    rag.chunk_entity_relation_graph.get_all_edges = AsyncMock(return_value=edges)
    for role, field, values in (
        ("full_entities", "entity_names", [row["id"] for row in nodes]),
        ("full_relations", "relation_pairs", [["Earth", "Moon"]] if edges else []),
    ):
        store = getattr(rag, role)
        store.table_name = role
        store.rows = {doc_id: {"id": doc_id, field: values, "count": len(values)}} if values else {}

        async def query(sql, params, *, store=store, **kwargs):
            return list(deepcopy(store.rows).values())

        async def get(key, *, store=store):
            return deepcopy(store.rows.get(key))

        store.db = SimpleNamespace(query=query)
        store.get_by_id = get
    rag.ainsert = AsyncMock()
    rag._index_storages = lambda: []
    rag.finalize_storages = AsyncMock()
    return rag


@pytest.mark.asyncio
@pytest.mark.parametrize("role,field", [
    ("full_entities", "entity_names"), ("full_relations", "relation_pairs")])
@pytest.mark.parametrize("damage", ["missing", "empty", "wrong"])
async def test_build_cannot_seal_graph_with_missing_or_incomplete_document_index(tmp_path, role, field, damage):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    doc_id = "doc-" + hashlib.sha256(CLAIM["submission_id"].encode()).hexdigest()
    rag = indexed_rag(doc_id)
    store = getattr(rag, role)
    if damage == "missing":
        store.rows.clear()
    else:
        store.rows[doc_id][field] = [] if damage == "empty" else (
            ["Unrelated"] if role == "full_entities" else [["Other", "Unrelated"]])
    repo = Repository()
    adapter = CatalogLightRAGAdapter(repo, rag_factory=AsyncMock(return_value=rag))
    with pytest.raises(RuntimeError, match="index"):
        await adapter.submit(with_evidence(repo, CLAIM, tmp_path), "actual content", "paper/one")
    assert repo.accepted and not repo.sealed and repo.receipt is None
    rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_verified_build_records_document_graph_contribution_independently_of_indexes(empty):
    from papervault.knowledge.store.catalog_adapter import verify_outputs

    inventory, _ = await verify_outputs(indexed_rag("doc", empty=empty), CLAIM, "doc", "actual content")
    graph = next(row for row in inventory if row["role"] == "graph")
    assert graph["verification"]["evidence"]["document_contribution"] == {
        "doc_id": "doc", "chunk_ids": ["chunk"],
        "entity_ids": [] if empty else ["Earth", "Moon"],
        "relation_pairs": [] if empty else [["Earth", "Moon"]],
    }
    if empty:
        assert all(row["verification"]["state"] == "verified_empty" for row in inventory
                   if row["role"] in {"graph", "full_entities", "full_relations"})


@pytest.mark.asyncio
@pytest.mark.parametrize("role,key", [("entity_chunks", "Earth"), ("relation_chunks", "Earth<SEP>Moon")])
@pytest.mark.parametrize("indexed", [False, True])
async def test_source_limited_membership_requires_index_even_without_private_graph_write(role, key, indexed):
    from papervault.knowledge.store.catalog_adapter import verify_outputs

    rag = indexed_rag("doc", empty=True)
    tracker = getattr(rag, role)
    tracker.table_name = role
    tracker.db = SimpleNamespace(query=AsyncMock(return_value=[{
        "id": key, "chunk_ids": '["older", "chunk"]', "count": 2}]))
    rag.chunk_entity_relation_graph.get_node = AsyncMock(return_value={"entity_id": "Earth", "source_id": "older"})
    rag.chunk_entity_relation_graph.get_edge = AsyncMock(return_value={"source_id": "older"})
    if not indexed:
        with pytest.raises(RuntimeError, match="index"):
            await verify_outputs(rag, CLAIM, "doc", "actual content")
    else:
        entity = role == "entity_chunks"
        index = rag.full_entities if entity else rag.full_relations
        index.rows["doc"] = {"id": "doc", "entity_names" if entity else "relation_pairs":
                             ["Earth"] if entity else [["Earth", "Moon"]], "count": 1}
        inventory, _ = await verify_outputs(rag, CLAIM, "doc", "actual content")
        proof = next(row for row in inventory if row["role"] == "graph")["verification"]["evidence"]["document_contribution"]
        assert proof["entity_ids"] == (["Earth"] if entity else [])
        assert proof["relation_pairs"] == ([] if entity else [["Earth", "Moon"]])


@pytest.mark.asyncio
async def test_rebuilt_other_document_rows_do_not_become_successor_contributions():
    from papervault.knowledge.store.catalog_adapter import verify_outputs

    rag = indexed_rag("doc", empty=True)
    rag.chunk_entity_relation_graph.get_all_nodes.return_value = [
        {"entity_id": "Other", "id": "Other", "source_id": "another-document-chunk"}]
    inventory, _ = await verify_outputs(rag, CLAIM, "doc", "actual content")
    proof = next(row for row in inventory if row["role"] == "graph")["verification"]["evidence"]["document_contribution"]
    assert proof["entity_ids"] == proof["relation_pairs"] == []


def contribution_receipt(entities, relations):
    return {"provider_observations": {"doc_id": "old-doc"}, "output_inventory": [{
        "role": "graph", "verification": {"evidence": {"document_contribution": {
            "doc_id": "old-doc", "chunk_ids": ["old:0"],
            "entity_ids": entities, "relation_pairs": relations,
        }}}}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,role,field", [
    ("entity", "full_entities", "entity_names"),
    ("relationship", "full_relations", "relation_pairs"),
])
@pytest.mark.parametrize("damage", ["missing", "empty", "wrong"])
async def test_retirement_checks_recorded_contribution_before_original_target_discovery(kind, role, field, damage):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.store.knowledge import RetirementFailure

    claim = {**CLAIM, "retired_receipt_ids": ["retired"]}
    rag = await memory_rag_for_retirement(claim, kind=kind)
    retired = contribution_receipt(["Shared"] if kind == "entity" else [],
                                   [["Neighbor", "Shared"]] if kind == "relationship" else [])
    store = getattr(rag, role)
    if damage == "missing":
        store.rows.clear()
    else:
        store.rows["old-doc"][field] = [] if damage == "empty" else (
            ["Unrelated"] if role == "full_entities" else [["Other", "Unrelated"]])
    repo = Repository()
    repo.get_receipt = lambda key: retired
    original = rag.adelete_by_doc_id
    rag.adelete_by_doc_id = AsyncMock(wraps=original)
    with pytest.raises(RetirementFailure) as caught:
        await CatalogLightRAGAdapter(repo)._retire(rag, claim)
    assert "index" in caught.value.evidence["index_validation_error"]
    rag.adelete_by_doc_id.assert_not_awaited()
    assert "old:0" in (await rag.chunk_entity_relation_graph.get_node("Shared"))["source_id"]
    assert await rag.doc_status.get_by_id("old-doc") is not None


@pytest.mark.asyncio
async def test_retirement_cannot_invent_missing_historical_contribution_evidence():
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.store.knowledge import RetirementFailure

    claim = {**CLAIM, "retired_receipt_ids": ["retired"]}
    rag = await memory_rag_for_retirement(claim)
    rag.adelete_by_doc_id = AsyncMock(wraps=rag.adelete_by_doc_id)
    repo = Repository()
    repo.get_receipt = lambda key: {"provider_observations": {"doc_id": "old-doc"}}
    with pytest.raises(RetirementFailure):
        await CatalogLightRAGAdapter(repo)._retire(rag, claim)
    rag.adelete_by_doc_id.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[], ["old:0"]])
async def test_retirement_requires_status_to_discover_every_recorded_document_chunk(chunks):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.store.knowledge import RetirementFailure

    claim = {**CLAIM, "retired_receipt_ids": ["retired"]}
    rag = await memory_rag_for_retirement(claim)
    retired = contribution_receipt(["Shared"], [])
    retired["output_inventory"][0]["verification"]["evidence"]["document_contribution"]["chunk_ids"].append("old:1")
    rag.doc_status.rows["old-doc"]["chunks_list"] = chunks
    repo = Repository()
    repo.get_receipt = lambda key: retired
    rag.adelete_by_doc_id = AsyncMock(wraps=rag.adelete_by_doc_id)
    with pytest.raises(RetirementFailure):
        await CatalogLightRAGAdapter(repo)._retire(rag, claim)
    rag.adelete_by_doc_id.assert_not_awaited()
