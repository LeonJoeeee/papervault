"""Retirement must not seal a partially rebuilt inherited contribution."""
from types import SimpleNamespace
from uuid import uuid4
from copy import deepcopy

import pytest

from tests.knowledge.test_catalog_adapter import (
    CLAIM, RECIPE, Repository, verified_rag, with_evidence,
)

@pytest.fixture(autouse=True)
def effective_recipe(monkeypatch):
    from papervault.knowledge.scheduler import catalog_intake

    recipe = deepcopy(RECIPE)
    recipe["prompts"].update(DEFAULT_TUPLE_DELIMITER="<|#|>",
                             DEFAULT_COMPLETION_DELIMITER="<|COMPLETE|>")
    monkeypatch.setattr(catalog_intake, "effective_distill_recipe", lambda: deepcopy(recipe))
    monkeypatch.setattr(Repository, "get_recipe", lambda self, digest: deepcopy(recipe))


async def _pipeline(workspace):
    from lightrag.kg.shared_storage import (
        get_namespace_data, initialize_pipeline_status, initialize_share_data,
    )

    initialize_share_data()
    await initialize_pipeline_status(workspace=workspace)
    return await get_namespace_data("pipeline_status", workspace=workspace)


async def _delivery(tmp_path, deletion):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    rag, repo = verified_rag(), Repository()
    rag.workspace = "retirement_" + uuid4().hex
    for value in vars(rag).values():
        if hasattr(value, "workspace"):
            value.workspace = rag.workspace
    finalized = []

    async def remove(doc_id):
        assert doc_id == "old-doc"
        pipeline = await _pipeline(rag.workspace)
        pipeline["history_messages"][:] = [f"Starting deletion for document: {doc_id}"]
        return await deletion(pipeline)

    async def insert(*args, **kwargs):
        pass

    async def finalize():
        finalized.append(True)

    async def factory(*args, **kwargs):
        return rag

    rag.adelete_by_doc_id, rag.ainsert = remove, insert
    rag.finalize_storages, rag._index_storages = finalize, lambda: []
    from papervault.store.knowledge import canonical_recipe
    claim = with_evidence(repo, {**CLAIM, "namespace": rag.workspace,
        "recipe_digest": canonical_recipe(repo.get_recipe(None))[1],
        "retired_receipt_ids": ["retired"]}, tmp_path)
    return CatalogLightRAGAdapter(repo, rag_factory=factory), repo, claim, finalized


def _result(status="success", message="Document old-doc successfully deleted"):
    from lightrag.base import DeletionResult

    return DeletionResult(status=status, doc_id="old-doc", message=message,
                          status_code=200 if status == "success" else 404,
                          file_path="paper/old")


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "KG rebuild completed. Failed: 1 entities, 0 relationships.",
    "KG rebuild completed. Failed: 0 entities, 1 relationships.",
])
async def test_success_result_with_partial_rebuild_counts_never_seals(tmp_path, message):
    async def delete(pipeline):
        return _result(message=message)

    adapter, repo, claim, finalized = await _delivery(tmp_path, delete)
    with pytest.raises(RuntimeError, match="retir|rebuild"):
        await adapter.submit(claim, "actual content", "one")
    assert repo.accepted and repo.receipt is None and not repo.sealed
    assert finalized == [True]


@pytest.mark.asyncio
async def test_not_found_cannot_prove_inherited_contribution_was_removed(tmp_path):
    async def delete(pipeline):
        return _result(status="not_found", message="Document old-doc not found")

    adapter, repo, claim, finalized = await _delivery(tmp_path, delete)
    with pytest.raises(RuntimeError, match="retir|rebuild"):
        await adapter.submit(claim, "actual content", "one")
    assert repo.accepted and repo.receipt is None and not repo.sealed
    assert finalized == [True]


@pytest.mark.asyncio
async def test_clean_retirement_seals_verified_successor(tmp_path):
    async def delete(pipeline):
        pipeline["history_messages"].append(
            "KG rebuild completed: 1 entities and 1 relationships rebuilt successfully.")
        return _result()

    adapter, repo, claim, finalized = await _delivery(tmp_path, delete)
    receipt = await adapter.submit(claim, "actual content", "one")
    assert repo.sealed and receipt["sealed"] and receipt["durable"]
    assert finalized == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["entity", "relationship", "no-cache", "parse"])
async def test_pinned_rebuilder_hidden_failure_history_prevents_retirement_seal(tmp_path, failure):
    """Run the pinned reducer: its failures do not reach DeletionResult.message."""
    from lightrag.kg.shared_storage import get_namespace_lock
    from lightrag.operate import rebuild_knowledge_from_chunks

    class Cache:
        async def get_by_ids(self, ids):
            return [{"cache_type": "extract", "chunk_id": "remaining",
                     "return": 7 if failure == "parse" else "<|COMPLETE|>", "create_time": 1}]

    class Chunks:
        async def get_by_ids(self, ids):
            return [{"content": "remaining evidence", "file_path": "other",
                     "llm_cache_list": [] if failure == "no-cache" else ["cached"]}]

        async def get_by_id(self, key):
            return {"content": "remaining evidence", "file_path": "other"}

    class Graph:
        async def get_node(self, key):
            if failure == "entity":
                raise OSError("entity backend rejected rebuild")
            return None

        async def get_edge(self, src, tgt):
            raise OSError("relationship backend rejected rebuild")

    observed = []

    async def delete(pipeline):
        await rebuild_knowledge_from_chunks(
            entities_to_rebuild={} if failure == "relationship" else {"Shared": ["remaining"]},
            relationships_to_rebuild={("Shared", "Neighbor"): ["remaining"]}
                if failure == "relationship" else {},
            knowledge_graph_inst=Graph(), entities_vdb=SimpleNamespace(),
            relationships_vdb=SimpleNamespace(), text_chunks_storage=Chunks(),
            llm_response_cache=Cache(), global_config={"workspace": claim["namespace"]},
            pipeline_status=pipeline,
            pipeline_status_lock=get_namespace_lock("pipeline_status", workspace=claim["namespace"]))
        observed.extend(pipeline["history_messages"])
        result = _result()
        assert "Failed" not in result.message  # Actual dependency's deletion summary omits these failures.
        return result

    adapter, repo, claim, finalized = await _delivery(tmp_path, delete)
    with pytest.raises(RuntimeError, match="retir|rebuild"):
        await adapter.submit(claim, "actual content", "one")
    assert any("Failed" in message or "cannot rebuild" in message for message in observed)
    assert repo.accepted and repo.receipt is None and not repo.sealed
    assert finalized == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [[], ["Starting deletion for document: another-document"]])
async def test_unattributed_success_history_cannot_prove_retirement(tmp_path, history):
    async def delete(pipeline):
        pipeline["history_messages"][:] = history
        return _result()

    adapter, repo, claim, _ = await _delivery(tmp_path, delete)
    with pytest.raises(RuntimeError, match="retir|rebuild"):
        await adapter.submit(claim, "actual content", "one")
    assert repo.accepted and repo.receipt is None and not repo.sealed


@pytest.mark.asyncio
async def test_clean_retirement_ignores_entity_names_and_other_workspace_failures(tmp_path):
    other = await _pipeline("unrelated_" + uuid4().hex)
    other["history_messages"][:] = ["KG rebuild completed. Failed: 1 entities, 0 relationships."]

    async def delete(pipeline):
        pipeline["history_messages"].append("Rebuild `Failed Star` from 1 chunks")
        pipeline["history_messages"].append(
            "KG rebuild completed: 1 entities and 0 relationships rebuilt successfully.")
        return _result()

    adapter, repo, claim, _ = await _delivery(tmp_path, delete)
    await adapter.submit(claim, "actual content", "one")
    assert repo.sealed
