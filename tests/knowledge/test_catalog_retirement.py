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


async def memory_rag_for_retirement(claim, *, pause=None, kind="entity", cache_case="complete", wrap_pause=False,
                                    summary_result="summary from remaining evidence", summary_cache=None, on_summary=None):
    """Real pinned deletion/reducer over memory stores; only the LLM/backend are doubles."""
    from functools import partial
    from lightrag import LightRAG
    from lightrag.base import DocStatus
    from lightrag.prompt import PROMPTS
    from lightrag.utils import compute_mdhash_id

    class Store:
        def __init__(self, namespace, rows=None):
            self.namespace, self.workspace = namespace, claim["namespace"]
            self.rows = deepcopy(rows or {})

        async def get_by_id(self, key):
            if summary_cache is not None and key.startswith("default:summary:"):
                return {"return": summary_cache, "create_time": 1, "cache_type": "summary"}
            return deepcopy(self.rows.get(key))

        async def get_by_ids(self, keys):
            return [deepcopy(self.rows.get(key)) for key in keys]

        async def upsert(self, rows):
            self.rows.update(deepcopy(rows))

        async def delete(self, keys):
            for key in keys:
                self.rows.pop(key, None)

        async def index_done_callback(self):
            pass

    class Graph:
        workspace = claim["namespace"]

        def __init__(self):
            sources = "old:0<SEP>remain:1<SEP>remain:2"
            self.nodes = {name: {"entity_id": name, "entity_type": "concept", "source_id": sources,
                "description": "retired and remaining evidence", "file_path": "old<SEP>other"}
                for name in ("Shared", "Neighbor")}
            self.edges = {("Neighbor", "Shared"): {"source_id": sources,
                "description": "retired and remaining relationship evidence", "keywords": "related",
                "weight": 2.0, "file_path": "old<SEP>other"}}

        async def get_node(self, key):
            return deepcopy(self.nodes.get(key))

        async def get_edge(self, source, target):
            return deepcopy(self.edges.get(tuple(sorted((source, target)))))

        async def get_nodes_batch(self, keys):
            return {key: await self.get_node(key) for key in keys}

        async def get_edges_batch(self, pairs):
            return {(pair["src"], pair["tgt"]): await self.get_edge(pair["src"], pair["tgt"])
                    for pair in pairs}

        async def get_node_edges(self, key):
            return [pair for pair in self.edges if key in pair]

        async def has_node(self, key):
            return key in self.nodes

        async def upsert_node(self, key, node_data):
            self.nodes[key] = deepcopy(node_data)

        async def upsert_edge(self, source, target, edge_data):
            self.edges[tuple(sorted((source, target)))] = deepcopy(edge_data)

        async def remove_nodes(self, keys):
            for key in keys:
                self.nodes.pop(key, None)

        async def remove_edges(self, pairs):
            for pair in pairs:
                self.edges.pop(tuple(pair), None)

        async def index_done_callback(self):
            pass

    class Tokenizer:
        def encode(self, text):
            return list(text.encode())

        def decode(self, values):
            return bytes(values).decode()

    calls = []

    async def extract(prompt, **kwargs):
        calls.append(prompt)
        if on_summary is not None:
            await on_summary()
        if pause is not None:
            if wrap_pause:
                raise RuntimeError("LLM callback wrapped the build pause") from pause
            raise pause
        return summary_result

    rag = SimpleNamespace(workspace=claim["namespace"], chunk_entity_relation_graph=Graph(),
                          llm_calls=calls, finalized=False)
    for role in ("full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks",
                 "relation_chunks", "llm_response_cache", "doc_status", "chunks_vdb", "entities_vdb",
                 "relationships_vdb"):
        setattr(rag, role, Store(role))
    rag.full_docs.rows["old-doc"] = {"content": "old evidence", "file_path": "old"}
    rag.doc_status.rows["old-doc"] = {"status": DocStatus.PROCESSED.value, "file_path": "old",
        "content_summary": "old", "content_length": 12, "chunks_list": ["old:0"], "chunks_count": 1,
        "metadata": {}, "created_at": "2026-10-01T00:00:00+00:00", "updated_at": "2026-10-01T00:00:00+00:00"}
    rag.full_entities.rows["old-doc"] = {"entity_names": ["Shared"] if kind == "entity" else [], "count": 1}
    rag.full_relations.rows["old-doc"] = {"relation_pairs": [["Shared", "Neighbor"]]
        if kind == "relationship" else [], "count": 1}
    # This fixture starts with an already verified publication. Preserve its
    # contribution independently of later mutations to the discovery indexes.
    rag.retired_receipt = {"provider_observations": {"doc_id": "old-doc"}, "output_inventory": [{
        "role": "graph", "verification": {"evidence": {"document_contribution": {
            "doc_id": "old-doc", "chunk_ids": ["old:0"],
            "entity_ids": ["Shared"] if kind == "entity" else [],
            "relation_pairs": [["Neighbor", "Shared"]] if kind == "relationship" else [],
        }}}}]}
    for chunk in ("old:0", "remain:1", "remain:2"):
        rag.text_chunks.rows[chunk] = {"id": chunk, "content": "remaining evidence", "file_path": "other",
            "llm_cache_list": [] if chunk == "old:0" else ["cache:" + chunk]}
        if chunk == "old:0":
            continue
        text = f"entity<|#|>Shared<|#|>concept<|#|>remaining entity description {chunk}"
        if kind == "relationship":
            text = f"relation<|#|>Shared<|#|>Neighbor<|#|>related<|#|>remaining relation description {chunk}"
        rag.llm_response_cache.rows["cache:" + chunk] = {"cache_type": "extract", "chunk_id": chunk,
            "return": text + "\n<|COMPLETE|>", "create_time": 1}
    if cache_case == "mixed":
        rag.text_chunks.rows["remain:1"]["llm_cache_list"] = []
    elif cache_case == "missing-entry":
        rag.llm_response_cache.rows.pop("cache:remain:1")
    elif cache_case in {"wrong-target", "limited-no-data"}:
        affected = ["remain:1", "remain:2"] if cache_case == "wrong-target" else ["remain:1"]
        for chunk in affected:
            entry = rag.llm_response_cache.rows["cache:" + chunk]
            entry["return"] = entry["return"].replace("Shared", "Unrelated")
    if cache_case == "normalized-source":
        for chunk in ("remain:1", "remain:2"):
            rag.llm_response_cache.rows["cache:" + chunk]["return"] = (
                "relation<|#|>Shared<|#|>Neighbor<|#|>related<|#|>&amp;#x01;\n<|COMPLETE|>")
    rag.entities_vdb.rows[compute_mdhash_id("Shared", prefix="ent-")] = {"source_id": "old:0<SEP>remain:1<SEP>remain:2"}
    config = {"workspace": rag.workspace, "llm_model_max_async": 2, "llm_model_name": "fixture",
        "role_llm_funcs": {"extract": extract}, "source_ids_limit_method": "KEEP",
        "max_source_ids_per_entity": 1 if cache_case == "limited-no-data" else 100,
        "max_source_ids_per_relation": 1 if cache_case == "limited-no-data" else 100,
        "summary_context_size": 10000, "summary_max_tokens": 10000,
        "summary_length_recommended": 100, "force_llm_summary_on_merge": 2, "tokenizer": Tokenizer()}
    rag.llm_response_cache.global_config = {"enable_llm_cache": False,
        "enable_llm_cache_for_entity_extract": summary_cache is not None}
    rag._build_global_config = lambda: config
    rag._index_storages = lambda: [value for value in vars(rag).values() if isinstance(value, Store)]
    for method in ("adelete_by_doc_id", "_insert_done", "_update_delete_retry_state"):
        setattr(rag, method, partial(getattr(LightRAG, method), rag))

    async def finalize():
        rag.finalized = True

    rag.finalize_storages = finalize
    rag.default_prompts = deepcopy(PROMPTS)
    await _pipeline(rag.workspace)
    return rag


def retirement_repository(rag):
    repo = Repository()
    repo.get_receipt = lambda receipt_id: deepcopy(rag.retired_receipt)
    return repo


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["entity", "relationship"])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_real_rebuild_summary_pause_reaches_metadata_delivery_as_typed_deferral(tmp_path, monkeypatch, kind, wrapped):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.store.build_breaker import BuildPausedError
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.knowledge.scheduler import catalog_intake
    from lightrag.prompt import PROMPTS
    from papervault.store.knowledge import canonical_recipe

    recipe = deepcopy(RECIPE)
    recipe["prompts"] = deepcopy(PROMPTS)
    monkeypatch.setattr(catalog_intake, "effective_distill_recipe", lambda: deepcopy(recipe))
    monkeypatch.setattr(Repository, "get_recipe", lambda self, digest: deepcopy(recipe))
    claim = {**CLAIM, "namespace": "pause_retirement_" + uuid4().hex,
             "retired_receipt_ids": ["retired"], "recipe_digest": canonical_recipe(recipe)[1]}
    pause = BuildPausedError("build summary paused by schedule")
    rag = await memory_rag_for_retirement(claim, pause=pause, kind=kind, wrap_pause=wrapped)
    store = ImmutableInputStore(tmp_path / "inputs")
    artifact = store.capture(b'{"title":"metadata"}', role="metadata")
    repo = retirement_repository(rag)
    repo.manifest = manifest = {"tier": "metadata", "artifacts": [artifact]}

    async def factory(*args, **kwargs):
        return rag

    adapter = CatalogLightRAGAdapter(repo, rag_factory=factory)
    with pytest.raises(BuildPausedError) as caught:
        await adapter.submit_metadata(claim, manifest, store)
    assert caught.value is pause
    assert len(rag.llm_calls) == 1
    assert rag.finalized
    assert repo.accepted and repo.receipt is None and not repo.sealed


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nested", "parallel"])
async def test_retirement_rebuild_observer_is_scoped_to_its_graph_and_task(mode):
    import asyncio
    from papervault.knowledge.store.build_breaker import BuildPausedError
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    claim = {**CLAIM, "namespace": "pause_scope_" + uuid4().hex, "retired_receipt_ids": ["retired"]}
    pause = BuildPausedError("only this retirement is paused")
    paused = await memory_rag_for_retirement(claim, pause=pause)
    other = await memory_rag_for_retirement({"namespace": "other_scope_" + uuid4().hex}, cache_case="mixed")
    unrelated_results = []
    original = paused.adelete_by_doc_id

    async def unrelated():
        result = await other.adelete_by_doc_id("old-doc")
        unrelated_results.append(result)

    async def nested(doc_id):
        await unrelated()
        return await original(doc_id)

    if mode == "nested":
        paused.adelete_by_doc_id = nested

    async def retire():
        with pytest.raises(BuildPausedError) as caught:
            await CatalogLightRAGAdapter(retirement_repository(paused))._retire(paused, claim)
        assert caught.value is pause

    if mode == "parallel":
        await asyncio.gather(retire(), unrelated())
    else:
        await retire()
    assert [result.status for result in unrelated_results] == ["success"]
    # A subsequent caller outside retirement must use the unmodified reducer too.
    outside = await memory_rag_for_retirement({"namespace": "after_scope_" + uuid4().hex}, cache_case="mixed")
    assert (await outside.adelete_by_doc_id("old-doc")).status == "success"



@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["entity", "relationship"])
@pytest.mark.parametrize("cache_case", ["missing-entry", "wrong-target", "limited-no-data"])
async def test_retirement_requires_attributable_cached_descriptions_after_source_limits(kind, cache_case):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.store.knowledge import RetirementFailure

    claim = {**CLAIM, "namespace": "evidence_retirement_" + uuid4().hex,
             "retired_receipt_ids": ["retired"]}
    rag = await memory_rag_for_retirement(claim, kind=kind, cache_case=cache_case)
    with pytest.raises(RetirementFailure) as caught:
        await CatalogLightRAGAdapter(retirement_repository(rag))._retire(rag, claim)
    assert "isolated rebuild required" in caught.value.evidence["rebuild_validation_error"]
    assert await rag.doc_status.get_by_id("old-doc") is not None
    assert rag.llm_calls == []



@pytest.mark.asyncio
@pytest.mark.parametrize("result,cached", [
    ("", False), ("   ", False), ("<think>no final evidence</think>", False),
    ("\x00", False), ("", True), ("<think>no final evidence</think>", True),
])
async def test_real_retirement_rejects_empty_normalized_or_cached_rebuild_summary(result, cached):
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.store.knowledge import RetirementFailure

    claim = {**CLAIM, "namespace": "empty_summary_" + uuid4().hex, "retired_receipt_ids": ["retired"]}
    rag = await memory_rag_for_retirement(claim, kind="relationship",
        summary_result=result, summary_cache=result if cached else None)
    with pytest.raises(RetirementFailure) as caught:
        await CatalogLightRAGAdapter(retirement_repository(rag))._retire(rag, claim)
    assert "empty" in caught.value.evidence["rebuild_validation_error"]
    assert "isolated rebuild required" in caught.value.evidence["rebuild_validation_error"]
    assert len(rag.llm_calls) == (0 if cached else 1)
    assert await rag.doc_status.get_by_id("old-doc") is not None



@pytest.mark.asyncio
async def test_retirement_rejects_normalized_empty_single_description_without_llm():
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.store.knowledge import RetirementFailure

    claim = {**CLAIM, "namespace": "source_summary_" + uuid4().hex, "retired_receipt_ids": ["retired"]}
    rag = await memory_rag_for_retirement(claim, kind="relationship", cache_case="normalized-source")
    with pytest.raises(RetirementFailure) as caught:
        await CatalogLightRAGAdapter(retirement_repository(rag))._retire(rag, claim)
    assert "empty summary" in caught.value.evidence["rebuild_validation_error"]
    assert rag.llm_calls == []
    assert await rag.doc_status.get_by_id("old-doc") is not None


@pytest.mark.asyncio
async def test_retirement_summary_validation_ignores_other_config_inside_observed_callback():
    import lightrag.operate as operate
    from papervault.knowledge.store.build_breaker import BuildPausedError
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter

    claim = {**CLAIM, "namespace": "summary_scope_" + uuid4().hex, "retired_receipt_ids": ["retired"]}
    pause = BuildPausedError("only the retirement callback is paused")
    unrelated = []

    async def nested():
        async def empty_summary(*args, **kwargs):
            return ""

        config = {**rag._build_global_config(), "role_llm_funcs": {"extract": empty_summary}}
        unrelated.append(await operate._summarize_descriptions(
            "Relation", "Unrelated", ["first evidence", "second evidence"], config))

    rag = await memory_rag_for_retirement(claim, pause=pause, on_summary=nested)
    with pytest.raises(BuildPausedError) as caught:
        await CatalogLightRAGAdapter(retirement_repository(rag))._retire(rag, claim)
    assert caught.value is pause
    assert unrelated == [""]
