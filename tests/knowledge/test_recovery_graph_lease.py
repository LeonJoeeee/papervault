"""Legacy and catalog graph lifetimes participate in coordinated recovery."""
from types import SimpleNamespace

import pytest

from papervault.store.fencing import WriterFenced, exclusive_fence


@pytest.mark.parametrize("namespace", [None, "pv_isolated_attempt"])
async def test_recovery_blocks_graph_construction_before_backend_initialization(tmp_path, monkeypatch, namespace):
    from papervault.knowledge.store import graph

    # The barrier is real; only expensive backend construction is replaced.
    calls = []

    async def construct(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    assert hasattr(graph, "_construct_graph")
    monkeypatch.setattr(graph, "_construct_graph", construct)
    with exclusive_fence(tmp_path):
        with pytest.raises(WriterFenced):
            await graph._initialize_graph(namespace=namespace, working_dir=str(tmp_path))
    assert calls == []


async def test_graph_holds_lease_until_backend_finalization_drains(tmp_path, monkeypatch):
    from papervault.knowledge.store import graph

    finalized = []

    async def finalize():
        with pytest.raises(WriterFenced):
            with exclusive_fence(tmp_path):
                pytest.fail("export entered before graph finalization completed")
        finalized.append(True)

    async def construct(**kwargs):
        return SimpleNamespace(finalize_storages=finalize)

    assert hasattr(graph, "_construct_graph")
    monkeypatch.setattr(graph, "_construct_graph", construct)
    rag = await graph._initialize_graph(working_dir=str(tmp_path))
    with pytest.raises(WriterFenced):
        with exclusive_fence(tmp_path):
            pytest.fail("recovery barrier entered while graph was active")
    await rag.finalize_storages()
    assert finalized == [True]
    with exclusive_fence(tmp_path):
        pass
def test_immutable_capture_obeys_the_knowledge_root_barrier(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.store.fencing import exclusive_fence

    store = ImmutableInputStore(tmp_path / "catalog_inputs")
    with exclusive_fence(tmp_path), pytest.raises(RuntimeError, match="fenc|writer|lease"):
        store.capture(b"must wait until coordinated snapshot completes", role="metadata")
    assert not list(store.root.iterdir())


@pytest.mark.parametrize("failure", ["exception", "pending"])
async def test_upstream_swallowed_flush_failure_keeps_recovery_fenced(tmp_path, monkeypatch, failure):
    import threading
    from lightrag.lightrag import LightRAG, StoragesStatus
    from papervault.knowledge.store import graph

    async def fail():
        if failure == "exception":
            raise OSError("injected unflushed vector buffer")

    store = SimpleNamespace(finalize=fail, _pending_vector_docs={"unfinished": {}}, _pending_vector_deletes=set())
    names = ("full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks", "relation_chunks",
        "entities_vdb", "relationships_vdb", "chunks_vdb", "chunk_entity_relation_graph", "llm_response_cache", "doc_status")
    rag = object.__new__(LightRAG)
    rag.__dict__.update(dict.fromkeys(names), _storages_status=StoragesStatus.INITIALIZED,
                        _parser_shutdown_event=threading.Event(), _parser_executor=None)
    rag.entities_vdb = store

    async def construct(**kwargs):
        return rag

    monkeypatch.setattr(graph, "_construct_graph", construct)
    await graph._initialize_graph(working_dir=str(tmp_path))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="finaliz|flush|pending"):
            await rag.finalize_storages()
        with pytest.raises(WriterFenced), exclusive_fence(tmp_path):
            pytest.fail("unflushed storage allowed recovery to enter")
