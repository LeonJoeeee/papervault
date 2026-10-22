"""The eval CLI uses the same pinned serving context as production queries."""
import argparse
import asyncio
import json

import pytest

from papervault.eval import run_eval
from papervault.knowledge.store import catalog_serving, graph
from tests.knowledge.test_catalog_query_pinning import Repository


@pytest.mark.parametrize("path, source", [
    ("textbook:Book#s2", "textbook:Book"),
    ("notebook:Notes#s1", "notebook:Notes"),
    ("web:nasa", "web:nasa"),
])
def test_slim_chunk_preserves_operator_citation_identity(path, source):
    chunk = run_eval._slim_chunk({"file_path": path, "content": "evidence"})
    assert chunk["paper_key"] is None
    assert chunk["source_key"] == source


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "retrieval", "cleanup"])
async def test_eval_cli_acquires_and_releases_catalog_pin_per_question(monkeypatch, tmp_path, failure):
    from papervault.store import knowledge_runtime as runtime

    repos = []
    finalized = []

    class Rag:
        async def aquery_data(self, intent, param):
            if failure == "retrieval":
                raise RuntimeError("retrieval unavailable")
            return {"status": "success", "data": {"chunks": [{"file_path": "A", "content": "evidence"}]},
                    "metadata": {"processing_info": {"total_entities_found": 20}}}

        async def finalize_storages(self):
            finalized.append(True)
            if failure == "cleanup":
                raise RuntimeError("storage finalization unavailable")

    async def factory(*args, **kwargs):
        return Rag()

    real_pin = catalog_serving.CatalogServingFacade.pin

    async def pin(repo, **kwargs):
        repos.append(repo)
        return await real_pin(repo, rag_factory=factory)

    def forbidden():
        pytest.fail("catalog evaluation reached legacy get_graph")

    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "postgres")
    monkeypatch.setattr(runtime, "assert_knowledge_authority", lambda expected: {"authority": expected})
    monkeypatch.setattr(runtime, "get_knowledge_repository", Repository)
    monkeypatch.setattr(catalog_serving.CatalogServingFacade, "pin", pin)
    monkeypatch.setattr(graph, "get_graph", forbidden)
    monkeypatch.setattr(graph, "assert_safe_workspace", lambda: "l0_probe")
    monkeypatch.setattr(run_eval, "_require_probe_workspace", lambda: "l0_probe")
    monkeypatch.setattr(run_eval, "_require_exclusive_gpu", lambda: None)
    monkeypatch.setattr(run_eval, "_rerank_fail_count", lambda: 0)
    monkeypatch.setattr(run_eval, "_reset_rerank_fails", lambda: None)
    monkeypatch.setattr(run_eval, "eval_path", lambda name: tmp_path / name)
    gold = tmp_path / "gold.jsonl"
    gold.write_text("\n".join(json.dumps({"qid": str(i), "intent": "question", "gold_keys": ["A"]})
                               for i in range(2)))
    args = argparse.Namespace(gold=str(gold), limit=0, tag="catalog", variant="baseline",
                              no_synth=True, concurrency=2)
    await run_eval.main_async(args)
    records = [json.loads(line) for line in (tmp_path / "catalog.jsonl").read_text().splitlines()]
    assert len(repos) == 2
    assert all(repo.released for repo in repos)
    assert len(finalized) == 2
    for record in records:
        if failure == "retrieval":
            assert "retrieval unavailable" in record["meta"]["error"]
        elif failure == "cleanup":
            assert "storage finalization unavailable" in record["meta"]["error"]
            assert record["retrieved_papers"] == ["A"]
        else:
            assert "error" not in record["meta"]
            assert record["query_receipt"]["manifest_revision"] == 1
            assert record["query_receipt"]["indexed_scope"] == ["paper:A"]


@pytest.mark.asyncio
async def test_owned_eval_context_is_closed_on_cancellation(monkeypatch):
    started, released = asyncio.Event(), asyncio.Event()

    class Rag:
        async def aquery_data(self, *args):
            started.set()
            await asyncio.Event().wait()

        async def close(self):
            released.set()

    async def acquire():
        return Rag()

    monkeypatch.setattr(catalog_serving, "get_query_graph", acquire)
    task = asyncio.create_task(run_eval.run_one(None, {"qid": "cancel", "intent": "question"}, {}))
    try:
        await asyncio.wait_for(started.wait(), 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert released.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("retrieval_fails", [False, True])
async def test_owned_eval_cleanup_failure_stays_in_question_record(monkeypatch, retrieval_fails):
    class Rag:
        async def aquery_data(self, *args):
            if retrieval_fails:
                raise RuntimeError("retrieval unavailable")
            return {"status": "success", "data": {"chunks": [{"file_path": "A", "content": "evidence"}]},
                    "metadata": {"processing_info": {"total_entities_found": 20}}}

        async def close(self):
            raise RuntimeError("release unavailable")

    async def acquire():
        return Rag()

    monkeypatch.setattr(catalog_serving, "get_query_graph", acquire)
    record = await run_eval.run_one(None, {"qid": "cleanup", "intent": "question"}, {}, no_synth=True)
    if retrieval_fails:
        assert record["meta"]["error"] == "RuntimeError: retrieval unavailable"
        assert record["meta"]["cleanup_error"] == "RuntimeError: release unavailable"
    else:
        assert record["meta"]["error"] == "RuntimeError: release unavailable"
        assert record["retrieved_papers"] == ["A"]
        assert record["data"]["chunks"][0]["content"] == "evidence"


@pytest.mark.asyncio
async def test_legacy_eval_opens_singleton_before_concurrent_questions(monkeypatch, tmp_path):
    acquisitions = []

    class Rag:
        async def aquery_data(self, *args):
            return {"status": "success", "data": {}}

    async def acquire():
        acquisitions.append(True)
        await asyncio.sleep(0)
        return Rag()

    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "legacy")
    monkeypatch.setattr(catalog_serving, "get_query_graph", acquire)
    monkeypatch.setattr(graph, "assert_safe_workspace", lambda: "l0_probe")
    monkeypatch.setattr(run_eval, "_require_probe_workspace", lambda: "l0_probe")
    monkeypatch.setattr(run_eval, "_require_exclusive_gpu", lambda: None)
    monkeypatch.setattr(run_eval, "_rerank_fail_count", lambda: 0)
    monkeypatch.setattr(run_eval, "_reset_rerank_fails", lambda: None)
    monkeypatch.setattr(run_eval, "eval_path", lambda name: tmp_path / name)
    gold = tmp_path / "gold.jsonl"
    gold.write_text("\n".join(json.dumps({"qid": str(i), "intent": "question", "gold_keys": []})
                               for i in range(2)))
    await run_eval.main_async(argparse.Namespace(gold=str(gold), limit=0, tag="legacy",
        variant="baseline", no_synth=True, concurrency=2))
    assert len(acquisitions) == 1
    assert len((tmp_path / "legacy.jsonl").read_text().splitlines()) == 2
