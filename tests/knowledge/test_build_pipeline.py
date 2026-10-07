"""The real pinned LightRAG pipeline, with fixture stores and no GPU/LLM/network."""
import asyncio
import importlib.metadata
import tomllib
from pathlib import Path

import numpy as np
import pytest
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

from papervault.knowledge.store import build_breaker
from papervault.knowledge.store.build_breaker import BuildBreaker, BuildPausedError


def test_private_pipeline_pin_moves_only_with_an_explicit_port():
    lock = tomllib.loads(Path('uv.lock').read_text())
    pinned = next(p['version'] for p in lock['package'] if p['name'] == 'lightrag-hku')
    assert pinned == importlib.metadata.version('lightrag-hku') == '1.5.4'


def test_pipeline_guard_rejects_unsupported_dependency_before_io(monkeypatch, tmp_path):
    from papervault.knowledge.store.build_pipeline import BuildAwareLightRAG

    monkeypatch.setattr(importlib.metadata, 'version', lambda _: '1.5.5')
    with pytest.raises(RuntimeError, match='1.5.4'):
        BuildAwareLightRAG(working_dir=str(tmp_path / 'must-not-exist'))
    assert not (tmp_path / 'must-not-exist').exists()


@pytest.mark.parametrize('adapter', [False, True])
@pytest.mark.parametrize('stage', ['extract', 'merge'])
@pytest.mark.parametrize('failure', ['paused', 'tripping-call'])
async def test_paused_pipeline_keeps_queue_and_never_writes_failed(adapter, stage, failure, monkeypatch, tmp_path):
    # Stock pipeline is a characterization of the defect; adapter must prevent every FAILED write.
    cls = LightRAG
    if adapter:
        from papervault.knowledge.store.build_pipeline import BuildAwareLightRAG
        cls = BuildAwareLightRAG
    now = [0.0]
    breaker = BuildBreaker(threshold=1, cooldown_s=600, clock=lambda: now[0])
    monkeypatch.setattr(build_breaker, '_BREAKER', breaker)

    async def embed(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    async def no_llm(*args, **kwargs):
        pytest.fail('fixture extraction must not contact an LLM')

    rag = cls(working_dir=str(tmp_path), workspace=f'fixture-{stage}-{adapter}-{failure}',
              llm_model_func=no_llm,
              embedding_func=EmbeddingFunc(embedding_dim=4, max_token_size=8192, func=embed),
              max_parallel_insert=1)
    await rag.initialize_storages()
    await initialize_pipeline_status(workspace=rag.workspace)
    writes = []
    original_upsert = rag.doc_status.upsert

    async def record(data):
        writes.extend(row['status'] for row in data.values())
        await original_upsert(data)

    monkeypatch.setattr(rag.doc_status, 'upsert', record)
    attempts = 0

    async def paused_extract(*args):
        nonlocal attempts
        attempts += 1
        breaker.record_failure()
        if failure == 'tripping-call':
            import httpx
            from openai import APIConnectionError
            raise APIConnectionError(request=httpx.Request('POST', 'https://fixture.invalid'))
        raise BuildPausedError('fixture pause')

    async def extracted(*args):
        return []

    import lightrag.pipeline as pipeline
    original_merge = pipeline.merge_nodes_and_edges
    if stage == 'merge':
        monkeypatch.setattr(rag, '_process_extract_entities', extracted)
        async def paused_merge(**kwargs):
            return await paused_extract()
        monkeypatch.setattr(pipeline, 'merge_nodes_and_edges', paused_merge)
    else:
        monkeypatch.setattr(rag, '_process_extract_entities', paused_extract)
    try:
        await rag.apipeline_enqueue_documents(
            input=['Unique fixture body one.', 'Unique fixture body two.', 'Unique fixture body three.'],
            ids=['paper:A', 'paper:B', 'paper:C'], file_paths=['A', 'B', 'C'])
        if adapter:
            with pytest.raises(BuildPausedError):
                await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), 5)
            assert 'failed' not in writes
            statuses = await rag.aget_docs_by_ids(['paper:A', 'paper:B', 'paper:C'])
            assert {st['status'] for st in statuses.values()} == {'pending'}
            assert attempts == 1
            # The same parsed bodies resume after cooldown, without deleting/re-enqueueing.
            now[0] = 601

            async def recovered_extract(*args):
                breaker.record_success()
                return []

            monkeypatch.setattr(rag, '_process_extract_entities', recovered_extract)
            monkeypatch.setattr(pipeline, 'merge_nodes_and_edges', original_merge)
            await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), 5)
            statuses = await rag.aget_docs_by_ids(['paper:A', 'paper:B', 'paper:C'])
            assert {st['status'] for st in statuses.values()} == {'processed'}
            assert 'failed' not in writes
        else:
            await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), 5)
            assert 'failed' in writes
    finally:
        await rag.finalize_storages()
