"""Abstract document boundaries survive retrieval facets and served-window cuts."""
import pytest

from papervault.knowledge.ingest.abstract_doc import ABSTRACT_HEADER
from papervault.knowledge.query import multiquery as mq


class Rag:
    async def aquery_data(self, intent, param):
        chunks = ([{"file_path": "A", "content": ABSTRACT_HEADER + "\nAbstract start"},
                   {"file_path": "B", "content": "Full text B"}]
                  if intent == "anchor" else [{"file_path": "A", "content": "Abstract continuation"}])
        return {"status": "success", "data": {"chunks": chunks},
                "metadata": {"processing_info": {"total_entities_found": 30}}}


@pytest.mark.asyncio
async def test_abstract_tier_propagates_across_facets_before_rrf_served_cut(monkeypatch):
    async def facets(*args):
        return [{"query": "facet", "hl": ["topic"], "ll": []}]

    monkeypatch.setattr(mq, "_decompose", facets)
    monkeypatch.setenv("KS_MQ_FANOUT", "0")
    monkeypatch.setattr(mq.aq, "_CHUNK_TOP_K", 1)
    monkeypatch.setattr(mq, "_MIN_COV", "thin")
    data, _, fused = await mq.retrieve_fused("anchor", Rag())
    assert fused
    assert [chunk["file_path"] for chunk in data["chunks"]] == ["B"]


@pytest.mark.asyncio
async def test_standard_merge_propagates_abstract_tier_across_facets(monkeypatch):
    async def facets(*args):
        return [{"query": query, "hl": ["topic"], "ll": []} for query in ("anchor", "facet")]

    monkeypatch.setattr(mq, "_decompose_standard", facets)
    monkeypatch.setattr(mq, "_count_tokens", lambda text: 1)
    data, metadata = await mq.retrieve_standard("anchor", Rag())
    abstract_chunks = [chunk for chunk in data["chunks"] if chunk["file_path"] == "A"]
    assert len(abstract_chunks) == 2
    assert all(chunk["source_tier"] == "abstract_only" for chunk in abstract_chunks)
    assert metadata["processing_info"]["total_entities_found"] == 30
