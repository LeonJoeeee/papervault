"""Eval and production must expose the same captured evidence contract."""
from copy import deepcopy

import pytest

from papervault.eval.run_eval import run_one
from papervault.knowledge.query import aquery as aq
from tests.knowledge.test_query_s2 import META, evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("no_synth", [False, True])
async def test_eval_shares_production_citations_and_retains_all_raw_evidence(monkeypatch, no_synth):
    async def synth(*args, **kwargs):
        return "Answer [Alpha]."
    monkeypatch.setattr(aq, "synth_answer", synth)
    class Rag:
        calls = 0
        async def aquery_data(self, intent, param):
            self.calls += 1
            return {"status": "success", "data": evidence(), "metadata": deepcopy(META)}
    rag = Rag()
    rec = await run_one(rag, {"qid": "s2", "intent": "query"}, {}, no_synth=no_synth)
    assert "error" not in rec["meta"]
    assert rec["cited_papers"] == (["Zeta", "Alpha", "Beta"] if no_synth else ["Alpha"])
    assert rec["retrieved_papers"] == ["Zeta", "Alpha", "Beta"]
    assert len(rec["data"]["chunks"]) == 4 and rag.calls == 1
    assert rec["data"]["chunks"][2]["source_tier"] == "abstract_only"
    assert rec["meta"]["query_receipt"] == rec["query_receipt"]
    assert rec["answer"] is None if no_synth else rec["answer"] == "Answer [Alpha]."


@pytest.mark.asyncio
async def test_empty_eval_coverage_never_synthesizes_or_loses_retrieved_candidates(monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("empty coverage must skip synthesis")
    monkeypatch.setattr(aq, "synth_answer", forbidden)
    class Rag:
        async def aquery_data(self, intent, param):
            return {"status": "success", "data": evidence(), "metadata": {}}
    rec = await run_one(Rag(), {"qid": "empty", "intent": "query"}, {})
    assert "error" not in rec["meta"]
    assert rec["kb_coverage"] == "empty" and rec["cited_papers"] == []
    assert len(rec["data"]["chunks"]) == 4
    assert rec["query_receipt"]["retrieved_paper_count"] == 3


@pytest.mark.parametrize("path, key", [("paper/A", "A"), ("BareKey", "BareKey"),
    ("textbook:Book#s2", None), ("unknown_source", None)])
def test_eval_trace_uses_the_same_citation_identity_for_both_paper_eras(path, key):
    from papervault.eval.run_eval import _slim_chunk
    assert _slim_chunk({"file_path": path, "content": "evidence"})["paper_key"] == key
