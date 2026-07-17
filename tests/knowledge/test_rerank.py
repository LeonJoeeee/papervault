"""V1 reranker contract + wiring tests (SDD §6.9.1). No GPU, no model download, no DB.

`_bge_rerank` is exercised with `_get_bge_reranker` monkeypatched to a fake CrossEncoder, so
these run offline. We verify the EXACT contract LightRAG's apply_rerank_if_enabled
(utils.py:2671-2752, lightrag 1.4.16) depends on:
  - called with keyword args query= / documents= / top_n=
  - returns list[{"index": int, "relevance_score": float}] with index → original doc position
  - results sorted by descending score, truncated to top_n
  - empty documents → [] (LightRAG then keeps original chunks)
Backend is sentence-transformers CrossEncoder (transformers-5-native; NOT FlagReranker, which
crashes on transformers 5.x — SDD §6.9.1 drill fix). The fake mirrors CrossEncoder.predict
(returns a numpy array of per-pair scores, default Sigmoid → already 0..1).
Plus the wiring: graph.get_graph() passes rerank_model_func + min_rerank_score=0.0 and calls
apply_ks_extraction_prompt BEFORE the ctor; aquery widens the QueryParam.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from papervault.knowledge.store import lightrag_init


class _FakeReranker:
    """Stand-in for CrossEncoder: returns a fixed score per (query, doc) pair, by doc order."""

    def __init__(self, scores):
        self._scores = scores
        self.calls = []

    def predict(self, pairs, batch_size=32, **kwargs):
        # batch_size accepted because the hardened _bge_rerank now calls predict(pairs,
        # batch_size=bs) via _predict_with_oom_retry (OOM-survival shrink 32→8→2→1).
        self.calls.append(pairs)
        # mirror CrossEncoder.predict: numpy array of per-pair scores, by pair order
        return np.array([self._scores[i] for i in range(len(pairs))], dtype=np.float32)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def _stub_reranker(monkeypatch):
    """Install a fake reranker so _bge_rerank never touches FlagEmbedding / a GPU."""

    def _make(scores):
        fake = _FakeReranker(scores)
        monkeypatch.setattr(lightrag_init, "_get_bge_reranker", lambda: fake)
        return fake

    return _make


# ---- _bge_rerank contract ---------------------------------------------------

def test_rerank_returns_index_score_dicts_sorted_desc(_stub_reranker):
    docs = ["d0", "d1", "d2"]
    _stub_reranker([0.2, 0.9, 0.5])  # by doc index
    out = _run(lightrag_init._bge_rerank(query="q", documents=docs, top_n=None))

    # shape: list of {"index", "relevance_score"}
    assert all(set(r.keys()) == {"index", "relevance_score"} for r in out)
    assert all(isinstance(r["index"], int) for r in out)
    assert all(isinstance(r["relevance_score"], float) for r in out)
    # sorted by descending score → d1 (0.9), d2 (0.5), d0 (0.2)
    assert [r["index"] for r in out] == [1, 2, 0]
    # float32 (CrossEncoder.predict) → approx; values + order are what matter
    assert [r["relevance_score"] for r in out] == pytest.approx([0.9, 0.5, 0.2])


def test_rerank_respects_top_n(_stub_reranker):
    docs = ["d0", "d1", "d2", "d3"]
    _stub_reranker([0.1, 0.4, 0.3, 0.95])
    out = _run(lightrag_init._bge_rerank(query="q", documents=docs, top_n=2))
    assert [r["index"] for r in out] == [3, 1]  # top-2 by score
    assert len(out) == 2


def test_rerank_index_points_into_original_documents(_stub_reranker):
    # The index must address retrieved_docs[index] in LightRAG; verify it is a valid
    # original position, not a rank position.
    docs = ["alpha", "beta", "gamma"]
    _stub_reranker([0.0, 1.0, 0.5])
    out = _run(lightrag_init._bge_rerank(query="q", documents=docs, top_n=None))
    for r in out:
        assert 0 <= r["index"] < len(docs)
    # top result's index addresses "beta"
    assert docs[out[0]["index"]] == "beta"


def test_rerank_empty_documents_returns_empty(_stub_reranker):
    fake = _stub_reranker([])
    out = _run(lightrag_init._bge_rerank(query="q", documents=[], top_n=5))
    assert out == []
    # short-circuits before touching the model
    assert fake.calls == []


def test_rerank_builds_query_doc_pairs(_stub_reranker):
    docs = ["d0", "d1"]
    fake = _stub_reranker([0.3, 0.7])
    _run(lightrag_init._bge_rerank(query="qq", documents=docs, top_n=None))
    # CrossEncoder.predict is fed (query, doc) pairs in original doc order; its default
    # Sigmoid activation yields 0..1 scores (no separate normalize kwarg — SDD §6.9.1).
    assert fake.calls[0] == [("qq", "d0"), ("qq", "d1")]


def test_rerank_single_pair_scores_one_doc(_stub_reranker):
    # CrossEncoder.predict on one pair returns a length-1 array; _bge_rerank indexes it fine.
    _stub_reranker([0.42])
    out = _run(lightrag_init._bge_rerank(query="q", documents=["only"], top_n=None))
    assert out == [{"index": 0, "relevance_score": pytest.approx(0.42)}]


def test_rerank_top_n_zero_or_none_keeps_all(_stub_reranker):
    docs = ["d0", "d1", "d2"]
    _stub_reranker([0.1, 0.2, 0.3])
    # top_n=None → all
    assert len(_run(lightrag_init._bge_rerank(query="q", documents=docs, top_n=None))) == 3
    # top_n=0 is falsy → no truncation (keep all), matching SDD `if top_n:`
    assert len(_run(lightrag_init._bge_rerank(query="q", documents=docs, top_n=0))) == 3


def test_rerank_func_is_async():
    # LightRAG awaits rerank_model_func(...); it MUST be a coroutine function.
    assert asyncio.iscoroutinefunction(lightrag_init._bge_rerank)


# ---- V1 wiring: aquery QueryParam -------------------------------------------

def test_aquery_query_param_widened(monkeypatch):
    """The SINGLE-query path builds QueryParam(mode='mix', top_k=100, chunk_top_k=60,
    enable_rerank=True) — the #5 default-promote values (was 40/12). Force the single path
    (default is now 'multiquery') so this asserts the single-query QueryParam, not the V-MQ route."""
    import papervault.knowledge.query.aquery as _aq
    monkeypatch.setattr(_aq, "_QUERY_VARIANT", "single")
    captured = {}

    class _CapturingRag:
        async def aquery_data(self, intent, param):  # noqa: ARG002
            captured["param"] = param
            return {"status": "failure", "data": {}, "metadata": {}}

    _run(_aq.query("some intent", rag=_CapturingRag()))

    p = captured["param"]
    assert p.mode == "mix"
    assert p.top_k == 100
    assert p.chunk_top_k == 60
    assert p.enable_rerank is True


# ---- V1 + V2 wiring: graph.get_graph() --------------------------------------

def test_get_graph_wires_rerank_and_prompt_before_ctor(monkeypatch):
    """get_graph() must (V2) call apply_ks_extraction_prompt BEFORE building LightRAG, and
    (V1) pass rerank_model_func + min_rerank_score=0.0 into the ctor.

    Everything heavy (assert_safe_workspace, LightRAG ctor, storage init) is stubbed so this
    runs with no DB / no model. We assert ordering via a shared event log.
    """
    from papervault.knowledge.store import graph as graph_mod

    events = []
    monkeypatch.setattr(graph_mod, "_RAG", None)
    monkeypatch.setattr(graph_mod, "assert_safe_workspace", lambda: "l0_probe")
    monkeypatch.setattr(
        graph_mod, "apply_ks_extraction_prompt", lambda **kw: events.append(("prompt", kw))
    )

    captured_ctor = {}

    class _FakeRAG:
        def __init__(self, **kwargs):
            events.append(("ctor", None))
            captured_ctor.update(kwargs)

        async def initialize_storages(self):
            events.append(("init_storages", None))

    monkeypatch.setattr(graph_mod, "LightRAG", _FakeRAG)
    # initialize_pipeline_status is imported inside get_graph from shared_storage
    import lightrag.kg.shared_storage as shared

    async def _fake_pipeline_status():
        events.append(("pipeline_status", None))

    monkeypatch.setattr(shared, "initialize_pipeline_status", _fake_pipeline_status)

    rag = _run(graph_mod.get_graph())
    assert isinstance(rag, _FakeRAG)

    # V2: prompt mutation happened, with the right flags, BEFORE the ctor.
    assert ("prompt", {"examples": True, "exclusions": True}) in events
    prompt_idx = events.index(("prompt", {"examples": True, "exclusions": True}))
    ctor_idx = events.index(("ctor", None))
    assert prompt_idx < ctor_idx

    # V1: ctor received the rerank func + explicit 0.0 floor.
    assert captured_ctor["rerank_model_func"] is graph_mod._bge_rerank
    assert captured_ctor["min_rerank_score"] == 0.0

    # cleanup the module singleton we just populated
    monkeypatch.setattr(graph_mod, "_RAG", None)


# ---- V2 format-safety: the REAL mutated PROMPTS must survive operate.py's two-stage .format()

def test_extraction_prompt_real_mutation_is_format_safe():
    """SDD §6.9.2 line 492 declares '.format-safe (only LightRAG placeholders, no bare {}/})'
    a HARD constraint. The wiring test above monkeypatches apply_ks_extraction_prompt to a
    lambda (it checks CALL ORDERING only, never the real mutation) — so nothing else exercises
    the actual content. This replicates operate.py's exact two-stage .format() (operate.py
    2909-2926, 2951-2960, lightrag 1.4.16) against the REAL mutated PROMPTS and asserts it does
    not raise: a future edit that introduces a stray '{' in an example or the exclusions block
    would crash the next BUILD with KeyError/ValueError — this catches it cheaply (no DB/model).
    """
    import copy

    from lightrag import prompt as lr_prompt

    from papervault.knowledge.store.extraction_prompt import apply_ks_extraction_prompt
    from papervault.knowledge.store.graph import ENTITY_TYPES

    # Snapshot + restore: this mutates the module-level lightrag.prompt.PROMPTS dict in place.
    saved_examples = copy.deepcopy(lr_prompt.PROMPTS["entity_extraction_examples"])
    saved_sysp = lr_prompt.PROMPTS["entity_extraction_system_prompt"]
    try:
        apply_ks_extraction_prompt(examples=True, exclusions=True)

        # --- Stage 1: examples joined then .format()'d (operate.py:2909-2918) ---
        examples = "\n".join(lr_prompt.PROMPTS["entity_extraction_examples"])
        example_context_base = dict(
            tuple_delimiter=lr_prompt.PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            completion_delimiter=lr_prompt.PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=", ".join(ENTITY_TYPES),
            language="English",
        )
        examples = examples.format(**example_context_base)  # raises KeyError/ValueError on a stray brace

        # --- Stage 2: system + user + continue prompts .format()'d (operate.py:2920-2960) ---
        context_base = dict(
            tuple_delimiter=lr_prompt.PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            completion_delimiter=lr_prompt.PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=",".join(ENTITY_TYPES),
            examples=examples,
            language="English",
        )
        sysp = lr_prompt.PROMPTS["entity_extraction_system_prompt"].format(**context_base)
        user = lr_prompt.PROMPTS["entity_extraction_user_prompt"].format(
            **{**context_base, "input_text": "BODY"}
        )
        cont = lr_prompt.PROMPTS["entity_continue_extraction_user_prompt"].format(
            **{**context_base, "input_text": "BODY"}
        )

        # placeholders were actually substituted (no literal {tuple_delimiter}/{examples}/… left)
        for rendered in (examples, sysp, user, cont):
            assert "{tuple_delimiter}" not in rendered
            assert "{completion_delimiter}" not in rendered
            assert "{entity_types}" not in rendered
            assert "{language}" not in rendered
        assert "{examples}" not in sysp
        assert lr_prompt.PROMPTS["DEFAULT_TUPLE_DELIMITER"] in examples  # delimiter really landed

        # V2 took effect: stock fiction/finance examples GONE, space-physics examples PRESENT.
        for stock in ("Alex", "Nexon Technologies", "Noah Carter"):
            assert stock not in examples
        for ours in ("Solar Energetic Particles", "Physics-Informed Neural Network", "Galactic Cosmic Rays"):
            assert ours in examples
        # V3 exclusions block reached the system prompt.
        assert "Exclusions (do NOT extract" in sysp
    finally:
        lr_prompt.PROMPTS["entity_extraction_examples"] = saved_examples
        lr_prompt.PROMPTS["entity_extraction_system_prompt"] = saved_sysp


def test_extraction_prompt_idempotent():
    """SDD §6.9.2: apply is idempotent (own sentinel). get_graph() builds the singleton once,
    but a repeated call must not stack a 2nd exclusions block or a 4th example.
    """
    import copy

    from lightrag import prompt as lr_prompt

    from papervault.knowledge.store.extraction_prompt import apply_ks_extraction_prompt

    saved_examples = copy.deepcopy(lr_prompt.PROMPTS["entity_extraction_examples"])
    saved_sysp = lr_prompt.PROMPTS["entity_extraction_system_prompt"]
    try:
        apply_ks_extraction_prompt(examples=True, exclusions=True)
        apply_ks_extraction_prompt(examples=True, exclusions=True)
        apply_ks_extraction_prompt(examples=True, exclusions=True)
        sysp = lr_prompt.PROMPTS["entity_extraction_system_prompt"]
        assert sysp.count("Exclusions (do NOT extract") == 1
        assert sysp.count("Acronym canonicalization") == 1
        assert len(lr_prompt.PROMPTS["entity_extraction_examples"]) == 3
    finally:
        lr_prompt.PROMPTS["entity_extraction_examples"] = saved_examples
        lr_prompt.PROMPTS["entity_extraction_system_prompt"] = saved_sysp
