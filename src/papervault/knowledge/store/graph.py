"""LightRAG graph instance (v3, SDD §6.0) — the full PROVEN config from
experiments/lightrag_l0/run.py:

- Neo4j graph + Postgres KV/vector/doc_status backends
- research entity-type ontology (drops Person/Organization → author/journal/cite
  strings don't become nodes; the verified noise source)
- ALL THREE concurrency knobs that actually feed 24-way (§6.0/§7): llm_model_max_async
  (LLM ceiling) + max_parallel_insert (docs in flight; default 2 is the hidden
  bottleneck) + embedding_func_max_async (default 8 = hidden embedding bottleneck).
- enable_llm_cache_for_entity_extract (deletion-rebuild reuses cached extraction, §6.1)
- force_llm_summary_on_merge (shared-entity re-summary threshold)

Workspace isolation (SDD §6.0 prod-safety) is driven by the env vars
NEO4J_WORKSPACE + POSTGRES_WORKSPACE (NOT LightRAG(workspace=), which only names
the in-process pipeline_status). Each var independently governs its storage layer;
an empty NEO4J_WORKSPACE falls back to 'base' and an empty POSTGRES_WORKSPACE to
'default' — never 'l0'. Prod = 'l0'; dev/probe/test = 'l0_probe'.
⚠️ The repo .env currently DEFAULTS both vars to prod 'l0' (flipping it to 'l0_probe'
is a user-pending OPEN item, SDD §6.0 line 182) — so a bare run with no env override
points at PROD. assert_safe_workspace() turns that into an explicit refusal on guarded
paths instead of a silent prod write; dev/probe/test must set 'l0_probe' explicitly.

assert_safe_workspace() (called by get_graph before instance creation) sinks the
prod-safety guard into the data-mutable layer: it requires BOTH env vars equal and
non-empty, expects 'l0_probe' by default, and refuses to touch prod 'l0' unless
KS_ALLOW_PROD_WORKSPACE=1 is set explicitly. So distill_batch / remove_one / the S4
loop / the probe / tests all inherit ONE gate instead of re-copying the probe's `if`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

from papervault import config as _pv
from papervault.domain import get_domain
from papervault.knowledge.config import CONFIG
from papervault.knowledge.ingest.chunking import chunking_by_sentence_boundary
from papervault.knowledge.store.extraction_prompt import apply_ks_extraction_prompt
from papervault.knowledge.store.lightrag_init import _bge_embed, _bge_rerank
from papervault.knowledge.store.llm import mimo_complete

log = logging.getLogger("ks.store.graph")

PROD_WORKSPACE = "l0"
DEV_WORKSPACE = "l0_probe"
_ALLOW_PROD_ENV = "KS_ALLOW_PROD_WORKSPACE"


class UnsafeWorkspaceError(RuntimeError):
    """Raised when the resolved workspace would touch prod 'l0' without explicit opt-in,
    or when NEO4J_WORKSPACE / POSTGRES_WORKSPACE disagree / are empty (half-isolation)."""


def assert_safe_workspace() -> str:
    """Structural prod-safety gate (SDD §6.0/§6.5). Returns the resolved workspace.

    Isolation rests on TWO independent env vars (NEO4J_WORKSPACE drives Neo4j,
    POSTGRES_WORKSPACE drives the PG KV/vector/doc_status). Reject:
      - either empty (a missing var silently falls back to 'base'/'default' next to prod),
      - the two disagreeing (half the data lands in a different namespace),
      - prod 'l0' unless KS_ALLOW_PROD_WORKSPACE=1 is set on purpose.
    """
    neo = os.environ.get("NEO4J_WORKSPACE", "").strip()
    pg = os.environ.get("POSTGRES_WORKSPACE", "").strip()
    allow_prod = os.environ.get(_ALLOW_PROD_ENV, "").strip() == "1"

    if not neo or not pg:
        raise UnsafeWorkspaceError(
            f"workspace env incomplete: NEO4J_WORKSPACE={neo!r} POSTGRES_WORKSPACE={pg!r}. "
            "Both must be set (a missing one silently falls back to base/default next to prod)."
        )
    if neo != pg:
        raise UnsafeWorkspaceError(
            f"workspace env mismatch: NEO4J_WORKSPACE={neo!r} != POSTGRES_WORKSPACE={pg!r}. "
            "Graph and KV/vector/doc_status would split across namespaces."
        )
    if neo == PROD_WORKSPACE and not allow_prod:
        raise UnsafeWorkspaceError(
            f"refusing to open prod workspace {PROD_WORKSPACE!r}: set {_ALLOW_PROD_ENV}=1 to "
            f"opt in on purpose (dev/probe/test should use {DEV_WORKSPACE!r})."
        )
    log.info("workspace=%s (prod_opt_in=%s)", neo, allow_prod)
    return neo

# 研究本体(SDD §6.0 / run.py:55):丢 Person/Organization,作者/期刊/引文串不进图。
# Externalized to the active domain pack (ADR-0003, papervault.domain); the bundled
# space_physics factory pack reproduces the 11-type list verbatim.
ENTITY_TYPES = get_domain().entity_types

_RAG: Optional[LightRAG] = None


async def get_graph() -> LightRAG:
    """Singleton LightRAG (v3 full config). Workspace via NEO4J_WORKSPACE/POSTGRES_WORKSPACE env;
    gated by assert_safe_workspace() (refuses prod 'l0' without KS_ALLOW_PROD_WORKSPACE=1)."""
    global _RAG
    if _RAG is not None:
        return _RAG

    assert_safe_workspace()  # SDD §6.0/§6.5 — structural prod-safety, before any instance/IO

    working_dir = CONFIG.lightrag.working_dir
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    # V2 (SDD §6.0/§6.9.2): mutate module-level lightrag.prompt.PROMPTS BEFORE the LightRAG
    # ctor — operate.py reads PROMPTS at EXTRACTION time, so the space-physics few-shot
    # examples + negative exclusions must be in place before any ainsert. Idempotent (own
    # sentinel) and build-time-only (takes effect only on the next BUILD, not queries).
    apply_ks_extraction_prompt(examples=True, exclusions=True)

    # Build-plane LLM (2026-07-16, user call): LightRAG-internal calls — entity/relation
    # extraction at ainsert (the token sink: 2 calls/chunk × ~10-13k tok) plus its small
    # query-path keyword extraction — run on the CHEAP deployment with thinking ON; pro
    # was overkill-priced for build-scale extraction. Research-plane calls (decompose +
    # synth in query/multiquery.py + query/synth.py) call mimo_complete directly and stay
    # on the pool default (mimo-v2.5-pro). Env-revertable without code:
    #   KS_BUILD_MODEL=mimo-v2.5-pro  → revert model
    #   KS_BUILD_THINKING=0           → stop forwarding enable_thinking
    #   KS_KW_THINKING=0              → enable_thinking=False for the query-path
    #       keyword-extraction call ONLY (LightRAG tags it keyword_extraction=True,
    #       operate.py:3465). It is a mechanical task measured burning ~67s of reasoning
    #       CoT (2026-07-16). Default 1 = today's behavior; the knob exists so the FAST
    #       recall A/B can arbitrate before any flip (issue #3).
    async def _build_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        kwargs.setdefault("model", os.getenv("KS_BUILD_MODEL") or _pv.BUILD_MODEL)
        if kwargs.get("keyword_extraction") and os.getenv("KS_KW_THINKING", "1") == "0":
            kwargs.setdefault("enable_thinking", False)
        elif os.getenv("KS_BUILD_THINKING", "1") == "1":
            kwargs.setdefault("enable_thinking", True)
        return await mimo_complete(
            prompt, system_prompt=system_prompt, history_messages=history_messages, **kwargs
        )

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=_build_llm,
        llm_model_name=os.getenv("KS_BUILD_MODEL") or _pv.BUILD_MODEL,
        llm_model_max_async=int(os.getenv("KS_LLM_MAX_ASYNC", "32")),        # 硬 LLM 天花板。默认 32(2026-06-04):6-key 时代 conc16 把 2400-chunk 抽取拖过 480s worker(59/100 error)→曾锁 8;补到 24 key 后 conc_probe 实测 conc12/24/36/48 全 0 error、max ≤83s«480s(争抢消失,~1.3 调用/key)。env 可调,应大致随活 key 数走(§7)
        max_parallel_insert=int(os.getenv("KS_MAX_PARALLEL_INSERT", "16")),  # 文档在飞数(默认 2 = 隐藏瓶颈)
        embedding_func_max_async=int(os.getenv("KS_EMBED_MAX_ASYNC", "16")), # embedding 并发(默认 8 = 隐藏瓶颈);三旋钮缺一不可(§7)
        chunk_token_size=CONFIG.lightrag.chunk_token_size,
        chunk_overlap_token_size=CONFIG.lightrag.chunk_overlap_token_size,
        # Boundary-aware chunker (ingest.chunking): same token target as the default
        # token splitter (now 2400 via CONFIG.lightrag.chunk_token_size), but a chunk
        # never ends mid-sentence (pure hygiene — only the cut location moves, not the
        # size). Drop-in for LightRAG's default chunking_by_token_size; uses the
        # tokenizer LightRAG passes in.
        chunking_func=chunking_by_sentence_boundary,
        entity_extract_max_gleaning=1,
        # Per-LLM-call timeout. LightRAG derives the extraction-WORKER timeout from this as
        # 2× (utils.limit_async_func_call): worker kills a chunk's extraction task at 2×this.
        # At the 2400-token chunk size a reasoning-model extraction (initial + 1 gleaning) is
        # ~80-110s alone but balloons under concurrency contention, so the worker cap must
        # leave headroom — env-tunable (KS_LLM_TIMEOUT) so a build can raise it without a code
        # change. 240 → 480s worker (default); a high-concurrency build should raise both this
        # and lower KS_LLM_MAX_ASYNC. See experiments/retry_test100.py.
        default_llm_timeout=int(os.getenv("KS_LLM_TIMEOUT", "240")),
        enable_llm_cache_for_entity_extract=True,
        force_llm_summary_on_merge=8,
        addon_params={"language": "English", "entity_types": ENTITY_TYPES},
        graph_storage="Neo4JStorage",
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        doc_status_storage="PGDocStatusStorage",
        embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=_bge_embed),
        # V1 (SDD §6.9.1): wire the bge-reranker-v2-m3 rerank path. Without rerank_model_func,
        # LightRAG's enable_rerank=True default (base.py:160) silently no-ops (F18). lazy-loaded.
        rerank_model_func=_bge_rerank,
        # Pin 0.0 explicitly: rerank only REORDERS chunks, never drops by absolute score
        # (process_chunks_unified filters only when min_rerank_score > 0.0; F20). Default is
        # already 0.0 but the env MIN_RERANK_SCORE could drift it → kill niche-query chunks.
        min_rerank_score=0.0,
    )
    await rag.initialize_storages()
    from lightrag.kg.shared_storage import initialize_pipeline_status

    await initialize_pipeline_status()
    _RAG = rag
    return _RAG


async def close_graph() -> None:
    global _RAG
    _RAG = None
