"""S3 query out-feed (SDD §5.4 / §6.4) — the v1 `query(intent)` entrypoint.

Pipeline (verified against LightRAG 1.4.16 source):
  res = await rag.aquery_data(intent,
            QueryParam(mode='mix', top_k=100, chunk_top_k=60, enable_rerank=True))  # #5 default 2026-06-14
    → plain dict, NO LLM generation:
      success: {status:'success', message, data:{entities,relationships,chunks,references},
                metadata:{query_mode, keywords, processing_info}}
      failure: {status:'failure', message, data:{}, metadata:{failure_reason, mode}}  # no processing_info
  guard status != 'success' or empty data → empty out-feed
  answer       = KS-written synth prose over data.{entities,relationships,chunks}
  cited_papers = strip 'paper/' from data.references[].file_path (chunk-level, deduped,
                 single-valued, unknown_source filtered — NOT entity-level which is
                 FIFO-truncated 100/300; SDD §4.3 F12). NEVER regex-scraped from prose.
  kb_coverage  = assess(metadata.processing_info.total_entities_found); None-safe (empty
                 on failure/empty where processing_info is absent — SDD §6.4).

mode='mix' (graph + vector): naive clears entities/relationships → cited can't aggregate
(F11). top_k=100 / chunk_top_k=60 / enable_rerank=True (#5 default-promote 2026-06-14; was 40/12)
per §6.4 / §6.9.1 (wide recall → bge rerank → 60 chunks to synth; the rerank_model_func is wired in
store/graph.get_graph()). NOTE: the live default variant is now 'multiquery' (retrieve_fused), not
this single-query path; both share these top_k/chunk_top_k constants.

`_cited_papers` and `_assess_coverage` are pure functions over an aquery_data dict so they
unit-test offline against canned dicts with no LightRAG running (tests/test_query.py).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from lightrag import QueryParam

from papervault.knowledge.query.synth import synth_answer

logger = logging.getLogger("ks.query.aquery")

_QUERY_MODE = "mix"   # SDD §6.4: graph+vector. naive → entities/relationships empty (F11).
# V1 wide-recall→rerank (SDD §6.9.1): top_k 20→40 widens the graph entity/relation candidate
# pool; chunk_top_k=12 is now PASSED explicitly (was unset → inherited env CHUNK_TOP_K=20,
# base.py:112) = the post-rerank chunk cap handed to synth. enable_rerank=True pins intent
# (already the LightRAG default). chunk_top_k is dual-use in process_chunks_unified: it is
# both the reranker's top_n (how many recalled chunks reach synth) AND the post-rerank hard
# cut. kb_coverage is unaffected — it reads the graph signal total_entities_found, and the
# reranker only touches chunks, not entities/relations (§6.9.1 F19).
_TOP_K = int(os.getenv("KS_TOP_K", "100"))   # #5 default-promote 2026-06-14 (was 40): lctx@60 sweet spot
# env-tunable for the #5 optimization loop (default now 60 = the lctx@60 promote 2026-06-14; was 12).
# The fused-chunk cap is HARD-BINDING on the V-MQ path (multiquery.py reads aquery._CHUNK_TOP_K as
# the fused top cap); raising it surfaces more DISTINCT papers into the served list. The eval scores
# @12-distinct over the FULL served list (backbone keeps first-12-distinct, top_n=None) so the
# headline budget stays fair regardless of this cap.
_CHUNK_TOP_K = int(os.getenv("KS_CHUNK_TOP_K", "60"))   # #5 default-promote 2026-06-14 (was 12): the unimodal peak
_ENABLE_RERANK = True
# Live retrieval variant for the shipped query() / MCP path. Default 'single' = the original
# single-query mix path (byte-identical when unset). 'multiquery' routes through
# multiquery.retrieve_fused (V-MQ) — the #5-loop winner when paired with KS_SYNTH_STRICT_REFUSAL=1
# + the long-context env (MAX_TOTAL_TOKENS=300000, KS_MQ_ENABLE_RERANK=false, KS_CHUNK_TOP_K=60,
# KS_MQ_SUB_CHUNK_TOP_K=60, KS_MQ_N_SUBQ=5, KS_TOP_K=100). Flag-gated deploy, never auto-on.
_QUERY_VARIANT = os.getenv("KS_QUERY_VARIANT", "multiquery")  # #5 default-promote 2026-06-14 (was 'single'): V-MQ live

_EMPTY = {"answer": "(KB 无相关知识)", "cited_papers": [], "kb_coverage": "empty"}


def _cited_papers(data: dict[str, Any]) -> list[str]:
    """Aggregate citation keys from data.references[].file_path (SDD §6.4 / §4.3).

    references[] = chunk-level, deduped, single-valued file_path, unknown_source already
    filtered out by LightRAG (utils.py generate_reference_list_from_chunks). We keep only
    the 'paper/' prefix ones and strip the prefix → pl citation key. Sorted + deduped.
    """
    refs = data.get("references") or []
    keys = set()
    for r in refs:
        fp = r.get("file_path") or ""
        if fp.startswith("paper/"):
            key = fp.split("/", 1)[1]
            if key:
                keys.add(key)
    return sorted(keys)


def _assess_coverage(metadata: dict[str, Any] | None, data: dict[str, Any]) -> str:
    """kb_coverage from graph signal (SDD §6.4). None-safe.

    e = metadata.processing_info.total_entities_found = len(final_entities), the graph-search
    entity count (LightRAG operate.py:4337). NOT a pre-truncation count: LightRAG exposes a
    SEPARATE entities_after_truncation = len(filtered_entities) (operate.py:4339) for the
    post-truncation figure. We deliberately bin on the (pre-truncation) graph-search count as a
    genuine graph-richness signal; the <5/<20/>=20 bins are KS's own design over that count.
    Failure/empty responses carry no processing_info → e is None → empty.
      e is None or chunks empty → empty
      e < 5  → empty
      5 <= e < 20 → thin
      e >= 20 → strong
    cosine (still present in v3, threshold 0.2) is only an auxiliary signal; graph-signal
    coverage is the deliberate design choice (§6.4 注), not 'no cosine'.
    """
    if not (data.get("chunks") or []):
        return "empty"
    pinfo = (metadata or {}).get("processing_info")
    if not pinfo:
        return "empty"
    e = pinfo.get("total_entities_found")
    if e is None or e < 5:
        return "empty"
    if e < 20:
        return "thin"
    return "strong"


async def query(intent: str, rag: Any = None) -> dict[str, Any]:
    """v1 out-feed (§5.4): query(intent) → {answer, cited_papers, kb_coverage}.

    `rag` is the LightRAG singleton; if None it's resolved via store.graph.get_graph()
    (workspace-gated). Passing it in lets callers (and the MCP server) share one gated
    instance with the ingest loop.
    """
    if rag is None:
        from papervault.knowledge.store.graph import get_graph

        rag = await get_graph()

    # #29: surface reranker degradation on the LIVE path (run_eval already does, via rec["meta"]).
    # The rerank hook (store/lightrag_init) increments a process-global counter + logs LOUD on each
    # OOM/failure, then silently falls back to original-order chunks — so at the data layer a
    # degraded query looks identical to a healthy one. Snapshot the counter around retrieval and log
    # LOUD if it advanced, so an operator watching the service sees THIS query got degraded retrieval
    # (more likely now the live reranker runs at max_length=4096). We deliberately do NOT reset the
    # counter: run_eval can (it runs at concurrency=1), but the live service serves queries
    # concurrently, so a reset would clobber a sibling query's count. A monotonic before/after delta
    # OVER-reports under concurrency (a sibling's OOM can bump this query's delta) but NEVER
    # under-reports — the safe side, since a shared-GPU reranker OOM degrades every concurrent query.
    try:
        from papervault.knowledge.store.lightrag_init import rerank_failure_count as _rr_count
    except Exception:  # noqa: BLE001 — observability must never break the query path
        _rr_count = None
    _rr0 = _rr_count() if _rr_count else None

    if _QUERY_VARIANT == "multiquery":
        # V-MQ live path (#5 winner). retrieve_fused gates on coverage and falls back to the
        # baseline single-query data when coverage is thin / decomposition fails, so this is safe:
        # a thin query behaves exactly like the single path. Lazy import avoids the aquery↔multiquery
        # import cycle (multiquery imports this module as `aq`).
        from papervault.knowledge.query.multiquery import retrieve_fused

        data, metadata, _fused = await retrieve_fused(intent, rag)
    else:
        res = await rag.aquery_data(
            intent,
            QueryParam(
                mode=_QUERY_MODE,
                top_k=_TOP_K,
                chunk_top_k=_CHUNK_TOP_K,
                enable_rerank=_ENABLE_RERANK,
                # A/B-fairness fix (bug-hunt 2026-06-15): without this the single-query branch
                # inherits LightRAG's DEFAULT_MAX_TOTAL_TOKENS=30000 and silently truncates the
                # requested chunk_top_k=60 down to ~6-12 served chunks — the exact 30k cap the lctx
                # work lifted ONLY on the multiquery path. Share the lifted cap so a KS_QUERY_VARIANT
                # =single baseline is measured under the same budget as the live multiquery default.
                max_total_tokens=int(os.getenv("KS_MAX_TOTAL_TOKENS", "300000")),
            ),
        )
        if not isinstance(res, dict) or res.get("status") != "success" or not res.get("data"):
            return dict(_EMPTY)
        data = res["data"]
        metadata = res.get("metadata")

    if _rr_count is not None and _rr0 is not None and _rr_count() > _rr0:
        logger.warning(
            "rerank DEGRADED during query — reranker OOM/failure fell back to original-order "
            "chunks, so retrieval quality is reduced for this answer. intent=%r", intent[:160]
        )

    # Nothing retrieved (status success but all arrays empty — e.g. a query whose
    # keywords matched no graph/vector content): skip the LLM round-trip, return empty.
    if not (data.get("entities") or data.get("relationships") or data.get("chunks")):
        return dict(_EMPTY)

    answer = await synth_answer(data, intent)
    cited_papers = _cited_papers(data)
    kb_coverage = _assess_coverage(metadata, data)

    return {"answer": answer, "cited_papers": cited_papers, "kb_coverage": kb_coverage}
