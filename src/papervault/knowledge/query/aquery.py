"""Read-only retrieval and evidence-bound synthesis with a shared pure out-feed.

A query pins one catalog generation across retrieval and synthesis. Citations are
verified inline keys; retrieval candidates and observation counts remain separate.
FAST eval uses the same builder with an explicit no-synthesis citation fallback.
"""
from __future__ import annotations

import logging
import asyncio
import os
import re
import time
from collections import Counter
from copy import deepcopy
from typing import Any

from lightrag import QueryParam

from papervault.knowledge.query.synth import _source_label, annotate_sources, source_tier, synth_answer
from papervault.llm_routing import route

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
# Live retrieval variant for the shipped query() / MCP path. 'single' selects the
# single-query mix path. The default 'multiquery' routes through
# multiquery.retrieve_fused (V-MQ) — the #5-loop winner when paired with KS_SYNTH_STRICT_REFUSAL=1
# + the long-context env (MAX_TOTAL_TOKENS=300000, KS_MQ_ENABLE_RERANK=false, KS_CHUNK_TOP_K=60,
# KS_MQ_SUB_CHUNK_TOP_K=60, KS_MQ_N_SUBQ=5, KS_TOP_K=100).
_QUERY_VARIANT = os.getenv("KS_QUERY_VARIANT", "multiquery")  # #5 default-promote 2026-06-14 (was 'single'): V-MQ live

_EMPTY = {"answer": "(KB 无相关知识)", "cited_papers": [], "cited_sources": [],
          "abstract_only_papers": [], "kb_coverage": "empty"}

# Operator-supplied source classes (knowledge/ingest/operator_docs.py) carry the source
# in a COLON prefix (`textbook:AuthorYear`, `notebook:idea-scope`) that survives LightRAG
# 1.5's file_path basenaming (no slash to strip). These are NOT pl citation keys: they must
# stay OUT of cited_papers (a slash-free file_path would otherwise be read as a bare paper
# key) and surface in a SEPARATE cited_sources list. web: is anticipated by the synth prompt
# / SDD §12 credibility bands; keep it here so a future web upstream is handled consistently.
_SOURCE_PREFIXES = ("textbook:", "notebook:", "web:")


def _is_source_key(fp: str) -> bool:
    return any(fp.startswith(p) for p in _SOURCE_PREFIXES)


def _retrieved_keys(data: dict[str, Any], *, sources: bool = False) -> list[str]:
    """Deduplicate by evidence identity in retrieval rank, with references as fallback."""
    keys = []
    for item in [*(data.get("chunks") or []), *(data.get("references") or [])]:
        fp = item.get("file_path") or item.get("paper_key") or ""
        label, _ = _source_label(fp)
        if label == "unknown":
            continue
        if sources:
            valid = _is_source_key(fp)
        else:
            valid = fp.startswith("paper/") or ("/" not in fp and ":" not in fp)
        if valid and label not in keys:
            keys.append(label)
    return keys


def _cited_papers(data: dict[str, Any]) -> list[str]:
    """Retrieval-ranked paper candidates (the FAST no-synthesis citation fallback).

    Synthesized answers use build_outfeed to restrict this pool to verified inline keys.
    """
    return _retrieved_keys(data)


def _cited_sources(data: dict[str, Any]) -> list[str]:
    """Legacy sorted retrieval accessor using the shared out-feed identity logic.

    Actual cited sources are verified against inline citations by build_outfeed.
    """
    return sorted(_retrieved_keys(data, sources=True))


def _abstract_only_papers(data: dict[str, Any]) -> list[str]:
    """Cited paper keys whose served chunks come from an abstract-only doc (#144). Sorted.

    Read from data.chunks (the chunk text starts with the abstract-only header) with the same
    source-key rule the synth label uses, then kept only if the key is also in cited_papers."""
    cited = set(_cited_papers(data))
    keys = set()
    for c in data.get("chunks") or []:
        if source_tier(c) != "abstract_only":
            continue
        label, _cred = _source_label(c.get("file_path") or c.get("paper_key") or "")
        if label in cited:  # cited_papers already excludes operator sources (textbook:/…)
            keys.add(label)
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
      e >= 5 with exclusively abstract-only chunks → weak
      5 <= e < 20 with full-text evidence → thin
      e >= 20 with full-text evidence → strong
    cosine (still present in v3, threshold 0.2) is only an auxiliary signal; graph-signal
    coverage is the deliberate design choice (§6.4 注), not 'no cosine'.
    """
    chunks = [c for c in annotate_sources(data)["chunks"] if (c.get("content") or "").strip()]
    if not chunks:
        return "empty"
    pinfo = (metadata or {}).get("processing_info")
    if not pinfo:
        return "empty"
    e = pinfo.get("total_entities_found")
    if e is None or e < 5:
        return "empty"
    if all(source_tier(c) == "abstract_only" for c in chunks):
        return "weak"
    if e < 20:
        return "thin"
    return "strong"


def build_outfeed(
    data: dict[str, Any], metadata: dict[str, Any] | None, answer: str | None,
    *, no_synth: bool = False, generation_id: str | None = None,
    query_mode: str = _QUERY_MODE, variant: str = "single", rerank_degraded: bool = False,
    synthesis_model: str | None = None, synthesis_latency_s: float = 0.0,
    serving_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure production/eval contract over one captured retrieval and its answer.

    No-synthesis eval retains retrieval-ranked candidates as cited_papers for recall
    compatibility. All synthesized outputs admit only exact inline keys in that pool.
    The model records the requested routing level; the gateway's underlying model is
    undisclosed and is never inferred from configuration.
    """
    data = annotate_sources(data)
    coverage = _assess_coverage(metadata, data)
    papers = _retrieved_keys(data)
    sources = _retrieved_keys(data, sources=True)
    if coverage == "empty" and not no_synth:
        answer = _EMPTY["answer"]
    counts = Counter(re.findall(r"\[([^\[\]\n]+)\]", answer or ""))
    def cited(keys):
        if no_synth:
            return list(keys)
        return sorted((k for k in keys if counts[k]), key=lambda k: (-counts[k], keys.index(k)))
    cited_papers, cited_sources = cited(papers), cited(sources)
    abstract = set(_abstract_only_papers(data))
    fulltext = {_source_label(c.get("file_path") or c.get("paper_key") or "")[0] for c in data["chunks"]
                if source_tier(c) != "abstract_only"}
    receipt = {
        "generation_id": generation_id, "query_mode": query_mode, "variant": variant,
        "kb_coverage": coverage, "retrieved_chunk_count": len(data["chunks"]),
        "retrieved_paper_count": len(papers), "cited_paper_count": len(cited_papers),
        "rerank_degraded": rerank_degraded, "synthesis_model": synthesis_model,
        "synthesis_latency_s": synthesis_latency_s, "no_synth": no_synth,
    }
    snapshot = serving_manifest or {}
    manifest = snapshot.get("manifest") or {}
    receipt.update(generation_id=snapshot.get("generation_id", generation_id),
                   manifest_revision=snapshot.get("manifest_revision"),
                   indexed_scope=None, freshness=None)
    if serving_manifest is not None:
        selected = manifest.get("selected_unit_receipts", {})
        receipt["indexed_scope"] = sorted(selected)
        revisions = {entry["unit_id"]: entry["consumed_input_revision"]
                     for entry in manifest.get("entries", [])
                     if entry["unit_id"] in selected and entry.get("consumed_input_revision") is not None}
        if revisions:
            receipt["freshness"] = {"input_revisions": revisions}
        # Only declared read metadata leaves the pinned manifest, never internal
        # receipts, backend details or a newly selected generation's timestamps.
        for key in ("indexed_scope", "freshness", "indexed_at", "freshness_timestamp",
                    "cohort", "cohort_id", "requirements_revision"):
            if key in snapshot or key in manifest:
                receipt[key] = deepcopy(snapshot[key] if key in snapshot else manifest[key])
    return {
        "answer": answer, "cited_papers": cited_papers, "cited_sources": cited_sources,
        "retrieved_papers": papers, "retrieved_sources": sources,
        "abstract_only_papers": [k for k in cited_papers if k in abstract and k not in fulltext],
        "kb_coverage": coverage, "query_receipt": receipt,
    }


async def synthesize_outfeed(
    data: dict[str, Any], metadata: dict[str, Any] | None, intent: str,
    *, no_synth: bool = False, **receipt_fields: Any,
) -> dict[str, Any]:
    """Apply the coverage gate before synthesis, then assemble one observable out-feed."""
    data = annotate_sources(data)
    coverage = _assess_coverage(metadata, data)
    answer = None
    model, latency = None, 0.0
    if not no_synth and coverage != "empty":
        model = route("synth")
        started = time.monotonic()
        answer = await synth_answer(data, intent, kb_coverage=coverage)
        latency = time.monotonic() - started
    out = build_outfeed(data, metadata, answer, no_synth=no_synth,
                        synthesis_model=model, synthesis_latency_s=latency, **receipt_fields)
    logger.info("query_receipt %s", out["query_receipt"], extra={"query_receipt": out["query_receipt"]})
    return out


async def query(intent: str, rag: Any = None) -> dict[str, Any]:
    """Pin catalog selection once across all retrieval facets and synthesis."""
    owned = rag is None
    if owned:
        from papervault.knowledge.store.catalog_serving import get_query_graph

        rag = await get_query_graph()
    try:
        if hasattr(rag, "validate"):
            await asyncio.to_thread(rag.validate)
        result = await _query(intent, rag)
        if hasattr(rag, "validate"):
            await asyncio.to_thread(rag.validate)
        return result
    finally:
        if owned and hasattr(rag, "close"):
            await rag.close()


async def _query(intent: str, rag: Any) -> dict[str, Any]:
    """Retrieve and synthesize over one already pinned store view."""
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
        # baseline single-query data when the coverage gate rejects fusion or decomposition fails. Lazy import avoids the aquery↔multiquery
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
            data = {}
        else:
            data = res["data"]
        metadata = res.get("metadata") if isinstance(res, dict) else None

    degraded = bool(_rr_count is not None and _rr0 is not None and _rr_count() > _rr0)
    if degraded:
        logger.warning("rerank DEGRADED during query intent=%r", intent[:160])
    return await synthesize_outfeed(
        data, metadata, intent, variant=_QUERY_VARIANT, rerank_degraded=degraded,
        generation_id=getattr(rag, "snapshot", {}).get("generation_id"),
        serving_manifest=getattr(rag, "snapshot", None),
    )
