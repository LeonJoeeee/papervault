"""BGE-M3 embedding + reranker funcs for the v3 LightRAG instance (SDD §6.0 / §6.9.1).

`_bge_embed` (+ its lazy model singleton `_get_bge_model`) is the embedding_func wired into
store/graph.get_graph()'s LightRAG ctor and the experiments baseline run.py.

`_bge_rerank` (+ its lazy singleton `_get_bge_reranker`, V1 / SDD §6.9.1) is the
rerank_model_func wired into the same ctor. It loads `bge-reranker-v2-m3` (same BAAI family
as BGE-M3, in-process on KS's pinned RTX 3090 via device cuda:0 + CUDA_VISIBLE_DEVICES=1) lazily, mirroring `_bge_embed`'s
lazy-singleton + in-process serving shape. WITHOUT it, LightRAG 1.4's `enable_rerank=True`
default (base.py:160) silently no-ops: `apply_rerank_if_enabled` (utils.py:2671) warns
"no rerank model is configured" and returns chunks unranked (SDD §11 F18).

★ Backend = `sentence_transformers.CrossEncoder`, NOT `FlagEmbedding.FlagReranker`
(drill fix 2026-06-02, SDD §6.9.1). This env runs transformers 5.9; FlagEmbedding 1.4.0's
compute_score_single_gpu (inference/reranker/encoder_only/base.py:147) calls
`tokenizer.prepare_for_model(...)`, removed from fast tokenizers in transformers 5.x →
`FlagReranker.compute_score` raises `AttributeError: XLMRobertaTokenizer has no attribute
prepare_for_model`. LightRAG's apply_rerank_if_enabled (utils.py:2750-2752) SWALLOWS it
(`except Exception: logger.error("Error during reranking ... using original chunks")`) →
chunks returned UNRANKED on EVERY query (a noisier no-op than the unconfigured case, F18).
CrossEncoder uses the standard transformers encode path (no `prepare_for_model`), is
transformers-5-native, runs the same model, and ships in the already-present
`sentence-transformers>=3.0` dep (no new infra). Global transformers downgrade was rejected:
it would risk the working BGE-M3 embedding path + LightRAG.

The old v2-era `get_rag()`/`close_rag()` singleton factory (and its module-level _RAG /
_INIT_LOCK) was removed with the dead-v2 sweep (build-loop ④): it built an UNGUARDED,
mis-configured second LightRAG (no assert_safe_workspace gate, no Neo4j/PG backend → local
file storage, none of the §6.0 concurrency/entity-type knobs) parallel to the real
store/graph.get_graph() singleton. The real v3 instance is built in store/graph.py.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
import os
import threading
import time

from papervault.knowledge.config import CONFIG

log = logging.getLogger("ks.store.lightrag_init")

_BGE_MODEL: object = None
_BGE_RERANKER: object = None
# Guards the two lazy GPU-model singletons (issue #31). The models are now LOADED inside the
# offload worker thread (not on the event loop), so first-use construction can race across
# threads (embed sem may be >1; rerank sem defaults 2). Double-checked locking keeps the load
# single-flight without adding any steady-state cost (the None fast-path skips the lock).
_MODEL_LOAD_LOCK = threading.Lock()

# One idle clock covers both models. Activity is counted around synchronous inference,
# including lazy load, so the timer cannot discard a model while a worker uses it.
_IDLE_UNLOAD_ENABLED = os.getenv("KS_BGE_IDLE_UNLOAD", "0") == "1"
_IDLE_UNLOAD_TIMEOUT = float(os.getenv("KS_BGE_IDLE_TIMEOUT", "600"))
_IDLE_UNLOAD_LOCK = threading.Lock()
_IDLE_UNLOAD_TIMER: threading.Timer | None = None
_IDLE_UNLOAD_GENERATION = 0
_IDLE_UNLOAD_ACTIVE = 0


def _unload_idle_models(generation: int) -> None:
    global _BGE_MODEL, _BGE_RERANKER, _IDLE_UNLOAD_TIMER
    with _IDLE_UNLOAD_LOCK:
        if generation != _IDLE_UNLOAD_GENERATION or _IDLE_UNLOAD_ACTIVE:
            return
        _IDLE_UNLOAD_TIMER = None
        if _BGE_MODEL is None and _BGE_RERANKER is None:
            return
        started = time.perf_counter()
        _BGE_MODEL = None
        _BGE_RERANKER = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            log.exception("bge idle unload: CUDA cache release failed")
        log.info("bge embedder and reranker unloaded after idle in %.3fs",
                 time.perf_counter() - started)


@contextmanager
def _model_activity():
    global _IDLE_UNLOAD_ACTIVE, _IDLE_UNLOAD_GENERATION, _IDLE_UNLOAD_TIMER
    if not _IDLE_UNLOAD_ENABLED:
        yield
        return
    with _IDLE_UNLOAD_LOCK:
        _IDLE_UNLOAD_GENERATION += 1
        if _IDLE_UNLOAD_TIMER is not None:
            _IDLE_UNLOAD_TIMER.cancel()
            _IDLE_UNLOAD_TIMER = None
        _IDLE_UNLOAD_ACTIVE += 1
    try:
        yield
    finally:
        with _IDLE_UNLOAD_LOCK:
            _IDLE_UNLOAD_ACTIVE -= 1
            if _IDLE_UNLOAD_ACTIVE == 0:
                _IDLE_UNLOAD_GENERATION += 1
                _IDLE_UNLOAD_TIMER = threading.Timer(
                    _IDLE_UNLOAD_TIMEOUT, _unload_idle_models,
                    args=(_IDLE_UNLOAD_GENERATION,),
                )
                _IDLE_UNLOAD_TIMER.daemon = True
                _IDLE_UNLOAD_TIMER.start()

# Rerank concurrency gate + integrity counter (2026-06-07, eval-trust drill).
# Unlike embedding (embedding_func_max_async=16 in graph.py), LightRAG calls rerank_model_func
# UNGATED — at query concurrency>1 multiple CrossEncoder.predict pile onto the single shared
# GPU0 → CUDA OOM. LightRAG's apply_rerank_if_enabled (utils.py:2750-2752) then SWALLOWS the
# OOM and returns chunks UNRANKED with no error → a SILENT per-query quality degradation that
# corrupts the eval ruler invisibly. Fix: (A) a semaphore bounds concurrent GPU rerank
# (KS_RERANK_MAX_ASYNC, default 1 = serialize on the shared card); (B) on OOM, empty_cache +
# retry at smaller batch sizes so transient pressure is survived; (C) a terminal failure
# increments _RERANK_FAILURES and logs LOUD so run_eval/callers detect a degraded run instead
# of trusting a silent one. rerank_failure_count() exposes the counter.
_RERANK_SEM: object = None
_RERANK_FAILURES: int = 0

# ★ CrossEncoder rerank input cap. CORRECTION 2026-06-15: the old comment claimed chunks are
# ~1200 tokens (chunk_token_size=1200); the live chunker is actually chunk_token_size=2400 (config.py,
# tiktoken: served chunks mean ~2257 / max ~2503) and the BGE-M3 reranker tokenizer counts the SAME
# text ~16% higher (mean ~2617 / p50 ~2750 / max ~3295, never above ~3485). The old 1024 truncated
# ~97% of chunks to their leading ~37%. 4096 (user-chosen 2026-06-16) = NEVER TRUNCATE: every chunk is
# below it, so the cross-encoder scores the FULL chunk. MECHANISM (measured, not padded-to-max_length):
# cost is by ACTUAL token length (min(chunk_tokens, max_length)) x chunk count — verified by same
# max_length=4096 taking 0.92s for short chunks vs 23.4s for long, and halving the chunk count halving
# the time. So 4096 just means "pay the real ~2750-token length" (= identical to setting ~3584, no chunk
# exceeds either); it does NOT pad to 4096. Cost: ~0.24s/(query,chunk) pair -> ~4 min/query rerank over
# base+5 facets x ~171 chunks, serialized. Memory is also actual-token-bound: ~12-16GB reserved for a
# single pool on the 24GB 3090 (the 22GB seen once was a concurrency-4 EVAL high-water mark + frag, not
# live; live rerank is serialized via KS_RERANK_MAX_ASYNC=1, with the #28 OOM-retry as backstop).
# Env-tunable (KS_RERANK_MAX_LENGTH) to trade fidelity for latency. Runs on KS's pinned RTX 3090
# (device cuda:0 resolves to the physical 3090 via CUDA_VISIBLE_DEVICES=1, .env 2026-06-07).
_RERANK_MAX_LENGTH = int(os.getenv("KS_RERANK_MAX_LENGTH", "4096"))
# Initial rerank batch_size — the dominant VRAM-peak lever (32@4096 ≈ 16.6GB; 8 ≈ ~1/4). Lowering it
# is QUALITY-NEUTRAL (still scores full max_length chunks, just in smaller batches → slightly slower);
# the #28 OOM-retry shrinks further (→2→1). Set lower (e.g. 8) when co-renting the GPU with pl/MinerU.
_RERANK_BATCH_SIZE = int(os.getenv("KS_RERANK_BATCH_SIZE", "32"))
# Round-2 latency lever (issue #4, benchmark-arbitrated, default OFF = byte-identical):
# cap the candidate pool BEFORE the cross-encoder. In KS's shipped `mix` mode the pool is
# LightRAG's round-robin interleave of the vector/entity/relation sources (_merge_all_chunks,
# operate.py 1.5.4) — each source pre-sorted best-first — so a PREFIX cut drops each source's
# lowest-ranked TAIL, keeps every source represented, and index positions stay valid against
# the original list. The cross-encoder is ~78% of retrieval wall-clock at avg pool 220
# (2026-07-18 pair). ARBITRATED 2026-07-18 (issue #21, 3 interleaved reps): cap 120 is
# recall-neutral (paired mean −0.8pp, within the noise floor, direction-inconsistent) at
# 1.90× retrieval speed → DEFAULT 120. cap 80 refuted (±8pp instability). 0 = uncapped.
# Never caps below the requested top_n; negative = off.
_RERANK_POOL_CAP = max(0, int(os.getenv("KS_RERANK_POOL_CAP", "120")))


def _get_bge_model() -> object:
    """Load BGE-M3 model once and cache (singleton). Thread-safe (loaded off-loop, issue #31)."""
    global _BGE_MODEL
    if _BGE_MODEL is None:
        with _MODEL_LOAD_LOCK:
            if _BGE_MODEL is None:
                from FlagEmbedding import BGEM3FlagModel

                started = time.perf_counter()
                _BGE_MODEL = BGEM3FlagModel(
                    CONFIG.bge_m3.model_path,
                    use_fp16=True,
                    devices=[CONFIG.bge_m3.device],
                )
                if _IDLE_UNLOAD_ENABLED:
                    log.info("bge embedder loaded in %.3fs", time.perf_counter() - started)
    return _BGE_MODEL


# Embedding GPU-offload gate (issue #31, 2026-07-23 — MCP event-loop robustness).
# BGE-M3 `model.encode` is a SYNCHRONOUS CUDA call. Historically it ran DIRECTLY on the S4
# single event loop, so a query's query/keyword embedding froze the loop for the encode's
# duration — and under the reviewer fan-out (many concurrent retrievals) those blocking bursts
# starved the MCP `initialize`/`tools/list` handshake path (idle 0.2s → 26-60s under load), the
# same event-loop-starvation class the rerank path already fixed via `asyncio.to_thread`
# (Fix A/B above) and the pl OCR-gate decouple. Two coupled changes restore loop liveness:
#   (1) KS_EMBED_OFFLOAD (default ON): run encode in a worker thread so the loop stays free.
#   (2) A GPU-concurrency semaphore (_get_embed_sem, KS_EMBED_GPU_MAX_ASYNC, default 1). The OLD
#       sync-on-loop behaviour ALSO had a hidden side effect: it SERIALIZED embedding (the loop
#       ran one encode at a time) and so pinned the embed VRAM peak at a single forward pass.
#       Moving to to_thread without a gate would let LightRAG's embedding_func_max_async (16)
#       fan 16 concurrent forward passes onto the shared BGE-M3 singleton → a VRAM-peak +
#       shared-nn.Module concurrency change on the co-rented 3090. Default 1 preserves BOTH the
#       old serialization AND the old single-pass VRAM peak — strictly loop-liveness with zero
#       throughput/VRAM regression on the query path (query embeds are tiny). Raise it for build
#       throughput when the card has headroom; this mirrors the rerank Fix-A semaphore exactly.
_EMBED_OFFLOAD = os.getenv("KS_EMBED_OFFLOAD", "1").strip().lower() in ("1", "true", "yes")
_EMBED_SEM: object = None


def _get_embed_sem() -> "asyncio.Semaphore":
    """Lazy per-process semaphore bounding concurrent GPU embed forward passes (issue #31).

    Lazy so it binds to the running event loop on first use, inheriting KS's one-loop-per-process
    invariant (same rationale as _get_rerank_sem). Default 1 = serialize embed on the shared GPU0,
    matching the VRAM/serialization profile of the old sync-on-loop path. KS_EMBED_GPU_MAX_ASYNC
    raises it (build throughput) when the card has headroom."""
    global _EMBED_SEM
    if _EMBED_SEM is None:
        _EMBED_SEM = asyncio.Semaphore(int(os.getenv("KS_EMBED_GPU_MAX_ASYNC", "1")))
    return _EMBED_SEM


def _bge_encode_sync(texts: list[str]):
    import numpy as np

    # Load INSIDE the worker thread (issue #31): the first-use model load is a multi-second
    # synchronous CUDA op — keeping it off the event loop is the whole point of the offload.
    with _model_activity():
        model = _get_bge_model()
        # BGE-M3 supports 8192; the live chunker feeds chunk_token_size=2400 (tiktoken) chunks
        # (BGE-M3 tokenizer counts them ~2617 mean / ~3485 max). max_length=512 silently truncated
        # most of every chunk before embedding (fixed 2026-05-30 per the harden audit); 8192 covers them.
        result = model.encode(texts, batch_size=32, max_length=8192)
        return np.array(result["dense_vecs"])


async def _bge_embed(texts: list[str]):
    """LightRAG embedding func — wraps BGE-M3 inference.

    The synchronous CUDA load+encode is offloaded to a thread (KS_EMBED_OFFLOAD) under a
    GPU-concurrency semaphore so it never blocks the S4 single event loop (issue #31).
    KS_EMBED_OFFLOAD=0 restores the legacy sync-on-loop call (kept only as a diagnostic/revert
    lever — it is the pre-#31 bug)."""
    if not _EMBED_OFFLOAD:
        return _bge_encode_sync(texts)
    async with _get_embed_sem():
        return await asyncio.to_thread(_bge_encode_sync, texts)


def _get_bge_reranker() -> object:
    """Load bge-reranker-v2-m3 once and cache (singleton, V1 / SDD §6.9.1).

    Mirrors `_get_bge_model`'s lazy in-process pattern. Backend is
    `sentence_transformers.CrossEncoder` (transformers-5-native; see module docstring for why
    NOT FlagReranker). CrossEncoder's default ``activation_fn`` is ``Sigmoid()`` for this
    single-label model, so ``predict`` already returns scores in 0..1 — aligning with
    LightRAG's ``relevance_score`` / ``min_rerank_score`` semantics (utils.py:2794-2805)
    WITHOUT a separate normalize step (sigmoid is applied exactly once).

    ★ One-time preflight (SDD §6.9.1 / §11 F21): runs a tiny ``predict`` smoke test at load
    time so a broken reranker (e.g. a transformers-version incompatibility) fails LOUDLY here,
    once, instead of being swallowed into a per-query ERROR-fallback inside LightRAG's
    apply_rerank_if_enabled — which would silently degrade every query with no signal to the
    caller. If this raises, the reranker is genuinely unusable and the build/query path should
    surface it, not mask it."""
    global _BGE_RERANKER
    if _BGE_RERANKER is not None:
        return _BGE_RERANKER
    with _MODEL_LOAD_LOCK:  # single-flight: the load runs off-loop in a worker thread (issue #31)
        if _BGE_RERANKER is not None:
            return _BGE_RERANKER
        import torch
        from sentence_transformers import CrossEncoder
        started = time.perf_counter()

        # ★ KS_RERANK_DTYPE (2026-07-16 latency survey, issue #3): the checkpoint declares
        # float32 and this ctor historically passed no dtype, so the reranker — the query
        # path's dominant compute (~0.24s/pair fp32) — never touched Ampere's half-precision
        # tensor cores (contrast: the embed loader sets use_fp16=True). bfloat16 ≈ halves
        # per-pair latency AND weight+activation VRAM; quality risk = half-precision
        # rounding only (same weights). Flag-gated: default float32 = behavior unchanged;
        # deploy .env flips to bfloat16 ONLY after the paired FAST --no-synth eval
        # (fp32 vs bf16 @ max_length 4096) confirms retrieval metrics neutral.
        dtype_name = os.getenv("KS_RERANK_DTYPE", "float32")
        model_kwargs = (
            {"torch_dtype": getattr(torch, dtype_name)} if dtype_name != "float32" else None
        )
        model = CrossEncoder(
            CONFIG.bge_reranker.model_path,
            device=CONFIG.bge_reranker.device,  # KS's GPU0/cuda:0, NOT pl's 3090
            max_length=_RERANK_MAX_LENGTH,      # ★ score whole chunk, not first 512 tokens
            model_kwargs=model_kwargs,
        )
        # ★ Preflight smoke test: fail loud at load, not silently per-query (F18/F21).
        score = model.predict([("preflight query", "preflight passage")])
        log.info(
            "bge reranker preflight OK (model=%s, max_length=%d, dtype=%s, sample_score=%.4f)",
            CONFIG.bge_reranker.model_path,
            _RERANK_MAX_LENGTH,
            next(model.model.parameters()).dtype,
            float(score[0]),
        )
        _BGE_RERANKER = model
        if _IDLE_UNLOAD_ENABLED:
            log.info("bge reranker loaded in %.3fs", time.perf_counter() - started)
    return _BGE_RERANKER


def _get_rerank_sem() -> "asyncio.Semaphore":
    """Lazy per-process semaphore bounding concurrent GPU rerank (Fix A).

    Lazy so it binds to the running event loop on first contended use. This inherits — does
    NOT newly introduce — LightRAG's existing system-wide ONE-LOOP-PER-PROCESS invariant: the
    rag singleton, the embedding limiter's worker task + PriorityQueue, and the Neo4j/PG async
    pools are all pinned to the first loop (see mcp/server.py:9-14). So every consumer (build,
    run_eval, MCP service) already runs a single asyncio loop for the process's lifetime; the
    semaphore is safe under that invariant. (Tests that call across loops must reset it via
    reset_rerank_failures(reset_sem=True).) Default 1 = serialize rerank on the single shared
    GPU0; KS_RERANK_MAX_ASYNC can raise it if the card has headroom. Embedding already has its
    own gate (embedding_func_max_async=16); the reranker had none — that gap caused the OOM."""
    global _RERANK_SEM
    if _RERANK_SEM is None:
        import os
        _RERANK_SEM = asyncio.Semaphore(int(os.getenv("KS_RERANK_MAX_ASYNC", "1")))
    return _RERANK_SEM


def reset_rerank_failures(reset_sem: bool = False) -> None:
    """Zero the rerank-failure counter so a fresh run's count reflects ONLY that run.

    Callers that measure per-run integrity (e.g. run_eval) MUST call this once at run start;
    otherwise the process-global counter accumulates monotonically (a long-lived MCP server
    would report stale failures forever). reset_sem=True also drops the cached semaphore (for
    tests that switch event loops)."""
    global _RERANK_FAILURES, _RERANK_SEM
    _RERANK_FAILURES = 0
    if reset_sem:
        _RERANK_SEM = None


def _predict_with_oom_retry(model, pairs):
    """CrossEncoder.predict with CUDA-OOM survival (Fix B).

    On an OOM RuntimeError, empty the CUDA cache and retry at a smaller batch_size
    (32→8→2→1). Non-OOM errors propagate immediately. If even batch_size=1 OOMs, re-raise the
    last OOM (genuinely no room — handled as a terminal failure by the caller)."""
    last = None
    for bs in ([b for b in (32, 8, 2, 1) if b <= _RERANK_BATCH_SIZE] or [1]):
        try:
            return model.predict(pairs, batch_size=bs)
        except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError subclass
            if "out of memory" not in str(e).lower():
                raise
            last = e
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — best-effort cache drop
                pass
    raise last


def rerank_failure_count() -> int:
    """Number of rerank calls that terminally failed this process (Fix C).

    >0 means some queries were served with UNRANKED chunks — eval/quality metrics for those
    queries are degraded and MUST NOT be trusted as if rerank had applied. run_eval checks
    this after a run and surfaces it loudly."""
    return _RERANK_FAILURES


async def _bge_rerank(
    query: str, documents: list[str], top_n: int | None = None, **kwargs
):
    """LightRAG rerank_model_func (V1 / SDD §6.9.1).

    Contract (verified against utils.py:2671-2752 apply_rerank_if_enabled, lightrag 1.4.16):
      - LightRAG calls it with keyword args ``query=``, ``documents=`` (list[str]), ``top_n=``.
      - Must return ``list[{"index": int, "relevance_score": float}]`` where ``index`` points
        back into the original ``documents`` list; LightRAG reorders chunks by this list and
        stamps each kept chunk with ``rerank_score`` (the relevance_score).
      - Empty/short result is fine: LightRAG falls back to the original chunks.

    ``CrossEncoder.predict`` returns a numpy array of per-pair scores in 0..1 (default
    Sigmoid activation; see _get_bge_reranker). The synchronous GPU call is offloaded to a
    thread so it does not block the S4 single event loop (SDD §6.6 / §11 F21; same trade-off
    shape used for `_bge_embed`).

    Hardened 2026-06-07 (eval-trust drill): the GPU call is bounded by _get_rerank_sem()
    (Fix A) and survives transient CUDA OOM via _predict_with_oom_retry (Fix B). A TERMINAL
    failure (even batch_size=1 OOMs, or any other error) is NOT swallowed silently: it bumps
    _RERANK_FAILURES + logs LOUD and returns [] so LightRAG falls back to original chunks —
    but rerank_failure_count() now lets the caller KNOW the run was degraded (Fix C)."""
    global _RERANK_FAILURES
    if not documents:
        return []
    if _RERANK_POOL_CAP and len(documents) > _RERANK_POOL_CAP:
        cap = max(_RERANK_POOL_CAP, top_n or 0)
        if len(documents) > cap:
            log.debug("rerank pool capped %d -> %d (KS_RERANK_POOL_CAP)", len(documents), cap)
            documents = documents[:cap]
    pairs = [(query, d) for d in documents]
    # Load AND predict inside the worker thread (issue #31): _get_bge_reranker() runs a
    # multi-second CrossEncoder load + preflight on first use — doing it here, off the event
    # loop, keeps the handshake path live at cold start. A load/preflight failure surfaces via
    # the except below (counted + logged loud), preserving the F21 fail-loud contract.
    def _load_and_predict(_pairs):
        with _model_activity():
            return _predict_with_oom_retry(_get_bge_reranker(), _pairs)

    async with _get_rerank_sem():
        try:
            scores = await asyncio.to_thread(_load_and_predict, pairs)
        except Exception as e:  # noqa: BLE001 — terminal: record + loud, never silently degrade
            _RERANK_FAILURES += 1
            log.error(
                "bge rerank FAILED on %d docs (%s: %s) — chunks returned UNRANKED; this run's "
                "rerank is DEGRADED (failure #%d). Lower KS_RERANK_MAX_ASYNC or free GPU0.",
                len(documents), type(e).__name__, str(e)[:120], _RERANK_FAILURES,
            )
            return []
    ranked = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
    if top_n:  # falsy (0/None) → keep all; real caller always passes a positive int (test codifies this)
        ranked = ranked[:top_n]
    return [{"index": i, "relevance_score": float(scores[i])} for i in ranked]
