"""Knowledge System MCP server.

v3 S3 (SDD §5.4/§6.4): query(intent) goes through LightRAG aquery_data(mode='mix')
→ KS-written synth prose. Response: {answer (prose with [paper_key] inline cites),
cited_papers (from references[].file_path), kb_coverage (from processing_info)}.

v3 S4 (SDD §6.6 "S4 运行时接线"): the paper incremental-sync scheduler runs as an
asyncio.Task in THIS server's event loop (via the FastMCP `lifespan` below), NOT a
separate thread with its own asyncio.run. This is a HARD single-loop constraint:
get_graph()'s module-level LightRAG singleton (+ its asyncpg pool and shared_storage
asyncio.Lock) binds to whichever loop first touches it; sharing one instance across two
loops would raise "got Future attached to a different loop" and break the §6.6
delete/insert mutex. So query and ingest/delete share ONE gated rag in ONE loop —
writes serialize on LightRAG's pipeline mutex, query is a lock-free read (§6.4 boundary).

The scheduler is DEFAULT-OFF (KS_AUTO_INGEST_ENABLED=false): ks_ledger is now
workspace-isolated (SDD §4.1, workspace column + per-workspace filtering), so run_round's
ledger writes/deletes can no longer pollute the prod ledger from a probe workspace. The
remaining reason for default-OFF is the §6.5 copy-throughput go/no-go + an explicit human
start — opt in on purpose once that gate is cleared. The dead v2 store.scheduler.
_scheduler_loop (paper_layer_status / chunk_metadata, un-gated prod PG) has been
physically removed in the dead-v2 sweep (§3).

MCP surface (Executor-facing): a SINGLE natural-language tool —
- query(intent) — ask the knowledge base a question; get a sourced prose answer.

propose_ideas / idea-engine is SDD【defer】(不进 v1) — NOT exposed here.
Operator-only ops (diagnostics, by-id chunk inspection, knob-tuning) live in the operator
CLI (`python -m papervault.knowledge.cli`), NOT on the MCP surface.
"""
from __future__ import annotations

from papervault.store import knowledge_runtime

import asyncio
import logging
import math
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

from mcp.server.fastmcp import Context, FastMCP

log = logging.getLogger("ks.mcp.server")


def _auto_ingest_enabled() -> bool:
    """Default OFF (SDD §6.6 S4 接线节)。ks_ledger 现已 workspace 隔离(§4.1,run_round 写/删
    账本只命中当前 workspace,不再污染生产);默认仍 OFF 是等 §6.5 副本吞吐 go/no-go + 人工
    显式启动。需显式 opt-in (KS_AUTO_INGEST_ENABLED=true/1/yes)。"""
    return os.getenv("KS_AUTO_INGEST_ENABLED", "false").strip().lower() in ("1", "true", "yes")


def _skip_embed_selfcheck() -> bool:
    """Escape hatch for the boot-time embedding self-check (issue #84). Set
    KS_SKIP_EMBED_SELFCHECK=1 (CI / tests / a known-good boot) to bypass the smoke-test and
    start the scheduler without exercising the GPU embed path."""
    return os.getenv("KS_SKIP_EMBED_SELFCHECK", "").strip().lower() in ("1", "true", "yes")


# Boot-time embedding self-check timeout (issue #84). The smoke-test loads BGE-M3 (a
# multi-second GPU model load on a cold boot) + does ONE tiny encode; 60s is generous
# headroom for the load while still bounding a HANG so a wedged embed stack can never block
# boot forever. Env-tunable.
_EMBED_SELFCHECK_TIMEOUT_SEC = float(os.getenv("KS_EMBED_SELFCHECK_TIMEOUT_SEC", "60"))
_embed_probe_task: Optional[asyncio.Task] = None


async def _embedding_stack_ok() -> bool:
    """Boot-time embedding smoke-test (issue #84) — FAIL SAFE.

    On 2026-07-23 a torch/torchvision ABI drift crashed BGE-M3 embedding on EVERY call, yet
    the service booted fine and the scheduler CHURNED for an hour — re-distilling then failing
    every document (deleting + re-inserting ~218 graph docs/round) before anyone noticed. This
    check embeds ONE short string through the REAL embedder path (`_bge_embed` → the same
    BGE-M3 / torch / torchvision stack the build uses), BEFORE the scheduler starts. If it
    raises — or hangs past KS_EMBED_SELFCHECK_TIMEOUT_SEC — we DO NOT start `main_loop`:
    auto-ingest stays off and the destructive re-distill loop never begins. Queries can still
    be served (degraded) since the query path opens the graph lazily on its own.

    Escape hatch: KS_SKIP_EMBED_SELFCHECK=1 bypasses it (CI / tests / a known-good boot).

    Returns True to proceed (passed OR skipped), False to refuse to start the scheduler.
    Never raises — any failure is caught, logged LOUD + actionable, and turned into False.
    """
    if _skip_embed_selfcheck():
        log.info("KS embedding self-check SKIPPED (KS_SKIP_EMBED_SELFCHECK set)")
        return True

    # Imported here (not at module load) so a broken FlagEmbedding/torch import chain surfaces
    # inside the guarded probe below, exactly like the per-call build path — not at server import.
    from papervault.knowledge.store.lightrag_init import _bge_embed

    async def _probe() -> int:
        vec = await _bge_embed(["knowledge-system boot embedding self-check"])
        # A real embed returns a (1, dim) array → len == 1. A silent empty result is also a fault.
        return len(vec) if vec is not None else 0

    global _embed_probe_task
    try:
        # GPU to_thread work cannot be cancelled. Retain ONE probe across timeouts so
        # retries neither release its GPU semaphore nor pile up executor/native threads.
        if _embed_probe_task is None:
            _embed_probe_task = asyncio.create_task(_probe(), name="ks-embedding-probe")
        n = await asyncio.wait_for(asyncio.shield(_embed_probe_task),
                                   timeout=_EMBED_SELFCHECK_TIMEOUT_SEC)
    except Exception as e:  # noqa: BLE001 — ANY failure (import/ABI/CUDA/timeout) must fail safe
        log.error(
            "KS embedding self-check FAILED — scheduler NOT started to avoid churning a broken "
            "build (re-distilling + failing every document, ~218 graph docs/round). Fix the "
            "torch/embedding stack (e.g. torchvision ABI drift), automatic retry is scheduled. Queries still "
            "served (degraded). Bypass with KS_SKIP_EMBED_SELFCHECK=1 once known-good. "
            "Error: %s: %s",
            type(e).__name__, e, exc_info=True,
        )
        return False
    finally:
        if _embed_probe_task is not None and _embed_probe_task.done():
            _embed_probe_task = None
    if n < 1:
        log.error(
            "KS embedding self-check FAILED — embedder returned an EMPTY result (len=%d); "
            "scheduler NOT started to avoid churning a broken build. Fix the embedding stack, "
            "automatic retry is scheduled. Bypass with KS_SKIP_EMBED_SELFCHECK=1 once known-good.",
            n,
        )
        return False
    log.info("KS embedding self-check passed (embedded 1 probe string → %d vector).", n)
    return True


# The S4 auto-ingest scheduler + the LightRAG graph / asyncpg pool are SERVER-LIFETIME singletons,
# NOT per-session. CRITICAL (bug fixed 2026-06-19): under streamable-http the MCP SDK runs the FastMCP
# `lifespan` (below) ONCE PER CLIENT SESSION — lowlevel Server.run() does
# `await stack.enter_async_context(self.lifespan(self))` and the streamable-http session manager calls
# Server.run() per connection. The real SERVER-STARTUP hook is the Starlette app lifespan, which
# FastMCP hardcodes to `session_manager.run()` only (fastmcp/server.py); __main__ wraps that app
# lifespan to call start_background()/stop_background() ONCE at boot/shutdown. So the scheduler must
# live here, not in the per-session lifespan. (Old behaviour: scheduler in the per-session lifespan →
# never started at boot, a DUPLICATE started per client, and close_graph/close_pool fired on EVERY
# session exit.) SDD §6.6 single-loop still holds: the app lifespan AND the session lifespan both run
# in uvicorn's loop — the same loop query() uses — so get_graph()'s singleton binds correctly.
_bg_task: Optional[asyncio.Task] = None
_startup_task: Optional[asyncio.Task] = None
_authority_task: Optional[asyncio.Task] = None


def _authority_checked(task: asyncio.Task) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        log.error("Knowledge authority check failed; scheduler disabled and knowledge queries refused until restart: %s", error)


async def _ensure_knowledge_authority() -> dict:
    """Retain one process-lifetime check; a slow catalog never blocks the event loop.

    Cancellation of a client/startup waiter must not cancel the database thread
    or cause a second connection. Both the result and rejection stay cached.
    Mutating runtime entry points continue to enforce authority independently.
    """
    global _authority_task
    if _authority_task is None:
        expected = "catalog" if knowledge_runtime.knowledge_jobs_mode() == "postgres" else "legacy"
        _authority_task = asyncio.create_task(
            asyncio.to_thread(knowledge_runtime.assert_knowledge_authority, expected), name="ks-authority-check")
        _authority_task.add_done_callback(_authority_checked)
    return await asyncio.shield(_authority_task)


def _startup_finished(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()  # Startup may outlive its bounded initial waiter.


async def _wait_selfcheck_retry(delay: float) -> None:
    await asyncio.sleep(delay)


def _selfcheck_retry_delay() -> float:
    """Fixed retry interval: never spin or postpone recovery indefinitely."""
    try:
        delay = float(os.getenv("KS_EMBED_SELFCHECK_RETRY_SEC", "60"))
    except ValueError:
        delay = 60.0
    return min(300.0, max(1.0, delay))


async def _start_once() -> bool:
    mode = knowledge_runtime.knowledge_jobs_mode()
    await _ensure_knowledge_authority()
    if not await _embedding_stack_ok():
        return False
    from papervault.knowledge.scheduler.round import DEFAULT_ROUND_INTERVAL

    global _bg_task
    interval = float(os.getenv("KS_AUTO_INGEST_INTERVAL_SEC", str(DEFAULT_ROUND_INTERVAL)))
    if mode == "postgres":
        from papervault.knowledge.scheduler.catalog_service import run_catalog_loop
        _bg_task = asyncio.create_task(run_catalog_loop(interval=interval), name="ks-auto-ingest")
    else:
        from papervault.knowledge.scheduler.round import main_loop
        from papervault.knowledge.store.graph import get_graph
        rag = await get_graph()
        _bg_task = asyncio.create_task(main_loop(rag, interval=interval), name="ks-auto-ingest")
    log.info("S4 scheduler task started at server startup (interval=%ss)", interval)
    return True


async def _retry_start() -> None:
    """Own one startup attempt at a time, retrying failures without blocking serving."""
    global _startup_task
    try:
        await _ensure_knowledge_authority()
        if not _auto_ingest_enabled():
            log.info("auto-ingest DISABLED (set KS_AUTO_INGEST_ENABLED=true to opt in)")
            return
        while _auto_ingest_enabled():
            try:
                if await _start_once():
                    return
            except Exception:
                log.exception("KS background startup failed; HTTP serving continues; retry scheduled")
            delay = _selfcheck_retry_delay()
            log.warning("KS startup not ready; retrying in %ss", delay)
            await _wait_selfcheck_retry(delay)
    finally:
        _startup_task = None


async def start_background() -> None:
    """Start the S4 auto-ingest scheduler ONCE, in the server's event loop. Idempotent — safe to call
    from BOTH the server-startup app lifespan (boot, __main__) and the per-session FastMCP lifespan
    (no-op after boot). Default-OFF; opt in with KS_AUTO_INGEST_ENABLED=true.

    Wait at most KS_STARTUP_AWAIT_SEC (default 3s; 0 skips waiting). A slow graph/model
    initialization continues in the background, with failed attempts retried automatically.
    Authority is checked once off-loop even when ingestion is disabled. A rejected
    authority disables knowledge queries/writers while library serving continues.
    """
    knowledge_runtime.knowledge_jobs_mode()  # Validate the local selector without database I/O.
    global _startup_task
    # Also called per HTTP session: it must never reopen serving while shutdown
    # is draining its fixed task snapshot. A restarted process gets fresh state.
    if _query_stopping:
        return
    if _bg_task is not None or _startup_task is not None:
        return
    if _authority_task is not None and (not _auto_ingest_enabled() or
            (_authority_task.done() and not _authority_task.cancelled() and _authority_task.exception() is not None)):
        return
    try:
        timeout = float(os.getenv("KS_STARTUP_AWAIT_SEC", "3"))
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("startup wait must be finite and nonnegative")
    except ValueError:
        log.warning("Invalid KS_STARTUP_AWAIT_SEC; using the default 3s startup wait")
        timeout = 3.0
    # Publish before awaiting: concurrent sessions cannot start another initialization.
    # This task owns both the initial attempt and retries; timeout must not cancel an
    # in-progress graph initialization or launch a competing attempt against the singleton.
    _startup_task = asyncio.create_task(_retry_start(), name="ks-embedding-retry")
    _startup_task.add_done_callback(_startup_finished)
    try:
        await asyncio.wait_for(asyncio.shield(_startup_task), timeout=timeout)
    except asyncio.TimeoutError:
        log.warning(
            "KS startup still pending after %ss (KS_STARTUP_AWAIT_SEC); "
            "HTTP serving continues while initialization/retries run in the background",
            timeout,
        )
    except Exception:
        # Knowledge rejection must not prevent the unified library HTTP service
        # from binding. The retained authority task fails every knowledge query.
        log.error("Knowledge background startup refused; library serving remains available")


async def stop_background() -> None:
    """Graceful SERVER-shutdown cleanup (called once from the app-lifespan shutdown, NOT per session):
    drain retained queries, cancel the scheduler + close the graph/pool."""
    global _bg_task, _startup_task, _embed_probe_task, _query_stopping
    _query_stopping = True
    queries = [entry.task for entry in _queries_by_id.values()]
    for task in queries:
        task.cancel()
    await asyncio.gather(*queries, return_exceptions=True)
    _queries_by_id.clear()
    _queries_by_intent.clear()
    if _startup_task is not None:
        startup = _startup_task
        startup.cancel()
        try:
            await startup
        except asyncio.CancelledError:
            pass
        _startup_task = None
    if _embed_probe_task is not None:
        _embed_probe_task.cancel()
        try:
            await _embed_probe_task
        except (asyncio.CancelledError, Exception):
            pass
        _embed_probe_task = None
    if _bg_task is not None:
        _bg_task.cancel()
        try:
            await _bg_task
        except asyncio.CancelledError:
            pass
        log.info("S4 scheduler task cancelled (graceful stop)")
        _bg_task = None
    from papervault.knowledge.store.graph import close_graph
    from papervault.knowledge.ledger.store import close_pool

    await close_graph()
    await close_pool()


@asynccontextmanager
async def _lifespan(_app: "FastMCP") -> AsyncIterator[None]:
    """FastMCP PER-SESSION lifespan (streamable-http runs it once per client connection — see the
    start_background note above). It only idempotently ensures the scheduler is up (a no-op after the
    boot-time start) and tears NOTHING down on session exit — graph/pool/scheduler are server-lifetime,
    cleaned up by stop_background() at app shutdown. (stdio transport runs this once = the single
    session = process lifetime, so start here covers stdio too; cleanup then falls to process exit.)"""
    await start_background()
    yield


mcp = FastMCP("knowledge-system", lifespan=_lifespan, instructions="""
knowledge-system (KS) is your bolt-on expert knowledge base. It has ingested everything
the lab has read in space physics, space computing, and AI4Science — papers,
textbooks, satellite + onboard-instrument references, spaceborne hardware/networking
studies, and distilled insights — and digested it so you can just ask.

- What it's for: accurate, reliable, in-depth answers about what the field already knows,
  grounded in real sources (every answer carries [paper_key] cites) — not guesses.
- When to use it: any time you need prior knowledge — scouting a new topic, verifying a
  claim, finding known failure modes, confirming a mechanism while debugging, choosing
  whom to cite.
- How to use it: query(intent) — ask in ONE full natural-language sentence, like
  briefing a senior colleague (what you're doing + what you want to know), NOT keywords.
  You get a prose answer + a cited_papers list to trace.

(Diagnostics / by-id chunk lookup / knob-tuning are operator ops in the operator CLI
`python -m papervault.knowledge.cli`, not here.)

Design: docs/architecture.md.
""".strip())


# S17 (2026-07-16, issue #3): progress heartbeat for long queries. Claude Code (and other
# MCP clients) abort a tool call after ~300s of "no response or progress"; a real KS query
# currently runs ~6-7 min, so every default-config caller timed out and the server kept
# burning a full ~150-250k-token synth for a client that was already gone. A progress
# notification every ~45s resets the client's idle timer. Interval env-tunable
# (KS_PROGRESS_HEARTBEAT_S; <=0 disables).
_HEARTBEAT_INTERVAL_S = float(os.getenv("KS_PROGRESS_HEARTBEAT_S", "45"))


@dataclass
class _QueryEntry:
    query_id: str
    intent: str
    task: asyncio.Task
    started_at: float
    completed_at: float | None = None
    collected: bool = False


# Process-local, single-serving-loop state. Active work is never evicted: when
# every slot is active, new questions get a busy answer. Completed detached work
# expires lazily after retention or earlier if capacity is needed for new work.
_QUERY_CAPACITY = 128
_queries_by_id: dict[str, _QueryEntry] = {}
_queries_by_intent: dict[str, _QueryEntry] = {}
_query_stopping = False


def _query_setting(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        if not math.isfinite(value):
            raise ValueError("query setting must be finite")
        return value
    except ValueError:
        log.warning("Invalid %s; using default %ss", name, default)
        return default


def _forget_query(entry: _QueryEntry) -> None:
    if _queries_by_id.get(entry.query_id) is entry:
        _queries_by_id.pop(entry.query_id)
    if _queries_by_intent.get(entry.intent) is entry:
        _queries_by_intent.pop(entry.intent)


def _find_query(intent: str) -> _QueryEntry | None:
    """Shared with admission: resume/collect calls need no new executing permit."""
    now = time.monotonic()
    retention = max(0.0, _query_setting("KS_MCP_QUERY_RETENTION_S", 900.0))
    for entry in list(_queries_by_id.values()):
        if entry.completed_at is not None and now - entry.completed_at >= retention:
            _forget_query(entry)
    key = intent.strip()
    return _queries_by_id.get(key) or _queries_by_intent.get(key)


def _query_finished(entry: _QueryEntry) -> None:
    entry.completed_at = time.monotonic()
    if not entry.task.cancelled():
        entry.task.exception()  # Detached failures are delivered on collection.


async def _execute_query(intent: str) -> dict[str, Any]:
    from papervault.knowledge.query.aquery import query as _query

    await _ensure_knowledge_authority()
    result = await _query(intent=intent)
    receipt = result.get("query_receipt")
    if receipt is not None:
        log.info("query_receipt %s", receipt, extra={"query_receipt": receipt})
    return result


def _new_query(intent: str) -> _QueryEntry | None:
    from papervault.mcp.admission import _retain_query_task

    if len(_queries_by_id) >= _QUERY_CAPACITY:
        completed = [entry for entry in _queries_by_id.values() if entry.task.done()]
        if not completed:
            return None
        victim = min(completed, key=lambda entry: (
            not entry.collected, entry.completed_at or entry.started_at))
        _forget_query(victim)
    query_id = f"q-{uuid4().hex[:12]}"
    task = asyncio.create_task(_execute_query(intent), name=f"ks-query-{query_id}")
    entry = _QueryEntry(query_id, intent, task, time.monotonic())
    # Register synchronously before any await, so concurrent normalized intents
    # cannot dispatch duplicate retrieval/synthesis work on the serving loop.
    _queries_by_id[query_id] = _queries_by_intent[intent] = entry
    task.add_done_callback(lambda _: _query_finished(entry))
    _retain_query_task(task)
    return entry


async def _progress_heartbeat(ctx: Context, stop: asyncio.Event) -> None:
    """Send MCP progress notifications until ``stop`` is set.

    Uses ``session.send_progress_notification(..., related_request_id=...)`` DIRECTLY —
    NOT ``ctx.report_progress()``: in the installed mcp SDK (1.27.1, unchanged through
    1.28.1) report_progress omits related_request_id on streamable-http so the
    notification is silently dropped client-side (upstream python-sdk #953/#2001; the fix
    lives only in the unreleased v2 server module). ctx.info() passes it and works — this
    mirrors that path. No-ops when the caller sent no progressToken (per MCP spec the
    client's own opt-in), so non-progress callers see byte-identical behavior.
    """
    if _HEARTBEAT_INTERVAL_S <= 0:
        return
    meta = getattr(getattr(ctx, "request_context", None), "meta", None)
    token = getattr(meta, "progressToken", None)
    if token is None:
        return
    t0 = time.monotonic()
    tick = 0
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=_HEARTBEAT_INTERVAL_S)
            return  # stop set → query finished, exit quietly
        except asyncio.TimeoutError:
            pass  # interval elapsed → send a tick
        tick += 1
        try:
            await ctx.session.send_progress_notification(
                progress_token=token,
                progress=float(tick),
                total=None,
                message=f"query running — {int(time.monotonic() - t0)}s elapsed",
                related_request_id=ctx.request_id,
            )
        except Exception:  # noqa: BLE001 — heartbeat must never kill the query itself
            log.debug("progress heartbeat send failed (tick=%d)", tick, exc_info=True)


@mcp.tool()
async def query(intent: str, ctx: Optional[Context] = None) -> dict[str, Any]:
    """Ask the lab's bolt-on expert knowledge base a question, in plain language.

    KS covers space physics, space computing, and AI4Science. It has ingested what
    the lab has read — papers, textbooks, satellite + onboard-instrument references,
    spaceborne hardware/networking studies, and distilled insights — and digested it
    into something you can just ask. It gives you an accurate, sourced answer about what
    the field already knows, instead of you guessing.

    HOW: pass ONE full natural-language sentence, like briefing a senior colleague — say
    what you're doing AND what you want to know. NOT keywords. You get back a prose answer
    with [paper_key] inline cites + a cited_papers list to trace.

    WHEN: any time you need "what's already known about this?" — scouting a new topic,
    verifying a claim, finding known failure modes, confirming a mechanism while debugging,
    or deciding whom to cite.

    Good examples (rich intent — match these):
      ✓ "Compare radiation-tolerant on-board AI hardware and inter-satellite task
         offloading for a LEO computing constellation. What power, latency, and
         thermal limits do the sources establish, and which faults can TMR mask?"
      ✓ "我做 Voyager 外日球层 XPINN 反演 cycle 3, 想看 PINN stiff PDE 训练稳定性
         最近进展, 尤其 adaptive sampling 那条线, 看能不能借鉴到 Parker transport"
      ✓ "What's the cross-paper consensus on whether AD-only PINN under-constrains
         in sparse-collocation regimes? Both supporting + contradicting evidence,
         so I can map the boundary for idea22 SEP GNN-PINN"
      ✓ "Find recent (last 2-3 yr) papers on graph neural networks + physics
         constraints for multi-sensor space-weather inverse problems. Want method
         comparison, not a survey."
      ✓ "我在 cycle 5 debug GNN-PINN 训练不收敛, 多 loss 项相互压制. 列已知
         failure modes + 各自缓解方法 + 适用 regime"

    Bad examples (thin keyword — results drift):
      ✗ "PINN review"   ✗ "SEP papers"   ✗ "GNN ML"

    Returns:
      {
        "answer":        "<prose answer with [paper_key] inline cites>",
        "cited_papers":  ["Reames2023", ...],   # feed these to get_paper
        "cited_sources": ["textbook:Schlickeiser2002", ...],  # cited operator sources
        "retrieved_papers": ["Reames2023", ...],  # all retrieval candidates
        "query_receipt": {"generation_id": "...", ...},  # read counts and synthesis timing
        "abstract_only_papers": ["Schwartz2022", ...],  # cited, but only the abstract is known
        "kb_coverage":   "strong" | "thin" | "weak" | "empty",   # honesty signal
      }
    kb_coverage="empty"/"weak"/"thin" → rephrase more specifically, or fall back to
    paper-library `search_papers` for fresher external literature.

    Latency: minutes, not seconds — p50 ~4 min under load (p90 ~7 min); plan other
    work around the call. After KS_MCP_QUERY_WAIT_S (default 90 seconds), unfinished
    work returns {"status": "running", "query_id": "q-...", "retry_after_s": 5, ...}.
    Call query again with the SAME intent or intent="<query_id>" to collect the
    answer without restarting work. Completed answers can be replayed by ID;
    after collection, the same intent starts fresh work. Results are retained for
    KS_MCP_QUERY_RETENTION_S (default 900 seconds) after completion, subject to
    bounded capacity. KS_MCP_QUERY_WAIT_S <= 0 waits until completion.
    When the service is saturated the call returns a
    normal result {"status": "busy", "busy": true, "retry_after_s": N, ...}
    instead of an answer: nothing is wrong — wait retry_after_s seconds, then
    retry, rather than treating papervault as down or retrying at once.
    """
    if _query_stopping:
        return {"status": "busy", "busy": True, "retry_after_s": 5,
                "reason": "Knowledge server is shutting down; retry after restart."}
    entry = _find_query(intent)
    if entry is None:
        intent = intent.strip()
        if re.fullmatch(r"q-[0-9a-f]{12}", intent):
            return {"status": "not_found", "query_id": intent,
                    "reason": "Query expired or server restarted; resubmit the original intent."}
        entry = _new_query(intent)
        if entry is None:
            return {"status": "busy", "busy": True, "retry_after_s": 5,
                    "reason": "Query registry is at capacity; retry after running work finishes."}

    return await _wait_query(entry, ctx)


async def _wait_query(entry: _QueryEntry, ctx: Optional[Context] = None) -> dict[str, Any]:
    """Wait on an already selected entry, including admission's atomic resumes."""
    # Heartbeats belong to this waiter, never to the detached execution: a lost
    # request must not keep trying to notify a dead session in the background.
    stop = asyncio.Event()
    heartbeat = asyncio.create_task(_progress_heartbeat(ctx, stop)) if ctx else None
    collected = False
    try:
        if ctx:
            try:
                await ctx.info(f"query intent={entry.intent!r}")
            except Exception:
                log.debug("query log notification failed; execution remains collectable", exc_info=True)
        wait = _query_setting("KS_MCP_QUERY_WAIT_S", 90.0)
        if wait <= 0:
            result = await asyncio.shield(entry.task)
        else:
            try:
                result = await asyncio.wait_for(asyncio.shield(entry.task), timeout=wait)
            except asyncio.TimeoutError:
                if entry.task.done():
                    result = entry.task.result()  # Includes an underlying TimeoutError.
                else:
                    return {"status": "running", "running": True, "query_id": entry.query_id,
                            "elapsed_s": int(time.monotonic() - entry.started_at), "retry_after_s": 5,
                            "reason": f'Query continues in the background. Call query with the same intent '
                                      f'or intent="{entry.query_id}" to collect the answer.'}
        collected = True
        return result
    except asyncio.CancelledError:
        # Shutdown owns cancellation/drain; a disconnected caller only stops its
        # own wait. The registered task and its execution permit remain alive.
        raise
    except Exception:
        collected = entry.task.done()
        raise
    finally:
        if heartbeat is not None:
            stop.set()
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        # No await may follow consumption: cancellation during notification
        # teardown must leave even a completed answer available for resume.
        if collected:
            if entry.task.exception() is not None:
                _forget_query(entry)
            else:
                entry.collected = True
                if _queries_by_intent.get(entry.intent) is entry:
                    _queries_by_intent.pop(entry.intent, None)


# NOTE: propose_ideas / idea-engine is SDD §1/§3/§5.4【defer】 — confirmed要、不进 v1
# (项目灵魂, 单独脑暴). It is intentionally NOT exposed on the v1 MCP surface; the v2
# implementation (pgvector/retrieve_core stack) was physically removed in the dead-v2
# sweep (§3) and will be redesigned from scratch. v1's sole Executor-facing tool is
# query(intent).
