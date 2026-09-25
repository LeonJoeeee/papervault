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

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

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

    try:
        # wait_for bounds a HANG. On timeout the coroutine is cancelled; if the encode is stuck
        # in the offload worker thread that thread cannot be cancelled, but it is abandoned and
        # boot proceeds WITHOUT the scheduler — the fail-safe outcome we want.
        n = await asyncio.wait_for(_probe(), timeout=_EMBED_SELFCHECK_TIMEOUT_SEC)
    except Exception as e:  # noqa: BLE001 — ANY failure (import/ABI/CUDA/timeout) must fail safe
        log.error(
            "KS embedding self-check FAILED — scheduler NOT started to avoid churning a broken "
            "build (re-distilling + failing every document, ~218 graph docs/round). Fix the "
            "torch/embedding stack (e.g. torchvision ABI drift), then restart. Queries still "
            "served (degraded). Bypass with KS_SKIP_EMBED_SELFCHECK=1 once known-good. "
            "Error: %s: %s",
            type(e).__name__, e, exc_info=True,
        )
        return False
    if n < 1:
        log.error(
            "KS embedding self-check FAILED — embedder returned an EMPTY result (len=%d); "
            "scheduler NOT started to avoid churning a broken build. Fix the embedding stack, "
            "then restart. Bypass with KS_SKIP_EMBED_SELFCHECK=1 once known-good.",
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


async def start_background() -> None:
    """Start the S4 auto-ingest scheduler ONCE, in the server's event loop. Idempotent — safe to call
    from BOTH the server-startup app lifespan (boot, __main__) and the per-session FastMCP lifespan
    (no-op after boot). Default-OFF; opt in with KS_AUTO_INGEST_ENABLED=true."""
    global _bg_task
    if _bg_task is not None:
        return  # already running (server-lifetime singleton)
    if not _auto_ingest_enabled():
        log.info("auto-ingest DISABLED (set KS_AUTO_INGEST_ENABLED=true to opt in)")
        return

    # issue #84 — boot-time embedding self-check BEFORE the scheduler. A broken embed stack
    # (torch/torchvision ABI drift) must NOT let the destructive re-distill loop start and
    # churn a broken build. Runs before get_graph() so a broken stack fails fast without even
    # opening the graph/PG pools; queries still work (they open the graph lazily). FAIL SAFE:
    # on failure we log LOUD and return without starting main_loop.
    if not await _embedding_stack_ok():
        return

    from papervault.knowledge.scheduler.round import DEFAULT_ROUND_INTERVAL, main_loop
    from papervault.knowledge.store.graph import get_graph

    interval = float(os.getenv("KS_AUTO_INGEST_INTERVAL_SEC", str(DEFAULT_ROUND_INTERVAL)))
    # get_graph() runs assert_safe_workspace(); refusing prod 'l0' (without KS_ALLOW_PROD_WORKSPACE=1)
    # raises here = explicit failure, never a silent prod write. Binds the singleton to THIS loop.
    rag = await get_graph()
    _bg_task = asyncio.create_task(main_loop(rag, interval=interval), name="ks-auto-ingest")
    log.info("S4 scheduler task started at server startup (interval=%ss)", interval)


async def stop_background() -> None:
    """Graceful SERVER-shutdown cleanup (called once from the app-lifespan shutdown, NOT per session):
    cancel the scheduler + close the graph/pool."""
    global _bg_task
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
the lab has read — papers, space-physics textbooks, satellite + onboard-instrument
references, and distilled insights — and digested it so you can just ask.

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

    KS has ingested everything the lab has read — papers, space-physics textbooks,
    satellite + onboard-instrument references, and distilled insights — and digested it
    into something you can just ask. It gives you an accurate, sourced answer about what
    the field already knows, instead of you guessing.

    HOW: pass ONE full natural-language sentence, like briefing a senior colleague — say
    what you're doing AND what you want to know. NOT keywords. You get back a prose answer
    with [paper_key] inline cites + a cited_papers list to trace.

    WHEN: any time you need "what's already known about this?" — scouting a new topic,
    verifying a claim, finding known failure modes, confirming a mechanism while debugging,
    or deciding whom to cite.

    Good examples (rich intent — match these):
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
        "cited_sources": ["textbook:Schlickeiser2002", ...],  # operator sources (not papers)
        "abstract_only_papers": ["Schwartz2022", ...],  # cited, but only the abstract is known
        "kb_coverage":   "strong" | "thin" | "empty",   # honesty signal
      }
    kb_coverage="empty"/"thin" → rephrase more specifically, or fall back to
    paper-library `search_papers` for fresher external literature.

    Latency: minutes, not seconds — p50 ~4 min under load (p90 ~7 min); plan other
    work around the call. When the service is saturated the call returns a
    normal result {"status": "busy", "busy": true, "retry_after_s": N, ...}
    instead of an answer: nothing is wrong — wait retry_after_s seconds, then
    retry, rather than treating papervault as down or retrying at once.
    """
    if ctx:
        await ctx.info(f"query intent={intent!r}")

    from papervault.knowledge.query.aquery import query as _query

    # S17: heartbeat keeps idle-timeout clients alive for the multi-minute pipeline.
    stop = asyncio.Event()
    heartbeat = asyncio.create_task(_progress_heartbeat(ctx, stop)) if ctx else None
    try:
        return await _query(intent=intent)
    finally:
        if heartbeat is not None:
            stop.set()
            try:
                await asyncio.wait_for(heartbeat, timeout=10)
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 — CancelledError is BaseException
                heartbeat.cancel()


# NOTE: propose_ideas / idea-engine is SDD §1/§3/§5.4【defer】 — confirmed要、不进 v1
# (项目灵魂, 单独脑暴). It is intentionally NOT exposed on the v1 MCP surface; the v2
# implementation (pgvector/retrieve_core stack) was physically removed in the dead-v2
# sweep (§3) and will be redesigned from scratch. v1's sole Executor-facing tool is
# query(intent).
