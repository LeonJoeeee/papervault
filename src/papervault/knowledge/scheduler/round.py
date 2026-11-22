"""S4 scheduler — round orchestration + doc_status terminal writeback (SDD §6.1/§6.6).

`run_round()` wires the pure `diff()` (scheduler/reconcile.py) to the S2 atoms
(ingest/distill.py: distill_batch / remove_one) into one incremental-sync round,
preserving the §6.6 delete-then-insert mutex (先全删, 后全插).

`reconcile_terminal()` closes paper `processing` rows with their own document:
apipeline_enqueue_documents only returns a track_id, never `done`. So each round
first reads LightRAG doc_status terminal states for the still-`processing` rows and
writes back `done` (PROCESSED) / `error` (FAILED). 中途态(PROCESSING/PENDING)留下轮;
查不到 / 非预期态(PREPROCESSED, F10)计入 stuck_guard, 连续 N 轮不前进 → error
(防 enqueue 早返 / 内容去重 F16 孤儿成为永久死状态).

`reconcile_library()` (#148) schedules terminal papers with available content and no graph doc,
and records content duplicates as done_meta with an explicit surviving twin/content snapshot.

`reconcile_healed()` (#131, #148) closes the other drift: LightRAG's own pipeline retries a
FAILED-with-content doc until it is PROCESSED, but a ledger row that already reached
`error` / `error_parked` was never revisited — the ledger kept reporting a failure (and an
`error` row was re-distilled, deleting the good doc) while the graph held the paper. Any
`error` / `error_parked` row whose doc is PROCESSED is written back `done`.

Round shape (SDD §13):
  reconcile_library → reconcile_terminal → reconcile_healed → diff(含 status 维度) → REMOVE
  → REDISTILL_DELETE(只回报删成功) → DISTILL_BATCH(只吃删成功的 redistill ∪ to_distill)
  → tail process for orphan recovery unless the build paused this round.

Build-path breaker (#143): while the process-wide breaker (store/build_breaker.py) is OPEN — the
upstream LLM path failed persistently — a round stops after the reconcile steps (DB reads, no
LLM): no diff-driven REMOVE / REDISTILL / DISTILL and no tail process, since each of them drives
LLM calls. The main loop also defers the operator-doc pickup. After the cool-down the next round
runs whole as the probe; the breaker logs its own transitions, a paused round logs only at DEBUG.
The opt-in off-peak schedule (#209) uses the same gates before failure-state checks: new build
work and operator-doc pickup wait through peak windows, then resume on the next off-peak round.

Prod-safety: get_graph() (passed in as `rag`) already gated by assert_safe_workspace().
This module never opens a graph itself; the caller owns the instance + workspace gate.
"""
from __future__ import annotations

from papervault.store import knowledge_runtime

import asyncio
import logging
import os
import time
from collections import defaultdict
from typing import Iterable, Optional

from lightrag.base import DocStatus

from papervault.knowledge.ingest.abstract_doc import done_status_for
from papervault.knowledge.ingest.distill import distill_batch, doc_status_of, docs_by_ids, remove_one
from papervault.knowledge.ingest.fingerprint import fingerprint
from papervault.knowledge.ingest.paper_library_client import catalog_state_token
from papervault.knowledge.ingest.vault import load_clean_index
from papervault.knowledge.ledger import store as ledger
from papervault.knowledge.ledger.store import _max_attempts
from papervault.knowledge.scheduler.opdoc_pickup import drain_pending
from papervault.knowledge.scheduler.reconcile import Diff, diff, reconcile_library
from papervault.knowledge.store.build_breaker import BuildPausedError, get_breaker
from papervault.library.services.concurrency import trim_process_memory

log = logging.getLogger("ks.scheduler.round")

PAPER = "paper"
DEFAULT_STUCK_LIMIT = 3        # processing 行连续 N 轮不前进 → error(§6.6 永卡兜底)
DEFAULT_ROUND_INTERVAL = 60.0  # main_loop sleep 秒
DEFAULT_FULL_RECONCILE_INTERVAL_S = 1800.0  # periodic safety full-reconcile during idle
_ACTIVE_LEDGER_STATUSES = frozenset({"processing", "error", "pending_remove"})
_STABLE_DOC_STATUSES = frozenset({"processed", "failed"})

# 进程内 stuck 计数(重启从 0 重数 —— 重启本身触发 LightRAG 自愈,见 §6.6)。
_stuck: dict[str, int] = defaultdict(int)


def _full_reconcile_interval() -> float:
    try:
        return max(0.0, float(os.getenv("KS_FULL_RECONCILE_INTERVAL_S", str(DEFAULT_FULL_RECONCILE_INTERVAL_S))))
    except ValueError:
        return DEFAULT_FULL_RECONCILE_INTERVAL_S


async def _ledger_idle_token():
    """Return a cheap ledger state token when no active rows exist, else None."""
    if hasattr(ledger, "_conn") and hasattr(ledger, "_workspace"):
        ws = ledger._workspace()
        async with ledger._conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT status, count(*), max(last_processed_at) "
                    "FROM ks_ledger WHERE workspace=%s GROUP BY status ORDER BY status",
                    (ws,),
                )
                rows = await cur.fetchall()
        token = []
        for status, count, max_ts in rows:
            if status in _ACTIVE_LEDGER_STATUSES and int(count) > 0:
                return None
            token.append((status, int(count), max_ts.isoformat() if max_ts is not None else ""))
        return tuple(token)
    # Test/fake ledger fallback (in-memory FakeLedger).
    if hasattr(ledger, "rows"):
        active = await ledger.load_by_status(tuple(_ACTIVE_LEDGER_STATUSES))
        if active:
            return None
        return tuple(sorted(
            (r.ingest_source, r.source_id, r.status, r.fingerprint, r.doc_id)
            for r in ledger.rows.values()
        ))
    return None


async def _doc_status_idle_token(rag):
    """Return a cheap LightRAG doc_status token when zero in-flight docs exist, else None.

    Terminal ``failed`` rows (e.g. historical ``dup-*`` audit rows or ``error_parked``
    docs) remain stable across clean idle rounds and are hashed into the token so any
    count change (new failure or external heal) still invalidates the fast-path.
    """
    doc_status = getattr(rag, "doc_status", None)
    if doc_status is not None and hasattr(doc_status, "get_status_counts"):
        counts = await doc_status.get_status_counts()
        normalized: dict[str, int] = {}
        for k, v in counts.items():
            st = str(k).lower()
            n = int(v)
            if st == "all":
                continue
            if st not in _STABLE_DOC_STATUSES:
                if n > 0:
                    return None
                continue
            if n > 0:
                normalized[st] = normalized.get(st, 0) + n
        return tuple(sorted(normalized.items()))
    if hasattr(rag, "rows") and isinstance(rag.rows, dict):
        counts: dict[str, int] = defaultdict(int)
        for row in rag.rows.values():
            st = str(row.get("status", "")).lower()
            if st not in _STABLE_DOC_STATUSES:
                return None
            counts[st] += 1
        return tuple(sorted(counts.items()))
    return None


async def _idle_state_token(rag):
    """Cheap combined state token for skipping redundant 46k-row idle rounds.

    Returns None whenever active work is in flight or any probe cannot confirm
    an idle state (fail-open to full round).
    """
    try:
        cat_token = await asyncio.to_thread(catalog_state_token)
        led_token = await _ledger_idle_token()
        if led_token is None:
            return None
        doc_token = await _doc_status_idle_token(rag)
        if doc_token is None:
            return None
        return (cat_token, led_token, doc_token)
    except Exception:
        log.debug("idle state probe unavailable; running full round", exc_info=True)
        return None


def _is_clean_idle_summary(summary: dict) -> bool:
    if summary.get("paused") or summary.get("skipped_idle"):
        return False
    if summary.get("to_distill") or summary.get("to_redistill") or summary.get("to_remove"):
        return False
    term = summary.get("terminal") or {}
    if any(term.get(k, 0) for k in ("done", "error", "pending", "stuck_error")):
        return False
    if (summary.get("healed") or {}).get("healed", 0):
        return False
    lib = summary.get("library") or {}
    if lib.get("reopened", 0) or lib.get("duplicates", 0):
        return False
    return True


async def reconcile_terminal(
    rag,
    *,
    stuck_limit: int = DEFAULT_STUCK_LIMIT,
    only_keys: Optional[Iterable[str]] = None,
) -> dict:
    """阶段0(b)(SDD §6.6):读 doc_status 终态把 processing 行回写 done/error。

    PROCESSED→done · FAILED→error · 中途态(PROCESSING/PENDING)留下轮 ·
    查不到行 / 非预期态(PREPROCESSED 等, F10)→ stuck_guard,N 轮不前进 → error。

    only_keys (subset entry, blocker ③): limit the processing-row scan to this key set.
    Without it a subset run_round would still收口 (and stuck_guard→error) processing rows
    OUTSIDE the subset — defeating the "一小撮固定 keys" intent and racing other rounds.
    """
    knowledge_runtime.assert_knowledge_authority("legacy")
    if hasattr(ledger, "load_by_status"):
        proc = [r for r in await ledger.load_by_status(("processing",)) if r.ingest_source == PAPER]
    else:
        proc = [r for r in (await ledger.load(PAPER)).values() if r.status == "processing"]
    if only_keys is not None:
        ks = set(only_keys)
        proc = [r for r in proc if r.source_id in ks]
    counters = {"done": 0, "error": 0, "pending": 0, "stuck_error": 0}
    if not proc:
        return counters

    by_doc_id = {r.doc_id: r for r in proc}
    statuses = await docs_by_ids(rag, list(by_doc_id))

    seen_terminal: set[str] = set()
    for doc_id, rec in by_doc_id.items():
        st = statuses.get(doc_id)
        ds = doc_status_of(st)  # dict-aware (SDD §6.6 ★): aget_docs_by_ids returns plain dicts
        if ds == DocStatus.PROCESSED:
            # done, or done_abstract for an abstract-only doc (#144: the class stays countable).
            await ledger.upsert(PAPER, rec.source_id, doc_id=doc_id, status=done_status_for(rec.fingerprint))
            counters["done"] += 1
            seen_terminal.add(rec.source_id)
        elif ds == DocStatus.FAILED:
            # F9: FAILED-无-content 不自愈 → error;下轮 diff(status 维度)走 delete-then-insert 重投。
            await ledger.upsert(PAPER, rec.source_id, doc_id=doc_id, status="error")
            counters["error"] += 1
            seen_terminal.add(rec.source_id)
            log.info("reconcile_terminal: %s doc_status FAILED → ledger error", rec.source_id)
        elif ds in (DocStatus.PROCESSING, DocStatus.PENDING,
                    DocStatus.PARSING, DocStatus.ANALYZING):
            # 中途态:不写终态,留下轮(本轮末 process 会推进)。PARSING/ANALYZING 是
            # LightRAG 1.5.x 新增的管线相位 — 不列入这里会落进 stuck_guard,3 轮后把
            # 慢文档误判成 error(1.5 移植面 #6)。
            counters["pending"] += 1
            seen_terminal.add(rec.source_id)  # in-flight work is queued, not stuck
        else:
            # st is None(doc_status 无此行 — enqueue 早返/F16 内容去重孤儿)或非预期态(PREPROCESSED, F10)。
            # 记日志不静默跳过(§8 TOTAL);计 stuck_guard。
            if ds is not None:
                log.warning("reconcile_terminal: %s unexpected doc_status=%s (F10)", rec.source_id, ds)
            else:
                log.warning("reconcile_terminal: %s has NO doc_status row (enqueue-drop/F16 orphan)", rec.source_id)

    # stuck_guard:本轮没拿到终态的 processing 行计数,连续 N 轮不前进 → error。
    for rec in proc:
        if rec.source_id in seen_terminal:
            _stuck.pop(rec.source_id, None)
            continue
        if not get_breaker().allows():
            continue  # a paused round is not a failed attempt
        _stuck[rec.source_id] += 1
        if _stuck[rec.source_id] >= stuck_limit:
            await ledger.upsert(PAPER, rec.source_id, doc_id=rec.doc_id, status="error")
            counters["stuck_error"] += 1
            _stuck.pop(rec.source_id, None)
            log.warning(
                "reconcile_terminal: %s stuck in processing %d rounds with no terminal state → error",
                rec.source_id, stuck_limit,
            )
    return counters


# Ledger statuses that record a failed build. LightRAG may still finish the doc afterwards.
_FAILED_LEDGER_STATUSES = ("error", "error_parked")


async def reconcile_healed(rag, *, only_keys: Optional[Iterable[str]] = None,
                           eligible_paper_keys: Optional[Iterable[str]] = None) -> dict:
    """#131: an `error` / `error_parked` ledger row whose LightRAG doc is PROCESSED → `done`.

    LightRAG resets FAILED-with-content docs to PENDING on every pipeline pass, so a build that
    failed on a transient backend outage heals on its own — after the ledger had already written
    `error` and, three failures later, parked the key. Nothing read doc_status for those rows
    again, so `ks stats` undercounted and an `error` row was re-distilled (its good doc deleted
    and rebuilt). The processed doc is the truth: the row becomes `done` (attempts reset, its
    fingerprint — the one the processed build was enqueued with — kept, so a later content change
    still re-distills through diff). Rows whose doc is absent or not yet PROCESSED are untouched.

    Covers every ingest_source (operator-doc rows drift the same way). only_keys (subset entry,
    blocker ③): limit to those paper keys, like reconcile_terminal.
    """
    knowledge_runtime.assert_knowledge_authority("legacy")
    # Operator documents paused mid-ingest retain processing; finish those on a later round too.
    rows = await ledger.load_by_status((*_FAILED_LEDGER_STATUSES, "processing"))
    if only_keys is not None:
        ks = set(only_keys)
        rows = [r for r in rows if r.ingest_source == PAPER and r.source_id in ks]
    if eligible_paper_keys is not None:
        eligible = set(eligible_paper_keys)
        rows = [r for r in rows if r.ingest_source != PAPER or r.source_id in eligible]
    counters = {"healed": 0}
    if not rows:
        return counters
    statuses = await docs_by_ids(rag, [r.doc_id for r in rows])
    for rec in rows:
        if doc_status_of(statuses.get(rec.doc_id)) == DocStatus.PROCESSED:
            await ledger.upsert(rec.ingest_source, rec.source_id, doc_id=rec.doc_id,
                                status=done_status_for(rec.fingerprint))
            counters["healed"] += 1
    if counters["healed"]:
        log.info("reconcile_healed: %d failed ledger row(s) whose doc is processed → done",
                 counters["healed"])
    return counters


def _clear_stuck(source_id: str) -> None:
    _stuck.pop(source_id, None)


async def run_round(
    rag,
    *,
    stuck_limit: int = DEFAULT_STUCK_LIMIT,
    only_keys: Optional[Iterable[str]] = None,
    _idle_state: Optional[dict] = None,
) -> dict:
    """一个增量同步 round(SDD §6.1/§6.6)。进程内串行,删插互斥。

    only_keys (subset entry, blocker ③): None = 全量(production main_loop). When given, the
    whole round — reconcile_terminal scan, diff classification (so to_remove stays inside
    the subset), REMOVE / REDISTILL / DISTILL — is confined to that key set. This is the
    run.py --keys口径 that lets experiments/probe_round_s4.py真跑 run_round over 3-5 fixed
    keys instead of the full ~4k-paper vault (which would burn the §6.5 throughput budget
    and break PROD-SAFETY rule (4)'s small-subset/serial intent).
    """
    if knowledge_runtime.knowledge_jobs_mode() == "postgres":
        knowledge_runtime.assert_knowledge_authority("catalog")
        from .catalog_service import run_catalog_round
        return await run_catalog_round(only_keys=only_keys)
    knowledge_runtime.assert_knowledge_authority("legacy")
    only = set(only_keys) if only_keys is not None else None

    if _idle_state is not None and only is None and get_breaker().allows():
        full_interval = _full_reconcile_interval()
        prev_token = _idle_state.get("token")
        last_full_at = _idle_state.get("last_full_at", 0.0)
        if prev_token is not None and full_interval > 0 and (time.monotonic() - last_full_at) < full_interval:
            current_token = await _idle_state_token(rag)
            if current_token is not None and current_token == prev_token:
                log.debug("run_round: idle fast-path (catalog/ledger/doc_status unchanged)")
                return {"skipped_idle": True}

    idx = await asyncio.to_thread(load_clean_index)
    library = await reconcile_library(rag, idx, only_keys=only)

    # 阶段0(b):先收口上一轮/重启遗留的 processing(subset 时只扫 subset 行)。
    eligible = set(idx) if only is None else set(idx) & only
    term = await reconcile_terminal(rag, stuck_limit=stuck_limit, only_keys=eligible)
    # #131:失败行(error/error_parked)的 doc 已被 LightRAG 自愈为 PROCESSED → done(先于 diff,
    # 免得 error 行被 redistill 删掉好 doc)。
    healed = await reconcile_healed(rag, only_keys=only, eligible_paper_keys=eligible)

    # #143/#209: upstream failure or peak window → start no build work this round (the gate
    # already logged the transition; the reconcile writes above needed no LLM).
    if not get_breaker().allows():
        if _idle_state is not None and only is None:
            _idle_state["token"] = None
        summary = {"terminal": term, "healed": healed, "paused": True}
        log.debug("run_round: build paused by the build-path breaker: %s", summary)
        return summary

    # 阶段1:纯 KS diff(含 status 维度,error 行重投)。subset 时 idx+led 都先裁到 only。
    led = await ledger.load(PAPER)
    d: Diff = await asyncio.to_thread(diff, idx, led, fp_of=fingerprint, only_keys=only)
    reopened = {k for k, _ in library["to_distill"]}
    d.to_redistill = [(k, fp) for k, fp in d.to_redistill if k not in reopened]
    d.to_distill.extend(library["to_distill"])

    # 阶段2:先全删(from idle,§6.6 删插互斥)。
    removed_count = 0
    for key in d.to_remove:
        r = await remove_one(rag, key, delete_ledger=True)
        if r == "removed":
            removed_count += 1

    # REDISTILL_DELETE:内容变/error 者先删旧 doc(留 ledger 行);只回报删成功的 key。
    redistill_removed: set[str] = set()
    for key, _fp in d.to_redistill:
        r = await remove_one(rag, key, delete_ledger=False)
        if r == "removed":
            redistill_removed.add(key)

    # 阶段3:后全插。只吃删成功的 redistill ∪ to_distill —— 删失败的 doc 仍在图,
    # 直插会触发 F2(dup-<hash>);pending/error 本轮跳过,下轮 diff 再来。
    rec_by_key = {key: rec for key, rec in idx.items()}
    batch: list[tuple] = [(rec_by_key[k], fp) for k, fp in d.to_distill]
    batch += [(rec_by_key[k], fp) for k, fp in d.to_redistill if k in redistill_removed]
    counters = await distill_batch(rag, batch) if batch else {}

    # 每轮末无条件再触发一次 process,让 LightRAG 自愈任意来源的 PROCESSING/FAILED 孤儿
    # (入口扫全 workspace 非终态 doc;无则廉价 return,lightrag.py:1769)。即使本轮 batch 空也调。
    paused = bool(counters.get("deferred")) or not get_breaker().allows()
    if not paused:
        try:
            await rag.apipeline_process_enqueue_documents()
        except BuildPausedError:
            paused = True

    summary = {
        "terminal": term,
        "healed": healed,
        "library": {k: v for k, v in library.items() if k != "to_distill"},
        "to_distill": len(d.to_distill),
        "to_redistill": len(d.to_redistill),
        "redistill_removed": len(redistill_removed),
        "to_remove": len(d.to_remove),
        "removed": removed_count,
        "distill": counters,
    }
    if paused:
        summary["paused"] = True
    if paused:
        log.debug("run_round: %s", summary)
    else:
        log.info("run_round: %s", summary)

    # OPTIONAL global gate (#84, nice-to-have): a round that ATTEMPTED builds (to_distill/to_redistill
    # non-empty) but landed ZERO progress AND saw batch-level (F17) enqueue/process failures is the
    # signature of a SYSTEMIC build break (embedding / LLM / graph backend down) — not per-paper bad
    # content. Per-key parking already bounds the churn within KS_REDISTILL_MAX_ATTEMPTS rounds; this
    # LOUD log surfaces the systemic cause in minutes instead of an hour of near-silent churn.
    build_errored = counters.get("errored", 0)
    round_progress = term["done"] + counters.get("queued", 0)
    if build_errored > 0 and round_progress == 0 and (d.to_distill or d.to_redistill):
        log.error(
            "run_round: BUILD APPEARS SYSTEMICALLY BROKEN — %d key(s) failed at batch enqueue/process "
            "with 0 successful builds this round. Keys PARK after %d consecutive failures "
            "(status=error_parked, no longer re-distilled). Check the embedding/LLM/graph backend NOW (#84).",
            build_errored, _max_attempts(),
        )
    if _idle_state is not None and only is None:
        _idle_state["last_full_at"] = time.monotonic()
        if _is_clean_idle_summary(summary):
            _idle_state["token"] = await _idle_state_token(rag)
        else:
            _idle_state["token"] = None
            await asyncio.to_thread(trim_process_memory)
    return summary


def _drain_did_work(picked) -> bool:
    """True iff drain_pending() actually processed or deferred at least one file."""
    if not picked:
        return False
    if isinstance(picked, dict):
        return any(bool(v) for v in picked.values())
    return bool(picked)


async def main_loop(rag, *, interval: float = DEFAULT_ROUND_INTERVAL,
                    stuck_limit: int = DEFAULT_STUCK_LIMIT) -> None:
    """进程内增量主循环(SDD §6.6):启动 ensure_schema,然后 sleep→run_round→repeat。

    取消(asyncio.CancelledError)即优雅退出(进程死=回 ledger + 重启走 reconcile 续上)。
    """
    if knowledge_runtime.knowledge_jobs_mode() == "postgres":
        knowledge_runtime.assert_knowledge_authority("catalog")
        from .catalog_service import run_catalog_loop
        return await run_catalog_loop(interval=interval)
    knowledge_runtime.assert_knowledge_authority("legacy")
    await ledger.ensure_schema()
    log.info("scheduler main_loop started (interval=%ss, stuck_limit=%s)", interval, stuck_limit)
    idle_state: dict = {"token": None, "last_full_at": 0.0}
    try:
        while True:
            try:
                await run_round(rag, stuck_limit=stuck_limit, _idle_state=idle_state)
            except BuildPausedError:
                idle_state["token"] = None
                log.debug("run_round: deferred paused build")
            except Exception:  # noqa: BLE001 — 单轮异常不杀循环;下轮重试(distill 批级已自兜底,这里是兜底之兜底)
                idle_state["token"] = None
                log.exception("run_round failed; retrying next interval")
            # #81 windowless operator-doc pickup: scan the pending dir and ingest dropped
            # textbook/notebook md against the SAME singleton rag. Runs HERE, after run_round,
            # in this one serial task → never concurrent with a paper round (single-writer
            # invariant free; off-loop inherited via ingest_document). drain_pending never
            # raises (it routes a bad file to failed/), but wrap it anyway so a bug there can
            # never break the while-True loop or the paper sync.
            # #143: an open build-path breaker defers the pickup too — ingesting now would fail
            # fast and route a good file to failed/; it stays in pending/ for a later round.
            try:
                if get_breaker().allows():
                    picked = await drain_pending(rag)
                    if _drain_did_work(picked):
                        idle_state["token"] = None
            except Exception:  # noqa: BLE001 — 兜底之兜底:pickup 异常不杀循环、不影响 paper sync
                idle_state["token"] = None
                log.exception("drain_pending failed; retrying next interval")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        log.info("scheduler main_loop cancelled — graceful stop")
        raise
