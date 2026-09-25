"""S4 scheduler — round orchestration + doc_status terminal writeback (SDD §6.1/§6.6).

`run_round()` wires the pure `diff()` (scheduler/reconcile.py) to the S2 atoms
(ingest/distill.py: distill_batch / remove_one) into one incremental-sync round,
preserving the §6.6 delete-then-insert mutex (先全删, 后全插).

`reconcile_terminal()` is the ONLY way a ledger row leaves `processing`:
apipeline_enqueue_documents only returns a track_id, never `done`. So each round
first reads LightRAG doc_status terminal states for the still-`processing` rows and
writes back `done` (PROCESSED) / `error` (FAILED). 中途态(PROCESSING/PENDING)留下轮;
查不到 / 非预期态(PREPROCESSED, F10)计入 stuck_guard, 连续 N 轮不前进 → error
(防 enqueue 早返 / 内容去重 F16 孤儿成为永久死状态).

`reconcile_healed()` (#131) closes the other drift: LightRAG's own pipeline retries a
FAILED-with-content doc until it is PROCESSED, but a ledger row that already reached
`error` / `error_parked` was never revisited — the ledger kept reporting a failure (and an
`error` row was re-distilled, deleting the good doc) while the graph held the paper. Any
`error` / `error_parked` row whose doc is PROCESSED is written back `done`.

Round shape (SDD §13):
  reconcile_terminal → reconcile_healed → diff(含 status 维度) → REMOVE
  → REDISTILL_DELETE(只回报删成功) → DISTILL_BATCH(只吃删成功的 redistill ∪ to_distill)
  → 末尾无条件 process 一次(自愈孤儿).

Prod-safety: get_graph() (passed in as `rag`) already gated by assert_safe_workspace().
This module never opens a graph itself; the caller owns the instance + workspace gate.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Iterable, Optional

from lightrag.base import DocStatus

from papervault.knowledge.ingest.distill import distill_batch, doc_status_of, remove_one
from papervault.knowledge.ingest.fingerprint import fingerprint
from papervault.knowledge.ingest.vault import load_clean_index
from papervault.knowledge.ledger import store as ledger
from papervault.knowledge.ledger.store import _max_attempts
from papervault.knowledge.scheduler.opdoc_pickup import drain_pending
from papervault.knowledge.scheduler.reconcile import Diff, diff

log = logging.getLogger("ks.scheduler.round")

PAPER = "paper"
DEFAULT_STUCK_LIMIT = 3        # processing 行连续 N 轮不前进 → error(§6.6 永卡兜底)
DEFAULT_ROUND_INTERVAL = 60.0  # main_loop sleep 秒

# 进程内 stuck 计数(重启从 0 重数 —— 重启本身触发 LightRAG 自愈,见 §6.6)。
_stuck: dict[str, int] = defaultdict(int)


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
    led = await ledger.load(PAPER)
    proc = [r for r in led.values() if r.status == "processing"]
    if only_keys is not None:
        ks = set(only_keys)
        proc = [r for r in proc if r.source_id in ks]
    counters = {"done": 0, "error": 0, "pending": 0, "stuck_error": 0}
    if not proc:
        return counters

    by_doc_id = {r.doc_id: r for r in proc}
    statuses = await rag.aget_docs_by_ids(list(by_doc_id.keys()))  # {doc_id: DocProcessingStatus}; 未命中省略

    seen_terminal: set[str] = set()
    for doc_id, rec in by_doc_id.items():
        st = statuses.get(doc_id)
        ds = doc_status_of(st)  # dict-aware (SDD §6.6 ★): aget_docs_by_ids returns plain dicts
        if ds == DocStatus.PROCESSED:
            await ledger.upsert(PAPER, rec.source_id, doc_id=doc_id, status="done")
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


async def reconcile_healed(rag, *, only_keys: Optional[Iterable[str]] = None) -> dict:
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
    rows = await ledger.load_by_status(_FAILED_LEDGER_STATUSES)
    if only_keys is not None:
        ks = set(only_keys)
        rows = [r for r in rows if r.ingest_source == PAPER and r.source_id in ks]
    counters = {"healed": 0}
    if not rows:
        return counters
    statuses = await rag.aget_docs_by_ids([r.doc_id for r in rows])
    for rec in rows:
        if doc_status_of(statuses.get(rec.doc_id)) == DocStatus.PROCESSED:
            await ledger.upsert(rec.ingest_source, rec.source_id, doc_id=rec.doc_id, status="done")
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
) -> dict:
    """一个增量同步 round(SDD §6.1/§6.6)。进程内串行,删插互斥。

    only_keys (subset entry, blocker ③): None = 全量(production main_loop). When given, the
    whole round — reconcile_terminal scan, diff classification (so to_remove stays inside
    the subset), REMOVE / REDISTILL / DISTILL — is confined to that key set. This is the
    run.py --keys口径 that lets experiments/probe_round_s4.py真跑 run_round over 3-5 fixed
    keys instead of the full ~4k-paper vault (which would burn the §6.5 throughput budget
    and break PROD-SAFETY rule (4)'s small-subset/serial intent).
    """
    only = set(only_keys) if only_keys is not None else None

    # 阶段0(b):先收口上一轮/重启遗留的 processing(subset 时只扫 subset 行)。
    term = await reconcile_terminal(rag, stuck_limit=stuck_limit, only_keys=only)
    # #131:失败行(error/error_parked)的 doc 已被 LightRAG 自愈为 PROCESSED → done(先于 diff,
    # 免得 error 行被 redistill 删掉好 doc)。
    healed = await reconcile_healed(rag, only_keys=only)

    # 阶段1:纯 KS diff(含 status 维度,error 行重投)。subset 时 idx+led 都先裁到 only。
    idx = load_clean_index()
    led = await ledger.load(PAPER)
    d: Diff = diff(idx, led, fp_of=fingerprint, only_keys=only)

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
    await rag.apipeline_process_enqueue_documents()

    summary = {
        "terminal": term,
        "healed": healed,
        "to_distill": len(d.to_distill),
        "to_redistill": len(d.to_redistill),
        "redistill_removed": len(redistill_removed),
        "to_remove": len(d.to_remove),
        "removed": removed_count,
        "distill": counters,
    }
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
    return summary


async def main_loop(rag, *, interval: float = DEFAULT_ROUND_INTERVAL,
                    stuck_limit: int = DEFAULT_STUCK_LIMIT) -> None:
    """进程内增量主循环(SDD §6.6):启动 ensure_schema,然后 sleep→run_round→repeat。

    取消(asyncio.CancelledError)即优雅退出(进程死=回 ledger + 重启走 reconcile 续上)。
    """
    await ledger.ensure_schema()
    log.info("scheduler main_loop started (interval=%ss, stuck_limit=%s)", interval, stuck_limit)
    try:
        while True:
            try:
                await run_round(rag, stuck_limit=stuck_limit)
            except Exception:  # noqa: BLE001 — 单轮异常不杀循环;下轮重试(distill 批级已自兜底,这里是兜底之兜底)
                log.exception("run_round failed; retrying next interval")
            # #81 windowless operator-doc pickup: scan the pending dir and ingest dropped
            # textbook/notebook md against the SAME singleton rag. Runs HERE, after run_round,
            # in this one serial task → never concurrent with a paper round (single-writer
            # invariant free; off-loop inherited via ingest_document). drain_pending never
            # raises (it routes a bad file to failed/), but wrap it anyway so a bug there can
            # never break the while-True loop or the paper sync.
            try:
                await drain_pending(rag)
            except Exception:  # noqa: BLE001 — 兜底之兜底:pickup 异常不杀循环、不影响 paper sync
                log.exception("drain_pending failed; retrying next interval")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        log.info("scheduler main_loop cancelled — graceful stop")
        raise
