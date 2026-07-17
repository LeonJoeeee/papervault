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

Round shape (SDD §13):
  reconcile_terminal → diff(含 status 维度) → REMOVE → REDISTILL_DELETE(只回报删成功)
  → DISTILL_BATCH(只吃删成功的 redistill ∪ to_distill) → 末尾无条件 process 一次(自愈孤儿).

Prod-safety: get_graph() (passed in as `rag`) already gated by assert_safe_workspace().
This module never opens a graph itself; the caller owns the instance + workspace gate.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Iterable, Optional

from lightrag.base import DocStatus

from papervault.knowledge.ingest.distill import distill_batch, remove_one
from papervault.knowledge.ingest.fingerprint import fingerprint
from papervault.knowledge.ingest.vault import load_clean_index
from papervault.knowledge.ledger import store as ledger
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
        ds = _doc_status(st)  # dict-aware (SDD §6.6 ★): aget_docs_by_ids returns plain dicts
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


def _doc_status(st) -> object | None:
    """Pull `.status` out of an aget_docs_by_ids entry (SDD §6.6 ★ contract).

    LightRAG 1.4.16's aget_docs_by_ids is type-hinted dict[str, DocProcessingStatus] but at
    RUNTIME returns {doc_id: plain dict}: it passes doc_status.get_by_id() straight through
    (lightrag.py:3192/3209) and both configured backends return a plain dict
    (PGDocStatusStorage.get_by_id postgres_impl.py:3818 `return dict(...)`;
    JsonDocStatusStorage.get_by_id json_doc_status_impl.py:238 `return self._data.get(id)`),
    whose `status` is a BARE STRING ('processed'/'failed'/…), not a DocProcessingStatus.
    So `getattr(st, "status")` on the dict is always None — which silently misclassifies every
    PROCESSED doc, trips stuck_guard, and flips successful docs to error (breaks §6.5 closure).
    Use dict-subscript; fall back to getattr only for an object-shaped st. DocStatus is a
    str-Enum, so the bare string compares equal to DocStatus.PROCESSED etc. downstream.
    """
    if st is None:
        return None
    if isinstance(st, dict):
        return st.get("status")
    return getattr(st, "status", None)


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
        "to_distill": len(d.to_distill),
        "to_redistill": len(d.to_redistill),
        "redistill_removed": len(redistill_removed),
        "to_remove": len(d.to_remove),
        "removed": removed_count,
        "distill": counters,
    }
    log.info("run_round: %s", summary)
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
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        log.info("scheduler main_loop cancelled — graceful stop")
        raise
