"""Reconcile / diff — the round entry (SDD §6.1 阶段1, §6.6).

`diff()` is PURE (inject `fp_of`), so it unit-tests without DB or files. It
classifies the clean vault index vs the KS ledger into the three action lists
the round drives:
  - to_distill   : key in index, not in ledger             → DISTILL_BATCH
  - to_redistill : key in both, fingerprint changed         → REDISTILL_DELETE + DISTILL_BATCH
                   OR ledger status == 'error' (F9 retry)   → 同上(指纹常未变, 只比指纹会永卡)
                   OR ledger status == 'pending_remove'     → 同上(F3 续删出口, 见下)
  - to_remove    : key in ledger, gone from index           → REMOVE_PHASE
(仅 ingest_source=paper 走 pl-镜子;textbook/web 不在此环,§6.1。)

★ status 维度(SDD §6.1 阶段1 / F9):一条 status='error' 的行(ainsert 抛错 / FAILED-无-content)
其指纹常已等于当前 extract 的 hash,只比指纹会让它既不进 to_distill(led 有行)也不进 to_redistill
(指纹没变)→ 永卡 error,违 §6.1.e『下轮重试』。故 error 行无条件并入 to_redistill 走 delete-then-insert
重投(REDISTILL 幂等)。

★ error_parked 熔断出口(#84):'error' 会**无条件**重投(上一条),对 systemic build 失败(如 embedding
stack 断,每篇每轮都失败)= 每 60s 一轮 redistill(删旧 doc)+重投+再失败,无限 churn 且不断删图。
故 ledger 在连续失败达 KS_REDISTILL_MAX_ATTEMPTS 后把该 key 泊到 **'error_parked'**(store._next_attempts_status)。
error_parked 是**终态**:它既不等于 'error' 也不等于 'pending_remove',故落到下面所有 elif 之外 = **不再进
to_redistill**,churn 被有界收住。仅指纹变化(下面第一支,内容真变)或显式 force re-ingest(清 ledger 行)能
复活它 —— 这正是我们要的:人来决定是否重试,而不是无限自动重投。
(#131: a parked row whose doc LightRAG later finishes is not a failure at all — round.reconcile_healed
writes it back `done` before this diff runs.)

★ pending_remove 死状态修(SDD §6.1.e/F3, drill-r7):pending_remove = adelete 撞 403 busy 的中间态。
其指纹被 ledger.upsert 的 COALESCE 保留(§4.1)。它有两类:
  - key∉idx(真删 REMOVE_PHASE 撞 403):由 to_remove 兜(to_remove = key∉idx),下轮 REMOVE_ONE 再删。
  - key∈idx(REDISTILL_DELETE 撞 403,旧 doc 还在图、key 没从 vault 消失):**永不进 to_remove**(它只收 key∉idx);
    若该 redistill 又源于 error(F9, fp 早已==当前 hash、status 不再 'error')→ diff 三分支全不命中 = 永卡死状态
    (旧 doc 永留图、新内容永不重插、ledger 谎报 pending_remove)。故对 *in-index* 的 pending_remove 无条件
    并入 to_redistill,与 error 同处置(先删后插,REDISTILL 幂等),给「下轮续删」一个真实出口。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional

from lightrag.base import DocStatus
from lightrag.utils import sanitize_text_for_encoding
from lightrag.utils_pipeline import compute_text_content_hash, strip_lightrag_doc_prefix

from papervault.knowledge.ingest.abstract_doc import build_abstract_doc
from papervault.knowledge.ingest.distill import clean, doc_status_of, record_duplicate
from papervault.knowledge.ingest.fingerprint import META, fingerprint
from papervault.knowledge.ingest.vault import read_extract_raw
from papervault.knowledge.ledger import store as ledger


@dataclass
class Diff:
    to_distill: list[tuple[str, str]] = field(default_factory=list)    # (key, fp)
    to_redistill: list[tuple[str, str]] = field(default_factory=list)  # (key, fp)
    to_remove: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.to_distill or self.to_redistill or self.to_remove)


def diff(
    idx: Mapping,
    led: Mapping,
    fp_of: Callable,
    *,
    only_keys: Optional[Iterable[str]] = None,
) -> Diff:
    """idx: {key: rec}; led: {key: LedgerRecord}; fp_of(rec) -> fingerprint str.

    only_keys (subset entry, blocker ③): when given, BOTH idx and led are first narrowed
    to that key set before classifying. This is what lets a probe (experiments/
    probe_round_s4.py) drive run_round over a fixed handful of keys instead of the full
    ~4k-paper vault (the run.py --keys口径). Filtering led too is load-bearing: to_remove =
    (led − idx); without narrowing led, every subset-EXTERNAL ledger row would fall into
    to_remove and get deleted. With the subset applied to both, to_remove can only ever
    name keys inside the subset, so a subset round never touches out-of-subset rows.
    """
    if only_keys is not None:
        ks = set(only_keys)
        idx = {k: v for k, v in idx.items() if k in ks}
        led = {k: v for k, v in led.items() if k in ks}
    d = Diff()
    for key, rec in idx.items():
        fp = fp_of(rec)
        existing = led.get(key)
        if existing is None:
            d.to_distill.append((key, fp))
        elif fp != existing.fingerprint:
            d.to_redistill.append((key, fp))
        elif existing.status == "error":            # F9: 指纹未变也重投(否则永卡 error,§6.1.e)
            d.to_redistill.append((key, fp))
        elif existing.status == "pending_remove":   # F3: in-index REDISTILL-撞-403 的续删出口(否则永卡死状态)
            d.to_redistill.append((key, fp))
        # #84: error_parked (连续失败达阈值后泊车) 是终态 —— 故意不进任何 list(fp 变已在上面复活它)。
        # 无 elif:落空即"本轮不动它",systemic build 失败不再每轮 redistill+删图无限 churn。
    d.to_remove = [key for key in led if key not in idx]
    return d


async def reconcile_library(rag, idx: Mapping, *, only_keys=None) -> dict:
    """Re-open terminal papers with available content but no graph doc; adopt content twins.

    Duplicate resolution comes first and covers every source. Expected absences use storage
    reads directly: LightRAG's aget_docs_by_ids warns on every deliberately missing id.
    A represented twin must be processed before a historical failure becomes terminal.
    """
    rows = await ledger.load_by_status(("done", "done_meta", "done_abstract", "error_parked",
                                       "error", "processing"))
    selected = [r for r in rows if only_keys is None or
                (r.ingest_source == "paper" and r.source_id in only_keys)]
    twins = {}
    for r in sorted(rows, key=lambda r: (r.ingest_source, r.source_id)):
        if r.status == "done" and r.fingerprint and r.fingerprint != META:
            twins.setdefault(r.fingerprint, []).append(r)
    ids = list(dict.fromkeys(r.doc_id for r in selected))
    cache = dict(zip(ids, await rag.doc_status.get_by_ids(ids))) if ids else {}

    async def status(did):
        if did not in cache:
            cache[did] = await rag.doc_status.get_by_id(did)
        return cache[did]

    async def twin_hash(did):
        st = await status(did)
        if isinstance(st, dict) and st.get("content_hash"):
            return st["content_hash"]
        doc = await rag.full_docs.get_by_id(did)
        if isinstance(doc, dict) and doc.get("content"):
            return compute_text_content_hash(strip_lightrag_doc_prefix(
                doc["content"], doc.get("parse_format")))
        return None

    async def clear_rejection(r, own):
        meta = own.get("metadata", {}) if isinstance(own, dict) else {}
        if (doc_status_of(own) == DocStatus.FAILED and
                meta.get("duplicate_kind") == "content_hash" and
                not await rag.full_docs.get_by_id(r.doc_id)):
            # Post-parse rejection uses the source's own id (not a dup-* marker) and
            # deletes its full_docs body. It is an audit row, never a graph document.
            await rag.doc_status.delete([r.doc_id])
            cache[r.doc_id] = None
            return None
        return own

    counters = {"reopened": 0, "duplicates": 0, "to_distill": []}
    for r in selected:
        own = await status(r.doc_id)
        if doc_status_of(own) == DocStatus.PROCESSED:
            continue
        if r.duplicate_of:
            own = await clear_rejection(r, own)
            if (doc_status_of(await status(r.duplicate_of)) == DocStatus.PROCESSED and
                    await twin_hash(r.duplicate_of) == r.duplicate_content_hash):
                continue  # represented content is still in the graph; no warning or rewrite
            # This source is no longer represented. Do not re-adopt a changed twin merely
            # because the old raw fingerprints match; schedule the available source below.
        twin_id = None
        if own is None and not r.duplicate_of:
            for twin in twins.get(r.fingerprint, []):
                if (twin.ingest_source, twin.source_id) == (r.ingest_source, r.source_id):
                    continue
                if doc_status_of(await status(twin.doc_id)) == DocStatus.PROCESSED:
                    twin_id = twin.doc_id
                    break
        # LightRAG's post-parse rejection can leave a FAILED own row. Enqueue rejection
        # instead leaves a dup-* marker on the source filename, with the same metadata.
        marker = own
        if marker is None and not r.duplicate_of:
            basename = r.source_id if r.ingest_source == "paper" else r.doc_id
            match = await rag.doc_status.get_doc_by_file_basename(basename)
            marker = match[1] if match else None
        meta = marker.get("metadata", {}) if isinstance(marker, dict) else {}
        original = meta.get("original_doc_id") if meta.get("duplicate_kind") == "content_hash" else None
        if not r.duplicate_of and original and original != r.doc_id and doc_status_of(await status(original)) == DocStatus.PROCESSED:
            # Post-parse rejection records the rejected content's hash. A historical
            # rejection is no longer proof of representation after that twin is rebuilt.
            rejected_hash = marker.get("content_hash")
            if not rejected_hash and r.ingest_source == "paper" and r.source_id in idx:
                rec = idx[r.source_id]
                raw = read_extract_raw(rec)
                text = clean(raw) if raw and raw.strip() else build_abstract_doc(rec)
                if text:
                    rejected_hash = compute_text_content_hash(sanitize_text_for_encoding(text))
            same_fingerprint = any(t.doc_id == original for t in twins.get(r.fingerprint, []))
            if ((rejected_hash and rejected_hash == await twin_hash(original)) or
                    (not rejected_hash and same_fingerprint)):
                twin_id = original
            else:
                twin_id = None
                own = await clear_rejection(r, own)
        if twin_id:
            content_hash = await twin_hash(twin_id)
            if content_hash:
                await clear_rejection(r, own)
                await record_duplicate(r.ingest_source, r.source_id, r.doc_id,
                                       twin_id, content_hash, r.fingerprint)
                counters["duplicates"] += 1
                continue
        if own is not None or r.ingest_source != "paper" or r.source_id not in idx:
            continue
        if r.status not in ("done", "done_meta", "done_abstract", "error_parked") or r.attempts != 0:
            continue
        fp = fingerprint(idx[r.source_id])
        if fp != META:
            # Queue directly without deleting a nonexistent doc or counting a failure.
            # distill_batch persists processing only when this round can actually build.
            counters["reopened"] += 1
            counters["to_distill"].append((r.source_id, fp))
    return counters
