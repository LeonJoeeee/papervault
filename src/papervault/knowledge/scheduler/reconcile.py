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
    d.to_remove = [key for key in led if key not in idx]
    return d
