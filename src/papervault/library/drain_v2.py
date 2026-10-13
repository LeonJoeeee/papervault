"""Read-only, single-writer-process evidence for the A3 deploy drain barrier."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from .claims_v2 import BudgetLedger
from .services import concurrency
from .state_v2 import check_index_readable


@dataclass(frozen=True)
class DrainEvidence:
    checked_at: float
    intake_stopped: bool
    execution_tracking_verified: bool
    claims: tuple[str, ...]
    in_flight: tuple[str, ...]
    executors: tuple[str, ...]
    merge_intents: tuple[str, ...]
    queued: int
    unfinished: int
    pending: int

    @property
    def drained(self) -> bool:
        return (self.intake_stopped and self.execution_tracking_verified
                and not (self.claims or self.in_flight or self.executors
                or self.merge_intents or self.queued or self.unfinished or self.pending))


async def verify_drained(library, *, stop_intake: Callable[[], Awaitable[bool]],
                         execution_tracking_verified: bool = False,
                         ledger: BudgetLedger | None = None, clock=time.time) -> DrainEvidence:
    """Seal intake first, then snapshot; a failed seal can never pass.

    The callback must stop *all* acquisition writers/producers and keep intake
    sealed through the flip. It must return exactly True on confirmation. This
    helper runs in the owning process with its live queue/executor registries;
    opening a fresh Library elsewhere cannot certify a running service's drain.
    No work is dequeued, settled, refunded or released here. Invalid accounting
    raises rather than certifying an empty ledger.

    execution_tracking_verified must be exactly True only when the deployment
    has verified all executor lifetimes were tracked from admission (including
    standalone writers). Empty registries or flags enabled only now cannot
    establish that history; without this pre-flight evidence, drained is False.
    """
    sealed = await stop_intake()
    async with concurrency.lib_write_lock:
        with library._merge_execution_lock:
            check_index_readable(library.root)
            ledger = ledger or BudgetLedger(library.root)
            rows = ledger.events()
            keys = {r["claim"]["key"] for r in rows if "claim" in r}
            papers = library.all_papers(include_quarantined=True)
            claims = {p.reopen_claim.lease_id for p in papers
                      if p.reopen_claim and p.reopen_claim.phase in {"queued", "dispatched"}}
            claims.update(t.lease_id for t in ledger.outstanding(keys))
            queues = tuple(library._merge_queues)
            executors = {k for k, count in library._merge_active.items() if count}
            executors.update(k for k, count in getattr(library, "_stoprule_active", {}).items() if count)
            return DrainEvidence(
                checked_at=clock(), intake_stopped=sealed is True,
                execution_tracking_verified=execution_tracking_verified is True,
                claims=tuple(sorted(claims)), in_flight=tuple(sorted(concurrency.in_flight_keys())),
                executors=tuple(sorted(executors)),
                merge_intents=tuple(sorted(p.key for p in papers if p.merge_intent is not None)),
                queued=sum(q.qsize() for q in queues),
                unfinished=sum(q._queue._unfinished_tasks for q in queues),
                pending=sum(len(q._pending) for q in queues))
