"""Single-writer identity consolidation with persisted quiescence (spec A1/A3).

The library remains a single-process writer. The durable budget event is the
irreversible point; an uncertain following index save requires operator repair,
never an automatic repeat. Checkpoints release the write lock while work drains.
"""

from __future__ import annotations

import asyncio
import math
import inspect
import threading
import time
import uuid
from dataclasses import replace
from functools import wraps
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from papervault.config import merge_v2_enabled

from .state_v2 import Merge, Policy, Retrieval, TerminalState, check_index_readable, project_domain_policy, retain_observed_restrictions

_sync_writer_lock = threading.RLock()


class MergeIntent(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    merge_id: str = Field(min_length=1)
    canonical_key: str = Field(min_length=1)
    absorbed_key: str = Field(min_length=1)
    started_at: float
    deadline: float
    canonical_quarantine: bool = False
    absorbed_quarantine: bool = False
    status: Literal["waiting", "expired", "uncertain"] = "waiting"
    metadata: dict = Field(default_factory=dict)


class MergeBudget(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    merge_id: str = Field(min_length=1)
    generation: int = Field(ge=1)
    spent: int = Field(ge=0)
    budget: int = Field(ge=0)


def resolve_alias(library, key: str):
    """Return the canonical row, refusing broken or cyclic redirects."""
    check_index_readable(library.root)
    seen = set()
    while key and key not in seen:
        seen.add(key)
        paper = library.get(key)
        if paper is None:
            return None
        state = paper.acquisition_state
        if state.merge == Merge.CANONICAL and not state.alias_key:
            return paper
        if state.merge != Merge.ALIAS:
            return None
        key = state.alias_key
    return None


def blocked(paper) -> bool:
    """Persisted ownership fences survive turning off the feature flag."""
    return bool(paper and (paper.merge_intent is not None
                          or ((merge_v2_enabled() or paper.merge_budget is not None)
                              and (paper.acquisition_state.merge == Merge.ALIAS or paper.acquisition_state.alias_key))))


def guard_execution(work):
    """Count the actual executor lifetime, including threads whose awaiter left.

    Format refusal is independent of merge tracking and feature flags.
    Lifetime tracking remains a merge-specific fence, not an admission predicate.
    Cancelled async extraction is uncertain, so its merge stays quarantined.
    """
    def enter(paper, library):
        tracked = merge_v2_enabled() or paper.merge_budget is not None or blocked(paper)
        if not tracked:
            # Kill switches cannot authorize an old writer to downgrade newer data.
            check_index_readable(library.root)
            return False
        with library._merge_execution_lock:
            check_index_readable(library.root)
            if blocked(library.get(paper.key) or paper):
                return None
            from .claims_v2 import BudgetLedger, LedgerError, record_generation
            current = library.get(paper.key) or paper
            try:
                # Detached campaign staging uses its live vault's ledger;
                # persisted claim phases never substitute for current ownership.
                ledger = getattr(library, "_merge_ledger", None) or BudgetLedger(library.root)
                owned = (not ledger.merged_into(paper.key)
                         and ledger.latest_generation(paper.key) == record_generation(current))
            except LedgerError:
                owned = False
            if not owned:
                return None
            library._merge_active[paper.key] = library._merge_active.get(paper.key, 0) + 1
        return True

    def leave(paper, library, tracked):
        if tracked:
            with library._merge_execution_lock:
                remaining = library._merge_active[paper.key] - 1
                if remaining:
                    library._merge_active[paper.key] = remaining
                else:
                    library._merge_active.pop(paper.key)

    if inspect.iscoroutinefunction(work):
        @wraps(work)
        async def asynchronous(paper, library, *args, **kwargs):
            tracked = enter(paper, library)
            if tracked is None:
                return None
            try:
                return await work(paper, library, *args, **kwargs)
            except asyncio.CancelledError:
                if tracked:
                    tracked = False  # keep resources counted before any cancellable await
                    from .services import concurrency
                    async def persist_uncertainty():
                        async with concurrency.lib_write_lock:
                            check_index_readable(library.root)
                            current = library.get(paper.key)
                            if current is not None:
                                intent = current.merge_intent
                                keys = (intent.canonical_key, intent.absorbed_key) if intent else (paper.key,)
                                for key in keys:
                                    row = library.get(key)
                                    if row is not None:
                                        row.terminal_state = replace(row.acquisition_state, quarantine=True)
                                        if row.merge_intent:
                                            row.merge_intent = row.merge_intent.model_copy(update={"status": "uncertain"})
                                library.save(force=True, durable=True)
                    publication = asyncio.create_task(persist_uncertainty())
                    while not publication.done():
                        try:
                            await asyncio.shield(publication)
                        except asyncio.CancelledError:
                            continue
                    publication.result()
                raise
            finally:
                leave(paper, library, tracked)
        return asynchronous

    @wraps(work)
    def synchronous(paper, library, *args, **kwargs):
        tracked = enter(paper, library)
        if tracked is None:
            return False
        try:
            return work(paper, library, *args, **kwargs)
        finally:
            leave(paper, library, tracked)
    return synchronous


def write_locked(call):
    """Bridge standalone synchronous callers to the existing async writer lock.

    No event-loop blocking and no cross-thread waiting on an asyncio lock:
    runtime callers must already own it. Standalone thread-pool callers serialize
    here before using the same lock in a short-lived loop.
    """
    from .services import concurrency
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        with _sync_writer_lock:
            if concurrency.lib_write_lock.locked():
                raise RuntimeError("active async library writer; schedule consolidation on its loop")
            async def run():
                async with concurrency.lib_write_lock:
                    return call()
            return asyncio.run(run())
    if not concurrency.lib_write_lock.owned():
        raise RuntimeError("identity consolidation requires the library write-lock owner")
    return call()


class MergeStore:
    def __init__(self, library, *, ledger=None, clock=time.time):
        from .claims_v2 import BudgetLedger
        from .services import concurrency
        self.library, self.clock = library, clock
        check_index_readable(library.root)
        self.ledger = ledger or BudgetLedger(library.root, clock=clock)
        self.lock = concurrency.lib_write_lock

    def _require_writer(self):
        if not merge_v2_enabled():
            raise ValueError("PAPERVAULT_MERGE_V2 is disabled")
        if not self.lock.owned():
            raise RuntimeError("merge writer does not own the library write-lock")
        check_index_readable(self.library.root)

    def _pair(self, intent):
        if intent is None:
            return None
        papers = [self.library.get(key) for key in (intent.canonical_key, intent.absorbed_key)]
        if any(p is None or p.merge_intent is None or p.merge_intent.merge_id != intent.merge_id
               or p.acquisition_state.merge != Merge.CANONICAL for p in papers):
            return None
        return papers

    def _can_quiesce(self, key):
        from .claims_v2 import record_generation
        paper = self.library.get(key)
        if (paper is None or paper.acquisition_state.merge != Merge.CANONICAL
                or paper.acquisition_state.alias_key
                or (paper.acquisition_state.quarantine and paper.merge_intent is None)
                or self.ledger.merged_into(key)
                or record_generation(paper) != self.ledger.latest_generation(key)):
            return None
        return paper

    def _observe(self, paper, intent, metadata):
        if not metadata:
            return intent
        owned = {"key", "reopen_claim", "merge_intent", "merge_budget"}
        values = {k: v for k, v in metadata.items() if k not in owned and v is not None}
        identities = {"doi", "arxiv_id", "paper_id", "url", "inspire_record_id", "inspire_document_urls",
                      "title", "authors", "year", "download_status", "pdf_path", "md_path", "txt_path",
                      "extraction_completion_id", "md_engine", "md_engine_version"}
        self.library._merge(paper, {k: v for k, v in values.items() if k not in identities})
        if "domain_status" in values:
            observed = project_domain_policy(TerminalState(policy=Policy.CLEAR), values)
            paper.terminal_state = retain_observed_restrictions(paper.unquiesced_state, observed)
        pending = {**values, **intent.metadata}  # first observed identity value wins
        quarantine = values.get("terminal_state")
        extra_quarantine = (quarantine.get("quarantine") is True if isinstance(quarantine, dict)
                            else getattr(quarantine, "quarantine", False))
        return intent.model_copy(update={"metadata": pending,
                                        "canonical_quarantine": intent.canonical_quarantine or extra_quarantine})

    def _quiesce(self, canonical_key, absorbed_key, *, wait_seconds=60, metadata=None):
        self._require_writer()
        if not math.isfinite(wait_seconds) or wait_seconds <= 0:
            raise ValueError("merge wait must be positive and finite")
        if canonical_key == absorbed_key:
            return None
        left, right = (self._can_quiesce(key) for key in (canonical_key, absorbed_key))
        if left is None or right is None:
            return None
        if left.merge_intent or right.merge_intent:
            intent = left.merge_intent
            if not (intent and intent.canonical_key == canonical_key
                    and intent.absorbed_key == absorbed_key and self._pair(intent)):
                return None
            updated = self._observe(left, intent, metadata)
            if metadata:
                left.merge_intent = right.merge_intent = updated
                self.library.save(force=True, durable=True)
            return updated
        intent = MergeIntent(merge_id=uuid.uuid4().hex, canonical_key=canonical_key,
                             absorbed_key=absorbed_key, started_at=float(self.clock()),
                             deadline=float(self.clock() + wait_seconds),
                             canonical_quarantine=left.acquisition_state.quarantine,
                             absorbed_quarantine=right.acquisition_state.quarantine)
        with self.library._merge_execution_lock:
            intent = self._observe(left, intent, metadata)
            for paper in (left, right):
                # The intent projects an admission quarantine; do not mark
                # existing dispatched execution as uncertain before it settles.
                paper.merge_intent = intent
            self.library.save(force=True, durable=True)
        return intent

    async def quiesce(self, canonical_key, absorbed_key, *, wait_seconds=60):
        async with self.lock:
            return self._quiesce(canonical_key, absorbed_key, wait_seconds=wait_seconds)

    async def restart(self, intent: MergeIntent, *, wait_seconds=60):
        """Explicitly replace an expired wait, keeping both admission fences.

        This does not establish drain or release claims, quarantine or budgets.
        Uncertain execution/commit still requires separate operator recovery.
        """
        async with self.lock:
            self._require_writer()
            if not math.isfinite(wait_seconds) or wait_seconds <= 0:
                raise ValueError("merge wait must be positive and finite")
            with self.library._merge_execution_lock:
                papers = self._pair(intent)
                if (papers is None or intent.status != "expired"
                        or any(paper.merge_intent != intent or paper.unquiesced_state.quarantine
                               or self._can_quiesce(paper.key) is None for paper in papers)):
                    return None
                now = float(self.clock())
                fresh = intent.model_copy(update={"merge_id": uuid.uuid4().hex, "status": "waiting",
                                                  "started_at": now, "deadline": now + wait_seconds})
                for paper in papers:
                    paper.merge_intent = fresh
                self.library.save(force=True, durable=True)
                self.library.log({"event": "merge_intent_restarted", "previous_merge_id": intent.merge_id,
                                  "merge_id": fresh.merge_id, "canonical_key": fresh.canonical_key,
                                  "absorbed_key": fresh.absorbed_key})
                return fresh

    def _drain_ready(self, intent):
        self._require_writer()
        papers = self._pair(intent)
        if papers is None:
            return False
        current = papers[0].merge_intent
        if current.status != "waiting":
            return False
        if self.clock() >= current.deadline:
            expired = current.model_copy(update={"status": "expired"})
            for paper in papers:
                paper.merge_intent = expired
            self.library.save(force=True, durable=True)
            return False
        keys = (current.canonical_key, current.absorbed_key)
        if (self.ledger.merged_into(keys[0]) or self.ledger.merged_into(keys[1])
                or self.ledger.outstanding(keys)):
            return False
        from .services import concurrency
        for paper in papers:
            if paper.reopen_claim and paper.reopen_claim.phase in {"queued", "dispatched"}:
                return False
            if any(concurrency.is_in_flight(stage + paper.key) for stage in ("", "dl:", "ex:")):
                return False
            if self.library._merge_active.get(paper.key, 0):
                return False
            if any(paper.key in queue._pending for queue in self.library._merge_queues):
                return False
        return True

    async def drain_ready(self, intent):
        async with self.lock:
            with self.library._merge_execution_lock:
                return self._drain_ready(intent)

    def _merge_records(self, canonical_key, absorbed_key, *, wait_seconds=60, metadata=None):
        self._require_writer()
        intent = self._quiesce(canonical_key, absorbed_key, wait_seconds=wait_seconds, metadata=metadata)
        with self.library._merge_execution_lock:
            if not self._drain_ready(intent):
                return None
            left, right = self._pair(intent)
            md_fields = ("md_path", "extraction_completion_id", "md_engine", "md_engine_version")
            selected_md = {field: getattr(left, field) for field in md_fields}
            artifacts = {}
            for kind in ("pdf", "md", "txt"):
                source = getattr(self.library, kind + "_path")(right.key)
                target = getattr(self.library, kind + "_path")(left.key)
                if not target.is_file() and source.is_file() and source.stat().st_size:
                    artifacts[kind] = (right.md_path if kind == "md" and right.extraction_completion_id
                                       else str(source.relative_to(self.library.root)))
            state = retain_observed_restrictions(
                replace(left.unquiesced_state, quarantine=left.unquiesced_state.quarantine or intent.canonical_quarantine),
                replace(right.unquiesced_state, quarantine=right.unquiesced_state.quarantine or intent.absorbed_quarantine))
            event = self.ledger.append_merge(canonical_key, absorbed_key, merge_id=intent.merge_id)
            budget = MergeBudget(merge_id=intent.merge_id, generation=event["generation_after"],
                                 spent=event["spent"], budget=event["budget"])
            # Work is settled before the existing in-place blank-fill writer runs.
            self.library._merge(left, right.model_dump())
            if intent.metadata:
                self.library._merge(left, intent.metadata)
            if "md" in artifacts:
                selected_md = {field: getattr(right, field) for field in md_fields}
            for field, value in selected_md.items():
                setattr(left, field, value)
            for kind, path in artifacts.items():
                setattr(left, kind + "_path", path)
            rank = {Retrieval.ACQUIRED: 1, Retrieval.VERIFIED: 2}
            if artifacts and rank.get(right.acquisition_state.retrieval, 0) > rank.get(state.retrieval, 0):
                left.download_status = right.download_status
                state = replace(state, retrieval=right.acquisition_state.retrieval, cause=right.acquisition_state.cause)
            left.terminal_state = state
            right.terminal_state = replace(right.acquisition_state, merge=Merge.ALIAS,
                                           alias_key=canonical_key, quarantine=True)
            for paper in (left, right):
                paper.merge_budget = budget
                paper.merge_intent = None
            self.library._reindex(left)
            for index in (self.library._by_doi, self.library._by_arxiv, self.library._by_title):
                for value, key in list(index.items()):
                    if key == absorbed_key:
                        index[value] = canonical_key
            try:
                self.library.save(force=True, durable=True)
            except Exception:
                uncertain = intent.model_copy(update={"status": "uncertain"})
                for paper in (left, right):
                    paper.merge_intent = uncertain
                    paper.terminal_state = replace(paper.acquisition_state, quarantine=True)
                raise
            return left

    async def merge_records(self, canonical_key, absorbed_key, *, wait_seconds=60):
        """One bounded-wait checkpoint; retry before the persisted deadline only.

        The caller can await other work between checkpoints. This never sleeps
        under the library lock or mistakes deadline expiry for drain evidence.
        """
        async with self.lock:
            return self._merge_records(canonical_key, absorbed_key, wait_seconds=wait_seconds)
