"""Explicit bounded campaigns and generation-fenced staged publication."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from papervault.config import reopen_v2_enabled, stoprules_v2_enabled

from .claims_v2 import BudgetLedger, ClaimStore, LedgerError, record_generation, reopenable
from .services import concurrency
from .services.identity_backfill import read_identity_records
from .state_v2 import Merge, Retrieval, effective_policy, resolved_identity
from .store import Library, _fsync_dir
from .telemetry_v2 import identity_snapshot, publication_fields, record_censored

_IDENTITY_FIELDS = {"key", "doi", "arxiv_id", "paper_id", "inspire_record_id", "url", "title", "authors", "year"}
_RESULT_FIELDS = {"download_source", "download_source_member", "ads_availability", "arxiv_withdrawal",
                  "firecrawl_pdf_hunt_exhausted", "txt_engine", "txt_engine_version", "md_engine", "md_engine_version"}


def _require_enabled():
    if not reopen_v2_enabled():
        raise ValueError("PAPERVAULT_REOPEN_V2 is disabled")


def _selection(root, cap, ledger):
    # Both modes inspect the complete snapshot; domain exclusions and scoped
    # overrides are decided by reopenable(), rather than the library clean view.
    papers = read_identity_records(Path(root))
    eligible = []
    for paper in papers:
        if not reopenable(paper):
            continue
        if paper.merge_budget is not None:
            account = ledger.key_budget(paper.key)
            if account["spent"] >= account["budget"]:
                continue
        eligible.append(paper)
    selected = sorted(eligible, key=lambda paper: paper.key)[:cap]
    return papers, selected


def inspect_campaign(root: Path, *, cap: int, clock=time.time) -> dict:
    """Read-only selection and authorization validation; no constructor/lock writes."""
    _require_enabled()
    ledger = BudgetLedger(root, clock=clock)
    ledger.check_campaign(cap=cap)
    papers, selected = _selection(root, cap, ledger)
    if any(p.reopen_claim and p.reopen_claim.phase in {"queued", "dispatched"} for p in papers):
        raise LedgerError("unsettled campaign claims require recovery")
    for paper in selected:
        generation = record_generation(paper)
        if generation != ledger.latest_generation(paper.key):
            raise LedgerError("index generation disagrees with the budget ledger")
    return {"dry_run": True, "cap": cap, "keys": [paper.key for paper in selected],
            "selected": len(selected), "completed": 0}


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


class StagedAttempt:
    """Detached record and isolated vault; assets are copied, never linked.

    Publication carries deltas and tombstones: unchanged copies cannot overwrite
    newer live extracts, and rejected text is removed. Cancellation retains the
    stage until verified drain or human recovery.
    """

    def __init__(self, library: Library, token):
        self.live, self.token = library, token
        self.before = library.get(token.key).model_copy(deep=True)
        self.root = library.root / ".reopen-staging" / token.lease_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.library = Library(self.root)
        self.library._telemetry_live = library
        # The isolated artifacts do not own accounting. Merge generation
        # fencing must consult the same durable history as the live executor.
        self.library._merge_ledger = BudgetLedger(library.root)
        candidate = self.before.model_copy(deep=True)
        candidate.terminal_state = replace(candidate.acquisition_state, retrieval=Retrieval.ELIGIBLE)
        self.library._papers[candidate.key] = candidate
        self.library._reindex(candidate)
        self.baseline = {}
        for kind in ("pdf", "md", "txt"):
            source = getattr(library, f"{kind}_path")(token.key)
            self.baseline[kind] = _digest(source)
            if source.is_file():
                shutil.copyfile(source, self.path(kind))

    def path(self, kind: str) -> Path:
        return getattr(self.library, f"{kind}_path")(self.token.key)

    def cleanup(self) -> None:
        shutil.rmtree(self.root)

    def _publication_refusal(self, store, paper, result):
        state = paper.unquiesced_state
        if (state.quarantine or state.retrieval != Retrieval.IN_PROGRESS
                or state.merge != Merge.CANONICAL or state.alias_key
                or not effective_policy(state, key=paper.key) or not resolved_identity(paper)
                or paper.model_dump(include=_IDENTITY_FIELDS) != self.before.model_dump(include=_IDENTITY_FIELDS)
                or not store.ledger.admitted(self.token)):
            return "publication policy/identity/budget fence"
        for field in _IDENTITY_FIELDS:
            old, new = getattr(self.before, field), getattr(result, field)
            if old != new and (field not in {"doi", "arxiv_id"} or old):
                return "result changed an adopted identity"
        for name in ("doi", "arxiv_id"):
            if not getattr(self.before, name) and getattr(result, name):
                holder = self.live.find(**{name: getattr(result, name)})
                if holder is not None and holder.key != paper.key:
                    return "verified identity collides at publication"
        return ""

    async def writeback(self, store: ClaimStore) -> bool:
        async with store.lock:
            paper = store.current(self.token)
            if (not reopen_v2_enabled() or paper is None
                    or paper.reopen_claim.phase != "dispatched"):
                return False  # Replay/stale generation cannot mutate even a marker.
            if store.clock() >= paper.reopen_claim.lease_expiry:
                store._quarantine(paper, "completion after lease expiry")
                return False
            result = self.library.get(self.token.key)
            refusal = self._publication_refusal(store, paper, result)
            if refusal:
                store._quarantine(paper, refusal)
                return False
            changed = {}
            for kind, baseline in self.baseline.items():
                digest = _digest(self.path(kind))
                if digest != baseline:
                    live_path = getattr(self.live, f"{kind}_path")(paper.key)
                    if _digest(live_path) != baseline:
                        store._quarantine(paper, "live artifact changed during execution")
                        return False
                    changed[kind] = digest
            fills = {name: getattr(result, name) for name in ("doi", "arxiv_id")
                     if not getattr(self.before, name) and getattr(result, name)}
            if fills:
                # The publication fences above authorize only this existing
                # dispatched result, including same-row verified blank fills.
                applied = self.live._fill_verified_identifiers(paper, **fills,
                    provenance={"operation": "reopen", "generation": self.token.generation})
                if applied != fills:
                    store._quarantine(paper, "identity fill refused at publication")
                    return False
            for kind, digest in changed.items():
                live_path = getattr(self.live, f"{kind}_path")(paper.key)
                if digest is None:
                    live_path.unlink(missing_ok=True)
                else:
                    temporary = live_path.with_suffix(live_path.suffix + ".reopen.tmp")
                    with temporary.open("wb") as stream:
                        stream.write(self.path(kind).read_bytes())
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, live_path)
                _fsync_dir(live_path.parent, durable=True)
                setattr(paper, f"{kind}_path", str(live_path.relative_to(self.live.root)) if digest else None)
            for field in _RESULT_FIELDS:
                if getattr(result, field) != getattr(self.before, field) and getattr(paper, field) == getattr(self.before, field):
                    setattr(paper, field, getattr(result, field))
            paper.download_status = result.download_status
            paper.reopen_claim = paper.reopen_claim.model_copy(update={"phase": "completed"})
            store.save()
            store.ledger.complete(self.token)
            self.live.log({"event": "reopen_complete", "key": paper.key,
                           "generation": self.token.generation, "lease_id": self.token.lease_id,
                           **publication_fields(identity_snapshot(self.before), paper,
                               verified=getattr(self.library, "_stoprule_identity_verified", False))})
            return True


async def process_attempt(queue, token, priority):
    """Recheck after the slot wait, then stage execution and fence publication."""
    from .state_v2 import valid_lease
    from .download_telemetry import network_slot, span
    from .services import download_queue

    store = ClaimStore(queue.library, clock=queue._reopen_clock)
    stage = None
    drained = False
    expiry = None
    drain_started = None
    try:
        async with network_slot(concurrency.network_sem, queue.library, token.key):
            if token.campaign_id in queue._reopen_halted:
                await store.not_sent(token)
                return
            if not await store.pre_dispatch(token):
                return
            stage = StagedAttempt(queue.library, token)
            async def watch_lease():
                await asyncio.sleep(max(0, token.lease_expiry - store.clock()))
                if await store.expire(token.key, token=token):
                    queue._reopen_halted.add(token.campaign_id)
                    record_censored(queue.library, token.key, reason="lease expired without executor drain evidence")
            expiry = asyncio.create_task(watch_lease())
            def execute():
                # Scheduling may outlast the async recheck. This last read emits
                # no upstream call if the current claim ceased to admit.
                if token.campaign_id in queue._reopen_halted or not store.allowed(token, executing=True):
                    return False
                with span("executor") if stoprules_v2_enabled() else nullcontext():
                    download_queue.download_paper(stage.library.get(token.key), stage.library,
                        lease_provider=lambda p: valid_lease(p, lease_id=token.lease_id, now=store.clock()),
                        budget_provider=lambda p: store.ledger.admitted(token))
                return True
            context = contextvars.copy_context()
            executor = asyncio.get_running_loop().run_in_executor(None, context.run, execute)
            cancelled = False
            quarantine_error = None
            while True:
                try:
                    sent = await asyncio.shield(executor)
                    drained = True
                    break
                except asyncio.CancelledError:
                    # Cancellation cannot release the permit while the thread
                    # still consumes it. Quarantine promptly, then await real
                    # drain, including repeated cancellation during shutdown.
                    cancelled = True
                    if stoprules_v2_enabled() and drain_started is None:
                        drain_started = time.monotonic()
                        record_censored(queue.library, token.key, reason="cancelled executor awaiting drain")
                    queue._reopen_halted.add(token.campaign_id)
                    quarantine = asyncio.create_task(store.quarantine(token, reason="cancelled executor awaiting drain"))
                    while not quarantine.done():
                        try:
                            await asyncio.shield(quarantine)
                        except asyncio.CancelledError:
                            continue
                        except BaseException as exc:
                            quarantine_error = exc
                            break
                    try:
                        quarantine.result()
                    except BaseException as exc:
                        quarantine_error = exc
                    if executor.done():
                        drained = True
                        if not executor.cancelled():
                            executor.exception()  # retrieve a terminated thread's failure
                        if quarantine_error is not None:
                            raise quarantine_error
                        raise
                except BaseException:
                    drained = executor.done()
                    raise
            if quarantine_error is not None:
                raise quarantine_error
            if cancelled:
                raise asyncio.CancelledError
        if not sent:
            await store.not_sent(token, executor_proof=True)
            return
        if await stage.writeback(store):
            paper = queue.library.get(token.key)
            if paper.download_status == "ok" and queue._on_success is not None:
                download_queue._call_with_optional_priority(queue._on_success, token.key, priority)
        else:
            queue._reopen_halted.add(token.campaign_id)
    except BaseException:
        queue._reopen_halted.add(token.campaign_id)
        record_censored(queue.library, token.key, reason="executor/publication outcome uncertain")
        await store.quarantine(token, reason="executor/publication outcome uncertain")
        raise
    finally:
        if drain_started is not None and drained:
            queue.library.log({"event": "download_telemetry", "key": token.key,
                               "phase": "executor_drain", "mark": "end",
                               "drain_s": max(0.0, time.monotonic() - drain_started)})
        if expiry is not None:
            expiry.cancel()
            await asyncio.gather(expiry, return_exceptions=True)
        if stage is not None and drained:
            stage.cleanup()


async def run_campaign(library: Library, *, cap: int, queue=None, clock=time.time,
                       lease_seconds: float = 3600) -> dict:
    """Explicit campaigns only; boot/scheduler never selects re-open candidates."""
    from .services.download_queue import DownloadQueue
    _require_enabled()
    store = ClaimStore(library, clock=clock)
    async with store.lock:
        store.ledger.check_campaign(cap=cap)
        papers, selected = _selection(library.root, cap, store.ledger)
        if any(p.reopen_claim and p.reopen_claim.phase in {"queued", "dispatched"} for p in papers):
            raise LedgerError("unsettled campaign claims require recovery")
        if any(concurrency.is_in_flight(f"dl:{p.key}") for p in selected):
            raise LedgerError("duplicate in-flight identity; campaign halted")
        if not selected:
            return {"dry_run": False, "cap": cap, "keys": [], "selected": 0, "completed": 0}
        campaign = store.ledger.start_campaign(cap=cap, keys=[p.key for p in selected])
    owned = queue is None
    queue = queue or DownloadQueue(library)
    queue._reopen_clock = clock
    tokens = []
    try:
        for paper in selected:
            token = await store.claim(paper.key, expected_state=paper.acquisition_state,
                expected_generation=record_generation(paper),
                campaign_id=campaign, ttl=lease_seconds)
            if token is None:
                raise LedgerError("claim lost; campaign halted")
            tokens.append(token)
            queue.add_reopen(token)
        await queue.start(recover=False)
        await queue._queue.join()
        if stoprules_v2_enabled():
            from .stop_rules import evaluate_round
            await evaluate_round(library, ledger=store.ledger)
        if any(store.current(t) is None or store.current(t).reopen_claim.phase not in {"completed", "not_sent", "released"}
               for t in tokens):
            raise LedgerError("campaign has uncertain attempts; drain/fencing recovery required")
        store.ledger.finish_campaign(campaign, drained=True)
    finally:
        if owned:
            await queue.stop()
    return {"dry_run": False, "campaign_id": campaign, "cap": cap, "keys": [p.key for p in selected],
            "selected": len(selected), "completed": sum(store.current(t).reopen_claim.phase == "completed" for t in tokens)}
