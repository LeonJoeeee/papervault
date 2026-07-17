"""Background worker pool for the extract (md) stage.

Independent of :class:`DownloadQueue`. Started by the daemon lifespan;
receives work from two sources:

  1. **Chain trigger** — ``DownloadQueue`` calls ``extract_queue.add(key)``
     after a successful PDF download.
  2. **Recovery** — on ``start()`` scans the library and enqueues every
     paper that ``classify()`` routes to EXTRACT (D7: has a PDF, no md yet,
     status ``ok``, attempts under the ceiling).

2026-06-06 (MinerU migration, SDD §4.1): **worker count is a fixed explicit
knob** ``_EXTRACT_CONCURRENCY`` (env ``PAPER_LIBRARY_EXTRACT_CONCURRENCY``,
default 8), DECOUPLED from GPU count. The persistent MinerU2.5-Pro vLLM server
owns its card 24/7 and batches the per-page sub-requests server-side via
``--max-num-seqs``; an explicit ``asyncio.Semaphore`` (``concurrency.acquire_extract_slot``)
caps how many whole-PDF extractions are in flight. ``ocr-pool.conf`` is
REDEFINED (SDD §4.3) to a binary **pause-dispatch** toggle: an EMPTY file pauses
admission (workers idle, the queue holds), a non-empty / absent file runs.
"Free the GPU" is now ``systemctl stop`` of the MinerU server unit, not a
conf edit.

Worker body runs (SDD §2.1):
  1. ``await extract_md`` — ONE whole-PDF MinerU call under a concurrency permit
     + a round-robin endpoint (acquired in ``concurrency.acquire_extract_slot``);
     the per-paper GPU pin + the pypdf ``extract_txt`` foreground pass are gone.
  2. ``library.save()``
  3. ``on_success(key)`` callback (optional; pre-route-B this fed the
     insight queue, but Phase 28 removed that stage — kept in the
     signature for tests / future consumers).

✦ Phase 28 (2026-05-24, route B): the third-stage insight queue was
deleted. ``on_success`` is no longer wired by ``build_server`` since
there's no downstream consumer; the callback parameter remains for
test fixtures and for future Librarian-side hooks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from ..extract import (
    _EXTRACT_CONCURRENCY,
    _EXTRACT_WORKERS,
    _LONG_PAPER_PAGE_THRESHOLD,
    extract_md,
)
from ..store import Library
from . import concurrency
from .classify import EXTRACT, classify

log = logging.getLogger("papervault.library.extract_queue")


def _note_mineru_activity() -> None:
    """Stamp on-demand MinerU activity (no-op unless on-demand ON). Cheap and
    must never raise on the hot path."""
    try:
        from .mineru_server import get_server_controller
        get_server_controller().note_activity()
    except Exception:  # noqa: BLE001
        pass


class ExtractQueue:
    """Async worker pool consuming extract tasks.

    2026-06-06 (MinerU migration): worker count is the fixed
    ``_EXTRACT_CONCURRENCY`` knob; ``ocr-pool.conf`` gates dispatch (pause/run),
    not GPU selection.
    """

    # How often a worker re-checks the pause-dispatch toggle while idle.
    _WORKER_POLL_INTERVAL_SECONDS = 5.0

    def __init__(
        self,
        library: Library,
        *,
        num_workers: Optional[int] = None,  # ignored; kept for back-compat
        on_success: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            library: Loaded :class:`Library`.
            num_workers: IGNORED. Worker count = ``_EXTRACT_CONCURRENCY``
                (SDD §4.1). Kept in signature for back-compat with callers
                passing it.
            on_success: Callback invoked with the paper key after a
                successful md extract on disk. Pre-route-B this fed
                ``insight_queue.add``; post-route-B (Phase 28) the
                production wiring passes None and the callback is a
                no-op. Tests may still pass a stub to observe completion.
        """
        self.library = library
        self._on_success = on_success
        # PriorityQueue items: (priority, seq, key). See concurrency.PRIORITY_*.
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq: int = 0
        self._workers: list[asyncio.Task] = []
        self._monitor_task: Optional[asyncio.Task] = None
        self._started = False
        # In-flight dedup (F7), priority-aware (R1): maps each key enqueued but
        # not yet dequeued to the priority it was last enqueued at. A
        # same-or-lower-priority re-add of a still-waiting key is deduped (a
        # +600s reconcile sweep cannot double-enqueue it); a strictly-more-urgent
        # re-add falls through at the higher priority so it jumps the backlog.
        # Cleared on the FIRST dequeue; the stale lower-prio duplicate it leaves
        # behind is dropped at pop time (T1: the worker only processes the first
        # tuple to claim _pending[key], so the drop is a no-op by construction
        # and does not depend on the has_extract short-circuit).
        self._pending: dict[str, int] = {}

    def _target_worker_count(self) -> int:
        """How many worker coroutines this queue runs (SDD §4.1).

        A fixed explicit concurrency knob ``_EXTRACT_CONCURRENCY``, decoupled
        from GPU count — the persistent MinerU server batches per-page
        sub-requests server-side, so the daemon-side knob just caps how many
        whole-PDF calls are in flight. (Dispatch pause/run is a SEPARATE,
        per-iteration check via ``ocr-pool.conf``; it does not change the
        worker COUNT.)

        Worker count (``_EXTRACT_WORKERS``, default 16) is DECOUPLED from the OCR
        slot cap (``extract_slots`` = _EXTRACT_CONCURRENCY, ~8): more workers than
        OCR slots means a worker in its off-loop LLM-gate phase never pins a GPU
        slot — another worker keeps MinerU fed (the GPU-idle fix).
        """
        return _EXTRACT_WORKERS

    async def start(self) -> None:
        """Enqueue recovery items + spawn ``_EXTRACT_CONCURRENCY`` workers.

        Recovery criterion (D7): ``classify(paper) == EXTRACT`` — i.e. a PDF
        on disk, no md yet, status ``ok``, attempts under the ceiling. The
        ``classify`` router is the single source of routing truth, so the
        ``has_pdf`` / ``has_extract`` / attempts guards live in one place.
        Long papers (>90 pages) get PRIORITY_LOW so they queue after
        normal-sized backlog.
        """
        if self._started:
            return
        self._started = True

        from ..extract import pdf_probe
        long_threshold = _LONG_PAPER_PAGE_THRESHOLD  # 90 pages by default

        recovered_normal = 0
        recovered_low = 0
        for paper in self.library.all_papers():
            if classify(paper, self.library) != EXTRACT:
                continue
            pdf_path = self.library.pdf_path(paper.key)
            # D10: page count comes from the isolated probe subprocess, never
            # an in-thread pypdf parse — one bad PDF must not wedge the
            # recovery scan at startup. A bad probe → n_pages 0 → normal lane
            # (extract_md re-probes and routes it to extract_failed there).
            n_pages = (pdf_probe(str(pdf_path)).n_pages
                       if pdf_path.exists() else 0)
            if n_pages > long_threshold:
                self._enqueue(paper.key, concurrency.PRIORITY_LOW)
                recovered_low += 1
            else:
                self._enqueue(paper.key, concurrency.PRIORITY_NORMAL)
                recovered_normal += 1
        log.info(
            "extract queue: recovered %d ok-without-md tasks "
            "(%d normal ≤%dp, %d low %dp+)",
            recovered_normal + recovered_low,
            recovered_normal, long_threshold,
            recovered_low, long_threshold + 1,
        )

        # Spawn a fixed pool of _EXTRACT_CONCURRENCY workers (SDD §4.1).
        target = self._target_worker_count()
        for i in range(target):
            self._workers.append(
                asyncio.create_task(
                    self._worker_loop(i), name=f"extract-worker-{i}"
                )
            )
        log.info("extract queue: spawned %d workers (OCR slot cap=%d, gate cap=%d)",
                 target, _EXTRACT_CONCURRENCY, concurrency._GATE_CONCURRENCY)

    async def stop(self) -> None:
        # The dynamic worker-count monitor is gone (SDD §4.1: fixed pool); the
        # ``_monitor_task`` slot is kept None for back-compat with any external
        # introspection. Cancel + drain the fixed worker pool.
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._monitor_task = None
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._pending.clear()  # F7: drop stale in-flight marks (queue is gone)
        self._started = False

    def add(self, key: str, priority: int = concurrency.PRIORITY_NORMAL) -> None:
        """Enqueue a paper key for md extraction."""
        self._enqueue(key, priority)

    def _enqueue(self, key: str, priority: int) -> None:
        # In-flight dedup (F7), priority-aware (R1): a still-waiting key is
        # deduped ONLY for a same-or-lower-priority re-add (larger number = less
        # urgent). A strictly-more-urgent re-add falls through to a fresh
        # put_nowait at the higher priority so it jumps the backlog; the stale
        # lower-prio tuple it leaves behind is dropped at pop time (T1: the
        # worker only processes the first tuple to claim _pending[key]).
        queued = self._pending.get(key)
        if queued is not None and priority >= queued:
            return
        self._pending[key] = priority
        self._seq += 1
        self._queue.put_nowait((priority, self._seq, key))
        # On-demand MinerU: queued work means "don't stop the server" (no-op
        # unless on-demand ON). _enqueue is the single chokepoint for add() AND
        # boot recovery, so one call covers both.
        _note_mineru_activity()

    def qsize(self) -> int:
        return self._queue.qsize()

    # ------------------------ worker loop ----------------------------------

    async def _worker_loop(self, worker_id: int) -> None:
        log.debug("extract worker %d starting", worker_id)
        while True:
            # Pause-dispatch gate (SDD §4.3): an EMPTY ``ocr-pool.conf`` pauses
            # admission — the worker idles (does NOT pull from the queue, so the
            # backlog holds + the in-flight set drains) and re-checks each poll
            # interval. A non-empty / absent file runs. This replaces the old
            # GPU-index list semantics (the persistent MinerU server owns its
            # card; "free the GPU" = stop the server unit).
            try:
                dispatch_on = concurrency.extract_dispatch_enabled()
            except Exception:
                dispatch_on = True
            if not dispatch_on:
                try:
                    await asyncio.sleep(self._WORKER_POLL_INTERVAL_SECONDS)
                except asyncio.CancelledError:
                    raise
                continue

            # Wait briefly for a task; timeout to re-check the dispatch toggle.
            try:
                priority, _seq, key = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self._WORKER_POLL_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise

            # Pop-time dedup (F7/R1/T1): the in-flight ``_pending`` mark is the
            # single source of truth for "this key's queued work is still
            # outstanding". The FIRST tuple to surface for a key claims it
            # (``pop`` returns its priority → process); a STALE duplicate tuple
            # — left behind when a strictly-more-urgent re-add jumped ahead (R1)
            # — finds the key already gone (``pop`` returns None) and is dropped
            # WITHOUT calling ``extract_md``. The stale-tuple "no-op" is thus
            # true by construction (it no longer depends on the worker's
            # ``has_extract`` short-circuit). A re-add that arrives while we
            # PROCESS this key re-populates ``_pending`` and is still honored.
            if self._pending.pop(key, None) is None:
                self._queue.task_done()
                continue
            # On-demand MinerU: refresh activity on the work CRITICAL PATH (not
            # just at enqueue) so a paper that waited in the queue past the idle
            # timeout doesn't let the monitor stop the server out from under the
            # extract we're about to run (review fix #5).
            _note_mineru_activity()
            try:
                await self._process_one(key, priority)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "extract worker %d crashed processing %s",
                    worker_id, key,
                )
            finally:
                self._queue.task_done()

    # ------------------------ task processing ------------------------------

    async def _process_one(self, key: str,
                           priority: int = concurrency.PRIORITY_NORMAL) -> None:
        # Cross-actor per-key mutual exclusion (verified drill finding) — a
        # re-add during processing could otherwise run extract_md on the same
        # live Paper in a second worker concurrently. Stage-scoped key ("ex:")
        # so it does not collapse with the download stage of the same paper.
        await concurrency.with_dedup(
            f"ex:{key}", lambda: self._process_one_inner(key, priority))

    async def _process_one_inner(self, key: str, priority: int) -> None:
        paper = self.library.get(key)
        if paper is None:
            log.warning("extract queue: %s not in library, dropping", key)
            return
        if not self.library.has_pdf(key):
            log.warning(
                "extract queue: %s has no PDF, dropping (download chain bug?)",
                key,
            )
            return
        # Idempotence — md already extracted (concurrent worker, prior
        # run, firecrawl text-only path, etc.). Skip extract; forward
        # to on_success if present (no-op in post-route-B production).
        if self.library.has_extract(key, "md"):
            log.debug(
                "extract queue: %s already has md, skipping extract", key
            )
            if self._on_success is not None:
                self._on_success(key)
            return

        # extract_md — ONE whole-PDF MinerU call (SDD §2.1), awaited directly
        # (extract_md is now a coroutine — no asyncio.to_thread). The pypdf
        # ``extract_txt`` foreground pass + its early save are GONE (txt-drop,
        # SDD §3.4): a PDF paper mid-extraction serves ``text_status="pending"``,
        # never a dirty pypdf body.
        #
        # Concurrency admission (SDD §4.1): hold a semaphore permit for the
        # whole call so at most ``_EXTRACT_CONCURRENCY`` whole-PDF extractions
        # run in flight; the round-robin endpoint is resolved inside extract_md
        # (slot-acquire here is the cap, the env endpoint set is the failover
        # target). vLLM's ``--max-num-seqs`` batches the per-page sub-requests
        # server-side. The wall-clock cap + per-request HTTP timeout live inside
        # extract_mineru (the D10 replacement), so no outer per-paper timeout.
        # OCR-slot admission now lives INSIDE extract_md, bracketing ONLY the
        # MinerU call and released before the LLM gates — so the (off-loop) gate
        # phase never pins a GPU slot. Here we just run extract_md + log a crash.
        try:
            await extract_md(paper, self.library)
        except Exception:
            log.exception("extract queue: extract_md failed for %s", key)

        async with concurrency.lib_write_lock:
            self.library.save()

        # 3. Optional on_success chain. Pre-route-B this fed the
        # insight queue; post-route-B production wiring sets on_success
        # to None (extract is the terminal pipeline stage). The branch
        # remains for test fixtures and future Librarian-side hooks.
        if self.library.has_extract(key, "md") and self._on_success is not None:
            log.info("extract queue: %s OK → on_success", key)
            try:
                self._on_success(key, priority=priority)
            except TypeError:
                self._on_success(key)
        else:
            # Log the ACTUAL md outcome — this line used to say "no md" even on
            # success (it only meant "no on_success hook wired"), which made the
            # 2026-06/07 three-week silent extract outage invisible in the logs.
            log.info("extract queue: %s done (md=%s, status=%s)",
                     key, self.library.has_extract(key, "md"), paper.download_status)
