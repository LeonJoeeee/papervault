"""Background worker pool for the PDF-download stage.

Independent of :class:`ExtractQueue`. Started by the daemon lifespan;
receives work from two sources:

  1. **External trigger** — MCP ``search_papers`` upserts metadata-only
     candidates and calls ``download_queue.add(key)``. Foreground
     ``get_paper`` enqueues at ``PRIORITY_URGENT`` via the same ``add(key)``
     (fire-and-forget; the record carries ``text_status=pending`` until the
     background workers finish).
  2. **Recovery** — on ``start()`` scans the library and enqueues every
     paper with ``status == "pending"`` AND no PDF on disk. Catches
     pending work from prior daemon sessions; restart-survival is by
     construction (no separate queue persistence file).

Worker body wraps :func:`download_paper` under ``concurrency.network_sem``
(cascade tier HTTP cap) and saves the library under
``concurrency.lib_write_lock``. On success the worker chains to:

- ``extract_queue.add(key)`` for the normal case (PDF on disk; needs txt+md)
- For a firecrawl text-only win (D7: ``status="ok"`` + md already on disk,
  no PDF) the worker takes the same ``on_success`` chain — extract queue
  self-detects the md presence and forwards without re-extracting.
- No downstream chaining for ``failed`` / ``metadata_only`` — these are
  terminal states and recovery on next startup will NOT re-enqueue them.

✦ Phase 28 (2026-05-24, route B): the third-stage insight queue was
removed (see ``src/papervault/library/insight/__init__.py``). The
previous ``on_text_only`` callback that bypassed
extract straight into insight is gone; text-only papers now route
through extract_queue (which short-circuits on already-present md and
just records the completion).

Self-check idempotence — at the top of ``_process_one`` we re-read the
paper and short-circuit if has_pdf is already True. This protects
against duplicate enqueues (BG recovery + foreground both adding the
same key) without needing a global with_dedup.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from ..download import download_paper
from ..models import DOWNLOAD_STATUS_OK, DOWNLOAD_STATUS_PENDING
from ..store import Library
from . import concurrency

log = logging.getLogger("papervault.library.download_queue")


class DownloadQueue:
    """Async worker pool consuming PDF-download tasks.

    Lifecycle::

        dq = DownloadQueue(library, on_success=extract_q.add)
        await dq.start()
        ...
        await dq.stop()
    """

    def __init__(
        self,
        library: Library,
        *,
        num_workers: int = 4,
        on_success: Optional[Callable[[str], None]] = None,
        # Phase 28 (2026-05-24, route B): the insight queue was removed,
        # so the dedicated text-only callback is now redundant. Accept
        # the kwarg for back-compat with any standalone caller that
        # still passes it, but route everything through ``on_success``
        # (extract_queue.add) — extract queue self-detects md presence
        # and short-circuits the no-op extract for firecrawl text-only
        # papers (D7: status="ok" + md on disk), then forwards. Will be
        # removed once stale call sites are cleaned up.
        on_text_only: Optional[Callable[[str], None]] = None,  # noqa: ARG002
    ):
        """
        Args:
            library: Loaded :class:`Library`.
            num_workers: Concurrent download workers. Default 4 matches
                the ``network_sem(4)`` cap — adding more workers just
                queues them on the semaphore.
            on_success: Callback invoked with the paper key after a
                successful PDF download. Typically ``extract_queue.add``.
                None during tests / standalone use. Also used for the
                firecrawl text-only branch (D7: status="ok" + md on disk;
                extract queue handles the md-already-present idempotence).
            on_text_only: Deprecated (Phase 28 route B). Accepted for
                back-compat but ignored — text-only papers now use the
                same ``on_success`` chain.
        """
        self.library = library
        self.num_workers = num_workers
        self._on_success = on_success
        self._on_text_only = on_success
        # PriorityQueue items are tuples (priority, seq, key). Lower priority
        # wins; seq breaks ties FIFO so two same-priority items pop in insertion
        # order. See concurrency.PRIORITY_* constants.
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq: int = 0
        self._workers: list[asyncio.Task] = []
        self._started = False
        # In-flight dedup (F7), priority-aware (R1): maps each key currently
        # sitting in the queue (enqueued, not yet dequeued by a worker) to the
        # priority it was last enqueued at. A second reconcile sweep at +600s
        # that re-routes the SAME firecrawl-md key at the SAME-OR-LOWER priority
        # before the first enqueue drains is skipped here, so it cannot re-run
        # download_paper + the firecrawl re-gate (LLM call) a second time within
        # one queue-residency window. A STRICTLY-MORE-URGENT re-add (e.g. a
        # foreground get_paper at PRIORITY_URGENT of a key already queued at
        # NORMAL) is NOT deduped — it is re-enqueued at the higher priority so
        # the urgent request actually jumps the backlog; the stale lower-prio
        # tuple it leaves behind is dropped by the worker's POP-TIME dedup
        # (T1): the first tuple to surface for a key pops its _pending entry
        # and processes; a later stale duplicate finds the entry gone and is
        # discarded WITHOUT processing — a true no-op for every key type
        # (including firecrawl-md, where the worker's has_pdf short-circuit
        # would NOT fire and would otherwise re-run all 18 PDF tiers). The
        # entry is cleared the moment a worker dequeues the key (so a genuine
        # re-add AFTER processing starts — the case the persistent
        # firecrawl_pdf_hunt_exhausted stamp bounds — still enqueues).
        self._pending: dict[str, int] = {}

    async def start(self) -> None:
        """Enqueue every paper needing download and spawn worker tasks.

        Recovery criterion: ``status == "pending"`` AND ``not has_pdf``.
        ``failed`` / ``metadata_only`` / ``extract_failed`` are
        intentionally skipped — they are terminal states (D7); ``audit
        --retry-failed`` resets them to ``pending`` if the operator wants a
        retry. (Reconcile (D8) additionally re-routes firecrawl-md ``ok``
        papers back here to hunt the real PDF — out of scope for this
        recovery scan, which only handles ``pending``.)

        Idempotent: calling ``start()`` twice is a no-op.
        """
        if self._started:
            return
        self._started = True

        recovered = 0
        for paper in self.library.all_papers():
            if paper.download_status != DOWNLOAD_STATUS_PENDING:
                continue
            if self.library.has_pdf(paper.key):
                continue
            self._enqueue(paper.key, concurrency.PRIORITY_NORMAL)
            recovered += 1
        log.info("download queue: recovered %d pending tasks", recovered)

        for i in range(self.num_workers):
            self._workers.append(
                asyncio.create_task(
                    self._worker_loop(i), name=f"download-worker-{i}"
                )
            )
        log.info("download queue: spawned %d workers", self.num_workers)

    async def stop(self) -> None:
        """Cancel workers; queue contents are lost on restart but
        recovery rebuilds them from on-disk state anyway."""
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._pending.clear()  # F7: drop stale in-flight marks (queue is gone)
        self._started = False

    def add(self, key: str, priority: int = concurrency.PRIORITY_NORMAL) -> None:
        """Enqueue a paper key for PDF download.

        ``priority`` is one of :data:`concurrency.PRIORITY_URGENT` (0) or
        :data:`concurrency.PRIORITY_NORMAL` (5). Foreground MCP-tool calls
        should pass ``PRIORITY_URGENT`` to jump past a background backlog.

        Safe to call duplicates. Two layers guard idempotence:

          * In-flight dedup (F7), priority-aware (R1): a key already sitting in
            the queue (enqueued, not yet dequeued) is skipped **for a
            same-or-lower-priority re-add**, so a +600s reconcile sweep cannot
            double-enqueue a still-waiting firecrawl-md key and re-run
            ``download_paper`` + the firecrawl re-gate (LLM call) a second time
            within one queue-residency window. A **strictly-more-urgent** re-add
            (e.g. a foreground ``get_paper`` at ``PRIORITY_URGENT`` of a key the
            reconcile sweep queued at ``PRIORITY_NORMAL``) is **not** deduped —
            it is re-enqueued at the higher priority so it jumps the backlog.
            The stale lower-prio tuple it leaves behind is dropped at POP TIME
            (T1): the worker processes a dequeued tuple only if it still owns
            the ``_pending`` entry (the first tuple to surface claims it); a
            later stale duplicate finds the entry gone and is discarded without
            processing. The mark is cleared on the FIRST dequeue.
          * Pop-time dedup makes the stale-tuple drop true **by construction**
            for every key type — it does NOT rely on the worker's ``has_pdf``
            short-circuit (which fires only for ``has_pdf`` papers and never for
            a firecrawl-md key, ``has_md`` ∧ ``¬has_pdf``, the population
            reconcile re-routes here to hunt the real PDF). Convergence to
            "PDF hunt + re-gate at most once per paper" rests on the persistent
            ``firecrawl_pdf_hunt_exhausted`` stamp (set on gate PASS) flipping
            ``classify`` rule (3) off; the in-flight + pop-time dedup closes the
            narrow same-window double-sweep race, the stamp covers the broader
            cross-cycle bound.
        """
        self._enqueue(key, priority)

    def _enqueue(self, key: str, priority: int) -> None:
        # In-flight dedup (F7), priority-aware (R1): a key already waiting in the
        # queue is deduped ONLY for a same-or-lower-priority re-add (lower
        # urgency = larger number). A strictly-MORE-urgent re-add falls through
        # to a fresh put_nowait at the higher priority so it jumps the backlog;
        # the stale lower-prio tuple it leaves behind is dropped at pop time
        # (T1: the worker only processes the first tuple to claim _pending[key]).
        queued = self._pending.get(key)
        if queued is not None and priority >= queued:
            return
        self._pending[key] = priority
        self._seq += 1
        self._queue.put_nowait((priority, self._seq, key))

    def qsize(self) -> int:
        return self._queue.qsize()

    async def _worker_loop(self, worker_id: int) -> None:
        log.debug("download worker %d starting", worker_id)
        while True:
            priority, _seq, key = await self._queue.get()
            # Pop-time dedup (F7/R1/T1): the in-flight ``_pending`` mark is the
            # single source of truth for "this key's queued work is still
            # outstanding". The FIRST tuple to surface for a key claims it
            # (``pop`` returns its priority → process); any STALE duplicate
            # tuple — left behind when a strictly-more-urgent re-add jumped
            # ahead at a higher priority (R1) — finds the key already gone
            # (``pop`` returns None) and is dropped WITHOUT calling
            # ``download_paper``. This makes the stale-tuple "no-op" claim true
            # BY CONSTRUCTION for every key type: it no longer leans on the
            # worker's ``has_pdf`` short-circuit, which never fires for a
            # firecrawl-md key (``has_md`` ∧ ``¬has_pdf``) and would otherwise
            # re-run all 18 PDF tiers a second time (T1). A re-add that arrives
            # while we PROCESS this key re-populates ``_pending`` with a fresh
            # tuple and is still honored (bounded by the firecrawl stamp).
            if self._pending.pop(key, None) is None:
                self._queue.task_done()
                continue
            try:
                await self._process_one(key, priority)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "download worker %d crashed processing %s",
                    worker_id, key,
                )
            finally:
                self._queue.task_done()

    async def _process_one(self, key: str,
                           priority: int = concurrency.PRIORITY_NORMAL) -> None:
        # Cross-actor per-key mutual exclusion (verified drill finding: the
        # pop-time _pending dedup does NOT cover a re-add that arrives WHILE a
        # worker is processing this key — a second worker then runs download_paper
        # on the same live Paper concurrently). with_dedup makes a concurrent
        # processing of the same key SUBSCRIBE to the in-flight run instead of
        # running a duplicate. Stage-scoped key ("dl:") so the download and the
        # chained extract stage of the same paper are NOT collapsed together.
        await concurrency.with_dedup(
            f"dl:{key}", lambda: self._process_one_inner(key, priority))

    async def _process_one_inner(self, key: str, priority: int) -> None:
        paper = self.library.get(key)
        if paper is None:
            log.warning("download queue: %s not in library, dropping", key)
            return
        # Idempotence — paper may already have a PDF (concurrent
        # foreground materialize, prior worker, etc.). In that case
        # short-circuit but still forward to the next stage if md is
        # missing, so a recovery-enqueued paper that already downloaded
        # in some prior run still moves through the pipeline.
        if self.library.has_pdf(key):
            log.debug("download queue: %s already has PDF, forwarding", key)
            self._forward_post_download(paper, priority)
            return

        async with concurrency.network_sem:
            ok = await asyncio.to_thread(download_paper, paper, self.library)

        async with concurrency.lib_write_lock:
            self.library.save()

        # Re-read paper for the freshest status (download_paper mutates).
        paper = self.library.get(key) or paper
        if ok:
            log.info("download queue: %s OK (status=%s)",
                     key, paper.download_status)
            self._forward_post_download(paper, priority)
        elif (paper.download_status == DOWNLOAD_STATUS_OK
                and self.library.has_extract(key, "md")):
            # D7: download_paper returned False (no PDF binary) but status is
            # "ok" with an md on disk — the firecrawl text-only fallback won.
            # Chain to extract_queue, which self-detects the md and
            # short-circuits the OCR step (no PDF to feed the cascade anyway).
            log.info("download queue: %s firecrawl text-only, chain to extract",
                     key)
            if self._on_text_only is not None:
                _call_with_optional_priority(self._on_text_only, key, priority)
        else:
            # status is "failed" or "metadata_only" — terminal,
            # no downstream chaining. Status is already written.
            log.info("download queue: %s terminal (status=%s)",
                     key, paper.download_status)

    def _forward_post_download(self, paper, priority: int) -> None:
        """Route a successfully-downloaded paper to the next stage,
        preserving the original work's priority all the way through.

        ✦ Phase 28 (route B): pre-route-B this branched between extract
        (no md yet) and insight (md already present, e.g. firecrawl). The
        insight stage is gone; both branches now go to ``on_success``
        (extract queue), which self-detects md presence and short-circuits
        accordingly.
        """
        if self._on_success is not None:
            _call_with_optional_priority(self._on_success, paper.key, priority)


def _call_with_optional_priority(cb, key: str, priority: int) -> None:
    """Invoke a callback that accepts (key) or (key, priority=...).
    The downstream queue's ``add()`` accepts an optional ``priority`` kwarg
    (D11); legacy callbacks (e.g., test stubs) accept only ``key``."""
    try:
        cb(key, priority=priority)
    except TypeError:
        cb(key)
