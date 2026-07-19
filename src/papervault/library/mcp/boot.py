"""Library background boot — the single source of truth for starting the library
plane's stage queues + safety-net tasks, shared by the standalone paper-library
entrypoint and the unified papervault MCP server.

Given an already-built server (from :func:`build_server`, which stashes the
Library + both queues on the instance), this:
  1. runs the one-time download_status migration (idempotent),
  2. starts the extract then download queues (downstream-first),
  3. launches the reconcile sweep + on-demand MinerU idle monitor,
and returns an async ``shutdown()`` that tears them down in reverse.

**Bind-the-port-before-the-sweep (issue #34 remaining scope).** Steps 1-3 include
the potentially-long backlog recovery: the extract queue's ``start()`` runs an
isolated ``pdf_probe`` *subprocess* per EXTRACT-routed paper (~1100 of them on
the production vault), and the first reconcile sweep scans the whole library.
Historically this ran *synchronously inside the ASGI lifespan startup*, and an
ASGI server does not bind its socket until lifespan-startup returns — so uvicorn
did not accept a single request for the 60-120 s the recovery took (worst case:
a wedged probe or a down MinerU turned it into a multi-minute / zombie boot).

So this function now returns as soon as the cheap prerequisites are scheduled:
the heavy recovery + the reconcile/monitor loops run in a DETACHED background
task (``server._paper_boot_task``). The caller's lifespan yields immediately →
uvicorn binds :8080 and serves while the backlog warms up in the background.

Graceful degradation during the warm-up window (the "sweep" is now background):
  * ``search_papers`` / ``get_paper`` work — they read the already-loaded
    ``Library`` (built in ``build_server``, before this runs) and return metadata
    + an honest ``text_status``.
  * ``get_paper``'s fire-and-forget URGENT download/extract kick is already gated
    on ``download_queue._started`` (mcp/server.py), so before the queues start it
    is simply skipped: a not-yet-reconciled paper reports ``text_status=pending``
    (honest) and progresses once the background boot has started the queues —
    typically sub-second, at most the length of the recovery scan. Nothing is
    lost; the only change is the URGENT kick may be deferred by that window.

Boot-failure observability: if the detached boot RAISES (queues/reconcile never
come up), the failure is loud in the log AND on a monitored signal —
``server._paper_boot_failed`` is set to a short summary. The heavy MCP tools
(``get_paper`` / ``search_papers``) read it and surface a ``service_degraded``
field so a caller sees the degradation instead of it living only in a log line.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable
from papervault.library.services import concurrency


async def start_background(server, log) -> Callable[[], Awaitable[None]]:
    """Schedule the library queues + BG tasks on ``server``; return a ``shutdown``
    coro. Returns PROMPTLY (issue #34): the heavy backlog recovery + reconcile
    run in a detached task (``server._paper_boot_task``) so the caller's ASGI
    lifespan can yield and uvicorn can bind the port before the sweep runs.

    Start order inside the background task is downstream-first so each upstream
    queue's ``on_success`` callback fires into an already-running consumer;
    shutdown is the reverse.
    """
    library = server._paper_library                 # type: ignore[attr-defined]
    download_queue = server._paper_download_queue   # type: ignore[attr-defined]
    extract_queue = server._paper_extract_queue     # type: ignore[attr-defined]

    # Handles filled in by the background boot; shutdown reads whatever exists.
    reconcile_task: asyncio.Task | None = None
    mineru_monitor_task: asyncio.Task | None = None

    # Boot-failure signal (monitored by the heavy MCP tools). None while the boot
    # is healthy / in-flight; set to a short exception summary if _deferred_boot
    # raises, so get_paper / search_papers can surface a ``service_degraded``
    # field to callers instead of the failure being visible ONLY in the log.
    server._paper_boot_failed = None  # type: ignore[attr-defined]

    async def _dirty_flusher() -> None:
        # Trailing-edge flush for debounced saves (issue #34 review): a burst that
        # ends inside the debounce window must not stay dirty until the next writer.
        while True:
            await asyncio.sleep(10)
            try:
                async with concurrency.lib_write_lock:
                    library.flush_if_dirty()
            except Exception:  # noqa: BLE001 — flusher must never die
                log.exception("dirty-flusher failed (will retry)")

    # The flusher is cheap and independent — start it immediately so any debounced
    # save made during the background recovery scan is guaranteed to reach disk.
    flusher_task = asyncio.create_task(_dirty_flusher(), name="library-dirty-flusher")

    async def _deferred_boot() -> None:
        """The potentially-long library warm-up — runs AFTER the port binds."""
        nonlocal reconcile_task, mineru_monitor_task
        try:
            # One-time download_status migration — normalize every legacy status to
            # the clean routing enum BEFORE the queues' recovery scans read a
            # record. Idempotent (a no-op on an already-migrated vault).
            from ..services.migrate_status import (
                STUB_DELETION_RULE, migrate_status, should_persist)

            report = migrate_status(library)
            if should_persist(report):
                async with concurrency.lib_write_lock:
                    library.save()
                stub_deletions = report.get("by_rule", {}).get(STUB_DELETION_RULE, 0)
                log.info("download_status migration: normalized %d/%d records "
                         "(%d stub deletions) %s",
                         report["migrated"], report["total"], stub_deletions,
                         report["by_rule"])

            # Backlog recovery (the long pole: pdf_probe subprocess per EXTRACT
            # paper). Now off the port-bind critical path.
            await extract_queue.start()
            log.info("paper-extract queue started")
            await download_queue.start()
            log.info("paper-download queue started")

            from ..services.reconcile import reconcile_loop
            reconcile_task = asyncio.create_task(
                reconcile_loop(library, download_queue, extract_queue),
                name="paper-reconcile-loop",
            )
            log.info("paper-reconcile loop started")

            from ..services.mineru_server import get_server_controller
            mineru_monitor_task = asyncio.create_task(
                get_server_controller().monitor_loop(extract_queue),
                name="mineru-idle-monitor",
            )
            log.info("mineru on-demand monitor task created (active only when on-demand ON)")
            log.info("library background boot complete (queues + reconcile live)")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a boot failure must be loud, not silent
            # Loud in the LOG *and* on a MONITORED signal: stash a short summary
            # on the server so the heavy MCP tools can tell callers the service
            # is degraded (queues/reconcile may be down) instead of the failure
            # being buried in a log line no caller ever sees.
            server._paper_boot_failed = (  # type: ignore[attr-defined]
                f"library background boot failed: {type(exc).__name__}: {exc}"[:300])
            log.exception("library background boot FAILED; server keeps serving "
                          "reads but queues/reconcile may be down")

    boot_task = asyncio.create_task(_deferred_boot(), name="library-boot")
    server._paper_boot_task = boot_task  # type: ignore[attr-defined]

    async def shutdown() -> None:
        # If the background boot is still running, stop it; if it finished, this
        # is a no-op and the await returns its (already-logged) result.
        boot_task.cancel()
        try:
            await boot_task
        except (asyncio.CancelledError, Exception):
            pass
        flusher_task.cancel()
        for _t in (flusher_task, reconcile_task, mineru_monitor_task):
            if _t is None:
                continue
            _t.cancel()
        for _t in (flusher_task, reconcile_task, mineru_monitor_task):
            if _t is None:
                continue
            try:
                await _t
            except (asyncio.CancelledError, Exception):
                pass
        # Stop both queues UNCONDITIONALLY. Each ``stop()`` is idempotent-safe
        # when the queue was never started (the boot raised before start(), or
        # a shutdown races an in-flight boot): with no workers spawned the
        # cancel/gather over an empty ``_workers`` list is a no-op, ``_pending``
        # is already empty, and ``_started`` is simply re-set False. Relying on
        # that is more robust than the old coarse ``queues_started`` flag, which
        # went stale if the boot raised BETWEEN the two start() calls (extract
        # started, download not) and then skipped stopping the started one.
        await download_queue.stop()
        await extract_queue.stop()
        # Graceful-shutdown flush: debounced state must reach disk before exit —
        # NEW records have no disk artifact and are NOT reconcile-recoverable.
        async with concurrency.lib_write_lock:
            library.save(force=True)

    return shutdown
