"""Library background boot — the single source of truth for starting the library
plane's stage queues + safety-net tasks, shared by the standalone paper-library
entrypoint and the unified papervault MCP server.

Given an already-built server (from :func:`build_server`, which stashes the
Library + both queues on the instance), this:
  1. runs the one-time download_status migration (idempotent),
  2. starts the extract then download queues (downstream-first),
  3. launches the reconcile sweep + on-demand MinerU idle monitor,
and returns an async ``shutdown()`` that tears them down in reverse.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable


async def start_background(server, log) -> Callable[[], Awaitable[None]]:
    """Start the library queues + BG tasks on ``server``; return a ``shutdown`` coro.

    Start order is downstream-first so each upstream queue's ``on_success`` callback
    fires into an already-running consumer; shutdown is the reverse.
    """
    library = server._paper_library                 # type: ignore[attr-defined]
    download_queue = server._paper_download_queue   # type: ignore[attr-defined]
    extract_queue = server._paper_extract_queue     # type: ignore[attr-defined]

    # One-time download_status migration — normalize every legacy status to the clean
    # routing enum BEFORE the queues' recovery scans read a record. Idempotent.
    from ..services.migrate_status import (
        STUB_DELETION_RULE, migrate_status, should_persist)

    report = migrate_status(library)
    if should_persist(report):
        library.save()
        stub_deletions = report.get("by_rule", {}).get(STUB_DELETION_RULE, 0)
        log.info("download_status migration: normalized %d/%d records "
                 "(%d stub deletions) %s",
                 report["migrated"], report["total"], stub_deletions,
                 report["by_rule"])

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

    async def shutdown() -> None:
        reconcile_task.cancel()
        mineru_monitor_task.cancel()
        for _t in (reconcile_task, mineru_monitor_task):
            try:
                await _t
            except (asyncio.CancelledError, Exception):
                pass
        await download_queue.stop()
        await extract_queue.stop()

    return shutdown
