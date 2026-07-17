"""Entry point for the paper-library MCP server.

Default transport is **streamable-http** (a long-running daemon on
127.0.0.1:8765) because paper-library is designed as a single shared
service for all consumer projects on this machine. ``--stdio`` is
retained for clients that spawn the server as a subprocess (Claude
Desktop / Cursor's MCP integration).

Usage::

    paper-library-mcp                       # streamable-http on 127.0.0.1:8765
    paper-library-mcp --port 9999           # streamable-http on a different port
    paper-library-mcp --host 0.0.0.0        # bind all interfaces (cross-machine)
    paper-library-mcp --stdio               # subprocess transport for Claude Desktop
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from .server import build_server


async def _run_async(args, log):
    """Construct the server, start the 2 BG stage-queue worker pools
    (download → extract), run the transport, stop the pools on shutdown.

    Start order is **downstream-first** so each upstream queue's
    ``on_success`` callback fires into an already-running consumer.
    Stop order is the reverse, so an in-flight upstream task can still
    push to a live downstream queue until cancellation cascades.

    ✦ Phase 28 (2026-05-24, route B): the insight queue (3rd stage) was
    removed. paper-library is now a mechanical fetch/extract/MCP service;
    all 5-Q digest intelligence lives in the research-side ``librarian/``
    curators. See ``services/paper-library/src/papervault.library/insight/
    __init__.py`` for the read-side shim that preserves ``Paper.insight``
    deserialization for the ~800 legacy records on disk.
    """
    server = build_server(library_path=args.library_path)
    library = server._paper_library  # type: ignore[attr-defined]
    download_queue = server._paper_download_queue  # type: ignore[attr-defined]
    extract_queue = server._paper_extract_queue  # type: ignore[attr-defined]

    # D7 one-time status migration — normalize every legacy download_status
    # ("ok:<src>", "text-only:firecrawl", "extract_low_quality") to the clean
    # routing enum BEFORE the queues' recovery scans (which route on
    # classify()/the enum) ever read a record. Idempotent: a no-op once the
    # index is already in canonical shape.
    from ..services.migrate_status import (
        STUB_DELETION_RULE, migrate_status, should_persist)
    report = migrate_status(library)
    # R2/redrill: the save-guard predicate lives in exactly ONE place
    # (``should_persist`` in services.migrate_status) so it can't drift from the
    # regression test, which calls the SAME helper. The save must fire when the
    # migration normalized any download_status VALUE (``report["migrated"]``) OR
    # when the ONLY mutation was a Pass A stub deletion: Pass A clears md_path /
    # md_engine / extract_attempts in memory and deletes the on-disk stub but
    # does NOT bump ``migrated`` (that counter tracks download_status value
    # migrations). A boot whose sole mutation is a stub deletion (e.g. a row
    # already canonical ``ok`` carrying an un-gated stub) would otherwise skip
    # save() → md_path=None never persists → index.json still points at the
    # now-deleted file.
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

    # D8 reconcile sweep — the automatic safety net for papers that fell
    # between the two conveyor belts (esp. the ~48 firecrawl-md papers the
    # download queue's pending-only recovery scan forgets). ``reconcile_loop``
    # runs ``reconcile_once`` immediately at the top of its first iteration
    # (before its first sleep), so we do NOT call it explicitly here — doing
    # so would run an identical full-library scan (one pdf_probe subprocess per
    # EXTRACT paper) twice back-to-back at boot for no benefit (low fix #2).
    # The queues are already up, so the loop's first sweep's add()s land.
    from ..services.reconcile import reconcile_loop
    reconcile_task = asyncio.create_task(
        reconcile_loop(library, download_queue, extract_queue),
        name="paper-reconcile-loop",
    )
    log.info("paper-reconcile loop started")

    # On-demand MinerU server lifecycle (no-op unless PAPER_LIBRARY_MINERU_ONDEMAND=1):
    # stop the vLLM server after the extract queue is idle for IDLE_TIMEOUT,
    # freeing the GPU; ensure_ready (on the extract worker path) starts it back.
    from ..services.mineru_server import get_server_controller
    mineru_monitor_task = asyncio.create_task(
        get_server_controller().monitor_loop(extract_queue),
        name="mineru-idle-monitor",
    )
    log.info("mineru on-demand monitor task created (active only when on-demand ON)")

    async def _shutdown_queues():
        reconcile_task.cancel()
        mineru_monitor_task.cancel()
        for _t in (reconcile_task, mineru_monitor_task):
            try:
                await _t
            except (asyncio.CancelledError, Exception):
                pass
        await download_queue.stop()
        await extract_queue.stop()

    if args.stdio:
        log.info("MCP stdio server ready")
        try:
            await server.run_stdio_async()
        finally:
            await _shutdown_queues()
        return

    server.settings.host = args.host
    server.settings.port = args.port
    log.info("streamable-http MCP server listening on %s:%d",
             args.host, args.port)

    # Wire SIGINT/SIGTERM to clean BG shutdown
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal(signame):
        log.info("received %s; shutting down", signame)
        stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(signal, sig_name),
                                    _on_signal, sig_name)
        except NotImplementedError:
            # Windows / some sandboxed environments
            pass

    server_task = asyncio.create_task(server.run_streamable_http_async())
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        await asyncio.wait(
            {server_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        if not server_task.done():
            server_task.cancel()
            try:
                await server_task
            except (asyncio.CancelledError, Exception):
                pass
        await _shutdown_queues()


def main():
    parser = argparse.ArgumentParser(prog="paper-library-mcp")
    parser.add_argument("--stdio", action="store_true",
                        help="serve over stdio instead of streamable-http "
                             "(use for Claude Desktop / subprocess clients)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="streamable-http bind host (default 127.0.0.1, "
                             "loopback only — change to 0.0.0.0 for cross-machine)")
    parser.add_argument("--port", type=int, default=8765,
                        help="streamable-http bind port (default 8765)")
    parser.add_argument("--library-path", default=None,
                        help="paper library root (default $PAPER_LIBRARY_PATH "
                             "or ~/paper-vault)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("paper-library-mcp")

    if (not args.stdio
            and args.host != "127.0.0.1"
            and not os.environ.get("PAPER_LIBRARY_MCP_TOKEN")):
        log.warning(
            "HTTP server binding to %s without PAPER_LIBRARY_MCP_TOKEN — "
            "server is OPEN. Acceptable on Tailscale or private-LAN-only "
            "networks; do NOT expose to the public internet.", args.host
        )

    asyncio.run(_run_async(args, log))


if __name__ == "__main__":
    main()
