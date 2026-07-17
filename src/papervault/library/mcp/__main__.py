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
    from .boot import start_background as _start_library_bg
    _shutdown_queues = await _start_library_bg(server, log)

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
