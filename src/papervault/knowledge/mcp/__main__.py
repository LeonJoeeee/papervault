"""KS MCP server entry point.

Run via:
    uv run python -m papervault.knowledge.mcp [--port 8001] [--host 0.0.0.0]

The S4 paper incremental-sync scheduler is NOT spawned here. It runs as an asyncio.Task
inside the MCP server's own event loop, owned by the FastMCP `lifespan` in
papervault.knowledge.mcp.server (SDD §6.6 single-loop hard constraint: get_graph()'s LightRAG
singleton + asyncpg pool + shared_storage asyncio.Lock bind to the loop that first touches
them; a separate thread + asyncio.run would crash with "Future attached to a different
loop" and break the §6.6 delete/insert mutex). It is default-OFF; enable with
KS_AUTO_INGEST_ENABLED=true only after the §6.5 copy-throughput go/no-go (ledger
workspace-isolation已落地, SDD §4.1 — no longer a precondition).
The dead v2 store.scheduler._scheduler_loop was physically removed (dead-v2 sweep, §3).
"""
from __future__ import annotations

import argparse
import logging
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Knowledge System MCP server")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port (default 8001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    parser.add_argument(
        "--transport",
        type=str,
        default="streamable-http",
        choices=["streamable-http", "stdio", "sse"],
        help="MCP transport (default streamable-http to match paper-library)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Importing server builds the FastMCP instance whose `lifespan` owns the S4
    # scheduler task (in this same event loop, default-OFF). See server.py docstring.
    from papervault.knowledge.mcp.server import mcp

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        # Start the S4 scheduler + graph/pool at SERVER STARTUP, not per MCP session. FastMCP's
        # mcp.run("streamable-http") wires ONLY session_manager.run() into the Starlette app lifespan
        # and runs our FastMCP lifespan once PER CLIENT SESSION (lowlevel Server.run → enter lifespan;
        # session manager calls it per connection). So we replicate run_streamable_http_async and WRAP
        # the app lifespan to add the server-lifetime background lifecycle (start/stop_background).
        import uvicorn
        from contextlib import asynccontextmanager as _acm

        from papervault.knowledge.mcp.server import start_background, stop_background

        app = mcp.streamable_http_app()
        _orig_lifespan = app.router.lifespan_context

        @_acm
        async def _server_lifespan(_a):  # runs ONCE at server boot/shutdown (uvicorn's loop)
            async with _orig_lifespan(_a):          # FastMCP's session_manager.run()
                await start_background()             # scheduler + graph/pool, once at boot
                try:
                    yield
                finally:
                    await stop_background()          # graceful cleanup at server shutdown

        app.router.lifespan_context = _server_lifespan
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level.lower())
        uvicorn.Server(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
