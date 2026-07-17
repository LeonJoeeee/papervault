"""Entry point for the unified papervault MCP server (ADR-0002: one process, three tools).

Default transport is **streamable-http** (a long-running daemon). ``--stdio`` is retained
for clients that spawn the server as a subprocess (Claude Desktop / some IDE integrations).

Both background planes are started ONCE at server boot (not per client session): the library
stage queues + reconcile + on-demand MinerU monitor, and the knowledge scheduler + graph/pool.
Under streamable-http a FastMCP lifespan runs once per client connection, so — mirroring the
knowledge server's proven pattern — we wrap the Starlette app lifespan to own the
server-lifetime lifecycle.

Usage::

    papervault-mcp                      # streamable-http on 127.0.0.1:8080
    papervault-mcp --port 9999
    papervault-mcp --stdio              # subprocess transport
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import os
import sys
from contextlib import asynccontextmanager

log = logging.getLogger("papervault.mcp")


class _BearerAuthMiddleware:
    """Pure-ASGI gate requiring ``Authorization: Bearer <token>`` on every HTTP request.

    Wraps the streamable-http app ONLY when ``PAPERVAULT_MCP_TOKEN`` is set (non-empty).
    It inspects the request headers on the incoming scope and short-circuits a 401 JSON
    before the app runs — it never touches the response stream, so it's safe for the
    server's SSE/streaming responses. Non-HTTP scopes (lifespan, websocket) pass straight
    through, so the server-lifetime lifecycle is unaffected. Constant-time compare avoids
    leaking the token via timing.
    """

    def __init__(self, app, token: str) -> None:
        self._app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            provided = dict(scope.get("headers") or {}).get(b"authorization", b"")
            if not hmac.compare_digest(provided, self._expected):
                body = (b'{"error":"unauthorized",'
                        b'"detail":"missing or invalid bearer token"}')
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
        await self._app(scope, receive, send)


def main() -> int:
    parser = argparse.ArgumentParser(prog="papervault-mcp", description="papervault MCP server")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="bind port (default 8080)")
    parser.add_argument("--stdio", action="store_true", help="serve over stdio instead of HTTP")
    parser.add_argument("--library-path", default=None, help="vault root (default from config)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from papervault.knowledge.mcp.server import start_background as start_knowledge_bg
    from papervault.knowledge.mcp.server import stop_background as stop_knowledge_bg
    from papervault.library.mcp.boot import start_background as start_library_bg
    from papervault.mcp.server import build_server

    server = build_server(library_path=args.library_path)

    if args.stdio:
        asyncio.run(_run_stdio(server, start_library_bg, start_knowledge_bg, stop_knowledge_bg))
        return 0

    import uvicorn

    server.settings.host = args.host
    server.settings.port = args.port
    app = server.streamable_http_app()
    _orig_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _server_lifespan(_a):  # runs ONCE at server boot/shutdown (uvicorn's loop)
        async with _orig_lifespan(_a):                       # FastMCP session_manager.run()
            shutdown_library = await start_library_bg(server, log)   # queues + reconcile + mineru
            await start_knowledge_bg()                       # scheduler + graph/pool
            log.info("papervault MCP server ready on %s:%d", args.host, args.port)
            try:
                yield
            finally:
                await stop_knowledge_bg()
                await shutdown_library()

    app.router.lifespan_context = _server_lifespan

    # Optional bearer-token auth (unset ⇒ byte-identical to prior behavior). The unified
    # server binds loopback with no auth by default; a token lets an operator expose it on
    # a trusted non-loopback address behind `Authorization: Bearer <token>`.
    asgi_app = app
    token = os.environ.get("PAPERVAULT_MCP_TOKEN", "").strip()
    if token:
        asgi_app = _BearerAuthMiddleware(app, token)
        log.info("bearer-token auth ENABLED on the HTTP transport (PAPERVAULT_MCP_TOKEN set)")

    config = uvicorn.Config(asgi_app, host=args.host, port=args.port, log_level=args.log_level.lower())
    uvicorn.Server(config).run()
    return 0


async def _run_stdio(server, start_library_bg, start_knowledge_bg, stop_knowledge_bg) -> None:
    shutdown_library = await start_library_bg(server, log)
    await start_knowledge_bg()
    log.info("papervault MCP stdio server ready")
    try:
        await server.run_stdio_async()
    finally:
        await stop_knowledge_bg()
        await shutdown_library()


if __name__ == "__main__":
    sys.exit(main())
