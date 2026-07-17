"""MCP server exposing the paper library to outside LLM clients.

A single entry point is provided — ``python -m papervault.library.mcp`` (see
:mod:`__main__`). It defaults to the streamable-http transport (a long-running
daemon for cross-machine access over Tailscale / LAN); pass ``--stdio`` for
clients that spawn the server as a subprocess (Claude Desktop / Cursor). Both
transports share the same tool/resource definitions (see :mod:`server`).
"""

from .server import build_server

__all__ = ["build_server"]
