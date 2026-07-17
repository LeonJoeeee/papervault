"""MCP server exposing the paper library to outside LLM clients.

Two entry points are provided:

  - ``paper-library-mcp-stdio`` — for Claude Desktop / Cursor (per-conversation
    process, stdio transport).
  - ``paper-library-mcp-http`` — for cross-machine access over Tailscale / LAN.

Both share the same tool/resource definitions (see :mod:`server`).
"""

from .server import build_server

__all__ = ["build_server"]
