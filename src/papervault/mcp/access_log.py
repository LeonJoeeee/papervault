"""Structured per-call access log for the MCP tools (issue #13, item 1).

Emits ONE INFO line per tool call on logger ``papervault.mcp.access``, anchored by
the grep-stable token ``MCPCALL``:

    MCPCALL seq=12 tool=query req=abc123 dur=171.42s outcome=ok args=intent:len=184

Fields:
  * ``seq``     — per-process monotonically increasing call number
  * ``tool``    — registered tool name (query / search_papers / get_paper)
  * ``req``     — MCP request id when a Context was injected, else ``-`` (correlates
                  with the S17 progress-heartbeat lines, which carry the same id)
  * ``dur``     — wall-clock seconds, monotonic
  * ``outcome`` — ``ok`` | ``error:<ExceptionType>`` | ``cancelled`` (request aborted /
                  client gone before completion)
  * ``args``    — size fingerprint only (string lengths, list arity). Never contents:
                  call payloads stay out of the log by design.

The wrap replaces ``Tool.fn`` AFTER registration, so parameter schemas (built from
the original signatures at registration time) are untouched, and validation still
runs before the wrapper — a schema-rejected call never reaches it (that failure is
logged by the SDK layer, not here).

If the FastMCP internals this hooks (``_tool_manager._tools[*].fn``) ever change
shape under an SDK bump, installation logs a WARNING and leaves serving untouched:
observability must never take the server down.
"""
from __future__ import annotations

import asyncio
import functools
import itertools
import logging
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

log = logging.getLogger("papervault.mcp.access")

_seq = itertools.count(1)


def _fingerprint(kwargs: dict[str, Any]) -> str:
    parts = []
    for k, v in kwargs.items():
        if hasattr(v, "request_id"):  # injected Context — logged via req=, not args=
            continue
        if isinstance(v, str):
            parts.append(f"{k}:len={len(v)}")
        elif isinstance(v, (list, tuple)):
            parts.append(f"{k}:n={len(v)}")
        elif isinstance(v, (int, float, bool)) or v is None:
            parts.append(f"{k}={v}")
    return ",".join(parts) if parts else "-"


def _request_id(kwargs: dict[str, Any]) -> str:
    for v in kwargs.values():
        rid = getattr(v, "request_id", None)
        if rid is not None:
            return str(rid)
    return "-"


def _wrap(name: str, fn):
    @functools.wraps(fn)
    async def logged(**kwargs):
        seq = next(_seq)
        t0 = time.monotonic()
        outcome = "ok"
        try:
            return await fn(**kwargs)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException as exc:
            outcome = f"error:{type(exc).__name__}"
            raise
        finally:
            log.info(
                "MCPCALL seq=%d tool=%s req=%s dur=%.2fs outcome=%s args=%s",
                seq, name, _request_id(kwargs),
                time.monotonic() - t0, outcome, _fingerprint(kwargs),
            )

    return logged


def install_access_log(mcp: FastMCP) -> None:
    """Wrap every registered tool's ``fn`` with the MCPCALL access logger."""
    try:
        tools = mcp._tool_manager._tools
    except AttributeError:
        log.warning("access log NOT installed: FastMCP tool-manager internals changed")
        return
    for name, tool in tools.items():
        if not getattr(tool, "is_async", True):
            # All papervault tools are async; a future sync tool would need an
            # is_async flip alongside the async wrapper — refuse rather than guess.
            log.warning("access log skipped sync tool %s", name)
            continue
        tool.fn = _wrap(name, tool.fn)
    log.info("access log installed on %d tools: %s", len(tools), ", ".join(tools))
