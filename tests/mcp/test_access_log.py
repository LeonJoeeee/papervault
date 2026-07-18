"""Unit tests for the MCPCALL access log (issue #13, item 1).

Exercises the wrapper through the real FastMCP Tool.run path (arg validation +
keyword invocation), not by calling the wrapper directly — so a change in how the
SDK invokes Tool.fn breaks these tests, exactly when the wrap would break live.
"""
import asyncio
import logging

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from papervault.mcp.access_log import install_access_log


def _build_toy() -> FastMCP:
    mcp = FastMCP("toy")

    @mcp.tool()
    async def echo(text: str, items: list[str] | None = None) -> dict:
        return {"text": text, "n": len(items or [])}

    @mcp.tool()
    async def boom(text: str) -> dict:
        raise ValueError("kaput")

    @mcp.tool()
    async def hang(text: str) -> dict:
        await asyncio.sleep(60)
        return {}

    install_access_log(mcp)
    return mcp


def _mcpcall_lines(caplog):
    return [r.getMessage() for r in caplog.records if "MCPCALL" in r.getMessage()]


async def test_ok_call_logs_and_preserves_result(caplog):
    mcp = _build_toy()
    with caplog.at_level(logging.INFO, logger="papervault.mcp.access"):
        result = await mcp._tool_manager._tools["echo"].run(
            {"text": "hello", "items": ["a", "b"]}
        )
    assert result == {"text": "hello", "n": 2}
    (line,) = _mcpcall_lines(caplog)
    assert "tool=echo" in line
    assert "outcome=ok" in line
    assert "text:len=5" in line and "items:n=2" in line
    assert "dur=" in line


async def test_error_call_logs_type_and_still_raises(caplog):
    mcp = _build_toy()
    with caplog.at_level(logging.INFO, logger="papervault.mcp.access"):
        with pytest.raises(ToolError, match="kaput"):
            await mcp._tool_manager._tools["boom"].run({"text": "x"})
    (line,) = _mcpcall_lines(caplog)
    assert "tool=boom" in line
    assert "outcome=error:ValueError" in line


async def test_cancelled_call_logs_cancelled(caplog):
    mcp = _build_toy()
    with caplog.at_level(logging.INFO, logger="papervault.mcp.access"):
        task = asyncio.create_task(
            mcp._tool_manager._tools["hang"].run({"text": "x"})
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    (line,) = _mcpcall_lines(caplog)
    assert "tool=hang" in line
    assert "outcome=cancelled" in line


async def test_schema_rejected_call_never_reaches_wrapper(caplog):
    mcp = _build_toy()
    with caplog.at_level(logging.INFO, logger="papervault.mcp.access"):
        with pytest.raises(ToolError):
            await mcp._tool_manager._tools["echo"].run({"nope": 1})
    assert _mcpcall_lines(caplog) == []


def test_install_survives_missing_internals(caplog):
    class Hollow:
        pass

    with caplog.at_level(logging.WARNING, logger="papervault.mcp.access"):
        install_access_log(Hollow())  # must not raise
    assert any("NOT installed" in r.getMessage() for r in caplog.records)
