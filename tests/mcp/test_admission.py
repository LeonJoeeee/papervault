"""Unit tests for the admission layers (issue #28) — through the real Tool.run path."""
import asyncio

import pytest
from mcp.server.fastmcp import FastMCP

from papervault.mcp import admission
from papervault.mcp.admission import install_admission


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(admission, "_session_sems", {})
    monkeypatch.setattr(admission, "_depth", {})
    monkeypatch.setattr(admission, "_recent", {})


def _build(gate: asyncio.Event | None = None):
    mcp = FastMCP("toy")
    running = []

    @mcp.tool()
    async def query(intent: str) -> dict:
        running.append(intent)
        if gate is not None:
            await gate.wait()
        return {"answer": intent}

    @mcp.tool()
    async def get_paper(identifiers: str) -> dict:
        return {"ok": identifiers}

    install_admission(mcp)
    return mcp, running


async def test_light_tool_not_wrapped():
    mcp, _ = _build()
    fn = mcp._tool_manager._tools["get_paper"].fn
    assert fn.__name__ != "admitted"
    assert (await mcp._tool_manager._tools["get_paper"].run({"identifiers": "x"})) == {"ok": "x"}


async def test_normal_call_passes_and_records(monkeypatch):
    mcp, _ = _build()
    out = await mcp._tool_manager._tools["query"].run({"intent": "q1"})
    assert out == {"answer": "q1"}
    assert len(admission._recent["query"]) == 1


async def test_session_cap_serializes_same_session(monkeypatch):
    monkeypatch.setattr(admission, "_SESSION_INFLIGHT", 1)
    gate = asyncio.Event()
    mcp, running = _build(gate)
    t = mcp._tool_manager._tools["query"]
    a = asyncio.create_task(t.run({"intent": "first"}))
    await asyncio.sleep(0.05)
    b = asyncio.create_task(t.run({"intent": "second"}))
    await asyncio.sleep(0.05)
    # cap 1, same (no-context) session bucket: second must NOT have started
    assert running == ["first"]
    gate.set()
    assert (await a) == {"answer": "first"}
    assert (await b) == {"answer": "second"}
    assert running == ["first", "second"]


async def test_busy_answer_when_projected_wait_exceeds(monkeypatch):
    monkeypatch.setattr(admission, "_MAX_WAIT_S", 100.0)
    monkeypatch.setattr(admission, "_LANES", 1)
    gate = asyncio.Event()
    mcp, running = _build(gate)
    t = mcp._tool_manager._tools["query"]
    admission._record("query", 300.0)  # avg 300s: depth 2 -> projected 300 > 100
    a = asyncio.create_task(t.run({"intent": "occupies"}))
    await asyncio.sleep(0.05)
    out = await t.run({"intent": "rejected"})
    assert out["busy"] is True
    assert out["queue_depth"] == 2
    assert out["expected_wait_s"] == 300
    assert out["retry_after_s"] == 360
    assert running == ["occupies"]      # rejected call never executed
    gate.set()
    await a
    assert admission._depth["query"] == 0  # busy path never leaked depth


async def test_layer2_disabled_by_zero(monkeypatch):
    monkeypatch.setattr(admission, "_MAX_WAIT_S", 0.0)
    monkeypatch.setattr(admission, "_LANES", 0)
    mcp, _ = _build()
    admission._record("query", 10_000.0)
    out = await mcp._tool_manager._tools["query"].run({"intent": "still served"})
    assert out == {"answer": "still served"}


async def test_depth_recovers_after_exception():
    mcp = FastMCP("toy")

    @mcp.tool()
    async def query(intent: str) -> dict:
        raise ValueError("boom")

    install_admission(mcp)
    from mcp.server.fastmcp.exceptions import ToolError
    with pytest.raises(ToolError):
        await mcp._tool_manager._tools["query"].run({"intent": "x"})
    assert admission._depth["query"] == 0
