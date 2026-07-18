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


def _ctx_for(mcp, session_obj, rid="R1"):
    from mcp.server.fastmcp import Context
    from mcp.shared.context import RequestContext
    rc = RequestContext(request_id=rid, meta=None, session=session_obj, lifespan_context=None)
    return Context(request_context=rc, fastmcp=mcp)


def _build_ctx_tool(gate):
    from typing import Optional

    from mcp.server.fastmcp import Context, FastMCP
    mcp = FastMCP("toy")
    running = []

    @mcp.tool()
    async def query(intent: str, ctx: Optional[Context] = None) -> dict:
        running.append(intent)
        await gate.wait()
        return {"answer": intent}

    install_admission(mcp)
    return mcp, running


async def test_two_sessions_do_not_starve_each_other(monkeypatch):
    # The layer-1 promise itself: a session queues behind ITSELF, other sessions run.
    monkeypatch.setattr(admission, "_SESSION_INFLIGHT", 1)
    gate = asyncio.Event()
    mcp, running = _build_ctx_tool(gate)
    t = mcp._tool_manager._tools["query"]
    s1, s2 = object(), object()
    a = asyncio.create_task(t.run({"intent": "s1-first"}, context=_ctx_for(mcp, s1)))
    await asyncio.sleep(0.05)
    b = asyncio.create_task(t.run({"intent": "s1-second"}, context=_ctx_for(mcp, s1)))
    c = asyncio.create_task(t.run({"intent": "s2-first"}, context=_ctx_for(mcp, s2)))
    await asyncio.sleep(0.05)
    assert "s1-first" in running and "s2-first" in running   # other session NOT starved
    assert "s1-second" not in running                        # same session queues
    gate.set()
    await asyncio.gather(a, b, c)
    assert admission._session_sems == {} and admission._session_active == {}  # eviction


async def test_cancel_mid_run_recovers_depth_and_semaphore(monkeypatch):
    monkeypatch.setattr(admission, "_SESSION_INFLIGHT", 1)
    gate = asyncio.Event()
    mcp, running = _build_ctx_tool(gate)
    t = mcp._tool_manager._tools["query"]
    s1 = object()
    a = asyncio.create_task(t.run({"intent": "doomed"}, context=_ctx_for(mcp, s1)))
    await asyncio.sleep(0.05)
    a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await a
    assert admission._depth["query"] == 0
    assert admission._session_sems == {}                     # permit released + evicted
    gate.set()
    out = await asyncio.wait_for(t.run({"intent": "next"}, context=_ctx_for(mcp, s1)), 1.0)
    assert out == {"answer": "next"}                         # lane usable after cancel


async def test_busy_answer_survives_convert_result(monkeypatch):
    monkeypatch.setattr(admission, "_MAX_WAIT_S", 1.0)
    monkeypatch.setattr(admission, "_LANES", 0)
    mcp, _ = _build()
    admission._record("query", 500.0)
    out = await mcp._tool_manager._tools["query"].run({"intent": "x"}, convert_result=True)
    text = str(out)
    assert "busy" in text and "retry_after_s" in text        # serialized, not rejected


async def test_combined_stack_busy_still_logs_mcpcall(monkeypatch, caplog):
    import logging as _logging

    from papervault.mcp.access_log import install_access_log
    monkeypatch.setattr(admission, "_MAX_WAIT_S", 1.0)
    monkeypatch.setattr(admission, "_LANES", 0)
    mcp = FastMCP("toy")

    @mcp.tool()
    async def query(intent: str) -> dict:
        return {"answer": intent}

    install_admission(mcp)
    install_access_log(mcp)   # production order: access log wraps admission
    admission._record("query", 500.0)
    with caplog.at_level(_logging.INFO):
        out = await mcp._tool_manager._tools["query"].run({"intent": "x"})
    assert out["busy"] is True
    lines = [r.getMessage() for r in caplog.records if "MCPCALL" in r.getMessage()]
    assert len(lines) == 1 and "tool=query" in lines[0]      # busy still leaves the audit line
