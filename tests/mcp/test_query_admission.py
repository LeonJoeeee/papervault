"""Admission follows a real MCP query's execution, independently of its waiters."""
import asyncio
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from papervault.knowledge.mcp import server
from papervault.mcp import admission
from papervault.mcp.access_log import install_access_log


@pytest.fixture
async def query_tool(monkeypatch):
    for name in ("_session_sems", "_session_active", "_depth", "_recent"):
        monkeypatch.setattr(admission, name, {})
    monkeypatch.setattr(server, "_queries_by_id", {})
    monkeypatch.setattr(server, "_queries_by_intent", {})
    monkeypatch.setattr(server, "_query_stopping", False)
    monkeypatch.setenv("KS_MCP_QUERY_WAIT_S", "0.01")
    monkeypatch.setattr(admission, "_SESSION_INFLIGHT", 1)
    monkeypatch.setattr(admission, "_MAX_WAIT_S", 1)
    monkeypatch.setattr(admission, "_LANES", 1)
    state = SimpleNamespace(started=asyncio.Event(), release=asyncio.Event(), calls=[], now=1000.0)

    async def authority():
        pass

    async def run(*, intent):
        state.calls.append(intent)
        state.started.set()
        await state.release.wait()
        return {"answer": intent}

    monkeypatch.setattr(server, "_ensure_knowledge_authority", authority)
    monkeypatch.setattr("papervault.knowledge.query.aquery.query", run)
    monkeypatch.setattr(admission, "time", SimpleNamespace(monotonic=lambda: state.now))
    mcp = FastMCP("query-admission-test")
    mcp.tool()(server.query)
    admission.install_admission(mcp)
    install_access_log(mcp)
    state.tool = mcp._tool_manager._tools["query"]
    state.mcp = mcp
    yield state
    tasks = [entry.task for entry in server._queries_by_id.values()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_detached_query_holds_permit_and_records_total_service_once(query_tool):
    running = await query_tool.tool.run({"intent": "question"})
    assert running["status"] == "running"
    assert admission._depth["query"] == 1
    assert admission._session_active == {0: 1}
    assert admission._recent.get("query", []) == []
    # Resume bypasses both a saturated session semaphore and a full Layer 2.
    again = await asyncio.wait_for(query_tool.tool.run({"intent": running["query_id"]}), 0.5)
    assert again["status"] == "running" and admission._depth["query"] == 1
    query_tool.now += 375.0
    query_tool.release.set()
    entry = server._queries_by_id[running["query_id"]]
    await entry.task
    assert admission._depth["query"] == 0
    assert list(admission._recent["query"]) == [375.0]
    assert admission._session_sems == {} and admission._session_active == {}
    # Collecting after completion also bypasses Layer 2 (even with zero lanes).
    admission._LANES = 0
    assert await query_tool.tool.run({"intent": "question"}) == {"answer": "question"}
    assert list(admission._recent["query"]) == [375.0]
    assert query_tool.calls == ["question"]


async def test_cancelled_waiter_keeps_admission_until_execution_finishes(query_tool, monkeypatch):
    monkeypatch.setenv("KS_MCP_QUERY_WAIT_S", "0")
    caller = asyncio.create_task(query_tool.tool.run({"intent": "question"}))
    await query_tool.started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert admission._depth["query"] == 1 and admission._recent.get("query", []) == []
    query_tool.now += 200.0
    query_tool.release.set()
    assert await query_tool.tool.run({"intent": "question"}) == {"answer": "question"}
    assert query_tool.calls == ["question"]
    assert admission._depth["query"] == 0 and list(admission._recent["query"]) == [200.0]


async def test_new_question_queues_behind_detached_session_work(query_tool):
    first = await query_tool.tool.run({"intent": "first"})
    queued = asyncio.create_task(query_tool.tool.run({"intent": "second"}))
    await asyncio.sleep(0.03)
    assert not queued.done() and query_tool.calls == ["first"]
    assert admission._depth["query"] == 1
    query_tool.release.set()
    assert (await queued)["answer"] == "second"
    assert (await query_tool.tool.run({"intent": first["query_id"]}))["answer"] == "first"
    assert len(admission._recent["query"]) == 2


async def test_resume_from_another_session_bypasses_busy_but_new_work_does_not(query_tool):
    from mcp.server.fastmcp import Context
    from mcp.shared.context import RequestContext

    async def send_log_message(**kwargs):
        pass

    def context():
        session = SimpleNamespace(send_log_message=send_log_message)
        request = RequestContext(request_id=1, meta=None, session=session, lifespan_context=None)
        return Context(request_context=request, fastmcp=query_tool.mcp)

    original, new_session = context(), context()
    first = await query_tool.tool.run({"intent": "first"}, context=original)
    busy = await query_tool.tool.run({"intent": "second"}, context=new_session)
    assert busy["status"] == "busy"
    resumed = await query_tool.tool.run({"intent": first["query_id"]}, context=new_session)
    assert resumed["status"] == "running"
    assert admission._depth["query"] == 1 and query_tool.calls == ["first"]


async def test_simultaneous_same_intent_calls_only_consume_one_slot(query_tool):
    a, b = await asyncio.gather(query_tool.tool.run({"intent": " question"}),
                                query_tool.tool.run({"intent": "question "}))
    assert a["query_id"] == b["query_id"]
    assert admission._depth["query"] == 1 and query_tool.calls == ["question"]
    assert admission._session_active == {0: 1}


async def test_fast_query_records_one_completion_and_preserves_public_schema(query_tool):
    query_tool.release.set()
    assert await query_tool.tool.run({"intent": "question"}) == {"answer": "question"}
    assert admission._depth["query"] == 0 and len(admission._recent["query"]) == 1
    assert admission._session_active == {} and server._queries_by_id == {}
    tools = await query_tool.mcp.list_tools()
    schema = tools[0].inputSchema
    assert schema["required"] == ["intent"]
    assert schema["properties"] == {"intent": {"title": "Intent", "type": "string"}}


async def test_failed_background_query_releases_and_records_once(query_tool, monkeypatch):
    from mcp.server.fastmcp.exceptions import ToolError

    async def fail(*, intent):
        query_tool.started.set()
        await query_tool.release.wait()
        raise ValueError("failed query")

    monkeypatch.setattr("papervault.knowledge.query.aquery.query", fail)
    running = await query_tool.tool.run({"intent": "question"})
    query_tool.now += 250
    query_tool.release.set()
    with pytest.raises(ToolError, match="failed query"):
        await query_tool.tool.run({"intent": running["query_id"]})
    assert admission._depth["query"] == 0
    assert list(admission._recent["query"]) == [250.0]
    assert admission._session_active == {} and server._queries_by_id == {}


async def test_shutdown_releases_detached_admission_before_store_cleanup(query_tool, monkeypatch):
    await query_tool.tool.run({"intent": "question"})

    async def close():
        assert admission._depth["query"] == 0
        assert admission._session_active == {} and admission._session_sems == {}

    monkeypatch.setattr("papervault.knowledge.store.graph.close_graph", close)
    monkeypatch.setattr("papervault.knowledge.ledger.store.close_pool", close)
    for name in ("_bg_task", "_startup_task", "_embed_probe_task"):
        monkeypatch.setattr(server, name, None)
    await server.stop_background()
    assert len(admission._recent["query"]) == 1


@pytest.mark.parametrize("kind", ["missing_id", "capacity", "shutdown"])
async def test_control_replies_do_not_replace_real_query_service_estimates(query_tool, monkeypatch, kind):
    if kind == "capacity":
        monkeypatch.setattr(admission, "_SESSION_INFLIGHT", 2)
        monkeypatch.setattr(admission, "_MAX_WAIT_S", 0)
        monkeypatch.setattr(server, "_QUERY_CAPACITY", 1)
        await query_tool.tool.run({"intent": "first"})
    elif kind == "shutdown":
        monkeypatch.setattr(server, "_query_stopping", True)
    intent = "q-000000000000" if kind == "missing_id" else "new question"
    result = await query_tool.tool.run({"intent": intent})
    assert result["status"] == ("not_found" if kind == "missing_id" else "busy")
    if kind == "capacity":
        assert "registry" in result["reason"]
    assert admission._recent.get("query", []) == []
    assert admission._avg_service_s("query") == 300.0
