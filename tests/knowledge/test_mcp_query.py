"""MCP query owns work across bounded waits, disconnects, and result collection."""
import asyncio
from types import SimpleNamespace

import pytest

from papervault.knowledge.mcp import server


@pytest.fixture
async def pipeline(monkeypatch):
    monkeypatch.setattr(server, "_queries_by_id", {}, raising=False)
    monkeypatch.setattr(server, "_queries_by_intent", {}, raising=False)
    monkeypatch.setattr(server, "_query_stopping", False, raising=False)
    monkeypatch.setenv("KS_MCP_QUERY_WAIT_S", "0.01")
    monkeypatch.setenv("KS_MCP_QUERY_RETENTION_S", "900")
    state = SimpleNamespace(started=asyncio.Event(), release=asyncio.Event(),
                            cancelled=asyncio.Event(), calls=[], checks=0, error=None)

    async def authority():
        state.checks += 1

    async def run(*, intent):
        state.calls.append(intent)
        state.started.set()
        try:
            await state.release.wait()
        except asyncio.CancelledError:
            state.cancelled.set()
            raise
        if state.error:
            raise state.error
        return {"answer": intent, "cited_papers": ["Paper2026"]}

    monkeypatch.setattr(server, "_ensure_knowledge_authority", authority)
    monkeypatch.setattr("papervault.knowledge.query.aquery.query", run)
    yield state
    tasks = [entry.task for entry in server._queries_by_id.values()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _running(intent="question"):
    # An absent soft wait must fail promptly, rather than hang a unit-test worker.
    result = await asyncio.wait_for(server.query(intent), 0.5)
    assert result["status"] == "running"
    assert result["running"] is True
    assert result["query_id"].startswith("q-")
    assert result["elapsed_s"] >= 0
    assert result["retry_after_s"] == 5
    assert "intent" in result["reason"] and result["query_id"] in result["reason"]
    return result


async def test_fast_completions_are_not_cached(pipeline):
    pipeline.release.set()
    for _ in range(2):
        assert await server.query(" question ") == {
            "answer": "question", "cited_papers": ["Paper2026"]}
        assert server._queries_by_id == {} and server._queries_by_intent == {}
    assert pipeline.calls == ["question", "question"]
    assert pipeline.checks == 2


@pytest.mark.parametrize("by_id", [False, True])
async def test_soft_wait_resume_collects_once_without_restarting(pipeline, by_id):
    running = await _running(" question ")
    retry = running["query_id"] if by_id else "question"
    assert (await _running(retry))["query_id"] == running["query_id"]
    pipeline.release.set()
    assert await server.query(retry) == {"answer": "question", "cited_papers": ["Paper2026"]}
    assert pipeline.calls == ["question"] and pipeline.checks == 1
    assert server._queries_by_id == {} and server._queries_by_intent == {}


@pytest.mark.parametrize("wait", ["0", "-1"])
async def test_nonpositive_wait_blocks_until_completion(pipeline, monkeypatch, wait):
    monkeypatch.setenv("KS_MCP_QUERY_WAIT_S", wait)
    caller = asyncio.create_task(server.query("question"))
    await pipeline.started.wait()
    await asyncio.sleep(0.02)
    assert not caller.done()
    pipeline.release.set()
    assert (await caller)["answer"] == "question"
    assert server._queries_by_id == {}


@pytest.mark.parametrize("by_id", [False, True])
async def test_client_cancellation_keeps_execution_resumable(pipeline, monkeypatch, by_id):
    monkeypatch.setenv("KS_MCP_QUERY_WAIT_S", "0")
    caller = asyncio.create_task(server.query(" question "))
    await pipeline.started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert not pipeline.cancelled.is_set()
    assert len(server._queries_by_id) == 1
    query_id = next(iter(server._queries_by_id))
    pipeline.release.set()
    result = await server.query(query_id if by_id else "question")
    assert result["answer"] == "question" and pipeline.calls == ["question"]


async def test_authority_check_is_also_inside_shielded_execution(pipeline, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def authority():
        pipeline.checks += 1
        entered.set()
        await release.wait()

    monkeypatch.setattr(server, "_ensure_knowledge_authority", authority)
    caller = asyncio.create_task(server.query("question"))
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    release.set()
    pipeline.release.set()
    assert (await server.query("question"))["answer"] == "question"
    assert pipeline.checks == 1 and pipeline.calls == ["question"]


async def test_concurrent_callers_share_execution(pipeline):
    a, b = await asyncio.gather(_running(" question"), _running("question "))
    assert a["query_id"] == b["query_id"]
    collectors = [asyncio.create_task(server.query(intent)) for intent in ("question", a["query_id"])]
    await asyncio.sleep(0)  # Both collectors attach before execution finishes.
    pipeline.release.set()
    first, second = await asyncio.gather(*collectors)
    assert first == second and first["answer"] == "question"
    assert pipeline.calls == ["question"]


async def test_failure_after_detach_is_delivered_on_collect_and_removed(pipeline):
    running = await _running()
    pipeline.error = ValueError("synthesis failed")
    pipeline.release.set()
    with pytest.raises(ValueError, match="synthesis failed"):
        await server.query(running["query_id"])
    assert pipeline.calls == ["question"]
    assert server._queries_by_id == {} and server._queries_by_intent == {}


async def test_failed_session_logging_cannot_discard_a_completed_answer(pipeline):
    from mcp.server.fastmcp import Context
    from mcp.shared.context import RequestContext

    running = await _running()
    pipeline.release.set()
    await server._queries_by_id[running["query_id"]].task

    async def send_log_message(**kwargs):
        raise RuntimeError("request notification stream closed")

    request = RequestContext(request_id=1, meta=None,
                             session=SimpleNamespace(send_log_message=send_log_message),
                             lifespan_context=None)
    context = Context(request_context=request, fastmcp=server.mcp)
    assert (await server.query(running["query_id"], context))["answer"] == "question"
    assert pipeline.calls == ["question"]


async def test_cancellation_during_heartbeat_cleanup_keeps_completed_answer(pipeline, monkeypatch):
    from mcp.server.fastmcp import Context
    from mcp.shared.context import RequestContext
    from mcp.types import RequestParams

    running = await _running()
    monkeypatch.setenv("KS_MCP_QUERY_WAIT_S", "0")
    monkeypatch.setattr(server, "_HEARTBEAT_INTERVAL_S", 0.001)
    sending, cleaning = asyncio.Event(), asyncio.Event()

    async def send_progress_notification(**kwargs):
        sending.set()
        try:
            await asyncio.Event().wait()
        finally:
            # Transport sends can have asynchronous cancellation cleanup.
            cleaning.set()
            await asyncio.Event().wait()

    async def send_log_message(**kwargs):
        pass

    session = SimpleNamespace(send_progress_notification=send_progress_notification,
                              send_log_message=send_log_message)
    request = RequestContext(request_id=1, meta=RequestParams.Meta(progressToken="progress"),
                             session=session, lifespan_context=None)
    caller = asyncio.create_task(server.query(running["query_id"],
                                Context(request_context=request, fastmcp=server.mcp)))
    await asyncio.wait_for(sending.wait(), 1)
    pipeline.release.set()
    await asyncio.wait_for(cleaning.wait(), 1)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert (await server.query(running["query_id"]))["answer"] == "question"
    assert pipeline.calls == ["question"]


async def test_retention_expires_completed_results_but_not_active_work(pipeline, monkeypatch):
    monkeypatch.setenv("KS_MCP_QUERY_RETENTION_S", "0")
    running = await _running()
    # Retention must never evict a still-executing query and launch duplicate work.
    assert (await _running())["query_id"] == running["query_id"]
    entry = server._queries_by_id[running["query_id"]]
    pipeline.release.set()
    await entry.task
    result = await server.query(running["query_id"])
    assert result["status"] == "not_found"
    assert server._queries_by_id == {} and server._queries_by_intent == {}
    assert pipeline.calls == ["question"]


async def test_capacity_never_evicts_active_work(pipeline, monkeypatch):
    monkeypatch.setattr(server, "_QUERY_CAPACITY", 1, raising=False)
    first = await _running("first")
    refused = await asyncio.wait_for(server.query("second"), 0.5)
    assert refused["status"] == "busy"
    assert pipeline.calls == ["first"]
    assert (await _running("first"))["query_id"] == first["query_id"]
    pipeline.release.set()
    await server._queries_by_id[first["query_id"]].task
    assert (await server.query("second"))["answer"] == "second"
    assert len(server._queries_by_id) <= 1 and pipeline.calls == ["first", "second"]


async def test_shutdown_drains_queries_before_closing_stores(pipeline, monkeypatch):
    await _running()
    closed = []

    async def close():
        assert pipeline.cancelled.is_set()
        assert server._queries_by_id == {} and server._queries_by_intent == {}
        closed.append(True)

    monkeypatch.setattr("papervault.knowledge.store.graph.close_graph", close)
    monkeypatch.setattr("papervault.knowledge.ledger.store.close_pool", close)
    monkeypatch.setattr(server, "_bg_task", None)
    monkeypatch.setattr(server, "_startup_task", None)
    monkeypatch.setattr(server, "_embed_probe_task", None)
    await server.stop_background()
    assert closed == [True, True]
    # A caller racing shutdown must not create new work after the drain snapshot.
    result = await server.query("new question")
    assert result["status"] == "busy" and pipeline.calls == ["question"]


async def test_per_session_startup_cannot_reopen_query_admission_during_shutdown(pipeline, monkeypatch):
    cancelling, drain = asyncio.Event(), asyncio.Event()
    active = []

    async def run(*, intent):
        active.append(intent)
        pipeline.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await drain.wait()
            raise
        finally:
            active.remove(intent)

    async def close():
        assert active == []

    monkeypatch.setattr("papervault.knowledge.query.aquery.query", run)
    monkeypatch.setattr("papervault.knowledge.store.graph.close_graph", close)
    monkeypatch.setattr("papervault.knowledge.ledger.store.close_pool", close)
    for name in ("_bg_task", "_startup_task", "_embed_probe_task", "_authority_task"):
        monkeypatch.setattr(server, name, None)
    await _running()
    stopping = asyncio.create_task(server.stop_background())
    await cancelling.wait()
    try:
        await server.start_background()  # The hook is also called by new HTTP sessions.
        result = await asyncio.wait_for(server.query("second"), 0.5)
        assert result["status"] == "busy" and active == ["question"]
    finally:
        drain.set()
        await stopping
