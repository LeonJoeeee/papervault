"""Admission and eval-caller tests for bounded-wait resumable search_papers (#354)."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.context import RequestContext

from papervault.knowledge.mcp import server as knowledge_server
from papervault.library import Library
from papervault.library.mcp import server as library_server
from papervault.mcp import admission
from papervault.mcp.access_log import install_access_log

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx_for(mcp: FastMCP, session_obj: object, rid: str = "R1") -> Context:
    rc = RequestContext(request_id=rid, meta=None, session=session_obj, lifespan_context=None)
    return Context(request_context=rc, fastmcp=mcp)


@pytest.fixture
async def search_tool(tmp_path, monkeypatch):
    for name in ("_session_sems", "_session_active", "_depth", "_recent"):
        monkeypatch.setattr(admission, name, {})
    monkeypatch.setenv("PAPERVAULT_MCP_SEARCH_WAIT_S", "0.01")
    monkeypatch.setattr(admission, "_SESSION_INFLIGHT", 1)
    monkeypatch.setattr(admission, "_MAX_WAIT_S", 1.0)
    monkeypatch.setattr(admission, "_LANES", 1)

    state = SimpleNamespace(
        started=asyncio.Event(),
        release=asyncio.Event(),
        calls=[],
        now=1000.0,
    )

    monkeypatch.setattr(library_server, "get_llm", lambda: SimpleNamespace())
    monkeypatch.setattr(
        library_server,
        "parse_intent",
        lambda q, llm=None: {
            "search_terms": [q],
            "filters": {},
            "limit_suggested": 5,
            "ranking_hint": "by_relevance",
            "reasoning": "test",
        },
    )

    async def fake_external(terms, *, year_min=None, year_max=None, ranking_hint="by_relevance"):
        state.calls.append(list(terms))
        state.started.set()
        await state.release.wait()
        return [
            {
                "title": f"Paper for {terms[0]}",
                "authors": ["Author, A."],
                "year": 2025,
                "venue": "ApJ",
                "abstract": "Detailed space physics abstract.",
                "doi": "10.1234/test-search",
                "arxiv_id": "",
                "citation_count": 10,
                "publication_types": ["JournalArticle"],
                "_source_origin": "external",
                "term_idx": 0,
                "rank": 0,
            }
        ], {}

    async def fake_ingest(cands, *, llm=None):
        return {i: {"reason": "", "tier": "1A", "ingest_ok": True, "llm_is_paper": True} for i in range(len(cands))}, 0

    async def fake_return(cands, intent, *, search_terms=None, filters=None, llm=None):
        return {i: {"reason": "", "score": 0.9} for i in range(len(cands))}, 0

    monkeypatch.setattr(library_server, "search_external_async", fake_external)
    monkeypatch.setattr(library_server, "judge_ingest", fake_ingest)
    monkeypatch.setattr(library_server, "judge_return", fake_return)
    monkeypatch.setattr(admission, "time", SimpleNamespace(monotonic=lambda: state.now))

    lib = Library(tmp_path)
    mcp = library_server.build_server(library=lib)
    admission.install_admission(mcp)
    install_access_log(mcp)
    state.tool = mcp._tool_manager._tools["search_papers"]
    state.mcp = mcp
    yield state
    registry = getattr(mcp, "_paper_search_registry", None)
    if registry is not None:
        await registry.shutdown()


async def test_detached_search_holds_permit_and_records_total_service_once(search_tool):
    running = await search_tool.tool.run({"query": "cosmic ray transport"})
    assert running["status"] == "running"
    assert running["running"] is True
    assert running["search_id"].startswith("s-")
    assert admission._depth["search_papers"] == 1
    assert admission._session_active == {0: 1}
    assert admission._recent.get("search_papers", []) == []

    # Resume bypasses both a saturated session semaphore and Layer 2.
    again = await asyncio.wait_for(search_tool.tool.run({"query": running["search_id"]}), 0.5)
    assert again["status"] == "running"
    assert admission._depth["search_papers"] == 1

    search_tool.now += 240.0
    search_tool.release.set()
    entry = search_tool.mcp._paper_search_registry.by_id[running["search_id"]]
    await entry.task

    assert admission._depth["search_papers"] == 0
    assert list(admission._recent["search_papers"]) == [240.0]
    assert admission._session_sems == {} and admission._session_active == {}

    # Collecting after completion also bypasses Layer 2 even with zero lanes.
    admission._LANES = 0
    collected = await search_tool.tool.run({"query": "cosmic ray transport"})
    assert collected["status"] == "ok" and len(collected["results"]) == 1
    replayed = await search_tool.tool.run({"query": running["search_id"]})
    assert replayed["status"] == "ok" and replayed["results"] == collected["results"]
    assert list(admission._recent["search_papers"]) == [240.0]
    assert search_tool.calls == [["cosmic ray transport"]]


@pytest.mark.parametrize("kind", ["stopping", "missing_id", "empty_query"])
async def test_preadmission_fast_paths_bypass_saturated_layers_and_preserve_estimates(
    search_tool, monkeypatch, kind
):
    # Occupy the sole session slot AND force Layer 2 busy if reached.
    s1, s2 = object(), object()
    running = await search_tool.tool.run(
        {"query": "in-flight search"}, context=_ctx_for(search_tool.mcp, s1)
    )
    assert running["status"] == "running"
    admission._LANES = 0
    admission._MAX_WAIT_S = 1.0

    if kind == "stopping":
        search_tool.mcp._paper_search_registry.stopping = True
        arg = "brand new search"
        expected = "busy"
    elif kind == "missing_id":
        arg = "s-000000000000"
        expected = "not_found"
    else:
        arg = "   \t\n  "
        expected = "error"

    # Runs on BOTH saturated session s1 (would deadlock on sem.acquire without fast-path)
    # and fresh session s2 (would fail Layer 2 busy with queue_depth without fast-path).
    for sess in (s1, s2):
        out = await asyncio.wait_for(
            search_tool.tool.run({"query": arg}, context=_ctx_for(search_tool.mcp, sess)),
            0.5,
        )
        assert out["status"] == expected
        if kind == "stopping":
            assert "shutting down" in out["reason"]
            assert "queue_depth" not in out

    assert admission._depth["search_papers"] == 1
    assert admission._recent.get("search_papers", []) == []
    assert admission._avg_service_s("search_papers") == 320.0

    search_tool.mcp._paper_search_registry.stopping = False
    search_tool.release.set()
    await search_tool.mcp._paper_search_registry.by_id[running["search_id"]].task


async def test_registry_capacity_busy_does_not_leak_depth_or_pollute_estimates(
    search_tool, monkeypatch
):
    monkeypatch.setattr(admission, "_SESSION_INFLIGHT", 2)
    monkeypatch.setattr(admission, "_MAX_WAIT_S", 0.0)
    search_tool.mcp._paper_search_registry.capacity = 1

    first = await search_tool.tool.run({"query": "slot one"})
    assert first["status"] == "running"
    busy = await search_tool.tool.run({"query": "slot two"})
    assert busy["status"] == "busy" and busy["busy"] is True
    assert "registry" in busy["reason"]
    assert admission._depth["search_papers"] == 1
    assert admission._recent.get("search_papers", []) == []

    search_tool.release.set()
    await search_tool.mcp._paper_search_registry.by_id[first["search_id"]].task


async def test_query_preadmission_fast_paths_bypass_saturated_session_and_layer2(monkeypatch):
    for name in ("_session_sems", "_session_active", "_depth", "_recent"):
        monkeypatch.setattr(admission, name, {})
    monkeypatch.setattr(knowledge_server, "_queries_by_id", {})
    monkeypatch.setattr(knowledge_server, "_queries_by_intent", {})
    monkeypatch.setattr(knowledge_server, "_query_stopping", False)
    monkeypatch.setenv("KS_MCP_QUERY_WAIT_S", "0.01")
    monkeypatch.setattr(admission, "_SESSION_INFLIGHT", 1)
    monkeypatch.setattr(admission, "_MAX_WAIT_S", 1.0)
    monkeypatch.setattr(admission, "_LANES", 0)

    release = asyncio.Event()

    async def authority():
        pass

    async def fake_query(*, intent):
        await release.wait()
        return {"answer": intent}

    monkeypatch.setattr(knowledge_server, "_ensure_knowledge_authority", authority)
    monkeypatch.setattr("papervault.knowledge.query.aquery.query", fake_query)

    mcp = FastMCP("query-fastpath-test")
    mcp.tool()(knowledge_server.query)
    admission.install_admission(mcp)
    tool = mcp._tool_manager._tools["query"]

    s1 = object()
    admission._LANES = 1
    first = await tool.run({"intent": "active question"}, context=_ctx_for(mcp, s1))
    assert first["status"] == "running"
    admission._LANES = 0  # Saturate Layer 2 as well as s1's Layer 1 permit.

    missing = await asyncio.wait_for(
        tool.run({"intent": "q-deadbeef0000"}, context=_ctx_for(mcp, s1)), 0.5
    )
    assert missing["status"] == "not_found"

    monkeypatch.setattr(knowledge_server, "_query_stopping", True)
    stopping = await asyncio.wait_for(
        tool.run({"intent": "another question"}, context=_ctx_for(mcp, s1)), 0.5
    )
    assert stopping["status"] == "busy" and "shutting down" in stopping["reason"]
    assert admission._recent.get("query", []) == []

    release.set()
    await knowledge_server._queries_by_id[first["query_id"]].task


async def test_queued_same_query_attaches_after_semaphore_acquire(search_tool):
    # Session s1 already holds its 1 permit on "first"; s1 queues "second" while s2
    # starts "second" concurrently. When "first" finishes, s1's queued "second"
    # wakes from sem.acquire() and attaches to s2's already-started "second".
    s1, s2 = object(), object()
    first = await search_tool.tool.run({"query": "first"}, context=_ctx_for(search_tool.mcp, s1))
    assert first["status"] == "running"

    admission._MAX_WAIT_S = 0.0  # Disable Layer 2 so s2 can start "second".
    queued_s1 = asyncio.create_task(
        search_tool.tool.run({"query": " second "}, context=_ctx_for(search_tool.mcp, s1))
    )
    await asyncio.sleep(0.02)
    assert not queued_s1.done()

    started_s2 = await search_tool.tool.run(
        {"query": "second"}, context=_ctx_for(search_tool.mcp, s2)
    )
    assert started_s2["status"] == "running"
    assert search_tool.calls == [["first"], ["second"]]

    # Finish "first" only by completing its task directly so s1's semaphore releases
    # while "second" is still running.
    first_entry = search_tool.mcp._paper_search_registry.by_id[first["search_id"]]
    first_entry.task.cancel()
    await asyncio.gather(first_entry.task, return_exceptions=True)
    search_tool.mcp._paper_search_registry.forget(first_entry)

    attached = await asyncio.wait_for(queued_s1, 0.5)
    assert attached["status"] == "running"
    assert attached["search_id"] == started_s2["search_id"]
    assert search_tool.calls == [["first"], ["second"]]
    assert admission._depth["search_papers"] == 1

    search_tool.release.set()
    await search_tool.mcp._paper_search_registry.by_id[started_s2["search_id"]].task


async def test_toy_search_papers_tool_remains_non_resumable(monkeypatch):
    for name in ("_session_sems", "_session_active", "_depth", "_recent"):
        monkeypatch.setattr(admission, name, {})
    mcp = FastMCP("toy-search")

    @mcp.tool()
    async def search_papers(query: str) -> dict:
        return {"echo": query}

    admission.install_admission(mcp)
    tool = mcp._tool_manager._tools["search_papers"]
    # A toy search_papers must NOT intercept s-<12hex> or empty strings via resumable fast-paths.
    assert await tool.run({"query": "s-000000000000"}) == {"echo": "s-000000000000"}
    assert await tool.run({"query": "   "}) == {"echo": "   "}
    assert len(admission._recent["search_papers"]) == 2


def _install_fake_mcp_client(monkeypatch, target_module, responses: list[dict]):
    calls: list[dict] = []
    sleeps: list[float] = []
    it = iter(responses)

    @asynccontextmanager
    async def fake_streamablehttp_client(url, **kwargs):
        yield (object(), object(), None)

    class FakeSession:
        def __init__(self, r, w):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments, **kwargs):
            calls.append({"name": name, "arguments": dict(arguments)})
            payload = next(it)
            return SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text=json.dumps(payload))],
            )

    async def fake_sleep(delay: float):
        sleeps.append(delay)

    monkeypatch.setattr(target_module, "streamablehttp_client", fake_streamablehttp_client)
    monkeypatch.setattr(target_module, "ClientSession", FakeSession)
    monkeypatch.setattr(target_module.asyncio, "sleep", fake_sleep)
    return calls, sleeps


@pytest.mark.parametrize("script_name", ["mustfind_recall", "search_relevance_baseline"])
async def test_eval_scripts_poll_running_search_by_id_until_ok(monkeypatch, script_name):
    mod = _load_script(script_name)
    calls, sleeps = _install_fake_mcp_client(
        monkeypatch,
        mod,
        [
            {"status": "running", "running": True, "search_id": "s-112233445566", "retry_after_s": 5},
            {"status": "running", "running": True, "search_id": "s-112233445566", "retry_after_s": 3},
            {"status": "ok", "results": [{"title": "Found Paper", "doi": "10.1/found"}]},
        ],
    )
    if script_name == "mustfind_recall":
        results = await mod._search_once("natural language intent")
    else:
        results, dt = await mod._search("natural language intent")
        assert dt >= 0.0

    assert results == [{"title": "Found Paper", "doi": "10.1/found"}]
    assert [c["arguments"]["query"] for c in calls] == [
        "natural language intent",
        "s-112233445566",
        "s-112233445566",
    ]
    assert sleeps == [5.0, 3.0]


@pytest.mark.parametrize("script_name", ["mustfind_recall", "search_relevance_baseline"])
@pytest.mark.parametrize(
    "terminal_payload",
    [
        {"status": "busy", "busy": True, "retry_after_s": 5, "reason": "saturated"},
        {"status": "not_found", "search_id": "s-112233445566", "reason": "expired"},
        {"status": "error", "results": [], "message": "unparseable intent"},
    ],
)
async def test_eval_scripts_reject_non_ok_terminal_statuses(monkeypatch, script_name, terminal_payload):
    mod = _load_script(script_name)
    _install_fake_mcp_client(
        monkeypatch,
        mod,
        [
            {"status": "running", "running": True, "search_id": "s-112233445566", "retry_after_s": 2},
            terminal_payload,
        ],
    )
    fn = mod._search_once if script_name == "mustfind_recall" else mod._search
    with pytest.raises(RuntimeError, match=terminal_payload["status"]):
        await fn("natural language intent")


@pytest.mark.parametrize("script_name", ["mustfind_recall", "search_relevance_baseline"])
async def test_eval_scripts_bound_polling_deadline(monkeypatch, script_name):
    mod = _load_script(script_name)
    clock = SimpleNamespace(t=100.0)
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock.t))

    @asynccontextmanager
    async def fake_streamablehttp_client(url, **kwargs):
        yield (object(), object(), None)

    class FakeSession:
        def __init__(self, r, w):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments, **kwargs):
            return SimpleNamespace(
                isError=False,
                content=[
                    SimpleNamespace(
                        text=json.dumps(
                            {
                                "status": "running",
                                "running": True,
                                "search_id": "s-abcdef012345",
                                "retry_after_s": 10,
                            }
                        )
                    )
                ],
            )

    async def advancing_sleep(delay: float):
        clock.t += max(delay, 1.0)

    monkeypatch.setattr(mod, "streamablehttp_client", fake_streamablehttp_client)
    monkeypatch.setattr(mod, "ClientSession", FakeSession)
    monkeypatch.setattr(mod.asyncio, "sleep", advancing_sleep)

    fn = mod._search_once if script_name == "mustfind_recall" else mod._search
    with pytest.raises(TimeoutError, match="s-abcdef012345"):
        await fn("natural language intent", timeout_s=25.0)
