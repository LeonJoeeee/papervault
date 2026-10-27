"""Resilient session behavior over a disposable real HTTP/SSE server."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import socket

import anyio
import httpx
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.fastmcp import Context
import pytest
import uvicorn

from papervault.mcp import server


_HEADERS = {"Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-03-26"}


@pytest.fixture
def make_server(monkeypatch):
    # Keep transport, tools, admission, access logging and SDK lifecycle real;
    # only replace library/storage startup and the knowledge backend.
    monkeypatch.setattr(server, "_build_library", lambda **kwargs: None)

    async def start_background():
        pass

    sessions = []

    async def query(intent: str, ctx: Context) -> dict:
        sessions.append(ctx.session)  # Keep identities distinct even after teardown.
        if intent == "notify":
            await ctx.session.send_tool_list_changed()
        return {"intent": intent, "session": id(ctx.session)}

    monkeypatch.setattr(server, "_start_knowledge_bg", start_background)
    monkeypatch.setattr(server, "_knowledge_query", query)
    return server.build_server


@asynccontextmanager
async def _serve(mcp):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    mcp_app = mcp.streamable_http_app()

    async def app(scope, receive, send):
        # Authentication is deployment-owned; exercise the manager's actual
        # principal comparison with real SDK user/token objects at its ASGI seam.
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            principal = headers.get(b"x-test-principal")
            if principal:
                token = AccessToken(
                    token="test-token", client_id=principal.decode(), scopes=[],
                )
                # SDK 1.27 tokens predate these optional identity fields.
                # model_copy also lets those tokens exercise supplied fields.
                token = token.model_copy(update={
                    "subject": headers.get(b"x-test-subject", b"subject").decode(),
                    "claims": {"iss": headers.get(b"x-test-issuer", b"issuer").decode()},
                })
                scope["user"] = AuthenticatedUser(token)
        await mcp_app(scope, receive, send)
    runner = uvicorn.Server(uvicorn.Config(
        app, log_level="critical", timeout_graceful_shutdown=0.2,
    ))
    task = asyncio.create_task(runner.serve(sockets=[sock]))
    try:
        with anyio.fail_after(5):
            while not runner.started:
                if task.done():
                    task.result()
                await anyio.sleep(0.001)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", headers=_HEADERS, timeout=3,
        ) as client:
            yield client
    finally:
        runner.should_exit = True
        with anyio.fail_after(5):
            await task
        sock.close()


def _message(method="tools/call", request_id=1, intent="test"):
    params = {"name": "query", "arguments": {"intent": intent}}
    if method == "initialize":
        params = {"protocolVersion": "2025-03-26", "capabilities": {},
                  "clientInfo": {"name": "resilience-test", "version": "1"}}
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


async def _post(client, session_id=None, **kwargs):
    headers = {"Mcp-Session-Id": session_id} if session_id is not None else {}
    return await client.post("/mcp", headers=headers, json=_message(**kwargs))


def _result(response):
    assert response.status_code == 200, response.text
    if response.headers["content-type"].startswith("text/event-stream"):
        message = next(json.loads(line[6:]) for line in response.text.splitlines()
                       if line.startswith("data: "))
    else:
        message = response.json()
    assert "error" not in message, message
    result = message["result"]
    assert not result.get("isError"), result
    return result


def _tool_result(response):
    return json.loads(_result(response)["content"][0]["text"])


async def test_unknown_id_accepts_tool_calls_and_keeps_session_identity(make_server):
    async with _serve(make_server()) as client:
        first = await _post(client, "remembered-session")
        identity = _tool_result(first)["session"]
        assert first.headers["mcp-session-id"] == "remembered-session"
        again = _tool_result(await _post(client, "remembered-session", request_id=2))
        other = _tool_result(await _post(client, "other-session", request_id=3))
        assert again["session"] == identity
        assert other["session"] != identity


async def test_restart_recovers_previous_session_without_initialize(make_server):
    async with _serve(make_server()) as client:
        initialized = await _post(client, method="initialize")
        _result(initialized)
        session_id = initialized.headers["mcp-session-id"]
    async with _serve(make_server()) as client:
        assert _tool_result(await _post(client, session_id))["intent"] == "test"


async def test_stale_initialize_reuses_requested_id(make_server):
    async with _serve(make_server()) as client:
        response = await _post(client, "stale-initialize", method="initialize")
        assert _result(response)["serverInfo"]["name"] == "papervault"
        assert response.headers["mcp-session-id"] == "stale-initialize"
        _result(await _post(client, "stale-initialize", request_id=2))


async def test_unknown_get_opens_sse_and_does_not_hold_creation_lock(make_server):
    async with _serve(make_server()) as client:
        async with client.stream("GET", "/mcp", headers={"Mcp-Session-Id": "stale-sse"}) as stream:
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith("text/event-stream")
            assert stream.headers["mcp-session-id"] == "stale-sse"
            # A new session must work while the first request is a live GET.
            _result(await _post(client, "independent-session"))
            _result(await _post(client, "stale-sse", request_id=2, intent="notify"))
            with anyio.fail_after(3):
                async for line in stream.aiter_lines():
                    if line.startswith("data: "):
                        assert json.loads(line[6:])["method"] == "notifications/tools/list_changed"
                        break
                else:
                    pytest.fail("SSE ended before the tool notification")


async def test_concurrent_recovery_creates_one_server_session(make_server):
    async with _serve(make_server()) as client:
        responses = await asyncio.gather(*(
            _post(client, "concurrent-session", request_id=i) for i in range(1, 9)
        ))
        identities = {_tool_result(response)["session"] for response in responses}
        assert len(identities) == 1


async def test_deleted_session_never_rehydrates(make_server):
    async with _serve(make_server()) as client:
        _result(await _post(client, "deleted-session"))
        response = await client.delete("/mcp", headers={"Mcp-Session-Id": "deleted-session"})
        assert response.status_code == 200
        await anyio.sleep(0)  # Let session runner cleanup execute.
        assert (await _post(client, "deleted-session", request_id=2)).status_code == 404
        assert (await _post(client, "deleted-session", method="initialize", request_id=3)).status_code == 404
        assert (await client.get("/mcp", headers={"Mcp-Session-Id": "deleted-session"})).status_code == 404


@pytest.mark.parametrize(("headers", "status"), [
    ({"Host": "untrusted.example:8080"}, 421),
    ({"Origin": "https://untrusted.example:8080"}, 403),
])
async def test_unknown_session_rejects_security_errors_before_creation(make_server, headers, status):
    mcp = make_server()
    async with _serve(mcp) as client:
        response = await client.post("/mcp", headers={"Mcp-Session-Id": "untrusted", **headers},
                                     json=_message())
        assert response.status_code == status
        assert "untrusted" not in mcp.session_manager._server_instances


async def _disconnect_session(manager, session_id):
    transport = manager._server_instances[session_id]
    await transport._read_stream_writer.aclose()
    with anyio.fail_after(3):
        while manager._server_instances.get(session_id) is transport:
            await anyio.sleep(0.001)


async def test_disconnected_runner_recovers_with_owner_checks(make_server):
    mcp = make_server()
    async with _serve(mcp) as client:
        client.headers["X-Test-Principal"] = "original-owner"
        initialized = await _post(client, method="initialize")
        _result(initialized)
        session_id = initialized.headers["mcp-session-id"]
        await _disconnect_session(mcp.session_manager, session_id)
        client.headers["X-Test-Principal"] = "other-owner"
        assert (await _post(client, session_id)).status_code == 404
        client.headers["X-Test-Principal"] = "original-owner"
        _result(await _post(client, session_id, request_id=2))
        client.headers.pop("X-Test-Principal")
        assert (await _post(client, session_id, request_id=3)).status_code == 404


async def test_anonymous_disconnected_session_cannot_be_claimed_by_principal(make_server):
    mcp = make_server()
    async with _serve(mcp) as client:
        initialized = await _post(client, method="initialize")
        _result(initialized)
        session_id = initialized.headers["mcp-session-id"]
        await _disconnect_session(mcp.session_manager, session_id)
        client.headers["X-Test-Principal"] = "new-owner"
        assert (await _post(client, session_id)).status_code == 404
        client.headers.pop("X-Test-Principal")
        _result(await _post(client, session_id, request_id=2))


@pytest.mark.parametrize("session_id", ["", "invalid id"])
async def test_invalid_recovery_id_is_bad_request(make_server, session_id):
    mcp = make_server()
    async with _serve(mcp) as client:
        assert (await _post(client, session_id)).status_code == 400
        assert not mcp.session_manager._server_instances


async def test_crashed_runner_cleans_up_and_recovers(make_server):
    mcp = make_server()
    crash = anyio.Event()
    launches = 0

    @asynccontextmanager
    async def crashable_lifespan(_app):
        nonlocal launches
        launches += 1
        if launches == 1:
            async def fail():
                await crash.wait()
                raise RuntimeError("simulated session runner failure")

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(fail)
                yield {}
        else:
            yield {}

    mcp._mcp_server.lifespan = crashable_lifespan
    async with _serve(mcp) as client:
        initialized = await _post(client, method="initialize")
        _result(initialized)
        session_id = initialized.headers["mcp-session-id"]
        crash.set()
        with anyio.fail_after(3):
            while session_id in mcp.session_manager._server_instances:
                await anyio.sleep(0.001)
        _tool_result(await _post(client, session_id, request_id=2))
        assert launches == 2


async def test_unknown_delete_does_not_allocate_session(make_server):
    mcp = make_server()
    async with _serve(mcp) as client:
        response = await client.delete("/mcp", headers={"Mcp-Session-Id": "missing"})
        assert response.status_code == 404
        assert not mcp.session_manager._server_instances


async def test_idle_eviction_is_recoverable(make_server):
    mcp = make_server()
    mcp.session_manager.session_idle_timeout = 0.05
    async with _serve(mcp) as client:
        first = _tool_result(await _post(client, "idle-session"))
        with anyio.fail_after(3):
            while "idle-session" in mcp.session_manager._server_instances:
                await anyio.sleep(0.001)
        second = _tool_result(await _post(client, "idle-session", request_id=2))
        assert second["session"] != first["session"]


async def test_delete_stays_terminated_when_idle_deadline_interrupts_cleanup(make_server):
    mcp = make_server()
    mcp.session_manager.session_idle_timeout = 0.1
    cleanup_finished = anyio.Event()

    @asynccontextmanager
    async def slow_cleanup(_app):
        try:
            yield {}
        finally:
            try:
                # Simulate teardown still running when the idle deadline fires.
                await anyio.sleep_forever()
            finally:
                cleanup_finished.set()

    mcp._mcp_server.lifespan = slow_cleanup
    async with _serve(mcp) as client:
        _tool_result(await _post(client, "delete-idle-race"))
        deleted = await client.delete("/mcp", headers={"Mcp-Session-Id": "delete-idle-race"})
        assert deleted.status_code == 200
        with anyio.fail_after(3):
            await cleanup_finished.wait()
        assert (await _post(client, "delete-idle-race", request_id=2)).status_code == 404


@pytest.mark.parametrize("changed_header", ["X-Test-Subject", "X-Test-Issuer"])
async def test_session_owner_includes_subject_and_issuer(make_server, changed_header):
    mcp = make_server()
    async with _serve(mcp) as client:
        client.headers["X-Test-Principal"] = "same-client"
        _tool_result(await _post(client, "principal-session"))
        client.headers[changed_header] = "other-identity"
        assert (await _post(client, "principal-session", request_id=2)).status_code == 404


async def test_manager_shutdown_clears_recovery_owners(make_server):
    mcp = make_server()
    async with _serve(mcp) as client:
        _tool_result(await _post(client, "cleanup-session"))
        assert mcp.session_manager._session_principals
    assert not mcp.session_manager._session_principals
    assert not mcp.session_manager._server_instances
