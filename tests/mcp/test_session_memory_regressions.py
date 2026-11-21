"""Regression tests for Batch 2 MCP session idle timeout and bounded tombstone/principal caches."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anyio
import pytest

from papervault.mcp import server


def test_build_server_sets_session_idle_timeout_from_env(monkeypatch):
    monkeypatch.setattr(server, "_build_library", lambda **kwargs: None)
    monkeypatch.delenv("PAPERVAULT_MCP_SESSION_IDLE_TIMEOUT_S", raising=False)
    mcp_default = server.build_server()
    assert mcp_default.session_manager.session_idle_timeout == 1800.0

    monkeypatch.setenv("PAPERVAULT_MCP_SESSION_IDLE_TIMEOUT_S", "42.5")
    mcp_custom = server.build_server()
    assert mcp_custom.session_manager.session_idle_timeout == 42.5


def test_session_manager_bounds_tombstones_and_principals():
    mgr = server.ResilientHTTPSessionManager(
        app=SimpleNamespace(),
        session_idle_timeout=10.0,
        max_tombstones=3,
        max_principals=4,
        tombstone_ttl=30.0,
        principal_ttl=60.0,
    )

    # Seed 5 terminated tombstones -> pruned down to max_tombstones=3 (oldest evicted).
    for i in range(5):
        sid = f"del-{i}"
        mgr._server_instances[sid] = SimpleNamespace(is_terminated=True)
        mgr._session_principals[sid] = ("client", None, None)
        mgr._principal_seen[sid] = float(i)
        mgr._terminated_at[sid] = float(i)

    mgr._prune_session_state(now=10.0)
    assert list(mgr._terminated_at) == ["del-2", "del-3", "del-4"]
    assert "del-0" not in mgr._server_instances
    assert "del-1" not in mgr._server_instances
    assert "del-0" not in mgr._session_principals

    # Advance past tombstone_ttl -> remaining tombstones expire cleanly.
    mgr._prune_session_state(now=100.0)
    assert mgr._terminated_at == {}
    assert mgr._server_instances == {}

    # Active sessions are never evicted from _session_principals, while excess inactive principals are.
    mgr._server_instances["active-1"] = SimpleNamespace(is_terminated=False)
    mgr._session_principals["active-1"] = ("owner-a", None, None)
    mgr._principal_seen["active-1"] = 100.0
    for i in range(6):
        sid = f"inactive-{i}"
        mgr._session_principals[sid] = (f"owner-{i}", None, None)
        mgr._principal_seen[sid] = 100.0 + i

    mgr._prune_session_state(now=110.0)
    assert "active-1" in mgr._session_principals
    assert len(mgr._session_principals) == 4  # active-1 + 3 newest inactive principals
    assert "inactive-0" not in mgr._session_principals
    assert "inactive-5" in mgr._session_principals

    # Advance past principal_ttl -> all inactive principals expire, active-1 stays.
    mgr._prune_session_state(now=250.0)
    assert list(mgr._session_principals) == ["active-1"]


@pytest.mark.asyncio
async def test_session_manager_real_http_delete_tombstone_evicts_after_cap(monkeypatch):
    """End-to-end check: deleted sessions stay 404 while within tombstone cap, and oldest tombstones evict."""
    from tests.mcp.test_session_resilience import _post, _result, _serve

    monkeypatch.setattr(server, "_build_library", lambda **kwargs: None)

    async def start_background():
        pass

    async def query(intent: str) -> dict:
        return {"intent": intent}

    monkeypatch.setattr(server, "_start_knowledge_bg", start_background)
    monkeypatch.setattr(server, "_knowledge_query", query)
    monkeypatch.setenv("PAPERVAULT_MCP_MAX_TOMBSTONES", "2")

    mcp = server.build_server()
    async with _serve(mcp) as client:
        for i in range(4):
            sid = f"tomb-{i}"
            _result(await _post(client, sid))
            resp = await client.delete("/mcp", headers={"Mcp-Session-Id": sid})
            assert resp.status_code == 200
            await asyncio.sleep(0.02)

        # Wait briefly for session runner cleanup tasks to register tombstones.
        with anyio.fail_after(3):
            while len(mcp.session_manager._terminated_at) > 2:
                await anyio.sleep(0.01)

        assert len(mcp.session_manager._terminated_at) <= 2
        assert len(mcp.session_manager._server_instances) <= 2
        # Most recently deleted session still returns 404.
        assert (await _post(client, "tomb-3", request_id=2)).status_code == 404
