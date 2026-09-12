"""Host allowlist configuration and real HTTP initialization, without server boot."""
import logging
import runpy

import pytest
from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

from papervault import config
from papervault.mcp import server


@pytest.mark.parametrize(("value", "expected"), [
    (None, []),
    ("", []),
    (" , \t, ", []),
    (" node.example.ts.net:8080, , node.example.ts.net, node.example.ts.net:* ",
     ["node.example.ts.net:8080", "node.example.ts.net", "node.example.ts.net:*"]),
])
def test_config_parses_allowed_hosts(monkeypatch, value, expected):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    if value is None:
        monkeypatch.delenv("PAPERVAULT_MCP_ALLOWED_HOSTS", raising=False)
    else:
        monkeypatch.setenv("PAPERVAULT_MCP_ALLOWED_HOSTS", value)
    parsed = runpy.run_path(config.__file__)["MCP_ALLOWED_HOSTS"]
    assert parsed == expected


def test_empty_hosts_keep_sdk_defaults():
    assert server.build_transport_security([]) is None


def test_builder_preserves_entries_and_derives_origins():
    hosts = ["node.example.ts.net:8080", "node.example.ts.net", "node.example.ts.net:*"]
    settings = server.build_transport_security(hosts)
    assert settings.enable_dns_rebinding_protection is True
    assert settings.allowed_hosts == [
        "127.0.0.1:*", "localhost:*", "[::1]:*",
        "node.example.ts.net:8080", "node.example.ts.net", "node.example.ts.net:*",
    ]
    assert settings.allowed_origins == [
        "http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*",
        "https://node.example.ts.net:8080", "http://node.example.ts.net:8080",
        "https://node.example.ts.net", "http://node.example.ts.net",
        "https://node.example.ts.net:*", "http://node.example.ts.net:*",
    ]
    assert hosts == ["node.example.ts.net:8080", "node.example.ts.net", "node.example.ts.net:*"]


def _initialize(hosts, host, origin=None):
    settings = server.build_transport_security(hosts)
    mcp = FastMCP("test", **({"transport_security": settings} if settings else {}))
    headers = {"Host": host, "Accept": "application/json, text/event-stream"}
    if origin is not None:
        headers["Origin"] = origin
    with TestClient(mcp.streamable_http_app()) as client:
        return client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                       "clientInfo": {"name": "test", "version": "0"}},
        })


def test_tailnet_host_rejected_when_var_unset():
    response = _initialize([], "node.example.ts.net:8080")
    assert response.status_code == 421
    assert response.text == "Invalid Host header"


def test_tailnet_host_accepted_when_listed():
    assert _initialize(["node.example.ts.net:8080"], "node.example.ts.net:8080").status_code == 200


@pytest.mark.parametrize("hosts", [[], ["node.example.ts.net:8080"]])
@pytest.mark.parametrize("host", ["127.0.0.1:8080", "localhost:8080", "[::1]:8080"])
def test_loopback_host_accepted_in_both_cases(hosts, host):
    assert _initialize(hosts, host).status_code == 200


def test_bare_host_entry_matches_portless_host():
    assert _initialize(["node.example.ts.net"], "node.example.ts.net").status_code == 200


@pytest.mark.parametrize(("entry", "host", "status"), [
    ("node.example.ts.net", "node.example.ts.net:8080", 421),
    ("node.example.ts.net:8080", "node.example.ts.net", 421),
    ("node.example.ts.net:8080", "node.example.ts.net:9090", 421),
    ("node.example.ts.net:*", "node.example.ts.net", 421),
    ("node.example.ts.net:*", "node.example.ts.net:9090", 200),
    ("node.example.ts.net:*", "other.example.ts.net:9090", 421),
])
def test_host_port_matching(entry, host, status):
    assert _initialize([entry], host).status_code == status


@pytest.mark.parametrize(("origin", "status"), [
    ("https://node.example.ts.net:8080", 200),
    ("http://node.example.ts.net:8080", 200),
    ("http://127.0.0.1:8080", 200),
    ("https://other.example.ts.net:8080", 403),
])
def test_origin_check_remains_enabled(origin, status):
    assert _initialize(["node.example.ts.net:8080"], "node.example.ts.net:8080",
                       origin).status_code == status


@pytest.mark.parametrize("hosts", [[], ["node.example.ts.net:8080"]])
def test_build_server_uses_config_and_logs_extra_hosts(monkeypatch, caplog, hosts):
    monkeypatch.setattr(config, "MCP_ALLOWED_HOSTS", hosts)
    monkeypatch.setattr(server, "_build_library", lambda **kwargs: None)
    with caplog.at_level(logging.INFO, logger=server.__name__):
        mcp = server.build_server()
    if hosts:
        assert "node.example.ts.net:8080" in mcp.settings.transport_security.allowed_hosts
        records = [r.getMessage() for r in caplog.records
                   if "PAPERVAULT_MCP_ALLOWED_HOSTS" in r.getMessage()]
        assert len(records) == 1 and "node.example.ts.net:8080" in records[0]
    else:
        assert mcp.settings.transport_security == FastMCP("default").settings.transport_security
        assert "PAPERVAULT_MCP_ALLOWED_HOSTS" not in caplog.text
