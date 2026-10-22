"""Operator checks must honor the selected transport and catalog authority."""

from contextlib import asynccontextmanager, contextmanager
import json
from types import SimpleNamespace
import subprocess
import sys

import pytest

from papervault import cli


@pytest.mark.asyncio
async def test_smoke_child_cannot_inherit_live_catalog_authority(monkeypatch, tmp_path):
    import mcp
    import mcp.client.stdio

    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")
    monkeypatch.setenv("PAPERVAULT_CATALOG_DSN", "postgresql://live.invalid/catalog")
    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "postgres")
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    operator_env = tmp_path / "operator.env"
    operator_env.write_text("PAPERVAULT_CATALOG_DSN=postgresql://operator.invalid/catalog\n")
    monkeypatch.setenv("PAPERVAULT_ENV", str(operator_env))
    captured = []
    loaded = []

    @asynccontextmanager
    async def child(params):
        captured.append(params.env)
        result = subprocess.run([sys.executable, "-c",
            "import papervault.config, os, json; "
            "print(json.dumps(os.environ.get('PAPERVAULT_CATALOG_DSN')))"],
            env=params.env, capture_output=True, text=True, check=True)
        loaded.append(json.loads(result.stdout))
        yield None, None

    class Session:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(name=name) for name in
                                          ("search_papers", "get_paper", "query")])

        async def call_tool(self, *args):
            return SimpleNamespace(content=[SimpleNamespace(
                type="text", text='{"results": [{"status": "not_found"}]}')])

    monkeypatch.setattr(mcp.client.stdio, "stdio_client", child)
    monkeypatch.setattr(mcp, "ClientSession", Session)
    steps = await cli._smoke_server_phase(False)
    assert all(ok for ok, _ in steps), steps
    assert captured[0]["PAPERVAULT_CATALOG_AUTHORITY"] == "json"
    assert "PAPERVAULT_CATALOG_DSN" not in captured[0]
    assert captured[0]["PAPERVAULT_KNOWLEDGE_JOBS"] == "legacy"
    assert captured[0]["KS_AUTO_INGEST_ENABLED"] == "false"
    assert loaded == [None], "child config must not restore the operator's catalog DSN"


@pytest.mark.parametrize("gateway_key, expected_failures", [("gateway-only", 0), ("", 1)])
def test_doctor_validates_selected_gateway_credentials(monkeypatch, tmp_path, gateway_key,
                                                       expected_failures):
    monkeypatch.setattr(cli.config, "USE_GATEWAY", True)
    monkeypatch.setattr(cli.config, "GATEWAY_KEY", gateway_key)
    monkeypatch.setattr(cli.config, "LLM_API_KEY", "" if gateway_key else "direct-only")
    monkeypatch.setattr(cli.config, "LLM_KEYS_FILE", tmp_path / "absent.json")
    report = cli._Report()
    cli._check_config(report)
    assert report.failures == expected_failures


@pytest.fixture
def databases(monkeypatch):
    import psycopg

    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")
    monkeypatch.setenv("PAPERVAULT_CATALOG_DSN", "postgresql://catalog.invalid/catalog")
    monkeypatch.setenv("PAPERVAULT_CATALOG_SCHEMA", "catalog_probe")
    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "legacy")
    monkeypatch.setenv("POSTGRES_DATABASE", "papervault")
    monkeypatch.setenv("POSTGRES_DB", "papervault")
    calls = []
    catalog = SimpleNamespace(versions=[1, 2], missing=False, unavailable=False)

    class Connection:
        def execute(self, query, params=None):
            query = query.as_string() if hasattr(query, "as_string") else query
            assert query.lstrip().upper().startswith("SELECT"), query
            calls.append((query, params))
            if "schema_version" in query:
                return SimpleNamespace(fetchall=lambda: [{"version": v} for v in catalog.versions])
            if catalog.missing:
                raise psycopg.errors.UndefinedTable("missing catalog relation")
            return SimpleNamespace(fetchall=lambda: [], fetchone=lambda: (True,))

    @contextmanager
    def connect(dsn, **kwargs):
        if dsn == "postgresql://catalog.invalid/catalog":
            calls.append((dsn, kwargs))
            if catalog.unavailable:
                raise psycopg.OperationalError("unreachable")
        connection = Connection()
        connection.autocommit = kwargs.get("autocommit", False)
        yield connection

    class Driver:
        def verify_connectivity(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(psycopg, "connect", connect)
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(
        GraphDatabase=SimpleNamespace(driver=lambda *args, **kwargs: Driver())))
    return catalog, calls


def test_doctor_checks_selected_catalog_dsn_and_schema_read_only(databases):
    _, calls = databases
    report = cli._Report()
    cli._check_databases(report)
    assert report.failures == 0
    catalog_connections = [kwargs for dsn, kwargs in calls
                           if dsn == "postgresql://catalog.invalid/catalog"]
    assert len(catalog_connections) == 1
    assert "default_transaction_read_only=on" in catalog_connections[0]["options"]
    assert any('"catalog_probe"."schema_version"' in query for query, _ in calls)
    assert any('"catalog_probe"."records"' in query for query, _ in calls)


@pytest.mark.parametrize("defect", ["unavailable", "missing", "old_schema", "missing_dsn"])
def test_doctor_fails_for_unready_catalog(databases, monkeypatch, defect):
    catalog, _ = databases
    if defect == "old_schema":
        catalog.versions = [1]
    elif defect == "missing_dsn":
        monkeypatch.delenv("PAPERVAULT_CATALOG_DSN")
    else:
        setattr(catalog, defect, True)
    report = cli._Report()
    cli._check_databases(report)
    assert report.failures == 1
