"""Exercise pytest isolation in fresh processes before application imports."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import psycopg
import pytest


TEST_DSN = "postgresql://tester@127.0.0.1:1/papervault_catalog_test"
LIVE_DSN = "postgresql://tester@127.0.0.1:1/papervault_catalog"
HOSTILE_ENV = {
    "PAPERVAULT_CATALOG_AUTHORITY": "postgres",
    "PAPERVAULT_CATALOG_DSN": LIVE_DSN,
    "PAPERVAULT_TERMINAL_V2": "1",
    "PAPERVAULT_IDENTITY_V2": "1",
    "PAPERVAULT_REOPEN_V2": "1",
    "PAPERVAULT_POLITE_V2": "1",
    "PAPERVAULT_MERGE_V2": "1",
    "PAPERVAULT_STOPRULES_V2": "1",
    "PAPERVAULT_LLM_GATEWAY": "1",
    "PAPERVAULT_LLM_GATEWAY_URL": "http://production.invalid/v1",
    "PAPERVAULT_LLM_GATEWAY_KEY": "production-key",
    "PAPERVAULT_LLM_API_KEY": "production-key",
    "PAPERVAULT_LLM_BASE_URL": "http://production.invalid/v1",
    "PAPERVAULT_DATA": "/production/data",
    "PAPERVAULT_VAULT": "/production/vault",
    "PAPERVAULT_FUTURE_FLAG": "1",
    "KS_AUTO_INGEST_ENABLED": "true",
    "KS_ALLOW_PROD_WORKSPACE": "1",
    "KS_CHUNK_TOKEN_SIZE": "99",
    "KS_FUTURE_SETTING": "1",
    "NEO4J_WORKSPACE": "l0",
    "NEO4J_URI": "bolt://production.invalid:7687",
    "POSTGRES_WORKSPACE": "l0",
    "POSTGRES_DB": "production",
    "POSTGRES_DATABASE": "production",
    "POSTGRES_HOST": "production.invalid",
    "PGSERVICE": "production",
    "PGDATABASE": "papervault_catalog",
    "MINERU_URL": "http://production.invalid:30000",
    "OPENALEX_API_KEY": "production-key",
    "OPENALEX_MAILTO": "production@example.invalid",
    "OPENAI_API_KEY": "production-key",
    "ANTHROPIC_API_KEY": "production-key",
    "MIMO_API_KEY_1": "tp-production-key",
    "ADS_API_TOKEN": "production-key",
    "CORE_API_KEY": "production-key",
    "FIRECRAWL_API_KEY": "production-key",
    "FIRECRAWL_API_URL": "http://production.invalid",
    "UNPAYWALL_EMAIL": "production@example.invalid",
    "WILEY_TDM_TOKEN": "production-key",
    "ELSEVIER_TDM_API_KEY": "production-key",
    "ANNAS_ARCHIVE_API_KEY": "production-key",
    "ZOTERO_API_KEY": "production-key",
    "PAPER_LIBRARY_PATH": "/production/vault",
    "PAPER_LIBRARY_EXTRACT_CONCURRENCY": "99",
    "PAPER_PIPELINE_USE_SCIHUB": "1",
    "PAPER_SEARCH_DEFAULT_LIMIT": "99",
    "LLM_KEYS_FILE": "/production/llm_keys.json",
    "LIGHTRAG_WORKING_DIR": "/production/knowledge",
    "BGE_M3_DEVICE": "cuda:99",
    "ENTITY_EXTRACTION_USE_JSON": "1",
    "RERANK_TIMEOUT": "9999",
}


@pytest.mark.parametrize("test_dsn,explicit_env,dotenv_version", [
    (None, False, None), (TEST_DSN, True, None), (None, False, "1.1.1"),
])
def test_hostile_environment_cannot_escape_test_isolation(
        tmp_path, test_dsn, explicit_env, dotenv_version):
    # Copy the real config so its module-level guard runs in a fresh interpreter.
    source = Path(__file__).resolve().parents[1] / "conftest.py"
    shutil.copyfile(source, tmp_path / "conftest.py")
    with (tmp_path / "conftest.py").open("a") as stream:
        stream.write(textwrap.dedent('''
            import psycopg
            from dotenv.main import DotEnv

            def forbidden(*args, **kwargs):
                raise AssertionError("test attempted a PG connection or dotenv read")

            psycopg.connect = forbidden
            DotEnv._get_stream = forbidden
        '''))
    env_file = tmp_path / ("explicit.env" if explicit_env else ".env")
    env_file.write_text("DOTENV_ISOLATION_SENTINEL=loaded\nPAPERVAULT_CATALOG_AUTHORITY=postgres\n")
    (tmp_path / "test_probe.py").write_text(textwrap.dedent('''
        import json
        import os
        from pathlib import Path
        import pytest
        from papervault import config
        from papervault.knowledge.config import CONFIG
        from papervault.library import extract, mineru_client
        from papervault.library.store import Library

        def assert_test_defaults():
            assert os.environ["PAPERVAULT_CATALOG_AUTHORITY"] == "json"
            assert "PAPERVAULT_CATALOG_DSN" not in os.environ
            assert os.environ.get("PAPERVAULT_CATALOG_TEST_DSN") == os.environ.get("EXPECTED_TEST_DSN")
            assert not any((config.terminal_v2_enabled(), config.identity_v2_enabled(),
                            config.reopen_v2_enabled(), config.polite_v2_enabled(),
                            config.merge_v2_enabled(), config.stoprules_v2_enabled()))
            assert not config.USE_GATEWAY
            assert config.LLM_API_KEY == config.LLM_BASE_URL == config.GATEWAY_KEY == ""
            assert config.DATA_DIR.is_dir()
            assert config.DATA_DIR != Path.home() / ".local/share/papervault"
            assert config.VAULT_PATH == config.DATA_DIR / "vault"
            assert config.LLM_KEYS_FILE == config.DATA_DIR / "llm_keys.json"
            assert not config.LLM_KEYS_FILE.exists()
            assert CONFIG.postgres.host == "localhost"
            assert CONFIG.postgres.db == "papervault"
            assert CONFIG.lightrag.chunk_token_size == 2400
            assert CONFIG.lightrag.working_dir == str(config.DATA_DIR / "knowledge_store")
            assert CONFIG.mimo.endpoints == ()
            assert CONFIG.anthropic.api_key == ""
            assert CONFIG.bge_m3.device == "cuda:0"
            assert mineru_client.endpoints_from_env()[0].url == "http://127.0.0.1:30000"
            assert extract._EXTRACT_CONCURRENCY == 8
            for name in ("PAPERVAULT_FUTURE_FLAG", "KS_FUTURE_SETTING", "KS_AUTO_INGEST_ENABLED",
                         "KS_ALLOW_PROD_WORKSPACE", "NEO4J_WORKSPACE", "POSTGRES_WORKSPACE",
                         "PGSERVICE", "PGDATABASE", "OPENALEX_API_KEY", "OPENALEX_MAILTO",
                         "OPENAI_API_KEY", "ADS_API_TOKEN", "CORE_API_KEY", "FIRECRAWL_API_KEY",
                         "FIRECRAWL_API_URL", "WILEY_TDM_TOKEN", "ELSEVIER_TDM_API_KEY",
                         "ANNAS_ARCHIVE_API_KEY", "ZOTERO_API_KEY", "UNPAYWALL_EMAIL",
                         "PAPER_PIPELINE_USE_SCIHUB", "PAPER_SEARCH_DEFAULT_LIMIT",
                         "ENTITY_EXTRACTION_USE_JSON", "RERANK_TIMEOUT"):
                assert name not in os.environ, name

        # Collection must already be safe, before any autouse fixture runs.
        assert_test_defaults()
        assert "DOTENV_ISOLATION_SENTINEL" not in os.environ

        @pytest.fixture
        def library(tmp_path):
            assert_test_defaults()
            lib = Library(tmp_path)
            yield lib
            lib.close()

        def test_write_uses_json(library):
            paper, _ = library.upsert({"title": "Isolated fixture", "doi": "10.1234/isolation"})
            library.save(force=True)
            assert json.loads(library.index_path.read_text())["papers"][paper.key]["title"] == "Isolated fixture"

        def test_unscoped_environment_change():
            os.environ["PAPERVAULT_CATALOG_AUTHORITY"] = "postgres"
            os.environ["PAPERVAULT_CATALOG_DSN"] = "host=127.0.0.1 port=1 dbname=papervault_catalog"
            os.environ["PAPERVAULT_MERGE_V2"] = "1"
            os.environ["KS_AUTO_INGEST_ENABLED"] = "true"
            os.environ["MINERU_URL"] = "http://production.invalid:30000"
            os.environ["CORE_API_KEY"] = "production-key"

        def test_next_fixture_still_uses_json(library):
            test_write_uses_json(library)

        def test_flags_require_explicit_opt_in(monkeypatch):
            monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
            assert config.merge_v2_enabled()

        def test_explicit_opt_in_does_not_leak():
            assert_test_defaults()
    '''))
    env = dict(os.environ)
    for key in ("PAPERVAULT_CATALOG_TEST_DSN", "PAPERVAULT_ENV", "PYTHON_DOTENV_DISABLED",
                "DOTENV_ISOLATION_SENTINEL", "EXPECTED_TEST_DSN"):
        env.pop(key, None)
    env.update(HOSTILE_ENV, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    if test_dsn:
        env["PAPERVAULT_CATALOG_TEST_DSN"] = test_dsn
        env["EXPECTED_TEST_DSN"] = test_dsn
    if explicit_env:
        env["PAPERVAULT_ENV"] = str(env_file)
    runner = textwrap.dedent('''
        import importlib.metadata
        import sys

        if sys.argv[1] != "current":
            installed_version = importlib.metadata.version
            importlib.metadata.version = lambda name: (
                sys.argv[1] if name == "python-dotenv" else installed_version(name)
            )
        import pytest
        raise SystemExit(pytest.main(sys.argv[2:]))
    ''')
    result = subprocess.run(
        [sys.executable, "-c", runner, dotenv_version or "current", "-q", "-p", "no:cacheprovider",
         "--basetemp", str(tmp_path / "run")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    if dotenv_version:
        assert result.returncode == 4, result.stdout + result.stderr
        assert "python-dotenv>=1.2" in result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "5 passed" in result.stdout


@pytest.mark.parametrize("dsn", [
    LIVE_DSN,
    "host=127.0.0.1 port=1 dbname=papervault_catalog",
    "postgresql://tester@127.0.0.1:1/papervault%5fcatalog",
    TEST_DSN + "?dbname=papervault_catalog",
    "host=127.0.0.1 port=1 user=papervault_catalog",
    "postgresql://tester@127.0.0.1:1/",
    "service=production",
    "dbname='host=127.0.0.1 dbname=papervault_catalog'",
    "dbname='postgresql://tester@127.0.0.1:1/papervault_catalog'",
    "dbname='unterminated",
])
def test_pg_fixture_refuses_unsafe_test_dsn_before_connecting(monkeypatch, request, dsn):
    monkeypatch.setenv("PAPERVAULT_CATALOG_TEST_DSN", dsn)
    monkeypatch.setenv("PGDATABASE", "papervault_catalog")

    def forbidden(*args, **kwargs):
        raise AssertionError("unsafe test DSN reached psycopg.connect")

    monkeypatch.setattr(psycopg, "connect", forbidden)
    with pytest.raises(pytest.fail.Exception, match="PAPERVAULT_CATALOG_TEST_DSN"):
        request.getfixturevalue("repo")


def test_pg_fixture_never_falls_back_to_inherited_live_dsn(monkeypatch, request):
    monkeypatch.delenv("PAPERVAULT_CATALOG_TEST_DSN", raising=False)
    monkeypatch.setenv("PAPERVAULT_CATALOG_DSN", LIVE_DSN)

    def forbidden(*args, **kwargs):
        raise AssertionError("live DSN reached psycopg.connect")

    monkeypatch.setattr(psycopg, "connect", forbidden)
    with pytest.raises(pytest.skip.Exception, match="PAPERVAULT_CATALOG_TEST_DSN"):
        request.getfixturevalue("repo")


@pytest.mark.parametrize("dsn", [TEST_DSN, "host=127.0.0.1 port=1 dbname=papervault_catalog_test"])
def test_pg_fixture_uses_only_explicit_test_dsn(monkeypatch, request, dsn):
    monkeypatch.setenv("PAPERVAULT_CATALOG_DSN", LIVE_DSN)
    monkeypatch.setenv("PAPERVAULT_CATALOG_TEST_DSN", dsn)

    class ConnectionSelected(Exception):
        pass

    def connect(selected, **kwargs):
        assert selected == dsn
        raise ConnectionSelected

    monkeypatch.setattr(psycopg, "connect", connect)
    with pytest.raises(ConnectionSelected):
        request.getfixturevalue("repo")


@pytest.mark.parametrize("disposable", [None, "1"])
def test_backend_test_connections_are_scrubbed_and_only_exposed_to_integration_fixture(tmp_path, disposable):
    source = Path(__file__).resolve().parents[1] / "conftest.py"
    shutil.copyfile(source, tmp_path / "conftest.py")
    targets = {"PAPERVAULT_BACKEND_TEST_DSN": "host=staging.invalid dbname=p4_rehearsal user=test password=fixture",
        "PAPERVAULT_BACKEND_TEST_NEO4J_URI": "bolt://staging.invalid:7687",
        "PAPERVAULT_BACKEND_TEST_NEO4J_USER": "fixture",
        "PAPERVAULT_BACKEND_TEST_NEO4J_PASSWORD": "fixture",
        "PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE": "p4_rehearsal",
        "PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE": disposable,
        "NEO4J_URI": "bolt://production.invalid:7687"}
    (tmp_path / "test_probe.py").write_text("TARGETS = " + repr(targets) + "\n" + textwrap.dedent('''
        import os
        import pytest

        assert all(name not in os.environ for name in TARGETS)

        @pytest.mark.integration
        def test_explicit_targets(backend_test_targets):
            assert backend_test_targets == TARGETS
            assert all(name not in os.environ for name in TARGETS)
            assert os.environ["PYTHON_DOTENV_DISABLED"] == "1"
            assert "POSTGRES_HOST" not in os.environ
            assert "NEO4J_URI" not in os.environ

        def test_unit_cannot_request_targets(request):
            with pytest.raises(pytest.fail.Exception, match="integration"):
                request.getfixturevalue("backend_test_targets")
    '''))
    env = dict(os.environ)
    env.update(HOSTILE_ENV, **{key: value for key, value in targets.items() if value is not None},
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
