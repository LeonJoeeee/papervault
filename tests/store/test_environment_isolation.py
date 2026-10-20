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
        import pytest
        from papervault import config
        from papervault.library.store import Library

        # Collection must already be safe, before any autouse fixture runs.
        assert os.environ["PAPERVAULT_CATALOG_AUTHORITY"] == "json"
        assert "PAPERVAULT_CATALOG_DSN" not in os.environ
        assert "DOTENV_ISOLATION_SENTINEL" not in os.environ

        @pytest.fixture
        def library(tmp_path):
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

        def test_next_fixture_still_uses_json(library):
            test_write_uses_json(library)
    '''))
    env = dict(os.environ)
    for key in ("PAPERVAULT_CATALOG_TEST_DSN", "PAPERVAULT_ENV", "PYTHON_DOTENV_DISABLED",
                "DOTENV_ISOLATION_SENTINEL"):
        env.pop(key, None)
    env.update(PAPERVAULT_CATALOG_AUTHORITY="postgres", PAPERVAULT_CATALOG_DSN=LIVE_DSN,
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    if test_dsn:
        env["PAPERVAULT_CATALOG_TEST_DSN"] = test_dsn
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
        assert "3 passed" in result.stdout


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
