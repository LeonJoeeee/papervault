"""PostgresConfig default alignment (hardening item 3).

The code defaults MUST match deploy/docker-compose.yml (which creates user/db
'papervault'); a divergent default let `papervault doctor` pass against one database
while the runtime targeted another. This pins the aligned contract so it can't drift back.
"""
from __future__ import annotations

import pytest

from papervault.knowledge.config import PostgresConfig

_PG_ENV = ("POSTGRES_USER", "POSTGRES_DB", "POSTGRES_HOST", "POSTGRES_PORT",
           "POSTGRES_PASSWORD")


@pytest.fixture
def _no_pg_env(monkeypatch):
    for var in _PG_ENV:
        monkeypatch.delenv(var, raising=False)


def test_postgres_defaults_match_docker_compose(_no_pg_env):
    pg = PostgresConfig.from_env()
    assert pg.user == "papervault"      # was 'ks' — now matches docker-compose
    assert pg.db == "papervault"        # was 'papervault.knowledge' — now matches
    assert pg.host == "localhost"
    assert pg.port == 5432
    # DSN is well-formed against the aligned defaults.
    assert pg.dsn == "postgresql://papervault:changeme@localhost:5432/papervault"


def test_postgres_env_overrides_win(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "custom")
    monkeypatch.setenv("POSTGRES_DB", "customdb")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    pg = PostgresConfig.from_env()
    assert (pg.user, pg.db, pg.port) == ("custom", "customdb", 6543)
