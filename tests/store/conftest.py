"""Opt-in isolated catalog schemas; a configured but broken DSN is a failure."""

import os
import uuid

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
import pytest


@pytest.fixture
def catalog_test_dsn():
    """Require an explicit non-live database; never discover one through libpq env."""
    dsn = os.environ.get("PAPERVAULT_CATALOG_TEST_DSN")
    if not dsn:
        pytest.skip("PAPERVAULT_CATALOG_TEST_DSN not provided")
    try:
        database = conninfo_to_dict(dsn).get("dbname", "")
    except psycopg.ProgrammingError:
        pytest.fail("PAPERVAULT_CATALOG_TEST_DSN is invalid", pytrace=False)
    # A missing dbname falls back to PGDATABASE/user/service settings. A dbname
    # containing a connection string can itself be expanded by libpq.
    if (not database or database.casefold() == "papervault_catalog"
            or "=" in database or database.startswith(("postgres://", "postgresql://"))):
        pytest.fail("PAPERVAULT_CATALOG_TEST_DSN must name an explicit non-live database",
                    pytrace=False)
    return dsn


@pytest.fixture
def repo(catalog_test_dsn):
    from papervault.store.repository import AuditEvent, PostgresCatalogRepository

    conn = psycopg.connect(catalog_test_dsn, autocommit=True, connect_timeout=3)
    schema = "catalog_test_" + uuid.uuid4().hex
    repository = PostgresCatalogRepository(conn, schema=schema)
    try:
        repository.ensure_schema(audit=AuditEvent("test", "schema"))
        yield repository
    finally:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        conn.close()
