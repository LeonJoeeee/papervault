"""Opt-in isolated catalog schemas; a configured but broken DSN is a failure."""

import os
import uuid

import psycopg
from psycopg import sql
import pytest


@pytest.fixture
def repo():
    from papervault.store.repository import AuditEvent, PostgresCatalogRepository

    dsn = os.environ.get("PAPERVAULT_CATALOG_TEST_DSN")
    if not dsn:
        pytest.skip("PAPERVAULT_CATALOG_TEST_DSN not provided")
    conn = psycopg.connect(dsn, autocommit=True, connect_timeout=3)
    schema = "catalog_test_" + uuid.uuid4().hex
    repository = PostgresCatalogRepository(conn, schema=schema)
    try:
        repository.ensure_schema(audit=AuditEvent("test", "schema"))
        yield repository
    finally:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        conn.close()
