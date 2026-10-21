"""Lossless derivative-store export, explicit inventories and empty target guards."""
import importlib.util
from uuid import uuid4

from psycopg import sql
import pytest


ROLES = ("full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks",
         "relation_chunks", "llm_response_cache", "doc_status", "chunks_vdb", "entities_vdb", "relationships_vdb")


def backends():
    assert importlib.util.find_spec("papervault.store.recovery_backends") is not None
    from papervault.store import recovery_backends
    return recovery_backends


@pytest.fixture
def backend_tables(repo):
    schema = "recovery_backend_source_" + uuid4().hex
    connection = repo.connection
    connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    tables = {role: "lightrag_" + role for role in ROLES}
    try:
        for name in tables.values():
            connection.execute(sql.SQL("CREATE TABLE {} (workspace text NOT NULL,id text NOT NULL, "
                "content jsonb NOT NULL,PRIMARY KEY(workspace,id))").format(sql.Identifier(schema, name)))
            connection.execute(sql.SQL("INSERT INTO {} VALUES('retained','doc','{{\"content\":\"retained evidence\"}}')").format(
                sql.Identifier(schema, name)))
        yield connection, schema, tables
    finally:
        connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_postgres_store_preserves_all_roles_and_rows_in_fresh_schema(backend_tables):
    module = backends()
    connection, schema, tables = backend_tables
    store = module.PostgresStore(connection, schema=schema, roles=tables)
    with store.barrier():
        exported = store.export()
    assert set(exported["roles"]) == set(ROLES)
    target = "recovery_backend_target_" + uuid4().hex
    try:
        module.restore_postgres_store(connection, target, exported)
        for table in tables.values():
            actual = connection.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(target, table))).fetchone()
            assert actual == {"workspace": "retained", "id": "doc", "content": {"content": "retained evidence"}}
        with pytest.raises(ValueError, match="empty|absent"):
            module.restore_postgres_store(connection, target, exported)
    finally:
        connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(target)))


@pytest.mark.parametrize("role", ROLES)
def test_postgres_store_refuses_omitted_role(backend_tables, role):
    module = backends()
    connection, schema, tables = backend_tables
    del tables[role]
    with pytest.raises(ValueError, match="role|store"):
        module.PostgresStore(connection, schema=schema, roles=tables)


def test_postgres_export_includes_retained_extra_tables(backend_tables):
    module = backends()
    connection, schema, tables = backend_tables
    connection.execute(sql.SQL("CREATE TABLE {} (id text PRIMARY KEY,content text)").format(sql.Identifier(schema, "ks_ledger")))
    connection.execute(sql.SQL("INSERT INTO {} VALUES ('old','history')").format(sql.Identifier(schema, "ks_ledger")))
    store = module.PostgresStore(connection, schema=schema, roles=tables)
    with store.barrier():
        exported = store.export()
    assert exported["tables"]["ks_ledger"]["rows"] == [{"id": "old", "content": "history"}]


def test_backend_resume_is_bound_to_operation_and_checks_committed_rows(backend_tables):
    module = backends()
    connection, schema, tables = backend_tables
    store = module.PostgresStore(connection, schema=schema, roles=tables)
    with store.barrier():
        exported = store.export()
    target = "recovery_backend_target_" + uuid4().hex
    try:
        first = module.restore_postgres_store(connection, target, exported, operation_id="restore-owned")
        assert module.restore_postgres_store(connection, target, exported, operation_id="restore-owned") == first
        with pytest.raises(ValueError, match="operation|owned"):
            module.restore_postgres_store(connection, target, exported, operation_id="different")
        connection.execute(sql.SQL("UPDATE {} SET content='{{}}'").format(sql.Identifier(target, tables["full_docs"])))
        with pytest.raises(ValueError, match="changed|checksum"):
            module.restore_postgres_store(connection, target, exported, operation_id="restore-owned")
    finally:
        connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(target)))


def test_actual_vector_hnsw_arrays_and_precise_json_survive_restore(backend_tables):
    module = backends()
    connection, schema, tables = backend_tables
    target = "recovery_backend_target_" + uuid4().hex
    table = sql.Identifier(schema, tables["chunks_vdb"])
    connection.execute(sql.SQL("DROP TABLE {}").format(table))
    connection.execute(sql.SQL("CREATE TABLE {} (workspace varchar(255),id varchar(255),content_vector vector(3),"
        "chunk_ids varchar(255)[],created timestamp(0),detail jsonb,PRIMARY KEY(workspace,id))").format(table))
    connection.execute(sql.SQL("CREATE INDEX recovery_hnsw ON {} USING hnsw(content_vector vector_cosine_ops)").format(table))
    connection.execute(sql.SQL("INSERT INTO {} VALUES ('retained','v','[0.1,0.2,0.3]',ARRAY['one','two'],"
        "'2026-10-21T01:00:00','{{\"precise\":0.123456789012345678901234567890}}')").format(table))
    store = module.PostgresStore(connection, schema=schema, roles=tables)
    with store.barrier():
        exported = store.export()
    assert any(row["extname"] == "vector" for row in exported["extensions"])
    try:
        module.restore_postgres_store(connection, target, exported, operation_id="vector-restore")
        row = connection.execute(sql.SQL("SELECT content_vector::text AS v,chunk_ids,detail->>'precise' AS exact "
            "FROM {}").format(sql.Identifier(target, tables["chunks_vdb"]))).fetchone()
        assert row == {"v": "[0.1,0.2,0.3]", "chunk_ids": ["one", "two"],
                       "exact": "0.123456789012345678901234567890"}
    finally:
        connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(target)))


def test_missing_extension_identity_refuses_before_target_creation(backend_tables):
    module = backends()
    connection, schema, tables = backend_tables
    store = module.PostgresStore(connection, schema=schema, roles=tables)
    with store.barrier():
        exported = store.export()
    exported["extensions"].append({"extname": "missing_recovery_test_extension", "extversion": "1.0"})
    target = "recovery_backend_target_" + uuid4().hex
    with pytest.raises(ValueError, match="extension"):
        module.restore_postgres_store(connection, target, exported)
    assert connection.execute("SELECT to_regnamespace(%s) AS name", (target,)).fetchone()["name"] is None


def test_backend_resume_rejects_different_index_with_same_count(backend_tables):
    module = backends()
    connection, schema, tables = backend_tables
    target = "recovery_backend_target_" + uuid4().hex
    connection.execute(sql.SQL("CREATE INDEX retained_lookup ON {} (id)").format(sql.Identifier(schema, tables["full_docs"])))
    store = module.PostgresStore(connection, schema=schema, roles=tables)
    with store.barrier():
        exported = store.export()
    try:
        module.restore_postgres_store(connection, target, exported, operation_id="index-restore")
        connection.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(target, "retained_lookup")))
        connection.execute(sql.SQL("CREATE INDEX retained_lookup ON {} (workspace)").format(
            sql.Identifier(target, tables["full_docs"])))
        with pytest.raises(ValueError, match="index"):
            module.restore_postgres_store(connection, target, exported, operation_id="index-restore")
    finally:
        connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(target)))


def test_physical_replay_never_resolves_a_missing_owned_table_outside_schema(backend_tables, monkeypatch):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    runtime = importlib.import_module("recovery_runtime")
    connection, schema, _ = backend_tables
    # pg_catalog supplies a populated fallback without requiring CREATE in the
    # dedicated test role's deliberately read-only public schema.
    assert connection.execute("SELECT count(*) AS count FROM pg_class").fetchone()["count"] > 0
    assert hasattr(runtime, "verify_runtime_tables")
    with pytest.raises(ValueError, match="isolated schema"):
        runtime.verify_runtime_tables(connection, schema, ["pg_class"])


def test_physical_replay_preserves_explicit_connection_password(backend_tables, monkeypatch):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    runtime = importlib.import_module("recovery_runtime")
    connection, _, _ = backend_tables
    assert hasattr(runtime, "postgres_runtime_values")
    assert runtime.postgres_runtime_values(connection)["password"] == connection.info.password


@pytest.mark.parametrize("dsn", ["host=example.invalid hostaddr=127.0.0.1 dbname=test user=test",
                               "host=127.0.0.1,127.0.0.2 dbname=test user=test"])
def test_physical_replay_rejects_unsupported_connection_routing(monkeypatch, dsn):
    import importlib
    from pathlib import Path
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    runtime = importlib.import_module("recovery_runtime")
    connection = SimpleNamespace(info=SimpleNamespace(dsn=dsn, password="fixture"))
    with pytest.raises(ValueError, match="routing"):
        runtime.postgres_runtime_values(connection)
