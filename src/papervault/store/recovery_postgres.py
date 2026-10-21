"""Exact catalog checkpoint export and fenced restoration to a fresh test schema."""
from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from importlib.resources import files

from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from .repository import AuditEvent, PostgresCatalogRepository, payload_hash, render_catalog_ddl


def _tables(repository):
    return [row["name"] for row in repository.connection.execute(
        "SELECT tablename AS name FROM pg_tables WHERE schemaname=%s ORDER BY tablename",
        (repository.schema,))]


def catalog_indexes(repository):
    """Read every installed index, independently of bundled migration DDL.

    Stable deparsing quotes every identifier and qualifies relations regardless
    of the caller's search path. Roll back this read-only scope to restore both
    settings, including when nested inside the coordinated checkpoint.
    """
    with repository.connection.transaction(force_rollback=True):
        repository.connection.execute("SET LOCAL search_path=pg_catalog")
        repository.connection.execute("SET LOCAL quote_all_identifiers=on")
        rows = repository.connection.execute(
            "SELECT c.relname AS name,t.relname AS table,pg_get_indexdef(i.indexrelid) AS definition,"
            "i.indisvalid AND i.indisready AND i.indislive AS usable "
            "FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
            "JOIN pg_class t ON t.oid=i.indrelid JOIN pg_namespace n ON n.oid=t.relnamespace "
            "WHERE n.nspname=%s ORDER BY c.relname", (repository.schema,)).fetchall()
        if any(not row["usable"] for row in rows):
            raise ValueError("catalog index is not valid/ready for recovery")
        return {row["name"]: {key: row[key] for key in ("table", "definition")} for row in rows}


def _rebind_catalog_index(name, index, source, target):
    # Change only the indexed relation. Expressions and predicate literals may
    # contain the source schema's spelling and must retain their exact meaning.
    prefix = (r'\A(CREATE (?:UNIQUE )?INDEX ' + re.escape(sql.Identifier(name).as_string())
              + r' ON (?:ONLY )?)' + re.escape(sql.Identifier(source, index["table"]).as_string()) + ' ')
    match = re.match(prefix, index["definition"])
    if not match:
        raise ValueError("invalid recovery catalog index definition: " + name)
    return match[1] + sql.Identifier(target, index["table"]).as_string() + ' ' + index["definition"][match.end():]


def _target_indexes(state, schema):
    return {name: {**index, "definition": _rebind_catalog_index(
        name, index, state["checkpoint"]["schema"], schema)} for name, index in state["indexes"].items()}


def verify_catalog_indexes(repository, state):
    if catalog_indexes(repository) != _target_indexes(state, repository.schema):
        raise ValueError("restored catalog index inventory/definitions differ")


@contextmanager
def catalog_barrier(repository, backup_id):
    """Drain backend IO first, then stop every cooperative catalog command.

    Nonblocking locks avoid reversing the namespace->catalog command lock order.
    A crash releases locks and leaves the incomplete filesystem package unselected.
    Source filesystem and legacy graph lifetime barriers are owned by the caller.
    """
    connection = repository.connection
    keys = ["papervault:" + repository.schema + ":recovery", "papervault:" + repository.schema]
    acquired = []
    try:
        for key in keys:
            row = connection.execute("SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS locked", (key,)).fetchone()
            if not row["locked"]:
                raise RuntimeError("recovery requires catalog/backend writers to drain")
            acquired.append(key)
        with connection.transaction():
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            connection.execute("SET LOCAL TIME ZONE 'UTC'")
            connection.execute("SET LOCAL lock_timeout='1s'")
            connection.execute("SET LOCAL statement_timeout='7200s'")
            names = _tables(repository)
            if not {"records", "audit", "writer_fence", "knowledge_runtime"}.issubset(names):
                raise ValueError("recovery requires the installed P4/P5 catalog schema")
            connection.execute(sql.SQL("LOCK TABLE {} IN SHARE MODE").format(
                sql.SQL(",").join(repository._table(name) for name in names)))
            clock = connection.execute("SELECT pg_current_snapshot()::text AS catalog_snapshot, "
                "pg_current_wal_lsn()::text AS wal_lsn, current_database() AS database, "
                "current_setting('server_version') AS server_version").fetchone()
            runtime = connection.execute(sql.SQL("SELECT authority_epoch FROM {}").format(
                repository._table("knowledge_runtime"))).fetchone()
            audit = connection.execute(sql.SQL("SELECT COALESCE(max(id),0) AS head FROM {}").format(
                repository._table("audit"))).fetchone()
            checkpoint = {**clock, "schema": repository.schema, "backup_id": backup_id,
                          "authority_epoch": runtime["authority_epoch"], "audit_head": audit["head"],
                          "timeline": "not_observed_logical_snapshot", "catalog_tables": names,
                          "catalog_indexes_sha256": payload_hash(catalog_indexes(repository))}
            yield checkpoint
    finally:
        for key in reversed(acquired):
            connection.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))", (key,))


def export_catalog_state(repository, checkpoint):
    if checkpoint.get("schema") != repository.schema or checkpoint.get("catalog_tables") != _tables(repository):
        raise ValueError("catalog schema changed after checkpoint")
    indexes = catalog_indexes(repository)
    if payload_hash(indexes) != checkpoint.get("catalog_indexes_sha256"):
        raise ValueError("catalog index inventory changed after checkpoint")
    tables, columns, rows_json = {}, {}, {}
    for name in checkpoint["catalog_tables"]:
        table = repository._table(name)
        # jsonb supplies PostgreSQL's canonical text forms for timestamps/numeric
        # values; json_populate_recordset restores each declared column type.
        rows_json[name] = [row["encoded"] for row in repository.connection.execute(
            sql.SQL("SELECT to_jsonb(t)::text AS encoded FROM {} t").format(table))]
        tables[name] = [json.loads(encoded) for encoded in rows_json[name]]
        columns[name] = [row["column_name"] for row in repository.connection.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s "
            "ORDER BY ordinal_position", (repository.schema, name))]
    names = ["schema.sql", "knowledge_schema.sql"]
    if "sample" not in columns.get("generations", []):
        raise ValueError("coordinated recovery requires the explicit P5b catalog migration")
    if "knowledge_import_events" in tables:
        names.append("knowledge_migration_schema.sql")
    schema_files = {}
    for name in names:
        content = files("papervault.store").joinpath(name).read_text()
        schema_files[name] = {"content": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}
    sequences = {}
    for row in repository.connection.execute(
            "SELECT sequencename FROM pg_sequences WHERE schemaname=%s", (repository.schema,)):
        name = row["sequencename"]
        sequences[name] = repository.connection.execute(sql.SQL("SELECT last_value,is_called FROM {}").format(
            sql.Identifier(repository.schema, name))).fetchone()
    return {"schema_version": 2, "checkpoint": checkpoint, "tables": tables, "columns": columns, "indexes": indexes,
            "schema_files": schema_files, "sequences": sequences, "rows_json": rows_json}


def validate_catalog_state(state):
    if state.get("schema_version") != 2:
        raise ValueError("unsupported recovery catalog schema; installed index state required")
    indexes = state.get("indexes")
    if not isinstance(indexes, dict) or payload_hash(indexes) != state["checkpoint"].get("catalog_indexes_sha256"):
        raise ValueError("recovery catalog index inventory differs from checkpoint")
    for name, index in indexes.items():
        if (not isinstance(index, dict) or index.get("table") not in state["tables"]
                or not isinstance(index.get("definition"), str)):
            raise ValueError("invalid recovery catalog index inventory")
        _rebind_catalog_index(name, index, state["checkpoint"]["schema"], state["checkpoint"]["schema"])
    names = set(state.get("schema_files", {}))
    if names not in ({"schema.sql", "knowledge_schema.sql"},
                     {"schema.sql", "knowledge_schema.sql", "knowledge_migration_schema.sql"}):
        raise ValueError("recovery catalog schema inventory differs")
    for name, artifact in state["schema_files"].items():
        current = files("papervault.store").joinpath(name).read_text()
        if (artifact["content"] != current or artifact["sha256"] != hashlib.sha256(current.encode()).hexdigest()):
            raise ValueError("recovery catalog schema is incompatible with this restore tool")
    if (set(state["tables"]) != set(state["checkpoint"]["catalog_tables"])
            or set(state["columns"]) != set(state["tables"])
            or set(state.get("rows_json", {})) != set(state["tables"])):
        raise ValueError("recovery catalog table inventory differs")
    if any([json.loads(row) for row in state["rows_json"][name]] != rows
           for name, rows in state["tables"].items()):
        raise ValueError("recovery catalog exact row inventory differs")
    runtime = state["tables"]["knowledge_runtime"]
    if (len(runtime) != 1 or runtime[0]["authority_epoch"] != state["checkpoint"]["authority_epoch"]
            or max((row["id"] for row in state["tables"]["audit"]), default=0) != state["checkpoint"]["audit_head"]):
        raise ValueError("recovery checkpoint differs from captured catalog state")


def validate_catalog_target(connection, schema):
    database = conninfo_to_dict(connection.info.dsn).get("dbname", "")
    if (database.lower() in {"papervault_catalog", "papervault", "knowledge_system", "postgres"}
            or not re.search(r"(?:^|[_-])(test|rehearsal)(?:$|[_-])", database.lower())
            or not re.fullmatch(r"catalog_recovery_[a-z0-9_]+", schema)):
        raise ValueError("catalog recovery requires explicit isolated test targets")


def restore_catalog_state(connection, schema, state, *, recovery_id):
    """Restore all rows atomically; preserve old receipts and fence new dispatch.

    The caller must also verify filesystem/backend packages. An existing schema,
    even an empty one, is never overwritten. No production/default database is
    accepted, and the returned fence is usable only by explicit recovery tooling.
    """
    validate_catalog_state(state)
    validate_catalog_target(connection, schema)
    if not recovery_id:
        raise ValueError("catalog recovery requires an operation identity")
    connection.row_factory = dict_row
    if connection.execute("SELECT to_regnamespace(%s) AS name", (schema,)).fetchone()["name"]:
        raise ValueError("restore requires an absent empty catalog schema")
    repository = PostgresCatalogRepository(connection, schema=schema)
    fence = "recovery:" + recovery_id
    with connection.transaction():
        connection.execute("SET LOCAL TIME ZONE 'UTC'")
        # JSON object ordering is not a DDL dependency order (canonical packages
        # sort their keys). The base schema must precede its additive migrations.
        for name in ("schema.sql", "knowledge_schema.sql", "knowledge_migration_schema.sql"):
            if name not in state["schema_files"]:
                continue
            artifact = state["schema_files"][name]
            connection.execute(render_catalog_ddl(artifact["content"], schema))
        if set(_tables(repository)) != set(state["tables"]):
            raise ValueError("recovery schema does not represent every source catalog table")
        installed = catalog_indexes(repository)
        expected = _target_indexes(state, schema)
        constrained = {row["name"] for row in connection.execute(
            "SELECT i.relname AS name FROM pg_constraint c JOIN pg_class i ON i.oid=c.conindid "
            "JOIN pg_namespace n ON n.oid=i.relnamespace WHERE n.nspname=%s AND c.contype IN ('p','u','x')", (schema,))}
        for name, index in installed.items():
            if expected.get(name) != index:
                if name in constrained:
                    raise ValueError("recovery catalog index conflicts with bundled constraint: " + name)
                # Reconcile only fresh target defaults, before loading rows
                # that may be legal under a different installed unique index.
                connection.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(schema, name)))
        constraints = list(connection.execute("SELECT c.conname,t.relname,c.condeferrable,c.condeferred "
            "FROM pg_constraint c JOIN pg_class t ON t.oid=c.conrelid JOIN pg_namespace n ON n.oid=t.relnamespace "
            "WHERE n.nspname=%s AND c.contype='f'", (schema,)))
        for row in constraints:
            connection.execute(sql.SQL("ALTER TABLE {} ALTER CONSTRAINT {} DEFERRABLE INITIALLY DEFERRED").format(
                repository._table(row["relname"]), sql.Identifier(row["conname"])))
        connection.execute("SET CONSTRAINTS ALL DEFERRED")
        # Historical tombstones are evidence, not new execution. Suppress this
        # one insertion guard only inside the fresh schema's atomic import.
        connection.execute(sql.SQL("ALTER TABLE {} DISABLE TRIGGER tombstone_insert_guard").format(
            repository._table("knowledge_output_tombstones")))
        # Only raw DDL singleton defaults exist yet, not historical data.
        for name in ("writer_fence", "knowledge_runtime"):
            connection.execute(sql.SQL("DELETE FROM {}").format(repository._table(name)))
        for name, rows in state["tables"].items():
            actual = [row["column_name"] for row in connection.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s "
                "ORDER BY ordinal_position", (schema, name))]
            if actual != state["columns"][name]:
                raise ValueError("recovery schema column inventory differs: " + name)
            if rows:
                connection.execute(sql.SQL("INSERT INTO {} SELECT * FROM json_populate_recordset(NULL::{}, %s::json)").format(
                    repository._table(name), repository._table(name)), ("[" + ",".join(state["rows_json"][name]) + "]",))
            observed = [row["encoded"] for row in connection.execute(sql.SQL(
                "SELECT to_jsonb(t)::text AS encoded FROM {} t").format(repository._table(name)))]
            if sorted(observed) != sorted(state["rows_json"][name]):
                raise ValueError("restored catalog rows differ: " + name)
        connection.execute(sql.SQL("ALTER TABLE {} ENABLE TRIGGER tombstone_insert_guard").format(
            repository._table("knowledge_output_tombstones")))
        connection.execute("SET CONSTRAINTS ALL IMMEDIATE")
        for row in constraints:
            mode = ("DEFERRABLE INITIALLY DEFERRED" if row["condeferred"] else "DEFERRABLE INITIALLY IMMEDIATE"
                    ) if row["condeferrable"] else "NOT DEFERRABLE"
            connection.execute(sql.SQL("ALTER TABLE {} ALTER CONSTRAINT {} " + mode).format(
                repository._table(row["relname"]), sql.Identifier(row["conname"])))
        for name, sequence in state["sequences"].items():
            connection.execute("SELECT setval(%s::regclass,%s,%s)",
                (str(sql.Identifier(schema, name).as_string(connection)), sequence["last_value"], sequence["is_called"]))
        for name, index in expected.items():
            if installed.get(name) != index:
                connection.execute(index["definition"])
        # Bundled primary/unique indexes are already installed. Compare those
        # definitions too; mismatches roll back the entire new target schema.
        verify_catalog_indexes(repository, state)
        connection.execute(sql.SQL("UPDATE {} SET fence_id=%s").format(repository._table("writer_fence")), (fence,))
        epoch = state["checkpoint"]["authority_epoch"] + 1
        connection.execute(sql.SQL("UPDATE {} SET authority_epoch=%s").format(repository._table("knowledge_runtime")), (epoch,))
        repository.fence_id = fence
        repository._finish(None, AuditEvent("recovery", "recovery_restored", {
            "recovery_id": recovery_id, "checkpoint": state["checkpoint"],
            "dispatch_disabled": True, "authority_epoch": epoch}))
    return {"recovery_id": recovery_id, "schema": schema, "authority_epoch": epoch,
            "dispatch_disabled": True, "fence_id": fence}
