"""Physical derivative inventories; restore preserves workspaces in isolated stores."""
from __future__ import annotations

import re
import json
from contextlib import contextmanager

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .repository import canonical_json, payload_hash


PG_ROLES = frozenset({"full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks",
                     "relation_chunks", "llm_response_cache", "doc_status", "chunks_vdb",
                     "entities_vdb", "relationships_vdb"})


def _json_projection(columns):
    names = [column["name"] for column in columns if column["type"] in {"json", "jsonb"}]
    return (sql.SQL("jsonb_build_object({})").format(sql.SQL(",").join(
        sql.SQL("{},{}::text").format(sql.Literal(name), sql.Identifier(name)) for name in names))
        if names else sql.SQL("'{}'::jsonb"))


def validate_postgres_rows(table):
    if [json.loads(row) for row in table["rows_json"]] != table["rows"]:
        raise ValueError("backend exact row inventory differs")
    texts = table.get("json_text_columns", [])
    columns = {column["name"] for column in table["columns"] if column["type"] in {"json", "jsonb"}}
    if len(texts) != len(table["rows"]):
        raise ValueError("backend exact JSON text inventory differs")
    for row, raw in zip(table["rows"], texts, strict=True):
        if set(raw) != columns or any(
                (json.loads(value) if value is not None else None) != row[name] for name, value in raw.items()):
            raise ValueError("backend exact JSON text projection differs")


class PostgresStore:
    format_name = "postgres-physical-rows-v1"

    def __init__(self, connection, *, schema, roles):
        if set(roles) != PG_ROLES or len(set(roles.values())) != len(PG_ROLES):
            raise ValueError("PostgreSQL recovery requires every configured derivative role/store")
        if not connection.autocommit:
            raise ValueError("recovery backend requires an explicit autocommit connection")
        self.connection, self.schema, self.roles = connection, schema, dict(roles)
        connection.row_factory = dict_row
        self._frozen = None

    def _tables(self):
        return [row["tablename"] for row in self.connection.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname=%s ORDER BY tablename", (self.schema,))]

    def estimated_bytes(self):
        return sum(self.connection.execute("SELECT pg_total_relation_size(%s::regclass) AS bytes",
            (sql.Identifier(self.schema, name).as_string(self.connection),)).fetchone()["bytes"]
            for name in self._tables())

    @contextmanager
    def barrier(self):
        with self.connection.transaction():
            self.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            self.connection.execute("SET LOCAL TIME ZONE 'UTC'")
            self.connection.execute("SET LOCAL lock_timeout='1s'")
            self.connection.execute("SET LOCAL statement_timeout='7200s'")
            names = self._tables()
            if not set(self.roles.values()).issubset(names):
                raise ValueError("a required PostgreSQL derivative store is absent")
            self.connection.execute(sql.SQL("LOCK TABLE {} IN SHARE MODE").format(sql.SQL(",").join(
                sql.Identifier(self.schema, name) for name in names)))
            self._frozen = names
            try:
                yield
            finally:
                self._frozen = None

    def export(self):
        if self._frozen is None or self._frozen != self._tables():
            raise ValueError("PostgreSQL export requires an unchanged fenced store inventory")
        tables = {}
        for name in self._frozen:
            columns = list(self.connection.execute(
                "SELECT a.attname AS name,format_type(a.atttypid,a.atttypmod) AS type,a.attnotnull AS not_null,"
                "pg_get_expr(d.adbin,d.adrelid) AS default,a.attgenerated AS generated "
                "FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid JOIN pg_namespace n ON n.oid=c.relnamespace "
                "LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum "
                "WHERE n.nspname=%s AND c.relname=%s AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum",
                (self.schema, name)))
            if any(column["generated"] or "nextval(" in (column["default"] or "") for column in columns):
                raise ValueError("backend generated/sequence columns require an explicit recovery adapter")
            constraints = list(self.connection.execute(
                "SELECT conname AS name,contype AS kind,pg_get_constraintdef(oid) AS definition FROM pg_constraint "
                "WHERE conrelid=%s::regclass", (sql.Identifier(self.schema, name).as_string(self.connection),)))
            if any(row["kind"] == "f" for row in constraints):
                raise ValueError("backend foreign keys require an explicit recovery adapter")
            indexes = list(self.connection.execute(
                "SELECT pg_get_indexdef(i.indexrelid) AS definition,i.indexprs IS NOT NULL AS expression "
                "FROM pg_index i WHERE i.indrelid=%s::regclass AND NOT EXISTS "
                "(SELECT FROM pg_constraint c WHERE c.conindid=i.indexrelid)",
                (sql.Identifier(self.schema, name).as_string(self.connection),)))
            if any(row["expression"] for row in indexes):
                raise ValueError("backend expression indexes require an explicit recovery adapter")
            order = sql.SQL(" ORDER BY workspace,id") if {"workspace", "id"}.issubset(
                column["name"] for column in columns) else sql.SQL("")
            rows = list(self.connection.execute(sql.SQL(
                "SELECT to_jsonb(t)::text AS value,{} AS json_text FROM {} t{}").format(
                    _json_projection(columns), sql.Identifier(self.schema, name), order)))
            rows_json = [row["value"] for row in rows]
            tables[name] = {"columns": columns, "constraints": constraints, "indexes": indexes,
                           "rows": [json.loads(row) for row in rows_json], "rows_json": rows_json,
                           "json_text_columns": [row["json_text"] for row in rows]}
        return {"schema_version": 1, "source_schema": self.schema, "roles": self.roles, "tables": tables,
                "extensions": list(self.connection.execute("SELECT extname,extversion FROM pg_extension ORDER BY extname")),
                "server_version": self.connection.execute("SHOW server_version").fetchone()["server_version"]}


def validate_postgres_target(connection, schema, exported):
    connection.row_factory = dict_row
    database = connection.execute("SELECT current_database() AS name").fetchone()["name"]
    if (not re.search(r"(?:^|[_-])(test|rehearsal)(?:$|[_-])", database.lower())
            or not re.fullmatch(r"recovery_backend_[a-z0-9_]+", schema)):
        raise ValueError("backend restore requires an explicit isolated test target")
    if (exported.get("schema_version") != 1 or set(exported.get("roles", {})) != PG_ROLES
            or not set(exported["roles"].values()).issubset(exported["tables"])):
        raise ValueError("backend recovery store inventory is incomplete")
    if "_papervault_recovery" in exported["tables"]:
        raise ValueError("backend source contains recovery bookkeeping")
    installed = list(connection.execute("SELECT extname,extversion FROM pg_extension ORDER BY extname"))
    if any(item not in installed for item in exported["extensions"]):
        raise ValueError("isolated PostgreSQL target requires matching extension versions before restore")
    for table in exported["tables"].values():
        validate_postgres_rows(table)


def _verify_postgres_rows(connection, schema, exported):
    names = {row["tablename"] for row in connection.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname=%s", (schema,))}
    if names - {"_papervault_recovery"} != set(exported["tables"]):
        raise ValueError("restored backend table inventory changed")
    for name, table in exported["tables"].items():
        rows = list(connection.execute(sql.SQL("SELECT to_jsonb(t)::text AS value,{} AS json_text FROM {} t").format(
            _json_projection(table["columns"]), sql.Identifier(schema, name))))
        expected_rows = [{"value": value, "json_text": raw} for value, raw in
                         zip(table["rows_json"], table["json_text_columns"], strict=True)]
        if sorted(map(canonical_json, rows)) != sorted(map(canonical_json, expected_rows)):
            raise ValueError("restored backend rows changed or checksum differs: " + name)
        indexes = list(connection.execute(
            "SELECT indexrelid FROM pg_index WHERE indrelid=%s::regclass AND indisvalid",
            (sql.Identifier(schema, name).as_string(connection),)))
        if len(indexes) != len(table["indexes"]) + sum(row["kind"] in {"p", "u", "x"} for row in table["constraints"]):
            raise ValueError("restored backend index inventory differs: " + name)
        definitions = [row["definition"] for row in connection.execute(
            "SELECT pg_get_indexdef(i.indexrelid) AS definition FROM pg_index i "
            "WHERE i.indrelid=%s::regclass AND NOT EXISTS (SELECT FROM pg_constraint c WHERE c.conindid=i.indexrelid)",
            (sql.Identifier(schema, name).as_string(connection),))]
        expected = [_rebind_index(row["definition"], exported["source_schema"], schema) for row in table["indexes"]]
        if sorted(definitions) != sorted(expected):
            raise ValueError("restored backend index definition differs: " + name)
        constraints = list(connection.execute(
            "SELECT conname AS name,contype AS kind,pg_get_constraintdef(oid) AS definition FROM pg_constraint "
            "WHERE conrelid=%s::regclass", (sql.Identifier(schema, name).as_string(connection),)))
        if sorted(constraints, key=lambda row: row["name"]) != sorted(table["constraints"], key=lambda row: row["name"]):
            raise ValueError("restored backend constraint inventory differs: " + name)


def _rebind_index(definition, source, target):
    definition = definition.replace(" ON " + sql.Identifier(source).as_string() + ".", " ON " + target + ".")
    return definition.replace(" ON " + source + ".", " ON " + target + ".")


def restore_postgres_store(connection, schema, exported, *, operation_id=None):
    validate_postgres_target(connection, schema, exported)
    binding = {"operation_id": operation_id, "export_sha256": payload_hash(exported)}
    result = {"schema": schema, "tables": sorted(exported["tables"]), "verified_export_sha256": payload_hash(exported)}
    if connection.execute("SELECT to_regnamespace(%s) AS name", (schema,)).fetchone()["name"]:
        marker = connection.execute("SELECT to_regclass(%s) AS name", (schema + "._papervault_recovery",)).fetchone()["name"]
        if not operation_id or not marker:
            raise ValueError("backend restore requires an absent empty schema or its owned operation")
        row = connection.execute(sql.SQL("SELECT binding FROM {}").format(
            sql.Identifier(schema, "_papervault_recovery"))).fetchone()
        if not row or row["binding"] != binding:
            raise ValueError("backend target belongs to another operation")
        with connection.transaction():
            connection.execute("SET LOCAL TIME ZONE 'UTC'")
            _verify_postgres_rows(connection, schema, exported)
        return result
    with connection.transaction():
        connection.execute("SET LOCAL TIME ZONE 'UTC'")
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        for name, table in exported["tables"].items():
            definitions = []
            for column in table["columns"]:
                # Type/default/constraint definitions come from the verified,
                # operator-trusted database package, never from document text.
                definition = sql.SQL("{} {}" + (" NOT NULL" if column["not_null"] else "")).format(
                    sql.Identifier(column["name"]), sql.SQL(column["type"]))
                if column["default"] is not None:
                    definition += sql.SQL(" DEFAULT " + column["default"])
                definitions.append(definition)
            for constraint in table["constraints"]:
                definitions.append(sql.SQL("CONSTRAINT {} {}").format(sql.Identifier(constraint["name"]),
                    sql.SQL(constraint["definition"])))
            target = sql.Identifier(schema, name)
            connection.execute(sql.SQL("CREATE TABLE {} ({})").format(target, sql.SQL(",").join(definitions)))
            if table["rows"]:
                # Preserve SQL NULL, JSON literal null and lexical JSON text.
                # json_populate_record alone cannot distinguish those inputs.
                values = [sql.SQL("(j.texts->>{})::{}").format(sql.Literal(column["name"]), sql.SQL(column["type"]))
                          if column["type"] in {"json", "jsonb"} else sql.Identifier("r", column["name"])
                          for column in table["columns"]]
                connection.execute(sql.SQL("INSERT INTO {} SELECT {} FROM "
                    "json_array_elements(%s::json) WITH ORDINALITY AS s(value,idx) "
                    "CROSS JOIN LATERAL json_populate_record(NULL::{},s.value) AS r "
                    "CROSS JOIN LATERAL (SELECT (%s::jsonb -> (s.idx-1)::int) AS texts) AS j").format(
                        target, sql.SQL(",").join(values), target),
                    ("[" + ",".join(table["rows_json"]) + "]", Jsonb(table["json_text_columns"])))
            for index in table["indexes"]:
                connection.execute(_rebind_index(index["definition"], exported["source_schema"], schema))
        _verify_postgres_rows(connection, schema, exported)
        if operation_id:
            connection.execute(sql.SQL("CREATE TABLE {} (binding jsonb NOT NULL)").format(
                sql.Identifier(schema, "_papervault_recovery")))
            connection.execute(sql.SQL("INSERT INTO {} VALUES(%s)").format(sql.Identifier(schema, "_papervault_recovery")),
                               (Jsonb(binding),))
    return result


class Neo4jStore:
    """Physical graph export inside the coordinator's graph lifetime barrier.

    The read transaction alone is not a cross-store snapshot. Cooperating graph
    clients must have drained and the knowledge-root exclusive lease must remain
    held by CoordinatedSource until registration completes.
    """
    format_name = "neo4j-physical-graph-v1"

    def __init__(self, driver, *, database, uri):
        self.driver, self.database, self.uri = driver, database, safe_uri(uri)
        self.transaction = None

    def estimated_bytes(self):
        with self.driver.session(database=self.database, default_access_mode="READ") as session:
            count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        # Admission includes per-object overhead and a held disk reservation;
        # oversized properties can still exceed this estimate and fail capture.
        return (count + edges) * 16384

    @contextmanager
    def barrier(self):
        with self.driver.session(database=self.database, default_access_mode="READ") as session:
            with session.begin_transaction(timeout=7200) as transaction:
                self.transaction = transaction
                try:
                    yield
                finally:
                    self.transaction = None

    def export(self):
        if self.transaction is None:
            raise ValueError("Neo4j export requires coordinated barrier scope")
        transaction = self.transaction
        nodes = [dict(row) for row in transaction.run(
            "MATCH (n) RETURN elementId(n) AS id,labels(n) AS labels,properties(n) AS properties ORDER BY id")]
        relationships = [dict(row) for row in transaction.run(
            "MATCH (a)-[r]->(b) RETURN elementId(r) AS id,elementId(a) AS source,elementId(b) AS target,"
            "type(r) AS type,properties(r) AS properties ORDER BY id")]
        schemas = [dict(row) for row in transaction.run("SHOW CONSTRAINTS YIELD name,createStatement RETURN name,createStatement")]
        indexes = [dict(row) for row in transaction.run("SHOW INDEXES YIELD name,type,owningConstraint,createStatement "
            "WHERE type <> 'LOOKUP' AND owningConstraint IS NULL RETURN name,createStatement")]
        engine = [dict(row) for row in transaction.run("CALL dbms.components() YIELD name,versions,edition RETURN name,versions,edition")]
        value = {"schema_version": 1, "source_uri": self.uri, "source_database": self.database,
                 "database_identity": dict(transaction.run("CALL db.info() YIELD id,name RETURN id,name").single()),
                 "nodes": nodes, "relationships": relationships, "constraints": schemas, "indexes": indexes,
                 "engine": engine}
        # Fail explicitly on unsupported spatial/temporal property encodings.
        # str(value) would silently change their Neo4j property types on restore.
        json.dumps(value, allow_nan=False)
        return value


def _cypher_identifier(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("invalid Neo4j schema identity")
    return "`" + value.replace("`", "``") + "`"


def safe_uri(value):
    from urllib.parse import urlsplit, urlunsplit
    parsed = urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("recovery endpoint must exclude credentials, queries and fragments")
    return urlunsplit(parsed)


def _graph_endpoint(value):
    import socket
    from urllib.parse import urlsplit
    parsed = urlsplit(safe_uri(value))
    return {address[4][0] for address in socket.getaddrinfo(parsed.hostname, parsed.port or 7687)}, parsed.port or 7687


def _verify_graph(transaction, exported, mapping):
    nodes = [dict(row) for row in transaction.run("MATCH (n) WHERE NOT n:_PapervaultRecovery "
        "RETURN elementId(n) AS id,labels(n) AS labels,properties(n) AS properties")]
    expected = [{**row, "id": mapping[row["id"]], "labels": sorted(row["labels"])} for row in exported["nodes"]]
    actual = [{**row, "labels": sorted(row["labels"])} for row in nodes]
    edges = [dict(row) for row in transaction.run("MATCH (a)-[r]->(b) "
        "RETURN elementId(a) AS source,elementId(b) AS target,type(r) AS type,properties(r) AS properties")]
    expected_edges = [{"source": mapping[row["source"]], "target": mapping[row["target"]],
        "type": row["type"], "properties": row["properties"]} for row in exported["relationships"]]
    if (sorted(map(canonical_json, actual)) != sorted(map(canonical_json, expected))
            or sorted(map(canonical_json, edges)) != sorted(map(canonical_json, expected_edges))):
        raise ValueError("restored graph properties or inventory changed")


def _graph_schema(session):
    constraints = list(session.run("SHOW CONSTRAINTS YIELD name,createStatement RETURN name,createStatement"))
    indexes = list(session.run("SHOW INDEXES YIELD name,type,owningConstraint,createStatement "
        "WHERE type<>'LOOKUP' AND owningConstraint IS NULL RETURN name,createStatement"))
    return {row["name"]: row["createStatement"] for row in constraints + indexes}


def restore_neo4j_store(driver, exported, *, database, uri, disposable=False, operation_id=None):

    hosts, port = _graph_endpoint(uri)
    source_hosts, source_port = _graph_endpoint(exported["source_uri"])
    if (exported.get("schema_version") != 1 or not disposable
            or (hosts & source_hosts and port == source_port)):
        raise ValueError("Neo4j restore requires an explicit disposable endpoint isolated from the source")
    if database == "system":
        raise ValueError("Neo4j system database is not a recovery target")
    ids = {row["id"] for row in exported["nodes"]}
    if len(ids) != len(exported["nodes"]) or any(
            row["source"] not in ids or row["target"] not in ids for row in exported["relationships"]) or any(
            "_PapervaultRecovery" in row["labels"] for row in exported["nodes"]):
        raise ValueError("Neo4j package has duplicate nodes or dangling relationships")
    with driver.session(database=database) as session:
        identity = dict(session.run("CALL db.info() YIELD id,name RETURN id,name").single())
        if identity["id"] == exported["database_identity"]["id"]:
            raise ValueError("Neo4j target is the source database through an alias")
        fingerprint = payload_hash(exported)
        with session.begin_transaction(timeout=7200) as transaction:
            markers = list(transaction.run("MATCH (n:_PapervaultRecovery) RETURN properties(n) AS marker"))
            count = transaction.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            if count:
                if (not operation_id or len(markers) != 1 or markers[0]["marker"].get("operation_id") != operation_id
                        or markers[0]["marker"].get("export_sha256") != fingerprint):
                    raise ValueError("Neo4j restore requires an empty target or its owned operation")
                mapping = json.loads(markers[0]["marker"]["mapping"])
            else:
                schemas = list(transaction.run("SHOW CONSTRAINTS YIELD name RETURN name"))
                indexes = list(transaction.run("SHOW INDEXES YIELD type WHERE type<>'LOOKUP' RETURN type"))
                if schemas or indexes:
                    raise ValueError("Neo4j restore requires an empty target schema")
                mapping = {}
                for node in exported["nodes"]:
                    labels = "".join(":" + _cypher_identifier(label) for label in node["labels"])
                    created = transaction.run("CREATE (n" + labels + ") SET n=$properties RETURN elementId(n) AS id",
                                              properties=node["properties"]).single()
                    mapping[node["id"]] = created["id"]
                for edge in exported["relationships"]:
                    transaction.run("MATCH (a),(b) WHERE elementId(a)=$source AND elementId(b)=$target "
                        "CREATE (a)-[r:" + _cypher_identifier(edge["type"]) + "]->(b) SET r=$properties",
                        source=mapping[edge["source"]], target=mapping[edge["target"]], properties=edge["properties"]).consume()
                if operation_id:
                    transaction.run("CREATE (n:_PapervaultRecovery) SET n=$properties", properties={
                        "operation_id": operation_id, "export_sha256": fingerprint, "mapping": canonical_json(mapping)}).consume()
            _verify_graph(transaction, exported, mapping)
            transaction.commit()
        # Neo4j does not mix schema updates with graph updates in one transaction.
        # A failed schema install leaves an explicit unaccepted target for resume.
        expected = {row["name"]: row["createStatement"] for row in exported["constraints"] + exported["indexes"]}
        existing = _graph_schema(session)
        if any(expected.get(name) != definition for name, definition in existing.items()):
            raise ValueError("restored graph schema changed")
        for name, definition in expected.items():
            if name not in existing:
                session.run(definition).consume()
        session.run("CALL db.awaitIndexes(300)").consume()
        if _graph_schema(session) != expected:
            raise ValueError("restored graph schema inventory differs")
    return {"database": database, "uri": uri, "nodes": len(ids),
            "relationships": len(exported["relationships"]), "node_id_mapping": mapping,
            "verified_export_sha256": fingerprint}
