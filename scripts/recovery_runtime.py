"""Explicit isolated physical replay wiring for the standalone recovery CLI.

Run in a fresh process. Runtime defaults and service dotenv files never select
targets; all backend connection variables are replaced from validated inputs.
"""
from __future__ import annotations

import json
import os


def postgres_runtime_values(connection):
    import ipaddress
    from psycopg.conninfo import conninfo_to_dict
    values = conninfo_to_dict(connection.info.dsn)
    if (values.get("hostaddr") and values["hostaddr"] != values.get("host")
            or any("," in values.get(key, "") for key in ("host", "port"))):
        raise ValueError("physical replay does not support ambiguous hostaddr or multihost connection routing")
    try:
        ipaddress.ip_address(values.get("host", ""))
    except ValueError as exc:
        raise ValueError("physical replay routing requires one explicit IP address") from exc
    # ConnectionInfo.dsn intentionally redacts credentials.
    values["password"] = connection.info.password
    if not all(values.get(key) for key in ("host", "dbname", "user")):
        raise ValueError("physical replay requires explicit backend host, database and user")
    return values


def verify_runtime_tables(connection, schema, tables):
    """Public remains visible for pgvector operators, never as a table fallback."""
    from psycopg import sql
    with connection.transaction():
        connection.execute(sql.SQL("SET LOCAL search_path TO {},public").format(sql.Identifier(schema)))
        for name in tables:
            owned = sql.Identifier(schema, name).as_string(connection)
            row = connection.execute("SELECT to_regclass(%s)::oid AS owned,to_regclass(%s)::oid AS resolved",
                (owned, sql.Identifier(name).as_string(connection))).fetchone()
            if row["owned"] is None or row["owned"] != row["resolved"]:
                raise ValueError("physical replay table is absent from the isolated schema: " + name)


def physical_replay_factory(config, targets, package, root):
    from psycopg import sql
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    from papervault.store.recovery import child, digest_file, verify_recovery_set
    from papervault.store.recovery_backends import _graph_endpoint, validate_postgres_target
    from papervault.store.repository import payload_hash
    from prepare_knowledge_backend import preparation_plan

    verify_recovery_set(package)
    pg, graph = targets["postgres"], targets["neo4j"]
    exported_pg = json.loads(child(package, "postgres/data.json").read_bytes())
    exported_graph = json.loads(child(package, "neo4j/data.json").read_bytes())
    validate_postgres_target(pg["connection"], pg["schema"], exported_pg)
    hosts, port = _graph_endpoint(graph["uri"])
    source_hosts, source_port = _graph_endpoint(exported_graph["source_uri"])
    if not graph.get("disposable") or (hosts & source_hosts and port == source_port):
        raise ValueError("physical replay requires a disposable graph endpoint isolated from the source")
    with graph["driver"].session(database=graph["database"]) as session:
        identity = session.run("CALL db.info() YIELD id RETURN id").single()["id"]
        if identity == exported_graph["database_identity"]["id"]:
            raise ValueError("physical replay graph is the source through an alias")
    values = postgres_runtime_values(pg["connection"])
    identity = {"postgres": {key: values[key] for key in ("host", "dbname", "user")},
        "postgres_port": values.get("port", "5432"), "schema": pg["schema"],
        "neo4j_uri": graph["uri"], "neo4j_database": graph["database"],
        "manifest_sha256": digest_file(package / "manifest.json")["sha256"]}
    graph_password = os.environ[graph["password_env"]] if graph.get("password_env") else ""
    # Remove inherited targets/options before any runtime import. These variables
    # are process-local; the caller's service environment is never edited.
    for name in list(os.environ):
        if name.startswith(("POSTGRES_", "NEO4J_")):
            os.environ.pop(name)
    for field, variable in (("host", "HOST"), ("port", "PORT"), ("user", "USER"), ("password", "PASSWORD"),
                            ("dbname", "DATABASE"), ("sslmode", "SSL_MODE"), ("sslcert", "SSL_CERT"),
                            ("sslkey", "SSL_KEY"), ("sslrootcert", "SSL_ROOT_CERT"), ("sslcrl", "SSL_CRL")):
        os.environ["POSTGRES_" + variable] = values.get(field, "5432" if field == "port" else "")
    os.environ["POSTGRES_DB"] = values["dbname"]
    os.environ["POSTGRES_SERVER_SETTINGS"] = "search_path=" + pg["schema"] + ",public"
    os.environ["NEO4J_URI"] = graph["uri"]
    os.environ["NEO4J_DATABASE"] = graph["database"]
    os.environ["NEO4J_USERNAME"] = graph.get("user", "neo4j")
    os.environ["NEO4J_PASSWORD"] = graph_password
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["PAPERVAULT_STORAGE"] = str(root / "knowledge")

    def factory(k, **kwargs):
        operation = k.repository.fence_id
        binding = {"operation": operation, "target": identity}
        connection = pg["connection"]
        connection.row_factory = dict_row
        plan = preparation_plan(embedding_dim=1024)
        present = connection.execute("SELECT to_regnamespace(%s) AS name", (pg["schema"],)).fetchone()["name"]
        if present:
            marker = connection.execute("SELECT to_regclass(%s) AS name", (pg["schema"] + "._papervault_replay",)).fetchone()["name"]
            if not marker or connection.execute(sql.SQL("SELECT binding FROM {}").format(
                    sql.Identifier(pg["schema"], "_papervault_replay"))).fetchone()["binding"] != binding:
                raise ValueError("physical replay backend is not this operation's empty/owned target")
        else:
            with connection.transaction():
                connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(pg["schema"])))
                connection.execute(sql.SQL("SET LOCAL search_path TO {},public").format(sql.Identifier(pg["schema"])))
                for statement in plan["postgres"]:
                    if not statement.startswith("CREATE EXTENSION"):
                        connection.execute(statement)
                connection.execute(sql.SQL("CREATE TABLE {} (binding jsonb NOT NULL)").format(
                    sql.Identifier(pg["schema"], "_papervault_replay")))
                connection.execute(sql.SQL("INSERT INTO {} VALUES(%s)").format(sql.Identifier(pg["schema"], "_papervault_replay")),
                    (Jsonb(binding),))
        verify_runtime_tables(connection, pg["schema"], plan["tables"])
        with graph["driver"].session(database=graph["database"]) as session:
            with session.begin_transaction() as transaction:
                markers = list(transaction.run("MATCH (n:_PapervaultReplay) RETURN n.binding AS binding"))
                count = transaction.run("MATCH(n) RETURN count(n) AS count").single()["count"]
                if count and (len(markers) != 1 or markers[0]["binding"] != payload_hash(binding)):
                    raise ValueError("physical replay graph is not this operation's empty/owned target")
                if not count:
                    transaction.run("CREATE (:_PapervaultReplay {binding:$binding})", binding=payload_hash(binding)).consume()
                transaction.commit()
            for statement in plan["neo4j"]:
                session.run(statement).consume()
        from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
        return CatalogLightRAGAdapter(k, **kwargs)

    return factory, identity
