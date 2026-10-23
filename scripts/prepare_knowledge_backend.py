#!/usr/bin/env python3
"""Review and explicitly prepare P4 backend schema, without loading runtime CONFIG.

Default: print an additive SQL/Cypher plan and make no connection. Execution
requires explicit connection arguments. Test/rehearsal database names are the
default boundary; any production administration needs a separate authorization
and the explicit --allow-live override. No legacy rows are migrated or deleted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from importlib import metadata

_BASE_TABLES = (
    "LIGHTRAG_DOC_FULL", "LIGHTRAG_DOC_CHUNKS", "LIGHTRAG_FULL_ENTITIES",
    "LIGHTRAG_FULL_RELATIONS", "LIGHTRAG_ENTITY_CHUNKS", "LIGHTRAG_RELATION_CHUNKS",
    "LIGHTRAG_LLM_CACHE", "LIGHTRAG_DOC_STATUS", "LIGHTRAG_VDB_CHUNKS",
    "LIGHTRAG_VDB_ENTITY", "LIGHTRAG_VDB_RELATION",
)
_NEO4J = (
    "CREATE INDEX IF NOT EXISTS FOR (n:PaperVaultCatalogEntity) ON (n.entity_id,n._pv_namespace)",
    "CREATE INDEX IF NOT EXISTS FOR (n:PaperVaultCatalogEntity) ON (n._pv_namespace)",
    "CREATE INDEX IF NOT EXISTS FOR ()-[r:DIRECTED]-() ON (r._pv_namespace)",
)
_KEY_CHECK = (
    "SELECT EXISTS(SELECT 1 FROM pg_index i WHERE i.indrelid=to_regclass(%s) "
    "AND i.indisunique AND i.indisvalid AND i.indpred IS NULL AND i.indexprs IS NULL "
    "AND i.indnkeyatts=2 AND i.indnatts=2 AND "
    "(SELECT array_agg(a.attname::text ORDER BY a.attname) "
    "FROM unnest(i.indkey::smallint[]) AS k(attnum) "
    "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum) "
    "=ARRAY['id','workspace']::text[]) AS prepared"
)


def _pinned_tables():
    distribution = metadata.distribution("lightrag-hku")
    if distribution.version != "1.5.7":
        raise RuntimeError("backend preparation requires reviewed lightrag-hku==1.5.7")
    # Read literal schema data without importing LightRAG's dotenv/bootstrap code.
    source = distribution.locate_file("lightrag/kg/postgres_impl.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "TABLES"
                                               for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("pinned backend schema literal unavailable")


def _check_target(database, allow_live):
    known_live = {"papervault", "papervault_catalog", "knowledge_system", "postgres", "template0", "template1"}
    if not allow_live and (database.lower() in known_live or
            re.search(r"(?:^|[_-])(test|rehearsal)(?:$|[_-])", database.lower()) is None):
        raise ValueError("backend preparation requires an explicit test/rehearsal database; live administration is refused")


def preparation_plan(*, embedding_dim, embedding_model_name=None, vector_type="vector"):
    if type(embedding_dim) is not int or not 0 < embedding_dim <= 16000 or vector_type not in {"vector", "halfvec"}:
        raise ValueError("explicit supported embedding dimension/vector type required")
    suffix = None
    if embedding_model_name and embedding_model_name.strip():
        suffix = re.sub(r"[^a-zA-Z0-9_]", "_", embedding_model_name.strip().lower()) + f"_{embedding_dim}d"
    definitions = _pinned_tables()
    statements = ["CREATE EXTENSION IF NOT EXISTS vector"]
    tables, columns = [], {}
    for base in _BASE_TABLES:
        vector = base.startswith("LIGHTRAG_VDB_")
        table = (base + "_" + suffix if vector and suffix else base).lower()
        if len(table) > 63:
            raise ValueError("embedding model suffix exceeds PostgreSQL identifier length")
        ddl = definitions[base]["ddl"].replace(base, table)
        ddl = ddl.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1)
        ddl = ddl.replace("VECTOR(dimension)", f"{vector_type.upper()}({embedding_dim})")
        statements.append(ddl)
        index_key = hashlib.sha256(table.encode()).hexdigest()[:24]
        statements.append(f'CREATE INDEX IF NOT EXISTS pv_p4_id_{index_key} ON "{table}"(id)')
        if vector:
            statements.append(f'CREATE INDEX IF NOT EXISTS pv_p4_vec_{index_key} ON "{table}" '
                              f'USING hnsw(content_vector {vector_type}_cosine_ops)')
        columns[table] = re.findall(
            r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:varchar|text|jsonb|integer|int4|bigint|timestamp|vector|halfvec)\b",
            ddl, flags=re.I | re.M)
        tables.append(table)
    return {"dependency": "lightrag-hku==1.5.7", "embedding_dim": embedding_dim,
            "embedding_model_name": embedding_model_name, "vector_type": vector_type,
            "tables": tables, "columns": columns, "postgres": statements, "neo4j": list(_NEO4J)}


def prepare_postgres(connection, *, embedding_dim, embedding_model_name=None,
                     vector_type="vector", allow_live=False):
    _check_target(connection.execute("SELECT current_database()").fetchone()[0], allow_live)
    plan = preparation_plan(embedding_dim=embedding_dim, embedding_model_name=embedding_model_name,
                            vector_type=vector_type)
    with connection.transaction():
        for statement in plan["postgres"]:
            connection.execute(statement)
        if vector_type == "halfvec":
            version = connection.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
            match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
            if match is None or tuple(map(int, match.groups())) < (0, 7, 0):
                raise RuntimeError("halfvec requires explicitly upgraded pgvector >=0.7.0")
        for table in plan["tables"]:
            selected = ",".join(f'"{column}"' for column in plan["columns"][table])
            connection.execute(f'SELECT {selected} FROM "{table}" LIMIT 0')
            if not connection.execute(_KEY_CHECK, (table,)).fetchone()[0]:
                raise RuntimeError("existing backend requires a reviewed migration for its ownership key")
    return plan


def prepare_neo4j(driver, *, database, allow_live=False):
    _check_target(database, allow_live)
    with driver.session(database=database) as session:
        for statement in _NEO4J:
            session.run(statement).consume()
    return {"indexes": len(_NEO4J)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--embedding-model-name", default=None,
                        help="EmbeddingFunc.model_name, omitted for the current unsuffixed runtime tables")
    parser.add_argument("--vector-type", choices=("vector", "halfvec"), default="vector")
    parser.add_argument("--execute", action="store_true", help="execute the reviewed additive plan")
    parser.add_argument("--allow-live", action="store_true", help="explicit separately authorized admin override")
    parser.add_argument("--dsn", help="explicit dedicated backend DSN, never read from runtime CONFIG")
    parser.add_argument("--neo4j-uri")
    parser.add_argument("--neo4j-database")
    parser.add_argument("--neo4j-user")
    parser.add_argument("--neo4j-password-env", help="explicit environment variable containing the admin password")
    args = parser.parse_args(argv)
    options = {key: getattr(args, key) for key in ("embedding_dim", "embedding_model_name", "vector_type")}
    plan = preparation_plan(**options)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.dsn and not args.neo4j_uri:
        parser.error("--execute requires an explicit --dsn and/or --neo4j-uri")
    if args.dsn:
        import psycopg
        from psycopg.conninfo import conninfo_to_dict

        database = conninfo_to_dict(args.dsn).get("dbname")
        if not database:
            parser.error("backend DSN must contain an explicit database name")
        _check_target(database, args.allow_live)
        with psycopg.connect(args.dsn, autocommit=True) as connection:
            prepare_postgres(connection, **options, allow_live=args.allow_live)
    if args.neo4j_uri:
        from neo4j import GraphDatabase

        if not all((args.neo4j_database, args.neo4j_user, args.neo4j_password_env)):
            parser.error("Neo4j execution requires explicit database, user, and password environment variable")
        _check_target(args.neo4j_database, args.allow_live)
        password = os.environ.get(args.neo4j_password_env)
        if not password:
            parser.error("explicit Neo4j password environment variable is empty")
        with GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, password)) as driver:
            prepare_neo4j(driver, database=args.neo4j_database, allow_live=args.allow_live)
    print(json.dumps({"prepared": True, "tables": plan["tables"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
