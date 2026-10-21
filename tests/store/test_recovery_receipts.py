"""A table inventory alone cannot prove retained physical output completeness."""
import copy
import importlib.util
from uuid import uuid4

import asyncpg
from pgvector.asyncpg import register_vector
from psycopg import sql
import pytest

from test_recovery_backends import backend_tables as backend_tables


@pytest.mark.parametrize("fault", [None, "empty_stores", "changed_row", "missing_graph", "ancestor"])
async def test_exported_materializations_match_runtime_receipt_hashes(backend_tables, tmp_path, fault):
    from papervault.knowledge.store.catalog_adapter import _inventory, _graph_inventory
    from papervault.store.recovery_backends import PostgresStore
    from papervault.store.repository import payload_hash
    connection, schema, roles = backend_tables
    # Include the representations that differ between jsonb export and asyncpg:
    # JSON null versus SQL NULL, JSON whitespace, vector float32, timestamp and
    # exact numeric scale. Runtime's original encoder supplies the reference.
    vector = sql.Identifier(schema, roles["chunks_vdb"])
    connection.execute(sql.SQL("DROP TABLE {}").format(vector))
    connection.execute(sql.SQL("CREATE TABLE {} (workspace text,id text,content_vector vector(3),"
        "nullable jsonb,literal_null jsonb,original_json json,n numeric(20,5),r real,stamp timestamp(6),"
        "zoned timestamptz,labels varchar[],PRIMARY KEY(workspace,id))").format(vector))
    connection.execute(sql.SQL("INSERT INTO {} VALUES('retained','vector','[0.1,0.2,0.3]',NULL,'null',"
        "'{{ \"spaced\": 1 }}',12.12000,0.1,'2026-01-01 01:00:00.120000',"
        "'2026-01-01 01:00:00+08',ARRAY['first',NULL])").format(vector))
    runtime = await asyncpg.connect(host=connection.info.host, port=connection.info.port,
        database=connection.info.dbname, user=connection.info.user, password=connection.info.password)
    try:
        await register_vector(runtime)
        outputs = []
        for role, table in roles.items():
            rows = [dict(row) for row in await runtime.fetch(f'SELECT * FROM "{schema}"."{table}" ORDER BY workspace,id')]
            output = _inventory(role, "postgres", "retained", rows, [row["id"] for row in rows], "2026-10-21T00:00:00+00:00")
            output["verification"]["evidence"]["backend_table"] = table
            outputs.append(output)
    finally:
        await runtime.close()
    node = {"id": "source-node", "labels": ["PaperVaultCatalogEntity"],
            "properties": {"_pv_namespace": "retained", "entity_id": "Earth", "description": "retained graph"}}
    outputs.append(_graph_inventory("retained", [{**node["properties"], "id": "Earth"}], [], "2026-10-21T00:00:00+00:00"))
    receipt = {"receipt_id": "retained-receipt", "namespace": "retained", "sealed": True, "durable": True,
               "aggregate_layers": ["retained"], "output_inventory": outputs}
    catalog = {"tables": {"knowledge_receipts": [{"receipt_id": receipt["receipt_id"], "receipt": receipt,
        "digest": payload_hash(receipt)}], "input_manifests": [], "knowledge_output_tombstones": [],
        "generation_manifests": [], "derivations": [], "selections": []}}
    store = PostgresStore(connection, schema=schema, roles=roles)
    with store.barrier():
        exported = store.export()
    graph = {"nodes": [node], "relationships": []}
    if fault == "empty_stores":
        for table in exported["tables"].values():
            table.update(rows=[], rows_json=[], json_text_columns=[])
    elif fault == "changed_row":
        table = exported["tables"][roles["full_docs"]]
        table["rows_json"][0] = table["rows_json"][0].replace("retained evidence", "changed evidence")
        import json
        table["rows"] = [json.loads(row) for row in table["rows_json"]]
    elif fault == "missing_graph":
        graph["nodes"] = []
    elif fault == "ancestor":
        receipt["aggregate_layers"].insert(0, "missing-ancestor")
        catalog["tables"]["knowledge_receipts"][0]["digest"] = payload_hash(receipt)
    assert importlib.util.find_spec("papervault.store.recovery_receipts") is not None
    from papervault.store.recovery_receipts import verify_receipt_closure
    if fault:
        with pytest.raises(ValueError, match="receipt|material|ancestor"):
            verify_receipt_closure(tmp_path, copy.deepcopy(catalog), exported, graph)
    else:
        verify_receipt_closure(tmp_path, catalog, exported, graph)
        from papervault.store.recovery_backends import restore_postgres_store
        target = "recovery_backend_target_" + uuid4().hex
        runtime = await asyncpg.connect(host=connection.info.host, port=connection.info.port,
            database=connection.info.dbname, user=connection.info.user, password=connection.info.password)
        try:
            await register_vector(runtime)
            restore_postgres_store(connection, target, exported, operation_id="typed-restore")
            for expected in outputs:
                if expected["store"] != "postgres":
                    continue
                table = roles[expected["role"]]
                values = [dict(row) for row in await runtime.fetch(f'SELECT * FROM "{target}"."{table}" ORDER BY workspace,id')]
                actual = _inventory(expected["role"], "postgres", "retained", values, [row["id"] for row in values], "restored")
                assert actual["sha256"] == expected["sha256"]
            connection.execute("SET TIME ZONE 'Asia/Shanghai'")
            restore_postgres_store(connection, target, exported, operation_id="typed-restore")
            for column, replacement in (("literal_null", None), ("original_json", '{"spaced":1}')):
                connection.execute(sql.SQL("UPDATE {} SET {}=%s::json").format(
                    sql.Identifier(target, roles["chunks_vdb"]), sql.Identifier(column)), (replacement,))
                with pytest.raises(ValueError, match="changed|checksum"):
                    restore_postgres_store(connection, target, exported, operation_id="typed-restore")
                original = exported["tables"][roles["chunks_vdb"]]["json_text_columns"][0][column]
                connection.execute(sql.SQL("UPDATE {} SET {}=%s::json").format(
                    sql.Identifier(target, roles["chunks_vdb"]), sql.Identifier(column)), (original,))
        finally:
            await runtime.close()
            connection.execute("SET TIME ZONE 'UTC'")
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(target)))
