"""Opt-in recovery against two explicitly supplied disposable Neo4j instances."""
import os
from uuid import uuid4

from neo4j import GraphDatabase
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def isolated_graphs():
    source = os.environ.get("RECOVERY_TEST_NEO4J_SOURCE")
    target = os.environ.get("RECOVERY_TEST_NEO4J_TARGET")
    if not source or not target:
        pytest.skip("two disposable recovery Neo4j test endpoints not supplied")
    assert source != target
    assert all(uri.startswith("bolt://127.0.0.1:") and not uri.endswith(":7687") for uri in (source, target))
    with GraphDatabase.driver(source, auth=None) as first, GraphDatabase.driver(target, auth=None) as second:
        first.verify_connectivity()
        second.verify_connectivity()
        yield (first, source), (second, target)


def test_graph_restore_preserves_nodes_edges_properties_and_refuses_populated_target(isolated_graphs):
    from papervault.store import recovery_backends

    assert hasattr(recovery_backends, "Neo4jStore")
    (source, source_uri), (target, target_uri) = isolated_graphs
    token = uuid4().hex
    with source.session(database="neo4j") as session:
        session.run("CREATE (a:PaperVaultCatalogEntity {entity_id:$a,_pv_namespace:$ns})-"
                    "[:DIRECTED {_pv_namespace:$ns,description:'measured relation',weight:0.5}]->"
                    "(b:PaperVaultCatalogEntity {entity_id:$b,_pv_namespace:$ns})",
                    a="Earth", b="Sun", ns=token).consume()
    try:
        store = recovery_backends.Neo4jStore(source, database="neo4j", uri=source_uri)
        with store.barrier():
            exported = store.export()
        assert len(exported["nodes"]) == 2 and len(exported["relationships"]) == 1
        result = recovery_backends.restore_neo4j_store(target, exported, database="neo4j", uri=target_uri,
                                                      disposable=True)
        assert result["nodes"] == 2 and result["relationships"] == 1
        with target.session(database="neo4j") as session:
            row = session.run("MATCH (a)-[r]->(b) RETURN a.entity_id AS a,b.entity_id AS b,"
                              "r.description AS description,r.weight AS weight").single()
            assert dict(row) == {"a": "Earth", "b": "Sun", "description": "measured relation", "weight": 0.5}
        with pytest.raises(ValueError, match="empty"):
            recovery_backends.restore_neo4j_store(target, exported, database="neo4j", uri=target_uri, disposable=True)
        with pytest.raises(ValueError, match="isolated|source"):
            recovery_backends.restore_neo4j_store(source, exported, database="neo4j", uri=source_uri, disposable=True)
    finally:
        for driver in (source, target):
            with driver.session(database="neo4j") as session:
                session.run("MATCH (n:PaperVaultCatalogEntity {_pv_namespace:$ns}) DETACH DELETE n", ns=token).consume()


def test_graph_resume_verifies_its_operation_and_committed_data(isolated_graphs):
    from papervault.store.recovery_backends import Neo4jStore, restore_neo4j_store
    (source, source_uri), (target, target_uri) = isolated_graphs
    token = uuid4().hex
    with source.session() as session:
        session.run("CREATE (:PaperVaultCatalogEntity {_pv_namespace:$ns,entity_id:'Earth'})", ns=token).consume()
    try:
        store = Neo4jStore(source, database="neo4j", uri=source_uri)
        with store.barrier():
            exported = store.export()
        options = {"database": "neo4j", "uri": target_uri, "disposable": True, "operation_id": token}
        first = restore_neo4j_store(target, exported, **options)
        assert restore_neo4j_store(target, exported, **options) == first
        with pytest.raises(ValueError, match="owned|operation"):
            restore_neo4j_store(target, exported, **{**options, "operation_id": "different"})
        with target.session() as session:
            session.run("MATCH (n:PaperVaultCatalogEntity {_pv_namespace:$ns}) SET n.entity_id='changed'", ns=token).consume()
        with pytest.raises(ValueError, match="changed"):
            restore_neo4j_store(target, exported, **options)
    finally:
        for driver in (source, target):
            with driver.session() as session:
                session.run("MATCH (n:PaperVaultCatalogEntity {_pv_namespace:$ns}) DETACH DELETE n", ns=token).consume()
        with target.session() as session:
            session.run("MATCH (n:_PapervaultRecovery {operation_id:$op}) DELETE n", op=token).consume()
