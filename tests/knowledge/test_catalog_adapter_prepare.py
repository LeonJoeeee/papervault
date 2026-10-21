"""Explicit additive backend preparation is independent of runtime CONFIG."""
from contextlib import contextmanager

import pytest


class Connection:
    def __init__(self, database):
        self.database = database
        self.statements = []

    @contextmanager
    def transaction(self):
        yield

    def execute(self, sql, params=None):
        self.statements.append(sql)
        assert sql.lstrip().upper().startswith(("SELECT", "CREATE"))
        return self

    def fetchone(self):
        if "current_database()" in self.statements[-1]:
            return (self.database,)
        if "extversion" in self.statements[-1]:
            return ("0.8.0",)
        return (True,)


def test_backend_preparation_creates_only_pinned_tables_and_model_indexes_additively():
    from scripts.prepare_knowledge_backend import prepare_postgres

    connection = Connection("p4_backend_rehearsal")
    result = prepare_postgres(connection, embedding_dim=1024, embedding_model_name="BGE/M3")
    assert "lightrag_vdb_chunks_bge_m3_1024d" in result["tables"]
    assert "lightrag_doc_status" in result["tables"]
    assert len(result["tables"]) == 11
    ddl = [sql for sql in connection.statements if sql.lstrip().upper().startswith("CREATE")]
    assert ddl
    assert all("IF NOT EXISTS" in sql for sql in ddl)
    assert not any(word in " ".join(ddl).upper() for word in ("DROP ", "DELETE ", "UPDATE ", "INSERT "))
    before = list(connection.statements)
    prepare_postgres(connection, embedding_dim=1024, embedding_model_name="BGE/M3")
    assert connection.statements == before + before


@pytest.mark.parametrize("database", ["papervault", "papervault_catalog", "knowledge_system", "postgres", "ambiguous-name"])
def test_preparation_refuses_live_or_unqualified_database_before_any_mutation(database):
    from scripts.prepare_knowledge_backend import prepare_postgres

    connection = Connection(database)
    with pytest.raises(ValueError, match="rehearsal|live"):
        prepare_postgres(connection, embedding_dim=1024)
    assert connection.statements == ["SELECT current_database()"]


def test_preparation_plan_matches_unsuffixed_runtime_without_model_name():
    from scripts.prepare_knowledge_backend import preparation_plan

    plan = preparation_plan(embedding_dim=1024)
    assert "lightrag_vdb_chunks" in plan["tables"]
    assert len(plan["neo4j"]) == 3


def test_neo4j_preparation_only_creates_common_catalog_indexes():
    from scripts.prepare_knowledge_backend import prepare_neo4j

    class Driver:
        def __init__(self):
            self.statements = []

        @contextmanager
        def session(self, *, database):
            assert database == "p4_backend_rehearsal"
            yield self

        def run(self, statement):
            self.statements.append(statement)
            return self

        def consume(self):
            pass

    driver = Driver()
    assert prepare_neo4j(driver, database="p4_backend_rehearsal") == {"indexes": 3}
    assert all(query.startswith("CREATE INDEX IF NOT EXISTS") for query in driver.statements)
    assert all("PaperVaultCatalogEntity" in query or "-[r:DIRECTED]" in query for query in driver.statements)
    with pytest.raises(ValueError, match="rehearsal|live"):
        prepare_neo4j(driver, database="neo4j")
    assert len(driver.statements) == 3
