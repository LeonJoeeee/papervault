"""Runtime authority and connection ownership, without network access."""

import asyncio
from contextlib import asynccontextmanager

import pytest


class Connection:
    autocommit = True

    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True


class Knowledge:
    def __init__(self, catalog):
        self.catalog = catalog

    def get_authority_state(self):
        assert not self.catalog.connection.closed
        return {"authority": "legacy", "authority_epoch": 0}

    def command(self, action, operation_id, payload):
        assert not self.catalog.connection.closed
        return {"action": action, "operation_id": operation_id, "payload": payload}

    @asynccontextmanager
    async def namespace_access(self, claim, seal=False):
        assert not self.catalog.connection.closed
        yield self


def test_each_command_owns_and_closes_its_connection(monkeypatch):
    from papervault.store.knowledge_runtime import KnowledgeRepositoryProxy

    connections = []

    def connect(dsn, **kwargs):
        assert dsn == "postgresql://test.invalid/task_test"
        assert kwargs["autocommit"] is True
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr("psycopg.connect", connect)
    repository = KnowledgeRepositoryProxy("postgresql://test.invalid/task_test", factory=Knowledge)
    assert repository.command("enqueue", "op-1", {"unit_id": "u-1"})["operation_id"] == "op-1"
    assert repository.get_authority_state()["authority"] == "legacy"
    assert len(connections) == 2
    assert all(connection.closed for connection in connections)


def test_failed_database_read_does_not_fall_back_to_legacy(monkeypatch):
    from papervault.store.knowledge_runtime import KnowledgeRepositoryProxy

    def connect(*args, **kwargs):
        raise ConnectionError("test database unavailable")

    monkeypatch.setattr("psycopg.connect", connect)
    repository = KnowledgeRepositoryProxy("postgresql://test.invalid/task_test", factory=Knowledge)
    with pytest.raises(ConnectionError):
        repository.get_authority_state()


async def test_namespace_connection_is_reused_only_by_its_own_task(monkeypatch):
    from papervault.store.knowledge_runtime import KnowledgeRepositoryProxy

    connections = []

    def connect(*args, **kwargs):
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr("psycopg.connect", connect)
    proxy = KnowledgeRepositoryProxy("postgresql://test.invalid/task_test", factory=Knowledge)
    async with proxy.namespace_access({"namespace": "private-1"}) as actual:
        proxy.get_authority_state()
        assert len(connections) == 1

        async def child():
            proxy.get_authority_state()

        await asyncio.create_task(child())
        assert len(connections) == 2
        assert connections[0] is actual.catalog.connection
        assert not connections[0].closed
        assert connections[1].closed
    assert connections[0].closed


@pytest.mark.parametrize("value", ["json", "", "Postgre", "typo"])
def test_invalid_knowledge_selector_refuses_startup(monkeypatch, value):
    from papervault.store.knowledge_runtime import knowledge_jobs_mode

    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", value)
    with pytest.raises(ValueError):
        knowledge_jobs_mode()


def test_unconfigured_install_keeps_legacy_default(monkeypatch):
    from papervault.store.knowledge_runtime import assert_knowledge_authority, knowledge_jobs_mode

    monkeypatch.delenv("PAPERVAULT_KNOWLEDGE_JOBS", raising=False)
    monkeypatch.delenv("PAPERVAULT_CATALOG_DSN", raising=False)
    assert knowledge_jobs_mode() == "legacy"
    assert assert_knowledge_authority("legacy")["authority"] == "legacy"


def test_postgres_selector_requires_explicit_catalog_connection(monkeypatch):
    from papervault.store.knowledge_runtime import assert_knowledge_authority

    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "postgres")
    monkeypatch.delenv("PAPERVAULT_CATALOG_DSN", raising=False)
    with pytest.raises(ValueError, match="PAPERVAULT_CATALOG_DSN"):
        assert_knowledge_authority("catalog")


@pytest.mark.parametrize("mode,authority", [("legacy", "catalog"), ("postgres", "legacy")])
def test_flag_cannot_override_committed_authority(monkeypatch, mode, authority):
    from papervault.store import knowledge_runtime as runtime

    class Repository:
        def get_authority_state(self):
            return {"authority": authority, "authority_epoch": 7}

    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", mode)
    monkeypatch.setenv("PAPERVAULT_CATALOG_DSN", "postgresql://test.invalid/task_test")
    monkeypatch.setattr(runtime, "get_knowledge_repository", lambda: Repository())
    with pytest.raises(runtime.KnowledgeAuthorityError):
        runtime.assert_knowledge_authority("catalog" if mode == "postgres" else "legacy")


def test_catalog_epoch_rejects_json_library_configuration(monkeypatch):
    from papervault.store import knowledge_runtime as runtime

    class Repository:
        def get_authority_state(self):
            return {"authority": "catalog", "authority_epoch": 7}

    monkeypatch.setenv("PAPERVAULT_KNOWLEDGE_JOBS", "postgres")
    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "json")
    monkeypatch.setenv("PAPERVAULT_CATALOG_DSN", "postgresql://test.invalid/task_test")
    monkeypatch.setattr(runtime, "get_knowledge_repository", lambda: Repository())
    with pytest.raises(runtime.KnowledgeAuthorityError, match="postgres"):
        runtime.assert_knowledge_authority("catalog")
