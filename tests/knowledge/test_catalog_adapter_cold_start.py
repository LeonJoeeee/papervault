"""Runtime catalog pool acquisition is read-only even on the first connection."""
import asyncio

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared", [True, False])
async def test_cold_catalog_pool_never_bootstraps_extensions_or_runs_legacy_migrations(monkeypatch, prepared):
    from lightrag.kg import postgres_impl as backend
    from papervault.knowledge.store.catalog_storage import acquire_catalog_client

    config = backend.ClientManager.get_config(vector_storage="PGVectorStorage")
    config.update(user="fixture", password="fixture", database="catalog_backend_fixture",
                  connection_retry_attempts=1, connection_retry_backoff=0,
                  connection_retry_backoff_max=0, vector_index_type="HNSW")
    statements = []
    pools = []

    class Connection:
        async def execute(self, sql, *args):
            statements.append(sql)
            raise AssertionError("runtime initialization must not issue DDL or migrations")

        async def fetchrow(self, sql, *args):
            statements.append(sql)
            assert sql.lstrip().upper().startswith("SELECT")
            return {"extversion": "0.8.0"} if prepared else None

        async def close(self):
            pass

    class Pool:
        closed = False

        async def close(self):
            self.closed = True

    async def connect(**kwargs):
        return Connection()

    async def create_pool(**kwargs):
        await kwargs["init"](Connection())
        pool = Pool()
        pools.append(pool)
        return pool

    async def register_vector(connection):
        pass  # Python codec installation, not a database schema operation.

    async def forbidden_check_tables(self):
        raise AssertionError("runtime acquisition must not call legacy check_tables")

    monkeypatch.setattr(backend.ClientManager, "_instances", {"db": None, "ref_count": 0, "vector_signature": None})
    monkeypatch.setattr(backend.ClientManager, "_lock", asyncio.Lock())
    monkeypatch.setattr(backend.ClientManager, "get_config", lambda **kwargs: config)
    monkeypatch.setattr(backend.asyncpg, "connect", connect)
    monkeypatch.setattr(backend.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(backend, "register_vector", register_vector)
    monkeypatch.setattr(backend.PostgreSQLDB, "check_tables", forbidden_check_tables)

    if not prepared:
        with pytest.raises(RuntimeError, match="prepared|extension"):
            await acquire_catalog_client()
        assert not pools
        assert backend.ClientManager._instances["db"] is None
    else:
        first = await acquire_catalog_client()
        second = await acquire_catalog_client()
        assert first is second
        assert backend.ClientManager._instances["ref_count"] == 2
        await backend.ClientManager.release_client(first)
        assert not pools[0].closed
        await backend.ClientManager.release_client(second)
        assert pools[0].closed
        assert backend.ClientManager._instances["db"] is None
    assert statements
    assert all(sql.lstrip().upper().startswith("SELECT") for sql in statements)


@pytest.mark.asyncio
@pytest.mark.parametrize("read_only", [True, False])
@pytest.mark.parametrize("prepared", [True, False])
async def test_storage_runtime_requires_prepared_namespace_schema_without_legacy_acquisition(monkeypatch, read_only, prepared):
    from papervault.knowledge.store import catalog_storage
    from tests.knowledge.test_catalog_adapter_storage import Gate

    statements = []
    released = []

    class DB:
        async def query(self, sql, params=None, **kwargs):
            statements.append(sql)
            assert sql.lstrip().upper().startswith("SELECT")
            return {"prepared": prepared}

    db = DB()

    async def acquire():
        return db

    async def legacy(**kwargs):
        raise AssertionError("catalog storage must not use upstream schema-initializing acquisition")

    async def release(value):
        released.append(value)

    monkeypatch.setattr(catalog_storage, "acquire_catalog_client", acquire)
    monkeypatch.setattr(catalog_storage.ClientManager, "get_client", legacy)
    monkeypatch.setattr(catalog_storage.ClientManager, "release_client", release)
    with catalog_storage.storage_binding(Gate(), {"namespace": "private"}, read_only=read_only):
        storage = catalog_storage.CatalogPGKVStorage(namespace="full_docs", workspace="private",
                                                     global_config={}, embedding_func=None)
    if prepared:
        await storage.initialize()
        await storage.finalize()
    else:
        with pytest.raises(catalog_storage.BackendNotPrepared, match="prepared|schema"):
            await storage.initialize()
    assert statements
    assert released == [db]
