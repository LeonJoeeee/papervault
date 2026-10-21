"""Explicit catalog connections and fail-closed knowledge authority selection.

The selector chooses a compatible implementation. Only the committed catalog
epoch grants authority. A connection belongs to one command or namespace gate;
concurrent scheduler, serving and operator tasks never share transactions.
"""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

import psycopg

from .repository import PostgresCatalogRepository


class KnowledgeAuthorityError(RuntimeError):
    """Configuration cannot write under the committed knowledge authority."""


def knowledge_jobs_mode() -> str:
    value = os.environ.get("PAPERVAULT_KNOWLEDGE_JOBS", "legacy").strip().lower()
    if value not in {"legacy", "postgres"}:
        raise ValueError("PAPERVAULT_KNOWLEDGE_JOBS must be legacy or postgres")
    return value


class KnowledgeRepositoryProxy:
    """Open explicit connections; retain one only while a namespace gate owns it."""

    def __init__(self, dsn: str, *, schema: str = "catalog", factory=None):
        if not dsn:
            raise ValueError("knowledge catalog requires PAPERVAULT_CATALOG_DSN")
        self.dsn, self.schema, self.factory = dsn, schema, factory
        self._active: ContextVar = ContextVar("knowledge_repository", default=None)

    @contextmanager
    def repository(self):
        current = self._active.get()
        if current is not None and current[0] == self._owner():
            yield current[1]
            return
        factory = self.factory
        if factory is None:
            from .knowledge import KnowledgeRepository

            factory = KnowledgeRepository
        with psycopg.connect(self.dsn, autocommit=True, connect_timeout=5) as connection:
            yield factory(PostgresCatalogRepository(connection, schema=self.schema))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def invoke(*args, **kwargs):
            with self.repository() as repository:
                return getattr(repository, name)(*args, **kwargs)

        return invoke

    @asynccontextmanager
    async def namespace_access(self, claim: dict, seal: bool = False):
        with self.repository() as repository:
            token = self._active.set((self._owner(), repository))
            try:
                async with repository.namespace_access(claim, seal=seal):
                    yield repository
            finally:
                self._active.reset(token)

    @staticmethod
    def _owner():
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return threading.get_ident(), task


def get_knowledge_repository() -> KnowledgeRepositoryProxy:
    return KnowledgeRepositoryProxy(
        os.environ.get("PAPERVAULT_CATALOG_DSN", ""),
        schema=os.environ.get("PAPERVAULT_CATALOG_SCHEMA", "catalog"),
    )


def assert_knowledge_authority(expected: str) -> dict:
    """Reject incompatible startup and stale legacy entry points, including reads.

    Existing installations with no catalog connection retain the legacy default.
    Once a catalog DSN is configured its epoch is always checked, including when
    the operator selects the legacy implementation. Database failures propagate.
    """
    if expected not in {"legacy", "catalog"}:
        raise ValueError("expected knowledge authority must be legacy or catalog")
    mode = knowledge_jobs_mode()
    if not os.environ.get("PAPERVAULT_CATALOG_DSN"):
        if mode == "postgres" or expected == "catalog":
            raise ValueError("postgres knowledge jobs require PAPERVAULT_CATALOG_DSN")
        return {"authority": "legacy", "authority_epoch": 0}
    state = get_knowledge_repository().get_authority_state()
    selected = "catalog" if mode == "postgres" else "legacy"
    if state["authority"] != selected or state["authority"] != expected:
        raise KnowledgeAuthorityError(
            f"knowledge configuration selects {selected}, committed authority is "
            f"{state['authority']} at epoch {state['authority_epoch']}; "
            "a configuration flag cannot transfer authority"
        )
    if selected == "catalog" and os.environ.get("PAPERVAULT_CATALOG_AUTHORITY") != "postgres":
        raise KnowledgeAuthorityError("catalog knowledge authority requires postgres library authority")
    return state
