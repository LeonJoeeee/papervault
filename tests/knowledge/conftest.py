"""pytest prod-safety gate (SDD §6.0/§6.5).

铁律(1)「绝不碰生产 l0」从程序性升级为结构性:bare `pytest` against the prod-pinned
`.env` (NEO4J_WORKSPACE=POSTGRES_WORKSPACE=l0) must NOT silently connect to the
production Postgres and mutate the shared `ks_ledger` / dead `chunk_metadata` tables.

Pure-logic tests (fingerprint / reconcile diff / monkeypatched query) never
open a DB pool, so they keep passing with no env set. Tests that DO open a real pool
(test_ledger) are gated: the very first pool open asserts a safe
workspace (l0_probe) or an explicit prod opt-in (KS_ALLOW_PROD_WORKSPACE=1) — same gate
as get_graph()'s assert_safe_workspace(). Otherwise the pool open raises and the DB test
fails loudly instead of corrupting prod.

NOTE: ks_ledger is now workspace-isolated (SDD §4.1: a `workspace` column + per-workspace
filtering on every CRUD entry, workspace sourced from assert_safe_workspace()). So the
workspace env is BOTH a prod-vs-probe prod-safety signal AND the ledger's filter dimension
— an l0_probe pool can no longer read/delete prod 'l0' ledger rows even within the single
shared POSTGRES_DB=papervault.knowledge table. (chunk_metadata is dead v2, removed.)
"""
from __future__ import annotations

import pytest

from papervault.knowledge.store.graph import assert_safe_workspace


@pytest.fixture(autouse=True)
def _prod_safety_gate(monkeypatch):
    """Wrap the DB pool factories so the first real pool open asserts a safe workspace.

    Pure tests that never call get_pool() are unaffected.
    """
    import papervault.knowledge.ledger.store as ledger_store

    _orig_ledger_get_pool = ledger_store.get_pool

    async def _guarded_ledger_get_pool():
        assert_safe_workspace()
        return await _orig_ledger_get_pool()

    monkeypatch.setattr(ledger_store, "get_pool", _guarded_ledger_get_pool)
    # (The v2 sidecar.crud pool guard was removed with the dead-v2 sweep — ledger.store is
    # now the only real-pool entry, so this one wrap is the whole gate.)

    yield
