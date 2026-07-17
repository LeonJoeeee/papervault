"""Pure unit tests for the structural prod-safety guard (SDD §6.0/§6.5).

No DB, no LightRAG, no IO — only env-var logic in assert_safe_workspace().
"""
import pytest

from papervault.knowledge.store.graph import (
    DEV_WORKSPACE,
    PROD_WORKSPACE,
    UnsafeWorkspaceError,
    assert_safe_workspace,
)

_ENV = ("NEO4J_WORKSPACE", "POSTGRES_WORKSPACE", "KS_ALLOW_PROD_WORKSPACE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    yield


def _set(monkeypatch, neo=None, pg=None, allow=None):
    if neo is not None:
        monkeypatch.setenv("NEO4J_WORKSPACE", neo)
    if pg is not None:
        monkeypatch.setenv("POSTGRES_WORKSPACE", pg)
    if allow is not None:
        monkeypatch.setenv("KS_ALLOW_PROD_WORKSPACE", allow)


def test_probe_workspace_both_set_passes(monkeypatch):
    _set(monkeypatch, neo=DEV_WORKSPACE, pg=DEV_WORKSPACE)
    assert assert_safe_workspace() == DEV_WORKSPACE


def test_both_empty_rejected(monkeypatch):
    # nothing set → must NOT silently fall back to base/default next to prod
    with pytest.raises(UnsafeWorkspaceError):
        assert_safe_workspace()


def test_neo_set_pg_missing_rejected(monkeypatch):
    # the exact half-guard hole: NEO4J set to probe, POSTGRES unset (.env would fill it with prod)
    _set(monkeypatch, neo=DEV_WORKSPACE)
    with pytest.raises(UnsafeWorkspaceError):
        assert_safe_workspace()


def test_pg_set_neo_missing_rejected(monkeypatch):
    _set(monkeypatch, pg=DEV_WORKSPACE)
    with pytest.raises(UnsafeWorkspaceError):
        assert_safe_workspace()


def test_mismatch_rejected(monkeypatch):
    # split-brain: graph -> probe, PG -> prod
    _set(monkeypatch, neo=DEV_WORKSPACE, pg=PROD_WORKSPACE)
    with pytest.raises(UnsafeWorkspaceError):
        assert_safe_workspace()


def test_prod_without_optin_rejected(monkeypatch):
    _set(monkeypatch, neo=PROD_WORKSPACE, pg=PROD_WORKSPACE)
    with pytest.raises(UnsafeWorkspaceError):
        assert_safe_workspace()


def test_prod_with_explicit_optin_allowed(monkeypatch):
    _set(monkeypatch, neo=PROD_WORKSPACE, pg=PROD_WORKSPACE, allow="1")
    assert assert_safe_workspace() == PROD_WORKSPACE


def test_prod_optin_must_be_exactly_1(monkeypatch):
    # a truthy-looking but non-"1" value must NOT open prod
    _set(monkeypatch, neo=PROD_WORKSPACE, pg=PROD_WORKSPACE, allow="true")
    with pytest.raises(UnsafeWorkspaceError):
        assert_safe_workspace()


def test_whitespace_treated_as_empty(monkeypatch):
    _set(monkeypatch, neo="  ", pg=DEV_WORKSPACE)
    with pytest.raises(UnsafeWorkspaceError):
        assert_safe_workspace()
