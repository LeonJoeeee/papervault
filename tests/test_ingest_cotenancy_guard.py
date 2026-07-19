"""Co-tenancy gate for `papervault ingest-doc` (PR #54 blocker).

Operator-doc ingest is a maintenance-window WRITE to the same ledger + LightRAG graph the
live service serves, so it must refuse while the service is systemd-active — reusing the
SHARED gate (papervault.ops_guards) that the eval also uses (issue #32). These mirror
tests/test_eval_cotenancy_guard.py but pin the INGEST override env + the CLI wiring.
"""
import subprocess
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import papervault.cli as cli_mod
from papervault import ops_guards

INGEST_OVERRIDE = "KS_INGEST_ALLOW_COTENANCY"


def _fake_run(stdout):
    def fake(cmd, capture_output=True, text=True, timeout=10):
        return SimpleNamespace(stdout=stdout, returncode=0)
    return fake


# --------------------------------------------------------------------------- #
#  shared gate, driven with the INGEST override env                            #
# --------------------------------------------------------------------------- #

def _reason() -> str:
    return "operator-doc ingest is a maintenance-window write — DB co-write corruption risk."


def test_ingest_gate_aborts_when_service_active(monkeypatch):
    monkeypatch.delenv(INGEST_OVERRIDE, raising=False)
    monkeypatch.setattr(subprocess, "run", _fake_run("active\n"))
    with pytest.raises(SystemExit):
        ops_guards.require_services_stopped(override_env=INGEST_OVERRIDE, reason=_reason())


def test_ingest_gate_passes_when_inactive(monkeypatch):
    monkeypatch.delenv(INGEST_OVERRIDE, raising=False)
    monkeypatch.setattr(subprocess, "run", _fake_run("inactive\n"))
    ops_guards.require_services_stopped(override_env=INGEST_OVERRIDE, reason=_reason())  # no raise


def test_ingest_override_env_bypasses_without_probing(monkeypatch):
    monkeypatch.setenv(INGEST_OVERRIDE, "1")

    def boom(*a, **k):
        raise AssertionError("must not even check systemd under override")

    monkeypatch.setattr(subprocess, "run", boom)
    ops_guards.require_services_stopped(override_env=INGEST_OVERRIDE, reason=_reason())  # no raise


def test_ingest_gate_no_systemd_is_permissive(monkeypatch):
    monkeypatch.delenv(INGEST_OVERRIDE, raising=False)

    def no_systemd(*a, **k):
        raise FileNotFoundError("systemctl not found")

    monkeypatch.setattr(subprocess, "run", no_systemd)
    ops_guards.require_services_stopped(override_env=INGEST_OVERRIDE, reason=_reason())  # no raise


def test_ingest_gate_mineru_unit_active_caught(monkeypatch):
    monkeypatch.delenv(INGEST_OVERRIDE, raising=False)
    calls = []

    def fake(cmd, capture_output=True, text=True, timeout=10):
        calls.append(cmd[-1])
        return SimpleNamespace(stdout="inactive\n" if len(calls) == 1 else "activating\n")

    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(SystemExit):
        ops_guards.require_services_stopped(override_env=INGEST_OVERRIDE, reason=_reason())
    assert len(calls) == 2  # reached the MinerU unit


# --------------------------------------------------------------------------- #
#  CLI wiring — `papervault ingest-doc` honors the gate before booting graph   #
# --------------------------------------------------------------------------- #

def _invoke(tmp_path, *, env):
    p = tmp_path / "book.md"
    p.write_text("# A\nalpha beta gamma", encoding="utf-8")
    return CliRunner().invoke(
        cli_mod.main,
        ["ingest-doc", str(p), "--kind", "textbook", "--key", "textbook:Schlickeiser2002"],
        env=env,
    )


def test_cli_ingest_doc_refuses_when_service_active(monkeypatch, tmp_path):
    # source guard ON so we reach the co-tenancy gate; service reads active → refuse.
    monkeypatch.setattr(subprocess, "run", _fake_run("active\n"))
    res = _invoke(tmp_path, env={"PAPERVAULT_OPERATOR_SOURCES": "1", "KS_INGEST_ALLOW_COTENANCY": ""})
    assert res.exit_code != 0
    combined = (res.stdout or "") + (res.stderr or "")
    assert "RUNNING" in combined and "papervault.service" in combined


def test_cli_ingest_doc_override_bypasses_gate(monkeypatch, tmp_path):
    # override set → gate does not fire. It then proceeds toward graph boot; stub get_graph
    # to fail fast so we assert the gate was PASSED (not the boot). subprocess must not even
    # be probed under override.
    def boom(*a, **k):
        raise AssertionError("gate must not probe systemd under override")

    monkeypatch.setattr(subprocess, "run", boom)

    async def _explode():
        raise RuntimeError("BOOT-REACHED")

    import papervault.knowledge.store.graph as graph_mod
    monkeypatch.setattr(graph_mod, "get_graph", lambda: _explode())

    res = _invoke(tmp_path, env={"PAPERVAULT_OPERATOR_SOURCES": "1", "KS_INGEST_ALLOW_COTENANCY": "1"})
    # got past the gate (override) into _go(); our stub boot raised → non-zero, but crucially
    # NOT the gate refusal.
    assert res.exit_code != 0
    combined = (res.stdout or "") + (res.stderr or "")
    assert "RUNNING" not in combined  # never hit the co-tenancy refusal
