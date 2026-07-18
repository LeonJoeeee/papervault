"""Unit tests for the eval co-tenancy guard (issue #32)."""
import subprocess
from types import SimpleNamespace

import pytest

from papervault.eval import run_eval


def _fake_run(stdout):
    def fake(cmd, capture_output=True, text=True, timeout=10):
        return SimpleNamespace(stdout=stdout, returncode=0)
    return fake


def test_aborts_when_service_active(monkeypatch):
    monkeypatch.delenv("KS_EVAL_ALLOW_COTENANCY", raising=False)
    monkeypatch.setattr(subprocess, "run", _fake_run("active\n"))
    with pytest.raises(SystemExit):
        run_eval._require_exclusive_gpu()


def test_passes_when_services_inactive(monkeypatch):
    monkeypatch.delenv("KS_EVAL_ALLOW_COTENANCY", raising=False)
    monkeypatch.setattr(subprocess, "run", _fake_run("inactive\n"))
    run_eval._require_exclusive_gpu()  # no raise


def test_override_env_bypasses(monkeypatch):
    monkeypatch.setenv("KS_EVAL_ALLOW_COTENANCY", "1")
    def boom(*a, **k):
        raise AssertionError("must not even check systemd under override")
    monkeypatch.setattr(subprocess, "run", boom)
    run_eval._require_exclusive_gpu()  # no raise


def test_no_systemd_environment_is_permissive(monkeypatch):
    monkeypatch.delenv("KS_EVAL_ALLOW_COTENANCY", raising=False)
    def no_systemd(*a, **k):
        raise FileNotFoundError("systemctl not found")
    monkeypatch.setattr(subprocess, "run", no_systemd)
    run_eval._require_exclusive_gpu()  # no raise (CI/container)
