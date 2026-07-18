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


def test_second_unit_active_still_caught(monkeypatch):
    monkeypatch.delenv("KS_EVAL_ALLOW_COTENANCY", raising=False)
    calls = []
    def fake(cmd, capture_output=True, text=True, timeout=10):
        calls.append(cmd[-1])
        return SimpleNamespace(stdout="inactive\n" if len(calls) == 1 else "activating\n")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(SystemExit):
        run_eval._require_exclusive_gpu()
    assert len(calls) == 2  # loop reached the second unit


def test_positive_detection_survives_later_probe_error(monkeypatch):
    # Review finding 2: unit1 active + unit2 probe throwing must STILL abort.
    monkeypatch.delenv("KS_EVAL_ALLOW_COTENANCY", raising=False)
    calls = []
    def fake(cmd, capture_output=True, text=True, timeout=10):
        calls.append(cmd[-1])
        if len(calls) == 1:
            return SimpleNamespace(stdout="active\n")
        raise subprocess.TimeoutExpired(cmd, timeout)
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(SystemExit):
        run_eval._require_exclusive_gpu()


def test_mineru_unit_name_honors_operator_override(monkeypatch):
    monkeypatch.delenv("KS_EVAL_ALLOW_COTENANCY", raising=False)
    monkeypatch.setenv("PAPER_LIBRARY_MINERU_UNIT", "custom-mineru.service")
    probed = []
    def fake(cmd, capture_output=True, text=True, timeout=10):
        probed.append(cmd[-1])
        return SimpleNamespace(stdout="inactive\n")
    monkeypatch.setattr(subprocess, "run", fake)
    run_eval._require_exclusive_gpu()
    assert "custom-mineru.service" in probed
