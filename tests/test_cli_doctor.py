"""Doctor checks shared OCR endpoint health without importing MinerU locally."""

import builtins
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.error import URLError
import urllib.request

from click.testing import CliRunner
import pytest

from papervault import cli


@pytest.fixture
def doctor(monkeypatch):
    for name in ("_check_config", "_check_paths", "_check_workspace", "_check_domain", "_check_gpu"):
        monkeypatch.setattr(cli, name, lambda report: None)
    monkeypatch.setattr(cli, "_model_available", lambda spec: True)
    original = builtins.__import__
    imports = []

    def without_mineru(name, *args, **kwargs):
        if name == "mineru" or name.startswith(("mineru.", "mineru_vl_utils")):
            imports.append(name)
            raise ImportError("MinerU belongs to the shared service")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mineru)
    return imports


@pytest.mark.parametrize("base", ["http://127.0.0.1:30000", "http://ocr:30000/api/"])
def test_doctor_reports_health_without_local_mineru(doctor, monkeypatch, base):
    monkeypatch.setenv("MINERU_URL", f"{base},http://unused")
    requested = []

    @contextmanager
    def health(url, timeout):
        requested.append((url, timeout))
        yield SimpleNamespace(status=200)

    monkeypatch.setattr(urllib.request, "urlopen", health)
    result = CliRunner().invoke(cli.main, ["doctor", "--skip-db"])
    assert result.exit_code == 0, result.output
    assert "✓ mineru" in result.output
    assert "OK — 0 warning(s)" in result.output
    assert requested == [(f"{base.rstrip('/')}/health", 3)]
    assert doctor == []


@pytest.mark.parametrize("failure", [503, URLError("connection refused")])
def test_doctor_warns_when_shared_ocr_is_unavailable(doctor, monkeypatch, failure):
    @contextmanager
    def health(url, timeout):
        if isinstance(failure, Exception):
            raise failure
        yield SimpleNamespace(status=failure)

    monkeypatch.setattr(urllib.request, "urlopen", health)
    result = CliRunner().invoke(cli.main, ["doctor", "--skip-db"])
    assert result.exit_code == 0, result.output
    assert "! mineru" in result.output
    assert "OK — 1 warning(s)" in result.output
    assert doctor == []
