"""Keep library unit tests independent of live OCR and systemd services."""

import subprocess

import pytest

from papervault import ops_guards
from papervault.library.services import mineru_server as ms


@pytest.fixture(autouse=True)
def isolate_mineru_server(monkeypatch):
    # Extraction tests fake the OCR call. Give that consumer a ready server;
    # readiness tests create their own instances and exercise the HTTP probes.
    controller = ms.MineruServerController()

    async def ready():
        return True

    controller._ready = ready
    monkeypatch.setattr(ms, "_controller", controller)


@pytest.fixture(autouse=True)
def isolate_live_systemd(monkeypatch):
    # Mutating CLI unit tests run on hosts where papervault.service may be live.
    # Default only systemctl probes to inactive (passing all other subprocesses
    # through untouched); tests exercising active/error states override
    # ops_guards.subprocess.run explicitly.
    real_run = subprocess.run

    def _inactive_systemctl(argv, *args, **kwargs):
        if isinstance(argv, (list, tuple)) and argv and argv[0] == "systemctl":
            return subprocess.CompletedProcess(argv, 3, stdout="inactive\n", stderr="")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(ops_guards.subprocess, "run", _inactive_systemctl)
