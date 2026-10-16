"""Keep library unit tests independent of the live OCR server."""

import pytest

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
