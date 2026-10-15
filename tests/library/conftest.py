"""Keep library unit tests independent of the live OCR server and user bus."""

import pytest

from papervault.library.services import mineru_server as ms


@pytest.fixture(autouse=True)
def isolate_mineru_server(monkeypatch):
    # Extraction tests fake the OCR call. Give that consumer a ready server;
    # controller tests create their own instances and exercise real lifecycle code.
    controller = ms.MineruServerController(watchdog=False)

    async def ready():
        return True

    async def systemctl(verb):
        pytest.fail(f"unit test attempted systemctl {verb} through the shared controller")

    controller._ready = ready
    controller._run_systemctl = systemctl
    monkeypatch.setattr(ms, "_controller", controller)
