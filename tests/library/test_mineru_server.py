"""MinerU compatibility hooks never probe HTTP or manage the independent service."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from papervault.library.mineru_client import MineruTransportError
from papervault.library.services.mineru_server import (
    MineruServerController,
    MineruServerUnavailable,
)


@pytest.fixture(autouse=True)
def forbid_preflight_and_lifecycle_operations(monkeypatch):
    def unexpected_http(*args, **kwargs):
        pytest.fail("readiness must not construct an HTTP client")

    async def unexpected_subprocess(*args, **kwargs):
        pytest.fail("readiness must not invoke a subprocess")

    monkeypatch.setattr(httpx, "AsyncClient", unexpected_http)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_subprocess)


@pytest.mark.parametrize("base_url", [None, "http://unreachable.invalid:32000/"])
async def test_ensure_ready_returns_none_without_http(monkeypatch, base_url):
    monkeypatch.setenv("MINERU_URL", "http://unreachable.invalid:30000")
    controller = MineruServerController(base_url=base_url)
    assert await controller.ensure_ready() is None
    assert await controller.ensure_ready() is None


async def test_operator_session_admits_ocr_without_http():
    async with MineruServerController().ocr_session():
        assert await MineruServerController().ensure_ready() is None


async def test_operator_session_propagates_workload_transport_failure():
    with pytest.raises(MineruServerUnavailable):
        async with MineruServerController().ocr_session():
            raise MineruServerUnavailable("client discovery failed")
    assert issubclass(MineruServerUnavailable, MineruTransportError)
