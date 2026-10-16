"""HTTP-boundary tests for consumer-side MinerU readiness; no live service is used."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from papervault.library.mineru_client import MineruTransportError
from papervault.library.services import mineru_server as ms
from papervault.library.services.mineru_server import (
    MineruServerController,
    MineruServerUnavailable,
)


@pytest.fixture(autouse=True)
def forbid_lifecycle_subprocesses(monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail(f"readiness must not invoke a subprocess: {args}")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected)


def _http(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(handler), **kwargs))


def test_stopped_backend_is_ready_without_model_probe(monkeypatch):
    requests = []

    def handler(request):
        requests.append((request.method, str(request.url)))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "backend": "stopped"})
        # Probes other than health fail while stopped and do not start the backend.
        return httpx.Response(503)

    _http(monkeypatch, handler)
    asyncio.run(MineruServerController().ensure_ready())
    assert requests == [("GET", "http://127.0.0.1:30000/health")]


@pytest.mark.parametrize("health_status", [503, 302])
async def test_unready_responses_raise_transport_error(monkeypatch, health_status):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/healthy":
            return httpx.Response(200)
        return httpx.Response(health_status, headers={"Location": "/healthy"})

    _http(monkeypatch, handler)
    with pytest.raises(MineruServerUnavailable) as exc:
        await MineruServerController().ensure_ready()
    assert isinstance(exc.value, MineruTransportError)
    assert paths == ["/health"]


@pytest.mark.parametrize("body", ["not JSON", ""])
async def test_health_200_does_not_require_json_body(monkeypatch, body):
    def handler(request):
        return httpx.Response(200, text=body)

    _http(monkeypatch, handler)
    await MineruServerController().ensure_ready()


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_http_failure_is_unavailable(monkeypatch, error_type):
    def handler(request):
        raise error_type("unreachable", request=request)

    _http(monkeypatch, handler)
    with pytest.raises(MineruServerUnavailable):
        await MineruServerController().ensure_ready()


async def test_entire_probe_has_a_bounded_budget(monkeypatch):
    monkeypatch.setattr(ms, "_HEALTH_HTTP_TIMEOUT", 0.01)

    async def handler(request):
        await asyncio.Event().wait()

    _http(monkeypatch, handler)
    with pytest.raises(MineruServerUnavailable):
        await asyncio.wait_for(MineruServerController().ensure_ready(), timeout=0.5)


async def test_each_call_probes_once_and_recovers_without_cooldown(monkeypatch):
    paths = []
    healthy = False

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200 if healthy else 503)

    _http(monkeypatch, handler)
    controller = MineruServerController()
    for _ in range(2):
        with pytest.raises(MineruServerUnavailable):
            await controller.ensure_ready()
    healthy = True
    await controller.ensure_ready()
    assert paths == ["/health", "/health", "/health"]


@pytest.mark.parametrize("override,want", [
    (None, "http://ocr.example:31000/prefix"),
    ("http://explicit.example:32000/", "http://explicit.example:32000"),
])
async def test_endpoint_resolution_keeps_first_env_endpoint(monkeypatch, override, want):
    monkeypatch.setenv("MINERU_URL", "http://ocr.example:31000/prefix/,http://other.example:31000")
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200)

    _http(monkeypatch, handler)
    await MineruServerController(base_url=override).ensure_ready()
    assert urls == [f"{want}/health"]


async def test_cancellation_propagates(monkeypatch):
    entered = asyncio.Event()

    async def handler(request):
        entered.set()
        await asyncio.Event().wait()

    _http(monkeypatch, handler)
    task = asyncio.create_task(MineruServerController().ensure_ready())
    await asyncio.wait_for(entered.wait(), timeout=0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("ready", [True, False])
async def test_operator_session_probes_before_ocr(monkeypatch, ready):
    def handler(request):
        return httpx.Response(200 if ready else 503, json={"backend": "stopped"})

    _http(monkeypatch, handler)
    entered = False
    try:
        async with MineruServerController().ocr_session():
            entered = True
    except MineruServerUnavailable:
        assert not ready
    assert entered is ready
