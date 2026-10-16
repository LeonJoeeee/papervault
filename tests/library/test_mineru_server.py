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


_MODELS = {"object": "list", "data": [{"id": "MinerU2.5-Pro-2605-1.2B", "object": "model"}]}


@pytest.fixture(autouse=True)
def forbid_lifecycle_subprocesses(monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail(f"readiness must not invoke a subprocess: {args}")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected)


def _http(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(handler), **kwargs))


def test_sleeping_backend_is_ready_without_waking_or_inspecting_sleep(monkeypatch):
    requests = []

    def handler(request):
        requests.append((request.method, str(request.url)))
        # The gate returns these same observations while the backend is asleep.
        return httpx.Response(200, json=_MODELS if request.url.path == "/v1/models" else {})

    _http(monkeypatch, handler)
    asyncio.run(MineruServerController().ensure_ready())
    assert requests == [("GET", "http://127.0.0.1:30000/health"),
                        ("GET", "http://127.0.0.1:30000/v1/models")]


@pytest.mark.parametrize("health_status,models_status,payload", [
    (503, 200, _MODELS),
    (200, 503, _MODELS),
    (302, 200, _MODELS),
    (200, 302, _MODELS),
    (200, 200, {"data": []}),
    (200, 200, {}),
    (200, 200, None),
    (200, 200, []),
    (200, 200, {"data": "not a model list"}),
    (200, 200, {"data": [{}]}),
])
async def test_unready_responses_raise_transport_error(
    monkeypatch, health_status, models_status, payload,
):
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(health_status)
        return httpx.Response(models_status, json=payload)

    _http(monkeypatch, handler)
    with pytest.raises(MineruServerUnavailable) as exc:
        await MineruServerController().ensure_ready()
    assert isinstance(exc.value, MineruTransportError)


async def test_invalid_json_is_unavailable(monkeypatch):
    def handler(request):
        return httpx.Response(200, text="not JSON" if request.url.path == "/v1/models" else "")

    _http(monkeypatch, handler)
    with pytest.raises(MineruServerUnavailable):
        await MineruServerController().ensure_ready()


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_http_failure_is_unavailable(monkeypatch, error_type):
    def handler(request):
        raise error_type("unreachable", request=request)

    _http(monkeypatch, handler)
    with pytest.raises(MineruServerUnavailable):
        await MineruServerController().ensure_ready()


@pytest.mark.parametrize("blocked_path", ["/health", "/v1/models"])
async def test_entire_probe_has_a_bounded_budget(monkeypatch, blocked_path):
    monkeypatch.setattr(ms, "_HEALTH_HTTP_TIMEOUT", 0.01)

    async def handler(request):
        if request.url.path == blocked_path:
            await asyncio.Event().wait()
        return httpx.Response(200)

    _http(monkeypatch, handler)
    with pytest.raises(MineruServerUnavailable):
        await asyncio.wait_for(MineruServerController().ensure_ready(), timeout=0.5)


async def test_each_call_probes_once_and_recovers_without_cooldown(monkeypatch):
    paths = []
    healthy = False

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200 if healthy else 503, json=_MODELS)

    _http(monkeypatch, handler)
    controller = MineruServerController()
    for _ in range(2):
        with pytest.raises(MineruServerUnavailable):
            await controller.ensure_ready()
    healthy = True
    await controller.ensure_ready()
    assert paths == ["/health", "/health", "/health", "/v1/models"]


@pytest.mark.parametrize("override,want", [
    (None, "http://ocr.example:31000/prefix"),
    ("http://explicit.example:32000/", "http://explicit.example:32000"),
])
async def test_endpoint_resolution_keeps_first_env_endpoint(monkeypatch, override, want):
    monkeypatch.setenv("MINERU_URL", "http://ocr.example:31000/prefix/,http://other.example:31000")
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_MODELS)

    _http(monkeypatch, handler)
    await MineruServerController(base_url=override).ensure_ready()
    assert urls == [f"{want}/health", f"{want}/v1/models"]


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
        return httpx.Response(200 if ready else 503, json=_MODELS)

    _http(monkeypatch, handler)
    entered = False
    try:
        async with MineruServerController().ocr_session():
            entered = True
    except MineruServerUnavailable:
        assert not ready
    assert entered is ready
