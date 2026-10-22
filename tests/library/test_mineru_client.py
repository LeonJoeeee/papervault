"""HTTP task protocol, C1 error classification and resource hygiene without live OCR."""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import time
from email import policy
from email.parser import BytesParser
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from papervault.library import mineru_client as mc


@pytest.fixture
def server(monkeypatch):
    """Use real httpx clients and serialization, replacing only the network."""
    real_client = httpx.AsyncClient
    state = SimpleNamespace(requests=[], clients=[], options=[])
    monkeypatch.setattr(mc, "_INLINE_RETRIES", 0)
    monkeypatch.setattr(mc, "_BACKOFF_BASE", 0)
    monkeypatch.setattr(mc, "_TASK_POLL_INTERVAL", 0, raising=False)

    def install(handler):
        async def record(request):
            state.requests.append(request)
            result = handler(request)
            return await result if inspect.isawaitable(result) else result

        def client(**kwargs):
            state.options.append(kwargs)
            obj = real_client(transport=httpx.MockTransport(record), **kwargs)
            state.clients.append(obj)
            return obj

        monkeypatch.setattr(httpx, "AsyncClient", client)
        return state

    return install


def scripted(server, steps):
    remaining = list(steps)

    def handle(request):
        assert remaining, f"unexpected request: {request.method} {request.url}"
        method, path, result = remaining.pop(0)
        assert (request.method, request.url.path) == (method, path)
        if isinstance(result, Exception):
            raise result
        return result

    return server(handle)


def submitted(status="pending", **kwargs):
    return httpx.Response(202, json={"task_id": "t1", "status": status, **kwargs})


def completed():
    return httpx.Response(200, json={"task_id": "t1", "status": "completed"})


def markdown(results=None):
    return httpx.Response(200, json={"results": (
        {"x": {"md_content": "# body"}} if results is None else results)})


def run(**kwargs):
    return asyncio.run(mc.extract_mineru(
        b"%PDF-1.4 body", [mc.Endpoint("a", "http://a:30000/")], stem="x", **kwargs))


def test_task_flow_submits_pdf_and_polls_without_health_preflight(server, monkeypatch):
    # A local MinerU install must never be needed by this HTTP consumer.
    monkeypatch.setitem(sys.modules, "mineru", None)
    state = scripted(server, [
        ("POST", "/tasks", submitted(queued_ahead=2)),
        ("GET", "/tasks/t1", httpx.Response(200, json={"status": "pending", "queued_ahead": 1})),
        ("GET", "/tasks/t1", httpx.Response(200, json={"status": "processing"})),
        ("GET", "/tasks/t1", completed()),
        ("GET", "/tasks/t1/result", markdown()),
    ])
    assert run() == "# body"
    request = state.requests[0]
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content)
    parts = {part.get_param("name", header="content-disposition"): part
             for part in message.iter_parts()}
    pdf = parts.pop("files")
    assert pdf.get_filename() == "x.pdf"
    assert pdf.get_content_type() == "application/pdf"
    assert pdf.get_payload(decode=True) == b"%PDF-1.4 body"
    assert {name: part.get_content() for name, part in parts.items()} == {
        "backend": "vlm-engine", "lang_list": "en", "formula_enable": "true",
        "table_enable": "true", "image_analysis": "false", "return_md": "true",
        "return_images": "false",
    }
    assert len(state.requests) == 5
    assert all(client.is_closed for client in state.clients)


@pytest.mark.parametrize("results,want", [
    ({"normalized_x": {"md_content": "# normalized"}}, "# normalized"),
    ({"x": {"md_content": "# selected"}, "other": {"md_content": "wrong"}}, "# selected"),
])
def test_result_stem_lookup_and_single_entry_fallback(server, results, want):
    scripted(server, [("POST", "/tasks", submitted()), ("GET", "/tasks/t1", completed()),
                      ("GET", "/tasks/t1/result", markdown(results))])
    assert run() == want


@pytest.mark.parametrize("results", [
    {}, {"x": {}}, {"x": {"md_content": None}}, {"x": {"md_content": "  \n"}},
    {"x": {"md_content": 42}}, {"x": []},
    {"other": {"md_content": "wrong"}, "another": {"md_content": "wrong"}},
])
def test_unusable_markdown_is_extraction_error(server, results):
    state = scripted(server, [("POST", "/tasks", submitted()), ("GET", "/tasks/t1", completed()),
                              ("GET", "/tasks/t1/result", markdown(results))])
    with pytest.raises(mc.MineruExtractionError):
        run()
    assert all(client.is_closed for client in state.clients)


def test_missing_results_is_extraction_error(server):
    scripted(server, [("POST", "/tasks", submitted()), ("GET", "/tasks/t1", completed()),
                      ("GET", "/tasks/t1/result", httpx.Response(200, json={}))])
    with pytest.raises(mc.MineruExtractionError):
        run()


@pytest.mark.parametrize("phase", ["submit", "status", "result"])
@pytest.mark.parametrize("code,error", [
    (400, mc.MineruExtractionError), (422, mc.MineruExtractionError),
    (409, mc.MineruExtractionError), (502, mc.MineruTransportError),
    (503, mc.MineruTransportError), (504, mc.MineruTransportError),
    (429, mc.MineruTransportError), (500, mc.MineruTransportError),
])
def test_http_error_classification_at_each_phase(server, phase, code, error):
    steps = [("POST", "/tasks", submitted()), ("GET", "/tasks/t1", completed()),
             ("GET", "/tasks/t1/result", markdown())]
    index = {"submit": 0, "status": 1, "result": 2}[phase]
    method, path, _ = steps[index]
    steps[index] = (method, path, httpx.Response(code, text="plain error"))
    state = scripted(server, steps[:index + 1])
    with pytest.raises(error):
        run()
    assert len(state.requests) == index + 1
    assert all(client.is_closed for client in state.clients)


@pytest.mark.parametrize("payload", [
    {"object": "error", "message": "bad document"},
    {"detail": "cannot render document"}, {"error": {"message": "bad PDF"}},
])
def test_structured_500_is_extraction_error(server, payload):
    scripted(server, [("POST", "/tasks", httpx.Response(500, json=payload))])
    with pytest.raises(mc.MineruExtractionError):
        run()


@pytest.mark.parametrize("payload", [{}, "Internal Server Error", None])
def test_500_without_structured_verdict_is_transport_error(server, payload):
    scripted(server, [("POST", "/tasks", httpx.Response(500, json=payload))])
    with pytest.raises(mc.MineruTransportError):
        run()


@pytest.mark.parametrize("phase", ["submit", "status", "result"])
def test_image_decode_400_retains_transport_subclass(server, monkeypatch, phase):
    monkeypatch.setattr(mc, "_INLINE_RETRIES", 3)
    steps = [("POST", "/tasks", submitted()), ("GET", "/tasks/t1", completed()),
             ("GET", "/tasks/t1/result", markdown())]
    index = {"submit": 0, "status": 1, "result": 2}[phase]
    method, path, _ = steps[index]
    steps[index] = (method, path, httpx.Response(400, json={
        "detail": "Failed to load image: cannot identify image file"}))
    state = scripted(server, steps[:index + 1])
    with pytest.raises(mc.MineruImageDecodeError):
        run()
    assert len(state.requests) == index + 1
    assert issubclass(mc.MineruImageDecodeError, mc.MineruTransportError)


@pytest.mark.parametrize("code", [400, 422, 409])
def test_per_document_error_never_fails_over(server, monkeypatch, code):
    monkeypatch.setattr(mc, "_INLINE_RETRIES", 3)
    state = scripted(server, [("POST", "/tasks", httpx.Response(code, text="bad PDF"))])
    with pytest.raises(mc.MineruExtractionError):
        run()
    assert len(state.requests) == 1


def test_failed_task_fetches_409_result_and_charges_extraction(server):
    scripted(server, [("POST", "/tasks", submitted()),
                      ("GET", "/tasks/t1", httpx.Response(200, json={
                          "status": "failed", "error": "invalid PDF"})),
                      ("GET", "/tasks/t1/result", httpx.Response(409, json={
                          "status": "failed", "error": "invalid PDF"}))])
    with pytest.raises(mc.MineruExtractionError, match="invalid PDF"):
        run()


def test_failed_status_cannot_be_overridden_by_successful_result(server):
    scripted(server, [("POST", "/tasks", submitted("failed", error="invalid PDF")),
                      ("GET", "/tasks/t1/result", markdown())])
    with pytest.raises(mc.MineruExtractionError, match="invalid PDF"):
        run()


@pytest.mark.parametrize("exception", [
    httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
    httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError,
])
def test_failure_before_processing_is_transport_error(server, exception):
    scripted(server, [("POST", "/tasks", submitted(queued_ahead=3)),
                      ("GET", "/tasks/t1", exception("network failure"))])
    with pytest.raises(mc.MineruTransportError):
        run()


@pytest.mark.parametrize("status,error", [
    ("pending", mc.MineruTransportError), ("processing", mc.MineruExtractionError),
])
def test_read_timeout_uses_last_observed_task_status(server, status, error):
    scripted(server, [("POST", "/tasks", submitted()),
                      ("GET", "/tasks/t1", httpx.Response(200, json={
                          "status": status, "queued_ahead": 3 if status == "pending" else 0})),
                      ("GET", "/tasks/t1", httpx.ReadTimeout("read stalled"))])
    with pytest.raises(error):
        run()


@pytest.mark.parametrize("failure", [
    httpx.ConnectError("refused"), httpx.ConnectTimeout("connect timeout"),
    httpx.Response(503, text="service restarting"),
])
def test_connection_outage_stays_transport_during_processing(server, failure):
    scripted(server, [("POST", "/tasks", submitted("processing")),
                      ("GET", "/tasks/t1", failure)])
    with pytest.raises(mc.MineruTransportError):
        run()


@pytest.mark.parametrize("status,error", [
    ("pending", mc.MineruTransportError), ("processing", mc.MineruExtractionError),
])
def test_wall_clock_expiry_uses_task_status(server, monkeypatch, status, error):
    clock = [100.0]
    monkeypatch.setattr(mc.time, "time", lambda: clock[0])

    def handle(request):
        if request.method == "POST":
            return submitted()
        assert request.url.path == "/tasks/t1"
        clock[0] = 102.0
        return httpx.Response(200, json={"status": status, "queued_ahead": 3})

    state = server(handle)
    with pytest.raises(error):
        run(wall_clock_deadline=101.0)
    assert len(state.requests) == 2


@pytest.mark.parametrize("status,error", [
    ("pending", mc.MineruTransportError), ("processing", mc.MineruExtractionError),
])
async def test_deadline_interrupts_inflight_status_request(server, status, error):
    async def handle(request):
        if request.method == "POST":
            return submitted(status, queued_ahead=2 if status == "pending" else 0)
        await asyncio.Event().wait()

    state = server(handle)
    with pytest.raises(error):
        await mc.extract_mineru(b"%PDF", [mc.Endpoint("a", "http://a")],
                               wall_clock_deadline=time.time() + 0.1)
    assert all(client.is_closed for client in state.clients)


def test_expired_deadline_does_not_submit(server):
    state = server(lambda request: pytest.fail("expired request was submitted"))
    with pytest.raises(mc.MineruTransportError):
        run(wall_clock_deadline=time.time() - 1)
    assert state.requests == []


def test_deadline_expiring_during_client_setup_does_not_submit(server, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mc.time, "time", lambda: clock[0])
    state = server(lambda request: submitted("completed"))
    client = httpx.AsyncClient

    def slow_setup(**kwargs):
        clock[0] = 102.0
        return client(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", slow_setup)
    with pytest.raises(mc.MineruTransportError):
        run(wall_clock_deadline=101.0)
    assert state.requests == []
    assert all(client.is_closed for client in state.clients)


@pytest.mark.parametrize("failure", [httpx.Response(502), httpx.ConnectError("refused")])
def test_transport_error_fails_over_to_next_endpoint(server, monkeypatch, failure):
    monkeypatch.setattr(mc, "_INLINE_RETRIES", 1)
    state = scripted(server, [("POST", "/tasks", failure), ("POST", "/tasks", submitted()),
                              ("GET", "/tasks/t1", completed()),
                              ("GET", "/tasks/t1/result", markdown())])
    assert asyncio.run(mc.extract_mineru(b"%PDF", [mc.Endpoint("a", "http://a"),
                        mc.Endpoint("b", "http://b")], stem="x")) == "# body"
    assert [request.url.host for request in state.requests] == ["a", "b", "b", "b"]
    assert all(client.is_closed for client in state.clients)


def test_processing_state_does_not_leak_into_failover_queue(server, monkeypatch):
    monkeypatch.setattr(mc, "_INLINE_RETRIES", 1)
    scripted(server, [("POST", "/tasks", submitted("processing")),
                      ("GET", "/tasks/t1", httpx.Response(503)),
                      ("POST", "/tasks", submitted(queued_ahead=2)),
                      ("GET", "/tasks/t1", httpx.ReadTimeout("queue wait"))])
    with pytest.raises(mc.MineruTransportError):
        asyncio.run(mc.extract_mineru(b"%PDF", [mc.Endpoint("a", "http://a"),
                    mc.Endpoint("b", "http://b")]))


@pytest.mark.parametrize("payload", [{}, {"task_id": ""}, ["unexpected"]])
def test_malformed_submission_is_transport_error(server, payload):
    scripted(server, [("POST", "/tasks", httpx.Response(202, json=payload))])
    with pytest.raises(mc.MineruTransportError):
        run()


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="invalid JSON"), httpx.Response(200, json={"status": "unknown"}),
    httpx.Response(200, json={}),
])
def test_malformed_status_is_transport_error(server, response):
    scripted(server, [("POST", "/tasks", submitted()), ("GET", "/tasks/t1", response)])
    with pytest.raises(mc.MineruTransportError):
        run()


async def test_cancellation_closes_http_client(server):
    entered = asyncio.Event()

    async def handle(request):
        if request.method == "POST":
            return submitted("processing")
        entered.set()
        await asyncio.Event().wait()

    state = server(handle)
    task = asyncio.create_task(mc.extract_mineru(b"%PDF", [mc.Endpoint("a", "http://a")]))
    try:
        await asyncio.wait_for(entered.wait(), 1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert all(client.is_closed for client in state.clients)


def test_separate_event_loops_own_and_close_separate_clients(server):
    def handle(request):
        if request.method == "POST":
            return submitted()
        return markdown() if request.url.path.endswith("/result") else completed()

    state = server(handle)
    assert run() == "# body"
    assert run() == "# body"
    assert len(state.clients) == 2
    assert all(client.is_closed for client in state.clients)


def test_configured_limits_and_timeout_reach_http_client(server, monkeypatch):
    monkeypatch.setenv("PAPER_LIBRARY_MINERU_MAX_CONCURRENCY", "12")
    monkeypatch.setenv("PAPER_LIBRARY_MINERU_MAX_CONNECTIONS", "24")
    state = scripted(server, [("POST", "/tasks", submitted()), ("GET", "/tasks/t1", completed()),
                              ("GET", "/tasks/t1/result", markdown())])
    assert run(http_timeout=17) == "# body"
    assert state.options[0]["limits"].max_connections == 24
    assert state.clients[0].timeout.read == 17


def test_no_endpoints_is_transport_error():
    with pytest.raises(mc.MineruTransportError):
        asyncio.run(mc.extract_mineru(b"%PDF", []))


def test_endpoints_default_when_unset(monkeypatch):
    monkeypatch.delenv("MINERU_URL", raising=False)
    eps = mc.endpoints_from_env()
    assert len(eps) == 1
    assert eps[0].url == "http://127.0.0.1:30000"


def test_endpoints_parses_comma_separated(monkeypatch):
    monkeypatch.setenv("MINERU_URL", "http://a:30000, http://b:30001 ")
    eps = mc.endpoints_from_env()
    assert [e.url for e in eps] == ["http://a:30000", "http://b:30001"]


def test_alternate_round_robins():
    eps = [mc.Endpoint("a", "http://a"), mc.Endpoint("b", "http://b")]
    assert mc.alternate(eps, 0).url == "http://a"
    assert mc.alternate(eps, 1).url == "http://b"
    assert mc.alternate(eps, 2).url == "http://a"      # wraps
    # Single endpoint degenerates to the same URL (steady state).
    one = [mc.Endpoint("a", "http://a")]
    assert mc.alternate(one, 5).url == "http://a"


def test_fd_watermark_warns_once_then_throttles(monkeypatch):
    """check_fd_watermark warns when open fds cross 60% of the soft limit, then
    THROTTLES to ≤1 warn / _FD_WARN_INTERVAL s on immediate re-calls."""
    monkeypatch.setattr(mc.resource, "getrlimit", lambda _which: (10, 1000))
    monkeypatch.setattr(mc, "_FD_WARN_FRACTION", 0.6)
    monkeypatch.setattr(mc, "_fd_warn_last_log", 0.0)
    real_listdir = os.listdir
    monkeypatch.setattr(
        mc.os, "listdir",
        lambda p: ["fd"] * 8 if p == "/proc/self/fd" else real_listdir(p))

    logger = Mock()
    mc.check_fd_watermark(logger)   # 8/10 = 80% >= 60% -> WARN
    mc.check_fd_watermark(logger)   # immediate -> throttled
    assert logger.warning.call_count == 1


def test_fd_watermark_never_raises_on_proc_error(monkeypatch):
    """A failed /proc/self/fd read is swallowed — the guard must never raise
    (nor warn) on the hot loop."""
    monkeypatch.setattr(mc.resource, "getrlimit", lambda _which: (10, 1000))
    monkeypatch.setattr(mc, "_FD_WARN_FRACTION", 0.6)
    monkeypatch.setattr(mc, "_fd_warn_last_log", 0.0)
    real_listdir = os.listdir

    def _boom(p):
        if p == "/proc/self/fd":
            raise OSError("proc read failed")
        return real_listdir(p)

    monkeypatch.setattr(mc.os, "listdir", _boom)

    logger = Mock()
    mc.check_fd_watermark(logger)   # must swallow, no raise
    assert logger.warning.call_count == 0


def test_client_limits_defaults(monkeypatch):
    monkeypatch.delenv("PAPER_LIBRARY_MINERU_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("PAPER_LIBRARY_MINERU_MAX_CONNECTIONS", raising=False)
    assert mc.client_limits() == (32, 64)


def test_client_limits_env_override(monkeypatch):
    monkeypatch.setenv("PAPER_LIBRARY_MINERU_MAX_CONCURRENCY", "12")
    monkeypatch.setenv("PAPER_LIBRARY_MINERU_MAX_CONNECTIONS", "24")
    assert mc.client_limits() == (12, 24)


def test_client_limits_ignore_invalid_env(monkeypatch):
    monkeypatch.setenv("PAPER_LIBRARY_MINERU_MAX_CONCURRENCY", "zero")
    monkeypatch.setenv("PAPER_LIBRARY_MINERU_MAX_CONNECTIONS", "0")
    assert mc.client_limits() == (32, 64)


def _count_trims(monkeypatch):
    calls = []
    monkeypatch.setattr(mc, "trim_heap", lambda: calls.append(1) or True)
    return calls


def test_trim_runs_once_per_successful_parse(monkeypatch):
    calls = _count_trims(monkeypatch)

    async def _impl(*_a, **_k):
        assert calls == []                    # not before the parse
        return "MD"

    monkeypatch.setattr(mc, "_extract_mineru_impl", _impl)
    eps = [mc.Endpoint("a", "http://a:30000")]
    assert asyncio.run(mc.extract_mineru(b"%PDF", eps, stem="x")) == "MD"
    assert calls == [1]


def test_trim_runs_once_per_failed_parse(monkeypatch):
    calls = _count_trims(monkeypatch)

    async def _impl(*_a, **_k):
        raise mc.MineruExtractionError("thin")

    monkeypatch.setattr(mc, "_extract_mineru_impl", _impl)
    eps = [mc.Endpoint("a", "http://a:30000")]
    with pytest.raises(mc.MineruExtractionError):
        asyncio.run(mc.extract_mineru(b"%PDF", eps, stem="x"))
    assert calls == [1]


def test_trim_failure_never_fails_the_parse(monkeypatch):
    def _boom():
        raise RuntimeError("trim exploded")

    monkeypatch.setattr(mc, "trim_heap", _boom)

    async def _impl(*_a, **_k):
        return "MD"

    monkeypatch.setattr(mc, "_extract_mineru_impl", _impl)
    eps = [mc.Endpoint("a", "http://a:30000")]
    assert asyncio.run(mc.extract_mineru(b"%PDF", eps, stem="x")) == "MD"


def test_trim_heap_is_noop_off_glibc(monkeypatch):
    def _no_glibc(_name):
        raise ValueError("unrecognized configuration name")

    monkeypatch.setattr(mc.os, "confstr", _no_glibc)
    monkeypatch.setattr(mc, "_malloc_trim_fn", mc._UNLOADED)
    assert mc.trim_heap() is False
    assert mc._malloc_trim_fn is None         # resolved once, cached as absent


def test_trim_heap_calls_malloc_trim_zero(monkeypatch):
    seen = []
    monkeypatch.setattr(mc, "_malloc_trim_fn", lambda pad: seen.append(pad) or 1)
    monkeypatch.setattr(mc, "_MALLOC_TRIM_ENABLED", True)
    assert mc.trim_heap() is True
    assert seen == [0]


def test_trim_heap_disabled_by_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(mc, "_malloc_trim_fn", lambda pad: seen.append(pad) or 1)
    monkeypatch.setattr(mc, "_MALLOC_TRIM_ENABLED", False)
    assert mc.trim_heap() is False
    assert seen == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc host only")
def test_trim_heap_resolves_real_glibc_symbol(monkeypatch):
    try:
        if not os.confstr("CS_GNU_LIBC_VERSION"):
            pytest.skip("not glibc")
    except (AttributeError, ValueError, OSError):
        pytest.skip("not glibc")
    monkeypatch.setattr(mc, "_malloc_trim_fn", mc._UNLOADED)
    monkeypatch.setattr(mc, "_MALLOC_TRIM_ENABLED", True)
    assert mc.trim_heap() is True
