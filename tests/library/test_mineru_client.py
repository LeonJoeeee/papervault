"""Tests for ``papervault.library.mineru_client`` — the in-process MinerU vLLM client.

These exercise the C1 transport-vs-extraction discriminator (SDD §2.2/§2.3)
WITHOUT a real MinerU server: the status/connect/structured-500 helpers are pure
string functions, the endpoint parser reads the env, and the two typed
exceptions are the contract ``extract.extract_md`` branches on. The real
``aio_do_parse`` http path needs a live vLLM server and is exercised by the live
smoke (SDD §8.B), not here.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from unittest.mock import Mock

import pytest

from papervault.library import mineru_client as mc


# ----------------------------- endpoints -----------------------------------


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


# ---------------- the C1 discriminator (status / connect / 500) -------------


@pytest.mark.parametrize("msg,code", [
    ("Unexpected status code: [502], response body: ...", 502),
    ("Unexpected status code: [400], response body: bad", 400),
    ("... Status code: 503, response body: ...", 503),
    ("no code here at all", None),
])
def test_status_of_parses_code(msg, code):
    assert mc._status_of(Exception(msg)) == code


def test_is_connect_fail_detects_restart_window():
    assert mc._is_connect_fail(Exception("Failed to connect to server http://x"))
    assert mc._is_connect_fail(Exception("Connection refused"))
    assert not mc._is_connect_fail(Exception("Unexpected status code: [400]"))


def test_is_structured_500_only_on_structured_body():
    structured = Exception("Error from server: {'object': 'error', 'message': 'bad'}")
    assert mc._is_structured_500(structured)
    # A BARE 500 (no structured object:error body) is NOT structured → transport.
    assert not mc._is_structured_500(Exception("Unexpected status code: [500]"))


def test_zero_byte_timeout_is_transport():
    assert mc._is_zero_byte_timeout(Exception("Read timed out"))
    assert mc._is_zero_byte_timeout(Exception("connection timeout"))
    assert not mc._is_zero_byte_timeout(Exception("Unexpected status code: [422]"))


# ----------------------- extract_mineru guard paths -------------------------


def test_extract_mineru_no_endpoints_is_transport():
    """No endpoint configured → MineruTransportError (deploy/transport-class,
    never a per-doc charge)."""
    with pytest.raises(mc.MineruTransportError):
        asyncio.run(mc.extract_mineru(b"%PDF", [], stem="x"))


def test_extract_mineru_import_failure_is_transport(monkeypatch):
    """A DOWN/absent MinerU surfaces as a TRANSPORT error (deploy problem, not a
    per-doc defect) — never an extraction charge. With the #76 reachability
    pre-check ON (default), an unreachable ``127.0.0.1:30000`` is classified
    transport BEFORE the lazy import even runs; in a mineru-less venv the absent
    import would classify the same way if the pre-check were bypassed. Either
    path proves the contract: no per-doc charge for a deploy/transport failure,
    and ``import papervault.library.extract`` never requires mineru at import."""
    eps = [mc.Endpoint("a", "http://127.0.0.1:30000")]
    with pytest.raises(mc.MineruTransportError):
        asyncio.run(mc.extract_mineru(b"%PDF-1.4 fake", eps, stem="x"))


# ------------- issue #76: async reachability pre-check (loop-safety) ---------
# aio_do_parse's client construction runs SYNCHRONOUS work on the S4 event loop
# and stalls when MinerU is unreachable (py-spy caught MainThread there), which
# starved the MCP handshake. The pre-check fast-fails a down server off-block.


def test_precheck_unreachable_short_circuits_before_parse(monkeypatch):
    """Pre-check ON + an unreachable server ⇒ MineruTransportError raised BEFORE
    aio_do_parse is ever entered (the loop-blocking construction never runs)."""
    _install_fake_mineru(monkeypatch, None)

    async def _boom(**_kw):
        raise AssertionError("aio_do_parse must NOT be called when unreachable")
    import sys
    sys.modules["mineru.cli.common"].aio_do_parse = _boom

    async def _down(*_a, **_k):
        return False
    monkeypatch.setattr(mc, "_any_endpoint_ready", _down)
    monkeypatch.setattr(mc, "_PRECHECK_ENABLED", True, raising=False)

    eps = [mc.Endpoint("a", "http://a:30000")]
    with pytest.raises(mc.MineruTransportError) as ei:
        asyncio.run(mc.extract_mineru(b"%PDF-1.4", eps, stem="x"))
    assert "precheck" in str(ei.value)


def test_precheck_off_reaches_parse(monkeypatch):
    """PAPER_LIBRARY_MINERU_PRECHECK=0 (flag off) skips the probe entirely and
    goes straight to aio_do_parse — the pre-#76 behaviour (revert lever)."""
    md_written = {}

    async def _fake_parse(**kw):
        from pathlib import Path
        out = Path(kw["output_dir"]) / kw["pdf_file_names"][0] / "vlm"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{kw['pdf_file_names'][0]}.md").write_text("# body")
        md_written["ok"] = True

    _install_fake_mineru(monkeypatch, _fake_parse)

    async def _must_not_probe(*_a, **_k):
        raise AssertionError("pre-check must be skipped when flag off")
    monkeypatch.setattr(mc, "_any_endpoint_ready", _must_not_probe)
    monkeypatch.setattr(mc, "_PRECHECK_ENABLED", False, raising=False)

    eps = [mc.Endpoint("a", "http://a:30000")]
    md = asyncio.run(mc.extract_mineru(b"%PDF-1.4", eps, stem="x"))
    assert md == "# body" and md_written.get("ok")


def test_any_endpoint_ready_passes_if_one_up(monkeypatch):
    """``_any_endpoint_ready`` is ANY, not ALL — one healthy endpoint in a
    2-endpoint set is enough to proceed (preserves round-robin failover)."""
    async def _health(url, _timeout):
        return url.endswith(":30001")  # only the 2nd endpoint is up
    monkeypatch.setattr(mc, "_endpoint_health_ok", _health)
    eps = [mc.Endpoint("a", "http://a:30000"), mc.Endpoint("b", "http://b:30001")]
    assert asyncio.run(mc._any_endpoint_ready(eps, 1.0)) is True

    async def _all_down(_url, _timeout):
        return False
    monkeypatch.setattr(mc, "_endpoint_health_ok", _all_down)
    assert asyncio.run(mc._any_endpoint_ready(eps, 1.0)) is False


def test_endpoint_health_ok_down_is_false_fast():
    """The real async probe against a closed localhost port returns False
    quickly (connection-refused) without raising — the loop-safe fast path."""
    import socket
    import time
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    t0 = time.monotonic()
    ok = asyncio.run(mc._endpoint_health_ok(f"http://127.0.0.1:{port}", 2.5))
    assert ok is False
    assert time.monotonic() - t0 < 2.5  # refused resolves well under the timeout


def test_read_md_from_outdir_missing_is_extraction(tmp_path):
    """A 200-OK that wrote no md (or an empty md) is EXTRACTION-class — the
    server ran but produced nothing usable."""
    from pathlib import Path
    with pytest.raises(mc.MineruExtractionError):
        mc._read_md_from_outdir(Path(tmp_path), "nope")

    # Empty md body → also extraction-class.
    md = Path(tmp_path) / "stem" / "vlm" / "stem.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("   \n")
    with pytest.raises(mc.MineruExtractionError):
        mc._read_md_from_outdir(Path(tmp_path), "stem")


def test_read_md_from_outdir_returns_body(tmp_path):
    from pathlib import Path
    md = Path(tmp_path) / "stem" / "vlm" / "stem.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("# real md body")
    assert mc._read_md_from_outdir(Path(tmp_path), "stem") == "# real md body"


# ----------------- wall-clock / slow-doc classification (reviewer must-fix) --
# A read-timeout = the server CONNECTED but this doc is too slow → per-doc
# pathology → EXTRACTION (charge → budget-terminal), so a pathologically slow
# PDF can't black-hole forever. A connect-fail / outage stays TRANSPORT.


def _install_fake_mineru(monkeypatch, aio_do_parse):
    """Inject fake mineru.cli.common + mineru_vl_utils.vlm_client.base_client so
    extract_mineru's lazy import succeeds without a real MinerU install, and the
    full retry loop runs against a stubbed aio_do_parse."""
    import sys
    import types

    class RequestError(ValueError):
        pass

    class ServerError(RuntimeError):
        pass

    class UnsupportedError(NotImplementedError):
        pass

    common = types.ModuleType("mineru.cli.common")
    common.aio_do_parse = aio_do_parse
    base = types.ModuleType("mineru_vl_utils.vlm_client.base_client")
    base.RequestError = RequestError
    base.ServerError = ServerError
    base.UnsupportedError = UnsupportedError
    for name, mod in (
        ("mineru", types.ModuleType("mineru")),
        ("mineru.cli", types.ModuleType("mineru.cli")),
        ("mineru.cli.common", common),
        ("mineru_vl_utils", types.ModuleType("mineru_vl_utils")),
        ("mineru_vl_utils.vlm_client", types.ModuleType("mineru_vl_utils.vlm_client")),
        ("mineru_vl_utils.vlm_client.base_client", base),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(mc, "_BACKOFF_BASE", 0.0, raising=False)  # no real sleeps

    # Issue #76: force the async reachability pre-check to "reachable" so these
    # tests exercise the retry-loop classification (the server is stubbed as
    # up-and-answering; the pre-check is covered separately below).
    async def _reachable(*_a, **_k):
        return True
    monkeypatch.setattr(mc, "_any_endpoint_ready", _reachable)
    return ServerError


def test_slow_doc_read_timeout_exhausted_is_extraction(monkeypatch):
    """A server that keeps read-timing-out (alive but slow) → after the inline
    budget is spent → EXTRACTION (charge), NOT a forever transport hot-loop."""
    async def read_timeout(**_kw):
        raise type("ReadTimeout", (Exception,), {})("body too slow")

    _install_fake_mineru(monkeypatch, read_timeout)
    eps = [mc.Endpoint("a", "http://a:30000")]
    with pytest.raises(mc.MineruExtractionError):
        asyncio.run(mc.extract_mineru(b"%PDF-1.4", eps, stem="x"))


def test_connect_fail_exhausted_is_transport(monkeypatch):
    """A server that's DOWN (connect-fail) stays TRANSPORT even after the inline
    budget — reconcile retries, never charges/terminalizes (C1)."""
    ServerError = _install_fake_mineru(monkeypatch, None)

    async def connect_fail(**_kw):
        raise ServerError("Failed to connect to server: connection refused")

    import sys
    sys.modules["mineru.cli.common"].aio_do_parse = connect_fail
    eps = [mc.Endpoint("a", "http://a:30000")]
    with pytest.raises(mc.MineruTransportError):
        asyncio.run(mc.extract_mineru(b"%PDF-1.4", eps, stem="x"))


# =================== transport-lifecycle: per-loop client close (#93) =========
#
# These are PURE-PYTHON regression tests for the socket-leak fix: NO mineru
# install and NO live server. They fake ``sys.modules["mineru.backend.vlm.
# vlm_analyze"]`` with a stub ``ModelSingleton`` -> predictor -> client ->
# ``_aio_client_cache`` graph, exactly the internals ``_close_mineru_client_for_loop``
# walks, and assert the refcount / eviction / kill-switch / fd-guard behaviour so a
# future refactor can't silently reintroduce the leak (issue #93).


class _StubAioClient:
    """Stand-in for a mineru per-loop ``httpx.AsyncClient``; records aclose()."""

    def __init__(self) -> None:
        self.aclosed = False

    async def aclose(self) -> None:
        self.aclosed = True


class _RaisingAioClient:
    """aclose() records the call THEN raises — proves eviction is aclose-safe."""

    def __init__(self) -> None:
        self.aclose_called = False

    async def aclose(self) -> None:
        self.aclose_called = True
        raise RuntimeError("aclose boom")


class _StubHttpClient:
    def __init__(self, cache: dict) -> None:
        self._aio_client_cache = cache


class _StubPredictor:
    def __init__(self, cache: dict) -> None:
        self.client = _StubHttpClient(cache)


class _StubModelSingleton:
    """Mimics mineru's process-global singleton: ``ModelSingleton()._models`` is a
    class-level dict of predictors (populated per test)."""

    _models: dict = {}


def _install_fake_singleton(monkeypatch, models: dict) -> None:
    """Install a fake ``mineru.backend.vlm.vlm_analyze`` module whose
    ``ModelSingleton()._models`` is ``models``."""
    _StubModelSingleton._models = dict(models)
    mod = types.ModuleType("mineru.backend.vlm.vlm_analyze")
    mod.ModelSingleton = _StubModelSingleton
    monkeypatch.setitem(sys.modules, "mineru.backend.vlm.vlm_analyze", mod)


def test_client_close_refcount_closes_only_at_zero(monkeypatch):
    """Two in-flight parses on ONE loop: the first decref does NOT close the
    shared client (another parse still using it); the second (count 0) does."""
    monkeypatch.setattr(mc, "_CLIENT_CLOSE_ENABLED", True)
    mc._loop_parse_refcounts.clear()
    stub = _StubAioClient()

    async def _body():
        loop = asyncio.get_running_loop()
        _install_fake_singleton(monkeypatch, {"k": _StubPredictor({loop: stub})})
        mc._incref_current_loop()
        mc._incref_current_loop()
        await mc._decref_current_loop_and_maybe_close()      # 2 -> 1, NO close
        assert stub.aclosed is False
        assert mc._loop_parse_refcounts.get(loop) == 1
        await mc._decref_current_loop_and_maybe_close()      # 1 -> 0, CLOSE
        assert stub.aclosed is True
        assert loop not in mc._loop_parse_refcounts

    asyncio.run(_body())


def test_client_close_evicts_only_current_loop(monkeypatch):
    """Close-at-zero acloses + evicts ONLY the current loop's cached client;
    another loop's cached client in the same cache is left untouched."""
    monkeypatch.setattr(mc, "_CLIENT_CLOSE_ENABLED", True)
    mc._loop_parse_refcounts.clear()
    cur = _StubAioClient()
    other = _StubAioClient()
    other_loop = object()   # stand-in key for a DIFFERENT event loop

    async def _body():
        loop = asyncio.get_running_loop()
        cache = {loop: cur, other_loop: other}
        _install_fake_singleton(monkeypatch, {"k": _StubPredictor(cache)})
        mc._incref_current_loop()
        await mc._decref_current_loop_and_maybe_close()      # 0 -> close current only
        assert cur.aclosed is True
        assert loop not in cache
        assert other.aclosed is False
        assert cache.get(other_loop) is other

    asyncio.run(_body())


def test_client_close_evicts_even_if_aclose_raises(monkeypatch):
    """If aclose() raises, the entry is STILL evicted (pop precedes aclose) and
    the error is swallowed, so the next parse rebuilds a fresh client cleanly."""
    monkeypatch.setattr(mc, "_CLIENT_CLOSE_ENABLED", True)
    mc._loop_parse_refcounts.clear()
    bad = _RaisingAioClient()

    async def _body():
        loop = asyncio.get_running_loop()
        cache = {loop: bad}
        _install_fake_singleton(monkeypatch, {"k": _StubPredictor(cache)})
        mc._incref_current_loop()
        await mc._decref_current_loop_and_maybe_close()      # aclose raises -> swallowed
        assert bad.aclose_called is True
        assert loop not in cache                             # evicted -> rebuild path clear

    asyncio.run(_body())


def test_client_close_disabled_leaves_client_cached(monkeypatch):
    """PAPER_LIBRARY_MINERU_CLIENT_CLOSE=0 (``_CLIENT_CLOSE_ENABLED`` False):
    the wrapper is inert — refcount still cleaned up, but the client is left
    OPEN + cached (no aclose, no eviction)."""
    monkeypatch.setattr(mc, "_CLIENT_CLOSE_ENABLED", False)
    mc._loop_parse_refcounts.clear()
    stub = _StubAioClient()

    async def _body():
        loop = asyncio.get_running_loop()
        cache = {loop: stub}
        _install_fake_singleton(monkeypatch, {"k": _StubPredictor(cache)})
        mc._incref_current_loop()
        await mc._decref_current_loop_and_maybe_close()      # 0, but close DISABLED
        assert stub.aclosed is False
        assert cache.get(loop) is stub                       # left open + cached
        assert loop not in mc._loop_parse_refcounts          # refcount still popped

    asyncio.run(_body())


def test_extract_mineru_wrapper_brackets_and_closes_on_success(monkeypatch):
    """The public wrapper increfs BEFORE the parse (refcount==1 during it) and,
    in its finally, decrefs + closes this loop's client on the success path."""
    monkeypatch.setattr(mc, "_CLIENT_CLOSE_ENABLED", True)
    mc._loop_parse_refcounts.clear()
    closed: list = []

    async def _fake_impl(*_a, **_k):
        assert mc._loop_parse_refcounts.get(asyncio.get_running_loop()) == 1
        return "MD"

    async def _fake_close(loop):
        closed.append(loop)

    monkeypatch.setattr(mc, "_extract_mineru_impl", _fake_impl)
    monkeypatch.setattr(mc, "_close_mineru_client_for_loop", _fake_close)

    async def _body():
        loop = asyncio.get_running_loop()
        eps = [mc.Endpoint("a", "http://127.0.0.1:30000")]
        assert await mc.extract_mineru(b"%PDF", eps, stem="x") == "MD"
        assert loop not in mc._loop_parse_refcounts
        assert closed == [loop]

    asyncio.run(_body())


def test_extract_mineru_wrapper_decrefs_and_closes_on_error(monkeypatch):
    """The finally runs on the ERROR path too: a raising parse still decrefs +
    closes, so an exception never leaks a refcount or a client."""
    monkeypatch.setattr(mc, "_CLIENT_CLOSE_ENABLED", True)
    mc._loop_parse_refcounts.clear()
    closed: list = []

    async def _boom_impl(*_a, **_k):
        raise mc.MineruTransportError("boom")

    async def _fake_close(loop):
        closed.append(loop)

    monkeypatch.setattr(mc, "_extract_mineru_impl", _boom_impl)
    monkeypatch.setattr(mc, "_close_mineru_client_for_loop", _fake_close)

    async def _body():
        loop = asyncio.get_running_loop()
        eps = [mc.Endpoint("a", "http://127.0.0.1:30000")]
        with pytest.raises(mc.MineruTransportError):
            await mc.extract_mineru(b"%PDF", eps, stem="x")
        assert loop not in mc._loop_parse_refcounts
        assert closed == [loop]

    asyncio.run(_body())


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


# ========= issue #134: server-side image decode failure is TRANSPORT =========
# A MinerU vLLM server whose Pillow plugins could not load (a stale process on a
# deleted venv) answers EVERY page with HTTP 400 "Failed to load image". The client
# renders the PNGs itself, so this is never the paper's fault: it must be a
# transport-class error (no attempt charged), not a per-doc extraction verdict.

_IMAGE_DECODE_400 = (
    'Unexpected status code: [400], response body: {"error":{"message":'
    '"Failed to load image: cannot identify image file <_io.BytesIO object>",'
    '"type":"BadRequestError","param":null,"code":400}}')


def _raising_parse(ServerError_holder, message, calls):
    async def _parse(**_kw):
        calls.append(1)
        raise ServerError_holder[0](message)
    return _parse


def test_image_decode_400_is_transport_not_extraction(monkeypatch):
    holder: list = [None]
    calls: list = []
    holder[0] = _install_fake_mineru(
        monkeypatch, _raising_parse(holder, _IMAGE_DECODE_400, calls))
    eps = [mc.Endpoint("a", "http://a:30000")]
    with pytest.raises(mc.MineruTransportError) as ei:
        asyncio.run(mc.extract_mineru(b"%PDF-1.4", eps, stem="x"))
    assert not isinstance(ei.value, mc.MineruExtractionError)
    assert isinstance(ei.value, mc.MineruImageDecodeError)
    assert "Failed to load image" in str(ei.value)
    # No inline retry: a broken server fails every page the same way.
    assert len(calls) == 1


def test_plain_400_is_still_extraction(monkeypatch):
    holder: list = [None]
    calls: list = []
    holder[0] = _install_fake_mineru(monkeypatch, _raising_parse(
        holder, 'Unexpected status code: [400], response body: {"error":'
                '{"message":"prompt too long","code":400}}', calls))
    eps = [mc.Endpoint("a", "http://a:30000")]
    with pytest.raises(mc.MineruExtractionError):
        asyncio.run(mc.extract_mineru(b"%PDF-1.4", eps, stem="x"))


def test_plain_422_is_still_extraction(monkeypatch):
    holder: list = [None]
    calls: list = []
    holder[0] = _install_fake_mineru(monkeypatch, _raising_parse(
        holder, "Unexpected status code: [422], response body: unprocessable", calls))
    eps = [mc.Endpoint("a", "http://a:30000")]
    with pytest.raises(mc.MineruExtractionError):
        asyncio.run(mc.extract_mineru(b"%PDF-1.4", eps, stem="x"))


def test_image_decode_text_on_non_400_is_not_reclassified():
    """Only the 400 + 'Failed to load image' pair is the stale-server signature."""
    assert mc._is_server_image_decode_fail(Exception(_IMAGE_DECODE_400))
    assert not mc._is_server_image_decode_fail(Exception(
        "Unexpected status code: [422], response body: Failed to load image"))
    assert not mc._is_server_image_decode_fail(Exception(
        "Unexpected status code: [400], response body: bad prompt"))


# ============ issue #96: bounded request fan-out + socket ceiling ============
# Each parse fanned out to mineru's default 100 concurrent requests with an
# uncapped httpx pool (8 slots x 100 = 747 sockets observed). The client now
# passes max_concurrency (default 32 = the unit's --max-num-seqs 32) and a
# max_connections ceiling. mineru 3.4.4's ModelSingleton forwards
# max_concurrency but DROPS max_connections, so the ceiling is also applied to
# the singleton's HTTP client before the parse creates its per-loop pool.


class _CapClient:
    def __init__(self):
        self.max_connections = None
        self.max_concurrency = 100


class _CapPredictor:
    def __init__(self):
        self.client = _CapClient()
        self.max_concurrency = 100


def _install_fake_capping_singleton(monkeypatch, order):
    predictor = _CapPredictor()
    seen = {}

    class _Singleton:
        def get_model(self, backend, model_path, server_url, **kwargs):
            order.append("get_model")
            seen.update(backend=backend, model_path=model_path,
                        server_url=server_url, kwargs=kwargs)
            return predictor

    mod = types.ModuleType("mineru.backend.vlm.vlm_analyze")
    mod.ModelSingleton = _Singleton
    monkeypatch.setitem(sys.modules, "mineru.backend.vlm.vlm_analyze", mod)
    return predictor, seen


def _recording_parse(order, captured):
    async def _parse(**kw):
        order.append("aio_do_parse")
        captured.update(kw)
        from pathlib import Path
        out = Path(kw["output_dir"]) / kw["pdf_file_names"][0] / "vlm"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{kw['pdf_file_names'][0]}.md").write_text("# body")
    return _parse


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


def test_parse_passes_caps_to_mineru_client(monkeypatch):
    monkeypatch.setenv("PAPER_LIBRARY_MINERU_MAX_CONCURRENCY", "16")
    monkeypatch.setenv("PAPER_LIBRARY_MINERU_MAX_CONNECTIONS", "40")
    order: list = []
    captured: dict = {}
    _install_fake_mineru(monkeypatch, _recording_parse(order, captured))
    predictor, seen = _install_fake_capping_singleton(monkeypatch, order)

    eps = [mc.Endpoint("a", "http://a:30000")]
    assert asyncio.run(mc.extract_mineru(b"%PDF-1.4", eps, stem="x")) == "# body"
    # Passed through aio_do_parse (mineru forwards max_concurrency) ...
    assert captured["max_concurrency"] == 16
    assert captured["max_connections"] == 40
    # ... and the singleton predictor is built with the same kwargs for the same
    # (backend, model_path, server_url) key, then capped, BEFORE the parse runs.
    assert order == ["get_model", "aio_do_parse"]
    assert (seen["backend"], seen["model_path"], seen["server_url"]) == (
        "http-client", None, "http://a:30000")
    assert seen["kwargs"]["max_concurrency"] == 16
    assert predictor.client.max_connections == 40
    assert predictor.client.max_concurrency == 16
    assert predictor.max_concurrency == 16


def test_parse_without_mineru_singleton_still_passes_caps(monkeypatch):
    """A mineru build without the singleton internals degrades to the kwargs
    alone — never an error."""
    order: list = []
    captured: dict = {}
    _install_fake_mineru(monkeypatch, _recording_parse(order, captured))
    monkeypatch.delitem(sys.modules, "mineru.backend.vlm.vlm_analyze", raising=False)
    eps = [mc.Endpoint("a", "http://a:30000")]
    assert asyncio.run(mc.extract_mineru(b"%PDF-1.4", eps, stem="x")) == "# body"
    assert order == ["aio_do_parse"]
    assert captured["max_concurrency"] == mc.client_limits()[0]


# ============ issue #133: return freed parse memory to the OS ============
# A whole-doc parse renders + base64-encodes page windows in worker threads;
# glibc keeps the freed memory in per-thread arenas (+5.5 GB retained after an
# 8-way burst). After each parse the client calls malloc_trim(0) — glibc only,
# a no-op elsewhere, never an error.


def _count_trims(monkeypatch):
    calls = []
    monkeypatch.setattr(mc, "trim_heap", lambda: calls.append(1) or True)
    monkeypatch.setattr(mc, "_close_mineru_client_for_loop",
                        lambda _loop: asyncio.sleep(0))
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
    monkeypatch.setattr(mc, "_close_mineru_client_for_loop",
                        lambda _loop: asyncio.sleep(0))

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
