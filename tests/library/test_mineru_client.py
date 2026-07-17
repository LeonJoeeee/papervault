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
    """mineru is an optionally-absent heavy dep (it is NOT in the test venv).
    Calling extract_mineru with an endpoint surfaces the missing import as a
    TRANSPORT error (deploy problem, not a per-doc defect) — never an extraction
    charge. This proves the lazy-import contract: ``import papervault.library.extract``
    works without mineru, and the failure is classified transport at call time."""
    eps = [mc.Endpoint("a", "http://127.0.0.1:30000")]
    with pytest.raises(mc.MineruTransportError):
        asyncio.run(mc.extract_mineru(b"%PDF-1.4 fake", eps, stem="x"))


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
