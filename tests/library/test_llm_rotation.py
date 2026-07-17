"""Tests for the central hot-reloaded key pool (llm.KeyPool / get_llm).

Covers: read active groups (skip ``disabled``), per-request random failover,
401/403 AUTO-DISABLE written to the file (+ subsequent reads skip it), 429 stays
transient (no disable), mtime hot-reload, all-fail raises, env fallback when the
file is missing, and get_llm building a pool. See research/docs/llm-key-pool.md.
"""
from __future__ import annotations

import json
import os

import pytest

from papervault import config
from papervault.library import llm as L

# api_key -> None (ok) or int HTTP code (raise). Tests set this before calling.
BEHAVIOR: dict[str, int | None] = {}


class FakeErr(Exception):
    def __init__(self, code):
        super().__init__(f"http {code}")
        self.status_code = code


class FakeLLM:
    """Stand-in for crewai.LLM: replies key-tagged, or raises FakeErr(code) per
    BEHAVIOR[api_key]."""

    def __init__(self, **kw):
        self.kw = kw

    def call(self, *args, **kwargs):
        code = BEHAVIOR.get(self.kw.get("api_key"))
        if code:
            raise FakeErr(code)
        return f"resp-{self.kw.get('api_key')}"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(L, "LLM", FakeLLM)
    BEHAVIOR.clear()
    yield
    BEHAVIOR.clear()


@pytest.fixture
def no_shuffle(monkeypatch):
    """Keep file order so failover/auto-disable is deterministic."""
    monkeypatch.setattr(L.random, "shuffle", lambda x: None)


def _write(path, groups):
    path.write_text(json.dumps(groups))


def test_reads_active_and_skips_disabled(tmp_path):
    f = tmp_path / "keys.json"
    _write(f, [
        {"model": "m", "api_key": "k1", "base_url": "b", "disabled": {"code": 401}},
        {"model": "m", "api_key": "k2", "base_url": "b"},
    ])
    p = L.KeyPool(f, 10)
    assert [g["api_key"] for g in p._groups] == ["k2"]      # k1 skipped
    assert p.call([{"role": "user", "content": "x"}]) == "resp-k2"


def test_random_failover_returns_a_working_key(tmp_path):
    f = tmp_path / "keys.json"
    _write(f, [{"model": "m", "api_key": "k1", "base_url": "b"},
               {"model": "m", "api_key": "k2", "base_url": "b"}])
    BEHAVIOR["k1"] = 429                                    # k1 throttled, k2 fine
    p = L.KeyPool(f, 10)
    # whichever is tried first, a 429 fails over to the working one
    assert p.call([{"role": "user", "content": "x"}]) == "resp-k2"


def test_401_auto_disables_key_in_file(tmp_path, no_shuffle):
    f = tmp_path / "keys.json"
    _write(f, [{"model": "m", "api_key": "k1", "base_url": "b"},
               {"model": "m", "api_key": "k2", "base_url": "b"}])
    BEHAVIOR["k1"] = 401                                    # bad/expired credential
    p = L.KeyPool(f, 10)
    assert p.call([{"role": "user", "content": "x"}]) == "resp-k2"   # k1 → failover → k2
    data = json.loads(f.read_text())
    g1 = next(g for g in data if g["api_key"] == "k1")
    g2 = next(g for g in data if g["api_key"] == "k2")
    assert g1.get("disabled") and g1["disabled"]["code"] == 401 and g1["disabled"]["at"]
    assert not g2.get("disabled")                          # only the 401 key disabled


def test_429_does_not_disable(tmp_path, no_shuffle):
    f = tmp_path / "keys.json"
    _write(f, [{"model": "m", "api_key": "k1", "base_url": "b"},
               {"model": "m", "api_key": "k2", "base_url": "b"}])
    BEHAVIOR["k1"] = 429                                    # transient
    p = L.KeyPool(f, 10)
    p.call([{"role": "user", "content": "x"}])
    data = json.loads(f.read_text())
    assert not next(g for g in data if g["api_key"] == "k1").get("disabled")


def test_hot_reload_picks_up_new_key(tmp_path):
    f = tmp_path / "keys.json"
    _write(f, [{"model": "m", "api_key": "k1", "base_url": "b"}])
    p = L.KeyPool(f, 10)
    assert {g["api_key"] for g in p._groups} == {"k1"}
    _write(f, [{"model": "m", "api_key": "k1", "base_url": "b"},
               {"model": "m", "api_key": "k2", "base_url": "b"}])
    os.utime(f, (p._mtime + 10, p._mtime + 10))             # force a newer mtime
    p._reload_if_changed()
    assert {g["api_key"] for g in p._groups} == {"k1", "k2"}


def test_all_keys_fail_raises(tmp_path, no_shuffle):
    f = tmp_path / "keys.json"
    _write(f, [{"model": "m", "api_key": "k1", "base_url": "b"},
               {"model": "m", "api_key": "k2", "base_url": "b"}])
    BEHAVIOR["k1"] = 500
    BEHAVIOR["k2"] = 500
    p = L.KeyPool(f, 10)
    with pytest.raises(FakeErr):
        p.call([{"role": "user", "content": "x"}])


def test_env_fallback_when_file_missing(tmp_path, monkeypatch):
    # NEW contract: a SINGLE-endpoint env fallback from config.LLM_API_KEY (+ base_url) when the
    # key-pool file is missing. config is import-frozen → patch the attr; _env_fallback_groups
    # reads config.LLM_API_KEY live.
    monkeypatch.setattr(config, "LLM_API_KEY", "envk1")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://x/v1")
    p = L.KeyPool(tmp_path / "nope.json", 10)
    assert [g["api_key"] for g in p._groups] == ["envk1"]


def test_get_llm_builds_keypool(tmp_path, monkeypatch):
    f = tmp_path / "keys.json"
    _write(f, [{"model": "m", "api_key": "k1", "base_url": "b"}])
    monkeypatch.setattr(L, "_KEYS_FILE", f)
    monkeypatch.setattr(L, "_pools", {})
    monkeypatch.setattr(config, "USE_GATEWAY", False)   # gateway OFF → direct KeyPool path
    got = L.get_llm()
    assert isinstance(got, L.KeyPool)
    assert got.call([{"role": "user", "content": "x"}]) == "resp-k1"


# ---- Phase 3: gateway mode (config.USE_GATEWAY, DEFAULT OFF) --------------------------------------

def test_gateway_off_is_default_keypool(tmp_path, monkeypatch):
    """With the gateway OFF (config.USE_GATEWAY False, the live default), get_llm STILL returns a
    KeyPool — the gateway code must not touch the direct path."""
    f = tmp_path / "keys.json"
    _write(f, [{"model": "m", "api_key": "k1", "base_url": "b"}])
    monkeypatch.setattr(L, "_KEYS_FILE", f)
    monkeypatch.setattr(L, "_pools", {})
    monkeypatch.setattr(config, "USE_GATEWAY", False)   # gateway OFF (live default)
    assert isinstance(L.get_llm(), L.KeyPool)


def test_gateway_group_routing(monkeypatch):
    """Gateway maps a call-site model to the proxy's REAL model groups: default/None and an explicit
    synth request → the strong SYNTH_MODEL group; only an explicit BUILD_MODEL request → the build
    group. Default matches the direct path (_DEFAULT_MODEL is the synth model). config is
    import-frozen → patch the config attrs directly (_gateway_group reads them live)."""
    monkeypatch.setattr(config, "SYNTH_MODEL", "big")
    monkeypatch.setattr(config, "BUILD_MODEL", "small")
    monkeypatch.setattr(L, "_DEFAULT_MODEL", "openai/big")
    assert L._gateway_group(None) == "openai/big"                 # default → synth
    assert L._gateway_group("openai/small") == "openai/small"     # explicit build model
    assert L._gateway_group("small") == "openai/small"
    assert L._gateway_group("openai/big") == "openai/big"         # explicit synth tier
    assert L._gateway_group(L._DEFAULT_MODEL) == "openai/big"     # _DEFAULT_MODEL is the synth model


def test_gateway_on_returns_proxy_llm(monkeypatch):
    """config.USE_GATEWAY=True → an LLM pointed at the proxy: group model, config base_url/key,
    max_tokens passthrough, num_retries=0 (no double-retry), cached per (max_tokens, group)."""
    monkeypatch.setattr(config, "USE_GATEWAY", True)
    monkeypatch.setattr(config, "GATEWAY_URL", "http://proxy:4000/v1")
    monkeypatch.setattr(config, "GATEWAY_KEY", "sk-test")
    monkeypatch.setattr(config, "SYNTH_MODEL", "big")
    monkeypatch.setattr(config, "BUILD_MODEL", "small")
    monkeypatch.setattr(L, "_gw_llms", {})
    got = L.get_llm()                       # default site → synth group
    assert isinstance(got, FakeLLM)         # L.LLM is monkeypatched to FakeLLM by _reset
    assert got.kw == {"model": "openai/big", "base_url": "http://proxy:4000/v1",
                      "api_key": "sk-test", "max_tokens": L._DEFAULT_MAX_TOKENS, "num_retries": 0}
    # an explicit build-model request lands on the build group, with max_tokens preserved
    cheap = L.get_llm(model="openai/small", max_tokens=4096)
    assert cheap.kw["model"] == "openai/small" and cheap.kw["max_tokens"] == 4096
    # cached: same (max_tokens, group) → same object
    assert L.get_llm() is got


def test_gateway_on_default_url_and_raises_on_failure(monkeypatch):
    """The gateway client is pointed at config.GATEWAY_URL (documented default 127.0.0.1:4000);
    the proxy client still RAISES on error so the callers' fail-open/closed try/except keeps working
    (it must not be swallowed into None)."""
    monkeypatch.setattr(config, "USE_GATEWAY", True)
    monkeypatch.setattr(config, "GATEWAY_URL", "http://127.0.0.1:4000/v1")   # documented default
    monkeypatch.setattr(config, "GATEWAY_KEY", "sk-test")
    monkeypatch.setattr(config, "SYNTH_MODEL", "big")
    monkeypatch.setattr(config, "BUILD_MODEL", "small")
    monkeypatch.setattr(L, "_gw_llms", {})
    got = L.get_llm()
    assert got.kw["base_url"] == "http://127.0.0.1:4000/v1"
    BEHAVIOR["sk-test"] = 500               # proxy returns an error
    with pytest.raises(FakeErr):
        got.call([{"role": "user", "content": "x"}])
