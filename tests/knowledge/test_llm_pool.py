"""Unit tests for the central hot-reloaded MiMo key pool (store/llm.py).

Contract: research/docs/llm-key-pool.md ("给 KS 的实现契约"). Covers the 7 requirements:
  * active-only selection (skip ``disabled`` + keyless groups)               [req 1]
  * mtime hot-reload (edit the temp file → next call re-reads)               [req 1]
  * per-request random.shuffle failover, NO cross-request state              [req 2]
  * 429/timeout/5xx = transient (fail over, nothing written)                 [req 3]
  * 401/403 = permanent → file-locked + atomic ``disabled`` write, pl schema [req 3]
  * mimo_complete signature unchanged + valid_endpoint_count = active count  [req 4]
  * env fallback when the file is missing                                    [req 6]

ALL HTTP is mocked — no network, and we ALWAYS point LLM_KEYS_FILE at a tmp_path file via
monkeypatch, never the real research/llm_keys.json (which the suite must not corrupt).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import papervault.knowledge.store.llm as llm_mod
from papervault import config
from papervault.knowledge.store.llm import KeyPool, _sdk_model, mimo_complete


# ---- helpers ----------------------------------------------------------------

def _group(key: str, base: str = "https://sgp.example/v1", model: str = "openai/test-model",
           disabled: dict | None = None) -> dict:
    g = {"model": model, "api_key": key, "base_url": base}
    if disabled is not None:
        g["disabled"] = disabled
    return g


def _write(path: Path, groups: list[dict]) -> None:
    path.write_text(json.dumps(groups, indent=2, ensure_ascii=False) + "\n")


class _FakeResp:
    def __init__(self, text: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class _FakeError(Exception):
    """openai-style error carrying a status_code (what _error_code reads)."""
    def __init__(self, status_code: int):
        super().__init__(f"fake http {status_code}")
        self.status_code = status_code


def _pin_order(monkeypatch):
    """Make random.shuffle a no-op so the pool tries groups in file order — lets the disable
    tests deterministically exercise a specific (bad-then-good) order. (The genuine per-request
    randomness is covered separately by test_shuffle_is_per_request_no_cross_state.)"""
    monkeypatch.setattr(llm_mod.random, "shuffle", lambda seq: None)


def _install_fake_client(monkeypatch, behavior):
    """Replace store.llm.AsyncOpenAI with a fake whose chat.completions.create dispatches by
    api_key via ``behavior(api_key) -> str | Exception``. Records the call order in ``calls``."""
    calls: list[str] = []

    class _Completions:
        def __init__(self, key):
            self._key = key

        async def create(self, *, model, messages, **kw):
            calls.append(self._key)
            out = behavior(self._key)
            if isinstance(out, Exception):
                raise out
            return _FakeResp(out)

    class _Chat:
        def __init__(self, key):
            self.completions = _Completions(key)

    class _FakeAsyncOpenAI:
        def __init__(self, *, api_key, base_url, timeout=None, max_retries=None):
            self.chat = _Chat(api_key)

    monkeypatch.setattr(llm_mod, "AsyncOpenAI", _FakeAsyncOpenAI)
    return calls


# ---- req 4 / req 6: model-prefix mapping + signature ------------------------

def test_sdk_model_strips_litellm_prefix(monkeypatch):
    assert _sdk_model("openai/test-model") == "test-model"
    assert _sdk_model("test-model") == "test-model"
    # _sdk_model(None) falls back to the module default; pin it, don't hardcode a provider model.
    monkeypatch.setattr(llm_mod, "_DEFAULT_MODEL", "default-model")
    assert _sdk_model(None) == "default-model"


# ---- req 1: active-only selection -------------------------------------------

def test_active_skips_disabled_and_keyless(tmp_path):
    f = tmp_path / "keys.json"
    _write(f, [
        _group("k-ok-1"),
        _group("k-dead", disabled={"at": "x", "code": 401, "reason": "r"}),
        {"model": "openai/test-model", "base_url": "b"},  # no api_key
        _group("k-ok-2"),
    ])
    pool = KeyPool(f)
    keys = {g["api_key"] for g in pool.active_groups}
    assert keys == {"k-ok-1", "k-ok-2"}
    assert len(pool.active_groups) == 2


# ---- req 1: mtime hot-reload ------------------------------------------------

def test_hot_reload_on_mtime_change(tmp_path):
    f = tmp_path / "keys.json"
    _write(f, [_group("k1")])
    pool = KeyPool(f)
    assert {g["api_key"] for g in pool.active_groups} == {"k1"}

    # rewrite with a newer mtime; next access must re-read
    _write(f, [_group("k1"), _group("k2")])
    os.utime(f, (time.time() + 10, time.time() + 10))
    assert {g["api_key"] for g in pool.active_groups} == {"k1", "k2"}


# ---- req 6: env fallback when file missing ----------------------------------

def test_env_fallback_when_file_missing(tmp_path, monkeypatch):
    missing = tmp_path / "nope.json"
    # NEW contract: a SINGLE-endpoint env fallback built from config.LLM_API_KEY (+ base_url),
    # used ONLY when the key-pool file is missing. config is import-frozen, so patch the config
    # attribute directly — _env_fallback_groups reads config.LLM_API_KEY live.
    monkeypatch.setattr(config, "LLM_API_KEY", "tp-envkey1")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://sgp.example/v1")
    pool = KeyPool(missing)
    keys = {g["api_key"] for g in pool.active_groups}
    assert keys == {"tp-envkey1"}


# ---- req 2: success returns first good; per-request shuffle, stateless ------

async def test_complete_returns_first_success(tmp_path, monkeypatch):
    f = tmp_path / "keys.json"
    _write(f, [_group("kA"), _group("kB"), _group("kC")])
    _install_fake_client(monkeypatch, lambda key: f"hello from {key}")
    pool = KeyPool(f)
    out = await pool.complete("ping")
    assert out.startswith("hello from k")


async def test_shuffle_is_per_request_no_cross_state(tmp_path, monkeypatch):
    """Over many calls every active key should get picked first at least once → proves
    per-request random.shuffle and that there is no fixed sequential counter."""
    f = tmp_path / "keys.json"
    _write(f, [_group("k1"), _group("k2"), _group("k3"), _group("k4")])
    calls = _install_fake_client(monkeypatch, lambda key: "ok")
    pool = KeyPool(f)
    firsts = set()
    for _ in range(60):
        calls.clear()
        await pool.complete("ping")
        firsts.add(calls[0])  # first endpoint tried this request
    assert firsts == {"k1", "k2", "k3", "k4"}


# ---- req 3: transient failover (429/timeout/5xx) — nothing written ----------

async def test_transient_429_fails_over_and_does_not_disable(tmp_path, monkeypatch):
    f = tmp_path / "keys.json"
    _write(f, [_group("bad429"), _group("good")])

    def behavior(key):
        return _FakeError(429) if key == "bad429" else "ok"

    _install_fake_client(monkeypatch, behavior)
    pool = KeyPool(f)
    out = await pool.complete("ping")
    assert out == "ok"
    # file must be UNCHANGED — no disabled written for a transient code
    data = json.loads(f.read_text())
    assert all("disabled" not in g for g in data)


async def test_all_transient_raises_after_max_rounds(tmp_path, monkeypatch):
    f = tmp_path / "keys.json"
    _write(f, [_group("a"), _group("b")])
    _install_fake_client(monkeypatch, lambda key: _FakeError(500))
    # speed up: no real backoff sleeps
    monkeypatch.setattr(llm_mod, "_BASE_BACKOFF", 0.0)
    pool = KeyPool(f)
    with pytest.raises(RuntimeError, match="All active MiMo keys failed"):
        await pool.complete("ping")
    data = json.loads(f.read_text())
    assert all("disabled" not in g for g in data)  # 500 is transient → never disabled


# ---- req 3: permanent 401/403 → atomic+locked disabled write (pl schema) ----

async def test_401_auto_disables_with_pl_schema(tmp_path, monkeypatch):
    f = tmp_path / "keys.json"
    _write(f, [_group("dead401"), _group("good")])

    def behavior(key):
        return _FakeError(401) if key == "dead401" else "ok"

    _pin_order(monkeypatch)
    _install_fake_client(monkeypatch, behavior)
    pool = KeyPool(f)
    out = await pool.complete("ping")
    assert out == "ok"

    data = json.loads(f.read_text())
    by_key = {g["api_key"]: g for g in data}
    assert "disabled" not in by_key["good"]
    dis = by_key["dead401"]["disabled"]
    # EXACT pl schema: keys {at, code, reason}; ISO-Z timestamp; numeric code.
    assert set(dis.keys()) == {"at", "code", "reason"}
    assert dis["code"] == 401
    assert isinstance(dis["reason"], str) and dis["reason"]
    assert dis["at"].endswith("Z") and "T" in dis["at"]


async def test_403_also_disables(tmp_path, monkeypatch):
    f = tmp_path / "keys.json"
    _write(f, [_group("dead403"), _group("good")])
    _pin_order(monkeypatch)
    _install_fake_client(monkeypatch,
                         lambda key: _FakeError(403) if key == "dead403" else "ok")
    pool = KeyPool(f)
    await pool.complete("ping")
    by_key = {g["api_key"]: g for g in json.loads(f.read_text())}
    assert by_key["dead403"]["disabled"]["code"] == 403


async def test_disable_uses_pl_lock_filename(tmp_path, monkeypatch):
    """The lock file MUST be ``<keyfile>.lock`` (same name pl uses) so concurrent writes from
    both processes are mutually safe. We assert FileLock is constructed with that exact path."""
    f = tmp_path / "keys.json"
    _write(f, [_group("dead401"), _group("good")])
    _pin_order(monkeypatch)
    _install_fake_client(monkeypatch,
                         lambda key: _FakeError(401) if key == "dead401" else "ok")

    seen_lock_paths: list[str] = []
    import filelock
    real_filelock = filelock.FileLock

    def _spy_filelock(path, *a, **kw):
        seen_lock_paths.append(path)
        return real_filelock(path, *a, **kw)

    monkeypatch.setattr(filelock, "FileLock", _spy_filelock)
    pool = KeyPool(f)
    await pool.complete("ping")
    assert seen_lock_paths == [str(f) + ".lock"]


async def test_disable_is_atomic_no_partial_file(tmp_path, monkeypatch):
    """Atomic write = temp→os.replace; after the disable the file is still valid JSON with all
    original groups intact (only a `disabled` block appended to the dead one)."""
    f = tmp_path / "keys.json"
    _write(f, [_group("dead401", base="https://a/v1"),
               _group("good", base="https://b/v1", model="openai/test-model")])
    _pin_order(monkeypatch)
    _install_fake_client(monkeypatch,
                         lambda key: _FakeError(401) if key == "dead401" else "ok")
    pool = KeyPool(f)
    await pool.complete("ping")

    data = json.loads(f.read_text())          # parses → not corrupted
    assert len(data) == 2                      # both groups still present
    by_key = {g["api_key"]: g for g in data}
    # untouched fields preserved
    assert by_key["dead401"]["base_url"] == "https://a/v1"
    assert by_key["dead401"]["model"] == "openai/test-model"
    assert not (tmp_path / "keys.json.tmp").exists()   # temp cleaned up by os.replace


async def test_disable_then_hot_reload_drops_key(tmp_path, monkeypatch):
    """After a 401 disable, the SAME pool's next call must skip the now-disabled key
    (the disable bumps mtime/forces a re-read)."""
    f = tmp_path / "keys.json"
    _write(f, [_group("dead401"), _group("good")])
    _pin_order(monkeypatch)
    _install_fake_client(monkeypatch,
                         lambda key: _FakeError(401) if key == "dead401" else "ok")
    pool = KeyPool(f)
    await pool.complete("ping")                 # triggers the disable
    keys = {g["api_key"] for g in pool.active_groups}
    assert keys == {"good"}                      # dead401 gone after re-read


# ---- req 4: public mimo_complete signature + delegation ---------------------

async def test_mimo_complete_delegates_to_pool(tmp_path, monkeypatch):
    f = tmp_path / "keys.json"
    _write(f, [_group("kX")])
    monkeypatch.setenv("LLM_KEYS_FILE", str(f))
    monkeypatch.setattr(llm_mod, "_KEYS_FILE", f)
    monkeypatch.setattr(llm_mod, "_pool", None)   # fresh singleton for this temp file
    _install_fake_client(monkeypatch, lambda key: "delegated-ok")
    out = await mimo_complete("ping", system_prompt="sys", history_messages=[], temperature=0.1)
    assert out == "delegated-ok"


def test_active_endpoint_count_and_pool_model(tmp_path, monkeypatch):
    f = tmp_path / "keys.json"
    _write(f, [_group("k1"), _group("k2"),
               _group("kdead", disabled={"at": "x", "code": 401, "reason": "r"})])
    monkeypatch.setattr(llm_mod, "_KEYS_FILE", f)
    monkeypatch.setattr(llm_mod, "_pool", None)
    assert llm_mod.active_endpoint_count() == 2
    assert llm_mod.pool_model() == "test-model"   # litellm prefix stripped
