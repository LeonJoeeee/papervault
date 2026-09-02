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

import httpx
import openai
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


@pytest.fixture(autouse=True)
def _stream_on(monkeypatch):
    """Pin the documented default (streaming ON) regardless of the developer's / deployment's
    .env, so the streaming tests exercise the streaming branch; the kill-switch test overrides."""
    monkeypatch.setattr(llm_mod, "_STREAM", True)


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
            if kw.get("stream"):
                # Mirror the real SDK: stream=True hands back an async iterator of chunks
                # (the pool streams by default since issue #100 and joins the deltas).
                async def _gen():
                    yield _chunk(content=out, finish="stop")
                return _gen()
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
    with pytest.raises(RuntimeError, match="All configured LLM keys failed"):
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


# ---- streaming (issue #100): long generations must not die at the relay's 120 s idle cut ----
#
# The relay behind the gateway closes any request that sends no bytes for ~120 s. A streamed
# completion keeps bytes flowing, so the pool streams by default and joins the deltas back into
# the str the LightRAG contract expects. ALL HTTP is faked — the fake `create` returns an async
# iterator of chunk objects when called with stream=True.

def _chunk(content=None, reasoning=None, finish=None, usage=None, index=0):
    """A minimal chat-completion CHUNK: `.choices[0].delta.content` (+ optional
    `reasoning_content`), `.choices[0].finish_reason`, `.choices[0].index` (the real API's
    choice index, non-zero only with n>1) and `.usage` (None except on the trailing usage-only
    chunk, which carries EMPTY choices like the real API)."""
    if usage is not None:
        return type("Chunk", (), {"choices": [], "usage": type("U", (), usage)()})()
    delta = type("Delta", (), {"content": content, "reasoning_content": reasoning})()
    choice = type("Choice", (), {"delta": delta, "finish_reason": finish, "index": index})()
    return type("Chunk", (), {"choices": [choice], "usage": None})()


def _install_fake_stream_client(monkeypatch, behavior):
    """Like _install_fake_client, but `behavior(api_key)` returns either a plain str (served for
    a NON-stream create) or a list of chunk objects / exceptions (served, in order, from an async
    iterator for a stream=True create; an Exception element is raised mid-stream). Records every
    create() kwargs dict in ``created``."""
    created: list[dict] = []

    class _Completions:
        def __init__(self, key):
            self._key = key

        async def create(self, *, model, messages, **kw):
            created.append(dict(model=model, **kw))
            out = behavior(self._key)
            if isinstance(out, Exception):
                raise out
            if not kw.get("stream"):
                return _FakeResp(out if isinstance(out, str) else "".join(
                    c.choices[0].delta.content or "" for c in out if c.choices))

            async def _gen():
                for item in out:
                    if isinstance(item, Exception):
                        raise item
                    yield item
            return _gen()

    class _Chat:
        def __init__(self, key):
            self.completions = _Completions(key)

    class _FakeAsyncOpenAI:
        def __init__(self, *, api_key, base_url, timeout=None, max_retries=None):
            self.chat = _Chat(api_key)

    monkeypatch.setattr(llm_mod, "AsyncOpenAI", _FakeAsyncOpenAI)
    return created


def _gateway_mode(monkeypatch):
    monkeypatch.setattr(config, "USE_GATEWAY", True)
    monkeypatch.setattr(config, "GATEWAY_URL", "http://gw.example/v1")
    monkeypatch.setattr(config, "GATEWAY_KEY", "gw-key")
    monkeypatch.setattr(config, "SYNTH_MODEL", "standard")
    monkeypatch.setattr(config, "BUILD_MODEL", "flash")


_HELLO_STREAM = [
    _chunk(reasoning="let me think"),          # reasoning delta: NOT part of the answer
    _chunk(content="Hel"),
    _chunk(content=None),                      # keep-alive / role-only delta
    _chunk(content="lo", finish="stop"),
    _chunk(usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
]


async def test_gateway_path_streams_and_joins_content_deltas(tmp_path, monkeypatch):
    _gateway_mode(monkeypatch)
    created = _install_fake_stream_client(monkeypatch, lambda key: list(_HELLO_STREAM))
    pool = KeyPool(tmp_path / "unused.json")
    out = await pool.complete("ping")
    assert out == "Hello"
    assert len(created) == 1
    assert created[0]["stream"] is True
    assert created[0]["stream_options"] == {"include_usage": True}


async def test_gateway_stream_usage_chunk_reaches_llmtok(tmp_path, monkeypatch, caplog):
    _gateway_mode(monkeypatch)
    _install_fake_stream_client(monkeypatch, lambda key: list(_HELLO_STREAM))
    pool = KeyPool(tmp_path / "unused.json")
    with caplog.at_level("INFO", logger="papervault.knowledge.store.llm"):
        await pool.complete("ping")
    tok = [r.getMessage() for r in caplog.records if r.getMessage().startswith("LLMTOK ")]
    assert len(tok) == 1
    assert "ptok=10 ctok=2 ttok=12" in tok[0]


async def test_stream_kill_switch_falls_back_to_plain_completion(tmp_path, monkeypatch):
    _gateway_mode(monkeypatch)
    monkeypatch.setattr(llm_mod, "_STREAM", False)
    created = _install_fake_stream_client(monkeypatch, lambda key: "plain-ok")
    pool = KeyPool(tmp_path / "unused.json")
    assert await pool.complete("ping") == "plain-ok"
    assert "stream" not in created[0] and "stream_options" not in created[0]


async def test_caller_stream_kwarg_never_leaks_a_stream_object(tmp_path, monkeypatch):
    """The contract returns str. A caller-supplied stream=False must not switch the wrapper
    back to the 120 s-vulnerable plain call, and stream=True must not hand back an iterator."""
    _gateway_mode(monkeypatch)
    created = _install_fake_stream_client(monkeypatch, lambda key: list(_HELLO_STREAM))
    pool = KeyPool(tmp_path / "unused.json")
    assert await pool.complete("ping", stream=False) == "Hello"
    assert await pool.complete("ping", stream=True) == "Hello"
    assert all(c["stream"] is True for c in created)


_MIDSTREAM_ERRORS = [
    # What openai's AsyncStream actually raises while ITERATING (it reads response.aiter_bytes()
    # directly; only request setup is normalised to APIConnectionError/APITimeoutError):
    httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
    httpx.ReadTimeout("timed out"),
    openai.APIError("upstream error event", httpx.Request("POST", "http://gw.example/v1/chat/completions"), body=None),
]


@pytest.mark.parametrize("err", _MIDSTREAM_ERRORS, ids=lambda e: type(e).__name__)
async def test_gateway_midstream_failure_becomes_stream_truncated(tmp_path, monkeypatch, caplog, err):
    """A failure while iterating the stream is normalised to StreamTruncated (cause chained) so
    every caller's classifier sees ONE transient type instead of status-less httpx/APIError
    instances; the gateway path still logs the attempt-fail before re-raising."""
    _gateway_mode(monkeypatch)
    _install_fake_stream_client(monkeypatch, lambda key: [_chunk(content="par"), err])
    pool = KeyPool(tmp_path / "unused.json")
    with caplog.at_level("WARNING", logger="papervault.knowledge.store.llm"):
        with pytest.raises(llm_mod.StreamTruncated) as ei:
            await pool.complete("ping")
    assert ei.value.__cause__ is err
    assert any("attempt-fail" in r.getMessage() and "err=StreamTruncated" in r.getMessage()
               for r in caplog.records)


async def test_request_time_error_is_not_wrapped(tmp_path, monkeypatch, caplog):
    """An error raised by create() itself (before any chunk) keeps its type and status — the
    401/403 auto-disable and the 429/5xx routing depend on it."""
    _gateway_mode(monkeypatch)
    _install_fake_stream_client(monkeypatch, lambda key: _FakeError(429))
    pool = KeyPool(tmp_path / "unused.json")
    with caplog.at_level("WARNING", logger="papervault.knowledge.store.llm"):
        with pytest.raises(_FakeError):
            await pool.complete("ping")
    assert any("code=429" in r.getMessage() for r in caplog.records)


async def test_direct_pool_streams_and_fails_over_on_midstream_error(tmp_path, monkeypatch):
    """The direct KeyPool path streams too, and a stream that dies mid-way is a TRANSIENT
    failure: fail over to the next key, write nothing."""
    monkeypatch.setattr(config, "USE_GATEWAY", False)
    f = tmp_path / "keys.json"
    _write(f, [_group("dies"), _group("good")])
    _pin_order(monkeypatch)

    def behavior(key):
        return ([_chunk(content="par"), httpx.RemoteProtocolError("peer closed connection")]
                if key == "dies" else list(_HELLO_STREAM))

    created = _install_fake_stream_client(monkeypatch, behavior)
    pool = KeyPool(f)
    assert await pool.complete("ping") == "Hello"
    assert [c["stream"] for c in created] == [True, True]
    assert all("disabled" not in g for g in json.loads(f.read_text()))


async def test_stream_cut_before_finish_is_an_error_not_an_empty_answer(tmp_path, monkeypatch, caplog):
    """A stream that ends WITHOUT any finish_reason was truncated upstream (the relay's cut, a
    dropped connection). Returning "" would be a silent success that the callers' retry /
    fail-over can never see — it must raise like any other failed attempt."""
    _gateway_mode(monkeypatch)
    _install_fake_stream_client(monkeypatch, lambda key: [_chunk(reasoning="hmm"), _chunk(content="par")])
    pool = KeyPool(tmp_path / "unused.json")
    with caplog.at_level("WARNING", logger="papervault.knowledge.store.llm"):
        with pytest.raises(llm_mod.StreamTruncated):
            await pool.complete("ping")
    assert any("attempt-fail" in r.getMessage() for r in caplog.records)


async def test_stream_that_finishes_with_empty_content_is_a_legit_empty_answer(tmp_path, monkeypatch):
    """finish_reason present + no content = the model genuinely answered nothing (same as the plain
    call's `content: None` → ""); NOT a truncation."""
    _gateway_mode(monkeypatch)
    _install_fake_stream_client(monkeypatch, lambda key: [_chunk(reasoning="hmm"), _chunk(content=None, finish="stop")])
    pool = KeyPool(tmp_path / "unused.json")
    assert await pool.complete("ping") == ""


async def test_stream_collects_only_choice_index_0_when_n_gt_1(tmp_path, monkeypatch):
    """The plain path returns choices[0] only; with n>1 a stream interleaves the alternatives'
    chunks, so the join must keep index 0 and drop the rest instead of concatenating them."""
    _gateway_mode(monkeypatch)
    _install_fake_stream_client(monkeypatch, lambda key: [
        _chunk(content="A1", index=0), _chunk(content="B1", index=1),
        _chunk(content="A2", index=0, finish="stop"), _chunk(content="B2", index=1, finish="stop"),
        _chunk(usage={"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5}),
    ])
    pool = KeyPool(tmp_path / "unused.json")
    assert await pool.complete("ping", n=2) == "A1A2"


async def test_reserved_stream_keys_inside_extra_body_are_stripped(tmp_path, monkeypatch):
    """The SDK merges extra_body OVER the request fields, so a caller could re-enable the plain
    call (or drop the usage chunk) through it. The override is unconditional: reserved keys are
    removed from a COPY of extra_body; the caller's other keys and dict are untouched."""
    _gateway_mode(monkeypatch)
    created = _install_fake_stream_client(monkeypatch, lambda key: list(_HELLO_STREAM))
    pool = KeyPool(tmp_path / "unused.json")
    body = {"stream": False, "stream_options": {"include_usage": False}, "keep": 1}
    assert await pool.complete("ping", extra_body=body) == "Hello"
    assert created[0]["extra_body"] == {"keep": 1}
    assert created[0]["stream"] is True and created[0]["stream_options"] == {"include_usage": True}
    assert body == {"stream": False, "stream_options": {"include_usage": False}, "keep": 1}


async def test_error_after_finish_reason_keeps_the_complete_answer(tmp_path, monkeypatch, caplog):
    """The collector keeps reading after the finish chunk to pick up the usage chunk. A failure
    in THAT tail (between the terminal answer chunk and usage/[DONE]) must not discard an answer
    that is already complete — it returns the answer with usage unavailable and logs it."""
    _gateway_mode(monkeypatch)
    _install_fake_stream_client(monkeypatch, lambda key: [
        _chunk(content="Hel"), _chunk(content="lo", finish="stop"),
        httpx.RemoteProtocolError("peer closed connection before the usage chunk"),
    ])
    pool = KeyPool(tmp_path / "unused.json")
    with caplog.at_level("INFO", logger="papervault.knowledge.store.llm"):
        assert await pool.complete("ping") == "Hello"
    tok = [r.getMessage() for r in caplog.records if r.getMessage().startswith("LLMTOK ")]
    assert len(tok) == 1 and "ptok=? ctok=? ttok=?" in tok[0]
    assert any("after finish_reason" in r.getMessage() for r in caplog.records)
