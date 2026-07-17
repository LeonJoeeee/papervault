"""MiMo (Xiaomi) LLM wrapper for LightRAG — central hot-reloaded key pool (2026-06-02).

pl and KS share ONE central key file (``PAPERVAULT_LLM_KEYS`` / legacy
``LLM_KEYS_FILE``, default ``<data>/llm_keys.json``) — a JSON array of
``{model, api_key, base_url}`` groups, optionally ``disabled``. Each ``mimo_complete``:
  * HOT-READS the file if its mtime changed (edit the file → next call sees it, no
    restart — add a group to add a key, delete/``disabled`` a group to drop one);
  * uses only ACTIVE groups (has ``api_key`` AND no ``disabled``), ``random.shuffle``s them
    per request and tries them in order, returning the first success — per-request random
    load-balancing, NO cross-request state (no round-robin counter, no cooldown);
  * routes failures by HTTP status: 429 / timeout / 5xx = TRANSIENT → just fail over;
    401 / 403 = PERMANENT (credential rejected) → on the FIRST occurrence AUTO-DISABLES
    that key by writing ``disabled: {at, code, reason}`` into the file (file-locked +
    atomic), so it (and pl, reading the same file) skip it thereafter.
On top of the per-request shuffle sits a small ``_MAX_ROUNDS`` retry-with-backoff (LightRAG
hammers this and has a 360s worker timeout). If the key file is missing/broken it falls back
to the old ``MIMO_API_KEY_*`` env (kept for config-slip resilience).

pl ALSO writes this file: we use the SAME lock filename (``<file>.lock``) and the SAME
``disabled`` schema so concurrent writes from both processes are safe + mutually understood.

Gateway mode (Phase 1B, ``KS_USE_GATEWAY=1``, DEFAULT OFF — docs/history/2026-06-04-gateway/2026-06-04-llm-gateway.md §9b):
when set, ``mimo_complete`` instead makes a SINGLE call to the running LiteLLM proxy (``KS_GATEWAY_URL``,
default ``http://127.0.0.1:4000/v1``, with virtual key ``KS_VIRTUAL_KEY``). The proxy owns the real
keys + failover/retry/cooldown/401-disable, so there is NO shuffle / _MAX_ROUNDS — just one proxy
call, bounded by the caller's outer deadline. The default (unset / "0") path is BYTE-FOR-BYTE the
KeyPool behavior below; gateway mode is a thin, reversible branch (unset env + restart to fall back).

Interface compatible with LightRAG's llm_model_func signature (DO NOT CHANGE):
    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from papervault import config

logger = logging.getLogger(__name__)

# Central key file (shared with the library plane). Path + provider settings come from
# papervault.config (one .env). The key-pool file is a generic {model, api_key, base_url}
# array — nothing here is tied to a specific provider.
_KEYS_FILE = config.LLM_KEYS_FILE

# A litellm-form model may carry a provider prefix ("openai/<model>"); the raw OpenAI SDK
# (what the knowledge plane uses) wants the bare name, so we strip a leading provider prefix.
_DEFAULT_MODEL = config.SYNTH_MODEL
_DEFAULT_BASE = config.LLM_BASE_URL

# Permanent (auto-disable) vs transient (just fail over) HTTP statuses.
_PERMANENT_CODES = {401, 403}

# Retry the full (re-shuffled) active set this many times, with exponential backoff (+ jitter)
# between rounds, before giving up — guards a transient all-endpoints-busy / 429 moment. Sits ON
# TOP of the per-request random.shuffle selection (NOT the old sequential round-robin counter).
# NOTE (gateway must-fix, docs/history/2026-06-04-gateway/2026-06-04-llm-gateway.md): the per-attempt timeout is now the
# CALLER's per-request `timeout=` (or the long _CLIENT_TIMEOUT backstop), and the REAL LightRAG
# extraction-worker cap is 2×default_llm_timeout = 480s (default 240), NOT the 360s an older
# comment claimed. Making `_MAX_ROUNDS × per_attempt + backoff < 480s` a proven invariant for the
# extraction path is tracked as a gateway must-fix; for now extraction passes no per-attempt
# timeout and relies on LightRAG's worker to cancel (which mimo_complete re-raises cleanly).
_MAX_ROUNDS = 2
_BASE_BACKOFF = 2.0

# Per-attempt observability (2026-06-03). A successful call slower than this is logged (INFO) so
# slow-under-contention calls are visible BEFORE they trip an outer worker timeout; failures +
# cancellations are always logged (WARNING). Grep `mimo ` to see them. env-tunable threshold.
_SLOW_CALL_S = float(os.getenv("KS_SLOW_CALL_LOG_S", "60"))

# Transport-level backstop (Phase 0, 2026-06-04 — see docs/history/2026-06-04-gateway/2026-06-04-llm-gateway.md). The
# AsyncOpenAI client timeout is a LONG dead-connection catch (~15min), NOT a business deadline:
# the SAME client serves short extraction AND a 16k-token synth, so a one-size-short value (the
# old hardcoded 120s) killed every long generation 100%. Per-call BUSINESS deadlines belong to
# the caller — passed as `timeout=` (already whitelisted → forwarded to .create()) or enforced
# by an outer asyncio.wait_for (LightRAG worker for extraction, synth_answer's timeout_s). env-tunable.
_CLIENT_TIMEOUT = float(os.getenv("KS_LLM_CLIENT_TIMEOUT", "900"))
# Explicit output ceiling at MiMo's max (128K = 131072; validated accepted). max_tokens is a CEILING,
# not a target → FREE for normal calls (a dense extraction finishes in ~hundreds of tokens, finish=
# stop). This is a DETERMINISTIC SAFETY VALUE, not a fix for an observed bug: we did NOT observe any
# production output-truncation — the build log has 0 `finish_reason` entries, and MiMo's unset default
# is already far above any extraction's reasoning+answer (model max is 128K; a sane default ≫ 20K). The
# reasoning-eats-budget failure mode is real ONLY at a tiny cap (probed: max_tokens=30 → 29 reasoning,
# 0 answer, finish=length) and does NOT occur at the default. We pin the cap explicitly so it can never
# silently bite regardless of MiMo's default; a genuine runaway would be caught by the gateway per-
# attempt timeout (loud 408 → retry). setdefault in complete() → explicit callers (e.g. synth) win.
_DEFAULT_MAX_TOKENS = int(os.getenv("KS_LLM_MAX_TOKENS", "131072"))
# Jitter added to the inter-round backoff so concurrent workers don't retry in lockstep against
# the same endpoints (thundering-herd / retry-storm guard — the #1 naive-retry mistake).
_BACKOFF_JITTER = float(os.getenv("KS_LLM_BACKOFF_JITTER", "0.75"))
# GATEWAY-path only: when the proxy itself is UNREACHABLE (down/restarting), back off + wait for it
# to return instead of fast-failing (else a brief gateway restart = thundering herd of doc failures).
# Distinct from no-retry-on-slow-key: only APIConnectionError triggers this; timeout/4xx/5xx still surface.
_GW_CONN_MAX_RETRIES = int(os.getenv("KS_GATEWAY_CONN_RETRIES", "6"))
_GW_CONN_BACKOFF_CAP = float(os.getenv("KS_GATEWAY_CONN_BACKOFF_CAP", "15"))


def _endpoint_label(g: dict) -> str:
    """Short, secret-free endpoint id for logs: last-4 of the key + base-url host (no key body)."""
    k = (g.get("api_key") or "")
    host = (g.get("base_url") or "").split("//", 1)[-1].split("/", 1)[0]
    return f"…{k[-4:]}@{host}"

# OpenAI SDK ChatCompletions.create() accepted kwargs (whitelist).
# LightRAG passes its own internal kwargs + MiMo-specific kwargs (e.g. enable_cot)
# that openai SDK rejects. Whitelist is safer than blacklist.
_OPENAI_STANDARD_KWARGS = {
    "model",
    "messages",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "temperature",
    "tools",
    "tool_choice",
    "top_logprobs",
    "top_p",
    "user",
    "extra_headers",
    "extra_query",
    "extra_body",
    "timeout",
}

# MiMo-specific kwargs (LightRAG may pass these) — forward via extra_body.
# "thinking" + "reasoning_effort" added 2026-07-16 (issue #3): litellm's xiaomi_mimo
# transformation natively supports both (probed live per its own docstring), but this
# whitelist silently dropped them one layer earlier — a dead-end for any graduated
# reasoning-effort lever. Inert until a caller passes them.
_MIMO_EXTRA_KWARGS = {"enable_cot", "enable_thinking", "thinking_mode", "thinking", "reasoning_effort"}


def _env_fallback_groups() -> list[dict]:
    """Single-endpoint fallback (``PAPERVAULT_LLM_API_KEY`` + base + model) — used ONLY when
    the key-pool file is missing/broken, so the pipeline never hard-fails on a config slip.
    The key-pool file is the primary, multi-key path."""
    if config.LLM_API_KEY:
        return [{
            "model": _DEFAULT_MODEL,
            "api_key": config.LLM_API_KEY,
            "base_url": config.LLM_BASE_URL,
        }]
    return []


def _sdk_model(model: str | None) -> str:
    """Map a shared-file ``model`` (litellm form ``openai/mimo-v2.5-pro``) to the bare model
    name the raw OpenAI SDK / MiMo gateway expects (``mimo-v2.5-pro``)."""
    m = model or _DEFAULT_MODEL
    if "/" in m:
        m = m.split("/", 1)[1]
    return m


def _error_code(exc: Exception) -> int | None:
    """Best-effort HTTP status from an openai/httpx exception; None if unknown.
    Falls back to the type name (``*AuthenticationError*`` → 401)."""
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    if code is None and "authentication" in type(exc).__name__.lower():
        return 401
    try:
        return int(code) if code is not None else None
    except (ValueError, TypeError):
        return None


class KeyPool:
    """Central hot-reloaded MiMo key pool (see module docstring). Mirrors paper-library's
    ``KeyPool``: mtime-cached hot-read, active-only selection, 401/403 auto-disable written
    back to the shared file with the SAME lock filename + ``disabled`` schema pl uses."""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._mtime: float | None = -1.0   # sentinel: force the first load
        self._groups: list[dict] = []
        self._clients: dict[tuple, AsyncOpenAI] = {}
        # Gateway-mode (KS_USE_GATEWAY=1) cache: ONE AsyncOpenAI pointing at the LiteLLM proxy
        # (vs the per-(base,key) self._clients above). Lazily built on first gateway call.
        self._gw_client: AsyncOpenAI | None = None
        self._reload_if_changed()

    # ---- hot-read (mtime-cached) ----
    def _reload_if_changed(self) -> None:
        try:
            mtime: float | None = self._path.stat().st_mtime
        except OSError:
            mtime = None
        if mtime == self._mtime:
            return
        with self._lock:
            if mtime == self._mtime:
                return
            self._groups = self._read_active(mtime)
            self._mtime = mtime

    def _read_active(self, mtime: float | None) -> list[dict]:
        if mtime is None:
            groups = _env_fallback_groups()
            if not groups:
                logger.error("LLM key file %s missing and no PAPERVAULT_LLM_API_KEY fallback",
                             self._path)
            return groups
        try:
            data = json.loads(self._path.read_text())
            if not isinstance(data, list):
                raise ValueError("key file is not a JSON array")
        except Exception as e:  # noqa: BLE001
            logger.error("LLM key file %s unreadable (%s); falling back to env", self._path, e)
            return _env_fallback_groups()
        active = [g for g in data
                  if isinstance(g, dict) and g.get("api_key") and not g.get("disabled")]
        if not active:
            logger.error("LLM key file %s has no active (non-disabled) keys", self._path)
        return active

    def _client(self, g: dict) -> AsyncOpenAI:
        # NB: read AsyncOpenAI off the module (not the import binding) so probe_throughput.py's
        # monkeypatch of ``store.llm.AsyncOpenAI`` is honored.
        import papervault.knowledge.store.llm as _self_mod
        base = g["base_url"]
        key = g["api_key"]
        cache_key = (base, key)
        c = self._clients.get(cache_key)
        if c is None:
            # max_retries=0: the OpenAI SDK's built-in retry (default 2) would retry the SAME endpoint
            # and MULTIPLY the per-attempt timeout (2-3× _CLIENT_TIMEOUT → blows the LightRAG 480s
            # worker on a slow endpoint). We do our OWN cross-key failover (shuffle/_MAX_ROUNDS), so
            # the SDK must NOT retry — one try per endpoint, then mimo_complete fails over. (2026-06-04)
            c = _self_mod.AsyncOpenAI(api_key=key, base_url=base, timeout=_CLIENT_TIMEOUT, max_retries=0)
            self._clients[cache_key] = c
        return c

    def _gateway_client(self) -> AsyncOpenAI:
        # Gateway mode (KS_USE_GATEWAY=1): ONE cached client pointing at the LiteLLM proxy. The
        # proxy holds the real keys + does failover/retry/cooldown/401-disable, so KS just talks
        # to it with a virtual key. base_url/key are read on first use (env-overridable); same
        # _CLIENT_TIMEOUT transport backstop as the direct path. Read AsyncOpenAI off the module
        # (not the import binding) so a monkeypatch of ``store.llm.AsyncOpenAI`` is honored.
        import papervault.knowledge.store.llm as _self_mod
        c = self._gw_client
        if c is None:
            base = config.GATEWAY_URL
            key = config.GATEWAY_KEY
            # max_retries=0: the OpenAI SDK's built-in retry (default 2) would retry the SAME endpoint
            # and MULTIPLY the per-attempt timeout (2-3× _CLIENT_TIMEOUT → blows the LightRAG 480s
            # worker on a slow endpoint). We do our OWN cross-key failover (shuffle/_MAX_ROUNDS), so
            # the SDK must NOT retry — one try per endpoint, then mimo_complete fails over. (2026-06-04)
            c = _self_mod.AsyncOpenAI(api_key=key, base_url=base, timeout=_CLIENT_TIMEOUT, max_retries=0)
            self._gw_client = c
        return c

    @property
    def active_groups(self) -> list[dict]:
        """The current ACTIVE groups (hot-reloaded). Snapshot copy; caller may shuffle."""
        self._reload_if_changed()
        return list(self._groups)

    # ---- the call: per-request random shuffle, stateless failover, on top a backoff loop ----
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        # Pin an explicit output ceiling at MiMo's max (deterministic safety value, NOT a fix for an
        # observed bug — see _DEFAULT_MAX_TOKENS). Single chokepoint for BOTH gateway + direct paths;
        # an explicit caller max_tokens (e.g. synth) wins via setdefault. Applies to extraction too,
        # which uses the RAW mimo_complete (LightRAG's llm_model_kwargs only reaches the query path).
        kwargs.setdefault("max_tokens", _DEFAULT_MAX_TOKENS)
        # Phase 1B (KS_USE_GATEWAY=1, default OFF — docs/history/2026-06-04-gateway/2026-06-04-llm-gateway.md §9b): route to
        # the LiteLLM proxy instead of the in-process KeyPool. The proxy owns failover/retry/
        # cooldown/401-disable, so gateway mode is a SINGLE call — no shuffle, no _MAX_ROUNDS. The
        # default (OFF) path below is UNCHANGED. Reversible: unset the env + restart to fall back.
        if config.USE_GATEWAY:
            return await self._complete_via_gateway(
                prompt, system_prompt=system_prompt,
                history_messages=history_messages, **kwargs,
            )
        self._reload_if_changed()
        base_groups = list(self._groups)
        if not base_groups:
            raise RuntimeError(
                f"No active LLM keys (file={self._path}; no PAPERVAULT_LLM_API_KEY fallback)."
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Whitelist openai-standard kwargs; MiMo-specific kwargs route to extra_body;
        # everything else (LightRAG internals like keyword_extraction, hashing_kv) drop.
        openai_kwargs: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in _OPENAI_STANDARD_KWARGS:
                openai_kwargs[k] = v
            elif k in _MIMO_EXTRA_KWARGS:
                extra_body[k] = v
            # else: drop silently (LightRAG internal kwarg)
        if extra_body:
            existing = openai_kwargs.get("extra_body", {})
            if isinstance(existing, dict):
                existing.update(extra_body)
                openai_kwargs["extra_body"] = existing
            else:
                openai_kwargs["extra_body"] = extra_body
        model_override = openai_kwargs.pop("model", None)

        last_error: Exception | None = None
        for round_idx in range(_MAX_ROUNDS):
            groups = list(base_groups)
            random.shuffle(groups)   # per-request random order; NO cross-request state
            for g in groups:
                model = _sdk_model(model_override or g.get("model"))
                ep = _endpoint_label(g)
                t0 = time.monotonic()
                try:
                    client = self._client(g)
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **openai_kwargs,
                    )
                    dt = time.monotonic() - t0
                    if dt >= _SLOW_CALL_S:
                        # Slow but OK — the call we want to see before an outer worker timeout kills it.
                        logger.info("mimo slow-ok %.0fs endpoint=%s round=%d", dt, ep, round_idx)
                    return resp.choices[0].message.content or ""
                except asyncio.CancelledError:
                    # Killed from OUTSIDE (e.g. LightRAG's worker timeout cancels this still-running
                    # call). This is the slow-call-that-got-axed case — log WHERE + how long it ran,
                    # then RE-RAISE (never swallow cancellation).
                    logger.warning("mimo CANCELLED after %.0fs endpoint=%s round=%d "
                                   "(outer timeout / shutdown — call did not finish)",
                                   time.monotonic() - t0, ep, round_idx)
                    raise
                except Exception as e:  # noqa: BLE001 — fail over; auto-disable only on 401/403
                    dt = time.monotonic() - t0
                    last_error = e
                    code = _error_code(e)
                    # Always log the attempt outcome: elapsed + endpoint + HTTP code + error type,
                    # so a slow/timeout-prone build is diagnosable (429 rate-limit vs hung endpoint
                    # vs slow generation, and on which endpoint). Additive only — flow unchanged.
                    logger.warning("mimo attempt-fail %.0fs endpoint=%s code=%s err=%s round=%d",
                                   dt, ep, code, type(e).__name__, round_idx)
                    if code in _PERMANENT_CODES:
                        # PERMANENT: disable on first occurrence + drop from this call's pool.
                        await self._disable_key(g["api_key"], code, type(e).__name__)
                        base_groups = [x for x in base_groups
                                       if x.get("api_key") != g["api_key"]]
                    # else TRANSIENT (429 / timeout / 5xx): just fail over, nothing remembered.
                    continue
            if not base_groups:
                break
            if round_idx < _MAX_ROUNDS - 1:
                await asyncio.sleep(_BASE_BACKOFF * (2 ** round_idx) + random.uniform(0, _BACKOFF_JITTER))

        raise RuntimeError(
            f"All active MiMo keys failed after {_MAX_ROUNDS} rounds. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )

    # ---- gateway mode (KS_USE_GATEWAY=1): ONE call to the LiteLLM proxy, no shuffle/rounds ----
    async def _complete_via_gateway(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        """Thin path: build the SAME messages + apply the SAME kwarg whitelist / extra_body
        routing / silent-drop as ``complete()``, then make a SINGLE ``chat.completions.create``
        against the proxy. The PROXY does failover/retry/cooldown, so there is NO shuffle and NO
        ``_MAX_ROUNDS`` here — the caller's outer deadline (LightRAG worker / synth ``wait_for``)
        governs. Preserves the contract: ``CancelledError`` re-raised, str return (None→"")."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Whitelist openai-standard kwargs; MiMo-specific kwargs route to extra_body; everything
        # else (LightRAG internals) drop — IDENTICAL to the direct path so the contract is one.
        openai_kwargs: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in _OPENAI_STANDARD_KWARGS:
                openai_kwargs[k] = v
            elif k in _MIMO_EXTRA_KWARGS:
                extra_body[k] = v
            # else: drop silently (LightRAG internal kwarg)
        if extra_body:
            existing = openai_kwargs.get("extra_body", {})
            if isinstance(existing, dict):
                existing.update(extra_body)
                openai_kwargs["extra_body"] = existing
            else:
                openai_kwargs["extra_body"] = extra_body
        # The model is the gateway GROUP name, which (locked design) is the REAL model name:
        # default "mimo-v2.5-pro"; send the cheap group "mimo-v2.5" ONLY when a caller explicitly
        # asked cheap (model="mimo-v2.5" or the legacy alias "mimo-cheap"). Ignore the per-key
        # _sdk_model map — the proxy owns key→deployment selection.
        requested = openai_kwargs.pop("model", None)
        # Strip any provider prefix (openai/<name>) before hitting the proxy: the gateway's
        # model GROUP is the bare real name, and PAPERVAULT_MODEL may legitimately carry a
        # litellm-form prefix (the library plane builds one). Mirrors the direct path's _sdk_model.
        model = _sdk_model(config.BUILD_MODEL if requested in (config.BUILD_MODEL, "cheap") else config.SYNTH_MODEL)

        # Gateway owns retry + timeout (config.yaml request_timeout/num_retries/retry_policy), so
        # send NO per-request retry/timeout to the proxy. synth.py / multiquery.py pass timeout=…
        # for the DIRECT KeyPool path; drop it (and any header kwarg) on the gateway path. The outer
        # asyncio.wait_for at those call sites still enforces the caller deadline locally.
        openai_kwargs.pop("timeout", None)
        openai_kwargs.pop("extra_headers", None)

        ep = f"gateway:{model}"
        conn_try = 0
        while True:
            t0 = time.monotonic()
            try:
                client = self._gateway_client()
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **openai_kwargs,
                )
                dt = time.monotonic() - t0
                if dt >= _SLOW_CALL_S:
                    # Slow but OK — the call we want to see before an outer worker timeout kills it.
                    logger.info("mimo slow-ok %.0fs endpoint=%s", dt, ep)
                return resp.choices[0].message.content or ""
            except asyncio.CancelledError:
                # Killed from OUTSIDE (LightRAG worker timeout / synth wait_for / shutdown). Log
                # WHERE + how long, then RE-RAISE (never swallow cancellation).
                logger.warning("mimo CANCELLED after %.0fs endpoint=%s "
                               "(outer timeout / shutdown — call did not finish)",
                               time.monotonic() - t0, ep)
                raise
            except APIConnectionError as e:
                # APITimeoutError SUBCLASSES APIConnectionError (openai/_exceptions.py): a per-request/
                # transport timeout = a slow/stuck call (gateway stuck > client timeout), NOT the gateway
                # being unreachable. Re-raise it — conn-backoff is ONLY for "can't connect", else we'd
                # retry a slow call 6× and break no-retry-on-slow-key.
                if isinstance(e, APITimeoutError):
                    raise
                # The GATEWAY ITSELF is unreachable (down/restarting) — NOT a slow key. Back off and
                # WAIT for it to return instead of fast-failing. Without this, a brief gateway restart
                # is a thundering herd: every worker instant-fails (conn refused, 0s) and churns docs,
                # nuking hundreds. Only APIConnectionError triggers this; a timeout/4xx/5xx still
                # surfaces below (preserve no-retry-on-slow-key). Jittered exp backoff so the workers
                # don't all reconnect in lockstep the instant the gateway is back.
                conn_try += 1
                if conn_try > _GW_CONN_MAX_RETRIES:
                    logger.warning("mimo gateway UNREACHABLE after %d conn-retries endpoint=%s — giving up",
                                   conn_try - 1, ep)
                    raise
                delay = min(2.0 ** (conn_try - 1), _GW_CONN_BACKOFF_CAP) * (0.5 + random.random())
                logger.warning("mimo gateway unreachable (conn err) endpoint=%s — backoff %.1fs (try %d/%d)",
                               ep, delay, conn_try, _GW_CONN_MAX_RETRIES)
                await asyncio.sleep(delay)
                continue
            except Exception as e:  # noqa: BLE001 — proxy owns failover; surface its final error
                dt = time.monotonic() - t0
                code = _error_code(e)
                logger.warning("mimo attempt-fail %.0fs endpoint=%s code=%s err=%s",
                               dt, ep, code, type(e).__name__)
                raise

    # ---- 401/403 auto-disable: the only persistent state, written to the shared file ----
    async def _disable_key(self, api_key: str, code: int, reason: str) -> None:
        """Write ``disabled: {at, code, reason}`` for this key. KS is async + this write is
        rare, so we hop to a thread (file lock + atomic replace are blocking I/O)."""
        await asyncio.to_thread(self._disable_key_sync, api_key, code, reason)
        # File changed → force a re-read on the next call (this process + pl both skip it).
        self._mtime = -1.0

    def _disable_key_sync(self, api_key: str, code: int, reason: str) -> None:
        try:
            try:
                from filelock import FileLock
                lock_ctx: Any = FileLock(str(self._path) + ".lock", timeout=10)
            except ImportError:
                lock_ctx = contextlib.nullcontext()
            with lock_ctx:
                data = json.loads(self._path.read_text())
                changed = False
                for g in data:
                    if (isinstance(g, dict) and g.get("api_key") == api_key
                            and not g.get("disabled")):
                        g["disabled"] = {
                            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "code": code,
                            "reason": reason,
                        }
                        changed = True
                if changed:
                    tmp = self._path.with_name(self._path.name + ".tmp")
                    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
                    os.replace(tmp, self._path)
                    logger.warning("auto-disabled LLM key …%s in %s (code=%s reason=%s)",
                                   api_key[-4:], self._path.name, code, reason)
        except Exception as e:  # noqa: BLE001 — disabling is best-effort; never crash the call
            logger.error("failed to auto-disable LLM key …%s: %s", api_key[-4:], e)


# One process-wide pool (hot-reloads internally; no cross-request state besides the file).
_pool: KeyPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> KeyPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = KeyPool(_KEYS_FILE)
    return _pool


def active_endpoint_count() -> int:
    """Number of ACTIVE groups in the central key file (or env fallback). Hot-reloaded.
    This is what ``CONFIG.mimo.valid_endpoint_count`` now reports."""
    return len(get_pool().active_groups)


def pool_model() -> str:
    """The bare model name the SDK should use, from the first active group (or env default).
    This is what ``CONFIG.mimo.model`` now reports (used as LightRAG's ``llm_model_name``)."""
    groups = get_pool().active_groups
    if groups:
        return _sdk_model(groups[0].get("model"))
    return _sdk_model(_DEFAULT_MODEL)


async def mimo_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs: Any,
) -> str:
    """Run a chat completion against the central MiMo key pool.

    LightRAG-compatible signature (DO NOT CHANGE). Per request: shuffles the ACTIVE keys and
    tries them in order, returning the first success; 401/403 auto-disables a key in the
    shared file; 429/timeout/5xx just fail over. Raises only if every active key fails.

    If ``KS_USE_GATEWAY=1`` (default off), routes a SINGLE call to the LiteLLM proxy instead
    (the proxy owns failover/retry/cooldown); same signature, whitelist, and str return.
    """
    return await get_pool().complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
