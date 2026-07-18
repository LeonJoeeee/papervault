"""LLM configuration — central hot-reloaded key pool (2026-06-02).

The XIAOMI/MiMo path reads a CENTRAL key file (``PAPERVAULT_LLM_KEYS`` / legacy
``LLM_KEYS_FILE``, default ``<data>/llm_keys.json``) shared with
KS — a JSON array of ``{model, api_key, base_url}`` groups. Each ``.call()``:
  * HOT-READS the file if its mtime changed (edit the file → next call sees it,
    no restart — add a group to add a key, delete/``disabled`` a group to drop one);
  * RANDOM-shuffles the ACTIVE (non-``disabled``) groups and tries them in order,
    returning the first success — per-request random load-balancing, NO
    cross-request state (no cooldown);
  * on a PERMANENT error (HTTP 401/403 = bad/expired credential) AUTO-DISABLES that
    key by writing ``disabled: {at, code, reason}`` into the file (file-locked +
    atomic), so it (and KS, reading the same file) skip it thereafter; a TRANSIENT
    error (429 / timeout / 5xx) just fails over with nothing remembered.
If the key file is missing/broken it falls back to the old ``XIAOMI_API_KEY*`` env.
"""

import json
import logging
import os
import random
import threading
import time
from pathlib import Path

from papervault import config
from papervault.llm_routing import route

logger = logging.getLogger(__name__)


def _litellm_model(model: str) -> str:
    """litellm needs a provider prefix to route a custom OpenAI-compatible endpoint.
    Bare model names get an ``openai/`` prefix; anything already prefixed is left as-is."""
    if not model:
        return model
    return model if "/" in model else f"openai/{model}"


class LLM:
    """Thin litellm-backed chat client (replaces the old ``crewai.LLM`` wrapper).

    pl only ever used ``crewai.LLM`` as a ``litellm.completion`` shim — never the
    Agent / Crew / Task framework — so crewai was pure dependency weight (and it
    hard-pinned ``json-repair==0.25.2``, blocking MinerU's ``>=0.46.2``). This
    class reproduces the EXACT surface the :class:`KeyPool` + gateway path depend
    on: construct with ``LLM(model=, base_url=, api_key=, max_tokens=,
    num_retries=)`` and ``.call(messages) -> str``.

    litellm raises typed exceptions that carry ``.status_code`` (e.g.
    ``AuthenticationError`` → 401), so the pool's 401/403 auto-disable + transient
    (429/5xx/timeout) failover in :meth:`KeyPool.call` are byte-unchanged.
    """

    def __init__(self, model, *, base_url=None, api_key=None, max_tokens=None,
                 num_retries=0, **_ignored):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.num_retries = num_retries

    def call(self, messages, **_ignored):
        import litellm
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resp = litellm.completion(
            model=self.model,
            messages=messages,
            base_url=self.base_url,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            num_retries=self.num_retries,
        )
        return resp.choices[0].message.content or ""

# Default well above litellm's small default so a long generation isn't truncated.
_DEFAULT_MAX_TOKENS = int(os.environ.get("PAPERVAULT_LIB_MAX_TOKENS", "32000"))
_DEFAULT_MODEL = _litellm_model(config.SYNTH_MODEL)
_DEFAULT_BASE = config.LLM_BASE_URL
_KEYS_FILE = config.LLM_KEYS_FILE

# Permanent (auto-disable) vs transient (just fail over) HTTP statuses.
_PERMANENT_CODES = {401, 403}


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


def _error_code(exc: Exception):
    """Best-effort HTTP status from an LLM/litellm exception; None if unknown.
    Falls back to the type name (``*AuthenticationError*`` → 401)."""
    code = getattr(exc, "status_code", None)
    if code is None and "authentication" in type(exc).__name__.lower():
        return 401
    try:
        return int(code) if code is not None else None
    except (ValueError, TypeError):
        return None


class KeyPool:
    """Central hot-reloaded MiMo key pool (see module docstring). ``.call`` mirrors
    :class:`LLM`.call (parse_intent / judge_* go through ``llm.call``); other attribute
    access delegates to a representative client."""

    def __init__(self, path, max_tokens: int, model_override: str | None = None):
        self._path = Path(path)
        self._max_tokens = max_tokens
        self._model_override = model_override   # force a specific model (else per-group/default)
        self._lock = threading.Lock()
        self._mtime = -1.0            # sentinel: force the first load
        self._groups: list[dict] = []
        self._clients: dict[tuple, LLM] = {}
        self._reload_if_changed()

    # ---- hot-read (mtime-cached) ----
    def _reload_if_changed(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            mtime = None
        if mtime == self._mtime:
            return
        with self._lock:
            if mtime == self._mtime:
                return
            self._groups = self._read_active(mtime)
            self._mtime = mtime

    def _read_active(self, mtime) -> list[dict]:
        if mtime is None:
            groups = _env_fallback_groups()
            if not groups:
                logger.error("LLM key file %s missing and no env fallback keys", self._path)
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

    def _client(self, g: dict) -> LLM:
        key = (self._model_override or g.get("model") or _DEFAULT_MODEL,
               g["base_url"], g["api_key"])
        c = self._clients.get(key)
        if c is None:
            c = LLM(model=key[0], base_url=key[1], api_key=key[2], max_tokens=self._max_tokens)
            self._clients[key] = c
        return c

    # ---- the call: random failover, stateless ----
    def call(self, *args, **kwargs):
        self._reload_if_changed()
        groups = list(self._groups)
        if not groups:
            raise RuntimeError(f"no active LLM keys (file={self._path})")
        random.shuffle(groups)                       # per-request random order; stateless
        last_exc: Exception | None = None
        for g in groups:
            try:
                return self._client(g).call(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 — fail over; auto-disable only on 401/403
                last_exc = e
                code = _error_code(e)
                if code in _PERMANENT_CODES:
                    self._disable_key(g["api_key"], code, type(e).__name__)
        raise last_exc                               # every active key failed this call

    # ---- 401 auto-disable: the only persistent state, written to the shared file ----
    def _disable_key(self, api_key: str, code: int, reason: str) -> None:
        try:
            import contextlib
            try:
                from filelock import FileLock
                lock_ctx = FileLock(str(self._path) + ".lock", timeout=10)
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
                            "code": code, "reason": reason,
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

    def __getattr__(self, name):
        if name.startswith("_") or "_path" not in self.__dict__:
            raise AttributeError(name)
        if self._groups:
            return getattr(self._client(self._groups[0]), name)
        raise AttributeError(name)


# One pool per distinct (max_tokens, model) combo, reused (hot-reloads internally).
_pools: dict[tuple, KeyPool] = {}
_pools_lock = threading.Lock()

# Gateway-mode (PAPERVAULT_LLM_GATEWAY=1) cache: one :class:`LLM` per (max_tokens, group) pointed at the
# LiteLLM proxy — same reuse shape as the direct-path pool cache. Phase 3, docs/history/2026-06-04-gateway/2026-06-04-llm-gateway.md.
_gw_llms: dict[tuple, "LLM"] = {}
_gw_lock = threading.Lock()


def _gateway_group(model: str | None) -> str:
    """Map a library call-site model to the LiteLLM proxy's GROUP name. The proxy exposes the
    real model names as groups; the ``openai/`` provider strips the prefix and sends the bare
    group name. Routing: an explicit request for the cheaper BUILD_MODEL routes there; anything
    else (default / None) routes to the strong SYNTH_MODEL — matching the direct path's default."""
    m = (model or "").split("/")[-1]
    if config.BUILD_MODEL and m == config.BUILD_MODEL:
        return _litellm_model(config.BUILD_MODEL)
    return _litellm_model(config.SYNTH_MODEL)


def _get_llm_via_gateway(mt: int, model: str | None):
    """PAPERVAULT_LLM_GATEWAY=1 path: an :class:`LLM` (litellm under the hood) pointed at the running LiteLLM
    proxy. The proxy holds the real MiMo keys + does failover/retry/cooldown/401-disable, so pl just
    talks to it with a virtual key. Same :class:`LLM` interface + max_tokens passthrough as the direct
    path, so callers (and their fail-open/closed exception handling) are byte-unchanged. Cached per
    (max_tokens, group), mirroring the direct pool cache. num_retries=0 so litellm does NOT stack its
    own retry on top of the proxy's failover (avoid double-retry)."""
    base = config.GATEWAY_URL
    key = config.GATEWAY_KEY
    group = _gateway_group(model)
    cache_key = (mt, group, base)
    with _gw_lock:
        c = _gw_llms.get(cache_key)
        if c is None:
            c = LLM(model=group, base_url=base, api_key=key, max_tokens=mt, num_retries=0)
            _gw_llms[cache_key] = c
        return c


def get_llm(*, max_tokens: int | None = None, model: str | None = None):
    """Shared LLM handle. XIAOMI/MiMo → the central hot-reloaded :class:`KeyPool`
    (random failover + 401 auto-disable); else OpenAI / Anthropic single client.

    ``model`` forces a specific model for this handle (e.g. a cheaper tier for a
    simple judge) while keeping the pool's key failover; default = per-group /
    ``_DEFAULT_MODEL``.

    Fallback (no ``PAPERVAULT_LLM_*`` key or pool set): a bare ``OPENAI_API_KEY``
    in the environment is used with ``OPENAI_MODEL`` (default ``openai/gpt-4o-mini``),
    then a bare ``ANTHROPIC_API_KEY`` with ``ANTHROPIC_MODEL`` (default
    ``anthropic/claude-sonnet-4-5``). NOTE: ``papervault doctor`` validates only the
    ``PAPERVAULT_LLM_*`` surface, NOT these fallbacks — a stray ``OPENAI_API_KEY`` in
    your shell silently routes library LLM calls to OpenAI while doctor stays green.

    Gateway mode (``PAPERVAULT_LLM_GATEWAY=1``, DEFAULT OFF — Phase 3, docs/history/2026-06-04-gateway/2026-06-04-llm-gateway.md):
    when set, return an :class:`LLM` pointed at the running LiteLLM proxy
    (``PAPERVAULT_LLM_GATEWAY_URL`` default ``http://127.0.0.1:4000/v1``, virtual key ``PAPERVAULT_LLM_GATEWAY_KEY``),
    with the model mapped to the proxy GROUP (default/None → ``mimo-v2.5-pro``; explicit cheap → ``mimo-v2.5``). The proxy owns the
    real keys + failover. The DEFAULT (OFF) path below is UNCHANGED."""
    mt = max_tokens or _DEFAULT_MAX_TOKENS
    # Default handle → the "judge" role (issue #8): library judges / intent parser / search all
    # ride get_llm() with no explicit model. Default routes to the SYNTH slot (today's behavior);
    # an operator can move them to a cheaper model via PAPERVAULT_LLM_JUDGE. An explicit model=
    # (the verify + extract-gate call sites) bypasses this. Normalize to litellm form; an empty
    # resolution → None → the pool's per-group model, byte-identical to today when nothing is set.
    if model is None:
        model = route("judge")[0]
    model = _litellm_model(model) or None
    if config.USE_GATEWAY:
        return _get_llm_via_gateway(mt, model)
    if _KEYS_FILE.exists() or config.LLM_API_KEY:
        with _pools_lock:
            pkey = (mt, model)
            pool = _pools.get(pkey)
            if pool is None:
                pool = KeyPool(_KEYS_FILE, mt, model_override=model)
                _pools[pkey] = pool
            return pool
    if os.environ.get("OPENAI_API_KEY"):
        return LLM(model=os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini"), max_tokens=mt)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLM(model=os.environ.get("ANTHROPIC_MODEL", "anthropic/claude-sonnet-4-5"),
                   max_tokens=mt)
    raise RuntimeError(
        "No LLM credentials. Set PAPERVAULT_LLM_API_KEY (+ PAPERVAULT_LLM_BASE_URL) or "
        "provide a key-pool file at PAPERVAULT_LLM_KEYS. See .env.example."
    )
