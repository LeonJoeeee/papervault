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

import contextlib
import contextvars
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from papervault import config
from papervault.llm_routing import (
    ModelTarget,
    clear_fallback_cooldowns as clear_fallback_cooldowns,
    format_targets,
    is_model_cooled_down,
    litellm_wire_kwargs,
    parse_targets,
    record_model_cooldown,
    route,
)
from papervault.llm_usage import log_usage

logger = logging.getLogger(__name__)


# Same transport policy as knowledge.store.llm: pre-first-chunk cutoff is disabled
# by default (0) so reasoning-heavy gateway calls are governed by _CLIENT_TIMEOUT_S;
# positive PAPERVAULT_LIB_FIRST_CHUNK_S opts into an early first-chunk deadline for
# short-idle relays. The absolute retry budget admits a retry only when its backoff
# and any positive first-chunk allowance still fit.
_FIRST_CHUNK_S = float(os.getenv("PAPERVAULT_LIB_FIRST_CHUNK_S", "0"))
_RETRY_BUDGET_S = float(os.getenv("PAPERVAULT_LIB_RETRY_BUDGET_S", "480"))
_CLIENT_TIMEOUT_S = 900.0


@dataclass(frozen=True)
class CallObservation:
    """Response disclosure, never a reconstruction from requested routing."""

    model: str
    provider: str
    latency_ms: float
    degraded: bool
    code: str | None = None


_call_observer = contextvars.ContextVar("library_llm_call_observer", default=None)


@contextlib.contextmanager
def observe_llm_calls(callback):
    """Observe calls in this execution context, including copied worker contexts."""
    token = _call_observer.set(callback)
    try:
        yield
    finally:
        _call_observer.reset(token)


def _disclosed(value):
    return value.strip()[:256] if isinstance(value, str) and value.strip() else "unknown"


class _DisclosureStream:
    """Read response fields before LiteLLM replaces model with a routing label."""

    def __init__(self, source, disclosure):
        self.source = source
        self.disclosure = disclosure

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self.source)
        for name in ("model", "provider"):
            value = chunk.get(name) if isinstance(chunk, dict) else getattr(chunk, name, None)
            if _disclosed(value) != "unknown":
                self.disclosure[name] = _disclosed(value)
        return chunk

    def __getattr__(self, name):
        return getattr(self.source, name)


class StreamTruncated(RuntimeError):
    """The provider did not finish its streamed answer."""


def _close_stream(stream):
    # LiteLLM's wrapper has no synchronous close; its underlying SDK stream does.
    target = getattr(stream, "completion_stream", stream)
    close = getattr(target, "close", None)
    if close is not None:
        with contextlib.suppress(Exception):
            close()


def _open_stream(kwargs):
    """Open a streamed completion and read its first chunk on a daemon worker.

    When ``_FIRST_CHUNK_S > 0``, completion creation plus first ``next()`` share one
    wall-clock deadline; when ``<= 0`` (default), no pre-first-chunk cutoff is applied
    and ``_CLIENT_TIMEOUT_S`` governs. An abandoned read closes its late stream itself
    so no executor shutdown can block the timeout or the retry.
    """
    import litellm

    ready = Future()

    def open_first():
        stream = None
        try:
            stream = litellm.completion(**kwargs)
            if hasattr(stream, "completion_stream"):
                # CustomStreamWrapper synthesizes chunk.model from the requested
                # route and custom_llm_provider from transport configuration.
                # Neither field proves what the endpoint actually served.
                disclosure = {}
                stream._papervault_response_disclosure = disclosure
                source = stream.completion_stream
                if source is not None and hasattr(source, "__next__"):
                    stream.completion_stream = _DisclosureStream(source, disclosure)
            iterator = iter(stream)
            first = next(iterator, None)
        except BaseException as exc:
            _close_stream(stream)
            with contextlib.suppress(InvalidStateError):
                ready.set_exception(exc)
        else:
            try:
                ready.set_result((stream, iterator, first))
            except InvalidStateError:
                _close_stream(stream)

    first_chunk_s = _FIRST_CHUNK_S
    deadline = time.monotonic() + first_chunk_s if first_chunk_s > 0 else None
    context = contextvars.copy_context()
    threading.Thread(target=context.run, args=(open_first,), daemon=True).start()
    handed_off = False
    try:
        timeout = max(0.0, deadline - time.monotonic()) if deadline is not None else None
        result = ready.result(timeout=timeout)
        handed_off = True
        return result
    except TimeoutError as exc:
        if deadline is None or ready.done():  # A timeout raised by the SDK itself keeps its type.
            raise
        raise StreamTruncated(f"no first chunk within {first_chunk_s:g}s") from exc
    finally:
        if not handed_off and not ready.cancel() and ready.exception() is None:
            # The worker won the race with cancellation: this caller owns cleanup.
            _close_stream(ready.result()[0])


def _create_completion(**kwargs):
    """Join answer deltas and usage, rejecting an EOF without a provider finish."""
    if kwargs.get("max_tokens") is None:
        kwargs.pop("max_tokens", None)
    stream, iterator, chunk = _open_stream({
        **kwargs, "stream": True, "stream_options": {"include_usage": True},
        "num_retries": 0, "timeout": _CLIENT_TIMEOUT_S,
    })
    parts = []
    usage = None
    finish = None
    model = provider = "unknown"
    disclosure = getattr(stream, "_papervault_response_disclosure", None)
    try:
        while chunk is not None:
            if disclosure is None:
                if _disclosed(getattr(chunk, "model", None)) != "unknown":
                    model = _disclosed(chunk.model)
                if _disclosed(getattr(chunk, "provider", None)) != "unknown":
                    provider = _disclosed(chunk.provider)
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            for choice in getattr(chunk, "choices", None) or []:
                if (getattr(choice, "index", 0) or 0) != 0:
                    continue
                piece = getattr(getattr(choice, "delta", None), "content", None)
                if piece:
                    parts.append(piece)
                reason = getattr(choice, "finish_reason", None)
                # CustomStreamWrapper invents stop at EOF. Only its received marker
                # proves the provider finished; raw SDK iterators use the chunk.
                if reason and getattr(stream, "received_finish_reason", reason):
                    finish = reason
            chunk = next(iterator, None)
    except Exception as exc:  # noqa: BLE001 — normalize transport/stream errors only
        code = _error_code(exc)
        if code is not None and 400 <= code < 500 and code not in {408, 429}:
            raise
        if finish is None:
            raise StreamTruncated(
                f"stream broke after {len(parts)} content pieces ({type(exc).__name__})"
            ) from exc
        logger.warning("library stream error after finish_reason=%s (%s); answer kept",
                       finish, type(exc).__name__)
    finally:
        _close_stream(stream)
    if finish is None:
        raise StreamTruncated(f"stream ended without a provider finish after {len(parts)} pieces")
    if disclosure is not None:
        model = disclosure.get("model", "unknown")
        provider = disclosure.get("provider", "unknown")
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="".join(parts)),
                                 finish_reason=finish)], usage=usage,
        model=model, provider=provider,
    )


def _is_transient(exc):
    import httpx
    import litellm

    code = _error_code(exc)
    if code is not None:
        return code in {408, 429} or 500 <= code <= 599
    return isinstance(exc, (StreamTruncated, TimeoutError, litellm.Timeout,
                            litellm.APIConnectionError, httpx.TransportError))


def _is_failover_eligible(exc: Exception) -> bool:
    import httpx
    import litellm
    from openai import APITimeoutError

    code = _error_code(exc)
    if code is not None:
        return code in {408, 429} or 500 <= code <= 599
    return isinstance(
        exc,
        (StreamTruncated, TimeoutError, APITimeoutError, litellm.Timeout, httpx.TimeoutException),
    )


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
    Calls stream internally; multi-target routes fail over immediately on transient
    primary errors with process-local cooldown, and single/final targets retry
    transient failures once within a retry admission budget.
    """

    def __init__(self, model, *, base_url=None, api_key=None, max_tokens=None,
                 num_retries=0, targets: tuple[ModelTarget, ...] | None = None,
                 **_ignored):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.targets = tuple(targets) if targets is not None else None

    def call(self, messages, **_ignored):
        started = time.monotonic()
        response = None
        degraded = True
        try:
            response, degraded = self._call(messages)
            return response.choices[0].message.content or ""
        finally:
            callback = _call_observer.get()
            if callback is not None:
                observation = CallObservation(
                    model=_disclosed(getattr(response, "model", None)),
                    provider=_disclosed(getattr(response, "provider", None)),
                    latency_ms=(time.monotonic() - started) * 1000,
                    degraded=degraded,
                    code="llm_call_failed" if response is None else None,
                )
                try:
                    callback(observation)
                except Exception:
                    logger.warning("LLM call observation unavailable")

    def _call(self, messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        t0 = time.monotonic()
        deadline = t0 + _RETRY_BUDGET_S
        mt_kw = {"max_tokens": self.max_tokens} if self.max_tokens is not None else {}
        if self.targets is None:
            for attempt in range(2):
                try:
                    resp = _create_completion(
                        model=self.model, messages=messages, base_url=self.base_url,
                        api_key=self.api_key, **mt_kw,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — retain pool status handling
                    delay = 0.5 + random.random()
                    remaining = deadline - time.monotonic()
                    if (attempt == 1 or not _is_transient(exc)
                            or remaining - delay <= max(_FIRST_CHUNK_S, 0.0)):
                        raise
                    logger.warning("library transient failure (%s); retrying once in %.1fs "
                                   "(%.1fs budget left)", type(exc).__name__, delay, remaining)
                    time.sleep(delay)
            # Bare model name (strip the litellm provider prefix) so pl and ks lines
            # aggregate under one label in per-model roll-ups.
            log_usage(logger, "pl", self.model.rsplit("/", 1)[-1], time.monotonic() - t0, resp)
            return resp, attempt > 0

        for target_idx, target in enumerate(self.targets):
            is_final = target_idx == len(self.targets) - 1
            if not is_final and is_model_cooled_down(target.model, now=time.monotonic()):
                continue
            wire = litellm_wire_kwargs(target)
            if not is_final:
                try:
                    resp = _create_completion(
                        **wire, messages=messages, base_url=self.base_url,
                        api_key=self.api_key, **mt_kw,
                    )
                    log_usage(logger, "pl", target.model, time.monotonic() - t0, resp)
                    return resp, target_idx > 0
                except Exception as exc:  # noqa: BLE001
                    if not _is_failover_eligible(exc):
                        raise
                    now = time.monotonic()
                    ttl = record_model_cooldown(target.model, now=now)
                    remaining = deadline - now
                    fallback_model = self.targets[target_idx + 1].model
                    code = _error_code(exc)
                    if remaining <= max(_FIRST_CHUNK_S, 0.0):
                        logger.warning(
                            "library primary failover skipped (budget exhausted: %.1fs left) "
                            "%s -> %s code=%s err=%s",
                            max(0.0, remaining), target.model, fallback_model,
                            code, type(exc).__name__,
                        )
                        raise
                    logger.warning(
                        "library primary failover %s -> %s code=%s err=%s cooldown=%.0fs",
                        target.model, fallback_model, code, type(exc).__name__, ttl,
                    )
                    continue

            for attempt in range(2):
                try:
                    resp = _create_completion(
                        **wire, messages=messages, base_url=self.base_url,
                        api_key=self.api_key, **mt_kw,
                    )
                    log_usage(logger, "pl", target.model, time.monotonic() - t0, resp)
                    return resp, (target_idx > 0 or attempt > 0)
                except Exception as exc:  # noqa: BLE001
                    delay = 0.5 + random.random()
                    remaining = deadline - time.monotonic()
                    if (attempt == 1 or not _is_transient(exc)
                            or remaining - delay <= max(_FIRST_CHUNK_S, 0.0)):
                        raise
                    logger.warning("library transient failure (%s); retrying once in %.1fs "
                                   "(%.1fs budget left)", type(exc).__name__, delay, remaining)
                    time.sleep(delay)
        raise AssertionError("unreachable: self.targets is non-empty")


def _optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


# Omit max_tokens by default (None) unless PAPERVAULT_LIB_MAX_TOKENS is explicitly set.
_DEFAULT_MAX_TOKENS: int | None = _optional_int_env("PAPERVAULT_LIB_MAX_TOKENS")
_DEFAULT_MODEL = litellm_wire_kwargs(parse_targets(route("judge"))[0])["model"]
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

    def __init__(
        self,
        path,
        max_tokens: int | None = None,
        model_override: str | None = None,
        *,
        targets: tuple[ModelTarget, ...] | None = None,
    ):
        self._path = Path(path)
        self._max_tokens = max_tokens
        self._model_override = model_override   # force a specific route/model (else per-group/default)
        self._targets = (
            tuple(targets)
            if targets is not None
            else (parse_targets(model_override) if model_override is not None else None)
        )
        self._lock = threading.Lock()
        self._mtime = -1.0            # sentinel: force the first load
        self._groups: list[dict] = []
        self._clients: dict[tuple, LLM] = {}
        self._reload_if_changed()

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

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
        if self._targets is not None:
            wire_model = litellm_wire_kwargs(self._targets[0])["model"]
            key = (wire_model, g["base_url"], g["api_key"], self._targets)
            c = self._clients.get(key)
            if c is None:
                c = LLM(
                    model=wire_model,
                    base_url=g["base_url"],
                    api_key=g["api_key"],
                    max_tokens=self._max_tokens,
                    targets=self._targets,
                )
                self._clients[key] = c
            return c
        key = (g.get("model") or _DEFAULT_MODEL, g["base_url"], g["api_key"])
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


def _get_llm_via_gateway(mt: int | None, spec: str, targets: tuple[ModelTarget, ...]):
    """PAPERVAULT_LLM_GATEWAY=1 path: an :class:`LLM` pointed at the transparent LLM Gateway.
    Same :class:`LLM` interface + max_tokens passthrough as the direct path, cached per
    ``(max_tokens, spec, base_url)``. ``num_retries=0`` avoids hidden SDK retry stacking."""
    base = config.GATEWAY_URL
    key = config.GATEWAY_KEY
    wire_model = litellm_wire_kwargs(targets[0])["model"]
    cache_key = (mt, spec, base)
    with _gw_lock:
        c = _gw_llms.get(cache_key)
        if c is None:
            c = LLM(
                model=wire_model,
                base_url=base,
                api_key=key,
                max_tokens=mt,
                num_retries=0,
                targets=targets,
            )
            _gw_llms[cache_key] = c
        return c


def get_llm(*, max_tokens: int | None = None, model: str | None = None):
    """Shared LLM handle, defaulting to the ``judge`` route; ``intent``, ``gate``, and ``verify``
    pass their own resolved ``route(role)`` target chains.

    ``model`` is parsed via :func:`papervault.llm_routing.parse_targets` into one or more
    concrete :class:`ModelTarget` entries with per-target thinking/effort wire parameters
    and caller-owned primary cooldown failover. Direct :class:`KeyPool` retains its hot
    reload, key failover, and 401/403 credential auto-disable behavior.
    """
    mt = max_tokens if max_tokens is not None else _DEFAULT_MAX_TOKENS
    targets = parse_targets(model or route("judge"))
    spec = format_targets(targets)
    if config.USE_GATEWAY:
        return _get_llm_via_gateway(mt, spec, targets)
    if _KEYS_FILE.exists() or config.LLM_API_KEY:
        with _pools_lock:
            pkey = (mt, spec)
            pool = _pools.get(pkey)
            if pool is None:
                pool = KeyPool(_KEYS_FILE, mt, model_override=spec, targets=targets)
                _pools[pkey] = pool
            return pool
    raise RuntimeError(
        "No LLM credentials. Set PAPERVAULT_LLM_API_KEY (+ PAPERVAULT_LLM_BASE_URL) or "
        "provide a key-pool file at PAPERVAULT_LLM_KEYS. See .env.example."
    )
