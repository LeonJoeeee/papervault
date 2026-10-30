"""Direct role → concrete model + thinking/effort routing and GLM fallback cooldowns.

Operators override a role with ``PAPERVAULT_LLM_<ROLE>=<model>[@<modifier>][,<fallback>...]``.
The five retired level aliases (``flash``/``standard``/``pro``/``max``/``ultra`` and
``papervault-*``) are rejected with ``ValueError`` so stale ``.env`` files fail loudly
at startup and in ``papervault doctor``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
_RETIRED_LEVELS = frozenset({"flash", "standard", "pro", "max", "ultra"})
_RETIRED_GATEWAY_ALIASES = frozenset(f"papervault-{lvl}" for lvl in _RETIRED_LEVELS)

_GLM_PREFIXES = ("glm-", "relay-glm-")
_GLM_EFFORTS = frozenset({"low", "high", "max"})
_CODEX_ALWAYS_ON_PREFIXES = ("gpt-6-astra", "gpt-6.1-sol")
_CODEX_ALWAYS_ON_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_CODEX_DISABLE_CAPABLE_PREFIXES = ("gpt-6.1-terra", "gpt-6-luna")
_CODEX_DISABLE_CAPABLE_EFFORTS = frozenset(EFFORTS)
_DEEPSEEK_GATEWAY_PREFIXES = ("deepseek-v4", "deepseek-flash", "relay-deepseek-")
_DEEPSEEK_EFFORTS = frozenset({"low", "high", "max"})
_MIMO_GATEWAY_PREFIXES = ("mimo-v2.6-", "relay-mimo-v2.6-")

_DEFAULT_ROUTES: dict[str, str] = {
    "intent": "gpt-6.1-terra@high",
    "build": "glm-5.3@max,gpt-6.1-terra@high",
    "keyword": "gpt-6.1-terra@low",
    "synth": "gpt-6-astra@xhigh",
    "gate": "glm-5.3-flash@high,gpt-6.1-terra@medium",
    "verify": "glm-5.3-flash@low,gpt-6.1-terra@low",
    "judge": "gpt-6.1-terra@medium",
    "decompose": "gpt-6.1-terra@high",
    "eval_judge": "glm-5.3@max",
}
ROLES = tuple(_DEFAULT_ROUTES)

_DEFAULT_COOLDOWN_S = 120.0
_FALLBACK_LOCK = threading.Lock()
_FALLBACK_COOLDOWNS: dict[str, float] = {}


@dataclass(frozen=True)
class ModelTarget:
    """Normalized single model target and optional reasoning/thinking wire directive."""

    model: str
    reasoning_effort: str | None = None
    thinking: bool | None = None


def _spec_error(spec: str, detail: str, *, var_name: str | None) -> ValueError:
    if var_name:
        return ValueError(f"invalid {var_name}={spec!r}: {detail}")
    return ValueError(f"invalid LLM target spec {spec!r}: {detail}")


def _parse_single_target(token: str, *, spec: str, var_name: str | None) -> ModelTarget:
    if not token:
        raise _spec_error(spec, "empty target segment", var_name=var_name)
    if any(ch.isspace() for ch in token):
        raise _spec_error(
            spec,
            f"whitespace is not allowed inside target token {token!r}",
            var_name=var_name,
        )
    if token.count("@") > 1:
        raise _spec_error(
            spec,
            f"multiple '@' modifiers in target {token!r}",
            var_name=var_name,
        )
    if "@" in token:
        raw_model, raw_mod = token.split("@", 1)
        if not raw_model or not raw_mod:
            raise _spec_error(
                spec,
                f"expected '<model>@<modifier>', got {token!r}",
                var_name=var_name,
            )
        mod: str | None = raw_mod.lower()
    else:
        raw_model, mod = token, None

    model = raw_model[7:] if raw_model.lower().startswith("openai/") else raw_model
    if not model:
        raise _spec_error(spec, f"missing model name in {token!r}", var_name=var_name)

    low_model = model.lower()
    if low_model in _RETIRED_LEVELS or low_model in _RETIRED_GATEWAY_ALIASES:
        raise _spec_error(
            spec,
            f"{raw_model!r} is a retired level alias; specify a concrete model "
            "(e.g. 'gpt-6.1-terra@high' or 'glm-5.3@max,gpt-6.1-terra@high')",
            var_name=var_name,
        )

    if low_model.startswith(_GLM_PREFIXES):
        if mod is None:
            return ModelTarget(model=model, reasoning_effort=None, thinking=None)
        if mod in _GLM_EFFORTS:
            return ModelTarget(model=model, reasoning_effort=mod, thinking=None)
        raise _spec_error(
            spec,
            f"invalid modifier @{raw_mod} for GLM model {model!r}; expected bare model "
            "or one of @low, @high, @max",
            var_name=var_name,
        )

    if low_model.startswith(_CODEX_ALWAYS_ON_PREFIXES):
        if mod is None:
            return ModelTarget(model=model, reasoning_effort=None, thinking=None)
        if mod in _CODEX_ALWAYS_ON_EFFORTS:
            return ModelTarget(model=model, reasoning_effort=mod, thinking=None)
        raise _spec_error(
            spec,
            f"invalid modifier @{raw_mod} for always-on Codex model {model!r}; expected "
            "bare model or one of @low, @medium, @high, @xhigh, @max",
            var_name=var_name,
        )

    if low_model.startswith(_CODEX_DISABLE_CAPABLE_PREFIXES):
        if mod is None:
            return ModelTarget(model=model, reasoning_effort=None, thinking=None)
        if mod in _CODEX_DISABLE_CAPABLE_EFFORTS:
            return ModelTarget(model=model, reasoning_effort=mod, thinking=None)
        raise _spec_error(
            spec,
            f"invalid modifier @{raw_mod} for Codex model {model!r}; expected bare model "
            f"or one of {', '.join(f'@{e}' for e in EFFORTS)}",
            var_name=var_name,
        )

    if low_model.startswith(_DEEPSEEK_GATEWAY_PREFIXES):
        if mod is None:
            return ModelTarget(model=model, reasoning_effort=None, thinking=None)
        if mod in _DEEPSEEK_EFFORTS:
            return ModelTarget(model=model, reasoning_effort=mod, thinking=None)
        if mod in {"nothinking", "none"}:
            return ModelTarget(model=model, reasoning_effort=None, thinking=False)
        raise _spec_error(
            spec,
            f"invalid modifier @{raw_mod} for DeepSeek model {model!r}; expected bare model "
            "or one of @low, @high, @max, @nothinking, @none",
            var_name=var_name,
        )

    if low_model.startswith(_MIMO_GATEWAY_PREFIXES):
        if mod is None:
            return ModelTarget(model=model, reasoning_effort=None, thinking=None)
        if mod == "thinking":
            return ModelTarget(model=model, reasoning_effort=None, thinking=True)
        if mod in {"nothinking", "none"}:
            return ModelTarget(model=model, reasoning_effort=None, thinking=False)
        raise _spec_error(
            spec,
            f"invalid modifier @{raw_mod} for MiMo model {model!r}; expected bare model "
            "or one of @thinking, @nothinking, @none",
            var_name=var_name,
        )

    if mod is None:
        return ModelTarget(model=model, reasoning_effort=None, thinking=None)
    if mod == "thinking":
        return ModelTarget(model=model, reasoning_effort=None, thinking=True)
    if mod == "nothinking":
        return ModelTarget(model=model, reasoning_effort=None, thinking=False)
    if mod in EFFORTS:
        return ModelTarget(model=model, reasoning_effort=mod, thinking=None)
    raise _spec_error(
        spec,
        f"unknown modifier @{raw_mod} for {model!r}; expected one of "
        f"{', '.join(f'@{e}' for e in EFFORTS)}, @thinking, or @nothinking",
        var_name=var_name,
    )


def parse_targets(spec: str, *, var_name: str | None = None) -> tuple[ModelTarget, ...]:
    """Parse a comma-separated ``primary[,fallback...]`` spec into ``ModelTarget`` items."""
    if not isinstance(spec, str) or not spec.strip():
        raise _spec_error(str(spec), "target specification cannot be empty", var_name=var_name)
    raw_parts = spec.split(",")
    targets = tuple(
        _parse_single_target(part.strip(), spec=spec, var_name=var_name)
        for part in raw_parts
    )
    return targets


def format_target(target: ModelTarget) -> str:
    """Serialize a ``ModelTarget`` into canonical ``route(role)`` syntax."""
    if target.reasoning_effort is not None:
        return f"{target.model}@{target.reasoning_effort}"
    if target.thinking is not None:
        return f"{target.model}@{'thinking' if target.thinking else 'nothinking'}"
    return target.model


def format_targets(targets: Sequence[ModelTarget]) -> str:
    """Serialize one or more ``ModelTarget`` entries as ``primary[,fallback...]``."""
    if not targets:
        raise ValueError("targets must not be empty")
    return ",".join(format_target(t) for t in targets)


def route(role: str) -> str:
    """Return the canonical target chain for ``role``; unset/blank overrides use defaults."""
    r = role.strip().lower()
    if r not in _DEFAULT_ROUTES:
        raise ValueError(f"unknown LLM role {role!r}; known roles: {', '.join(ROLES)}")
    var = f"PAPERVAULT_LLM_{r.upper()}"
    raw = os.getenv(var, "").strip()
    if not raw:
        targets = parse_targets(_DEFAULT_ROUTES[r])
    else:
        targets = parse_targets(raw, var_name=var)
    return format_targets(targets)


def openai_wire_kwargs(target: ModelTarget) -> dict[str, Any]:
    """Wire kwargs for ``openai.AsyncOpenAI`` chat completions."""
    wire: dict[str, Any] = {"model": target.model}
    if target.reasoning_effort is not None:
        wire["reasoning_effort"] = target.reasoning_effort
    if target.thinking is not None:
        wire["extra_body"] = {
            "thinking": {"type": "enabled" if target.thinking else "disabled"}
        }
    return wire


def litellm_wire_kwargs(target: ModelTarget) -> dict[str, Any]:
    """Wire kwargs for client-side ``litellm.completion``.

    ``reasoning_effort`` is always placed inside ``extra_body`` rather than top-level
    so client-side LiteLLM forwards it cleanly to OpenAI-compatible endpoints without
    raising ``UnsupportedParamsError`` on custom model IDs.
    """
    model = target.model if "/" in target.model else f"openai/{target.model}"
    wire: dict[str, Any] = {"model": model}
    extra_body: dict[str, Any] = {}
    if target.reasoning_effort is not None:
        extra_body["reasoning_effort"] = target.reasoning_effort
    if target.thinking is not None:
        extra_body["thinking"] = {"type": "enabled" if target.thinking else "disabled"}
    if extra_body:
        wire["extra_body"] = extra_body
    return wire


def fallback_cooldown_seconds() -> float:
    """Active primary-model cooldown window in seconds (``<= 0`` disables cooldown)."""
    raw = os.getenv("PAPERVAULT_LLM_FALLBACK_COOLDOWN_S", "").strip()
    if not raw:
        return _DEFAULT_COOLDOWN_S
    val = float(raw)
    return val if val > 0 else 0.0


def is_model_cooled_down(model: str, *, now: float | None = None) -> bool:
    """Return True iff ``model`` is currently inside an active fallback cooldown window."""
    if fallback_cooldown_seconds() <= 0:
        return False
    key = model.strip()
    ts = time.monotonic() if now is None else now
    with _FALLBACK_LOCK:
        until = _FALLBACK_COOLDOWNS.get(key)
        if until is None:
            return False
        if ts >= until:
            _FALLBACK_COOLDOWNS.pop(key, None)
            return False
        return True


def record_model_cooldown(model: str, *, now: float | None = None) -> float:
    """Record a cooldown window for ``model`` and return the applied duration in seconds."""
    key = model.strip()
    ttl = fallback_cooldown_seconds()
    if ttl <= 0:
        with _FALLBACK_LOCK:
            _FALLBACK_COOLDOWNS.pop(key, None)
        return 0.0
    ts = time.monotonic() if now is None else now
    with _FALLBACK_LOCK:
        _FALLBACK_COOLDOWNS[key] = ts + ttl
    return ttl


def clear_fallback_cooldowns() -> None:
    """Clear all process-local fallback cooldown entries (for deterministic tests)."""
    with _FALLBACK_LOCK:
        _FALLBACK_COOLDOWNS.clear()


def http_status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status code from an OpenAI/LiteLLM/httpx exception."""
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    if code is None and "authentication" in type(exc).__name__.lower():
        return 401
    try:
        return int(code) if code is not None else None
    except (ValueError, TypeError):
        return None


def _is_openai_failover_eligible(exc: Exception, code: int | None) -> bool:
    if code is not None:
        return code in {408, 429} or 500 <= code <= 599
    if isinstance(exc, TimeoutError):
        return True
    try:
        from openai import APITimeoutError

        return isinstance(exc, APITimeoutError)
    except Exception:  # noqa: BLE001
        return type(exc).__name__ == "APITimeoutError"


async def create_openai_chat_completion_with_fallback(
    client: Any,
    targets: Sequence[ModelTarget],
    *,
    log: logging.Logger | None = None,
    budget_s: float | None = None,
    **kwargs: Any,
) -> Any:
    """Execute ``client.chat.completions.create`` across ``targets`` with primary cooldown failover.

    Used by async OpenAI-SDK callers (``judge_mimo``, ``_ab_u1_judge``,
    ``scripts/search_relevance_baseline.py``). Non-final targets in active cooldown are
    skipped; transient upstream errors (``429``, ``408``, ``500..599``, timeouts) on a
    non-final target record cooldown, log a warning, and fail over immediately in the
    same call without burning same-target retries. ``CancelledError`` and deterministic
    ``4xx`` errors always re-raise immediately.
    """
    if not targets:
        raise ValueError("targets must not be empty")
    deadline = (time.monotonic() + budget_s) if budget_s is not None else None
    active_log = log or logger
    for target_idx, target in enumerate(targets):
        is_final = target_idx == len(targets) - 1
        if not is_final and is_model_cooled_down(target.model):
            continue
        wire = openai_wire_kwargs(target)
        try:
            return await client.chat.completions.create(**kwargs, **wire)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            if is_final:
                raise
            code = http_status_code(exc)
            if not _is_openai_failover_eligible(exc, code):
                raise
            ttl = record_model_cooldown(target.model)
            fallback_model = targets[target_idx + 1].model
            if deadline is not None and deadline - time.monotonic() <= 0.0:
                active_log.warning(
                    "openai primary failover skipped (budget exhausted) %s -> %s code=%s err=%s",
                    target.model,
                    fallback_model,
                    code,
                    type(exc).__name__,
                )
                raise
            active_log.warning(
                "openai primary failover %s -> %s code=%s err=%s cooldown=%.0fs",
                target.model,
                fallback_model,
                code,
                type(exc).__name__,
                ttl,
            )
    raise AssertionError("unreachable: targets is non-empty")
