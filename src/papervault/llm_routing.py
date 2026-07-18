"""Role → (model, thinking) routing for every papervault LLM call site (issue #8).

ONE resolver so call sites stop reading model/thinking env vars ad-hoc. Each ROLE
maps to a first-class ``.env`` line ``PAPERVAULT_LLM_<ROLE>`` whose value is
``model[:think|:nothink]``:

  * ``model:think``   → send ``enable_thinking=True``
  * ``model:nothink`` → send ``enable_thinking=False``
  * ``model``  (bare) → send NO thinking param (safe for a non-reasoning provider)

``route(role)`` returns ``(model, thinking)`` with ``thinking`` in ``{True, False, None}``.
A call site forwards ``model`` and — ONLY when ``thinking is not None`` — an
``enable_thinking`` kwarg.

DEFAULTS preserve today's behavior (this layer is behavior-neutral until an operator
sets a new var):

  * synth / decompose / judge / gate → the SYNTH slot (``config.SYNTH_MODEL``), no
    thinking param (they ride the pool default today);
  * build / keyword → the BUILD slot (``config.BUILD_MODEL``) and reproduce
    ``knowledge/store/graph.py``'s ``KS_BUILD_THINKING`` / ``KS_KW_THINKING`` logic
    (build-extraction defaults thinking ON; the query-path keyword call — the one
    carrying ``response_format`` — defaults ON too but flips OFF under ``KS_KW_THINKING=0``);
  * verify → its current model (``PAPER_PIPELINE_VERIFY_MODEL`` or ``openai/mimo-v2.5``).

LEGACY vars remain as back-compat overrides (the experiment layer eval arms use) and
WIN over the new ``PAPERVAULT_LLM_*`` line where they apply:

  * ``KS_BUILD_MODEL``    — model for build + keyword
  * ``KS_BUILD_THINKING`` — thinking for build + keyword (``"1"`` → True, else None)
  * ``KS_KW_THINKING``    — ``"0"`` forces the keyword call's thinking OFF (False)
  * ``PAPER_PIPELINE_VERIFY_MODEL`` — model for verify
"""
from __future__ import annotations

import os

from papervault import config

ROLES = ("synth", "decompose", "build", "keyword", "judge", "gate", "verify")

# Sentinel: "the new PAPERVAULT_LLM_<ROLE> line said nothing about thinking" — distinct
# from ``None`` (a bare model spec, which DOES say "send no thinking param").
_UNSET = object()

_VERIFY_DEFAULT_MODEL = "openai/mimo-v2.5"


def _parse_spec(spec: str) -> tuple[str, bool | None]:
    """Parse ``model[:think|:nothink]`` → ``(model, thinking)``.

    Bare model → thinking ``None`` (send no thinking param). Rejects an empty model
    (e.g. ``:think`` or ``""``) with a clear error.
    """
    s = spec.strip()
    low = s.lower()
    if low.endswith(":think"):
        model, thinking = s[: -len(":think")], True
    elif low.endswith(":nothink"):
        model, thinking = s[: -len(":nothink")], False
    else:
        model, thinking = s, None
    model = model.strip()
    if not model:
        raise ValueError(
            f"invalid LLM route spec {spec!r}: empty model "
            "(expected 'model', 'model:think', or 'model:nothink')"
        )
    return model, thinking


def _new_line(role: str) -> tuple[str | None, bool | object]:
    """The parsed ``PAPERVAULT_LLM_<ROLE>`` line as ``(model, thinking)``.

    Returns ``(None, _UNSET)`` when the var is unset/blank (treated as "not
    configured" → fall back to the default, no error).
    """
    raw = os.getenv(f"PAPERVAULT_LLM_{role.upper()}")
    if raw is None or not raw.strip():
        return (None, _UNSET)
    model, thinking = _parse_spec(raw)
    return (model, thinking)


def _resolve_synth_slot(role: str) -> tuple[str, bool | None]:
    """synth / decompose / judge / gate: default = (SYNTH slot, no thinking param);
    the new ``PAPERVAULT_LLM_<ROLE>`` line, when set, overrides both axes."""
    new_model, new_thinking = _new_line(role)
    if new_model is not None:
        return (new_model, None if new_thinking is _UNSET else new_thinking)  # type: ignore[return-value]
    return (config.SYNTH_MODEL, None)


def _build_slot_model(role: str, new_model: str | None) -> str:
    """Model for the BUILD slot (build + keyword share it). Reproduces graph.py's
    ``os.getenv("KS_BUILD_MODEL") or _pv.BUILD_MODEL`` with the new var slotted between
    the legacy override and the config default: legacy KS_BUILD_MODEL > new var > BUILD_MODEL."""
    return os.getenv("KS_BUILD_MODEL") or new_model or config.BUILD_MODEL


def _resolve_build() -> tuple[str, bool | None]:
    new_model, new_thinking = _new_line("build")
    model = _build_slot_model("build", new_model)
    # thinking — graph.py build-extraction path: `elif os.getenv("KS_BUILD_THINKING","1")=="1"`.
    # Legacy KS_BUILD_THINKING (when set) wins; else the new var; else today's default (True).
    kbt = os.getenv("KS_BUILD_THINKING")
    if kbt is not None:
        thinking: bool | None = True if kbt == "1" else None
    elif new_thinking is not _UNSET:
        thinking = new_thinking  # type: ignore[assignment]
    else:
        thinking = True  # KS_BUILD_THINKING unset → graph.py default "1" → True
    return (model, thinking)


def _resolve_keyword() -> tuple[str, bool | None]:
    new_model, new_thinking = _new_line("keyword")
    model = _build_slot_model("keyword", new_model)
    # thinking — graph.py query-path keyword call (the one carrying response_format):
    #   if KS_KW_THINKING=="0":        enable_thinking=False
    #   elif KS_BUILD_THINKING=="1":   enable_thinking=True   (default "1")
    #   else:                          (no thinking param)
    kkt = os.getenv("KS_KW_THINKING")
    kbt = os.getenv("KS_BUILD_THINKING")
    if kkt == "0":
        thinking: bool | None = False
    elif kbt is not None:
        thinking = True if kbt == "1" else None
    elif new_thinking is not _UNSET:
        thinking = new_thinking  # type: ignore[assignment]
    else:
        thinking = True  # KS_BUILD_THINKING unset → graph.py default "1" → True
    return (model, thinking)


def _resolve_verify() -> tuple[str, bool | None]:
    """verify: legacy PAPER_PIPELINE_VERIFY_MODEL > new var > openai/mimo-v2.5. The library
    plane sends no thinking param today, so thinking follows the new var (default None)."""
    new_model, new_thinking = _new_line("verify")
    model = os.getenv("PAPER_PIPELINE_VERIFY_MODEL") or new_model or _VERIFY_DEFAULT_MODEL
    thinking = None if new_thinking is _UNSET else new_thinking
    return (model, thinking)  # type: ignore[return-value]


def route(role: str) -> tuple[str, bool | None]:
    """Resolve a role to ``(model, thinking)``. ``thinking``: True (``:think``),
    False (``:nothink``), or None (bare model → send no thinking param)."""
    r = role.strip().lower()
    if r in ("synth", "decompose", "judge", "gate"):
        return _resolve_synth_slot(r)
    if r == "build":
        return _resolve_build()
    if r == "keyword":
        return _resolve_keyword()
    if r == "verify":
        return _resolve_verify()
    raise ValueError(f"unknown LLM role {role!r}; known roles: {', '.join(ROLES)}")
