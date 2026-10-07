"""Role → gateway level routing for every PaperVault text LLM call.

Operators override a role with PAPERVAULT_LLM_<ROLE>=flash|standard|pro.
The endpoint owns each level's model and effort. Direct endpoints must also
serve these level aliases; caller-side provider models are no longer selected.
"""
from __future__ import annotations

import os

LEVELS = ("flash", "standard", "pro")
_DEFAULT_LEVELS = {
    "build": "standard",
    "keyword": "flash",
    "synth": "pro",
    "gate": "flash",
    "verify": "flash",
    "judge": "standard",
    "decompose": "standard",
    "eval_judge": "standard",
}
ROLES = tuple(_DEFAULT_LEVELS)


def route(role: str) -> str:
    """Return the role's level; unset/blank overrides use the agreed default."""
    r = role.strip().lower()
    if r not in _DEFAULT_LEVELS:
        raise ValueError(f"unknown LLM role {role!r}; known roles: {', '.join(ROLES)}")
    var = f"PAPERVAULT_LLM_{r.upper()}"
    level = os.getenv(var, "").strip() or _DEFAULT_LEVELS[r]
    if level not in LEVELS:
        raise ValueError(f"invalid {var}={level!r}; expected one of {', '.join(LEVELS)}")
    return level
