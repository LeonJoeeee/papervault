"""Token account book: one LLMTOK line per successful LLM completion (issue #4, round 2).

Every LLM choke point (knowledge gateway path, knowledge direct KeyPool path, library
litellm wrapper) calls :func:`log_usage` right after a successful completion:

    LLMTOK plane=ks model=mimo-v2.5-pro dur=84.2s ptok=198432 ctok=1810 ttok=200242

``ptok``/``ctok``/``ttok`` are the endpoint-reported prompt/completion/total token
counts — real tokenizer numbers, not char estimates. A response without a usable
``usage`` block logs ``ptok=? ctok=? ttok=?`` so gaps are visible, not silent.

Scope: the LIVE-SERVICE planes only. The offline eval/arbitration tools
(``papervault.eval._ab_u1_judge``, ``papervault.eval.judge_mimo``) call the API
directly and are deliberately outside the book — it accounts what serving costs,
not what benchmarking costs.

``model`` labels are normalized to the bare gateway/group name (no litellm
provider prefix) so per-model roll-ups aggregate across planes.

Emission can never affect the call: any exception inside is swallowed (the same
never-change-the-outcome contract as the MCPCALL access log, PR #14).
"""
from __future__ import annotations

import logging
from typing import Any


def log_usage(logger: logging.Logger, plane: str, model: str, dur_s: float, resp: Any) -> None:
    """Emit the LLMTOK accounting line for a completed LLM response."""
    try:
        u = getattr(resp, "usage", None)
        ptok = getattr(u, "prompt_tokens", None)
        ctok = getattr(u, "completion_tokens", None)
        ttok = getattr(u, "total_tokens", None)
        logger.info(
            "LLMTOK plane=%s model=%s dur=%.1fs ptok=%s ctok=%s ttok=%s",
            plane, model, dur_s,
            ptok if ptok is not None else "?",
            ctok if ctok is not None else "?",
            ttok if ttok is not None else "?",
        )
    except Exception:  # noqa: BLE001 — accounting must never change a call's outcome
        # Distinct prefix (not "LLMTOK ") so a grep-based accounting parser never
        # hits a fieldless line even with DEBUG enabled.
        logger.debug("LLMTOK-emit-failed", exc_info=True)
