# 0012 — Direct role model routing and caller-owned GLM quota fallback

Status: Accepted (2026-10-30)

## Context

PaperVault previously routed its nine text LLM stages through five symbolic gateway levels (`flash`, `standard`, `pro`, `max`, `ultra`) prefixed on the wire as `papervault-<level>`, delegating concrete model selection, reasoning effort, and cross-model fallback to gateway-side aliases. The transparent LLM Gateway (`v0.7.0`) removed project-specific `papervault-*` aliases and hidden server-side cross-model fallback: callers now pass concrete model IDs (`gpt-6-astra`, `gpt-6.1-terra`, `glm-5.3`, `glm-5.3-flash`) plus explicit reasoning/thinking parameters directly, and own their own retry and fallback policy.

At the same time, high-volume background extraction, gating, PDF verification, and evaluation judging benefit from routing primary traffic to Zhipu GLM coding-plan models (`glm-5.3`, `glm-5.3-flash`) with adaptive thinking enabled, while Zhipu's 5-hour and weekly subscription quota windows require deterministic, non-stalling failover to `gpt-6.1-terra` without wasting per-chunk round-trips against a throttled primary. Client-side `litellm.completion` in the Library plane also rejects top-level `reasoning_effort` on custom `openai/<model>` names unless forwarded via `extra_body`.

## Decision

1. **Direct concrete role routes.** Retire `LEVELS` and `gateway_model(...)`. Keep the nine `ROLES` (`intent`, `build`, `keyword`, `synth`, `gate`, `verify`, `judge`, `decompose`, `eval_judge`) and `PAPERVAULT_LLM_<ROLE>` overrides, where each role resolves to an ordered tuple of immutable `ModelTarget(model, reasoning_effort, thinking)` entries (`primary[,fallback...]`). Reject retired bare level names and `papervault-*` aliases with `ValueError` naming `PAPERVAULT_LLM_<ROLE>` so stale `.env` files fail loudly in `papervault doctor` and startup checks.
2. **Agreed role defaults and modifier grammar.**
   - Single-target GPT roles (no cross-model fallback): `synth` → `gpt-6-astra@xhigh`; `intent`, `keyword`, `decompose`, `judge` → `gpt-6.1-terra@high`.
   - Multi-target GLM-primary roles with GPT fallback: `build`, `eval_judge` → `glm-5.3,gpt-6.1-terra@high`; `gate`, `verify` → `glm-5.3-flash,gpt-6.1-terra@high`.
   - Omitted `@modifier` on `glm-*` defaults to `thinking=True` (`@nothinking` / `@none` disables; effort tiers on `glm-*` raise `ValueError`). Non-`glm-*` models default to plain OpenAI-compatible wire calls (`reasoning_effort=None, thinking=None`), preserving arbitrary OpenAI-compatible endpoints, and accept `@none|@low|@medium|@high|@xhigh|@max` or `@thinking|@nothinking`.
   - Per-role `max_tokens` is not added to routing; existing transport ceilings (`KS_LLM_MAX_TOKENS`, `PAPERVAULT_LIB_MAX_TOKENS`) remain unchanged.
3. **Centralized wire builders.** `openai_wire_kwargs(target)` emits top-level `reasoning_effort` and `extra_body.thinking` for `openai.AsyncOpenAI`; `litellm_wire_kwargs(target)` prefixes bare models with `openai/` and places both `reasoning_effort` and `thinking` inside `extra_body` so client-side LiteLLM never raises `UnsupportedParamsError`.
4. **Caller-owned GLM → GPT failover with process-local cooldown.** Multi-target callers (`knowledge/store/llm.py`, `library/llm.py`, `eval/judge_mimo.py`, `eval/_ab_u1_judge.py`, `scripts/search_relevance_baseline.py`) share a thread-safe process-local cooldown map keyed by `target.model` (`PAPERVAULT_LLM_FALLBACK_COOLDOWN_S`, default `120`s, `<= 0` disables). On `HTTP 429`, `408`, `500..599`, `APITimeoutError` / `TimeoutError`, or `StreamTruncated` from a non-final target, the caller records cooldown on the failed primary, logs `primary -> fallback`, and fails over immediately in the same call without burning same-target retries (within remaining retry budget). Active cooldown skips non-final targets on subsequent calls and automatically re-probes the primary once expired. `CancelledError` and deterministic `4xx` (`400`, `401`, `403`, `422`) never fail over; plain `APIConnectionError` on the S2 gateway path retains the local-proxy unreachable backoff without cooling down the primary model.

Rejected alternatives:
- *Keep `flash`..`ultra` aliases in PaperVault and map them internally to provider models*: rejected because hiding concrete model/thinking choices behind local aliases obscures actual served models in receipts/logs and blocks operators from configuring per-role `primary,fallback` chains directly in `.env`.
- *Burn same-target transient retries on GLM before failing over to GPT*: rejected because Zhipu quota exhaustion (`429`) persists across windows; retrying the throttled primary multiplies latency and exhausts worker budgets before fallback runs.

## Consequences

- PaperVault works directly against the transparent LLM Gateway (`v0.7.0`) and any standard OpenAI-compatible endpoint without proxy-owned `papervault-*` aliases.
- Sustained GLM `429` or timeout windows fail over to `gpt-6.1-terra@high` on the first failed attempt and bypass the throttled primary for 120 seconds before automatic recovery probing; library serving observations mark fallback calls `degraded=True` and usage logs record the actual served `target.model`.
- Operators upgrading an existing `.env` that pinned `PAPERVAULT_LLM_<ROLE>=flash|standard|pro|max|ultra` must remove those lines (to adopt the new defaults) or replace them with concrete `<model>[@<modifier>][,<fallback>...]` specs; `papervault doctor` surfaces any unmigrated variable immediately.
