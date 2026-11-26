# 0014 — Three-tier GLM → relay → Codex role routing, gpt-6.2-astra synthesis, and quota-exhaustion failover

Status: Accepted (2026-11-18), Amends 0013 (2026-10-30)

## Context

[ADR-0012](0012-direct-role-model-routing-and-glm-fallback.md) and [ADR-0013](0013-effort-tiered-role-routing-and-omitted-max-tokens.md) established concrete per-role `PAPERVAULT_LLM_<ROLE>` chains, per-family modifier validation, and caller-owned cooldown failover. Three operational changes motivate this amendment:

1. **Arrival of `gpt-6.2-astra` and `gpt-6.2-sol`.** The transparent LLM Gateway catalog exposes `gpt-6.2-astra` and `gpt-6.2-sol` as always-on Codex reasoning models (`low|medium|high|xhigh|max`, rejecting `@none`), succeeding `gpt-6-astra@xhigh` on evidence-grounded answer synthesis (`synth`).
2. **Three-tier subscription and relay priority.** Under ADR-0013, `keyword`, `judge`, and `decompose` spent Codex subscription quota directly as single-target primaries, `build`/`gate`/`verify` skipped pre-funded third-party `relay-glm-*` balances when Zhipu's weekly or 5-hour subscription window throttled, and `eval_judge` lacked same-family relay continuity. The owner established an explicit three-tier order for non-interactive stages — **Tier 1 official GLM subscription (`glm-*`) → Tier 2 third-party `relay-*` balance (`relay-glm-*`) → Tier 3 Codex subscription (`gpt-6.1-terra`)** — while keeping interactive `intent` directly on low-latency `gpt-6.1-terra@high` and keeping `eval_judge` strictly within the GLM-5.3 family (`glm-5.3@max,relay-glm-5.3@max`, without Tier 3 GPT fallback) to preserve the benchmark ruler invariant.
3. **Relay balance and quota-exhaustion failover gap.** Multi-target callers previously failed over only on `HTTP 408`, `429`, `500..599`, timeouts, and `StreamTruncated`. Third-party OpenAI-compatible relays (`NewAPI`/`OneAPI` and provider billing gates) and Zhipu quota endpoints also report exhausted credit or subscription windows via `HTTP 402` or `HTTP 400`/`403` payloads (e.g., `预扣费额度失败`, `Insufficient balance`, `insufficient_user_quota`, or Zhipu codes `1113|1304|1308|1309`). Treating those responses as non-retryable `4xx` aborted non-final `relay-*` legs instead of cooling them down and advancing to Tier 3 `gpt-6.1-terra` (and could misclassify a quota `403` as a revoked credential in direct `KeyPool` mode).

## Decision

Amend [ADR-0012](0012-direct-role-model-routing-and-glm-fallback.md) Decision item 4 and [ADR-0013](0013-effort-tiered-role-routing-and-omitted-max-tokens.md) Decision items 1 and 2 as follows (keeping omitted default `max_tokens`, wire builders, and retired alias rejection unchanged):

1. **Codex `gpt-6.2` always-on modifier registration and `synth` upgrade.** Register `gpt-6.2-astra` and `gpt-6.2-sol` in `_CODEX_ALWAYS_ON_PREFIXES` alongside `gpt-6-astra` and `gpt-6.1-sol` (accepting bare or `@low|@medium|@high|@xhigh|@max`; rejecting `@none`, `@thinking`, `@nothinking`, and unknown modifiers), and upgrade default `synth` from `gpt-6-astra@xhigh` to `gpt-6.2-astra@xhigh`.
2. **Three-tier default role routes.**
   - Single-target Codex roles: `intent` → `gpt-6.1-terra@high` (interactive low-latency planning); `synth` → `gpt-6.2-astra@xhigh` (flagship synthesis).
   - Three-tier GLM → `relay-glm-*` → Codex roles:
     - `build` → `glm-5.3@max,relay-glm-5.3@max,gpt-6.1-terra@high`
     - `gate` → `glm-5.3-flash@high,relay-glm-5.3-flash@high,gpt-6.1-terra@medium`
     - `verify` → `glm-5.3-flash@low,relay-glm-5.3-flash@low,gpt-6.1-terra@low`
     - `keyword` → `glm-5.3-flash@low,relay-glm-5.3-flash@low,gpt-6.1-terra@low`
     - `judge` → `glm-5.3-flash@high,relay-glm-5.3-flash@high,gpt-6.1-terra@medium`
     - `decompose` → `glm-5.3-flash@high,relay-glm-5.3-flash@high,gpt-6.1-terra@high`
   - Two-tier same-family evaluation judge (single-family ruler invariant): `eval_judge` → `glm-5.3@max,relay-glm-5.3@max` (no cross-family GPT fallback).
3. **Quota and balance-exhaustion failover across all multi-target callers.** Centralize `is_quota_exhaustion_error(exc, code)` in `papervault.llm_routing` and apply it in `create_openai_chat_completion_with_fallback`, `library.llm._is_failover_eligible`, and `knowledge.store.llm._is_failover_eligible`. Treat `HTTP 402` unconditionally, `BudgetExceededError`, and `HTTP 400/402/403/429` responses carrying quota/balance indicators (`quota`, `insufficient`, `balance`, `credit`, `exhausted`, `套餐`, `额度`, `余额`, `限额`, `预扣费`, `欠费`, `weekly`, `5小时`, `5-hour`) or Zhipu quota/billing codes (`1113|1304|1308|1309`) as failover-eligible on non-final targets so exhausted Tier 1/Tier 2 legs record `PAPERVAULT_LLM_FALLBACK_COOLDOWN_S` per `target.model` and advance immediately in the same call. Exclude quota-indicated `403` responses from permanent key auto-disable in `KeyPool`, while genuine non-quota `400/401/403/422` errors continue failing fast.

Rejected alternatives:
- *Add Tier 3 `gpt-6.1-terra@high` to default `eval_judge`*: rejected because crossing model families mid-evaluation violates the single-ruler invariant established in ADR-0013; failing over only from official `glm-5.3@max` to same-family `relay-glm-5.3@max` preserves measurement consistency across Zhipu subscription windows.
- *Fail over on every `HTTP 400` or `HTTP 403` regardless of payload*: rejected because genuine schema/parameter validation errors (`400`) and revoked-token/permission denials (`403`) are deterministic caller or credential defects that must surface immediately rather than masking misconfiguration behind fallback models.

## Consequences

- Background extraction (`build`, `gate`, `verify`), search/query helpers (`judge`, `keyword`, `decompose`), and evaluation judging (`eval_judge`) drain official GLM subscription quota first, absorb overflows on `relay-glm-*`, and (for production roles) step cleanly to `gpt-6.1-terra` when relay credit is exhausted (`402` or quota-indicated `400/403/429`), cooling down each exhausted leg independently for 120 seconds.
- Knowledge distillation and library extraction recipe digests deterministically reflect the updated `configured_models` strings for `build`, `keyword`, and `gate`.

## Amendment (2026-11-21, `v0.11.0`–`v0.11.2`)

Three failover refinements extend the policy above without changing default role routes or per-family modifier rules:

1. **Dedicated quota/auth cooldown (`PAPERVAULT_LLM_QUOTA_COOLDOWN_S`, default `600` s).** Quota/balance exhaustion and upstream OAuth expiry use a 600-second per-model cooldown (`<= 0` disables; also disabled if unset while `PAPERVAULT_LLM_FALLBACK_COOLDOWN_S <= 0`), keeping `PAPERVAULT_LLM_FALLBACK_COOLDOWN_S` (`120` s) scoped to transient errors (`408`, `429`, `5xx`, timeouts, transport errors, `StreamTruncated`).
2. **Gateway-wrapped upstream OAuth `token_expired` failover (`is_upstream_auth_expired_error`, `v0.11.2`).** Multi-target routes fail over with the quota cooldown (`600` s) when a gateway-wrapped upstream channel returns HTTP `400`/`401` indicating expired upstream OAuth (`token_expired` / `your authentication token has expired`), without disabling valid keys in direct `KeyPool` mode. Genuine client/proxy auth failures (`Invalid proxy server token`, generic `Invalid API key`) remain fail-fast.
3. **Truncation and transport-error handling.** Empty-content or thinking-only completions normalize to `StreamTruncated` (engaging retries and multi-target failover), partial `finish_reason="length"` completions return `TruncatedResponse` (a `str` subclass) so callers can inspect truncation explicitly, and library LLM calls treat transport/connection errors as failover-eligible on non-final targets with per-target timeout bounds.

## Amendment (2026-11-26)

1. **Upgrade full-size GLM primaries to `glm-5.4`.** With official direct `glm-5.4` available on the transparent LLM Gateway (while third-party relay remains on `relay-glm-5.3` and no `glm-5.4-flash` exists yet), update the full-size GLM primaries for `build` (`glm-5.4@max,relay-glm-5.3@max,gpt-6.1-terra@high`) and `eval_judge` (`glm-5.4@max,relay-glm-5.3@max`). All `glm-5.3-flash` roles (`keyword`, `gate`, `verify`, `judge`, `decompose`) remain unchanged. Note: until the third-party relay exposes `glm-5.4`, a quota failover on `eval_judge` stays within the GLM family (no cross-family GPT fallback) but steps down one minor version (`glm-5.4` → `relay-glm-5.3`); operators requiring strict single-version arbitration or historical `glm-5.3` baseline reproduction should pin `PAPERVAULT_LLM_EVAL_JUDGE=glm-5.4@max` (or `glm-5.3@max`).
2. **`glm-5.4` effort modifier support.** Allow `@medium` alongside `@low`, `@high`, and `@max` on `glm-5.4*` (matching the live gateway capability), while keeping other `glm-*` and `relay-glm-*` legs restricted to `@low|@high|@max`.
3. **Retire third-party `relay-glm-*` default legs.** Because the third-party OpenAI-compatible relay (`api.aibvv.cn`) became unavailable (hanging chat timeouts / 503), remove `relay-glm-*` from all default routes so production roles fall back directly from official GLM to Codex (`build` → `glm-5.4@max,gpt-6.1-terra@high`; `keyword`/`verify` → `glm-5.3-flash@low,gpt-6.1-terra@low`; `gate`/`judge` → `glm-5.3-flash@high,gpt-6.1-terra@medium`; `decompose` → `glm-5.3-flash@high,gpt-6.1-terra@high`), and return `eval_judge` to single-target `glm-5.4@max` (strict single-version ruler invariant). Generic `relay-*` modifier parsing and quota-exhaustion failover classifiers remain supported for explicit env overrides.
