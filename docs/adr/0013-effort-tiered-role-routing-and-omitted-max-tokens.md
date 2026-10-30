# 0013 — Effort-tiered role routing, Gateway v0.7.3 modifier grammar, and omitted default max_tokens

Status: Accepted (2026-10-30), Amends 0012 (2026-10-30)

## Context

[ADR-0012](0012-direct-role-model-routing-and-glm-fallback.md) moved PaperVault to direct concrete `PAPERVAULT_LLM_<ROLE>` routes and caller-owned primary cooldown failover against the transparent LLM Gateway `v0.7.0`. Subsequent Gateway `v0.7.3` catalog standardization and first-principles stage review surfaced three mismatches in ADR-0012's initial defaults and modifier rules:

1. **Per-family thinking and effort contracts.** Under Gateway `v0.7.3`, `glm-*` and `relay-glm-*` are always-on reasoning models (`can_disable_thinking=no`) exposing three explicit `reasoning_effort` tiers (`low`, `high`, `max`) rather than binary `@thinking`/`@nothinking` toggles; `gpt-6-astra*` and `gpt-6.1-sol*` are always-on Codex models (`low..max`, rejecting `@none`); `gpt-6.1-terra*` and `gpt-6-luna*` disable thinking via `reasoning_effort="none"` rather than `extra_body.thinking`; `deepseek-v4*`, `deepseek-flash*`, and `relay-deepseek-*` accept `low|high|max` or `@nothinking`/`@none`; and `mimo-v2.6-*` / `relay-mimo-v2.6-*` support binary `@thinking`/`@nothinking` only.
2. **Stage effort calibration and the `eval_judge` ruler invariant.** Running every GPT helper role at `@high` over-spends latency on serial query keyword extraction (`keyword`) and batch relevance scoring (`judge`), while un-tiered `glm-5.3` / `glm-5.3-flash` under-specifies effort across `verify`, `gate`, and `build`. Conversely, defaulting `eval_judge` to `glm-5.3,gpt-6.1-terra@high` silently mixes GLM and GPT verdicts mid-benchmark whenever Zhipu returns a transient `429`, violating the evaluation ruler invariant that both sides of an A/B or rubric run be scored by the same judge family.
3. **Artificial client-side `max_tokens` ceilings.** Gateway `v0.7.3` injects each `official-direct` model's official output ceiling (`131072` on `glm-5.3*`/`mimo-v2.6-*`, `393216` on `deepseek-*`) whenever `max_tokens` is omitted, while `codex-subscription` ignores `max_tokens`. Legacy hardcoded defaults in PaperVault (`131000` in `knowledge/store/llm.py`, `32000` in `library/llm.py` and `eval/judge_mimo.py`, `16000` in `knowledge/query/synth.py`, `4000` in `eval/_ab_u1_judge.py`) restrict combined reasoning + answer headroom, particularly at `@max` effort.

## Decision

Amend [ADR-0012](0012-direct-role-model-routing-and-glm-fallback.md) Decision items 2 and 4 as follows (keeping Decision items 1, 3, and caller-owned cooldown mechanics unchanged):

1. **Settled nine-role effort-tiered defaults.**
   - Single-target Codex roles (no cross-model fallback): `synth` → `gpt-6-astra@xhigh`; `intent`, `decompose` → `gpt-6.1-terra@high`; `judge` → `gpt-6.1-terra@medium`; `keyword` → `gpt-6.1-terra@low`.
   - Multi-target GLM-primary production roles with Codex quota fallback: `verify` → `glm-5.3-flash@low,gpt-6.1-terra@low`; `gate` → `glm-5.3-flash@high,gpt-6.1-terra@medium`; `build` → `glm-5.3@max,gpt-6.1-terra@high`.
   - Single-target cross-family evaluation judge: `eval_judge` → `glm-5.3@max` with no default cross-family GPT fallback so benchmarks preserve a single-ruler invariant. Explicit multi-target overrides in `PAPERVAULT_LLM_EVAL_JUDGE` (for example same-family `glm-5.3@max,relay-glm-5.3@max`) remain supported through `create_openai_chat_completion_with_fallback`.
2. **Gateway `v0.7.3` per-family modifier grammar.** After stripping an optional `openai/` prefix, `parse_targets` validates modifiers case-insensitively per family and raises `ValueError` (naming `PAPERVAULT_LLM_<ROLE>` when present) on unsupported combinations:
   - `glm-*`, `relay-glm-*`: bare (`reasoning_effort=None, thinking=None`, formatted `<model>`) or `@low|@high|@max` (`reasoning_effort=mod`, formatted `<model>@<mod>`); reject `@none`, `@nothinking`, `@thinking`, `@medium`, `@xhigh`, and unknown modifiers.
   - `gpt-6-astra*`, `gpt-6.1-sol*`: bare or `@low|@medium|@high|@xhigh|@max` (`reasoning_effort`); reject `@none`, `@thinking`, `@nothinking`, and unknown modifiers.
   - `gpt-6.1-terra*`, `gpt-6-luna*`: bare or `@none|@low|@medium|@high|@xhigh|@max` (`reasoning_effort`); reject `@thinking`, `@nothinking`, and unknown modifiers.
   - `deepseek-v4*`, `deepseek-flash*`, `relay-deepseek-*`: bare, `@low|@high|@max` (`reasoning_effort`), or `@nothinking|@none` (`thinking=False`, formatted `<model>@nothinking`); reject `@medium`, `@xhigh`, `@thinking`, and unknown modifiers.
   - `mimo-v2.6-*`, `relay-mimo-v2.6-*`: bare, `@thinking` (`thinking=True`), or `@nothinking|@none` (`thinking=False`, formatted `<model>@nothinking`); reject effort tiers (`@low..@max`) and unknown modifiers.
   - Other OpenAI-compatible model IDs (not matching the five gateway family prefixes above): continue accepting bare, `@none|@low|@medium|@high|@xhigh|@max`, or `@thinking|@nothinking`.
3. **Omit default `max_tokens` on the wire unless explicitly configured.** Default `KS_LLM_MAX_TOKENS`, `PAPERVAULT_LIB_MAX_TOKENS`, `KS_SYNTH_MAX_TOKENS`, and `JUDGE_MAX_TOKENS` to `None`, remove `_ab_u1_judge`'s hardcoded `max_tokens=4000`, and strip `max_tokens=None` from OpenAI and LiteLLM completion kwargs so `max_tokens` is sent only when an operator env var or caller passes an explicit integer. Record `max_tokens` (`None` when unset, `int` when set) in `effective_distill_recipe()` and `extraction_registration.effective_recipe()` so recipe digests remain sensitive to explicit output caps.

Rejected alternatives:
- *Retain default `glm-5.3,gpt-6.1-terra@high` failover on `eval_judge`*: rejected because silent mid-run switching between GLM and Codex invalidates relative A/B comparisons; failing loudly or retrying within `glm-5.3@max` (unless an operator explicitly opts into same-family relay failover) preserves measurement integrity.
- *Pin higher hardcoded `max_tokens` numbers in PaperVault instead of omitting*: rejected because per-provider ceilings differ (`131072` vs `393216` vs ignored) and Gateway `v0.7.3` owns authoritative per-model ceiling injection.

## Consequences

- `papervault doctor` and startup route parsing reject stale `glm-*@nothinking`, `glm-*@thinking`, `gpt-6-astra@none`, `mimo-v2.6-*@high`, and retired level aliases immediately with actionable variable names.
- High-effort stages (`build` on `glm-5.3@max`, `synth` on `gpt-6-astra@xhigh`, `eval_judge` on `glm-5.3@max`) receive full upstream reasoning and output headroom without client-imposed token caps, while outer wall-clock timeouts continue bounding runaway calls.
- Knowledge distillation and library extraction recipe digests canonically serialize `"max_tokens": null` under default configuration and change deterministically when an explicit token cap is configured.
