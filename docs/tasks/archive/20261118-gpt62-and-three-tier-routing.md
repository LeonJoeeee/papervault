# Task: Upgrade `synth` to `gpt-6.2-astra` and Establish 3-Tier GLM → Relay → Codex Routing

## Goal & Why

1. Upgrade `synth` default route from `gpt-6-astra@xhigh` to `gpt-6.2-astra@xhigh` and register `gpt-6.2-astra` and `gpt-6.2-sol` in `_CODEX_ALWAYS_ON_PREFIXES` (`low|medium|high|xhigh|max`).
2. Align role defaults with the owner's explicit 3-tier priority (**1. Official GLM weekly subscription → 2. Third-party `relay-*` balance → 3. Codex subscription**), keeping interactive `intent` on low-latency `gpt-6.1-terra@high` and preserving the single-family GLM-5.3 ruler invariant on `eval_judge`:
   - `intent`: `gpt-6.1-terra@high`
   - `synth`: `gpt-6.2-astra@xhigh`
   - `build`: `glm-5.3@max,relay-glm-5.3@max,gpt-6.1-terra@high`
   - `gate`: `glm-5.3-flash@high,relay-glm-5.3-flash@high,gpt-6.1-terra@medium`
   - `verify`: `glm-5.3-flash@low,relay-glm-5.3-flash@low,gpt-6.1-terra@low`
   - `keyword`: `glm-5.3-flash@low,relay-glm-5.3-flash@low,gpt-6.1-terra@low`
   - `judge`: `glm-5.3-flash@high,relay-glm-5.3-flash@high,gpt-6.1-terra@medium`
   - `decompose`: `glm-5.3-flash@high,relay-glm-5.3-flash@high,gpt-6.1-terra@high`
   - `eval_judge`: `glm-5.3@max,relay-glm-5.3@max`
3. Fix the relay balance-exhaustion failover gap across all multi-target callers (`llm_routing.create_openai_chat_completion_with_fallback`, `library/llm._is_failover_eligible`, and `knowledge/store/llm._is_failover_eligible`): recognize upstream quota/balance exhaustion responses (`HTTP 402`, or `HTTP 400/402/403/429` carrying quota/balance indicators such as `quota`, `insufficient`, `balance`, `credit`, `exhausted`, `套餐`, `额度`, `余额`, `限额`, `预扣费`, `weekly`, `5小时`, `5-hour`, or Zhipu codes `1304|1308|1309|1113`) as failover-eligible on non-final targets so exhausted `relay-*` legs record cooldown and advance cleanly to Tier 3 (`gpt-6.1-terra`), while genuine non-quota `400/401/403/422` errors continue failing fast.
4. Record `docs/adr/0014-three-tier-glm-relay-codex-routing-and-gpt62.md` (amending ADR-0013), update living docs/examples, and bump the synchronized single repo version from `0.10.30` to `0.10.31` per `AGENTS.md` and `README.md` ("Releases", owner-authorized for live cutover).

## Bounds

- Allowed paths:
  - `src/papervault/llm_routing.py`
  - `src/papervault/library/llm.py`
  - `src/papervault/knowledge/store/llm.py`
  - `docs/adr/0013-effort-tiered-role-routing-and-omitted-max-tokens.md` (status amendment link only)
  - `docs/adr/0014-three-tier-glm-relay-codex-routing-and-gpt62.md`
  - `.env.example`, `README.md`, `docs/INSTALL.md`, `docs/architecture.md`, `docs/eval.md`
  - Synchronized version manifests (`0.10.31`): `pyproject.toml`, `uv.lock`, `plugin/plugin.json`, `plugin/.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`
  - Unit tests under `tests/` (`tests/test_llm_routing.py`, `tests/library/test_llm_rotation.py`, `tests/library/test_llm_stream.py`, `tests/knowledge/test_llm_pool.py`, `tests/knowledge/test_catalog_intake.py`, `tests/knowledge/test_synth_retry.py`, and any other unit test asserting old default role strings)
- Non-goals: Do not touch the primary checkout's `.venv`, `.env`, live database, or `papervault.service`.

## Done-check

- Inside the worktree: `UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev`
- `.venv/bin/ruff check src tests` passes cleanly.
- `.venv/bin/pytest -m 'not integration' -q` passes cleanly (including `tests/test_plugin_version_sync.py` and new 3-hop `glm -> relay (402 / 403 quota) -> gpt-6.1-terra` failover tests).
