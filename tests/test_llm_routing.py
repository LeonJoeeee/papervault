"""Direct per-role model + thinking/effort routing, wire kwargs, and cooldown tests (issue #350)."""
from __future__ import annotations

import pytest

from papervault import config, llm_routing
from papervault.llm_routing import (
    EFFORTS,
    ROLES,
    ModelTarget,
    clear_fallback_cooldowns,
    fallback_cooldown_seconds,
    format_target,
    format_targets,
    is_model_cooled_down,
    litellm_wire_kwargs,
    openai_wire_kwargs,
    parse_targets,
    record_model_cooldown,
    route,
)

ROLE_DEFAULTS = [
    ("intent", "gpt-6.1-terra@high"),
    ("build", "glm-5.3@max,gpt-6.1-terra@high"),
    ("keyword", "gpt-6.1-terra@low"),
    ("synth", "gpt-6-astra@xhigh"),
    ("gate", "glm-5.3-flash@high,gpt-6.1-terra@medium"),
    ("verify", "glm-5.3-flash@low,gpt-6.1-terra@low"),
    ("judge", "gpt-6.1-terra@medium"),
    ("decompose", "gpt-6.1-terra@high"),
    ("eval_judge", "glm-5.3@max"),
]


@pytest.fixture(autouse=True)
def clean_routes(monkeypatch):
    clear_fallback_cooldowns()
    for role, _ in ROLE_DEFAULTS:
        monkeypatch.delenv(f"PAPERVAULT_LLM_{role.upper()}", raising=False)
    monkeypatch.delenv("PAPERVAULT_LLM_FALLBACK_COOLDOWN_S", raising=False)
    for var in ("KS_BUILD_MODEL", "PAPER_PIPELINE_VERIFY_MODEL", "MIMO_MODEL"):
        monkeypatch.delenv(var, raising=False)
    yield
    clear_fallback_cooldowns()


def test_retired_level_exports_removed():
    assert not hasattr(llm_routing, "LEVELS")
    assert not hasattr(llm_routing, "gateway_model")
    assert ROLES == (
        "intent",
        "build",
        "keyword",
        "synth",
        "gate",
        "verify",
        "judge",
        "decompose",
        "eval_judge",
    )
    assert EFFORTS == ("none", "low", "medium", "high", "xhigh", "max")


@pytest.mark.parametrize("role,want", ROLE_DEFAULTS)
def test_agreed_role_default_and_round_trip(role, want):
    resolved = route(role)
    assert resolved == want
    parsed = parse_targets(resolved)
    assert format_targets(parsed) == want


def test_role_default_target_objects():
    assert parse_targets(route("synth")) == (
        ModelTarget("gpt-6-astra", reasoning_effort="xhigh"),
    )
    for role in ("intent", "decompose"):
        assert parse_targets(route(role)) == (
            ModelTarget("gpt-6.1-terra", reasoning_effort="high"),
        )
    assert parse_targets(route("keyword")) == (
        ModelTarget("gpt-6.1-terra", reasoning_effort="low"),
    )
    assert parse_targets(route("judge")) == (
        ModelTarget("gpt-6.1-terra", reasoning_effort="medium"),
    )
    assert parse_targets(route("verify")) == (
        ModelTarget("glm-5.3-flash", reasoning_effort="low"),
        ModelTarget("gpt-6.1-terra", reasoning_effort="low"),
    )
    assert parse_targets(route("gate")) == (
        ModelTarget("glm-5.3-flash", reasoning_effort="high"),
        ModelTarget("gpt-6.1-terra", reasoning_effort="medium"),
    )
    assert parse_targets(route("build")) == (
        ModelTarget("glm-5.3", reasoning_effort="max"),
        ModelTarget("gpt-6.1-terra", reasoning_effort="high"),
    )
    assert parse_targets(route("eval_judge")) == (
        ModelTarget("glm-5.3", reasoning_effort="max"),
    )


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("role,want", ROLE_DEFAULTS)
def test_blank_override_uses_role_default(monkeypatch, role, want, blank):
    monkeypatch.setenv(f"PAPERVAULT_LLM_{role.upper()}", blank)
    assert route(role) == want


def test_whitespace_and_openai_prefix_normalized(monkeypatch):
    monkeypatch.setenv(
        "PAPERVAULT_LLM_BUILD",
        "  openai/glm-5.3@MAX , openai/gpt-6.1-terra@HIGH  ",
    )
    assert route(" BUILD ") == "glm-5.3@max,gpt-6.1-terra@high"
    assert parse_targets(route("build")) == (
        ModelTarget("glm-5.3", reasoning_effort="max"),
        ModelTarget("gpt-6.1-terra", reasoning_effort="high"),
    )


def test_glm_and_relay_glm_modifier_rules_and_wire_kwargs():
    for raw_model, want_model in (
        ("glm-5.3", "glm-5.3"),
        ("GLM-5.3-Flash", "GLM-5.3-Flash"),
        ("relay-glm-5.3", "relay-glm-5.3"),
        ("openai/relay-glm-5.3-flash", "relay-glm-5.3-flash"),
    ):
        bare = parse_targets(raw_model)
        assert bare == (ModelTarget(want_model, reasoning_effort=None, thinking=None),)
        assert format_target(bare[0]) == want_model
        assert openai_wire_kwargs(bare[0]) == {"model": want_model}
        assert litellm_wire_kwargs(bare[0]) == {"model": f"openai/{want_model}"}

        for effort in ("low", "high", "max"):
            tiered = parse_targets(f"{raw_model}@{effort.upper()}")
            assert tiered == (ModelTarget(want_model, reasoning_effort=effort, thinking=None),)
            assert format_target(tiered[0]) == f"{want_model}@{effort}"
            assert parse_targets(format_target(tiered[0])) == tiered
            assert openai_wire_kwargs(tiered[0]) == {
                "model": want_model,
                "reasoning_effort": effort,
            }
            assert litellm_wire_kwargs(tiered[0]) == {
                "model": f"openai/{want_model}",
                "extra_body": {"reasoning_effort": effort},
            }


def test_codex_family_modifier_rules_and_wire_kwargs():
    # Always-on Codex models (gpt-6-astra*, gpt-6.1-sol*): bare or @low..@max.
    for raw_model, want_model in (
        ("gpt-6-astra", "gpt-6-astra"),
        ("gpt-6.1-sol", "gpt-6.1-sol"),
        ("openai/GPT-6-Astra-Preview", "GPT-6-Astra-Preview"),
    ):
        bare = parse_targets(raw_model)
        assert bare == (ModelTarget(want_model, reasoning_effort=None, thinking=None),)
        assert format_target(bare[0]) == want_model
        for effort in ("low", "medium", "high", "xhigh", "max"):
            tiered = parse_targets(f"{raw_model}@{effort}")
            assert tiered == (ModelTarget(want_model, reasoning_effort=effort, thinking=None),)
            assert format_target(tiered[0]) == f"{want_model}@{effort}"

    # Disable-capable Codex models (gpt-6.1-terra*, gpt-6-luna*): bare or @none..@max.
    for raw_model, want_model in (
        ("gpt-6.1-terra", "gpt-6.1-terra"),
        ("gpt-6-luna", "gpt-6-luna"),
        ("openai/GPT-6.1-Terra", "GPT-6.1-Terra"),
    ):
        bare = parse_targets(raw_model)
        assert bare == (ModelTarget(want_model, reasoning_effort=None, thinking=None),)
        assert format_target(bare[0]) == want_model
        for effort in EFFORTS:
            tiered = parse_targets(f"{raw_model}@{effort}")
            assert tiered == (ModelTarget(want_model, reasoning_effort=effort, thinking=None),)
            assert format_target(tiered[0]) == f"{want_model}@{effort}"
            assert openai_wire_kwargs(tiered[0]) == {
                "model": want_model,
                "reasoning_effort": effort,
            }


def test_deepseek_gateway_family_modifier_rules_and_wire_kwargs():
    for raw_model, want_model in (
        ("deepseek-v4", "deepseek-v4"),
        ("deepseek-flash", "deepseek-flash"),
        ("relay-deepseek-v4", "relay-deepseek-v4"),
        ("openai/DeepSeek-V4", "DeepSeek-V4"),
    ):
        bare = parse_targets(raw_model)
        assert bare == (ModelTarget(want_model, reasoning_effort=None, thinking=None),)
        assert format_target(bare[0]) == want_model

        for effort in ("low", "high", "max"):
            tiered = parse_targets(f"{raw_model}@{effort}")
            assert tiered == (ModelTarget(want_model, reasoning_effort=effort, thinking=None),)
            assert format_target(tiered[0]) == f"{want_model}@{effort}"
            assert openai_wire_kwargs(tiered[0]) == {
                "model": want_model,
                "reasoning_effort": effort,
            }
            assert litellm_wire_kwargs(tiered[0]) == {
                "model": f"openai/{want_model}",
                "extra_body": {"reasoning_effort": effort},
            }

        for mod in ("nothinking", "none"):
            disabled = parse_targets(f"{raw_model}@{mod}")
            assert disabled == (ModelTarget(want_model, reasoning_effort=None, thinking=False),)
            assert format_target(disabled[0]) == f"{want_model}@nothinking"
            assert parse_targets(format_target(disabled[0])) == disabled
            assert openai_wire_kwargs(disabled[0]) == {
                "model": want_model,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
            assert litellm_wire_kwargs(disabled[0]) == {
                "model": f"openai/{want_model}",
                "extra_body": {"thinking": {"type": "disabled"}},
            }


def test_mimo_gateway_family_modifier_rules_and_wire_kwargs():
    for raw_model, want_model in (
        ("mimo-v2.6-pro", "mimo-v2.6-pro"),
        ("mimo-v2.6-flash", "mimo-v2.6-flash"),
        ("relay-mimo-v2.6-pro", "relay-mimo-v2.6-pro"),
        ("openai/MiMo-v2.6-Pro", "MiMo-v2.6-Pro"),
    ):
        bare = parse_targets(raw_model)
        assert bare == (ModelTarget(want_model, reasoning_effort=None, thinking=None),)
        assert format_target(bare[0]) == want_model

        enabled = parse_targets(f"{raw_model}@thinking")
        assert enabled == (ModelTarget(want_model, reasoning_effort=None, thinking=True),)
        assert format_target(enabled[0]) == f"{want_model}@thinking"
        assert parse_targets(format_target(enabled[0])) == enabled
        assert openai_wire_kwargs(enabled[0]) == {
            "model": want_model,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
        assert litellm_wire_kwargs(enabled[0]) == {
            "model": f"openai/{want_model}",
            "extra_body": {"thinking": {"type": "enabled"}},
        }

        for mod in ("nothinking", "none"):
            disabled = parse_targets(f"{raw_model}@{mod}")
            assert disabled == (ModelTarget(want_model, reasoning_effort=None, thinking=False),)
            assert format_target(disabled[0]) == f"{want_model}@nothinking"
            assert parse_targets(format_target(disabled[0])) == disabled
            assert openai_wire_kwargs(disabled[0]) == {
                "model": want_model,
                "extra_body": {"thinking": {"type": "disabled"}},
            }


def test_arbitrary_openai_compatible_modifier_rules_and_canonical_formatting():
    for plain in ("gpt-4.1-mini", "deepseek-r1", "provider-model"):
        targets = parse_targets(plain)
        assert targets == (ModelTarget(plain, reasoning_effort=None, thinking=None),)
        assert format_target(targets[0]) == plain

        for effort in EFFORTS:
            tiered = parse_targets(f"{plain}@{effort}")
            assert tiered == (ModelTarget(plain, reasoning_effort=effort, thinking=None),)
            assert format_target(tiered[0]) == f"{plain}@{effort}"

        t_think = parse_targets(f"{plain}@thinking")
        assert t_think == (ModelTarget(plain, reasoning_effort=None, thinking=True),)
        assert format_target(t_think[0]) == f"{plain}@thinking"
        assert parse_targets(format_target(t_think[0])) == t_think

        t_nothink = parse_targets(f"{plain}@nothinking")
        assert t_nothink == (ModelTarget(plain, reasoning_effort=None, thinking=False),)
        assert format_target(t_nothink[0]) == f"{plain}@nothinking"
        assert parse_targets(format_target(t_nothink[0])) == t_nothink


@pytest.mark.parametrize(
    "bad",
    [
        # Retired bare levels (case-insensitive, with or without openai/ or @modifier)
        "flash",
        "standard",
        "pro",
        "max",
        "ultra",
        "ULTRA",
        "openai/pro",
        "flash@high",
        # Retired papervault-* aliases (case-insensitive)
        "papervault-flash",
        "papervault-standard",
        "papervault-pro",
        "papervault-max",
        "papervault-ultra",
        "PAPERVAULT-MAX",
        "openai/papervault-standard",
        # Syntax errors: empty comma segments, internal whitespace, malformed @
        "gpt-6.1-terra,",
        ",gpt-6.1-terra",
        "glm-5.3@max, ,gpt-6.1-terra@high",
        "gpt 6.1",
        "gpt-6.1-terra @high",
        "gpt-6.1-terra@ high",
        "gpt-6.1-terra@high@thinking",
        "@high",
        "gpt-6.1-terra@",
        "openai/",
        "openai/@high",
        # GLM / relay-glm rejections: @none, @nothinking, @thinking, @medium, @xhigh, unknown
        "glm-5.3@none",
        "glm-5.3@nothinking",
        "glm-5.3@thinking",
        "glm-5.3@medium",
        "glm-5.3@xhigh",
        "glm-5.3@bogus",
        "relay-glm-5.3@none",
        "relay-glm-5.3@nothinking",
        "relay-glm-5.3@thinking",
        "relay-glm-5.3@medium",
        "relay-glm-5.3@xhigh",
        "openai/GLM-5.3-Flash@medium",
        # Codex always-on rejections: @none, @thinking, @nothinking, unknown
        "gpt-6-astra@none",
        "gpt-6-astra@thinking",
        "gpt-6-astra@nothinking",
        "gpt-6-astra@bogus",
        "gpt-6.1-sol@none",
        "gpt-6.1-sol@thinking",
        "gpt-6.1-sol@nothinking",
        # Codex disable-capable rejections: @thinking, @nothinking, unknown
        "gpt-6.1-terra@thinking",
        "gpt-6.1-terra@nothinking",
        "gpt-6.1-terra@bogus",
        "gpt-6-luna@thinking",
        "gpt-6-luna@nothinking",
        # DeepSeek gateway rejections: @medium, @xhigh, @thinking, unknown
        "deepseek-v4@medium",
        "deepseek-v4@xhigh",
        "deepseek-v4@thinking",
        "deepseek-v4@bogus",
        "deepseek-flash@medium",
        "deepseek-flash@thinking",
        "relay-deepseek-v4@xhigh",
        "relay-deepseek-v4@thinking",
        # MiMo chat gateway rejections: effort tiers + unknown
        "mimo-v2.6-pro@low",
        "mimo-v2.6-pro@medium",
        "mimo-v2.6-pro@high",
        "mimo-v2.6-pro@xhigh",
        "mimo-v2.6-pro@max",
        "mimo-v2.6-pro@bogus",
        "mimo-v2.6-flash@high",
        "relay-mimo-v2.6-pro@low",
        "relay-mimo-v2.6-pro@max",
        # Arbitrary OpenAI-compatible unknown modifier
        "gpt-4.1-mini@bogus",
        "deepseek-r1@bogus",
    ],
)
def test_invalid_override_names_the_variable(monkeypatch, bad):
    monkeypatch.setenv("PAPERVAULT_LLM_SYNTH", bad)
    with pytest.raises(ValueError, match="PAPERVAULT_LLM_SYNTH"):
        route("synth")
    with pytest.raises(ValueError):
        parse_targets(bad)


def test_parse_targets_rejects_empty_spec():
    with pytest.raises(ValueError):
        parse_targets("")
    with pytest.raises(ValueError):
        parse_targets("   ")


def test_unknown_role_rejected():
    with pytest.raises(ValueError, match="unknown LLM role"):
        route("unknown")


def test_retired_legacy_env_vars_ignored(monkeypatch):
    for var in (
        "PAPERVAULT_MODEL",
        "PAPERVAULT_BUILD_MODEL",
        "KS_BUILD_MODEL",
        "PAPER_PIPELINE_VERIFY_MODEL",
        "MIMO_MODEL",
    ):
        monkeypatch.setenv(var, "legacy-ignored-model")
    assert route("build") == "glm-5.3@max,gpt-6.1-terra@high"
    assert route("keyword") == "gpt-6.1-terra@low"
    assert route("verify") == "glm-5.3-flash@low,gpt-6.1-terra@low"
    assert route("eval_judge") == "glm-5.3@max"


def test_wire_kwargs_builders():
    gpt = ModelTarget("gpt-6-astra", reasoning_effort="xhigh")
    assert openai_wire_kwargs(gpt) == {
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
    }
    litellm_gpt = litellm_wire_kwargs(gpt)
    assert litellm_gpt == {
        "model": "openai/gpt-6-astra",
        "extra_body": {"reasoning_effort": "xhigh"},
    }
    assert "reasoning_effort" not in litellm_gpt

    glm_max = ModelTarget("glm-5.3", reasoning_effort="max")
    assert openai_wire_kwargs(glm_max) == {
        "model": "glm-5.3",
        "reasoning_effort": "max",
    }
    assert litellm_wire_kwargs(glm_max) == {
        "model": "openai/glm-5.3",
        "extra_body": {"reasoning_effort": "max"},
    }

    mimo_on = ModelTarget("mimo-v2.6-pro", thinking=True)
    assert openai_wire_kwargs(mimo_on) == {
        "model": "mimo-v2.6-pro",
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert litellm_wire_kwargs(mimo_on) == {
        "model": "openai/mimo-v2.6-pro",
        "extra_body": {"thinking": {"type": "enabled"}},
    }

    ds_off = ModelTarget("deepseek-v4", thinking=False)
    assert openai_wire_kwargs(ds_off) == {
        "model": "deepseek-v4",
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    assert litellm_wire_kwargs(ds_off) == {
        "model": "openai/deepseek-v4",
        "extra_body": {"thinking": {"type": "disabled"}},
    }

    plain = ModelTarget("gpt-4.1-mini")
    assert openai_wire_kwargs(plain) == {"model": "gpt-4.1-mini"}
    assert litellm_wire_kwargs(plain) == {"model": "openai/gpt-4.1-mini"}

    # Explicit non-openai/ provider prefix constructed directly is preserved.
    explicit = ModelTarget("anthropic/claude-sonnet-4-6", reasoning_effort="high")
    assert litellm_wire_kwargs(explicit) == {
        "model": "anthropic/claude-sonnet-4-6",
        "extra_body": {"reasoning_effort": "high"},
    }


def test_fallback_cooldown_lifecycle(monkeypatch):
    assert fallback_cooldown_seconds() == 120.0
    assert not is_model_cooled_down("glm-5.3", now=100.0)

    record_model_cooldown("glm-5.3", now=100.0)
    assert is_model_cooled_down("glm-5.3", now=100.0)
    assert is_model_cooled_down("glm-5.3", now=219.9)
    assert not is_model_cooled_down("glm-5.3", now=220.0)

    # Disabling cooldown via <= 0 prevents recording/skipping.
    monkeypatch.setenv("PAPERVAULT_LLM_FALLBACK_COOLDOWN_S", "0")
    assert fallback_cooldown_seconds() == 0.0
    record_model_cooldown("glm-5.3", now=300.0)
    assert not is_model_cooled_down("glm-5.3", now=300.0)

    monkeypatch.setenv("PAPERVAULT_LLM_FALLBACK_COOLDOWN_S", "-10")
    assert fallback_cooldown_seconds() == 0.0


@pytest.mark.parametrize("gateway", [False, True])
@pytest.mark.parametrize(
    "spec,want_model,want_extra_body",
    [
        ("gpt-6.1-terra@medium", "openai/gpt-6.1-terra", {"reasoning_effort": "medium"}),
        ("glm-5.3-flash@high,gpt-6.1-terra@medium", "openai/glm-5.3-flash", {"reasoning_effort": "high"}),
        ("gpt-4.1-mini", "openai/gpt-4.1-mini", None),
    ],
)
def test_library_get_llm_sends_litellm_wire_kwargs(
    tmp_path, monkeypatch, gateway, spec, want_model, want_extra_body,
):
    from types import SimpleNamespace
    from papervault.library import llm

    monkeypatch.setattr(config, "USE_GATEWAY", gateway)
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://direct.example/v1")
    monkeypatch.setattr(config, "GATEWAY_KEY", "test-key")
    monkeypatch.setattr(config, "GATEWAY_URL", "https://gateway.example/v1")
    monkeypatch.setattr(llm, "_KEYS_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(llm, "_pools", {})
    monkeypatch.setattr(llm, "_gw_llms", {})
    monkeypatch.setenv("PAPERVAULT_LLM_JUDGE", spec)
    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr(llm, "_DEFAULT_MAX_TOKENS", None)
    monkeypatch.setattr(llm, "_create_completion", complete)
    assert llm.get_llm().call("judge") == "ok"
    expected = {
        "model": want_model,
        "messages": [{"role": "user", "content": "judge"}],
        "base_url": "https://gateway.example/v1" if gateway else "https://direct.example/v1",
        "api_key": "test-key",
    }
    if want_extra_body is not None:
        expected["extra_body"] = want_extra_body
    assert requests == [expected]
    assert "max_tokens" not in requests[0]

    assert llm.get_llm(max_tokens=4096).call("judge") == "ok"
    assert requests[1] == {**expected, "max_tokens": 4096}


def test_doctor_accepts_concrete_route_defaults(tmp_path, monkeypatch, capsys):
    from papervault import cli

    monkeypatch.setattr(config, "LLM_KEYS_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    report = cli._Report()
    cli._check_config(report)
    assert report.failures == 0
    output = capsys.readouterr().out
    assert "build=glm-5.3@max,gpt-6.1-terra@high" in output
    assert "synth=gpt-6-astra@xhigh" in output
    assert "keyword=gpt-6.1-terra@low" in output
    assert "judge=gpt-6.1-terra@medium" in output
    assert "gate=glm-5.3-flash@high,gpt-6.1-terra@medium" in output
    assert "verify=glm-5.3-flash@low,gpt-6.1-terra@low" in output
    assert "eval_judge=glm-5.3@max" in output


def test_doctor_reports_retired_or_invalid_role_override(tmp_path, monkeypatch, capsys):
    from papervault import cli

    monkeypatch.setattr(config, "LLM_KEYS_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setenv("PAPERVAULT_LLM_SYNTH", "ultra")
    report = cli._Report()
    cli._check_config(report)
    assert report.failures == 1
    assert "PAPERVAULT_LLM_SYNTH" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_eval_and_baseline_callers_default_single_target_and_multi_target_override(
    tmp_path, monkeypatch, caplog,
):
    import asyncio
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from papervault.eval import _ab_u1_judge, judge_mimo

    spec_srb = importlib.util.spec_from_file_location(
        "search_relevance_baseline",
        Path(__file__).resolve().parents[1] / "scripts" / "search_relevance_baseline.py",
    )
    assert spec_srb and spec_srb.loader
    srb = importlib.util.module_from_spec(spec_srb)
    spec_srb.loader.exec_module(srb)
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    class FakeHTTPError(Exception):
        def __init__(self, status_code: int):
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    def make_client(outcomes):
        calls = []
        seq = iter(outcomes)

        async def create(**kwargs):
            calls.append(kwargs)
            nxt = next(seq)
            if isinstance(nxt, BaseException):
                raise nxt
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=nxt))]
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return client, calls

    valid_judge_reply = json.dumps(
        {
            "qid": "q1",
            "relevance": 90,
            "faithfulness": 1.0,
            "n_substantive_claims": 1,
            "n_claims_with_citation": 1,
            "citation_recall": 1.0,
            "citation_support_precision": 1.0,
            "citation_checks": [{"claim": "c", "cited_key": "K1", "verdict": "entail", "note": "ok"}],
            "nugget_recall": 1.0,
            "nugget_judgements": [{"nugget_id": 0, "coverage": "covered", "score": 1.0, "note": "ok"}],
            "trap_correct_refusal": None,
            "overall_note": "ok",
        }
    )
    judge_dir = tmp_path / "judge" / "vmq_r1"
    judge_dir.mkdir(parents=True)
    (judge_dir / "q1.prompt.txt").write_text("judge prompt")
    monkeypatch.setattr(
        judge_mimo, "eval_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    monkeypatch.setattr(judge_mimo.JA, "validate_judge_json", lambda d, n_gold_nuggets=None: None)

    # --- A. Default eval_judge='glm-5.3@max': single-target ruler invariant (no cross-family failover) ---
    clear_fallback_cooldowns()
    srb_single, srb_single_calls = make_client([FakeHTTPError(429), FakeHTTPError(429)])
    with pytest.raises(FakeHTTPError):
        await srb._judge(
            srb_single, "solar wind", [{"title": "Paper A", "year": 2024, "abstract": "abc"}]
        )
    assert [c["model"] for c in srb_single_calls] == ["glm-5.3", "glm-5.3"]
    assert all(
        c["reasoning_effort"] == "max" and "extra_body" not in c
        for c in srb_single_calls
    )
    assert not is_model_cooled_down("glm-5.3")

    u1_single, u1_single_calls = make_client(
        [TimeoutError("t1"), json.dumps({"faithful": True, "reason": "ok"})]
    )
    assert (
        await _ab_u1_judge._judge(
            u1_single,
            asyncio.Semaphore(1),
            {"head": "A", "tail": "B", "description": "rel", "source_chunks": ["A causes B"]},
        )
        is True
    )
    assert [c["model"] for c in u1_single_calls] == ["glm-5.3", "glm-5.3"]
    assert all(c["reasoning_effort"] == "max" for c in u1_single_calls)

    jm_single, jm_single_calls = make_client([FakeHTTPError(429), valid_judge_reply])
    counters_single = {"ok": 0, "skip": 0, "fail": 0}
    res_single = await judge_mimo._judge_one(
        jm_single,
        asyncio.Semaphore(1),
        "vmq_r1",
        "q1",
        {"qid": "q1", "gold_keys": ["K1"], "nuggets": ["n1"]},
        True,
        counters_single,
    )
    assert res_single == ("vmq_r1", "q1", "ok", "attempt=2")
    assert [c["model"] for c in jm_single_calls] == ["glm-5.3", "glm-5.3"]
    assert all(c["reasoning_effort"] == "max" for c in jm_single_calls)

    # --- B. Explicit multi-target PAPERVAULT_LLM_EVAL_JUDGE override fails over & honors cooldown ---
    monkeypatch.setenv("PAPERVAULT_LLM_EVAL_JUDGE", "glm-5.3@max,relay-glm-5.3@max")
    clear_fallback_cooldowns()
    srb_client, srb_calls = make_client(
        [
            FakeHTTPError(429),
            json.dumps({"items": [{"idx": 0, "score": 0.9}]}),
            json.dumps({"items": [{"idx": 0, "score": 0.8}]}),
        ]
    )
    with caplog.at_level("WARNING", logger="papervault.llm_routing"):
        scores_1 = await srb._judge(
            srb_client, "solar wind", [{"title": "Paper A", "year": 2024, "abstract": "abc"}]
        )
        # Second call while glm-5.3 is cooled down skips primary directly:
        scores_2 = await srb._judge(
            srb_client, "solar wind", [{"title": "Paper A", "year": 2024, "abstract": "abc"}]
        )
    assert scores_1 == [0.9] and scores_2 == [0.8]
    assert [c["model"] for c in srb_calls] == ["glm-5.3", "relay-glm-5.3", "relay-glm-5.3"]
    assert all(c["reasoning_effort"] == "max" and "extra_body" not in c for c in srb_calls)
    assert any("glm-5.3 -> relay-glm-5.3" in r.getMessage() for r in caplog.records)

    # Deterministic 401 does NOT fail over:
    clear_fallback_cooldowns()
    auth_client, auth_calls = make_client(
        [FakeHTTPError(401), json.dumps({"items": [{"idx": 0, "score": 1.0}]})]
    )
    with pytest.raises(FakeHTTPError):
        await srb._judge(auth_client, "solar wind", [{"title": "Paper A", "year": 2024}])
    assert [c["model"] for c in auth_calls] == ["glm-5.3"]

    # CancelledError propagates immediately without failover:
    cancel_client, cancel_calls = make_client([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await srb._judge(cancel_client, "solar wind", [{"title": "Paper A", "year": 2024}])
    assert [c["model"] for c in cancel_calls] == ["glm-5.3"]

    # _ab_u1_judge._judge: timeout on glm-5.3 fails over immediately to relay-glm-5.3@max
    clear_fallback_cooldowns()
    u1_client, u1_calls = make_client(
        [TimeoutError("upstream timeout"), json.dumps({"faithful": True, "reason": "direct"})]
    )
    verdict = await _ab_u1_judge._judge(
        u1_client,
        asyncio.Semaphore(1),
        {"head": "A", "tail": "B", "description": "rel", "source_chunks": ["A causes B"]},
    )
    assert verdict is True
    assert [c["model"] for c in u1_calls] == ["glm-5.3", "relay-glm-5.3"]

    # judge_mimo._judge_one: 429 on glm-5.3 fails over in attempt 1 without burning attempt
    clear_fallback_cooldowns()
    jm_client, jm_calls = make_client([FakeHTTPError(429), valid_judge_reply])
    counters = {"ok": 0, "skip": 0, "fail": 0}
    res = await judge_mimo._judge_one(
        jm_client,
        asyncio.Semaphore(1),
        "vmq_r1",
        "q1",
        {"qid": "q1", "gold_keys": ["K1"], "nuggets": ["n1"]},
        True,
        counters,
    )
    assert res == ("vmq_r1", "q1", "ok", "attempt=1")
    assert [c["model"] for c in jm_calls] == ["glm-5.3", "relay-glm-5.3"]


def test_extraction_registration_effective_recipe_records_max_tokens_and_digest(tmp_path, monkeypatch):
    import json
    import os
    import subprocess
    import sys

    from papervault.library import llm as lib_llm
    from papervault.library.services.extraction_registration import effective_recipe
    from papervault.store.knowledge import canonical_recipe

    monkeypatch.setattr(lib_llm, "_DEFAULT_MAX_TOKENS", None)
    default_recipe = effective_recipe("mineru2.5-pro")
    assert default_recipe["gate"]["configured_model"] == "glm-5.3-flash@high,gpt-6.1-terra@medium"
    assert default_recipe["gate"]["max_tokens"] is None
    canon_default, digest_default = canonical_recipe(default_recipe)
    assert canon_default["gate"]["max_tokens"] is None

    # KeyPool with no active groups still reports its configured max_tokens cleanly (None or int).
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    empty_pool = lib_llm.KeyPool(tmp_path / "absent.json", None, model_override="glm-5.3-flash@high")
    assert effective_recipe("mineru2.5-pro", llm=empty_pool)["gate"]["max_tokens"] is None
    capped_pool = lib_llm.KeyPool(tmp_path / "absent.json", 8192, model_override="glm-5.3-flash@high")
    capped_recipe = effective_recipe("mineru2.5-pro", llm=capped_pool)
    assert capped_recipe["gate"]["max_tokens"] == 8192
    assert canonical_recipe(capped_recipe)[1] != digest_default

    # Process startup check for PAPERVAULT_LIB_MAX_TOKENS unset vs explicit integer:
    program = """
import json
from papervault.library.services.extraction_registration import effective_recipe
from papervault.store.knowledge import canonical_recipe
r, d = canonical_recipe(effective_recipe('mineru2.5-pro'))
print(json.dumps({'digest': d, 'max_tokens': r['gate']['max_tokens']}))
"""
    base_env = {k: v for k, v in os.environ.items() if k != "PAPERVAULT_LIB_MAX_TOKENS"}
    base_env["PYTHON_DOTENV_DISABLED"] = "1"
    unset_proc = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=base_env,
        capture_output=True,
        text=True,
        check=True,
    )
    set_proc = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={**base_env, "PAPERVAULT_LIB_MAX_TOKENS": "16384"},
        capture_output=True,
        text=True,
        check=True,
    )
    unset_out = json.loads(unset_proc.stdout.strip().splitlines()[-1])
    set_out = json.loads(set_proc.stdout.strip().splitlines()[-1])
    assert unset_out["max_tokens"] is None
    assert set_out["max_tokens"] == 16384
    assert unset_out["digest"] != set_out["digest"]


@pytest.mark.asyncio
async def test_eval_judges_omit_default_max_tokens_and_honor_judge_max_tokens(tmp_path, monkeypatch):
    import asyncio
    import json
    import os
    import subprocess
    import sys
    from types import SimpleNamespace

    from papervault.eval import _ab_u1_judge, judge_mimo

    # Subprocess check: JUDGE_MAX_TOKENS defaults to None when unset/blank and parses explicit int.
    env = {k: v for k, v in os.environ.items() if k != "JUDGE_MAX_TOKENS"}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    for unset_val in (None, "", "   "):
        run_env = dict(env)
        if unset_val is not None:
            run_env["JUDGE_MAX_TOKENS"] = unset_val
        proc = subprocess.run(
            [sys.executable, "-c", "import json, papervault.eval.judge_mimo as m; print(json.dumps(m.MAX_TOKENS))"],
            cwd=tmp_path,
            env=run_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(proc.stdout.strip().splitlines()[-1]) is None

    proc_set = subprocess.run(
        [sys.executable, "-c", "import json, papervault.eval.judge_mimo as m; print(json.dumps(m.MAX_TOKENS))"],
        cwd=tmp_path,
        env={**env, "JUDGE_MAX_TOKENS": "16000"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(proc_set.stdout.strip().splitlines()[-1]) == 16000

    # Wire check for _ab_u1_judge and judge_mimo:
    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "faithful": True,
                                "reason": "ok",
                                "qid": "q1",
                                "relevance": 90,
                                "faithfulness": 1.0,
                                "n_substantive_claims": 0,
                                "n_claims_with_citation": 0,
                                "citation_recall": 1.0,
                                "citation_support_precision": 1.0,
                                "citation_checks": [],
                                "nugget_recall": 1.0,
                                "nugget_judgements": [],
                                "trap_correct_refusal": True,
                                "overall_note": "ok",
                            }
                        )
                    )
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    # 1. _ab_u1_judge omits max_tokens on wire (no hardcoded 4000):
    assert (
        await _ab_u1_judge._judge(
            client,
            asyncio.Semaphore(1),
            {"head": "A", "tail": "B", "description": "rel", "source_chunks": ["A causes B"]},
        )
        is True
    )
    assert "max_tokens" not in calls[-1]

    # 2. judge_mimo omits max_tokens when MAX_TOKENS is None and forwards when set:
    judge_dir = tmp_path / "judge" / "vmq_r1"
    judge_dir.mkdir(parents=True)
    (judge_dir / "q1.prompt.txt").write_text("judge prompt")
    monkeypatch.setattr(judge_mimo, "eval_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(judge_mimo.JA, "validate_judge_json", lambda d, n_gold_nuggets=None: None)

    monkeypatch.setattr(judge_mimo, "MAX_TOKENS", None)
    await judge_mimo._judge_one(
        client,
        asyncio.Semaphore(1),
        "vmq_r1",
        "q1",
        {"qid": "q1", "gold_keys": [], "nuggets": []},
        True,
        {"ok": 0, "skip": 0, "fail": 0},
    )
    assert "max_tokens" not in calls[-1]

    monkeypatch.setattr(judge_mimo, "MAX_TOKENS", 16000)
    await judge_mimo._judge_one(
        client,
        asyncio.Semaphore(1),
        "vmq_r1",
        "q1",
        {"qid": "q1", "gold_keys": [], "nuggets": []},
        True,
        {"ok": 0, "skip": 0, "fail": 0},
    )
    assert calls[-1]["max_tokens"] == 16000
