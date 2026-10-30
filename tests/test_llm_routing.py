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
    ("build", "glm-5.3,gpt-6.1-terra@high"),
    ("keyword", "gpt-6.1-terra@high"),
    ("synth", "gpt-6-astra@xhigh"),
    ("gate", "glm-5.3-flash,gpt-6.1-terra@high"),
    ("verify", "glm-5.3-flash,gpt-6.1-terra@high"),
    ("judge", "gpt-6.1-terra@high"),
    ("decompose", "gpt-6.1-terra@high"),
    ("eval_judge", "glm-5.3,gpt-6.1-terra@high"),
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
    for role in ("keyword", "intent", "decompose", "judge"):
        assert parse_targets(route(role)) == (
            ModelTarget("gpt-6.1-terra", reasoning_effort="high"),
        )
    for role in ("build", "eval_judge"):
        assert parse_targets(route(role)) == (
            ModelTarget("glm-5.3", thinking=True),
            ModelTarget("gpt-6.1-terra", reasoning_effort="high"),
        )
    for role in ("gate", "verify"):
        assert parse_targets(route(role)) == (
            ModelTarget("glm-5.3-flash", thinking=True),
            ModelTarget("gpt-6.1-terra", reasoning_effort="high"),
        )


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("role,want", ROLE_DEFAULTS)
def test_blank_override_uses_role_default(monkeypatch, role, want, blank):
    monkeypatch.setenv(f"PAPERVAULT_LLM_{role.upper()}", blank)
    assert route(role) == want


def test_whitespace_and_openai_prefix_normalized(monkeypatch):
    monkeypatch.setenv(
        "PAPERVAULT_LLM_BUILD",
        "  openai/glm-5.3@thinking , openai/gpt-6.1-terra@high  ",
    )
    assert route(" BUILD ") == "glm-5.3,gpt-6.1-terra@high"
    assert parse_targets(route("build")) == (
        ModelTarget("glm-5.3", thinking=True),
        ModelTarget("gpt-6.1-terra", reasoning_effort="high"),
    )


def test_glm_modifier_rules_and_canonical_formatting():
    # Omitted modifier on glm-* defaults to thinking=True and formats bare.
    t_default = parse_targets("glm-5.3")
    assert t_default == (ModelTarget("glm-5.3", thinking=True),)
    assert format_target(t_default[0]) == "glm-5.3"

    # Case-insensitive glm-* detection.
    t_upper = parse_targets("GLM-5.3-Flash")
    assert t_upper == (ModelTarget("GLM-5.3-Flash", thinking=True),)

    # @thinking explicitly sets thinking=True and canonicalizes to bare glm model.
    t_on = parse_targets("glm-5.3-flash@thinking")
    assert t_on == (ModelTarget("glm-5.3-flash", thinking=True),)
    assert format_target(t_on[0]) == "glm-5.3-flash"

    # @nothinking and @none both disable thinking on glm-* and format as @nothinking.
    for spec in ("glm-5.3@nothinking", "glm-5.3@none"):
        t_off = parse_targets(spec)
        assert t_off == (ModelTarget("glm-5.3", thinking=False),)
        assert format_target(t_off[0]) == "glm-5.3@nothinking"
        assert parse_targets(format_target(t_off[0])) == t_off


def test_non_glm_modifier_rules_and_canonical_formatting():
    # Plain non-glm model leaves reasoning_effort=None and thinking=None.
    for plain in ("gpt-4.1-mini", "provider-model"):
        targets = parse_targets(plain)
        assert targets == (ModelTarget(plain, reasoning_effort=None, thinking=None),)
        assert format_target(targets[0]) == plain

    # Effort modifiers on non-glm models.
    for effort in EFFORTS:
        targets = parse_targets(f"gpt-6.1-terra@{effort}")
        assert targets == (ModelTarget("gpt-6.1-terra", reasoning_effort=effort, thinking=None),)
        assert format_target(targets[0]) == f"gpt-6.1-terra@{effort}"

    # @thinking / @nothinking on non-glm models round-trip cleanly.
    t_think = parse_targets("deepseek-r1@thinking")
    assert t_think == (ModelTarget("deepseek-r1", reasoning_effort=None, thinking=True),)
    assert format_target(t_think[0]) == "deepseek-r1@thinking"
    assert parse_targets(format_target(t_think[0])) == t_think

    t_nothink = parse_targets("deepseek-r1@nothinking")
    assert t_nothink == (ModelTarget("deepseek-r1", reasoning_effort=None, thinking=False),)
    assert format_target(t_nothink[0]) == "deepseek-r1@nothinking"
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
        "glm-5.3, ,gpt-6.1-terra@high",
        "gpt 6.1",
        "gpt-6.1-terra @high",
        "gpt-6.1-terra@ high",
        "gpt-6.1-terra@high@thinking",
        "@high",
        "gpt-6.1-terra@",
        "openai/",
        "openai/@high",
        # GLM effort tiers & unknown modifiers rejected
        "glm-5.3@low",
        "glm-5.3@medium",
        "glm-5.3@high",
        "glm-5.3@xhigh",
        "glm-5.3@max",
        "glm-5.3@bogus",
        "gpt-6.1-terra@bogus",
    ],
)
def test_invalid_override_names_the_variable(monkeypatch, bad):
    monkeypatch.setenv("PAPERVAULT_LLM_SYNTH", bad)
    with pytest.raises(ValueError, match="PAPERVAULT_LLM_SYNTH"):
        route("synth")


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
    assert route("build") == "glm-5.3,gpt-6.1-terra@high"
    assert route("keyword") == "gpt-6.1-terra@high"
    assert route("verify") == "glm-5.3-flash,gpt-6.1-terra@high"
    assert route("eval_judge") == "glm-5.3,gpt-6.1-terra@high"


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

    glm_on = ModelTarget("glm-5.3", thinking=True)
    assert openai_wire_kwargs(glm_on) == {
        "model": "glm-5.3",
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert litellm_wire_kwargs(glm_on) == {
        "model": "openai/glm-5.3",
        "extra_body": {"thinking": {"type": "enabled"}},
    }

    glm_off = ModelTarget("glm-5.3-flash", thinking=False)
    assert openai_wire_kwargs(glm_off) == {
        "model": "glm-5.3-flash",
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    assert litellm_wire_kwargs(glm_off) == {
        "model": "openai/glm-5.3-flash",
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
        ("gpt-6.1-terra@high", "openai/gpt-6.1-terra", {"reasoning_effort": "high"}),
        ("glm-5.3-flash,gpt-6.1-terra@high", "openai/glm-5.3-flash", {"thinking": {"type": "enabled"}}),
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

    monkeypatch.setattr(llm, "_create_completion", complete)
    assert llm.get_llm().call("judge") == "ok"
    expected = {
        "model": want_model,
        "messages": [{"role": "user", "content": "judge"}],
        "base_url": "https://gateway.example/v1" if gateway else "https://direct.example/v1",
        "api_key": "test-key",
        "max_tokens": llm._DEFAULT_MAX_TOKENS,
    }
    if want_extra_body is not None:
        expected["extra_body"] = want_extra_body
    assert requests == [expected]


def test_doctor_accepts_concrete_route_defaults(tmp_path, monkeypatch, capsys):
    from papervault import cli

    monkeypatch.setattr(config, "LLM_KEYS_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    report = cli._Report()
    cli._check_config(report)
    assert report.failures == 0
    output = capsys.readouterr().out
    assert "build=glm-5.3,gpt-6.1-terra@high" in output
    assert "synth=gpt-6-astra@xhigh" in output
    assert "decompose=gpt-6.1-terra@high" in output


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
async def test_eval_and_baseline_callers_fail_over_glm_to_gpt_and_honor_cooldown(
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

    # 1. search_relevance_baseline._judge: 429 on glm-5.3 fails over immediately to gpt-6.1-terra@high
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
    assert [c["model"] for c in srb_calls] == ["glm-5.3", "gpt-6.1-terra", "gpt-6.1-terra"]
    assert srb_calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert srb_calls[1]["reasoning_effort"] == "high" and "extra_body" not in srb_calls[1]
    assert any("glm-5.3 -> gpt-6.1-terra" in r.getMessage() for r in caplog.records)

    # Deterministic 401 does NOT fail over to gpt-6.1-terra:
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

    # 2. _ab_u1_judge._judge: timeout on glm-5.3 fails over immediately to gpt-6.1-terra@high
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
    assert [c["model"] for c in u1_calls] == ["glm-5.3", "gpt-6.1-terra"]

    # 3. judge_mimo._judge_one: 429 on glm-5.3 fails over in attempt 1 without sleeping/burning attempt
    clear_fallback_cooldowns()
    judge_dir = tmp_path / "judge" / "vmq_r1"
    judge_dir.mkdir(parents=True)
    (judge_dir / "q1.prompt.txt").write_text("judge prompt")
    monkeypatch.setattr(
        judge_mimo, "eval_path", lambda *parts: tmp_path.joinpath(*parts)
    )
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
    monkeypatch.setattr(judge_mimo.JA, "validate_judge_json", lambda d, n_gold_nuggets=None: None)
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
    assert [c["model"] for c in jm_calls] == ["glm-5.3", "gpt-6.1-terra"]
