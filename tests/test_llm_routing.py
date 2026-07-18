"""Tests for the per-role LLM routing resolver (papervault.llm_routing, issue #8).

Covers: spec-syntax parse (bare / :think / :nothink / empty-error), per-role default
resolution (behavior-neutral defaults), the new PAPERVAULT_LLM_<ROLE> env override, and
the legacy back-compat precedence (KS_BUILD_MODEL / KS_BUILD_THINKING / KS_KW_THINKING /
PAPER_PIPELINE_VERIFY_MODEL win over the new var + reproduce graph.py's thinking logic).
"""
from __future__ import annotations

import pytest

from papervault import config
from papervault import llm_routing
from papervault.llm_routing import _parse_spec, route

# Every env var the resolver reads — cleared before each test for a hermetic baseline.
_ROUTING_ENV = (
    "PAPERVAULT_LLM_SYNTH",
    "PAPERVAULT_LLM_DECOMPOSE",
    "PAPERVAULT_LLM_BUILD",
    "PAPERVAULT_LLM_KEYWORD",
    "PAPERVAULT_LLM_JUDGE",
    "PAPERVAULT_LLM_GATE",
    "PAPERVAULT_LLM_VERIFY",
    "KS_BUILD_MODEL",
    "KS_BUILD_THINKING",
    "KS_KW_THINKING",
    "PAPER_PIPELINE_VERIFY_MODEL",
)

_SYNTH = "synth-model"
_BUILD = "build-model"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip every routing env var and pin the two config model slots to known sentinels."""
    for var in _ROUTING_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "SYNTH_MODEL", _SYNTH)
    monkeypatch.setattr(config, "BUILD_MODEL", _BUILD)


# --------------------------------------------------------------------------- spec parse

def test_parse_bare_model_sends_no_thinking():
    assert _parse_spec("mimo-v2.5-pro") == ("mimo-v2.5-pro", None)


def test_parse_think_suffix():
    assert _parse_spec("mimo-v2.5-pro:think") == ("mimo-v2.5-pro", True)


def test_parse_nothink_suffix():
    assert _parse_spec("mimo-v2.5:nothink") == ("mimo-v2.5", False)


def test_parse_is_case_insensitive_on_suffix():
    assert _parse_spec("mimo-v2.5:NoThink") == ("mimo-v2.5", False)
    assert _parse_spec("mimo-v2.5:THINK") == ("mimo-v2.5", True)


def test_parse_strips_whitespace():
    assert _parse_spec("  mimo-v2.5-pro  ") == ("mimo-v2.5-pro", None)
    assert _parse_spec("mimo-v2.5 :think") == ("mimo-v2.5", True)


def test_parse_preserves_provider_prefix():
    # A litellm-form model (openai/<name>) has "/" not ":" — the suffix parse must leave it intact.
    assert _parse_spec("openai/mimo-v2.5:nothink") == ("openai/mimo-v2.5", False)
    assert _parse_spec("openai/mimo-v2.5") == ("openai/mimo-v2.5", None)


@pytest.mark.parametrize("bad", ["", "   ", ":think", ":nothink", "  :think"])
def test_parse_empty_model_raises(bad):
    with pytest.raises(ValueError, match="empty model"):
        _parse_spec(bad)


# ------------------------------------------------------------------- default resolution

@pytest.mark.parametrize("role", ["synth", "decompose", "judge", "gate"])
def test_synth_slot_roles_default_to_synth_model_no_thinking(role):
    assert route(role) == (_SYNTH, None)


def test_build_defaults_to_build_model_thinking_on():
    # graph.py default: KS_BUILD_THINKING unset ("1") → enable_thinking=True.
    assert route("build") == (_BUILD, True)


def test_keyword_defaults_to_build_model_thinking_on():
    # graph.py keyword branch default (KS_KW_THINKING unset, KS_BUILD_THINKING unset "1") → True.
    assert route("keyword") == (_BUILD, True)


def test_verify_defaults_to_current_verify_model():
    assert route("verify") == ("openai/mimo-v2.5", None)


def test_synth_slot_roles_pass_through_empty_config_model(monkeypatch):
    # SYNTH_MODEL empty (PAPERVAULT_MODEL unset) → the resolver returns "" so the call site
    # rides the pool's per-group model (byte-identical to today).
    monkeypatch.setattr(config, "SYNTH_MODEL", "")
    assert route("synth") == ("", None)


def test_build_passes_through_empty_config_model(monkeypatch):
    monkeypatch.setattr(config, "BUILD_MODEL", "")
    assert route("build") == ("", True)


# --------------------------------------------------------------------- new-var override

def test_new_var_overrides_synth_with_thinking(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_LLM_SYNTH", "mimo-v2.5-pro:think")
    assert route("synth") == ("mimo-v2.5-pro", True)


def test_new_var_overrides_decompose_nothink(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_LLM_DECOMPOSE", "mimo-v2.5:nothink")
    assert route("decompose") == ("mimo-v2.5", False)


def test_new_var_bare_model_clears_thinking(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_LLM_JUDGE", "some-model")
    assert route("judge") == ("some-model", None)


def test_new_var_overrides_build(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_LLM_BUILD", "mimo-v2.5:nothink")
    assert route("build") == ("mimo-v2.5", False)


def test_new_var_overrides_keyword(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_LLM_KEYWORD", "mimo-v2.5:nothink")
    assert route("keyword") == ("mimo-v2.5", False)


def test_new_var_overrides_gate(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_LLM_GATE", "cheap:nothink")
    assert route("gate") == ("cheap", False)


def test_new_var_overrides_verify(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_LLM_VERIFY", "openai/mimo-v2.5:nothink")
    assert route("verify") == ("openai/mimo-v2.5", False)


def test_blank_new_var_falls_back_to_default(monkeypatch):
    # A blank line is treated as "not configured" (no error) → the default resolves.
    monkeypatch.setenv("PAPERVAULT_LLM_SYNTH", "   ")
    assert route("synth") == (_SYNTH, None)


def test_invalid_new_var_raises(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_LLM_SYNTH", ":think")
    with pytest.raises(ValueError, match="empty model"):
        route("synth")


# ------------------------------------------------------------ legacy back-compat precedence

def test_ks_build_model_overrides_build_model_and_new_var(monkeypatch):
    monkeypatch.setenv("KS_BUILD_MODEL", "legacy-build")
    monkeypatch.setenv("PAPERVAULT_LLM_BUILD", "new-build:nothink")
    # Legacy KS_BUILD_MODEL wins the MODEL; thinking still comes from the new var (:nothink).
    assert route("build") == ("legacy-build", False)


def test_ks_build_model_applies_to_keyword_too(monkeypatch):
    monkeypatch.setenv("KS_BUILD_MODEL", "legacy-build")
    assert route("keyword")[0] == "legacy-build"


def test_ks_build_thinking_zero_disables_build_thinking(monkeypatch):
    monkeypatch.setenv("KS_BUILD_THINKING", "0")
    assert route("build") == (_BUILD, None)


def test_ks_build_thinking_one_keeps_build_thinking(monkeypatch):
    monkeypatch.setenv("KS_BUILD_THINKING", "1")
    assert route("build") == (_BUILD, True)


def test_ks_build_thinking_wins_over_new_var(monkeypatch):
    monkeypatch.setenv("KS_BUILD_THINKING", "0")
    monkeypatch.setenv("PAPERVAULT_LLM_BUILD", "mimo:think")
    # Legacy thinking override wins → None (thinking off); model from the new var.
    assert route("build") == ("mimo", None)


def test_ks_kw_thinking_zero_disables_keyword_thinking(monkeypatch):
    monkeypatch.setenv("KS_KW_THINKING", "0")
    assert route("keyword") == (_BUILD, False)


def test_ks_kw_thinking_zero_wins_over_new_var_and_build_thinking(monkeypatch):
    monkeypatch.setenv("KS_KW_THINKING", "0")
    monkeypatch.setenv("KS_BUILD_THINKING", "1")
    monkeypatch.setenv("PAPERVAULT_LLM_KEYWORD", "mimo:think")
    assert route("keyword") == ("mimo", False)


def test_ks_kw_thinking_does_not_affect_build(monkeypatch):
    # KS_KW_THINKING is keyword-only; the build-extraction call ignores it.
    monkeypatch.setenv("KS_KW_THINKING", "0")
    assert route("build") == (_BUILD, True)


def test_keyword_follows_build_thinking_when_kw_not_zero(monkeypatch):
    # KS_KW_THINKING set but not "0" → falls through to the KS_BUILD_THINKING branch.
    monkeypatch.setenv("KS_KW_THINKING", "1")
    monkeypatch.setenv("KS_BUILD_THINKING", "0")
    assert route("keyword") == (_BUILD, None)


def test_paper_pipeline_verify_model_overrides_verify(monkeypatch):
    monkeypatch.setenv("PAPER_PIPELINE_VERIFY_MODEL", "legacy-verify")
    assert route("verify") == ("legacy-verify", None)


def test_paper_pipeline_verify_model_wins_over_new_var_model(monkeypatch):
    monkeypatch.setenv("PAPER_PIPELINE_VERIFY_MODEL", "legacy-verify")
    monkeypatch.setenv("PAPERVAULT_LLM_VERIFY", "new-verify:nothink")
    # Legacy wins MODEL; thinking still tracks the new var.
    assert route("verify") == ("legacy-verify", False)


# ------------------------------------------------------------------------- role validation

def test_unknown_role_raises():
    with pytest.raises(ValueError, match="unknown LLM role"):
        route("nonsense")


def test_role_is_case_insensitive():
    assert route("SYNTH") == (_SYNTH, None)
    assert route("Build") == (_BUILD, True)


def test_roles_tuple_is_complete():
    for r in llm_routing.ROLES:
        assert isinstance(route(r), tuple) and len(route(r)) == 2
