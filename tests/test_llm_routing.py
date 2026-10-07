"""Role defaults, operator overrides, and library request routing (issue #152)."""
from types import SimpleNamespace

import pytest

from papervault import config
from papervault.llm_routing import route

ROLE_LEVELS = [
    ("build", "standard"), ("keyword", "flash"), ("synth", "pro"),
    ("gate", "flash"), ("verify", "flash"), ("judge", "standard"),
    ("decompose", "standard"), ("eval_judge", "standard"),
]


@pytest.fixture(autouse=True)
def clean_routes(monkeypatch):
    for role, _ in ROLE_LEVELS:
        monkeypatch.delenv(f"PAPERVAULT_LLM_{role.upper()}", raising=False)
    for var in ("KS_BUILD_MODEL", "PAPER_PIPELINE_VERIFY_MODEL", "MIMO_MODEL"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("role,want", ROLE_LEVELS)
def test_agreed_role_level(role, want):
    assert route(role) == want


@pytest.mark.parametrize("role,_", ROLE_LEVELS)
@pytest.mark.parametrize("level", ["flash", "standard", "pro"])
def test_each_role_can_be_overridden(monkeypatch, role, _, level):
    monkeypatch.setenv(f"PAPERVAULT_LLM_{role.upper()}", level)
    assert route(role) == level


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("role,want", ROLE_LEVELS)
def test_blank_override_uses_role_default(monkeypatch, role, want, blank):
    monkeypatch.setenv(f"PAPERVAULT_LLM_{role.upper()}", blank)
    assert route(role) == want


def test_whitespace_is_normalized(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_LLM_SYNTH", " pro ")
    assert route(" SYNTH ") == "pro"


@pytest.mark.parametrize("bad", ["cheap", "openai/pro", "provider-model", "pro:suffix", "PRO"])
def test_invalid_override_names_the_variable(monkeypatch, bad):
    monkeypatch.setenv("PAPERVAULT_LLM_SYNTH", bad)
    with pytest.raises(ValueError, match="PAPERVAULT_LLM_SYNTH.*flash.*standard.*pro"):
        route("synth")


def test_unknown_role_rejected():
    with pytest.raises(ValueError, match="unknown LLM role"):
        route("unknown")


def test_retired_model_overrides_cannot_replace_levels(monkeypatch):
    for var in ("PAPERVAULT_MODEL", "PAPERVAULT_BUILD_MODEL", "KS_BUILD_MODEL",
                "PAPER_PIPELINE_VERIFY_MODEL", "MIMO_MODEL"):
        monkeypatch.setenv(var, "provider-model")
    assert route("build") == "standard"
    assert route("keyword") == "flash"
    assert route("verify") == "flash"
    assert route("eval_judge") == "standard"


@pytest.mark.parametrize("gateway", [False, True])
@pytest.mark.parametrize("level", ["flash", "standard", "pro"])
def test_library_request_preserves_each_level(tmp_path, monkeypatch, gateway, level):
    from papervault.library import llm

    monkeypatch.setattr(config, "USE_GATEWAY", gateway)
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://direct.example/v1")
    monkeypatch.setattr(config, "GATEWAY_KEY", "test-key")
    monkeypatch.setattr(config, "GATEWAY_URL", "https://gateway.example/v1")
    monkeypatch.setattr(llm, "_KEYS_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(llm, "_pools", {})
    monkeypatch.setattr(llm, "_gw_llms", {})
    monkeypatch.setenv("PAPERVAULT_LLM_JUDGE", level)
    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr(llm, "_create_completion", complete)
    assert llm.get_llm().call("judge") == "ok"
    assert requests == [{
        "model": f"openai/{level}", "messages": [{"role": "user", "content": "judge"}],
        "base_url": "https://gateway.example/v1" if gateway else "https://direct.example/v1",
        "api_key": "test-key", "max_tokens": llm._DEFAULT_MAX_TOKENS,
    }]


def test_doctor_accepts_level_defaults_without_model_slots(tmp_path, monkeypatch, capsys):
    from papervault import cli

    monkeypatch.setattr(config, "LLM_KEYS_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    report = cli._Report()
    cli._check_config(report)
    assert report.failures == 0
    assert "synth=pro" in capsys.readouterr().out


def test_doctor_reports_invalid_role_override(tmp_path, monkeypatch, capsys):
    from papervault import cli

    monkeypatch.setattr(config, "LLM_KEYS_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setenv("PAPERVAULT_LLM_SYNTH", "provider-model")
    report = cli._Report()
    cli._check_config(report)
    assert report.failures == 1
    assert "PAPERVAULT_LLM_SYNTH" in capsys.readouterr().out
