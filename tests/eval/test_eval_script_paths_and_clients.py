"""Unit tests for eval script output paths and judge client endpoint resolution."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from papervault import config
from papervault.eval import _ab_u1_judge, judge_mimo

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(script_name: str):
    path = REPO_ROOT / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"test_loaded_{script_name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("script_name", ["mustfind_recall", "search_relevance_baseline"])
def test_eval_scripts_resolve_output_via_eval_path(monkeypatch, tmp_path, script_name):
    eval_root = tmp_path / "external-eval-dir"
    monkeypatch.setenv("PAPERVAULT_EVAL_DIR", str(eval_root))
    mod = _load_script(script_name)

    out_path = mod._output_path("smoke_tag")
    assert out_path == eval_root.resolve() / "smoke_tag.jsonl"
    assert "src" not in out_path.parts

    # Reject any PAPERVAULT_EVAL_DIR located inside src/
    monkeypatch.setenv("PAPERVAULT_EVAL_DIR", str(REPO_ROOT / "src" / "papervault" / "eval" / "results"))
    with pytest.raises(ValueError, match="outside src/"):
        mod._output_path("smoke_tag")


async def test_mustfind_recall_main_creates_parent_and_writes_under_eval_dir(monkeypatch, tmp_path):
    eval_root = tmp_path / "nested" / "eval-out"
    monkeypatch.setenv("PAPERVAULT_EVAL_DIR", str(eval_root))
    mod = _load_script("mustfind_recall")

    pools_file = tmp_path / "pools.jsonl"
    intents_file = tmp_path / "intents.jsonl"
    pools_file.write_text(
        json.dumps({"qid": "q1", "pool": [{"title": "Landmark Reconnection Study", "doi": "10.1/a"}]}) + "\n"
    )
    intents_file.write_text(json.dumps({"qid": "q1", "intent": "magnetic reconnection"}) + "\n")

    async def fake_search_once(intent: str, *, timeout_s: float = 900.0):
        return [{"title": "Landmark Reconnection Study", "doi": "10.1/a"}]

    monkeypatch.setattr(mod, "_search_once", fake_search_once)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mustfind_recall.py", "--tag", "mfr_unit", "--pools", str(pools_file), "--intents", str(intents_file)],
    )

    await mod.main()

    out_file = eval_root / "mfr_unit.jsonl"
    assert out_file.is_file()
    row = json.loads(out_file.read_text().strip())
    assert row["qid"] == "q1"
    assert row["recall"] == 1.0


def test_search_relevance_baseline_respects_gateway_and_direct_config(monkeypatch):
    mod = _load_script("search_relevance_baseline")

    captured: list[dict[str, object]] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(mod, "AsyncOpenAI", FakeAsyncOpenAI)

    # Gateway enabled -> GATEWAY_URL / GATEWAY_KEY
    monkeypatch.setattr(config, "USE_GATEWAY", True)
    monkeypatch.setattr(config, "GATEWAY_URL", "http://127.0.0.1:4000/v1")
    monkeypatch.setattr(config, "GATEWAY_KEY", "gw-secret")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://direct.example/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "direct-secret")

    mod._build_judge_client()
    assert captured[-1] == {
        "base_url": "http://127.0.0.1:4000/v1",
        "api_key": "gw-secret",
        "max_retries": 0,
    }

    # Gateway disabled -> LLM_BASE_URL / LLM_API_KEY
    monkeypatch.setattr(config, "USE_GATEWAY", False)
    mod._build_judge_client()
    assert captured[-1] == {
        "base_url": "https://direct.example/v1",
        "api_key": "direct-secret",
        "max_retries": 0,
    }

    # Gateway enabled with empty GATEWAY_URL -> fails closed (never leaks gateway key to api.openai.com)
    monkeypatch.setattr(config, "USE_GATEWAY", True)
    monkeypatch.setattr(config, "GATEWAY_URL", "")
    with pytest.raises(RuntimeError, match="PAPERVAULT_LLM_GATEWAY_URL is empty"):
        mod._build_judge_client()


@pytest.mark.parametrize("mod", [judge_mimo, _ab_u1_judge])
def test_judge_modules_fall_back_to_config_and_honor_legacy_overrides(monkeypatch, mod):
    monkeypatch.delenv("KS_GATEWAY_URL", raising=False)
    monkeypatch.delenv("KS_VIRTUAL_KEY", raising=False)

    # 1. Gateway mode fallback via papervault.config
    monkeypatch.setattr(config, "USE_GATEWAY", True)
    monkeypatch.setattr(config, "GATEWAY_URL", "http://127.0.0.1:4010/v1")
    monkeypatch.setattr(config, "GATEWAY_KEY", "cfg-gw-key")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://direct.example/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "cfg-direct-key")
    assert mod._resolve_endpoint() == ("http://127.0.0.1:4010/v1", "cfg-gw-key")

    # 2. Direct mode fallback via papervault.config
    monkeypatch.setattr(config, "USE_GATEWAY", False)
    assert mod._resolve_endpoint() == ("https://direct.example/v1", "cfg-direct-key")

    # 3. Explicit legacy env overrides take precedence over config
    monkeypatch.setenv("KS_GATEWAY_URL", "http://127.0.0.1:4999/v1")
    monkeypatch.setenv("KS_VIRTUAL_KEY", "legacy-vk")
    assert mod._resolve_endpoint() == ("http://127.0.0.1:4999/v1", "legacy-vk")

    # 4. Partial legacy override (KS_VIRTUAL_KEY alone under direct mode) NEVER sends key to direct URL
    monkeypatch.delenv("KS_GATEWAY_URL", raising=False)
    assert mod._resolve_endpoint() == ("http://127.0.0.1:4000/v1", "legacy-vk")

    # 5. Partial legacy override (KS_GATEWAY_URL alone under direct mode) NEVER leaks direct provider key
    monkeypatch.delenv("KS_VIRTUAL_KEY", raising=False)
    monkeypatch.setenv("KS_GATEWAY_URL", "http://127.0.0.1:4999/v1")
    assert mod._resolve_endpoint() == ("http://127.0.0.1:4999/v1", "")

    # 6. Partial legacy override (KS_GATEWAY_URL alone under gateway mode) can reuse gateway key
    monkeypatch.setattr(config, "USE_GATEWAY", True)
    assert mod._resolve_endpoint() == ("http://127.0.0.1:4999/v1", "cfg-gw-key")


async def test_judge_mimo_run_raises_clear_error_when_no_key_or_empty_gateway_url(monkeypatch):
    monkeypatch.delenv("KS_GATEWAY_URL", raising=False)
    monkeypatch.delenv("KS_VIRTUAL_KEY", raising=False)
    monkeypatch.setattr(config, "USE_GATEWAY", False)
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "GATEWAY_KEY", "")

    with pytest.raises(SystemExit, match="No judge API key configured"):
        await judge_mimo._run(["vmq_r1"], concurrency=1, force=False)

    monkeypatch.setattr(config, "USE_GATEWAY", True)
    monkeypatch.setattr(config, "GATEWAY_KEY", "gw-secret")
    monkeypatch.setattr(config, "GATEWAY_URL", "")
    with pytest.raises(SystemExit, match="PAPERVAULT_LLM_GATEWAY_URL is empty"):
        await judge_mimo._run(["vmq_r1"], concurrency=1, force=False)


async def test_ab_u1_judge_main_rejects_empty_gateway_url(monkeypatch, tmp_path):
    monkeypatch.delenv("KS_GATEWAY_URL", raising=False)
    monkeypatch.delenv("KS_VIRTUAL_KEY", raising=False)
    monkeypatch.setattr(config, "USE_GATEWAY", True)
    monkeypatch.setattr(config, "GATEWAY_KEY", "gw-secret")
    monkeypatch.setattr(config, "GATEWAY_URL", "")
    edges_file = tmp_path / "u1.jsonl"
    edges_file.write_text("{}\n")
    monkeypatch.setattr(sys, "argv", ["_ab_u1_judge.py", str(edges_file)])

    with pytest.raises(SystemExit, match="PAPERVAULT_LLM_GATEWAY_URL is empty"):
        await _ab_u1_judge.main()


async def test_ab_u1_judge_main_uses_config_fallback_without_legacy_key(monkeypatch, tmp_path):
    monkeypatch.delenv("KS_GATEWAY_URL", raising=False)
    monkeypatch.delenv("KS_VIRTUAL_KEY", raising=False)
    monkeypatch.setattr(config, "USE_GATEWAY", False)
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://direct.example/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "direct-key")

    edges_file = tmp_path / "u1.jsonl"
    edges_file.write_text(
        json.dumps({"head": "A", "tail": "B", "description": "A drives B", "source_chunks": ["A drives B"]}) + "\n"
    )
    monkeypatch.setattr(sys, "argv", ["_ab_u1_judge.py", str(edges_file)])

    captured: list[dict[str, object]] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    async def fake_judge(client, sem, edge):
        return True

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(_ab_u1_judge, "_judge", fake_judge)

    await _ab_u1_judge.main()
    assert captured == [{"api_key": "direct-key", "timeout": 300, "max_retries": 0, "base_url": "https://direct.example/v1"}]
