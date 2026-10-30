"""Serving receipts describe observed calls without storing queries or response bodies."""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from papervault.library import llm as transport
from papervault.library.services import intent_parser
from papervault.library.services.registration import ServingTrace


@pytest.mark.parametrize("disclosed_model,expected", [("actual-v4", "actual-v4"), (None, "unknown")])
def test_litellm_routing_normalization_cannot_become_served_model(monkeypatch, disclosed_model, expected):
    with patch.dict(os.environ, {"LITELLM_MODE": "PRODUCTION"}):
        import litellm
    from openai.types.chat import ChatCompletionChunk

    chunk = ChatCompletionChunk(
        id="receipt", model=disclosed_model or "", created=0, object="chat.completion.chunk",
        choices=[{"index": 0, "delta": {"content": "{}"}, "finish_reason": "stop"}],
    )
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}}
    logging_obj.messages = [{"role": "user", "content": "ping"}]
    logging_obj.optional_params = {}
    stream = litellm.CustomStreamWrapper(
        iter([chunk]), "configured-standard", logging_obj, custom_llm_provider="openai",
    )
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: stream)
    observations = []
    with transport.observe_llm_calls(observations.append):
        assert transport.LLM("openai/standard").call("ping") == "{}"
    assert observations[0].model == expected
    assert observations[0].provider == "unknown"


def response_stream(monkeypatch, *, model=None, provider=None):
    chunk = SimpleNamespace(
        model=model, provider=provider, usage=None,
        choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content=json.dumps({
            "search_terms": ["cosmic ray transport"], "filters": {},
        })), finish_reason="stop")],
    )
    stream = iter([chunk])
    monkeypatch.setattr(transport, "_open_stream", lambda kwargs: (stream, stream, next(stream)))


@pytest.mark.parametrize("actual_model,actual_provider,expected_model,expected_provider", [
    ("served-model-v3", "served-provider", "served-model-v3", "served-provider"),
    (None, None, "unknown", "unknown"),
    ("served-model-v3", None, "served-model-v3", "unknown"),
])
def test_intent_receipt_uses_response_disclosure_not_requested_routing(
    monkeypatch, actual_model, actual_provider, expected_model, expected_provider,
):
    response_stream(monkeypatch, model=actual_model, provider=actual_provider)
    events = []
    trace = ServingTrace(SimpleNamespace(log=events.append), trace_id="trace-251")
    with intent_parser.observe_intent(trace.record_intent):
        plan = intent_parser.parse_intent("cosmic ray transport", transport.LLM("openai/standard"))

    assert plan["search_terms"] == ["cosmic ray transport"]
    assert len(events) == 1
    receipt = events[0]
    assert receipt["stage"] == "intent"
    assert receipt["trace_id"] == "trace-251"
    assert receipt["model"] == expected_model
    assert receipt["provider"] == expected_provider
    assert receipt["latency_ms"] >= 0
    assert receipt["degraded"] is False
    assert "query" not in receipt and "response" not in receipt


def test_backend_receipt_retains_error_taxonomy_without_query():
    events = []
    trace = ServingTrace(SimpleNamespace(log=events.append), trace_id="trace-251")
    trace.record_backend(SimpleNamespace(
        backend="ads", term="private term", term_idx=2, status="degraded", hit_count=0,
        latency_ms=12.5, code="source_rate_limited", scope="service",
        retryability="transient-with-delay", retry_after_s=15.0,
    ))

    receipt = events[0]
    assert receipt["stage"] == "retrieval" and receipt["arm"] == "external"
    assert {key: receipt[key] for key in (
        "backend", "term_idx", "status", "hit_count", "latency_ms", "code", "scope",
        "retryability", "retry_after_s",
    )} == {
        "backend": "ads", "term_idx": 2, "status": "degraded", "hit_count": 0,
        "latency_ms": 12.5, "code": "source_rate_limited", "scope": "service",
        "retryability": "transient-with-delay", "retry_after_s": 15.0,
    }
    assert "private" not in str(receipt)


@pytest.mark.parametrize("failover", [False, True])
def test_intent_receipt_marks_recovered_call_failure_degraded(monkeypatch, tmp_path, failover):
    response = SimpleNamespace(
        model="actual-recovery-model", provider="actual-provider", usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "search_terms": ["cosmic ray transport"], "filters": {},
        })))],
    )
    attempts = iter([transport.StreamTruncated("private failure body"), response])

    def completion(**kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(transport, "_create_completion", completion)
    monkeypatch.setattr(transport.time, "sleep", lambda delay: None)
    client = transport.LLM("openai/standard")
    if failover:
        path = tmp_path / "test-keys.json"
        path.write_text(json.dumps([
            {"model": "configured-first", "base_url": "https://first.test", "api_key": "test-1"},
            {"model": "configured-second", "base_url": "https://second.test", "api_key": "test-2"},
        ]))
        client = transport.KeyPool(path, max_tokens=256)
        monkeypatch.setattr(transport.random, "shuffle", lambda items: None)
        monkeypatch.setattr(transport, "_RETRY_BUDGET_S", 0)

    observations = []
    with intent_parser.observe_intent(observations.append):
        intent_parser.parse_intent("cosmic ray transport", client)
    assert len(observations) == 1
    assert observations[0].model == "actual-recovery-model"
    assert observations[0].provider == "actual-provider"
    assert observations[0].degraded is True
    assert observations[0].code is None


@pytest.mark.asyncio
async def test_intent_observers_stay_request_local_across_to_thread(monkeypatch):
    from threading import Barrier

    barrier = Barrier(2)

    def completion(**kwargs):
        barrier.wait(timeout=5)
        model = kwargs["model"]
        return SimpleNamespace(
            model=model + "-served", provider="disclosed-provider", usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "search_terms": [model], "filters": {},
            })))],
        )

    monkeypatch.setattr(transport, "_create_completion", completion)

    async def request(model):
        observations = []
        with intent_parser.observe_intent(observations.append):
            await asyncio.to_thread(intent_parser.parse_intent, model, transport.LLM(model))
        return observations

    first, second = await asyncio.gather(request("first"), request("second"))
    assert [event.model for event in first] == ["first-served"]
    assert [event.model for event in second] == ["second-served"]


def test_intent_receipt_marks_multi_target_route_fallback_degraded(monkeypatch):
    from papervault.llm_routing import parse_targets

    transport.clear_fallback_cooldowns()
    response = SimpleNamespace(
        model="gpt-6.1-terra-served", provider="openai-gateway", usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "search_terms": ["cosmic ray transport"], "filters": {},
        })))],
    )
    attempts = iter([transport.StreamTruncated("primary cut"), response])

    def completion(**kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(transport, "_create_completion", completion)
    client = transport.LLM(
        "openai/glm-5.3-flash",
        targets=parse_targets("glm-5.3-flash,gpt-6.1-terra@high"),
    )
    observations = []
    try:
        with intent_parser.observe_intent(observations.append):
            intent_parser.parse_intent("cosmic ray transport", client)
        assert len(observations) == 1
        assert observations[0].model == "gpt-6.1-terra-served"
        assert observations[0].provider == "openai-gateway"
        assert observations[0].degraded is True
        assert observations[0].code is None
    finally:
        transport.clear_fallback_cooldowns()
