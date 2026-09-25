"""#143: process-wide build-path breaker — pause the background knowledge build on a sustained
upstream failure, resume by itself, never touch the query path.

Deterministic: an injected fake clock, a fake `mimo_complete`, and the fake ledger / LightRAG
from test_round. No network, no DB, no real LightRAG.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
import openai
import pytest

import papervault.knowledge.ingest.distill as distill
import papervault.knowledge.scheduler.round as rnd
import papervault.knowledge.store.build_breaker as bb
import papervault.knowledge.store.graph as graph
import papervault.knowledge.store.llm as llm_mod
from papervault.knowledge.ledger.store import LedgerRecord
from papervault.knowledge.store.build_breaker import BuildBreaker, BuildPausedError, is_upstream_failure
from papervault.knowledge.store.llm import StreamTruncated
from tests.knowledge.test_round import FakeLedger, FakeRag, _rec

_REQ = httpx.Request("POST", "http://127.0.0.1:4000/v1/chat/completions")
_LOGGER = "ks.store.build_breaker"


def _status_error(code: int) -> openai.APIStatusError:
    return openai.APIStatusError(f"upstream {code}", response=httpx.Response(code, request=_REQ), body=None)


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _breaker(clock: FakeClock, *, threshold: int = 3, window_s: float = 60.0,
             cooldown_s: float = 600.0) -> BuildBreaker:
    return BuildBreaker(threshold=threshold, window_s=window_s, cooldown_s=cooldown_s, clock=clock)


class FakeComplete:
    """Stand-in for mimo_complete: records every call; raises `exc` if set, else returns 'ok'."""

    def __init__(self, exc: BaseException | None = None):
        self.exc = exc
        self.calls: list[dict] = []

    async def __call__(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return "ok"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def breaker(monkeypatch, clock) -> BuildBreaker:
    """Install a fresh, fake-clocked breaker as THE process-wide one for this test."""
    b = _breaker(clock)
    monkeypatch.setattr(bb, "_BREAKER", b)
    return b


# ---------------------------------------------------------------- breaker state machine


def test_opens_after_n_failures_within_window(clock):
    b = _breaker(clock, threshold=3, window_s=60)
    b.record_failure()
    clock.t += 10
    b.record_failure()
    assert b.allows() and b.state == "closed"
    clock.t += 10
    b.record_failure()
    assert not b.allows() and b.state == "open"


def test_failures_spread_beyond_the_window_do_not_open(clock):
    b = _breaker(clock, threshold=3, window_s=60)
    for _ in range(10):
        b.record_failure()
        clock.t += 31   # at most 2 failures ever fall inside one 60 s window
    assert b.allows() and b.state == "closed"


def test_successes_while_closed_do_not_hide_a_failure_burst(clock):
    # The count is failures IN THE WINDOW, not consecutive: intermittent 400s between
    # successes (the 2026-09-25 MaaS pattern) still trip it.
    b = _breaker(clock, threshold=3, window_s=60)
    for _ in range(3):
        b.record_failure()
        b.record_success()
    assert b.state == "open"


def test_stays_open_until_cooldown_then_half_open(clock):
    b = _breaker(clock, threshold=1, cooldown_s=600)
    b.record_failure()
    clock.t += 599
    assert not b.allows() and b.state == "open"
    clock.t += 1
    assert b.allows() and b.state == "half_open"


def test_half_open_probe_success_closes(clock):
    b = _breaker(clock, threshold=2, cooldown_s=600)
    b.record_failure()
    b.record_failure()
    clock.t += 600
    assert b.allows()
    b.record_success()
    assert b.state == "closed" and b.allows()
    # the window was cleared on close: one fresh failure does not re-open
    b.record_failure()
    assert b.state == "closed"


def test_half_open_probe_failure_reopens_for_a_fresh_cooldown(clock):
    b = _breaker(clock, threshold=2, cooldown_s=600)
    b.record_failure()
    b.record_failure()
    clock.t += 600
    assert b.allows()
    b.record_failure()                 # ONE failure in half-open re-opens
    assert b.state == "open" and not b.allows()
    clock.t += 599
    assert not b.allows()
    clock.t += 1
    assert b.allows() and b.state == "half_open"


def test_results_arriving_while_open_are_ignored(clock):
    # In-flight calls started before the trip finish after it: neither closes nor extends it.
    b = _breaker(clock, threshold=1, cooldown_s=600)
    b.record_failure()
    clock.t += 300
    b.record_success()
    b.record_failure()
    assert b.state == "open"
    clock.t += 300
    assert b.allows() and b.state == "half_open"


def test_one_log_line_per_transition_not_per_call(clock, caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    b = _breaker(clock, threshold=30, window_s=300, cooldown_s=600)

    def lines():
        return [r for r in caplog.records if r.name == _LOGGER]

    for _ in range(29):
        b.record_failure()
    assert lines() == []                               # below threshold: silent
    b.record_failure()
    assert len(lines()) == 1 and "OPEN" in lines()[0].getMessage()
    for _ in range(500):                               # fast-fails + late results while open
        assert not b.allows()
        b.record_failure()
    assert len(lines()) == 1
    clock.t += 600
    for _ in range(5):
        assert b.allows()
    assert len(lines()) == 2 and "HALF-OPEN" in lines()[1].getMessage()
    b.record_failure()
    assert len(lines()) == 3 and "RE-OPENED" in lines()[2].getMessage()
    clock.t += 600
    assert b.allows()
    b.record_success()
    b.record_success()
    assert [r.getMessage().split()[2] for r in lines()] == ["OPEN", "HALF-OPEN", "RE-OPENED",
                                                            "HALF-OPEN", "CLOSED"]
    assert [r.levelno for r in lines()] == [logging.WARNING, logging.INFO, logging.WARNING,
                                           logging.INFO, logging.INFO]


def test_threshold_non_positive_disables_the_breaker(clock):
    b = _breaker(clock, threshold=0)
    for _ in range(1000):
        b.record_failure()
    assert b.allows() and b.state == "closed"


def test_from_env_defaults_and_overrides():
    d = BuildBreaker.from_env({})
    assert (d.threshold, d.window_s, d.cooldown_s) == (30, 300.0, 600.0)
    o = BuildBreaker.from_env({"KS_BUILD_BREAKER_FAILURES": "5", "KS_BUILD_BREAKER_WINDOW_S": "60",
                               "KS_BUILD_BREAKER_COOLDOWN_S": "120"})
    assert (o.threshold, o.window_s, o.cooldown_s) == (5, 60.0, 120.0)
    bad = BuildBreaker.from_env({"KS_BUILD_BREAKER_FAILURES": "lots"})
    assert bad.threshold == 30


# ---------------------------------------------------------------- what counts as an upstream failure


@pytest.mark.parametrize("exc", [
    _status_error(400), _status_error(429), _status_error(503),
    openai.APIConnectionError(request=_REQ), openai.APITimeoutError(request=_REQ),
    StreamTruncated("cut"),
])
def test_upstream_failures_are_counted(exc):
    assert is_upstream_failure(exc)


def test_direct_pool_exhaustion_counts_through_its_cause():
    try:
        try:
            raise _status_error(502)
        except openai.APIStatusError as e:
            raise RuntimeError("All configured LLM keys failed") from e
    except RuntimeError as wrapped:
        assert is_upstream_failure(wrapped)


@pytest.mark.parametrize("exc", [
    ValueError("caller bug"), RuntimeError("No active LLM keys"),
    BuildPausedError("paused"), asyncio.CancelledError(),
])
def test_non_upstream_errors_are_not_counted(exc):
    assert not is_upstream_failure(exc)


def test_direct_pool_raises_exhaustion_chained_to_the_upstream_error(tmp_path, monkeypatch):
    # The direct (non-gateway) KeyPool wraps the last upstream error in a RuntimeError;
    # the breaker can only classify it if the cause is chained.
    import json

    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps([{"model": "m", "api_key": "sk-test-0001", "base_url": "http://x/v1"}]))
    monkeypatch.setattr(llm_mod.config, "USE_GATEWAY", False)
    monkeypatch.setattr(llm_mod, "_MAX_ROUNDS", 1)

    async def boom(*a, **k):
        raise _status_error(503)

    monkeypatch.setattr(llm_mod, "_create_completion", boom)
    pool = llm_mod.KeyPool(keys)
    with pytest.raises(RuntimeError) as ei:
        asyncio.run(pool.complete("p"))
    assert is_upstream_failure(ei.value)


# ---------------------------------------------------------------- build_llm (LightRAG's llm_model_func)


async def test_build_calls_fail_fast_after_n_upstream_failures(breaker, monkeypatch):
    fake = FakeComplete(exc=_status_error(503))
    monkeypatch.setattr(graph, "mimo_complete", fake)
    for _ in range(breaker.threshold):
        with pytest.raises(openai.APIStatusError):
            await graph.build_llm("extract entities")
    assert len(fake.calls) == breaker.threshold

    with pytest.raises(BuildPausedError):
        await graph.build_llm("extract entities")
    assert len(fake.calls) == breaker.threshold        # the gateway was NOT contacted


async def test_keyword_calls_pass_and_do_not_count_while_open(breaker, monkeypatch):
    for _ in range(breaker.threshold):
        breaker.record_failure()
    assert breaker.state == "open"

    fake = FakeComplete()
    monkeypatch.setattr(graph, "mimo_complete", fake)
    kw = {"response_format": {"type": "json_object"}}
    assert await graph.build_llm("keywords of the query", **kw) == "ok"
    assert len(fake.calls) == 1

    # keyword-route failures never feed the breaker (it is build-role only)
    breaker_closed = _breaker(FakeClock(), threshold=1)
    monkeypatch.setattr(bb, "_BREAKER", breaker_closed)
    fake.exc = _status_error(503)
    with pytest.raises(openai.APIStatusError):
        await graph.build_llm("keywords of the query", **kw)
    assert breaker_closed.state == "closed"


async def test_direct_mimo_complete_is_not_gated_while_open(breaker, monkeypatch):
    # decompose / synth call mimo_complete directly — the breaker must not sit on that path.
    for _ in range(breaker.threshold):
        breaker.record_failure()
    assert breaker.state == "open"

    class Pool:
        async def complete(self, prompt, **kw):
            return "synth answer"

    monkeypatch.setattr(llm_mod, "get_pool", lambda: Pool())
    assert await llm_mod.mimo_complete("synthesize") == "synth answer"


async def test_cancellations_and_caller_errors_are_not_counted(breaker, monkeypatch):
    fake = FakeComplete(exc=asyncio.CancelledError())
    monkeypatch.setattr(graph, "mimo_complete", fake)
    for _ in range(breaker.threshold * 3):
        with pytest.raises(asyncio.CancelledError):
            await graph.build_llm("extract entities")
    fake.exc = ValueError("bad kwarg")
    for _ in range(breaker.threshold * 3):
        with pytest.raises(ValueError):
            await graph.build_llm("extract entities")
    assert breaker.state == "closed"


async def test_build_probe_success_closes_and_failure_reopens(breaker, clock, monkeypatch):
    fake = FakeComplete(exc=_status_error(503))
    monkeypatch.setattr(graph, "mimo_complete", fake)
    for _ in range(breaker.threshold):
        with pytest.raises(openai.APIStatusError):
            await graph.build_llm("x")
    clock.t += breaker.cooldown_s

    with pytest.raises(openai.APIStatusError):         # the probe call reaches the gateway…
        await graph.build_llm("x")
    assert breaker.state == "open"                     # …fails → re-open
    with pytest.raises(BuildPausedError):
        await graph.build_llm("x")

    clock.t += breaker.cooldown_s
    fake.exc = None
    assert await graph.build_llm("x") == "ok"          # probe succeeds → closed
    assert breaker.state == "closed"


# ---------------------------------------------------------------- scheduler round gate


@pytest.fixture
def fake_ledger(monkeypatch):
    fl = FakeLedger()
    monkeypatch.setattr(distill, "ledger", fl)
    monkeypatch.setattr(rnd, "ledger", fl)
    return fl


def _vault_with_new_paper(monkeypatch, fake_ledger):
    from lightrag.base import DocStatus

    monkeypatch.setattr(rnd, "load_clean_index", lambda *a, **k: {"NEW": _rec("NEW")})
    monkeypatch.setattr(rnd, "fingerprint", lambda rec: "fNEW")
    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: "body of NEW")
    fake_ledger.rows = {
        "DONE": LedgerRecord("l0_probe", "paper", "DONE", "f", "paper:DONE", "processing"),
        "GONE": LedgerRecord("l0_probe", "paper", "GONE", "f", "paper:GONE", "done"),
    }
    return FakeRag(doc_statuses={"paper:DONE": DocStatus.PROCESSED})


async def test_run_round_starts_no_build_work_while_open(breaker, fake_ledger, monkeypatch):
    rag = _vault_with_new_paper(monkeypatch, fake_ledger)
    for _ in range(breaker.threshold):
        breaker.record_failure()

    summary = await rnd.run_round(rag)

    kinds = [k for k, _ in rag.calls]
    assert "enqueue" not in kinds and "delete" not in kinds and "process" not in kinds
    assert summary["paused"] is True
    # ledger reconciliation (DB reads, no LLM) still runs: the finished doc is written back
    assert fake_ledger.rows["DONE"].status == "done"
    assert "GONE" in fake_ledger.rows                  # removal (LLM re-summary) deferred too


async def test_run_round_after_cooldown_is_the_probe_round(breaker, clock, fake_ledger, monkeypatch):
    rag = _vault_with_new_paper(monkeypatch, fake_ledger)
    for _ in range(breaker.threshold):
        breaker.record_failure()
    clock.t += breaker.cooldown_s

    summary = await rnd.run_round(rag)

    kinds = [k for k, _ in rag.calls]
    assert "enqueue" in kinds and "process" in kinds and "delete" in kinds
    assert summary.get("paused") is not True
    assert breaker.state == "half_open"


async def test_run_round_unchanged_while_closed(breaker, fake_ledger, monkeypatch):
    rag = _vault_with_new_paper(monkeypatch, fake_ledger)
    summary = await rnd.run_round(rag)
    assert "paused" not in summary
    assert ("process", None) in rag.calls


async def test_main_loop_defers_operator_doc_pickup_while_open(breaker, fake_ledger, monkeypatch):
    for _ in range(breaker.threshold):
        breaker.record_failure()
    seen = {"rounds": 0, "drains": 0}

    async def fake_round(rag, **kw):
        seen["rounds"] += 1
        return {}

    async def fake_drain(rag):
        seen["drains"] += 1

    async def stop(_s):
        raise asyncio.CancelledError

    monkeypatch.setattr(rnd, "run_round", fake_round)
    monkeypatch.setattr(rnd, "drain_pending", fake_drain)
    monkeypatch.setattr(rnd.asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        await rnd.main_loop(object())
    assert seen == {"rounds": 1, "drains": 0}

    # closed → the pickup runs as before
    monkeypatch.setattr(bb, "_BREAKER", _breaker(FakeClock()))
    with pytest.raises(asyncio.CancelledError):
        await rnd.main_loop(object())
    assert seen == {"rounds": 2, "drains": 1}
