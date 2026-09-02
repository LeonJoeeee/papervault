"""Bounded internal retry around the S3 synth MiMo call (issue #70).

Retrieval already SUCCEEDED by the time synth runs, so a single TRANSIENT synth hiccup
(timeout / connection reset / 5xx / 429 / empty-200) must be retried a few times BEFORE the
honest SYNTH_FAILED fallback — not surfaced to the caller who then manually re-sends. A
DETERMINISTIC failure (400 validation / all-deployments-blocked 401/403) must FAIL FAST, and a
true full outage of transient errors must still exhaust the retries and then fall back.

Pure unit tests: mimo_complete is stubbed (no LightRAG / no DB / no real LLM), backoff is
zeroed, so they run instantly.
"""
import asyncio

import pytest

from papervault.knowledge.query import synth as synth_mod
from papervault.knowledge.query.synth import SYNTH_FAILED_PREFIX, _is_transient_synth_error, synth_answer
from papervault.knowledge.store.llm import StreamTruncated

# Minimal aquery_data subgraph — one paper chunk is enough for _build_prompt.
_DATA = {"chunks": [{"content": "SEP flux drops during Forbush decreases.", "file_path": "paper/X2023"}]}
_INTENT = "What does the retrieved material say about SEP observation noise models?"


class _FakeStatusError(Exception):
    """openai-style error carrying a status_code (what synth's _error_code classifier reads)."""

    def __init__(self, status_code: int):
        super().__init__(f"fake http {status_code}")
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch):
    """Zero the exponential backoff so retry tests don't actually sleep."""
    monkeypatch.setattr(synth_mod, "_SYNTH_BACKOFF_BASE", 0.0)
    monkeypatch.setattr(synth_mod, "_SYNTH_MAX_RETRIES", 2)  # → up to 3 attempts (pin the default)


def _install(monkeypatch, outcomes):
    """Stub mimo_complete to yield ``outcomes`` in order: a str is returned, an Exception is
    raised. Returns a mutable ``calls`` list recording how many attempts happened."""
    seq = iter(outcomes)
    calls: list[int] = []

    async def _fake(*args, **kwargs):
        calls.append(1)
        out = next(seq)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(synth_mod, "mimo_complete", _fake)
    return calls


# ---- transient: retried, then succeeds --------------------------------------

async def test_transient_5xx_retried_then_succeeds(monkeypatch):
    calls = _install(monkeypatch, [_FakeStatusError(503), "real synthesized answer [X2023]"])
    out = await synth_answer(_DATA, _INTENT)
    assert out == "real synthesized answer [X2023]"     # the successful retry's prose, not the fallback
    assert len(calls) == 2                                # failed once, succeeded on retry


async def test_transient_timeout_retried_then_succeeds(monkeypatch):
    calls = _install(monkeypatch, [asyncio.TimeoutError(), "recovered answer [X2023]"])
    out = await synth_answer(_DATA, _INTENT)
    assert out == "recovered answer [X2023]"
    assert len(calls) == 2


async def test_empty_200_retried_then_succeeds(monkeypatch):
    # S16: an empty/whitespace 200 is the "malformed response" transient class → retry it too.
    calls = _install(monkeypatch, ["   \n  ", "non-empty answer [X2023]"])
    out = await synth_answer(_DATA, _INTENT)
    assert out == "non-empty answer [X2023]"
    assert len(calls) == 2


# ---- deterministic: fail fast, NO retry -------------------------------------

async def test_deterministic_400_fails_fast(monkeypatch):
    calls = _install(monkeypatch, [_FakeStatusError(400), "should never be reached"])
    out = await synth_answer(_DATA, _INTENT)
    assert out.startswith(SYNTH_FAILED_PREFIX)           # honest fallback
    assert len(calls) == 1                                # NOT retried


async def test_all_deployments_blocked_403_fails_fast(monkeypatch):
    # KeyPool's terminal error when every deployment is 403-blocked — deterministic, fail fast.
    err = RuntimeError("All configured LLM keys failed after 2 rounds. Last error: PermissionDeniedError")
    calls = _install(monkeypatch, [err, "should never be reached"])
    out = await synth_answer(_DATA, _INTENT)
    assert out.startswith(SYNTH_FAILED_PREFIX)
    assert len(calls) == 1


# ---- true outage: transient errors exhaust retries, then fall back ----------

async def test_transient_exhausts_retries_then_falls_back(monkeypatch):
    calls = _install(monkeypatch, [_FakeStatusError(503)] * 3)  # every attempt fails transiently
    out = await synth_answer(_DATA, _INTENT)
    assert out.startswith(SYNTH_FAILED_PREFIX)           # terminal graceful degradation unchanged
    assert len(calls) == 3                                # MAX_RETRIES(2) + 1 = 3 attempts, then give up


async def test_empty_200_exhausts_retries_then_falls_back(monkeypatch):
    calls = _install(monkeypatch, ["", "  ", "\n"])
    out = await synth_answer(_DATA, _INTENT)
    assert out.startswith(SYNTH_FAILED_PREFIX)
    assert len(calls) == 3


# ---- transient: a stream cut before its finish_reason (issue #100) ---------------

def test_stream_truncated_is_classified_transient():
    # No HTTP status, not an APIConnectionError — it must be recognised by type, or a cut
    # synthesis stream would fall back immediately instead of using the bounded retry.
    assert _is_transient_synth_error(StreamTruncated("stream ended without a finish_reason")) is True


async def test_stream_truncated_retried_then_succeeds(monkeypatch):
    calls = _install(monkeypatch, [StreamTruncated("stream ended without a finish_reason"),
                                   "answer after the cut [X2023]"])
    out = await synth_answer(_DATA, _INTENT)
    assert out == "answer after the cut [X2023]"
    assert len(calls) == 2
