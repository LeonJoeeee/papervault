"""Library transport regressions: deadlines, cut streams and bounded retries."""
import logging
import os
import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai.types.chat import ChatCompletionChunk

from papervault.library import llm as L

# LiteLLM defaults to development mode and loads dotenv at import. The main
# checkout's venv can otherwise inject live paths after conftest isolated them.
with patch.dict(os.environ, {"LITELLM_MODE": "PRODUCTION"}):
    import litellm
    from litellm import CustomStreamWrapper


def chunk(content=None, finish=None, *, index=0, reasoning=None, usage=None):
    return ChatCompletionChunk(
        id="test", model="test", created=0, object="chat.completion.chunk",
        choices=[{"index": index, "delta": {"content": content,
                  "reasoning_content": reasoning}, "finish_reason": finish}],
        usage=usage,
    )


class Stream:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.chunks)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed.set()


def wrapped(chunks):
    source = Stream(chunks)
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}}
    logging_obj.messages = [{"role": "user", "content": "ping"}]
    logging_obj.optional_params = {}
    stream = CustomStreamWrapper(
        source, "test", logging_obj, custom_llm_provider="openai",
        stream_options={"include_usage": True},
    )
    return stream, source


def responses(monkeypatch, *items):
    calls = []
    pending = iter(items)

    def completion(**kwargs):
        calls.append(kwargs)
        item = next(pending)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(litellm, "completion", completion)
    return calls


@pytest.fixture(autouse=True)
def timing(monkeypatch):
    monkeypatch.setattr(L, "_FIRST_CHUNK_S", 0.1, raising=False)
    monkeypatch.setattr(L, "_RETRY_BUDGET_S", 5.0, raising=False)
    monkeypatch.setattr(L.random, "random", lambda: 0.0)
    monkeypatch.setattr(litellm, "disable_streaming_logging", True)


def test_real_wrapper_joins_content_and_retains_usage(monkeypatch, caplog):
    stream, source = wrapped([
        chunk(reasoning="private reasoning"), chunk("an"), chunk("swer"),
        chunk(finish="stop"),
        ChatCompletionChunk(id="test", model="test", created=0,
                            object="chat.completion.chunk", choices=[],
                            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}),
    ])
    calls = responses(monkeypatch, stream)
    with caplog.at_level(logging.INFO, logger=L.__name__):
        assert L.LLM("openai/test", base_url="https://example.test/v1", api_key="test",
                     max_tokens=123, num_retries=8).call("ping") == "answer"
    assert source.closed.is_set()
    assert len(calls) == 1
    assert calls[0]["stream"] is True
    assert calls[0]["stream_options"] == {"include_usage": True}
    assert calls[0]["num_retries"] == 0
    assert calls[0]["messages"] == [{"role": "user", "content": "ping"}]
    assert calls[0]["model"] == "openai/test"
    assert calls[0]["base_url"] == "https://example.test/v1"
    assert calls[0]["api_key"] == "test" and calls[0]["max_tokens"] == 123
    lines = [r.message for r in caplog.records if "LLMTOK" in r.message]
    assert len(lines) == 1
    assert "plane=pl model=test" in lines[0]
    assert "ptok=7 ctok=3 ttok=10" in lines[0]


@pytest.mark.parametrize("pieces", [[], [chunk("partial")]])
def test_real_wrapper_synthetic_stop_is_retried(monkeypatch, pieces):
    cut, cut_source = wrapped(pieces)
    good, good_source = wrapped([chunk("answer"), chunk(finish="stop")])
    calls = responses(monkeypatch, cut, good)
    assert L.LLM("openai/test").call("ping") == "answer"
    assert len(calls) == 2
    assert cut.received_finish_reason is None
    assert cut_source.closed.is_set() and good_source.closed.is_set()


@pytest.mark.parametrize("phase", ["create", "first_next"])
def test_first_chunk_deadline_retries_and_closes_late_stream(monkeypatch, phase, caplog):
    unblock_evt = threading.Event()
    source = Stream([chunk("late"), chunk(finish="stop")])

    class Blocked(Stream):
        def __next__(self):
            assert unblock_evt.wait(3)
            return next(source)

    late = source if phase == "create" else Blocked([])
    calls = []

    def completion(**kwargs):
        calls.append(time.monotonic())
        if len(calls) == 1:
            if phase == "create":
                assert unblock_evt.wait(3)
            return late
        return Stream([chunk("answer"), chunk(finish="stop")])

    monkeypatch.setattr(litellm, "completion", completion)
    started = time.monotonic()
    try:
        assert L.LLM("openai/test").call("ping") == "answer"
        assert len(calls) == 2
        # 0.1s deadline + 0.5s backoff, well before the blocked provider can return.
        assert 0.1 <= calls[1] - calls[0] < 1.5
        assert time.monotonic() - started < 1.5
        assert "StreamTruncated" in caplog.text
    finally:
        unblock_evt.set()
    assert late.closed.wait(1)


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
@pytest.mark.parametrize("during_stream", [False, True])
def test_deterministic_4xx_is_not_retried(monkeypatch, code, during_stream):
    err = litellm.BadRequestError("bad request", model="test", llm_provider="openai")
    err.status_code = code
    stream = Stream([chunk("partial"), err])
    calls = responses(monkeypatch, stream if during_stream else err)
    with pytest.raises(type(err)) as caught:
        L.LLM("openai/test").call("ping")
    assert caught.value is err
    assert len(calls) == 1
    if during_stream:
        assert stream.closed.is_set()


@pytest.mark.parametrize("code", [408, 429, 500, 503])
def test_transient_http_failure_retried_once(monkeypatch, code):
    err = litellm.APIError(code, "temporary", llm_provider="openai", model="test")
    calls = responses(monkeypatch, err, Stream([chunk("answer"), chunk(finish="stop")]))
    assert L.LLM("openai/test").call("ping") == "answer"
    assert len(calls) == 2


def test_second_cut_raises_instead_of_retrying_again(monkeypatch):
    calls = responses(monkeypatch, Stream([]), Stream([]))
    with pytest.raises(L.StreamTruncated):
        L.LLM("openai/test").call("ping")
    assert len(calls) == 2


def test_no_retry_without_backoff_and_first_chunk_room(monkeypatch):
    monkeypatch.setattr(L, "_RETRY_BUDGET_S", 0.5)
    calls = responses(monkeypatch, Stream([]))
    with pytest.raises(L.StreamTruncated):
        L.LLM("openai/test").call("ping")
    assert len(calls) == 1


def test_read_failure_discards_partial_answer(monkeypatch):
    cut = Stream([chunk("discard"), httpx.ReadError("connection cut")])
    calls = responses(monkeypatch, cut, Stream([chunk("answer"), chunk(finish="stop")]))
    assert L.LLM("openai/test").call("ping") == "answer"
    assert len(calls) == 2 and cut.closed.is_set()


def test_usage_tail_failure_keeps_finished_answer(monkeypatch):
    source = Stream([chunk("answer"), chunk(finish="stop"), httpx.ReadError("tail cut")])
    calls = responses(monkeypatch, source)
    assert L.LLM("openai/test").call("ping") == "answer"
    assert len(calls) == 1 and source.closed.is_set()


def test_first_chunk_deadline_does_not_limit_generation(monkeypatch):
    def chunks():
        yield chunk(reasoning="thinking")
        time.sleep(0.15)
        yield chunk("answer")
        yield chunk(finish="length")

    responses(monkeypatch, Stream(chunks()))
    assert L.LLM("openai/test").call("ping") == "answer"


def test_only_first_choice_content_is_returned(monkeypatch):
    responses(monkeypatch, Stream([
        chunk("answer"), chunk("other", index=1), chunk(finish="stop"),
    ]))
    assert L.LLM("openai/test").call("ping") == "answer"


def test_creation_and_first_next_share_one_deadline(monkeypatch):
    monkeypatch.setattr(L, "_FIRST_CHUNK_S", 0.3)
    unblock_evt = threading.Event()
    first_started = threading.Event()

    class Blocked(Stream):
        def __next__(self):
            first_started.set()
            assert unblock_evt.wait(3)
            return chunk("late")

    source = Blocked([])

    def completion(**kwargs):
        time.sleep(0.2)
        return source

    monkeypatch.setattr(litellm, "completion", completion)
    started = time.monotonic()
    try:
        with pytest.raises(L.StreamTruncated, match="no first chunk"):
            L._create_completion(model="openai/test", messages=[])
        elapsed = time.monotonic() - started
        assert first_started.is_set()
        # A separate 0.3s allowance for next() would return at 0.5s or later.
        assert 0.28 <= elapsed < 0.45
    finally:
        unblock_evt.set()
    assert source.closed.wait(1)


def test_elapsed_attempt_time_counts_against_retry_budget(monkeypatch):
    from types import SimpleNamespace

    clock = [0.0]
    calls = []
    monkeypatch.setattr(L, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=time.sleep))

    def completion(**kwargs):
        calls.append(kwargs)
        clock[0] = 4.5  # Only 0.5s of the original 5s budget remains.
        raise litellm.InternalServerError("busy", model="test", llm_provider="openai")

    monkeypatch.setattr(litellm, "completion", completion)
    with pytest.raises(litellm.InternalServerError):
        L.LLM("openai/test").call("ping")
    assert len(calls) == 1


def test_unexpected_request_error_is_not_retried(monkeypatch):
    calls = responses(monkeypatch, ValueError("invalid local input"))
    with pytest.raises(ValueError, match="invalid local input"):
        L.LLM("openai/test").call("ping")
    assert len(calls) == 1
