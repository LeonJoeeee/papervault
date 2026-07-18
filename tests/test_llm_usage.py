"""Unit tests for the LLMTOK token account book (issue #4, round 2)."""
import logging
from types import SimpleNamespace

from papervault.llm_usage import log_usage

log = logging.getLogger("test.llmtok")


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if "LLMTOK" in r.getMessage()]


def test_full_usage_logged(caplog):
    resp = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=198432, completion_tokens=1810, total_tokens=200242)
    )
    with caplog.at_level(logging.INFO, logger="test.llmtok"):
        log_usage(log, "ks", "mimo-v2.5-pro", 84.21, resp)
    (line,) = _lines(caplog)
    assert "plane=ks" in line
    assert "model=mimo-v2.5-pro" in line
    assert "dur=84.2s" in line
    assert "ptok=198432" in line and "ctok=1810" in line and "ttok=200242" in line


def test_missing_usage_logs_question_marks(caplog):
    with caplog.at_level(logging.INFO, logger="test.llmtok"):
        log_usage(log, "pl", "m", 1.0, SimpleNamespace())  # no .usage at all
        log_usage(log, "pl", "m", 1.0, SimpleNamespace(usage=None))
    lines = _lines(caplog)
    assert len(lines) == 2
    for line in lines:
        assert "ptok=? ctok=? ttok=?" in line


def test_partial_usage_mixes_values_and_gaps(caplog):
    resp = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=None, total_tokens=None)
    )
    with caplog.at_level(logging.INFO, logger="test.llmtok"):
        log_usage(log, "ks", "m", 0.5, resp)
    (line,) = _lines(caplog)
    assert "ptok=10" in line and "ctok=?" in line and "ttok=?" in line


def test_library_call_site_wired(caplog, monkeypatch):
    # Wiring regression guard (review nit 4): LLM.call must emit an LLMTOK line
    # with the bare (prefix-stripped) model label.
    import litellm

    from papervault.library.llm import LLM

    fake_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
    )
    monkeypatch.setattr(litellm, "completion", lambda **kw: fake_resp)
    client = LLM("openai/mimo-v2.5-pro")
    with caplog.at_level(logging.INFO, logger="papervault.library.llm"):
        out = client.call("ping")
    assert out == "hi"
    lines = [r.getMessage() for r in caplog.records if "LLMTOK" in r.getMessage()]
    (line,) = lines
    assert "plane=pl" in line
    assert "model=mimo-v2.5-pro" in line  # provider prefix stripped
    assert "ptok=7" in line and "ctok=3" in line


def test_emission_never_raises(caplog):
    class Poison:
        @property
        def usage(self):
            raise RuntimeError("boom")

    with caplog.at_level(logging.INFO, logger="test.llmtok"):
        log_usage(log, "ks", "m", 0.1, Poison())  # must not raise
    assert _lines(caplog) == []
