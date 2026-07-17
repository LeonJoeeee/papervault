"""Tests for _verify_pdf_matches_metadata — now an LLM (MiMo v2.5) identity judge
on the first 2 pages, behind cheap accept-on-doubt valves.

Verified: the valves (short title / no text / garbage font) short-circuit BEFORE
any LLM call; the LLM verdict is honored; LLM/parse errors fail OPEN (accept).
"""
from __future__ import annotations

import json
from unittest.mock import patch

from papervault.library import download
from papervault.library.models import Paper


class _FakePage:
    def __init__(self, text):
        self._t = text

    def extract_text(self):
        return self._t


class _FakeReader:
    def __init__(self, pages):
        self.pages = [_FakePage(t) for t in pages]


class _FakeLLM:
    """Records call count; returns a fixed raw string, or a SEQUENCE of raws (one
    per successive call, last repeated), or raises every call."""
    def __init__(self, responses="", *, raises=False):
        self._resp = responses if isinstance(responses, list) else [responses]
        self.raises = raises
        self.calls = 0

    def call(self, messages):
        i = self.calls
        self.calls += 1
        if self.raises:
            raise RuntimeError("llm down")
        return self._resp[min(i, len(self._resp) - 1)]


def _verdict(match: bool, reason="r"):
    return json.dumps({"match": match, "reason": reason})


def _verify(title, authors, pages, llm):
    paper = Paper(key="k", title=title, authors=authors)
    with patch("pypdf.PdfReader", return_value=_FakeReader(pages)):
        return download._verify_pdf_matches_metadata(b"%PDF-fake", paper, llm=llm)


_REAL_PAGE = ("Cosmic-Ray Transport Coefficients\nJoe Giacalone\nThe University "
              "of Arizona\nAbstract. A review of cosmic-ray transport coefficients "
              "is presented with emphasis on cross-field transport. " * 3)


# ----------------------------- LLM verdict honored ------------------------

def test_llm_match_accepts():
    llm = _FakeLLM(_verdict(True))
    ok, reason = _verify("Cosmic-Ray Transport Coefficients", ["Giacalone"],
                         [_REAL_PAGE], llm)
    assert ok is True and llm.calls == 1


def test_llm_mismatch_rejects():
    llm = _FakeLLM(_verdict(False, "this is a different paper"))
    ok, reason = _verify("Cosmic-Ray Transport Coefficients", ["Giacalone"],
                         [_REAL_PAGE], llm)
    assert ok is False and llm.calls == 1


# ------------------------------- fail-open --------------------------------

def test_llm_error_retries_then_fails_open():
    llm = _FakeLLM(raises=True)
    ok, reason = _verify("A Sufficiently Long Paper Title", ["Author"],
                         [_REAL_PAGE], llm)
    assert ok is True and "fail_open" in reason
    assert llm.calls == 3                      # retried before failing open


def test_llm_unparseable_retries_then_fails_open():
    llm = _FakeLLM("sorry, I cannot answer in JSON")
    ok, reason = _verify("A Sufficiently Long Paper Title", ["Author"],
                         [_REAL_PAGE], llm)
    assert ok is True and "fail_open" in reason
    assert llm.calls == 3


def test_llm_retry_recovers_from_a_bad_reply():
    """First reply is malformed, second is valid JSON → the retry recovers and
    the verdict is honored (no fail-open)."""
    llm = _FakeLLM(["garbled non-json reply", _verdict(False, "wrong paper")])
    ok, reason = _verify("A Sufficiently Long Paper Title", ["Author"],
                         [_REAL_PAGE], llm)
    assert ok is False and llm.calls == 2 and "mismatch" in reason


# --------------------- valves short-circuit BEFORE the LLM ----------------

def test_short_title_accepts_without_llm():
    llm = _FakeLLM(_verdict(False))      # would reject if reached
    ok, reason = _verify("CRs", ["Author"], [_REAL_PAGE], llm)
    assert ok is True and llm.calls == 0 and "short" in reason


def test_no_text_accepts_without_llm():
    llm = _FakeLLM(_verdict(False))
    ok, reason = _verify("A Sufficiently Long Paper Title", ["Author"],
                         ["tiny"], llm)
    assert ok is True and llm.calls == 0


def test_garbage_font_accepts_without_llm():
    llm = _FakeLLM(_verdict(False))
    garbage = "/BT/D6/D8/CX/AC " * 60    # low alpha, no English stopwords
    ok, reason = _verify("A Sufficiently Long Paper Title", ["Author"],
                         [garbage], llm)
    assert ok is True and llm.calls == 0 and "garbage" in reason
