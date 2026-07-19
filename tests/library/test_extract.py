"""Tests for extract.py — single-engine whole-PDF MinerU2.5-Pro extraction.

2026-06-06 (MinerU migration): the dots/chandra/marker per-chunk OCR cascade +
per-paper GPU pin + chunk cache are gone. ``extract_md`` is now an async coroutine
that makes ONE whole-PDF ``mineru_client.extract_mineru`` call. These tests stub
``extract.extract_mineru`` (so no real MinerU server is needed) and exercise:

  - the surviving ``review_extract`` / ``completeness_gate`` LLM judges (unchanged);
  - ``pdf_probe`` (now only ``bad`` / ``n_pages`` / ``reason`` — no chunk fields);
  - the ``extract_md`` spine: probe-bad, gate-reject, gate-pass-save, firecrawl
    no-re-OCR;
  - the C1 transport-vs-extraction split (SDD §2.2): a transport error charges NO
    attempt and never terminalizes; an extraction error charges and terminalizes
    at the retry BUDGET only; the completeness-gate / clarity rejects terminalize
    WITHOUT a charge.
"""

from __future__ import annotations

import asyncio

import pytest

from papervault.library import Library, extract


# ----------- review_extract -------------------------------------------------


def test_review_extract_short_returns_fail():
    out = extract.review_extract("too short")
    assert out["ok"] is False
    assert "extract_too_short" in out["issues"]


def test_review_extract_llm_unavailable_is_conservative():
    class BoomLLM:
        def call(self, msgs):
            raise RuntimeError("api down")
    out = extract.review_extract("X" * 1000, llm=BoomLLM())
    # Conservative ok=True so we don't lose work over a flaky API.
    assert out["ok"] is True
    assert "review_llm_unavailable" in out["issues"]


def test_review_extract_parses_valid_json():
    class FakeLLM:
        def call(self, msgs):
            return '{"ok": false, "issues": ["loop"], "confidence": 0.9}'
    out = extract.review_extract("X" * 1000, llm=FakeLLM())
    assert out["ok"] is False
    assert out["issues"] == ["loop"]


def test_review_extract_no_braces_returns_conservative():
    class FakeLLM:
        def call(self, msgs):
            return "no JSON whatsoever"
    out = extract.review_extract("X" * 1000, llm=FakeLLM())
    assert out["ok"] is True
    assert "review_parse_failed" in out["issues"]


def test_review_extract_malformed_json_returns_conservative():
    class FakeLLM:
        def call(self, msgs):
            return '{"ok": true, "issues": ['
    out = extract.review_extract("X" * 1000, llm=FakeLLM())
    assert out["ok"] is True
    assert "review_parse_failed" in out["issues"]


# ----------- review_extract is CLARITY-ONLY (D4 / S4) -----------------------


def test_review_extract_dropped_broken_pdf_axis():
    """S4/D4: review_extract is narrowed to clarity only — it no longer carries
    the ``broken_pdf_suspected`` completeness/PDF-source axis (that moved
    entirely to completeness_gate). Every return path omits the key."""
    # short-floor path
    assert "broken_pdf_suspected" not in extract.review_extract("short")

    class BoomLLM:
        def call(self, msgs):
            raise RuntimeError("api down")
    assert "broken_pdf_suspected" not in extract.review_extract(
        "X" * 1000, llm=BoomLLM())

    class FakeLLM:
        def call(self, msgs):
            return '{"ok": true, "issues": [], "confidence": 0.9}'
    out = extract.review_extract("X" * 1000, llm=FakeLLM())
    assert "broken_pdf_suspected" not in out
    assert set(out.keys()) == {"ok", "issues", "confidence"}


def test_review_extract_ignores_broken_pdf_from_llm():
    """Even if the LLM volunteers a ``broken_pdf_suspected`` field, the
    clarity-only reviewer drops it — completeness/paywall is not its call (D4).
    A readable chunk the LLM tags broken_pdf stays ``ok=True`` and the key is
    not propagated."""
    class FakeLLM:
        def call(self, msgs):
            return ('{"ok": true, "broken_pdf_suspected": true, '
                    '"issues": ["looks like a landing page"], '
                    '"confidence": 0.8}')
    out = extract.review_extract("X" * 1000, llm=FakeLLM())
    assert out["ok"] is True
    assert "broken_pdf_suspected" not in out


def test_review_extract_clarity_pass_short_chunk_is_not_penalized():
    """D4: review_extract NEVER judges by length — a legitimately short but
    perfectly readable chunk (above the blank-floor) passes. Only the
    <_BLANK_CHUNK_FLOOR (100-char) blank-floor fails, and that is a
    blank/refused chunk, not a 'short' verdict. (See the F2 tests below for the
    formerly-fatal 100-499 char band.)"""
    class FakeLLM:
        def call(self, msgs):
            # The reviewer is asked the clarity question and says: readable.
            return '{"ok": true, "issues": [], "confidence": 0.95}'
    # ~600 chars: short for a paper, but well above the blank floor → readable.
    short_but_readable = "A readable sentence. " * 30
    assert len(short_but_readable) >= 500
    out = extract.review_extract(short_but_readable, llm=FakeLLM())
    assert out["ok"] is True


def test_review_extract_flags_garbled_chunk():
    """The one thing review_extract DOES flag: an unreadable (garbled/looping)
    chunk → ok=False, with the concrete issue surfaced."""
    class FakeLLM:
        def call(self, msgs):
            return ('{"ok": false, "issues": ["looping: Hull, Hull, Hulf"], '
                    '"confidence": 0.95}')
    out = extract.review_extract("X" * 1000, llm=FakeLLM())
    assert out["ok"] is False
    assert out["issues"] == ["looping: Hull, Hull, Hulf"]


def test_review_extract_string_false_is_not_passed():
    """An LLM that emits the JSON string ``"false"`` for ``ok`` must NOT be
    read as truthy (``bool('false')`` is True) — a genuinely unreadable chunk
    would otherwise slip through."""
    class FakeLLM:
        def call(self, msgs):
            return '{"ok": "false", "issues": ["garbled"], "confidence": 0.9}'
    out = extract.review_extract("X" * 1000, llm=FakeLLM())
    assert out["ok"] is False


# ----------- completeness_gate (D3, the whole-document gate) ----------------


class _GateLLM:
    """Fake LLM that returns a fixed completeness verdict, and records the
    text it was asked to judge so a test can assert it was NOT mutated."""

    def __init__(self, reply: str):
        self._reply = reply
        self.seen: str | None = None

    def call(self, msgs):
        self.seen = msgs[-1]["content"]
        return self._reply


def test_completeness_gate_complete_full_paper():
    """A normal, complete paper → complete=true."""
    llm = _GateLLM('{"complete": true, "reason": "complete"}')
    out = extract.completeness_gate("Intro... Methods... Results... " * 50,
                                    llm=llm)
    assert out["complete"] is True
    assert out["reason"] == "complete"


def test_completeness_gate_truncated_is_incomplete():
    """A mid-word truncation → complete=false with the reason carried through."""
    llm = _GateLLM(
        '{"complete": false, "reason": "truncated: ends \'we therefore conclu\'"}'
    )
    out = extract.completeness_gate("long body ... we therefore conclu", llm=llm)
    assert out["complete"] is False
    assert "truncated" in out["reason"]


def test_completeness_gate_paywall_stub_is_incomplete():
    """A login / paywall stub (no body) → complete=false."""
    llm = _GateLLM('{"complete": false, "reason": "paywall: \'Sign in to access\'"}')
    out = extract.completeness_gate("Title\nAuthors\nSign in to access", llm=llm)
    assert out["complete"] is False
    assert "paywall" in out["reason"]


def test_completeness_gate_short_but_complete_is_complete():
    """A 2-page letter is SHORT but whole — must NOT be judged incomplete by
    length (the core D3 rule). The gate trusts the LLM verdict; with a
    'short but whole' verdict it serves."""
    llm = _GateLLM('{"complete": true, "reason": "short but whole letter"}')
    out = extract.completeness_gate("A brief two-page research note. The end.",
                                    llm=llm)
    assert out["complete"] is True
    assert "whole" in out["reason"]


def test_completeness_gate_is_judge_only_never_mutates_text():
    """Lossless (D3): the gate inspects the EXACT text and never alters it."""
    text = "Verbatim body  with  odd   spacing and a $\\frac{a}{b}$ formula."
    llm = _GateLLM('{"complete": true, "reason": "complete"}')
    extract.completeness_gate(text, llm=llm)
    assert llm.seen == text  # passed through byte-for-byte


def test_completeness_gate_llm_unavailable_is_fail_open():
    """A real LLM error → fail-OPEN (complete=true): the text already passed
    every per-chunk review; a flaky API must not throw it away (SDD §5)."""
    class BoomLLM:
        def call(self, msgs):
            raise RuntimeError("api down")
    out = extract.completeness_gate("X" * 1000, llm=BoomLLM())
    assert out["complete"] is True
    assert out["reason"] == "gate_llm_unavailable"


def test_completeness_gate_no_braces_is_fail_open():
    llm = _GateLLM("no JSON whatsoever")
    out = extract.completeness_gate("X" * 1000, llm=llm)
    assert out["complete"] is True
    assert out["reason"] == "gate_parse_failed"


def test_completeness_gate_malformed_json_is_fail_open():
    llm = _GateLLM('{"complete": false, ')  # truncated JSON
    out = extract.completeness_gate("X" * 1000, llm=llm)
    assert out["complete"] is True
    assert out["reason"] == "gate_parse_failed"


def test_completeness_gate_empty_text_is_incomplete():
    """Empty / whitespace-only text is the one hard-coded incomplete — there is
    nothing to serve, so the LLM is never even called."""
    out = extract.completeness_gate("   \n  ", llm=_GateLLM("unused"))
    assert out["complete"] is False
    assert out["reason"] == "empty_text"


def test_completeness_gate_string_false_is_incomplete():
    """LLMs commonly emit the JSON string "false" instead of a boolean. The
    gate must read that as incomplete (``bool("false")`` would be True → a
    fail-OPEN in the one direction the gate exists to catch — serving a
    genuinely-incomplete paper as complete)."""
    llm = _GateLLM('{"complete": "false", "reason": "truncated"}')
    out = extract.completeness_gate("long body ... we therefore conclu", llm=llm)
    assert out["complete"] is False
    assert out["reason"] == "truncated"


def test_completeness_gate_string_true_is_complete():
    """The mirror: a string "true" verdict reads as complete (so the string
    coercion doesn't accidentally flip a genuine pass to incomplete)."""
    llm = _GateLLM('{"complete": "true", "reason": "complete"}')
    out = extract.completeness_gate("Intro... Methods... Results... " * 50,
                                    llm=llm)
    assert out["complete"] is True


# ----------- pdf_probe (isolated subprocess health-check, SDD §2.1) ---------
#
# PDFProbe carries only ``bad`` / ``n_pages`` / ``reason`` now — the chunking
# fields (``n_chunks`` / ``single_chunk``) and the ``chunk_size`` param are gone
# (whole-PDF call, terminal is budget-only).


def _write_pdf(path, n_pages=1):
    """Write a real (tiny) multi-page PDF via pypdf so pdf_probe's child can
    actually parse it."""
    from pypdf import PdfWriter
    w = PdfWriter()
    for _ in range(n_pages):
        w.add_blank_page(width=72, height=72)
    with open(path, "wb") as f:
        w.write(f)


def test_pdf_probe_real_pdf_counts_pages(tmp_path):
    pdf = tmp_path / "good.pdf"
    _write_pdf(pdf, n_pages=3)
    probe = extract.pdf_probe(str(pdf))
    assert probe.bad is False
    assert probe.n_pages == 3
    assert probe.reason == "ok"


def test_pdf_probe_long_pdf_counts_pages(tmp_path):
    """A long PDF is just a big ``n_pages`` — there is no chunk count anymore."""
    pdf = tmp_path / "long.pdf"
    _write_pdf(pdf, n_pages=65)
    probe = extract.pdf_probe(str(pdf))
    assert probe.bad is False
    assert probe.n_pages == 65


def test_pdf_probe_has_no_chunk_fields():
    """PDFProbe no longer exposes ``n_chunks`` / ``single_chunk`` (SDD §6.1)."""
    probe = extract.PDFProbe(bad=False, n_pages=5, reason="ok")
    assert not hasattr(probe, "n_chunks")
    assert not hasattr(probe, "single_chunk")
    assert {f for f in probe.__dataclass_fields__} == {"bad", "n_pages", "reason"}


def test_pdf_probe_not_a_pdf_is_bad(tmp_path):
    """A file without %PDF magic → bad, child never even imports pypdf."""
    f = tmp_path / "garbage.pdf"
    f.write_bytes(b"this is not a pdf at all")
    probe = extract.pdf_probe(str(f))
    assert probe.bad is True
    assert probe.n_pages == 0
    assert probe.reason == "not_pdf"


def test_pdf_probe_timeout_is_bad(tmp_path, monkeypatch):
    """A wedged child (simulated via a TimeoutExpired) → conservatively bad,
    so the daemon's main thread is never the one that hangs."""
    import subprocess as _sp

    def boom(*a, **kw):
        raise _sp.TimeoutExpired(cmd="probe", timeout=1)

    monkeypatch.setattr(extract.subprocess, "run", boom)
    probe = extract.pdf_probe(str(tmp_path / "whatever.pdf"))
    assert probe.bad is True
    assert probe.reason == "probe_timeout"


# ----------- extract_md spine (single whole-PDF MinerU call, SDD §2.1) -------
#
# ``extract_md`` is now an async coroutine making ONE ``extract_mineru`` call.
# We stub ``extract.extract_mineru`` (and ``pdf_probe``) so the spine runs
# without a real MinerU server / real PDF, and stub the two LLM judges
# (``review_extract`` / ``completeness_gate``) so they don't reach a real LLM.


def _pdf_paper(tmp_path):
    lib = Library(tmp_path)
    p, _ = lib.upsert({"title": "Gate spine paper", "authors": ["A"],
                       "year": 2024, "doi": "10.1/gate-spine"})
    lib.pdf_path(p.key).write_bytes(b"%PDF-1.4 fake")
    lib.save()
    return lib, p


def _stub_good_probe(monkeypatch, n_pages=10):
    monkeypatch.setattr(
        extract, "pdf_probe",
        lambda *a, **kw: extract.PDFProbe(bad=False, n_pages=n_pages,
                                          reason="ok"))


def _stub_mineru(monkeypatch, *, returns=None, raises=None):
    """Stub the whole-PDF ``extract_mineru`` coroutine to a fixed md (or to
    raise one of the C1 typed exceptions). Records the call count."""
    calls = {"n": 0}

    async def fake(pdf_bytes, endpoints, *, stem="doc", **kw):
        calls["n"] += 1
        if raises is not None:
            raise raises
        return returns
    monkeypatch.setattr(extract, "extract_mineru", fake)
    return calls


def _stub_judges_pass(monkeypatch):
    """Make both LLM judges pass so the spine reaches save on a good md."""
    monkeypatch.setattr(extract, "review_extract",
                        lambda text, *, llm=None: {"ok": True, "issues": [],
                                                   "confidence": 1.0})
    monkeypatch.setattr(extract, "completeness_gate",
                        lambda text, *, llm=None: {"complete": True,
                                                   "reason": "complete"})


def test_extract_md_is_coroutine():
    """The single-engine spine is async (it awaits ``extract_mineru``)."""
    assert asyncio.iscoroutinefunction(extract.extract_md)


def test_extract_md_gate_pass_saves_and_sets_ok(tmp_path, monkeypatch):
    """A complete md → both judges pass → md saved (mineru2.5-pro) + status ok."""
    lib, p = _pdf_paper(tmp_path)
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, returns="REAL ASSEMBLED BODY of the paper. " * 30)
    _stub_judges_pass(monkeypatch)

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out == "REAL ASSEMBLED BODY of the paper. " * 30
    assert lib.has_extract(p.key, "md")
    assert p.download_status == "ok"
    assert p.md_engine == "mineru2.5-pro"
    # Hardcoded model id (SDD §3.3 item 4c) — non-empty even without mineru.
    assert p.md_engine_version == "MinerU2.5-Pro-2605-1.2B"


def test_extract_md_gate_reject_blocks_save_and_flips_status(tmp_path, monkeypatch):
    """An incomplete md → completeness_gate rejects → NO md on disk,
    status=extract_failed, attempts NOT charged (D5), return None."""
    lib, p = _pdf_paper(tmp_path)
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, returns="paywall stub body")
    monkeypatch.setattr(extract, "review_extract",
                        lambda text, *, llm=None: {"ok": True, "issues": [],
                                                   "confidence": 1.0})
    monkeypatch.setattr(
        extract, "completeness_gate",
        lambda text, *, llm=None: {"complete": False, "reason": "truncated"})

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    assert not lib.has_extract(p.key, "md")          # md NOT saved
    assert p.download_status == "extract_failed"      # terminal
    assert p.extract_attempts == 0                    # gate reject does NOT charge


def test_extract_md_clarity_reject_blocks_save_no_charge(tmp_path, monkeypatch):
    """A readable-but-garbled md → review_extract clarity FAIL → extract_failed
    WITHOUT charging an attempt (a retry would hit the same garbled bytes)."""
    lib, p = _pdf_paper(tmp_path)
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, returns="Hull, Hull, Hulf, Hull, Hulf " * 50)
    monkeypatch.setattr(
        extract, "review_extract",
        lambda text, *, llm=None: {"ok": False, "issues": ["looping garble"],
                                   "confidence": 0.95})
    # completeness_gate must never be reached on a clarity reject.
    monkeypatch.setattr(
        extract, "completeness_gate",
        lambda text, *, llm=None: pytest.fail("gate after clarity reject"))

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    assert not lib.has_extract(p.key, "md")
    assert p.download_status == "extract_failed"
    assert p.extract_attempts == 0


def test_extract_md_already_has_md_short_circuits(tmp_path, monkeypatch):
    """An md already on disk is terminal truth — extract_md returns it verbatim
    and never calls extract_mineru (idempotent, no re-OCR)."""
    lib, p = _pdf_paper(tmp_path)
    lib.md_path(p.key).write_text("existing md body")
    p.md_path = str(lib.md_path(p.key).relative_to(lib.root))
    calls = _stub_mineru(monkeypatch, returns="SHOULD NOT BE USED")
    monkeypatch.setattr(extract, "pdf_probe",
                        lambda *a, **kw: pytest.fail("must not probe when md on disk"))

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out == "existing md body"
    assert calls["n"] == 0


def test_extract_md_firecrawl_md_not_reocrd_when_pdf_present(tmp_path, monkeypatch):
    """A firecrawl md on disk is terminal — extract_md NEVER re-OCRs it just
    because a real PDF also exists (D5). The early ``has_extract(md)`` guard
    returns the firecrawl md verbatim; extract_mineru/probe/gate never run, and
    classify routes the paper to TERMINAL while serve-safety keeps the md."""
    from papervault.library.mcp.server import _attach_text_reference
    from papervault.library.services import classify as classify_mod

    lib, p = _pdf_paper(tmp_path)              # a real PDF is on disk
    md_path = lib.md_path(p.key)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("firecrawl body")
    p.md_path = str(md_path.relative_to(lib.root))
    p.md_engine = "firecrawl"
    p.download_status = "ok"

    calls = _stub_mineru(monkeypatch, returns="SHOULD NOT BE USED")
    monkeypatch.setattr(extract, "pdf_probe",
                        lambda *a, **kw: pytest.fail("re-OCR must not run"))

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out == md_path.read_text()          # returned the firecrawl md verbatim
    assert calls["n"] == 0                      # extract_mineru never consulted
    assert lib.has_extract(p.key, "md")
    assert p.download_status == "ok"            # untouched

    # classify routes it to rule-2 TERMINAL (real PDF + md = done; no re-hunt).
    assert classify_mod.classify(p, lib) == classify_mod.TERMINAL
    rec: dict = {}
    _attach_text_reference(rec, p, lib)
    assert "text_path" in rec
    assert "text_status" not in rec


# ----------- extract_md probe-bad accounting (SDD §2.1 step 1) ---------------


def test_extract_md_bad_probe_charges_attempt_and_terminates(tmp_path, monkeypatch):
    """A bad local PDF is a genuine per-doc defect → attempts++. With
    MAX_EXTRACT_ATTEMPTS==1 a single bad probe terminalizes; the server is never
    called."""
    from papervault.library.models import MAX_EXTRACT_ATTEMPTS
    lib, p = _pdf_paper(tmp_path)
    p.extract_attempts = MAX_EXTRACT_ATTEMPTS - 1
    monkeypatch.setattr(
        extract, "pdf_probe",
        lambda *a, **kw: extract.PDFProbe(bad=True, n_pages=0, reason="not_pdf"))
    calls = _stub_mineru(monkeypatch, returns="SHOULD NOT BE USED")

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    assert p.extract_attempts == MAX_EXTRACT_ATTEMPTS
    assert p.download_status == "extract_failed"
    assert calls["n"] == 0                      # no server call on a bad probe


def test_extract_md_bad_probe_under_budget_not_terminal(tmp_path, monkeypatch):
    """A bad probe below the retry budget charges an attempt but does NOT
    terminalize (terminal is budget-only — the old n_chunks==1 instant-terminal
    disjunct is dropped, SDD §2.1)."""
    from papervault.library.models import MAX_EXTRACT_ATTEMPTS
    if MAX_EXTRACT_ATTEMPTS < 2:
        pytest.skip("budget is 1 — no under-budget window to test")
    lib, p = _pdf_paper(tmp_path)
    monkeypatch.setattr(
        extract, "pdf_probe",
        lambda *a, **kw: extract.PDFProbe(bad=True, n_pages=0, reason="not_pdf"))
    _stub_mineru(monkeypatch, returns="x")

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    assert p.extract_attempts == 1
    assert p.download_status != "extract_failed"


# ----------- C1: transport-vs-extraction failure split (SDD §2.2) -----------
#
# The single failure bucket of the old cascade is split into two typed
# exceptions. ``MineruTransportError`` → NO charge, NO terminal (non-mutation
# lets classify re-route next sweep). ``MineruExtractionError`` → charge,
# terminal at the retry BUDGET only.


def test_c1_transport_error_charges_no_attempt_not_terminal(tmp_path, monkeypatch):
    """C1 CORE: a transport failure (server down / bare-500 / both endpoints
    dead) leaves status AND attempts UNMUTATED so classify re-routes the paper
    next reconcile sweep — a server outage can never terminalize a paper."""
    from papervault.library.mineru_client import MineruTransportError
    lib, p = _pdf_paper(tmp_path)
    p.download_status = "ok"
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, raises=MineruTransportError("server unreachable"))

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    assert p.extract_attempts == 0              # NO charge on transport
    assert p.download_status == "ok"            # NOT terminalized
    assert not lib.has_extract(p.key, "md")


def test_c1_transport_error_at_budget_minus_one_still_not_terminal(tmp_path, monkeypatch):
    """Even a paper already at MAX-1 attempts stays non-terminal on a transport
    error — a transport error NEVER charges, so it can never tip the budget."""
    from papervault.library.models import MAX_EXTRACT_ATTEMPTS
    from papervault.library.mineru_client import MineruTransportError
    lib, p = _pdf_paper(tmp_path)
    p.download_status = "pending"
    p.extract_attempts = MAX_EXTRACT_ATTEMPTS - 1
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, raises=MineruTransportError("bare 500 during restart"))

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    assert p.extract_attempts == MAX_EXTRACT_ATTEMPTS - 1   # untouched
    assert p.download_status == "pending"                   # not terminal


def test_c1_extraction_error_charges_attempt(tmp_path, monkeypatch):
    """C1: an extraction-class failure (400/422/structured-500/truncation/thin
    md) is a per-doc defect → charge an attempt. Below budget it does NOT
    terminalize."""
    from papervault.library.models import MAX_EXTRACT_ATTEMPTS
    from papervault.library.mineru_client import MineruExtractionError
    lib, p = _pdf_paper(tmp_path)
    p.download_status = "ok"
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch,
                 raises=MineruExtractionError("truncated: unexpected finish_reason"))

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    assert p.extract_attempts == 1              # charged
    if MAX_EXTRACT_ATTEMPTS > 1:
        assert p.download_status != "extract_failed"   # not terminal yet


def test_c1_extraction_error_terminal_at_budget_only(tmp_path, monkeypatch):
    """C1: an extraction-class failure terminalizes ONLY when the charge hits
    MAX_EXTRACT_ATTEMPTS — never on first contact (the dropped n_chunks==1
    instant-terminal disjunct)."""
    from papervault.library.models import MAX_EXTRACT_ATTEMPTS
    from papervault.library.mineru_client import MineruExtractionError
    lib, p = _pdf_paper(tmp_path)
    p.download_status = "ok"
    p.extract_attempts = MAX_EXTRACT_ATTEMPTS - 1
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, raises=MineruExtractionError("400 malformed pdf"))

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    assert p.extract_attempts == MAX_EXTRACT_ATTEMPTS
    assert p.download_status == "extract_failed"


def test_c1_empty_md_belt_and_suspenders_charges(tmp_path, monkeypatch):
    """A 200-OK-but-empty md body (extract_mineru returned empty rather than
    raising) is treated as a per-doc extraction failure → charge an attempt."""
    from papervault.library.models import MAX_EXTRACT_ATTEMPTS
    lib, p = _pdf_paper(tmp_path)
    p.download_status = "ok"
    p.extract_attempts = MAX_EXTRACT_ATTEMPTS - 1
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, returns="   \n  ")   # whitespace-only

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    assert p.extract_attempts == MAX_EXTRACT_ATTEMPTS
    assert p.download_status == "extract_failed"
    assert not lib.has_extract(p.key, "md")


# ----------- issue #43: transport-defer stamping (busy-loop fix) ------------
#
# A transport failure still charges nothing / does not terminalize (C1 above),
# but it now ALSO stamps a per-artifact deferral marker so reconcile can stop
# re-enqueuing the paper every sweep while the backend stays down. A NON-transport
# outcome (success / per-doc defect) clears the marker; a success also advances
# the process-global success epoch (backend-recovery signal).


def test_transport_error_stamps_deferral_against_current_pdf(tmp_path, monkeypatch):
    from papervault.library.mineru_client import MineruTransportError
    from papervault.library.services import extract_defer
    extract_defer.reset_for_test()
    lib, p = _pdf_paper(tmp_path)
    p.download_status = "ok"
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, raises=MineruTransportError("mineru_import_failed"))

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    # Deferred against the current PDF artifact + the current success epoch.
    assert p.extract_deferred_sig == extract_defer.pdf_sig(lib, p.key)
    assert p.extract_deferred_epoch == extract_defer.success_epoch()
    # And, per C1, still non-terminal with no attempt charged.
    assert p.extract_attempts == 0
    assert p.download_status == "ok"
    # The paper now reads as "deferred against this artifact".
    assert extract_defer.is_extract_deferred(p, lib) is True


def test_success_clears_deferral_and_advances_epoch(tmp_path, monkeypatch):
    from papervault.library.services import extract_defer
    extract_defer.reset_for_test()
    lib, p = _pdf_paper(tmp_path)
    p.download_status = "ok"
    # Pretend a prior sweep deferred it.
    extract_defer.mark_extract_deferred(p, lib)
    assert extract_defer.is_extract_deferred(p, lib) is True
    epoch_before = extract_defer.success_epoch()

    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, returns="REAL ASSEMBLED BODY of the paper. " * 30)
    _stub_judges_pass(monkeypatch)
    out = asyncio.run(extract.extract_md(p, lib, llm=object()))

    assert out is not None and lib.has_extract(p.key, "md")
    assert p.extract_deferred_sig is None            # cleared on success
    assert extract_defer.success_epoch() == epoch_before + 1   # backend-recovery signal
    assert extract_defer.is_extract_deferred(p, lib) is False


def test_per_doc_defect_clears_stale_deferral(tmp_path, monkeypatch):
    """A genuine per-doc extraction failure must NOT leave the paper marked
    transport-deferred — it charges an attempt and keeps retrying up to budget,
    so reconcile must not skip it."""
    from papervault.library.mineru_client import MineruExtractionError
    from papervault.library.services import extract_defer
    extract_defer.reset_for_test()
    lib, p = _pdf_paper(tmp_path)
    p.download_status = "ok"
    extract_defer.mark_extract_deferred(p, lib)      # stale marker from a prior blip
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, raises=MineruExtractionError("truncated"))

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    assert p.extract_attempts == 1                   # charged (per-doc defect)
    assert p.extract_deferred_sig is None            # marker cleared → not skipped
    assert extract_defer.is_extract_deferred(p, lib) is False


def test_transport_defer_stamp_survives_save_load_roundtrip(tmp_path, monkeypatch):
    """Worker save-to-disk pin: the transport-defer stamp the extract worker writes
    (via ``extract_md``'s transport arm, then the worker's ``library.save()``) is a
    PERSISTED ``Paper`` field, so it must survive a Library save/load round-trip —
    otherwise a restart could not read the stamp back at all. (The process success
    epoch is separately process-local; only the per-paper sig + epoch stamp is
    persisted, and this pins that persistence.)"""
    from papervault.library.mineru_client import MineruTransportError
    from papervault.library.services import extract_defer
    extract_defer.reset_for_test()
    lib, p = _pdf_paper(tmp_path)
    p.download_status = "ok"
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, raises=MineruTransportError("mineru_import_failed"))

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    sig, epoch = p.extract_deferred_sig, p.extract_deferred_epoch
    assert sig is not None                          # stamped by the (worker) transport arm

    lib.save()                                      # the worker's post-extract library.save()
    # Reload from the SAME vault root (a restart) → deserialize the persisted stamp.
    lib2 = Library(str(tmp_path))
    p2 = lib2.get(p.key)
    assert p2 is not None
    assert p2.extract_deferred_sig == sig           # sig survived the round-trip
    assert p2.extract_deferred_epoch == epoch        # epoch survived the round-trip


def test_extract_md_no_pdf_returns_none(tmp_path, monkeypatch):
    """No PDF on disk → return None without touching the server or attempts."""
    lib = Library(tmp_path)
    p, _ = lib.upsert({"title": "No pdf paper here for the gate", "authors": ["A"],
                       "year": 2020, "doi": "10.1/nopdf"})
    calls = _stub_mineru(monkeypatch, returns="x")

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out is None
    assert calls["n"] == 0
    assert p.extract_attempts == 0


# ----------- _save_md provenance (SDD §3.3 item 4c) -------------------------


def test_save_md_mineru_stamps_hardcoded_version(tmp_path):
    """_save_md for a mineru md stamps the HARDCODED model id (non-empty even if
    mineru is not importable), not a _pkg_version lookup."""
    lib = Library(tmp_path)
    p, _ = lib.upsert({"title": "Save md provenance paper title", "authors": ["A"],
                       "year": 2024, "doi": "10.1/savemd"})
    extract._save_md(p, lib, "mineru2.5-pro", "# body\nreal md")
    assert p.md_engine == "mineru2.5-pro"
    assert p.md_engine_version == "MinerU2.5-Pro-2605-1.2B"
    assert lib.has_extract(p.key, "md")
    # Raw body, NO YAML frontmatter (OCR md never carries any).
    assert lib.md_path(p.key).read_text() == "# body\nreal md"


# ----------- confirmed_completeness_gate (D3 hardening, 2026-06-10) ---------
# The single-pass gate measurably false-rejects (~0.8%/pass; 24/30 of the
# full-vault backfill rejects were overturned on re-read). A reject must be
# confirmed by a second judging pass; a split verdict fails OPEN (complete).


class _SeqGateLLM:
    """Fake LLM returning a SEQUENCE of replies (one per call), recording calls."""

    def __init__(self, *replies: str):
        self._replies = list(replies)
        self.calls = 0

    def call(self, msgs):
        self.calls += 1
        return self._replies.pop(0) if self._replies else '{"complete": true, "reason": "complete"}'


def test_confirmed_gate_pass_is_accepted_immediately():
    """A first-pass PASS is final — exactly ONE judging call (false-accept is
    not the measured failure mode; don't double every healthy paper's cost)."""
    llm = _SeqGateLLM('{"complete": true, "reason": "complete"}')
    out = extract.confirmed_completeness_gate("Intro... body... refs " * 50, llm=llm)
    assert out["complete"] is True
    assert llm.calls == 1


def test_confirmed_gate_flaky_reject_is_overturned():
    """reject → re-judge says complete ⇒ OVERTURNED to complete (the 24/30
    false-positive class survives), with the overruled reason kept for audit."""
    llm = _SeqGateLLM(
        '{"complete": false, "reason": "truncated: flaky"}',
        '{"complete": true, "reason": "complete"}',
    )
    out = extract.confirmed_completeness_gate("Intro... body... refs " * 50, llm=llm)
    assert out["complete"] is True
    assert "overturned" in out["reason"]
    assert "flaky" in out["reason"]                  # audit trail of pass 1
    assert llm.calls == 2


def test_confirmed_gate_double_reject_is_confirmed():
    """reject → re-judge ALSO rejects ⇒ confirmed reject, both reasons carried."""
    llm = _SeqGateLLM(
        '{"complete": false, "reason": "paywall stub"}',
        '{"complete": false, "reason": "no body sections"}',
    )
    out = extract.confirmed_completeness_gate("Title\nSign in to access", llm=llm)
    assert out["complete"] is False
    assert "paywall stub" in out["reason"] and "no body sections" in out["reason"]
    assert llm.calls == 2


def test_confirmed_gate_empty_text_still_hard_rejects():
    """The deterministic empty-text reject confirms at zero LLM cost."""
    llm = _SeqGateLLM()
    out = extract.confirmed_completeness_gate("   \n", llm=llm)
    assert out["complete"] is False
    assert llm.calls == 0                            # hard-coded branch, no LLM


def test_extract_md_flaky_gate_reject_no_longer_kills_the_paper(tmp_path, monkeypatch):
    """END-TO-END: a one-off flaky gate reject during extract_md is overturned
    by the confirmation pass → md SAVED, status ok (the pre-hardening behavior
    terminalized the paper on a single flaky verdict)."""
    lib, p = _pdf_paper(tmp_path)
    _stub_good_probe(monkeypatch)
    _stub_mineru(monkeypatch, returns="# body\nreal md")
    monkeypatch.setattr(extract, "review_extract",
                        lambda text, *, llm=None: {"ok": True, "issues": [],
                                                   "confidence": 1.0})
    verdicts = [{"complete": False, "reason": "flaky one-off"},
                {"complete": True, "reason": "complete"}]
    monkeypatch.setattr(
        extract, "completeness_gate",
        lambda text, *, llm=None: verdicts.pop(0))

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out == "# body\nreal md"
    assert lib.has_extract(p.key, "md")              # md SAVED
    assert p.download_status == "ok"
    assert not verdicts                              # both passes consumed


def test_confirmed_gate_failopen_second_pass_does_not_overturn():
    """Review must-fix: pass-1 REJECT + pass-2 LLM-error (which fail-opens to
    complete inside completeness_gate) must NOT overturn — a confirmation pass
    satisfiable by its own failure is not a confirmation. The reject stands."""

    class _RejectThenBoom:
        calls = 0

        def call(self, msgs):
            self.calls += 1
            if self.calls == 1:
                return '{"complete": false, "reason": "paywall stub (genuine)"}'
            raise RuntimeError("transient API outage")

    out = extract.confirmed_completeness_gate("Title\nSign in to access",
                                              llm=_RejectThenBoom())
    assert out["complete"] is False                   # reject STANDS
    assert "no_second_opinion" in out["reason"]
    assert "paywall stub (genuine)" in out["reason"]  # pass-1 reason kept
