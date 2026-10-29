"""Unit tests for the scored judges (``judge_ingest`` / ``judge_return``) and the
shared projection — V6 §7.

These exercise the judge in isolation with a fake LLM that captures the prompts it is
handed, so we can assert on:
- the SHARED ``_candidate_to_item`` projection (now carries ``citation_count`` +
  ``is_review`` for BOTH gates — §7);
- the V6 return rubric (``_RETURN_SYSTEM``): score each paper on its BEST-matching
  sub-part, 0.7-0.9 for fully answering ONE intended sub-topic, the 0.2-0.4
  "same field" guard band, soft prefs as gentle nudges never cuts;
- the return-judge USER prompt carrying the parser's ``search_terms`` as
  "intended sub-topics" + the soft prefs / year-window channel;
- backward-compatible public signature (the new params are keyword-only with defaults).
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from papervault.library.services import judge
from papervault.library.services.judge import (
    _RETURN_SYSTEM,
    _candidate_to_item,
    judge_ingest,
    judge_return,
)


@pytest.mark.parametrize("worker_fails", [False, True])
async def test_cancelled_judge_keeps_permit_until_thread_finishes(monkeypatch, worker_fails):
    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(judge.concurrency, "llm_sem", sem)
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    calls = []

    def call(messages):
        calls.append(messages)
        if len(calls) == 1:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5), "test failed to release LLM worker"
            if worker_fails:
                raise ValueError("cancelled worker failed")
        return "answer"

    llm = SimpleNamespace(call=call)
    first = asyncio.create_task(judge._call_with_retry("system", "first", llm))
    second = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert sem.locked(), "cancelled caller released a running worker's permit"
        second = asyncio.create_task(judge._call_with_retry("system", "second", llm))
        await asyncio.sleep(0)
        assert len(calls) == 1 and not second.done()
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        if second is not None:
            assert await asyncio.wait_for(second, 2) == "answer"
        await asyncio.wait_for(sem.acquire(), 2)
        sem.release()
    assert len(calls) == 2  # cancellation must never retry the abandoned call


async def test_cancelled_judge_waiter_does_not_start_worker_or_release_owned_permit(monkeypatch):
    sem = asyncio.Semaphore(0)
    monkeypatch.setattr(judge.concurrency, "llm_sem", sem)
    caller = asyncio.create_task(judge._call_with_retry("system", "user", None))
    await asyncio.sleep(0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert sem.locked()


class _CaptureLLM:
    """Fake LLM: records every (system, user) prompt pair, returns a canned reply.

    ``reply_for`` maps each captured user prompt to a JSON ``{"items":[...]}`` string;
    by default it scores/judges every ``i`` it can find in the user payload.
    """

    def __init__(self, mode: str):
        self.mode = mode          # "return" or "ingest"
        self.calls: list[dict] = []

    def call(self, messages):
        system = messages[0]["content"]
        user = messages[1]["content"]
        self.calls.append({"system": system, "user": user})
        # The user payload always ends with the JSON candidate list; pull the i's.
        items_json = user[user.index("[") : user.rindex("]") + 1]
        items = json.loads(items_json)
        if self.mode == "return":
            out = [{"i": it["i"], "reason": "ok", "score": 0.8} for it in items]
        else:
            out = [{"i": it["i"], "reason": "ok", "is_paper": True, "tier": "1A"}
                   for it in items]
        return json.dumps({"items": out})


_CANDS = [
    {"title": "Cosmic ray PINN transport", "venue": "ApJ", "year": 2024,
     "authors": ["A", "B"], "abstract": "physics-informed transport",
     "citation_count": 42, "is_review": True},
    {"title": "Solar wind modulation review", "venue": "JGR", "year": 2013,
     "authors": ["C"], "abstract": "review", "citation_count": None},
]


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False), ("false", False), ("true", False),
    ("0", False), ("no", False), (0, False), (1, False), (None, False),
])
def test_ingest_paper_backstop_requires_a_real_boolean(value, expected):
    raw = json.dumps({"items": [{"i": 0, "tier": "1A", "is_paper": value}]})
    assert judge._parse_ingest(raw, {0})[0]["llm_is_paper"] is expected


def test_ingest_missing_paper_backstop_cannot_admit_a_record():
    assert judge._parse_ingest('{"items":[{"i":0,"tier":"1A"}]}', {0})[0]["llm_is_paper"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity"])
def test_return_nonfinite_relevance_scores_default_to_zero(value):
    raw = json.dumps({"items": [{"i": 0, "score": value}]})
    assert judge._parse_score(raw, {0})[0]["score"] == 0.0


# ───────────────────────── shared projection ─────────────────────────

def test_candidate_to_item_carries_citation_and_review():
    """§7: the shared projection now ALSO emits citation_count + is_review so the
    return rubric's soft prefs reference fields the LLM can actually see."""
    item = _candidate_to_item(3, _CANDS[0])
    assert item["i"] == 3
    assert item["title"] == "Cosmic ray PINN transport"
    assert item["authors"] == ["A", "B"]
    assert item["citation_count"] == 42
    assert item["is_review"] is True


def test_candidate_to_item_coerces_missing_soft_fields():
    """citation_count None/absent → 0 (int); is_review absent → False (bool)."""
    item = _candidate_to_item(0, {"title": "t"})
    assert item["citation_count"] == 0 and isinstance(item["citation_count"], int)
    assert item["is_review"] is False
    # explicit None citation_count also coerces to 0
    assert _candidate_to_item(1, {"citation_count": None})["citation_count"] == 0


@pytest.mark.asyncio
async def test_ingest_projection_includes_soft_fields_unsplit():
    """The projection is SHARED and unsplit: the ingest payload ALSO carries
    citation_count + is_review even though the ingest prompt ignores them (§7)."""
    llm = _CaptureLLM("ingest")
    judged, dropped = await judge_ingest(_CANDS, llm=llm)
    assert dropped == 0
    payload = llm.calls[0]["user"]
    sent = json.loads(payload[payload.index("[") : payload.rindex("]") + 1])
    assert all("citation_count" in it and "is_review" in it for it in sent)


# ───────────────────────── return rubric ─────────────────────────

def test_return_rubric_scores_best_sub_part():
    """The V6 rubric scores each paper on its BEST-matching sub-part, not whole-query
    coverage; a focused paper nailing ONE intended sub-topic lands 0.7-0.9."""
    r = _RETURN_SYSTEM
    assert "BEST-MATCHING SUB-PART" in r
    assert "0.7-0.9 : fully answers >=1 intended sub-topic" in r
    assert "Only a paper answering the WHOLE intent reaches 0.9+." in r
    assert "Do NOT penalize a paper" in r and "for being focused" in r


def test_return_rubric_has_same_field_guard_band():
    """The 0.2-0.4 guard band is the precision floor that stops this becoming an
    OR-gate: same field, no sub-part answered → 0.2-0.4."""
    r = _RETURN_SYSTEM
    assert "0.2-0.4 : same field, addresses none of the sub-parts" in r
    assert "do NOT\nreward mere field membership" in r


def test_return_rubric_soft_prefs_are_nudges_never_cuts():
    """Soft prefs (citation/review/recency) are gentle nudges that only break
    near-ties — NEVER a cut; unknown-year is recency-unverifiable, still never cut."""
    r = _RETURN_SYSTEM
    assert "SOFT PREFERENCES (gentle nudges, NEVER a cut)" in r
    assert "prefs only break near-ties" in r
    assert "still NEVER cut it" in r


def test_return_rubric_forbids_renumbering():
    """Prompt-side belt against the index-rebasing silent-drop mode (§6)."""
    assert 'do NOT renumber' in _RETURN_SYSTEM


# ───────────────────────── return user prompt (sub-topics + soft prefs) ─────────────────────────

@pytest.mark.asyncio
async def test_return_user_prompt_carries_search_terms_as_subtopics():
    """The parser's search_terms reach the return judge as 'intended sub-topics' (§7)."""
    llm = _CaptureLLM("return")
    await judge_return(
        _CANDS, "find cosmic ray PINN transport work",
        search_terms=["cosmic ray transport", "physics-informed neural net"],
        filters={}, llm=llm)
    user = llm.calls[0]["user"]
    assert "User query intent: find cosmic ray PINN transport work" in user
    assert ("The intended sub-topics include: "
            "cosmic ray transport, physics-informed neural net") in user


@pytest.mark.asyncio
async def test_return_user_prompt_formats_year_window_as_nl():
    """The year window is the hand-built NL channel: 'from 2016 to any', not a raw
    [2016, None] token; soft prefs ride raw (presence/direction only)."""
    llm = _CaptureLLM("return")
    await judge_return(
        _CANDS, "q", search_terms=["t"],
        filters={"year_min": 2016, "year_max": None,
                 "citation_pref": "prefer", "review_pref": "off"}, llm=llm)
    user = llm.calls[0]["user"]
    assert "year_window=from 2016 to any" in user
    assert "citation_pref=prefer" in user and "review_pref=off" in user


@pytest.mark.asyncio
async def test_return_user_prompt_no_window_when_unset():
    """No year bound → year_window=none."""
    llm = _CaptureLLM("return")
    await judge_return(_CANDS, "q", search_terms=["t"], filters={}, llm=llm)
    assert "year_window=none" in llm.calls[0]["user"]


@pytest.mark.asyncio
async def test_return_public_signature_backward_compatible():
    """New params are keyword-only with safe defaults — the older
    judge_return(candidates, intent, *, llm=) call still works (no NameError, no
    aspects arg). Scores are clamped + returned per global index."""
    llm = _CaptureLLM("return")
    judged, dropped = await judge_return(_CANDS, "q", llm=llm)
    assert dropped == 0
    assert set(judged) == {0, 1}
    assert all(0.0 <= j["score"] <= 1.0 for j in judged.values())
    # empty search_terms → the sub-topics line renders empty, no crash
    assert "The intended sub-topics include: \n" in llm.calls[0]["user"]


# ───────────────────────── judge_batches_dropped counter (§8) ─────────────────────────


class _RebasingLLM:
    """Fake LLM that IGNORES the provided global ``i`` and renumbers its output
    ``0..k-1`` per batch (the index-rebasing silent-drop mode). Batch-0's response
    happens to match expected {0..k-1}; any LATER batch (global i>=BATCH_SIZE)
    parses OK but matches ZERO expected indices → DROPPED + WARN + counter."""

    def __init__(self, mode: str):
        self.mode = mode
        self.calls: int = 0

    def call(self, messages):
        self.calls += 1
        user = messages[1]["content"]
        items = json.loads(user[user.index("[") : user.rindex("]") + 1])
        if self.mode == "return":
            out = [{"i": local, "reason": "ok", "score": 0.8}
                   for local, _ in enumerate(items)]
        else:
            out = [{"i": local, "reason": "ok", "is_paper": True, "tier": "1A"}
                   for local, _ in enumerate(items)]
        return json.dumps({"items": out})


def _many_cands(n: int) -> list[dict]:
    return [{"title": f"Paper number {i} on cosmic ray transport", "venue": "ApJ",
             "year": 2024, "authors": ["A"], "abstract": "x", "citation_count": 0,
             "is_review": False} for i in range(n)]


@pytest.mark.asyncio
async def test_judge_return_counts_parse_ok_zero_matched_drop(caplog, monkeypatch):
    """§8: a batch that parses OK but matches ZERO expected global indices (index
    rebasing) across all retries is DROPPED, WARN-logged (not ERROR-logged), and
    counted in judge_batches_dropped. With >BATCH_SIZE cands, batch-0 matches
    (rebased 0..k-1 == expected) while batch-1 (global i>=30) matches nothing →
    exhausts MAX_RETRIES + 1 attempts and drops 1 batch."""
    import logging
    monkeypatch.setattr(judge, "_BACKOFF_BASE", 0.0)
    cands = _many_cands(judge.BATCH_SIZE + 5)   # 2 batches
    llm = _RebasingLLM("return")
    with caplog.at_level(logging.WARNING, logger=judge.__name__):
        judged, dropped = await judge_return(cands, "q", llm=llm)
    # Batch-0 (1 call) matched; batch-1 retried across MAX_RETRIES + 1 calls then dropped.
    assert llm.calls == 1 + (judge.MAX_RETRIES + 1)
    assert dropped == 1
    assert set(judged) == set(range(judge.BATCH_SIZE))   # only batch-0 survives
    assert any(r.levelno == logging.WARNING and "0 items matched expected" in r.message
               for r in caplog.records)
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


@pytest.mark.asyncio
async def test_judge_ingest_counts_parse_ok_zero_matched_drop(monkeypatch):
    """§8: the ingest gate threads the same dropped-batch counter out after retries."""
    monkeypatch.setattr(judge, "_BACKOFF_BASE", 0.0)
    cands = _many_cands(judge.BATCH_SIZE + 5)
    llm = _RebasingLLM("ingest")
    judged, dropped = await judge_ingest(cands, llm=llm)
    assert llm.calls == 1 + (judge.MAX_RETRIES + 1)
    assert dropped == 1
    assert set(judged) == set(range(judge.BATCH_SIZE))


@pytest.mark.asyncio
async def test_judge_no_drop_when_all_batches_match():
    """Canary: a well-behaved LLM (keeps global i) drops nothing."""
    cands = _many_cands(judge.BATCH_SIZE + 5)
    _judged, dropped = await judge_return(cands, "q", llm=_CaptureLLM("return"))
    assert dropped == 0


class _UnparseableLLM:
    """Fake LLM returning an UNPARSEABLE reply — a refusal (no JSON object) or
    malformed JSON. The fail-closed gate must DROP the batch (counted), NEVER let
    the exception escape and crash search_papers."""

    def __init__(self, payload: str):
        self._payload = payload
        self.calls: int = 0

    def call(self, messages):
        self.calls += 1
        return self._payload


@pytest.mark.asyncio
async def test_judge_drops_unparseable_response_never_crashes(monkeypatch):
    """R1 regression: a judge LLM reply with no JSON object (a refusal) OR malformed
    JSON inside braces must exhaust MAX_RETRIES + 1 attempts and then be a conservative
    DROP (fail-CLOSED), not an exception out of judge_ingest/judge_return."""
    monkeypatch.setattr(judge, "_BACKOFF_BASE", 0.0)
    for bad in ("I cannot comply with this request.", "{this is not: valid json}"):
        llm_i = _UnparseableLLM(bad)
        judged_i, dropped_i = await judge_ingest(_CANDS, llm=llm_i)
        assert judged_i == {} and dropped_i == 1
        assert llm_i.calls == judge.MAX_RETRIES + 1
        llm_r = _UnparseableLLM(bad)
        judged_r, dropped_r = await judge_return(_CANDS, "q", llm=llm_r)
        assert judged_r == {} and dropped_r == 1
        assert llm_r.calls == judge.MAX_RETRIES + 1


# ── per-item JSON salvage (one bad item must not void the whole 30-batch) ──────────
def test_extract_json_obj_fast_path_unchanged():
    """A well-formed envelope parses identically to before (success path untouched)."""
    raw = '{"items": [{"i": 0, "score": 0.5}, {"i": 1, "score": 0.6}]}'
    assert len(judge._extract_json_obj(raw)["items"]) == 2


def test_extract_json_obj_salvages_one_bad_item():
    """Envelope json.loads fails on ONE malformed middle item; the good items (each
    carrying ``"i"``) are recovered and only the broken one is dropped — so the
    fail-CLOSED 'one bad item voids all 30' mode is gone."""
    raw = '{"items": [{"i": 0, "score": 0.9}, {"i": 1, "score": }, {"i": 2, "score": 0.7}]}'
    got = sorted(it["i"] for it in judge._extract_json_obj(raw)["items"])
    assert got == [0, 2]


def test_extract_json_obj_strips_markdown_fence():
    """A ```json ... ``` fenced reply still parses."""
    raw = '```json\n{"items": [{"i": 4, "score": 0.8}]}\n```'
    assert judge._extract_json_obj(raw)["items"][0]["i"] == 4


def test_extract_json_obj_no_json_still_raises():
    """A refusal (no JSON object at all) still raises -> caller DROPs (fail-closed)."""
    with pytest.raises(ValueError):
        judge._extract_json_obj("I cannot comply.")


def test_extract_json_obj_nothing_salvageable_raises():
    """Malformed AND no recoverable item (no ``"i"``) -> raises -> caller DROPs."""
    with pytest.raises(Exception):
        judge._extract_json_obj("{this is not: valid json}")


# ── retry of transient malformed or zero-matched judge replies (#348) ──────────


@pytest.mark.parametrize("gate", ["ingest", "return"])
@pytest.mark.parametrize("bad_attempts", [1, 2])
async def test_judge_recovers_from_transient_malformed_json(monkeypatch, gate, bad_attempts):
    monkeypatch.setattr(judge, "_BACKOFF_BASE", 0.0)
    calls = 0

    def call(messages):
        nonlocal calls
        calls += 1
        if calls <= bad_attempts:
            return '{items: [{i: 0, reason: "unquoted property name", score: 0.9}]}'
        return _CaptureLLM(gate).call(messages)

    llm = SimpleNamespace(call=call)
    if gate == "ingest":
        judged, dropped = await judge_ingest(_CANDS, llm=llm)
    else:
        judged, dropped = await judge_return(_CANDS, "q", llm=llm)
    assert dropped == 0
    assert set(judged) == {0, 1}
    assert calls == bad_attempts + 1


@pytest.mark.parametrize("gate", ["ingest", "return"])
@pytest.mark.parametrize("bad_attempts", [1, 2])
async def test_judge_recovers_from_transient_zero_matched_reply(monkeypatch, gate, bad_attempts):
    monkeypatch.setattr(judge, "_BACKOFF_BASE", 0.0)
    cands = _many_cands(judge.BATCH_SIZE + 2)
    batch1_calls = 0

    def call(messages):
        nonlocal batch1_calls
        user = messages[1]["content"]
        items = json.loads(user[user.index("[") : user.rindex("]") + 1])
        if items[0]["i"] >= judge.BATCH_SIZE:
            batch1_calls += 1
            if batch1_calls <= bad_attempts:
                return _RebasingLLM(gate).call(messages)
        return _CaptureLLM(gate).call(messages)

    llm = SimpleNamespace(call=call)
    if gate == "ingest":
        judged, dropped = await judge_ingest(cands, llm=llm)
    else:
        judged, dropped = await judge_return(cands, "q", llm=llm)
    assert dropped == 0
    assert set(judged) == set(range(judge.BATCH_SIZE + 2))
    assert batch1_calls == bad_attempts + 1


async def test_judge_retry_backoff_semaphore_and_3arg_compat(monkeypatch):
    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(judge.concurrency, "llm_sem", sem)
    # Positional 3-argument _call_with_retry remains backward-compatible.
    raw = await judge._call_with_retry("sys", "usr", SimpleNamespace(call=lambda msgs: "plain-raw"))
    assert raw == "plain-raw"

    sleeps: list[tuple[float, bool]] = []
    sem_locked_in_worker: list[bool] = []

    async def fake_sleep(delay: float):
        # Permit must be released BEFORE sleeping between retry attempts.
        sleeps.append((delay, sem.locked()))

    monkeypatch.setattr(judge.asyncio, "sleep", fake_sleep)
    seq = iter([
        "{malformed: json}",
        json.dumps({"items": [{"i": 999, "reason": "wrong index", "score": 0.9}]}),
        json.dumps({"items": [{"i": 0, "reason": "ok", "score": 0.9},
                              {"i": 1, "reason": "ok", "score": 0.7}]}),
    ])

    def call(messages):
        sem_locked_in_worker.append(sem.locked())
        return next(seq)

    judged, dropped = await judge_return(_CANDS, "q", llm=SimpleNamespace(call=call))
    assert dropped == 0
    assert set(judged) == {0, 1}
    assert sem_locked_in_worker == [True, True, True]
    assert sleeps == [(2.0, False), (4.0, False)]


async def test_judge_final_attempt_determines_warning_vs_error_log(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(judge, "_BACKOFF_BASE", 0.0)

    # Ends on parse-OK zero-matched -> WARNING, never ERROR.
    seq_warn = iter([
        "{malformed: json}",
        RuntimeError("transient transport error"),
        json.dumps({"items": [{"i": 999, "reason": "rebased", "score": 0.8}]}),
    ])

    def call_warn(messages):
        item = next(seq_warn)
        if isinstance(item, Exception):
            raise item
        return item

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=judge.__name__):
        judged, dropped = await judge_return(_CANDS, "q", llm=SimpleNamespace(call=call_warn))
    assert judged == {} and dropped == 1
    assert any(r.levelno == logging.WARNING and "0 items matched expected idx" in r.message
               for r in caplog.records)
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)

    # Ends on malformed JSON after earlier zero-matched replies -> ERROR, not zero-matched WARNING.
    seq_err = iter([
        json.dumps({"items": [{"i": 999, "reason": "rebased", "score": 0.8}]}),
        json.dumps({"items": [{"i": 999, "reason": "rebased", "score": 0.8}]}),
        "{malformed: json}",
    ])
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=judge.__name__):
        judged, dropped = await judge_return(
            _CANDS, "q", llm=SimpleNamespace(call=lambda msgs: next(seq_err)),
        )
    assert judged == {} and dropped == 1
    assert any(r.levelno == logging.ERROR and "dropping batch" in r.message
               for r in caplog.records)
    assert not any("0 items matched expected idx" in r.message for r in caplog.records)


async def test_cancelled_judge_during_parse_retry_backoff_does_not_continue(monkeypatch):
    in_sleep = asyncio.Event()
    calls = 0

    async def slow_sleep(_delay: float):
        in_sleep.set()
        await asyncio.Future()

    monkeypatch.setattr(judge.asyncio, "sleep", slow_sleep)

    def call(messages):
        nonlocal calls
        calls += 1
        return "{malformed: json}"

    task = asyncio.create_task(judge_return(_CANDS, "q", llm=SimpleNamespace(call=call)))
    await asyncio.wait_for(in_sleep.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == 1
