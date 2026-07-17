"""Tests for services/resolver.py — fuzzy in-library lookup."""

from __future__ import annotations

import pytest

from papervault.library import Library
from papervault.library.services.resolver import ResolverService, _keyword_score


@pytest.fixture
def populated_lib(tmp_path):
    lib = Library(tmp_path)
    lib.upsert({"title": "Solar wind modulation review", "authors": ["Potgieter"],
                "year": 2013, "abstract": "Cosmic ray transport in the heliosphere.",
                "doi": "10.1/a", "is_review": True})
    lib.upsert({"title": "Deep learning for image classification", "authors": ["He"],
                "year": 2016, "abstract": "ResNet architecture", "doi": "10.1/b"})
    lib.upsert({"title": "Cosmic ray spectrum", "authors": ["Aslam"], "year": 2020,
                "abstract": "high energy spectrum measurements", "doi": "10.1/c"})
    return lib


# ----------- _keyword_score -------------------------------------------------


def test_keyword_score_perfect_overlap():
    paper = type("P", (), {"title": "cosmic ray transport", "venue": "",
                            "abstract": "", "authors": [], "year": 2020})
    s = _keyword_score("cosmic ray", paper())
    assert s == 1.0


def test_keyword_score_no_match():
    paper = type("P", (), {"title": "deep learning", "venue": "",
                            "abstract": "", "authors": [], "year": 2020})
    s = _keyword_score("cosmic ray transport", paper())
    assert s == 0.0


def test_keyword_score_empty_query_zero():
    paper = type("P", (), {"title": "anything", "venue": "",
                            "abstract": "", "authors": [], "year": 2020})
    assert _keyword_score("", paper()) == 0.0


# ----------- ResolverService.resolve ---------------------------------------


def test_resolve_empty_library_returns_empty(tmp_path):
    lib = Library(tmp_path)
    svc = ResolverService(lib)
    assert svc.resolve("anything") == []


def test_resolve_no_llm_returns_keyword_ranked(populated_lib):
    svc = ResolverService(populated_lib)
    out = svc.resolve("cosmic ray", top_k=5, use_llm=False)
    assert out
    titles = [r["title"] for r in out]
    # Both cosmic-ray papers should be there; deep learning should not.
    assert any("Cosmic" in t or "cosmic" in t for t in titles)
    assert not any("Deep learning" in t for t in titles)


def test_resolve_no_keyword_match_returns_empty(populated_lib):
    svc = ResolverService(populated_lib)
    out = svc.resolve("entirely unrelated topic xyzzy", use_llm=False)
    assert out == []


class _FakeLLM:
    def __init__(self, response: str):
        self._response = response

    def call(self, messages):
        return self._response


def test_resolve_llm_filters_by_min_confidence(populated_lib):
    # Pretend LLM returns two matches: one above threshold, one below.
    fake = _FakeLLM('{"matches": ['
                    '{"i": 1, "confidence": 0.9, "reason": "exact"},'
                    '{"i": 2, "confidence": 0.4, "reason": "weak"}]}')
    svc = ResolverService(populated_lib, llm=fake)
    out = svc.resolve("cosmic ray", min_confidence=0.7)
    assert len(out) == 1
    assert out[0]["score"] == pytest.approx(0.9)


def test_resolve_llm_failure_falls_back_to_keyword(populated_lib):
    class BoomLLM:
        def call(self, msgs):
            raise RuntimeError("api down")
    svc = ResolverService(populated_lib, llm=BoomLLM())
    out = svc.resolve("cosmic ray")
    # Got something via the keyword fallback.
    assert out
    assert all(r["in_library"] for r in out)


def test_resolve_llm_returns_invalid_json_falls_back(populated_lib):
    fake = _FakeLLM("not json at all")
    svc = ResolverService(populated_lib, llm=fake)
    out = svc.resolve("cosmic ray")
    # Falls back to keyword-scored results.
    assert out


def test_resolve_llm_returns_malformed_matches(populated_lib):
    fake = _FakeLLM('{"matches": [{"missing_keys": "x"}]}')
    svc = ResolverService(populated_lib, llm=fake)
    out = svc.resolve("cosmic ray")
    # Malformed match silently skipped → falls through with [], NOT keyword
    # fallback (the LLM did respond, just produced no usable matches).
    assert out == []


def test_resolve_top_k_caps_result_count(populated_lib):
    svc = ResolverService(populated_lib)
    # use_llm=False → straight keyword result, capped to top_k
    out = svc.resolve("cosmic", top_k=1, use_llm=False)
    assert len(out) == 1


# ----------- score provenance (degraded-path hardening, fix #7) -------------


def test_resolve_llm_path_tags_score_kind_llm(populated_lib):
    """An LLM-graded match carries score_kind='llm' so the server's auto-resolve
    gate uses the lenient 0.85 floor (not the stricter keyword floor)."""
    fake = _FakeLLM('{"matches": [{"i": 1, "confidence": 0.9, "reason": "x"}]}')
    svc = ResolverService(populated_lib, llm=fake)
    out = svc.resolve("cosmic ray", min_confidence=0.7)
    assert out
    assert all(c["score_kind"] == "llm" for c in out)


def test_resolve_keyword_fallback_tags_score_kind_keyword(populated_lib):
    """The degraded (LLM-down) keyword fallback tags candidates
    score_kind='keyword' so the server holds them to the stricter ≥0.95 floor —
    a token-saturated overlap of 1.0 must NOT pass the lenient 0.85 LLM floor."""
    class BoomLLM:
        def call(self, msgs):
            raise RuntimeError("api down")
    svc = ResolverService(populated_lib, llm=BoomLLM())
    out = svc.resolve("cosmic ray")
    assert out
    assert all(c["score_kind"] == "keyword" for c in out)


def test_resolve_no_llm_path_tags_score_kind_keyword(populated_lib):
    """use_llm=False is also a keyword-only path → score_kind='keyword'."""
    svc = ResolverService(populated_lib)
    out = svc.resolve("cosmic ray", use_llm=False)
    assert out
    assert all(c["score_kind"] == "keyword" for c in out)


def test_resolve_token_disjoint_target_reachable_via_llm(tmp_path):
    """fix #7b: the keyword pre-filter is no longer a hard >0 admission veto.
    A query whose distinctive tokens are absent from the alpha-token blob (here:
    a year/volume-only style query that scores 0 on overlap) must still reach the
    LLM, which recognises it from the full metadata. Before the fix prelim was
    empty → not_found; the LLM never saw the paper."""
    lib = Library(tmp_path)
    lib.upsert({"title": "Heliospheric modulation of galactic cosmic rays",
                "authors": ["Potgieter"], "year": 2013, "venue": "Living Reviews",
                "abstract": "transport theory", "doi": "10.1/z"})

    seen_candidate_count = {"n": 0}

    class RecognisingLLM:
        def call(self, msgs):
            # The user message embeds the candidate JSON; count how many the LLM
            # was offered, then confidently pick #1.
            user = msgs[-1]["content"]
            seen_candidate_count["n"] = user.count('"key"')
            return '{"matches": [{"i": 1, "confidence": 0.97, "reason": "match"}]}'

    svc = ResolverService(lib, llm=RecognisingLLM())
    # "2013" is digits-only (stripped by the alpha tokenizer) → keyword score 0.
    out = svc.resolve("2013")
    assert seen_candidate_count["n"] >= 1, "0-overlap target was NOT admitted to the LLM"
    assert out and out[0]["key"] == lib.all_papers()[0].key
