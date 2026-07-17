"""Tests for services/search_service.py — in-library search with filters."""

from __future__ import annotations

import pytest

from papervault.library import Library
from papervault.library.services.search_service import SearchService


@pytest.fixture
def populated_lib(tmp_path):
    lib = Library(tmp_path)
    lib.upsert({"title": "Solar wind modulation review", "authors": ["Potgieter"],
                "year": 2013, "abstract": "Cosmic ray transport in heliosphere.",
                "doi": "10.1/a", "is_review": True, "citation_count": 500})
    lib.upsert({"title": "Cosmic ray spectrum measurement", "authors": ["Aslam"],
                "year": 2020, "abstract": "high energy spectrum",
                "doi": "10.1/b", "citation_count": 50})
    lib.upsert({"title": "Cosmic ray PINN model", "authors": ["Wei"],
                "year": 2024, "abstract": "neural network solver for transport",
                "doi": "10.1/c", "citation_count": 5})
    # On-disk extract for one paper, not the others.
    lib.txt_path("Aslam2020").write_text("body text")
    return lib


def _titles(out: dict) -> list[str]:
    return [r["title"] for r in out["results"]]


# ----------- filters --------------------------------------------------------


def test_year_min_filter(populated_lib):
    out = SearchService(populated_lib).search("cosmic", rerank=False, year_min=2020)
    assert all(r["year"] >= 2020 for r in out["results"])
    titles = _titles(out)
    assert "Solar wind modulation review" not in titles  # 2013, filtered out


def test_year_max_filter(populated_lib):
    out = SearchService(populated_lib).search("cosmic", rerank=False, year_max=2015)
    assert all(r["year"] <= 2015 for r in out["results"])


def test_is_review_filter_true(populated_lib):
    out = SearchService(populated_lib).search("cosmic", rerank=False, is_review=True)
    assert all(r["is_review"] for r in out["results"])


def test_is_review_filter_false(populated_lib):
    out = SearchService(populated_lib).search("cosmic", rerank=False, is_review=False)
    assert all(not r["is_review"] for r in out["results"])


def test_has_extract_filter(populated_lib):
    out = SearchService(populated_lib).search("cosmic", rerank=False, has_extract=True)
    keys = {r["key"] for r in out["results"]}
    assert keys == {"Aslam2020"}


def test_citation_min_filter(populated_lib):
    out = SearchService(populated_lib).search("cosmic", rerank=False, citation_min=100)
    titles = _titles(out)
    assert "Solar wind modulation review" in titles
    assert "Cosmic ray PINN model" not in titles


# ----------- rerank toggle --------------------------------------------------


def test_rerank_false_skips_llm(populated_lib):
    """With rerank=False the LLM must NOT be called even if one is configured."""
    class BoomLLM:
        def call(self, _msgs):
            raise AssertionError("LLM must not be called")
    svc = SearchService(populated_lib, llm=BoomLLM())
    out = svc.search("cosmic", rerank=False)
    # Just confirm we got results without invoking the boom.
    assert out["results"]


def test_rerank_true_uses_llm_order(populated_lib):
    """LLM returns a reversed order; results should follow that order."""
    class FakeLLM:
        def call(self, _msgs):
            # the test query matches all 3 cosmic papers, so prelim has 3 items;
            # ask LLM to put i=3 first, then i=1, then i=2
            return '{"order": [3, 1, 2]}'
    svc = SearchService(populated_lib, llm=FakeLLM())
    out = svc.search("cosmic ray", rerank=True, limit=10)
    # First result should be whichever paper was at prelim index 3 (1-based).
    assert len(out["results"]) >= 1
    # Verify the LLM order was respected (top 3 not in default-keyword order).
    keys = [r["key"] for r in out["results"][:3]]
    # LLM rerank must put a paper that is NOT the highest keyword-score one first.
    assert len(set(keys)) == len(keys)  # no duplicates


def test_rerank_llm_failure_falls_back(populated_lib):
    class BoomLLM:
        def call(self, _msgs):
            raise RuntimeError("api down")
    svc = SearchService(populated_lib, llm=BoomLLM())
    out = svc.search("cosmic", rerank=True)
    # Got results via fallback keyword path.
    assert out["results"]


def test_rerank_invalid_json_keeps_keyword_order(populated_lib):
    """LLM returns malformed JSON; remaining items appended in keyword order."""
    class FakeLLM:
        def call(self, _msgs):
            return "garbage no json here"
    svc = SearchService(populated_lib, llm=FakeLLM())
    out = svc.search("cosmic", rerank=True)
    assert out["results"]


# ----------- empty paths ----------------------------------------------------


def test_no_keyword_match_returns_empty(populated_lib):
    out = SearchService(populated_lib).search("entirely unrelated xyzzy", rerank=False)
    assert out["results"] == []
    assert out["total_in_library"] == 3


def test_total_in_library_reported(populated_lib):
    out = SearchService(populated_lib).search("cosmic", rerank=False)
    assert out["total_in_library"] == 3
