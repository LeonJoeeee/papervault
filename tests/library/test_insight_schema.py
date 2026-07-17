"""Unit tests for ``papervault.library.insight.schema`` and ``Paper.insight``
backward-compat with old index.json shapes.
"""

from __future__ import annotations

import json

import pytest

from papervault.library.insight.schema import Insight, InsightAnswers
from papervault.library.models import Paper


def test_insight_answers_default_empty_strings():
    """Default-construct with no args — all 5 fields empty strings."""
    a = InsightAnswers()
    assert a.q1_new_understanding == ""
    assert a.q2_reusable_methods == ""
    assert a.q3_assumptions_failures == ""
    assert a.q4_open_questions == ""
    assert a.q5_idea_seeds == ""


def test_insight_serialize_roundtrip():
    """Insight → model_dump → dict → re-validate yields equal object."""
    a = InsightAnswers(
        q1_new_understanding="finding 1",
        q2_reusable_methods="trick A",
        q3_assumptions_failures="assumption Z",
        q4_open_questions="follow-up Q",
        q5_idea_seeds="apply to task B",
    )
    src = Insight(
        version="v1",
        ingested_at="2026-05-16T14:23:01Z",
        model="openai/mimo-v2.5-pro",
        academic_knowledge_hash="sha256:abc123",
        answers=a,
    )
    raw = src.model_dump()
    rebuilt = Insight(**raw)
    assert rebuilt == src
    # Specifically test JSON round-trip too
    rebuilt2 = Insight(**json.loads(json.dumps(raw)))
    assert rebuilt2 == src


def test_paper_with_insight_serializes_inline(tmp_path):
    """Paper.model_dump() with insight set must yield nested dict, not a string."""
    p = Paper(key="X2026", title="t", authors=["A"], year=2026)
    p.insight = Insight(
        version="v1",
        ingested_at="2026-05-16T00:00:00Z",
        model="m",
        academic_knowledge_hash="sha256:0",
        answers=InsightAnswers(q1_new_understanding="ok"),
    )
    dumped = p.model_dump()
    assert isinstance(dumped["insight"], dict)
    assert dumped["insight"]["version"] == "v1"
    assert dumped["insight"]["answers"]["q1_new_understanding"] == "ok"


def test_paper_backward_compat_no_insight_key():
    """Old index.json shape (no `insight` key) loads with insight=None."""
    raw = {"key": "Y2020", "title": "t", "authors": ["B"], "year": 2020}
    p = Paper(**raw)
    assert p.insight is None
    # Dumping it back includes "insight": None (Pydantic emits explicit null)
    assert p.model_dump()["insight"] is None


def test_insight_version_field_required():
    """Insight without a version is a programming error (validation fails)."""
    with pytest.raises(Exception):
        Insight(  # type: ignore[call-arg]
            ingested_at="2026-05-16T00:00:00Z",
            model="m",
            academic_knowledge_hash="sha256:0",
            answers=InsightAnswers(),
        )
