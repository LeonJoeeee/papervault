"""Pydantic models for Paper.insight.

Output of the LLM ingest call is validated against ``InsightAnswers``; the
surrounding provenance fields (version / ingested_at / model /
academic_knowledge_hash) are filled by the worker.
"""

from __future__ import annotations

from pydantic import BaseModel


class InsightAnswers(BaseModel):
    """The five fixed answers, one per question. Each is a free-form
    string (1-3 entries × 2-3 sentences per spec §4.2). Defaults are
    empty strings so partial LLM output doesn't fail validation —
    the worker decides whether to accept or retry.

    **D20 schema migration (2026-05-18)**: question set was reframed
    knowledge-centric. New field names (q1_domain_addition /
    q2_method_addition / q3_boundary_warning / q4_ak_revision /
    q5_collision_idea) replace v1 names. v1 fields are retained for
    back-compat with the 800+ insights ingested under v1 — per D19
    they remain valid snapshots until the operator manually
    re-ingests them. v1 records load with v1 fields populated and v2
    fields empty; v2 records load with the reverse. Consumers should
    check ``Insight.version`` to know which set carries content."""

    # v2 fields (current question set, D20 onwards).
    q1_domain_addition: str = ""
    q2_method_addition: str = ""
    q3_boundary_warning: str = ""
    q4_ak_revision: str = ""
    q5_collision_idea: str = ""

    # v1 legacy fields — kept so the 800+ pre-D20 insights still load
    # cleanly. Do not write these from new ingest paths.
    q1_new_understanding: str = ""
    q2_reusable_methods: str = ""
    q3_assumptions_failures: str = ""
    q4_open_questions: str = ""
    q5_idea_seeds: str = ""


class Insight(BaseModel):
    """Outer wrapper stored on Paper.insight. Versioned so callers can
    deterministically skip already-ingested papers."""

    version: str
    """``QUESTION_SET_VERSION`` at ingest time. When the spec/questions
    are revised, bumping this constant triggers re-ingest of all papers
    whose stored version is older."""

    ingested_at: str
    """ISO 8601 UTC timestamp when this insight was successfully written."""

    model: str
    """LLM model identifier (e.g., ``openai/mimo-v2.5-pro``). Read from
    ``llm.model`` attribute at ingest time."""

    academic_knowledge_hash: str
    """SHA-256 hex of the academic-knowledge concat content at ingest
    time. Used for audit ("which papers were ingested under AK state X")."""

    answers: InsightAnswers
