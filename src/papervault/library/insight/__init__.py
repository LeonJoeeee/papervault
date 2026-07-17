"""Paper-insight layer — Pydantic schema for the deprecated 5-Q LLM
digest. Kept **read-only** after route B (Phase 28, 2026-05-24).

Background: through Phase 24 this package owned a write-side pipeline
(`worker.py` + `services/insight_queue.py` + `academic_knowledge.py` +
`prompts.py`) that ran a MiMo 5-question digest on every freshly
extracted paper and stored the result inline on ``Paper.insight``. In
Phase 28 (Knowledge System v2, route B) the entire write-side pipeline
moved to the research-side `librarian/` curators; paper-library was
demoted to a mechanical fetch/extract/MCP service with no LLM.

What remains here:

- ``schema.py`` — Pydantic models (`Insight` / `InsightAnswers`) so the
  ~800 legacy ``Paper.insight`` records on disk still deserialize when
  the library is loaded. Read-only — no new ingest path writes these.
- The original 5-Q prompts have been archived as a markdown reference at
  ``librarian/legacy-references/insight-prompts-pre-routeB.md``.

Import discipline: only the schema types are exported. Anything that
used to import ``worker.ingest_insight`` / ``worker.should_skip_insight``
/ ``prompts.QUESTION_SET_VERSION`` / ``academic_knowledge.*`` /
``dead_letter.*`` has been deleted along with those modules. If you see
such an import in a downstream consumer, that consumer is stale.
"""

from .schema import Insight, InsightAnswers

__all__ = [
    "Insight",
    "InsightAnswers",
]
