"""Lightweight paper-library client.

Reads from paper-library's on-disk data store directly (index.json + extracts/).
We bypass the MCP HTTP layer for batch operations (massive throughput diff:
disk read ~ms vs MCP HTTP round-trip ~100ms). Ad-hoc single-paper lookups go
through paper-library's own MCP server (used by Executor/Reviewer LLMs, not by KS).

The knowledge plane depends on the library plane ONLY through this on-disk contract:
the vault path (papervault.config.VAULT_PATH — the SAME source the library plane
writes through, so the co-hosted reader and writer always agree) and the index.json
shape (version-checked on load, so a producer-side format change fails loud here
instead of silently yielding empty records).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger("ks.ingest.paper_library_client")

# Resolve the vault from the SAME source the library plane writes through
# (papervault.config.VAULT_PATH: PAPERVAULT_VAULT -> legacy PAPER_LIBRARY_PATH ->
# PAPERVAULT_DATA/vault, expanduser'd). In the unified single process the library
# writer and this ingest reader MUST agree on the directory, or the scheduler reads
# an empty/absent vault and the knowledge graph never populates.
from papervault import config as _pv_config

DEFAULT_VAULT = _pv_config.VAULT_PATH

# paper-library writes index.json as {"version": 1, "papers": {...}}.
_EXPECTED_INDEX_VERSION = 1


@dataclass
class PaperRecord:
    """Subset of paper-library Paper model fields needed by KS."""

    key: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    abstract: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    citation_count: int = 0
    is_review: bool = False
    publication_types: list[str] = field(default_factory=list)
    # Paths (relative to vault root)
    pdf_path: Optional[str] = None
    txt_path: Optional[str] = None
    md_path: Optional[str] = None
    # Distillation state
    distilled_at: Optional[str] = None
    insight: Optional[dict[str, Any]] = None  # raw dict; insight_v1 or v2 schema
    insight_invalid_reason: Optional[str] = None
    # New fields (Phase 27)
    source_type: str = "research_paper"  # research_paper / textbook / review
    # Domain membership (paper-library Phase 32). None = in-domain. A non-None
    # status (off_domain / non_paper / bad_extract) means the paper is quarantined
    # at the source; load_vault_index() excludes these by default so the KS graph
    # build sees the library's clean view (no KS-side domain classifier needed).
    domain_status: Optional[str] = None
    # paper-library download_status (#144): "metadata_only" = no PDF could be found — the class
    # that gets an abstract-only graph doc. Empty when the index entry predates the field.
    download_status: str = ""

    @classmethod
    def from_index_entry(cls, entry: dict[str, Any]) -> "PaperRecord":
        return cls(
            key=entry["key"],
            title=entry.get("title", ""),
            authors=entry.get("authors", []) or [],
            year=entry.get("year"),
            venue=entry.get("venue", ""),
            abstract=entry.get("abstract", ""),
            doi=entry.get("doi", ""),
            arxiv_id=entry.get("arxiv_id", ""),
            url=entry.get("url", ""),
            citation_count=entry.get("citation_count", 0),
            is_review=entry.get("is_review", False),
            publication_types=entry.get("publication_types", []) or [],
            pdf_path=entry.get("pdf_path"),
            txt_path=entry.get("txt_path"),
            md_path=entry.get("md_path"),
            distilled_at=entry.get("distilled_at"),
            insight=entry.get("insight"),
            insight_invalid_reason=entry.get("insight_invalid_reason"),
            source_type=entry.get("source_type", "research_paper"),
            domain_status=entry.get("domain_status"),
            download_status=entry.get("download_status") or "",
        )

    @property
    def has_insight(self) -> bool:
        if not self.insight:
            return False
        answers = self.insight.get("answers") or {}
        # Either v1 or v2 fields must have non-empty content
        v1_keys = (
            "q1_new_understanding",
            "q2_reusable_methods",
            "q3_assumptions_failures",
            "q4_open_questions",
            "q5_idea_seeds",
        )
        v2_keys = (
            "q1_domain_addition",
            "q2_method_addition",
            "q3_boundary_warning",
            "q4_ak_revision",
            "q5_collision_idea",
        )
        has_v1 = any(answers.get(k) for k in v1_keys)
        has_v2 = any(answers.get(k) for k in v2_keys)
        return has_v1 or has_v2

    @property
    def insight_version(self) -> Optional[str]:
        """Return 'v1' or 'v2' based on which Q fields are populated."""
        if not self.insight:
            return None
        version = self.insight.get("version")
        if version in ("v1", "v2"):
            return version
        # Fall back to content detection
        answers = self.insight.get("answers") or {}
        if any(answers.get(k) for k in ("q1_domain_addition", "q2_method_addition")):
            return "v2"
        if any(answers.get(k) for k in ("q1_new_understanding", "q2_reusable_methods")):
            return "v1"
        return None


def load_vault_index(
    vault_path: Path = DEFAULT_VAULT, *, include_quarantined: bool = False
) -> dict[str, PaperRecord]:
    """Load papers from {vault}/index.json.

    By DEFAULT returns the library's clean view — papers domain-quarantined at the
    source (``domain_status`` set: off_domain / non_paper / bad_extract, per
    paper-library Phase 32) are excluded, so the KS graph build never ingests
    off-domain contamination and no KS-side domain classifier is needed. Pass
    ``include_quarantined=True`` for audits.

    Version-checks the index so a paper-library export-format change is loud (a
    warning) rather than silently mapping every field to its default.
    """
    index_file = vault_path / "index.json"
    with open(index_file) as f:
        index = json.load(f)
    version = index.get("version")
    if version != _EXPECTED_INDEX_VERSION:
        logger.warning(
            "paper-vault index.json version=%r, expected %r — paper-library may "
            "have changed its export format; KS field mapping (from_index_entry) "
            "may be stale.",
            version,
            _EXPECTED_INDEX_VERSION,
        )
    papers = index.get("papers", {})
    recs = {key: PaperRecord.from_index_entry(entry) for key, entry in papers.items()}
    if include_quarantined:
        return recs
    return {k: r for k, r in recs.items() if r.domain_status is None}


def iter_distilled(vault_path: Path = DEFAULT_VAULT) -> Iterator[PaperRecord]:
    """Iterate papers that have a valid insight."""
    for paper in load_vault_index(vault_path).values():
        if paper.has_insight and not paper.insight_invalid_reason:
            yield paper


def iter_undistilled(vault_path: Path = DEFAULT_VAULT) -> Iterator[PaperRecord]:
    """Iterate papers that have no valid insight yet."""
    for paper in load_vault_index(vault_path).values():
        if not paper.has_insight and not paper.insight_invalid_reason:
            yield paper


_REF_HEADING = re.compile(r"(?im)^\s{0,3}#{0,4}\s*(references|bibliography)\b")


def _strip_references(text: str) -> str:
    """Cut the bibliography/references section before graph ingest — it is the single
    biggest source of junk graph entities (cited author surnames, journal abbreviations)
    and adds nothing to 'what is known'. Only cuts a references heading in the latter half
    (avoids an early in-text mention of the word)."""
    if not text:
        return text
    for m in _REF_HEADING.finditer(text):
        if m.start() > len(text) * 0.5:
            return text[: m.start()].rstrip()
    return text


def read_paper_text(paper: PaperRecord, vault_path: Path = DEFAULT_VAULT) -> Optional[str]:
    """Read paper's extracted text from disk (markdown preferred), references section
    stripped (junk-entity source for the knowledge-graph build)."""
    for rel in (paper.md_path, paper.txt_path):
        if rel:
            full = vault_path / rel
            if full.exists():
                return _strip_references(full.read_text(encoding="utf-8", errors="replace"))
    return None
