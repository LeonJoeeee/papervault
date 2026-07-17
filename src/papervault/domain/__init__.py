"""Domain pack loader (ADR-0003) — the space-physics-specific layer as editable config.

The knowledge-graph ontology, the LightRAG few-shot extraction examples, the extraction
exclusion/canonicalization rules, and the ingest domain-gate rubric were all hardcoded in
Python. This package externalizes them into a *domain pack* — a directory of plain config
files — so swapping the research domain needs NO code edits, only a different pack.

A pack is a directory containing:
  - ``domain.toml``               — ``label`` (str): the domain description string.
  - ``ontology.json``             — JSON list of entity-type strings (the KG ontology).
  - ``extraction_examples.txt``   — LightRAG few-shot examples, separated by a line that is
                                    exactly ``===EXAMPLE===``.
  - ``extraction_exclusions.txt`` — the extraction exclusion / canonicalization rules block.
  - ``search_gate.md``            — the ingest domain-gate rubric (markdown).

Resolution (``get_domain()``):
  - ``PAPERVAULT_DOMAIN_PATH`` — if set, an external directory the operator points at.
  - else the bundled default ``<this dir>/<PAPERVAULT_DOMAIN>`` (``PAPERVAULT_DOMAIN``
    defaults to ``space_physics``, the factory pack).

The constructed pack is cached in a module global, so repeated ``get_domain()`` calls do
not re-read disk. ``space_physics`` is the factory default and reproduces the previously
hardcoded values byte-for-byte.
"""
from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

_EXAMPLE_SEP_RE = re.compile(r"(?m)^===EXAMPLE===$")


@dataclass(frozen=True)
class DomainPack:
    """The active research-domain configuration, loaded from a pack directory."""

    label: str
    """Domain description string (e.g. ``"space physics + AI4Science"``)."""
    entity_types: list[str]
    """Knowledge-graph entity ontology (the entity-type names)."""
    extraction_examples: list[str]
    """LightRAG few-shot extraction example strings."""
    extraction_exclusions: str
    """Extraction exclusion / canonicalization rules block."""
    search_gate: str
    """Ingest domain-gate rubric (markdown text)."""

    @property
    def entity_types_guidance(self) -> str:
        """The closed-ontology guidance string for LightRAG >= 1.5.

        1.5.x replaced ``addon_params['entity_types']`` (a list) with
        ``addon_params['entity_types_guidance']`` (one free-text string rendered into the
        extraction system prompt's ``---Entity Types---`` section). This renders the pack's
        ontology as a CLOSED list plus the negative-exclusion rules, so the whole domain
        layer travels through the one supported channel.
        """
        types = ", ".join(f"`{t}`" for t in self.entity_types)
        return (
            f"Allowed entity types (a CLOSED list): {types}.\n"
            "This list is CLOSED — never invent a new type and never use `Other`/`Unknown`. "
            "If a candidate entity does not clearly fit one of these types, DO NOT extract "
            "it at all (omit it entirely rather than forcing a catch-all).\n\n"
            + self.extraction_exclusions
        )


_CACHE: DomainPack | None = None


def _resolve_pack_dir() -> Path:
    """Resolve the active pack directory from the environment (see module docstring)."""
    override = os.environ.get("PAPERVAULT_DOMAIN_PATH")
    if override:
        return Path(override)
    return Path(__file__).parent / os.environ.get("PAPERVAULT_DOMAIN", "space_physics")


def _load_pack(pack_dir: Path) -> DomainPack:
    if not pack_dir.is_dir():
        raise RuntimeError(
            f"papervault domain pack not found: {pack_dir} "
            "(set PAPERVAULT_DOMAIN_PATH to an external pack dir, or PAPERVAULT_DOMAIN to a "
            "bundled pack name; the factory default is 'space_physics')."
        )

    with (pack_dir / "domain.toml").open("rb") as fh:
        meta = tomllib.load(fh)
    label = str(meta["label"])

    entity_types = json.loads((pack_dir / "ontology.json").read_text(encoding="utf-8"))

    raw_examples = (pack_dir / "extraction_examples.txt").read_text(encoding="utf-8")
    extraction_examples = [
        part.strip() for part in _EXAMPLE_SEP_RE.split(raw_examples) if part.strip()
    ]

    extraction_exclusions = (pack_dir / "extraction_exclusions.txt").read_text(encoding="utf-8")
    search_gate = (pack_dir / "search_gate.md").read_text(encoding="utf-8")

    return DomainPack(
        label=label,
        entity_types=list(entity_types),
        extraction_examples=extraction_examples,
        extraction_exclusions=extraction_exclusions,
        search_gate=search_gate,
    )


def get_domain() -> DomainPack:
    """Return the active :class:`DomainPack` (cached singleton).

    The pack directory is resolved once from ``PAPERVAULT_DOMAIN_PATH`` /
    ``PAPERVAULT_DOMAIN`` (see module docstring) and the result cached in a module
    global; subsequent calls do not re-read disk. Raises ``RuntimeError`` if the
    resolved pack directory does not exist.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_pack(_resolve_pack_dir())
    return _CACHE
