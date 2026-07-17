"""KS domain-tuned entity-extraction prompt (research variants V2 + V3, KS_RAG_SOTA_RESEARCH.md).

WHY: LightRAG's stock few-shot examples are a fiction scene / finance blurb / sports headline
whose <Output> blocks emit person/organization/equipment/category — so the extractor is TAUGHT
to pull people, orgs, and citation-shaped proper nouns, which is the root cause of KS's
off-ontology + citation-entity noise. KS only injected the 11-type ontology via addon_params
(a weak hint) while the EXAMPLES showed the opposite. This module replaces the examples with
space-physics / AI-for-science ones (V2) and adds explicit negative exclusions + an acronym
canonicalization rule to the system prompt (V3).

Mutates the module-level lightrag.prompt.PROMPTS dict, so call apply_ks_extraction_prompt()
BEFORE the extraction runs (before ainsert / get_graph builds). operate.py reads PROMPTS at
extraction time, .format()-substituting {tuple_delimiter}/{completion_delimiter}/{entity_types}/
{language} — so the strings below MUST contain no other literal braces.
"""
from __future__ import annotations

from lightrag import prompt as lr_prompt

from papervault.domain import get_domain

# V2 domain few-shot examples (the exact LightRAG 4-field entity / 5-field relation format)
# are externalized to the active domain pack (ADR-0003, papervault.domain). The bundled
# space_physics factory pack reproduces the previously-hardcoded few-shot strings verbatim
# (each still contains no literal braces other than the {tuple_delimiter}/{completion_delimiter}
# placeholders LightRAG substitutes at extraction time). Read at import as a module-level
# constant; the loader caches, so this is a single cached disk read, not a per-call one.
KS_EXAMPLES = get_domain().extraction_examples

# --- V3: negative exclusions + acronym rule, inserted before the system prompt's ---Examples--- block.
# V3 negative exclusions + acronym/substance rules are externalized to the active domain
# pack (ADR-0003); the space_physics factory pack reproduces the previous block verbatim.
_EXCLUSIONS = get_domain().extraction_exclusions


# --- V3: closed-ontology replacement for LightRAG's `Other` catch-all (system prompt item 1).
# The stock entity_type bullet ends with an escape hatch ("classify it as `Other`") that
# re-admits exactly the off-ontology proper-noun noise the 11-type ontology exists to drop
# (LightRAG lowercases entity_type on store, so `Other` lands as a real `other` node, not dropped).
# Replace it with a CLOSED-list + DROP instruction. Anchored on the exact vendored sentence.
_OTHER_CATCHALL = (
    "Categorize the entity using one of the following types: `{entity_types}`. "
    "If none of the provided entity types apply, do not add new entity type and classify it as `Other`."
)
_CLOSED_ONTOLOGY = (
    "Categorize the entity using EXACTLY one of the following types: `{entity_types}`. "
    "This list is CLOSED — never invent a new type and never use `Other`/`Unknown`. "
    "If a candidate entity does not clearly fit one of these types, DO NOT extract it at all "
    "(omit it entirely rather than forcing a catch-all)."
)


def apply_ks_extraction_prompt(examples: bool = True, exclusions: bool = True) -> None:
    """Mutate lightrag.prompt.PROMPTS in place. Idempotent. Call before extraction.

    This monkeypatch anchors on LightRAG 1.4.x internals (the ``\\n---Examples---`` marker and
    the `Other` catch-all in the entity-extraction system prompt). A LightRAG version that
    restructures that prompt makes the patch a silent no-op — the space-physics noise-suppression
    and closed ontology would vanish with no error, degrading every graph build. So we FAIL LOUD
    if neither anchor is found (the pin is `lightrag-hku>=1.4,<1.5`; this backs it up)."""
    if exclusions:
        sysp = lr_prompt.PROMPTS["entity_extraction_system_prompt"]
        already = "Exclusions (do NOT extract" in sysp
        has_examples_anchor = "\n---Examples---" in sysp
        has_other = _OTHER_CATCHALL in sysp
        if not already and not has_examples_anchor and not has_other:
            raise RuntimeError(
                "apply_ks_extraction_prompt: neither the '---Examples---' anchor nor the "
                "'Other' catch-all was found in LightRAG's entity_extraction_system_prompt. "
                "The extraction-prompt monkeypatch would silently no-op — this usually means an "
                "unsupported LightRAG version (pin is >=1.4,<1.5). Refusing to build a degraded graph."
            )
        if not already and has_examples_anchor:
            sysp = sysp.replace("\n---Examples---", "\n" + _EXCLUSIONS, 1)
        # Drop the `Other` catch-all: close the ontology so off-ontology candidates are omitted.
        if has_other:
            sysp = sysp.replace(_OTHER_CATCHALL, _CLOSED_ONTOLOGY, 1)
        lr_prompt.PROMPTS["entity_extraction_system_prompt"] = sysp
    if examples:
        lr_prompt.PROMPTS["entity_extraction_examples"] = list(KS_EXAMPLES)
