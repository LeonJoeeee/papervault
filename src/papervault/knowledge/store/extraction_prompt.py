"""Domain-tuned entity-extraction prompt for LightRAG 1.5.x (research variants V2 + V3).

WHY: LightRAG's stock few-shot examples are a fiction scene / finance blurb / sports headline
whose <Output> blocks emit person/organization/equipment/category — so the extractor is TAUGHT
to pull people, orgs, and citation-shaped proper nouns, which is the root cause of the
off-ontology + citation-entity noise the domain ontology exists to drop. The domain pack
replaces the examples with in-domain ones (V2) and closes the ontology + adds negative
exclusions (V3).

HOW, on LightRAG 1.5.x (ported 2026-07-17; the 1.4.x mechanics are retired):
  - The ontology + negative exclusions travel through the OFFICIAL channel:
    ``addon_params["entity_types_guidance"]`` (one string, rendered into the system prompt's
    ``---Entity Types---`` section). ``store/graph.py`` passes
    ``get_domain().entity_types_guidance`` — nothing to monkeypatch for that.
  - The few-shot examples still go through ``PROMPTS["entity_extraction_examples"]``
    (read at LightRAG ctor time in text mode), replaced here.
  - ONE monkeypatch remains: the 1.5.x system prompt's entity_type bullet still ends with an
    ``Other`` escape hatch ("If none of the provided entity types apply, classify it as
    `Other`."), which re-admits exactly the noise the closed ontology drops (LightRAG stores
    `Other` as a real node). We replace that sentence with a DROP instruction.

Call ``apply_ks_extraction_prompt()`` BEFORE any ainsert (``get_graph()`` does). The strings
must contain no literal braces other than the placeholders LightRAG .format()-substitutes.

FAIL-LOUD: this anchors on 1.5.x prompt internals (pin: ``lightrag-hku>=1.5,<1.6``). If the
anchor sentence is missing, we RAISE instead of building a silently degraded graph. Same for
``ENTITY_EXTRACTION_USE_JSON``: JSON extraction mode reads ``entity_extraction_json_examples``,
which we do not patch — refuse rather than run with stock examples.
"""
from __future__ import annotations

import os

from lightrag import prompt as lr_prompt

from papervault.domain import get_domain

# V2 domain few-shot examples (LightRAG 4-field entity / 5-field relation record format),
# from the active domain pack (ADR-0003). Record shape + delimiters are unchanged between
# LightRAG 1.4.x and 1.5.x; the strings carry only {tuple_delimiter}/{completion_delimiter}.
KS_EXAMPLES = get_domain().extraction_examples

# V3 negative exclusions (from the pack). On 1.5.x these are DELIVERED via
# get_domain().entity_types_guidance (see store/graph.py addon_params), kept exported here
# for tests/consumers that assert the block's content.
_EXCLUSIONS = get_domain().extraction_exclusions

# --- The one remaining monkeypatch: kill 1.5.x's `Other` escape hatch. Anchored on the
# exact vendored sentence (prompt.py line ~62 in 1.5.4).
_OTHER_CATCHALL_15 = (
    "If none of the provided entity types apply, classify it as `Other`."
)
_CLOSED_ONTOLOGY_15 = (
    "If none of the provided entity types apply, DO NOT extract the entity at all "
    "(the type list is CLOSED — never use `Other`/`Unknown`, omit the entity instead)."
)


def apply_ks_extraction_prompt(examples: bool = True, exclusions: bool = True) -> None:
    """Mutate lightrag.prompt.PROMPTS in place. Idempotent. Call before extraction.

    ``exclusions=True`` applies the closed-ontology patch (the `Other` escape-hatch
    replacement — the exclusions TEXT itself travels via addon_params entity_types_guidance).
    ``examples=True`` swaps in the domain pack's few-shot examples. Raises RuntimeError when
    the anchors are missing (unsupported LightRAG version) or when JSON extraction mode is
    enabled (its example set is not patched) — never build a silently degraded graph.
    """
    if os.environ.get("ENTITY_EXTRACTION_USE_JSON", "").strip().lower() in ("1", "true", "yes"):
        raise RuntimeError(
            "apply_ks_extraction_prompt: ENTITY_EXTRACTION_USE_JSON is enabled, but the domain "
            "examples are only wired into TEXT extraction mode (entity_extraction_examples). "
            "Unset it, or port the domain pack to entity_extraction_json_examples first."
        )
    if exclusions:
        sysp = lr_prompt.PROMPTS["entity_extraction_system_prompt"]
        already = _CLOSED_ONTOLOGY_15 in sysp
        has_other = _OTHER_CATCHALL_15 in sysp
        if not already and not has_other:
            raise RuntimeError(
                "apply_ks_extraction_prompt: the `Other` escape-hatch anchor was not found in "
                "LightRAG's entity_extraction_system_prompt. The closed-ontology patch would "
                "silently no-op — this usually means an unsupported LightRAG version "
                "(pin is >=1.5,<1.6). Refusing to build a degraded graph."
            )
        if has_other:
            sysp = sysp.replace(_OTHER_CATCHALL_15, _CLOSED_ONTOLOGY_15, 1)
        lr_prompt.PROMPTS["entity_extraction_system_prompt"] = sysp
    if examples:
        lr_prompt.PROMPTS["entity_extraction_examples"] = list(KS_EXAMPLES)
