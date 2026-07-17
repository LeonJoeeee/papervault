"""1.4.16-parity keyword-extraction prompt (equivalence-gate residual fix, issue #4).

LightRAG 1.5.x rewrote PROMPTS["keywords_extraction"] ("keep the lists SHORT") and replaced
the THREE rich few-shot examples with a single bare placeholder — the extractor loses all
demonstration of keyword count/richness, thinning the base-arm hl/ll keywords that drive the
KG (entity/relation) retrieval arms. This module restores the 1.4.16 strings VERBATIM
(extracted programmatically from lightrag 1.4.16, never hand-typed).

Applied by get_graph() when KS_KW_PROMPT_COMPAT=1 — DEFAULT 0 (OFF): the frozen-window
arbitration (2026-07-18, issue #4) showed the 1.5.4 stock keyword prompt WINS (frozen arm
E' 0.8568 with compat off; drift-tainted arm F 0.7798 with it on). Kept as an experiment
knob only."""
from __future__ import annotations

import os

from lightrag import prompt as lr_prompt

KW_PROMPT_1416 = "---Role---\nYou are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.\n\n---Goal---\nGiven a user query, your task is to extract two distinct types of keywords:\n1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.\n2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.\n\n---Instructions & Constraints---\n1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.\n2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.\n3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from \"latest financial report of Apple Inc.\", you should extract \"latest financial report\" and \"Apple Inc.\" rather than \"latest\", \"financial\", \"report\", and \"Apple\".\n4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., \"hello\", \"ok\", \"asdfghjkl\"), you must return a JSON object with empty lists for both keyword types.\n5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.\n\n---Examples---\n{examples}\n\n---Real Data---\nUser Query: {query}\n\n---Output---\nOutput:"

KW_EXAMPLES_1416 = ["Example 1:\n\nQuery: \"How does international trade influence global economic stability?\"\n\nOutput:\n{\n  \"high_level_keywords\": [\"International trade\", \"Global economic stability\", \"Economic impact\"],\n  \"low_level_keywords\": [\"Trade agreements\", \"Tariffs\", \"Currency exchange\", \"Imports\", \"Exports\"]\n}\n\n", "Example 2:\n\nQuery: \"What are the environmental consequences of deforestation on biodiversity?\"\n\nOutput:\n{\n  \"high_level_keywords\": [\"Environmental consequences\", \"Deforestation\", \"Biodiversity loss\"],\n  \"low_level_keywords\": [\"Species extinction\", \"Habitat destruction\", \"Carbon emissions\", \"Rainforest\", \"Ecosystem\"]\n}\n\n", "Example 3:\n\nQuery: \"What is the role of education in reducing poverty?\"\n\nOutput:\n{\n  \"high_level_keywords\": [\"Education\", \"Poverty reduction\", \"Socioeconomic development\"],\n  \"low_level_keywords\": [\"School access\", \"Literacy rates\", \"Job training\", \"Income inequality\"]\n}\n\n"]


def apply_kw_prompt_compat() -> None:
    """Restore the 1.4.16 keyword-extraction prompt + rich few-shots. Idempotent."""
    if os.getenv("KS_KW_PROMPT_COMPAT", "0").strip().lower() in ("0", "false", "no"):
        return
    lr_prompt.PROMPTS["keywords_extraction"] = KW_PROMPT_1416
    if KW_EXAMPLES_1416 is not None:
        lr_prompt.PROMPTS["keywords_extraction_examples"] = KW_EXAMPLES_1416
