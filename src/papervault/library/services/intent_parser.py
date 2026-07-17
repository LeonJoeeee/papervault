"""LLM-based intent parser for ``search_papers``.

Translates a natural-language research intent into a structured search plan:
  - search_terms: list of keyword phrases to feed to backend search APIs
  - filters: year_min / year_max (the only REAL filters, emitted only on an
    explicit recency cue) + citation_pref / review_pref (SOFT ranking hints)
  - limit_suggested: count hint from intent (e.g. "5 papers" -> 5; None otherwise)
  - ranking_hint: ``by_importance`` / ``by_recency`` / ``by_relevance``
  - reasoning: short string for debug visibility

Why this exists: callers (LLM agents or humans) express what they want in
natural language; we don't make them translate to keywords + filter args.
See ``mcp-llm-to-llm-design`` memory entry: thin-in interface, LLM at
backend parses intent.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)


# Hard cap on search_terms. The prompt constrains LLM #0 to ~2-5 ORTHOGONAL
# FACETS (term 0 = the verbatim anchor; A3 facet-orthogonality, no synonym
# padding); we cap server-side as defensive truncation in case LLM ignores it.
# The cap is a PREFIX, so it never displaces the term-0 anchor (A1).
MAX_SEARCH_TERMS = 8


class SearchPlan(TypedDict):
    search_terms: list[str]
    filters: dict
    limit_suggested: Optional[int]
    ranking_hint: str
    reasoning: str


_SYSTEM_PROMPT = """You are a search planner for a research librarian working on
**space physics + AI4Science**. Your job: turn a natural-language research
intent into a structured search plan.

First THINK (briefly, in the "reasoning" field): what would a paper that is a
PERFECT hit for this intent actually be ABOUT? Name its core concept and its
distinct facets. THEN emit the terms. Do not skip this reasoning step — the
quality of the terms depends on it.

The downstream pipeline takes:
- search_terms: a list of keyword phrases sent to multiple academic backends.
  TWO HARD RULES govern this list:

  RULE 1 — TERM 0 IS THE ANCHOR (the verbatim core concept). The FIRST term must
  be the single core concept of the intent stated plainly — the phrase a
  perfectly on-target paper's title would literally contain. The pipeline gives
  term 0 priority so the rest of the terms (expansions) can never outvote the
  user's actual intent. Get term 0 right first; everything else expands around it.

  RULE 2 — TERMS COVER DISTINCT FACETS, NOT REWORDINGS. Each remaining term must
  cover a DIFFERENT facet of the intent — a different ANGLE — not a near-duplicate
  rewording of the same idea. The facets of a space-physics + AI4Science intent
  are usually some of:
    * phenomenon — what is being studied (e.g. cosmic ray transport, SEP events)
    * method — how (e.g. physics-informed neural network, Bayesian inversion)
    * system — which mission / instrument / object / dataset (e.g. Voyager,
      Parker Solar Probe, neutron monitor)
    * regime — which range / condition / horizon (e.g. outer heliosphere,
      solar maximum, multi-year forecast)
  Aim for ONE term per real facet present in the intent. Vary terminology ACROSS
  facets (expand acronyms, mix domain phrasing) — but the variation should change
  the FACET, not just reword the same facet.

  OVER-SPLIT GUARD (do NOT manufacture fake facets): one atomic concept is ONE
  facet, never several synonym shards. "PINN", "physics-informed neural network",
  "physics-informed deep learning" are the SAME method facet — pick the best one,
  do not spend three terms on it. Prefer FEWER orthogonal terms over many
  near-duplicates: an intent with only 2 real facets should yield ~2 terms, not 6
  padded ones. Never exceed 6 terms (hard ceiling 8); 2-5 is typical.

  Each term should be 2-6 words. Don't include single common words.
- filters: optional constraints derived from the intent. Two of them are
  SOFT HINTS — gentle ranking nudges, NEVER a hard cut:
    * citation_pref: an int when the intent wants well-cited / high-impact
      work (e.g. "highly cited" / "important" / "classic"), else null. Only its
      PRESENCE and DIRECTION matter downstream — the magnitude is a hint, not a gate.
    * review_pref: "prefer" when the intent asks for a review / survey /
      overview; "off" when it explicitly wants PRIMARY (non-review)
      work; null when the intent says nothing about review-vs-primary.
  year_min / year_max are the ONLY REAL filters. Emit them ONLY when the
  intent carries an EXPLICIT recency or date cue (see "Year rule" below);
  otherwise BOTH stay null. Do NOT add a year window just because a topic
  feels modern — absent a cue, no window.
- limit_suggested: count hint from the intent (e.g. "5 papers" -> 5; null if
  no count specified).
- ranking_hint: one of by_importance / by_recency / by_relevance based
  on intent cues like "important" / "recent" / "about ...".

Year rule (year_min / year_max emitted ONLY on an EXPLICIT recency cue):
- "in recent years" / "lately" -> a generous recent window (year_min ~ 8-10 years back).
- "latest" / "this year" / "last 1-2 years" -> year_min ~ 2-3 years back.
- "last decade" / "past ten years" -> year_min = 10 years back. A literal year
  like "2024" -> year_min = year_max = 2024.
- NO recency/date words at all -> year_min = year_max = null. A bare topic,
  a methodology question, or "important/highly cited" WITHOUT a recency word gets
  NO year window (those map to citation_pref / ranking_hint, not a year cut).
Prefer generous windows over narrow ones (a narrow window can over-exclude).

Output ONLY JSON:
{
  "search_terms": ["...", "...", ...],
  "filters": {
    "year_min": <int|null>,
    "year_max": <int|null>,
    "citation_pref": <int|null>,
    "review_pref": "prefer" | "off" | null
  },
  "limit_suggested": <int|null>,
  "ranking_hint": "by_importance" | "by_recency" | "by_relevance",
  "reasoning": "<short>"
}

==== Few-shot examples ====
(In each, term 0 is the verbatim ANCHOR; the rest are DISTINCT facets — never
synonym rewordings of term 0. The reasoning names the facets.)

Intent: "a review of PINN"
Output: {
  "search_terms": ["physics-informed neural network", "PINN survey review"],
  "filters": {"year_min": null, "year_max": null, "citation_pref": null, "review_pref": "prefer"},
  "limit_suggested": null,
  "ranking_hint": "by_relevance",
  "reasoning": "Core=PINN (anchor=method). Only ONE real facet is present (the PINN method itself); the survey intent already lives in review_pref. So just 2 terms: the anchor + one survey angle. Do NOT add a 3rd 'PINN benchmark' term — benchmark-form and review-form are the SAME PINN facet reworded (they would pull overlapping sets), not a distinct axis. No recency -> no year window"
}

Intent: "recent PINN work on SEP transport"
Output: {
  "search_terms": ["solar energetic particle transport", "physics-informed neural network", "Parker transport equation", "SEP propagation inversion"],
  "filters": {"year_min": 2023, "year_max": null, "citation_pref": null, "review_pref": null},
  "limit_suggested": null,
  "ranking_hint": "by_recency",
  "reasoning": "Core=SEP transport (anchor=phenomenon). Facets: method (PINN), system (Parker equation), regime (inversion) — each a different angle, not a reworded 'PINN SEP'. recency cue -> 2023+"
}

Intent: "Voyager cosmic ray measurements in the outer heliosphere"
Output: {
  "search_terms": ["Voyager outer heliosphere cosmic ray", "cosmic ray modulation", "termination shock heliopause", "anomalous cosmic ray interstellar"],
  "filters": {"year_min": null, "year_max": null, "citation_pref": null, "review_pref": null},
  "limit_suggested": null,
  "ranking_hint": "by_relevance",
  "reasoning": "Core=Voyager cosmic ray measurement (anchor=system+phenomenon). Facets: phenomenon (modulation), regime (termination shock/heliopause), regime (interstellar ACR). No recency -> no year window"
}

Intent: "latest progress on UHECR"
Output: {
  "search_terms": ["ultra-high-energy cosmic ray", "Pierre Auger Telescope Array", "UHECR composition spectrum", "extragalactic UHECR origin"],
  "filters": {"year_min": 2023, "year_max": null, "citation_pref": null, "review_pref": null},
  "limit_suggested": null,
  "ranking_hint": "by_recency",
  "reasoning": "Core=UHECR (anchor, acronym expanded). Facets: system (observatories), phenomenon (composition/spectrum), regime (origin). recency cue -> 2023+"
}

Intent: "AI applied to MHD simulation"
Output: {
  "search_terms": ["machine learning magnetohydrodynamics simulation", "neural network surrogate model", "operator learning PDE", "solar wind plasma simulation"],
  "filters": {"year_min": null, "year_max": null, "citation_pref": null, "review_pref": null},
  "limit_suggested": null,
  "ranking_hint": "by_relevance",
  "reasoning": "Core=ML for MHD simulation (anchor). Facets: method (surrogate), method (operator learning — a DIFFERENT technique, not a reword), system (solar wind). Topic only -> no year window"
}

Intent: "Bayesian inversion inverse-problem methodology"
Output: {
  "search_terms": ["Bayesian inversion inverse problem", "uncertainty quantification posterior", "Gaussian process regression", "variational inference"],
  "filters": {"year_min": null, "year_max": null, "citation_pref": null, "review_pref": null},
  "limit_suggested": null,
  "ranking_hint": "by_relevance",
  "reasoning": "Core=Bayesian inversion (anchor=method). Facets: regime (UQ/posterior), method (GP — distinct technique), method (variational — distinct technique). Pure methodology -> no year window"
}

Intent: "highly cited GCR review from 2024"
Output: {
  "search_terms": ["galactic cosmic ray review", "GCR solar modulation", "cosmic ray transport heliosphere"],
  "filters": {"year_min": 2024, "year_max": 2024, "citation_pref": 5, "review_pref": "prefer"},
  "limit_suggested": null,
  "ranking_hint": "by_importance",
  "reasoning": "Core=GCR review (anchor). Facets: phenomenon (solar modulation), regime (heliospheric transport). Literal 2024; high-cite -> citation_pref; survey -> review_pref"
}

Intent: "I want 5 of the most important papers in the PINN field, from the last 3 years"
Output: {
  "search_terms": ["physics-informed neural network", "PINN training optimization", "PINN benchmark applications"],
  "filters": {"year_min": 2023, "year_max": null, "citation_pref": 20, "review_pref": null},
  "limit_suggested": 5,
  "ranking_hint": "by_importance",
  "reasoning": "Core=PINN (anchor). One method field -> only 3 terms covering distinct sub-aspects (training, benchmarks), NOT 5 reworded synonyms. recency cue -> 2023+; 'most important' -> citation_pref"
}

Intent: "latest progress on PINN applied to fluid dynamics"
Output: {
  "search_terms": ["physics-informed neural network fluid dynamics", "Navier-Stokes neural solver", "turbulence modeling deep learning", "PINN CFD application"],
  "filters": {"year_min": 2024, "year_max": null, "citation_pref": null, "review_pref": null},
  "limit_suggested": null,
  "ranking_hint": "by_recency",
  "reasoning": "Core=PINN for fluids (anchor). Facets: system (Navier-Stokes), phenomenon (turbulence), regime (CFD). recency cue -> 2024+"
}

Intent: "classic foundational work in the PINN field"
Output: {
  "search_terms": ["physics-informed neural network", "PINN original formulation", "deep learning differential equations"],
  "filters": {"year_min": null, "year_max": null, "citation_pref": 50, "review_pref": "off"},
  "limit_suggested": null,
  "ranking_hint": "by_importance",
  "reasoning": "Core=PINN (anchor). Facets: regime (original/foundational), phenomenon (solving differential equations). classic/seminal -> high citation_pref + primary; no recency word -> no year window"
}

Intent: "machine learning"
Output: {
  "search_terms": ["machine learning"],
  "filters": {"year_min": null, "year_max": null, "citation_pref": null, "review_pref": null},
  "limit_suggested": null,
  "ranking_hint": "by_relevance",
  "reasoning": "Core=machine learning (anchor, verbatim). VERY broad and facet-less: no phenomenon/method/system/regime is named, so there is NOTHING orthogonal to split. Keep terms FEW — do NOT pad term 0 or invent synonym shards (deep learning / neural network are the SAME broad concept, not distinct facets). One honest anchor term; caller should narrow. No recency/review/impact cue -> all filters null."
}

==== End few-shots ====
"""


def parse_intent(query: str, llm=None) -> SearchPlan:
    """Parse a natural-language research intent into a structured search plan.

    The plan drives the downstream search pipeline (multi-source aggregation
    + LLM filter/rank). Caller passes the plan to ``search_external`` and
    ``llm_filter_and_rank``.

    Args:
        query: Natural language intent. May be Chinese or English. Can be
            terse ("PINN review") or expressive (full paragraph).
        llm: Optional pre-built LLM instance (for testing). Defaults to
            ``get_llm()``.

    Returns:
        SearchPlan dict with normalized fields.

    Raises:
        ValueError: LLM response not parseable as JSON (caller may retry).
    """
    if llm is None:
        from ..llm import get_llm
        llm = get_llm()

    user_prompt = f"Intent: {query}\nOutput:"

    raw = llm.call([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    # Extract JSON object from response (handles wrapping prose if LLM
    # ignores the "ONLY JSON" instruction)
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        raise ValueError(f"intent_parser: no JSON in LLM response: {(raw or '')[:200]}")

    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"intent_parser: malformed JSON: {e}; raw={m.group(0)[:200]}")

    # Normalize + validate. The raw-query fallback (LLM returned no/empty
    # search_terms) re-injects the query as a single literal anchor term — but
    # STRIPPED, never the raw unstripped string, so an all-whitespace query can
    # never become an all-whitespace anchor term.
    _query_anchor = (query or "").strip()
    search_terms = parsed.get("search_terms") or [_query_anchor]
    if not isinstance(search_terms, list) or not search_terms:
        search_terms = [_query_anchor]
    # Strip/filter-empty (NO cap yet) -> order-preserving DEDUP -> cap LAST.
    # Dedup BEFORE the cap so a duplicate term never spends a fair-share slot
    # and so the cap never loses a genuine distinct sub-topic to an early dup.
    _stripped = [str(t).strip() for t in search_terms if str(t).strip()]  # NO cap yet
    _seen: set = set()
    _distinct = [t for t in _stripped if not (t in _seen or _seen.add(t))]  # order-preserving dedup
    search_terms = _distinct[:MAX_SEARCH_TERMS]  # cap LAST — up to 8 DISTINCT terms
    if not search_terms:
        # No usable terms AND the query is itself empty-after-strip → there is
        # nothing to search. Fail CLOSED as a parse error (the caller maps a
        # ValueError to {status:error}) rather than fanning out / ingesting on
        # an all-whitespace anchor term.
        if not _query_anchor:
            raise ValueError("intent_parser: empty query yields no search terms")
        search_terms = [_query_anchor]

    filters = parsed.get("filters") or {}
    if not isinstance(filters, dict):
        filters = {}
    # Normalize filter values: coerce "null"/"None"/"" -> None; fold a legacy
    # pinned-old-model is_review bool into review_pref in the SAME pass (thin
    # tolerance — dead-on-arrival for any model running the rewritten prompt).
    for k in ("year_min", "year_max", "review_pref", "citation_pref"):
        v = filters.get(k)
        if v in ("null", "None", ""):
            filters[k] = None
        elif k == "review_pref" and isinstance(v, bool):  # legacy-bool guard, same pass
            filters[k] = "prefer" if v else "off"

    # Int-coerce the numeric bounds so a string/float from the LLM (e.g.
    # "2023") cannot TypeError-crash YEAR_DROP's int comparison downstream;
    # an uncoercible value degrades to None (= no bound), never blows up.
    for k in ("year_min", "year_max", "citation_pref"):
        v = filters.get(k)
        if v is None:
            continue
        try:
            filters[k] = int(v)
        except (ValueError, TypeError):
            filters[k] = None

    limit_suggested = parsed.get("limit_suggested")
    if limit_suggested is not None:
        try:
            limit_suggested = int(limit_suggested)
            if limit_suggested <= 0 or limit_suggested > 200:
                limit_suggested = None
        except (ValueError, TypeError):
            limit_suggested = None

    ranking_hint = parsed.get("ranking_hint", "by_relevance")
    if ranking_hint not in ("by_importance", "by_recency", "by_relevance"):
        ranking_hint = "by_relevance"

    reasoning = str(parsed.get("reasoning", ""))[:300]

    logger.info(
        "intent_parser: query=%r -> terms=%r filters=%r limit=%r hint=%s",
        query, search_terms, filters, limit_suggested, ranking_hint,
    )

    return {
        "search_terms": search_terms,
        "filters": filters,
        "limit_suggested": limit_suggested,
        "ranking_hint": ranking_hint,
        "reasoning": reasoning,
    }
