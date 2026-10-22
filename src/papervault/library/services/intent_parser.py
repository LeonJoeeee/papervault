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
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import NotRequired, Optional, TypedDict

from papervault.domain import get_domain

logger = logging.getLogger(__name__)


# Hard cap on search_terms. The prompt constrains LLM #0 to ~2-5 ORTHOGONAL
# FACETS (term 0 = the verbatim anchor; A3 facet-orthogonality, no synonym
# padding); we cap server-side as defensive truncation in case LLM ignores it.
# The cap is a PREFIX, so it never displaces the term-0 anchor (A1), or its
# first per-experiment replacement when the LLM merged named experiments.
MAX_SEARCH_TERMS = 8


# A conservative guard for known experiment names, not acronym-based NER:
# methods/phenomena such as PINN, MHD and SEP must never be split as instruments.
# Aliases share an identity; multiword names are matched as a whole.
_EXPERIMENT_PATTERN = re.compile(
    r"(?<!\w)(?:"
    r"(?P<pamela>PAMELA)|"
    r"(?P<ams02>Alpha Magnetic Spectrometer|AMS[-\s]?02)|"
    r"(?P<auger>Pierre Auger(?: Observatory)?)|"
    r"(?P<telescope_array>Telescope Array)"
    r")(?!\w)",
    re.IGNORECASE,
)
_ENTITY_SEPARATOR = re.compile(r"[\s,/;&+]*(?:(?:and|or|versus|vs)\b\.?[\s,/;&+]*)*", re.I)
_NEXT_MEASUREMENT_NOUN = {
    "spectrum": "abundance",
    "abundance": "fraction",
    "fraction": "flux",
    "flux": "measurement",
    "measurement": "spectrum",
}
# Fold ASCII case only for the dictionary keys; retain Unicode word boundaries.
_MEASUREMENT_NOUN = re.compile(r"\b(?ai:" + "|".join(_NEXT_MEASUREMENT_NOUN) + r")\b")
_FORBIDDEN_SIBLING_NOUN = re.compile(r"\b(?:features?|excess(?:es)?|hints?)\b", re.I)
_EXPERIMENT_ALIASES = {
    "pamela": ["PAMELA", "Payload for Antimatter Matter Exploration and Light-nuclei Astrophysics"],
    "ams02": ["AMS-02", "AMS02", "AMS 02", "Alpha Magnetic Spectrometer"],
    "auger": ["Pierre Auger", "Pierre Auger Observatory"],
    "telescope_array": ["Telescope Array"],
}


class SearchPlan(TypedDict):
    search_terms: list[str]
    filters: dict
    limit_suggested: Optional[int]
    ranking_hint: str
    reasoning: str
    named_entities: NotRequired[list[str]]


def _named_entity_match(name: str, query: str):
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", query, re.I)


def named_entity_groups(query: str, plan: SearchPlan) -> list[list[str]]:
    """Explicit instruments with aliases for per-need serving reservations.

    Known names work with older planners; held-out names come from the planner
    and must occur in the user query. These groups do not rewrite search terms.
    """
    found = []
    for aliases in _EXPERIMENT_ALIASES.values():
        matches = [match for alias in aliases if (match := _named_entity_match(alias, query))]
        if matches:
            found.append((min(match.start() for match in matches), aliases))
    for name in plan.get("named_entities", []):
        if not isinstance(name, str) or not (match := _named_entity_match(name, query)):
            continue
        known = next((aliases for aliases in _EXPERIMENT_ALIASES.values()
                      if name.casefold() in {alias.casefold() for alias in aliases}), None)
        found.append((match.start(), known or [name]))
    groups = {}
    for _, aliases in sorted(found, key=lambda item: item[0]):
        groups.setdefault(aliases[0].casefold(), list(aliases))
    return list(groups.values())


@dataclass(frozen=True)
class IntentObservation:
    model: str
    provider: str
    latency_ms: float
    degraded: bool
    code: str | None = None


_intent_observer = ContextVar("intent_observer", default=None)


@contextmanager
def observe_intent(callback):
    """Emit one bounded receipt per parse without changing SearchPlan payloads."""
    token = _intent_observer.set(callback)
    try:
        yield
    finally:
        _intent_observer.reset(token)


_SYSTEM_PROMPT = """You are a search planner for a research librarian working on
**__DOMAIN_LABEL__**. Your job: turn a natural-language research
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
  the FACET, not just reword the same facet. Instrument acronym/full-name pairs
  are the explicit ENTITY RULE exception below.

  ENTITY RULE — one named experiment/instrument per term; never merge entities into one term.
  Give each experiment/instrument named in the intent its own system facet, paired with the
  phenomenon and a canonical measurement noun — one of: fraction, spectrum,
  flux, abundance, measurement. FORBID "feature", "features", "spectral feature",
  "excess", "hint", and "debate" in entity terms: these describe the debate,
  not the measurement reported by the original papers.
  For an experiment/instrument named in the intent, when its common acronym
  and full name differ, emit BOTH
  name forms, each as its own term with the phenomenon and a canonical noun
  (e.g. "AMS-02 positron fraction" AND
  "Alpha Magnetic Spectrometer positron fraction"). Keep the name forms
  separate; never merge an acronym and its full name into one term. These
  required name forms are an exception to the no-synonym-rewording advice.
  Distinct experiments are distinct facets even when measuring the same
  phenomenon. For a multi-entity intent, term 0 can name the shared phenomenon;
  it must not concatenate the entity names. Separate abundance/fraction from
  energy-spectrum measurements when the intent asks for both observables.
  Keep term 0 as the verbatim anchor. Within the eight-term budget INCLUDING
  the anchor, first give each entity one canonical-noun term using its acronym,
  then add full-name terms, then additional requested observables. If the
  budget forces a choice, prefer one canonical-noun term per entity, acronym
  first, then full names.

  OVER-SPLIT GUARD (do NOT manufacture fake facets): one atomic concept is ONE
  facet, never several synonym shards. "PINN", "physics-informed neural network",
  "physics-informed deep learning" are the SAME method facet — pick the best one,
  do not spend three terms on it. Prefer FEWER orthogonal terms over many
  near-duplicates: an intent with only 2 real facets should yield ~2 terms, not 6
  padded ones. Aim for 2-5 terms; use up to 8 when required instrument name forms
  or requested observables need them. Never exceed 8 terms including the anchor.

  Each term should be 2-6 words, except when retaining a full instrument name
  requires more. Don't include single common words.
- named_entities: experiment/instrument/mission names explicitly written in the
  user's intent, copied using the user's spelling. Include held-out instruments
  such as Voyager 1 or ACE when named. Do not include methods or phenomena such
  as PINN, MHD or SEP, and do not invent instruments absent from the intent.
  This list tracks distinct requested needs independently of search expansion.
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
  "named_entities": ["<explicit experiment or instrument name>", ...],
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
synonym rewordings of term 0. Required instrument name forms follow the ENTITY
RULE exception. The reasoning names the facets.)

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
  "search_terms": ["Voyager heliospheric cosmic ray flux", "cosmic ray modulation", "termination shock heliopause", "anomalous cosmic ray interstellar"],
  "filters": {"year_min": null, "year_max": null, "citation_pref": null, "review_pref": null},
  "limit_suggested": null,
  "ranking_hint": "by_relevance",
  "reasoning": "Core=Voyager cosmic ray measurement (anchor=system+phenomenon). Facets: phenomenon (modulation), regime (termination shock/heliopause), regime (interstellar ACR). No recency -> no year window"
}

Intent: "latest progress on UHECR"
Output: {
  "search_terms": ["ultra-high-energy cosmic ray", "Pierre Auger cosmic ray spectrum", "Telescope Array cosmic ray spectrum", "UHECR composition spectrum", "extragalactic UHECR origin"],
  "filters": {"year_min": 2023, "year_max": null, "citation_pref": null, "review_pref": null},
  "limit_suggested": null,
  "ranking_hint": "by_recency",
  "reasoning": "Core=UHECR (anchor, acronym expanded). Facets: system (observatories), phenomenon (composition/spectrum), regime (origin). recency cue -> 2023+"
}

Intent: "Find original papers on cosmic ray positron measurements from PAMELA and AMS-02, covering positron abundance and energy spectra."
Output: {
  "search_terms": ["cosmic ray positron measurements", "PAMELA positron abundance", "AMS-02 positron fraction", "Payload for Antimatter Matter Exploration and Light-nuclei Astrophysics positron abundance", "Alpha Magnetic Spectrometer positron fraction", "PAMELA positron energy spectrum", "AMS-02 positron energy spectrum", "Alpha Magnetic Spectrometer positron energy spectrum"],
  "filters": {"year_min": null, "year_max": null, "citation_pref": null, "review_pref": "off"},
  "limit_suggested": null,
  "ranking_hint": "by_relevance",
  "reasoning": "Core=cosmic ray positron measurements (verbatim shared-phenomenon anchor). Cover each experiment's acronym first with abundance/fraction, then its full name in a separate term, then energy spectrum. These canonical nouns target original measurements, not spectral features or debate. Eight terms including the anchor; never merge entities or name forms. Original papers -> primary work; no date cue -> no year window."
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
# Domain label from the active domain pack (ADR-0003, papervault.domain), substituted at
# import (loader-cached). The facet-taxonomy examples below the label stay space-physics-
# specific (no domain-pack field covers them).
_SYSTEM_PROMPT = _SYSTEM_PROMPT.replace("__DOMAIN_LABEL__", get_domain().label)


def _split_experiment_term(term: str) -> list[str]:
    """Keep known experiments independently searchable with the shared topic.

    Unknown names rely on the prompt rule. Zero/one recognized identity is
    untouched, including multiple aliases referring to the same experiment.
    """
    matches = list(_EXPERIMENT_PATTERN.finditer(term))
    entities: dict[str, str] = {}
    spans: list[tuple[int, int]] = []
    for match in matches:
        entities.setdefault(match.lastgroup, match.group())
        start, end = match.span()
        # Remove wrappers around a name/alias, not parentheses in the topic.
        if term[max(0, start - 1):start] == "(" and term[end:end + 1] == ")":
            start, end = start - 1, end + 1
        spans.append((start, end))
    if len(entities) < 2:
        return [term]

    context = [term[:spans[0][0]]]
    for left, right in zip(spans, spans[1:]):
        gap = term[left[1]:right[0]]
        # Drop only connectors BETWEEN entity names. A topical conjunction
        # such as "electron and positron spectra" remains part of the query.
        if not _ENTITY_SEPARATOR.fullmatch(gap):
            context.append(gap)
    context.append(term[spans[-1][1]:])
    topic = " ".join(" ".join(context).split()).strip(" ,/;&+")
    return [f"{entity} {topic}".strip() for entity in entities.values()]


def _measurement_sibling_terms(term: str) -> list[str]:
    """Keep the parent and add one adjacent measurement-noun variant if eligible."""
    if not _EXPERIMENT_PATTERN.search(term) or _FORBIDDEN_SIBLING_NOUN.search(term):
        return [term]
    noun = _MEASUREMENT_NOUN.search(term)
    if noun is None:
        return [term]
    replacement = _NEXT_MEASUREMENT_NOUN[noun.group().lower()]
    sibling = term[:noun.start()] + replacement + term[noun.end():]
    return [term, sibling]


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
    from ..llm import get_llm, observe_llm_calls

    started = time.monotonic()
    calls = []
    code = "intent_call_failed"
    try:
        with observe_llm_calls(calls.append):
            if llm is None:
                llm = get_llm()
            raw = llm.call([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Intent: {query}\nOutput:"},
            ])
        code = "intent_response_invalid"
        plan = _parse_response(query, raw)
        code = None
        return plan
    finally:
        callback = _intent_observer.get()
        if callback is not None:
            served = calls[-1] if calls else None
            observation = IntentObservation(
                model=served.model if served else "unknown",
                provider=served.provider if served else "unknown",
                latency_ms=(time.monotonic() - started) * 1000,
                degraded=code is not None or any(call.degraded for call in calls),
                code=code,
            )
            try:
                callback(observation)
            except Exception:
                logger.warning("intent observation unavailable")


def _parse_response(query: str, raw: str) -> SearchPlan:

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
    # Strip/filter-empty -> split merged experiments -> noun siblings -> DEDUP -> cap LAST.
    # Dedup BEFORE the cap so a duplicate term never spends a fair-share slot
    # and so the cap never loses a genuine distinct sub-topic to an early dup.
    _stripped = [str(t).strip() for t in search_terms if str(t).strip()]  # NO cap yet
    if not _stripped:
        if not _query_anchor:
            raise ValueError("intent_parser: empty query yields no search terms")
        _stripped = [_query_anchor]
    _expanded = [
        variant
        for term in _stripped
        for part in _split_experiment_term(term)
        for variant in _measurement_sibling_terms(part)
    ]
    _seen: set = set()
    _distinct = [t for t in _expanded if not (t in _seen or _seen.add(t))]  # order-preserving dedup
    search_terms = _distinct[:MAX_SEARCH_TERMS]  # cap LAST — up to 8 DISTINCT terms

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

    plan: SearchPlan = {
        "search_terms": search_terms,
        "filters": filters,
        "limit_suggested": limit_suggested,
        "ranking_hint": ranking_hint,
        "reasoning": reasoning,
    }
    entities = parsed.get("named_entities")
    if isinstance(entities, list):
        names = list(dict.fromkeys(
            name.strip() for name in entities if isinstance(name, str) and name.strip()
            and len(name.strip()) <= 256 and _named_entity_match(name.strip(), query)
        ))[:MAX_SEARCH_TERMS]
        if names:
            plan["named_entities"] = names
    return plan
