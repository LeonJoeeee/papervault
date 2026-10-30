"""Intent normalization keeps experiment-specific searches independently retrievable."""

import json

import pytest

from papervault.library.services.intent_parser import parse_intent


def test_intent_observation_marks_undisclosed_serving_metadata_unknown():
    from papervault.library.services.intent_parser import observe_intent

    observations = []
    client = FakeLLM(["cosmic ray transport"])
    client.model = "configured-model-is-not-evidence"
    client.provider = "configured-provider-is-not-evidence"
    with observe_intent(observations.append):
        plan = parse_intent("Find cosmic ray transport", client)

    assert set(plan) == {"search_terms", "kg_concepts", "filters", "limit_suggested", "ranking_hint", "reasoning"}
    assert len(observations) == 1
    receipt = observations[0]
    assert receipt.model == receipt.provider == "unknown"
    assert receipt.latency_ms >= 0
    assert receipt.degraded is False
    assert receipt.code is None


@pytest.mark.parametrize("failure,code", [
    (RuntimeError("secret credential and response body"), "intent_call_failed"),
    ("private provider response without JSON", "intent_response_invalid"),
])
def test_intent_observation_records_errors_without_sensitive_bodies(failure, code):
    from dataclasses import asdict
    from papervault.library.services.intent_parser import observe_intent

    class BrokenLLM:
        def call(self, messages):
            if isinstance(failure, Exception):
                raise failure
            return failure

    observations = []
    with observe_intent(observations.append), pytest.raises((RuntimeError, ValueError)):
        parse_intent("a private research question", BrokenLLM())

    assert len(observations) == 1
    receipt = observations[0]
    assert receipt.degraded is True
    assert receipt.code == code
    assert receipt.model == receipt.provider == "unknown"
    assert "secret" not in str(asdict(receipt))
    assert "private" not in str(asdict(receipt))


def test_named_entity_groups_preserve_instrument_aliases_and_held_out_names():
    from papervault.library.services.intent_parser import named_entity_groups

    class NamedLLM:
        def call(self, messages):
            return json.dumps({
                "search_terms": ["cosmic ray measurements"], "filters": {},
                "named_entities": ["Voyager 1", "AMS-02", "not actually requested"],
            })

    query = "Compare Voyager 1 and AMS-02 cosmic ray measurements"
    plan = parse_intent(query, NamedLLM())
    assert plan["named_entities"] == ["Voyager 1", "AMS-02"]
    groups = named_entity_groups(query, plan)
    assert ["AMS-02", "AMS02", "AMS 02", "Alpha Magnetic Spectrometer"] in groups
    assert ["Voyager 1"] in groups
    assert len(groups) == 2


def test_named_entity_groups_fall_back_to_known_query_experiments():
    from papervault.library.services.intent_parser import named_entity_groups

    query = "PAMELA and Alpha Magnetic Spectrometer positron spectra with PINN and MHD"
    plan = parse_intent(query, FakeLLM(["positron spectra"]))
    groups = named_entity_groups(query, plan)
    assert len(groups) == 2
    assert groups[0][0] == "PAMELA"
    assert groups[1][0] == "AMS-02"


def test_named_entities_malformed_response_does_not_change_existing_plan():
    class NamedLLM:
        def call(self, messages):
            return json.dumps({"search_terms": ["solar wind"], "named_entities": "ACE"})

    plan = parse_intent("ACE solar wind", NamedLLM())
    assert "named_entities" not in plan


class FakeLLM:
    def __init__(self, terms):
        self.terms = terms

    def call(self, messages):
        return json.dumps({"search_terms": self.terms, "filters": {}, "reasoning": ""})


@pytest.mark.parametrize("term,expected", [
    ("PAMELA AMS-02 positron spectrum",
     ["PAMELA positron spectrum", "PAMELA positron abundance",
      "AMS-02 positron spectrum", "AMS-02 positron abundance"]),
    ("AMS-02 PAMELA positron spectrum",
     ["AMS-02 positron spectrum", "AMS-02 positron abundance",
      "PAMELA positron spectrum", "PAMELA positron abundance"]),
    ("PAMELA and AMS-02 electron and positron spectra",
     ["PAMELA electron and positron spectra", "AMS-02 electron and positron spectra"]),
    ("PAMELA/AMS-02 positron fraction",
     ["PAMELA positron fraction", "PAMELA positron flux",
      "AMS-02 positron fraction", "AMS-02 positron flux"]),
    ("PAMELA vs. AMS-02 positron spectrum",
     ["PAMELA positron spectrum", "PAMELA positron abundance",
      "AMS-02 positron spectrum", "AMS-02 positron abundance"]),
    ("pamela ams02 positron spectrum",
     ["pamela positron spectrum", "pamela positron abundance",
      "ams02 positron spectrum", "ams02 positron abundance"]),
    ("PAMELA Alpha Magnetic Spectrometer positron spectrum",
     ["PAMELA positron spectrum", "PAMELA positron abundance",
      "Alpha Magnetic Spectrometer positron spectrum",
      "Alpha Magnetic Spectrometer positron abundance"]),
    ("Pierre Auger Observatory and Telescope Array cosmic ray spectrum",
     ["Pierre Auger Observatory cosmic ray spectrum", "Pierre Auger Observatory cosmic ray abundance",
      "Telescope Array cosmic ray spectrum", "Telescope Array cosmic ray abundance"]),
    ("AMS-02 Alpha Magnetic Spectrometer PAMELA positron spectrum",
     ["AMS-02 positron spectrum", "AMS-02 positron abundance",
      "PAMELA positron spectrum", "PAMELA positron abundance"]),
    ("positron spectrum PAMELA AMS-02",
     ["PAMELA positron spectrum", "PAMELA positron abundance",
      "AMS-02 positron spectrum", "AMS-02 positron abundance"]),
    ("PAMELA AMS-02 (electron and positron) spectra",
     ["PAMELA (electron and positron) spectra", "AMS-02 (electron and positron) spectra"]),
])
def test_parse_intent_splits_merged_experiment_terms(term, expected):
    plan = parse_intent("What did the named experiments measure?", FakeLLM([term]))
    assert plan["search_terms"] == expected


@pytest.mark.parametrize("term,sibling", [
    ("PAMELA positron spectrum", "PAMELA positron abundance"),
    ("PAMELA positron abundance", "PAMELA positron fraction"),
    ("PAMELA positron fraction", "PAMELA positron flux"),
    ("PAMELA positron flux", "PAMELA positron measurement"),
    ("PAMELA positron measurement", "PAMELA positron spectrum"),
    ("AMS-02 positron fraction", "AMS-02 positron flux"),
    ("Alpha Magnetic Spectrometer positron spectrum",
     "Alpha Magnetic Spectrometer positron abundance"),
    ("Alpha Magnetic Spectrometer AMS-02 positron fraction",
     "Alpha Magnetic Spectrometer AMS-02 positron flux"),
    ("Pierre Auger Observatory cosmic ray spectrum", "Pierre Auger Observatory cosmic ray abundance"),
    ("XPAMELA AMS-02 positron spectrum", "XPAMELA AMS-02 positron abundance"),
    ("PAMELA AMS-020 positron spectrum", "PAMELA AMS-020 positron abundance"),
    ("pAmElA Positron SPECTRUM at EARTH", "pAmElA Positron abundance at EARTH"),
    ("PAMELA flux spectrum flux", "PAMELA measurement spectrum flux"),
    ("positron spectrum from PAMELA", "positron abundance from PAMELA"),
])
def test_parse_intent_adds_one_measurement_sibling_after_unchanged_parent(term, sibling):
    plan = parse_intent("Find measurement papers", FakeLLM([f"  {term}  "]))
    assert plan["search_terms"] == [term, sibling]


@pytest.mark.parametrize("term", [
    "cosmic-ray positron excess",
    "cosmic-ray positron spectrum",
    "XPAMELA positron spectrum",
    "AMS-020 positron spectrum",
    "Parker Solar Probe SEP measurements",
    "PINN MHD SEP transport",
    "PAMELA spectral measurements",
    "PAMELA spectrums measurement_method",
])
def test_parse_intent_leaves_terms_without_entity_or_canonical_noun_untouched(term):
    plan = parse_intent("Find measurement papers", FakeLLM([f"  {term}  "]))
    assert plan["search_terms"] == [term]


@pytest.mark.parametrize("noun", ["fractİon", "fractıon", "ſpectrum", "spectrumé"])
@pytest.mark.parametrize("use_query_fallback", [False, True])
def test_parse_intent_preserves_noncanonical_unicode_nouns(noun, use_query_fallback):
    term = f"PAMELA positron {noun}"
    plan = parse_intent(term, FakeLLM([] if use_query_fallback else [term]))
    assert plan["search_terms"] == [term]


@pytest.mark.parametrize("wording", [
    "feature", "features", "spectral feature", "excess", "hint", "FEATURES", "excesses", "hints",
])
def test_parse_intent_does_not_add_siblings_with_forbidden_nouns(wording):
    term = f"PAMELA positron spectrum {wording}"
    plan = parse_intent("Find measurement papers", FakeLLM([term]))
    assert plan["search_terms"] == [term]


@pytest.mark.parametrize("wording", [
    "feature", "features", "spectral feature", "excess", "hint", "debate",
])
def test_parse_intent_leaves_measurement_wording_to_llm(wording):
    plan = parse_intent("Find original positron measurement papers", FakeLLM([
        "cosmic ray positron measurements",
        f"PAMELA positron {wording}",
        f"AMS-02 positron {wording}",
        f"Alpha Magnetic Spectrometer positron {wording}",
    ]))
    assert plan["search_terms"] == [
        "cosmic ray positron measurements",
        f"PAMELA positron {wording}",
        f"AMS-02 positron {wording}",
        f"Alpha Magnetic Spectrometer positron {wording}",
    ]


def test_parse_intent_splits_before_deduplication_and_cap():
    plan = parse_intent("What did PAMELA and AMS-02 measure?", FakeLLM([
        "  cosmic ray positron  ",
        "  AMS-02 PAMELA positron spectrum  ",
        "",
        "PAMELA positron spectrum",
        "AMS-02 positron spectrum",
        "solar modulation",
        "charge sign dependence",
        "interstellar flux",
        "cosmic ray diffusion",
        "geomagnetic cutoff",
        "overflow facet",
    ]))
    assert plan["search_terms"] == [
        "cosmic ray positron",
        "AMS-02 positron spectrum",
        "AMS-02 positron abundance",
        "PAMELA positron spectrum",
        "PAMELA positron abundance",
        "solar modulation",
        "charge sign dependence",
        "interstellar flux",
    ]


def test_parse_intent_drops_alias_parentheses_before_deduplication():
    plan = parse_intent("What did PAMELA and AMS-02 measure?", FakeLLM([
        "PAMELA and AMS-02 (Alpha Magnetic Spectrometer) positron spectrum",
        "PAMELA positron spectrum",
        "AMS-02 positron spectrum",
    ]))
    assert plan["search_terms"] == [
        "PAMELA positron spectrum", "PAMELA positron abundance",
        "AMS-02 positron spectrum", "AMS-02 positron abundance",
    ]


def test_parse_intent_deduplicates_siblings_before_cap():
    plan = parse_intent("Find measurement papers", FakeLLM([
        "PAMELA positron spectrum",
        "PAMELA positron abundance",
        "PAMELA positron spectrum",
        "PAMELA positron abundance",
        "solar modulation",
        "charge sign dependence",
        "interstellar flux",
        "cosmic ray diffusion",
        "geomagnetic cutoff",
        "overflow facet",
    ]))
    assert plan["search_terms"] == [
        "PAMELA positron spectrum",
        "PAMELA positron abundance",
        "PAMELA positron fraction",
        "solar modulation",
        "charge sign dependence",
        "interstellar flux",
        "cosmic ray diffusion",
        "geomagnetic cutoff",
    ]


def test_parse_intent_caps_expanded_terms_in_entity_order():
    plan = parse_intent("What did PAMELA and AMS-02 measure?", FakeLLM([
        "cosmic ray positron",
        "solar modulation",
        "charge sign dependence",
        "interstellar flux",
        "cosmic ray diffusion",
        "geomagnetic cutoff",
        "heliosphere transport",
        "AMS-02 PAMELA positron spectrum",
    ]))
    assert plan["search_terms"] == [
        "cosmic ray positron",
        "solar modulation",
        "charge sign dependence",
        "interstellar flux",
        "cosmic ray diffusion",
        "geomagnetic cutoff",
        "heliosphere transport",
        "AMS-02 positron spectrum",
    ]


@pytest.mark.parametrize("padding,expected_tail", [
    (6, ["PAMELA positron spectrum", "PAMELA positron abundance"]),
    (7, ["PAMELA positron spectrum"]),
])
def test_parse_intent_cap_prioritizes_parent_before_sibling(padding, expected_tail):
    facets = [
        "cosmic ray positron", "solar modulation", "charge sign dependence",
        "interstellar flux", "cosmic ray diffusion", "geomagnetic cutoff", "heliosphere transport",
    ][:padding]
    plan = parse_intent("Find measurement papers", FakeLLM(facets + ["PAMELA positron spectrum"]))
    assert plan["search_terms"] == facets + expected_tail


@pytest.mark.parametrize("terms", [[], [" ", "\t"]])
def test_parse_intent_splits_merged_query_fallback(terms):
    plan = parse_intent("  PAMELA AMS-02 positron spectrum  ", FakeLLM(terms))
    assert plan["search_terms"] == [
        "PAMELA positron spectrum", "PAMELA positron abundance",
        "AMS-02 positron spectrum", "AMS-02 positron abundance",
    ]


@pytest.mark.parametrize("query", [
    "current understanding of cosmic ray transport",
    "Current Observational Picture of solar energetic particles",
    "current picture of magnetic reconnection",
    "current state of heliospheric modeling",
    "state of the art in particle acceleration",
    "an overview of solar wind turbulence",
    "related-work on cosmic rays",
    "related work on heliospheric transport",
    "an introduction to space weather",
    "a survey of magnetic reconnection",
    "canonical formulations and the more recent model implementations",
    "largest historical events",
    "FOUNDATIONAL work and latest progress",
    "classical transport models",
    "classic acceleration papers",
    "seminal work on cosmic rays",
])
@pytest.mark.parametrize("hint", ["by_recency", "by_relevance", "by_importance"])
def test_intent_clears_false_positive_recency_cut(query, hint):
    class FalseRecencyLLM:
        def call(self, messages):
            return json.dumps({
                "search_terms": ["cosmic ray transport"],
                "filters": {"year_min": "2020", "year_max": None, "review_pref": "prefer"},
                "ranking_hint": hint,
            })

    plan = parse_intent(query, FalseRecencyLLM())
    assert plan["filters"] == {"year_min": None, "year_max": None, "review_pref": "prefer"}
    assert plan["ranking_hint"] == ("by_relevance" if hint == "by_recency" else hint)


@pytest.mark.parametrize("query,year_min,year_max", [
    ("recent work on solar wind", 2020, None),
    ("latest progress in cosmic ray transport", 2023, None),
    ("recent overview of solar wind", 2020, None),
    ("state of the art in 2024", 2024, None),
    ("survey of the last five years", 2021, None),
    ("overview of the past decade", 2016, None),
    ("overview of the newest results", 2023, None),
    ("current understanding this year", 2026, None),
    ("overview of historical events between 1980 and 1990", 1980, 1990),
    ("classical acceleration papers from 1960", 1960, 1960),
    ("contemporary particle acceleration", 2020, None),
])
def test_intent_preserves_explicit_recency_and_closed_year_windows(query, year_min, year_max):
    class DatedLLM:
        def call(self, messages):
            return json.dumps({"search_terms": ["solar wind"],
                               "filters": {"year_min": year_min, "year_max": year_max},
                               "ranking_hint": "by_recency"})

    plan = parse_intent(query, DatedLLM())
    assert plan["filters"] == {"year_min": year_min, "year_max": year_max}
    assert plan["ranking_hint"] == "by_recency"


@pytest.mark.parametrize(
    "override",
    [None, "gpt-6.1-terra@high", "gpt-6-astra@xhigh", "glm-5.3,gpt-6.1-terra@high"],
)
def test_intent_default_client_uses_its_dedicated_route(monkeypatch, override):
    from papervault.library import llm

    monkeypatch.delenv("PAPERVAULT_LLM_INTENT", raising=False)
    if override:
        monkeypatch.setenv("PAPERVAULT_LLM_INTENT", override)
    requested = []

    def get_client(*, model=None):
        requested.append(model)
        return FakeLLM(["solar modulation"])

    monkeypatch.setattr(llm, "get_llm", get_client)
    assert parse_intent("solar modulation")["search_terms"] == ["solar modulation"]
    assert requested == [override or "gpt-6.1-terra@high"]


def test_injected_intent_client_bypasses_route_resolution(monkeypatch):
    monkeypatch.setenv("PAPERVAULT_LLM_INTENT", "flash")
    assert parse_intent("solar modulation", FakeLLM(["solar modulation"]))["search_terms"] == ["solar modulation"]


def test_kg_concepts_are_independent_normalized_canonical_terms():
    class ConceptLLM:
        def call(self, messages):
            return json.dumps({
                "search_terms": ["PINN SEP", "PAMELA positron spectrum"],
                "kg_concepts": ["  Parker transport equation  ", "", None, 3,
                                "physics-informed neural networks", "Parker transport equation"],
            })

    plan = parse_intent("Find transport methods", ConceptLLM())
    assert plan["kg_concepts"] == ["Parker transport equation", "physics-informed neural networks"]
    assert plan["search_terms"] == ["PINN SEP", "PAMELA positron spectrum", "PAMELA positron abundance"]


@pytest.mark.parametrize("concepts", [None, "solar modulation", {}, [], [None, 3, " "]])
def test_invalid_or_empty_kg_concepts_fall_back_to_final_search_terms(concepts):
    class ConceptLLM:
        def call(self, messages):
            return json.dumps({"search_terms": ["PAMELA AMS-02 positron spectrum"],
                               "kg_concepts": concepts})

    plan = parse_intent("positron measurements", ConceptLLM())
    assert plan["kg_concepts"] == ["PAMELA positron spectrum", "PAMELA positron abundance",
                                   "AMS-02 positron spectrum", "AMS-02 positron abundance"]


def test_missing_kg_concepts_fall_back_to_query_when_terms_are_missing():
    assert parse_intent("  solar modulation  ", FakeLLM([]))["kg_concepts"] == ["solar modulation"]


def test_kg_concepts_are_bounded_after_deduplication():
    class ConceptLLM:
        def call(self, messages):
            return json.dumps({"search_terms": ["solar modulation"],
                               "kg_concepts": ["concept 0"] * 8 + [f"concept {i}" for i in range(1, 10)]})

    assert parse_intent("solar modulation", ConceptLLM())["kg_concepts"] == [
        "concept 0", "concept 1", "concept 2", "concept 3", "concept 4", "concept 5", "concept 6", "concept 7",
    ]
