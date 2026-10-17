"""Intent normalization keeps experiment-specific searches independently retrievable."""

import json

import pytest

from papervault.library.services.intent_parser import parse_intent


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
