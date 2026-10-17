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
     ["PAMELA positron spectrum", "AMS-02 positron spectrum"]),
    ("AMS-02 PAMELA positron spectrum",
     ["AMS-02 positron spectrum", "PAMELA positron spectrum"]),
    ("PAMELA and AMS-02 electron and positron spectra",
     ["PAMELA electron and positron spectra", "AMS-02 electron and positron spectra"]),
    ("PAMELA/AMS-02 positron fraction",
     ["PAMELA positron fraction", "AMS-02 positron fraction"]),
    ("PAMELA vs. AMS-02 positron spectrum",
     ["PAMELA positron spectrum", "AMS-02 positron spectrum"]),
    ("pamela ams02 positron spectrum",
     ["pamela positron spectrum", "ams02 positron spectrum"]),
    ("PAMELA Alpha Magnetic Spectrometer positron spectrum",
     ["PAMELA positron spectrum", "Alpha Magnetic Spectrometer positron spectrum"]),
    ("Pierre Auger Observatory and Telescope Array cosmic ray spectrum",
     ["Pierre Auger Observatory cosmic ray spectrum", "Telescope Array cosmic ray spectrum"]),
    ("AMS-02 Alpha Magnetic Spectrometer PAMELA positron spectrum",
     ["AMS-02 positron spectrum", "PAMELA positron spectrum"]),
    ("positron spectrum PAMELA AMS-02",
     ["PAMELA positron spectrum", "AMS-02 positron spectrum"]),
    ("PAMELA AMS-02 (electron and positron) spectra",
     ["PAMELA (electron and positron) spectra", "AMS-02 (electron and positron) spectra"]),
])
def test_parse_intent_splits_merged_experiment_terms(term, expected):
    plan = parse_intent("What did the named experiments measure?", FakeLLM([term]))
    assert plan["search_terms"] == expected


@pytest.mark.parametrize("term", [
    "PAMELA positron spectrum",
    "AMS-02 positron fraction",
    "Alpha Magnetic Spectrometer AMS-02 positron fraction",
    "Parker Solar Probe SEP measurements",
    "Pierre Auger Observatory cosmic ray spectrum",
    "PINN MHD SEP transport",
    "XPAMELA AMS-02 positron spectrum",
    "PAMELA AMS-020 positron spectrum",
])
def test_parse_intent_leaves_single_entity_and_non_entity_terms_untouched(term):
    plan = parse_intent("Find measurement papers", FakeLLM([f"  {term}  "]))
    assert plan["search_terms"] == [term]


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
        "PAMELA positron spectrum",
        "solar modulation",
        "charge sign dependence",
        "interstellar flux",
        "cosmic ray diffusion",
        "geomagnetic cutoff",
    ]


def test_parse_intent_drops_alias_parentheses_before_deduplication():
    plan = parse_intent("What did PAMELA and AMS-02 measure?", FakeLLM([
        "PAMELA and AMS-02 (Alpha Magnetic Spectrometer) positron spectrum",
        "PAMELA positron spectrum",
        "AMS-02 positron spectrum",
    ]))
    assert plan["search_terms"] == ["PAMELA positron spectrum", "AMS-02 positron spectrum"]


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


@pytest.mark.parametrize("terms", [[], [" ", "\t"]])
def test_parse_intent_splits_merged_query_fallback(terms):
    plan = parse_intent("  PAMELA AMS-02 positron spectrum  ", FakeLLM(terms))
    assert plan["search_terms"] == ["PAMELA positron spectrum", "AMS-02 positron spectrum"]
