"""Identity ladder and P2 fixtures from the accepted acquisition spec, A4."""

import importlib

import pytest

from papervault.library.models import Paper, SourceIdentity


def v2():
    return importlib.import_module("papervault.library.identity_v2")


TITLE = "The transport of cosmic rays in the heliosheath"
REFERENCE = {"title": TITLE, "authors": ["R. D. Strauss"], "year": 2012}


@pytest.mark.parametrize("value", [
    "10.1234/ABC", "doi:10.1234/ABC", "https://doi.org/10.1234/ABC",
    "http://dx.doi.org/10.1234/ABC", "dx.doi.org/10.1234%2FABC",
])
def test_canonical_doi_equivalence(value):
    module = v2()
    assert module.canonical_doi(value) == "10.1234/abc"
    decision = module.correspondence_decision(value, "https://doi.org/10.1234/abc")
    assert decision.outcome == module.Outcome.RESOLVED
    assert decision.rejection is None


@pytest.mark.parametrize("namespace,identifier,want", [
    ("core:work", "123", "core:work:123"),
    ("core:output", "123", "core:output:123"),
    ("arxiv:paper", "1201.12345v2", "arxiv:1201.12345"),
    ("inspire:literature", "123", "inspire:123"),
    ("semantic_scholar:paper", "abc", "semantic_scholar:abc"),
])
def test_namespace_format_and_parse_roundtrip(namespace, identifier, want):
    module = v2()
    assert module.format_namespace(namespace, identifier) == want
    parsed = module.parse_namespace(want)
    assert parsed is not None and parsed.key == want


def test_core_work_output_collision_remains_distinct():
    module = v2()
    work, output = [module.parse_namespace(f"core:{kind}:123") for kind in ("work", "output")]
    assert work != output
    assert len({work, output}) == 2
    observations = [SourceIdentity(source="core", source_ids={f"core:{kind}": ["123"]})
                    for kind in ("work", "output")]
    choices = [module.select_canonical({"source_identities": [item]}) for item in observations]
    assert [item.key for item in choices] == ["core:work:123", "core:output:123"]
    assert choices[0] != choices[1]


@pytest.mark.parametrize("value", ["core:123", "core:works:123", "core:output:0",
                                        "core:work:-1", "inspire:abc", "unknown:123"])
def test_ambiguous_or_invalid_namespaces_are_not_identities(value):
    assert v2().parse_namespace(value) is None


@pytest.mark.parametrize("record,key,rung", [
    ({"doi": "doi:10.1234/ABC", "arxiv_id": "1201.12345"}, "10.1234/abc", 0),
    ({"arxiv_id": "1201.12345", "url": "https://example.org/a"}, "arxiv:1201.12345", 1),
    ({"inspire_record_id": "123"}, "inspire:123", 1),
    ({"source": "core", "paper_id": "123"}, "core:work:123", 1),
    ({"source": "semantic_scholar", "paper_id": "abc"}, "semantic_scholar:abc", 1),
    ({"paper_id": "123", "url": "https://example.org/a"}, "https://example.org/a", 2),
    ({"doi": "garbage", "url": "https://example.org/a"}, "https://example.org/a", 2),
    ({"url": "https://doi.org/10.1234/a"}, "https://doi.org/10.1234/a", 2),
])
def test_rung_classification_and_canonical_selection(record, key, rung):
    module = v2()
    assert module.select_canonical(record).key == key
    assert module.classify_rung(record) == rung
    assert module.select_canonical(Paper(key="K", **record)).key == key


def test_observed_doi_wins_and_lower_rungs_never_replace_current():
    module = v2()
    observation = SourceIdentity(source="core", source_ids={"core:work": ["123"]},
                                 dois=["10.1234/ABC"], urls=["https://example.org/a"])
    current = module.select_canonical({"source_identities": [observation]})
    assert current.key == "10.1234/abc"
    assert module.select_canonical({"source": "core", "paper_id": "123"}, current=current) == current
    assert module.select_canonical({"doi": "10.1234/different"}, current=current) == current


@pytest.mark.parametrize("value", [None, {}, 1, [], {"source_identities": [None, 3]},
                                        {"doi": [], "url": "http://[invalid"}])
def test_selection_is_total_for_missing_and_malformed_inputs(value):
    assert v2().select_canonical(value) is None
    assert v2().classify_rung(value) is None


def test_doi_url_contradiction_is_a_typed_route_rejection():
    module = v2()
    decision = module.correspondence_decision("10.1234/a", "https://doi.org/10.1234/b")
    assert decision.outcome == module.Outcome.CONTRADICTED
    assert decision.reason == module.Reason.DOI_MISMATCH
    assert decision.rejection == module.RejectionKind.ROUTE_REJECTED
    assert decision.rejection != module.RejectionKind.IDENTITY_REJECTED
    assert decision.rejection != module.RejectionKind.EXHAUSTED
    second = module.correspondence_decision("10.1234/a", "https://doi.org/10.1234/a")
    assert second.outcome == module.Outcome.RESOLVED


@pytest.mark.parametrize("field,value,reason", [
    ("doi", "10.1234/other", "doi_mismatch"),
    ("title", "Galactic Cosmic Rays in the Dynamic Heliosphere", "title_mismatch"),
    ("authors", ["M. S. Potgieter"], "author_mismatch"),
    ("year", 2015, "year_mismatch"),
])
def test_positive_metadata_contradictions_reject_only_the_route(field, value, reason):
    observed = {**REFERENCE, "doi": "10.1234/a", field: value}
    decision = v2().correspondence_decision("10.1234/a", "https://doi.org/10.1234/a",
                                           reference=REFERENCE, observed=observed)
    assert decision.outcome.value == "contradicted"
    assert decision.reason.value == reason
    assert decision.rejection.value == "route_rejected"


@pytest.mark.parametrize("year", [2011, 2012, 2013])
def test_bibliography_and_registry_evidence_resolve_an_opaque_route(year):
    decision = v2().correspondence_decision("10.1234/a", "https://publisher.example/item",
        reference=REFERENCE, observed={**REFERENCE, "doi": "10.1234/a", "year": year})
    assert decision.outcome.value == "resolved"


@pytest.mark.parametrize("observed", [None, {}, {"title": TITLE},
    {**REFERENCE, "authors": []}, {**REFERENCE, "year": None},
    {**REFERENCE, "doi": "garbage"}, {"title": "Short", "authors": ["Strauss"]}])
def test_incomplete_metadata_is_retryable_not_contradiction(observed):
    decision = v2().correspondence_decision("10.1234/a", "https://publisher.example/item",
                                           reference=REFERENCE, observed=observed)
    assert decision.outcome.value == "unresolved"
    assert decision.rejection is None


@pytest.mark.parametrize("value", [None, 1, [], {}, "garbage", "10.1/a"])
def test_correspondence_is_total_for_invalid_doi(value):
    assert v2().canonical_doi(value) == ""
    assert v2().correspondence_decision(value, "https://doi.org/10.1234/a").outcome.value == "unresolved"


@pytest.mark.parametrize("url", [None, [], "http://[invalid", "https://doi.org.evil/10.1234/a",
                                      "https://user@doi.org/10.1234/a", "file:///10.1234/a"])
def test_malformed_or_spoofed_urls_cannot_establish_correspondence(url):
    assert v2().correspondence_decision("10.1234/a", url).outcome.value == "unresolved"


@pytest.mark.parametrize("url", ["https://journals.aps.org/prl/abstract/10.1234/wrong",
                                      "https://link.springer.com/article/10.1234/wrong/pdf"])
def test_explicit_publisher_doi_contradiction_rejects_route(url):
    decision = v2().correspondence_decision("10.1234/a", url, reference=REFERENCE,
                                           observed={**REFERENCE, "doi": "10.1234/a"})
    assert decision.outcome.value == "contradicted"
    assert decision.reason.value == "doi_mismatch"


@pytest.mark.parametrize("field,value", [("authors", 1), ("authors", [3]),
                                               ("year", True), ("year", "2012")])
def test_malformed_supplied_metadata_cannot_resolve(field, value):
    reference = {**REFERENCE, "year": None}
    observed = {**REFERENCE, "doi": "10.1234/a", field: value}
    assert v2().correspondence_decision("10.1234/a", "https://doi.org/10.1234/a",
        reference=reference, observed=observed).outcome.value == "unresolved"


@pytest.mark.parametrize("candidate_authors,outcome", [
    (["Smith", "Jones"], "resolved"),  # .5 surname Jaccard, reference first included
    (["Smith", "Jones", "Brown"], "contradicted"),
    (["Jones"], "contradicted"),
])
def test_author_overlap_threshold_and_first_surname(candidate_authors, outcome):
    reference = {**REFERENCE, "authors": ["A. Smith"]}
    observed = {**REFERENCE, "doi": "10.1234/a", "authors": candidate_authors}
    assert v2().correspondence_decision("10.1234/a", "https://publisher.example/item",
        reference=reference, observed=observed).outcome.value == outcome


def test_same_rung_selection_is_independent_of_observation_order():
    observations = [SourceIdentity(source="core", source_ids={f"core:{kind}": ["123"]})
                    for kind in ("work", "output")]
    assert v2().select_canonical({"source_identities": observations}).key == "core:output:123"
    assert v2().select_canonical({"source_identities": observations[::-1]}).key == "core:output:123"


def test_existing_canonical_namespace_is_normalized_before_selection():
    module = v2()
    current = module.CanonicalIdentity(module.Rung.SOURCE, "core:work:00123")
    assert module.select_canonical({"url": "https://example.org/item"}, current=current).key == "core:work:123"
