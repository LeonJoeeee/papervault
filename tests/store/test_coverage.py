"""Coverage must reject convincing evidence for a different version or an old observation."""
import copy
from datetime import datetime, timezone

import pytest

NOW = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
IDENTITY = {"unit_id": "paper:A", "consumed_input_revision": "i1", "recipe_digest": "r1",
            "generation_id": "g1", "attempt_id": "a1", "submission_id": "s1",
            "authority_epoch": 1, "fencing_token": 1}


def row():
    output = {"role": "graph", "store": "neo4j", "output_id": "graph:a1",
              "namespace": "private-A",
              "verification": {"state": "present"}}
    return {"unit": {"unit_id": "paper:A", "kind": "paper", "record_key": "A", "quarantine": False},
            "record": {"key": "A", "title": "Test title", "doi": "10.1/a", "policy": "clear", "merge": "canonical"},
            "requirement": {**IDENTITY, "record_revision": 1, "inventory_revision": "whole1"},
            "recipe": {"output_contract": {"fulltext": {"required": ["graph"], "not_applicable": []}}},
            "input_manifest": {"metadata_revision": 1, "artifacts": [
                {"artifact_id": "text", "kind": "fulltext", "sha256": "a" * 64, "byte_length": 12}]},
            "selected_derivation": {**IDENTITY, "derivation_id": "d1", "receipt_id": "receipt1",
                                    "achieved_tier": "fulltext"},
            "receipt_validated": True,
            "receipt": {**IDENTITY, "receipt_id": "receipt1", "namespace": "private-A", "achieved_tier": "fulltext", "sealed": True, "durable": True,
                        "output_inventory": [output]},
            "observations": [{**IDENTITY, **output, "receipt_id": "receipt1", "state": "present",
                              "observed_at": "2026-10-20T11:59:00+00:00",
                              "ingested_at": "2026-10-20T12:00:00+00:00"},
                             {"unit_id": "paper:A", "consumed_input_revision": "i1", "role": "input_evidence",
                              "store": "library", "output_id": "text", "state": "present",
                              "observed_at": "2026-10-20T11:59:00Z", "detail": {
                                  "artifact_id": "text", "sha256": "a" * 64, "byte_length": 12}}],
            "inventories": [{"inventory_revision": "whole1", "parent_evidence_revision": "i1", "mode": "whole",
                "required_units": ["paper:A"], "optional_units": [], "section_mapping_revisions": {}}],
            "jobs": [{"state": "succeeded"}], "publication_history": [{"derivation_id": "d0"}]}


class CatalogRows:
    """Already materialized catalog rows, with no artifact/backend access."""
    def __init__(self, rows):
        self.rows = rows

    def coverage_rows(self, scope=None):
        return copy.deepcopy(self.rows)


def query(rows, **kwargs):
    from papervault.store.coverage import query_coverage
    return query_coverage(CatalogRows(rows), evaluation_time=NOW, **kwargs)


def test_current_exact_selected_publication_and_fresh_roles_are_covered_without_mutation():
    source = row()
    before = copy.deepcopy(source)
    result = query([source])
    unit = result["units"][0]
    assert unit["indexed"] is True
    assert unit["readable"] is True
    assert unit["requested_identity"]["recipe_digest"] == "r1"
    assert unit["selected_identity"]["recipe_digest"] == "r1"
    assert unit["roles"][0]["state"] == "present"
    assert unit["roles"][0]["age_seconds"] == 60
    assert result["records"][0]["indexed"] is True
    assert source == before


@pytest.mark.parametrize(("change", "state"), [
    ({"observed_at": "2026-10-20T11:00:00+00:00"}, "stale"),
    ({"state": "unavailable"}, "unknown"),
    ({"state": "missing"}, "missing"),
    ({"recipe_digest": "wrong"}, "unknown"),
    ({"observed_at": "2026-10-20T12:01:00+00:00"}, "unknown"),
])
def test_wrong_identity_old_positive_and_collector_outage_never_pass(change, state):
    source = row()
    source["observations"][0].update(change)
    unit = query([source])["units"][0]
    assert unit["indexed"] is False
    assert unit["roles"][0]["state"] == state


@pytest.mark.parametrize(("field", "value"), [("namespace", "another-version"), ("namespace", None),
                                               ("receipt_id", "another-receipt"), ("receipt_id", None)])
def test_observation_cannot_borrow_another_namespace_or_receipt(field, value):
    source = row()
    source["observations"][0][field] = value
    unit = query([source])["units"][0]
    assert unit["indexed"] is False
    assert unit["roles"][0]["state"] == "unknown"


def test_requested_recipe_stays_separate_from_selected_and_rejects_publication():
    unit = query([row()], recipe_digest="r2")["units"][0]
    assert unit["requested_identity"]["recipe_digest"] == "r2"
    assert unit["selected_identity"]["recipe_digest"] == "r1"
    assert unit["published"] is False
    assert unit["readable"] is True
    assert "selected_recipe_digest_mismatch" in unit["reasons"]
    assert unit["publication_history"] == [{"derivation_id": "d0"}]


def test_latest_actual_observation_wins_over_later_ingestion_of_old_positive():
    source = row()
    source["observations"].append({**source["observations"][0], "state": "missing",
                                   "observed_at": "2026-10-20T11:59:30+00:00",
                                   "ingested_at": "2026-10-20T11:59:31+00:00"})
    source["observations"][0]["ingested_at"] = "2026-10-20T12:00:00+00:00"
    assert query([source])["units"][0]["roles"][0]["state"] == "missing"


def test_metadata_and_abstract_never_infer_readable_fulltext():
    source = row()
    source["selected_derivation"]["achieved_tier"] = "abstract"
    source["receipt"]["achieved_tier"] = "abstract"
    source["recipe"]["output_contract"]["abstract"] = {"required": ["graph"], "not_applicable": []}
    source["input_manifest"]["artifacts"][0]["kind"] = "abstract"
    unit = query([source])["units"][0]
    assert unit["readable"] is False
    assert unit["indexed"] is False
    assert unit["tier_covered"] is True


def test_graph_empty_requires_receipt_verification_and_required_not_applicable_does_not_pass():
    source = row()
    source["observations"][0]["state"] = "verified_empty"
    assert query([source])["units"][0]["indexed"] is False
    source["receipt"]["output_inventory"][0]["verification"] = {
        "state": "verified_empty", "method": "count", "evidence": {"count": 0, "verified": True}}
    assert query([source])["units"][0]["indexed"] is True
    source["observations"][0]["state"] = "not_applicable"
    assert query([source])["units"][0]["indexed"] is False


def test_missing_contract_or_quarantine_blocks_coverage():
    source = row()
    source["recipe"] = {}
    assert query([source])["units"][0]["indexed"] is False
    source = row()
    source["unit"]["quarantine"] = True
    assert query([source])["units"][0]["indexed"] is False


@pytest.mark.parametrize("contract", [
    {"required": ["graph"]},
    {"required": ["graph"], "not_applicable": None},
    {"required": ["graph"], "not_applicable": "vectors"},
    {"required": ["graph"], "not_applicable": ["graph"]},
])
def test_incomplete_or_conflicting_output_contract_never_receives_coverage_credit(contract):
    source = row()
    source["recipe"]["output_contract"]["fulltext"] = contract
    unit = query([source])["units"][0]
    assert unit["readable"] is True
    assert unit["tier_covered"] is False
    assert unit["indexed"] is False
    assert "missing_or_invalid_recipe_output_contract" in unit["reasons"]


def test_unknown_expected_inventory_prevents_record_completeness():
    source = row()
    source["requirement"]["inventory_revision"] = None
    source["inventories"] = []
    source.pop("inventory", None)
    result = query([source])
    assert result["units"][0]["indexed"] is True
    assert result["records"][0]["indexed"] is False
    assert result["records"][0]["inventory_state"] == "unknown"


@pytest.mark.parametrize("change", ["quarantine", "inventory"])
@pytest.mark.parametrize("parent_kind", ["operator_document", "paper"])
def test_section_serving_eligibility_rechecks_typed_parent_and_current_inventory(change, parent_kind):
    parent, section = row(), row()
    parent_id = "paper:A" if parent_kind == "paper" else "operator:P"
    parent["unit"].update(kind=parent_kind, unit_id=parent_id, record_key="A" if parent_kind == "paper" else None)
    if parent_kind != "paper":
        parent["record"] = None
    section["unit"].update(kind="section", unit_id="section:S", record_key=None)
    section["record"] = None
    for source in (parent, section):
        for obj in (source["requirement"], source["selected_derivation"], source["receipt"], *source["observations"]):
            obj["unit_id"] = source["unit"]["unit_id"]
        source["requirement"]["inventory_revision"] = "v1"
        source["input_manifest"]["parent_evidence_revision"] = "e1"
    section["input_manifest"]["section_definition_revision"] = 1
    section["section_mappings"] = [{"unit_id": "section:S", "parent_unit_id": parent_id,
        "parent_evidence_revision": "e1", "mapping_revision": 1}]
    parent["inventories"] = [{"inventory_revision": "v1", "parent_evidence_revision": "e1", "mode": "sections",
        "required_units": ["section:S"], "optional_units": [], "section_mapping_revisions": {"section:S": 1}}]
    parent["current_inventory"] = {"inventory_revision": "v1", "invalidated": False}
    initial = query([section, parent])
    assert next(unit for unit in initial["units"] if unit["unit_id"] == "section:S")["source_eligible"]
    if parent_kind == "paper":
        assert initial["records"][0]["title"] == parent["record"]["title"]
        assert initial["records"][0]["doi"] == parent["record"]["doi"]
    if change == "quarantine":
        parent["unit"]["quarantine"] = True
    else:
        parent["inventories"].append({**parent["inventories"][0], "inventory_revision": "v2",
            "parent_evidence_revision": "e2", "section_mapping_revisions": {"section:S": 2}})
        parent["current_inventory"]["inventory_revision"] = "v2"
    child = next(unit for unit in query([parent, section])["units"] if unit["unit_id"] == "section:S")
    assert child["source_eligible"] is False


def test_sections_require_complete_exact_inventory_and_parent_evidence_without_double_counting():
    parent = row()
    parent["inventory"] = {"inventory_revision": "v1", "parent_evidence_revision": "i1", "mode": "sections",
        "required_units": ["section:A:1", "section:A:2"], "optional_units": [],
        "section_mapping_revisions": {"section:A:1": 1, "section:A:2": 1}}
    parent["requirement"]["inventory_revision"] = "v1"
    section = row()
    section["unit"].update(unit_id="section:A:1", kind="section")
    for obj in (section["requirement"], section["selected_derivation"], section["receipt"], section["observations"][0], section["observations"][1]):
        obj["unit_id"] = "section:A:1"
    section["requirement"]["inventory_revision"] = "v1"
    section["input_manifest"].update(parent_evidence_revision="i1", section_definition_revision=1)
    section["section_mappings"] = [{"unit_id": "section:A:1", "mapping_revision": 1,
        "parent_unit_id": "paper:A", "parent_evidence_revision": "i1"}]
    result = query([parent, section], expected_inventory_revision="v1")
    assert result["records"][0]["indexed"] is False
    assert result["records"][0]["sections"] == {"required": 2, "complete": 1, "missing": 1,
                                               "stale": 0, "unknown": 0, "excluded": 0}
    assert result["totals"]["registered"] == 1
    del parent["inventory"]
    assert query([parent, section])["records"][0]["sections"]["unknown"] > 0


def test_current_generation_does_not_choose_candidate_or_count_record_twice():
    active, candidate = row(), row()
    active["generation"] = {"generation_id": "g1", "state": "active"}
    candidate["generation"] = {"generation_id": "g2", "state": "candidate"}
    candidate["requirement"]["generation_id"] = "g2"
    candidate["selected_derivation"] = None
    result = query([candidate, active])
    assert len(result["units"]) == 1
    assert result["records"][0]["indexed"] is True
    assert query([candidate, active], generation_id="g2")["units"][0]["requested_identity"]["generation_id"] == "g2"


def test_manifest_must_belong_to_requested_unit_and_exact_input_revision():
    source = row()
    source["input_manifest"] = {"unit_id": "other", "consumed_input_revision": "i1",
                                "manifest": source["input_manifest"]}
    assert query([source])["units"][0]["readable"] is False


def test_catalog_rejected_receipt_cannot_be_published_even_with_valid_looking_json():
    source = row()
    source["receipt_validated"] = False
    assert query([source])["units"][0]["published"] is False


def test_section_inventory_mapping_revision_and_parent_evidence_must_match():
    parent, section = row(), row()
    parent["input_manifest"]["parent_evidence_revision"] = "e1"
    parent["requirement"]["inventory_revision"] = "v1"
    parent["inventories"] = [{"inventory_revision": "v1", "parent_evidence_revision": "e1",
        "mode": "sections", "required_units": ["section:A:1"], "optional_units": [],
        "section_mapping_revisions": {"section:A:1": 1}}]
    section["unit"].update(unit_id="section:A:1", kind="section")
    for obj in (section["requirement"], section["selected_derivation"], section["receipt"], section["observations"][0], section["observations"][1]):
        obj["unit_id"] = "section:A:1"
    section["requirement"]["inventory_revision"] = "v1"
    section["input_manifest"].update(parent_evidence_revision="e1", section_mapping_revision=1)
    section["section_mappings"] = [{"unit_id": "section:A:1", "mapping_revision": 1,
        "parent_unit_id": "paper:A", "parent_evidence_revision": "e1"}]
    assert query([parent, section])["records"][0]["indexed"] is True
    section["input_manifest"]["section_mapping_revision"] = 2
    result = query([parent, section])
    assert result["records"][0]["indexed"] is False
    assert result["records"][0]["sections"]["unknown"] == 1


def test_readability_requires_fresh_exact_bytes_observation_for_the_consumed_input():
    source = row()
    source["observations"] = [obs for obs in source["observations"] if obs["role"] != "input_evidence"]
    assert query([source])["units"][0]["readable"] is False
    source["observations"].append({"unit_id": "paper:A", "consumed_input_revision": "i1",
        "role": "input_evidence", "store": "library", "output_id": "text", "state": "present",
        "observed_at": "2026-10-20T11:59:00Z", "detail": {"artifact_id": "text", "sha256": "b" * 64, "byte_length": 12}})
    assert query([source])["units"][0]["readable"] is False
    source["observations"][-1]["detail"]["sha256"] = "a" * 64
    assert query([source])["units"][0]["readable"] is True
    source["observations"][-1]["observed_at"] = "2026-10-20T11:00:00Z"
    assert query([source])["units"][0]["readable"] is False


def test_current_record_quarantine_policy_and_metadata_revision_block_conforming_coverage():
    for field, value in (("quarantine", True), ("policy", "excluded"), ("revision", 2)):
        source = row()
        source["record"][field] = value
        assert query([source])["units"][0]["indexed"] is False


def test_byte_output_requires_fresh_observation_of_sealed_hash_and_length():
    source = row()
    source["receipt"]["output_inventory"][0].update(sha256="c" * 64, byte_length=7)
    assert query([source])["units"][0]["indexed"] is False
    source["observations"][0]["detail"] = {"sha256": "c" * 64, "byte_length": 7}
    assert query([source])["units"][0]["indexed"] is True
    source["observations"][0]["detail"]["byte_length"] = 8
    assert query([source])["units"][0]["indexed"] is False


def test_selected_unit_requires_every_pinned_aggregate_layer_with_fresh_exact_outputs():
    from papervault.store.repository import payload_hash

    source = row()
    ancestor = copy.deepcopy(source["receipt"])
    ancestor.update(unit_id="paper:B", receipt_id="receiptB", namespace="ancestor")
    ancestor["output_inventory"][0]["output_id"] = "graph:B"
    ancestor["output_inventory"][0]["namespace"] = "ancestor"
    ancestor["achieved_tier"] = "fulltext"
    source["receipt"].update(namespace="own", aggregate_layers=["ancestor", "own"],
        base_aggregate_layers=["ancestor"], selected_unit_receipts={"paper:A": "receipt1", "paper:B": "receiptB"})
    source["receipt"]["selected_unit_receipts_digest"] = payload_hash(source["receipt"]["selected_unit_receipts"])
    source["receipt"]["output_inventory"][0]["namespace"] = "own"
    source["observations"][0]["namespace"] = "own"
    source["aggregate_receipts"] = [{"receipt": ancestor, "recipe": source["recipe"]}]
    source["aggregate_observations"] = [{**IDENTITY, "unit_id": "paper:B", "namespace": "ancestor", "receipt_id": "receiptB",
        "role": "graph", "store": "neo4j", "output_id": "graph:B", "state": "present",
        "observed_at": "2026-10-20T11:00:00Z"}]
    unit = query([source])["units"][0]
    assert unit["indexed"] is False
    assert unit["aggregate_roles"][0]["state"] == "stale"
    source["aggregate_observations"][0]["observed_at"] = "2026-10-20T11:59:00Z"
    assert query([source])["units"][0]["indexed"] is True
    source["aggregate_receipts"] = []
    unit = query([source])["units"][0]
    assert unit["indexed"] is False
    assert "aggregate_layer_receipt_missing" in unit["reasons"]


def test_later_generation_aggregate_layer_also_blocks_an_earlier_selected_unit():
    source = row()
    source["receipt"]["namespace"] = "own"
    source["generation_manifest"] = {"manifest": {"aggregate_layers": ["own", "later"],
        "selected_unit_receipts": {"paper:A": "receipt1", "paper:B": "receiptB"}}}
    assert query([source])["units"][0]["indexed"] is False


def test_generation_aggregate_is_verified_once_and_reported_by_reference(monkeypatch):
    from papervault.store import coverage

    first, second = row(), row()
    second["unit"].update(unit_id="paper:B", record_key="B")
    second["record"].update(key="B")
    for obj in (second["requirement"], second["selected_derivation"], second["receipt"], *second["observations"]):
        obj["unit_id"] = "paper:B"
    second["selected_derivation"]["receipt_id"] = second["receipt"]["receipt_id"] = "receiptB"
    second["receipt"]["namespace"] = "private-B"
    second["receipt"]["output_inventory"][0]["namespace"] = "private-B"
    second["observations"][0].update(namespace="private-B", receipt_id="receiptB")
    third = copy.deepcopy(second)
    third["unit"].update(unit_id="paper:C", record_key="C")
    third["record"].update(key="C")
    for obj in (third["requirement"], third["selected_derivation"], third["receipt"], *third["observations"]):
        obj["unit_id"] = "paper:C"
    third["selected_derivation"]["receipt_id"] = third["receipt"]["receipt_id"] = "receiptC"
    third["receipt"]["namespace"] = third["receipt"]["output_inventory"][0]["namespace"] = "private-C"
    third["observations"][0].update(namespace="private-C", receipt_id="receiptC")
    manifest = {"generation_id": "g1", "manifest_revision": 4, "manifest": {
        "aggregate_layers": ["private-A", "private-B", "private-C"],
        "selected_unit_receipts": {"paper:A": "receipt1", "paper:B": "receiptB", "paper:C": "receiptC"}}}
    ancestors = [{"receipt": item["receipt"], "recipe": item["recipe"]} for item in (first, second, third)]
    observations = first["observations"] + second["observations"] + third["observations"]
    for source in (first, second, third):
        source.update(generation_manifest=manifest, aggregate_receipts=ancestors, aggregate_observations=observations)
    calls = []
    original = coverage._role

    def counted(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(coverage, "_role", counted)
    result = query([first, second, third])
    assert all(unit["indexed"] for unit in result["units"])
    assert len(calls) == 6  # Three selected roles plus three shared physical-layer verifications.
    assert len(result["aggregate_contexts"]) == 1
    assert all(unit["aggregate_roles"] == [] for unit in result["units"])
    assert len({unit["aggregate_context_id"] for unit in result["units"]}) == 1


def test_captured_raw_fulltext_evidence_role_is_readable_but_transformed_abstract_is_not():
    source = row()
    source["input_manifest"]["artifacts"][0].pop("kind")
    source["input_manifest"]["artifacts"][0]["role"] = "fulltext_evidence"
    assert query([source])["units"][0]["readable"] is True
    source["input_manifest"]["artifacts"][0]["role"] = "abstract_metadata"
    assert query([source])["units"][0]["readable"] is False


def test_standalone_operator_documents_remain_distinct_registered_units_and_scope_exact():
    first, second = row(), row()
    for source, unit_id in ((first, "operator:A"), (second, "operator:B")):
        source["unit"].update(unit_id=unit_id, record_key=None, kind="operator_document")
        source["record"] = None
        for obj in (source["requirement"], source["selected_derivation"], source["receipt"], *source["observations"]):
            obj["unit_id"] = unit_id
    result = query([first, second])
    assert result["totals"]["registered"] == 2
    assert {record["key"] for record in result["records"]} == {"operator:A", "operator:B"}
    assert len(query([first, second], scope="operator:A")["units"]) == 1


def test_receipt_requires_explicit_catalog_validation_and_matching_achieved_tier():
    source = row()
    source.pop("receipt_validated", None)
    assert query([source])["units"][0]["published"] is False
    source["receipt_validated"] = True
    source["receipt"]["achieved_tier"] = "metadata"
    assert query([source])["units"][0]["published"] is False


def test_partial_section_lower_tier_does_not_claim_entire_record_coverage():
    parent, section = row(), row()
    parent["inventory"] = {"inventory_revision": "v1", "parent_evidence_revision": "i1", "mode": "sections",
        "required_units": ["section:A:1", "section:A:2"], "optional_units": [],
        "section_mapping_revisions": {"section:A:1": 1, "section:A:2": 1}}
    parent["requirement"]["inventory_revision"] = "v1"
    section["unit"].update(unit_id="section:A:1", kind="section")
    for obj in (section["requirement"], section["selected_derivation"], section["receipt"], *section["observations"]):
        obj["unit_id"] = "section:A:1"
    section["requirement"]["inventory_revision"] = "v1"
    section["input_manifest"].update(parent_evidence_revision="i1", section_definition_revision=1)
    section["section_mappings"] = [{"unit_id": "section:A:1", "mapping_revision": 1,
        "parent_unit_id": "paper:A", "parent_evidence_revision": "i1"}]
    section["selected_derivation"]["achieved_tier"] = section["receipt"]["achieved_tier"] = "abstract"
    section["recipe"]["output_contract"]["abstract"] = {"required": ["graph"], "not_applicable": []}
    record = query([parent, section])["records"][0]
    assert record["tiers"]["abstract"] == 1
    assert record["tier_coverage"]["abstract"] is False
    parent["inventory"].update(inventory_revision="v2", required_units=["section:A:1"])
    parent["requirement"]["inventory_revision"] = section["requirement"]["inventory_revision"] = "v2"
    record = query([parent, section])["records"][0]
    assert record["tier_coverage"]["abstract"] is True
    assert record["indexed"] is False


def test_whole_inventory_counts_lower_tier_parent_without_letting_old_section_inventory_add_credit():
    parent = row()
    parent["requirement"]["inventory_revision"] = "whole1"
    parent["inventories"] = [{"inventory_revision": "whole1", "parent_evidence_revision": "i1", "mode": "whole",
        "required_units": ["paper:A"], "optional_units": [], "section_mapping_revisions": {}}]
    parent["selected_derivation"]["achieved_tier"] = parent["receipt"]["achieved_tier"] = "abstract"
    parent["recipe"]["output_contract"]["abstract"] = {"required": ["graph"], "not_applicable": []}
    record = query([parent])["records"][0]
    assert record["tier_coverage"]["abstract"] is True
    assert record["indexed"] is False


def test_verified_empty_byte_output_still_requires_exact_hash_and_length_observation():
    source = row()
    source["receipt"]["output_inventory"][0].update(sha256="c" * 64, byte_length=0,
        verification={"state": "verified_empty", "method": "count", "evidence": {"count": 0}})
    assert query([source])["units"][0]["indexed"] is False
    source["observations"][0]["detail"] = {"sha256": "c" * 64, "byte_length": 0}
    assert query([source])["units"][0]["indexed"] is True


@pytest.mark.parametrize("age", [float("nan"), float("inf"), float("-inf")])
def test_observation_age_must_be_finite(age):
    with pytest.raises(ValueError, match="finite"):
        query([row()], max_observation_age=age)


def operator_sections():
    parent, section = row(), row()
    parent["unit"].update(unit_id="operator:O", record_key=None, kind="operator_document")
    parent["record"] = None
    parent["requirement"].update(unit_id="operator:O", recipe_digest=None, inventory_revision="v1", inventory_parent_selector=True)
    parent["selected_derivation"] = parent["receipt"] = parent["recipe"] = None
    parent["receipt_validated"] = False
    parent["input_manifest"]["parent_evidence_revision"] = "e1"
    for obs in parent["observations"]:
        obs["unit_id"] = "operator:O"
    inventory = {"parent_unit_id": "operator:O", "inventory_revision": "v1", "mode": "sections",
        "parent_evidence_revision": "e1", "parent_consumed_input_revision": "i1",
        "required_units": ["section:S"], "optional_units": [], "section_mapping_revisions": {"section:S": 1}}
    parent["inventories"] = [inventory]
    parent["current_inventory"] = {"parent_unit_id": "operator:O", "generation_id": "g1", "inventory_revision": "v1", "invalidated": False}
    section["unit"].update(unit_id="section:S", record_key=None, kind="section")
    section["record"] = None
    for obj in (section["requirement"], section["selected_derivation"], section["receipt"], *section["observations"]):
        obj.update(unit_id="section:S", consumed_input_revision="iS")
    section["requirement"]["inventory_revision"] = "v1"
    section["input_manifest"].update(parent_evidence_revision="e1", section_definition_revision=1)
    section["section_mappings"] = [{"unit_id": "section:S", "mapping_revision": 1,
        "parent_unit_id": "operator:O", "parent_evidence_revision": "e1"}]
    return parent, section


@pytest.mark.parametrize("scope", [None, "operator:O", "section:S"])
def test_operator_section_aggregates_under_typed_parent_even_without_record_key(scope):
    parent, section = operator_sections()
    result = query([parent, section], scope=scope)
    assert len(result["records"]) == 1
    assert result["records"][0]["key"] == "operator:O"
    assert result["records"][0]["indexed"] is True
    assert result["records"][0]["sections"]["complete"] == 1


def test_required_missing_unit_and_parent_inventory_identity_survive_query_and_section_states():
    parent, section = operator_sections()
    parent["inventories"][0]["required_units"].append("section:S2")
    parent["inventories"][0]["section_mapping_revisions"]["section:S2"] = 1
    record = query([parent, section])["records"][0]
    absent = next(outcome for outcome in record["required_unit_outcomes"] if outcome["unit_id"] == "section:S2")
    assert absent["state"] == "missing"
    assert absent["reason"] == "required_unit_absent"
    assert absent["inventory_revision"] == "v1"
    assert absent["parent_evidence_revision"] == "e1"
    parent["inventories"][0]["required_units"] = ["section:S"]
    section["observations"][1]["observed_at"] = "2026-10-20T11:00:00Z"
    assert query([parent, section])["records"][0]["sections"]["stale"] == 1


def test_required_section_with_missing_aggregate_layer_is_counted_missing():
    parent, section = operator_sections()
    section["receipt"]["aggregate_layers"] = ["missing-ancestor", section["receipt"]["namespace"]]
    record = query([parent, section])["records"][0]
    assert record["sections"]["missing"] == 1
    assert record["required_unit_outcomes"][0]["aggregate_roles"][0]["state"] == "missing"
