"""Keep the F0 workload/denominator and distinguish historical aggregates from current evidence."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
POOLS = ROOT / "src/papervault/eval/gold_search_pools.jsonl"
FROZEN = ROOT / "docs/baselines/f0-snapshot-2026-10-18.json"


def inputs():
    pools = [json.loads(line) for line in POOLS.read_text().splitlines() if line.strip()]
    entry = pools[0]["pool"][0]
    record = {"key": "A", "title": entry["title"], "doi": entry.get("doi"),
              "registered": True, "readable": True, "indexed": False,
              "tiers": {"fulltext": 0, "abstract": 1, "metadata": 0},
              "tier_coverage": {"fulltext": False, "abstract": True, "metadata": False}, "unit_ids": ["paper:A"]}
    unit = {"unit_id": "paper:A", "requested_identity": {"recipe_digest": "new"},
            "selected_identity": {"recipe_digest": "old"}, "receipt_id": "r1",
            "reasons": ["selected_recipe_digest_mismatch"], "roles": [{"state": "stale"}]}
    return pools, {"records": [record], "units": [unit], "max_observation_age": 300,
                   "selectors": {"generation_id": "current", "recipe_digest": "current", "consumed_input_revision": "current"}}, json.loads(FROZEN.read_text())


def test_frozen_aggregates_retained_without_fabricating_historic_entry_matches():
    from papervault.store.coverage_r2 import coverage_r2
    pools, coverage, frozen = inputs()
    report = coverage_r2(pools, coverage, frozen_snapshot=frozen)
    assert report["workload"] == {"pools": 12, "entries_total": 111}
    assert report["frozen_legacy_ruler"]["aggregates"] == frozen["r2_coverage"]
    assert report["frozen_legacy_ruler"]["entry_provenance"] == "unknown"
    assert report["current_legacy_ruler"]["state"] == "unknown"
    assert len(report["entries"]) == 111
    assert report["current_version_qualified"]["metadata"] >= 1
    assert report["current_version_qualified"]["indexed"] == 0
    assert report["current_version_qualified"]["lower_tiers"]["abstract"] >= 1
    assert report["entries"][0]["units"][0]["receipt_id"] == "r1"
    assert report["entries"][0]["historic_match"] == "unknown"


def test_contemporaneous_legacy_comparison_has_exact_keyed_difference_and_version_reason():
    from papervault.store.coverage_r2 import coverage_r2
    pools, coverage, frozen = inputs()
    legacy = [{**coverage["records"][0], "readable": True, "indexed": True}]
    report = coverage_r2(pools, coverage, frozen_snapshot=frozen, current_legacy_records=legacy)
    delta = report["current_differences"][0]
    assert delta["qid"] == pools[0]["qid"]
    assert delta["entry_index"] == 0
    assert delta["legacy_key"] == delta["current_key"] == "A"
    assert delta["differences"]["indexed"] == {"legacy": True, "version_qualified": False}
    assert delta["units"][0]["reasons"] == ["selected_recipe_digest_mismatch"]
    assert report["current_legacy_ruler"]["indexed"] >= 1
    assert delta["difference_explanations"]["indexed"]["category"] == "recipe"


def test_new_positive_cannot_explain_negative_legacy_observation_without_its_evidence():
    from papervault.store.coverage_r2 import coverage_r2
    pools, coverage, frozen = inputs()
    coverage["records"][0]["indexed"] = True
    coverage["units"][0]["reasons"] = []
    coverage["units"][0]["roles"] = [{"state": "present"}]
    legacy = [{**coverage["records"][0], "indexed": False}]
    report = coverage_r2(pools, coverage, frozen_snapshot=frozen, current_legacy_records=legacy)
    delta = report["current_differences"][0]
    assert delta["difference_explanations"]["indexed"]["status"] == "unresolved"
    assert delta["legacy_evidence"]["indexed"] is False
    assert report["unresolved_difference_count"] > 0


def test_readable_difference_uses_input_evidence_independently_of_recipe():
    from papervault.store.coverage_r2 import coverage_r2
    pools, coverage, frozen = inputs()
    coverage["records"][0]["readable"] = False
    coverage["units"][0]["input_roles"] = [{"state": "missing"}]
    legacy = [{**coverage["records"][0], "readable": True}]
    report = coverage_r2(pools, coverage, frozen_snapshot=frozen, current_legacy_records=legacy)
    assert report["current_differences"][0]["difference_explanations"]["readable"]["category"] == "materialization"


def test_matcher_preserves_doi_precedence_and_exact_ascii_title_rule():
    from papervault.store.coverage_r2 import match_entries
    records = [{"key": "first", "doi": " 10.1/X ", "title": "Title"},
               {"key": "last", "doi": "10.1/x", "title": "TITLE"}]
    assert match_entries(records, [{"doi": "10.1/X", "title": "different"},
                                   {"title": "Tí-tle"}, {"title": "title"}]) == ["last", None, "first"]


def test_report_script_only_reads_explicit_inputs_and_writes_explicit_output(tmp_path):
    _, coverage, _ = inputs()
    supplied = tmp_path / "coverage.json"
    supplied.write_text(json.dumps(coverage))
    output = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/coverage_report.py"),
                             "--coverage", str(supplied), "--pools", str(POOLS),
                             "--frozen", str(FROZEN), "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["workload"]["entries_total"] == 111
    missing = subprocess.run([sys.executable, str(ROOT / "scripts/coverage_report.py")],
                             capture_output=True, text=True)
    assert missing.returncode != 0


def test_same_sized_modified_landmark_workload_is_not_mislabeled_as_frozen():
    import pytest
    from papervault.store.coverage_r2 import coverage_r2

    pools, coverage, frozen = inputs()
    pools[0]["pool"][0]["title"] = "replacement landmark"
    with pytest.raises(ValueError, match="frozen"):
        coverage_r2(pools, coverage, frozen_snapshot=frozen)


def test_r2_acceptance_refuses_noncurrent_or_relaxed_instrument_and_retains_policy():
    import copy
    import pytest
    from papervault.store.coverage_r2 import coverage_r2

    pools, coverage, frozen = inputs()
    coverage.update(selectors={"generation_id": "current", "recipe_digest": "current", "consumed_input_revision": "current"},
                    max_observation_age=300)
    report = coverage_r2(pools, coverage, frozen_snapshot=frozen)
    assert report["coverage_policy"] == {"selectors": coverage["selectors"], "max_observation_age": 300}
    for field, value in (("generation_id", "candidate"), ("recipe_digest", "old"), ("consumed_input_revision", "old")):
        changed = copy.deepcopy(coverage)
        changed["selectors"][field] = value
        with pytest.raises(ValueError, match="current"):
            coverage_r2(pools, changed, frozen_snapshot=frozen)
    for age in (301, float("nan"), float("inf")):
        changed = copy.deepcopy(coverage)
        changed["max_observation_age"] = age
        with pytest.raises(ValueError, match="age"):
            coverage_r2(pools, changed, frozen_snapshot=frozen)


def test_r2_difference_identifies_required_absent_section_not_just_existing_parent():
    from papervault.store.coverage_r2 import coverage_r2
    pools, coverage, frozen = inputs()
    coverage["records"][0]["required_unit_outcomes"] = [{"unit_id": "section:S2", "state": "missing",
        "reason": "required_unit_absent", "inventory_revision": "v1", "parent_evidence_revision": "e1"}]
    legacy = [{**coverage["records"][0], "indexed": True}]
    report = coverage_r2(pools, coverage, frozen_snapshot=frozen, current_legacy_records=legacy)
    assert report["current_differences"][0]["required_unit_outcomes"][0]["unit_id"] == "section:S2"


def test_alternate_r2_policy_requires_versioned_label_and_never_emits_current_headline():
    from papervault.store.coverage_r2 import coverage_r2
    pools, coverage, frozen = inputs()
    coverage["selectors"]["generation_id"] = "candidate"
    coverage["max_observation_age"] = 600
    report = coverage_r2(pools, coverage, frozen_snapshot=frozen, measurement_label="candidate_probe_v1")
    assert "current_version_qualified" not in report
    assert report["candidate_probe_v1"]["indexed"] == 0
    assert report["coverage_policy"]["selectors"]["generation_id"] == "candidate"
