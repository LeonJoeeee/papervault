"""Admission pages apply the same scoped grants as transactional claims."""
import copy

import pytest

from papervault.store.repository import AuditEvent, Record
from test_knowledge_postgres import knowledge as knowledge


@pytest.fixture
def admission(knowledge):
    k = knowledge
    k.command("create_generation", "generation", {"generation_id": "g", "candidate": False})
    recipe = k.command("register_recipe", "recipe", {"recipe": {"schema_version": 1,
        "output_contract": {"metadata": {"required": ["metadata"]}}}})["recipe_digest"]

    def enqueue(key, *, change=None, section=False):
        state = {"policy": "override", "policy_reasons": ["off_domain"], "required_revision": 2,
            "override": {"key": key, "scope": ["acquire"], "reasons": ["off_domain"], "revision": 2}}
        for name, value in (change or {}).items():
            if name == "override" and isinstance(value, dict):
                state[name].update(copy.deepcopy(value))
            else:
                state[name] = value
        k.repository.upsert(Record.from_payload(key, {"key": key, "title": key, "terminal_state": state}),
            expected_revision=0, audit=AuditEvent("test", "record"))
        parent = "paper:" + key
        k.command("register_unit", "unit:" + parent,
            {"unit_id": parent, "kind": "paper", "record_key": key, "identity": key})
        unit = "section:" + key if section else parent
        manifest = {"metadata_revision": "1", "tier": "metadata",
            "artifacts": [{"artifact_id": key, "sha256": "a" * 64, "byte_length": 1}]}
        if section:
            k.command("register_unit", "unit:" + unit, {"unit_id": unit, "kind": "section", "identity": key})
            k.command("register_section", "mapping:" + unit, {"unit_id": unit, "parent_unit_id": parent,
                "mapping_revision": 1, "parent_evidence_revision": "e", "definition": {"locator": key}})
            k.command("register_inventory", "inventory:" + parent, {"parent_unit_id": parent,
                "inventory_revision": "v1", "parent_evidence_revision": "e", "mode": "sections",
                "required_units": [unit], "optional_units": [], "section_mapping_revisions": {unit: 1}})
            k.command("set_inventory", "select-inventory:" + parent, {"parent_unit_id": parent,
                "generation_id": "g", "inventory_revision": "v1", "expected_revision": 0,
                "expected_record_revision": 1})
            manifest.update(section_definition_revision=1, parent_evidence_revision="e")
        k.command("register_input", "input:" + unit,
            {"unit_id": unit, "consumed_input_revision": unit, "manifest": manifest})
        identity = {"unit_id": unit, "generation_id": "g", "consumed_input_revision": unit, "recipe_digest": recipe}
        k.command("set_requirement", "requirement:" + unit, {**identity, "expected_revision": 0,
            "expected_record_revision": None if section else 1, "inventory_revision": "v1" if section else None})
        return k.command("enqueue", "enqueue:" + unit, identity)

    return k, enqueue


@pytest.mark.parametrize("section", [False, True])
def test_invalid_override_page_cannot_starve_a_later_valid_job(admission, section):
    k, enqueue = admission
    blocked = [enqueue(f"blocked-{i:03}", change={"required_revision": 3}, section=section)
        for i in range(100)]
    valid = enqueue("valid", section=section)
    assert [job["job_id"] for job in k.list_admission_jobs()] == [valid["job_id"]]
    assert all(k.get_job(job["job_id"])["attempts"] == 0 for job in blocked)


@pytest.mark.parametrize("change", [
    {"override": None}, {"override": {"key": "wrong"}},
    {"override": {"scope": ["other"]}}, {"override": {"reasons": ["other"]}},
    {"required_revision": 3}, {"policy": "clear"},
])
@pytest.mark.parametrize("section", [False, True])
def test_admission_limit_counts_only_jobs_with_valid_scoped_policy(admission, change, section):
    k, enqueue = admission
    enqueue("blocked", change=change, section=section)
    valid = enqueue("valid", section=section)
    assert [job["job_id"] for job in k.list_admission_jobs(limit=1)] == [valid["job_id"]]
