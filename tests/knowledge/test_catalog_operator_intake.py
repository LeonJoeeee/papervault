"""Operator section identities keep parent evidence and segmentation history."""

import pytest


class Tokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


class Repository:
    def __init__(self):
        self.units, self.sections, self.inputs, self.inventories = {}, {}, {}, {}
        self.current, self.requirements, self.jobs = {}, {}, {}
        self.commands = []
        self.operations = {}
        self.generation = {"generation_id": "active", "manifest_revision": 1}

    def get_unit(self, unit_id):
        return self.units.get(unit_id)

    def get_operation(self, operation_id):
        return self.operations.get(operation_id)

    def get_active_generation(self):
        return self.generation

    def get_generation(self, generation_id):
        return self.generation

    def get_authority_state(self):
        return {"requirements_revision": len(self.requirements)}

    def get_section(self, unit_id):
        return self.sections.get(unit_id)

    def get_requirement(self, unit_id, generation):
        return self.requirements.get((unit_id, generation))

    def get_logical_job(self, unit, input_revision, recipe, generation):
        return self.jobs.get((unit, input_revision, recipe, generation))

    def get_current_inventory(self, parent, generation):
        selector = self.current.get((parent, generation))
        return {**self.inventories[selector["inventory_revision"]], **selector} if selector else None

    def command(self, action, operation_id, payload):
        from papervault.store.repository import payload_hash
        self.commands.append((action, payload))
        if action == "register_recipe":
            return {"recipe_digest": payload_hash(payload["recipe"])}
        if action == "register_unit":
            self.units[payload["unit_id"]] = payload
        if action == "register_section":
            self.sections[payload["unit_id"]] = payload
        if action == "register_input":
            self.inputs[payload["consumed_input_revision"]] = payload
        if action == "register_inventory":
            self.inventories[payload["inventory_revision"]] = payload
        if action == "set_inventory":
            self.current[(payload["parent_unit_id"], payload["generation_id"])] = {
                **payload, "revision": payload["expected_revision"] + 1}
        if action == "set_requirement":
            self.requirements[(payload["unit_id"], payload["generation_id"])] = {
                **payload, "revision": payload["expected_revision"] + 1}
        if action == "enqueue":
            key = tuple(payload[field] for field in ("unit_id", "consumed_input_revision", "recipe_digest", "generation_id"))
            self.jobs.setdefault(key, {**payload, "job_id": "job" + str(len(self.jobs) + 1)})
            return self.jobs[key]
        if action == "fork_generation":
            result = {"generation_id": payload["generation_id"], "queued_jobs": len(self.requirements)}
            self.operations[operation_id] = result
            return result
        return payload


def intake(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake

    return CatalogIntake(Repository(), ImmutableInputStore(tmp_path / "inputs"), tmp_path,
                         recipe={"schema_version": 1})


def document(tmp_path):
    path = tmp_path / "textbook.md"
    path.write_text("# First\nalpha beta gamma delta epsilon zeta eta theta\n# Second\niota kappa lambda mu nu xi omicron pi")
    return path


def test_operator_sections_have_exact_parent_inventory_without_parent_job(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    capture = intake(tmp_path)
    result = capture.capture_document("textbook", "textbook:Tester2020", document(tmp_path),
                                      tokenizer=Tokenizer(), max_tokens=12)
    assert result["catalog_queued"] is True
    assert result["sections"] >= 2
    repo = capture.repository
    parent = next(unit for unit in repo.units.values() if unit["kind"] == "operator_document")
    inventory = repo.get_current_inventory(parent["unit_id"], "active")
    assert parent["unit_id"] not in inventory["required_units"]
    assert len(repo.jobs) == len(inventory["required_units"])
    for unit_id in inventory["required_units"]:
        mapping = repo.sections[unit_id]
        assert mapping["parent_evidence_revision"] == inventory["parent_evidence_revision"]
        assert mapping["mapping_revision"] == inventory["section_mapping_revisions"][unit_id]
    parent_input = repo.inputs[inventory["parent_consumed_input_revision"]]["manifest"]
    assert parent_input["parent_evidence_revision"] == inventory["parent_evidence_revision"]


def test_changed_operator_inputs_need_force_and_create_new_candidate(tmp_path, monkeypatch):
    from papervault.knowledge.ingest.operator_docs import AlreadyIngestedError

    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    capture, path = intake(tmp_path), document(tmp_path)
    capture.capture_document("textbook", "textbook:Tester2020", path, tokenizer=Tokenizer(), max_tokens=12)
    old_jobs = len(capture.repository.jobs)
    old_mapping = next(iter(capture.repository.sections.values())).copy()
    path.write_text("# Revised\nreplacement source evidence words.")
    with pytest.raises(AlreadyIngestedError):
        capture.capture_document("textbook", "textbook:Tester2020", path, tokenizer=Tokenizer(), max_tokens=12)
    assert len(capture.repository.jobs) == old_jobs
    result = capture.capture_document("textbook", "textbook:Tester2020", path,
                                      tokenizer=Tokenizer(), max_tokens=12, force=True)
    assert result["generation_id"] != "active"
    assert any(action == "fork_generation" for action, _ in capture.repository.commands)
    latest = capture.repository.sections[old_mapping["unit_id"]]
    assert latest["mapping_revision"] > old_mapping["mapping_revision"]
    assert latest["parent_evidence_revision"] != old_mapping["parent_evidence_revision"]


def test_changed_segmentation_is_a_new_mapping_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_OPERATOR_SOURCES", "1")
    capture, path = intake(tmp_path), document(tmp_path)
    capture.capture_document("textbook", "textbook:Tester2020", path, tokenizer=Tokenizer(), max_tokens=12)
    before = next(iter(capture.repository.sections.values())).copy()
    capture.capture_document("textbook", "textbook:Tester2020", path,
                              tokenizer=Tokenizer(), max_tokens=7, force=True)
    after = capture.repository.sections[before["unit_id"]]
    assert after["mapping_revision"] == before["mapping_revision"] + 1
    assert after["parent_evidence_revision"] == before["parent_evidence_revision"]
    assert after["definition"] != before["definition"]
