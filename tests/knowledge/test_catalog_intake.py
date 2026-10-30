"""Desired inputs are captured before admission and fallback creates a new job."""

import json
import os
import subprocess
import sys

import pytest

from papervault.store.repository import Record


class Repository:
    def __init__(self):
        self.unit = None
        self.requirement = None
        self.jobs = {}
        self.commands = []
        self.inventory = None

    def get_unit_for_record(self, key):
        return self.unit

    def get_active_generation(self):
        return {"generation_id": "active"}

    def get_requirement(self, unit, generation):
        return self.requirement

    def get_current_inventory(self, parent, generation):
        return self.inventory

    def get_logical_job(self, unit, input_revision, recipe, generation):
        return self.jobs.get(input_revision)

    def command(self, action, operation_id, payload):
        from papervault.store.repository import payload_hash
        self.commands.append((action, operation_id, payload))
        if action == "register_unit":
            self.unit = payload
        if action == "register_recipe":
            return {"recipe_digest": payload_hash(payload["recipe"])}
        if action == "set_requirement":
            self.requirement = {**payload, "revision": payload["expected_revision"] + 1}
        if action == "set_inventory":
            self.inventory = {**payload, "revision": payload["expected_revision"] + 1}
        if action == "enqueue":
            key = payload["consumed_input_revision"]
            self.jobs.setdefault(key, {**payload, "job_id": "job-" + str(len(self.jobs) + 1)})
            return self.jobs[key]
        return payload


def recipe():
    return {"schema_version": 1, "output_contract": {"fulltext": {"required": ["graph"]}}}


def test_fallback_enqueues_linked_new_logical_job(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake

    vault = tmp_path / "vault"
    vault.mkdir()
    text = vault / "A.md"
    text.write_text("A retained measurement paper.")
    record = Record.from_payload("A", {"key": "A", "md_path": "A.md",
        "download_status": "metadata_only", "abstract": "Its measured result.",
        "terminal_state": {"policy": "clear", "retrieval": "exhausted"}})
    repo = Repository()
    intake = CatalogIntake(repo, ImmutableInputStore(tmp_path / "inputs"), vault, recipe=recipe())
    first = intake.capture_record(record)
    text.unlink()
    second = intake.capture_record(record)
    assert first["job_id"] != second["job_id"]
    assert second["predecessor_job_id"] == first["job_id"]
    assert len(repo.jobs) == 2
    assert repo.requirement["expected_record_revision"] == record.revision


def test_unchanged_recapture_does_not_advance_desired_revision(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake

    repo = Repository()
    record = Record.from_payload("A", {"key": "A"})
    intake = CatalogIntake(repo, ImmutableInputStore(tmp_path / "inputs"), tmp_path, recipe=recipe())
    assert intake.capture_record(record)["job_id"] == intake.capture_record(record)["job_id"]
    assert len(repo.jobs) == 1
    assert sum(action == "set_requirement" for action, _, _ in repo.commands) == 1


def test_whole_document_inventory_never_double_counts_parent(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake

    repo = Repository()
    intake = CatalogIntake(repo, ImmutableInputStore(tmp_path / "inputs"), tmp_path, recipe=recipe())
    intake.capture_record(Record.from_payload("A", {"key": "A"}))
    inventory = next(payload for action, _, payload in repo.commands if action == "register_inventory")
    assert inventory["mode"] == "whole"
    assert inventory["required_units"] == [repo.unit["unit_id"]]
    assert inventory["optional_units"] == []


def test_recipe_declares_every_tier_and_no_credentials(monkeypatch):
    from papervault.knowledge.scheduler.catalog_intake import effective_distill_recipe
    from papervault.store.repository import canonical_json

    monkeypatch.setenv("PAPERVAULT_LLM_API_KEY", "secret-must-never-enter-recipe")
    actual = effective_distill_recipe()
    for tier in ("fulltext", "abstract", "metadata"):
        contract = actual["output_contract"][tier]
        assert contract["required"]
        assert not set(contract["required"]) & set(contract["not_applicable"])
    assert "graph" in actual["output_contract"]["metadata"]["not_applicable"]
    assert "secret-must-never-enter-recipe" not in canonical_json(actual)


def test_recipe_changes_when_boot_time_llm_token_limit_changes(tmp_path):
    # The wrapper reads this at import time. Test real process startup so the
    # recipe cannot claim a fresh environment value the running wrapper ignores,
    # and verify unset KS_LLM_MAX_TOKENS serializes max_tokens=None cleanly.
    program = """
import json
from papervault.knowledge.scheduler.catalog_intake import effective_distill_recipe
from papervault.knowledge.store.llm import _DEFAULT_MAX_TOKENS
from papervault.store.knowledge import canonical_recipe
recipe, digest = canonical_recipe(effective_distill_recipe())
print(json.dumps({'digest': digest, 'recipe': recipe, 'actual': _DEFAULT_MAX_TOKENS}))
"""
    base_env = {k: v for k, v in os.environ.items() if k != "KS_LLM_MAX_TOKENS"}
    base_env["PYTHON_DOTENV_DISABLED"] = "1"
    results = []
    for limit in (None, 4096, 8192):
        run_env = dict(base_env)
        if limit is not None:
            run_env["KS_LLM_MAX_TOKENS"] = str(limit)
        result = subprocess.run([sys.executable, "-c", program], cwd=tmp_path, check=True,
                                capture_output=True, text=True, env=run_env)
        results.append(json.loads(result.stdout.splitlines()[-1]))
    assert len({r["digest"] for r in results}) == 3
    for result, expected in zip(results, (None, 4096, 8192), strict=True):
        assert result["actual"] == expected
        assert result["recipe"]["execution_settings"]["llm_wrapper"]["max_tokens"] == expected


@pytest.mark.parametrize(("setting", "changed"), [
    ("_CLIENT_TIMEOUT", 301.0), ("_STREAM", False), ("_FIRST_CHUNK_S", 23.0),
    ("_MAX_ROUNDS", 4), ("_BASE_BACKOFF", 3.0), ("_BACKOFF_JITTER", 0.25),
    ("_GW_CONN_MAX_RETRIES", 3), ("_GW_CONN_BACKOFF_CAP", 11.0),
    ("_GW_TRANSIENT_RETRIES", 5), ("_GW_TRANSIENT_BACKOFF_CAP", 12.0),
    ("_GW_RETRY_BUDGET_S", 222.0),
])
def test_recipe_binds_effective_llm_transport_and_retry_settings(monkeypatch, setting, changed):
    from papervault.knowledge.scheduler.catalog_intake import effective_distill_recipe
    from papervault.knowledge.store import llm
    from papervault.store.knowledge import canonical_recipe

    before = canonical_recipe(effective_distill_recipe())[1]
    monkeypatch.setattr(llm, setting, changed)
    assert canonical_recipe(effective_distill_recipe())[1] != before


def test_recipe_binds_gateway_mode_without_credentials(monkeypatch):
    from papervault import config
    from papervault.knowledge.scheduler.catalog_intake import effective_distill_recipe
    from papervault.store.knowledge import canonical_recipe
    from papervault.store.repository import canonical_json

    monkeypatch.setattr(config, "USE_GATEWAY", False)
    before = canonical_recipe(effective_distill_recipe())[1]
    monkeypatch.setattr(config, "USE_GATEWAY", True)
    monkeypatch.setattr(config, "GATEWAY_KEY", "private-test-gateway-key")
    recipe = effective_distill_recipe()
    assert canonical_recipe(recipe)[1] != before
    assert "private-test-gateway-key" not in canonical_json(recipe)


def test_unavailable_input_records_discrepancy_and_keeps_other_intake_moving(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake

    repo = Repository()
    intake = CatalogIntake(repo, ImmutableInputStore(tmp_path / "inputs"), tmp_path, recipe=recipe())
    bad = Record.from_payload("A", {"key": "A", "md_path": "../outside.md"})
    good = Record.from_payload("B", {"key": "B"})
    result = intake._capture_rows([bad, good])
    assert result == {"captured": 1, "excluded": 0, "errors": 1}
    discrepancies = [payload for action, _, payload in repo.commands if action == "record_discrepancy"]
    assert discrepancies[0]["reason"] == "input_capture_unavailable"
    assert discrepancies[0]["evidence"]["record_key"] == "A"
