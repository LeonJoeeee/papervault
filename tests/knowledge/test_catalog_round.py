"""Catalog round contracts without backend credentials."""
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.store.conftest import catalog_test_dsn as catalog_test_dsn, repo as repo
from tests.store.test_knowledge_postgres import knowledge as knowledge


class EmptyCatalog:
    def __init__(self, authority="catalog"):
        self.authority = authority

    def get_authority_state(self):
        return {"authority": self.authority, "authority_epoch": 1}

    def list_jobs(self, states=None):
        return []

    def list_outbox(self, limit=100, kind=None):
        return []

    def list_collection_heads(self):
        return []

    def list_selected_receipts(self, scope=None):
        return []

    def list_reconcilable_receipts(self, scope=None):
        return []

    def list_admission_jobs(self, scope=None):
        return []

    def coverage_rows(self, scope=None):
        return []


class ClosedBreaker:
    def allows(self):
        return False


@pytest.mark.asyncio
async def test_round_preserves_closed_build_gate_without_attempt_or_delivery(tmp_path):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore

    scheduler = CatalogScheduler(EmptyCatalog(), object(), ImmutableInputStore(tmp_path), breaker=ClosedBreaker())
    result = await scheduler.run_round()
    assert result["paused"] is True
    assert result["claimed"] == result["delivered"] == 0


@pytest.mark.asyncio
async def test_uncommitted_authority_performs_no_scheduler_mutation(tmp_path):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore

    result = await CatalogScheduler(EmptyCatalog("legacy"), object(), ImmutableInputStore(tmp_path)).run_round()
    assert result["paused"] is True
    assert result["reason"] == "catalog_authority_not_committed"


def test_metadata_receipt_uses_exact_captured_bytes_and_declares_no_graph(tmp_path):
    from papervault.knowledge.scheduler.catalog import metadata_receipt
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.store.knowledge import validate_receipt

    store = ImmutableInputStore(tmp_path)
    artifact = store.capture(b'{"title":"one"}', role="metadata")
    claim = {"job_id": "j", "attempt_id": "a", "submission_id": "s", "authority_epoch": 1,
             "fencing_token": 2, "unit_id": "u", "consumed_input_revision": "i",
             "recipe_digest": "r", "generation_id": "g", "namespace": "delta", "lease_id": "l",
             "aggregate_base_manifest_revision": 1, "base_aggregate_layers": [],
             "base_selected_unit_receipts": {}, "retired_receipt_ids": []}
    receipt = metadata_receipt(claim, {"artifacts": [artifact], "tier": "metadata"}, store)
    output = receipt["output_inventory"][0]
    assert output["role"] == "metadata"
    assert output["sha256"] == hashlib.sha256(b'{"title":"one"}').hexdigest()
    assert output["byte_length"] == 15
    assert receipt["achieved_tier"] == "metadata"
    validate_receipt(receipt, claim, {"output_contract": {"metadata": {
        "required": ["metadata"], "not_applicable": ["graph", "vectors"]}}})
    (tmp_path / artifact["sha256"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError):
        metadata_receipt(claim, {"artifacts": [artifact], "tier": "metadata"}, store)


@pytest.mark.asyncio
async def test_metadata_collection_head_refreshes_inherited_backend_outputs(tmp_path):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.store.knowledge import CLAIM_FIELDS

    receipt = {key: "B" for key in CLAIM_FIELDS}
    receipt.update(receipt_id="receipt-B", namespace="metadata-B", achieved_tier="metadata",
        aggregate_layers=["fulltext-A", "metadata-B"], output_inventory=[{
            "role": "metadata", "store": "catalog_inputs", "namespace": "metadata-B", "output_id": "bytes-B",
            "sha256": "b" * 64, "byte_length": 1}])
    ancestor_observation = {**{key: "A" for key in CLAIM_FIELDS}, "namespace": "fulltext-A",
        "receipt_id": "receipt-A", "role": "graph", "store": "neo4j", "output_id": "graph-A",
        "observed_at": "2026-10-21T00:00:00Z", "state": "present"}
    commands = []
    repository = SimpleNamespace(get_input=lambda *a: {"manifest": {"artifacts": []}},
        get_section=lambda *a: None, command=lambda *args: commands.append(args))
    adapter = SimpleNamespace(collect_observations=AsyncMock(return_value=[ancestor_observation]))
    scheduler = CatalogScheduler(repository, adapter, ImmutableInputStore(tmp_path))
    summary = {"observations": 0, "discrepancies": 0}
    await scheduler._collect(receipt, summary)
    adapter.collect_observations.assert_awaited_once_with(receipt)
    assert any(action == "observe" and value["receipt_id"] == "receipt-A" for action, _, value in commands)


def test_paper_cleaning_does_not_rewrite_operator_section_transformation():
    from papervault.knowledge.scheduler.catalog import execution_text
    from papervault.knowledge.ingest.distill import clean

    raw = ("Measured source evidence. " * 100 + "\n\n## References\n\nAuthor, 2020. A citation.").encode()
    assert clean(raw.decode()) != raw.decode()
    assert execution_text({"kind": "paper"}, "fulltext", raw) == clean(raw.decode())
    assert execution_text({"kind": "section"}, "fulltext", raw) == raw.decode()


def test_backend_file_path_preserves_public_paper_and_operator_citation_keys():
    from papervault.knowledge.scheduler.catalog import citation_file_path

    assert citation_file_path({"unit_id": "random-uuid", "kind": "paper", "record_key": "doi-key"}, {}) == "paper/doi-key"
    assert citation_file_path({"unit_id": "another-uuid", "kind": "section", "record_key": None},
        {"metadata": {"source_key": "textbook:plasma#s7"}}) == "textbook:plasma#s7"
    with pytest.raises(ValueError):
        citation_file_path({"unit_id": "unmapped-uuid", "kind": "section"}, {"metadata": {}})


@pytest.mark.asyncio
async def test_completed_receipt_rebases_without_starving_later_job_in_single_slot_rounds(knowledge, tmp_path):
    from datetime import datetime, timezone

    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.scheduler.catalog_inputs import make_input_manifest
    from papervault.store.knowledge import CLAIM_FIELDS
    from tests.store.test_knowledge_postgres import (
        _NoBackend, _OpenBuildGate, claim, prepare_fulltext, receipt, seal,
    )

    k = knowledge
    store, first = prepare_fulltext(k, tmp_path)

    def enqueue_unit(key):
        unit_id = "operator:" + key
        k.command("register_unit", "unit:" + key,
            {"unit_id": unit_id, "kind": "operator_document", "identity": key})
        artifact = store.capture(("Evidence for " + key).encode(), role="fulltext_evidence")
        manifest, input_revision = make_input_manifest(unit_id, {"source_key": key}, [artifact], tier="fulltext")
        k.command("register_input", "input:" + key,
            {"unit_id": unit_id, "consumed_input_revision": input_revision, "manifest": manifest})
        identity = {"unit_id": unit_id, "consumed_input_revision": input_revision,
                    "recipe_digest": first["recipe_digest"], "generation_id": "g"}
        k.command("set_requirement", "requirement:" + key, {**identity, "expected_revision": 0})
        return k.command("enqueue", "job:" + key, identity)

    advancing = enqueue_unit("B")
    later = enqueue_unit("C")
    original_claim = claim(k, first, "claim:A")
    original_receipt = seal(k, original_claim)
    k.command("complete", "complete:" + original_receipt["receipt_id"], {
        **original_claim, "receipt": original_receipt, "expected_revision": original_claim["revision"]})

    other_claim = claim(k, advancing, "claim:B")
    other_receipt = seal(k, other_claim)
    k.command("publish_derivation", "publish:" + other_receipt["receipt_id"], {
        **other_claim, "receipt": other_receipt, "expected_revision": other_claim["revision"],
        "expected_selection_revision": 0, "select": True})
    current_base = k.get_generation("g")["manifest_revision"]
    assert current_base > original_claim["aggregate_base_manifest_revision"]

    class SealingBackend(_NoBackend):
        async def submit(self, claim, text, file_path, *, achieved_tier):
            k.command("accept_submission", "accept:" + claim["submission_id"], {"claim": claim})
            rec = receipt(claim)
            k.command("seal_namespace", "seal:" + claim["submission_id"], {"claim": claim, "receipt": rec})
            return rec

        async def collect_observations(self, rec):
            return [{**{key: rec[key] for key in CLAIM_FIELDS}, "receipt_id": rec["receipt_id"],
                **{key: output[key] for key in ("role", "store", "namespace", "output_id")},
                "state": "present", "observed_at": datetime.now(timezone.utc).isoformat(), "detail": {}}
                for output in rec["output_inventory"]]

    scheduler = CatalogScheduler(k, SealingBackend(), store, breaker=_OpenBuildGate())
    rebased = await scheduler.run_round()
    assert rebased["claimed"] == rebased["delivered"] == rebased["published"] == 1
    first_job = k.get_job(first["job_id"])
    assert first_job["attempts"] == 2
    attempt = k.get_attempt(first_job["current_attempt_id"])
    assert attempt["attempt_id"] != original_claim["attempt_id"]
    assert attempt["submission_id"] != original_claim["submission_id"]
    assert attempt["fencing_token"] > original_claim["fencing_token"]
    assert attempt["aggregate_base_manifest_revision"] == current_base
    assert attempt["base_selected_unit_receipts"] == {"operator:B": other_receipt["receipt_id"]}
    selected = k.get_selection("paper:A", "g")
    assert selected["derivation_id"] == "derivation:" + attempt["receipt_id"]
    assert k.get_operation("publish:" + attempt["receipt_id"]) is not None
    assert k.get_job(later["job_id"])["attempts"] == 0

    progressed = await scheduler.run_round()
    assert progressed["claimed"] == progressed["delivered"] == progressed["published"] == 1
    assert k.get_job(later["job_id"])["state"] == "succeeded"
    assert k.get_selection("operator:C", "g") is not None
    assert not k.list_outbox(kind="submission")
    assert k.get_attempt(original_claim["attempt_id"])["cost_state"] == "unknown"
    assert k.get_authority_state()["budget_reserved"] == 4
    replay = await scheduler.run_round()
    assert replay["claimed"] == replay["delivered"] == replay["published"] == 0
    assert k.get_job(first["job_id"])["attempts"] == 2
    assert k.get_job(later["job_id"])["attempts"] == 1
