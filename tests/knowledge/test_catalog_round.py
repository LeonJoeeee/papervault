"""Catalog round contracts without backend credentials."""
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


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
