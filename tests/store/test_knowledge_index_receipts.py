"""Old sealed backend receipts cannot bypass the document-index verifier."""
from copy import deepcopy

import pytest

from tests.store.test_knowledge_postgres import knowledge as knowledge


def backend_receipt(claim, tier="fulltext"):
    from papervault.knowledge.store.catalog_adapter import _graph_inventory, _inventory
    from tests.store.test_knowledge_postgres import receipt

    rec = receipt(claim)
    rec["achieved_tier"] = tier
    stamp = "2026-10-21T00:00:00Z"
    rec["output_inventory"] = [_graph_inventory(claim["namespace"],
        [{"entity_id": "Earth", "source_id": "chunk"}], [], stamp)]
    for role in ("full_entities", "full_relations"):
        rec["output_inventory"].append(_inventory(role, "postgres", claim["namespace"], [], [], stamp))
    return rec


@pytest.mark.parametrize("tier", ["fulltext", "abstract", "metadata"])
def test_legacy_physical_receipt_needs_versioned_index_verification_before_reuse(tier):
    from papervault.store.knowledge import validate_receipt
    from tests.knowledge.test_catalog_adapter import CLAIM

    rec = backend_receipt(CLAIM, tier)
    with pytest.raises(ValueError, match="index"):
        validate_receipt(rec, CLAIM, {"output_contract": {tier: {"required": ["graph"]}}})


@pytest.mark.parametrize("damage", ["proof", "doc", "chunks", "entity", "pairs", "index", "retired", "version"])
def test_catalog_checks_index_protocol_and_contribution_identity(damage):
    from papervault.store.knowledge import validate_receipt
    from tests.knowledge.test_catalog_adapter import CLAIM

    rec = backend_receipt(CLAIM)
    rec["provider_observations"]["doc_id"] = "doc"
    rec["document_index_verification"] = {"version": 1, "retired_receipt_ids": []}
    proof = {"doc_id": "doc", "chunk_ids": ["chunk"], "entity_ids": ["Earth"], "relation_pairs": []}
    rec["output_inventory"][0]["verification"]["evidence"]["document_contribution"] = proof
    index = rec["output_inventory"][1]
    index["backend_ids"] = ["doc"]
    index["verification"] = {"state": "present", "method": "readback", "evidence": {"record_count": 1}}
    recipe = {"output_contract": {"fulltext": {"required": ["graph"]}}}
    validate_receipt(deepcopy(rec), CLAIM, recipe)
    if damage == "proof":
        rec["output_inventory"][0]["verification"]["evidence"].pop("document_contribution")
    elif damage in {"doc", "chunks", "entity", "pairs"}:
        field, value = {"doc": ("doc_id", "other"), "chunks": ("chunk_ids", []),
                        "entity": ("entity_ids", [None]), "pairs": ("relation_pairs", [["Earth"]])}[damage]
        proof[field] = value
    elif damage == "index":
        index["backend_ids"] = []
    elif damage == "retired":
        rec["document_index_verification"]["retired_receipt_ids"] = ["other"]
    else:
        rec["document_index_verification"]["version"] = 0
    with pytest.raises(ValueError, match="index|contribution"):
        validate_receipt(rec, CLAIM, recipe)


@pytest.mark.asyncio
@pytest.mark.parametrize("completed", [False, True])
async def test_recovered_old_receipt_cannot_complete_or_publish_and_retains_history(knowledge, tmp_path, monkeypatch, completed):
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.store import knowledge_commands
    from tests.store.test_knowledge_postgres import claim, prepare

    k = knowledge
    c = claim(k, prepare(k))
    rec = backend_receipt(c)
    # Fixture an immutable receipt accepted by the earlier validation protocol.
    # Only setup bypasses the new gate; every completion/publication below uses it.
    with monkeypatch.context() as old_protocol:
        old_protocol.setattr(knowledge_commands, "validate_receipt", lambda *args: None)
        k.command("seal_namespace", "old-seal", {"claim": c, "receipt": rec})
        if completed:
            k.command("complete", "old-complete", {**c, "receipt": rec, "expected_revision": c["revision"]})
    before = k.get_serving_manifest()
    job = k.get_job(c["job_id"])
    scheduler = CatalogScheduler(k, object(), ImmutableInputStore(tmp_path))
    summary = {"completed": 0, "published": 0, "discrepancies": 0}
    assert await scheduler._finish(rec, summary) is False
    assert summary == {"completed": 0, "published": 0, "discrepancies": 1}
    with pytest.raises(ValueError, match="index"):
        k.command("publish_derivation", "recovered-publish", {**c, "receipt": rec,
            "expected_revision": job["revision"], "expected_selection_revision": 0, "select": True})
    assert k.get_job(c["job_id"]) == job
    assert k.get_serving_manifest() == before
    assert k.get_selection(c["unit_id"], c["generation_id"]) is None
    assert k.get_receipt(rec["receipt_id"]) == rec
