"""Pinned generation reads survive promotion without relaxing authority or evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from papervault.knowledge.query import aquery as aq
from papervault.knowledge.store.catalog_serving import CatalogServingFacade
from tests.store.test_coverage import row


class Repository:
    def __init__(self):
        self.row = row()
        self.receipt = self.row["receipt"]
        self.snapshot = {"generation_id": "g1", "manifest_revision": 1, "revision": 1,
            "manifest": {"aggregate_layers": ["private-A"], "selected_unit_receipts": {"paper:A": "receipt1"}},
            "entries": [{"unit_id": "paper:A", "eligible": True, "receipt": self.receipt}]}
        self.active_reads = 0
        self.released = False

    def get_serving_manifest(self, **kwargs):
        self.active_reads += 1
        return deepcopy(self.snapshot)

    def coverage_rows(self, scope=None):
        value = deepcopy(self.row)
        now = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        for obs in value["observations"]:
            obs["observed_at"] = now
        return [value]

    def command(self, action, operation_id, payload):
        if action == "release_pin":
            self.released = True
        return {"pin_id": "pin1", "revision": 1,
                "expires_at": datetime.now(timezone.utc) + timedelta(seconds=300)}


@pytest.mark.asyncio
@pytest.mark.parametrize("same_generation", [False, True])
async def test_query_uses_pinned_manifest_after_promotion_during_synthesis(monkeypatch, same_generation):
    repo = Repository()
    seen = []
    class Rag:
        async def aquery_data(self, intent, param):
            return {"status": "success", "data": {"chunks": [{"file_path": "A", "content": "evidence"}],
                "references": [{"file_path": "A"}]},
                "metadata": {"processing_info": {"total_entities_found": 30}}}
        async def finalize_storages(self):
            pass
    async def factory(repository, claim, **kwargs):
        seen.append(claim)
        return Rag()
    async def synth(*args, **kwargs):
        if not same_generation:
            repo.snapshot["generation_id"] = "g2"
        repo.snapshot["manifest_revision"] = 2
        repo.snapshot["manifest"]["aggregate_layers"].append("new-layer")
        # The newest per-generation manifest's new layer lacks observations. The
        # pinned aggregate does not contain that layer and must not validate it.
        repo.row["generation_manifest"] = {"generation_id": "g1", "manifest_revision": 2,
                                          "manifest": deepcopy(repo.snapshot["manifest"])}
        return "Finding [A]."
    facade = await CatalogServingFacade.pin(repo, rag_factory=factory)
    monkeypatch.setattr(aq, "_QUERY_VARIANT", "single")
    monkeypatch.setattr(aq, "synth_answer", synth)
    try:
        result = await aq.query("question", rag=facade)
        assert result["cited_papers"] == ["A"]
        assert result["query_receipt"]["generation_id"] == "g1"
        assert seen == [{"namespace": "private-A", "aggregate_layers": ["private-A"]}]
        assert repo.active_reads == 1
    finally:
        await facade.close()
    assert repo.released


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["policy", "quarantine", "missing_input", "changed_receipt", "missing_unit"])
async def test_pinned_validation_preserves_current_eligibility_and_evidence(change):
    repo = Repository()
    facade = await CatalogServingFacade.pin(repo)
    if change == "policy":
        repo.row["record"]["policy"] = "excluded"
    elif change == "quarantine":
        repo.row["unit"]["quarantine"] = True
    elif change == "missing_input":
        repo.row["observations"][1]["state"] = "missing"
    elif change == "changed_receipt":
        repo.row["selected_derivation"]["receipt_id"] = "new"
    else:
        repo.row["requirement"]["generation_id"] = "g2"
    # Active selection is intentionally unchanged: a snapshot-only validator is unsafe.
    try:
        with pytest.raises(RuntimeError, match="eligible|evidence|selection|receipt"):
            facade.validate()
    finally:
        await facade.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["expired", "renewal_failed", "unpinned"])
async def test_reader_pin_health_remains_mandatory(fault):
    repo = Repository()
    facade = await CatalogServingFacade.pin(repo)
    if fault == "expired":
        facade._pin["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    elif fault == "renewal_failed":
        facade._pin_error = RuntimeError("renewal unavailable")
    else:
        facade._pin = None
    try:
        with pytest.raises(RuntimeError, match="pin"):
            facade.validate()
    finally:
        await facade.close()
