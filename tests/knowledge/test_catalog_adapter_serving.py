"""A selected manifest is applied before backend retrieval and cache access."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest


def manifest():
    return {"generation_id": "active", "manifest_revision": 7, "revision": 9,
            "manifest": {"aggregate_layers": ["pv_selected"],
                         "selected_unit_receipts": {"selected": "receipt"}}, "entries": [
        {"unit_id": "selected", "eligible": True, "receipt": {
            "namespace": "pv_selected", "sealed": True, "durable": True, "receipt_id": "receipt"}},
    ]}


class Repository:
    def __init__(self):
        self.current = manifest()
        self.pin_active = False

    def command(self, action, operation_id, payload):
        if action == "pin_manifest":
            self.pin_active = True
            return {"pin_id": "pin", "revision": 1, "expires_at": "2100-01-01T00:00:00+00:00"}
        if action == "release_pin":
            self.pin_active = False
            return {"released": True}
        assert action == "renew_pin"
        return {"pin_id": "pin", "revision": payload["expected_revision"] + 1,
                "expires_at": "2100-01-01T00:00:00+00:00"}

    def get_serving_manifest(self, **kwargs):
        return deepcopy(self.current)

    def coverage_rows(self, scope=None):
        from tests.store.test_coverage import row
        rows = []
        for entry in self.current["entries"]:
            value = row()
            uid, receipt = entry["unit_id"], entry["receipt"]
            value["unit"].update(unit_id=uid, record_key=uid)
            value["record"]["key"] = uid
            value["unit"]["quarantine"] = not entry["eligible"]
            for item in (value["requirement"], value["selected_derivation"], value["receipt"], *value["observations"]):
                item.update(unit_id=uid, generation_id="active")
                if "receipt_id" in item:
                    item["receipt_id"] = receipt["receipt_id"]
                if "namespace" in item:
                    item["namespace"] = receipt["namespace"]
            value["receipt"]["output_inventory"][0]["namespace"] = receipt["namespace"]
            value["receipt"].update(receipt)
            for obs in value["observations"]:
                obs["observed_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            value["inventories"][0]["required_units"] = [uid]
            rows.append(value)
        receipts = [{"receipt": value["receipt"], "recipe": value["recipe"]} for value in rows]
        observations = [obs for value in rows for obs in value["observations"]]
        for value in rows:
            value["aggregate_receipts"] = receipts
            value["aggregate_observations"] = observations
        return rows


@pytest.mark.asyncio
async def test_runtime_embedding_change_fails_before_selected_vectors_are_read(monkeypatch):
    from papervault.knowledge.scheduler import catalog_intake
    from papervault.knowledge.store.catalog_serving import CatalogServingFacade

    repo = Repository()
    repo.current["manifest"]["retrieval_contract"] = {"schema_version": 1, "model": "old-model"}
    monkeypatch.setattr(catalog_intake, "effective_embedding_contract", lambda: {"schema_version": 1, "model": "new-model"})

    async def should_not_open(*args, **kwargs):
        raise AssertionError("incompatible embeddings must be rejected before opening a backend")

    facade = await CatalogServingFacade.pin(repo, rag_factory=should_not_open)
    try:
        with pytest.raises(RuntimeError, match="embedding.*contract"):
            await facade.aquery_data("query", None)
    finally:
        await facade.close()


@pytest.mark.asyncio
async def test_only_selected_namespaces_enter_retrieval_graph_aggregate_and_cache():
    from papervault.knowledge.store.catalog_serving import CatalogServingFacade

    repo = Repository()

    async def factory(repository, claim, **kwargs):
        # Constructing the private backend would itself expose private caches.
        assert claim["namespace"] == "pv_selected"
        assert claim["aggregate_layers"] == ["pv_selected"]
        assert kwargs["read_only"] is True

        class Rag:
            async def aquery_data(self, query, param):
                return {"status": "success", "data": {
                    "entities": [{"entity_name": "shared", "description": "selected"}],
                    "relationships": [], "chunks": [{"content": "selected"}],
                    "references": []}, "metadata": {}}

            async def finalize_storages(self):
                pass
        return Rag()

    facade = await CatalogServingFacade.pin(repo, rag_factory=factory)
    assert repo.pin_active
    result = await facade.aquery_data("query", None)
    assert result["data"]["entities"][0]["description"] == "selected"
    assert result["data"]["chunks"] == [{"content": "selected"}]
    await facade.close()
    assert not repo.pin_active


@pytest.mark.asyncio
async def test_retrieval_uses_one_selected_aggregate_instead_of_one_query_per_unit():
    from papervault.knowledge.store.catalog_serving import CatalogServingFacade

    repo = Repository()
    repo.current["manifest"]["aggregate_layers"].append("pv_second")
    repo.current["manifest"]["selected_unit_receipts"]["second"] = "receipt2"
    repo.current["entries"].append({"unit_id": "second", "eligible": True, "receipt": {
        "namespace": "pv_second", "sealed": True, "durable": True, "receipt_id": "receipt2"}})

    async def factory(repository, claim, **kwargs):
        assert claim["aggregate_layers"] == ["pv_selected", "pv_second"]

        class Rag:
            async def aquery_data(self, query, param):
                return {"status": "success", "data": {"chunks": [{"content": "one global retrieval"}]}}

            async def finalize_storages(self):
                pass
        return Rag()

    facade = await CatalogServingFacade.pin(repo, rag_factory=factory)
    result = await facade.aquery_data("query", None)
    assert result["data"]["chunks"] == [{"content": "one global retrieval"}]
    await facade.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_reason", ["no_results", "backend_unavailable", None])
async def test_empty_selected_retrieval_reaches_public_empty_coverage_but_real_failures_raise(monkeypatch, failure_reason):
    from papervault.knowledge.query import aquery
    from papervault.knowledge.store.catalog_serving import CatalogServingFacade

    async def factory(*args, **kwargs):
        class Rag:
            async def aquery_data(self, query, param):
                return {"status": "failure", "message": "Query returned no results", "data": {},
                        "metadata": {"failure_reason": failure_reason, "mode": "mix"}}

            async def finalize_storages(self):
                pass
        return Rag()

    repo = Repository()
    facade = await CatalogServingFacade.pin(repo, rag_factory=factory)
    monkeypatch.setattr(aquery, "_QUERY_VARIANT", "single")
    try:
        if failure_reason == "no_results":
            result = await aquery.query("empty", rag=facade)
            assert result["kb_coverage"] == "empty"
            assert result["cited_papers"] == [] and result["cited_sources"] == []
        else:
            with pytest.raises(RuntimeError, match="retrieval failed"):
                await aquery.query("empty", rag=facade)
    finally:
        await facade.close()
    assert not repo.pin_active


@pytest.mark.asyncio
async def test_no_results_still_rechecks_eligibility_after_retrieval():
    from papervault.knowledge.store.catalog_serving import CatalogServingFacade

    repo = Repository()

    async def factory(*args, **kwargs):
        class Rag:
            async def aquery_data(self, query, param):
                repo.current["entries"][0]["eligible"] = False
                return {"status": "failure", "data": {}, "metadata": {"failure_reason": "no_results"}}

            async def finalize_storages(self):
                pass
        return Rag()

    facade = await CatalogServingFacade.pin(repo, rag_factory=factory)
    try:
        with pytest.raises(RuntimeError, match="ineligible"):
            await facade.aquery_data("empty", None)
    finally:
        await facade.close()


@pytest.mark.asyncio
async def test_policy_change_fails_before_next_facet_reads_pinned_namespace():
    from papervault.knowledge.store.catalog_serving import CatalogServingFacade

    repo = Repository()

    async def should_not_open(*args, **kwargs):
        raise AssertionError("policy recheck must happen before backend retrieval")

    facade = await CatalogServingFacade.pin(repo, rag_factory=should_not_open)
    repo.current["entries"][0]["eligible"] = False
    with pytest.raises(RuntimeError, match="changed|eligible|manifest"):
        await facade.aquery_data("query", None)


@pytest.mark.asyncio
async def test_generation_promotion_keeps_the_original_manifest_across_facets():
    from papervault.knowledge.store.catalog_serving import CatalogServingFacade

    repo = Repository()
    async def factory(repository, claim, **kwargs):
        assert claim["aggregate_layers"] == ["pv_selected"]
        class Rag:
            async def aquery_data(self, query, param):
                return {"status": "success", "data": {"chunks": [{"content": "original view"}]}}
            async def finalize_storages(self):
                pass
        return Rag()
    facade = await CatalogServingFacade.pin(repo, rag_factory=factory)
    repo.current["manifest_revision"] = 8
    try:
        result = await facade.aquery_data("facet", None)
        assert result["data"]["chunks"] == [{"content": "original view"}]
        assert facade.snapshot["manifest_revision"] == 7
    finally:
        await facade.close()


@pytest.mark.asyncio
async def test_aggregate_cannot_serve_a_receipt_without_a_current_unit_entry():
    from papervault.knowledge.store.catalog_serving import CatalogServingFacade

    repo = Repository()
    repo.current["entries"] = []
    with pytest.raises(RuntimeError, match="receipt|aggregate"):
        await CatalogServingFacade.pin(repo)


@pytest.mark.asyncio
async def test_public_query_pins_once_and_releases_serving_facade(monkeypatch):
    from papervault.knowledge.query import aquery
    from papervault.knowledge.store import catalog_serving

    class Facade:
        closed = False

        async def aquery_data(self, *args, **kwargs):
            return {"status": "success", "data": {}}

        def validate(self):
            pass

        async def close(self):
            self.closed = True

    facade = Facade()

    async def resolve():
        return facade

    monkeypatch.setattr(catalog_serving, "get_query_graph", resolve)
    monkeypatch.setattr(aquery, "_QUERY_VARIANT", "single")
    result = await aquery.query("empty")
    assert result["cited_papers"] == []
    assert facade.closed
