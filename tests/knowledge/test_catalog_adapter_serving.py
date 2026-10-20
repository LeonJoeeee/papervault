"""A selected manifest is applied before backend retrieval and cache access."""
from copy import deepcopy

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
async def test_generation_promotion_does_not_mix_manifest_across_facets():
    from papervault.knowledge.store.catalog_serving import CatalogServingFacade

    repo = Repository()
    facade = await CatalogServingFacade.pin(repo)
    repo.current["manifest_revision"] = 8
    with pytest.raises(RuntimeError, match="manifest"):
        await facade.aquery_data("facet", None)


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
