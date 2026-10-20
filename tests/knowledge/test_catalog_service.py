"""Catalog orchestration captures durable work before dispatching it."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_round_captures_inputs_before_catalog_admission():
    from papervault.knowledge.scheduler.catalog_service import CatalogService

    order = []
    intake = SimpleNamespace(scan=lambda **kw: order.append(("capture", kw)) or {"captured": 1})
    scheduler = SimpleNamespace(run_round=AsyncMock(side_effect=lambda **kw: order.append(("dispatch", kw)) or {"claimed": 1}))
    repo = SimpleNamespace(list_outbox=lambda **kw: [])
    result = await CatalogService(repo, intake, scheduler).run_round(only_keys={"A"})
    assert order == [("capture", {"only_keys": {"A"}}), ("dispatch", {"only_keys": {"A"}})]
    assert result["intake"]["captured"] == 1
    assert result["claimed"] == 1


@pytest.mark.asyncio
async def test_capture_failure_never_acknowledges_or_dispatches():
    from papervault.knowledge.scheduler.catalog_service import CatalogService

    def broken(**kwargs):
        raise OSError("captured artifact unavailable")

    scheduler = SimpleNamespace(run_round=AsyncMock())
    repo = SimpleNamespace(command=lambda *a: pytest.fail("must retain pending work"))
    with pytest.raises(OSError):
        await CatalogService(repo, SimpleNamespace(scan=broken), scheduler).run_round()
    scheduler.run_round.assert_not_awaited()


def test_force_inventory_recovery_forks_once_and_never_enqueues_active_jobs():
    from papervault.knowledge.scheduler.catalog_service import CatalogService

    inventory = {"parent_unit_id": "parent", "generation_id": "active", "inventory_revision": "I",
                 "parent_evidence_revision": "E", "parent_consumed_input_revision": "PI",
                 "required_units": ["S"], "optional_units": [], "section_mapping_revisions": {"S": 2},
                 "enqueue_mode": "candidate", "request_id": "request-1", "desired_recipe_digest": "R"}
    desired = {}
    operations, commands = {}, []

    def command(action, op, payload):
        commands.append((action, payload))
        if action == "set_requirement":
            desired.update(payload, revision=1)
        result = {"generation_id": payload.get("generation_id")}
        operations[op] = result
        return result

    repo = SimpleNamespace(get_current_inventory=lambda *a: inventory,
        get_generation=lambda *a: {"state": "active", "manifest_revision": 4},
        get_authority_state=lambda: {"requirements_revision": 9}, get_requirement=lambda *a: desired or None,
        list_inputs_for_unit=lambda *a: [{"consumed_input_revision": "SI", "manifest": {
            "parent_evidence_revision": "E", "parent_consumed_input_revision": "PI", "section_definition_revision": 2}}],
        get_operation=operations.get, command=command)
    service = CatalogService(repo, None, None)
    service.recover_inventory(inventory)
    service.recover_inventory(inventory)
    assert [action for action, _ in commands].count("fork_generation") == 1
    assert all(action != "enqueue" for action, _ in commands)
    assert desired["consumed_input_revision"] == "SI"


def test_ambiguous_section_inputs_retain_recovery_work():
    from papervault.knowledge.scheduler.catalog_service import CatalogService

    inventory = {"parent_unit_id": "P", "generation_id": "G", "inventory_revision": "I",
                 "parent_evidence_revision": "E", "required_units": ["S"], "optional_units": [],
                 "section_mapping_revisions": {"S": 1}, "enqueue_mode": "active", "desired_recipe_digest": "R"}
    manifest = {"parent_evidence_revision": "E", "section_definition_revision": 1}
    repo = SimpleNamespace(get_current_inventory=lambda *a: inventory,
        list_inputs_for_unit=lambda *a: [{"consumed_input_revision": name, "manifest": manifest} for name in ("A", "B")])
    with pytest.raises(ValueError, match="ambiguous"):
        CatalogService(repo, None, None).recover_inventory(inventory)
