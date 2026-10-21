"""Exercise the real catalog-only query through the shared opt-in PostgreSQL fixture."""
from datetime import datetime, timezone

from psycopg import sql

from papervault.store.repository import AuditEvent, Record, payload_hash


def test_catalog_publication_is_not_indexed_without_fresh_exact_output_observation(repo):
    from papervault.store.coverage import query_coverage
    from papervault.store.knowledge import KnowledgeRepository

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "coverage-schema"))
    repo.connection.execute(sql.SQL("UPDATE {} SET authority='catalog'").format(repo._table("knowledge_runtime")))
    catalog = KnowledgeRepository(repo)
    repo.upsert(Record.from_payload("A", {"key": "A", "title": "coverage paper"}),
                expected_revision=0, audit=AuditEvent("test", "coverage-record"))
    catalog.command("register_unit", "unit", {"unit_id": "paper:A", "kind": "paper",
                    "record_key": "A", "identity": "A"})
    catalog.command("register_input", "input", {"unit_id": "paper:A", "consumed_input_revision": "i1",
        "manifest": {"metadata_revision": 1, "artifacts": [{"artifact_id": "text", "kind": "fulltext",
                     "sha256": "a" * 64, "byte_length": 12}], "upstream_derivation_ids": []}})
    digest = catalog.command("register_recipe", "recipe", {"recipe": {"schema_version": 1,
        "output_contract": {"fulltext": {"required": ["graph"], "not_applicable": []}}}})["recipe_digest"]
    catalog.command("create_generation", "generation", {"generation_id": "g1", "candidate": False})
    identity = {"unit_id": "paper:A", "consumed_input_revision": "i1", "recipe_digest": digest,
                "generation_id": "g1"}
    catalog.command("register_inventory", "inventory", {"parent_unit_id": "paper:A", "inventory_revision": "whole1",
        "parent_evidence_revision": "i1", "parent_consumed_input_revision": "i1", "mode": "whole",
        "required_units": ["paper:A"], "optional_units": [], "section_mapping_revisions": {}})
    catalog.command("set_inventory", "current-inventory", {"parent_unit_id": "paper:A", "generation_id": "g1",
        "inventory_revision": "whole1", "expected_revision": 0, "expected_record_revision": 1})
    catalog.command("set_requirement", "requirement", {**identity, "inventory_revision": "whole1",
        "expected_revision": 0, "expected_record_revision": 1})
    job = catalog.command("enqueue", "job", identity)
    claim = catalog.command("claim", "claim", {"job_id": job["job_id"], "expected_revision": job["revision"],
                                               "authority_epoch": 1, "lease_seconds": 300})
    stamp = datetime.now(timezone.utc)
    receipt = {**claim, "receipt_id": "r1", "sealed": True, "durable": True, "achieved_tier": "fulltext",
        "output_inventory": [{"role": "graph", "output_id": "graph1", "store": "neo4j",
           "namespace": claim["namespace"], "backend_ids": [], "sha256": None, "byte_length": None,
           "verified_at": stamp.isoformat(), "verification": {"state": "verified_empty", "method": "count",
                                                                "evidence": {"count": 0, "verified": True}}}]}
    receipt["aggregate_layers"] = claim["base_aggregate_layers"] + [claim["namespace"]]
    receipt["selected_unit_receipts"] = {**claim["base_selected_unit_receipts"], claim["unit_id"]: receipt["receipt_id"]}
    receipt["selected_unit_receipts_digest"] = payload_hash(receipt["selected_unit_receipts"])
    catalog.command("seal_namespace", "seal", {"claim": claim, "receipt": receipt})
    catalog.command("publish_derivation", "publish", {**claim, "receipt": receipt,
        "expected_revision": claim["revision"], "expected_selection_revision": 0, "select": True})
    catalog.command("observe", "input-observation", {**identity, "observation_id": "input:o1",
        "role": "input_evidence", "store": "library", "output_id": "text", "state": "present",
        "observed_at": stamp.isoformat(), "detail": {"artifact_id": "text", "sha256": "a" * 64, "byte_length": 12}})
    before = repo.connection.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(repo._table("audit"))).fetchone()["n"]
    coverage = query_coverage(catalog, "A", evaluation_time=stamp)
    assert coverage["totals"] == {"registered": 1, "readable": 1, "published": 1, "indexed": 0}
    catalog.command("observe", "observation", {**claim, "observation_id": "o1", "receipt_id": "r1",
        "role": "graph", "store": "neo4j", "output_id": "graph1", "state": "present",
        "observed_at": stamp.isoformat()})
    coverage = query_coverage(catalog, "A", evaluation_time=stamp)
    assert coverage["records"][0]["indexed"] is True
    assert query_coverage(catalog, "A", evaluation_time=stamp, recipe_digest="other")["records"][0]["indexed"] is False
    after = repo.connection.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(repo._table("audit"))).fetchone()["n"]
    assert after == before + 1  # Only the explicit observe command creates an audit event.


def test_real_operator_section_query_uses_inventory_parent_selector_and_unit_scope(repo):
    from papervault.store.coverage import query_coverage
    from papervault.store.knowledge import KnowledgeRepository

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "coverage-operator-schema"))
    repo.connection.execute(sql.SQL("UPDATE {} SET authority='catalog'").format(repo._table("knowledge_runtime")))
    catalog = KnowledgeRepository(repo)
    catalog.command("register_unit", "parent", {"unit_id": "operator:O", "kind": "operator_document", "identity": "O"})
    catalog.command("register_unit", "section", {"unit_id": "section:S", "kind": "section", "identity": "S"})
    catalog.command("register_section", "mapping", {"unit_id": "section:S", "parent_unit_id": "operator:O",
        "mapping_revision": 1, "parent_evidence_revision": "e1", "definition": {"locator": "chapter1"}})
    for unit_id, revision in (("operator:O", "parent-input"), ("section:S", "section-input")):
        manifest = {"metadata_revision": "m1", "parent_evidence_revision": "e1", "artifacts": [
            {"artifact_id": "text:" + unit_id, "sha256": "a" * 64, "byte_length": 12, "role": "fulltext_evidence"}],
            "upstream_derivation_ids": []}
        if unit_id == "section:S":
            manifest["section_definition_revision"] = 1
        catalog.command("register_input", "input:" + unit_id, {"unit_id": unit_id,
            "consumed_input_revision": revision, "manifest": manifest})
    digest = catalog.command("register_recipe", "recipe", {"recipe": {"schema_version": 1,
        "output_contract": {"fulltext": {"required": ["graph"], "not_applicable": []}}}})["recipe_digest"]
    catalog.command("create_generation", "generation", {"generation_id": "g1", "candidate": False})
    catalog.command("register_inventory", "inventory", {"parent_unit_id": "operator:O", "inventory_revision": "v1",
        "parent_evidence_revision": "e1", "parent_consumed_input_revision": "parent-input", "mode": "sections",
        "required_units": ["section:S"], "optional_units": [], "section_mapping_revisions": {"section:S": 1}})
    catalog.command("set_inventory", "current-inventory", {"parent_unit_id": "operator:O", "generation_id": "g1",
        "inventory_revision": "v1", "expected_revision": 0})
    identity = {"unit_id": "section:S", "generation_id": "g1", "consumed_input_revision": "section-input", "recipe_digest": digest}
    catalog.command("set_requirement", "requirement", {**identity, "inventory_revision": "v1", "expected_revision": 0})
    job = catalog.command("enqueue", "job", identity)
    claim = catalog.command("claim", "claim", {"job_id": job["job_id"], "expected_revision": job["revision"], "authority_epoch": 1})
    stamp = datetime.now(timezone.utc)
    selected = {**claim["base_selected_unit_receipts"], "section:S": "rS"}
    receipt = {**claim, "receipt_id": "rS", "sealed": True, "durable": True, "achieved_tier": "fulltext",
        "aggregate_layers": claim["base_aggregate_layers"] + [claim["namespace"]], "selected_unit_receipts": selected,
        "selected_unit_receipts_digest": payload_hash(selected), "output_inventory": [{"role": "graph", "store": "neo4j",
            "output_id": "graphS", "namespace": claim["namespace"], "sha256": None, "byte_length": None,
            "verified_at": stamp.isoformat(), "verification": {"state": "verified_empty", "method": "count", "evidence": {"count": 0}}}]}
    catalog.command("seal_namespace", "seal", {"claim": claim, "receipt": receipt})
    catalog.command("publish_derivation", "publish", {**claim, "receipt": receipt, "expected_revision": claim["revision"],
        "expected_selection_revision": 0, "select": True})
    for unit_id, revision in (("operator:O", "parent-input"), ("section:S", "section-input")):
        catalog.command("observe", "input-observation:" + unit_id, {"observation_id": "input:" + unit_id,
            "unit_id": unit_id, "consumed_input_revision": revision, "role": "input_evidence", "store": "library",
            "output_id": "text:" + unit_id, "state": "present", "observed_at": stamp.isoformat(), "detail": {
                "artifact_id": "text:" + unit_id, "sha256": "a" * 64, "byte_length": 12}})
    catalog.command("observe", "output-observation", {**claim, "receipt_id": "rS", "observation_id": "graphS",
        "role": "graph", "store": "neo4j", "output_id": "graphS", "state": "present", "observed_at": stamp.isoformat()})
    assert catalog.get_requirement("operator:O", "g1") is None
    for scope in (None, "operator:O", "section:S"):
        result = query_coverage(catalog, scope, evaluation_time=stamp)
        assert len(result["records"]) == 1
        record = result["records"][0]
        assert record["key"] == "operator:O"
        assert record["readable"] is True
        assert record["indexed"] is True
        assert record["required_unit_outcomes"][0]["unit_id"] == "section:S"
