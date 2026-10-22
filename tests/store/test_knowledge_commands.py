"""Pure identity/receipt contracts; database transitions live in opt-in PG tests."""
import copy

import pytest


def test_effective_recipe_digest_ignores_credentials_but_tracks_prompt_and_versions():
    from papervault.store.knowledge import canonical_recipe

    a = {"schema_version": 1, "prompt": "extract", "model": "m1",
         "engine": {"version": "1", "api_key": "secret", "base_url": "https://u:p@example.test/v1?api_key=secret"}}
    spec, digest = canonical_recipe(a)
    assert "secret" not in str(spec)
    assert "u:p" not in str(spec)
    b = copy.deepcopy(a)
    b["engine"]["api_key"] = "other"
    b["engine"]["base_url"] = "https://other:pw@example.test/v1?api_key=other"
    assert canonical_recipe(b)[1] == digest
    b["prompt"] = "new prompt"
    assert canonical_recipe(b)[1] != digest
    assert a["engine"]["api_key"] == "secret"


def test_input_manifest_requires_precise_bytes_and_explicit_metadata_revision():
    from papervault.store.knowledge import validate_input_manifest

    valid = {"artifacts": [{"artifact_id": "a", "sha256": "a" * 64, "byte_length": 0}],
             "metadata_revision": "m1", "upstream_derivation_ids": []}
    validate_input_manifest(valid)
    for field in ("sha256", "byte_length", "artifact_id"):
        bad = copy.deepcopy(valid)
        del bad["artifacts"][0][field]
        with pytest.raises(ValueError):
            validate_input_manifest(bad)
    with pytest.raises(ValueError):
        validate_input_manifest({"artifacts": []})


def test_receipt_rejects_wrong_recipe_unverified_outputs_and_invented_empty_graph():
    from papervault.store.knowledge import validate_receipt

    claim = {"job_id": "j", "attempt_id": "a", "submission_id": "s", "authority_epoch": 1,
             "fencing_token": 2, "unit_id": "paper:A", "consumed_input_revision": "i",
             "recipe_digest": "r", "generation_id": "g", "namespace": "private-a"}
    receipt = {**claim, "receipt_id": "receipt", "sealed": True, "durable": True,
               "achieved_tier": "fulltext", "provider_observations": {"model": "unknown"},
               "output_inventory": [{"role": "graph", "output_id": "graph:a", "store": "neo4j",
                    "namespace": "private-a", "backend_ids": [], "sha256": None, "byte_length": None,
                    "verified_at": "2026-10-20T00:00:00+00:00",
                    "verification": {"state": "verified_empty", "method": "count",
                                     "evidence": {"count": 0, "verified": True}}}]}
    recipe = {"output_contract": {"fulltext": {"required": ["graph"], "not_applicable": []}}}
    validate_receipt(receipt, claim, recipe)
    wrong = {**receipt, "recipe_digest": "wrong"}
    with pytest.raises(ValueError, match="recipe_digest"):
        validate_receipt(wrong, claim, recipe)
    bad = copy.deepcopy(receipt)
    bad["output_inventory"][0]["verification"]["state"] = "unknown"
    with pytest.raises(ValueError):
        validate_receipt(bad, claim, recipe)
    bad["output_inventory"][0]["verification"] = {"state": "verified_empty"}
    with pytest.raises(ValueError):
        validate_receipt(bad, claim, recipe)


def test_test_dsn_guard_refuses_live_database_before_connecting():
    from papervault.store.knowledge import validate_test_dsn

    for dsn in ("postgresql://localhost/papervault_catalog", "dbname=papervault_catalog",
                "postgresql://localhost/papervault%5Fcatalog"):
        with pytest.raises(ValueError, match="live"):
            validate_test_dsn(dsn)
    with pytest.raises(ValueError, match="explicit"):
        validate_test_dsn("host=localhost")
    validate_test_dsn("dbname=papervault_p4_test")


def test_receipt_binds_copy_on_write_aggregate_base_and_resulting_unit_set():
    import hashlib
    import json
    from papervault.store.knowledge import validate_receipt

    claim = {"job_id": "j", "attempt_id": "a", "submission_id": "s", "authority_epoch": 1,
             "fencing_token": 2, "unit_id": "u2", "consumed_input_revision": "i",
             "recipe_digest": "r", "generation_id": "g", "namespace": "delta",
             "aggregate_base_manifest_revision": 4, "base_aggregate_layers": ["base"],
             "base_selected_unit_receipts": {"u1": "r1"}, "retired_receipt_ids": []}
    selected = {"u1": "r1", "u2": "r2"}
    digest = hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {**claim, "receipt_id": "r2", "sealed": True, "durable": True, "achieved_tier": "metadata",
        "aggregate_layers": ["base", "delta"], "selected_unit_receipts": selected,
        "selected_unit_receipts_digest": digest, "output_inventory": [{"role": "graph", "store": "neo4j",
            "output_id": "g", "namespace": "delta", "sha256": None, "byte_length": None,
            "verified_at": "2026-10-20T00:00:00Z", "verification": {
                "state": "verified_empty", "method": "scan", "evidence": {"node_count": 0, "edge_count": 0}}}]}
    recipe = {"output_contract": {"metadata": {"required": ["graph"], "not_applicable": []}}}
    validate_receipt(receipt, claim, recipe)
    bad = copy.deepcopy(receipt)
    bad["aggregate_layers"] = ["delta"]
    with pytest.raises(ValueError, match="aggregate"):
        validate_receipt(bad, claim, recipe)
    bad = copy.deepcopy(receipt)
    bad["aggregate_base_manifest_revision"] = 3
    with pytest.raises(ValueError, match="aggregate"):
        validate_receipt(bad, claim, recipe)


def test_recipe_sanitizes_header_and_camelcase_credentials_and_database_urls():
    from papervault.store.knowledge import canonical_recipe

    recipe = {"schema_version": 1, "apiKey": "secret1", "headers": {"X-Api-Key": "secret2"},
              "database_url": "postgresql://user:secret3@db.test/catalog?password=secret4",
              "max_tokens": 500, "credentials": {"value": "secret5"}}
    spec, _ = canonical_recipe(recipe)
    assert "secret" not in str(spec)
    assert spec["max_tokens"] == 500
    assert "db.test/catalog" in spec["database_url"]


def test_schema_qualification_preserves_catalog_authority_and_store_literals():
    from papervault.store.repository import render_catalog_ddl

    ddl = "CREATE SCHEMA IF NOT EXISTS catalog; CREATE TABLE catalog.t (authority TEXT DEFAULT 'catalog'); SELECT 'catalog.jobs'::regclass; SELECT 'catalog' AS store;"
    rendered = render_catalog_ddl(ddl, "catalog_test_unit")
    assert rendered == 'CREATE SCHEMA IF NOT EXISTS "catalog_test_unit"; CREATE TABLE "catalog_test_unit".t (authority TEXT DEFAULT \'catalog\'); SELECT \'"catalog_test_unit".jobs\'::regclass; SELECT \'catalog\' AS store;'


def embedding_spec(model="model-a"):
    return {"schema_version": 1, "model": model, "dimension": 1024, "max_length": 8192,
            "use_fp16": True, "representation": "BGEM3FlagModel.encode:dense_vecs:cosine",
            "implementation_sha256": "a" * 64, "packages": {"FlagEmbedding": "1", "torch": "2"}}


def test_retrieval_contract_preserves_complete_embedding_identity_and_ignores_nonvector_recipes():
    from papervault.store.knowledge import retrieval_contract

    spec = embedding_spec()
    recipe = {"schema_version": 1, "embedding_contract": spec,
              "output_contract": {"fulltext": {"required": ["chunks_vdb", "graph"]},
                                  "metadata": {"required": ["metadata"], "not_applicable": ["chunks_vdb"]}}}
    assert retrieval_contract(recipe) == spec
    changed = copy.deepcopy(recipe)
    changed["embedding_contract"]["model"] = "model-b"
    assert retrieval_contract(changed) != retrieval_contract(recipe)
    assert retrieval_contract({"output_contract": {"fulltext": {"required": ["graph"]}}}) is None
    assert retrieval_contract({"output_contract": {"metadata": {"required": ["metadata"],
                                                               "not_applicable": ["vectors"]}}}) is None
    result = retrieval_contract(recipe)
    result["packages"]["torch"] = "mutated"
    assert recipe["embedding_contract"] == spec


@pytest.mark.parametrize("field,value", [("dimension", 0), ("dimension", True), ("model", ""),
    ("implementation_sha256", "unknown"), ("representation", ""), ("packages", {}), ("schema_version", 2)])
def test_vector_recipe_requires_explicit_valid_embedding_contract(field, value):
    from papervault.store.knowledge import retrieval_contract

    recipe = {"output_contract": {"abstract": {"required": ["vectors"]}}}
    with pytest.raises(ValueError, match="embedding"):
        retrieval_contract(recipe)
    recipe["embedding_contract"] = {**embedding_spec(), field: value}
    with pytest.raises(ValueError, match="embedding"):
        retrieval_contract(recipe)


def test_generation_contract_reducer_binds_once_and_metadata_never_overwrites():
    from papervault.store.knowledge import ClaimRejected
    from papervault.store.knowledge_commands import generation_contract

    class Repository:
        contract = None
        tier = "metadata"
        model = "model-a"
        def get_recipe(self, digest):
            return {"embedding_contract": embedding_spec(self.model), "output_contract": {
                "fulltext": {"required": ["vectors"]}, "metadata": {"required": ["metadata"]}}}
        def get_input(self, revision):
            return {"manifest": {"tier": self.tier}}
        def get_generation(self, generation):
            return {"retrieval_contract": self.contract}
        def _update(self, table, values, **keys):
            self.contract = values["retrieval_contract"]
    repository = Repository()
    job = {"recipe_digest": "r", "consumed_input_revision": "i", "target_generation": "g"}
    assert generation_contract(repository, job, bind=True) is False
    assert repository.contract is None
    repository.tier = "fulltext"
    assert generation_contract(repository, job, bind=True) is True
    assert generation_contract(repository, job, bind=True) is False
    repository.model = "model-b"
    with pytest.raises(ClaimRejected, match="fresh generation"):
        generation_contract(repository, job, bind=True)
    repository.tier = "metadata"
    assert generation_contract(repository, job, bind=True) is False
    assert repository.contract == embedding_spec("model-a")
    with pytest.raises(ClaimRejected, match="fresh generation"):
        generation_contract(repository, job, tier="fulltext")


def test_recipe_token_counts_remain_identity_while_authentication_tokens_are_redacted():
    from papervault.store.knowledge import canonical_recipe

    recipe = {"schema_version": 1, "chunking": {"token_size": 1200, "overlap_token_size": 100},
        "execution_settings": {"embedding_max_token_size": 8192, "max_token_size": 4096, "min_tokens": 5},
        "token": "private-one", "auth_token": "private-two", "apiToken": "private-three",
        "access_token": "private-four", "refresh_tokens": "private-five", "hf_token": "private-six"}
    spec, digest = canonical_recipe(recipe)
    assert spec["chunking"] == recipe["chunking"]
    assert spec["execution_settings"] == recipe["execution_settings"]
    assert "private" not in str(spec)
    changed = copy.deepcopy(recipe)
    changed["chunking"]["token_size"] = 2400
    assert canonical_recipe(changed)[1] != digest
    changed = copy.deepcopy(recipe)
    changed["execution_settings"]["embedding_max_token_size"] = 4096
    assert canonical_recipe(changed)[1] != digest
    changed = copy.deepcopy(recipe)
    changed["auth_token"] = "different-secret"
    assert canonical_recipe(changed)[1] == digest


@pytest.mark.parametrize("counter", [-1, None, True, "3", 2**31])
def test_invalid_legacy_checkpoint_preserves_raw_quarantines_unit_and_creates_no_reducer(counter):
    from psycopg import sql
    from papervault.store.knowledge_commands import migrate_legacy_checkpoint

    class Connection:
        def execute(self, query, args):
            return [{"id": "legacy", "detail": {"legacy_consecutive_failures": counter,
                     "evidence": {"raw": {"attempts": counter, "status": "error_parked"}}}}]
    class Catalog:
        def _table(self, table):
            return sql.Identifier("catalog", table)
    class Repository:
        connection = Connection()
        repository = Catalog()
        inserted = []
        updated = []
        def _one(self, table, **keys):
            return {"authority": "catalog"} if table == "knowledge_runtime" else None
        def get_unit(self, unit):
            return {"unit_id": unit, "revision": 1}
        def get_reducer(self, unit, revision):
            return None
        def _insert(self, table, values):
            self.inserted.append((table, values))
            return values
        def _update(self, table, values, **keys):
            self.updated.append((table, values))
            return values
    repository = Repository()
    result = migrate_legacy_checkpoint(repository, "bad-unit", "first-input")
    assert [table for table, _ in repository.inserted] == ["knowledge_legacy_checkpoint_applications"]
    assert result["evidence"]["counter_state"] == "unknown"
    assert result["evidence"]["historical_events_replayed"] is False
    assert result["evidence"]["legacy_jobs"][0]["detail"]["legacy_consecutive_failures"] == counter
    table, values = repository.updated[0]
    assert table == "units" and values["quarantine"] is True
    assert "invalid" in values["quarantine_reason"]["reason"]


@pytest.mark.parametrize("old_inventory,invalidated,expected", [("v1", False, True), ("v2", False, False), ("v2", True, True)])
def test_inventory_selection_invalidates_stale_included_requirements_before_snapshot(monkeypatch, old_inventory, invalidated, expected):
    from papervault.store import knowledge_commands as commands

    class Repository:
        desired = {"unit_id": "S", "inventory_revision": old_inventory, "invalidated": invalidated,
                   "excluded": False, "enqueue_mode": "active", "revision": 1,
                   "consumed_input_revision": "section-input"}
        repository = None
        def __init__(self):
            self.repository = self
        def get_inventory(self, parent, revision):
            return {"required_units": ["S"], "inventory_revision": revision}
        def _one(self, table, **keys):
            return None
        def get_generation(self, generation):
            return {"state": "active"}
        def get_unit(self, unit):
            return {"record_key": None}
        def get_input(self, identity):
            assert identity == "section-input"
            return {"manifest": {"artifacts": [], "upstream_derivation_ids": []}}
        def _rows(self, table, **keys):
            return [{"unit_id": "S"}] if table == "section_mappings" else [dict(self.desired)]
        def _update(self, table, values, **keys):
            assert table == "knowledge_requirements"
            self.desired.update(values)
            return dict(self.desired)
        def _insert(self, table, values):
            assert table == "current_inventories"
            return values
    repository = Repository()
    snapshots = []
    monkeypatch.setattr(commands, "snapshot", lambda k, g: snapshots.append(dict(k.desired)))
    monkeypatch.setattr(commands, "bump_requirements", lambda k: None)
    monkeypatch.setattr(commands, "outbox", lambda *a, **kw: None)
    commands.dispatch(repository, "set_inventory", "inventory-change", {"parent_unit_id": "O",
        "generation_id": "g", "inventory_revision": "v2", "expected_revision": 0})
    assert snapshots[0]["invalidated"] is expected
    assert repository.desired["inventory_revision"] == old_inventory
    assert repository.desired["excluded"] is False


def test_legacy_checkpoint_seeding_is_deferred_until_catalog_authority():
    from papervault.store.knowledge_commands import migrate_legacy_checkpoint

    class Repository:
        def _one(self, table, **keys):
            assert table == "knowledge_runtime"
            return {"authority": "legacy"}
    assert migrate_legacy_checkpoint(Repository(), "U", "input-at-C0") is None


def test_serving_manifest_keeps_pinned_excluded_parent_ineligible_when_read_view_is_synthetic(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from papervault.store import coverage
    from papervault.store.knowledge import KnowledgeRepository

    class Repository:
        connection = SimpleNamespace(info=SimpleNamespace(transaction_status=SimpleNamespace(name="INTRANS")),
                                     transaction=nullcontext)
        def _serving_pointer(self):
            return {"generation_id": "g", "manifest_revision": 2, "revision": 3, "manifest": {
                "entries": [{"unit_id": "O", "selected_derivation_id": "old-whole", "excluded": True}]}}
    monkeypatch.setattr(coverage, "query_coverage", lambda *args, **kwargs: {"units": [{
        "unit_id": "O", "derivation_id": "old-whole", "source_eligible": True, "tier_covered": True,
        "quarantine": False, "excluded": False, "readable": True, "achieved_tier": "fulltext"}]})
    result = KnowledgeRepository.get_serving_manifest(Repository())
    assert result["entries"][0]["eligible"] is False


@pytest.mark.parametrize("blocked_by", ["parked", "quarantined"])
def test_first_claim_commits_new_blocking_legacy_checkpoint_without_execution(monkeypatch, blocked_by):
    from papervault.store import knowledge_commands as commands
    from papervault.store.knowledge import ClaimRejected

    class Repository:
        applied = False
        job = {"id": "J", "kind": "distill", "revision": 1, "unit_id": "U", "consumed_input_revision": "I", "state": "queued"}
        def _one(self, table, **keys):
            if table == "knowledge_runtime":
                return {"authority": "catalog", "authority_epoch": 2}
            assert table == "knowledge_legacy_checkpoint_applications"
            return {"applied": True} if self.applied else None
        def get_job(self, job):
            return dict(self.job)
        def get_unit(self, unit):
            return {"quarantine": self.applied and blocked_by == "quarantined"}
        def get_reducer(self, unit, input_revision):
            return {"parked": self.applied and blocked_by == "parked"}
        def _update(self, table, values, **keys):
            assert table == "jobs" and set(values) == {"revision"}
            self.job.update(values)
            return dict(self.job)
    repository = Repository()
    def apply(k, unit, input_revision):
        k.applied = True
        return {"applied": True}
    def execution_forbidden(*args, **kwargs):
        raise ClaimRejected("blocking baseline must commit before execution eligibility")
    monkeypatch.setattr(commands, "migrate_legacy_checkpoint", apply)
    monkeypatch.setattr(commands, "eligible", execution_forbidden)
    result = commands.dispatch(repository, "claim", "claim:J:1", {"job_id": "J", "expected_revision": 1, "authority_epoch": 2})
    assert repository.applied
    assert result["claimed"] is False and result["attempt_id"] is None and result["revision"] == 2
    assert result[blocked_by] is True
    assert repository.job["state"] == "queued"


@pytest.mark.parametrize("change,allowed", [
    ({}, True), ({"override": None}, False),
    ({"override": {"key": "B"}}, False),
    ({"override": {"reasons": ["other"]}}, False),
    ({"override": {"scope": ["other"]}}, False),
    ({"required_revision": 3}, False),
    ({"policy": "clear"}, False),
])
@pytest.mark.parametrize("unit_kind", ["paper", "section"])
def test_knowledge_admission_checks_full_scoped_policy(change, allowed, unit_kind):
    from types import SimpleNamespace
    from papervault.library.models import Paper
    from papervault.store.knowledge import ClaimRejected
    from papervault.store.knowledge_commands import eligible
    from papervault.store.repository import Record

    state = {"policy": "override", "policy_reasons": ["off_domain"], "required_revision": 2,
             "override": {"key": "A", "scope": ["acquire"], "reasons": ["off_domain"], "revision": 2}}
    if isinstance(change.get("override"), dict):
        state["override"].update(change["override"])
    else:
        state.update(change)
    record = Record.from_payload("A", Paper(key="A", title="Policy fixture", terminal_state=state).model_dump(mode="json"))
    desired = {"invalidated": False, "excluded": False, "enqueue_mode": "active",
               "consumed_input_revision": "I", "recipe_digest": "R", "inventory_revision": "v1"}
    source = {"kind": "paper", "unit_id": "paper:A", "record_key": "A", "quarantine": False}
    section = {"kind": "section", "unit_id": "section:S", "record_key": None, "quarantine": False}
    repository = SimpleNamespace(repository=SimpleNamespace(get=lambda key: record),
        get_input=lambda revision: {"manifest": {"section_definition_revision": 1}},
        get_unit=lambda unit: section if unit == "section:S" else source,
        get_section=lambda *args: {"parent_unit_id": "paper:A"},
        get_current_inventory=lambda *args: {"inventory_revision": "v1", "invalidated": False},
        get_generation=lambda generation: {"state": "active"}, get_requirement=lambda *args: desired)
    job = {"unit_id": "section:S" if unit_kind == "section" else "paper:A",
           "quarantine": False, "target_generation": "G",
           "consumed_input_revision": "I", "recipe_digest": "R"}
    if allowed:
        assert eligible(repository, job) == desired
    else:
        with pytest.raises(ClaimRejected, match="record policy"):
            eligible(repository, job)
