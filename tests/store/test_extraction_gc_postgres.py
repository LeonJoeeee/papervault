"""P5c retention, admission races and recoverable deletion on dedicated PostgreSQL."""

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest

from papervault.store.extraction import ExtractionStore
from papervault.store.knowledge import ClaimRejected, KnowledgeRepository
from papervault.store.repository import AuditEvent, PostgresCatalogRepository, Record
from test_extraction_postgres import extraction_catalog as extraction_catalog, frozen, publish


def api():
    assert importlib.util.find_spec("papervault.store.extraction_gc") is not None
    from papervault.store import extraction_gc
    return extraction_gc


def command(k, action, payload):
    return k.command(action, action + ":" + uuid4().hex, payload)


def window(k):
    start = datetime.now(timezone.utc) - timedelta(seconds=1)
    return command(k, "begin_maintenance_window", {
        "window_id": "gc-window", "kind": "monthly_reboot", "starts_at": start.isoformat(),
        "ends_at": (start + timedelta(minutes=5)).isoformat(), "operator_nonce": "staging-window",
        "expected_revision": 0})


def lease(k, owner="gc-worker"):
    return command(k, "acquire_maintenance_lease", {
        "resource": "extract_gc", "window_id": "gc-window", "owner": owner, "lease_seconds": 120})


def policy(k, completion, vault, disposable=True):
    row = k._one("extract_retention", completion_id=completion["completion_id"])
    return command(k, "set_extract_retention", {"completion_id": completion["completion_id"],
        "vault": str(vault), "disposable": disposable, "expected_revision": row["revision"],
        "operator_nonce": uuid4().hex, "reason": "explicit isolated fixture disposition"})


@pytest.fixture
def superseded(extraction_catalog, tmp_path):
    k = extraction_catalog
    first = frozen(k, tmp_path)
    publish(k, first, tmp_path)
    second = frozen(k, tmp_path, previous=first["completion_id"])
    publish(k, second, tmp_path, expected_head_revision=1)
    return k, first, second, tmp_path


def age(k, completion, days=91):
    k.connection.execute(sql.SQL("UPDATE {} SET superseded_at=clock_timestamp()-%s::interval "
        "WHERE completion_id=%s").format(k.repository._table("extract_retention")),
        (str(days) + " days", completion["completion_id"]))


def ready(superseded):
    k, first, second, vault = superseded
    policy(k, first, vault)
    age(k, first)
    window(k)
    return k, first, second, vault, lease(k)


def row(plan, completion):
    return next(item for item in plan["extracts"] if item["completion_id"] == completion["completion_id"])


def test_supersession_is_recorded_at_publication_and_protection_is_default(superseded):
    api()
    k, first, second, vault = superseded
    old = k._one("extract_retention", completion_id=first["completion_id"])
    current = k._one("extract_retention", completion_id=second["completion_id"])
    assert old["superseded_at"] is not None
    assert current["superseded_at"] is None
    assert not old["disposable"] and not current["disposable"]
    age(k, first)
    assert "protected" in row(api().plan_gc(k, vault), first)["reasons"]
    assert ExtractionStore(vault).completion(first["completion_id"]) == first


def test_disposable_retention_is_ninety_days_after_supersession(superseded):
    k, first, second, vault = superseded
    policy(k, first, vault)
    age(k, first, days=89.999)
    assert not row(api().plan_gc(k, vault), first)["eligible"]
    age(k, first, days=90)
    assert row(api().plan_gc(k, vault), first)["eligible"]
    policy(k, second, vault)
    age(k, second)
    assert "current" in row(api().plan_gc(k, vault), second)["reasons"]


@pytest.mark.parametrize("kind", ["reader", "job", "recovery"])
def test_reader_job_and_recovery_pins_prevent_deletion(superseded, kind):
    k, first, _, vault, claim = ready(superseded)
    pin = command(k, "pin_extract", {"completion_id": first["completion_id"], "owner": "consumer",
        "kind": kind, **({"ttl_seconds": 30} if kind == "reader" else {})})
    assert "pinned" in row(api().plan_gc(k, vault), first)["reasons"]
    with pytest.raises(ClaimRejected, match="pinned"):
        api().collect_extract(k, vault, first["completion_id"], claim)
    command(k, "release_extract_pin", {"pin_id": pin["pin_id"], "owner": "consumer",
        "expected_revision": pin["revision"], "reason": "completed consumer",
        "operator_nonce": "reviewed", "disposition": "reviewed_recovery_disposition"})
    assert row(api().plan_gc(k, vault), first)["eligible"]


def test_gc_keeps_sources_receipts_and_current_version(superseded):
    k, first, second, vault, claim = ready(superseded)
    source = vault / first["input_manifest"]["artifacts"][0]["path"]
    receipt = vault / "extracts/versions" / first["completion_id"] / "completion.json"
    result = api().collect_extract(k, vault, first["completion_id"], claim)
    assert result["state"] == "deleted"
    assert not (vault / first["output_inventory"][0]["path"]).exists()
    assert source.read_bytes() == b"%PDF frozen source"
    assert receipt.is_file()
    assert k.get_extraction_receipt(first["completion_id"])["receipt"] == first
    assert ExtractionStore(vault).completion(second["completion_id"]) == second
    assert api().collect_extract(k, vault, first["completion_id"], claim)["state"] == "deleted"


def test_storage_failure_retains_tombstone_and_can_retry(superseded, monkeypatch):
    k, first, second, vault, claim = ready(superseded)
    unlink = os.unlink
    def broken(path, *args, **kwargs):
        if str(path) == "output.md":
            raise OSError("injected storage failure")
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os, "unlink", broken)
        with pytest.raises(OSError, match="injected storage"):
            api().collect_extract(k, vault, first["completion_id"], claim)
    assert k._one("extract_deletions", completion_id=first["completion_id"])["state"] == "deleting"
    assert (vault / first["output_inventory"][0]["path"]).is_file()
    with pytest.raises(ClaimRejected, match="tombston"):
        command(k, "pin_extract", {"completion_id": first["completion_id"], "owner": "late-reader",
            "kind": "reader", "ttl_seconds": 30})
    assert api().collect_extract(k, vault, first["completion_id"], claim)["state"] == "deleted"
    assert ExtractionStore(vault).completion(second["completion_id"]) == second


def test_unlink_before_catalog_acknowledgement_is_recoverable(superseded, monkeypatch):
    k, first, _, vault, claim = ready(superseded)
    original = k.command
    def lost(action, operation, data):
        if action == "finish_extract_deletion":
            raise ConnectionError("catalog unavailable after unlink")
        return original(action, operation, data)
    with monkeypatch.context() as patch:
        patch.setattr(k, "command", lost)
        with pytest.raises(ConnectionError, match="after unlink"):
            api().collect_extract(k, vault, first["completion_id"], claim)
    assert not (vault / first["output_inventory"][0]["path"]).exists()
    assert k._one("extract_deletions", completion_id=first["completion_id"])["state"] == "deleting"
    assert api().collect_extract(k, vault, first["completion_id"], claim)["state"] == "deleted"


def test_expired_or_replaced_maintenance_lease_cannot_unlink(superseded):
    k, first, _, vault, stale = ready(superseded)
    k.connection.execute(sql.SQL("UPDATE {} SET expires_at=clock_timestamp()-interval '1 second'").format(
        k.repository._table("maintenance_leases")))
    fresh = lease(k, "replacement")
    with pytest.raises(ClaimRejected, match="lease"):
        api().collect_extract(k, vault, first["completion_id"], stale)
    assert (vault / first["output_inventory"][0]["path"]).exists()
    assert fresh["fencing_token"] > stale["fencing_token"]
    assert api().collect_extract(k, vault, first["completion_id"], fresh)["state"] == "deleted"


@pytest.mark.parametrize("target", ["file", "parent", "root", "hardlink"])
def test_gc_refuses_symlinks_and_shared_files(superseded, target):
    k, first, _, vault, claim = ready(superseded)
    output = vault / first["output_inventory"][0]["path"]
    source = vault / first["input_manifest"]["artifacts"][0]["path"]
    before = source.read_bytes()
    if target == "file":
        output.unlink()
        output.symlink_to(source)
    elif target == "parent":
        retained = output.parent.with_name("retained-output")
        output.parent.rename(retained)
        output.parent.symlink_to(retained, target_is_directory=True)
    elif target == "root":
        alias = vault / "vault-alias"
        alias.symlink_to(vault, target_is_directory=True)
        vault = alias
    else:
        os.link(output, vault / "protected-shared-copy")
    with pytest.raises((ValueError, OSError, ClaimRejected)):
        api().collect_extract(k, vault, first["completion_id"], claim)
    assert source.read_bytes() == before
    if target == "hardlink":
        assert (vault / "protected-shared-copy").read_bytes() == output.read_bytes()


def test_catalog_record_reference_blocks_gc_and_late_reference_is_rejected(superseded):
    k, first, _, vault, claim = ready(superseded)
    record = Record.from_payload("B", {"key": "B", "title": "Retained research asset",
        "md_path": first["output_inventory"][0]["path"]})
    k.repository.upsert(record, expected_revision=0, audit=AuditEvent("test", "reference"))
    assert "referenced" in row(api().plan_gc(k, vault), first)["reasons"]
    k.repository.upsert(Record.from_payload("B", {"key": "B", "title": "New selection"}),
        expected_revision=1, audit=AuditEvent("test", "advance"))
    api().collect_extract(k, vault, first["completion_id"], claim)
    with pytest.raises(ClaimRejected, match="tombston"):
        k.repository.upsert(record, expected_revision=2, audit=AuditEvent("test", "late_reference"))
    assert k.repository.get("B").payload["title"] == "New selection"


@pytest.mark.parametrize("alias", ["dot", "parent", "absolute", "symlink"])
def test_record_aliases_protect_bytes_and_cannot_reenter_after_tombstone(superseded, alias):
    k, first, _, vault, claim = ready(superseded)
    path = first["output_inventory"][0]["path"]
    if alias == "symlink":
        (vault / "selected.md").symlink_to(vault / path)
        path = "selected.md"
    else:
        path = {"dot": "./" + path, "parent": "extracts/../" + path,
                "absolute": str(vault) + "/./" + path}[alias]
    record = Record.from_payload("B", {"key": "B", "title": "Selected alias", "md_path": path})
    k.repository.upsert(record, expected_revision=0, audit=AuditEvent("test", "alias"))
    with pytest.raises(ClaimRejected, match="referenced"):
        api().collect_extract(k, vault, first["completion_id"], claim)
    k.repository.upsert(Record.from_payload("B", {"key": "B", "title": "New selection"}),
        expected_revision=1, audit=AuditEvent("test", "advance"))
    api().collect_extract(k, vault, first["completion_id"], claim)
    with pytest.raises(ClaimRejected, match="tombston"):
        k.repository.upsert(record, expected_revision=2, audit=AuditEvent("test", "late_alias"))


def test_physical_input_alias_protects_disposable_extract(superseded):
    k, first, _, vault, claim = ready(superseded)
    command(k, "register_unit", {"unit_id": "alias-consumer", "kind": "operator_document", "identity": "alias-consumer"})
    artifact = {**first["output_inventory"][0], "artifact_id": "alias",
                "path": "./" + first["output_inventory"][0]["path"]}
    command(k, "register_input", {"unit_id": "alias-consumer", "consumed_input_revision": "alias-input",
        "manifest": {"metadata_revision": "m", "artifacts": [artifact], "upstream_derivation_ids": []}})
    with pytest.raises(ClaimRejected, match="referenced"):
        api().collect_extract(k, vault, first["completion_id"], claim)


def test_finish_cannot_acknowledge_a_dangling_symlink_as_absent(superseded):
    k, first, _, vault, claim = ready(superseded)
    payload = {"completion_id": first["completion_id"], "vault": str(vault), "lease": claim}
    intent = command(k, "prepare_extract_deletion", payload)
    output = vault / first["output_inventory"][0]["path"]
    output.unlink()
    output.symlink_to(vault / "missing-target")
    with pytest.raises((ClaimRejected, ValueError, OSError)):
        command(k, "finish_extract_deletion", {**payload, "deletion_id": intent["deletion_id"]})
    assert k._one("extract_deletions", completion_id=first["completion_id"])["state"] == "deleting"


def test_reference_admission_and_gc_are_serialized(superseded, catalog_test_dsn):
    k, first, _, vault, claim = ready(superseded)
    begun, done = Event(), Event()
    def deletion():
        with psycopg.connect(catalog_test_dsn, autocommit=True) as conn:
            other = KnowledgeRepository(PostgresCatalogRepository(conn, schema=k.repository.schema))
            begun.set()
            try:
                return api().collect_extract(other, vault, first["completion_id"], claim)
            finally:
                done.set()
    with ThreadPoolExecutor(max_workers=1) as workers:
        with k.repository._write(AuditEvent("test", "held_admission")):
            future = workers.submit(deletion)
            assert begun.wait(5)
            command(k, "pin_extract", {"completion_id": first["completion_id"], "owner": "reader",
                "kind": "reader", "ttl_seconds": 30})
            assert not done.is_set()
        with pytest.raises(ClaimRejected, match="pinned"):
            future.result(timeout=5)
    assert (vault / first["output_inventory"][0]["path"]).is_file()


def test_concurrent_successor_publication_and_gc_keep_current_and_shared_source(superseded, catalog_test_dsn):
    k, first, second, vault, claim = ready(superseded)
    successor = frozen(k, vault, previous=second["completion_id"])
    entered = [Event(), Event()]
    def run(index):
        with psycopg.connect(catalog_test_dsn, autocommit=True) as conn:
            other = KnowledgeRepository(PostgresCatalogRepository(conn, schema=k.repository.schema))
            entered[index].set()
            if index:
                return publish(other, successor, vault, expected_head_revision=2)
            return api().collect_extract(other, vault, first["completion_id"], claim)
    with ThreadPoolExecutor(max_workers=2) as workers:
        with k.repository._write(AuditEvent("test", "held_publication_and_gc")):
            futures = [workers.submit(run, index) for index in range(2)]
            assert all(event.wait(5) for event in entered)
        results = [future.result(timeout=10) for future in futures]
    assert results[0]["state"] == "deleted" and results[1]["registration"] == "registered"
    unit_id = k.get_extraction_receipt(successor["completion_id"])["unit_id"]
    assert k.get_extraction_head(unit_id)["completion_id"] == successor["completion_id"]
    for completion in (second, successor):
        assert (vault / completion["output_inventory"][0]["path"]).is_file()
    assert (vault / first["input_manifest"]["artifacts"][0]["path"]).read_bytes() == b"%PDF frozen source"


def test_expiry_during_prepare_verification_stops_before_unlink(superseded, monkeypatch):
    import time
    from papervault.store.extraction import ExtractionStore
    k, first, _, vault, claim = ready(superseded)
    original = ExtractionStore.verify
    def slow(self, completion):
        time.sleep(0.5)
        return original(self, completion)
    k.connection.execute(sql.SQL("UPDATE {} SET expires_at=clock_timestamp()+interval '0.15 seconds'").format(
        k.repository._table("maintenance_leases")))
    monkeypatch.setattr(ExtractionStore, "verify", slow)
    with pytest.raises(TimeoutError):
        api().collect_extract(k, vault, first["completion_id"], claim)
    assert (vault / first["output_inventory"][0]["path"]).is_file()


def consumer(k, completion):
    from papervault.store.repository import payload_hash
    command(k, "register_unit", {"unit_id": "consumer", "kind": "operator_document", "identity": "consumer"})
    manifest = {"metadata_revision": "m", "artifacts": [{"artifact_id": "copied-input",
        "sha256": completion["output_inventory"][0]["sha256"], "byte_length": completion["output_inventory"][0]["byte_length"]}],
        "upstream_derivation_ids": ["extraction-derivation:" + completion["completion_id"]]}
    identity = "input:" + payload_hash(manifest)
    command(k, "register_input", {"unit_id": "consumer", "consumed_input_revision": identity, "manifest": manifest})
    recipe = command(k, "register_recipe", {"recipe": completion["recipe"]})
    command(k, "create_generation", {"generation_id": "consumer-generation", "candidate": True})
    return {"unit_id": "consumer", "consumed_input_revision": identity, "recipe_digest": recipe["recipe_digest"],
        "generation_id": "consumer-generation", "expected_revision": 0}


def test_queued_job_protects_upstream_extract(superseded):
    k, first, _, vault, claim = ready(superseded)
    desired = consumer(k, first)
    command(k, "enqueue", desired)
    assert "referenced" in row(api().plan_gc(k, vault), first)["reasons"]
    with pytest.raises(ClaimRejected, match="referenced"):
        api().collect_extract(k, vault, first["completion_id"], claim)


def test_earlier_input_cannot_be_selected_after_tombstone(superseded):
    k, first, _, vault, claim = ready(superseded)
    desired = consumer(k, first)
    api().collect_extract(k, vault, first["completion_id"], claim)
    with pytest.raises(ClaimRejected, match="tombston"):
        command(k, "set_requirement", desired)
    with pytest.raises(ClaimRejected, match="tombston"):
        command(k, "enqueue", desired)


def test_lease_loss_during_storage_verification_stops_before_unlink(superseded, monkeypatch):
    k, first, _, vault, claim = ready(superseded)
    original = os.read
    expired = False
    def slow(fd, size):
        nonlocal expired
        if not expired:
            expired = True
            k.connection.execute(sql.SQL("UPDATE {} SET expires_at=clock_timestamp()-interval '1 second'").format(
                k.repository._table("maintenance_leases")))
        return original(fd, size)
    with monkeypatch.context() as patch:
        patch.setattr(os, "read", slow)
        with pytest.raises(ClaimRejected, match="lease"):
            api().collect_extract(k, vault, first["completion_id"], claim)
    assert expired
    assert (vault / first["output_inventory"][0]["path"]).is_file()


def test_deletion_tombstone_cannot_be_erased_or_rebound(superseded):
    k, first, _, vault, claim = ready(superseded)
    api().collect_extract(k, vault, first["completion_id"], claim)
    table = k.repository._table("extract_deletions")
    for statement in (sql.SQL("DELETE FROM {}"), sql.SQL("UPDATE {} SET state='deleting'"),
                      sql.SQL("UPDATE {} SET output='{{}}'::jsonb")):
        with pytest.raises(psycopg.Error, match="immutable|append-only|tombstone"):
            k.connection.execute(statement.format(table))


def test_completed_disposal_is_a_valid_snapshot_dependency(superseded, tmp_path):
    import shutil
    from papervault.store.recovery_postgres import catalog_barrier, export_catalog_state
    from papervault.store.recovery_source import verify_file_dependencies
    k, first, _, vault, claim = ready(superseded)
    api().collect_extract(k, vault, first["completion_id"], claim)
    with catalog_barrier(k.repository, "test-gc") as checkpoint:
        catalog = export_catalog_state(k.repository, checkpoint)
    package = tmp_path / "package"
    shutil.copytree(vault / "extracts", package / "extracts/vault")
    (package / "sources").mkdir()
    verify_file_dependencies(package, catalog)
    # The exception is exact and only for disposed outputs, never their sources.
    (package / "extracts/vault/inputs" / first["input_manifest"]["artifacts"][0]["sha256"]).unlink()
    with pytest.raises(ValueError, match="absent"):
        verify_file_dependencies(package, catalog)


def test_set_inventory_cannot_readmit_tombstoned_input(superseded):
    k, first, _, vault, lease = ready(superseded)
    desired = consumer(k, first)
    command(k, 'set_requirement', {**desired, 'excluded': True})
    command(k, 'register_inventory', {'parent_unit_id': 'consumer', 'inventory_revision': 'whole',
        'mode': 'whole', 'parent_evidence_revision': 'copied-input', 'required_units': ['consumer'],
        'parent_consumed_input_revision': desired['consumed_input_revision']})
    api().collect_extract(k, vault, first['completion_id'], lease)
    with pytest.raises(ClaimRejected, match='tombston'):
        command(k, 'set_inventory', {'parent_unit_id': 'consumer', 'inventory_revision': 'whole',
            'generation_id': desired['generation_id'], 'expected_revision': 0})


def test_reader_manifest_cannot_readmit_tombstoned_input(superseded):
    k, first, _, vault, lease = ready(superseded)
    desired = consumer(k, first)
    command(k, 'create_generation', {'generation_id': 'active', 'candidate': False})
    command(k, 'set_requirement', {**desired, 'generation_id': 'active', 'excluded': True})
    api().collect_extract(k, vault, first['completion_id'], lease)
    pointer = k.get_serving_manifest()
    with pytest.raises(ClaimRejected, match='tombston'):
        command(k, 'pin_manifest', {'owner': 'late-reader', 'generation_id': 'active',
            'manifest_revision': pointer['manifest_revision'], 'expected_serving_revision': pointer['revision']})


def test_current_selection_protects_old_input_after_desired_input_advances(superseded):
    from test_knowledge_postgres import claim, seal
    from papervault.store.repository import payload_hash
    k, first, _, vault, lease = ready(superseded)
    command(k, 'register_unit', {'unit_id': 'active-consumer', 'kind': 'operator_document', 'identity': 'active-consumer'})
    output = first['output_inventory'][0]
    manifest = {'metadata_revision': 'm1', 'artifacts': [{'artifact_id': 'copied-input',
        'sha256': output['sha256'], 'byte_length': output['byte_length']}],
        'upstream_derivation_ids': ['extraction-derivation:' + first['completion_id']]}
    input_id = 'input:' + payload_hash(manifest)
    command(k, 'register_input', {'unit_id': 'active-consumer', 'consumed_input_revision': input_id, 'manifest': manifest})
    recipe = command(k, 'register_recipe', {'recipe': {'schema_version': 1, 'code_version': 'test',
        'output_contract': {'fulltext': {'required': ['graph'], 'not_applicable': []}}}})['recipe_digest']
    command(k, 'create_generation', {'generation_id': 'active', 'candidate': False})
    desired = {'unit_id': 'active-consumer', 'consumed_input_revision': input_id, 'recipe_digest': recipe,
        'generation_id': 'active', 'expected_revision': 0}
    command(k, 'set_requirement', desired)
    job = command(k, 'enqueue', desired)
    execution = claim(k, job)
    receipt = seal(k, execution)
    published = command(k, 'publish_derivation', {**execution, 'receipt': receipt,
        'expected_revision': execution['revision'], 'expected_selection_revision': 0, 'select': True})
    fresh_manifest = {'metadata_revision': 'm2', 'artifacts': [], 'upstream_derivation_ids': []}
    command(k, 'register_input', {'unit_id': 'active-consumer', 'consumed_input_revision': 'fresh-input', 'manifest': fresh_manifest})
    command(k, 'set_requirement', {**desired, 'consumed_input_revision': 'fresh-input', 'expected_revision': 1})
    assert k.get_selection('active-consumer', 'active')['derivation_id'] == published['derivation_id']
    current = k.get_serving_manifest()
    assert current['manifest']['selected_unit_receipts']['active-consumer'] == receipt['receipt_id']
    assert current['manifest']['entries'][0]['selected_derivation_id'] is None
    assert 'referenced' in row(api().plan_gc(k, vault), first)['reasons']


def test_old_manifest_reader_pin_remains_protected(superseded):
    k, first, _, vault, lease = ready(superseded)
    desired = consumer(k, first)
    command(k, 'create_generation', {'generation_id': 'active', 'candidate': False})
    command(k, 'set_requirement', {**desired, 'generation_id': 'active'})
    pointer = k.get_serving_manifest()
    command(k, 'pin_manifest', {'owner': 'early-reader', 'generation_id': 'active',
        'manifest_revision': pointer['manifest_revision'], 'expected_serving_revision': pointer['revision']})
    fresh = {'metadata_revision': 'fresh', 'artifacts': [], 'upstream_derivation_ids': []}
    command(k, 'register_input', {'unit_id': 'consumer', 'consumed_input_revision': 'fresh', 'manifest': fresh})
    command(k, 'set_requirement', {**desired, 'generation_id': 'active', 'consumed_input_revision': 'fresh', 'expected_revision': 1})
    assert 'pinned' in row(api().plan_gc(k, vault), first)['reasons']


def test_failed_job_recovery_pin_remains_protected(superseded):
    from test_knowledge_postgres import claim
    k, first, _, vault, lease = ready(superseded)
    desired = consumer(k, first)
    command(k, 'set_requirement', desired)
    job = command(k, 'enqueue', desired)
    execution = claim(k, job)
    command(k, 'fail', {**execution, 'expected_revision': execution['revision'], 'error': {'reason': 'unknown worker disposition'}})
    command(k, 'set_requirement', {**desired, 'expected_revision': 1, 'excluded': True})
    assert 'pinned' in row(api().plan_gc(k, vault), first)['reasons']


def test_recovery_round_trip_preserves_deletion_tombstone(superseded):
    from uuid import uuid4
    from psycopg import sql
    from papervault.store.recovery_postgres import catalog_barrier, export_catalog_state, restore_catalog_state
    k, first, _, vault, lease = ready(superseded)
    api().collect_extract(k, vault, first['completion_id'], lease)
    with catalog_barrier(k.repository, 'review-gc') as checkpoint:
        catalog = export_catalog_state(k.repository, checkpoint)
    schema = 'catalog_recovery_' + uuid4().hex
    try:
        restored = restore_catalog_state(k.connection, schema, catalog, recovery_id='review-gc')
        assert restored['dispatch_disabled'] is True
        deletion = k.connection.execute(sql.SQL('SELECT * FROM {}.extract_deletions').format(sql.Identifier(schema))).fetchone()
        assert deletion['state'] == 'deleted'
        assert deletion['completion_id'] == first['completion_id']
    finally:
        k.connection.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))


def test_section_input_cannot_readmit_tombstoned_parent(superseded):
    k, first, _, vault, lease = ready(superseded)
    desired = consumer(k, first)
    api().collect_extract(k, vault, first['completion_id'], lease)
    command(k, 'register_unit', {'unit_id': 'section', 'kind': 'section', 'identity': 'section'})
    command(k, 'register_section', {'unit_id': 'section', 'parent_unit_id': 'consumer',
        'mapping_revision': 1, 'parent_evidence_revision': 'copied-input', 'definition': {'section': 1}})
    section = {'metadata_revision': 'section1', 'artifacts': [], 'upstream_derivation_ids': [],
        'parent_consumed_input_revision': desired['consumed_input_revision'],
        'parent_evidence_revision': 'copied-input', 'section_definition_revision': 1}
    with pytest.raises(ClaimRejected, match='tombston'):
        command(k, 'register_input', {'unit_id': 'section', 'consumed_input_revision': 'section-input', 'manifest': section})


def test_pinned_old_inventory_parent_is_protected_after_inventory_advances(superseded):
    k, first, _, vault, lease = ready(superseded)
    desired = consumer(k, first)
    command(k, 'create_generation', {'generation_id': 'active', 'candidate': False})
    command(k, 'register_unit', {'unit_id': 'section', 'kind': 'section', 'identity': 'section'})
    command(k, 'register_section', {'unit_id': 'section', 'parent_unit_id': 'consumer',
        'mapping_revision': 1, 'parent_evidence_revision': 'copied-input', 'definition': {'section': 1}})
    old_child = {'metadata_revision': 'section1', 'artifacts': [], 'upstream_derivation_ids': [],
        'parent_consumed_input_revision': desired['consumed_input_revision'],
        'parent_evidence_revision': 'copied-input', 'section_definition_revision': 1}
    command(k, 'register_input', {'unit_id': 'section', 'consumed_input_revision': 'old-section-input', 'manifest': old_child})
    command(k, 'register_inventory', {'parent_unit_id': 'consumer', 'inventory_revision': 'old-inventory',
        'mode': 'sections', 'parent_evidence_revision': 'copied-input', 'required_units': ['section'],
        'section_mapping_revisions': {'section': 1}, 'parent_consumed_input_revision': desired['consumed_input_revision']})
    command(k, 'set_inventory', {'parent_unit_id': 'consumer', 'inventory_revision': 'old-inventory',
        'generation_id': 'active', 'expected_revision': 0})
    command(k, 'set_requirement', {'unit_id': 'section', 'generation_id': 'active',
        'consumed_input_revision': 'old-section-input', 'recipe_digest': desired['recipe_digest'],
        'inventory_revision': 'old-inventory', 'expected_revision': 0})
    pointer = k.get_serving_manifest()
    command(k, 'pin_manifest', {'owner': 'inventory-reader', 'generation_id': 'active',
        'manifest_revision': pointer['manifest_revision'], 'expected_serving_revision': pointer['revision']})
    command(k, 'register_input', {'unit_id': 'consumer', 'consumed_input_revision': 'new-parent-input',
        'manifest': {'metadata_revision': 'new-parent', 'artifacts': [], 'upstream_derivation_ids': []}})
    command(k, 'register_section', {'unit_id': 'section', 'parent_unit_id': 'consumer',
        'mapping_revision': 2, 'parent_evidence_revision': 'new-parent', 'definition': {'section': 1}})
    fresh_child = {**old_child, 'parent_consumed_input_revision': 'new-parent-input',
        'parent_evidence_revision': 'new-parent', 'section_definition_revision': 2}
    command(k, 'register_input', {'unit_id': 'section', 'consumed_input_revision': 'new-section-input', 'manifest': fresh_child})
    command(k, 'register_inventory', {'parent_unit_id': 'consumer', 'inventory_revision': 'new-inventory',
        'mode': 'sections', 'parent_evidence_revision': 'new-parent', 'required_units': ['section'],
        'section_mapping_revisions': {'section': 2}, 'parent_consumed_input_revision': 'new-parent-input'})
    command(k, 'set_inventory', {'parent_unit_id': 'consumer', 'inventory_revision': 'new-inventory',
        'generation_id': 'active', 'expected_revision': 1})
    command(k, 'set_requirement', {'unit_id': 'section', 'generation_id': 'active',
        'consumed_input_revision': 'new-section-input', 'recipe_digest': desired['recipe_digest'],
        'inventory_revision': 'new-inventory', 'expected_revision': 2})
    assert 'pinned' in row(api().plan_gc(k, vault), first)['reasons']


def test_gc_refuses_outer_transaction_that_prevents_durable_intent(superseded):
    k, first, _, vault, lease = ready(superseded)
    output = vault / first['output_inventory'][0]['path']
    with k.connection.transaction():
        with pytest.raises((ValueError, RuntimeError), match='transaction'):
            api().collect_extract(k, vault, first['completion_id'], lease)
    assert output.is_file()


def test_outer_rollback_cannot_leave_unlink_without_tombstone(superseded):
    k, first, _, vault, lease = ready(superseded)
    output = vault / first['output_inventory'][0]['path']
    with pytest.raises(RuntimeError, match='rollback caller'):
        with k.connection.transaction():
            with pytest.raises(ValueError, match='transaction'):
                api().collect_extract(k, vault, first['completion_id'], lease)
            raise RuntimeError('rollback caller')
    assert output.is_file()
    assert k._one('extract_deletions', completion_id=first['completion_id']) is None
