"""Empty isolated restores, durable resume and isolated acceptance receipts."""
import importlib.util
import json
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest

from papervault.store.repository import AuditEvent, Record


@pytest.fixture
def package(repo, tmp_path):
    from papervault.store.knowledge import KnowledgeRepository
    from papervault.store.recovery import snapshot
    from papervault.store.recovery_source import CoordinatedSource
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake

    repo.ensure_knowledge_schema(audit=AuditEvent("test", "recovery_schema"))
    repo.connection.execute(sql.SQL("UPDATE {} SET authority='catalog'").format(repo._table("knowledge_runtime")))
    vault, knowledge = tmp_path / "vault", tmp_path / "knowledge"
    vault.mkdir()
    knowledge.mkdir()
    k = KnowledgeRepository(repo)
    k.command("create_generation", "source", {"generation_id": "source", "candidate": False})
    inputs = ImmutableInputStore(knowledge / "catalog_inputs")
    intake = CatalogIntake(k, inputs, vault, recipe={"schema_version": 1,
        "name": "metadata-test", "output_contract": {"metadata": {"required": ["metadata", "tombstones"]}}})
    for key in ("A", "B"):
        repo.upsert(Record.from_payload(key, {"key": key, "title": key, "terminal_state": {
            "retrieval": "eligible", "policy": "clear", "merge": "canonical"}}),
            expected_revision=0, audit=AuditEvent("test", "seed"))
        intake.capture_record(repo.get(key))
    source = CoordinatedSource(repo, vault=vault, knowledge_root=knowledge, metadata_sample=True)
    manifest = snapshot(source, tmp_path / "packages", apply=True, sample=True)
    return tmp_path / "packages" / manifest["snapshot_id"], k


@pytest.fixture
def target_catalog(catalog_test_dsn):
    connection = psycopg.connect(catalog_test_dsn, autocommit=True)
    schema = "catalog_recovery_" + uuid4().hex
    try:
        yield connection, schema
    finally:
        connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        connection.close()


def restore_module():
    assert importlib.util.find_spec("papervault.store.recovery_restore") is not None
    from papervault.store import recovery_restore
    return recovery_restore


def test_restore_dry_run_has_no_target_or_catalog_mutations(package, target_catalog, tmp_path):
    module = restore_module()
    path, source = package
    connection, schema = target_catalog
    target = tmp_path / "target"
    before = source.get_authority_state()
    result = module.restore_recovery_set(path, target, target_generation="restored",
        catalog_connection=connection, catalog_schema=schema)
    assert result["apply"] is False
    assert not target.exists()
    assert connection.execute("SELECT to_regnamespace(%s)", (schema,)).fetchone()[0] is None
    assert source.get_authority_state() == before


def test_restore_creates_new_receipt_and_never_moves_source_selection(package, target_catalog, tmp_path):
    module = restore_module()
    path, source = package
    connection, schema = target_catalog
    before = source.get_authority_state()
    receipt = module.restore_recovery_set(path, tmp_path / "target", target_generation="restored",
        catalog_connection=connection, catalog_schema=schema, apply=True)
    assert receipt["kind"] == "restore_materialization"
    assert receipt["acceptance_scope"] == "isolated_recovery"
    assert receipt["sample"] is True
    assert receipt["target_generation"] == "restored"
    assert receipt["source_snapshot_id"] == path.name
    assert receipt["operation_recipe"]["name"] == "papervault-restore"
    assert receipt["source_content_recipes"]
    assert source.get_authority_state() == before
    retained = json.loads((tmp_path / "target/restore-receipt.json").read_bytes())
    assert retained == receipt
    assert list((tmp_path / "target/knowledge/catalog_inputs").iterdir())
    from papervault.store.knowledge import validate_receipt
    with pytest.raises(ValueError, match="live acceptance"):
        validate_receipt(receipt, {}, {})


def test_restore_resumes_owned_partial_copy_and_refuses_changed_completed_bytes(package, target_catalog, tmp_path, monkeypatch):
    module = restore_module()
    path, source = package
    connection, schema = target_catalog
    target = tmp_path / "target"
    original = module.copy_verified_file
    copied = []

    def interrupt(source, destination, expected):
        if len(copied) == 2:
            raise OSError("injected copy interruption")
        original(source, destination, expected)
        copied.append(destination)

    monkeypatch.setattr(module, "copy_verified_file", interrupt)
    with pytest.raises(OSError, match="injected"):
        module.restore_recovery_set(path, target, target_generation="restored", catalog_connection=connection,
                                    catalog_schema=schema, apply=True)
    state = json.loads((target / ".restore.json").read_bytes())
    assert state["failures"][-1]["error_type"] == "OSError"
    copied[0].write_bytes(b"changed after successful checkpoint")
    monkeypatch.setattr(module, "copy_verified_file", original)
    with pytest.raises(ValueError, match="completed|checksum"):
        module.restore_recovery_set(path, target, target_generation="restored", catalog_connection=connection,
                                    catalog_schema=schema, apply=True)


def test_restore_resumes_interruption_without_overwriting_completed_files(package, target_catalog, tmp_path, monkeypatch):
    module = restore_module()
    path, source = package
    connection, schema = target_catalog
    target = tmp_path / "target"
    original = module.copy_verified_file
    copied = []

    def interrupt(source, destination, expected):
        if len(copied) == 1:
            raise OSError("interrupt")
        original(source, destination, expected)
        copied.append((destination, destination.stat().st_mtime_ns))

    monkeypatch.setattr(module, "copy_verified_file", interrupt)
    with pytest.raises(OSError):
        module.restore_recovery_set(path, target, target_generation="restored", catalog_connection=connection,
                                    catalog_schema=schema, apply=True)
    monkeypatch.setattr(module, "copy_verified_file", original)
    result = module.restore_recovery_set(path, target, target_generation="restored", catalog_connection=connection,
                                        catalog_schema=schema, apply=True)
    assert result["fencing_token"] == 2
    assert copied[0][0].stat().st_mtime_ns == copied[0][1]
    assert result["failures"] == [{"attempt": 1, "error_type": "OSError"}]


def test_populated_target_is_rejected_before_any_database_write(package, target_catalog, tmp_path):
    module = restore_module()
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "target"
    target.mkdir()
    (target / "unowned").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="empty"):
        module.restore_recovery_set(path, target, target_generation="restored", catalog_connection=connection,
                                    catalog_schema=schema, apply=True)
    assert (target / "unowned").read_bytes() == b"preserve"


def test_partial_copy_cannot_follow_a_symlink(tmp_path):
    module = restore_module()
    from papervault.store.recovery import digest_file
    source, target, victim = (tmp_path / name for name in ("source", "target", "victim"))
    source.write_bytes(b"source bytes")
    victim.write_bytes(b"preserve")
    (tmp_path / ".target.restore-partial").symlink_to(victim)
    with pytest.raises(ValueError, match="symlink"):
        module.copy_verified_file(source, target, digest_file(source))
    assert victim.read_bytes() == b"preserve"


def test_restore_recovers_sealed_receipt_after_lost_progress_write(package, target_catalog, tmp_path, monkeypatch):
    module = restore_module()
    path, _ = package
    connection, schema = target_catalog
    original = module._atomic_json
    failed = False

    def interrupt(path, state):
        nonlocal failed
        if state.get("phase") == "complete" and not failed:
            failed = True
            raise OSError("lost progress write after receipt")
        return original(path, state)

    target = tmp_path / "target"
    monkeypatch.setattr(module, "_atomic_json", interrupt)
    with pytest.raises(OSError, match="lost progress"):
        module.restore_recovery_set(path, target, target_generation="restored", catalog_connection=connection,
                                    catalog_schema=schema, apply=True)
    sealed = (target / "restore-receipt.json").read_bytes()
    monkeypatch.setattr(module, "_atomic_json", original)
    receipt = module.restore_recovery_set(path, target, target_generation="restored", catalog_connection=connection,
                                          catalog_schema=schema, apply=True)
    assert receipt == json.loads(sealed)
    assert (target / "restore-receipt.json").read_bytes() == sealed


@pytest.mark.parametrize("damage", ["missing", "record", "fence", "epoch"])
def test_completed_restore_revalidates_catalog(package, target_catalog, tmp_path, damage):
    module = restore_module()
    path, _ = package
    connection, schema = target_catalog
    target = tmp_path / "target"
    module.restore_recovery_set(path, target, target_generation="restored", catalog_connection=connection,
                               catalog_schema=schema, apply=True)
    if damage == "missing":
        connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
    elif damage == "record":
        connection.execute(sql.SQL("UPDATE {} SET payload=payload || '{{\"title\":\"damaged\"}}'::jsonb").format(
            sql.Identifier(schema, "records")))
    elif damage == "fence":
        connection.execute(sql.SQL("UPDATE {} SET fence_id='unfenced'").format(sql.Identifier(schema, "writer_fence")))
    else:
        connection.execute(sql.SQL("UPDATE {} SET authority_epoch=authority_epoch+1").format(
            sql.Identifier(schema, "knowledge_runtime")))
    with pytest.raises(ValueError, match="catalog"):
        module.restore_recovery_set(path, target, target_generation="restored", catalog_connection=connection,
                                   catalog_schema=schema, apply=True)
