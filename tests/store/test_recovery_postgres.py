"""Recovery checkpoints, history preservation and fences on dedicated PostgreSQL."""
import importlib.util
import json
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest

from papervault.store.knowledge import KnowledgeRepository
from papervault.store.repository import AuditEvent, PostgresCatalogRepository, Record
from test_extraction_postgres import extraction_catalog as extraction_catalog


def recovery_pg():
    assert importlib.util.find_spec("papervault.store.recovery_postgres") is not None
    from papervault.store import recovery_postgres
    return recovery_postgres


def seed(repo):
    repo.ensure_knowledge_schema(audit=AuditEvent("test", "recovery_schema"))
    repo.upsert(Record.from_payload("A", {"key": "A", "title": "retained metadata"}),
                expected_revision=0, audit=AuditEvent("test", "seed_recovery"))


def test_catalog_export_contains_exact_history_and_checkpoint(repo):
    module = recovery_pg()
    seed(repo)
    before = repo.connection.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(repo._table("audit"))).fetchone()["n"]
    with module.catalog_barrier(repo, "backup-test") as checkpoint:
        exported = module.export_catalog_state(repo, checkpoint)
        assert exported["checkpoint"] == checkpoint
        assert checkpoint["backup_id"] == "backup-test"
        assert checkpoint["catalog_snapshot"]
        assert checkpoint["wal_lsn"]
        assert len(exported["tables"]["audit"]) == before
        assert exported["tables"]["records"][0]["payload"]["title"] == "retained metadata"
        assert {"jobs", "rulings", "knowledge_outbox", "knowledge_receipts", "extraction_receipts",
                "recipes", "generation_manifests", "knowledge_runtime"}.issubset(exported["tables"])
    assert repo.get("A").revision == 1
    assert repo.connection.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(repo._table("audit"))).fetchone()["n"] == before


def test_snapshot_barrier_refuses_active_external_writer(repo, catalog_test_dsn):
    module = recovery_pg()
    seed(repo)
    with psycopg.connect(catalog_test_dsn, autocommit=True) as connection:
        key = "papervault:" + repo.schema + ":recovery"
        connection.execute("SELECT pg_advisory_lock_shared(hashtextextended(%s,0))", (key,))
        try:
            with pytest.raises(RuntimeError, match="writer"):
                with module.catalog_barrier(repo, "cannot-capture"):
                    pytest.fail("barrier admitted while backend IO was active")
        finally:
            connection.execute("SELECT pg_advisory_unlock_shared(hashtextextended(%s,0))", (key,))
    with module.catalog_barrier(repo, "after-drain") as checkpoint:
        assert checkpoint["backup_id"] == "after-drain"


async def test_snapshot_barrier_rejects_backend_io_before_namespace_writes(repo, catalog_test_dsn):
    module = recovery_pg()
    seed(repo)
    # Even a malformed claim must be refused by the global barrier before any
    # namespace work or backend operation is possible.
    with psycopg.connect(catalog_test_dsn, autocommit=True) as connection:
        other = KnowledgeRepository(PostgresCatalogRepository(connection, schema=repo.schema))
        with module.catalog_barrier(repo, "fenced"):
            with pytest.raises(RuntimeError, match="recovery"):
                async with other.namespace_access({"namespace": "private"}):
                    pytest.fail("backend mutation entered while recovery was frozen")


def test_restore_requires_absent_schema_and_preserves_history_with_dispatch_fenced(repo, catalog_test_dsn):
    module = recovery_pg()
    seed(repo)
    with module.catalog_barrier(repo, "restore-test") as checkpoint:
        exported = module.export_catalog_state(repo, checkpoint)
    target = "catalog_recovery_" + uuid4().hex
    connection = psycopg.connect(catalog_test_dsn, autocommit=True)
    try:
        result = module.restore_catalog_state(connection, target, exported, recovery_id="restore-op")
        recovered = PostgresCatalogRepository(connection, schema=target)
        assert recovered.get("A").payload == repo.get("A").payload
        history = connection.execute(sql.SQL("SELECT * FROM {} ORDER BY id").format(recovered._table("audit"))).fetchall()
        assert [row["event"] for row in history[:-1]] == [row["event"] for row in exported["tables"]["audit"]]
        assert history[-1]["event"] == "recovery_restored"
        assert result["recovery_id"] == "restore-op"
        assert KnowledgeRepository(recovered).get_authority_state()["authority_epoch"] > checkpoint["authority_epoch"]
        from papervault.store.fencing import WriterFenced
        with pytest.raises(WriterFenced):
            recovered.upsert(Record.from_payload("B", {"key": "B", "title": "unadmitted"}),
                             expected_revision=0, audit=AuditEvent("test", "unsafe"))
        with pytest.raises(ValueError, match="empty|absent"):
            module.restore_catalog_state(connection, target, exported, recovery_id="another")
        assert recovered.get("B") is None
    finally:
        connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(target)))
        connection.close()


def test_bad_schema_identity_refuses_restore_before_target_creation(repo, catalog_test_dsn):
    module = recovery_pg()
    seed(repo)
    with module.catalog_barrier(repo, "schema-test") as checkpoint:
        exported = module.export_catalog_state(repo, checkpoint)
    exported = json.loads(json.dumps(exported))
    exported["schema_files"]["schema.sql"]["sha256"] = "0" * 64
    target = "catalog_recovery_" + uuid4().hex
    with psycopg.connect(catalog_test_dsn, autocommit=True) as connection:
        with pytest.raises(ValueError, match="schema"):
            module.restore_catalog_state(connection, target, exported, recovery_id="bad")
        assert connection.execute("SELECT to_regnamespace(%s)", (target,)).fetchone()[0] is None


@pytest.mark.parametrize("field", ["authority_epoch", "audit_head"])
def test_checkpoint_must_match_the_captured_catalog_state(repo, field):
    module = recovery_pg()
    seed(repo)
    with module.catalog_barrier(repo, "checkpoint-consistency") as checkpoint:
        exported = module.export_catalog_state(repo, checkpoint)
    exported["checkpoint"][field] += 1
    with pytest.raises(ValueError, match="checkpoint"):
        module.validate_catalog_state(exported)


def recovery_source():
    assert importlib.util.find_spec("papervault.store.recovery_source") is not None
    from papervault.store.recovery_source import CoordinatedSource
    return CoordinatedSource


def test_metadata_sample_captures_bytes_recipes_and_full_catalog_under_one_barrier(repo, tmp_path):
    source_type = recovery_source()
    from papervault.store.recovery import snapshot, verify_recovery_set

    seed(repo)
    vault, knowledge = tmp_path / "vault", tmp_path / "knowledge"
    vault.mkdir()
    knowledge.mkdir()
    (vault / "pdfs").mkdir()
    (vault / "pdfs/A.pdf").write_bytes(b"retained source bytes")
    (vault / "extracts").mkdir()
    (vault / "extracts/A.txt").write_text("retained legacy text")
    source = source_type(repo, vault=vault, knowledge_root=knowledge, metadata_sample=True)
    package = snapshot(source, tmp_path / "recovery", apply=True, sample=True)
    root = tmp_path / "recovery" / package["snapshot_id"]
    assert verify_recovery_set(root) == package
    assert (root / "sources/pdfs/A.pdf").read_bytes() == b"retained source bytes"
    assert (root / "extracts/vault/A.txt").read_text() == "retained legacy text"
    assert package["parent_inventory"][0]["record_key"] == "A"
    assert package["components"]["neo4j"]["format"] == "metadata-sample-not-applicable-v1"
    event = repo.connection.execute(sql.SQL("SELECT event,detail FROM {} ORDER BY id DESC LIMIT 1").format(
        repo._table("audit"))).fetchone()
    assert event["event"] == "recovery_snapshot_registered"
    assert event["detail"]["checkpoint"] == package["checkpoint"]
    captured = json.loads((root / "catalog/data.json").read_bytes())
    assert captured["tables"]["audit"][-1]["event"] != "recovery_snapshot_registered"


def test_complete_snapshot_cannot_omit_configured_derivative_backends(repo, tmp_path):
    source_type = recovery_source()
    from papervault.store.recovery import snapshot

    seed(repo)
    vault, knowledge = tmp_path / "vault", tmp_path / "knowledge"
    vault.mkdir()
    knowledge.mkdir()
    source = source_type(repo, vault=vault, knowledge_root=knowledge)
    with pytest.raises(ValueError, match="backend"):
        snapshot(source, tmp_path / "recovery", apply=True)
    assert not (tmp_path / "recovery").exists()


def test_active_library_or_graph_prevents_snapshot(repo, tmp_path):
    source_type = recovery_source()
    from papervault.store.fencing import WriterFenced, writer_lease
    from papervault.store.recovery import snapshot

    seed(repo)
    vault, knowledge = tmp_path / "vault", tmp_path / "knowledge"
    vault.mkdir()
    knowledge.mkdir()
    source = source_type(repo, vault=vault, knowledge_root=knowledge, metadata_sample=True)
    for root in (vault, knowledge):
        lease = writer_lease(root, None)
        try:
            with pytest.raises(WriterFenced):
                snapshot(source, tmp_path / "recovery", apply=True, sample=True)
        finally:
            lease.close()


@pytest.mark.parametrize("fault", ["absent", "corrupt"])
def test_snapshot_requires_every_captured_input_byte(repo, tmp_path, fault):
    source_type = recovery_source()
    from papervault.store.recovery import snapshot
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake

    seed(repo)
    vault, knowledge = tmp_path / "vault", tmp_path / "knowledge"
    vault.mkdir()
    knowledge.mkdir()
    k = KnowledgeRepository(repo)
    k.command("create_generation", "active", {"generation_id": "active", "candidate": False})
    store = ImmutableInputStore(knowledge / "catalog_inputs")
    intake = CatalogIntake(k, store, vault, recipe={"schema_version": 1,
        "output_contract": {"metadata": {"required": ["metadata", "tombstones"]}}})
    intake.capture_record(repo.get("A"))
    blob = next(store.root.iterdir())
    if fault == "absent":
        blob.unlink()
    else:
        blob.write_bytes(b"corrupt captured input")
    source = source_type(repo, vault=vault, knowledge_root=knowledge, metadata_sample=True)
    with pytest.raises(ValueError, match="input|artifact"):
        snapshot(source, tmp_path / "recovery", apply=True, sample=True)
    assert not (tmp_path / "recovery/retention.json").exists()


def test_sample_generation_marker_survives_fork_and_blocks_promotion(repo):
    seed(repo)
    from papervault.store.knowledge import ClaimRejected
    k = KnowledgeRepository(repo)
    generation = k.command("create_generation", "sample", {
        "generation_id": "sample", "candidate": False, "sample": True})
    assert generation.get("sample") is True
    state = k.get_authority_state()
    fork = k.command("fork_generation", "fork-sample", {"generation_id": "sample-copy",
        "source_generation_id": "sample", "expected_manifest_revision": generation["manifest_revision"],
        "expected_requirements_revision": state["requirements_revision"]})
    assert fork["sample"] is True
    with pytest.raises(ClaimRejected, match="sample"):
        k.command("promote_generation", "cannot-promote", {"generation_id": "sample-copy"})
    with pytest.raises(psycopg.errors.RaiseException, match="sample"):
        repo.connection.execute(sql.SQL("UPDATE {} SET sample=FALSE WHERE generation_id='sample-copy'").format(
            repo._table("generations")))


def test_mutable_projection_is_backed_up_without_permanently_pinning_each_week(repo, tmp_path):
    seed(repo)
    from papervault.store.recovery import snapshot
    from papervault.store.recovery_source import CoordinatedSource
    vault, knowledge = tmp_path / "vault", tmp_path / "knowledge"
    vault.mkdir()
    knowledge.mkdir()
    (vault / "index.json").write_text('{"current":"projection"}')
    (vault / "pdfs").mkdir()
    (vault / "pdfs/source.pdf").write_bytes(b"retained research source")
    (knowledge / "mutable-cache.json").write_text('{"status":"mutable"}')
    (vault / "reopen-budget.jsonl").write_text('{"event":"fixture"}\n')
    (vault / "correction-ledger.json").write_text('{"revision":1}')
    source = CoordinatedSource(repo, vault=vault, knowledge_root=knowledge, metadata_sample=True)
    manifest = snapshot(source, tmp_path / "sets", apply=True, sample=True)
    assert "sources/index.json" in {row["path"] for row in manifest["files"]}
    assert "sources/index.json" not in manifest["protected_paths"]
    assert "inputs/mutable-cache.json" not in manifest["protected_paths"]
    assert "sources/reopen-budget.jsonl" not in manifest["protected_paths"]
    assert "sources/correction-ledger.json" not in manifest["protected_paths"]
    assert "sources/pdfs/source.pdf" in manifest["protected_paths"]


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("fault", ["absent", "corrupt", "receipt"])
def test_snapshot_verifies_unregistered_extraction_closure(repo, tmp_path, staged, fault):
    from test_extraction_postgres import frozen
    from papervault.store.recovery import snapshot
    from papervault.store.recovery_source import CoordinatedSource
    seed(repo)
    vault, knowledge = tmp_path / "vault", tmp_path / "knowledge"
    vault.mkdir()
    knowledge.mkdir()
    pending = vault / ".reopen-staging/owned" if staged else vault
    completion = frozen(KnowledgeRepository(repo), pending, captured_epoch=None)
    artifact = pending / completion["input_manifest"]["artifacts"][0]["path"]
    if fault == "absent":
        artifact.unlink()
    elif fault == "corrupt":
        artifact.write_bytes(b"corrupted original input")
    else:
        receipt = pending / "extracts/versions" / completion["completion_id"] / "completion.json"
        changed = json.loads(receipt.read_bytes())
        changed["record_key"] = "different"
        receipt.write_text(json.dumps(changed))
    source = CoordinatedSource(repo, vault=vault, knowledge_root=knowledge, metadata_sample=True)
    with pytest.raises(ValueError, match="input|artifact|completion|integrity"):
        snapshot(source, tmp_path / "sets", apply=True, sample=True)


def test_registered_extraction_inputs_use_their_retained_vault_location(extraction_catalog, tmp_path):
    import shutil
    from test_extraction_postgres import frozen, publish
    from papervault.store.recovery_postgres import catalog_barrier, export_catalog_state
    from papervault.store.recovery_source import verify_file_dependencies
    vault, package = tmp_path / "vault", tmp_path / "package"
    completion = frozen(extraction_catalog, vault)
    publish(extraction_catalog, completion, vault)
    with catalog_barrier(extraction_catalog.repository, "test-extraction") as checkpoint:
        catalog = export_catalog_state(extraction_catalog.repository, checkpoint)
    shutil.copytree(vault / "extracts", package / "extracts/vault")
    (package / "sources").mkdir()
    verify_file_dependencies(package, catalog)


def test_imported_identity_indexes_survive_coordinated_restore(repo, catalog_test_dsn, tmp_path):
    from papervault.store.adapter import LibraryAdapter
    from papervault.store.importer import import_catalog
    from papervault.store.recovery import snapshot, verify_recovery_set
    from papervault.store.recovery_restore import restore_recovery_set
    from papervault.store.recovery_source import CoordinatedSource

    vault, knowledge = tmp_path / "vault", tmp_path / "knowledge"
    vault.mkdir()
    knowledge.mkdir()
    source = vault / "index.json"
    source.write_text(json.dumps({"papers": {"A": {"key": "A", "title": "Imported identity",
        "doi": "10.1234/recovery", "arxiv_id": "2501.12345"}}}))
    report = import_catalog(LibraryAdapter.from_snapshot(source), repo)
    assert report["target_identity_indexes"]["doi"]["state"] == "created"
    assert report["target_identity_indexes"]["arxiv_id"]["state"] == "created"
    repo.ensure_knowledge_schema(audit=AuditEvent("test", "recovery_schema"))
    # A non-bundled expression/partial index, including a schema-looking literal,
    # catches index allowlists and unsafe global schema-name replacements.
    literal = " ON " + sql.Identifier(repo.schema, "records").as_string() + " "
    repo.connection.execute(sql.SQL("CREATE INDEX {} ON {} (lower(title)) WHERE source <> {}").format(
        sql.Identifier('extra "title" index'), repo._table("records"), sql.Literal(literal)))
    manifest = snapshot(CoordinatedSource(repo, vault=vault, knowledge_root=knowledge, metadata_sample=True),
                        tmp_path / "sets", sample=True, apply=True)
    package = tmp_path / "sets" / manifest["snapshot_id"]
    assert verify_recovery_set(package) == manifest
    target = "catalog_recovery_" + uuid4().hex
    with psycopg.connect(catalog_test_dsn, autocommit=True) as connection:
        try:
            restore_recovery_set(package, tmp_path / "restored", target_generation="restored",
                catalog_connection=connection, catalog_schema=target, apply=True)
            indexes = {row["indexname"] for row in connection.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname=%s", (target,))}
            assert {"records_doi_unique", "records_arxiv_id_unique", 'extra "title" index'} <= indexes
            for field, identity in (("doi", "10.1234/recovery"), ("arxiv_id", "2501.12345")):
                with pytest.raises(psycopg.errors.UniqueViolation):
                    connection.execute(sql.SQL("INSERT INTO {} (key,download_status,payload_version,payload,{}) "
                        "VALUES ('duplicate','pending',1,'{{\"key\":\"duplicate\"}}',%s)").format(
                            sql.Identifier(target, "records"), sql.Identifier(field)), (identity,))
            captured = json.loads((package / "catalog/data.json").read_bytes())
            assert set(captured["indexes"]) == indexes
            definition = connection.execute("SELECT indexdef FROM pg_indexes WHERE schemaname=%s AND indexname=%s",
                (target, 'extra "title" index')).fetchone()["indexdef"]
            assert literal in definition
        finally:
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(target)))


@pytest.mark.parametrize("damage", ["missing", "omitted", "changed"])
def test_catalog_index_inventory_is_bound_to_checkpoint(repo, damage):
    module = recovery_pg()
    seed(repo)
    with module.catalog_barrier(repo, "indexes") as checkpoint:
        exported = module.export_catalog_state(repo, checkpoint)
    # Default {} lets the old exporter reach the fail-closed assertion too.
    indexes = exported.setdefault("indexes", {})
    if damage == "missing":
        exported.pop("indexes")
    elif damage == "omitted":
        indexes.pop("records_status_idx", None)
    else:
        indexes["records_status_idx"] = {"table": "records", "definition": "corrupt"}
    with pytest.raises(ValueError, match="index"):
        module.validate_catalog_state(exported)


@pytest.mark.parametrize("changed", [False, True])
def test_restore_matches_installed_nonconstraint_index_changes(repo, catalog_test_dsn, changed):
    module = recovery_pg()
    seed(repo)
    repo.connection.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(repo.schema, "records_status_idx")))
    if changed:
        repo.connection.execute(sql.SQL("CREATE INDEX records_status_idx ON {} (title)").format(repo._table("records")))
    with module.catalog_barrier(repo, "changed-index") as checkpoint:
        exported = module.export_catalog_state(repo, checkpoint)
    target = "catalog_recovery_" + uuid4().hex
    with psycopg.connect(catalog_test_dsn, autocommit=True) as connection:
        try:
            module.restore_catalog_state(connection, target, exported, recovery_id="changed-index")
            index = connection.execute("SELECT indexdef FROM pg_indexes WHERE schemaname=%s AND indexname='records_status_idx'",
                                       (target,)).fetchone()
            if changed:
                assert index["indexdef"] == f"CREATE INDEX records_status_idx ON {target}.records USING btree (title)"
            else:
                assert index is None
        finally:
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(target)))


@pytest.mark.parametrize("changed", [False, True])
def test_restore_reconciles_stronger_default_index_before_loading(repo, catalog_test_dsn, changed):
    module = recovery_pg()
    seed(repo)
    repo.connection.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(repo.schema, "generations_one_active")))
    if changed:
        repo.connection.execute(sql.SQL("CREATE UNIQUE INDEX generations_one_active ON {} (generation_id)").format(
            repo._table("generations")))
    # These rows are legal under the captured installed indexes, but collide
    # under the bundled default. Restore must not impose a different index set.
    repo.connection.execute(sql.SQL("INSERT INTO {} (generation_id,state,requirements_revision) "
        "VALUES ('one','active',0),('two','active',0)").format(repo._table("generations")))
    with module.catalog_barrier(repo, "changed-unique-index") as checkpoint:
        exported = module.export_catalog_state(repo, checkpoint)
    target = "catalog_recovery_" + uuid4().hex
    with psycopg.connect(catalog_test_dsn, autocommit=True) as connection:
        try:
            module.restore_catalog_state(connection, target, exported, recovery_id="changed-unique-index")
            rows = connection.execute(sql.SQL("SELECT generation_id FROM {} WHERE state='active' ORDER BY generation_id").format(
                sql.Identifier(target, "generations"))).fetchall()
            assert [row["generation_id"] for row in rows] == ["one", "two"]
        finally:
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(target)))
