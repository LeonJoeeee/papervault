"""Verified, resumable restoration into explicit empty isolated targets."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from .extraction import fsync_directory, write_once
from .recovery import _atomic_json, checked_path, child, digest_file, verify_recovery_set
from .recovery_postgres import restore_catalog_state, validate_catalog_state, validate_catalog_target, verify_catalog_indexes
from .repository import AuditEvent, PostgresCatalogRepository, canonical_json, payload_hash


def copy_verified_file(source, destination, expected):
    source, destination = checked_path(source), checked_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest_file(destination) != expected:
            raise ValueError("completed restore file checksum differs")
        return
    partial = checked_path(destination.with_name("." + destination.name + ".restore-partial"))
    shutil.copyfile(source, partial, follow_symlinks=False)
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    if digest_file(partial) != expected:
        raise ValueError("restored copy checksum differs")
    os.replace(partial, destination)
    fsync_directory(destination.parent)


def _destination(relative):
    if relative.startswith("sources/") and relative != "sources/inventory.json":
        return "vault/" + relative.removeprefix("sources/")
    if relative.startswith("extracts/vault/"):
        return "vault/extracts/" + relative.removeprefix("extracts/vault/")
    if relative.startswith("inputs/") and relative != "inputs/inventory.json":
        return "knowledge/" + relative.removeprefix("inputs/")
    return "evidence/" + relative


def _catalog_component(connection, schema, catalog, operation_id):
    from psycopg.rows import dict_row

    connection.row_factory = dict_row
    exists = connection.execute("SELECT to_regnamespace(%s) AS name", (schema,)).fetchone()["name"]
    if exists:
        # Resolve a committed restore whose response/progress-file write was lost.
        row = connection.execute(sql.SQL("SELECT detail FROM {} WHERE event='recovery_restored' "
            "AND detail->>'recovery_id'=%s").format(sql.Identifier(schema, "audit")), (operation_id,)).fetchone()
        if not row or row["detail"]["checkpoint"] != catalog["checkpoint"]:
            raise ValueError("existing catalog is not this restore operation's isolated target")
        _verify_catalog_contents(connection, schema, catalog, operation_id)
        return {"schema": schema, "recovery_id": operation_id, "fence_id": "recovery:" + operation_id,
                "authority_epoch": row["detail"]["authority_epoch"], "dispatch_disabled": True}
    return restore_catalog_state(connection, schema, catalog, recovery_id=operation_id)


def _verify_catalog_contents(connection, schema, catalog, operation_id):
    from collections import Counter
    from psycopg.rows import dict_row
    connection.row_factory = dict_row
    if not connection.execute("SELECT to_regnamespace(%s) AS name", (schema,)).fetchone()["name"]:
        raise ValueError("restored catalog is missing")
    with connection.transaction():
        connection.execute("SET LOCAL TIME ZONE 'UTC'")
        verify_catalog_indexes(PostgresCatalogRepository(connection, schema=schema), catalog)
        tables = {row["tablename"] for row in connection.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname=%s", (schema,))}
        if tables != set(catalog["tables"]):
            raise ValueError("restored catalog table inventory changed")
        candidate = connection.execute(sql.SQL("SELECT detail FROM {} WHERE event='recovery_candidate_created' "
            "AND detail->>'operation_id'=%s").format(sql.Identifier(schema, "audit")), (operation_id,)).fetchone()
        for name, rows in catalog["rows_json"].items():
            if name in {"knowledge_runtime", "writer_fence"}:
                continue
            actual = [row["encoded"] for row in connection.execute(sql.SQL(
                "SELECT to_jsonb(t)::text AS encoded FROM {} t").format(sql.Identifier(schema, name)))]
            # Before candidate creation, only the fence/epoch and audit may
            # differ. Afterward, its atomic digest binds every added data row.
            if (Counter(rows) - Counter(actual)
                    or (not candidate and name != "audit" and Counter(rows) != Counter(actual))):
                raise ValueError("restored catalog history changed: " + name)
        row = connection.execute(sql.SQL("SELECT fence_id FROM {}").format(sql.Identifier(schema, "writer_fence"))).fetchone()
        if not row or row["fence_id"] != "recovery:" + operation_id:
            raise ValueError("restored catalog writer fence changed")
        matches = connection.execute(sql.SQL("SELECT (to_jsonb(t)-'authority_epoch' = %s::jsonb-'authority_epoch') "
            "AND authority_epoch=%s AS matches FROM {} t").format(sql.Identifier(schema, "knowledge_runtime")),
            (catalog["rows_json"]["knowledge_runtime"][0], catalog["checkpoint"]["authority_epoch"] + 1)).fetchone()
        if not matches or not matches["matches"]:
            raise ValueError("restored catalog runtime/epoch changed")
        if candidate and _catalog_rows_digest(connection, schema) != candidate["detail"]["catalog_contents_sha256"]:
            raise ValueError("restored catalog candidate/content rows changed")


def _catalog_rows_digest(connection, schema):
    """Bind all target rows after candidate creation, including added rows.

    Audit is append-only operation evidence: the original rows are checked above,
    and materialization appends its receipt hash after this digest is committed.
    """
    with connection.transaction():
        connection.execute("SET LOCAL TIME ZONE 'UTC'")
        tables = [row["tablename"] for row in connection.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname=%s AND tablename<>'audit' ORDER BY tablename", (schema,))]
        return payload_hash({name: sorted(row["encoded"] for row in connection.execute(sql.SQL(
            "SELECT to_jsonb(t)::text AS encoded FROM {} t").format(sql.Identifier(schema, name)))) for name in tables})


def _restore_candidate(restored, state, sample):
    from .knowledge import KnowledgeRepository
    op = state["operation_id"]
    with restored._write(AuditEvent("recovery", "recovery_candidate_created")):
        row = restored.connection.execute(sql.SQL("SELECT detail FROM {} WHERE event='recovery_candidate_created' "
            "AND detail->>'operation_id'=%s").format(restored._table("audit")), (op,)).fetchone()
        if row:
            digest = row["detail"]["catalog_contents_sha256"]
            if _catalog_rows_digest(restored.connection, restored.schema) != digest:
                raise ValueError("restored catalog candidate/content rows changed")
        else:
            KnowledgeRepository(restored).command("create_generation", op + ":generation", {
                "generation_id": state["binding"]["target_generation"], "candidate": True, "sample": sample})
            digest = _catalog_rows_digest(restored.connection, restored.schema)
            # Atomic with candidate creation: losing the progress-file write
            # cannot adopt a subsequently modified candidate on resume.
            restored._finish(None, AuditEvent("recovery", "recovery_candidate_created", {
                "operation_id": op, "catalog_contents_sha256": digest}))
    return digest


def restore_recovery_set(package, target, *, target_generation, catalog_connection=None,
                         catalog_schema=None, postgres_target=None, neo4j_target=None, apply=False):
    package, target = checked_path(package), checked_path(target)
    manifest = verify_recovery_set(package)
    if target.is_relative_to(package) or package.is_relative_to(target):
        raise ValueError("restore target must be isolated from the recovery package")
    for root in manifest.get("source_roots", []):
        root = checked_path(root)
        if target.is_relative_to(root) or root.is_relative_to(target):
            raise ValueError("restore target must be isolated from source roots")
    if not isinstance(target_generation, str) or not re.fullmatch(r"[A-Za-z0-9_:-]{1,128}", target_generation):
        raise ValueError("explicit new target generation required")
    catalog = json.loads((package / "catalog/data.json").read_bytes())
    validate_catalog_state(catalog)
    from .recovery_source import verify_file_dependencies
    verify_file_dependencies(package, catalog)
    if any(row["generation_id"] == target_generation for row in catalog["tables"]["generations"]):
        raise ValueError("restoration requires a new isolated generation identity")
    if not apply:
        return {"apply": False, "mode": "restore", "source_snapshot_id": manifest["snapshot_id"],
                "target": str(target), "target_generation": target_generation, "sample": manifest["sample"],
                "required_stores": manifest["required_stores"]}
    if catalog_connection is None or not catalog_schema:
        raise ValueError("applied restore requires an explicit isolated catalog target")
    validate_catalog_target(catalog_connection, catalog_schema)
    for name, destination in (("postgres", postgres_target), ("neo4j", neo4j_target)):
        component = manifest["components"][name]
        if component["format"] != "metadata-sample-not-applicable-v1" and destination is None:
            raise ValueError("restore requires an empty isolated " + name + " backend target")
        if component["format"] == "metadata-sample-not-applicable-v1" and not manifest["sample"]:
            raise ValueError("complete recovery set cannot omit backend stores")
    target_identity = {key: value for key, value in conninfo_to_dict(catalog_connection.info.dsn).items()
                       if key in {"host", "port", "dbname", "user"}}
    binding = {"manifest_sha256": digest_file(package / "manifest.json")["sha256"],
               "target_generation": target_generation, "catalog_schema": catalog_schema,
               "catalog_target": target_identity}
    if postgres_target:
        binding["postgres_target"] = {"schema": postgres_target["schema"], **{
            key: value for key, value in conninfo_to_dict(postgres_target["connection"].info.dsn).items()
            if key in {"host", "port", "dbname", "user"}}}
    if neo4j_target:
        from .recovery_backends import safe_uri
        binding["neo4j_target"] = {"uri": safe_uri(neo4j_target["uri"]), "database": neo4j_target["database"]}
    state_path = child(target, ".restore.json")
    if target.exists() and not state_path.exists() and any(target.iterdir()):
        raise ValueError("restore requires an empty isolated target")
    if not state_path.exists():
        present = catalog_connection.execute("SELECT to_regnamespace(%s)", (catalog_schema,)).fetchone()
        present = next(iter(present.values())) if isinstance(present, dict) else present[0]
        if present:
            raise ValueError("restore requires an absent empty catalog schema")
    target.mkdir(parents=True, exist_ok=True)
    with child(target, ".restore.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            state = json.loads(state_path.read_bytes())
            if state["binding"] != binding:
                raise ValueError("restore target belongs to another operation or checkpoint")
        else:
            state = {"binding": binding, "operation_id": "restore-" + uuid4().hex, "attempts": 0,
                     "completed_files": {}, "components": {}, "failures": [], "phase": "copying"}
            _atomic_json(state_path, state)
        for relative, expected in state["completed_files"].items():
            if digest_file(child(target, relative)) != expected:
                raise ValueError("completed restore file checksum differs")
        for name, destination in (("postgres", postgres_target), ("neo4j", neo4j_target)):
            if name not in state["components"] or state["components"][name].get("not_applicable"):
                continue
            from .recovery_backends import restore_neo4j_store, restore_postgres_store
            data = json.loads(child(package, name + "/data.json").read_bytes())
            if name == "postgres":
                restore_postgres_store(destination["connection"], destination["schema"], data, operation_id=state["operation_id"])
            else:
                restore_neo4j_store(destination["driver"], data, operation_id=state["operation_id"],
                    **{key: destination[key] for key in ("database", "uri", "disposable")})
        if "catalog" in state["components"]:
            _verify_catalog_contents(catalog_connection, catalog_schema, catalog, state["operation_id"])
        receipt_path = child(target, "restore-receipt.json")
        if receipt_path.exists() and state["phase"] != "complete":
            receipt = json.loads(receipt_path.read_bytes())
            if (receipt["operation_id"] != state["operation_id"]
                    or receipt["source_manifest_sha256"] != binding["manifest_sha256"]
                    or receipt["verified_files"] != state["completed_files"]):
                raise ValueError("sealed restore receipt differs from operation")
            _record_materialization(catalog_connection, catalog_schema, state, receipt_path)
            state.update(phase="complete", receipt_sha256=digest_file(receipt_path)["sha256"])
            _atomic_json(state_path, state)
        if state["phase"] == "complete":
            receipt_path = child(target, "restore-receipt.json")
            if digest_file(receipt_path)["sha256"] != state["receipt_sha256"]:
                raise ValueError("restore receipt checksum differs")
            return json.loads(receipt_path.read_bytes())
        state["attempts"] += 1
        _atomic_json(state_path, state)
        try:
            for item in manifest["files"]:
                relative = _destination(item["path"])
                expected = {key: item[key] for key in ("sha256", "byte_length")}
                if relative in state["completed_files"]:
                    continue
                copy_verified_file(child(package, item["path"]), child(target, relative), expected)
                state["completed_files"][relative] = expected
                _atomic_json(state_path, state)
            for name, destination in (("postgres", postgres_target), ("neo4j", neo4j_target)):
                if name in state["components"]:
                    continue
                if manifest["components"][name]["format"] == "metadata-sample-not-applicable-v1":
                    state["components"][name] = {"not_applicable": "metadata_sample"}
                else:
                    from .recovery_backends import restore_neo4j_store, restore_postgres_store
                    data = json.loads((package / name / "data.json").read_bytes())
                    if name == "postgres":
                        state["components"][name] = restore_postgres_store(destination["connection"], destination["schema"], data,
                            operation_id=state["operation_id"])
                    else:
                        state["components"][name] = restore_neo4j_store(destination["driver"], data,
                            **{key: destination[key] for key in ("database", "uri", "disposable")},
                            operation_id=state["operation_id"])
                _atomic_json(state_path, state)
            state["components"]["catalog"] = _catalog_component(catalog_connection, catalog_schema, catalog, state["operation_id"])
            _atomic_json(state_path, state)
            restored = PostgresCatalogRepository(catalog_connection, schema=catalog_schema)
            restored.fence_id = state["components"]["catalog"]["fence_id"]
            catalog_digest = _restore_candidate(restored, state, manifest["sample"])
            receipt = {"schema_version": 1, "kind": "restore_materialization",
                "receipt_id": "restore-receipt:" + state["operation_id"], "operation_id": state["operation_id"],
                "attempt_id": state["operation_id"] + ":" + str(state["attempts"]),
                "fencing_token": state["attempts"], "recovery_epoch": state["operation_id"],
                "acceptance_scope": "isolated_recovery", "target_generation": target_generation,
                "source_snapshot_id": manifest["snapshot_id"], "source_manifest_sha256": binding["manifest_sha256"],
                "sample": manifest["sample"], "target_root": str(target),
                "catalog_contents_sha256": catalog_digest,
                "original_receipt_ids": [row["receipt_id"] for row in catalog["tables"]["knowledge_receipts"]],
                "original_derivation_ids": [row.get("derivation_id", row["id"]) for row in catalog["tables"]["derivations"]],
                "source_content_recipes": [row["recipe_digest"] for row in catalog["tables"]["recipes"]],
                "operation_recipe": {"schema_version": 1, "name": "papervault-restore", "papervault": version("papervault"),
                    "implementation_sha256": digest_file(Path(__file__))["sha256"]},
                "verified_files": state["completed_files"], "materializations": state["components"],
                "failures": state["failures"], "dispatch_disabled": True}
            encoded = canonical_json(receipt).encode()
            write_once(target / "restore-receipt.json", encoded)
            _record_materialization(catalog_connection, catalog_schema, state, target / "restore-receipt.json")
            state.update(phase="complete", receipt_sha256=hashlib.sha256(encoded).hexdigest())
            _atomic_json(state_path, state)
            return receipt
        except BaseException as exc:
            state["failures"].append({"attempt": state["attempts"], "error_type": type(exc).__name__})
            state["phase"] = "interrupted"
            _atomic_json(state_path, state)
            raise


def _record_materialization(connection, schema, state, receipt_path):
    restored = PostgresCatalogRepository(connection, schema=schema)
    restored.fence_id = state["components"]["catalog"]["fence_id"]
    digest = digest_file(receipt_path)["sha256"]
    audit = AuditEvent("recovery", "recovery_materialized", {
        "operation_id": state["operation_id"], "receipt_sha256": digest})
    with restored._write(audit):
        row = connection.execute(sql.SQL("SELECT detail FROM {} WHERE event='recovery_materialized' "
            "AND detail->>'operation_id'=%s").format(restored._table("audit")), (state["operation_id"],)).fetchone()
        if row and row["detail"] != audit.detail:
            raise ValueError("restore receipt differs from catalog materialization")
        if not row:
            restored._finish(None, audit)
