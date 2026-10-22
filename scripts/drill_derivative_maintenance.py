#!/usr/bin/env python3
"""Run P5c GC/rebuild checks only against a new explicitly disposable fixture.

This is a synthetic metadata maintenance rehearsal, not a host-failure recovery,
production RTO, or graph/vector/fulltext build benchmark. No dotenv is loaded.
Failed runs retain their schemas and evidence; choose a new drill ID to retry.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

os.environ["PYTHON_DOTENV_DISABLED"] = "1"


def validate_scope(args):
    """Validate explicit paths and DSN before any connection or mutation."""
    from psycopg.conninfo import conninfo_to_dict
    from papervault.store.recovery import checked_path

    if not args.disposable or not args.drill_id.strip():
        raise ValueError("an explicit drill ID and --disposable acknowledgement are required")
    root, evidence = checked_path(args.root), checked_path(args.evidence)
    if (any(".." in path.parts for path in (args.root, args.evidence))
            or any(not path.is_relative_to(Path("/tmp")) or path == Path("/tmp")
                   for path in (root, evidence))
            or root == evidence or root.is_relative_to(evidence) or evidence.is_relative_to(root)
            or root.exists() or evidence.exists()):
        raise ValueError("drill requires distinct new nonoverlapping destinations inside /tmp")
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        raise ValueError("the explicitly named DSN environment variable is absent")
    info = conninfo_to_dict(dsn)
    if (info.get("host") not in {"127.0.0.1", "::1"}
            or info.get("hostaddr", info["host"]) != info["host"]
            or not info.get("port", "").isdigit() or not 1024 <= int(info["port"]) <= 65535
            or int(info["port"]) == 5432 or info.get("dbname") != "p5c_drill"
            or info.get("user") != "p5c_drill" or info.get("service") or info.get("password")):
        raise ValueError("drill requires p5c_drill user/database on an explicit disposable loopback high port")
    token = hashlib.sha256(args.drill_id.encode()).hexdigest()[:20]
    schemas = {name: "catalog_recovery_" + token + "_" + name for name in ("extract", "rebuild")}
    if any(not re.fullmatch(r"catalog_recovery_[a-z0-9_]+", schema) for schema in schemas.values()):
        raise ValueError("invalid isolated schema")
    return root, evidence, dsn, schemas


def metadata_recipe():
    import papervault
    package = Path(papervault.__file__).parent
    sources = ("knowledge/scheduler/catalog.py", "knowledge/scheduler/catalog_inputs.py",
               "knowledge/scheduler/catalog_intake.py", "store/knowledge_commands.py")
    return {"schema_version": 1, "name": "p5c-isolated-metadata-fixture",
            "packages": {name: version(name) for name in ("papervault", "lightrag-hku")},
            "code_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in sources},
            "output_contract": {"metadata": {"required": ["metadata", "tombstones"],
                "not_applicable": ["graph", "vectors", "fulltext_evidence"]}}}


async def execute(args):
    import psycopg
    from psycopg import sql
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake
    from papervault.knowledge.store.build_breaker import BuildBreaker, BuildSchedule
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.store.derivative_maintenance import maintain
    from papervault.store.extraction import ExtractionStore, write_once
    from papervault.store.extraction_gc import plan_gc
    from papervault.store.knowledge import ClaimRejected, KnowledgeRepository
    from papervault.store.recovery import digest_file, file_inventory
    from papervault.store.repository import AuditEvent, PostgresCatalogRepository, Record, canonical_json
    from papervault.store.scheduler_capacity import SchedulerCapacity

    root, evidence, dsn, schemas = validate_scope(args)
    started = time.monotonic()
    log, actions, denials = [], [], []
    # Explicit hostaddr/password/passfile overrides prevent ambient libpq
    # defaults from redirecting this fixture or consulting a credential file.
    from psycopg.conninfo import conninfo_to_dict
    with psycopg.connect(dsn, autocommit=True, hostaddr=conninfo_to_dict(dsn)["host"],
                         password="", passfile=str(root / ".unused-passfile"), connect_timeout=5) as conn:
        # Check all destinations before creating any one of them.
        if conn.execute("SELECT current_database(),current_user").fetchone() != ("p5c_drill", "p5c_drill"):
            raise ValueError("connected database/user differs from explicit disposable fixture")
        for schema in schemas.values():
            if conn.execute("SELECT to_regnamespace(%s)", (schema,)).fetchone()[0]:
                raise ValueError("drill refuses an existing schema")
        root.mkdir(parents=True)
        evidence.mkdir(parents=True)
        # CREATE without IF NOT EXISTS closes the absence-check race: only
        # schemas created by this transaction may reach the fixture bootstrap.
        with conn.transaction():
            for schema in schemas.values():
                conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

        def save(name, value):
            write_once(evidence / name, (canonical_json(KnowledgeRepository._json(value)) + "\n").encode())

        def initialize(name):
            repo = PostgresCatalogRepository(conn, schema=schemas[name])
            audit = AuditEvent("drill", "isolated_fixture_initialize")
            repo.ensure_schema(audit=audit)
            repo.ensure_knowledge_schema(audit=audit)
            with repo._write(audit):
                conn.execute(sql.SQL("UPDATE {} SET authority='catalog'").format(repo._table("knowledge_runtime")))
                repo._finish(None, audit)
            return KnowledgeRepository(repo)

        def call(k, action, payload, vault, **kwargs):
            operation = args.drill_id + ":" + action + ":" + str(len(actions))
            result = maintain(k, action, payload, vault=vault, operation_id=operation,
                apply=True, owner="isolated-drill", window_id="window", lease_seconds=120, **kwargs)
            actions.append({"action": action, "operation_id": operation, "result": result})
            return result["result"]

        def refuse(label, invoke, reason):
            try:
                invoke()
            except ClaimRejected as exc:
                if reason not in str(exc):
                    raise
                denials.append({"check": label, "error_type": type(exc).__name__, "reason": str(exc)})
            else:
                raise AssertionError(label + " unexpectedly admitted")

        def open_window(k, vault):
            start = datetime.now(timezone.utc) - timedelta(seconds=1)
            return call(k, "window", {"window_id": "window", "kind": "monthly_reboot",
                "starts_at": start.isoformat(), "ends_at": (start + timedelta(seconds=120)).isoformat(),
                "expected_revision": 0, "operator_nonce": args.drill_id}, vault)

        try:
            save("environment.json", {"drill_id": args.drill_id, "scope": "explicit disposable synthetic fixtures",
                "root": str(root), "schemas": schemas, "command": sys.argv,
                "endpoint": {key: conn.info.get_parameters()[key]
                    for key in ("host", "port", "dbname", "user")}, "python": sys.version,
                "version": version("papervault"), "postgres_version": conn.execute("SHOW server_version").fetchone()[0],
                "script_sha256": digest_file(Path(__file__))["sha256"],
                "tool_sha256": {str(path): digest_file(path)["sha256"] for path in
                    sorted(Path("src/papervault/store").glob("*.py"))}, "dotenv_disabled": True,
                "fixture": "Synthetic metadata catalog; independent extraction catalog; no tracked paper sample",
                "claims_excluded": ["host failure recovery", "production RTO", "fulltext/graph/vector build"]})
            extracts = initialize("extract")
            vault, knowledge = root / "source/vault", root / "source/knowledge"
            vault.mkdir(parents=True)
            knowledge.mkdir(parents=True)
            source_bytes = b"%PDF-1.4\nP5c synthetic immutable source fixture\n"
            store = ExtractionStore(vault)
            completions = []

            def record(key):
                return Record.from_payload(key, {"key": key, "title": "P5c fixture " + key,
                    "terminal_state": {"retrieval": "eligible", "policy": "clear", "merge": "canonical"}})

            def publish(key, previous=None, head_revision=0):
                current = extracts.repository.get(key)
                identity = {"key": key, "title": current.payload["title"]}
                request = store.capture(record_key=key, metadata=identity, source=source_bytes, source_role="pdf",
                    recipe={"schema_version": 1, "engine": "isolated-fixture-only",
                        "output_contract": {"fulltext": {"required": ["markdown"], "not_applicable": []}}},
                    guard={"identity": identity, "previous_completion_id": previous},
                    unit_evidence={"kind": "paper", "record_key": key, "identity": identity},
                    source_catalog_revision=current.revision, authority_epoch=1)
                completion = store.finalize(request, ("# Synthetic " + key + " version " + str(head_revision)).encode())
                extracts.command("publish_derivation", "extract:" + completion["completion_id"],
                    {"kind": "extraction", "record_key": key, "completion": completion, "authority_epoch": 1,
                     "expected_record_revision": current.revision, "expected_head_revision": head_revision, "vault": str(vault)})
                completions.append(completion)
                return completion

            for key in ("disposable", "protected"):
                extracts.repository.upsert(record(key), expected_revision=0, audit=AuditEvent("drill", "fixture_record"))
            old = publish("disposable")
            current = publish("disposable", old["completion_id"], 1)
            protected = publish("protected")
            protected_current = publish("protected", protected["completion_id"], 1)
            paths = {"source": vault / old["input_manifest"]["artifacts"][0]["path"],
                     "disposable": vault / old["output_inventory"][0]["path"],
                     "current": vault / current["output_inventory"][0]["path"],
                     "protected": vault / protected["output_inventory"][0]["path"],
                     "protected_current": vault / protected_current["output_inventory"][0]["path"]}
            hashes_before = {name: digest_file(path) for name, path in paths.items()}
            retained = extracts._one("extract_retention", completion_id=old["completion_id"])
            call(extracts, "disposition", {"completion_id": old["completion_id"], "disposable": True,
                "expected_revision": retained["revision"], "reason": "isolated GC drill fixture",
                "operator_nonce": args.drill_id}, vault)
            gc_window_started = time.monotonic()
            gc_window = open_window(extracts, vault)
            gc_payload = {"completion_ids": [old["completion_id"]]}
            refuse("90-day retention", lambda: call(extracts, "gc", gc_payload, vault), "retention")
            assert digest_file(paths["disposable"]) == hashes_before["disposable"]
            # TEST FIXTURE ONLY: actual wall-clock time is not advanced. These
            # two superseded rows are aged inside this new disposable catalog.
            conn.execute(sql.SQL("UPDATE {} SET superseded_at=clock_timestamp()-interval '91 days' "
                "WHERE completion_id=ANY(%s)").format(extracts.repository._table("extract_retention")),
                ([old["completion_id"], protected["completion_id"]],))
            refuse("default protected extract", lambda: call(extracts, "gc",
                {"completion_ids": [protected["completion_id"]]}, vault), "protected")
            refuse("current extract", lambda: call(extracts, "gc",
                {"completion_ids": [current["completion_id"]]}, vault), "current")
            before_gc_plan = plan_gc(extracts, vault)
            unlink = os.unlink
            injected = []

            def interrupted_unlink(path, *args, **kwargs):
                if path == "output.md" and not injected:
                    injected.append(True)
                    raise OSError("isolated drill injected unlink failure")
                return unlink(path, *args, **kwargs)

            with patch("os.unlink", interrupted_unlink):
                try:
                    call(extracts, "gc", gc_payload, vault)
                except OSError as exc:
                    assert "isolated drill injected" in str(exc)
                    denials.append({"check": "recoverable unlink failure", "reason": str(exc), "error_type": "OSError"})
                else:
                    raise AssertionError("GC fault was not exercised")
            deleting = extracts._one("extract_deletions", completion_id=old["completion_id"])
            assert deleting["state"] == "deleting" and injected == [True]
            assert digest_file(paths["disposable"]) == hashes_before["disposable"]
            deleted = call(extracts, "gc", gc_payload, vault)[0]
            assert deleted["state"] == "deleted" and deleted["deletion_id"] == deleting["deletion_id"]
            assert not paths["disposable"].exists()
            hashes_after = {name: digest_file(path) for name, path in paths.items() if name != "disposable"}
            assert all(hashes_after[name] == hashes_before[name] for name in hashes_after)
            for completion in completions:
                assert extracts.get_extraction_receipt(completion["completion_id"])["receipt"] == completion
                assert store.path("extracts/versions/" + completion["completion_id"] + "/completion.json").is_file()
            gc_elapsed = time.monotonic() - gc_window_started
            assert gc_elapsed < 119
            save("gc.json", {"before_plan": before_gc_plan, "after_plan": plan_gc(extracts, vault),
                "fixture_aging": {"days": 91, "test_only_direct_update": True,
                    "completion_ids": [old["completion_id"], protected["completion_id"]]},
                "deleting": deleting, "deleted": deleted, "hashes_before": hashes_before, "hashes_after": hashes_after,
                "source_bytes_before": len(source_bytes), "source_bytes_after": paths["source"].stat().st_size,
                "retained_receipts": len(completions), "window": gc_window, "elapsed_seconds": gc_elapsed})
            log.append("GC: real 90-day, protected/current refusals; injected unlink failure persisted deleting intent; retry deleted only disposable output; source/receipts/protected/current hashes retained")

            k = initialize("rebuild")
            k.command("create_generation", "active", {"generation_id": "active", "candidate": False, "sample": False})
            inputs = ImmutableInputStore(knowledge / "catalog_inputs")
            recipe = metadata_recipe()
            intake = CatalogIntake(k, inputs, vault, recipe=recipe)
            keys = ["metadata-1", "metadata-2", "metadata-3"]
            for key in keys:
                k.repository.upsert(record(key), expected_revision=0, audit=AuditEvent("drill", "metadata_fixture"))
                intake.capture_record(k.repository.get(key))
            metadata_before = {key: k.repository.get(key).payload for key in keys}
            def backend_usage(physical):
                return int(conn.execute("SELECT COALESCE(sum(pg_total_relation_size(c.oid)),0) AS bytes "
                    "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                    "WHERE n.nspname=ANY(%s) AND c.relkind='r'", (list(schemas.values()),)).fetchone()["bytes"])

            def scheduler(cap):
                return CatalogScheduler(k, CatalogLightRAGAdapter(k, working_dir=str(knowledge), input_store=inputs),
                    inputs, max_admissions=1, reserved_cost=0, capacity_gate=cap,
                    breaker=BuildBreaker(threshold=0, schedule=BuildSchedule(enabled=False)))

            roots = (vault / "extracts", knowledge)
            capacity_root = root / "capacity"
            blocked_gate = SchedulerCapacity(capacity_root, derivative_roots=roots, estimated_bytes=4096,
                backend_usage=backend_usage, soft_cap=1)
            blocked_files_before = {"vault": file_inventory(vault), "knowledge": file_inventory(knowledge)}
            blocked = await scheduler(blocked_gate).run_round(only_keys=keys)
            assert blocked["capacity_blocked"] > 0 and blocked["claimed"] == blocked["published"] == 0
            assert all(row["attempts"] == 0 for row in k.list_admission_jobs(set(keys)))
            assert all(digest_file(path) == hashes_before[name] for name, path in paths.items() if name != "disposable")
            blocked_files_after = {"vault": file_inventory(vault), "knowledge": file_inventory(knowledge)}
            assert blocked_files_after == blocked_files_before
            denials.append({"check": "capacity exhaustion", "reason": blocked["capacity_reason"], "summary": blocked})
            gate = SchedulerCapacity(capacity_root, derivative_roots=roots, estimated_bytes=4096,
                backend_usage=backend_usage, soft_cap=64 * 1024 * 1024)
            worker = scheduler(gate)
            source_reports = []
            for _ in keys:
                report = await worker.run_round(only_keys=keys)
                assert report["published"] == 1 and not report["paused"] and not report["discrepancies"]
                source_reports.append(report)
            serving = k.get_serving_manifest()
            assert all(entry["eligible"] for entry in serving["entries"])
            rebuild_started = time.monotonic()
            rebuild_window = open_window(k, vault)
            generation, state = k.get_generation("active"), k.get_authority_state()
            fork_payload = {"generation_id": "candidate", "source_generation_id": "active",
                "expected_manifest_revision": generation["manifest_revision"],
                "expected_requirements_revision": state["requirements_revision"]}
            fork = call(k, "fork", fork_payload, vault)
            assert fork["queued_jobs"] == len(keys) and not fork["sample"]

            def promotion_payload():
                candidate, runtime = k.get_generation("candidate"), k.get_authority_state()
                return {"generation_id": "candidate", "expected_manifest_revision": candidate["manifest_revision"],
                    "expected_requirements_revision": runtime["requirements_revision"],
                    "expected_serving_revision": runtime["serving_revision"], "max_observation_age": 300}

            incomplete_payload = promotion_payload()
            refuse("incomplete candidate promotion", lambda: call(k, "promote", incomplete_payload, vault), "incomplete")
            candidate_reports = []
            for _ in keys:
                report = await worker.run_round(only_keys=keys)
                assert report["published"] == 1 and not report["paused"] and not report["discrepancies"]
                assert k.get_serving_manifest() == serving
                candidate_reports.append(report)
            exact_payload = promotion_payload()
            stale_payload = {**exact_payload, "expected_serving_revision": exact_payload["expected_serving_revision"] - 1}
            refuse("stale promotion revisions", lambda: call(k, "promote", stale_payload, vault), "revision changed")
            assert k.get_serving_manifest() == serving
            candidate_manifest = k.get_manifest("candidate")
            promoted = call(k, "promote", exact_payload, vault)
            assert promoted["generation_id"] == "candidate" and all(row["eligible"] for row in promoted["entries"])
            assert promoted["manifest"] == candidate_manifest["manifest"]
            assert {key: k.repository.get(key).payload for key in keys} == metadata_before
            rebuild_elapsed = time.monotonic() - rebuild_started
            assert rebuild_elapsed < 119
            assert not list(capacity_root.glob(".capacity-reservation-*.reserve"))
            save("rebuild.json", {"recipe": recipe, "parents": len(keys), "tier": "metadata", "sample": False,
                "fork_payload": fork_payload, "fork": fork, "source_reports": source_reports,
                "capacity_denial": blocked, "capacity_denial_files_before": blocked_files_before,
                "capacity_denial_files_after": blocked_files_after, "backend_measured_bytes": backend_usage(False),
                "candidate_reports": candidate_reports, "serving_before": serving, "candidate_manifest": candidate_manifest,
                "promotion_payload": exact_payload, "serving_after": promoted, "window": rebuild_window,
                "elapsed_seconds": rebuild_elapsed, "serving_changed_during_candidate_publications": False,
                "observed_serving_downtime_seconds": 0, "downtime_method": "eligible serving manifest read after every candidate publication and refusal",
                "source_metadata_unchanged": True})
            log.append("Rebuild: capacity exhaustion blocked claims; three metadata parents built through intake/scheduler; candidate publications kept serving; incomplete/stale promotion refused; exact complete manifest promoted")
            results = {"drill_id": args.drill_id, "success": True, "elapsed_seconds": time.monotonic() - started,
                "gc_window_elapsed_seconds": gc_elapsed, "rebuild_window_elapsed_seconds": rebuild_elapsed,
                "window_seconds": 120, "observed_serving_downtime_seconds": 0,
                "source_hashes_preserved": True, "metadata_parents": len(keys), "extraction_receipts_preserved": len(completions),
                "gpu_calls": 0, "external_model_calls": 0, "host_failure_recovery": False,
                "production_rto": False, "fulltext_build": False, "schemas": schemas}
            save("actions.json", {"actions": actions, "denials": denials})
            save("results.json", results)
            write_once(evidence / "verification.log", ("\n".join(log) + "\n").encode())
            save("hashes.json", {"algorithm": "sha256", "files": file_inventory(evidence)})
            print(canonical_json(results), flush=True)
        except BaseException as exc:
            save("failure.json", {"error_type": type(exc).__name__, "error": str(exc),
                "elapsed_seconds": time.monotonic() - started, "actions_completed": len(actions)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn-env", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--drill-id", required=True)
    parser.add_argument("--disposable", action="store_true", required=True)
    args = parser.parse_args(argv)
    asyncio.run(execute(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
