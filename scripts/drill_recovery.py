#!/usr/bin/env python3
"""Execute a reproducible metadata census drill; optional real-store fixture roundtrip.

The tracked landmark pools contain 100 distinct real papers, metadata only. This
instrument discloses absent corpus strata and never counts synthetic backend rows
as parent documents or claims full-system recovery time from the sample.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time

os.environ["PYTHON_DOTENV_DISABLED"] = "1"


class FileUsageSampler:
    """Observe staged bytes, including temporary capacity reservations."""
    def __init__(self, root):
        self.root = root
        self.stop = threading.Event()
        self.peak_logical = self.peak_allocated = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def measure(self):
        logical = allocated = 0
        for path in self.root.rglob("*"):
            try:
                if path.is_file():
                    stat = path.stat()
                    logical += stat.st_size
                    allocated += stat.st_blocks * 512
            except FileNotFoundError:
                pass  # An atomic rename/removal between samples is expected.
        self.peak_logical = max(self.peak_logical, logical)
        self.peak_allocated = max(self.peak_allocated, allocated)

    def run(self):
        while not self.stop.is_set():
            self.measure()
            self.stop.wait(0.1)

    def close(self):
        self.stop.set()
        self.thread.join()
        self.measure()


def sample_inventory(path):
    parents = {}
    for line in path.read_text().splitlines():
        pool = json.loads(line)
        for paper in pool["pool"]:
            identity = " ".join(paper["title"].casefold().split())
            parent = parents.setdefault(identity, {"paper": paper, "pools": []})
            parent["pools"].append(pool["qid"])
    return [{**row, "key": "landmark-" + hashlib.sha256(identity.encode()).hexdigest()[:24]}
            for identity, row in sorted(parents.items())]


def metadata_recipe():
    import papervault
    root = Path(papervault.__file__).parent
    names = ("knowledge/scheduler/catalog.py", "knowledge/scheduler/catalog_inputs.py",
             "knowledge/store/catalog_adapter.py", "store/knowledge_commands.py")
    return {"schema_version": 1, "name": "papervault-metadata-drill",
        "packages": {name: version(name) for name in ("papervault", "lightrag-hku")},
        "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
        "output_contract": {"metadata": {"required": ["metadata", "tombstones"],
            "not_applicable": ["graph", "vectors", "fulltext_evidence"]}}}


async def execute(args, parents):
    import psycopg
    from psycopg import sql
    from papervault.store.extraction import write_once
    from papervault.store.knowledge import KnowledgeRepository
    from papervault.store.repository import AuditEvent, PostgresCatalogRepository, Record, canonical_json
    from papervault.store.recovery import checked_path, digest_file, file_inventory, snapshot, verify_recovery_set
    from papervault.store.recovery_postgres import validate_catalog_target
    from papervault.store.recovery_source import CoordinatedSource
    from papervault.store.recovery_restore import restore_recovery_set
    from papervault.store.recovery_replay import replay_recovery_set
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore
    from papervault.knowledge.scheduler.catalog_intake import CatalogIntake
    from papervault.knowledge.scheduler.catalog import CatalogScheduler
    from papervault.knowledge.store.catalog_adapter import CatalogLightRAGAdapter
    from papervault.knowledge.store.build_breaker import BuildBreaker, BuildSchedule

    root, evidence = checked_path(args.root), checked_path(args.evidence)
    if root.exists() or evidence.exists():
        raise ValueError("drill requires new empty staging/evidence destinations; failed runs remain evidence")
    token = hashlib.sha256(args.drill_id.encode()).hexdigest()[:20]
    schemas = {name: "catalog_recovery_" + token + "_" + name for name in ("source", "restore", "replay")}
    backend_schemas = {name: "recovery_backend_" + token + "_" + name for name in ("source", "restore")}
    started, cpu = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    log, timings, failures = [], {}, []
    dsn = os.environ[args.dsn_env]
    with ExitStack() as stack:
        source_conn = stack.enter_context(psycopg.connect(dsn, autocommit=True))
        target_conn = stack.enter_context(psycopg.connect(dsn, autocommit=True))
        replay_conn = stack.enter_context(psycopg.connect(dsn, autocommit=True))
        for schema in [*schemas.values(), *backend_schemas.values()]:
            validate_catalog_target(source_conn, schemas["source"])
            if source_conn.execute("SELECT to_regnamespace(%s)", (schema,)).fetchone()[0]:
                raise ValueError("drill schema already exists: " + schema)
        pg_store = neo_store = pg_target = neo_target = None
        if args.neo4j_source or args.neo4j_target:
            if not args.neo4j_source or not args.neo4j_target or args.neo4j_source == args.neo4j_target:
                raise ValueError("provide two distinct explicitly disposable Neo4j endpoints")
            from neo4j import GraphDatabase
            from papervault.store.recovery_backends import PostgresStore, Neo4jStore
            from prepare_knowledge_backend import preparation_plan
            first = stack.enter_context(GraphDatabase.driver(args.neo4j_source, auth=None))
            second = stack.enter_context(GraphDatabase.driver(args.neo4j_target, auth=None))
            if not all(uri.startswith("bolt://127.0.0.1:") and not uri.endswith(":7687")
                       for uri in (args.neo4j_source, args.neo4j_target)):
                raise ValueError("drill physical fixtures require explicitly disposable loopback endpoints")
            for driver in (first, second):
                with driver.session(database="neo4j") as session:
                    if session.run("MATCH(n) RETURN count(n) AS n").single()["n"]:
                        raise ValueError("drill graph target must be empty")
            pg_conn = stack.enter_context(psycopg.connect(dsn, autocommit=True))
            pg_restore = stack.enter_context(psycopg.connect(dsn, autocommit=True))
            plan = preparation_plan(embedding_dim=3)
            with pg_conn.transaction():
                pg_conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(backend_schemas["source"])))
                pg_conn.execute(sql.SQL("SET LOCAL search_path TO {},public").format(sql.Identifier(backend_schemas["source"])))
                for statement in plan["postgres"]:
                    if not statement.startswith("CREATE EXTENSION"):
                        pg_conn.execute(statement)
                pg_conn.execute("INSERT INTO lightrag_vdb_entity (workspace,id,entity_name,content_vector,chunk_ids) "
                                "VALUES ('drill_fixture','synthetic-row','fixture-only','[0.1,0.2,0.3]',ARRAY['fixture'])")
            roles = dict(zip(("full_docs", "text_chunks", "full_entities", "full_relations", "entity_chunks",
                "relation_chunks", "llm_response_cache", "doc_status", "chunks_vdb", "entities_vdb", "relationships_vdb"),
                plan["tables"], strict=True))
            pg_store = PostgresStore(pg_conn, schema=backend_schemas["source"], roles=roles)
            with first.session(database="neo4j") as session:
                session.run("CREATE (a:PaperVaultCatalogEntity {entity_id:'fixture-a',_pv_namespace:$ns})-"
                    "[:DIRECTED {_pv_namespace:$ns,weight:0.5}]->(b:PaperVaultCatalogEntity "
                    "{entity_id:'fixture-b',_pv_namespace:$ns})", ns=args.drill_id).consume()
            neo_store = Neo4jStore(first, database="neo4j", uri=args.neo4j_source)
            pg_target = {"connection": pg_restore, "schema": backend_schemas["restore"]}
            neo_target = {"driver": second, "database": "neo4j", "uri": args.neo4j_target, "disposable": True}
        root.mkdir(parents=True)
        evidence.mkdir(parents=True)
        sampler = FileUsageSampler(root)
        stack.callback(sampler.close)
        def save(name, value):
            write_once(evidence / name, (canonical_json(value) + "\n").encode())
        save("environment.json", {"drill_id": args.drill_id, "staging_root": str(root), "production_root": "/data",
            "production_window_owner": "orchestrator; monthly reboot window <=7200 seconds",
            "schemas": schemas, "backend_schemas": backend_schemas if pg_store else {},
            "physical_fixture_endpoints": [args.neo4j_source, args.neo4j_target] if pg_store else [],
            "python": sys.version, "platform": platform.platform(), "command": sys.argv,
            "tool_sha256": {str(path): digest_file(path)["sha256"] for path in sorted(Path("src/papervault/store").glob("recovery*.py"))},
            "drill_script_sha256": digest_file(Path(__file__))["sha256"]})
        try:
            source_repo = PostgresCatalogRepository(source_conn, schema=schemas["source"])
            source_repo.ensure_schema(audit=AuditEvent("drill", "initialize"))
            source_repo.ensure_knowledge_schema(audit=AuditEvent("drill", "initialize"))
            with source_repo._write(AuditEvent("drill", "isolated_authority_bootstrap")):
                source_conn.execute(sql.SQL("UPDATE {} SET authority='catalog'").format(source_repo._table("knowledge_runtime")))
                source_repo._finish(None, AuditEvent("drill", "isolated_authority_bootstrap"))
            k = KnowledgeRepository(source_repo)
            k.command("create_generation", args.drill_id, {"generation_id": "source", "candidate": False})
            vault, knowledge = root / "source/vault", root / "source/knowledge"
            vault.mkdir(parents=True)
            inputs = ImmutableInputStore(knowledge / "catalog_inputs")
            recipe = metadata_recipe()
            intake = CatalogIntake(k, inputs, vault, recipe=recipe)
            for parent in parents:
                paper = parent["paper"]
                payload = {"key": parent["key"], "title": paper["title"], "doi": paper.get("doi") or "",
                    "year": paper.get("year"), "authors": [paper["first_author"]], "source": "tracked-landmark-pool",
                    "terminal_state": {"retrieval": "eligible", "policy": "clear", "merge": "canonical"}}
                source_repo.upsert(Record.from_payload(parent["key"], payload), expected_revision=0,
                    audit=AuditEvent("drill", "metadata_source"))
                intake.capture_record(source_repo.get(parent["key"]))
            write_once(vault / "metadata-source.json", canonical_json(parents).encode())
            adapter = CatalogLightRAGAdapter(k, working_dir=str(knowledge), input_store=inputs)
            scheduler = CatalogScheduler(k, adapter, inputs, max_admissions=1, reserved_cost=0,
                breaker=BuildBreaker(threshold=0, schedule=BuildSchedule(enabled=False)))
            phase = time.monotonic()
            for index in range(len(parents)):
                report = await scheduler.run_round(only_keys=[row["key"] for row in parents])
                if report["discrepancies"] or report["paused"] or report["published"] != 1:
                    raise RuntimeError("source metadata pipeline did not publish exactly one admitted parent")
                if (index + 1) % 20 == 0:
                    print(f"source published {index+1}/{len(parents)}", flush=True)
            timings["source_admission_seconds"] = time.monotonic() - phase
            before = k.get_authority_state()
            source = CoordinatedSource(source_repo, vault=vault, knowledge_root=knowledge,
                postgres=pg_store, neo4j=neo_store, metadata_sample=pg_store is None)
            phase = time.monotonic()
            manifest = snapshot(source, root / "sets", apply=True, sample=True)
            timings["snapshot_seconds"] = time.monotonic() - phase
            package = root / "sets" / manifest["snapshot_id"]
            verify_recovery_set(package)
            save("snapshot-manifest.json", manifest)
            pg_export = json.loads((package / "postgres/data.json").read_bytes())
            graph_export = json.loads((package / "neo4j/data.json").read_bytes())
            save("backup-inclusion.json", {"snapshot_id": manifest["snapshot_id"],
                "checkpoint": manifest["checkpoint"], "components": manifest["components"],
                "catalog_export_hash": digest_file(package / "catalog/data.json"),
                "postgres_export_hash": digest_file(package / "postgres/data.json"),
                "neo4j_export_hash": digest_file(package / "neo4j/data.json"),
                "postgres_roles": pg_export.get("roles", {}),
                "postgres_table_rows": {name: len(table["rows_json"]) for name, table in pg_export.get("tables", {}).items()},
                "neo4j_nodes": len(graph_export.get("nodes", [])),
                "neo4j_relationships": len(graph_export.get("relationships", [])),
                "operational_backup_activated": False, "operational_root": "/data/p5b-recovery"})
            save("recipes.json", {"content_recipe": recipe, "source_recipe_digest": k._rows("recipes")[0]["recipe_digest"]})
            log.append("snapshot: exact catalog checkpoint and all seven component inventories/hash closures verified")
            print("snapshot verified", flush=True)
            phase = time.monotonic()
            restored = restore_recovery_set(package, root / "restore", target_generation="restored",
                catalog_connection=target_conn, catalog_schema=schemas["restore"], postgres_target=pg_target,
                neo4j_target=neo_target, apply=True)
            timings["restore_seconds"] = time.monotonic() - phase
            save("restore-receipt.json", restored)
            log.append(f"restore: {len(manifest['parent_inventory'])} parents; verified bytes, backend readback, original history, new fenced receipt")
            print("restoration verified", flush=True)
            phase = time.monotonic()
            replayed = await replay_recovery_set(package, root / "replay", source_generation="source", target_generation="replayed",
                catalog_connection=replay_conn, catalog_schema=schemas["replay"], apply=True)
            timings["replay_seconds"] = time.monotonic() - phase
            save("replay-receipt.json", replayed)
            assert len(replayed["receipts"]) == len(parents) >= 100
            assert k.get_authority_state() == before
            assert replayed["serving_generation_id"] is None
            assert set(replayed["original_receipt_ids"]).isdisjoint(row["receipt_id"] for row in replayed["receipts"])
            log.append("replay: 100 original metadata inputs/recipes, 100 new normal claims/sealed receipts, no source serving change")
            log.append("sample generation cannot promote; production-window drill remains with orchestrator")
            save("sample.json", {"selection": "census of every distinct title in the tracked 12 landmark pools; no sampling RNG",
                "source": str(args.pools), "source_hash": digest_file(args.pools), "parents": parents,
                "strata": {"kind": {"paper": len(parents)}, "tier": {"metadata": len(parents)},
                    "pool_memberships": dict(Counter(pool for row in parents for pool in row["pools"])),
                    "doi": dict(Counter("present" if row["paper"].get("doi") else "absent" for row in parents)),
                    "year_decade": dict(Counter(str((row["paper"].get("year", 0) // 10) * 10) for row in parents))},
                "missing_strata": ["abstract and full-text source bytes", "operator documents/notebooks/textbooks",
                    "section-built parents", "registered extraction engines/OCR", "real graph/vector derivations"],
                "physical_fixture_scope": "one synthetic 3D pgvector row and two graph nodes/one edge; never counted as parents" if pg_store else "not executed"})
            usage = resource.getrusage(resource.RUSAGE_SELF)
            sampler.close()
            relation_bytes = source_conn.execute("SELECT COALESCE(sum(pg_total_relation_size(c.oid)),0) AS bytes "
                "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=ANY(%s) AND c.relkind='r'",
                ([*schemas.values(), *backend_schemas.values()],)).fetchone()["bytes"]
            results = {"drill_id": args.drill_id, "success": True, "parents": len(parents), "timings": timings,
                "elapsed_seconds": time.monotonic() - started, "peak_worker_rss_kib": usage.ru_maxrss,
                "worker_cpu_seconds": usage.ru_utime + usage.ru_stime - cpu.ru_utime - cpu.ru_stime,
                "sampled_peak_staging_logical_bytes": sampler.peak_logical,
                "sampled_peak_staging_allocated_bytes": sampler.peak_allocated,
                "file_usage_sampling_interval_seconds": 0.1,
                "staging_file_bytes_at_completion": sum(row["byte_length"] for row in file_inventory(root)),
                "owned_database_bytes_at_completion": int(relation_bytes), "database_server_peak_resources": "not measured",
                "gpu_calls": 0, "external_model_calls": 0, "external_model_cost_usd": 0,
                "host_energy_cost": "not measured", "failures": failures, "source_serving_unchanged": True,
                "sample_timing_is_full_rto": False, "host_failure_recovery": False,
                "production_root": "/data", "staging_root": str(root)}
            save("results.json", results)
            write_once(evidence / "verification.log", ("\n".join(log) + "\n").encode())
            save("hashes.json", {"algorithm": "sha256", "files": file_inventory(evidence)})
            print(canonical_json(results), flush=True)
        except BaseException as exc:
            save("failure.json", {"error_type": type(exc).__name__, "elapsed_seconds": time.monotonic() - started,
                "completed_timings": timings, "staging_root": str(root)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--drill-id", required=True)
    parser.add_argument("--dsn-env", default="PAPERVAULT_CATALOG_TEST_DSN")
    parser.add_argument("--pools", type=Path, default=Path("src/papervault/eval/gold_search_pools.jsonl"))
    parser.add_argument("--neo4j-source", help="optional disposable loopback fixture source (no auth)")
    parser.add_argument("--neo4j-target", help="optional distinct empty disposable loopback fixture target (no auth)")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    parents = sample_inventory(args.pools)
    if len(parents) < 100:
        parser.error("drill requires at least 100 distinct real parent documents")
    if not args.apply:
        print(json.dumps({"apply": False, "parents": len(parents), "root": str(args.root), "evidence": str(args.evidence)}))
        return 0
    asyncio.run(execute(args, parents))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
