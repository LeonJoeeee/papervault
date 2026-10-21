#!/usr/bin/env python3
"""Coordinated weekly recovery sets. Explicit roots/connections; dry-run by default."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

# Neither application imports nor LightRAG may discover a service dotenv.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"


def connections(configuration, stack):
    import psycopg
    from psycopg.conninfo import conninfo_to_dict
    result = {}
    for name in ("catalog", "postgres"):
        if name not in configuration:
            continue
        item = configuration[name]
        dsn = os.environ[item["dsn_env"]]
        if not conninfo_to_dict(dsn).get("dbname"):
            raise ValueError("explicit database name required")
        result[name] = {**item, "connection": stack.enter_context(psycopg.connect(dsn, autocommit=True, connect_timeout=5))}
    if "neo4j" in configuration:
        from neo4j import GraphDatabase
        from papervault.store.recovery_backends import safe_uri
        item = configuration["neo4j"]
        password = os.environ[item["password_env"]] if item.get("password_env") else None
        if password is None and not item.get("disposable"):
            raise ValueError("Neo4j credentials required outside explicit disposable targets")
        auth = (item["user"], password) if password is not None else None
        driver = stack.enter_context(GraphDatabase.driver(safe_uri(item["uri"]), auth=auth))
        result["neo4j"] = {**item, "driver": driver}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="explicit source roots, schema identities and credential environment names")
    parser.add_argument("--root", type=Path, help="explicit recovery-set root")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--due", action="store_true", help="skip if a verified complete set is less than seven days old")
    parser.add_argument("--verify", type=Path, help="verify one package, without any connection or write")
    args = parser.parse_args(argv)
    from papervault.store.recovery import checked_path, snapshot, verify_recovery_set
    if args.verify:
        manifest = verify_recovery_set(args.verify)
        print(json.dumps({"verified": True, "snapshot_id": manifest["snapshot_id"], "sample": manifest["sample"]}))
        return 0
    if not args.config or not args.root:
        parser.error("--config and --root are required")
    root = checked_path(args.root)
    if args.due and (root / "retention.json").exists():
        retained = json.loads((root / "retention.json").read_bytes())["retained"]
        complete = [verify_recovery_set(root / identity) for identity in retained]
        if complete and datetime.fromisoformat(complete[-1]["created_at"]) > datetime.now(timezone.utc) - timedelta(days=7):
            print(json.dumps({"due": False, "latest": complete[-1]["snapshot_id"]}))
            return 0
    config = json.loads(args.config.read_bytes())
    from papervault.store.recovery_backends import Neo4jStore, PostgresStore
    from papervault.store.recovery_source import CoordinatedSource
    from papervault.store.repository import PostgresCatalogRepository
    with ExitStack() as stack:
        targets = connections(config, stack)
        catalog = targets["catalog"]
        pg, graph = targets.get("postgres"), targets.get("neo4j")
        source = CoordinatedSource(PostgresCatalogRepository(catalog["connection"], schema=catalog["schema"]),
            vault=config["vault"], knowledge_root=config["knowledge_root"], metadata_sample=config.get("metadata_sample", False),
            postgres=PostgresStore(pg["connection"], schema=pg["schema"], roles=pg["roles"]) if pg else None,
            neo4j=Neo4jStore(graph["driver"], database=graph["database"], uri=graph["uri"]) if graph else None)
        result = snapshot(source, root, apply=args.apply, sample=config.get("sample", False))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
