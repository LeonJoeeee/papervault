#!/usr/bin/env python3
"""Compare a JSON snapshot with the non-authoritative PG catalog (read-only).

PAPERVAULT_CATALOG_DSN=... .venv/bin/python scripts/shadow_read_compare.py --index /copy/index.json

The default workload is the union of all JSON/PG keys plus matches from the
frozen F0 R2 gold_search_pools.jsonl. --limit bounds the general keys; R2 keys
and repeated --pool-key arguments are always included. A nonzero difference
rate exits 1; invalid input/connection errors exit 2. No schema is initialized.
"""

import argparse
from importlib.resources import files
import json
import os
from pathlib import Path

import psycopg

from papervault.store.adapter import LibraryAdapter
from papervault.store.compare import compare_catalog
from papervault.store.repository import PostgresCatalogRepository


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("/data/paper-vault/index.json"))
    parser.add_argument("--schema", default="catalog")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pool", type=Path, default=Path(str(
        files("papervault").joinpath("eval/gold_search_pools.jsonl"),
    )))
    parser.add_argument("--pool-key", action="append", default=[])
    args = parser.parse_args(argv)
    dsn = os.environ.get("PAPERVAULT_CATALOG_DSN")
    if not dsn:
        parser.error("PAPERVAULT_CATALOG_DSN is required; no credential discovery is performed")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    try:
        adapter = LibraryAdapter.from_snapshot(args.index)
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5,
                             options="-c default_transaction_read_only=on") as conn:
            repository = PostgresCatalogRepository(conn, schema=args.schema, read_only=True)
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                revision = conn.execute("SELECT pg_current_snapshot()::text AS revision").fetchone()
                report = compare_catalog(adapter, repository, limit=args.limit,
                                         pool_path=args.pool, pool_keys=args.pool_key)
                report["postgres_snapshot"] = revision["revision"]
    except (OSError, ValueError, TypeError, KeyError, psycopg.Error) as exc:
        print(json.dumps({"error": type(exc).__name__}))
        return 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    return int(report["different"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
