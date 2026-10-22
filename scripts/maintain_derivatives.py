#!/usr/bin/env python3
"""Inspect or explicitly apply disposable-extract GC and isolated generation maintenance."""

import argparse
import json
import os
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "window", "disposition", "gc", "fork", "promote"))
    parser.add_argument("--dsn-env", required=True, help="environment variable containing the explicit catalog DSN")
    parser.add_argument("--catalog-schema", required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--payload", type=Path, help="JSON object with exact command inputs/revisions")
    parser.add_argument("--operation-id", required=True, help="retain this ID and payload across response-loss retries")
    parser.add_argument("--owner")
    parser.add_argument("--window-id")
    parser.add_argument("--lease-seconds", type=int, default=300)
    parser.add_argument("--apply", action="store_true", help="default is a read-only plan")
    args = parser.parse_args(argv)
    if args.action != "inspect" and not args.payload:
        parser.error("this action requires --payload")
    import psycopg
    from papervault.store.derivative_maintenance import maintain
    from papervault.store.knowledge import KnowledgeRepository
    from papervault.store.repository import PostgresCatalogRepository

    # Schema installation remains an explicit deployment step, never a side
    # effect of an inspection or maintenance command.
    with psycopg.connect(os.environ[args.dsn_env], autocommit=True, connect_timeout=5) as connection:
        repo = PostgresCatalogRepository(connection, schema=args.catalog_schema, read_only=not args.apply)
        result = maintain(KnowledgeRepository(repo), args.action,
            json.loads(args.payload.read_bytes()) if args.payload else {}, vault=args.vault,
            operation_id=args.operation_id, apply=args.apply, owner=args.owner,
            window_id=args.window_id, lease_seconds=args.lease_seconds)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
