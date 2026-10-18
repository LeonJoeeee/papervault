#!/usr/bin/env python3
"""Export PG authority to canonical index.json; --verify checks shadow parity.

The output path and PAPERVAULT_CATALOG_DSN are explicit. The database connection
is read-only; no schema changes, imports or credential discovery occur.
Export obeys handoff fencing and is permitted again after a preflight rollback.
"""
import argparse
import json
import os
from pathlib import Path

import psycopg

from papervault.store.exporter import export_catalog
from papervault.store.fencing import WriterFenced, writer_lease
from papervault.store.repository import PostgresCatalogRepository


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--schema", default="catalog")
    args = parser.parse_args(argv)
    dsn = os.environ.get("PAPERVAULT_CATALOG_DSN")
    if not dsn:
        parser.error("PAPERVAULT_CATALOG_DSN is required")
    lease = None
    try:
        lease = writer_lease(args.output.parent, None)
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5,
                             options="-c default_transaction_read_only=on") as connection:
            report = export_catalog(PostgresCatalogRepository(connection, schema=args.schema,
                                    read_only=True), args.output, verify=args.verify)
    except (OSError, ValueError, TypeError, KeyError, WriterFenced, psycopg.Error) as exc:
        print(json.dumps({"error": type(exc).__name__}))
        return 2
    finally:
        if lease:
            lease.close()
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
