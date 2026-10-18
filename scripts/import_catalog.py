#!/usr/bin/env python3
"""Import a snapshot into the non-authoritative catalog, never the live Library.

Examples (run from a checkout with its dependencies installed):
  .venv/bin/python scripts/import_catalog.py --dry-run --index /copy/index.json
  PAPERVAULT_CATALOG_DSN=... .venv/bin/python scripts/import_catalog.py --apply --index /copy/index.json

The default is a read-only dry run of /data/paper-vault/index.json. --apply writes
only to the explicit DSN's catalog schema; it never changes JSON or LightRAG.
The sibling correction-ledger.json is read if present; --ledger overrides it.
--jobs accepts an explicit {"jobs": [...]} snapshot, never a live job source.
"""

import argparse
import json
import os
from pathlib import Path

import psycopg

from papervault.store.adapter import LibraryAdapter
from papervault.store.importer import import_catalog
from papervault.store.repository import PostgresCatalogRepository


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--index", type=Path, default=Path("/data/paper-vault/index.json"))
    parser.add_argument("--schema", default="catalog")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--summary", action="store_true", help="omit per-key maps for a compact report")
    args = parser.parse_args(argv)
    dsn = os.environ.get("PAPERVAULT_CATALOG_DSN") if args.apply else None
    if args.apply and not dsn:
        parser.error("--apply requires PAPERVAULT_CATALOG_DSN; no credential discovery is performed")
    report = None
    try:
        adapter = LibraryAdapter.from_snapshot(args.index)
        report = import_catalog(adapter, ledger_path=args.ledger, jobs_path=args.jobs)
        if args.apply and not report["integrity_errors"]:
            with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
                report = import_catalog(adapter, PostgresCatalogRepository(conn, schema=args.schema),
                                        ledger_path=args.ledger, jobs_path=args.jobs)
    except (OSError, ValueError, TypeError, KeyError, psycopg.Error) as exc:
        # Database exceptions may contain credentials; report the class, never a DSN/message.
        report = report or {}
        report.update(error=type(exc).__name__, dry_run=not args.apply)
        print(json.dumps(report, sort_keys=True, ensure_ascii=False, default=str))
        return 2
    if args.summary:
        report = {key: value for key, value in report.items()
                  if key not in {"per_key_match", "payload_sha256"}}
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    return int(bool(report["integrity_errors"]) or report["difference_rate"] not in (None, 0))


if __name__ == "__main__":
    raise SystemExit(main())
