#!/usr/bin/env python3
"""P2 pre-flip rehearsal tooling: a vault COPY and staging DB are mandatory.

--preflight --apply fences and imports delta; --flip probes/exports while fenced;
--rollback exercises the narrow frozen window; --resume closes it durably BEFORE
executing explicit orchestrator-supplied restart and health commands. Commands
are argv (shlex), never shell scripts. Live authority activation is not enabled.
"""
import argparse
import json
import os
import shlex
from pathlib import Path

import psycopg

from papervault.store.cutover import Cutover
from papervault.store.repository import PostgresCatalogRepository


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("preflight", "flip", "rollback", "resume"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--rehearsal", action="store_true", required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--schema", default="catalog")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restart-command")
    parser.add_argument("--health-command")
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    if args.vault.resolve().is_relative_to(Path("/data/paper-vault").resolve()):
        parser.error("P2 rehearsal refuses the live vault")
    dsn = os.environ.get("PAPERVAULT_CATALOG_DSN")
    if not dsn:
        parser.error("PAPERVAULT_CATALOG_DSN is required")
    if args.resume and not (args.restart_command and args.health_command and args.reason):
        parser.error("--resume requires --restart-command, --health-command and --reason")
    os.environ["PAPERVAULT_CATALOG_SCHEMA"] = args.schema
    try:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as connection:
            if connection.info.dbname != "papervault_catalog_stage":
                raise ValueError("rehearsal requires the papervault_catalog_stage database")
            cutover = Cutover(args.vault, args.env_file,
                              PostgresCatalogRepository(connection, schema=args.schema))
            if args.preflight:
                report = cutover.preflight(apply=args.apply)
            elif args.flip:
                report = cutover.flip()
            elif args.rollback:
                report = cutover.rollback()
            else:
                report = cutover.resume(restart=shlex.split(args.restart_command),
                    health=shlex.split(args.health_command), reason=args.reason)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, psycopg.Error) as exc:
        # Never expose DB messages, credentials or command output.
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc) if isinstance(exc, RuntimeError) else "rehearsal refused"}))
        return 2
    # Payload hash maps are large and do not add evidence to counts/parity.
    report = {k: v for k, v in report.items() if k not in {"payload_sha256", "per_key_match"}}
    print(json.dumps(report, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
