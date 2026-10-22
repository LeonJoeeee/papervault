#!/usr/bin/env python3
"""Restore or replay a verified package in empty isolated targets; never promote."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("restore", "replay"), required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--source-generation")
    parser.add_argument("--parents", type=Path, help="JSON list of retained parent-unit identities for replay")
    parser.add_argument("--config", type=Path, help="explicit isolated target identities and credential environment names")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "replay" and not args.source_generation:
        parser.error("replay requires --source-generation")
    if args.apply and not args.config:
        parser.error("--apply requires --config with explicit isolated targets")
    from papervault.store.recovery import checked_path
    args.package, args.root = checked_path(args.package), checked_path(args.root)
    from papervault.store.recovery_restore import restore_recovery_set
    from papervault.store.recovery_replay import replay_recovery_set
    from snapshot_derivatives import connections
    with ExitStack() as stack:
        config = json.loads(args.config.read_bytes()) if args.config else {}
        targets = connections(config, stack) if args.apply else {}
        options = {"target_generation": args.generation, "apply": args.apply,
            "catalog_connection": targets.get("catalog", {}).get("connection"),
            "catalog_schema": targets.get("catalog", {}).get("schema")}
        if args.mode == "restore":
            if args.apply:
                from papervault.store.recovery import maintenance_deadline
                stack.enter_context(maintenance_deadline())
            result = restore_recovery_set(args.package, args.root, **options,
                postgres_target=targets.get("postgres"), neo4j_target=targets.get("neo4j"))
        else:
            from recovery_runtime import physical_replay_factory
            factory, identity = physical_replay_factory(config, targets, args.package, args.root) if args.apply and config.get("physical_replay") else (None, None)
            result = asyncio.run(replay_recovery_set(args.package, args.root, **options,
                source_generation=args.source_generation,
                parent_ids=json.loads(args.parents.read_bytes()) if args.parents else None,
                adapter_factory=factory, backend_identity=identity))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
