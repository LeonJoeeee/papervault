#!/usr/bin/env python3
"""Preview a scoped derivative census; --apply atomically publishes mode-0600 JSON.

Uses the explicit snapshot source configuration/connections, never runtime CONFIG
or dotenv. ``extra_derivative_roots`` is required (an explicit empty list affirms
there are none) and must enumerate retained snapshots and every other derivative
filesystem root outside vault/extracts and knowledge_root. Operators own scope
completeness; no live location is discovered. External stores use the established
PostgresStore/Neo4jStore estimates, not a claim of exact physical disk usage.
Catalog relation bytes are excluded because the runtime measures/adds them.
Refresh the applied census at least every 300 seconds for admission to continue.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4

os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from snapshot_derivatives import connections

from papervault.store.derivative_capacity import (
    _check_binding, _identity, _inventory, _open_directory, _path, _regular,
)


def _roots(configuration):
    if not isinstance(configuration, dict) or not all(
            name in configuration for name in ("vault", "knowledge_root", "catalog", "extra_derivative_roots")):
        raise ValueError("explicit source configuration requires catalog, vault, knowledge and extra derivative roots")
    if type(configuration.get("metadata_sample", False)) is not bool:
        raise ValueError("configuration metadata_sample must be an explicit boolean")
    if not configuration.get("metadata_sample", False) and not all(
            name in configuration for name in ("postgres", "neo4j")):
        raise ValueError("complete configuration requires explicit PostgreSQL and Neo4j adapters")
    extras = configuration["extra_derivative_roots"]
    if not isinstance(extras, list) or any(not isinstance(path, str) or not path for path in extras):
        raise ValueError("extra_derivative_roots must be an explicit list of paths")
    paths = [configuration["vault"], configuration["knowledge_root"], *extras]
    if any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths):
        raise ValueError("explicit census roots must be absolute paths")
    vault, knowledge, *extras = (_path(Path(path)) for path in paths)
    if vault.is_relative_to(knowledge) or knowledge.is_relative_to(vault):
        raise ValueError("source vault and knowledge roots must be disjoint")
    for root in extras:
        if any(root.is_relative_to(primary) or primary.is_relative_to(root) for primary in (vault, knowledge)):
            raise ValueError("extra derivative root overlaps protected source or primary derivative roots")
    for root in (vault, knowledge, *extras):
        fd = _open_directory(root)
        if fd is None:
            raise ValueError("configured census root is absent: " + str(root))
        os.close(fd)
    return vault, knowledge, extras


def validate_configuration(configuration):
    """Refuse omitted scope and unsafe paths before establishing connections."""
    _roots(configuration)


def _prove_metadata_only(catalog):
    from psycopg import sql

    queries = (
        ("input_manifests", "manifest->>'tier' IS DISTINCT FROM 'metadata'"),
        ("knowledge_receipts", "receipt->>'achieved_tier' IS DISTINCT FROM 'metadata' OR EXISTS "
         "(SELECT 1 FROM jsonb_array_elements(receipt->'output_inventory') AS item "
         "WHERE item->>'store' IS NULL OR item->>'store' NOT IN ('catalog','catalog_inputs'))"),
    )
    for table, condition in queries:
        row = catalog.connection.execute(sql.SQL("SELECT count(*) AS count FROM {} WHERE ").format(
            catalog._table(table)) + sql.SQL(condition)).fetchone()
        if row["count"] != 0:
            raise ValueError("metadata_sample cannot omit referenced physical backend derivatives")


def measure(configuration, *, catalog, postgres=None, neo4j=None):
    """Read measured backend/retained usage, with no filesystem or backend writes."""
    vault, knowledge, extras = _roots(configuration)
    measured_at = datetime.now(timezone.utc).isoformat()
    if configuration.get("metadata_sample", False):
        _prove_metadata_only(catalog)
    elif postgres is None or neo4j is None:
        raise ValueError("complete census requires both physical backend adapters")
    external = 0
    for adapter in (postgres, neo4j):
        if adapter is not None:
            size = adapter.estimated_bytes()
            if type(size) is not int or size < 0:
                raise ValueError("backend adapter supplied an invalid derivative size")
            external += size
    seen = set()
    retained = 0
    for root in extras:
        fd = _open_directory(root)
        if fd is None:
            raise ValueError("configured retained derivative root disappeared")
        try:
            retained += _inventory(fd, seen, os.fstat(fd).st_dev)
            _check_binding(root, fd)
        finally:
            os.close(fd)
    return {"schema_version": 1, "backend_bytes": external + retained,
            "measured_at": measured_at, "scope": {"vault_root": str(vault), "knowledge_root": str(knowledge)},
            "includes_catalog": False, "backend_estimated_bytes": external,
            "retained_filesystem_bytes": retained, "extra_derivative_roots": [str(root) for root in extras],
            "metadata_sample": configuration.get("metadata_sample", False),
            "measurement": "conservative recovery-adapter census; not an exact on-disk guarantee"}


def _existing_output(parent_fd, name, census):
    fd = None
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        info = os.fstat(fd)
        _regular(info)
        if info.st_size > 65536:
            raise ValueError("existing output is not an owned usage census")
        prior = json.loads(os.read(fd, 65537))
        if (not isinstance(prior, dict) or prior.get("scope") != census["scope"]
                or type(prior.get("backend_bytes")) is not int or prior["backend_bytes"] < 0
                or prior.get("includes_catalog") is not False
                or not isinstance(prior.get("measured_at"), str)):
            raise ValueError("existing output is not an owned scoped usage census")
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("existing output is not an owned usage census") from exc
    finally:
        if fd is not None:
            os.close(fd)


def publish(output, census):
    """Atomically replace an explicit regular output, preserving protected aliases."""
    output = _path(Path(output))
    parent_fd = _open_directory(output.parent)
    if parent_fd is None:
        raise ValueError("census output parent must already exist")
    temporary = ".derivative-usage-" + uuid4().hex + ".tmp"
    fd = None
    try:
        _existing_output(parent_fd, output.name, census)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent_fd)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write((json.dumps(census, sort_keys=True, indent=2) + "\n").encode())
            stream.flush()
            os.fsync(fd)
        _check_binding(output.parent, parent_fd)
        _existing_output(parent_fd, output.name, census)
        os.replace(temporary, output.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        raise ValueError("protected census publication failed") from exc
    finally:
        try:
            if fd is not None:
                try:
                    info = os.stat(temporary, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    if _identity(info) == _identity(os.fstat(fd)):
                        os.unlink(temporary, dir_fd=parent_fd)
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="explicit existing snapshot source JSON")
    parser.add_argument("--output", type=Path, required=True, help="explicit census path; parent must exist")
    parser.add_argument("--apply", action="store_true", help="publish measured census; default prints preview only")
    args = parser.parse_args(argv)
    from papervault.store.recovery import checked_path
    from papervault.store.recovery_backends import Neo4jStore, PostgresStore
    from papervault.store.repository import PostgresCatalogRepository

    configuration = json.loads(checked_path(args.config).read_bytes())
    validate_configuration(configuration)
    output = checked_path(args.output)
    with ExitStack() as stack:
        targets = connections(configuration, stack)
        catalog = targets["catalog"]
        pg, graph = targets.get("postgres"), targets.get("neo4j")
        postgres = PostgresStore(pg["connection"], schema=pg["schema"], roles=pg["roles"]) if pg else None
        if postgres is not None and not set(postgres.roles.values()).issubset(postgres._tables()):
            raise ValueError("complete PostgreSQL derivative role inventory is absent")
        neo4j = Neo4jStore(graph["driver"], database=graph["database"], uri=graph["uri"]) if graph else None
        census = measure(configuration,
            catalog=PostgresCatalogRepository(catalog["connection"], schema=catalog["schema"]),
            postgres=postgres, neo4j=neo4j)
    if args.apply:
        publish(output, census)
    print(json.dumps({"apply": args.apply, "output": str(output), **census}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
