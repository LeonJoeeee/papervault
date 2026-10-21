"""Read-only R2 report from explicit catalog evidence and frozen ruler inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from papervault.store.coverage import query_coverage
from papervault.store.coverage_r2 import coverage_r2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--coverage", type=Path, help="Previously evaluated query_coverage JSON")
    source.add_argument("--catalog-rows", type=Path, help="Consistent coverage_rows catalog snapshot JSON")
    source.add_argument("--dsn", help="Explicit PostgreSQL catalog DSN; read-only transaction")
    parser.add_argument("--pools", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--legacy-records", type=Path, help="Current F0 ruler records with key/title/doi/readable/indexed")
    parser.add_argument("--evaluation-time", help="Timezone-aware ISO timestamp")
    parser.add_argument("--max-observation-age", type=float, default=300)
    parser.add_argument("--input-revision", default="current")
    parser.add_argument("--recipe-digest", default="current")
    parser.add_argument("--generation-id", default="current")
    parser.add_argument("--inventory-revision")
    parser.add_argument("--measurement-label", help="Versioned label required for alternate instrument policy")
    parser.add_argument("--output", type=Path, help="Omit to write JSON to stdout")
    args = parser.parse_args(argv)
    if args.coverage:
        coverage = json.loads(args.coverage.read_text())
    else:
        kwargs = dict(evaluation_time=args.evaluation_time, max_observation_age=args.max_observation_age,
                      consumed_input_revision=args.input_revision, recipe_digest=args.recipe_digest,
                      generation_id=args.generation_id, expected_inventory_revision=args.inventory_revision)
        if args.catalog_rows:
            class Snapshot:
                def coverage_rows(self, scope=None):
                    return json.loads(args.catalog_rows.read_text())
            coverage = query_coverage(Snapshot(), **kwargs)
        else:
            # Never use the catalog constructor/schema initializer: report access
            # must work with a PostgreSQL role that cannot write or migrate.
            import psycopg
            from papervault.store.knowledge import KnowledgeRepository
            from papervault.store.repository import PostgresCatalogRepository
            with psycopg.connect(args.dsn, autocommit=True, options="-c default_transaction_read_only=on") as connection:
                catalog = PostgresCatalogRepository(connection, read_only=True)
                coverage = query_coverage(KnowledgeRepository(catalog), **kwargs)
    pools = [json.loads(line) for line in args.pools.read_text().splitlines() if line.strip()]
    report = coverage_r2(pools, coverage, frozen_snapshot=json.loads(args.frozen.read_text()),
                         current_legacy_records=json.loads(args.legacy_records.read_text()) if args.legacy_records else None,
                         measurement_label=args.measurement_label)
    output = json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
