"""Explicit legacy evidence dry-run, capture rehearsal, import and writer handoff.

No default DSN, credential discovery, .env loading, or automatic schema/capture
installation. Mutating actions refuse the known live databases in this tooling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from papervault.store.knowledge_cutover import CutoverRepository, SourceCapture, validate_rehearsal_dsn
from papervault.store.knowledge_import import LegacyImportRepository, plan_legacy_import
from papervault.store.repository import AuditEvent, PostgresCatalogRepository


def _read(path, default=None):
    return json.loads(path.read_text()) if path else default


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("dry-run", "apply", "initialize-catalog", "install-capture",
        "capture-snapshot", "capture-delta", "freeze-source", "handoff"), default="dry-run")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--changes", type=Path, help="SourceCapture.delta JSON; verifies capture identity")
    parser.add_argument("--mappings", type=Path)
    parser.add_argument("--docstatus", type=Path, help="Explicit current source document-status evidence")
    parser.add_argument("--configuration", type=Path, help="Non-secret relevant source configuration JSON")
    parser.add_argument("--source-dsn", help="Explicit DSN; dry-run opens a read-only transaction")
    parser.add_argument("--source-schema", default="public")
    parser.add_argument("--source-id")
    parser.add_argument("--catalog-dsn")
    parser.add_argument("--catalog-schema", default="catalog")
    parser.add_argument("--operation-id")
    parser.add_argument("--after", type=int)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--legacy-role", action="append", default=[])
    parser.add_argument("--expected-epoch", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action in {"dry-run", "apply"} and not (args.snapshot or args.source_dsn):
        parser.error("--snapshot or --source-dsn is required")
    if args.snapshot and args.source_dsn:
        parser.error("supply a snapshot or an explicit source connection")
    if args.action in {"apply", "initialize-catalog", "handoff"} and not args.catalog_dsn:
        parser.error("explicit --catalog-dsn required")
    if args.action in {"apply", "initialize-catalog", "handoff"} and not args.operation_id:
        parser.error("stable --operation-id required")
    configuration = _read(args.configuration, {})
    report = None
    if args.source_dsn:
        import psycopg
        if not args.source_id:
            parser.error("explicit --source-id required for source connection")
        mutating = args.action in {"install-capture", "capture-snapshot", "freeze-source"}
        if mutating:
            validate_rehearsal_dsn(args.source_dsn)
        options = "" if mutating else "-c default_transaction_read_only=on"
        with psycopg.connect(args.source_dsn, autocommit=True, options=options) as connection:
            capture = SourceCapture(connection, ledger_schema=args.source_schema, source=args.source_id,
                                    configuration=configuration, allow_production=not mutating)
            if args.action == "install-capture":
                report = capture.install()
            elif args.action == "capture-snapshot":
                report = capture.snapshot()
            elif args.action == "capture-delta":
                report = capture.delta(args.after)
            elif args.action == "freeze-source":
                report = capture.freeze(legacy_roles=args.legacy_role, drain=_read(args.evidence, {}))
            elif args.action in {"dry-run", "apply"}:
                snapshot = capture.readonly_snapshot()
            else:
                parser.error("source connection does not apply to this action")
    elif args.action in {"install-capture", "capture-snapshot", "capture-delta", "freeze-source"}:
        parser.error("explicit --source-dsn required")
    if args.action in {"dry-run", "apply"}:
        snapshot = _read(args.snapshot) if args.snapshot else snapshot
        delta = _read(args.changes, {})
        if delta and delta.get("capture_id") != snapshot.get("capture_id"):
            parser.error("delta capture identity differs from snapshot")
        plan = plan_legacy_import(snapshot, mappings=_read(args.mappings, []),
                                  doc_status_rows=_read(args.docstatus, []), changes=delta.get("changes", []),
                                  final_checkpoint=delta.get("final_checkpoint"))
        report = plan
    if args.action in {"apply", "initialize-catalog", "handoff"}:
        import psycopg
        validate_rehearsal_dsn(args.catalog_dsn)
        with psycopg.connect(args.catalog_dsn, autocommit=True) as connection:
            repository = PostgresCatalogRepository(connection, schema=args.catalog_schema)
            importer = LegacyImportRepository(repository)
            if args.action == "initialize-catalog":
                repository.ensure_schema(audit=AuditEvent("operator", "catalog-schema", {"operation_id": args.operation_id}))
                repository.ensure_knowledge_schema(audit=AuditEvent("operator", "knowledge-schema", {"operation_id": args.operation_id}))
                importer.ensure_schema(audit=AuditEvent("operator", "migration-schema", {"operation_id": args.operation_id}))
                report = {"initialized": args.catalog_schema}
            elif args.action == "apply":
                report = importer.apply(plan, operation_id=args.operation_id)
            else:
                if args.expected_epoch is None or args.evidence is None:
                    parser.error("--expected-epoch and --evidence required for handoff")
                report = CutoverRepository(repository).commit_handoff(_read(args.evidence),
                    expected_epoch=args.expected_epoch, operation_id=args.operation_id)
    output = json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
