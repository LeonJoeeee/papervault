"""Restartable shadow import and lossless reconciliation; never writes a vault."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .adapter import LibraryAdapter, read_json
from .compare import record_differences
from .repository import AuditEvent, PendingJob, Record, payload_hash


def identity_conflicts(records: list[Record]) -> list[dict]:
    conflicts = []
    for field in ("doi", "arxiv_id"):
        identities = defaultdict(list)
        for record in records:
            if value := getattr(record, field):
                identities[value].append(record.key)
        conflicts.extend({"field": field, "value": value, "keys": sorted(keys)}
                         for value, keys in sorted(identities.items()) if len(keys) > 1)
    return conflicts


def _supporting_inputs(adapter, records, ledger_path, jobs_path):
    if ledger_path is None and adapter.source.get("path"):
        sibling = Path(adapter.source["path"]).with_name("correction-ledger.json")
        ledger_path = sibling if sibling.exists() else None
    ledger = read_json(Path(ledger_path)) if ledger_path else {}
    if not isinstance(ledger, dict) or ledger.get("version", 1) != 1:
        raise ValueError("unsupported ruling ledger snapshot")
    rulings = ledger.get("policy_rulings", [])
    if not isinstance(rulings, list):
        raise ValueError("policy_rulings must be a list")
    for row in rulings:
        if (not isinstance(row, dict) or not isinstance(row.get("action"), str)
                or not row["action"].strip() or not isinstance(row.get("reason"), str)
                or not row["reason"].strip() or type(row.get("revision")) is not int
                or row["revision"] < 1 or not isinstance(row.get("keys"), list)
                or not row["keys"] or any(not isinstance(k, str) or not k for k in row["keys"])):
            raise ValueError("invalid policy ruling snapshot")
        if datetime.fromisoformat(row["timestamp"]).tzinfo is None:
            raise ValueError("ruling timestamp must be timezone-aware")
    job_data = read_json(Path(jobs_path)) if jobs_path else {"jobs": []}
    if not isinstance(job_data, dict) or not isinstance(job_data.get("jobs"), list):
        raise ValueError("job snapshot must contain a jobs list")
    jobs = [PendingJob(**row) for row in job_data["jobs"]]
    keys = {record.key for record in records}
    aliases, errors = {}, []
    for record in records:
        if record.merge == "alias":
            canonical = record.payload.get("terminal_state", {}).get("alias_key")
            if not canonical or canonical not in keys:
                errors.append({"key": record.key, "error": "dangling_alias", "canonical": canonical})
            else:
                aliases[record.key] = canonical
    for alias in aliases:
        seen, cursor = set(), alias
        while cursor in aliases:
            if cursor in seen:
                errors.append({"key": alias, "error": "alias_cycle"})
                break
            seen.add(cursor)
            cursor = aliases[cursor]
    for row in rulings:
        for key in row["keys"]:
            if key not in keys:
                errors.append({"key": key, "error": "dangling_ruling"})
    for job in jobs:
        if job.key not in keys:
            errors.append({"key": job.key, "error": "dangling_job", "id": job.id})
    if len({job.id for job in jobs}) != len(jobs):
        errors.append({"error": "duplicate_job_ids"})
    return aliases, rulings, jobs, errors, {
        "ledger": payload_hash(ledger) if ledger_path else None,
        "jobs": payload_hash(job_data) if jobs_path else None,
    }


def import_catalog(adapter: LibraryAdapter, repository=None, *, ledger_path=None,
                   jobs_path=None) -> dict:
    records = list(adapter.all())
    aliases, rulings, jobs, errors, support_hashes = _supporting_inputs(
        adapter, records, ledger_path, jobs_path,
    )
    hashes = {record.key: record.payload_sha256 for record in records}
    report = {
        "dry_run": repository is None, "source": adapter.source,
        "supporting_source_sha256": support_hashes,
        "total": len(records),
        "counts": {field: dict(sorted(Counter(getattr(r, field) for r in records).items()))
                   for field in ("download_status", "retrieval", "policy", "merge", "quarantine")},
        "identity_conflicts": identity_conflicts(records), "integrity_errors": errors,
        "payload_sha256": hashes, "payload_manifest_sha256": payload_hash(hashes),
        "per_key_match": dict.fromkeys(hashes), "difference_rate": None,
        "changed": 0, "supporting_changed": {"aliases": 0, "rulings": 0, "jobs": 0},
        "supporting_counts": {"aliases": len(aliases), "rulings": len(rulings), "jobs": len(jobs)},
        "target_identity_indexes": None, "extra_keys": [],
    }
    if repository is None or errors:
        return report

    def event(name, **detail):
        return AuditEvent("import_catalog", name, {
            "source": adapter.source, "supporting_source_sha256": support_hashes, **detail,
        })

    repository.ensure_schema(audit=event("schema_migration"))
    for record in records:
        report["changed"] += repository.upsert(record, audit=event(
            "snapshot_record", payload_sha256=record.payload_sha256,
        ))
    for alias, canonical in sorted(aliases.items()):
        report["supporting_changed"]["aliases"] += repository.add_alias(
            alias, canonical, audit=event("snapshot_alias", canonical=canonical),
        )
    for ruling in rulings:
        report["supporting_changed"]["rulings"] += repository.import_ruling(
            ruling, audit=event("snapshot_ruling", ruling=ruling),
        )
    for job in jobs:
        report["supporting_changed"]["jobs"] += repository.import_job(job, audit=event("snapshot_job"))
    # Each committed record is its own restart checkpoint. A retry emits events
    # only for remaining/changed effects; no checkpoint file or live writer is used.
    report["target_identity_indexes"] = repository.finalize_identity_indexes(
        audit=event("identity_indexes"),
    )
    target = {r.key: r for r in repository.all()}
    differences = []
    for record in records:
        fields = record_differences(record, target.get(record.key))
        report["per_key_match"][record.key] = not fields
        if fields:
            differences.append({"key": record.key, "fields": fields})
    report["extra_keys"] = sorted(target.keys() - hashes.keys())
    report["differences"] = differences
    denominator = len(records) + len(report["extra_keys"])
    report["difference_rate"] = (
        (len(differences) + len(report["extra_keys"])) / denominator if denominator else 0.0
    )
    report["target_total"] = len(target)
    return report
