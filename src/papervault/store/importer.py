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
    explicit = {job.id: job for job in jobs}
    for record in records:
        claim = record.payload.get("reopen_claim")
        if claim and claim["phase"] in {"queued", "dispatched"}:
            job = PendingJob("reopen:" + record.key, "reopen", record.key,
                state=claim["phase"], attempts=claim["generation"], lease_id=claim["lease_id"],
                detail={"campaign_id": claim["campaign_id"]})
            if job.id not in explicit:
                jobs.append(job)
            elif any(getattr(explicit[job.id], field) != getattr(job, field)
                     for field in ("kind", "key", "state", "attempts", "lease_id")):
                raise ValueError("job snapshot disagrees with active record claim")
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
        "supporting_differences": {"aliases": []}, "phase": "dry_run",
    }
    if repository is None or errors:
        return report

    def event(name, **detail):
        return AuditEvent("import_catalog", name, {
            "source": adapter.source, "supporting_source_sha256": support_hashes, **detail,
        })

    try:
        report["phase"] = "schema"
        repository.ensure_schema(audit=event("schema_migration"))
        report["phase"] = "identity_preflight"
        existing = {r.key: r for r in repository.all()}
        projected = existing | {r.key: r for r in records}
        report["identity_conflicts"] = identity_conflicts(list(projected.values()))
        constrained = repository.identity_index_fields()
        blocked = [conflict for conflict in report["identity_conflicts"] if conflict["field"] in constrained]
        if blocked:
            report["integrity_errors"].extend(
                {"error": "identity_conflict_with_unique_index", **conflict} for conflict in blocked
            )
            report["target_total"] = len(existing)
            return report
        report["phase"] = "records"
        def alias_depth(key):
            return 1 + alias_depth(aliases[key]) if key in aliases else 0

        for record in sorted(records, key=lambda row: (alias_depth(row.key), row.key)):
            previous = existing.get(record.key)
            report["changed"] += repository.upsert(record, audit=event(
                "snapshot_record", payload_sha256=record.payload_sha256,
            ), expected_revision=previous.revision if previous else 0)
        report["phase"] = "aliases"
        for alias, canonical in sorted(aliases.items()):
            report["supporting_changed"]["aliases"] += repository.add_alias(
                alias, canonical, audit=event("snapshot_alias", canonical=canonical),
            )
        report["phase"] = "rulings"
        for ruling in rulings:
            report["supporting_changed"]["rulings"] += repository.import_ruling(
                ruling, audit=event("snapshot_ruling", ruling=ruling),
            )
        report["phase"] = "jobs"
        for job in jobs:
            report["supporting_changed"]["jobs"] += repository.import_job(job, audit=event("snapshot_job"))
        # Each committed record is its own restart checkpoint. A retry emits
        # events only for remaining/changed effects, without a checkpoint file.
        report["phase"] = "identity_indexes"
        report["target_identity_indexes"] = repository.finalize_identity_indexes(
            audit=event("identity_indexes"),
        )
        report["phase"] = "reconciliation"
        target = {r.key: r for r in repository.all()}
        target_aliases = repository.all_aliases()
    except Exception as exc:
        # Preserve committed progress for the CLI without hiding the failure
        # from Python callers or including potentially sensitive DB messages.
        exc.catalog_report = report
        raise
    differences = []
    for record in records:
        fields = record_differences(record, target.get(record.key),
                                   json_alias=aliases.get(record.key), postgres_alias=target_aliases.get(record.key))
        report["per_key_match"][record.key] = not fields
        if fields:
            differences.append({"key": record.key, "fields": fields})
    report["supporting_differences"]["aliases"] = [
        {"key": key, "json": aliases.get(key), "postgres": target_aliases.get(key)}
        for key in sorted(aliases.keys() | target_aliases.keys())
        if aliases.get(key) != target_aliases.get(key)
    ]
    report["extra_keys"] = sorted((target.keys() | target_aliases.keys()) - hashes.keys())
    report["differences"] = differences
    denominator = len(records) + len(report["extra_keys"])
    report["difference_rate"] = (
        (len(differences) + len(report["extra_keys"])) / denominator if denominator else 0.0
    )
    report["target_total"] = len(target)
    report["phase"] = "reconciled"
    return report
