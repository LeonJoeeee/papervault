"""Lossless, resumable legacy evidence import without fabricated job provenance."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from importlib.resources import files
import json

from .knowledge import KnowledgeRepository, OperationConflict
from .repository import AuditEvent, payload_hash, render_catalog_ddl

SOURCE_FIELDS = ("workspace", "ingest_source", "source_id")
TIERS = {"done": "fulltext", "done_abstract": "abstract", "done_meta": "metadata"}


def _json(value):
    return json.loads(json.dumps(value, default=lambda item: item.isoformat()))


def source_identity(row):
    if any(not isinstance(row.get(field), str) or not row[field] for field in SOURCE_FIELDS):
        raise ValueError("legacy source identity requires workspace, ingest_source and source_id")
    return {field: row[field] for field in SOURCE_FIELDS}


def source_row_id(row):
    return payload_hash(source_identity(row))


def _doc_identity(row):
    return row.get("workspace"), row.get("doc_id", row.get("id"))


def plan_legacy_import(snapshot, *, mappings=(), doc_status_rows=(), changes=(), final_checkpoint=None):
    """Plan immutable source/version evidence and quarantined legacy projections.

    Mappings are explicit typed source identities, never guessed from basenames.
    Legacy ``attempts`` is a consecutive-failure counter, not attempt history.
    A plan is side-effect free; apply is a separate audited administrative action.
    """
    snapshot = _json(deepcopy(snapshot))
    if not isinstance(snapshot.get("source"), str) or not snapshot["source"]:
        raise ValueError("explicit source identity required")
    captured = bool(snapshot.get("capture_id")) and type(snapshot.get("checkpoint")) is int
    start = snapshot["checkpoint"] if captured else 0
    if start < 0:
        raise ValueError("checkpoint must be nonnegative")
    capture_id = snapshot.get("capture_id") or "uncaptured:" + payload_hash(snapshot)
    mappings_by_key = defaultdict(list)
    for mapping in mappings:
        if mapping.get("kind") not in {"paper", "operator_document", "section"} or not mapping.get("unit_id"):
            raise ValueError("mapping requires explicit typed unit_id")
        mappings_by_key[source_row_id(mapping)].append(deepcopy(mapping))
    current_rows = {source_row_id(row): row for row in snapshot.get("rows", [])}
    mapping_units = {identity: {mapping["unit_id"] for mapping in choices}
                     for identity, choices in mappings_by_key.items()}
    unit_counts = Counter(unit for identity in current_rows for unit in mapping_units.get(identity, ()))
    document_counts = Counter(_doc_identity(row) for row in current_rows.values())
    raw_events = []

    def index_row(row, increment):
        document_counts[_doc_identity(row)] += increment
        for unit in mapping_units.get(source_row_id(row), ()):
            unit_counts[unit] += increment

    def append_event(sequence, operation, row, deleted):
        choices = mappings_by_key[source_row_id(row)]
        duplicate_unit = len(choices) == 1 and unit_counts[choices[0]["unit_id"]] > 1
        raw_events.append((sequence, operation, row, deleted, duplicate_unit,
                           document_counts[_doc_identity(row)] > 1))

    # All snapshot rows share one aligned source state; journal dispositions
    # belong to the state after that atomic change, never the final row set.
    for row in snapshot.get("rows", []):
        append_event(start, "SNAPSHOT", row, False)
    expected = start + 1
    for change in changes:
        if not captured or change.get("sequence") != expected:
            raise ValueError("capture changes must be contiguous after snapshot checkpoint")
        expected += 1
        operation = change.get("operation")
        old, new = change.get("old"), change.get("new")
        if operation not in {"INSERT", "UPDATE", "DELETE"}:
            raise ValueError("unknown source change operation")
        if operation in {"UPDATE", "DELETE"} and not old:
            raise ValueError("source change requires old row")
        if operation in {"INSERT", "UPDATE"} and not new:
            raise ValueError("source change requires new row")
        if old and current_rows.get(source_row_id(old)) != old:
            raise ValueError("capture old image disagrees with aligned snapshot/journal")
        if operation == "INSERT" and source_row_id(new) in current_rows:
            raise ValueError("capture insert already exists in snapshot/journal")
        if old:
            current_rows.pop(source_row_id(old), None)
            index_row(old, -1)
        if new:
            current_rows[source_row_id(new)] = new
            index_row(new, 1)
        if old and (new is None or source_row_id(old) != source_row_id(new)):
            append_event(change["sequence"], "DELETE", old, True)
        if new:
            append_event(change["sequence"], operation, new, False)
    end = expected - 1
    if final_checkpoint is not None and (not captured or end != final_checkpoint):
        raise ValueError("capture changes must be contiguous through final checkpoint")
    latest = {}
    for sequence, operation, raw, deleted, _, _ in raw_events:
        latest[source_row_id(raw)] = (sequence, operation, raw, deleted)
    status_by_doc = defaultdict(list)
    for status in doc_status_rows:
        status_by_doc[_doc_identity(status)].append(status)
    sets = {key: [] for key in ("matches", "ledger_only", "docstatus_only", "duplicates", "conflicts", "provenance_gaps")}
    events = []
    ledger_doc_ids = {_doc_identity(raw) for _, _, raw, deleted in latest.values() if not deleted}
    for sequence, operation, raw, deleted, duplicate_unit, duplicate_document in raw_events:
        identity = source_row_id(raw)
        choices = mappings_by_key[identity]
        compatible = len(choices) == 1
        mapped = choices[0] if compatible else None
        dispositions = []
        matched = bool(status_by_doc.get(_doc_identity(raw)))
        dispositions.append("matched" if matched else "ledger_only")
        if len(choices) > 1:
            dispositions.append("conflicting_mapping")
        elif not choices:
            dispositions.append("unmatched_unit")
        if duplicate_unit:
            dispositions.append("duplicate_unit")
        if duplicate_document:
            dispositions.append("duplicate_document_identity")
        if len(status_by_doc.get(_doc_identity(raw), [])) > 1:
            dispositions.append("duplicate_docstatus")
        dispositions.append("provenance_gap")
        if deleted:
            dispositions.append("source_deleted")
        attempts = raw.get("attempts")
        if type(attempts) is not int or attempts < 0:
            dispositions.append("invalid_failure_counter")
        entry = {"source": snapshot["source"], "capture_id": capture_id, "source_row_id": identity,
                 "source_identity": source_identity(raw), "source_version": sequence, "operation": operation,
                 "raw": deepcopy(raw), "mapping": mapped, "unit_id": mapped["unit_id"] if mapped else None,
                 "claimed_execution_state": raw.get("status"), "claimed_tier": TIERS.get(raw.get("status")),
                 "legacy_consecutive_failures": attempts, "achieved_tier": None,
                 "consumed_input_revision": None, "recipe_digest": None, "generation_id": None,
                 "quarantine": True, "excluded": True, "executable": False, "deleted": deleted,
                 "dispositions": dispositions, "doc_status_evidence": deepcopy(status_by_doc.get(_doc_identity(raw), []))}
        entry["event_digest"] = payload_hash({key: entry[key] for key in (
            "source", "capture_id", "source_row_id", "source_version", "operation", "raw", "deleted")})
        events.append(entry)
        if latest[identity][0] == sequence:
            reference = {"source_row_id": identity, "source_identity": entry["source_identity"],
                         "source_version": sequence, "dispositions": dispositions}
            sets["matches" if matched else "ledger_only"].append(reference)
            sets["provenance_gaps"].append(reference)
            if any(reason.startswith("duplicate") for reason in dispositions):
                sets["duplicates"].append(reference)
            if "conflicting_mapping" in dispositions or "invalid_failure_counter" in dispositions:
                sets["conflicts"].append(reference)
    for status in doc_status_rows:
        if _doc_identity(status) not in ledger_doc_ids:
            sets["docstatus_only"].append({"identity": list(_doc_identity(status)), "raw": deepcopy(status),
                                           "disposition": "docstatus_only_unverified", "quarantine": True, "excluded": True})
    for identity, rows in status_by_doc.items():
        if identity not in ledger_doc_ids and len(rows) > 1:
            sets["duplicates"].append({"identity": list(identity), "raw": deepcopy(rows),
                "disposition": "duplicate_docstatus_only_identity", "quarantine": True, "excluded": True})
    checkpoint = {"source": snapshot["source"], "capture_id": capture_id, "snapshot_sequence": start if captured else None,
                  "last_sequence": end if captured else None, "handoff_eligible": captured,
                  "configuration": snapshot.get("configuration", {}), "schema": snapshot.get("schema", {}),
                  "source_rows": len(latest), "external_attempts": snapshot.get("external_attempts", "unknown"),
                  "final_row_count": len(current_rows), "final_rows_digest": payload_hash([
                      current_rows[key] for key in sorted(current_rows)])}
    return {"source": snapshot["source"], "capture_id": capture_id, "events": events, "sets": sets,
            "checkpoint": checkpoint, "counts": {key: len(values) for key, values in sets.items()},
            "legacy_states": dict(Counter(raw.get("status") for _, _, raw, deleted in latest.values() if not deleted))}


def validate_projection_unchanged(head, job):
    if not head or not head.get("projection_digest") or head["projection_digest"] != payload_hash(_json(job)):
        raise OperationConflict("legacy projection changed independently of the importer")


def _retains_evidence(previous, proposed):
    """Evidence may gain fields/observations, but accepted facts cannot disappear."""
    if isinstance(previous, dict):
        return isinstance(proposed, dict) and all(
            key in proposed and _retains_evidence(value, proposed[key]) for key, value in previous.items())
    if isinstance(previous, list):
        return isinstance(proposed, list) and all(
            any(_retains_evidence(value, candidate) for candidate in proposed) for value in previous)
    return previous == proposed


def _validate_enrichment(previous, proposed, old_job, values):
    enriched_fields = {"mapping", "unit_id", "dispositions", "doc_status_evidence"}
    fixed_evidence = {key: value for key, value in previous.items() if key not in enriched_fields}
    if not _retains_evidence(fixed_evidence, proposed):
        raise OperationConflict("same-version source evidence regression; enrichment required")
    old_mapping, new_mapping = previous.get("mapping"), proposed.get("mapping")
    verified = old_job["detail"].get("mapping_resolution") == "verified"
    if verified:
        if not new_mapping or values["detail"]["mapping_resolution"] != "verified":
            raise OperationConflict("same-version projection mapping regression; enrichment required")
        if any(old_mapping.get(key) != new_mapping.get(key) for key in ("unit_id", "kind", "record_key")):
            raise OperationConflict("conflicting verified mapping for the same source version; projection remains quarantined")
    if old_mapping and not _retains_evidence(old_mapping, new_mapping):
        if verified or values["detail"]["mapping_resolution"] != "verified":
            raise OperationConflict("same-version mapping evidence regression; enrichment required")
    if not _retains_evidence(previous.get("doc_status_evidence", []), proposed.get("doc_status_evidence", [])):
        raise OperationConflict("same-version document-status evidence regression; enrichment required")
    reasons = set(previous["dispositions"])
    if "matched" in proposed["dispositions"]:
        reasons.discard("ledger_only")
    if new_mapping:
        reasons.discard("unmatched_unit")
    if not reasons <= set(proposed["dispositions"]):
        raise OperationConflict("same-version disposition evidence regression; enrichment required")


class LegacyImportRepository:
    """Only owns migration history and ``distill_legacy_unverified`` projections."""
    def __init__(self, repository):
        self.repository = repository
        self.knowledge = KnowledgeRepository(repository)

    def ensure_schema(self, *, audit):
        with self.repository._write(audit):
            ddl = files("papervault.store").joinpath("knowledge_migration_schema.sql").read_text()
            ddl = render_catalog_ddl(ddl, self.repository.schema)
            self.repository.connection.execute(ddl)
            self.repository._finish(None, audit)

    def source_versions(self, source):
        return self.knowledge._rows("knowledge_import_events", source=source)

    def _projection_values(self, event, identity):
        mapping = event.get("mapping") or {}
        unit = self.knowledge.get_unit(mapping.get("unit_id")) if mapping else None
        compatible = bool(unit and unit["kind"] == mapping.get("kind")
                          and unit["record_key"] == mapping.get("record_key"))
        detail = {"migration_owner": identity, "source_version": event["source_version"],
                  "legacy_consecutive_failures": event["legacy_consecutive_failures"],
                  "counter_semantics": "consecutive_build_failures", "evidence": event,
                  "mapping_resolution": "verified" if compatible else "missing_or_conflicting_catalog_unit",
                  "excluded": True, "deleted": event["deleted"], "executable": False}
        return {"kind": "distill_legacy_unverified", "key": unit["record_key"] if compatible else None,
                "unit_id": unit["unit_id"] if compatible else None, "state": "quarantined", "attempts": 0,
                "achieved_tier": None, "quarantine": True, "detail": detail,
                "quarantine_reason": {"reasons": event["dispositions"], "legacy_unverified": True}}

    def _preflight(self, plan):
        """Validate the complete proposed projection under the catalog writer lock."""
        k = self.knowledge
        checkpoint_id = {"source": plan["source"], "capture_id": plan["capture_id"]}
        checkpoint = k._one("knowledge_import_checkpoints", **checkpoint_id)
        latest, seen = {}, {}
        for event in plan["events"]:
            if any(event[field] != checkpoint_id[field] for field in checkpoint_id):
                raise OperationConflict("source event does not belong to the proposed plan")
            identity = {**checkpoint_id, "source_row_id": event["source_row_id"]}
            version = {**identity, "source_version": event["source_version"]}
            event_digest = payload_hash({key: event[key] for key in (
                "source", "capture_id", "source_row_id", "source_version", "operation", "raw", "deleted")})
            if event_digest != event["event_digest"] or source_row_id(event["raw"]) != event["source_row_id"]:
                raise OperationConflict("source event has conflicting identity/evidence")
            stored = k._one("knowledge_import_events", **version)
            version_key = (event["source_row_id"], event["source_version"])
            if ((stored and stored["event_digest"] != event_digest)
                    or (version_key in seen and seen[version_key] != event_digest)):
                raise OperationConflict("source row/version has conflicting evidence")
            seen[version_key] = event_digest
            old = latest.get(event["source_row_id"])
            if old is None or old["source_version"] < event["source_version"]:
                latest[event["source_row_id"]] = event
            elif old["source_version"] == event["source_version"] and old != event:
                raise OperationConflict("source row/version has conflicting projection evidence")
        last = plan["checkpoint"].get("last_sequence")
        previous_last = checkpoint["checkpoint"].get("last_sequence") if checkpoint else None
        advance = checkpoint is None or previous_last is None or (last is not None and last >= previous_last)
        if checkpoint and last == previous_last:
            previous_checkpoint = deepcopy(checkpoint["checkpoint"])
            proposed_checkpoint = deepcopy(plan["checkpoint"])
            if previous_checkpoint.get("external_attempts") == "unknown":
                previous_checkpoint.pop("external_attempts", None)
            if not _retains_evidence(previous_checkpoint, proposed_checkpoint):
                raise OperationConflict("same-checkpoint evidence regression; enrichment required")
            previous_report = k._one("knowledge_import_reports", plan_digest=checkpoint["plan_digest"])["report"]
            for key in ("docstatus_only", "duplicates"):
                # Ledger-row dispositions are checked against their accepted projections below.
                previous_orphans = [row for row in previous_report["sets"][key] if "source_row_id" not in row]
                proposed_orphans = [row for row in plan["sets"][key] if "source_row_id" not in row]
                if not _retains_evidence(previous_orphans, proposed_orphans):
                    raise OperationConflict("same-checkpoint orphan evidence regression; enrichment required")
        heads = k._rows("knowledge_import_heads", **checkpoint_id)
        if advance and any(head["source_row_id"] not in latest or
                           latest[head["source_row_id"]]["source_version"] < head["source_version"] for head in heads):
            raise OperationConflict("proposed checkpoint loses accepted source projection evidence")
        projections = []
        for event in latest.values():
            identity = {**checkpoint_id, "source_row_id": event["source_row_id"]}
            head = k._one("knowledge_import_heads", **identity)
            if head and head["source_version"] > event["source_version"]:
                continue
            job_id = "legacy-unverified:" + payload_hash(identity)
            old_job = k._one("jobs", id=job_id)
            values = self._projection_values(event, identity)
            if old_job:
                if old_job["kind"] != "distill_legacy_unverified" or old_job["detail"].get("migration_owner") != identity:
                    raise OperationConflict("import refuses to overwrite catalog-owned job")
                if head is None or old_job["detail"].get("source_version") != head["source_version"]:
                    raise OperationConflict("legacy projection source version changed independently")
                validate_projection_unchanged(head, old_job)
                if head.get("plan_digest"):
                    lineage = k._one("knowledge_import_projections", **identity, revision=head["projection_revision"])
                    if (not lineage or any(lineage[field] != head[field] for field in (
                            "source_version", "plan_digest", "projection_digest"))
                            or lineage["evidence"] != old_job["detail"]["evidence"]):
                        raise OperationConflict("legacy projection accepted lineage changed independently")
                if head["source_version"] == event["source_version"]:
                    _validate_enrichment(old_job["detail"]["evidence"], event, old_job, values)
            elif head:
                raise OperationConflict("legacy projection disappeared independently of the importer")
            # A row can keep its source version while another row advances the
            # accepted source checkpoint. Its projection must still match that
            # checkpoint's complete report; an older report cannot enrich it.
            if not advance and (not old_job or any(old_job[field] != value for field, value in values.items())):
                raise OperationConflict("older source checkpoint cannot change an accepted projection")
            projections.append((identity, head, old_job, values))
        return checkpoint, advance, projections

    def apply(self, plan, *, operation_id):
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("stable operation_id required")
        digest = payload_hash(plan)
        audit = AuditEvent("knowledge-import", "legacy_import", {"operation_id": operation_id, "plan_digest": digest})
        k = self.knowledge
        with self.repository._write(audit):
            previous = k._one("knowledge_migration_operations", operation_id=operation_id)
            if previous:
                if previous["payload_digest"] != digest:
                    raise OperationConflict("migration operation ID payload conflict")
                return previous["result"]
            state = k.get_authority_state()
            if state["authority"] != "legacy":
                raise ValueError("legacy projection imports are fenced after catalog handoff")
            checkpoint, advance, projections = self._preflight(plan)
            if not k._one("knowledge_import_reports", plan_digest=digest):
                k._insert("knowledge_import_reports", {"plan_digest": digest, "source": plan["source"],
                    "capture_id": plan["capture_id"], "report": plan})
            inserted, projected = 0, 0
            for event in plan["events"]:
                version = {field: event[field] for field in ("source", "capture_id", "source_row_id", "source_version")}
                if not k._one("knowledge_import_events", **version):
                    k._insert("knowledge_import_events", {**version, "event_digest": event["event_digest"], "evidence": event})
                    inserted += 1
            for identity, head, old_job, values in projections:
                changed = not old_job or any(old_job[field] != value for field, value in values.items())
                if not changed and head.get("plan_digest"):
                    continue
                job_id = "legacy-unverified:" + payload_hash(identity)
                if changed:
                    projection = k._update("jobs", values, id=job_id) if old_job else k._insert("jobs", {"id": job_id, **values})
                    projected += 1
                else:
                    projection = old_job
                event = values["detail"]["evidence"]
                revision = (head.get("projection_revision") or 0) + 1 if head else 1
                projection_digest = payload_hash(k._json(projection))
                k._insert("knowledge_import_projections", {**identity, "revision": revision,
                    "source_version": event["source_version"], "plan_digest": digest,
                    "previous_plan_digest": head.get("plan_digest") if head else None,
                    "projection_digest": projection_digest, "evidence": event})
                head_values = {"source_version": event["source_version"], "job_id": job_id,
                    "deleted": event["deleted"], "event_digest": event["event_digest"],
                    "projection_digest": projection_digest, "plan_digest": digest, "projection_revision": revision}
                if head:
                    k._update("knowledge_import_heads", head_values, **identity)
                else:
                    k._insert("knowledge_import_heads", {**identity, **head_values})
            if advance and (not checkpoint or checkpoint["plan_digest"] != digest):
                values = {"checkpoint": plan["checkpoint"], "plan_digest": digest,
                          "revision": checkpoint["revision"] + 1 if checkpoint else 1}
                checkpoint_id = {"source": plan["source"], "capture_id": plan["capture_id"]}
                if checkpoint:
                    k._update("knowledge_import_checkpoints", values, **checkpoint_id)
                else:
                    k._insert("knowledge_import_checkpoints", {**checkpoint_id, **values})
            result = {"inserted": inserted, "projected": projected, "checkpoint": plan["checkpoint"]}
            k._insert("knowledge_migration_operations", {"operation_id": operation_id, "payload_digest": digest, "result": result})
            self.repository._finish(None, audit)
            return result
