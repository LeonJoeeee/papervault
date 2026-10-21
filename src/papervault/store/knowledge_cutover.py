"""Explicit durable ledger capture and forward-only catalog authority handoff.

Source administration and catalog commits are separate operations. Cross-store
backend fences require operator evidence; a configuration flag is never a fence.
"""
from __future__ import annotations

import uuid

from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from .knowledge import KnowledgeRepository, OperationConflict
from .knowledge_import import _json, source_row_id, validate_projection_unchanged
from .repository import AuditEvent, payload_hash

DRAIN_FIELDS = ("dispatch_stopped", "pipeline_admission_stopped", "inflight_drained",
                "backend_writes_fenced", "internal_workers_stopped")
VALIDATION_FIELDS = ("dispositions", "references", "quarantine", "accounting", "external_attempts")


def validate_rehearsal_dsn(dsn):
    database = conninfo_to_dict(dsn).get("dbname")
    if not database:
        raise ValueError("rehearsal DSN requires explicit database")
    if database.lower() in {"papervault_catalog", "knowledge_system"}:
        raise ValueError("rehearsal refuses live database")


def _validate_drain(drain):
    if any(drain.get(field) is not True for field in DRAIN_FIELDS) or not drain.get("receipt"):
        raise ValueError("handoff requires drained dispatch/admission and backend/internal-worker fence receipts")


def validate_handoff_evidence(checkpoint, evidence):
    if not checkpoint.get("handoff_eligible") or not checkpoint.get("capture_id"):
        raise ValueError("handoff requires durable capture checkpoint")
    for key in ("source", "capture_id"):
        if checkpoint.get(key) != evidence.get(key):
            raise ValueError("handoff source/capture mismatch")
    final = checkpoint.get("last_sequence")
    if type(final) is not int or final != evidence.get("final_sequence"):
        raise ValueError("handoff final delta checkpoint mismatch")
    fence = evidence.get("source_fence") or {}
    checks = fence.get("permission_checks") or []
    roles = fence.get("revoked_roles") or []
    if (fence.get("capture_id") != checkpoint["capture_id"] or fence.get("final_sequence") != final
            or fence.get("fenced") is not True or fence.get("legacy_dml_revoked") is not True
            or not fence.get("fence_id") or not roles or not checks
            or fence.get("final_row_count") != checkpoint.get("final_row_count")
            or not checkpoint.get("final_rows_digest")
            or fence.get("final_rows_digest") != checkpoint.get("final_rows_digest")
            or {check.get("role") for check in checks} != set(roles)
            or any(check.get("can_write") is not False for check in checks)):
        raise ValueError("legacy writes must be durably fenced and DML privileges verified revoked")
    _validate_drain(evidence.get("drain") or {})
    validation = evidence.get("validation") or {}
    if any(validation.get(field) is not True for field in VALIDATION_FIELDS) or not validation.get("receipt"):
        raise ValueError("key-level dispositions/references/quarantine/accounting/external-attempt validation required")


def validate_legacy_projection(head, job, event, *, expected_evidence=None):
    """Reject source/reference/counter drift before changing authority."""
    validate_projection_unchanged(head, job)
    source_evidence = event["evidence"] if event else {}
    expected = expected_evidence if expected_evidence is not None else source_evidence
    if (not job or not event or job["kind"] != "distill_legacy_unverified" or job["state"] != "quarantined"
            or not job["quarantine"] or job["detail"].get("source_version") != head["source_version"]
            or not job["detail"].get("excluded") or job.get("achieved_tier") is not None
            or job["attempts"] != 0 or job.get("consumed_input_revision") is not None
            or job.get("recipe_digest") is not None or job.get("target_generation") is not None
            or job["detail"].get("deleted") != head["deleted"]
            or head["event_digest"] != event["event_digest"]
            or job["detail"].get("evidence") != expected
            or any(expected.get(field) != source_evidence.get(field) for field in (
                "source", "capture_id", "source_row_id", "source_version", "operation", "raw", "deleted", "event_digest"))
            or job["detail"].get("legacy_consecutive_failures") != event["evidence"]["raw"].get("attempts")):
        raise ValueError("legacy projection source/disposition/quarantine/counter validation failed")


class SourceCapture:
    """Administrative capture of one explicit source ledger.

    A BEFORE STATEMENT counter-row lock serializes source transactions *before*
    their first row changes. Journal sequence increments are transactional: an
    abort leaves neither a sequence gap nor an uncommitted reservation. Snapshot
    takes a SHARE ledger lock before its first snapshot read. No runtime path
    installs these triggers implicitly.
    """
    def __init__(self, connection, *, ledger_schema, source, configuration, allow_production=False):
        if not connection.autocommit:
            raise ValueError("capture requires explicit autocommit connection")
        if not allow_production:
            validate_rehearsal_dsn(connection.info.dsn)
        self.connection = connection
        self.connection.row_factory = dict_row
        self.schema = ledger_schema
        self.source = source
        self.configuration = configuration

    def _table(self, name):
        return sql.Identifier(self.schema, name)

    @property
    def ledger(self):
        return self._table("ks_ledger")

    def _state(self):
        return self.connection.execute(sql.SQL("SELECT * FROM {} WHERE singleton").format(
            self._table("papervault_ledger_capture_state"))).fetchone()

    def install(self):
        from psycopg.types.json import Jsonb
        with self.connection.transaction():
            self.connection.execute(sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(self.ledger))
            self.connection.execute(sql.SQL("""CREATE TABLE IF NOT EXISTS {} (
                singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK(singleton), capture_id TEXT NOT NULL,
                source TEXT NOT NULL, configuration JSONB NOT NULL, sequence BIGINT NOT NULL DEFAULT 0,
                fenced BOOLEAN NOT NULL DEFAULT FALSE, fence_id TEXT)
            """).format(self._table("papervault_ledger_capture_state")))
            state = self._state()
            if state and (state["source"] != self.source or state["configuration"] != self.configuration):
                raise ValueError("existing capture source/configuration differs")
            self.connection.execute(sql.SQL("INSERT INTO {} (singleton,capture_id,source,configuration) "
                "VALUES(TRUE,%s,%s,%s) ON CONFLICT DO NOTHING").format(
                    self._table("papervault_ledger_capture_state")),
                (str(uuid.uuid4()), self.source, Jsonb(self.configuration)))
            self.connection.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} (sequence BIGINT PRIMARY KEY,"
                "capture_id TEXT NOT NULL, operation TEXT NOT NULL,old_row JSONB,new_row JSONB)").format(
                    self._table("papervault_ledger_capture_journal")))
            self.connection.execute(sql.SQL("""CREATE OR REPLACE FUNCTION {}() RETURNS trigger
                LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $capture$
                DECLARE blocked BOOLEAN;
                BEGIN
                  SELECT fenced INTO blocked FROM {} WHERE singleton FOR UPDATE;
                  IF blocked THEN RAISE EXCEPTION 'legacy ledger writer fenced'; END IF;
                  RETURN NULL;
                END $capture$
            """).format(self._table("papervault_ledger_capture_lock"), self._table("papervault_ledger_capture_state")))
            self.connection.execute(sql.SQL("""CREATE OR REPLACE FUNCTION {}() RETURNS trigger
                LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $capture$
                DECLARE n BIGINT; capture TEXT;
                BEGIN
                  UPDATE {} SET sequence=sequence+1 WHERE singleton RETURNING sequence,capture_id INTO n,capture;
                  INSERT INTO {}(sequence,capture_id,operation,old_row,new_row)
                  VALUES(n,capture,TG_OP,CASE WHEN TG_OP='INSERT' THEN NULL ELSE to_jsonb(OLD) END,
                         CASE WHEN TG_OP='DELETE' THEN NULL ELSE to_jsonb(NEW) END);
                  IF TG_OP='DELETE' THEN RETURN OLD; END IF;
                  RETURN NEW;
                END $capture$
            """).format(self._table("papervault_ledger_capture_row"),
                          self._table("papervault_ledger_capture_state"), self._table("papervault_ledger_capture_journal")))
            self.connection.execute(sql.SQL("""CREATE OR REPLACE FUNCTION {}() RETURNS trigger
                LANGUAGE plpgsql AS $capture$ BEGIN
                RAISE EXCEPTION 'TRUNCATE and capture-history mutation are forbidden'; END $capture$
            """).format(self._table("papervault_ledger_capture_reject")))
            triggers = [
                ("papervault_capture_statement", "BEFORE INSERT OR UPDATE OR DELETE", self.ledger, "STATEMENT", "papervault_ledger_capture_lock"),
                ("papervault_capture_row", "AFTER INSERT OR UPDATE OR DELETE", self.ledger, "ROW", "papervault_ledger_capture_row"),
                ("papervault_capture_truncate", "BEFORE TRUNCATE", self.ledger, "STATEMENT", "papervault_ledger_capture_reject"),
                ("papervault_capture_immutable", "BEFORE UPDATE OR DELETE OR TRUNCATE", self._table("papervault_ledger_capture_journal"), "STATEMENT", "papervault_ledger_capture_reject"),
            ]
            for name, event, table, granularity, function in triggers:
                exists = self.connection.execute("SELECT 1 FROM pg_trigger WHERE tgrelid=%s::regclass AND tgname=%s",
                                                  (table.as_string(self.connection), name)).fetchone()
                if not exists:
                    self.connection.execute(sql.SQL("CREATE TRIGGER {} {} ON {} FOR EACH {} EXECUTE FUNCTION {}()").format(
                        sql.Identifier(name), sql.SQL(event), table, sql.SQL(granularity), self._table(function)))
            for table in ("papervault_ledger_capture_state", "papervault_ledger_capture_journal"):
                self.connection.execute(sql.SQL("REVOKE ALL ON {} FROM PUBLIC").format(self._table(table)))
            return _json(self._state())

    def snapshot(self):
        with self.connection.transaction():
            # READ COMMITTED is intentional: acquiring the lock must precede the
            # first snapshot read, including the capture counter and ledger rows.
            self.connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            self.connection.execute(sql.SQL("LOCK TABLE {} IN SHARE MODE").format(self.ledger))
            state = self._state()
            rows = list(self.connection.execute(sql.SQL("SELECT * FROM {} ORDER BY workspace,ingest_source,source_id").format(self.ledger)))
            return {"source": self.source, "capture_id": state["capture_id"], "checkpoint": state["sequence"],
                    "rows": _json(rows), "configuration": state["configuration"],
                    "schema": self._schema(), "external_attempts": "unknown"}

    def _schema(self):
        columns = list(self.connection.execute("SELECT column_name,data_type,is_nullable FROM information_schema.columns "
            "WHERE table_schema=%s AND table_name='ks_ledger' ORDER BY ordinal_position", (self.schema,)))
        return {"table": self.ledger.as_string(self.connection), "columns": columns}

    def readonly_snapshot(self):
        """Dry-run snapshot: no trigger installation, DML, or handoff assertion."""
        with self.connection.transaction():
            self.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            rows = list(self.connection.execute(sql.SQL("SELECT * FROM {} ORDER BY workspace,ingest_source,source_id").format(self.ledger)))
            return {"source": self.source, "capture_id": None, "checkpoint": None, "rows": _json(rows),
                    "configuration": self.configuration, "schema": self._schema(), "external_attempts": "unknown"}

    def delta(self, after):
        if type(after) is not int or after < 0:
            raise ValueError("delta requires nonnegative capture sequence")
        with self.connection.transaction():
            self.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            state = self._state()
            rows = list(self.connection.execute(sql.SQL("SELECT * FROM {} WHERE sequence>%s AND sequence<=%s ORDER BY sequence").format(
                self._table("papervault_ledger_capture_journal")), (after, state["sequence"])))
            if [row["sequence"] for row in rows] != list(range(after + 1, state["sequence"] + 1)):
                raise ValueError("capture journal is not contiguous")
            return {"capture_id": state["capture_id"], "final_checkpoint": state["sequence"], "changes": [
                {"sequence": row["sequence"], "operation": row["operation"], "old": row["old_row"], "new": row["new_row"]} for row in rows]}

    def freeze(self, *, legacy_roles, drain):
        """Explicit final source fence; no running work is inferred from process flags."""
        _validate_drain(drain)
        if not legacy_roles:
            raise ValueError("explicit legacy writer roles required")
        with self.connection.transaction():
            self.connection.execute(sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(self.ledger))
            for role in ["PUBLIC", *legacy_roles]:
                self.connection.execute(sql.SQL("REVOKE INSERT,UPDATE,DELETE,TRUNCATE ON {} FROM {}").format(
                    self.ledger, sql.SQL("PUBLIC") if role == "PUBLIC" else sql.Identifier(role)))
            columns = list(self.connection.execute("SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass "
                "AND attnum>0 AND NOT attisdropped ORDER BY attnum", (self.ledger.as_string(self.connection),)))
            column_names = sql.SQL(",").join(sql.Identifier(column["attname"]) for column in columns)
            for role in ["PUBLIC", *legacy_roles]:
                self.connection.execute(sql.SQL("REVOKE INSERT({}),UPDATE({}) ON {} FROM {}").format(
                    column_names, column_names, self.ledger,
                    sql.SQL("PUBLIC") if role == "PUBLIC" else sql.Identifier(role)))
            checks = []
            for role in legacy_roles:
                result = self.connection.execute("SELECT has_table_privilege(%s,%s,'INSERT,UPDATE,DELETE,TRUNCATE') "
                    "OR has_any_column_privilege(%s,%s,'INSERT,UPDATE') AS can_write",
                    (role, self.ledger.as_string(self.connection), role, self.ledger.as_string(self.connection))).fetchone()
                if result["can_write"]:
                    raise ValueError("legacy role retains inherited/owner/superuser DML privilege")
                checks.append({"role": role, "can_write": False})
            fence_id = str(uuid.uuid4())
            self.connection.execute(sql.SQL("UPDATE {} SET fenced=TRUE,fence_id=COALESCE(fence_id,%s) WHERE singleton").format(
                self._table("papervault_ledger_capture_state")), (fence_id,))
            state = self._state()
            rows = list(self.connection.execute(sql.SQL("SELECT * FROM {}").format(self.ledger)))
            rows = sorted(_json(rows), key=source_row_id)
            return {"capture_id": state["capture_id"], "final_sequence": state["sequence"], "fenced": True,
                    "final_row_count": len(rows), "final_rows_digest": payload_hash(rows),
                    "legacy_dml_revoked": True, "fence_id": state["fence_id"],
                    "revoked_roles": list(legacy_roles), "permission_checks": checks, "drain": drain}


class CutoverRepository:
    def __init__(self, repository):
        self.repository = repository
        self.knowledge = KnowledgeRepository(repository)

    def commit_handoff(self, evidence, *, expected_epoch, operation_id):
        if not operation_id:
            raise ValueError("stable handoff operation_id required")
        digest = payload_hash({"evidence": evidence, "expected_epoch": expected_epoch})
        audit = AuditEvent("knowledge-cutover", "knowledge_authority_handoff", {"operation_id": operation_id, "evidence_digest": digest})
        k = self.knowledge
        with self.repository._write(audit):
            old = k._one("knowledge_migration_operations", operation_id=operation_id)
            if old:
                if old["payload_digest"] != digest:
                    raise OperationConflict("handoff operation payload conflict")
                return old["result"]
            state = k.get_authority_state()
            if state["authority"] != "legacy" or state["authority_epoch"] != expected_epoch:
                raise ValueError("authority/epoch changed; handoff is forward-only")
            active = k._rows("generations", state="active")
            if len(active) != 1 or not k.get_manifest(active[0]["generation_id"]):
                raise ValueError("handoff requires an initialized active generation")
            checkpoint = k._one("knowledge_import_checkpoints", source=evidence["source"], capture_id=evidence["capture_id"])
            if not checkpoint:
                raise ValueError("no imported final capture checkpoint")
            validate_handoff_evidence(checkpoint["checkpoint"], evidence)
            report = k._one("knowledge_import_reports", plan_digest=checkpoint["plan_digest"])
            if not report or report["source"] != evidence["source"] or report["capture_id"] != evidence["capture_id"]:
                raise ValueError("final keyed disposition report missing")
            expected_events = {}
            for event in report["report"]["events"]:
                previous = expected_events.get(event["source_row_id"])
                if previous is None or event["source_version"] > previous["source_version"]:
                    expected_events[event["source_row_id"]] = event
            imported = k._rows("knowledge_import_events", source=evidence["source"], capture_id=evidence["capture_id"])
            first, last = checkpoint["checkpoint"]["snapshot_sequence"], evidence["final_sequence"]
            versions = {event["source_version"] for event in imported if event["source_version"] > first}
            if versions != set(range(first + 1, last + 1)):
                raise ValueError("imported final capture sequence is not contiguous")
            current_rows = []
            for head in k._rows("knowledge_import_heads", source=evidence["source"], capture_id=evidence["capture_id"]):
                job = k._one("jobs", id=head["job_id"])
                event = next((event for event in imported if event["source_row_id"] == head["source_row_id"]
                              and event["source_version"] == head["source_version"]), None)
                expected = expected_events.pop(head["source_row_id"], None)
                if expected is None:
                    raise ValueError("legacy projection absent from final report")
                validate_legacy_projection(head, job, event, expected_evidence=expected)
                mapping = expected.get("mapping") or {}
                unit = k.get_unit(mapping.get("unit_id")) if mapping else None
                compatible = bool(unit and unit["kind"] == mapping.get("kind")
                                  and unit["record_key"] == mapping.get("record_key"))
                if (job["unit_id"] != (unit["unit_id"] if compatible else None)
                        or job["key"] != (unit["record_key"] if compatible else None)
                        or job["detail"].get("mapping_resolution") != (
                            "verified" if compatible else "missing_or_conflicting_catalog_unit")):
                    raise ValueError("legacy projection mapping is not resolved against current catalog units")
                if not head["deleted"]:
                    current_rows.append(event["evidence"]["raw"])
            if (expected_events or len(current_rows) != checkpoint["checkpoint"]["final_row_count"]
                    or payload_hash(sorted(current_rows, key=source_row_id)) != checkpoint["checkpoint"]["final_rows_digest"]):
                raise ValueError("final source row set differs from imported projection")
            if any(not event["evidence"].get("quarantine") or not event["evidence"].get("excluded")
                   or not event["evidence"].get("dispositions") for event in imported):
                raise ValueError("source evidence has unrecorded or conforming legacy dispositions")
            if k.list_jobs(states=["running"]):
                raise ValueError("catalog attempts remain running at handoff")
            registered = {"evidence_id": "handoff:" + operation_id, "source": evidence["source"],
                          "capture_id": evidence["capture_id"], "final_sequence": last, "evidence": evidence, "digest": digest}
            k._insert("knowledge_cutover_evidence", registered)
            checkpoint_value = {"migration": checkpoint["checkpoint"], "evidence_id": registered["evidence_id"],
                                "evidence_digest": digest, "validation": evidence["validation"]}
            result = k._update("knowledge_runtime", {"authority": "catalog", "authority_epoch": expected_epoch + 1,
                                                       "checkpoint": checkpoint_value}, singleton=True)
            result = k._json(result)
            k._insert("knowledge_migration_operations", {"operation_id": operation_id, "payload_digest": digest, "result": result})
            self.repository._finish(None, audit)
            return result

    def rollback(self, *, authority="catalog"):
        """Software rollback retains authority, audit, outbox and attempted effects."""
        state = self.knowledge.get_authority_state()
        if authority != "catalog" or state["authority"] != "catalog":
            raise ValueError("rollback must retain catalog authority; reverse migration requires separate rehearsal")
        return state
