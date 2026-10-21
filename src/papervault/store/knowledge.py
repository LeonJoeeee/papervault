"""Catalog-owned knowledge commands and immutable execution identities.

Construction is read-only. Schema installation is an explicit administrative
repository operation. External adapters own evidence collection; this module
checks its exact admitted identity and the declared output contract.
"""
from __future__ import annotations

import asyncio
import copy
import re
from contextlib import asynccontextmanager
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from psycopg.conninfo import conninfo_to_dict

from .repository import AuditEvent, PostgresCatalogRepository, canonical_json, payload_hash


class OperationConflict(ValueError):
    """An operation ID was reused for a different command."""


class ClaimRejected(RuntimeError):
    """Policy, revision, generation, budget or fencing forbids this transition."""


class KnownExecutionFailure(RuntimeError):
    """The adapter has definitive evidence this execution cannot succeed."""


class RetirementFailure(RuntimeError):
    """An inherited contribution could not be verifiably removed/rebuilt."""

    def __init__(self, retired_receipt_id, doc_id, status, messages=(), evidence=None):
        self.retired_receipt_id = retired_receipt_id
        self.doc_id = doc_id
        self.status = status
        self.messages = tuple(messages)
        self.evidence = evidence or {}
        super().__init__("retirement rebuild unverified: " + str(status))


# Token counts/sizes are execution identity; only credential token names are
# secrets. Generic suffix `_token` covers provider names such as `hf_token`.
_SECRET = re.compile(r"(^|_)(api_?key|password|passwd|secret|credentials?|authorization)(_|$)"
                     r"|(^|_)token$|(^|_)(auth|api|access|refresh|bearer|session|id)_tokens$", re.I)


def _secret_key(key):
    normalized = re.sub(r"([a-z])([A-Z])", r"\1_\2", key).replace("-", "_")
    return bool(_SECRET.search(normalized))


def canonical_recipe(recipe: dict) -> tuple[dict, str]:
    """Return a secret-free effective specification and its stable digest."""
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if not _secret_key(key)}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str) and "://" in value:
            parts = urlsplit(value)
            host = parts.hostname or ""
            if ":" in host:
                host = "[" + host + "]"
            if parts.port:
                host += ":" + str(parts.port)
            query = urlencode([(key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True)
                               if not _secret_key(key)])
            return urlunsplit((parts.scheme, host, parts.path, query, ""))
        return value
    if not isinstance(recipe, dict) or type(recipe.get("schema_version")) is not int:
        raise ValueError("recipe requires an integer schema_version")
    spec = clean(copy.deepcopy(recipe))
    return spec, payload_hash(spec)


VECTOR_OUTPUT_ROLES = frozenset({"vectors", "chunks_vdb", "entities_vdb", "relationships_vdb"})


def requires_vectors(recipe: dict, tier: str) -> bool:
    return bool(VECTOR_OUTPUT_ROLES.intersection(recipe.get("output_contract", {}).get(tier, {}).get("required", [])))


def retrieval_contract(recipe: dict) -> dict | None:
    """Return the exact embedding space identity, or None for nonvector recipes.

    Dimension alone cannot identify a vector space. A vector recipe must freeze
    its model, representation, implementation and dependency versions explicitly.
    """
    if not any(requires_vectors(recipe, tier) for tier in recipe.get("output_contract", {})):
        return None
    contract = recipe.get("embedding_contract")
    if not isinstance(contract, dict):
        raise ValueError("vector recipe requires an explicit embedding contract")
    valid = (type(contract.get("schema_version")) is int and contract["schema_version"] == 1
        and all(isinstance(contract.get(key), str) and contract[key].strip() for key in ("model", "representation"))
        and all(type(contract.get(key)) is int and contract[key] > 0 for key in ("dimension", "max_length"))
        and type(contract.get("use_fp16")) is bool
        and isinstance(contract.get("implementation_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", contract["implementation_sha256"])
        and isinstance(contract.get("packages"), dict) and bool(contract["packages"])
        and all(isinstance(key, str) and key and isinstance(value, str) and value
                for key, value in contract["packages"].items()))
    if not valid:
        raise ValueError("embedding contract requires exact model, dimensions, representation, implementation and packages")
    canonical_json(contract)
    return copy.deepcopy(contract)


def validate_input_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict) or manifest.get("metadata_revision") is None:
        raise ValueError("input manifest requires exact metadata_revision")
    if not isinstance(manifest.get("artifacts"), list):
        raise ValueError("input manifest requires artifact inventory")
    seen = set()
    for artifact in manifest["artifacts"]:
        if (not isinstance(artifact, dict) or not artifact.get("artifact_id")
                or not re.fullmatch(r"[0-9a-f]{64}", artifact.get("sha256", ""))
                or type(artifact.get("byte_length")) is not int or artifact["byte_length"] < 0):
            raise ValueError("artifact requires ID, SHA256 and precise nonnegative byte_length")
        if artifact["artifact_id"] in seen:
            raise ValueError("duplicate input artifact")
        seen.add(artifact["artifact_id"])
    if not isinstance(manifest.get("upstream_derivation_ids", []), list):
        raise ValueError("upstream derivations must be explicit identifiers")
    canonical_json(manifest)


CLAIM_FIELDS = ("job_id", "attempt_id", "submission_id", "authority_epoch", "fencing_token",
                "unit_id", "consumed_input_revision", "recipe_digest", "generation_id", "namespace")
OBSERVATION_IDENTITY_FIELDS = ("unit_id", "consumed_input_revision", "recipe_digest", "generation_id", "attempt_id",
                              "submission_id", "authority_epoch", "fencing_token", "receipt_id", "namespace", "role", "store", "output_id")


def validate_receipt(receipt: dict, claim: dict, recipe: dict) -> None:
    for key in CLAIM_FIELDS:
        if receipt.get(key) != claim.get(key) or receipt.get(key) is None:
            raise ValueError(f"receipt identity mismatch: {key}")
    if not receipt.get("receipt_id") or receipt.get("sealed") is not True or receipt.get("durable") is not True:
        raise ValueError("receipt must be sealed and durable")
    if "aggregate_base_manifest_revision" in claim:
        for key in ("aggregate_base_manifest_revision", "base_aggregate_layers", "base_selected_unit_receipts", "retired_receipt_ids"):
            if receipt.get(key) != claim[key]:
                raise ValueError("receipt aggregate base differs: " + key)
        layers = claim["base_aggregate_layers"] + [claim["namespace"]]
        selected = {**claim["base_selected_unit_receipts"], claim["unit_id"]: receipt["receipt_id"]}
        if (receipt.get("aggregate_layers") != layers or receipt.get("selected_unit_receipts") != selected
                or receipt.get("selected_unit_receipts_digest") != payload_hash(selected)):
            raise ValueError("receipt aggregate result does not match admitted base and selected units")
    tier = receipt.get("achieved_tier")
    if tier not in {"fulltext", "abstract", "metadata"}:
        raise ValueError("receipt requires known achieved_tier")
    contract = recipe.get("output_contract", {}).get(tier, {})
    required = contract.get("required")
    if not isinstance(required, list) or not required:
        raise ValueError("recipe must explicitly declare required output roles")
    nonapplicable = contract.get("not_applicable", [])
    if set(required) & set(nonapplicable):
        raise ValueError("required roles cannot be nonapplicable")
    inventory = receipt.get("output_inventory")
    if not isinstance(inventory, list):
        raise ValueError("receipt requires output inventory")
    seen = {}
    for output in inventory:
        role = output.get("role")
        if not role or role in seen or output.get("namespace") != claim["namespace"]:
            raise ValueError("output identity is missing, duplicated or wrong namespace")
        verification = output.get("verification", {})
        state = verification.get("state")
        if state not in {"present", "verified_empty", "not_applicable"}:
            raise ValueError("unverified output")
        if not verification.get("method") or not verification.get("evidence"):
            raise ValueError("output verification needs method and evidence")
        if state == "not_applicable" and role not in nonapplicable:
            raise ValueError("undeclared nonapplicable role")
        if state == "verified_empty":
            evidence = verification["evidence"]
            counts = [v for k, v in evidence.items() if k == "count" or k.endswith("_count")]
            if not counts or any(type(v) is not int or v != 0 for v in counts):
                raise ValueError("empty output needs explicit verified zero counts")
        stamp = output.get("verified_at")
        if not stamp or datetime.fromisoformat(stamp.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("verification time must be timezone aware")
        if not output.get("output_id") or not output.get("store"):
            raise ValueError("output requires stable identity and store")
        digest, size = output.get("sha256"), output.get("byte_length")
        if digest is not None or size is not None:
            if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or type(size) is not int or size < 0):
                raise ValueError("byte outputs require exact hash and length together")
        seen[role] = state
    if any(seen.get(role) not in {"present", "verified_empty"} for role in required):
        raise ValueError("required output contract is incomplete")
    outputs = {output["role"]: output for output in inventory}
    if claim.get("retired_receipt_ids") and outputs.get("graph", {}).get("store") == "neo4j":
        expected = {"version": 1, "retired_receipt_ids": claim["retired_receipt_ids"],
                    "graph_sha256": outputs["graph"].get("sha256")}
        protocol = receipt.get("retirement_verification")
        if not isinstance(protocol, dict) or type(protocol.get("version")) is not int or protocol != expected:
            raise ValueError("retirement graph/vector verification missing or incompatible; isolated rebuild required")
    if {"full_entities", "full_relations"} & outputs.keys():
        _validate_document_index_verification(receipt, claim, outputs)
    canonical_json(receipt)


def _validate_document_index_verification(receipt, claim, outputs):
    """Keep pre-verification physical receipts from bypassing sealing on recovery."""
    protocol = receipt.get("document_index_verification")
    if (not isinstance(protocol, dict) or type(protocol.get("version")) is not int
            or protocol != {"version": 1, "retired_receipt_ids": claim.get("retired_receipt_ids", [])}):
        raise ValueError("document index verification protocol missing or incompatible; isolated rebuild required")
    # Metadata deltas attest verified retirement but have no new graph-bearing
    # document. Pure catalog metadata receipts do not contain backend indexes.
    if receipt["achieved_tier"] == "metadata":
        return
    graph = outputs.get("graph", {})
    proof = graph.get("verification", {}).get("evidence", {}).get("document_contribution")
    if not isinstance(proof, dict):
        raise ValueError("document index contribution evidence missing")
    doc_id = proof.get("doc_id")
    chunks = proof.get("chunk_ids")
    if (not isinstance(doc_id, str) or not doc_id
            or doc_id != receipt.get("provider_observations", {}).get("doc_id")
            or not isinstance(chunks, list) or not chunks
            or any(not isinstance(chunk, str) or not chunk for chunk in chunks)):
        raise ValueError("document index contribution identity is invalid")
    entities, pairs = proof.get("entity_ids"), proof.get("relation_pairs")
    if (not isinstance(entities, list) or any(not isinstance(entity, str) or not entity for entity in entities)
            or not isinstance(pairs, list) or any(not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(part, str) or not part for part in pair) for pair in pairs)):
        raise ValueError("document index contribution graph identities are invalid")
    for role, expected in (("full_entities", entities), ("full_relations", pairs)):
        index = outputs.get(role, {})
        if expected and (doc_id not in index.get("backend_ids", [])
                         or index.get("verification", {}).get("state") != "present"):
            raise ValueError("document index inventory does not cover recorded contribution")


def validate_test_dsn(dsn: str) -> None:
    """Refuse a production/default database before opening any connection."""
    database = conninfo_to_dict(dsn).get("dbname")
    if not database:
        raise ValueError("test DSN requires explicit database name")
    if database.lower() == "papervault_catalog":
        raise ValueError("test DSN refuses live papervault_catalog database")


class KnowledgeRepository:
    """Focused SQL commands; no mutation loads a complete catalog snapshot."""

    def __init__(self, repository: PostgresCatalogRepository):
        self.repository = repository
        self.connection = repository.connection
        self._namespace_locks = {}
        self._namespace_owners = {}

    def command(self, action: str, operation_id: str, payload: dict) -> dict:
        from contextlib import nullcontext
        from .knowledge_commands import dispatch
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("stable operation_id required")
        if not isinstance(payload, dict):
            raise ValueError("command payload must be an object")
        digest = payload_hash(payload)
        audit = AuditEvent("knowledge", action, {"operation_id": operation_id})
        # Namespace first, writer second: a sealing caller may already own the
        # namespace guard across external IO. Reversing these locks deadlocks.
        namespace = payload.get("claim", payload).get("namespace") if action in {
            "seal_namespace", "record_tombstone", "defer_attempt", "begin_retirement", "fail_retirement"} else None
        guard = self._session_namespace_lock(namespace) if namespace else nullcontext()
        with guard, self.repository._write(audit):
            old = self._one("knowledge_operations", operation_id=operation_id)
            if old:
                if old["action"] != action or old["payload_digest"] != digest:
                    raise OperationConflict("operation ID reused with conflicting payload")
                return old["result"]
            result = dispatch(self, action, operation_id, payload)
            result = self._json(result)
            self._insert("knowledge_operations", {"operation_id": operation_id, "action": action,
                         "payload_digest": digest, "result": result})
            self.repository._finish(payload.get("record_key"), audit)
            return result

    @staticmethod
    def _json(value):
        import json
        from decimal import Decimal
        return json.loads(json.dumps(value, default=lambda v: v.isoformat() if isinstance(v, datetime)
                                     else float(v) if isinstance(v, Decimal) else str(v)))

    def _one(self, table, **where):
        rows = self._rows(table, **where)
        return rows[0] if rows else None

    def _rows(self, table, **where):
        from psycopg import sql
        predicate = (sql.SQL(" AND ").join(sql.SQL("{} IS NOT DISTINCT FROM %s").format(sql.Identifier(k))
                                          for k in where) if where else sql.SQL("TRUE"))
        return list(self.connection.execute(sql.SQL("SELECT * FROM {} WHERE {}").format(
            self.repository._table(table), predicate), list(where.values())))

    def _insert(self, table, values):
        from psycopg import sql
        from psycopg.types.json import Jsonb
        return self.connection.execute(sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING *").format(
            self.repository._table(table), sql.SQL(",").join(map(sql.Identifier, values)),
            sql.SQL(",").join(sql.Placeholder() for _ in values)),
            [Jsonb(v) if isinstance(v, (dict, list)) or (table == "units" and key == "identity") else v
             for key, v in values.items()]).fetchone()

    def _update(self, table, values, **where):
        from psycopg import sql
        from psycopg.types.json import Jsonb
        return self.connection.execute(sql.SQL("UPDATE {} SET {} WHERE {} RETURNING *").format(
            self.repository._table(table), sql.SQL(",").join(sql.SQL("{}=%s").format(sql.Identifier(k))
                for k in values), sql.SQL(" AND ").join(sql.SQL("{}=%s").format(sql.Identifier(k))
                for k in where)), [*[Jsonb(v) if isinstance(v, (dict, list)) else v for v in values.values()],
                                   *where.values()]).fetchone()

    def get_unit(self, unit_id):
        return self._one("units", unit_id=unit_id)

    def get_unit_for_record(self, record_key):
        rows = [r for r in self._rows("units", record_key=record_key) if r["kind"] != "section"]
        if len(rows) > 1:
            raise ValueError("record has ambiguous whole-document units")
        return rows[0] if rows else None

    def resolve_scope_units(self, scope):
        from psycopg import sql
        if scope is None:
            return None
        keys = [scope] if isinstance(scope, str) else list(scope)
        rows = self.connection.execute(sql.SQL(
            "WITH requested AS (SELECT unit_id,record_key,kind FROM {units} WHERE unit_id=ANY(%s) OR record_key=ANY(%s)), "
            "parents AS (SELECT unit_id FROM requested WHERE kind<>'section' UNION SELECT parent_unit_id FROM {sections} "
            "WHERE unit_id IN (SELECT unit_id FROM requested)) SELECT unit_id FROM {units} "
            "WHERE unit_id IN(SELECT unit_id FROM requested UNION SELECT unit_id FROM parents UNION "
            "SELECT unit_id FROM {sections} WHERE parent_unit_id IN(SELECT unit_id FROM parents)) "
            "OR record_key IN(SELECT record_key FROM requested)"
        ).format(units=self.repository._table("units"), sections=self.repository._table("section_mappings")), (keys, keys))
        return [r["unit_id"] for r in rows]

    def list_registered_records(self, limit=1000, after=None):
        from psycopg import sql
        return list(self.connection.execute(sql.SQL("SELECT * FROM {} WHERE (%s::text IS NULL OR key>%s) "
            "ORDER BY key LIMIT %s").format(self.repository._table("records")), (after, after, limit)))

    def list_pending_input_records(self, limit=1000):
        from psycopg import sql
        return list(self.connection.execute(sql.SQL(
            "SELECT DISTINCT r.* FROM {} r JOIN {} p ON p.record_key=r.key AND p.record_revision=r.revision "
            "WHERE r.merge='canonical' AND NOT EXISTS(SELECT 1 FROM {} u JOIN {} q ON q.unit_id=u.unit_id "
            "JOIN {} g ON g.generation_id=q.generation_id AND g.state='active' "
            "WHERE u.record_key=r.key AND q.record_revision=r.revision AND NOT q.invalidated) "
            "ORDER BY r.key LIMIT %s"
        ).format(self.repository._table("records"), self.repository._table("knowledge_pending_inputs"),
            self.repository._table("units"), self.repository._table("knowledge_requirements"),
            self.repository._table("generations")), (limit,)))

    def get_active_generation(self):
        return self._one("generations", state="active")

    def get_job(self, job_id):
        row = self._one("jobs", id=job_id)
        if row:
            row = {**row, "job_id": row["id"], "generation_id": row["target_generation"]}
        return row

    def get_logical_job(self, unit_id, consumed_input_revision, recipe_digest, generation_id):
        row = self._one("jobs", kind="distill", unit_id=unit_id,
            consumed_input_revision=consumed_input_revision, recipe_digest=recipe_digest,
            target_generation=generation_id)
        return self.get_job(row["id"]) if row else None

    def list_jobs(self, states=None):
        from psycopg import sql
        query = sql.SQL("SELECT * FROM {} WHERE kind='distill'").format(self.repository._table("jobs"))
        params = []
        if states is not None:
            query += sql.SQL(" AND state=ANY(%s)")
            params.append(list(states))
        return [{**r, "job_id": r["id"], "generation_id": r["target_generation"]}
                for r in self.connection.execute(query + sql.SQL(" ORDER BY enqueued_at,id"), params)]

    def get_attempt(self, attempt_id):
        return self._one("knowledge_attempts", attempt_id=attempt_id)

    def get_generation(self, generation_id):
        return self._one("generations", generation_id=generation_id)

    def get_requirement(self, unit_id, generation_id):
        return self._one("knowledge_requirements", unit_id=unit_id, generation_id=generation_id)

    def get_input(self, consumed_input_revision):
        return self._one("input_manifests", consumed_input_revision=consumed_input_revision)

    def get_inputs(self, consumed_input_revisions):
        from psycopg import sql
        return list(self.connection.execute(sql.SQL("SELECT * FROM {} WHERE consumed_input_revision=ANY(%s)").format(
            self.repository._table("input_manifests")), (list(consumed_input_revisions),)))

    def list_inputs_for_unit(self, unit_id):
        return self._rows("input_manifests", unit_id=unit_id)

    def get_active_maintenance_window(self):
        from psycopg import sql
        return self.connection.execute(sql.SQL("SELECT * FROM {} WHERE starts_at<=clock_timestamp() "
            "AND ends_at>clock_timestamp()").format(self.repository._table("knowledge_maintenance_windows"))).fetchone()

    def list_pending_repairs(self, limit=100):
        from psycopg import sql
        return list(self.connection.execute(sql.SQL(
            "SELECT r.* FROM {} r JOIN {} p USING(publication_id) JOIN {} s ON s.derivation_id=p.derivation_id "
            "JOIN {} g ON g.generation_id=s.generation_id LEFT JOIN {} target ON target.generation_id=r.target_generation "
            "WHERE g.state='active' AND (r.target_generation IS NULL OR target.source_manifest_revision<>g.manifest_revision) "
            "ORDER BY r.created_at,r.repair_id LIMIT %s"
        ).format(self.repository._table("knowledge_repairs"), self.repository._table("publications"),
            self.repository._table("selections"), self.repository._table("generations"),
            self.repository._table("generations")), (limit,)))

    def get_repair(self, repair_id):
        return self._one("knowledge_repairs", repair_id=repair_id)

    def get_recipe(self, recipe_digest):
        row = self._one("recipes", recipe_digest=recipe_digest)
        return row["spec"] if row else None

    def get_record(self, record_key):
        record = self.repository.get(record_key)
        return {**record.__dict__} if record else None

    def get_reducer(self, unit_id, consumed_input_revision):
        return self._one("knowledge_unit_reducers", unit_id=unit_id, consumed_input_revision=consumed_input_revision)

    def get_selection(self, unit_id, generation_id):
        return self._one("selections", unit_id=unit_id, generation_id=generation_id)

    def get_receipt_publications(self, receipt_id):
        from psycopg import sql
        return list(self.connection.execute(sql.SQL(
            "SELECT p.* FROM {} p JOIN {} d USING(derivation_id) WHERE d.receipt_id=%s"
        ).format(self.repository._table("publications"), self.repository._table("derivations")), (receipt_id,)))

    def get_derivation(self, derivation_id):
        from psycopg import sql
        row = self._one("derivations", derivation_id=derivation_id)
        if row is None:
            return None
        last = self.connection.execute(sql.SQL("SELECT event,at FROM {} WHERE derivation_id=%s ORDER BY id DESC LIMIT 1").format(
            self.repository._table("supersession_events")), (derivation_id,)).fetchone()
        return {**row, "superseded_at": last["at"] if last and last["event"] == "superseded" else None}

    def get_manifest(self, generation_id, revision=None):
        generation = self.get_generation(generation_id)
        return self._one("generation_manifests", generation_id=generation_id,
                         manifest_revision=revision if revision is not None else generation["manifest_revision"])

    def _serving_pointer(self):
        state = self._one("knowledge_runtime", singleton=True)
        if not state or not state["serving_generation_id"]:
            return None
        manifest = self.get_manifest(state["serving_generation_id"], state["serving_manifest_revision"])["manifest"]
        return {"generation_id": state["serving_generation_id"],
                "manifest_revision": state["serving_manifest_revision"], "revision": state["serving_revision"],
                "retrieval_contract": manifest.get("retrieval_contract"), "manifest": manifest}

    def get_serving_manifest(self, evaluation_time=None, max_observation_age=300):
        """Pin an exact manifest and project fresh eligibility from catalog evidence."""
        from .coverage import query_coverage
        own_transaction = self.connection.info.transaction_status.name == "IDLE"
        with self.connection.transaction():
            if own_transaction:
                self.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            pointer = self._serving_pointer()
            if pointer is None:
                return None
            coverage = query_coverage(self, generation_id=pointer["generation_id"],
                evaluation_time=evaluation_time, max_observation_age=max_observation_age)
            by_unit = {row["unit_id"]: row for row in coverage["units"]}
            entries = []
            for desired in pointer["manifest"]["entries"]:
                covered = by_unit.get(desired["unit_id"], {})
                policy_ok = covered.get("source_eligible") is True
                receipt = covered.get("receipt") or {}
                same_selection = covered.get("derivation_id") == desired.get("selected_derivation_id")
                eligible = bool(not desired.get("excluded") and same_selection and policy_ok and covered.get("tier_covered")
                                and not covered.get("quarantine") and not covered.get("excluded")
                                and (covered.get("achieved_tier") != "fulltext" or covered.get("readable")))
                entries.append({"unit_id": desired["unit_id"], "derivation_id": desired.get("selected_derivation_id"),
                    "eligible": eligible, "receipt": receipt,
                    "reason": "verified_current" if eligible else "ineligible_or_unavailable"})
            return {**pointer, "manifest_id": pointer["generation_id"] + ":" + str(pointer["manifest_revision"]),
                    "entries": entries}

    def validate_serving_manifest(self, manifest_id, expected_revision=None, **kwargs):
        current = self.get_serving_manifest(**kwargs)
        return bool(current and current["manifest_id"] == manifest_id and
                    (expected_revision is None or current["revision"] == expected_revision) and
                    all(entry["eligible"] for entry in current["entries"]))

    def get_receipt(self, receipt_id):
        row = self._one("knowledge_receipts", receipt_id=receipt_id)
        return row["receipt"] if row else None

    def get_operation(self, operation_id):
        row = self._one("knowledge_operations", operation_id=operation_id)
        return row["result"] if row else None

    def get_pin(self, pin_id):
        return self._one("knowledge_pins", pin_id=pin_id)

    def protected_namespaces(self, evaluation_time=None):
        from datetime import timezone
        from psycopg import sql
        stamp = evaluation_time or datetime.now(timezone.utc)
        if isinstance(stamp, str):
            stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("protection evaluation time must be timezone aware")
        rows = self.connection.execute(sql.SQL(
            "SELECT jsonb_array_elements_text(manifest->'aggregate_layers') AS namespace FROM {} "
            "UNION SELECT jsonb_array_elements_text(r.receipt->'aggregate_layers') FROM {} d JOIN {} r USING(receipt_id) "
            "UNION SELECT jsonb_array_elements_text(namespaces) FROM {} WHERE released_at IS NULL AND (expires_at IS NULL OR expires_at>%s) "
            "UNION SELECT namespace FROM {} WHERE outcome IS NULL OR cost_state='unknown'"
        ).format(self.repository._table("generation_manifests"), self.repository._table("derivations"),
            self.repository._table("knowledge_receipts"), self.repository._table("knowledge_pins"),
            self.repository._table("knowledge_attempts")), (stamp,))
        return sorted(row["namespace"] for row in rows)

    def get_submission(self, submission_id):
        attempt = self._one("knowledge_attempts", submission_id=submission_id)
        if not attempt:
            return None
        job = self.get_job(attempt["job_id"])
        from .knowledge_commands import claim_result
        return {"claim": claim_result(job, attempt), "accepted": attempt["accepted"],
                "receipt_id": attempt["receipt_id"],
                "receipt": self.get_receipt(attempt["receipt_id"]) if attempt["receipt_id"] else None}

    def lookup_submission(self, submission_id):
        row = self.get_submission(submission_id)
        return row["receipt"] if row else None

    def list_outbox(self, limit=100, kind=None):
        from psycopg import sql
        return list(self.connection.execute(sql.SQL("SELECT * FROM {} WHERE acknowledged_at IS NULL "
            "AND (%s::text IS NULL OR kind=%s) ORDER BY created_at,outbox_id LIMIT %s").format(
                self.repository._table("knowledge_outbox")), (kind, kind, limit)))

    def list_reconcilable_receipts(self, scope=None, limit=1000):
        from psycopg import sql
        keys = self.resolve_scope_units(scope)
        rows = self.connection.execute(sql.SQL(
            "SELECT r.receipt FROM {} j JOIN {} a ON a.attempt_id=j.current_attempt_id "
            "JOIN {} r ON r.receipt_id=a.receipt_id LEFT JOIN {} op ON op.operation_id='publish:'||r.receipt_id "
            "WHERE j.kind='distill' AND j.state IN ('running','succeeded') AND op.operation_id IS NULL "
            "AND (%s::text[] IS NULL OR j.key=ANY(%s) OR j.unit_id=ANY(%s)) ORDER BY j.enqueued_at,j.id LIMIT %s"
        ).format(self.repository._table("jobs"), self.repository._table("knowledge_attempts"),
            self.repository._table("knowledge_receipts"), self.repository._table("knowledge_operations")),
            (keys, keys, keys, limit))
        return [r["receipt"] for r in rows]

    def list_selected_receipts(self, scope=None):
        from psycopg import sql
        keys = self.resolve_scope_units(scope)
        rows = self.connection.execute(sql.SQL(
            "SELECT {projection} AS receipt FROM {selections} s JOIN {derivations} d USING(derivation_id) JOIN {receipts} r USING(receipt_id) "
            "JOIN {generations} g ON g.generation_id=s.generation_id JOIN {units} u ON u.unit_id=s.unit_id "
            "WHERE g.state IN ('active','candidate') AND (%s::text[] IS NULL OR u.record_key=ANY(%s) OR u.unit_id=ANY(%s))"
        ).format(projection=self._receipt_projection("r"), selections=self.repository._table("selections"),
            derivations=self.repository._table("derivations"), receipts=self.repository._table("knowledge_receipts"),
            generations=self.repository._table("generations"), units=self.repository._table("units")), (keys, keys, keys))
        return [r["receipt"] for r in rows]

    def list_collection_heads(self):
        from psycopg import sql
        rows = self.connection.execute(sql.SQL(
            "SELECT r.receipt FROM {} g JOIN {} m ON m.generation_id=g.generation_id AND m.manifest_revision=g.manifest_revision "
            "JOIN {} r ON r.receipt->>'namespace'=m.manifest->'aggregate_layers'->>-1 WHERE g.state IN ('active','candidate')"
        ).format(self.repository._table("generations"), self.repository._table("generation_manifests"),
            self.repository._table("knowledge_receipts")))
        return [r["receipt"] for r in rows]

    def list_admission_jobs(self, scope=None, limit=100):
        from psycopg import sql
        keys = self.resolve_scope_units(scope)
        rows = self.connection.execute(sql.SQL(
            "SELECT j.* FROM {jobs} j JOIN {requirements} q ON q.unit_id=j.unit_id AND q.generation_id=j.target_generation "
            "JOIN {generations} g ON g.generation_id=j.target_generation JOIN {units} u ON u.unit_id=j.unit_id "
            "JOIN {inputs} i ON i.consumed_input_revision=j.consumed_input_revision "
            "JOIN {recipes} recipe ON recipe.recipe_digest=j.recipe_digest "
            "LEFT JOIN {records} r ON r.key=u.record_key "
            "LEFT JOIN {attempts} a ON a.attempt_id=j.current_attempt_id "
            "LEFT JOIN {operations} op ON op.operation_id='publish:'||a.receipt_id "
            "LEFT JOIN {reducers} cp ON cp.unit_id=j.unit_id AND cp.consumed_input_revision=j.consumed_input_revision "
            "LEFT JOIN {mappings} sm ON sm.unit_id=u.unit_id AND sm.mapping_revision::text="
                "COALESCE(i.manifest->>'section_definition_revision',i.manifest->>'section_mapping_revision') "
            "LEFT JOIN {units} parent ON parent.unit_id=sm.parent_unit_id "
            "LEFT JOIN {records} pr ON pr.key=parent.record_key "
            "LEFT JOIN {current_inventories} ci ON ci.parent_unit_id=parent.unit_id AND ci.generation_id=j.target_generation "
            "LEFT JOIN {inventories} inv ON inv.parent_unit_id=ci.parent_unit_id AND inv.inventory_revision=ci.inventory_revision "
            "WHERE j.kind='distill' AND NOT q.invalidated AND NOT q.excluded AND NOT j.quarantine AND NOT u.quarantine "
            "AND NOT COALESCE(cp.parked,FALSE) "
            "AND (g.state='active' OR (g.state='candidate' AND EXISTS(SELECT 1 FROM {windows} "
                "WHERE starts_at<=clock_timestamp() AND ends_at>clock_timestamp()))) "
            "AND (g.state='candidate' OR q.enqueue_mode='active') "
            "AND (u.record_key IS NULL OR (r.policy IN ('clear','override') AND r.merge='canonical' AND NOT r.quarantine "
                "AND q.record_revision=r.revision)) "
            "AND (u.kind<>'section' OR (parent.unit_id IS NOT NULL AND NOT parent.quarantine "
                "AND NOT ci.invalidated AND ci.inventory_revision=q.inventory_revision AND inv.mode='sections' "
                "AND inv.section_mapping_revisions->>u.unit_id=sm.mapping_revision::text "
                "AND (inv.required_units ? u.unit_id OR inv.optional_units ? u.unit_id) "
                "AND inv.parent_evidence_revision=sm.parent_evidence_revision "
                "AND i.manifest->>'parent_evidence_revision'=sm.parent_evidence_revision "
                "AND (parent.record_key IS NULL OR (pr.policy IN ('clear','override') AND pr.merge='canonical' "
                    "AND NOT pr.quarantine AND ci.record_revision=pr.revision)))) "
            "AND (NOT COALESCE((recipe.spec->'output_contract'->COALESCE(i.manifest->>'tier','fulltext')->'required') "
                "?| ARRAY['vectors','chunks_vdb','entities_vdb','relationships_vdb'],FALSE) "
                "OR (jsonb_typeof(recipe.spec->'embedding_contract')='object' AND (g.retrieval_contract IS NULL "
                    "OR g.retrieval_contract=recipe.spec->'embedding_contract'))) "
            "AND q.consumed_input_revision=j.consumed_input_revision AND q.recipe_digest=j.recipe_digest "
            "AND (j.state IN ('queued','failed') OR (j.state='running' AND a.lease_expires_at<=clock_timestamp()) "
            "OR (j.state='succeeded' AND op.operation_id IS NULL AND a.aggregate_base_manifest_revision<>g.manifest_revision)) "
            "AND (%s::text[] IS NULL OR j.key=ANY(%s) OR j.unit_id=ANY(%s)) ORDER BY j.enqueued_at,j.id LIMIT %s"
        ).format(**{name: self.repository._table(table) for name, table in {
            "jobs": "jobs", "requirements": "knowledge_requirements", "generations": "generations", "units": "units",
            "inputs": "input_manifests", "recipes": "recipes", "records": "records", "attempts": "knowledge_attempts",
            "operations": "knowledge_operations", "reducers": "knowledge_unit_reducers", "mappings": "section_mappings",
            "current_inventories": "current_inventories", "inventories": "unit_inventories",
            "windows": "knowledge_maintenance_windows"}.items()}), (keys, keys, keys, limit))
        return [{**r, "job_id": r["id"], "generation_id": r["target_generation"]} for r in rows]

    def get_section(self, unit_id, mapping_revision=None):
        rows = self._rows("section_mappings", unit_id=unit_id)
        if mapping_revision is not None:
            return next((r for r in rows if r["mapping_revision"] == mapping_revision), None)
        return max(rows, key=lambda r: r["mapping_revision"], default=None)

    def get_inventory(self, parent_unit_id, inventory_revision):
        return self._one("unit_inventories", parent_unit_id=parent_unit_id, inventory_revision=inventory_revision)

    def get_current_inventory(self, parent_unit_id, generation_id):
        selection = self._one("current_inventories", parent_unit_id=parent_unit_id, generation_id=generation_id)
        if not selection:
            return None
        return {**self.get_inventory(parent_unit_id, selection["inventory_revision"]),
                "generation_id": generation_id, "revision": selection["revision"],
                "invalidated": selection["invalidated"], "record_revision": selection["record_revision"],
                "enqueue_mode": selection["enqueue_mode"], "request_id": selection["request_id"],
                "desired_recipe_digest": selection["desired_recipe_digest"]}

    def get_authority_state(self):
        present = self.connection.execute("SELECT to_regclass(%s) AS name",
                         (self.repository.schema + ".knowledge_runtime",)).fetchone()
        if not present["name"]:
            return {"authority": "legacy", "authority_epoch": 1, "checkpoint": {}}
        return self._one("knowledge_runtime", singleton=True)

    def record_tombstone(self, claim, role, output_id):
        return self.command("record_tombstone", "tombstone:" + payload_hash({"namespace": claim["namespace"],
            "role": role, "output_id": output_id}), {"claim": claim, "role": role, "output_id": output_id})

    def list_tombstones(self, namespace, role=None):
        from psycopg import sql
        namespaces = [namespace] if isinstance(namespace, str) else list(namespace)
        query = sql.SQL("SELECT * FROM {} WHERE namespace=ANY(%s)").format(self.repository._table("knowledge_output_tombstones"))
        params = [namespaces]
        if role is not None:
            query += sql.SQL(" AND role=%s")
            params.append(role)
        return list(self.connection.execute(query, params))

    def tombstone_inventory(self, namespace):
        import hashlib
        from datetime import timezone
        rows = sorted(({"role": r["role"], "output_id": r["output_id"]} for r in self.list_tombstones(namespace)),
                      key=lambda r: (r["role"], r["output_id"]))
        data = canonical_json(rows).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        return {"role": "tombstones", "store": "catalog", "namespace": namespace,
            "output_id": namespace+":tombstones:"+digest,
            "backend_ids": [r["role"]+":"+r["output_id"] for r in rows], "sha256": digest, "byte_length": len(data),
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "verification": {"state": "present" if rows else "verified_empty", "method": "catalog_tombstones_readback_v1",
                "evidence": {"record_count": len(rows), "rows_digest": digest}}}

    def _session_namespace_lock(self, namespace):
        from contextlib import contextmanager
        @contextmanager
        def locked():
            key = "papervault:" + self.repository.schema + ":namespace:" + namespace
            acquired = self.connection.execute("SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS locked", (key,)).fetchone()["locked"]
            if not acquired:
                raise ClaimRejected("namespace has an active writer; retry after it drains")
            try:
                yield
            finally:
                self.connection.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))", (key,))
        return locked()

    @asynccontextmanager
    async def namespace_access(self, claim, seal=False):
        """Serialize every mutator and final seal on the same PG namespace lock.

        A dedicated connection must remain bound throughout this scope. Reentry
        is permitted only from its owning asyncio task, never a child task.
        """
        from .knowledge_commands import valid_claim
        namespace = claim["namespace"]
        task = asyncio.current_task()
        owner = self._namespace_owners.get(namespace)
        if owner is task:
            valid_claim(self, claim, allow_sealed=False)
            yield self
            return
        lock = self._namespace_locks.setdefault(namespace, asyncio.Lock())
        async with lock:
            lock_key = "papervault:" + self.repository.schema + ":namespace:" + namespace
            while not self.connection.execute("SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS locked",
                                               (lock_key,)).fetchone()["locked"]:
                await asyncio.sleep(0.05)
            self._namespace_owners[namespace] = task
            try:
                valid_claim(self, claim, allow_sealed=False)
                yield self
            finally:
                self._namespace_owners.pop(namespace, None)
                self.connection.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))", (lock_key,))

    @staticmethod
    def _receipt_projection(alias=None):
        """Read projection; full immutable receipt remains available by ID."""
        from psycopg import sql
        receipt = sql.Identifier(alias, "receipt") if alias else sql.Identifier("receipt")
        digest = sql.Identifier(alias, "digest") if alias else sql.Identifier("digest")
        return sql.SQL("({r}-ARRAY['base_aggregate_layers','aggregate_layers','base_selected_unit_receipts','selected_unit_receipts']) "
            "||jsonb_build_object('aggregate_parent_namespace',{r}->'base_aggregate_layers'->>-1,"
            "'aggregate_layer_count',jsonb_array_length(COALESCE({r}->'aggregate_layers','[]'::jsonb)),"
            "'selected_unit_receipt_id',({r}->'selected_unit_receipts')->({r}->>'unit_id'),"
            "'catalog_validation_digest',{digest},'compact_projection',true)").format(r=receipt, digest=digest)

    def get_namespace_receipts(self, namespaces, *, compact=False):
        from psycopg import sql
        projection = self._receipt_projection() if compact else sql.Identifier("receipt")
        rows = self.connection.execute(sql.SQL("SELECT {} AS receipt FROM {} WHERE receipt->>'namespace'=ANY(%s)").format(
            projection, self.repository._table("knowledge_receipts")), (list(namespaces),))
        return [row["receipt"] for row in rows]

    def coverage_rows(self, scope=None):
        """A consistent snapshot using batched relational reads, including inherited layers."""
        from psycopg import sql
        own_transaction = self.connection.info.transaction_status.name == "IDLE"
        with self.connection.transaction():
            if own_transaction:
                self.connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            keys = None if scope is None else ([scope] if isinstance(scope, str) else list(scope))
            resolved_units = self.resolve_scope_units(scope)
            units = list(self.connection.execute(sql.SQL(
                "SELECT * FROM {units} WHERE %s::text[] IS NULL OR unit_id=ANY(%s)"
            ).format(units=self.repository._table("units")), (resolved_units, resolved_units)))
            unit_ids = [u["unit_id"] for u in units]
            def batch(table, column, values):
                return list(self.connection.execute(sql.SQL("SELECT * FROM {} WHERE {}=ANY(%s)").format(
                    self.repository._table(table), sql.Identifier(column)), (list(values),))) if values else []
            def index(rows, key):
                return {row[key]: row for row in rows}
            def grouped(rows, key):
                result = {}
                for row in rows:
                    result.setdefault(row[key], []).append(row)
                return result
            records = index(batch("records", "key", {u["record_key"] for u in units if u["record_key"]}), "key")
            requirements = grouped(batch("knowledge_requirements", "unit_id", unit_ids), "unit_id")
            inventory_rows = batch("unit_inventories", "parent_unit_id", unit_ids)
            inventory_index = {(r["parent_unit_id"], r["inventory_revision"]): r for r in inventory_rows}
            current_inventories = batch("current_inventories", "parent_unit_id", unit_ids)
            for selector in current_inventories:
                parent_id = selector["parent_unit_id"]
                inventory = inventory_index[(parent_id, selector["inventory_revision"])]
                existing = [r for r in requirements.get(parent_id, []) if r["generation_id"] == selector["generation_id"]]
                if existing and not (inventory["mode"] == "sections" and all(r["excluded"] for r in existing)):
                    continue
                if existing:
                    requirements[parent_id] = [r for r in requirements[parent_id]
                                               if r["generation_id"] != selector["generation_id"]]
                # A section-built parent is readable evidence, not a second
                # executable/covered unit. Its selectors come from inventory.
                requirements.setdefault(parent_id, []).append({"unit_id": parent_id,
                    "generation_id": selector["generation_id"],
                    "consumed_input_revision": inventory.get("parent_consumed_input_revision") or inventory["parent_evidence_revision"],
                    "recipe_digest": None, "inventory_revision": selector["inventory_revision"],
                    "record_revision": selector["record_revision"], "revision": selector["revision"],
                    "invalidated": selector["invalidated"], "excluded": False, "inventory_parent_selector": True})
            selections = {(r["unit_id"], r["generation_id"]): r for r in batch("selections", "unit_id", unit_ids)}
            derivations = index(batch("derivations", "derivation_id", {r["derivation_id"] for r in selections.values()}), "derivation_id")
            selected_receipts = list(self.connection.execute(sql.SQL("SELECT receipt_id,{} AS receipt FROM {} WHERE receipt_id=ANY(%s)").format(
                self._receipt_projection(), self.repository._table("knowledge_receipts")),
                (list({r["receipt_id"] for r in derivations.values()}),)))
            receipts = {r["receipt_id"]: r["receipt"] for r in selected_receipts}
            layers = {ns for receipt in receipts.values() for ns in receipt.get("aggregate_layers", [receipt["namespace"]])}
            generation_manifests = {r["generation_id"]: r for r in self.connection.execute(sql.SQL(
                "SELECT m.* FROM {} m JOIN {} g ON m.generation_id=g.generation_id AND m.manifest_revision=g.manifest_revision "
                "WHERE g.generation_id=ANY(%s)"
            ).format(self.repository._table("generation_manifests"), self.repository._table("generations")),
                (list({r["generation_id"] for rows in requirements.values() for r in rows}),))}
            layers.update(ns for m in generation_manifests.values() for ns in m["manifest"].get("aggregate_layers", []))
            aggregate_receipts = self.get_namespace_receipts(layers, compact=True)
            def latest_observations(column, values, *, inputs_only=False):
                predicate = sql.SQL(" AND o.role='input_evidence'") if inputs_only else sql.SQL("")
                return list(self.connection.execute(sql.SQL("SELECT o.* FROM {heads} h JOIN {table} o USING(observation_id) "
                    "WHERE {column}=ANY(%s){predicate}").format(heads=self.repository._table("observation_heads"),
                        table=self.repository._table("observations"), column=sql.Identifier("h", column), predicate=predicate),
                        (list(values),))) if values else []
            observations = latest_observations("unit_id", unit_ids, inputs_only=True)
            aggregate_observations = latest_observations("namespace", layers)
            by_observation_id = {r["observation_id"]: r for r in observations + aggregate_observations}
            observations = grouped(by_observation_id.values(), "unit_id")
            inventories = grouped(inventory_rows, "parent_unit_id")
            sections = list(self.connection.execute(sql.SQL("SELECT * FROM {} WHERE unit_id=ANY(%s) OR parent_unit_id=ANY(%s)").format(
                self.repository._table("section_mappings")), (unit_ids, unit_ids)))
            jobs = grouped(batch("jobs", "unit_id", unit_ids), "unit_id")
            publications = grouped(batch("publications", "unit_id", unit_ids), "unit_id")
            all_requirements = [r for rows in requirements.values() for r in rows]
            recipe_ids = {r["recipe_digest"] for r in all_requirements} | {r["recipe_digest"] for r in aggregate_receipts}
            recipes = index(batch("recipes", "recipe_digest", recipe_ids), "recipe_digest")
            shared_aggregate_receipts = [{"receipt": receipt, "recipe": recipes[receipt["recipe_digest"]]["spec"]}
                                         for receipt in aggregate_receipts]
            shared_receipt_index = {r["receipt"]["namespace"]: r for r in shared_aggregate_receipts}
            shared_observation_index = grouped(aggregate_observations, "receipt_id")
            shared_section_index = grouped(sections, "unit_id")
            shared_parent_section_index = grouped(sections, "parent_unit_id")
            current_inventory_index = {(r["parent_unit_id"], r["generation_id"]): r for r in current_inventories}
            inputs = index(batch("input_manifests", "consumed_input_revision", {r["consumed_input_revision"] for r in all_requirements}), "consumed_input_revision")
            generations = index(batch("generations", "generation_id", {r["generation_id"] for r in all_requirements}), "generation_id")
            serving = self._serving_pointer()
            result = []
            for unit in units:
                uid = unit["unit_id"]
                for requirement in requirements.get(uid) or [None]:
                    gid = requirement["generation_id"] if requirement else None
                    selection = selections.get((uid, gid))
                    derivation = derivations.get(selection["derivation_id"]) if selection else None
                    receipt = receipts.get(derivation["receipt_id"]) if derivation else None
                    if receipt:
                        derivation = {**derivation, **{key: receipt[key] for key in CLAIM_FIELDS if key not in derivation}}
                    result.append({"unit": unit, "record": records.get(unit["record_key"]),
                        "requirement": requirement, "selection": selection, "selected_derivation": derivation,
                        "receipt": receipt, "receipt_validated": bool(receipt),
                        "input_manifest": inputs.get(requirement["consumed_input_revision"]) if requirement else None,
                        "observations": observations.get(uid, []), "aggregate_observations": aggregate_observations,
                        "aggregate_receipts": shared_aggregate_receipts,
                        "aggregate_receipts_by_namespace": shared_receipt_index,
                        "aggregate_observations_by_receipt": shared_observation_index,
                        "inventories": inventories.get(uid, []),
                        "section_mappings": shared_section_index.get(uid, []) + shared_parent_section_index.get(uid, []),
                        "jobs": jobs.get(uid, []), "publication_history": publications.get(uid, []),
                        "recipe": recipes.get(requirement["recipe_digest"], {}).get("spec") if requirement else None,
                        "current_inventory": current_inventory_index.get((uid, gid)),
                        "generation": generations.get(gid), "generation_manifest": generation_manifests.get(gid),
                        "serving_manifest": serving})
            unregistered = self.connection.execute(sql.SQL(
                "SELECT r.* FROM {} r WHERE NOT EXISTS(SELECT 1 FROM {} u WHERE u.record_key=r.key AND u.kind<>'section') "
                "AND (%s::text[] IS NULL OR r.key=ANY(%s))"
            ).format(self.repository._table("records"), self.repository._table("units")), (keys, keys))
            for record in unregistered:
                result.append({"unit": {"unit_id": None, "kind": "paper", "record_key": record["key"]},
                    "record": record, "requirement": None, "selection": None, "selected_derivation": None,
                    "receipt": None, "receipt_validated": False, "input_manifest": None,
                    "observations": [], "aggregate_observations": [], "aggregate_receipts": [], "inventories": [],
                    "section_mappings": [], "jobs": [], "publication_history": [], "recipe": None,
                    "generation": None, "generation_manifest": None, "serving_manifest": serving})
            return result
