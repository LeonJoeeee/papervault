"""Real lifecycle commands retain operator evidence at their storage boundary."""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from papervault.store.knowledge import KnowledgeRepository


COMPLETION = "a" * 32


class MemoryCatalog:
    """Replace SQL storage only; command, dispatch and lifecycle reducers stay real."""

    def __init__(self):
        self.rows = {"extraction_receipts": [{"completion_id": COMPLETION, "receipt": {}}]}
        self.audits = []
        self.connection = self
        self.writing = False

    @contextmanager
    def _write(self, audit):
        before = deepcopy((self.rows, self.audits))
        self.writing = True
        try:
            yield
        except Exception:
            self.rows, self.audits = before
            raise
        finally:
            self.writing = False

    def _finish(self, key, audit):
        assert self.writing, "audit must be in the same transaction as lifecycle mutation"
        self.audits.append(audit)

    def execute(self, query):
        assert query == "SELECT clock_timestamp() AS now"
        return self

    def fetchone(self):
        return {"now": datetime(2026, 10, 22, tzinfo=timezone.utc)}


class StoredKnowledge(KnowledgeRepository):
    def _rows(self, table, **where):
        return [deepcopy(row) for row in self.repository.rows.get(table, [])
                if all(row.get(key) == value for key, value in where.items())]

    def _insert(self, table, values):
        assert self.repository.writing, "lifecycle mutation must share the audit transaction"
        row = deepcopy(values)
        if table == "extract_pins":
            row.update(revision=1, released_at=None)
        self.repository.rows.setdefault(table, []).append(row)
        return deepcopy(row)

    def _update(self, table, values, **where):
        assert self.repository.writing
        row = next(row for row in self.repository.rows[table]
                   if all(row.get(key) == value for key, value in where.items()))
        row.update(deepcopy(values))
        return deepcopy(row)


@pytest.fixture
def knowledge():
    return StoredKnowledge(MemoryCatalog())


def pin(knowledge, **extra):
    return knowledge.command("pin_extract", "pin-op", {
        "completion_id": COMPLETION, "kind": "recovery", "owner": "recovery-job", **extra,
    })


def test_pin_records_operator_actor_even_when_real_result_omits_it(knowledge):
    result = pin(knowledge, actor="leon")
    assert "actor" not in result
    audit = knowledge.repository.audits[-1]
    assert audit.actor == "leon"
    assert audit.event == "pin_extract"
    assert audit.detail == {"operation_id": "pin-op", "completion_id": COMPLETION,
                            "kind": "recovery", "owner": "recovery-job"}


def test_reviewed_release_preserves_reason_nonce_revision_and_actor_once(knowledge):
    acquired = pin(knowledge)
    payload = {"pin_id": acquired["pin_id"], "owner": "recovery-job", "expected_revision": 1,
        "disposition": "reviewed_recovery_disposition", "reason": "verified original executor drained",
        "operator_nonce": "owner-review-2026", "actor": "leon"}
    result = knowledge.command("release_extract_pin", "release-op", payload)
    assert result["released_at"] is not None
    assert result["revision"] == 2
    assert "reason" not in result
    assert "operator_nonce" not in result
    audit = knowledge.repository.audits[-1]
    assert audit.actor == "leon"
    assert audit.detail == {"operation_id": "release-op", "pin_id": acquired["pin_id"],
        "owner": "recovery-job", "expected_revision": 1,
        "disposition": "reviewed_recovery_disposition", "reason": "verified original executor drained",
        "operator_nonce": "owner-review-2026"}
    assert knowledge.command("release_extract_pin", "release-op", payload) == result
    assert len(knowledge.repository.audits) == 2
    assert len(knowledge.repository.rows["knowledge_operations"]) == 2


@pytest.mark.parametrize("actor", ["", " ", None, 1])
def test_invalid_explicit_actor_cannot_mutate_or_emit_audit(knowledge, actor):
    before = deepcopy(knowledge.repository.rows)
    with pytest.raises(ValueError, match="actor"):
        pin(knowledge, actor=actor)
    assert knowledge.repository.rows == before
    assert knowledge.repository.audits == []


def test_programmatic_pin_without_actor_keeps_existing_knowledge_actor(knowledge):
    pin(knowledge)
    assert knowledge.repository.audits[-1].actor == "knowledge"


def test_extract_audit_omits_unrelated_credentials_and_arbitrary_payloads(knowledge):
    result = pin(knowledge, actor="leon", reason="manual protection", operator_nonce="review-1",
                 api_key="credential-must-not-leak", provider={"authorization": "also-secret"})
    assert result["kind"] == "recovery"
    detail = knowledge.repository.audits[-1].detail
    assert detail == {"operation_id": "pin-op", "completion_id": COMPLETION,
        "kind": "recovery", "owner": "recovery-job", "reason": "manual protection",
        "operator_nonce": "review-1"}


def test_other_knowledge_actions_keep_their_existing_audit_contract(knowledge):
    knowledge.command("register_unit", "unit-op", {"unit_id": "paper:A", "kind": "paper",
        "identity": "A", "actor": "other-actor", "reason": "unrelated reason"})
    audit = knowledge.repository.audits[-1]
    assert audit.actor == "knowledge"
    assert audit.detail == {"operation_id": "unit-op"}


def test_retention_audit_keeps_review_evidence_beyond_real_retention_row(knowledge, tmp_path):
    from papervault.store.extraction import ExtractionStore

    store = ExtractionStore(tmp_path)
    request = store.capture(record_key="A", metadata={"title": "Fixture"}, source=b"source pdf",
        source_role="pdf", recipe={"schema_version": 1},
        guard={"identity": {"doi": "10.1234/fixture"}, "previous_completion_id": None},
        unit_evidence={"kind": "paper", "record_key": "A"})
    receipt = store.finalize(request, b"immutable markdown")
    completion_id = receipt["completion_id"]
    knowledge.repository.rows["extraction_receipts"].append({"completion_id": completion_id, "receipt": receipt})
    knowledge.repository.rows["extract_retention"] = [{"completion_id": completion_id,
        "vault": str(tmp_path), "revision": 7, "disposable": False}]
    result = knowledge.command("set_extract_retention", "retention-op", {
        "completion_id": completion_id, "vault": str(tmp_path), "expected_revision": 7,
        "disposable": True, "reason": "reviewed supersession", "operator_nonce": "retention-review",
        "actor": "leon"})
    assert result["disposable"] is True
    assert "reason" not in result and "operator_nonce" not in result
    audit = knowledge.repository.audits[-1]
    assert audit.actor == "leon"
    assert audit.detail == {"operation_id": "retention-op", "completion_id": completion_id,
        "vault": str(tmp_path), "expected_revision": 7, "disposable": True,
        "reason": "reviewed supersession", "operator_nonce": "retention-review"}


def test_maintenance_audit_keeps_explicit_window_owner_and_bounded_duration(knowledge):
    from datetime import timedelta

    stamp = datetime(2026, 10, 22, tzinfo=timezone.utc)
    knowledge.repository.rows["knowledge_runtime"] = [{"singleton": True, "authority": "catalog",
                                                       "authority_epoch": 3}]
    knowledge.repository.rows["knowledge_maintenance_windows"] = [{"window_id": "window-1",
        "starts_at": stamp - timedelta(seconds=1), "ends_at": stamp + timedelta(seconds=600)}]
    lease = knowledge.command("acquire_maintenance_lease", "lease-op", {"resource": "extract_gc",
        "window_id": "window-1", "owner": "gc-operator", "lease_seconds": 120, "actor": "leon"})
    assert lease["authority_epoch"] == 3
    audit = knowledge.repository.audits[-1]
    assert audit.actor == "leon"
    assert audit.detail == {"operation_id": "lease-op", "resource": "extract_gc",
                            "window_id": "window-1", "owner": "gc-operator", "lease_seconds": 120}
    knowledge.command("renew_maintenance_lease", "renew-op", {"lease": {**lease, "api_key": "secret"},
        "lease_seconds": 100, "actor": "leon"})
    audit = knowledge.repository.audits[-1]
    assert audit.detail == {"operation_id": "renew-op", "lease_seconds": 100, "lease": {
        "resource": "extract_gc", "lease_id": lease["lease_id"], "owner": "gc-operator",
        "authority_epoch": 3, "fencing_token": 1, "window_id": "window-1"}}
