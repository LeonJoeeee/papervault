"""Operator decisions share the record CAS and transactional commit boundary."""

import copy
from contextlib import contextmanager
from dataclasses import replace

import pytest

from papervault.library.models import Paper
from papervault.library.state_v2 import Merge, TerminalState
from papervault.store.persistence import CatalogConflict, PostgresPersistence
from papervault.store.repository import AuditEvent, PostgresCatalogRepository, Record


class TransactionalCatalog:
    """Database boundary double; real persistence batching/CAS stays in use."""

    def __init__(self, record):
        self.records = {record.key: record}
        self.rulings, self.audit = [], []
        self.reject_ruling = False

    @contextmanager
    def _write(self, audit):
        before = copy.deepcopy((self.records, self.rulings, self.audit))
        try:
            yield
        except Exception:
            self.records, self.rulings, self.audit = before
            raise

    def get(self, key):
        return self.records.get(key)

    def flush(self, records, *, expected_revisions, audit):
        for row in records:
            assert self.records[row.key].revision == expected_revisions[row.key]
            self.records[row.key] = replace(row, revision=expected_revisions[row.key] + 1)
            self.audit.append(audit)
        return {r.key: self.records[r.key].revision for r in records}

    def import_ruling(self, ruling, *, audit):
        if self.reject_ruling:
            raise ValueError("injected ruling insert failure")
        self.rulings.append(ruling)
        self.audit.append(audit)


@pytest.fixture
def operator_batch(monkeypatch):
    paper = Paper(key="A", title="initial")
    record = replace(Record.from_payload("A", paper.model_dump(mode="json")), revision=1)
    repository = TransactionalCatalog(record)
    backend = PostgresPersistence("unused")
    backend.persisted = {"A": record}
    backend.normalized = {"A": paper.model_dump(mode="json")}
    @contextmanager
    def connect():
        yield repository
    monkeypatch.setattr(backend, "repository", connect)
    backend.operator_rulings.append({"timestamp": "2026-10-22T00:00:00+00:00",
        "action": "reopen_download", "event": "operator_reopen", "scope": "acquire",
        "reason": "investigated", "actor": "owner", "keys": ["A"], "revision": 1})
    paper.download_status = "failed"
    return backend, repository, paper


def test_operator_insert_failure_rolls_back_state_audit_and_pending_work(operator_batch):
    backend, repository, paper = operator_batch
    repository.reject_ruling = True
    with pytest.raises(ValueError, match="ruling insert"):
        backend.save({"A": paper})
    assert repository.get("A").revision == 1
    assert repository.get("A").payload["download_status"] == "pending"
    assert repository.rulings == repository.audit == []
    assert backend.persisted["A"].revision == 1 and backend.dirty({"A": paper})
    assert len(backend.operator_rulings) == 1


def test_operator_batch_commits_named_audit_ruling_and_outbox_once(operator_batch):
    backend, repository, paper = operator_batch
    backend.save({"A": paper})
    assert repository.get("A").revision == 2
    assert len(repository.rulings) == 1
    assert repository.audit[0].event == "operator_reopen"
    assert repository.audit[0].actor == "owner"
    assert repository.audit[0].pending_jobs[0].id == "catalog-export:A"
    backend.save({"A": paper})
    assert len(repository.rulings) == 1 and len(repository.audit) == 2


def test_stale_operator_cannot_commit_ruling_or_merge_disjoint_metadata(operator_batch):
    backend, repository, paper = operator_batch
    repository.records["A"] = replace(repository.records["A"], revision=2)
    with pytest.raises(CatalogConflict):
        backend.save({"A": paper})
    assert repository.rulings == repository.audit == []
    assert backend.persisted["A"].revision == 1


def test_retained_alias_identifiers_do_not_compete_for_resolved_ownership():
    payload = Paper(key="B", doi="10.1234/retained", arxiv_id="1234.5678",
        terminal_state=TerminalState(merge=Merge.ALIAS, alias_key="A")).model_dump(mode="json")
    record = Record.from_payload("B", payload)
    assert record.doi is record.arxiv_id is None
    assert record.payload["doi"] == "10.1234/retained"
    assert record.payload["arxiv_id"] == "1234.5678"


def test_alias_import_cannot_release_an_identifier_without_a_canonical_owner():
    from papervault.store.repository import _check_alias_ownership
    alias = Record.from_payload("B", Paper(key="B", doi="10.1234/observation",
        terminal_state=TerminalState(merge=Merge.ALIAS, alias_key="A")).model_dump(mode="json"))
    canonical = Record.from_payload("A", {"key": "A", "doi": "10.1234/other"})
    with pytest.raises(ValueError, match="retained identifier"):
        _check_alias_ownership(alias, canonical)
    transferred = Record.from_payload("A", {"key": "A", "doi": "10.1234/observation"})
    _check_alias_ownership(alias, transferred)
    # A legacy alias's existing unique owner column remains valid until an
    # explicit transfer; reading it never relinquishes that reservation.
    _check_alias_ownership(replace(alias, doi="10.1234/observation"), canonical)


def test_public_canonical_upsert_checks_retained_alias_owners(monkeypatch):
    """Use the real public write boundary with a narrow SQL storage double."""
    alias = Record.from_payload("B", Paper(key="B", doi="10.1234/retained",
        terminal_state=TerminalState(merge=Merge.ALIAS, alias_key="A")).model_dump(mode="json"))
    original = replace(Record.from_payload("A", {"key": "A", "doi": "10.1234/retained"}), revision=1)
    storage = TransactionalCatalog(original)
    repository = PostgresCatalogRepository.__new__(PostgresCatalogRepository)
    monkeypatch.setattr(repository, "_write", storage._write)
    monkeypatch.setattr(repository, "get", storage.get)
    def write(record, audit, expected):
        storage.records[record.key] = record
        return True
    monkeypatch.setattr(repository, "_upsert", write)
    def validate():
        from papervault.store.repository import _check_alias_ownership
        _check_alias_ownership(alias, storage.get("A"))
    monkeypatch.setattr(repository, "_validate_alias_ownership", validate, raising=False)
    changed = Record.from_payload("A", {"key": "A", "doi": "10.1234/different"})
    with pytest.raises(ValueError, match="retained identifier"):
        repository.upsert(changed, expected_revision=1, audit=AuditEvent("test", "change"))
    assert storage.get("A") == original
