"""Offline reconciliation proofs using the real command and persistence adapter.

The double models revision checks/transactions, not PostgreSQL SQL semantics;
test_registration_pg.py covers the same two-writer sequence on opt-in PG.
"""

import copy
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from psycopg import sql

from test_persistence import pg_library
from test_registration_pg import candidate
from papervault.library.services.registration import RegistrationQueue, RegistrationService, ServingTrace
from papervault.store.repository import StaleWrite


class MemoryCatalog:
    schema = "catalog_test_memory"

    def __init__(self):
        self.records, self.events, self.jobs = {}, [], []
        self.connection = SimpleNamespace(execute=lambda *a: [])

    @contextmanager
    def _write(self, audit):
        before = copy.deepcopy((self.records, self.events, self.jobs))
        try:
            yield
        except Exception:
            self.records, self.events, self.jobs = before
            raise

    def _table(self, name):
        return sql.Identifier(self.schema, name)

    def all(self):
        return iter(list(self.records.values()))

    def get(self, key):
        return self.records.get(key)

    def _finish(self, key, audit):
        self.events.append((key, audit))
        self.jobs.extend(audit.pending_jobs)

    def upsert(self, record, *, expected_revision, audit):
        current = self.get(record.key)
        actual = current.revision if current else 0
        if actual != expected_revision:
            raise StaleWrite(record.key, expected_revision, actual)
        if current and current.payload == record.payload:
            return False
        self.records[record.key] = replace(record, revision=actual + 1)
        self._finish(record.key, audit)
        return True

    def flush(self, records, *, expected_revisions, audit):
        with self._write(audit):
            return {r.key: expected_revisions[r.key] + int(self.upsert(
                r, expected_revision=expected_revisions[r.key], audit=audit)) for r in records}


@pytest.fixture
def memory_catalog():
    return MemoryCatalog()


@pytest.mark.parametrize("field,local,winner", [
    ("doi", "10.1234/E", "10.1234/D"),
    ("arxiv_id", "2401.00001v1", "2401.00002v1"),
    ("paper_id", "E", "D"),
])
async def test_registration_refreshes_divergent_local_identity_before_later_save(
        memory_catalog, tmp_path, monkeypatch, field, local, winner):
    repo = memory_catalog
    one = pg_library(repo, tmp_path / "one", monkeypatch)
    paper, _ = one.upsert(candidate(doi="", paper_id=""))
    one.save(force=True)
    two = pg_library(repo, tmp_path / "two", monkeypatch)
    assert one.adopt_identifiers(paper.key, **{field: local}) == {field: "adopted"}
    assert two.adopt_identifiers(paper.key, **{field: winner}) == {field: "adopted"}
    two.save(force=True)
    committed = repo.get(paper.key)

    incoming = candidate(doi="", paper_id="")
    incoming[field] = winner
    outcomes = await RegistrationService(one, SimpleNamespace(add=lambda key: None)).register_candidates(
        [incoming],
        actor="test", reason="two-writer-reconciliation")

    # Check the externally visible hazard before checking diagnostic details.
    one.save(force=True)
    assert repo.get(paper.key).payload[field] == winner
    assert getattr(paper, field) == winner
    assert one.get(paper.key) is paper
    assert one._catalog_backend.persisted[paper.key].revision >= committed.revision
    assert outcomes[0]["status"] == "duplicate"
    assert outcomes[0]["key"] == paper.key
    assert outcomes[0]["reason"] == "identity_conflict"
    conflicts = [{"field": field, "authority": winner, "observed": local, "origin": "local"}]
    assert outcomes[0]["identity_conflicts"] == conflicts
    events = [event for key, event in repo.events
              if key == paper.key and event.event == "registration"]
    assert len(events) == 1
    assert events[0].detail["identity_conflicts"] == conflicts
    assert events[0].detail["observations"][0]["identity_conflicts"] == conflicts


async def test_registration_reloads_existing_row_at_accept_time(memory_catalog, tmp_path, monkeypatch):
    repo = memory_catalog
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper, _ = lib.upsert(candidate())
    lib.save(force=True)
    original_all = repo.all

    def stale_lookup_snapshot():
        records = list(original_all())
        current = repo.get(paper.key)
        repo.records[paper.key] = replace(current,
            payload={**current.payload, "abstract": "Authoritative fresh abstract"},
            revision=current.revision + 1)
        return iter(records)

    monkeypatch.setattr(repo, "all", stale_lookup_snapshot)
    outcomes = await RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [candidate()], actor="test", reason="fresh-accept")
    assert outcomes[0]["status"] == "duplicate"
    assert paper.abstract == "Authoritative fresh abstract"
    assert repo.get(paper.key).payload["abstract"] == "Authoritative fresh abstract"
    assert lib._catalog_backend.persisted[paper.key].revision == repo.get(paper.key).revision


@pytest.mark.parametrize("field,winner,alternative,status", [
    ("doi", "10.1234/D", "10.1234/E", "registered"),
    ("arxiv_id", "2401.00002v1", "2401.00001v1", "registered"),
    ("arxiv_id", "2401.00002v1", "2401.00002v2", "rejected"),
    ("paper_id", "D", "E", "duplicate"),
])
async def test_candidate_cannot_replace_authoritative_identity(
        memory_catalog, tmp_path, monkeypatch, field, winner, alternative, status):
    repo = memory_catalog
    lib = pg_library(repo, tmp_path, monkeypatch)
    data = candidate(doi="", paper_id="")
    data[field] = winner
    paper, _ = lib.upsert(data)
    lib.save(force=True)
    incoming = candidate(doi="", paper_id="")
    incoming[field] = alternative
    outcomes = await RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [incoming], actor="test", reason="conflicting-observation")
    result = outcomes[0]
    assert result["status"] == status
    if status == "registered":
        assert result["key"] != paper.key
        assert getattr(lib.get(result["key"]), field) == alternative
        assert repo.get(result["key"]).payload[field] == alternative
        assert len(list(repo.all())) == 2
    else:
        assert result["reason"] == "identity_conflict"
        assert len(list(repo.all())) == 1
        if status == "duplicate":
            assert result["key"] == paper.key
            assert result["identity_conflicts"][0]["decision"] == "refused-existing"
        else:
            assert result["key"] is None
    lib.save(force=True)
    assert getattr(paper, field) == winner
    assert repo.get(paper.key).payload[field] == winner


@pytest.mark.parametrize("claim", ["owner", "reservation"])
async def test_registration_source_id_adoption_checks_live_claims(
        memory_catalog, tmp_path, monkeypatch, claim):
    repo = memory_catalog
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper, _ = lib.upsert(candidate(doi="", paper_id="", title="Independent detector observations"))
    lib.save(force=True)
    if claim == "owner":
        assert lib.adopt_identifiers(paper.key, paper_id="123") == {"paper_id": "adopted"}
    else:
        lib._identity_reservations.add(("paper_id", "core:work:123", paper.key, object()))
    outcomes = await RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [candidate()], actor="test", reason="claimed-source-id")
    registered = lib.get(outcomes[0]["key"])
    assert registered.key != paper.key
    assert registered.paper_id == ""
    assert outcomes[0]["reason"] == "identity_conflict"
    assert repo.get(registered.key).payload["paper_id"] == ""


async def test_identity_edit_during_commit_is_refreshed_and_traced(memory_catalog, tmp_path, monkeypatch):
    import asyncio
    import json

    repo = memory_catalog
    local = pg_library(repo, tmp_path / "local", monkeypatch)
    paper, _ = local.upsert(candidate(doi="", paper_id=""))
    local.save(force=True)
    other = pg_library(repo, tmp_path / "other", monkeypatch)
    other.adopt_identifiers(paper.key, doi="10.1234/D")
    other.save(force=True)
    upsert = repo.upsert

    def edit_during_io(record, **kwargs):
        result = upsert(record, **kwargs)
        assert local.adopt_identifiers(paper.key, doi="10.1234/E") == {"doi": "adopted"}
        paper.download_status = "metadata_only"
        return result

    monkeypatch.setattr(repo, "upsert", edit_during_io)
    queue = RegistrationQueue(RegistrationService(local, SimpleNamespace(add=lambda key: None)))
    await queue.start()
    try:
        queue.submit([candidate(doi="10.1234/D", paper_id="")], actor="test", reason="commit-edit",
                     trace=ServingTrace(local, "commit-edit"))
        await asyncio.wait_for(queue._queue.join(), 2)
    finally:
        await queue.stop()
    assert paper.doi == "10.1234/D"
    assert paper.download_status == "metadata_only"
    events = [json.loads(line) for line in local.manifest_path.read_text().splitlines()]
    decisions = [e for e in events if e.get("decision") == "duplicate"]
    assert len(decisions) == 1
    assert decisions[0]["reason"] == "identity_conflict"
    assert decisions[0]["identity_conflicts"] == 1
    monkeypatch.setattr(repo, "upsert", upsert)
    local.save(force=True)
    assert repo.get(paper.key).payload["doi"] == "10.1234/D"
    assert repo.get(paper.key).payload["download_status"] == "metadata_only"


async def test_fresh_row_load_does_not_hold_the_serving_identity_lock(memory_catalog, tmp_path, monkeypatch):
    import asyncio
    import threading

    repo = memory_catalog
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper, _ = lib.upsert(candidate())
    lib.save(force=True)
    entered, release = threading.Event(), threading.Event()
    get = repo.get

    def stalled_get(key):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return get(key)

    monkeypatch.setattr(repo, "get", stalled_get)
    command = asyncio.create_task(RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [candidate()], actor="test", reason="stalled-row-load"))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert await asyncio.wait_for(asyncio.to_thread(lib.find, doi="10.1234/registration"), 1) is paper
    finally:
        release.set()
        await command


def test_existing_authoritative_source_id_with_another_live_owner_is_rejected(tmp_path):
    from papervault.library import Library
    from papervault.library.services.registration import _view

    lib = Library(tmp_path)
    original, _ = lib.upsert(candidate(paper_id=""))
    worker, _ = lib.upsert(candidate(doi="", paper_id="123", title="Independent detector observations"))
    view = _view(lib, lib._papers)
    view.get(original.key).paper_id = "123"  # authoritative identity versus stale local owner
    service = RegistrationService(lib, SimpleNamespace(add=lambda key: None))
    outcomes, changed = service._stage(view, [candidate()])
    assert outcomes[0]["status"] == "rejected"
    assert outcomes[0]["reason"] == "identity_conflict"
    assert changed == {}
    assert lib.get(worker.key).paper_id == "123"


async def test_failed_registration_does_not_acknowledge_a_newer_identity_revision(
        memory_catalog, tmp_path, monkeypatch):
    from papervault.store.persistence import CatalogConflict

    repo = memory_catalog
    lib = pg_library(repo, tmp_path / "one", monkeypatch)
    paper, _ = lib.upsert(candidate(doi=""))
    lib.save(force=True)
    original_revision = lib._catalog_backend.persisted[paper.key].revision
    other = pg_library(repo, tmp_path / "two", monkeypatch)
    lib.adopt_identifiers(paper.key, doi="10.1234/E")
    other.adopt_identifiers(paper.key, doi="10.1234/D")
    other.save(force=True)
    committed = repo.get(paper.key)
    upsert = repo.upsert

    def fail_after_mutation(record, **kwargs):
        upsert(record, **kwargs)
        raise OSError("injected transaction failure")

    monkeypatch.setattr(repo, "upsert", fail_after_mutation)
    outcomes = await RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [candidate(doi="10.1234/D")], actor="test", reason="failed-reconciliation")
    assert outcomes[0]["status"] == "failed"
    assert repo.get(paper.key) == committed
    assert paper.doi == "10.1234/E"
    assert lib._catalog_backend.persisted[paper.key].revision == original_revision
    assert not any(event.event == "registration" for _, event in repo.events)
    monkeypatch.setattr(repo, "upsert", upsert)
    with pytest.raises(CatalogConflict):
        lib.save(force=True)
    assert repo.get(paper.key) == committed


async def test_refused_inspire_identity_does_not_publish_its_document_links(
        memory_catalog, tmp_path, monkeypatch):
    repo = memory_catalog
    lib = pg_library(repo, tmp_path, monkeypatch)
    worker, _ = lib.upsert(candidate(doi="", source="inspire", paper_id="",
                                    title="Independent detector observations"))
    lib.save(force=True)
    lib.adopt_identifiers(worker.key, inspire_record_id="1412451")
    outcomes = await RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [candidate(source="inspire", paper_id="", inspire_record_id="1412451",
                   inspire_document_urls=["https://inspirehep.net/files/test.pdf"])],
        actor="test", reason="claimed-inspire-id")
    registered = lib.get(outcomes[0]["key"])
    assert outcomes[0]["reason"] == "identity_conflict"
    assert registered.inspire_record_id == ""
    assert registered.inspire_document_urls == []
    assert repo.get(registered.key).payload["inspire_document_urls"] == []


async def test_stalled_adoption_evidence_does_not_hold_the_serving_lock(memory_catalog, tmp_path, monkeypatch):
    import asyncio
    import threading

    repo = memory_catalog
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper, _ = lib.upsert(candidate())
    lib.save(force=True)
    entered, release = threading.Event(), threading.Event()
    stage_fields = lib.corrections.stage_fields

    def stalled_evidence(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return stage_fields(*args, **kwargs)

    monkeypatch.setattr(lib.corrections, "stage_fields", stalled_evidence)
    command = asyncio.create_task(RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [candidate()], actor="test", reason="stalled-adoption-evidence"))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert await asyncio.wait_for(asyncio.to_thread(lib.find, doi="10.1234/registration"), 1) is paper
    finally:
        release.set()
        await command


async def test_registration_preserves_unsaved_worker_metadata_from_before_staging(
        memory_catalog, tmp_path, monkeypatch):
    from papervault.library.state_v2 import Policy, Retrieval, TerminalState

    repo = memory_catalog
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper, _ = lib.upsert(candidate())
    paper.terminal_state = TerminalState(policy=Policy.CLEAR, retrieval=Retrieval.ELIGIBLE)
    lib.save(force=True)
    paper.download_status = "metadata_only"
    paper.abstract = "Worker abstract pending its normal save"
    outcomes = await RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [candidate()], actor="test", reason="preexisting-worker-edits")
    assert outcomes[0]["reason"] == "identity_exists"
    assert paper.download_status == "metadata_only"
    assert paper.abstract == "Worker abstract pending its normal save"
    lib.save(force=True)
    assert repo.get(paper.key).payload["download_status"] == "metadata_only"
    assert repo.get(paper.key).payload["abstract"] == "Worker abstract pending its normal save"


async def test_registration_keeps_worker_document_links_for_the_same_inspire_identity(
        memory_catalog, tmp_path, monkeypatch):
    repo = memory_catalog
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper, _ = lib.upsert(candidate(source="inspire", paper_id="", inspire_record_id="1412451"))
    lib.save(force=True)
    upsert = repo.upsert

    def worker_links(record, **kwargs):
        result = upsert(record, **kwargs)
        assert lib.adopt_identifiers(paper.key, inspire_record_id="1412451") == {
            "inspire_record_id": "already-mine"}
        paper.inspire_document_urls = ["https://inspirehep.net/files/worker.pdf"]
        return result

    monkeypatch.setattr(repo, "upsert", worker_links)
    outcomes = await RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [candidate(source="inspire", paper_id="", inspire_record_id="1412451")],
        actor="test", reason="worker-document-links")
    assert outcomes[0]["reason"] == "identity_exists"
    assert paper.inspire_record_id == "1412451"
    assert paper.inspire_document_urls == ["https://inspirehep.net/files/worker.pdf"]
    monkeypatch.setattr(repo, "upsert", upsert)
    lib.save(force=True)
    assert repo.get(paper.key).payload["inspire_document_urls"] == ["https://inspirehep.net/files/worker.pdf"]


@pytest.mark.parametrize("recid", [1412451, "01412451"])
async def test_registration_normalizes_inspire_identity_before_adoption(
        memory_catalog, tmp_path, monkeypatch, recid):
    repo = memory_catalog
    lib = pg_library(repo, tmp_path, monkeypatch)
    outcomes = await RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [candidate(source="inspire", paper_id="", inspire_record_id=recid,
                   inspire_document_urls=["https://inspirehep.net/files/paper.pdf"])],
        actor="test", reason="normalized-inspire-id")
    assert outcomes[0]["status"] == "registered"
    paper = lib.get(outcomes[0]["key"])
    assert paper.inspire_record_id == "1412451"
    assert paper.inspire_document_urls == ["https://inspirehep.net/files/paper.pdf"]
    assert repo.get(paper.key).payload["inspire_record_id"] == "1412451"
    evidence = next(event for _, event in repo.events if event.event == "registration")
    assert evidence.detail["observations"][0]["inspire_record_id"] == recid
