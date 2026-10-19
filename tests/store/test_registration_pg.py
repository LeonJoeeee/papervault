"""Opt-in PG proofs for the separately admitted registration command."""

import asyncio
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from psycopg import sql

from test_persistence import pg_library
from papervault.library.services.registration import RegistrationService
from papervault.library.state_v2 import Policy, Retrieval, TerminalState
from papervault.store.repository import AuditEvent, Record


def candidate(**fields):
    return {"title": "Registration of cosmic ray transport measurements",
            "authors": ["Tester, A."], "year": 2024, "doi": "10.1234/registration",
            "source": "core", "paper_id": "123", "abstract": "Cosmic ray measurements.",
            "publication_types": ["JournalArticle"],
            "_registration": {"ingest_ok": True, "llm_is_paper": True, "tier": "1A"}, **fields}


def rows(repo, table, where="TRUE"):
    return repo.connection.execute(sql.SQL("SELECT * FROM {} WHERE " + where).format(
        repo._table(table))).fetchall()


@pytest.mark.parametrize("download_budget", [0, 1])
async def test_registration_commits_record_audit_and_pending_jobs_together(
        repo, tmp_path, monkeypatch, download_budget):
    lib = pg_library(repo, tmp_path, monkeypatch)
    queued = []
    service = RegistrationService(lib, SimpleNamespace(add=queued.append), max_downloads=download_budget)
    outcomes = await service.register_candidates([candidate()], actor="test", reason="registration-1")
    key = outcomes[0]["key"]
    assert outcomes[0]["status"] == "registered"
    assert repo.get(key).payload["domain_tier"] == "1A"
    events = rows(repo, "audit", "event='registration'")
    assert len(events) == 1
    assert events[0]["key"] == key
    assert events[0]["actor"] == "test"
    assert events[0]["detail"]["command_id"] == "registration-1"
    assert events[0]["detail"]["source_identities"][0]["source_ids"] == {"core:work": ["123"]}
    assert len(rows(repo, "jobs", "kind='catalog_export'")) == 1
    assert len(rows(repo, "jobs", "kind='download' AND state='pending'")) == download_budget
    assert queued == ([key] if download_budget else [])
    before = repo.get(key)
    replay = await service.register_candidates([candidate()], actor="test", reason="registration-1")
    assert replay[0]["status"] == "duplicate"
    assert repo.get(key) == before
    assert len(rows(repo, "audit", "event='registration'")) == 1
    assert len(queued) == download_budget
    assert not lib.index_path.exists()


async def test_registration_job_failure_rolls_back_catalog_audit_and_live_publication(
        repo, tmp_path, monkeypatch):
    lib = pg_library(repo, tmp_path, monkeypatch)
    queued = []
    service = RegistrationService(lib, SimpleNamespace(add=queued.append))
    original = repo._write_job

    def fail(job):
        original(job)
        if job.kind == "download":
            raise RuntimeError("injected pending-job failure")

    monkeypatch.setattr(repo, "_write_job", fail)
    outcomes = await service.register_candidates([candidate()], actor="test", reason="rollback-test")
    assert outcomes[0]["status"] == "failed"
    assert list(repo.all()) == []
    assert rows(repo, "audit", "event='registration'") == []
    assert rows(repo, "jobs") == []
    assert lib.keys() == []
    assert lib._catalog_backend.persisted == {}
    assert queued == []


async def test_registration_resolves_fresh_pg_identity_from_another_writer(repo, tmp_path, monkeypatch):
    stale = pg_library(repo, tmp_path / "stale", monkeypatch)
    current = pg_library(repo, tmp_path / "current", monkeypatch)
    first = RegistrationService(current, SimpleNamespace(add=lambda key: None))
    registered = await first.register_candidates([candidate()], actor="test", reason="first-writer")
    second = RegistrationService(stale, SimpleNamespace(add=lambda key: pytest.fail("duplicate enqueue")))
    duplicate = await second.register_candidates([candidate()], actor="test", reason="stale-writer")
    assert duplicate[0]["status"] == "duplicate"
    assert duplicate[0]["key"] == registered[0]["key"]
    assert len(list(repo.all())) == 1
    assert len(stale.keys()) == 1


async def test_registration_reuses_unsaved_worker_identifier_without_creating_twin(
        repo, tmp_path, monkeypatch):
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper, _ = lib.upsert(candidate(doi=""))
    lib.save(force=True)
    paper.doi = "10.1234/worker-filled"
    incoming = candidate(title="Another source title for the already registered work", doi=paper.doi)
    result = await RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [incoming], actor="test", reason="worker-identity")
    assert result[0]["status"] == "duplicate"
    assert result[0]["key"] == paper.key
    assert len(list(repo.all())) == 1
    assert lib.get(paper.key) is paper
    assert paper.doi == "10.1234/worker-filled"


async def test_registration_preserves_unknown_payload_and_concurrent_worker_fields(
        repo, tmp_path, monkeypatch):
    repo.upsert(Record.from_payload("Existing", {"key": "Existing", "title": candidate()["title"],
        "doi": "10.1234/registration", "download_status": "pending", "future": {"keep": 42}}),
        audit=AuditEvent("test", "seed"), expected_revision=0)
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper = lib.get("Existing")
    original = repo.upsert

    def concurrent(record, **kwargs):
        paper.download_status = "metadata_only"  # executor writes while command IO is in progress
        return original(record, **kwargs)

    monkeypatch.setattr(repo, "upsert", concurrent)
    result = await RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [candidate()], actor="test", reason="preservation")
    assert result[0]["status"] == "duplicate"
    assert lib.get("Existing") is paper
    assert paper.download_status == "metadata_only"
    assert repo.get("Existing").payload["future"] == {"keep": 42}
    assert lib._catalog_backend.dirty(lib._papers)


async def test_registration_cannot_override_uncommitted_local_policy(repo, tmp_path, monkeypatch):
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper, _ = lib.upsert(candidate())
    lib.save(force=True)
    before = repo.get(paper.key)
    paper.terminal_state = TerminalState(policy=Policy.EXCLUDED, retrieval=Retrieval.ELIGIBLE,
                                         policy_reasons=("off_domain",))
    outcomes = await RegistrationService(lib, SimpleNamespace(add=lambda key: None)).register_candidates(
        [candidate()], actor="test", reason="policy-race")
    assert outcomes[0]["reason"] == "registration_policy"
    assert repo.get(paper.key) == before
    assert paper.terminal_state.policy == Policy.EXCLUDED


async def test_noop_duplicate_retains_bibliographic_alternative_once_per_command(
        repo, tmp_path, monkeypatch):
    lib = pg_library(repo, tmp_path, monkeypatch)
    service = RegistrationService(lib, SimpleNamespace(add=lambda key: None))
    first = await service.register_candidates([candidate()], actor="test", reason="original")
    key = first[0]["key"]
    before = repo.get(key)
    alternate = candidate(title="Alternative title for the same observed cosmic ray work")
    result = await service.register_candidates([alternate], actor="test", reason="alternate")
    assert result[0]["status"] == "duplicate"
    assert repo.get(key) == before
    events = rows(repo, "audit", "event='registration'")
    assert len(events) == 2
    evidence = next(e for e in events if e["detail"]["command_id"] == "alternate")
    assert evidence["detail"]["observations"][0]["title"] == alternate["title"]
    assert len(rows(repo, "jobs")) == 2  # no-op evidence does not reschedule export/acquisition
    await service.register_candidates([alternate], actor="test", reason="alternate")
    assert len(rows(repo, "audit", "event='registration'")) == 2
    assert repo.get(key) == before


@pytest.mark.parametrize("field,verified,incoming", [
    ("doi", "10.1234/registration", "10.1234/registration"),
    ("arxiv_id", "2401.12345v1", "2401.12345v1"),
])
async def test_registration_rolls_back_staged_twin_when_verified_fill_wins(
        repo, tmp_path, monkeypatch, field, verified, incoming):
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper, _ = lib.upsert(candidate(doi="", title="Independent proton detector observations"))
    lib.save(force=True)
    entered, release = threading.Event(), threading.Event()
    upsert = repo.upsert

    def paused_upsert(record, **kwargs):
        result = upsert(record, **kwargs)
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(repo, "upsert", paused_upsert)
    queued = []
    discovery = candidate(doi="")
    discovery[field] = incoming
    command = asyncio.create_task(RegistrationService(lib, SimpleNamespace(add=queued.append)).register_candidates(
        [discovery], actor="test", reason="pg-save-boundary-race"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert await asyncio.to_thread(lib.fill_verified_identifiers, paper.key, **{field: verified}) == {
            field: verified}
    finally:
        release.set()
        outcomes = await command

    assert outcomes[0]["status"] == "duplicate"
    assert outcomes[0]["key"] == paper.key
    assert lib.keys() == [paper.key]
    assert lib.get(paper.key) is paper
    assert [(r.key, r.payload[field]) for r in repo.all()] == [(paper.key, verified)]
    events = rows(repo, "audit", "event='registration'")
    assert [(e["key"], e["detail"]["outcome"]) for e in events] == [(paper.key, "duplicate")]
    assert events[0]["detail"]["observations"][0][field] == incoming
    assert rows(repo, "jobs", "kind='download'") == []
    assert queued == []
    assert not lib._identity_reservations


@pytest.mark.parametrize("abort", [False, True])
async def test_registration_commit_reservation_is_read_safe_and_released_on_rollback(
        repo, tmp_path, monkeypatch, abort):
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper, _ = lib.upsert(candidate(doi="", title="Independent proton detector observations"))
    lib.save(force=True)
    entered, release = threading.Event(), threading.Event()
    write, depth = repo._write, 0

    @contextmanager
    def paused_commit(audit):
        nonlocal depth
        depth += 1
        try:
            with write(audit):
                yield
                if depth == 1 and audit.event == "registration":
                    entered.set()
                    assert release.wait(5)
                    if abort:
                        raise OSError("injected registration commit failure")
        finally:
            depth -= 1

    monkeypatch.setattr(repo, "_write", paused_commit)
    queued = []
    command = asyncio.create_task(RegistrationService(lib, SimpleNamespace(add=queued.append)).register_candidates(
        [candidate()], actor="test", reason="pg-commit-gap"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert await asyncio.wait_for(asyncio.to_thread(lib.find, doi="10.1234/registration"), 1) is None
        filled = await asyncio.wait_for(asyncio.to_thread(
            lib.fill_verified_identifiers, paper.key, doi="10.1234/registration"), 1)
        assert filled == {}
        assert paper.doi == ""
        assert lib.keys() == [paper.key]
    finally:
        release.set()
        outcomes = await command

    assert not lib._identity_reservations
    if abort:
        assert outcomes[0]["status"] == "failed"
        assert [r.key for r in repo.all()] == [paper.key]
        assert rows(repo, "audit", "event='registration'") == []
        assert rows(repo, "jobs", "kind='download'") == []
        assert queued == []
        assert lib.fill_verified_identifiers(paper.key, doi="10.1234/registration") == {
            "doi": "10.1234/registration"}
    else:
        assert outcomes[0]["status"] == "registered"
        key = outcomes[0]["key"]
        assert key != paper.key
        assert lib.find(doi="10.1234/registration").key == key
        assert [r.key for r in repo.all() if r.doi == "10.1234/registration"] == [key]
        assert len(rows(repo, "audit", "event='registration'")) == 1
        assert [j["key"] for j in rows(repo, "jobs", "kind='download'")] == [key]
        assert queued == [key]
    lib.save(force=True)
    assert len([r for r in repo.all() if r.doi == "10.1234/registration"]) == 1
