"""Library authority boundary, delta flushes, and observable conflicts."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from psycopg import sql
from psycopg.errors import CheckViolation

from test_postgres import count
from papervault.library.models import Paper
from papervault.library.store import Library
from papervault.store.repository import AuditEvent, Record, PendingJob


def pg_library(repo, tmp_path, monkeypatch):
    from papervault.store.persistence import PostgresPersistence

    backend = PostgresPersistence("unused", schema=repo.schema)

    @contextmanager
    def repository():
        yield repo

    monkeypatch.setattr(backend, "repository", repository)
    monkeypatch.setattr(PostgresPersistence, "from_environment", lambda: backend)
    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")
    return Library(tmp_path)


def test_pg_load_save_delta_and_noop_leave_json_untouched(repo, tmp_path, monkeypatch):
    path = tmp_path / "index.json"
    path.write_bytes(b'old export must remain untouched')
    lib = pg_library(repo, tmp_path, monkeypatch)
    assert lib.all_papers() == []
    lib._papers["A"] = Paper(key="A", title="first")
    lib._papers["B"] = Paper(key="B", title="second")
    before = count(repo, "audit")
    lib.save(force=True)
    assert count(repo, "audit") == before + 2
    assert count(repo, "jobs") == 2  # pending export markers in the same transaction
    lib.save(force=True)
    assert count(repo, "audit") == before + 2
    lib.get("A").authors.append("nested edit")
    lib.save(force=True)
    assert repo.get("A").revision == 2
    assert repo.get("B").revision == 1
    assert repo.get("A").payload["authors"] == ["nested edit"]
    assert path.read_bytes() == b'old export must remain untouched'
    assert not (tmp_path / "index.json.bak").exists()
    loaded = pg_library(repo, tmp_path / "reload", monkeypatch)
    assert loaded.get("A").authors == ["nested edit"]


def test_pg_preserves_unknown_payload_and_does_not_flush_load_defaults(repo, tmp_path, monkeypatch):
    repo.upsert(Record.from_payload("A", {"key": "A", "future": {"keep": 42}}),
                expected_revision=0, audit=AuditEvent("test", "seed"))
    lib = pg_library(repo, tmp_path, monkeypatch)
    before = count(repo, "audit")
    lib.save(force=True)
    assert count(repo, "audit") == before
    lib.get("A").title = "new title"
    lib.save(force=True)
    assert repo.get("A").payload == {"key": "A", "title": "new title", "future": {"keep": 42}}


def test_pg_reloads_and_merges_disjoint_metadata_once_but_reports_overlaps(repo, tmp_path, monkeypatch):
    from papervault.store.persistence import CatalogConflict

    repo.upsert(Record.from_payload("A", {"key": "A", "title": "before"}),
                expected_revision=0, audit=AuditEvent("test", "seed"))
    one = pg_library(repo, tmp_path / "one", monkeypatch)
    two = pg_library(repo, tmp_path / "two", monkeypatch)
    one.get("A").title = "new title"
    one.save(force=True)
    two.get("A").venue = "new venue"
    two.save(force=True)
    assert repo.get("A").payload["title"] == two.get("A").title == "new title"
    assert repo.get("A").payload["venue"] == "new venue"
    one.get("A").title = "winner"
    one.save(force=True)
    two.get("A").title = "stale loser"
    with pytest.raises(CatalogConflict) as failure:
        two.save(force=True)
    assert failure.value.keys == ("A",)
    assert two._index_dirty
    assert repo.get("A").payload["title"] == "winner"
    assert '"event": "catalog_conflict"' in two.manifest_path.read_text()
    # Repeated saves cannot silently rebase the rejected edit and overwrite it.
    with pytest.raises(CatalogConflict):
        two.save(force=True)


def test_pg_purge_and_collision_refused_before_mutation_or_unlink(repo, tmp_path, monkeypatch):
    from papervault.store.repository import DeletionRefused

    lib = pg_library(repo, tmp_path, monkeypatch)
    for key in ("A", "B"):
        lib._papers[key] = Paper(key=key, title=key, doi="10.1/A" if key == "A" else "")
        lib._reindex(lib._papers[key])
    lib.save(force=True)
    lib.pdf_path("B").write_bytes(b"retained")
    with pytest.raises(DeletionRefused):
        lib.purge("B")
    with pytest.raises(DeletionRefused):
        lib.set_resolved_doi("B", "10.1/A")
    assert lib.get("B") is not None
    assert lib.pdf_path("B").read_bytes() == b"retained"
    lib._papers.pop("B")
    with pytest.raises(DeletionRefused):
        lib.save(force=True)
    assert repo.get("B") is not None


def test_pg_failed_batch_does_not_advance_snapshot(repo, tmp_path, monkeypatch):
    lib = pg_library(repo, tmp_path, monkeypatch)
    lib._papers["A"] = Paper(key="A")
    repo.connection.execute(sql.SQL(
        "ALTER TABLE {} ADD CONSTRAINT reject_flush CHECK (event <> 'state_flush')"
    ).format(sql.Identifier(repo.schema, "audit")))
    with pytest.raises(CheckViolation):
        lib.save(force=True)
    assert repo.get("A") is None
    repo.connection.execute(sql.SQL("ALTER TABLE {} DROP CONSTRAINT reject_flush").format(
        sql.Identifier(repo.schema, "audit")))
    lib.save(force=True)
    assert repo.get("A").revision == 1


def test_completion_guard_rejects_stale_before_publication(repo, tmp_path, monkeypatch):
    from papervault.store.repository import StaleCompletion

    lib = pg_library(repo, tmp_path, monkeypatch)
    lib._papers["A"] = Paper(key="A")
    lib.save(force=True)
    repo.import_job(PendingJob("reopen:A", "reopen", "A", state="dispatched", attempts=2,
                              lease_id="current"), audit=AuditEvent("test", "queue"))
    token = SimpleNamespace(key="A", generation=1, lease_id="old")
    with pytest.raises(StaleCompletion):
        with lib.catalog_completion(token):
            lib.pdf_path("A").write_bytes(b"stale publication")
    assert not lib.pdf_path("A").exists()


def test_json_default_and_explicit_authority_have_identical_bytes(tmp_path, monkeypatch):
    results = []
    for authority in (None, "json"):
        if authority is None:
            monkeypatch.delenv("PAPERVAULT_CATALOG_AUTHORITY", raising=False)
        else:
            monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", authority)
        lib = Library(tmp_path / str(authority))
        lib._papers["A"] = Paper(key="A", title="test", added_at="2026-10-18T00:00:00")
        lib.save(force=True)
        results.append(lib.index_path.read_bytes())
        assert json.loads(results[-1])["version"] == 1
    assert results[0] == results[1]


def test_invalid_authority_refused_before_creating_vault(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgrez")
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match="PAPERVAULT_CATALOG_AUTHORITY"):
        Library(root)
    assert not root.exists()


@pytest.mark.parametrize("stale", [False, True])
async def test_pg_staged_reopen_uses_json_stage_and_catalog_completion(repo, tmp_path, monkeypatch, stale):
    from papervault.library.claims_v2 import ClaimStore
    from papervault.library.reopen_v2 import StagedAttempt
    from papervault.library.state_v2 import TerminalState, Retrieval, Policy

    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper = Paper(key="A", doi="10.1234/test", download_status="failed",
        terminal_state=TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR))
    lib._papers["A"] = paper
    lib.save(force=True)
    store = ClaimStore(lib)
    campaign = store.ledger.start_campaign(cap=1, keys=["A"])
    token = await store.claim("A", expected_state=paper.acquisition_state,
                              expected_generation=0, campaign_id=campaign)
    assert await store.pre_dispatch(token)
    stage = StagedAttempt(lib, token)
    assert stage.library._catalog_backend is None
    assert set(stage.library._papers) == {"A"}
    stage.path("pdf").write_bytes(b"%PDF staged result")
    stage.library.get("A").download_status = "ok"
    if stale:
        current = repo.get("A")
        repo.upsert(Record.from_payload("A", {**current.payload, "domain_status": "off_domain"}),
                    expected_revision=current.revision, audit=AuditEvent("other", "policy_change"))
    assert await stage.writeback(store) is (not stale)
    if stale:
        assert not lib.pdf_path("A").exists()
        assert repo.get("A").payload["domain_status"] == "off_domain"
    else:
        assert lib.pdf_path("A").read_bytes() == b"%PDF staged result"
        job = repo.connection.execute(sql.SQL("SELECT state FROM {} WHERE id='reopen:A'").format(
            sql.Identifier(repo.schema, "jobs"))).fetchone()
        assert job["state"] == "completed"
        assert repo.get("A").payload["reopen_claim"]["phase"] == "completed"
    stage.cleanup()


async def test_ordinary_save_cannot_bypass_attempt_guard(repo, tmp_path, monkeypatch):
    from papervault.library.claims_v2 import ReopenClaim
    from papervault.store.repository import StaleCompletion

    lib = pg_library(repo, tmp_path, monkeypatch)
    token = ReopenClaim(key="A", generation=2, lease_id="current", lease_expiry=9999999999., campaign_id="c")
    lib._papers["A"] = Paper(key="A", reopen_claim=token)
    lib.save(force=True)
    lib.get("A").reopen_claim = token.model_copy(update={"generation": 1, "lease_id": "stale", "phase": "completed"})
    before = count(repo, "audit")
    with pytest.raises(StaleCompletion):
        lib.save(force=True)
    assert repo.get("A").payload["reopen_claim"]["phase"] == "queued"
    assert count(repo, "audit") == before


def test_completion_multiple_saves_commit_latest_revision_once(repo, tmp_path, monkeypatch):
    from papervault.library.claims_v2 import ReopenClaim

    lib = pg_library(repo, tmp_path, monkeypatch)
    token = ReopenClaim(key="A", generation=1, lease_id="lease", lease_expiry=9999999999., campaign_id="c", phase="dispatched")
    lib._papers["A"] = Paper(key="A", reopen_claim=token)
    lib.save(force=True)
    with lib.catalog_completion(token):
        lib.get("A").venue = "first nested save"
        lib.save(force=True)
        lib.get("A").reopen_claim = token.model_copy(update={"phase": "completed"})
        lib.save(force=True)
    assert repo.get("A").revision == 3
    assert lib._catalog_backend.persisted["A"].revision == 3


def test_pg_flush_keeps_alias_projection_in_sync(repo, tmp_path, monkeypatch):
    from papervault.library.state_v2 import TerminalState, Merge
    from papervault.store.exporter import export_catalog

    lib = pg_library(repo, tmp_path, monkeypatch)
    lib._papers["A"] = Paper(key="A", terminal_state=TerminalState(merge=Merge.ALIAS, alias_key="Z"))
    lib._papers["Z"] = Paper(key="Z")
    lib.save(force=True)
    assert repo.all_aliases() == {"A": "Z"}
    assert export_catalog(repo, tmp_path / "export.json", verify=True)["difference_rate"] == 0


def test_save_acknowledgement_preserves_edits_made_during_commit(repo, tmp_path, monkeypatch):
    lib = pg_library(repo, tmp_path, monkeypatch)
    lib._papers["A"] = Paper(key="A", title="snapshot")
    flush = repo.flush

    def concurrent_edit(*args, **kwargs):
        revisions = flush(*args, **kwargs)
        lib.get("A").download_status = "ok"
        lib.get("A").pdf_path = "pdfs/A.pdf"
        return revisions

    monkeypatch.setattr(repo, "flush", concurrent_edit)
    lib.save(force=True)
    assert lib.get("A").download_status == "ok"
    assert lib.get("A").pdf_path == "pdfs/A.pdf"
    assert lib._index_dirty
    assert repo.get("A").payload["download_status"] == "pending"
    monkeypatch.setattr(repo, "flush", flush)
    lib.save(force=True)
    assert repo.get("A").payload["download_status"] == "ok"


def test_pg_delta_removes_cleared_optional_claim(repo, tmp_path, monkeypatch):
    from papervault.library.claims_v2 import ReopenClaim

    lib = pg_library(repo, tmp_path, monkeypatch)
    claim = ReopenClaim(key="A", generation=1, lease_id="one", lease_expiry=9999999999., campaign_id="c", phase="completed")
    lib._papers["A"] = Paper(key="A", reopen_claim=claim)
    lib.save(force=True)
    lib.get("A").reopen_claim = None
    lib.save(force=True)
    assert "reopen_claim" not in repo.get("A").payload
    assert lib.get("A").reopen_claim is None


async def test_failed_completion_preserves_stage_and_does_not_publish_completed_claim(repo, tmp_path, monkeypatch):
    import time
    from papervault.library.claims_v2 import ClaimStore
    from papervault.library.reopen_v2 import process_attempt
    from papervault.library.services import download_queue
    from papervault.library.state_v2 import TerminalState, Retrieval, Policy

    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    lib = pg_library(repo, tmp_path, monkeypatch)
    paper = Paper(key="A", doi="10.1234/test", download_status="failed",
                  terminal_state=TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR))
    lib._papers["A"] = paper
    lib.save(force=True)
    store = ClaimStore(lib)
    campaign = store.ledger.start_campaign(cap=1, keys=["A"])
    token = await store.claim("A", expected_state=paper.acquisition_state,
                              expected_generation=0, campaign_id=campaign)
    repo.connection.execute(sql.SQL(
        "ALTER TABLE {} ADD CONSTRAINT reject_completion CHECK (event <> 'job_completion')"
    ).format(sql.Identifier(repo.schema, "audit")))

    def download(candidate, stage, **kwargs):
        stage.pdf_path("A").write_bytes(b"%PDF recoverable stage")
        candidate.download_status = "ok"
        return True

    monkeypatch.setattr(download_queue, "download_paper", download)
    queue = SimpleNamespace(library=lib, _reopen_clock=time.time, _reopen_halted=set(), _on_success=None)
    with pytest.raises(CheckViolation):
        await process_attempt(queue, token, 1)
    assert (tmp_path / ".reopen-staging" / token.lease_id / "pdfs/A.pdf").read_bytes() == b"%PDF recoverable stage"
    assert repo.get("A").payload["reopen_claim"]["phase"] == "dispatched"
    assert paper.reopen_claim.phase == "dispatched"


def test_completion_and_save_use_one_reentrant_file_lock(repo, tmp_path, monkeypatch):
    from papervault.library.claims_v2 import ReopenClaim

    lib = pg_library(repo, tmp_path, monkeypatch)
    claim = ReopenClaim(key="A", generation=1, lease_id="lease", lease_expiry=9999999999., campaign_id="c", phase="dispatched")
    lib._papers["A"] = Paper(key="A", reopen_claim=claim)
    lib.save(force=True)
    with lib.catalog_completion(claim):
        # A peer must not take the JSON/BibTeX file lock while this transaction
        # owns the PG lock: otherwise the peer and completion deadlock on save.
        from filelock import FileLock, Timeout
        with pytest.raises(Timeout):
            with FileLock(str(tmp_path / ".write.lock"), timeout=0):
                pass
        lib.get("A").reopen_claim = claim.model_copy(update={"phase": "completed"})
        lib.save(force=True, lock_timeout=0)


def test_stale_optional_clear_is_not_retried_as_metadata(repo, tmp_path, monkeypatch):
    from papervault.library.claims_v2 import ReopenClaim
    from papervault.store.persistence import CatalogConflict

    one = pg_library(repo, tmp_path / "one", monkeypatch)
    claim = ReopenClaim(key="A", generation=1, lease_id="lease", lease_expiry=9999999999., campaign_id="c", phase="completed")
    one._papers["A"] = Paper(key="A", title="old", reopen_claim=claim)
    one.save(force=True)
    two = pg_library(repo, tmp_path / "two", monkeypatch)
    one.get("A").title = "new"
    one.save(force=True)
    two.get("A").reopen_claim = None
    with pytest.raises(CatalogConflict):
        two.save(force=True)
    assert repo.get("A").payload["reopen_claim"]["lease_id"] == "lease"
