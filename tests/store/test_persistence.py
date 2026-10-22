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


@pytest.fixture(params=[
    ("0", "0", "0"),
    ("1", "0", "0"),
    ("0", "1", "0"),
    ("0", "0", "1"),
    ("1", "1", "1"),
], ids=["legacy", "terminal", "reopen", "merge", "all-v2"])
def acquisition_flags(request, monkeypatch):
    for name, value in zip(("TERMINAL", "REOPEN", "MERGE"), request.param):
        monkeypatch.setenv(f"PAPERVAULT_{name}_V2", value)
    return "1" in request.param


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


def test_pg_preserves_unknown_payload_and_does_not_flush_load_defaults(repo, tmp_path, monkeypatch, acquisition_flags):
    repo.upsert(Record.from_payload("A", {"key": "A", "future": {"keep": 42}}),
                expected_revision=0, audit=AuditEvent("test", "seed"))
    lib = pg_library(repo, tmp_path, monkeypatch)
    before = count(repo, "audit")
    lib.save(force=True)
    assert count(repo, "audit") == before
    assert repo.get("A").revision == 1
    assert repo.get("A").payload == {"key": "A", "future": {"keep": 42}}
    lib.get("A").title = "new title"
    lib.save(force=True)
    assert repo.get("A").payload == {"key": "A", "title": "new title", "future": {"keep": 42}}
    assert repo.get("A").revision == 2
    assert count(repo, "audit") == before + 1
    lib.save(force=True)
    assert count(repo, "audit") == before + 1


def test_pg_reloads_and_merges_disjoint_metadata_once_but_reports_overlaps(repo, tmp_path, monkeypatch, acquisition_flags):
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


def test_pg_purge_refused_and_collision_retains_alias_or_waits(repo, tmp_path, monkeypatch, acquisition_flags):
    from papervault.config import merge_v2_enabled
    from papervault.store.repository import DeletionRefused

    lib = pg_library(repo, tmp_path, monkeypatch)
    for key in ("A", "B"):
        lib._papers[key] = Paper(key=key, title=key, doi="10.1/A" if key == "A" else "")
        lib._reindex(lib._papers[key])
    lib.save(force=True)
    lib.pdf_path("B").write_bytes(b"retained")
    before = {key: paper.model_dump(mode="json") for key, paper in lib._papers.items()}
    audit_before = count(repo, "audit")
    with pytest.raises(DeletionRefused):
        lib.purge("B")
    assert {key: paper.model_dump(mode="json") for key, paper in lib._papers.items()} == before
    assert count(repo, "audit") == audit_before
    assert not lib.manifest_path.exists()
    outcome = lib.set_resolved_doi("B", "10.1/A")
    if merge_v2_enabled():
        assert outcome == ("collision", "A")
        assert repo.resolve_alias("B") == "A"
        assert repo.get("B").merge == "alias"
        assert repo.get("B").payload["terminal_state"]["alias_key"] == "A"
        assert count(repo, "audit") > audit_before
        assert repo.connection.execute(sql.SQL(
            "SELECT count(*) AS n FROM {} WHERE key='B' AND event='identity_merge'"
        ).format(sql.Identifier(repo.schema, "audit"))).fetchone()["n"] == 1
    else:
        assert outcome == ("merge_waiting", "A")
        assert {key: paper.model_dump(mode="json") for key, paper in lib._papers.items()} == before
        assert count(repo, "audit") == audit_before
        assert repo.all_aliases() == {}
    assert lib.pdf_path("B").read_bytes() == b"retained"
    lib._papers.pop("B")
    with pytest.raises(DeletionRefused):
        lib.save(force=True)
    assert repo.get("B") is not None


def test_pg_status_write_persists_observed_state(repo, tmp_path, monkeypatch, acquisition_flags):
    repo.upsert(Record.from_payload("A", {"key": "A", "future": {"keep": 42}}),
                expected_revision=0, audit=AuditEvent("test", "seed"))
    lib = pg_library(repo, tmp_path, monkeypatch)
    lib.get("A").download_status = "failed"
    lib.save(force=True)
    payload = repo.get("A").payload
    assert payload["download_status"] == "failed"
    assert payload["future"] == {"keep": 42}
    if acquisition_flags:
        assert payload["terminal_state"]["retrieval"] == "exhausted"
        assert payload["terminal_state"]["cause"] == "retrieval_exhausted"
    else:
        assert "terminal_state" not in payload


@pytest.mark.parametrize("deadline", [0.0, 1e20], ids=["expired", "waiting"])
def test_pg_resumed_doi_collision_waits_if_expired_and_otherwise_aliases(repo, tmp_path, monkeypatch, deadline):
    from papervault.library.merge_v2 import MergeIntent

    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    intent = MergeIntent(merge_id="persisted", canonical_key="A", absorbed_key="B",
                         started_at=0.0, deadline=deadline)
    for key in ("A", "B"):
        paper = Paper(key=key, title=key, doi="10.1/A" if key == "A" else "",
                      merge_intent=intent)
        repo.upsert(Record.from_payload(key, paper.model_dump(mode="json")),
                    expected_revision=0, audit=AuditEvent("test", "seed"))
    lib = pg_library(repo, tmp_path, monkeypatch)
    lib.pdf_path("B").write_bytes(b"retained")
    audit_before = count(repo, "audit")
    outcome = lib.set_resolved_doi("B", "10.1/A")
    if deadline == 0.0:
        assert outcome == ("merge_waiting", "A")
        assert repo.all_aliases() == {}
        assert repo.get("B").merge == "canonical"
        assert repo.get("B").payload["merge_intent"]["status"] == "expired"
        assert not (tmp_path / "reopen-budget.jsonl").exists()
    else:
        assert outcome == ("collision", "A")
        assert repo.resolve_alias("B") == "A"
        assert repo.get("B").merge == "alias"
        assert repo.get("B").payload.get("merge_intent") is None
        assert (tmp_path / "reopen-budget.jsonl").exists()
    assert count(repo, "audit") > audit_before
    assert lib.pdf_path("B").read_bytes() == b"retained"


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


def test_pg_merge_transfers_resolved_identifiers_without_erasing_alias_evidence(repo, tmp_path, monkeypatch):
    from papervault.library.merge_v2 import MergeStore
    from papervault.library.services import concurrency

    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    lib = pg_library(repo, tmp_path, monkeypatch)
    lib._papers["A"] = Paper(key="A", title="canonical", doi="10.1/canonical")
    lib._papers["B"] = Paper(key="B", title="observation", arxiv_id="2401.12345")
    for paper in lib._papers.values():
        lib._reindex(paper)
    lib.save(force=True)
    repo.finalize_identity_indexes(audit=AuditEvent("test", "identity_indexes"))
    lib.pdf_path("B").write_bytes(b"retained source")
    before = count(repo, "audit")

    async def merge():
        async with concurrency.lib_write_lock:
            return MergeStore(lib)._merge_records("A", "B")

    import asyncio
    assert asyncio.run(merge()).key == "A"
    canonical, alias = repo.get("A"), repo.get("B")
    assert canonical.arxiv_id == "2401.12345"
    assert alias.arxiv_id is None
    assert alias.doi is None
    assert alias.payload["arxiv_id"] == "2401.12345"
    assert alias.payload["title"] == "observation"
    assert repo.resolve_alias("B") == "A"
    assert repo.identity_index_fields() == {"doi", "arxiv_id"}
    assert count(repo, "audit") > before
    assert lib.pdf_path("B").read_bytes() == b"retained source"
    assert lib.pdf_path("A").read_bytes() == b"retained source"


def test_pg_distinct_identity_merge_refusal_retains_unique_owner_reservation(repo, tmp_path, monkeypatch):
    from psycopg.errors import UniqueViolation
    from papervault.library.merge_v2 import MergeStore
    import asyncio

    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    lib = pg_library(repo, tmp_path, monkeypatch)
    lib._papers["A"] = Paper(key="A", arxiv_id="2401.12345v1")
    lib._papers["B"] = Paper(key="B", arxiv_id="2401.12345v2")
    lib.save(force=True)
    repo.finalize_identity_indexes(audit=AuditEvent("test", "identity_indexes"))
    with pytest.raises(ValueError, match="conflicting resolved"):
        asyncio.run(MergeStore(lib).merge_records("A", "B"))
    assert repo.all_aliases() == {}
    assert repo.get("B").arxiv_id == "2401.12345v2"
    with pytest.raises(UniqueViolation):
        repo.upsert(Record.from_payload("C", {"key": "C", "arxiv_id": "2401.12345v2"}),
                    expected_revision=0, audit=AuditEvent("test", "stale_registration"))


@pytest.mark.parametrize("source", ["core", "openalex", "operator_document"])
@pytest.mark.parametrize("export", [None, {"version": 1, "papers": {}}], ids=["absent-export", "stale-export"])
def test_pg_policy_ruling_uses_authority_and_commits_history(repo, tmp_path, monkeypatch, source, export):
    from papervault.library.services.policy_ruling import apply_ruling, inspect_ruling
    from papervault.library.state_v2 import Policy, Retrieval, TerminalState

    lib = pg_library(repo, tmp_path, monkeypatch)
    lib._papers["A"] = Paper(key="A", title="arbitrary source", source=source,
        download_status="metadata_only", terminal_state=TerminalState(
            policy=Policy.UNKNOWN, retrieval=Retrieval.EXHAUSTED))
    lib.save(force=True)
    if export is not None:
        lib.index_path.write_text(json.dumps(export))
    original_export = lib.index_path.read_bytes() if export is not None else None
    before = count(repo, "audit")
    assert inspect_ruling(tmp_path, keys=["A"], action="acquire", reason="operator reviewed")["matched_keys"] == ["A"]
    result = apply_ruling(tmp_path, keys=["A"], action="acquire", reason="operator reviewed")
    assert result["applied"] == result["rulings_added"] == 1
    record = repo.get("A")
    assert record.policy == "override"
    assert record.payload["terminal_state"]["override"]["key"] == "A"
    assert record.payload["terminal_state"]["override"]["scope"] == ["acquire"]
    ruling = repo.list_rulings()
    assert len(ruling) == 1
    assert ruling[0]["keys"] == ["A"]
    assert ruling[0]["reason"] == "operator reviewed"
    assert count(repo, "audit") > before
    assert repo.connection.execute(sql.SQL(
        "SELECT count(*) AS n FROM {} WHERE event='policy_ruling' AND detail->>'reason'='operator reviewed'"
    ).format(sql.Identifier(repo.schema, "audit"))).fetchone()["n"] >= 1
    replay_audit = count(repo, "audit")
    replay = apply_ruling(tmp_path, keys=["A"], action="acquire", reason="operator reviewed")
    assert replay["applied"] == replay["rulings_added"] == 0
    assert count(repo, "rulings") == 1
    assert count(repo, "audit") == replay_audit
    assert (lib.index_path.read_bytes() if lib.index_path.exists() else None) == original_export


def test_pg_policy_ruling_audit_failure_rolls_back_state_history_jobs_and_snapshot(repo, tmp_path, monkeypatch):
    from papervault.library.services.policy_ruling import apply_ruling
    from papervault.library.state_v2 import Policy, Retrieval, TerminalState

    lib = pg_library(repo, tmp_path, monkeypatch)
    lib._papers["A"] = Paper(key="A", source="publisher", download_status="metadata_only",
        terminal_state=TerminalState(policy=Policy.UNKNOWN, retrieval=Retrieval.EXHAUSTED))
    lib.save(force=True)
    backend = lib._catalog_backend
    before = repo.get("A")
    audit_before, jobs_before = count(repo, "audit"), count(repo, "jobs")
    # The record audit succeeds first; reject the ruling's own audit after its
    # INSERT, proving both prior writes roll back with the failed command.
    repo.connection.execute(sql.SQL(
        "ALTER TABLE {} ADD CONSTRAINT reject_ruling_audit "
        "CHECK (event <> 'policy_ruling' OR key IS NOT NULL)"
    ).format(sql.Identifier(repo.schema, "audit")))
    with pytest.raises(CheckViolation):
        apply_ruling(tmp_path, keys=["A"], action="acquire", reason="reviewed")
    assert repo.get("A") == before
    assert count(repo, "rulings") == 0
    assert count(repo, "audit") == audit_before
    assert count(repo, "jobs") == jobs_before
    assert backend.persisted["A"] == before
    assert backend.normalized["A"]["terminal_state"]["policy"] == "unknown"
    assert not (tmp_path / "correction-ledger.json").exists()


def test_pg_stale_policy_operator_cannot_replace_competing_ruling(repo, tmp_path, monkeypatch):
    from dataclasses import replace
    from papervault.library.services.operator_actions import stage_operator
    from papervault.library.services.policy_ruling import apply_ruling
    from papervault.library.state_v2 import Policy, Retrieval, TerminalState
    from papervault.store.persistence import CatalogConflict

    repo.upsert(Record.from_payload("A", Paper(key="A", source="publisher",
        download_status="metadata_only", terminal_state=TerminalState(
            policy=Policy.UNKNOWN, retrieval=Retrieval.EXHAUSTED)).model_dump(mode="json")),
        expected_revision=0, audit=AuditEvent("test", "seed"))
    stale = pg_library(repo, tmp_path / "stale", monkeypatch)
    pg_library(repo, tmp_path / "winner", monkeypatch)
    apply_ruling(tmp_path / "winner", keys=["A"], action="exclude", reason="current restriction")
    winner = repo.get("A")
    history_before, audit_before = repo.list_rulings(), count(repo, "audit")
    stage_operator(stale, action="clear", keys=["A"], reason="stale permission",
        event="policy_ruling", row={"timestamp": "2026-10-22T00:00:00+00:00",
            "action": "clear", "scope": "policy", "keys": ["A"], "revision": 1,
            "reason": "stale permission", "actor": "operator", "event": "policy_ruling"})
    stale.get("A").terminal_state = replace(stale.get("A").acquisition_state, policy=Policy.CLEAR)
    with pytest.raises(CatalogConflict):
        stale.save(force=True)
    assert repo.get("A") == winner
    assert winner.policy == "excluded"
    assert repo.list_rulings() == history_before
    assert count(repo, "audit") == audit_before
    assert stale._catalog_backend.persisted["A"].revision == 1


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
