"""Handoff sequence on a vault copy and an opt-in test database."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from psycopg import sql

from test_persistence import pg_library
from papervault.library.models import Paper
from papervault.library.store import Library


def setup_cutover(repo, tmp_path, monkeypatch):
    from papervault.store.cutover import Cutover

    (tmp_path / "index.json").write_text(json.dumps({"version": 1, "papers": {
        "A": Paper(key="A", title="fixture title").model_dump(mode="json"),
    }}))
    env = tmp_path / ".env"
    env.write_text("OTHER=retained\n")
    # Use the real Library facade, with only the external connection injected.
    pg_library(repo, tmp_path / "unused", monkeypatch).close()
    return Cutover(tmp_path, env, repo)


def test_fenced_flip_export_and_rollback(repo, tmp_path, monkeypatch):
    cutover = setup_cutover(repo, tmp_path, monkeypatch)
    assert cutover.preflight(apply=True)["go"]
    assert cutover.preflight(apply=True)["changed"] == 0
    assert cutover.flip()["difference_rate"] == 0
    state = cutover.state()
    assert state["phase"] == "flipped_fenced" and not state["writes_resumed"]
    assert "PAPERVAULT_CATALOG_AUTHORITY=postgres" in cutover.env.read_text()
    result = cutover.rollback()
    assert result["phase"] == "rolled_back"
    assert "PAPERVAULT_CATALOG_AUTHORITY=json" in cutover.env.read_text()
    assert (tmp_path / "index.json").read_bytes() == (tmp_path / ".catalog-frozen-index.json").read_bytes()


def test_resume_closes_rollback_before_restart_even_when_restart_fails(repo, tmp_path, monkeypatch):
    cutover = setup_cutover(repo, tmp_path, monkeypatch)
    cutover.preflight(apply=True)
    cutover.flip()
    with pytest.raises(RuntimeError, match="restart"):
        cutover.resume(restart=["/bin/false"], health=["/bin/true"], reason="isolated test")
    assert cutover.state()["writes_resumed"]
    with pytest.raises(RuntimeError, match="forward"):
        cutover.rollback()


def test_changed_json_after_preflight_refuses_flip(repo, tmp_path, monkeypatch):
    cutover = setup_cutover(repo, tmp_path, monkeypatch)
    cutover.preflight(apply=True)
    (tmp_path / "index.json").write_text('{"version":1,"papers":{}}')
    with pytest.raises(RuntimeError, match="changed"):
        cutover.flip()
    assert "PAPERVAULT_CATALOG_AUTHORITY" not in cutover.env.read_text()


def test_successful_preflight_can_abort_and_preflight_again(repo, tmp_path, monkeypatch):
    from papervault.store.cutover import Cutover
    from papervault.store.repository import PostgresCatalogRepository

    cutover = setup_cutover(repo, tmp_path, monkeypatch)
    assert cutover.preflight(apply=True)["go"]
    state = cutover.state()
    assert state["phase"] == "preflighted" and not state["writes_resumed"]
    assert repo.connection.execute(sql.SQL("SELECT fence_id FROM {}").format(
        repo._table("writer_fence"))).fetchone()["fence_id"] == state["fence_id"]
    original = cutover.index.read_bytes(), cutover.index.stat().st_mtime_ns
    original_env = cutover.env.read_bytes()

    # Abort through a fresh command, without a frozen post-flip snapshot.
    recovery = Cutover(tmp_path, cutover.env,
                       PostgresCatalogRepository(repo.connection, schema=repo.schema))
    assert recovery.rollback()["phase"] == "rolled_back"
    assert recovery.state()["phase"] == "rolled_back"
    assert not recovery.state()["writes_resumed"]
    assert (cutover.index.read_bytes(), cutover.index.stat().st_mtime_ns) == original
    assert cutover.env.read_bytes() == original_env
    assert not cutover.frozen.exists()
    assert repo.connection.execute(sql.SQL("SELECT fence_id FROM {}").format(
        repo._table("writer_fence"))).fetchone()["fence_id"] is None
    Library(tmp_path, authority="json").close()
    report = recovery.preflight(apply=True)
    assert report["go"] and report["difference_rate"] == 0
    assert recovery.state()["fence_id"] != state["fence_id"]


def test_preflighted_abort_refuses_changed_json_without_releasing_fences(repo, tmp_path, monkeypatch):
    from papervault.store.fencing import WriterFenced

    cutover = setup_cutover(repo, tmp_path, monkeypatch)
    cutover.preflight(apply=True)
    payload = json.loads(cutover.index.read_bytes())
    payload["papers"]["A"]["title"] = "unaccounted external edit"
    cutover.index.write_text(json.dumps(payload))
    original = {path: path.read_bytes() for path in (cutover.index, cutover.env, cutover.state_path)}
    with pytest.raises(RuntimeError, match="changed.*forward recovery required"):
        cutover.rollback()
    assert {path: path.read_bytes() for path in original} == original
    assert not cutover.frozen.exists()
    assert repo.connection.execute(sql.SQL("SELECT fence_id FROM {}").format(
        repo._table("writer_fence"))).fetchone()["fence_id"] == cutover.state()["fence_id"]
    with pytest.raises(WriterFenced):
        Library(tmp_path, authority="json")


def test_interrupted_first_probe_can_rollback_while_writers_still_fenced(repo, tmp_path, monkeypatch):
    from papervault.library.store import Library

    cutover = setup_cutover(repo, tmp_path, monkeypatch)
    cutover.preflight(apply=True)
    save = Library.save

    def interrupted(library, **kwargs):
        save(library, **kwargs)
        raise OSError("interrupted after first committed probe")

    monkeypatch.setattr(Library, "save", interrupted)
    with pytest.raises(OSError):
        cutover.flip()
    assert not cutover.state()["writes_resumed"]
    assert cutover.rollback()["phase"] == "rolled_back"


def test_interrupted_rollback_restarts_idempotently(repo, tmp_path, monkeypatch):
    from papervault.store import cutover as module

    cutover = setup_cutover(repo, tmp_path, monkeypatch)
    cutover.preflight(apply=True)
    cutover.flip()
    setter = module.set_authority
    monkeypatch.setattr(module, "set_authority", lambda *a: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        cutover.rollback()
    monkeypatch.setattr(module, "set_authority", setter)
    assert cutover.rollback()["phase"] == "rolled_back"


@pytest.mark.parametrize("failure", ["integrity", "parity"])
def test_failed_preflight_rollback_preserves_json_and_unfences_export(repo, tmp_path, monkeypatch, failure):
    from papervault.store.adapter import LibraryAdapter
    from papervault.store.cutover import Cutover
    from papervault.store.fencing import WriterFenced, writer_lease
    from papervault.store.importer import import_catalog
    from papervault.store.repository import AuditEvent, PostgresCatalogRepository, Record

    cutover = setup_cutover(repo, tmp_path, monkeypatch)
    import_catalog(LibraryAdapter.from_snapshot(cutover.index), repo)
    ledger = tmp_path / "correction-ledger.json"
    if failure == "integrity":
        ledger.write_text(json.dumps({"version": 1, "policy_rulings": [{
            "action": "exclude", "reason": "dangling test ruling", "revision": 1,
            "keys": ["missing"], "timestamp": "2026-10-18T00:00:00+00:00",
        }]}))
    else:
        repo.upsert(Record.from_payload("B", {"key": "B", "title": "PG-only record"}),
                    expected_revision=0, audit=AuditEvent("test", "seed"))
    original = cutover.index.read_bytes(), cutover.index.stat().st_mtime_ns
    original_env = cutover.env.read_bytes()
    with pytest.raises(RuntimeError, match="delta import failed integrity/parity"):
        cutover.preflight(apply=True)
    assert cutover.state()["phase"] == "preparing"
    assert not cutover.state()["writes_resumed"]
    with pytest.raises(WriterFenced):
        writer_lease(tmp_path, "json")

    # A fresh command must use the durable token to release the repository fence.
    writer = PostgresCatalogRepository(repo.connection, schema=repo.schema)
    record = Record.from_payload("C", {"key": "C", "title": "after recovery"})
    with pytest.raises(WriterFenced):
        writer.upsert(record, expected_revision=0, audit=AuditEvent("test", "write"))
    recovery = Cutover(tmp_path, cutover.env, writer)
    assert recovery.rollback()["phase"] == "rolled_back"
    assert not recovery.state()["writes_resumed"]
    assert (cutover.index.read_bytes(), cutover.index.stat().st_mtime_ns) == original
    assert cutover.env.read_bytes() == original_env
    assert not cutover.frozen.exists()
    library = Library(tmp_path, authority="json")
    library.close()
    assert writer.upsert(record, expected_revision=0, audit=AuditEvent("test", "write"))

    # Correct the bad supporting input; export may then reconcile PG-only rows.
    if failure == "integrity":
        ledger.unlink()
    result = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/export_catalog.py"),
        "--schema", repo.schema, "--output", str(cutover.index), "--verify",
    ], env={**os.environ, "PAPERVAULT_CATALOG_DSN": os.environ["PAPERVAULT_CATALOG_TEST_DSN"]},
        cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["difference_rate"] == 0
    assert report["json_count"] == report["postgres_count"] == (2 if failure == "integrity" else 3)
    assert recovery.preflight(apply=True)["go"]


@pytest.mark.parametrize("phase", ["preparing", "preflighted"])
def test_preflight_abort_retries_after_database_fence_was_cleared(repo, tmp_path, monkeypatch, phase):
    from papervault.store.cutover import Cutover

    cutover = setup_cutover(repo, tmp_path, monkeypatch)
    if phase == "preparing":
        # Interrupt after durable preparing intent and DB fencing, before import.
        import papervault.store.cutover as module
        with monkeypatch.context() as patch:
            patch.setattr(module, "import_catalog", lambda *a: (_ for _ in ()).throw(OSError("interrupted import")))
            with pytest.raises(OSError):
                cutover.preflight(apply=True)
    else:
        cutover.preflight(apply=True)
    original = cutover.index.read_bytes()
    record = cutover._record

    def interrupted(state, **updates):
        if updates.get("phase") == "rolled_back":
            raise OSError("interrupted after DB fence release")
        return record(state, **updates)

    monkeypatch.setattr(cutover, "_record", interrupted)
    with pytest.raises(OSError, match="DB fence release"):
        cutover.rollback()
    assert cutover.state()["phase"] == phase
    assert repo.connection.execute(sql.SQL("SELECT fence_id FROM {}").format(
        repo._table("writer_fence"))).fetchone()["fence_id"] is None
    assert Cutover(tmp_path, cutover.env, repo).rollback()["phase"] == "rolled_back"
    assert cutover.index.read_bytes() == original
    assert cutover.preflight(apply=True)["go"]


def test_preflight_abort_before_schema_or_database_fence_exists(repo, tmp_path, monkeypatch):
    cutover = setup_cutover(repo, tmp_path, monkeypatch)
    repo.connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(repo.schema)))
    with monkeypatch.context() as patch:
        patch.setattr(repo, "ensure_schema", lambda **kw: (_ for _ in ()).throw(OSError("interrupted migration")))
        with pytest.raises(OSError):
            cutover.preflight(apply=True)
    original = cutover.index.read_bytes()
    assert cutover.rollback()["phase"] == "rolled_back"
    assert cutover.index.read_bytes() == original
    assert cutover.preflight(apply=True)["go"]


@pytest.mark.parametrize("phase", ["preparing", "preflighted"])
def test_preflip_rollback_refuses_after_writes_resumed(repo, tmp_path, monkeypatch, phase):
    cutover = setup_cutover(repo, tmp_path, monkeypatch)
    cutover.preflight(apply=True)
    cutover._record(cutover.state(), phase=phase, writes_resumed=True)
    original = {path: path.read_bytes() for path in (cutover.index, cutover.env, cutover.state_path)}
    with pytest.raises(RuntimeError, match="forward recovery required"):
        cutover.rollback()
    assert {path: path.read_bytes() for path in original} == original
    assert repo.connection.execute(sql.SQL("SELECT fence_id FROM {}").format(
        repo._table("writer_fence"))).fetchone()["fence_id"] == cutover.state()["fence_id"]
