"""Handoff sequence on a vault copy and an opt-in test database."""
import json

import pytest

from test_persistence import pg_library
from papervault.library.models import Paper


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
