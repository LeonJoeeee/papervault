"""Fail-closed handoff fencing and rollback boundaries."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from papervault.library.store import Library


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_active_library_fences_cutover_and_exclusive_fence_blocks_new_library(tmp_path):
    from papervault.store.fencing import exclusive_fence, WriterFenced

    lib = Library(tmp_path)
    with pytest.raises(WriterFenced):
        with exclusive_fence(tmp_path):
            pass
    lib.close()
    with exclusive_fence(tmp_path):
        with pytest.raises(WriterFenced):
            Library(tmp_path)


def test_durable_fenced_phase_survives_command_exit(tmp_path):
    from papervault.store.fencing import WriterFenced

    (tmp_path / ".catalog-handoff.json").write_text(json.dumps({
        "phase": "flipped_fenced", "fence_id": "test", "writes_resumed": False,
    }))
    with pytest.raises(WriterFenced):
        Library(tmp_path)


def test_script_requires_explicit_copy_and_rejects_live_before_connecting(tmp_path):
    result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/cutover_catalog.py"), "--flip",
        "--rehearsal", "--vault", "/data/paper-vault", "--env-file", str(tmp_path / ".env")],
        cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "live vault" in result.stdout + result.stderr


def test_env_replaces_duplicates_and_preserves_other_bytes(tmp_path):
    from papervault.store.cutover import set_authority

    env = tmp_path / ".env"
    env.write_bytes(b"# credentials stay local\nOTHER=value\nPAPERVAULT_CATALOG_AUTHORITY=json\nPAPERVAULT_CATALOG_AUTHORITY=postgres\n")
    set_authority(env, "postgres")
    assert env.read_bytes() == (b"# credentials stay local\nOTHER=value\n"
                               b"PAPERVAULT_CATALOG_AUTHORITY=postgres\n")


def test_closed_library_cannot_write_after_releasing_its_fence(tmp_path):
    from papervault.store.fencing import exclusive_fence, WriterFenced

    lib = Library(tmp_path)
    lib.close()
    with exclusive_fence(tmp_path):
        with pytest.raises(WriterFenced):
            lib.save(force=True)


def test_export_cli_obeys_durable_writer_fence_before_connecting(tmp_path):
    import os

    (tmp_path / ".catalog-handoff.json").write_text(json.dumps({
        "phase": "flipped_fenced", "fence_id": "test", "writes_resumed": False,
    }))
    output = tmp_path / "index.json"
    output.write_bytes(b"frozen authority")
    result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/export_catalog.py"), "--output", str(output), "--verify"],
        env={**os.environ, "PAPERVAULT_CATALOG_DSN": "host=127.0.0.1 port=1 dbname=missing"},
        cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert json.loads(result.stdout)["error"] == "WriterFenced"
    assert output.read_bytes() == b"frozen authority"


@pytest.mark.parametrize("phase", ["preparing", "preflighted", "flipped_fenced", "rolled_back", "resumed"])
def test_export_lease_requires_unfenced_phase_without_changing_library_authority(tmp_path, phase):
    from papervault.store.fencing import WriterFenced, writer_lease

    (tmp_path / ".catalog-handoff.json").write_text(json.dumps({
        "phase": phase, "fence_id": "test", "writes_resumed": phase == "resumed",
    }))
    if phase not in {"rolled_back", "resumed"}:
        with pytest.raises(WriterFenced):
            writer_lease(tmp_path, None)
        return
    lease = writer_lease(tmp_path, None)
    lease.close()
    with pytest.raises(WriterFenced):
        writer_lease(tmp_path, "postgres" if phase == "rolled_back" else "json")
