"""Save-cost controls (issue #34): debounce, forced flush, O(1) rotation, crash recovery."""
import json

from papervault.library.models import Paper
from papervault.library.store import Library


def _lib(tmp_path, interval="5"):  # explicit opt-in; shipped default is 0/off
    import os
    os.environ["PAPERVAULT_SAVE_MIN_INTERVAL_S"] = interval
    os.environ["PAPERVAULT_BIB_MIN_INTERVAL_S"] = "60"
    lib = Library(root=tmp_path)
    del os.environ["PAPERVAULT_SAVE_MIN_INTERVAL_S"]
    del os.environ["PAPERVAULT_BIB_MIN_INTERVAL_S"]
    return lib


def _add(lib, key):
    lib._papers[key] = Paper(key=key, title=f"Paper {key}", authors=["A"], year=2020)
    lib._reindex(lib._papers[key])


def _index_keys(tmp_path):
    return set(json.loads((tmp_path / "index.json").read_text())["papers"].keys())


def test_debounce_skips_then_force_flushes(tmp_path):
    lib = _lib(tmp_path)
    _add(lib, "A1")
    lib.save()                       # first save: real (last_save was 0)
    assert _index_keys(tmp_path) == {"A1"}
    _add(lib, "A2")
    lib.save()                       # within interval: skipped, dirty
    assert _index_keys(tmp_path) == {"A1"}
    assert lib._index_dirty is True
    lib.save(force=True)             # drain flush
    assert _index_keys(tmp_path) == {"A1", "A2"}
    assert lib._index_dirty is False


def test_interval_elapsed_saves_again(tmp_path, monkeypatch):
    lib = _lib(tmp_path)
    _add(lib, "A1")
    lib.save()
    _add(lib, "A2")
    monkeypatch.setattr(lib, "_last_index_save", lib._last_index_save - 10)
    lib.save()                       # interval elapsed: real save
    assert _index_keys(tmp_path) == {"A1", "A2"}


def test_rotation_first_validates_then_renames(tmp_path):
    lib = _lib(tmp_path, interval="0")
    _add(lib, "A1")
    lib.save(force=True)             # rotation 1: parse-validated path
    assert lib._rotation_validated is True
    first = (tmp_path / "index.json").read_text()
    _add(lib, "A2")
    lib.save(force=True)             # rotation 2: O(1) rename path
    assert (tmp_path / "index.json.bak").read_text() == first  # bak == previous primary
    assert _index_keys(tmp_path) == {"A1", "A2"}


def test_missing_primary_recovers_from_bak(tmp_path):
    lib = _lib(tmp_path, interval="0")
    _add(lib, "A1")
    lib.save(force=True)
    _add(lib, "A2")
    lib.save(force=True)
    (tmp_path / "index.json").unlink()   # crash shape: rename happened, write didn't
    lib2 = Library(root=tmp_path)
    assert "A1" in lib2._papers          # recovered from .bak (previous primary)
