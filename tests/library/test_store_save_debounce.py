"""Save-cost controls (issue #34): debounce, forced flush, O(1) rotation, crash recovery."""
import json

from papervault.library.models import Paper
from papervault.library.store import Library


def _lib(tmp_path, interval="5"):  # explicit opt-in; shipped default is 0/off
    import os
    prev = (os.environ.get("PAPERVAULT_SAVE_MIN_INTERVAL_S"),
            os.environ.get("PAPERVAULT_BIB_MIN_INTERVAL_S"))
    os.environ["PAPERVAULT_SAVE_MIN_INTERVAL_S"] = interval
    os.environ["PAPERVAULT_BIB_MIN_INTERVAL_S"] = "60"
    lib = Library(root=tmp_path)
    # restore the conftest session pins (review finding 5: del leaked 60s to later tests)
    os.environ["PAPERVAULT_SAVE_MIN_INTERVAL_S"] = prev[0] if prev[0] is not None else "0"
    os.environ["PAPERVAULT_BIB_MIN_INTERVAL_S"] = prev[1] if prev[1] is not None else "0"
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



def test_flush_if_dirty_writes_only_when_dirty(tmp_path):
    lib = _lib(tmp_path)
    _add(lib, "A1")
    lib.save()
    assert lib.flush_if_dirty() is False       # clean: no write
    _add(lib, "A2")
    lib.save()                                  # debounced -> dirty
    assert lib.flush_if_dirty() is True
    assert _index_keys(tmp_path) == {"A1", "A2"}


def test_out_of_band_corruption_never_clobbers_bak(tmp_path):
    lib = _lib(tmp_path, interval="0")
    _add(lib, "A1")
    lib.save(force=True)                        # rotation validated
    _add(lib, "A2")
    lib.save(force=True)                        # bak now holds the A1 snapshot
    good_bak = (tmp_path / "index.json.bak").read_text()
    (tmp_path / "index.json").write_text('{"version": 1, "papers": {TRUNCATED')  # bit-rot
    _add(lib, "A3")
    lib.save(force=True)                        # sanity gate must PRESERVE .bak
    assert (tmp_path / "index.json.bak").read_text() == good_bak
    assert _index_keys(tmp_path) == {"A1", "A2", "A3"}  # primary rewritten good


def test_first_save_with_corrupt_primary_preserves_bak(tmp_path):
    lib = _lib(tmp_path, interval="0")
    _add(lib, "A1")
    lib.save(force=True)
    _add(lib, "A2")
    lib.save(force=True)
    good_bak = (tmp_path / "index.json.bak").read_text()
    (tmp_path / "index.json").write_text("NOT JSON")
    lib2 = Library(root=tmp_path)               # fresh process: recovers from bak
    assert "A1" in lib2._papers
    lib2.save(force=True)                       # first save, corrupt primary on entry
    assert (tmp_path / "index.json.bak").read_text() == good_bak  # bak untouched
