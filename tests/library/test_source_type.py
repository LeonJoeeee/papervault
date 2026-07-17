"""Back-compat tests for the removed source_type / book-only fields.

Phase 1 post-ingest redesign (D1/D2, 2026-05-31): books left
paper-library scope. The ``Paper.source_type`` field, the
``SOURCE_TYPE_*`` constants, the ``Chapter`` model, the
``papervault.library.chapters`` module, ``papervault.library.manual_ingest`` and the
``add-textbook`` / ``add-review`` CLI commands were all deleted. Review
handling stays via the bibliometric ``is_review`` field.

Existing ``/data/paper-vault/index.json`` rows still carry a
``source_type`` value (plus the removed ``publisher`` / ``isbn`` /
``edition`` / ``chapters`` columns). The ``Paper`` model declares
``model_config = ConfigDict(extra="ignore")`` so those leftover unknown
keys load cleanly instead of raising. These tests pin that contract.
"""

from __future__ import annotations

import json

from papervault.library import Library, Paper


def test_paper_tolerates_stray_source_type_key():
    """A record dict carrying a stray ``source_type`` (and other removed
    book-only fields) loads cleanly and round-trips — the unknown keys are
    ignored, surviving fields are preserved."""
    record = {
        "key": "Reames2017",
        "title": "Solar Energetic Particles",
        "authors": ["Reames"],
        "year": 2017,
        # Leftover columns from the deleted textbook taxonomy — must be
        # tolerated (extra="ignore"), not raise.
        "source_type": "textbook",
        "publisher": "Springer",
        "isbn": "9783319503714",
        "edition": "1st",
        "chapters": [
            {"chapter_num": 1, "title": "Intro", "page_start": 0, "page_end": 20},
        ],
        "is_review": False,
    }
    p = Paper.model_validate(record)
    assert p.key == "Reames2017"
    assert p.title == "Solar Energetic Particles"
    assert p.year == 2017
    assert p.is_review is False
    # The dropped fields are not surfaced on the model.
    assert not hasattr(p, "source_type")
    assert not hasattr(p, "chapters")
    assert not hasattr(p, "publisher")

    # Round-trips: model_dump → model_validate stays clean (no stray keys
    # re-introduced, no error).
    p2 = Paper.model_validate(p.model_dump())
    assert p2.key == "Reames2017"
    assert "source_type" not in p.model_dump()


def test_library_loads_legacy_index_json_with_source_type(tmp_path):
    """A Library loaded from an index.json whose rows carry the legacy
    ``source_type`` value (and book-only columns) loads without error."""
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "extracts").mkdir()
    legacy_index = {
        "version": 1,
        "papers": {
            "Smith2010": {
                "key": "Smith2010",
                "title": "An older paper from a vault that still has the field",
                "authors": ["Smith"],
                "year": 2010,
                # Pre-Phase-1 columns left on disk:
                "source_type": "research_paper",
                "publisher": "",
                "isbn": "",
                "chapters": [],
            },
        },
    }
    (lib_dir / "index.json").write_text(json.dumps(legacy_index))
    lib = Library(lib_dir)
    p = lib.get("Smith2010")
    assert p is not None
    assert p.title.startswith("An older paper")
    assert not hasattr(p, "source_type")
