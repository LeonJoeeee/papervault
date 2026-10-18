"""Lossless snapshot mapping and read-only Library access."""

import json

import pytest


def test_canonical_payload_hash_preserves_unknown_fields():
    from papervault.store.repository import Record, payload_hash

    assert payload_hash({"b": 2, "a": 1}) == (
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    )
    raw = {"key": "A", "download_status": "ok", "legacy": {"unicode": "宇宙"}}
    record = Record.from_payload("A", raw)
    assert record.payload == raw
    assert record.retrieval == "acquired"
    assert record.policy == "unknown"
    raw["legacy"]["unicode"] = "changed"
    assert record.payload["legacy"]["unicode"] == "宇宙"
    assert record.payload_sha256 != payload_hash(raw)


def test_mapping_retains_explicit_v2_axes_and_versioned_identity():
    from papervault.store.repository import Record

    record = Record.from_payload("Alias", {
        "key": "Alias", "doi": " 10.1/ABC ", "arxiv_id": " 1234.5678v2 ",
        "terminal_state": {"retrieval": "exhausted", "policy": "excluded",
                           "merge": "alias", "quarantine": True, "alias_key": "A"},
    })
    assert (record.doi, record.arxiv_id) == ("10.1/abc", "1234.5678v2")
    assert (record.retrieval, record.policy, record.merge, record.quarantine) == (
        "exhausted", "excluded", "alias", True,
    )


def test_library_adapter_includes_quarantined_records_without_writing(tmp_path, monkeypatch):
    from papervault.library.store import Library
    from papervault.store.adapter import LibraryAdapter

    path = tmp_path / "index.json"
    path.write_text(json.dumps({"version": 1, "papers": {
        "A": {"key": "A", "domain_status": "off_domain", "legacy": 42},
        "B": {"key": "B"},
    }}))
    before = path.read_bytes()
    monkeypatch.setattr(Library, "__init__", lambda *a, **k: pytest.fail("writable constructor"))
    adapter = LibraryAdapter.from_snapshot(path)
    assert [record.key for record in adapter.all()] == ["A", "B"]
    assert adapter.get("A").payload["legacy"] == 42
    assert adapter.get("A").policy == "excluded"
    assert adapter.get("missing") is None
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("text", [
    '{}', '{"papers": []}', '{"version": 999, "papers": {}}',
    '{"papers": {"A": {"key": "B"}}}',
    '{"papers": {"A": {"key": "A"}, "A": {"key": "A"}}}',
    '{"papers": {"A": {"key": "A", "year": "invalid"}}}',
])
def test_malformed_snapshot_is_rejected_without_creating_files(tmp_path, text):
    from papervault.store.adapter import LibraryAdapter

    path = tmp_path / "snapshot.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        LibraryAdapter.from_snapshot(path)
    assert list(tmp_path.iterdir()) == [path]
