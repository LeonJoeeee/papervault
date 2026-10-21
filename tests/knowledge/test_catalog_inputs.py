"""Immutable consumed bytes and their input identities."""

import hashlib

import pytest


def test_captured_input_survives_source_replacement(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore

    source = tmp_path / "source.md"
    source.write_bytes(b"original evidence")
    store = ImmutableInputStore(tmp_path / "captured")
    artifact = store.capture(source.read_bytes(), role="fulltext_evidence")
    source.write_bytes(b"later revision")
    assert store.read(artifact) == b"original evidence"
    assert artifact["sha256"] == hashlib.sha256(b"original evidence").hexdigest()
    assert artifact["byte_length"] == 17


def test_repeated_capture_does_not_rewrite_existing_artifact(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore

    store = ImmutableInputStore(tmp_path)
    first = store.capture(b"retained", role="fulltext_evidence")
    path = tmp_path / first["sha256"]
    before = path.stat().st_mtime_ns
    assert store.capture(b"retained", role="fulltext_evidence") == first
    assert path.stat().st_mtime_ns == before


def test_corrupt_existing_input_is_never_replaced_silently(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore, InputUnavailable

    store = ImmutableInputStore(tmp_path)
    artifact = store.capture(b"retained", role="fulltext_evidence")
    (tmp_path / artifact["sha256"]).write_bytes(b"corrupt!")
    with pytest.raises(InputUnavailable):
        store.capture(b"retained", role="fulltext_evidence")
    with pytest.raises(InputUnavailable):
        store.read(artifact)


def test_input_identity_changes_for_metadata_or_actual_consumed_bytes(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import make_input_manifest

    manifest, revision = make_input_manifest("unit-A", {"abstract": "first"}, [], tier="abstract")
    changed, other = make_input_manifest("unit-A", {"abstract": "second"}, [], tier="abstract")
    assert other != revision
    assert manifest["metadata_revision"] != changed["metadata_revision"]
    _, another_unit = make_input_manifest("unit-B", {"abstract": "first"}, [], tier="abstract")
    assert another_unit != revision
    assert "catalog_revision" not in manifest


def test_input_manifest_binds_section_and_parent_revision():
    from papervault.knowledge.scheduler.catalog_inputs import make_input_manifest

    first, rev1 = make_input_manifest("section-A", {}, [], tier="fulltext",
                                    parent_evidence_revision="parent-1", section_definition_revision=1)
    _, rev2 = make_input_manifest("section-A", {}, [], tier="fulltext",
                                parent_evidence_revision="parent-2", section_definition_revision=1)
    assert first["parent_evidence_revision"] == "parent-1"
    assert first["section_definition_revision"] == 1
    assert rev1 != rev2


def test_artifact_path_cannot_escape_capture_root(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore, InputUnavailable

    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    root = tmp_path / "captured"
    store = ImmutableInputStore(root)
    digest = hashlib.sha256(b"outside").hexdigest()
    with pytest.raises(InputUnavailable):
        store.read({"artifact_id": "sha256:" + digest, "sha256": digest,
                    "byte_length": 7, "locator": str(outside)})


def test_paper_capture_binds_raw_fulltext_and_exact_metadata(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore, capture_paper_input
    from papervault.store.repository import Record

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "extract.md").write_bytes(b"A full text measurement.\n")
    record = Record.from_payload("A", {"key": "A", "title": "First title", "md_path": "extract.md"})
    store = ImmutableInputStore(tmp_path / "captured")
    captured = capture_paper_input(record, "unit-A", store, vault)
    assert captured["tier"] == "fulltext"
    assert captured["text"] == "A full text measurement.\n"
    assert captured["manifest"]["metadata"]["title"] == "First title"
    assert store.read(captured["manifest"]["artifacts"][0]) == b"A full text measurement.\n"


def test_fulltext_to_abstract_fallback_has_new_consumed_identity(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore, capture_paper_input
    from papervault.store.repository import Record

    vault = tmp_path / "vault"
    vault.mkdir()
    path = vault / "extract.md"
    path.write_text("A full text measurement.")
    record = Record.from_payload("A", {"key": "A", "md_path": "extract.md",
        "download_status": "metadata_only", "abstract": "An abstract of the measurement."})
    store = ImmutableInputStore(tmp_path / "captured")
    first = capture_paper_input(record, "unit-A", store, vault)
    path.unlink()
    second = capture_paper_input(record, "unit-A", store, vault)
    assert first["tier"] == "fulltext"
    assert second["tier"] == "abstract"
    assert second["consumed_input_revision"] != first["consumed_input_revision"]
    assert second["text"].startswith("[ABSTRACT ONLY")


def test_source_path_escape_is_rejected_before_capture(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore, capture_paper_input, InputUnavailable
    from papervault.store.repository import Record

    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / "outside").write_text("outside evidence")
    record = Record.from_payload("A", {"key": "A", "md_path": "../outside"})
    with pytest.raises(InputUnavailable):
        capture_paper_input(record, "unit-A", ImmutableInputStore(tmp_path / "captured"), vault)


def test_artifact_location_does_not_become_consumed_input_identity(tmp_path):
    from papervault.knowledge.scheduler.catalog_inputs import ImmutableInputStore, make_input_manifest

    first = ImmutableInputStore(tmp_path / "first").capture(b"same bytes", role="fulltext_evidence")
    second = ImmutableInputStore(tmp_path / "restored").capture(b"same bytes", role="fulltext_evidence")
    _, before = make_input_manifest("unit-A", {}, [first], tier="fulltext")
    _, after = make_input_manifest("unit-A", {}, [second], tier="fulltext")
    assert before == after
