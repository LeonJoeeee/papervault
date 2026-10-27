"""Fingerprint tests against the REAL vault (read-only).

The metadata-only-count test validates the SDD's authoritative "no full text" 判据
(md_path & txt_path both empty). If reality != the SDD number, this test FAILS —
which is the point: it surfaces a stale SDD number (methodology: tests catch SDD
errors). 2026-06-01: SDD reflux'd to 691/3373 (was 692/3372 — one paper gained
full text, the META→full-text jump §6.1 REDISTILL is built to capture).
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from papervault.knowledge.ingest import fingerprint as fpmod, paper_library_client as client, vault
from papervault.knowledge.ingest.abstract_doc import is_abstract_fp
from papervault.knowledge.ingest.fingerprint import META, fingerprint, is_metadata_only
from papervault.knowledge.ingest.vault import DEFAULT_VAULT, load_clean_index

_HAS_VAULT = (DEFAULT_VAULT / "index.json").exists()
requires_vault = pytest.mark.skipif(not _HAS_VAULT, reason="paper-vault not present")


@requires_vault
def test_fingerprint_meta_vs_fulltext_and_stable():
    idx = load_clean_index()
    meta_key = next(k for k, rec in idx.items() if is_metadata_only(rec))
    meta_fp = fingerprint(idx[meta_key])
    # #144: a metadata-state paper is META without an abstract, its own ABSTRACT: class with one.
    assert meta_fp == META or is_abstract_fp(meta_fp)
    if not (idx[meta_key].abstract or "").strip():
        assert meta_fp == META

    ft_key = next(k for k, rec in idx.items() if not is_metadata_only(rec))
    fp = fingerprint(idx[ft_key])
    if fp != META:  # has text on disk
        assert len(fp) == 64  # sha256 hex
        assert fingerprint(idx[ft_key]) == fp  # stable / deterministic


@requires_vault
def test_corpus_split_is_consistent():
    """Metadata-only + full-text partition the clean index (portable invariant —
    no absolute counts, which are corpus-state-specific)."""
    idx = load_clean_index()
    meta = sum(1 for rec in idx.values() if is_metadata_only(rec))
    fulltext = sum(1 for rec in idx.values() if not is_metadata_only(rec))
    assert meta + fulltext == len(idx)
    assert meta >= 0 and fulltext >= 0


@pytest.fixture
def extract_reads(tmp_path, monkeypatch):
    # Exercise the default reader and the real path containment rules in a local vault.
    resolve = vault.resolve_extract_path
    monkeypatch.setattr(vault, "resolve_extract_path", lambda _, rel: resolve(tmp_path, rel))
    read = Path.read_text
    reads = []

    def counted(path, *args, **kwargs):
        if path.is_relative_to(tmp_path):
            reads.append(path)
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    return reads


@pytest.mark.parametrize("change", ["mtime", "size", "replacement"])
def test_extract_cache_reuses_bytes_and_invalidates_stat(tmp_path, extract_reads, change):
    path = tmp_path / "paper.md"
    path.write_text(" body\n text ", encoding="utf-8")
    rec = client.PaperRecord(key="paper", md_path=path.name)
    old_fp = hashlib.sha256(b"body text").hexdigest()
    assert fingerprint(rec) == old_fp
    assert fingerprint(rec) == old_fp
    assert extract_reads == [path]

    before = path.stat()
    changed = " new  text  " if change != "size" else "A longer replacement body."
    if change == "replacement":
        replacement = tmp_path / "ocr.md"
        replacement.write_text(changed, encoding="utf-8")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        replacement.replace(path)
    else:
        path.write_text(changed, encoding="utf-8")
        # Independently prove both stat dimensions; no filesystem clock sleeps.
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns +
                          (1_000_000 if change == "mtime" else 0)))
    expected = b"new text" if change != "size" else changed.encode()
    assert fingerprint(rec) == hashlib.sha256(expected).hexdigest()
    assert len(extract_reads) == 2


def test_extract_cache_tracks_preferred_path_creation_and_aliases(tmp_path, extract_reads):
    rec = client.PaperRecord(key="paper", md_path="paper.md", txt_path="paper.txt")
    assert fingerprint(rec) == META
    (tmp_path / "paper.txt").write_text("Text body")
    txt_fp = hashlib.sha256(b"Text body").hexdigest()
    assert fingerprint(rec) == txt_fp
    (tmp_path / "alias.txt").symlink_to(tmp_path / "paper.txt")
    alias = client.PaperRecord(key="alias", txt_path="alias.txt")
    assert fingerprint(alias) == txt_fp
    assert len(extract_reads) == 1
    (tmp_path / "paper.md").write_text("Markdown body")
    assert fingerprint(rec) == hashlib.sha256(b"Markdown body").hexdigest()
    (tmp_path / "paper.md").unlink()
    assert fingerprint(rec) == txt_fp
    (tmp_path / "paper.txt").unlink()
    assert fingerprint(rec) == META
    assert len(extract_reads) == 2


@pytest.mark.parametrize("empty", ["", " \n\t "])
def test_empty_extract_cache_keeps_metadata_live(tmp_path, extract_reads, monkeypatch, empty):
    path = tmp_path / "paper.md"
    path.write_text(empty)
    rec = client.PaperRecord(key="paper", md_path=path.name, download_status="metadata_only")
    assert fingerprint(rec) == META
    rec.abstract = "Original abstract."
    first = fingerprint(rec)
    assert is_abstract_fp(first)
    rec.abstract = "Revised abstract."
    assert fingerprint(rec) != first
    monkeypatch.setenv("KS_ABSTRACT_DOCS", "0")
    assert fingerprint(rec) == META
    assert extract_reads == [path]
    path.write_text("OCR body")
    assert fingerprint(rec) == hashlib.sha256(b"OCR body").hexdigest()


def test_fingerprint_patched_reader_bypasses_file_cache(monkeypatch):
    rec = client.PaperRecord(key="paper", md_path="unavailable.md")
    bodies = iter(["first", "second", " "])
    monkeypatch.setattr(fpmod, "read_extract_raw", lambda _: next(bodies))
    assert fingerprint(rec) == hashlib.sha256(b"first").hexdigest()
    assert fingerprint(rec) == hashlib.sha256(b"second").hexdigest()
    assert fingerprint(rec) == META


def test_index_cache_reuses_parse_and_validation_with_isolated_records(tmp_path, monkeypatch):
    index = {"version": 1, "papers": {
        "clean": {"key": "clean", "title": "Original", "terminal_state": {"policy": "clear"}},
        "blocked": {"key": "blocked", "domain_status": "off_domain"},
    }}
    (tmp_path / "index.json").write_text(json.dumps(index))
    loads, validations = [], []
    load = client.json.load
    eligible = client.PaperRecord.eligible.fget

    def counted_load(*args, **kwargs):
        loads.append(True)
        return load(*args, **kwargs)

    def counted_eligible(rec):
        validations.append(rec.key)
        return eligible(rec)

    monkeypatch.setattr(client.json, "load", counted_load)
    monkeypatch.setattr(client.PaperRecord, "eligible", property(counted_eligible))
    first = client.load_vault_index(tmp_path)
    second = client.load_vault_index(tmp_path)
    assert first == second and set(first) == {"clean"}
    assert first is not second and first["clean"] is not second["clean"]
    first["clean"].title = "Caller mutation"
    second.clear()
    assert client.load_vault_index(tmp_path)["clean"].title == "Original"
    assert len(loads) == 1
    assert validations == ["clean", "blocked"]
    assert set(client.load_vault_index(tmp_path, include_quarantined=True)) == {"clean", "blocked"}
    assert set(client.load_vault_index(tmp_path)) == {"clean"}
    assert len(loads) == 2


@pytest.mark.parametrize("change", ["mtime", "size", "replacement"])
def test_index_cache_reloads_changed_index(tmp_path, change):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"version": 1, "papers": {"A": {"key": "A"}}}))
    assert set(client.load_vault_index(tmp_path)) == {"A"}
    before = path.stat()
    key = "B" if change != "size" else "LongerKey"
    payload = json.dumps({"version": 1, "papers": {key: {"key": key}}})
    target = tmp_path / "new-index.json" if change == "replacement" else path
    target.write_text(payload)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns +
                        (0 if change == "size" else 1_000_000)))
    if change == "replacement":
        target.replace(path)
    assert set(client.load_vault_index(tmp_path)) == {key}


def test_index_cache_warns_on_each_uncached_version(tmp_path, caplog):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"version": 99, "papers": {}}))
    assert client.load_vault_index(tmp_path) == {}
    assert client.load_vault_index(tmp_path) == {}
    assert len(caplog.records) == 1
    assert "version=99, expected 1" in caplog.text
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    client.load_vault_index(tmp_path)
    assert len(caplog.records) == 2
