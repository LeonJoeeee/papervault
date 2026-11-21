"""Regression tests for Batch 2 persistence P0, pooling, memory/perf, schema, and ExtractionStore."""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from papervault.library.models import Paper
from papervault.library.store import Library
from papervault.store.extraction import ExtractionStore
from papervault.store.persistence import CatalogConflict, PostgresPersistence
from papervault.store.repository import PostgresCatalogRepository, Record, StaleWrite


class _MemoryRepo:
    """Minimal in-memory repository exercising PostgresPersistence retry/conflict paths."""

    def __init__(self, initial: dict[str, Record] | None = None):
        self.records: dict[str, Record] = dict(initial or {})
        self.flushes: list[list[str]] = []

    @contextmanager
    def _write(self, event):
        snapshot = dict(self.records)
        try:
            yield
        except Exception:
            self.records = snapshot
            raise

    def get(self, key: str) -> Record | None:
        return self.records.get(key)

    def all(self):
        return iter(self.records.values())

    def flush(self, records, *, expected_revisions, audit):
        self.flushes.append([r.key for r in records])
        for rec in records:
            cur = self.records.get(rec.key)
            exp = expected_revisions[rec.key]
            if (cur is None and exp != 0) or (cur is not None and cur.revision != exp):
                raise StaleWrite(rec.key, exp, cur.revision if cur else None)
        out = {}
        for rec in records:
            rev = expected_revisions[rec.key] + 1
            self.records[rec.key] = replace(rec, revision=rev)
            out[rec.key] = rev
        return out


def _backend_for(repo: _MemoryRepo) -> PostgresPersistence:
    backend = PostgresPersistence("postgresql://unused/test")

    @contextmanager
    def _repo_ctx():
        yield repo

    backend.repository = _repo_ctx  # type: ignore[method-assign]
    return backend


def test_catalog_conflict_retry_does_not_flag_brand_new_insert():
    """New inserts (rev=0, remote=None) survive a sibling mergeable StaleWrite retry."""
    initial_a = replace(Record.from_payload("A", {"key": "A", "title": "old title"}), revision=1)
    repo = _MemoryRepo({"A": initial_a})
    backend = _backend_for(repo)
    papers = backend.load()

    # Concurrent remote writer updates A.title (mergeable field) -> rev 2.
    repo.records["A"] = replace(
        Record.from_payload("A", {"key": "A", "title": "remote title"}), revision=2,
    )

    # Local writer edits disjoint mergeable field on A AND inserts brand-new B.
    papers["A"].venue = "ApJ"
    papers["B"] = Paper(key="B", title="Brand new paper inserted in same batch")

    assert backend.save(papers) == []
    assert repo.get("A").revision == 3
    assert repo.get("A").payload["title"] == "remote title"
    assert repo.get("A").payload["venue"] == "ApJ"
    assert repo.get("B") is not None
    assert repo.get("B").revision == 1
    assert not backend.dirty(papers)


def test_single_key_catalog_conflict_flushes_unrelated_dirty_keys():
    """One non-mergeable conflicted key must not permanently wedge unrelated dirty papers."""
    initial_a = replace(Record.from_payload("A", {"key": "A", "title": "v1"}), revision=1)
    initial_b = replace(Record.from_payload("B", {"key": "B", "title": "b1"}), revision=1)
    repo = _MemoryRepo({"A": initial_a, "B": initial_b})
    backend = _backend_for(repo)
    papers = backend.load()

    # Remote writer makes a conflicting edit on A.title -> rev 2.
    repo.records["A"] = replace(Record.from_payload("A", {"key": "A", "title": "remote"}), revision=2)

    # Local writer conflicts on A.title, updates unrelated B, and adds new C.
    papers["A"].title = "local-conflict"
    papers["B"].venue = "JGR"
    papers["C"] = Paper(key="C", title="Unrelated new paper")

    with pytest.raises(CatalogConflict) as exc_info:
        backend.save(papers)

    assert exc_info.value.keys == ("A",)
    # Unrelated B and new C were flushed and accepted cleanly despite A's conflict.
    assert repo.get("B").revision == 2
    assert repo.get("B").payload["venue"] == "JGR"
    assert repo.get("C") is not None and repo.get("C").revision == 1
    assert backend.persisted["B"].revision == 2
    assert backend.persisted["C"].revision == 1
    # A remains conflicted and remote winner is untouched.
    assert repo.get("A").payload["title"] == "remote"

    # A subsequent save with another new paper D still flushes D cleanly without wedging.
    papers["D"] = Paper(key="D", title="Another new paper after A conflicted")
    with pytest.raises(CatalogConflict) as exc_info2:
        backend.save(papers)
    assert exc_info2.value.keys == ("A",)
    assert repo.get("D") is not None and repo.get("D").revision == 1


def test_pg_library_skips_state_v2_fingerprints_double_dirty_full_reindex_and_noop_bib(tmp_path, monkeypatch):
    """Under PG authority: skip _state_v2_fingerprints, skip second dirty(), skip no-op bib, incremental reindex."""
    rec_a = replace(Record.from_payload("A", {"key": "A", "title": "Paper A", "doi": "10.1/a"}), revision=1)
    rec_b = replace(Record.from_payload("B", {"key": "B", "title": "Paper B", "doi": "10.1/b"}), revision=1)
    repo = _MemoryRepo({"A": rec_a, "B": rec_b})
    backend = _backend_for(repo)

    monkeypatch.setattr(PostgresPersistence, "from_environment", lambda: backend)
    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")

    lib = Library(tmp_path)
    try:
        # Dead _state_v2_fingerprints is not populated under PG.
        assert lib._state_v2_fingerprints == {}

        # Guard against redundant full-catalog dirty() calls inside Library.save().
        dirty_calls = []
        orig_dirty = backend.dirty

        def spy_dirty(papers):
            dirty_calls.append(True)
            return orig_dirty(papers)

        monkeypatch.setattr(backend, "dirty", spy_dirty)

        bib_calls = []
        orig_bib = lib._save_bib

        def spy_bib():
            bib_calls.append(True)
            return orig_bib()

        monkeypatch.setattr(lib, "_save_bib", spy_bib)

        # No-op non-forced save: skips dirty() AND skips _save_bib().
        lib.save(force=False, force_bib=False)
        assert dirty_calls == []
        assert bib_calls == []

        # Edit only A: incremental reindex touches ONLY A (not B), dirty() still not called.
        reindexed_batches = []
        orig_reindex_keys = lib._reindex_keys

        def spy_reindex_keys(keys):
            reindexed_batches.append(set(keys))
            return orig_reindex_keys(keys)

        monkeypatch.setattr(lib, "_reindex_keys", spy_reindex_keys)
        lib.get("A").doi = "10.1/a-new"
        lib.save(force=False, force_bib=False)

        assert dirty_calls == []
        assert reindexed_batches == [{"A"}]
        assert lib._by_doi == {"10.1/a-new": "A", "10.1/b": "B"}
        assert len(bib_calls) == 1

        # Subsequent no-op non-forced save skips bib again; explicit force_bib=True still writes bib.
        lib.save(force=False, force_bib=False)
        assert len(bib_calls) == 1
        lib.save(force=True)
        assert len(bib_calls) == 2
    finally:
        lib.close()


def test_persistence_lazy_pool_and_monkeypatched_connect_fallback(monkeypatch):
    """PostgresPersistence never connects in __init__, honors monkeypatched psycopg.connect, and close() is safe."""
    import psycopg

    backend = PostgresPersistence("postgresql://never-opened/db")
    assert backend._pool is None

    seen = []

    class FakeConn:
        autocommit = True
        row_factory = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_connect(dsn, **kwargs):
        seen.append((dsn, kwargs))
        return FakeConn()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    with backend.repository() as repo:
        assert isinstance(repo, PostgresCatalogRepository)
    assert len(seen) == 1
    assert backend._pool is None
    backend.close()
    backend.close()  # idempotent


def test_completion_avoids_deepcopy_of_normalized(monkeypatch):
    """PostgresPersistence.completion() uses copy-on-write shallow dicts instead of deepcopy."""
    import copy

    rec_a = replace(Record.from_payload("A", {"key": "A", "title": "Paper A"}), revision=1)
    repo = _MemoryRepo({"A": rec_a})
    repo.check_completion = lambda *a, **kw: {"state": "dispatched"}
    backend = _backend_for(repo)
    papers = backend.load()

    deepcopies = []
    orig_deepcopy = copy.deepcopy

    def spy_deepcopy(obj, *args, **kwargs):
        deepcopies.append(type(obj))
        return orig_deepcopy(obj, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", spy_deepcopy)
    token = SimpleNamespace(key="A", generation=1, lease_id="lease-1")
    with backend.completion(token):
        assert backend._working is not None
        # No deepcopy happened while entering completion()
        assert deepcopies == []
        papers["A"].venue = "Updated Venue"
        backend.save(papers)
    assert repo.get("A").payload["venue"] == "Updated Venue"


def test_schema_contains_alias_and_registration_audit_indexes():
    from importlib.resources import files
    from papervault.store.repository import render_catalog_ddl

    ddl = files("papervault.store").joinpath("schema.sql").read_text()
    assert "records_alias_idx" in ddl
    assert "WHERE merge = 'alias'" in ddl
    assert "audit_event_actor_cmd_idx" in ddl
    assert "WHERE event = 'registration'" in ddl
    rendered = render_catalog_ddl(ddl, "catalog_test")
    assert '"catalog_test".records' in rendered
    assert '"catalog_test".audit' in rendered


def test_repository_caches_to_regclass_and_clears_on_ensure_schema():
    queries = []

    class FakeResult:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class FakeConn:
        autocommit = True
        row_factory = None

        def execute(self, sql_stmt, params=None):
            queries.append((str(sql_stmt), params))
            return FakeResult({"name": "catalog.extract_deletions"})

        @contextmanager
        def transaction(self):
            yield

    repo = PostgresCatalogRepository(FakeConn(), schema="catalog")
    assert repo._table_exists("extract_deletions") is True
    assert repo._table_exists("extract_deletions") is True
    assert repo._table_exists("extract_deletions") is True
    # Only ONE SELECT to_regclass executed despite 3 checks!
    assert len(queries) == 1
    repo._regclass_cache.clear()
    assert repo._table_exists("extract_deletions") is True
    assert len(queries) == 2


def test_extraction_store_hardlinks_large_pdf_and_caches_completions_index(tmp_path, monkeypatch):
    """ExtractionStore.capture() hardlinks same-device PDFs >= threshold and caches completions(key=...)."""
    store = ExtractionStore(tmp_path)
    pdf_bytes = b"%PDF-1.4\n" + (b"0123456789abcdef" * 5000)  # ~80 KiB >= 64 KiB
    pdf_path = tmp_path / "pdfs" / "Smith2024.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)

    recipe = {"schema_version": 1, "engine": "mineru2.5-pro", "configured_model": "test",
              "output_contract": {"fulltext": {"required": ["markdown"], "not_applicable": []}}}
    req = store.capture(
        record_key="Smith2024", metadata={"key": "Smith2024"}, source=pdf_bytes, source_role="pdf",
        recipe=recipe, guard={"previous_completion_id": None, "identity": {"key": "Smith2024"}},
        unit_evidence={"kind": "paper", "record_key": "Smith2024"},
    )
    input_path = tmp_path / req["input_manifest"]["artifacts"][0]["path"]
    assert input_path.read_bytes() == pdf_bytes
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == req["input_manifest"]["artifacts"][0]["sha256"]
    # On ext4 (where FICLONE is unsupported), os.link shares the inode; on Btrfs/XFS reflink creates CoW copy.
    # Either way, atomic replacement of pdf_path must keep input_path intact!
    pdf_tmp = pdf_path.with_suffix(".pdf.tmp")
    pdf_tmp.write_bytes(b"%PDF-1.4 replaced atomically")
    os.replace(pdf_tmp, pdf_path)
    assert input_path.read_bytes() == pdf_bytes

    # Finalize two completions for different keys, then verify completions(key=...) cache avoids reading other files.
    c1 = store.finalize(req, b"# Smith2024 extracted body")
    req2 = store.capture(
        record_key="Jones2024", metadata={"key": "Jones2024"}, source=b"%PDF small", source_role="pdf",
        recipe=recipe, guard={"previous_completion_id": None, "identity": {"key": "Jones2024"}},
        unit_evidence={"kind": "paper", "record_key": "Jones2024"},
    )
    c2 = store.finalize(req2, b"# Jones2024 extracted body")

    # Prime the index cache once.
    assert [c["completion_id"] for c in store.completions(key="Smith2024")] == [c1["completion_id"]]

    # On subsequent per-key lookups, unrelated completion files are NOT read at all!
    reads = []
    orig_read_bytes = type(input_path).read_bytes

    def spy_read_bytes(self):
        reads.append(str(self))
        return orig_read_bytes(self)

    monkeypatch.setattr(type(input_path), "read_bytes", spy_read_bytes)
    assert [c["completion_id"] for c in store.completions(key="Jones2024")] == [c2["completion_id"]]
    assert len(reads) == 1 and c2["completion_id"] in reads[0]
    reads.clear()
    assert list(store.completions(key="NoSuchPaper")) == []
    assert reads == []


def test_registration_view_does_not_leak_preview_keys_into_live_index_entries(tmp_path):
    """_view() isolates _key_index_entries so previewing external candidates never mutates live state."""
    from papervault.library.services.registration import _view, preview_candidates

    lib = Library(tmp_path)
    try:
        p, _ = lib.upsert({"title": "Live paper in library", "authors": ["Smith"], "year": 2024, "doi": "10.1/live"})
        lib.save()
        before_entries = dict(lib._key_index_entries)

        view = _view(lib, lib._papers, deep_copy=True, frozen=True)
        assert view._key_index_entries is not lib._key_index_entries
        view._reindex(Paper(key="GhostPreview2026", title="Temporary preview candidate", doi="10.1/ghost"))
        assert "GhostPreview2026" not in lib._key_index_entries
        assert lib._key_index_entries == before_entries

        snapshots, _ = preview_candidates(lib, [{
            "title": "External candidate that should not leak into live index entries",
            "authors": ["Doe, J."],
            "year": 2025,
            "doi": "10.1/ext-preview",
            "abstract": "Detailed measurement of heliospheric plasma waves.",
            "_registration": {"ingest_ok": True, "tier": "1A", "llm_is_paper": True},
        }])
        assert len(snapshots) == 1
        assert lib._key_index_entries == before_entries
    finally:
        lib.close()


def test_reindex_keys_preserves_title_ownership_on_venue_edit(tmp_path, monkeypatch):
    """Editing a non-index field (venue) must not flip _by_title ownership between same-titled papers."""
    from papervault.library.models import normalize_title

    title = "Identical plasma wave title shared by two papers"
    rec_a = replace(Record.from_payload("A", {"key": "A", "title": title, "doi": "10.1/a"}), revision=1)
    rec_b = replace(Record.from_payload("B", {"key": "B", "title": title, "doi": "10.1/b"}), revision=1)
    repo = _MemoryRepo({"A": rec_a, "B": rec_b})
    backend = _backend_for(repo)
    monkeypatch.setattr(PostgresPersistence, "from_environment", lambda: backend)
    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")

    lib = Library(tmp_path)
    try:
        nt = normalize_title(title)
        assert lib._by_title[nt] == "B"  # B was indexed after A and owns the title bucket
        lib.get("A").venue = "ApJ"
        lib.save(force=False, force_bib=False)
        # Venue edit did NOT change A's title, so B retains _by_title ownership!
        assert lib._by_title[nt] == "B"

        # Changing A's DOI still updates _by_doi cleanly without touching _by_title.
        lib.get("A").doi = "10.1/a-v2"
        lib.save(force=False, force_bib=False)
        assert lib._by_doi.get("10.1/a") is None
        assert lib._by_doi["10.1/a-v2"] == "A"
        assert lib._by_title[nt] == "B"
    finally:
        lib.close()


def test_register_postgres_marks_bib_dirty_for_next_bib_refresh(tmp_path, monkeypatch):
    """RegistrationService._register_postgres() sets _bib_dirty so library.bib refreshes on next interval."""
    from papervault.library.services.registration import RegistrationService

    class _RegRepo(_MemoryRepo):
        def upsert(self, record, *, audit, expected_revision):
            cur = self.records.get(record.key)
            if cur is not None and cur.revision != expected_revision:
                raise StaleWrite(record.key, expected_revision, cur.revision)
            self.records[record.key] = replace(record, revision=expected_revision + 1)
            return True

    repo = _RegRepo()
    backend = _backend_for(repo)
    monkeypatch.setattr(PostgresPersistence, "from_environment", lambda: backend)
    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")

    lib = Library(tmp_path)
    try:
        lib._bib_dirty = False
        lib._last_bib_save = 0.0
        svc = RegistrationService(lib, SimpleNamespace(add=lambda *a, **k: None))
        outcomes = svc._register_postgres(
            [{
                "title": "Newly registered space physics paper for bibtex refresh",
                "authors": ["Parker, E."],
                "year": 2025,
                "doi": "10.1/reg-bib",
                "abstract": "Measurements of Alfven waves in the inner heliosphere.",
                "_registration": {"ingest_ok": True, "tier": "1A", "llm_is_paper": True},
            }],
            actor="search",
            reason="bib-refresh-test",
        )
        assert any(o["status"] == "registered" for o in outcomes)
        assert lib._bib_dirty is True

        # Next non-forced save after bib interval writes library.bib even with zero new PG changes!
        lib.save(force=False, force_bib=False)
        assert lib._bib_dirty is False
        assert "10.1/reg-bib" in lib.bib_path.read_text(encoding="utf-8")
    finally:
        lib.close()


def test_concurrent_edit_to_unrelated_paper_during_save_keeps_index_dirty(tmp_path, monkeypatch):
    """Mutating unrelated paper B while A is flushing to PG must keep lib._index_dirty=True."""
    rec_a = replace(Record.from_payload("A", {"key": "A", "title": "Paper A"}), revision=1)
    rec_b = replace(Record.from_payload("B", {"key": "B", "title": "Paper B"}), revision=1)
    repo = _MemoryRepo({"A": rec_a, "B": rec_b})
    backend = _backend_for(repo)
    monkeypatch.setattr(PostgresPersistence, "from_environment", lambda: backend)
    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")

    lib = Library(tmp_path)
    try:
        mutated_once = [False]
        orig_flush = repo.flush

        def flush_with_concurrent_edit(records, *, expected_revisions, audit):
            result = orig_flush(records, expected_revisions=expected_revisions, audit=audit)
            if not mutated_once[0]:
                mutated_once[0] = True
                # Simulate another worker mutating unrelated paper B during A's PG round-trip.
                lib.get("B").venue = "JGR Space Physics"
            return result

        monkeypatch.setattr(repo, "flush", flush_with_concurrent_edit)
        lib.get("A").venue = "ApJ"
        lib.save(force=True, force_bib=False)

        assert repo.get("A").payload["venue"] == "ApJ"
        assert repo.get("B").payload.get("venue", "") == ""
        assert lib._index_dirty is True  # concurrent edit to B was NOT lost!

        assert lib.flush_if_dirty() is True
        assert repo.get("B").payload["venue"] == "JGR Space Physics"
        assert lib._index_dirty is False
    finally:
        lib.close()


def test_stale_operator_ruling_conflict_does_not_wedge_subsequent_unrelated_saves():
    """A StaleWrite rejecting a queued operator ruling must clear it so unrelated saves succeed."""
    rec_a = replace(Record.from_payload("A", {"key": "A", "title": "Paper A"}), revision=1)
    rec_b = replace(Record.from_payload("B", {"key": "B", "title": "Paper B"}), revision=1)
    repo = _MemoryRepo({"A": rec_a, "B": rec_b})
    repo.import_ruling = lambda ruling, *, audit: None
    backend = _backend_for(repo)
    papers = backend.load()

    # Remote advances A -> rev 2 while local queues an operator ruling against rev 1.
    repo.records["A"] = replace(Record.from_payload("A", {"key": "A", "title": "Paper A remote"}), revision=2)
    backend.operator_rulings.append({
        "timestamp": "2026-10-22T00:00:00+00:00",
        "action": "reopen_download",
        "event": "operator_reopen",
        "scope": "acquire",
        "reason": "stale operator retry",
        "actor": "owner",
        "keys": ["A"],
        "revision": 1,
    })
    papers["B"].venue = "Solar Physics"

    with pytest.raises(CatalogConflict) as exc_info:
        backend.save(papers)
    assert exc_info.value.keys == ("A",)
    assert backend.operator_rulings == []

    # Subsequent save of unrelated B must succeed cleanly instead of staying wedged forever!
    assert backend.save(papers) == []
    assert repo.get("B").payload["venue"] == "Solar Physics"
    assert repo.get("B").revision == 2


def test_extraction_store_hardlink_toctou_mismatch_falls_back_to_write_once(tmp_path, monkeypatch):
    """If pdfs/<key>.pdf is replaced atomically before os.link, capture detects inode mismatch and writes exact source."""
    import papervault.store.extraction as ext_mod

    store = ExtractionStore(tmp_path)
    orig_bytes = b"%PDF-1.4\n" + (b"A123456789abcdef" * 5000)
    replaced_bytes = b"%PDF-1.4\n" + (b"B123456789abcdef" * 5000)
    pdf_path = tmp_path / "pdfs" / "Race2024.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(orig_bytes)

    # Disable reflink so we exercise the hardlink path deterministically on any filesystem.
    monkeypatch.setattr(ext_mod, "_try_reflink_fd", lambda fd, dest: False)

    real_link = os.link
    raced = [False]

    def racing_link(src, dst, *args, **kwargs):
        if not raced[0] and str(src) == str(pdf_path):
            raced[0] = True
            # Atomically swap candidate to a brand-new inode with different content right before os.link!
            tmp_swap = pdf_path.with_suffix(".pdf.swap")
            tmp_swap.write_bytes(replaced_bytes)
            os.replace(tmp_swap, pdf_path)
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(ext_mod.os, "link", racing_link)

    recipe = {"schema_version": 1, "engine": "mineru2.5-pro", "configured_model": "test",
              "output_contract": {"fulltext": {"required": ["markdown"], "not_applicable": []}}}
    req = store.capture(
        record_key="Race2024", metadata={"key": "Race2024"}, source=orig_bytes, source_role="pdf",
        recipe=recipe, guard={"previous_completion_id": None, "identity": {"key": "Race2024"}},
        unit_evidence={"kind": "paper", "record_key": "Race2024"},
    )
    assert raced[0] is True
    input_path = tmp_path / req["input_manifest"]["artifacts"][0]["path"]
    # Captured bytes MUST match orig_bytes (never replaced_bytes) and MUST NOT share inode with replaced pdf_path!
    assert input_path.read_bytes() == orig_bytes
    assert input_path.stat().st_ino != pdf_path.stat().st_ino
