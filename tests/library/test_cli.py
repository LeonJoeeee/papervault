"""CLI handler tests. Each one builds a fresh tmp library, runs
`cli.main([...])`, and asserts on exit code + captured stdout/stderr.

Service classes are monkeypatched at the cli import sites so we never hit
network / disk / LLM during these tests.
"""

from __future__ import annotations

import json

import pytest

from papervault.library import Library, cli


@pytest.fixture
def tmp_lib_env(tmp_path, monkeypatch):
    """Point $PAPER_LIBRARY_PATH at a fresh tmpdir and seed two papers."""
    monkeypatch.setenv("PAPER_LIBRARY_PATH", str(tmp_path))
    lib = Library(tmp_path)
    lib.upsert({"title": "Solar Modulation Review", "authors": ["Marius Potgieter"],
                "year": 2013, "doi": "10.1234/abc", "is_review": True,
                "citation_count": 200})
    lib.upsert({"title": "Cosmic Ray Spectrum", "authors": ["Aslam"], "year": 2020,
                "doi": "10.1234/xyz", "citation_count": 5})
    lib.save()
    return lib


def test_config_prints_root(tmp_lib_env, capsys):
    rc = cli.main(["config"])
    out = capsys.readouterr().out
    assert rc == 0
    assert str(tmp_lib_env.root) in out
    assert "papers_in_library: 2" in out


def test_show_known_key(tmp_lib_env, capsys):
    rc = cli.main(["show", "Potgieter2013"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["key"] == "Potgieter2013"
    assert payload["has_pdf"] is False  # nothing on disk yet


def test_show_unknown_key_exits_1(tmp_lib_env, capsys):
    rc = cli.main(["show", "GhostKey"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "GhostKey" in err


def test_list_review_filter(tmp_lib_env, capsys):
    rc = cli.main(["list", "--review"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Potgieter2013" in out
    assert "Aslam2020" not in out


def test_list_json_shape(tmp_lib_env, capsys):
    rc = cli.main(["list", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    items = json.loads(out)
    assert len(items) == 2
    assert {p["key"] for p in items} == {"Potgieter2013", "Aslam2020"}


def test_bibtex_subset(tmp_lib_env, capsys):
    rc = cli.main(["bibtex", "--keys", "Potgieter2013"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "@article{Potgieter2013" in out
    assert "Aslam2020" not in out


def test_bibtex_missing_key_exits_1(tmp_lib_env, capsys):
    rc = cli.main(["bibtex", "--keys", "Potgieter2013,GhostKey"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "GhostKey" in err


def test_cite_check_dangling(tmp_lib_env, tmp_path, capsys):
    tex = tmp_path / "doc.tex"
    tex.write_text(r"\citep{Potgieter2013,Ghost9999}")
    rc = cli.main(["cite-check", str(tex)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    assert "Ghost9999" in out


def test_cite_check_missing_file(tmp_lib_env, tmp_path, capsys):
    rc = cli.main(["cite-check", str(tmp_path / "nonexistent.tex")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "not found" in err


def test_audit_clean_returns_0(tmp_lib_env, capsys):
    rc = cli.main(["audit"])
    assert rc == 0


def test_audit_detects_drift_and_fix_backfills(tmp_lib_env, capsys):
    lib = tmp_lib_env
    # Plant an on-disk file but DO NOT update the index field — the drift case.
    lib.pdf_path("Potgieter2013").write_bytes(b"%PDF-1.0 fake")
    lib.txt_path("Potgieter2013").write_text("body")

    rc = cli.main(["audit", "--json"])
    out = capsys.readouterr().out
    assert rc == 1
    report = json.loads(out)
    kinds = {(d["kind"], d["key"]) for d in report["drift"]}
    assert ("pdf", "Potgieter2013") in kinds
    assert ("txt", "Potgieter2013") in kinds

    rc2 = cli.main(["audit", "--fix"])
    assert rc2 == 1  # still has drift logged in this run, but fix actually wrote
    # New Library() reload to confirm persistence.
    reloaded = Library(lib.root)
    p = reloaded.get("Potgieter2013")
    assert p.pdf_path == "pdfs/Potgieter2013.pdf"
    assert p.txt_path == "extracts/txt/Potgieter2013.txt"


def test_audit_orphan_file(tmp_lib_env, capsys):
    (tmp_lib_env.pdfs_dir / "Stranger2099.pdf").write_bytes(b"x")
    rc = cli.main(["audit", "--json"])
    out = capsys.readouterr().out
    assert rc == 1
    report = json.loads(out)
    assert any(o["file"].endswith("Stranger2099.pdf") for o in report["orphans"])


def test_audit_queue_reports_status_breakdown(tmp_lib_env, capsys):
    """Seed mixed download_status values; --queue --json should bucket them."""
    lib = tmp_lib_env
    # tmp_lib_env already has Potgieter2013 + Aslam2020 (both default 'pending').
    # Mutate them to spread across several statuses, plus add three more.
    lib.get("Potgieter2013").download_status = "ok"
    lib.get("Potgieter2013").download_source = "arxiv"
    lib.get("Aslam2020").download_status = "failed"
    lib.upsert({"title": "First helio paper for audit queue test", "authors": ["Helio"], "year": 2021,
                "doi": "10.1/h1", "download_status": "ok",
                "download_source": "oa_aggregators"})
    lib.upsert({"title": "Second helio paper for audit queue test", "authors": ["Helio"], "year": 2022,
                "doi": "10.1/h2", "download_status": "extract_failed"})
    lib.upsert({"title": "Third helio paper for audit queue test", "authors": ["Helio"], "year": 2023,
                "doi": "10.1/h3"})  # default pending
    # Plant a real md file so extract coverage is non-zero among ok papers.
    lib.md_path("Potgieter2013").write_text("# md body")
    lib.save()

    rc = cli.main(["audit", "--queue", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert "queue" in payload
    counts = payload["queue"]["status_counts"]
    # D7: status routes (one "ok" bucket), provenance is in download_source.
    assert counts.get("pending") == 1
    assert counts.get("ok") == 2
    assert counts.get("failed") == 1
    assert counts.get("extract_failed") == 1
    # Extract coverage among ok papers (2 total; 1 has md, 0 have txt).
    assert payload["queue"]["ok_total"] == 2
    assert payload["queue"]["ok_has_md"] == 1
    assert payload["queue"]["ok_has_txt"] == 0
    assert payload["queue"]["ok_missing_both"] == 1


def test_audit_queue_includes_recent_failed(tmp_lib_env, capsys):
    """Three failed papers with different added_at — recent_failed sorted desc."""
    lib = tmp_lib_env
    # Wipe seed entries' relevance: drop them out of failed bucket explicitly.
    lib.get("Potgieter2013").download_status = "ok"
    lib.get("Aslam2020").download_status = "ok"
    # Three failed papers; assign deterministic added_at so we can assert order.
    lib.upsert({"title": "Alpha failed paper from recent test", "authors": ["Alpha"], "year": 2018,
                "doi": "10.1/old", "download_status": "failed",
                "added_at": "2024-01-01T00:00:00+00:00"})
    lib.upsert({"title": "Beta failed paper from recent test", "authors": ["Beta"], "year": 2019,
                "doi": "10.1/mid", "download_status": "failed",
                "added_at": "2024-06-01T00:00:00+00:00"})
    lib.upsert({"title": "Gamma failed paper from recent test", "authors": ["Gamma"], "year": 2020,
                "doi": "10.1/new", "download_status": "failed",
                "added_at": "2025-01-01T00:00:00+00:00"})
    lib.save()

    rc = cli.main(["audit", "--queue", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    recent = payload["queue"]["recent_failed"]
    assert len(recent) == 3
    keys_in_order = [r["key"] for r in recent]
    assert keys_in_order == ["Gamma2020", "Beta2019", "Alpha2018"]


def test_audit_retry_failed_resets_to_pending(tmp_lib_env, capsys):
    """Two failed + one pending → both failed flip to pending, original pending intact."""
    lib = tmp_lib_env
    lib.get("Potgieter2013").download_status = "failed"
    lib.get("Aslam2020").download_status = "failed"
    lib.upsert({"title": "Already Pending", "authors": ["Delta"], "year": 2024,
                "doi": "10.1/dp"})  # default 'pending'
    lib.save()

    rc = cli.main(["audit", "--retry-failed"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reset 2 papers" in out

    # Reload to verify persistence on disk.
    reloaded = Library(lib.root)
    assert reloaded.get("Potgieter2013").download_status == "pending"
    assert reloaded.get("Aslam2020").download_status == "pending"
    assert reloaded.get("Delta2024").download_status == "pending"


def test_audit_retry_low_quality_resets_to_recovered(tmp_lib_env, capsys):
    """D7/D9: two extract_failed papers (with PDF) flip to ok + attempts
    cleared; an unrelated ok paper is untouched. (The flag name is kept for
    back-compat; it now targets the extract_failed terminal state.)"""
    lib = tmp_lib_env
    lib.get("Potgieter2013").download_status = "extract_failed"
    lib.get("Potgieter2013").extract_attempts = 3
    lib.pdf_path("Potgieter2013").write_bytes(b"%PDF-1.0 fake")
    lib.get("Aslam2020").download_status = "extract_failed"
    lib.get("Aslam2020").extract_attempts = 3
    lib.pdf_path("Aslam2020").write_bytes(b"%PDF-1.0 fake")
    lib.upsert({"title": "Already ok recovered paper title", "authors": ["Delta"],
                "year": 2024, "doi": "10.1/dp",
                "download_status": "ok"})
    lib.save()

    rc = cli.main(["audit", "--retry-low-quality"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reset 2 papers" in out
    assert "extract_failed" in out

    reloaded = Library(lib.root)
    assert reloaded.get("Potgieter2013").download_status == "ok"
    assert reloaded.get("Potgieter2013").extract_attempts == 0
    assert reloaded.get("Aslam2020").download_status == "ok"
    assert reloaded.get("Aslam2020").extract_attempts == 0
    # Untouched paper kept its original status.
    assert reloaded.get("Delta2024").download_status == "ok"


def _seed_metadata_only(lib):
    """Seed a mixed set of metadata_only rows + one untouched ok paper.

    Returns nothing; callers read back via reloaded Library. The two seed
    papers from the fixture become metadata_only with DOIs:
      - Potgieter2013 doi=10.1234/abc  (matches has-doi; not 10.3390)
      - Aslam2020     doi=10.1234/xyz  (matches has-doi; not 10.3390)
    plus the rows below.
    """
    lib.get("Potgieter2013").download_status = "metadata_only"
    lib.get("Aslam2020").download_status = "metadata_only"
    # MDPI gold-OA: 10.3390 prefix, metadata_only.
    lib.upsert({"title": "MDPI gold OA metadata only paper title", "authors": ["Mdpi"],
                "year": 2022, "doi": "10.3390/rs14010001",
                "download_status": "metadata_only"})
    # Real arxiv id, no DOI, metadata_only.
    lib.upsert({"title": "Arxiv preprint metadata only paper title", "authors": ["Arx"],
                "year": 2021, "arxiv_id": "2101.01234",
                "download_status": "metadata_only"})
    # No-DOI no-arxiv citation stub (must NOT be reset by any scoped filter).
    lib.upsert({"title": "No DOI citation stub metadata only paper title",
                "authors": ["Stub"], "year": 2019,
                "download_status": "metadata_only"})
    # Unrelated ok paper — never touched.
    lib.upsert({"title": "Already ok unrelated paper title", "authors": ["Ok"],
                "year": 2024, "doi": "10.1/ok", "download_status": "ok"})
    lib.save()


def test_audit_retry_metadata_only_doi_prefix(tmp_lib_env, capsys):
    """--filter 10.3390 flips ONLY the MDPI row metadata_only → pending;
    all other metadata_only rows (other DOIs, arxiv, no-DOI stub) untouched."""
    lib = tmp_lib_env
    _seed_metadata_only(lib)
    mdpi_key = lib.find(doi="10.3390/rs14010001").key

    rc = cli.main(["audit", "--retry-metadata-only", "--filter", "10.3390"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reset 1 papers from metadata_only → pending" in out
    assert mdpi_key in out

    reloaded = Library(lib.root)
    assert reloaded.get(mdpi_key).download_status == "pending"
    # Everything else stays metadata_only / ok.
    assert reloaded.get("Potgieter2013").download_status == "metadata_only"
    assert reloaded.get("Aslam2020").download_status == "metadata_only"
    arxiv_key = reloaded.find(arxiv_id="2101.01234").key
    assert reloaded.get(arxiv_key).download_status == "metadata_only"


def test_audit_retry_metadata_only_has_arxiv(tmp_lib_env, capsys):
    """--filter has-arxiv flips ONLY the row with a real arxiv_id; the no-DOI
    no-arxiv citation stub and DOI-only rows are left terminal."""
    lib = tmp_lib_env
    _seed_metadata_only(lib)
    arxiv_key = lib.find(arxiv_id="2101.01234").key

    rc = cli.main(["audit", "--retry-metadata-only", "--filter", "has-arxiv"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reset 1 papers from metadata_only → pending" in out
    assert arxiv_key in out

    reloaded = Library(lib.root)
    assert reloaded.get(arxiv_key).download_status == "pending"
    # DOI-only metadata_only rows (no arxiv) untouched.
    assert reloaded.get("Potgieter2013").download_status == "metadata_only"
    mdpi_key = reloaded.find(doi="10.3390/rs14010001").key
    assert reloaded.get(mdpi_key).download_status == "metadata_only"


def test_audit_retry_metadata_only_bare_refuses(tmp_lib_env, capsys):
    """A bare --retry-metadata-only (no --filter, no --all) must refuse and
    mutate NOTHING — the mass-reset safety gate."""
    lib = tmp_lib_env
    _seed_metadata_only(lib)

    rc = cli.main(["audit", "--retry-metadata-only"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "requires --filter" in captured.err

    # Nothing flipped: every metadata_only row still metadata_only.
    reloaded = Library(lib.root)
    assert reloaded.get("Potgieter2013").download_status == "metadata_only"
    assert reloaded.get("Aslam2020").download_status == "metadata_only"
    mdpi_key = reloaded.find(doi="10.3390/rs14010001").key
    assert reloaded.get(mdpi_key).download_status == "metadata_only"


def test_audit_retry_metadata_only_all_override(tmp_lib_env, capsys):
    """--all overrides the filter gate and resets EVERY metadata_only row
    (including the no-DOI stub), but leaves the ok paper untouched."""
    lib = tmp_lib_env
    _seed_metadata_only(lib)

    rc = cli.main(["audit", "--retry-metadata-only", "--all"])
    out = capsys.readouterr().out
    assert rc == 0
    # 5 metadata_only rows were seeded (2 fixture + MDPI + arxiv + stub).
    assert "reset 5 papers from metadata_only → pending" in out

    reloaded = Library(lib.root)
    for key in ["Potgieter2013", "Aslam2020"]:
        assert reloaded.get(key).download_status == "pending"
    # The unrelated ok paper is never touched.
    ok_key = reloaded.find(doi="10.1/ok").key
    assert reloaded.get(ok_key).download_status == "ok"


def test_audit_retry_metadata_only_bad_filter_errors(tmp_lib_env, capsys):
    """An unrecognised --filter spec fails loud (exit 1) rather than silently
    matching nothing."""
    lib = tmp_lib_env
    _seed_metadata_only(lib)

    rc = cli.main(["audit", "--retry-metadata-only", "--filter", "garbage"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "unrecognised --filter spec" in captured.err

    # No mutation on the bad-filter path.
    reloaded = Library(lib.root)
    assert reloaded.get("Potgieter2013").download_status == "metadata_only"


def test_audit_queue_alone_skips_drift_scan(tmp_lib_env, capsys):
    """--queue alone must NOT include drift / dangling / orphans in the JSON."""
    lib = tmp_lib_env
    # Plant a drift case: file exists, but index field is null.
    lib.pdf_path("Potgieter2013").write_bytes(b"%PDF-1.0 fake")

    rc = cli.main(["audit", "--queue", "--json"])
    out = capsys.readouterr().out
    assert rc == 0  # queue report alone never returns 1 for drift
    payload = json.loads(out)
    assert "queue" in payload
    assert "drift" not in payload
    assert "dangling" not in payload
    assert "orphans" not in payload


def test_audit_combined_flags(tmp_lib_env, capsys):
    """--queue + --fix → both behaviors run."""
    lib = tmp_lib_env
    # Plant a drift case so --fix has something to do.
    lib.pdf_path("Potgieter2013").write_bytes(b"%PDF-1.0 fake")

    rc = cli.main(["audit", "--queue", "--fix", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    # Queue payload present.
    assert "queue" in payload
    # Drift scan also present and the fix actually ran.
    assert "drift" in payload
    assert payload["fixed"] >= 1

    # Re-load the library to confirm --fix wrote the path through.
    reloaded = Library(lib.root)
    assert reloaded.get("Potgieter2013").pdf_path == "pdfs/Potgieter2013.pdf"
    # rc is 1 because the drift entries were detected this run.
    assert rc == 1


def test_status_basic(tmp_lib_env, capsys):
    """status command renders a dashboard with the right counts."""
    lib = tmp_lib_env
    lib.get("Potgieter2013").download_status = "ok"
    lib.get("Potgieter2013").download_source = "arxiv"
    lib.get("Aslam2020").download_status = "failed"
    lib.save()

    rc = cli.main(["status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "total papers:  2" in out
    assert "downloaded:" in out
    assert "✓ Potgieter2013" in out
    assert "✗ Aslam2020" in out


def test_status_json(tmp_lib_env, capsys):
    rc = cli.main(["status", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["total_papers"] == 2
    assert "recent_ok" in payload
    assert "recent_failed" in payload


def test_topics_list_and_show(tmp_lib_env, capsys):
    tmp_lib_env.write_topic("solar", {"keys": ["Potgieter2013"], "topic": "Solar"})
    rc = cli.main(["topics", "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "solar" in out

    rc2 = cli.main(["topics", "show", "solar"])
    out2 = capsys.readouterr().out
    payload = json.loads(out2)
    assert payload["topic"] == "Solar"


def test_topics_show_missing(tmp_lib_env, capsys):
    rc = cli.main(["topics", "show", "no-such-thing"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no-such-thing" in err


def test_read_missing_extract(tmp_lib_env, capsys):
    rc = cli.main(["read", "Potgieter2013", "--md"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no md extract" in err


def test_read_with_truncation(tmp_lib_env, capsys):
    tmp_lib_env.txt_path("Potgieter2013").write_text("X" * 10000)
    rc = cli.main(["read", "Potgieter2013", "--txt", "--max-chars", "200"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "truncated" in out
    assert len(out) < 1000  # truncated, not full 10000 bytes


def test_add_uses_addservice(tmp_lib_env, capsys, monkeypatch):
    # D11: a successful CLI add now reports "queued" (enqueue-only).
    canned = {"status": "queued", "key": "Foo2025", "metadata": {"title": "Padded test paper title for validation"},
              "candidates": None, "message": "queued"}

    class FakeAdd:
        def __init__(self, lib): pass
        def add(self, ident, force_refresh=False):
            assert ident == "10.1/x"
            return canned

    monkeypatch.setattr("papervault.library.services.add_service.AddService", FakeAdd)
    rc = cli.main(["add", "10.1/x"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "queued: Foo2025" in out


def test_add_failed_exits_1(tmp_lib_env, capsys, monkeypatch):
    class FakeAdd:
        def __init__(self, lib): pass
        def add(self, ident, force_refresh=False):
            return {"status": "not_found", "key": None, "metadata": None,
                    "candidates": None, "message": "nope"}

    monkeypatch.setattr("papervault.library.services.add_service.AddService", FakeAdd)
    rc = cli.main(["add", "garbage"])
    assert rc == 1


def test_add_batch_all_queued_reports_zero_failures(tmp_lib_env, tmp_path, capsys, monkeypatch):
    # D11/S7 regression guard: AddService.add's success status is "queued"
    # (never "added"). cmd_add_batch's failure tally must count both "queued"
    # and "exists" as successes — otherwise a fully successful batch is mis-
    # tallied as all-failures and exits 1.
    batch_file = tmp_path / "idents.txt"
    batch_file.write_text("10.1/a\n10.1/b\n10.1/c\n")

    canned = [
        {"status": "queued", "key": "A2025", "message": "queued"},
        {"status": "queued", "key": "B2025", "message": "queued"},
        {"status": "exists", "key": "C2025", "message": "already in library"},
    ]

    class FakeBatch:
        def __init__(self, lib, max_workers=4): pass
        def add_many(self, idents, on_result=None):
            return canned

    monkeypatch.setattr("papervault.library.services.batch.BatchAddService", FakeBatch)
    rc = cli.main(["add-batch", str(batch_file)])
    err = capsys.readouterr().err
    assert rc == 0
    assert "3 processed, 0 failure(s)" in err


def test_add_batch_counts_non_success_statuses_as_failures(tmp_lib_env, tmp_path, capsys, monkeypatch):
    # The dual: not_found / rejected / internal_error are real failures and
    # must be tallied (and force exit 1).
    batch_file = tmp_path / "idents.txt"
    batch_file.write_text("good\nbad\n")

    canned = [
        {"status": "queued", "key": "Good2025", "message": "queued"},
        {"status": "not_found", "key": None, "message": "nope"},
    ]

    class FakeBatch:
        def __init__(self, lib, max_workers=4): pass
        def add_many(self, idents, on_result=None):
            return canned

    monkeypatch.setattr("papervault.library.services.batch.BatchAddService", FakeBatch)
    rc = cli.main(["add-batch", str(batch_file)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "2 processed, 1 failure(s)" in err


def test_search_no_rerank(tmp_lib_env, capsys):
    rc = cli.main(["search", "cosmic ray", "--no-rerank", "-k", "5"])
    out = capsys.readouterr().out
    assert rc == 0
    # Aslam2020 has 'Cosmic Ray Spectrum' in title — must rank.
    assert "Aslam2020" in out


def test_library_path_override(tmp_path, capsys, monkeypatch):
    # When --library-path is passed, it overrides $PAPER_LIBRARY_PATH for this run.
    monkeypatch.delenv("PAPER_LIBRARY_PATH", raising=False)
    rc = cli.main(["--library-path", str(tmp_path), "config"])
    out = capsys.readouterr().out
    assert rc == 0
    assert str(tmp_path) in out


# =========== insight ========================================================
# Phase 28 (2026-05-24, route B): `paper-library insight` subcommand and
# the entire 5-Q write pipeline were removed. The CLI tests that
# exercised `insight regenerate / batch / dead-letters / retry-invalid
# / audit-meaningless` were deleted along with the implementation.
# Schema-side coverage stays in tests/test_insight_schema.py.


# ---- audit --retry-extract-keys (issue #134): targeted extract_failed reset ----
#
# Resets ONLY the listed keys that are extract_failed with a PDF on disk and no
# md; dry-run by default; a write refuses while papervault.service is active,
# because the running service saves its whole in-memory index and would
# overwrite the reset. systemctl is always mocked here.


class _FakeSystemctl:
    """Stand-in for ``subprocess.run(["systemctl", "--user", "is-active", unit])``."""

    def __init__(self, state="inactive"):
        self.state = state
        self.calls = []

    def __call__(self, argv, **_kw):
        import subprocess
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0 if self.state == "active" else 3,
                                           stdout=f"{self.state}\n", stderr="")


def _seed_retry_extract(lib, tmp_path):
    """Eligible: Potgieter2013 + Aslam2020. Ineligible: md present, no PDF,
    not extract_failed. Unlisted-but-eligible: Echo2022 (must stay untouched)."""
    def _paper(title, author, year, doi):
        p, _ = lib.upsert({"title": title, "authors": [author], "year": year, "doi": doi})
        return p

    for p in (lib.get("Potgieter2013"), lib.get("Aslam2020")):
        p.download_status = "extract_failed"
        p.extract_attempts = 3
        p.extract_deferred_sig = "1:1"
        p.extract_deferred_epoch = 7
        lib.pdf_path(p.key).write_bytes(b"%PDF-1.0 fake")
    has_md = _paper("Has markdown already paper", "Bravo", 2021, "10.1/b")
    has_md.download_status = "extract_failed"
    has_md.extract_attempts = 3
    lib.pdf_path(has_md.key).write_bytes(b"%PDF-1.0 fake")
    lib.md_path(has_md.key).write_text("# body")
    no_pdf = _paper("No pdf on disk paper", "Charlie", 2021, "10.1/c")
    no_pdf.download_status = "extract_failed"
    no_pdf.extract_attempts = 3
    ok = _paper("Healthy ok paper title", "Delta", 2024, "10.1/d")
    ok.download_status = "ok"
    ok.extract_attempts = 1
    lib.pdf_path(ok.key).write_bytes(b"%PDF-1.0 fake")
    unlisted = _paper("Unlisted failed paper", "Echo", 2022, "10.1/e")
    unlisted.download_status = "extract_failed"
    unlisted.extract_attempts = 3
    lib.pdf_path(unlisted.key).write_bytes(b"%PDF-1.0 fake")
    lib.save()
    keys = tmp_path / "keys.txt"
    keys.write_text("# incident keys\nPotgieter2013\n\nAslam2020\nPotgieter2013\n"
                    "Bravo2021\nCharlie2021\nDelta2024\nGhost1999\n")
    return keys


def _snapshot(root):
    return {p.key: (p.download_status, p.extract_attempts, p.extract_deferred_sig,
                    p.extract_deferred_epoch)
            for p in Library(root).all_papers(include_quarantined=True)}


def test_retry_extract_keys_is_dry_run_by_default(tmp_lib_env, tmp_path, capsys,
                                                  monkeypatch):
    from papervault import ops_guards
    fake = _FakeSystemctl("active")
    monkeypatch.setattr(ops_guards.subprocess, "run", fake)
    keys = _seed_retry_extract(tmp_lib_env, tmp_path)
    before = _snapshot(tmp_lib_env.root)

    rc = cli.main(["audit", "--retry-extract-keys", str(keys)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY-RUN" in out
    assert "7 listed" in out and "6 unique" in out
    assert "2 eligible" in out
    assert _snapshot(tmp_lib_env.root) == before       # nothing written
    assert fake.calls == []                             # dry-run needs no probe


def test_retry_extract_keys_resets_only_eligible_listed_keys(tmp_lib_env, tmp_path,
                                                            capsys, monkeypatch):
    from papervault import ops_guards
    fake = _FakeSystemctl("inactive")
    monkeypatch.setattr(ops_guards.subprocess, "run", fake)
    keys = _seed_retry_extract(tmp_lib_env, tmp_path)
    before = _snapshot(tmp_lib_env.root)

    rc = cli.main(["audit", "--retry-extract-keys", str(keys), "--no-dry-run", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    result = payload["retry_extract_keys"]
    assert result["dry_run"] is False
    assert result["listed"] == 7 and result["unique"] == 6
    assert sorted(result["reset_keys"]) == ["Aslam2020", "Potgieter2013"]
    assert result["skipped"] == {"not_found": ["Ghost1999"],
                                 "not_extract_failed": ["Delta2024"],
                                 "no_pdf": ["Charlie2021"],
                                 "has_md": ["Bravo2021"]}
    assert ["systemctl", "--user", "is-active", "papervault.service"] in fake.calls

    after = _snapshot(tmp_lib_env.root)
    for key in ("Potgieter2013", "Aslam2020"):
        assert after[key] == ("ok", 0, None, 0)
    for key in ("Bravo2021", "Charlie2021", "Delta2024", "Echo2022"):
        assert after[key] == before[key]                # untouched


def test_retry_extract_keys_write_refuses_while_service_active(tmp_lib_env, tmp_path,
                                                              capsys, monkeypatch):
    from papervault import ops_guards
    fake = _FakeSystemctl("active")
    monkeypatch.setattr(ops_guards.subprocess, "run", fake)
    keys = _seed_retry_extract(tmp_lib_env, tmp_path)
    before = _snapshot(tmp_lib_env.root)

    rc = cli.main(["audit", "--retry-extract-keys", str(keys), "--no-dry-run"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "papervault.service" in err
    assert _snapshot(tmp_lib_env.root) == before       # refused before any write


@pytest.mark.parametrize("state", ["activating", "deactivating", "reloading"])
def test_retry_extract_keys_refuses_in_transitional_states(tmp_lib_env, tmp_path,
                                                          capsys, monkeypatch, state):
    """A stopping service still runs its final save, so it counts as running."""
    from papervault import ops_guards
    monkeypatch.setattr(ops_guards.subprocess, "run", _FakeSystemctl(state))
    keys = _seed_retry_extract(tmp_lib_env, tmp_path)

    rc = cli.main(["audit", "--retry-extract-keys", str(keys), "--no-dry-run"])
    assert rc == 2


def test_retry_extract_keys_missing_file_is_an_error(tmp_lib_env, tmp_path, capsys):
    rc = cli.main(["audit", "--retry-extract-keys", str(tmp_path / "absent.txt")])
    assert rc == 2
    assert "absent.txt" in capsys.readouterr().err
