"""Tests for the one-time D7 ``download_status`` migration
(``services.migrate_status.migrate_status``).

The key safety property is a ROUND-TRIP over a PRE-SHAPED index.json: build
a vault directory whose ``index.json`` carries the legacy status zoo
(``ok:<src>``, ``text-only:firecrawl``, ``extract_low_quality``, empty),
load it, migrate, save, reload from disk, and assert every record landed on
a canonical D7 value with provenance split into ``download_source``. The
migration NEVER runs against the live vault — every fixture is a tmp dir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from papervault.library.models import DownloadStatus
from papervault.library.services.migrate_status import migrate_status
from papervault.library.store import Library

_CANONICAL = set(DownloadStatus.__args__)  # type: ignore[attr-defined]


def _shaped_index(tmp_path: Path, records: dict) -> Path:
    """Write a pre-shaped vault: index.json with the given {key: paper-dict}."""
    root = tmp_path / "vault"
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(
        json.dumps({"version": 1, "papers": records}, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def _write_pdf(lib: Library, key: str) -> None:
    p = lib.pdf_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-1.4\nfake")


def _write_md(lib: Library, key: str) -> None:
    p = lib.md_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# fake\n", encoding="utf-8")


def _rec(key: str, status: str, **kw) -> dict:
    base = {
        "key": key,
        "title": f"{key} a sufficiently long paper title for the gate",
        "authors": ["X"],
        "year": 2020,
        "abstract": "some abstract text",
        "download_status": status,
    }
    base.update(kw)
    return base


def test_round_trip_over_preshaped_index(tmp_path):
    """The headline test: a legacy index.json → migrate → save → reload →
    every record canonical, provenance preserved/backfilled."""
    records = {
        "Ok2020":        _rec("Ok2020", "ok:arxiv"),
        "OkAgg2021":     _rec("OkAgg2021", "ok:oa_aggregators"),
        "Recovered2019": _rec("Recovered2019", "ok:recovered"),
        "Manual2018":    _rec("Manual2018", "ok:manual"),
        "Fire2022":      _rec("Fire2022", "text-only:firecrawl"),
        "Lq2017":        _rec("Lq2017", "extract_low_quality"),
        "Pending2023":   _rec("Pending2023", "pending"),
        "Failed2016":    _rec("Failed2016", "failed"),
        "Empty2015":     _rec("Empty2015", ""),
    }
    root = _shaped_index(tmp_path, records)

    lib = Library(root)
    # Lq2017 carries no PDF and no md but has an abstract → metadata_only.
    report = migrate_status(lib)
    lib.save()

    assert report["migrated"] >= 6  # ok*4, firecrawl, lq, empty

    # Reload from disk — round-trip through index.json.
    reloaded = Library(root)

    # Every record is now a canonical D7 value.
    for key in records:
        p = reloaded.get(key)
        assert p is not None, key
        assert p.download_status in _CANONICAL, (key, p.download_status)

    # ok:<src> → ok + source backfilled from the suffix.
    assert reloaded.get("Ok2020").download_status == "ok"
    assert reloaded.get("Ok2020").download_source == "arxiv"
    assert reloaded.get("OkAgg2021").download_status == "ok"
    assert reloaded.get("OkAgg2021").download_source == "oa_aggregators"
    assert reloaded.get("Recovered2019").download_source == "recovered"
    assert reloaded.get("Manual2018").download_source == "manual"

    # text-only:firecrawl → ok + source firecrawl.
    assert reloaded.get("Fire2022").download_status == "ok"
    assert reloaded.get("Fire2022").download_source == "firecrawl"

    # extract_low_quality ghost (no PDF, no md, has abstract) → metadata_only.
    assert reloaded.get("Lq2017").download_status == "metadata_only"

    # Already-canonical values untouched.
    assert reloaded.get("Pending2023").download_status == "pending"
    assert reloaded.get("Failed2016").download_status == "failed"

    # Empty → pending.
    assert reloaded.get("Empty2015").download_status == "pending"


def test_idempotent_second_run_is_noop(tmp_path):
    """Running the migration twice is safe — the second pass migrates 0."""
    records = {
        "Ok2020":   _rec("Ok2020", "ok:scihub"),
        "Fire2022": _rec("Fire2022", "text-only:firecrawl"),
    }
    root = _shaped_index(tmp_path, records)
    lib = Library(root)

    first = migrate_status(lib)
    assert first["migrated"] == 2
    second = migrate_status(lib)
    assert second["migrated"] == 0


def test_lq_with_pdf_becomes_ok_and_resets_attempts(tmp_path):
    """extract_low_quality + a PDF on disk → ok (re-extract under new cascade)
    with attempts cleared so classify routes it back to EXTRACT."""
    records = {"Lq2017": _rec("Lq2017", "extract_low_quality", extract_attempts=2)}
    root = _shaped_index(tmp_path, records)
    lib = Library(root)
    _write_pdf(lib, "Lq2017")

    migrate_status(lib)

    p = lib.get("Lq2017")
    assert p.download_status == "ok"
    assert p.extract_attempts == 0


def test_lq_with_md_no_pdf_becomes_ok_firecrawl(tmp_path):
    """extract_low_quality with an md but no PDF (firecrawl-ish) → ok +
    source firecrawl (serveable text-only)."""
    records = {"Lq2017": _rec("Lq2017", "extract_low_quality")}
    root = _shaped_index(tmp_path, records)
    lib = Library(root)
    _write_md(lib, "Lq2017")

    migrate_status(lib)

    p = lib.get("Lq2017")
    assert p.download_status == "ok"
    assert p.download_source == "firecrawl"


def test_lq_no_disk_no_abstract_becomes_failed(tmp_path):
    """extract_low_quality with nothing on disk and no abstract → failed."""
    records = {"Lq2017": _rec("Lq2017", "extract_low_quality", abstract="")}
    root = _shaped_index(tmp_path, records)
    lib = Library(root)

    migrate_status(lib)

    assert lib.get("Lq2017").download_status == "failed"


def test_existing_download_source_not_clobbered(tmp_path):
    """If a record already has a download_source, the ok:<src> suffix does
    NOT overwrite it (the explicit field is authoritative)."""
    records = {
        "Ok2020": _rec("Ok2020", "ok:arxiv", download_source="crossref_tm"),
    }
    root = _shaped_index(tmp_path, records)
    lib = Library(root)

    migrate_status(lib)

    p = lib.get("Ok2020")
    assert p.download_status == "ok"
    assert p.download_source == "crossref_tm"  # preserved, not overwritten


# ---- BLOCKER: un-gated firecrawl stub + real PDF (state-machine) -----------


def _write_firecrawl_md(lib: Library, key: str, body: str = "stub body") -> None:
    """Write a firecrawl-sourced md (YAML frontmatter ``source: firecrawl``)."""
    p = lib.md_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\nsource: firecrawl\n---\n\n{body}\n", encoding="utf-8")


def test_ungated_firecrawl_stub_over_real_pdf_is_deleted_so_pdf_extracts(tmp_path):
    """The blocker: a firecrawl md stub (un-gated, predates the gate) coexisting
    with a real PDF and NO exhausted stamp would, after migration, rest at
    classify rule (2) TERMINAL serving the stub forever while the real PDF is
    never OCR'd. The migration must DELETE the stub so classify routes the real
    PDF to EXTRACT and (after extraction) the served full text is the real body.
    """
    from papervault.library.services import classify as classify_mod

    records = {
        # status ok:scihub = a real PDF WAS downloaded after the firecrawl md;
        # firecrawl_pdf_hunt_exhausted defaults False (never re-gated).
        "Provornikova2023": _rec("Provornikova2023", "ok:scihub",
                                 download_source="firecrawl",
                                 md_engine="firecrawl"),
    }
    root = _shaped_index(tmp_path, records)
    lib = Library(root)
    _write_pdf(lib, "Provornikova2023")
    _write_firecrawl_md(lib, "Provornikova2023",
                        body="You have been blocked (Cloudflare)")

    p = lib.get("Provornikova2023")
    assert lib.md_source(p.key) == "firecrawl"
    assert p.firecrawl_pdf_hunt_exhausted is False
    # Before the fix this would be rule (2) TERMINAL (has_pdf ∧ has_md).
    assert classify_mod.classify(p, lib) == classify_mod.TERMINAL

    report = migrate_status(lib)
    lib.save()

    assert report["by_rule"].get("firecrawl_stub_over_pdf_deleted") == 1

    reloaded = Library(root)
    rp = reloaded.get("Provornikova2023")
    # Stub md is gone; the real PDF remains.
    assert not reloaded.has_extract("Provornikova2023", "md")
    assert reloaded.has_pdf("Provornikova2023")
    assert rp.download_status == "ok"
    assert rp.extract_attempts == 0
    # classify now routes the real PDF to EXTRACT (no longer TERMINAL).
    assert classify_mod.classify(rp, reloaded) == classify_mod.EXTRACT


def test_ungated_stub_deletion_then_real_extract_serves_real_full_text(
        tmp_path, monkeypatch):
    """End-to-end: after the migration deletes the stub, extract_md OCRs the
    real PDF and serve-safety surfaces the REAL body (not the stub)."""
    import asyncio

    from papervault.library import extract
    from papervault.library.mcp.server import _attach_text_reference
    from papervault.library.services import classify as classify_mod

    records = {
        "Seager2003": _rec("Seager2003", "ok:arxiv",
                           download_source="firecrawl", md_engine="firecrawl"),
    }
    root = _shaped_index(tmp_path, records)
    lib = Library(root)
    lib.pdf_path("Seager2003").parent.mkdir(parents=True, exist_ok=True)
    lib.pdf_path("Seager2003").write_bytes(b"%PDF-1.4 real")
    _write_firecrawl_md(lib, "Seager2003", body="paywall landing-page stub")

    migrate_status(lib)
    lib.save()

    p = lib.get("Seager2003")
    assert classify_mod.classify(p, lib) == classify_mod.EXTRACT

    # Stub the single-engine MinerU spine so extract_md re-extracts the real PDF
    # to a real body (2026-06-06 migration: one whole-PDF ``extract_mineru`` call,
    # no chunk cascade / GPU pin). Both LLM judges pass.
    real_body = "REAL OCR BODY of the actual paper. " * 30
    monkeypatch.setattr(extract, "pdf_probe",
                        lambda *a, **kw: extract.PDFProbe(
                            bad=False, n_pages=1, reason="ok"))

    async def fake_mineru(pdf_bytes, endpoints, *, stem="doc", **kw):
        return real_body
    monkeypatch.setattr(extract, "extract_mineru", fake_mineru)
    monkeypatch.setattr(extract, "review_extract",
                        lambda text, *, llm=None: {"ok": True, "issues": [],
                                                   "confidence": 1.0})
    monkeypatch.setattr(extract, "completeness_gate",
                        lambda text, *, llm=None: {"complete": True,
                                                   "reason": "complete"})

    out = asyncio.run(extract.extract_md(p, lib, llm=object()))
    assert out == real_body
    assert p.download_status == "ok"
    # serve-safety surfaces the real OCR body via text_path.
    assert lib.md_path("Seager2003").read_text() == real_body
    rec: dict = {}
    _attach_text_reference(rec, p, lib)
    assert "text_path" in rec and "text_status" not in rec


def test_canonical_ok_row_with_stub_deletion_triggers_save(tmp_path):
    """R2: a row already canonical ``ok`` (so the status-migration counter
    ``migrated`` stays 0) carrying an un-gated firecrawl stub over a real PDF.

    Pass A deletes the stub and clears md_path / md_engine / extract_attempts in
    memory, but ``migrated`` is NOT bumped (it counts download_status value
    migrations only). The daemon's save guard must therefore still fire on the
    Pass A counter — otherwise migrated==0 → save() skipped → the in-memory
    md_path=None never persists → reloaded index.json still points at the
    now-deleted file (serve-safety would re-impersonate the stub).

    redrill: this test invokes the PRODUCTION guard ``should_persist`` (the same
    function ``mcp/__main__._run_async`` calls) — it does NOT replicate the
    predicate inline. So reverting __main__ to ``if report["migrated"]: save()``
    or renaming the by_rule key on one side would break this test, not slip past.
    """
    from papervault.library.services.migrate_status import (
        STUB_DELETION_RULE, should_persist)

    records = {
        # Already canonical ``ok`` — NOT a legacy value, so the status loop
        # leaves it untouched (migrated stays 0). The only mutation is Pass A.
        "Canonical2024": _rec("Canonical2024", "ok",
                              download_source="firecrawl",
                              md_engine="firecrawl",
                              md_path="extracts/md/Canonical2024.md"),
    }
    root = _shaped_index(tmp_path, records)
    lib = Library(root)
    _write_pdf(lib, "Canonical2024")
    _write_firecrawl_md(lib, "Canonical2024", body="blocked stub")

    report = migrate_status(lib)

    # The status-migration counter does NOT see this (the row was canonical).
    assert report["migrated"] == 0
    # But Pass A flags the deletion (under the shared rule-key constant).
    assert report["by_rule"].get(STUB_DELETION_RULE) == 1

    # The PRODUCTION save-guard (NOT an inline copy) MUST fire even though
    # migrated==0, because a Pass A deletion occurred.
    assert should_persist(report), \
        "save guard must fire on a stub-deletion-only boot"

    # Persist (what the guard authorizes) and confirm the deletion round-trips:
    # the reloaded record no longer points at the deleted md file.
    lib.save()
    reloaded = Library(root)
    rp = reloaded.get("Canonical2024")
    assert rp.md_path is None
    assert not reloaded.has_extract("Canonical2024", "md")


def test_should_persist_predicate_unit(tmp_path):
    """redrill: a focused unit test of the PRODUCTION save-guard ``should_persist``
    directly, covering every branch so a regression in the guard itself (not just
    its wiring) is caught:

      * pure stub-deletion boot (migrated==0, stub key>0) → True
      * pure value-migration boot (migrated>0, no stub) → True
      * a clean no-op boot (migrated==0, no stub) → False  ← the bug class:
        if someone reverts to ``if report["migrated"]: save()`` THIS still holds,
        but the stub-only case above flips to a False that breaks the wiring test.
    """
    from papervault.library.services.migrate_status import (
        STUB_DELETION_RULE, should_persist)

    # Stub-deletion-only boot: must persist (the R2 bug class).
    assert should_persist({"migrated": 0, "by_rule": {STUB_DELETION_RULE: 1}})
    # Value-migration-only boot: must persist.
    assert should_persist({"migrated": 3, "by_rule": {}})
    # Both: persist.
    assert should_persist({"migrated": 2, "by_rule": {STUB_DELETION_RULE: 1}})
    # Clean no-op boot: must NOT persist (nothing changed in memory).
    assert not should_persist({"migrated": 0, "by_rule": {}})
    # Tolerates a missing by_rule entirely.
    assert not should_persist({"migrated": 0})
