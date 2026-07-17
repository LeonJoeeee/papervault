"""Tests for the post-ingest router ``services.classify.classify`` (D7 / SDD §5).

``classify`` must be TOTAL (every paper lands somewhere) + mutually
exclusive (exactly one Action) + pure (no disk/state mutation). The five
priority-ordered rules are exercised directly here, plus the load-bearing
edge cases: terminal-wins-over-disk, firecrawl-md routes to DOWNLOAD, and
the attempts ceiling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from papervault.library.models import MAX_EXTRACT_ATTEMPTS
from papervault.library.services.classify import (
    DOWNLOAD,
    EXTRACT,
    TERMINAL,
    classify,
)
from papervault.library.store import Library


def _write_pdf(lib: Library, key: str) -> None:
    p = lib.pdf_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-1.4\nfake")


def _write_md(lib: Library, key: str, *, source: str | None = None) -> None:
    p = lib.md_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = "# fake\n"
    if source:
        body = f"---\nsource: {source}\n---\n\n" + body
    p.write_text(body, encoding="utf-8")


@pytest.fixture
def lib(tmp_path: Path) -> Library:
    return Library(tmp_path)


def _add(lib: Library, status: str, **kw):
    p, _ = lib.upsert({
        "title": kw.pop("title", "A sufficiently long paper title for the gate"),
        "authors": ["X"], "year": 2020, **kw,
    })
    p.download_status = status
    return p


# ---- rule 1: terminal states always win, even with stray disk artefacts ----


@pytest.mark.parametrize("status", ["extract_failed", "failed", "metadata_only"])
def test_terminal_status_is_terminal(lib, status):
    p = _add(lib, status)
    assert classify(p, lib) == TERMINAL


def test_terminal_wins_over_pdf_and_md_on_disk(lib):
    """A terminal record with a full extract on disk still reads TERMINAL
    (rule 1 short-circuits before the disk-fact rules) — audit-only revival."""
    p = _add(lib, "metadata_only")
    _write_pdf(lib, p.key)
    _write_md(lib, p.key)
    assert classify(p, lib) == TERMINAL


# ---- rule 2: real PDF already extracted = done ----


def test_pdf_plus_md_is_terminal(lib):
    p = _add(lib, "ok")
    _write_pdf(lib, p.key)
    _write_md(lib, p.key)
    assert classify(p, lib) == TERMINAL


# ---- rule 3: no PDF → DOWNLOAD (incl. firecrawl-md) ----


def test_pending_no_pdf_is_download(lib):
    p = _add(lib, "pending")
    assert classify(p, lib) == DOWNLOAD


def test_firecrawl_md_no_pdf_routes_to_download(lib):
    """The D8 rescue: status=ok + md on disk but NO PDF (firecrawl text-only)
    must route to DOWNLOAD to hunt the real PDF — NOT terminal, NOT extract."""
    p = _add(lib, "ok", download_source="firecrawl")
    _write_md(lib, p.key, source="firecrawl")
    assert not lib.has_pdf(p.key)
    assert lib.has_extract(p.key, "md")
    assert classify(p, lib) == DOWNLOAD


def test_ok_status_but_no_pdf_no_md_is_download(lib):
    p = _add(lib, "ok")
    assert classify(p, lib) == DOWNLOAD


def test_firecrawl_md_hunt_exhausted_rests_terminal_not_download(lib):
    """S3 (issue #1/#2): once a firecrawl-md paper's real-PDF hunt is exhausted
    (``firecrawl_pdf_hunt_exhausted`` stamped on a gate-PASS), classify must NOT
    keep re-routing it to DOWNLOAD every reconcile sweep. It RESTS in the rule-5
    catch-all (TERMINAL = skip), while serve-safety still hands out its md as
    text_path. This is the resting state that breaks the permanent hot-loop."""
    p = _add(lib, "ok", download_source="firecrawl")
    _write_md(lib, p.key, source="firecrawl")
    p.firecrawl_pdf_hunt_exhausted = True
    assert not lib.has_pdf(p.key)
    assert lib.has_extract(p.key, "md")
    # NOT DOWNLOAD (no hot-loop), NOT EXTRACT (no PDF) → resting TERMINAL.
    assert classify(p, lib) == TERMINAL


def test_firecrawl_md_not_yet_exhausted_still_downloads(lib):
    """The flag is a one-shot: a firecrawl-md paper whose hunt has NOT yet run
    (flag default False — e.g. a freshly-migrated historical paper) still routes
    DOWNLOAD exactly ONCE to look for the real PDF (D8 rescue)."""
    p = _add(lib, "ok", download_source="firecrawl")
    _write_md(lib, p.key, source="firecrawl")
    assert p.firecrawl_pdf_hunt_exhausted is False
    assert classify(p, lib) == DOWNLOAD


def test_exhausted_flag_does_not_block_extract_when_real_pdf_arrives(lib):
    """The exhausted flag only gates the firecrawl-md DOWNLOAD branch. If a real
    PDF lands while there is NO md on disk yet (has_pdf True, has_md False),
    rule 4 takes over regardless of the flag — it must NOT freeze a now-
    extractable paper."""
    p = _add(lib, "ok")
    p.firecrawl_pdf_hunt_exhausted = True
    _write_pdf(lib, p.key)  # real PDF arrived, no md yet
    assert classify(p, lib) == EXTRACT


def test_firecrawl_md_with_real_pdf_rests_terminal_no_upgrade(lib):
    """D5 / no PDF-upgrade re-OCR: a firecrawl-md paper that ALSO has a real PDF
    on disk classifies rule-2 TERMINAL (has_pdf ∧ has_md = done) — it is NEVER
    routed to EXTRACT to re-OCR the real PDF. "if firecrawl's scrape was no
    good, it stays no good": the
    firecrawl md is served as-is for good; there is no upgrade path (the dead
    force-upgrade branch was removed). serve-safety still serves the md via
    text_path; only the persisted status is the routing decision here."""
    p = _add(lib, "ok", download_source="firecrawl")
    _write_md(lib, p.key, source="firecrawl")
    _write_pdf(lib, p.key)  # a real PDF later appeared too
    assert lib.has_pdf(p.key) and lib.has_extract(p.key, "md")
    assert lib.md_source(p.key) == "firecrawl"
    assert classify(p, lib) == TERMINAL          # rule 2, NOT EXTRACT
    # the flag state is irrelevant — rule 2 fires before rule 3's flag check.
    p.firecrawl_pdf_hunt_exhausted = True
    assert classify(p, lib) == TERMINAL
    p.firecrawl_pdf_hunt_exhausted = False
    assert classify(p, lib) == TERMINAL


# ---- rule 4: PDF, no md, ok, attempts left → EXTRACT ----


def test_pdf_no_md_ok_is_extract(lib):
    p = _add(lib, "ok")
    _write_pdf(lib, p.key)
    assert classify(p, lib) == EXTRACT


def test_extract_attempts_below_ceiling_still_extract(lib):
    p = _add(lib, "ok")
    p.extract_attempts = MAX_EXTRACT_ATTEMPTS - 1
    _write_pdf(lib, p.key)
    assert classify(p, lib) == EXTRACT


# ---- rule 5: catch-all terminal ----


def test_pdf_no_md_attempts_exhausted_is_terminal(lib):
    """has_pdf, no md, ok, but attempts ran out (status not yet flipped) →
    rule 4 fails its attempts guard, so rule 5 catches it as TERMINAL."""
    p = _add(lib, "ok")
    p.extract_attempts = MAX_EXTRACT_ATTEMPTS
    _write_pdf(lib, p.key)
    assert classify(p, lib) == TERMINAL


def test_pending_with_pdf_no_md_extracts_not_blackhole(lib):
    """Restart-safety (§8 inv #4): a crash between the PDF write and the
    status flip leaves a REAL PDF under a still-'pending' record. Rule 4
    accepts status ∈ {ok, pending}, so it MUST extract — never black-hole to
    TERMINAL (a downloaded PDF lost to terminal would never get a full text)."""
    p = _add(lib, "pending")
    _write_pdf(lib, p.key)
    assert classify(p, lib) == EXTRACT


def test_nonfirecrawl_md_no_pdf_rests_terminal_not_hotloop(lib):
    """Issue #2: a REAL-OCR md whose PDF is gone (md_source != firecrawl, no
    PDF) must REST (TERMINAL, still served via text_path) — NOT re-route to
    DOWNLOAD forever. Only an un-exhausted firecrawl-md re-downloads to upgrade."""
    p = _add(lib, "ok")
    _write_md(lib, p.key)  # no source frontmatter → md_source != "firecrawl"
    assert not lib.has_pdf(p.key)
    assert lib.has_extract(p.key, "md")
    assert lib.md_source(p.key) != "firecrawl"
    assert classify(p, lib) == TERMINAL


# ---- totality + mutual exclusivity + purity ----


def test_classify_is_total_and_single_valued(lib):
    """Across a spread of (status × disk) combos, classify always returns
    exactly one of the three Actions — never None, never a surprise."""
    valid = {DOWNLOAD, EXTRACT, TERMINAL}
    combos = [
        ("pending", False, False),
        ("pending", True, False),
        ("ok", False, False),
        ("ok", True, False),
        ("ok", True, True),
        ("ok", False, True),   # firecrawl-md
        ("failed", False, False),
        ("metadata_only", False, False),
        ("extract_failed", True, False),
    ]
    for i, (status, pdf, md) in enumerate(combos):
        p = _add(lib, status, title=f"Totality probe paper number {i} long enough")
        if pdf:
            _write_pdf(lib, p.key)
        if md:
            _write_md(lib, p.key)
        result = classify(p, lib)
        assert result in valid, (status, pdf, md, result)


def test_classify_does_not_mutate(lib):
    """Pure function: classify reads but never writes paper state."""
    p = _add(lib, "ok")
    p.extract_attempts = 1
    _write_pdf(lib, p.key)
    before = (p.download_status, p.download_source, p.extract_attempts)
    classify(p, lib)
    after = (p.download_status, p.download_source, p.extract_attempts)
    assert before == after
