"""Unit tests for the ETA estimator (D14)."""
from __future__ import annotations


from papervault.library import Library
from papervault.library.services.download_queue import DownloadQueue
from papervault.library.services.eta import (
    DOWNLOAD_AVG_SECONDS,
    DOWNLOAD_INFLIGHT_SECONDS,
    EXTRACT_INFLIGHT_SECONDS,
    MINERU_SECONDS_PER_PAGE,
    TYPICAL_PAGES,
    available_fields,
    estimate_eta,
)
from papervault.library.services.extract_queue import ExtractQueue


def _make_lib(tmp_path):
    lib = Library(tmp_path)
    p, _ = lib.upsert({
        "title": "Padded test paper title for validation",
        "authors": ["Alpha"],
        "year": 2024,
        "doi": "10.1/eta",
        "abstract": "An abstract is here.",
    })
    return lib, p


def test_eta_for_paper_needing_download_and_extract(tmp_path):
    lib, paper = _make_lib(tmp_path)
    dq = DownloadQueue(lib, num_workers=0)
    eq = ExtractQueue(lib, num_workers=0)
    result = estimate_eta(paper, lib, dq, eq)
    expected = (
        DOWNLOAD_AVG_SECONDS
        + TYPICAL_PAGES * MINERU_SECONDS_PER_PAGE
        + DOWNLOAD_INFLIGHT_SECONDS
        + EXTRACT_INFLIGHT_SECONDS
    )
    assert result["eta_seconds"] == expected
    assert "URGENT" in result["eta_note"]
    assert "download" in result["eta_note"]
    assert "extract" in result["eta_note"]


def test_eta_for_paper_with_pdf_only(tmp_path):
    """PDF on disk → only extract is pending; ETA drops by download cost."""
    lib, paper = _make_lib(tmp_path)
    lib.pdf_path(paper.key).write_bytes(b"%PDF-1.0 minimal")
    dq = DownloadQueue(lib, num_workers=0)
    eq = ExtractQueue(lib, num_workers=0)
    result = estimate_eta(paper, lib, dq, eq)
    # Download terms drop out; extract terms remain. PDF page count
    # falls back to TYPICAL_PAGES because pypdf can't parse the fake.
    expected = TYPICAL_PAGES * MINERU_SECONDS_PER_PAGE + EXTRACT_INFLIGHT_SECONDS
    assert result["eta_seconds"] == expected
    assert "download" not in result["eta_note"]


def test_available_fields_includes_disk_artifacts(tmp_path):
    lib, paper = _make_lib(tmp_path)
    fields = available_fields(paper, lib)
    assert "title" in fields
    assert "abstract" in fields
    assert "doi" in fields
    assert "pdf" not in fields and "md" not in fields and "txt" not in fields

    lib.pdf_path(paper.key).write_bytes(b"x")
    lib.txt_path(paper.key).write_text("body")
    fields = available_fields(paper, lib)
    assert "pdf" in fields
    assert "txt" in fields
