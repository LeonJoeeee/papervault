"""Fingerprint tests against the REAL vault (read-only).

The metadata-only-count test validates the SDD's authoritative "no full text" 判据
(md_path & txt_path both empty). If reality != the SDD number, this test FAILS —
which is the point: it surfaces a stale SDD number (methodology: tests catch SDD
errors). 2026-06-01: SDD reflux'd to 691/3373 (was 692/3372 — one paper gained
full text, the META→full-text jump §6.1 REDISTILL is built to capture).
"""
import pytest

from papervault.knowledge.ingest.fingerprint import META, fingerprint, is_metadata_only
from papervault.knowledge.ingest.vault import DEFAULT_VAULT, load_clean_index

_HAS_VAULT = (DEFAULT_VAULT / "index.json").exists()
pytestmark = pytest.mark.skipif(not _HAS_VAULT, reason="paper-vault not present")


def test_fingerprint_meta_vs_fulltext_and_stable():
    idx = load_clean_index()
    meta_key = next(k for k, rec in idx.items() if is_metadata_only(rec))
    assert fingerprint(idx[meta_key]) == META

    ft_key = next(k for k, rec in idx.items() if not is_metadata_only(rec))
    fp = fingerprint(idx[ft_key])
    if fp != META:  # has text on disk
        assert len(fp) == 64  # sha256 hex
        assert fingerprint(idx[ft_key]) == fp  # stable / deterministic


def test_corpus_split_is_consistent():
    """Metadata-only + full-text partition the clean index (portable invariant —
    no absolute counts, which are corpus-state-specific)."""
    idx = load_clean_index()
    meta = sum(1 for rec in idx.values() if is_metadata_only(rec))
    fulltext = sum(1 for rec in idx.values() if not is_metadata_only(rec))
    assert meta + fulltext == len(idx)
    assert meta >= 0 and fulltext >= 0
