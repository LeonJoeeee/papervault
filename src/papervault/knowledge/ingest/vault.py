"""v3 vault reader — KS reads pl's paper-vault READ-ONLY (SDD §5.1).

Thin layer over the proven `paper_library_client` (run.py also uses it):
- load_clean_index(): {key: PaperRecord} clean view. pl purges off-domain at
  source, so the residual `domain_status` field is all-None and the client's
  exclude filter is a no-op (SDD §5.1).
- read_extract_raw(rec): RAW md|txt extract text, for fingerprint (SDD §6.1.b).
  Deliberately RAW (not reference-stripped): the fingerprint must stay stable
  across clean() regex tweaks, so only true content change flips it. clean()
  (reference/ack stripping, §6.1.c) lives in the DISTILL/LightRAG layer (slice 2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from papervault.knowledge.ingest.paper_library_client import (
    DEFAULT_VAULT,
    PaperRecord,
    load_vault_index,
)

__all__ = ["DEFAULT_VAULT", "PaperRecord", "load_clean_index", "read_extract_raw"]


def load_clean_index(vault_path: Path = DEFAULT_VAULT) -> dict[str, PaperRecord]:
    """pl 干净集 {key: PaperRecord}(全集即干净,§5.1)。只读。"""
    return load_vault_index(vault_path)  # include_quarantined=False → 排掉 domain_status!=None(现全 None)


def read_extract_raw(rec: PaperRecord, vault_path: Path = DEFAULT_VAULT) -> Optional[str]:
    """RAW md|txt 全文(md 优先);两者皆无/缺文件 → None。供 fingerprint 用,不剥引用。"""
    for rel in (rec.md_path, rec.txt_path):
        if rel:
            full = vault_path / rel
            if full.exists():
                return full.read_text(encoding="utf-8", errors="replace")
    return None
