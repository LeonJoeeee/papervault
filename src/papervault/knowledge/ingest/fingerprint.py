"""Content fingerprint for incremental sync (SDD §4.1 / §6.1.b).

fingerprint(rec) = sha256(normalized RAW extract)  |  META.

- META = 元数据态: index records neither md_path nor txt_path. This is the
  AUTHORITATIVE "no full text" 判据 (SDD §2/§4.1; 2026-06-01 实测 == 691 papers,
  口径由 test_fingerprint loud-assert 守, 随 pl OCR 补全文漂移).
- META → hash transition (a paper later gets OCR'd) is exactly what flips the
  fingerprint and triggers REDISTILL (§6.1).
- Hash RAW extract (not clean()'d) so the fingerprint is stable across clean()
  regex changes — only genuine content change flips it.
"""
from __future__ import annotations

import hashlib
import re

from papervault.knowledge.ingest.paper_library_client import PaperRecord
from papervault.knowledge.ingest.vault import read_extract_raw

META = "META"
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """折叠空白 + strip,使指纹与排版/换行噪声解耦。"""
    return _WS.sub(" ", text).strip()


def is_metadata_only(rec: PaperRecord) -> bool:
    """元数据态判据(权威):index 既无 md_path 又无 txt_path。"""
    return not rec.md_path and not rec.txt_path


def fingerprint(rec: PaperRecord) -> str:
    if is_metadata_only(rec):
        return META
    text = read_extract_raw(rec)
    if not text or not text.strip():
        return META  # 路径有但文件缺/空 → 当元数据态(待全文就绪)
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()
