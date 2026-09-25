"""Abstract-only docs (#144): a library paper held only as metadata + abstract enters the graph
as ONE document built from its metadata and abstract, marked abstract-only everywhere.

- Class: a paper in KS metadata state (no extract) whose library download_status is
  `metadata_only` (no PDF could be found) and whose abstract is non-empty. A paper still on its
  way to full text (PDF awaiting OCR, pending / failed download) stays META: its full text would
  only delete the abstract doc again.

- Text: an explicit first line `ABSTRACT_HEADER`, then title / authors / year / venue / DOI /
  arXiv id / abstract, each capped so the whole doc stays one chunk (KS_CHUNK_TOKEN_SIZE 2400).
- Fingerprint: `ABSTRACT:<sha256 of that text>` — a class of its own, distinct from `META` and
  from a full-text sha256, so a paper that later gains full text flips fingerprint and its
  abstract doc is deleted and replaced (distill / round REDISTILL path).
- Ledger: a processed abstract doc is `done_abstract` (not `done`), so the class is countable.
- Kill switch: KS_ABSTRACT_DOCS=0 makes `fingerprint()` return `META` again for these papers; the
  next scheduler rounds then delete the abstract docs and write `done_meta` back. The one-step
  offline version is `python -m papervault.knowledge.cli rollback-abstracts --apply`.

Pure helpers only (no ledger / LightRAG imports), so fingerprint, distill, round, synth and
aquery can all share them without import cycles.
"""
from __future__ import annotations

import hashlib
import html
import os
import re
from typing import Optional

from papervault.knowledge.ingest.paper_library_client import PaperRecord

ABSTRACT_HEADER = "[ABSTRACT ONLY — full text not available]"
# paper-library's DOWNLOAD_STATUS_METADATA_ONLY (library/models.py), read through the on-disk
# index contract like every other field (KS never imports the library plane).
ELIGIBLE_DOWNLOAD_STATUS = "metadata_only"
ABSTRACT_FP_PREFIX = "ABSTRACT:"
DONE_ABSTRACT = "done_abstract"

# Caps that keep the doc inside one chunk. Real abstracts run ~1-2.5k chars; 4000 chars of dense
# scientific prose is ~1.5-2k tiktoken tokens, under the 2400-token chunk with the metadata lines.
_MAX_TITLE_CHARS = 300
_MAX_VENUE_CHARS = 200
_MAX_AUTHORS = 10
_MAX_AUTHOR_CHARS = 80
_MAX_ABSTRACT_CHARS = 4000

_WS = re.compile(r"\s+")
# Markup tags only (JATS `<jats:p>`, `<p>`, `<sub>` from Crossref-style abstracts). A tag must open
# with a letter right after `<` (or `</`), so math like `x < 5 and y > 3` is left alone.
_TAG = re.compile(r"</?[A-Za-z][\w:.-]*(?:\s[^<>]*)?/?>")


def abstract_docs_enabled() -> bool:
    """KS_ABSTRACT_DOCS (default on). Read at call time so a restart with =0 takes effect."""
    return os.getenv("KS_ABSTRACT_DOCS", "1").strip().lower() not in ("0", "false", "no", "off")


def _clean(text: Optional[str]) -> str:
    return _WS.sub(" ", _TAG.sub(" ", html.unescape(text or ""))).strip()


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip() + " …"


def build_abstract_doc(rec: PaperRecord) -> Optional[str]:
    """The abstract-only doc text for `rec`, or None when `rec` is not in the class (not
    download_status=metadata_only, or no usable abstract)."""
    if rec.download_status != ELIGIBLE_DOWNLOAD_STATUS:
        return None
    abstract = _clean(rec.abstract)
    if not abstract:
        return None
    lines = [ABSTRACT_HEADER]
    title = _clean(rec.title)
    if title:
        lines.append("Title: " + _cap(title, _MAX_TITLE_CHARS))
    authors = [_cap(_clean(a), _MAX_AUTHOR_CHARS) for a in (rec.authors or []) if _clean(a)]
    if authors:
        shown = ", ".join(authors[:_MAX_AUTHORS])
        lines.append("Authors: " + shown + (" et al." if len(authors) > _MAX_AUTHORS else ""))
    if rec.year:
        lines.append(f"Year: {rec.year}")
    venue = _clean(rec.venue)
    if venue:
        lines.append("Venue: " + _cap(venue, _MAX_VENUE_CHARS))
    doi = _clean(rec.doi)
    if doi:
        lines.append("DOI: " + doi)
    arxiv = _clean(rec.arxiv_id)
    if arxiv:
        lines.append("arXiv: " + arxiv)
    lines.append("Abstract: " + _cap(abstract, _MAX_ABSTRACT_CHARS))
    return "\n".join(lines)


def abstract_fingerprint(text: str) -> str:
    return ABSTRACT_FP_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_abstract_fp(fp: Optional[str]) -> bool:
    return bool(fp) and fp.startswith(ABSTRACT_FP_PREFIX)


def is_abstract_text(text: Optional[str]) -> bool:
    """True when `text` (a doc's full_docs content or one of its chunks) is an abstract-only doc."""
    return bool(text) and text.lstrip().startswith(ABSTRACT_HEADER)


def strip_abstract_header(text: str) -> str:
    """The doc/chunk text without its header line (the label carries the class instead)."""
    body = text.lstrip()
    if body.startswith(ABSTRACT_HEADER):
        body = body[len(ABSTRACT_HEADER):]
    return body.strip()


def done_status_for(fp: Optional[str]) -> str:
    """The ledger success status for a processed doc enqueued with fingerprint `fp`."""
    return DONE_ABSTRACT if is_abstract_fp(fp) else "done"
