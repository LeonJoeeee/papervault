"""Zotero Web API sync — push paper-library metadata + abstracts to a
Zotero user library.

Design choices (see CLAUDE.md "Zotero integration" section):

  * **Metadata + abstract only.** Zotero Web API has no concept of
    linked-file attachments to arbitrary local paths (the JSON schema
    accepts ``linkMode: linked_file`` but stores no path — only the
    bare ``filename``). Uploaded attachments would burn Zotero's 300 MB
    free quota at scale. So we sync the JSON record only and rely on
    Zotero Desktop's BibTeX importer (``library.bib`` includes
    ``file = {...}`` fields) to attach PDFs / extracts when the user
    wants them locally.
  * **Idempotent.** Once a paper is pushed, its ``Paper.zotero_key`` is
    set. Subsequent ``sync_all()`` calls skip those papers entirely. To
    force a re-push, clear ``paper.zotero_key`` in index.json.
  * **First-run dedup.** If the user already has items in Zotero (e.g.
    from a prior BibTeX import), the first sync scans existing items by
    DOI and arXiv ID and links them back to library entries — no
    duplicate items get created.
  * **Batched.** Zotero accepts at most 50 items per ``create_items``
    POST. We chunk accordingly.
  * **Scale target.** Designed to handle 3-5k papers cleanly; the full
    sync is ~3000 items / 50 per batch ≈ 60 POSTs ≈ a few minutes.

Usage::

    from papervault.library.integrations.zotero import ZoteroSync
    from papervault.library.store import Library
    lib = Library()  # vault root from papervault.config
    sync = ZoteroSync(lib, api_key=os.environ["ZOTERO_API_KEY"],
                       user_id=int(os.environ["ZOTERO_USER_ID"]))
    report = sync.sync_all(dry_run=False)
    print(report)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable, Optional

from ..models import Paper
from ..store import Library

log = logging.getLogger(__name__)

_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+", re.I)
_ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")


def _normalize_doi(doi: str) -> str:
    return (doi or "").strip().lower().rstrip(".,;)")


def _normalize_arxiv(aid: str) -> str:
    if not aid:
        return ""
    s = aid.strip().lower()
    s = s.replace("arxiv:", "").replace("arxiv.org/abs/", "")
    s = re.sub(r"v\d+$", "", s)  # strip version
    return s


class ZoteroSync:
    """Push paper-library metadata to a Zotero user library.

    Stateless apart from the library passed in — all sync state is stored
    in :attr:`Paper.zotero_key`. Re-running ``sync_all()`` is safe.
    """

    BATCH_SIZE = 50  # Zotero Web API hard limit
    HTTP_TIMEOUT = 90.0  # seconds; pyzotero's default httpx timeout is short
    MAX_RETRIES = 3      # per-batch retry on ReadTimeout / network errors

    def __init__(self, library: Library, *, api_key: str, user_id: int):
        try:
            from pyzotero import zotero  # type: ignore
            import httpx  # pyzotero's HTTP backend
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pyzotero not installed. `pip install pyzotero` first."
            ) from exc
        if not api_key:
            raise ValueError("ZOTERO_API_KEY is required")
        if not user_id:
            raise ValueError("ZOTERO_USER_ID is required")
        self.library = library
        # Let pyzotero construct its own httpx.Client so the Authorization
        # / Zotero-API-Version / Content-Type default_headers() get attached.
        # Then bump the timeout — pyzotero's default is httpx's default (5 s),
        # which times out 50-item POSTs when Zotero's backend is busy.
        self._zot = zotero.Zotero(user_id, "user", api_key)
        try:
            self._zot.client.timeout = httpx.Timeout(self.HTTP_TIMEOUT)
        except Exception:
            log.warning("could not extend pyzotero httpx timeout; using default")

    # ------------------------------------------------------------------
    # Conversion: Paper → Zotero item dict
    # ------------------------------------------------------------------

    @staticmethod
    def _paper_to_item(paper: Paper, template: dict[str, Any]) -> dict[str, Any]:
        """Fill a Zotero item template from a Paper. Mutates and returns the
        template. ``template`` should come from
        ``zot.item_template('journalArticle' or 'preprint')``.
        """
        item = dict(template)  # shallow copy
        if paper.title:
            item["title"] = paper.title
        if paper.abstract:
            item["abstractNote"] = paper.abstract
        if paper.year:
            item["date"] = str(paper.year)
        if paper.venue:
            # 'publicationTitle' for journalArticle; 'repository' is set
            # separately for preprint below.
            if "publicationTitle" in item:
                item["publicationTitle"] = paper.venue
        if paper.doi:
            item["DOI"] = paper.doi
        if paper.url:
            item["url"] = paper.url
        if paper.authors:
            item["creators"] = [
                {"creatorType": "author", "name": a} for a in paper.authors
            ]
        # arXiv handling: if no DOI but arxiv_id present, fill preprint fields
        if paper.arxiv_id and "archiveID" in item:
            item["archiveID"] = paper.arxiv_id
            item["repository"] = "arXiv"
            if paper.url == "":
                item["url"] = f"https://arxiv.org/abs/{paper.arxiv_id}"
        # `extra` is free-form; round-trip our citation key + ingest
        # provenance so Zotero exports stay aligned with paper-library's
        # internal naming. Better-BibTeX users see this as a custom field.
        extra_lines = [f"Citation Key: {paper.key}"]
        if paper.arxiv_id and not paper.doi:
            extra_lines.append(f"arXiv: {paper.arxiv_id}")
        if paper.source:
            extra_lines.append(f"tex.ingestSource: {paper.source}")
        item["extra"] = "\n".join(extra_lines)
        return item

    def _build_item(self, paper: Paper) -> dict[str, Any]:
        """Choose the right Zotero item type and fill it from a Paper."""
        # Preprint type for arxiv-only records; otherwise journalArticle.
        is_preprint = bool(paper.arxiv_id) and not paper.doi
        item_type = "preprint" if is_preprint else "journalArticle"
        template = self._zot.item_template(item_type)
        return self._paper_to_item(paper, template)

    # ------------------------------------------------------------------
    # First-run dedup: scan existing Zotero items, map DOI/arxiv → key
    # ------------------------------------------------------------------

    def scan_existing(self) -> dict[str, str]:
        """Return a {identifier → zotero_item_key} map by walking the user's
        existing Zotero library. Identifiers include normalized DOI and
        arxiv ID. Useful when the user previously imported via BibTeX
        and we want to link those items rather than re-create.

        Fail-safe: if Zotero rate-limits or returns 403 (observed after
        bursts of timeouts on POST), return an empty map and log a warning
        instead of crashing the whole sync. Per-paper ``zotero_key`` is
        the primary dedup mechanism; scan_existing is only an extra layer
        for first-run dedup against user-pre-existing items.
        """
        index: dict[str, str] = {}
        # Pagination: 50 per page (smaller than max-100 to reduce timeout risk).
        start = 0
        page_size = 50
        # Iterate item types one at a time — the `||` (OR) syntax sometimes
        # triggers 403 from the Zotero API when the key has just been
        # exercised heavily.
        item_types = ["journalArticle", "preprint", "conferencePaper",
                       "book", "bookSection", "report"]
        for it_type in item_types:
            start = 0
            while True:
                try:
                    items = self._zot.items(start=start, limit=page_size,
                                              itemType=it_type)
                except Exception as exc:
                    log.warning("scan_existing(%s, start=%d) failed: %r — "
                                 "continuing without pre-existing dedup",
                                 it_type, start, exc)
                    break
                if not items:
                    break
                for it in items:
                    data = it.get("data") or {}
                    key = data.get("key")
                    if not key:
                        continue
                    doi = _normalize_doi(data.get("DOI") or "")
                    if doi:
                        index[f"doi:{doi}"] = key
                    aid = _normalize_arxiv(data.get("archiveID") or "")
                    if aid:
                        index[f"arxiv:{aid}"] = key
                    extra = data.get("extra") or ""
                    for m in _DOI_RE.finditer(extra):
                        index.setdefault(
                            f"doi:{_normalize_doi(m.group(0))}", key)
                    for m in _ARXIV_RE.finditer(extra):
                        index.setdefault(
                            f"arxiv:{_normalize_arxiv(m.group(1))}", key)
                if len(items) < page_size:
                    break
                start += page_size
        return index

    @staticmethod
    def _paper_identifiers(paper: Paper) -> list[str]:
        ids: list[str] = []
        if paper.doi:
            ids.append(f"doi:{_normalize_doi(paper.doi)}")
        if paper.arxiv_id:
            ids.append(f"arxiv:{_normalize_arxiv(paper.arxiv_id)}")
        return ids

    # ------------------------------------------------------------------
    # Main sync
    # ------------------------------------------------------------------

    def sync_all(
        self,
        *,
        dry_run: bool = False,
        on_progress: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Push every library paper without ``zotero_key`` to Zotero.

        Args:
            dry_run: if True, build the items and print plan, but POST nothing.
            on_progress: optional callable invoked as ``(stage, n_done, total)``
                — stages: ``"scan"``, ``"linked"``, ``"created"``.

        Returns ``{"linked": N, "created": M, "skipped": K, "errors": [...]}``.
        """
        report: dict[str, Any] = {"linked": 0, "created": 0,
                                  "skipped": 0, "errors": []}
        papers = self.library.all_papers()

        # Stage 1: scan existing Zotero items, dedup by DOI/arxiv
        existing = self.scan_existing() if any(
            p.zotero_key is None for p in papers) else {}
        if on_progress:
            on_progress("scan", len(existing), len(existing))

        to_create: list[tuple[Paper, dict[str, Any]]] = []
        for p in papers:
            if p.zotero_key:
                report["skipped"] += 1
                continue
            # Try to link to existing Zotero item
            linked_key: Optional[str] = None
            for ident in self._paper_identifiers(p):
                if ident in existing:
                    linked_key = existing[ident]
                    break
            if linked_key:
                if not dry_run:
                    p.zotero_key = linked_key
                report["linked"] += 1
                continue
            # Build new item
            try:
                item = self._build_item(p)
            except Exception as exc:
                report["errors"].append(
                    {"key": p.key, "stage": "build", "error": repr(exc)[:200]})
                continue
            to_create.append((p, item))

        # Stage 2: batched creates
        if not to_create:
            if not dry_run:
                self.library.save()
            return report

        if dry_run:
            report["created"] = len(to_create)  # would-be count
            return report

        import time as _time
        for i in range(0, len(to_create), self.BATCH_SIZE):
            chunk = to_create[i:i + self.BATCH_SIZE]
            payload = [item for _p, item in chunk]
            resp = None
            last_exc: Optional[Exception] = None
            for attempt in range(self.MAX_RETRIES):
                try:
                    resp = self._zot.create_items(payload)
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < self.MAX_RETRIES - 1:
                        # Exponential backoff: 2s, 4s, 8s
                        _time.sleep(2 ** (attempt + 1))
                    continue
            if resp is None:
                report["errors"].append({
                    "stage": "create",
                    "batch_start": i,
                    "error": repr(last_exc)[:200] if last_exc else "unknown",
                })
                continue
            # Response shape: {"successful": {"0": {"key": "XX", ...}, ...},
            #                  "failed": {"3": {"code":..., "message":...}}}
            successful = (resp or {}).get("successful") or {}
            failed = (resp or {}).get("failed") or {}
            for idx_str, returned_item in successful.items():
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                if idx >= len(chunk):
                    continue
                paper = chunk[idx][0]
                key = (returned_item.get("data") or {}).get("key") or returned_item.get("key")
                if key:
                    paper.zotero_key = key
                    report["created"] += 1
            for idx_str, fail in failed.items():
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                paper = chunk[idx][0] if idx < len(chunk) else None
                report["errors"].append({
                    "key": paper.key if paper else None,
                    "stage": "create_response",
                    "error": fail.get("message", "")[:200],
                })
            if on_progress:
                on_progress("created", i + len(chunk), len(to_create))
            # Persist incrementally so a crash mid-sync doesn't waste prior pushes
            self.library.save()

        return report
