"""Paper ingestion front door: identify → fetch metadata → upsert → ENQUEUE.

The orchestration layer behind the ``paper-library add`` CLI. Designed to be
called from any caller (CLI, pipeline, batch).

D11 (enqueue-only): ``add`` resolves the identifier, upserts the metadata card
into the library as ``download_status="pending"``, persists, and RETURNS. It
does **not** download the PDF or run OCR synchronously. The running daemon's
download queue picks the card up — ``DownloadQueue.start()`` recovery and
``reconcile_once`` (D8) both re-enqueue every ``pending ∧ ¬has_pdf`` paper from
on-disk state, so a CLI process (separate from the daemon, no in-memory queue
handle) "drops into the queue" simply by persisting the card as ``pending``.

Why: the old synchronous ``add`` ran ``download_paper`` + ``extract_md`` in the
CLI process, contending with the daemon for the SAME GPU across processes
(double-model OOM) and blocking the terminal for minutes. D11 routes every
download/OCR through the one background belt; the CLI is a thin enqueue.
"""

from __future__ import annotations

from typing import Optional

from .. import fetch as fetcher
from ..models import DOWNLOAD_STATUS_PENDING, DOWNLOAD_STATUS_TERMINAL, Paper
from ..store import Library
from .resolver import ResolverService


def _paper_metadata_dict(paper: Paper, library: Library) -> dict:
    return {
        "key": paper.key,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "venue": paper.venue,
        "abstract": paper.abstract,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "citation_count": paper.citation_count,
        "is_review": paper.is_review,
        "has_pdf": library.has_pdf(paper.key),
        "has_extract_md": library.has_extract(paper.key, "md"),
        "has_extract_txt": library.has_extract(paper.key, "txt"),
        "md_engine": paper.md_engine,
        "md_engine_version": paper.md_engine_version,
        "added_at": paper.added_at,
    }


class AddService:

    def __init__(self, library: Library, llm=None):
        self.library = library
        self._llm = llm
        self._resolver: Optional[ResolverService] = None

    def _resolver_svc(self) -> ResolverService:
        if self._resolver is None:
            self._resolver = ResolverService(self.library, llm=self._llm)
        return self._resolver

    def add(self, identifier: str, *, force_refresh: bool = False) -> dict:
        """Idempotent paper add. See module docstring for the spec.

        Returns a dict matching the MCP `add_paper` AddResult schema.
        """
        ident = (identifier or "").strip()
        if not ident:
            return {"status": "not_found", "key": None, "metadata": None,
                    "candidates": None, "message": "empty identifier"}

        # 1. exact-id resolution
        metadata: Optional[dict] = None
        existing: Optional[Paper] = None

        # An "exists" short-circuit is only valid when the existing record is
        # fully realized (PDF + extract). Many keyword-search candidates land
        # in the library as metadata-only via upsert(), so a later benchmark
        # add must still download + extract them.
        def _is_complete(p: Paper) -> bool:
            return self.library.has_pdf(p.key) and (
                self.library.has_extract(p.key, "md")
                or self.library.has_extract(p.key, "txt")
            )

        if fetcher.looks_like_doi(ident):
            existing = self.library.find(doi=ident)
            if existing and not force_refresh and _is_complete(existing):
                return self._exists_response(existing)
            if existing is None:
                metadata = fetcher.fetch_by_doi(ident)
        elif fetcher.looks_like_arxiv(ident):
            existing = self.library.find(arxiv_id=ident)
            if existing and not force_refresh and _is_complete(existing):
                return self._exists_response(existing)
            if existing is None:
                metadata = fetcher.fetch_by_arxiv(ident)
        elif ident in self.library.keys():
            existing = self.library.get(ident)
            if existing and not force_refresh and _is_complete(existing):
                return self._exists_response(existing)

        # 2. fuzzy: look in library first via resolver
        if metadata is None and existing is None:
            cands = self._resolver_svc().resolve(ident, top_k=5)
            in_lib = [c for c in cands if c.get("in_library")]
            if len(in_lib) == 1 and in_lib[0]["score"] >= 0.85:
                p = self.library.get(in_lib[0]["key"])
                if p:
                    return self._exists_response(p)
            if len(in_lib) > 1:
                return {
                    "status": "ambiguous",
                    "key": None,
                    "metadata": None,
                    "candidates": cands,
                    "message": ("multiple library entries match; call add_paper "
                                "again with a specific key/DOI/arxiv_id"),
                }
            # no library hit → no further auto-fetch (would need broader external search,
            # let the caller use search.search_all explicitly).
            return {
                "status": "not_found",
                "key": None,
                "metadata": None,
                "candidates": None,
                "message": "no library match and identifier is not a recognizable DOI/arxiv id",
            }

        if metadata is None and existing is None:
            return {
                "status": "not_found",
                "key": None,
                "metadata": None,
                "candidates": None,
                "message": f"could not fetch metadata for {ident!r}",
            }

        # 3. upsert into library (gives us a key + dedupes). If an existing
        # record was already complete (PDF + extract), short-circuit. Otherwise
        # we proceed to fill in what's missing.
        if metadata is not None:
            paper, was_new = self.library.upsert(metadata)
            if paper is None:
                # quality gate rejected — phantom-shaped metadata
                return {
                    "status": "rejected",
                    "key": None,
                    "metadata": None,
                    "candidates": None,
                    "message": "metadata rejected by quality gate (phantom-shaped); "
                               "see manifest.log for reason",
                }
        else:
            assert existing is not None
            paper, was_new = existing, False
        if not was_new and not force_refresh and _is_complete(paper):
            self.library.save()
            return self._exists_response(paper)

        # 4. ENQUEUE-ONLY (D11). The card is in the library; mark it ``pending``
        # and persist. We do NOT call download_paper / extract_md here — the
        # daemon's download queue owns all PDF fetch + OCR. A CLI process has no
        # in-memory queue handle, so "enqueue" = persist the card as ``pending``
        # ∧ ¬has_pdf; the daemon's ``DownloadQueue.start()`` recovery and the
        # ``reconcile_once`` (D8) sweep both re-route exactly that on-disk state
        # back into the belt. This removes the cross-process GPU contention the
        # old synchronous add caused (D11/D12).
        #
        # On ``force_refresh`` we also reset the attempt budget so the daemon
        # gives the (re-)download a fresh run rather than tripping a stale
        # ``extract_attempts`` ceiling left from a prior pass.
        if force_refresh:
            paper.download_status = DOWNLOAD_STATUS_PENDING
            paper.extract_attempts = 0
            paper.firecrawl_pdf_hunt_exhausted = False
        elif (paper.download_status or "") in DOWNLOAD_STATUS_TERMINAL:
            # Plain (non-force) re-add of an existing TERMINAL card
            # (failed / extract_failed / metadata_only). classify() returns
            # TERMINAL for these, so neither DownloadQueue.start() recovery
            # (pending ∧ ¬has_pdf) nor reconcile_once will pick it up — telling
            # the operator it was "queued" would be a lie. Leave the terminal
            # status untouched (no-clobber) and return an honest response so the
            # operator knows to use ``audit --retry-failed`` or ``--force-refresh``.
            self.library.save()
            return {
                "status": "exists",
                "key": paper.key,
                "metadata": _paper_metadata_dict(paper, self.library),
                "candidates": None,
                "message": (
                    f"already terminal (download_status={paper.download_status!r}); "
                    "not queued. Use `audit --retry-failed` or re-add with "
                    "--force-refresh to retry."
                ),
            }
        elif not self.library.has_pdf(paper.key):
            # Don't clobber a terminal/ok status into pending on a plain re-add;
            # only nudge a blank/empty status onto the pending track so a freshly
            # upserted card is guaranteed queue-visible.
            if not (paper.download_status or "").strip():
                paper.download_status = DOWNLOAD_STATUS_PENDING

        self.library.save()

        return {
            "status": "queued",
            "key": paper.key,
            "metadata": _paper_metadata_dict(paper, self.library),
            "candidates": None,
            "message": "queued for background download + extraction "
                       "(the daemon's download queue will pick it up)",
        }

    def _exists_response(self, paper: Paper) -> dict:
        return {
            "status": "exists",
            "key": paper.key,
            "metadata": _paper_metadata_dict(paper, self.library),
            "candidates": None,
            "message": "already in library",
        }
