from __future__ import annotations

import re
from typing import Optional

import requests

from ..models import Paper
from ._shared import TIMEOUT, USER_AGENT, _is_pdf_bytes, log


def _try_arxiv(paper: Paper) -> Optional[bytes]:
    if not paper.arxiv_id:
        return None
    arxiv_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        r = requests.get(url, timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT}, allow_redirects=True)
    except requests.RequestException:
        # Transient network error (timeout / connection reset): the id may be
        # perfectly real — we just couldn't reach arxiv. Never null on this.
        return None
    if r.ok and _is_pdf_bytes(r.content):
        return r.content
    # A definitive 404 means arxiv has no such paper: the arxiv_id is a
    # fabricated / mistyped identifier (issue #62 — the ingest gate emits
    # format-valid-but-nonexistent arxiv ids for ~84% of arxiv-bearing
    # metadata_only papers, e.g. "8977.2023", "2025.35859"). Null it so the
    # fake id stops misleading cite-check and future resolution; keep the DOI,
    # which is reliable. Only 404 is definitive — a 403/5xx/non-pdf-200 can be
    # transient throttling or a withdrawn-but-real record, so leave those be.
    # The in-place clear is persisted by the caller's library.save() after
    # download_paper (same write-back contract as _try_arxiv_by_title).
    if r.status_code == 404:
        log.warning("arxiv[%s]: 404 — nulling fabricated arxiv_id=%r (keeping doi=%r)",
                    paper.key, paper.arxiv_id, paper.doi)
        paper.arxiv_id = ""
    return None


def _try_arxiv_by_title(paper: Paper) -> Optional[bytes]:
    """When the paper has no arxiv_id but has a title, search arxiv by
    title to discover an arxiv preprint version. Many papers in
    paywalled journals have arxiv preprints whose ID didn't get captured
    in the original metadata fetch.

    On a confident match, persists the discovered arxiv_id back to the
    paper so future cascade attempts skip the search.
    """
    if paper.arxiv_id:
        return None  # already had arxiv_id; _try_arxiv would have used it
    title = (paper.title or "").strip()
    if len(title) < 20:  # too short to disambiguate reliably
        return None

    from ..sources.arxiv import search_arxiv
    try:
        # search_arxiv internally prefixes the query with `all:`, which
        # cross-field-searches title + abstract + comments. Don't add an
        # extra `ti:` prefix — `all:ti:"..."` is invalid syntax and
        # arxiv returns 0 hits for it. Just pass the title text; the
        # title-overlap match below filters non-title matches.
        # Strip troublesome punctuation that could confuse arxiv parser.
        clean = re.sub(r'["\\?<>]', '', title)
        # Cap query length — arxiv's URL-encoded query can hit length
        # limits with very long titles. 100 chars is plenty for ranking.
        clean = clean[:100]
        results = search_arxiv(clean, max_results=5)
    except Exception:
        return None

    if not results:
        return None

    # Find best match by title overlap
    paper_title_norm = re.sub(r'[^a-z0-9]+', ' ', title.lower()).strip()
    best = None
    for r in results:
        cand_title = (r.get('title') or '').strip()
        cand_norm = re.sub(r'[^a-z0-9]+', ' ', cand_title.lower()).strip()
        # Strict-enough overlap:
        # - normalized titles must share their first 30+ chars exactly, OR
        # - one is a prefix/suffix of the other after normalization
        first_chars = min(50, len(paper_title_norm), len(cand_norm))
        if first_chars >= 30 and paper_title_norm[:first_chars] == cand_norm[:first_chars]:
            best = r
            break
        if (paper_title_norm in cand_norm) or (cand_norm in paper_title_norm):
            if abs(len(cand_norm) - len(paper_title_norm)) < 30:
                best = r
                break

    if not best:
        return None

    discovered_arxiv_id = best.get('arxiv_id', '')
    if not discovered_arxiv_id:
        return None

    # Try downloading via arxiv directly. Do NOT persist the discovered arxiv_id
    # up-front: a title-match can be loose, and the bytes still face
    # _verify_pdf_matches_metadata in the caller. Persisting before any outcome
    # poisoned future sweeps even on a plain download failure (verified drill
    # finding). Persist ONLY once we actually have a PDF in hand.
    arxiv_id_clean = re.sub(r'v\d+$', '', discovered_arxiv_id.strip())
    url = f"https://arxiv.org/pdf/{arxiv_id_clean}"
    try:
        r = requests.get(url, timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT},
                         allow_redirects=True)
        if r.ok and _is_pdf_bytes(r.content):
            paper.arxiv_id = discovered_arxiv_id
            return r.content
    except Exception:
        return None
    return None
