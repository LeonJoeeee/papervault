from __future__ import annotations

import os
from typing import Optional

import requests

from ..models import Paper
from ._shared import TIMEOUT, USER_AGENT, _is_pdf_bytes


def _try_core(paper: Paper) -> Optional[bytes]:
    """CORE — UK-based OA aggregator indexing ~280M papers from institutional
    repositories worldwide. Coverage is largely complementary to Unpaywall
    (which uses BASE/PubMed indexing); CORE often surfaces author-uploaded
    postprints and conference proceedings on university servers that other
    aggregators miss. Requires CORE_API_KEY env var.

    Two-pass: DOI-exact first; if that misses or returns no downloadUrl,
    fall back to title search (fuzzy match for top result).
    """
    api_key = os.environ.get("CORE_API_KEY", "").strip()
    if not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT}
    # Note: trailing slash is required — CORE returns 301 to /works/
    api = "https://api.core.ac.uk/v3/search/works/"

    def _query(q: str, limit: int = 3) -> list[dict]:
        try:
            r = requests.get(api, params={"q": q, "limit": limit},
                             headers=headers, timeout=TIMEOUT,
                             allow_redirects=True)
        except Exception:
            return []
        if not r.ok:
            return []
        try:
            return r.json().get("results") or []
        except Exception:
            return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _harvest(work: dict) -> None:
        for field in ("downloadUrl", "fullTextLink"):
            url = work.get(field)
            if url and url not in seen:
                seen.add(url)
                candidates.append(url)
        for link in (work.get("links") or []):
            if isinstance(link, dict):
                u = link.get("url")
                if u and u not in seen:
                    seen.add(u)
                    candidates.append(u)

    # Pass 1: DOI lookup
    if paper.doi:
        for w in _query(f'doi:"{paper.doi}"', 3):
            _harvest(w)

    # Pass 2: title search fallback (CORE indexes plenty without DOI match)
    if not candidates and paper.title and len(paper.title) >= 20:
        # Sanitize title — keep alnum + spaces, drop anything CORE may
        # interpret as syntax
        clean_title = ''.join(c if c.isalnum() or c.isspace() else ' '
                               for c in paper.title)[:120]
        for w in _query(f'title:"{clean_title}"', 5):
            # require title sim > 0.7 to avoid pulling unrelated papers
            cand_title = (w.get("title") or '').lower()
            if not cand_title:
                continue
            our = paper.title.lower()
            from difflib import SequenceMatcher
            if SequenceMatcher(None, cand_title[:120], our[:120], autojunk=False).ratio() < 0.70:
                continue
            _harvest(w)

    for url in candidates:
        try:
            r2 = requests.get(url, timeout=TIMEOUT,
                              headers={"User-Agent": USER_AGENT},
                              allow_redirects=True)
            if r2.ok and _is_pdf_bytes(r2.content):
                return r2.content
        except Exception:
            continue
    return None
