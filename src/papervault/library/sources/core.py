"""CORE search API wrapper.

CORE (https://core.ac.uk) is a UK-based aggregator of ~280M open-access
papers from institutional repositories. Coverage is broad and largely
complementary to Semantic Scholar / OpenAlex — CORE often surfaces
postprints + conference proceedings that other aggregators miss.

Used by ``search_all`` as one of several keyword-search backends. The
download tier (``download.py:_try_core``) hits the same API for PDF
URLs separately; this module is metadata-only for the search blend.

Requires ``CORE_API_KEY`` env var. Without it, returns ``[]`` silently
so the upstream blend can no-op past this backend.
"""

import os
import time
from typing import Any

import requests

from .exceptions import BackendDegraded

API_URL = "https://api.core.ac.uk/v3/search/works/"
TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 5.0


def search_core(query: str, max_results: int = 30) -> list[dict[str, Any]]:
    """Search CORE and return paper metadata in the shared search-result shape.

    Empty query OR missing ``CORE_API_KEY`` → ``[]`` (silent no-op).
    """
    if not query:
        return []
    api_key = os.environ.get("CORE_API_KEY", "").strip()
    if not api_key:
        return []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "paper-library/0.1 (mailto:dev@example.invalid)",
    }
    params = {"q": query, "limit": max_results}

    # NEVER date-sort CORE (publishedDate is corrupt — years like 2967/2566 —
    # and citationCount is uniformly 0). No sort param ⇒ relevance default.
    # On a non-ok response or retry-exhaustion we RAISE ``BackendDegraded`` (a
    # transient failure must NOT masquerade as an authoritative 0-hit). A genuine
    # empty result set is an EMPTY ``data["results"]`` → returns [].
    resp = None
    for _attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, headers=headers,
                                timeout=TIMEOUT, allow_redirects=True)
            # CORE returns 429 when over rate limit
            if resp.status_code == 429:
                time.sleep(RETRY_DELAY)
                continue
            if not resp.ok:
                raise BackendDegraded(f"CORE HTTP {resp.status_code}")
            break
        except (requests.exceptions.ConnectionError,
                requests.exceptions.SSLError,
                requests.exceptions.Timeout):
            time.sleep(RETRY_DELAY)
            continue
    if resp is None or not resp.ok:
        raise BackendDegraded("CORE retry exhausted")

    try:
        data = resp.json()
    except Exception:
        raise BackendDegraded("CORE non-JSON response")

    out: list[dict[str, Any]] = []
    for work in data.get("results") or []:
        # CORE authors: list of {"name": "Last, F."} dicts
        authors_raw = work.get("authors") or []
        authors = [a.get("name", "") for a in authors_raw if isinstance(a, dict)]

        # Year: CORE uses yearPublished (int)
        year_raw = work.get("yearPublished")
        try:
            year = int(year_raw) if year_raw else None
        except (TypeError, ValueError):
            year = None

        # Identifiers: doi top-level; arxivId top-level (rarely set)
        doi = (work.get("doi") or "").strip()
        arxiv_id = (work.get("arxivId") or "").strip()

        # Venue: CORE uses "publisher" (publisher house) but the journal
        # name lives under "journals[0].title" when available
        journals = work.get("journals") or []
        venue = ""
        if isinstance(journals, list) and journals:
            j0 = journals[0] if isinstance(journals[0], dict) else {}
            venue = j0.get("title") or ""
        if not venue:
            venue = work.get("publisher") or ""

        # URL: prefer downloadUrl (direct PDF), then sourceFulltextUrls
        url = work.get("downloadUrl") or ""
        if not url:
            sf = work.get("sourceFulltextUrls") or []
            if isinstance(sf, list) and sf:
                url = sf[0] if isinstance(sf[0], str) else ""

        out.append({
            "title": (work.get("title") or "").strip(),
            "authors": authors,
            "abstract": (work.get("abstract") or "").strip(),
            "year": year,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "paper_id": str(work.get("id") or ""),  # CORE internal id
            "venue": venue,
            "publication_types": [],  # CORE doesn't expose this field
            "citation_count": int(work.get("citationCount") or 0),
            "url": url,
        })

    return out
