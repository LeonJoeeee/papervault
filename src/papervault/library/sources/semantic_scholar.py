"""Semantic Scholar API search tool."""

import random
import time
from typing import Any

import requests

from .exceptions import BackendDegraded

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = ("title,authors,year,abstract,citationCount,externalIds,"
          "publicationTypes,venue,publicationVenue,url")


def search_semantic_scholar(query: str, max_results: int = 30) -> list[dict[str, Any]]:
    """Search Semantic Scholar and return structured paper metadata.

    Keyless S2 is aggressively 429-rate-limited; retry on 429 is BOUNDED (≤3
    attempts, jittered backoff). On exhausted retries / a non-429 HTTPError /
    no response we RAISE ``BackendDegraded`` (S2 alone never aborts the call —
    ``_fetch_one_backend`` records it as DEGRADED and returns [] for this pair).
    Returns ``[]`` only for an empty query or a genuinely empty result set.
    """
    if not query:
        return []

    params = {
        "query": query,
        "limit": max_results,
        "fields": FIELDS,
    }

    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(min(2.0, 0.5 * (attempt + 1)) + random.random() * 0.5)
                continue
            resp.raise_for_status()
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.SSLError):
            time.sleep(0.5 * (attempt + 1))
            continue
        except requests.exceptions.HTTPError as e:
            raise BackendDegraded(
                f"S2 HTTP {getattr(e.response, 'status_code', '?')}"
            ) from e
    else:
        raise BackendDegraded("S2 429 exhausted")
    if resp is None:
        raise BackendDegraded("S2 no response")

    data = resp.json()

    papers = []
    for paper in data.get("data", []):
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        ext_ids = paper.get("externalIds") or {}
        doi = ext_ids.get("DOI", "")
        arxiv_id = ext_ids.get("ArXiv", "")
        venue = paper.get("venue") or (paper.get("publicationVenue") or {}).get("name", "")

        papers.append({
            "title": paper.get("title", ""),
            "authors": authors,
            "abstract": paper.get("abstract", "") or "",
            "year": paper.get("year", 0),
            "citation_count": paper.get("citationCount", 0),
            "doi": doi,
            "arxiv_id": arxiv_id,
            "paper_id": paper.get("paperId", ""),
            "venue": venue,
            "publication_types": paper.get("publicationTypes") or [],
            "url": paper.get("url", "") or "",
        })

    return papers
