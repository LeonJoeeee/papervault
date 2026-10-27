"""Semantic Scholar API search tool."""

import random
import time
from typing import Any

import requests

from papervault.config import polite_v2_enabled
from ..fetch import normalize_doi
from ..polite_http import PolitenessDenied, polite_get

from .exceptions import BackendDegraded, backend_degraded, source_errors, source_results
from .identity import semantic_scholar_identity

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = ("title,authors,year,abstract,citationCount,externalIds,"
          "publicationTypes,venue,publicationVenue,url")


@source_errors
def search_semantic_scholar(query: str, max_results: int = 30) -> list[dict[str, Any]]:
    """Search Semantic Scholar and return structured paper metadata.

    Keyless S2 is aggressively 429-rate-limited; retry on 429 is BOUNDED (≤3
    attempts, jittered backoff) with polite-v2 off. With polite-v2 on, a 429
    returns immediately so its cooldown remains owned by the ledger.
    On exhausted retries / a non-429 HTTPError /
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

    failure = BackendDegraded("S2 no response")
    for attempt in range(3):
        try:
            resp = polite_get(requests.get, API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                failure = backend_degraded(response=resp)
                if polite_v2_enabled():
                    raise failure
                time.sleep(min(2.0, 0.5 * (attempt + 1)) + random.random() * 0.5)
                continue
            resp.raise_for_status()
            break
        except PolitenessDenied as error:
            raise (failure if attempt else backend_degraded(error)) from error
        except requests.ConnectionError as error:
            failure = backend_degraded(error)
            time.sleep(0.5 * (attempt + 1))
        except requests.HTTPError as error:
            raise backend_degraded(error) from error
    else:
        raise failure

    data = resp.json()
    records = source_results(data, "data")

    papers = []
    for paper in records:
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        ext_ids = paper.get("externalIds") or {}
        doi = normalize_doi(ext_ids.get("DOI") or "")
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
            "source_identities": [semantic_scholar_identity(paper)],
        })

    return papers
