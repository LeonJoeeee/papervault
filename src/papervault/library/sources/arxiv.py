"""arXiv API search tool."""

import random
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

from ..polite_http import polite_get

from .exceptions import BackendDegraded, backend_degraded, source_errors
from .identity import source_identity

ARXIV_API_URL = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# arXiv allows ~1 request per 3 seconds per IP. With multiple concurrent callers
# we get 429d aggressively. The OUTER ``_pace`` limiter (search.py, ≥3s, OUTSIDE
# the fetch timeout) is the primary protection; this in-loop retry is a thin
# BOUNDED backoff (≤3 attempts, jittered, total wall ≲12s) so the ``to_thread``'d
# call cannot outlive the 30s ``wait_for`` cancel by the old 75s (the prior
# _MAX_RETRIES=5 × _BACKOFF_BASE*(attempt+1) sleep chain). On give-up we RAISE
# ``BackendDegraded`` (a transient empty/429 must NOT masquerade as an
# authoritative 0-hit) — distinct from a genuine empty feed (returns []).
_MAX_RETRIES = 3
_BACKOFF_CAP = 4.0   # per-attempt backoff ceiling; 3 attempts → total wall ≲12s


@source_errors
def search_arxiv(query: str, max_results: int = 30, *,
                 sort_by_recency: bool = False) -> list[dict[str, Any]]:
    """Search arXiv and return structured paper metadata.

    Sort defaults to ``relevance``; ``submittedDate`` descending is used ONLY
    when ``sort_by_recency`` is True (threaded from ``ranking_hint=="by_recency"``).
    arXiv exposes no citation count, so it is NEVER ranked on importance.
    ``id:`` queries pass through; other queries search all fields.

    Bounded retry on HTTP 429 / 5xx / connection errors with jittered backoff
    (≤3 attempts). RAISES ``BackendDegraded`` after all retries fail (a transient
    empty/429 is NOT an authoritative 0-hit). Entries missing a title or
    publication date are skipped; no remaining entries returns ``[]``.
    """
    if not query:
        return []

    params = {
        "search_query": query if query.startswith("id:") else f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate" if sort_by_recency else "relevance",
        "sortOrder": "descending",
    }

    failure = BackendDegraded("arXiv no response")
    for attempt in range(_MAX_RETRIES):
        try:
            resp = polite_get(requests.get, ARXIV_API_URL, params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as error:
            failure = backend_degraded(error)
            time.sleep(min(_BACKOFF_CAP, 1.0 * (attempt + 1)) + random.random() * 0.5)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            failure = backend_degraded(response=resp)
            time.sleep(min(_BACKOFF_CAP, 1.0 * (attempt + 1)) + random.random() * 0.5)
            continue
        if not resp.ok:
            raise backend_degraded(response=resp)
        break
    else:
        raise failure

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as error:
        raise backend_degraded(ValueError("invalid XML")) from error
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise backend_degraded(ValueError("invalid Atom feed"))
    papers = []

    for entry in root.findall("atom:entry", NS):
        title_elem = entry.find("atom:title", NS)
        published_elem = entry.find("atom:published", NS)
        if title_elem is None or published_elem is None:
            continue
        title = title_elem.text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
        published = published_elem.text
        year = published[:4]
        url = entry.find("atom:id", NS).text.strip()
        arxiv_id = url.split("/abs/")[-1]
        doi = (entry.findtext("arxiv:doi", "", NS) or "").strip()
        identity = source_identity(
            "arxiv", source_ids={"arxiv:paper": arxiv_id}, dois=doi, arxiv_ids=arxiv_id,
            urls=[url, *(link.get("href") for link in entry.findall("atom:link", NS))],
        )

        authors = []
        for author_elem in entry.findall("atom:author", NS):
            name = author_elem.find("atom:name", NS).text.strip()
            authors.append(name)

        papers.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year,
            "url": url,
            "arxiv_id": arxiv_id,
            "doi": doi,
            "source_identities": [identity],
        })

    return papers
