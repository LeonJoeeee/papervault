"""Inspire-HEP API search tool.

Inspire-HEP (CERN-hosted) covers high-energy physics, particle physics,
cosmic ray physics, and adjacent astrophysics. No auth, generous rate limits.
"""

from __future__ import annotations

import time
from typing import Any

import requests

API_URL = "https://inspirehep.net/api/literature"
FIELDS = "titles,authors,publication_info,arxiv_eprints,dois,abstracts,citation_count"
MAX_RETRIES = 5
RETRY_DELAY = 5.0


def search_inspire(query: str, max_results: int = 30) -> list[dict[str, Any]]:
    """Search Inspire-HEP and return structured paper metadata."""
    if not query:
        return []

    # Bare phrase-AND query: pass the raw term string. INSPIRE's default
    # operator AND-joins the tokens of a bare (unquoted, unfielded) phrase, so
    # ``q="magnetic reconnection"`` matches records containing BOTH tokens. We
    # deliberately do NOT quote (would force an exact-phrase match, far too
    # narrow for thin geospace coverage) nor wrap in a field qualifier.
    params = {
        "q": query,
        "size": max_results,
        "fields": FIELDS,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(RETRY_DELAY)
                continue
            resp.raise_for_status()
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.SSLError):
            time.sleep(RETRY_DELAY)
            continue
        except requests.exceptions.HTTPError:
            return []
    else:
        return []

    try:
        data = resp.json()
    except ValueError:
        return []

    papers: list[dict[str, Any]] = []
    for hit in (data.get("hits") or {}).get("hits", []) or []:
        meta = hit.get("metadata") or {}

        titles = meta.get("titles") or []
        title = (titles[0].get("title") if titles else "") or ""

        authors = []
        for a in (meta.get("authors") or []):
            name = (a.get("full_name") or "").strip()
            if name:
                authors.append(name)

        pub_info = meta.get("publication_info") or []
        year: Any = None
        venue = ""
        if pub_info:
            first = pub_info[0] or {}
            year = first.get("year")
            venue = (first.get("journal_title") or "") or ""
        if not year:
            preprint_date = meta.get("preprint_date") or ""
            if isinstance(preprint_date, str) and len(preprint_date) >= 4:
                prefix = preprint_date[:4]
                if prefix.isdigit():
                    year = int(prefix)

        arxiv_eprints = meta.get("arxiv_eprints") or []
        arxiv_id = ""
        if arxiv_eprints:
            arxiv_id = (arxiv_eprints[0].get("value") or "") or ""

        dois = meta.get("dois") or []
        doi = ""
        if dois:
            doi = (dois[0].get("value") or "") or ""

        abstracts = meta.get("abstracts") or []
        abstract = ""
        if abstracts:
            abstract = (abstracts[0].get("value") or "") or ""

        citation_count = int(meta.get("citation_count") or 0)

        url = ""
        if arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        elif doi:
            url = f"https://doi.org/{doi}"

        papers.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "url": url,
            "venue": venue,
            "citation_count": citation_count,
        })

    return papers
