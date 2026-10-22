"""NASA ADS API search tool.

NASA ADS (Harvard-Smithsonian Astrophysics Data System) covers astrophysics,
physics, and astronomy — including pre-arxiv-era literature (e.g. Parker 1958).

Requires an API token in the env var ``ADS_API_TOKEN`` (free, get one at
https://ui.adsabs.harvard.edu/user/settings/token). If unset the function
gracefully returns an empty list.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Any

import requests

from .exceptions import BackendDegraded, backend_degraded, source_errors, source_results
from .identity import source_identity

API_URL = "https://api.adsabs.harvard.edu/v1/search/query"
FIELDS = ("title,author,year,abstract,doi,identifier,bibcode,pub,"
         "citation_count,read_count")
MAX_RETRIES = 5
RETRY_DELAY = 5.0
MAX_AUTHORS = 30

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")
_AUTH_WARNED = False


def _warn_auth_once(detail: str) -> None:
    """Emit a one-time stderr warning for missing/invalid ADS auth."""
    global _AUTH_WARNED
    if _AUTH_WARNED:
        return
    _AUTH_WARNED = True
    print(f"[papervault.library.sources.ads] {detail}", file=sys.stderr)


def _extract_arxiv_id(identifiers: list[str]) -> str:
    """Pull a bare arxiv id (e.g. '2401.0001') from an ADS identifier list."""
    for ident in identifiers or []:
        if not isinstance(ident, str):
            continue
        # Direct prefix forms: "arXiv:2401.0001" or just "2401.0001"
        if ident.lower().startswith("arxiv:"):
            tail = ident.split(":", 1)[1].strip()
            m = _ARXIV_ID_RE.search(tail)
            if m:
                return m.group(1)
        m = _ARXIV_ID_RE.match(ident.strip())
        if m:
            return m.group(1)
    # Fallback: any string that contains a NEW-style arxiv id
    for ident in identifiers or []:
        if not isinstance(ident, str):
            continue
        m = _ARXIV_ID_RE.search(ident)
        if m:
            return m.group(1)
    return ""


@source_errors
def search_ads(query: str, max_results: int = 30) -> list[dict[str, Any]]:
    """Search NASA ADS and return structured paper metadata.

    Returns ``[]`` (without making an HTTP request) if ``ADS_API_TOKEN`` is not
    set, or for a genuine 0-hit query (200 + zero docs). On exhausted retries
    (repeated 429 / connection errors) we RAISE ``BackendDegraded`` — consistent
    with arxiv / semantic_scholar / core, so a transient failure is reported as a
    degraded source, NOT masqueraded as an authoritative 0-hit. On a present-but-
    rejected token (401/403) or a WAF/captcha challenge we also raise
    ``BackendDegraded``.
    """
    if not query:
        return []
    token = os.environ.get("ADS_API_TOKEN")
    if not token:
        return []

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": query,
        "fl": FIELDS,
        "rows": max_results,
        "sort": "relevance",
    }

    failure = BackendDegraded("ADS no response")
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                failure = backend_degraded(response=resp)
                time.sleep(RETRY_DELAY)
                continue
            # A WAF captcha is not evidence of rejected credentials.
            if resp.headers.get("x-amzn-waf-action") == "captcha" or resp.status_code == 405:
                raise BackendDegraded("ADS WAF", code="provider_blocked")
            if resp.status_code in (401, 403):
                _warn_auth_once(
                    f"ADS auth failed ({resp.status_code}); check ADS_API_TOKEN."
                )
                raise BackendDegraded(f"ADS auth {resp.status_code}",
                                      code="authentication_failed", retryability="permanent")
            resp.raise_for_status()
            break
        except requests.ConnectionError as error:
            failure = backend_degraded(error)
            time.sleep(RETRY_DELAY)
        except requests.HTTPError as error:
            raise backend_degraded(error) from error
    else:
        raise failure

    data = resp.json()
    docs = source_results(data, "response", "docs")

    papers: list[dict[str, Any]] = []
    for doc in docs:
        titles = doc.get("title") or []
        title = (titles[0] if titles else "") or ""

        authors_raw = doc.get("author") or []
        authors = [a for a in authors_raw[:MAX_AUTHORS] if isinstance(a, str)]

        year_raw = doc.get("year")
        year: Any = None
        if isinstance(year_raw, int):
            year = year_raw
        elif isinstance(year_raw, str) and year_raw.strip().isdigit():
            year = int(year_raw.strip())

        abstract = (doc.get("abstract") or "") or ""

        doi_list = doc.get("doi") or []
        doi = (doi_list[0] if doi_list else "") or ""

        identifiers = doc.get("identifier") or []
        arxiv_id = _extract_arxiv_id(identifiers)

        bibcode = doc.get("bibcode") or ""
        venue = (doc.get("pub") or "") or ""
        citation_count = int(doc.get("citation_count") or 0)

        url = ""
        if bibcode:
            url = f"https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract"
        elif doi:
            url = f"https://doi.org/{doi}"
        identity = source_identity(
            "ads", source_ids={"ads:bibcode": bibcode, "ads:identifier": identifiers},
            dois=doi_list,
            arxiv_ids=[ident.split(":", 1)[1] for ident in identifiers
                       if isinstance(ident, str) and ident.lower().startswith("arxiv:")]
                      or [arxiv_id],
            urls=url,
        )

        papers.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "paper_id": bibcode,
            "url": url,
            "venue": venue,
            "citation_count": citation_count,
            "source_identities": [identity],
        })

    return papers
