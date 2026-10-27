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
import re
import time
from typing import Any

import requests

from .. import fetch
from .exceptions import BackendDegraded, backend_degraded, source_errors, source_results
from .identity import identity_values, source_identity

API_URL = "https://api.core.ac.uk/v3/search/works/"
TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 5.0


@source_errors
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
    failure = BackendDegraded("CORE no response")
    for _attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, headers=headers,
                                timeout=TIMEOUT, allow_redirects=True)
            if resp.status_code == 429:
                failure = backend_degraded(response=resp)
                time.sleep(RETRY_DELAY)
                continue
            if not resp.ok:
                raise backend_degraded(response=resp)
            break
        except (requests.ConnectionError, requests.Timeout) as error:
            failure = backend_degraded(error)
            time.sleep(RETRY_DELAY)
    else:
        raise failure

    data = resp.json()
    works = source_results(data, "results")

    out: list[dict[str, Any]] = []
    for work in works:
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
        doi = fetch.normalize_doi(work.get("doi") or "")
        arxiv_id = fetch.extract_valid_arxiv_id(work.get("arxivId") or "")
        arxivs = [work["arxivId"]] if arxiv_id else []

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
        if not arxiv_id:
            arxiv_id = fetch.extract_arxiv_id_from_doi_or_url(doi, url)
            if arxiv_id:
                arxivs.append(arxiv_id)

        outputs = identity_values(work.get("outputs"))
        # Work responses also carry output API URLs. Label the numeric output
        # IDs separately from the work ID, while preserving the supplied URLs.
        output_ids = [re.sub(r"^https://api\.core\.ac\.uk/v3/outputs/([0-9]+)/?$", r"\1", v)
                      for v in outputs]
        ids = {"core:work": work.get("id"), "core:output": output_ids}
        for field, namespace in (("magId", "core:mag"), ("pubmedId", "core:pubmed")):
            if field in work:
                ids[namespace] = work[field]
        dois = [doi]
        identifiers = work.get("identifiers") or []
        if isinstance(identifiers, dict):
            identifiers = [{"type": kind, "identifier": value}
                           for kind, value in identifiers.items()]
        for entry in identifiers:
            if not isinstance(entry, dict):
                ids.setdefault("core:identifier", []).extend(identity_values(entry))
                arxivs.extend(v for v in identity_values(entry)
                              if v.lower().startswith(("arxiv:", "oai:arxiv.org:")))
                continue
            kind, value = entry.get("type") or "unknown", entry.get("identifier")
            ids.setdefault(f"core:identifier:{kind}", []).extend(identity_values(value))
            if str(kind).lower() == "doi":
                dois.extend(identity_values(value))
            elif str(kind).lower() in {"arxiv", "arxiv_id"}:
                arxivs.extend(identity_values(value))
            elif str(kind).lower() in {"oai", "oai_id"}:
                arxivs.extend(v for v in identity_values(value)
                              if v.lower().startswith("oai:arxiv.org:"))
        oai = identity_values([work.get("oai"), *(work.get("oaiIds") or [])])
        if oai:
            ids["core:oai"] = oai
            arxivs.extend(v for v in oai if v.lower().startswith("oai:arxiv.org:"))
        identity = source_identity(
            "core", source_ids=ids, dois=dois, arxiv_ids=arxivs,
            urls=[work.get("downloadUrl"), *(work.get("sourceFulltextUrls") or []),
                  *(v for v in outputs if v.startswith("https://")),
                  *(link.get("url") for link in work.get("links") or []
                    if isinstance(link, dict))],
        )

        out.append({
            "title": (work.get("title") or "").strip(),
            "authors": authors,
            "abstract": (work.get("abstract") or "").strip(),
            "year": year,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "paper_id": str(work.get("id") or ""),  # CORE work id, never an output id
            "venue": venue,
            "publication_types": [],  # CORE doesn't expose this field
            "citation_count": int(work.get("citationCount") or 0),
            "url": url,
            "source_identities": [identity],
        })

    return out
