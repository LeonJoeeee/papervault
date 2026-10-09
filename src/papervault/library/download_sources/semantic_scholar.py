from __future__ import annotations

from typing import Optional

import requests

from ..models import Paper
from ._shared import BROWSER_HEADERS, TIMEOUT, USER_AGENT, _is_pdf_bytes


def _try_semantic_scholar_oa(paper: Paper) -> Optional[bytes]:
    """Try Semantic Scholar's ``openAccessPdf`` field.

    S2 maintains its own OA crawl (separate from Unpaywall / OpenAlex)
    and surfaces ``openAccessPdf.url`` for papers it has crawled. Often
    catches BRONZE / HYBRID OA papers that Unpaywall missed.
    """
    if not paper.doi:
        return None
    api = (
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{paper.doi}"
        "?fields=openAccessPdf"
    )
    try:
        r = requests.get(api, timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
        if not r.ok:
            return None
        oa = (r.json().get("openAccessPdf") or {})
        url = oa.get("url")
        if not url:
            return None
        # Try with browser headers — S2 OA URLs are usually publisher
        # pages and need a browser-like UA to avoid 403.
        pdf = requests.get(url, timeout=TIMEOUT,
                           headers=BROWSER_HEADERS, allow_redirects=True)
        if pdf.ok and _is_pdf_bytes(pdf.content):
            return pdf.content
    except Exception:
        return None
    return None
