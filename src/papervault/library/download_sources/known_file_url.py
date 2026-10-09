from __future__ import annotations

import re
from typing import Callable, Optional
from urllib.parse import parse_qs, unquote, urlsplit

import requests

from ..models import Paper
from ._shared import BROWSER_HEADERS, TIMEOUT, _is_pdf_bytes


def _known_file_url(paper: Paper) -> Optional[str]:
    """Select a stored HTTP(S) file candidate, without resolving landing pages.

    A PDF path or a recognized download endpoint is only eligibility, never
    proof of availability or identity. Do not extract URLs embedded in prose
    (notably Otsuka2020's meeting-program link).
    """
    url = paper.url or ""
    # urlsplit removes some controls, and requests repairs malformed escapes.
    # Reject those strings before either can turn them into a different URL.
    if (not url or "\\" in url or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url)
            or re.search(r"%(?![0-9a-fA-F]{2})", url)):
        return None
    try:
        parts = urlsplit(url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.port == 0):
            return None
        # Apply the HTTP client's host/IDNA validation without any transport.
        # Preparation failures are prerequisites, not attempted downloads.
        requests.PreparedRequest().prepare_url(url, None)
    except (ValueError, requests.RequestException):
        return None
    path = unquote(parts.path)
    if path.lower().endswith(".pdf"):
        return url
    query = parse_qs(parts.query)
    # Digital Commons files need not have a PDF suffix (FIT / TRACE cohort).
    if (path == "/cgi/viewcontent.cgi"
            and len(query.get("article", [])) == 1
            and re.fullmatch(r"[0-9]+", query["article"][0])
            and len(query.get("context", [])) == 1
            and re.fullmatch(r"[A-Za-z0-9_-]+", query["context"][0])):
        return url
    # CiteSeer's stored file links can carry a legacy Java session parameter.
    if (parts.hostname == "citeseerx.ist.psu.edu"
            and re.fullmatch(r"/viewdoc/download(?:;jsessionid=[A-Za-z0-9._-]+)?", path)
            and len(query.get("doi", [])) == 1 and query["doi"][0]
            and query.get("rep") == ["rep1"] and query.get("type") == ["pdf"]):
        return url
    return None


def _try_known_file_url(paper: Paper) -> Optional[bytes]:
    """Fetch a stored file candidate; the cascade still verifies its identity."""
    url = _known_file_url(paper)
    if not url:
        return None
    return _fetch_pdf_url(url)


def _fetch_pdf_url(url: str, *, get: Optional[Callable[..., requests.Response]] = None) -> Optional[bytes]:
    """Fetch a whole PDF candidate; identity remains the cascade's responsibility.

    Repository tiers can supply a transport to enforce their request limits.
    """
    try:
        r = (get or requests.get)(url, timeout=TIMEOUT,
                                  headers=BROWSER_HEADERS, allow_redirects=True)
        # We asked for a whole file, so a partial response is not a download.
        if (200 <= r.status_code < 300 and r.status_code != 206
                and not r.headers.get("Content-Range") and _is_pdf_bytes(r.content)):
            return r.content
    except requests.RequestException:
        pass
    return None
