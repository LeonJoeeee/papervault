from __future__ import annotations

import re
from typing import Optional

import requests

from ..models import Paper
from ._shared import BROWSER_HEADERS, TIMEOUT, USER_AGENT, _is_pdf_bytes


# Highwire Press <meta name="citation_pdf_url"> is a near-universal
# convention among scholarly publishers. We match both single- and
# double-quoted variants and don't care about attribute order.
_CITATION_PDF_URL_RE = re.compile(
    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
# Some publishers put `content` before `name`. Match that ordering too.
_CITATION_PDF_URL_RE_REV = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
    re.IGNORECASE,
)


def _try_citation_pdf_url(paper: Paper) -> Optional[bytes]:
    """Generic publisher meta-tag PDF discovery.

    Most academic publishers (MDPI, Springer, Wiley, RSC, ACS, IEEE, AIP,
    APS, IOP, etc.) embed a Highwire Press style <meta name="citation_pdf_url"
    content="..."> tag on their article landing pages. Hitting the DOI
    redirect lands you on the publisher page; we then parse for the tag
    and follow it.

    Two-UA strategy: many publishers (MDPI, Springer) want browser-like
    headers and 403 bare UAs; others (IOPscience for AAS / IOP journals)
    do the opposite — bare UA returns the actual PDF, browser-like
    requests get redirected to a perfdrive bot-challenge. So we try BOTH
    UA strategies for both the landing page fetch and the final PDF
    fetch, taking the first that yields a valid PDF.
    """
    if not paper.doi:
        return None

    landing_url = f"https://doi.org/{paper.doi}"
    # Two-UA landing fetch: take whichever mode yields a citation_pdf_url
    # meta tag (or a direct PDF). Browser-headers can succeed for MDPI but
    # gets redirected to a bot-challenge for IOPscience; bare UA inverts
    # that. We must verify the meta tag is actually present, not just
    # that the response was 200 — perfdrive's bot challenge ALSO returns
    # 200 (with HTML but without the meta tag).
    landing = None
    for ua_kind, ua_headers in (("bare", {"User-Agent": USER_AGENT}),
                                 ("browser", BROWSER_HEADERS)):
        try:
            r = requests.get(landing_url, timeout=TIMEOUT,
                              headers=ua_headers, allow_redirects=True)
        except Exception:
            continue
        if not r.ok:
            continue
        if _is_pdf_bytes(r.content):
            return r.content
        if (_CITATION_PDF_URL_RE.search(r.text)
                or _CITATION_PDF_URL_RE_REV.search(r.text)):
            landing = r
            break
    if landing is None:
        return None

    html = landing.text
    m = _CITATION_PDF_URL_RE.search(html) or _CITATION_PDF_URL_RE_REV.search(html)
    if not m:
        return None
    pdf_url = m.group(1).strip()
    if pdf_url.startswith("//"):
        pdf_url = "https:" + pdf_url
    elif pdf_url.startswith("/"):
        from urllib.parse import urlparse
        base = urlparse(landing.url)
        pdf_url = f"{base.scheme}://{base.netloc}{pdf_url}"

    # Try BOTH UA modes for the PDF fetch. IOP is the canonical example
    # of "bare-UA wins": its `/article/{doi}/pdf` endpoint returns
    # application/pdf to USER_AGENT but redirects browser-like requests
    # to a perfdrive bot-challenge HTML page.
    for ua_kind, base_headers in (("bare", {"User-Agent": USER_AGENT}),
                                   ("browser", BROWSER_HEADERS)):
        headers = dict(base_headers)
        headers["Referer"] = landing.url
        headers["Accept"] = "application/pdf,*/*;q=0.8"
        try:
            r = requests.get(pdf_url, timeout=TIMEOUT,
                              headers=headers, allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None


def _try_curl_impersonate(paper: Paper) -> Optional[bytes]:
    """Variant of `_try_citation_pdf_url` using curl-cffi to impersonate
    a real browser's TLS fingerprint. Defeats Akamai EdgeSuite and similar
    CDN gates that classify our `requests`-library traffic as bot traffic
    and 403 us — even when the request looks browser-like at the HTTP
    layer (User-Agent, Accept-*, Sec-Fetch-*).

    Validated empirically: `requests` 403s on
    https://www.mdpi.com/2504-2289/6/4/140/pdf, but curl-cffi with
    chrome120 impersonation returns the actual 2.9MB PDF. Same code path,
    same headers — only the TLS fingerprint differs.

    Doesn't help against publisher walls that need JavaScript execution
    (Wiley/onlinelibrary, ACS, ASME) — those still 403 because the wall
    is a JS challenge not a TLS-fingerprint check. For those we'd need
    a real headless browser (Playwright).
    """
    if not paper.doi:
        return None
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return None
    try:
        landing = curl_requests.get(
            f"https://doi.org/{paper.doi}",
            impersonate="chrome120",
            timeout=TIMEOUT,
            allow_redirects=True,
        )
    except Exception:
        return None
    if not landing.ok:
        return None
    if _is_pdf_bytes(landing.content):
        return landing.content
    html = landing.text
    m = _CITATION_PDF_URL_RE.search(html) or _CITATION_PDF_URL_RE_REV.search(html)
    if not m:
        return None
    pdf_url = m.group(1).strip()
    if pdf_url.startswith("//"):
        pdf_url = "https:" + pdf_url
    elif pdf_url.startswith("/"):
        from urllib.parse import urlparse
        base = urlparse(landing.url)
        pdf_url = f"{base.scheme}://{base.netloc}{pdf_url}"
    try:
        r = curl_requests.get(
            pdf_url, impersonate="chrome120", timeout=TIMEOUT,
            headers={"Referer": landing.url,
                     "Accept": "application/pdf,*/*;q=0.8"},
            allow_redirects=True,
        )
        if r.ok and _is_pdf_bytes(r.content):
            return r.content
    except Exception:
        return None
    return None
