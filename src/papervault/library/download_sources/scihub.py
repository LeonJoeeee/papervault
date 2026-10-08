from __future__ import annotations

import concurrent.futures
import os
import re
import time
from typing import Optional

import requests

from ..models import Paper
from ._shared import BROWSER_HEADERS, TIMEOUT, USER_AGENT, _is_pdf_bytes


# Bootstrap list of discovery sources; each is fetched, scanned for
# sci-hub.<tld> domains, then liveness-checked. Order doesn't matter
# beyond "first response is fastest".
_SCIHUB_DISCOVERY_SOURCES = (
    "https://sci-hub.41610.org/",   # PyPaperBot's source — explicit mirror index
    "https://en.wikipedia.org/wiki/Sci-Hub",
    "https://lovescihub.wordpress.com/",
    "https://sci-hub.ru/",       # any alive mirror's footer lists siblings
    "https://sci-hub.ee/",
)

# If all discovery sources fail, fall back to these historical mirrors.
_SCIHUB_FALLBACK_DOMAINS = (
    "sci-hub.ru", "sci-hub.ee", "sci-hub.ren",
    "sci-hub.wf", "sci-hub.box", "sci-hub.41610.org",
)

# Cache: (timestamp_set_at, list_of_alive_mirror_urls)
_scihub_mirrors_cache: tuple[float, list[str]] = (0.0, [])
_SCIHUB_CACHE_TTL = 6 * 3600   # 6 hours

# Regex: only match valid TLD-like suffixes 2-8 lowercase chars,
# followed by word boundary. Avoids matching accidental text.
_SCIHUB_DOMAIN_RE = re.compile(r"\bsci-hub\.[a-z]{2,8}(?:\.[a-z]{2,8})?\b")


def _discover_scihub_mirrors() -> list[str]:
    """Discover alive Sci-Hub mirrors from multiple sources, cache for TTL.

    Process:
      1. If cache fresh, return cached list.
      2. Fetch each source URL; regex out sci-hub.<tld> domains.
      3. If discovery yielded nothing, use _SCIHUB_FALLBACK_DOMAINS.
      4. Liveness-check each domain via HEAD request (timeout 5s).
      5. Return alive ones (as full https://<domain> URLs).
      6. Update cache. If liveness yielded nothing, return all discovered
         (last-resort: try them anyway in the actual download).
    """
    global _scihub_mirrors_cache
    now = time.time()
    cached_at, cached = _scihub_mirrors_cache
    if cached and (now - cached_at) < _SCIHUB_CACHE_TTL:
        return cached

    domains: set[str] = set()
    for src in _SCIHUB_DISCOVERY_SOURCES:
        try:
            r = requests.get(src, timeout=10,
                             headers={"User-Agent": USER_AGENT})
            if r.ok:
                for m in _SCIHUB_DOMAIN_RE.finditer(r.text):
                    domains.add(m.group(0))
        except Exception:
            continue

    if not domains:
        domains = set(_SCIHUB_FALLBACK_DOMAINS)

    # Liveness check
    alive: list[str] = []
    for domain in sorted(domains):
        url = f"https://{domain}"
        try:
            r = requests.head(url, timeout=5, allow_redirects=True,
                              headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                alive.append(url)
        except Exception:
            continue

    # If liveness check yielded nothing (e.g., HEAD blocked everywhere),
    # fall back to all discovered domains as URLs — let _try_scihub's
    # GET handle it.
    if not alive:
        alive = [f"https://{d}" for d in sorted(domains)]

    _scihub_mirrors_cache = (now, alive)
    return alive


# PDF URL extraction patterns for sci-hub landing pages, in priority
# order. Sci-Hub mirror HTML structure changes occasionally; this list
# covers every layout we've observed:
#   1. <meta name="citation_pdf_url" content="..."> — Highwire-standard
#      meta, present on all current sci-hub.ru pages. Most reliable.
#   2. <object type="application/pdf" data="..."> — current sci-hub.ru
#      embed (replaced the old <iframe>/<embed> in 2024).
#   3. <iframe|embed src="...pdf..."> — legacy mirrors / older pages.
#   4. location.href = "...pdf..." — JS-redirect pattern on some mirrors.
_SCIHUB_PDF_PATTERNS = (
    re.compile(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', re.I),
    re.compile(r'<object[^>]+type=["\']application/pdf["\'][^>]+data=["\']([^"\']+)["\']', re.I),
    re.compile(r'(?:iframe|embed)[^>]+src=["\']([^"\']+\.pdf[^"\']*)["\']', re.I),
    re.compile(r'location\.href\s*=\s*[\'"]([^\'"]+\.pdf[^\'"]*)[\'"]'),
)


def _scihub_one_mirror(mirror: str, target: str, max_attempts: int = 3) -> Optional[bytes]:
    """Probe ONE scihub mirror; return PDF bytes or None. Extracted from
    _try_scihub so the mirror loop can race in parallel rather than
    iterate serially (40+s under racing-cascade concurrency, well past
    the per-paper deadline).

    Some mihomo exit IPs get stripped/captcha landing pages with no PDF
    link. Page size varies across healthy layouts, so inspect every OK
    response for a link. Retry up to ``max_attempts`` with fresh connections
    (each requests.get opens a new TCP socket → new mihomo round-robin
    exit) when the response fails or lacks the PDF link.
    """
    landing_url = f"{mirror}/{target}"
    for attempt in range(max_attempts):
        try:
            r = requests.get(landing_url, timeout=TIMEOUT,
                             headers=BROWSER_HEADERS, allow_redirects=True)
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            continue
        if not r.ok:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            continue
        pdf_url: Optional[str] = None
        for pat in _SCIHUB_PDF_PATTERNS:
            m = pat.search(r.text)
            if m:
                pdf_url = m.group(1)
                break
        if not pdf_url:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            continue
        pdf_url = pdf_url.split("#", 1)[0]
        if pdf_url.startswith("//"):
            pdf_url = "https:" + pdf_url
        elif pdf_url.startswith("/"):
            pdf_url = mirror + pdf_url
        pdf_headers = dict(BROWSER_HEADERS)
        pdf_headers["Referer"] = landing_url
        pdf_headers["Accept"] = "application/pdf,*/*;q=0.8"
        try:
            pdf_resp = requests.get(pdf_url, timeout=TIMEOUT,
                                    headers=pdf_headers, allow_redirects=True)
            if pdf_resp.ok and _is_pdf_bytes(pdf_resp.content):
                return pdf_resp.content
        except Exception:
            pass
        if attempt < max_attempts - 1:
            time.sleep(2 ** attempt)
    return None


def _try_scihub(paper: Paper) -> Optional[bytes]:
    """Last-resort fetch via Sci-Hub. Off by default; opt in with
    PAPER_PIPELINE_USE_SCIHUB=1. Be aware this may not be legal in your
    jurisdiction.

    Mirrors race in parallel; first valid PDF wins. Serial iteration was
    the dominant bottleneck under racing-cascade concurrency — 8 mirrors
    × per-mirror timeout = 40+s, exceeding the per-paper deadline.
    """
    if os.environ.get("PAPER_PIPELINE_USE_SCIHUB", "").strip() not in {"1", "true", "yes"}:
        return None
    target = paper.doi or paper.arxiv_id or paper.url
    if not target:
        return None
    mirrors = _discover_scihub_mirrors()
    if not mirrors:
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(mirrors)) as ex:
        futures = [ex.submit(_scihub_one_mirror, m, target) for m in mirrors]
        try:
            for fut in concurrent.futures.as_completed(futures, timeout=TIMEOUT * 2):
                data = fut.result()
                if data:
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    return data
        except concurrent.futures.TimeoutError:
            pass
    return None
