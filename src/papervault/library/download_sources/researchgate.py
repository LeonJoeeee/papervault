from __future__ import annotations

import re
from typing import Optional

from ..models import Paper
from ._shared import _is_pdf_bytes, _load_stealthy_fetcher


# ResearchGate publication-detail page link from search results. The
# canonical URL shape is `/publication/<numeric_id>_<slug>` — we accept
# absolute or root-relative.
_RG_PUBLICATION_RE = re.compile(
    r'href=["\'](?:https?://(?:www\.)?researchgate\.net)?(/publication/\d+[^"\']*)["\']',
    re.IGNORECASE,
)
# PDF link inside a publication detail page. RG uses several patterns;
# we look for any `.pdf` inside an href, especially `full-text.pdf`,
# `publicationDownloadFile`, or `documentDownload`.
_RG_PDF_LINK_RE = re.compile(
    r'href=["\']([^"\']+(?:\.pdf|publicationDownloadFile[^"\']*|'
    r'documentDownload[^"\']*))["\']',
    re.IGNORECASE,
)


def _try_researchgate(paper: Paper) -> Optional[bytes]:
    """ResearchGate via Scrapling + real-browser click simulation.

    RG's anti-bot is multi-layered: Cloudflare interstitial in front,
    login wall behind, and a click-event check on the actual Download
    button. Plain ``requests`` (the historical implementation here) was
    blocked at layer one; even cookie replay was blocked at layer three.
    The only working path so far is:

      1. Find the RG publication URL — search-engine result (Google
         Scholar via Scrapling) usually exposes it as
         ``/profile/<author>/publication/<id>_<slug>``. This is more
         reliable than RG's own /search/ endpoint, which often hides
         recent uploads behind a login wall.
      2. Drive Scrapling's StealthyFetcher (Playwright + Cloudflare
         Turnstile solver) to the publication's landing page.
      3. Locate the ``[data-testid="research-header-cta-download-fulltext"]``
         button and trigger ``page.expect_download() / page.click()`` —
         a real browser click event, which is what RG validates. RG
         then issues the actual PDF binary in response.

    Slow (~30-60 s per call: ~20 s Cloudflare solve + ~10 s page render
    + ~5 s download). Costs Playwright launch + ~300 MB browser. Position
    as a near-last tier in the cascade. Returns None on any failure;
    safe under repeat invocations.

    All Scrapling imports are lazy so the daemon doesn't pull patchright
    + Playwright until this tier actually runs.
    """
    title = (paper.title or "").strip()
    doi = (paper.doi or "").strip()
    if len(title) < 20 and not doi:
        return None

    StealthyFetcher = _load_stealthy_fetcher("researchgate")
    if StealthyFetcher is None:
        return None

    # Stage 1: find an RG publication URL via Google Scholar.
    # Scholar surfaces the canonical /profile/<author>/publication/<id>/...
    # form that RG itself sometimes hides behind a login wall.
    if doi:
        scholar_q = f"{title[:120]} {doi}"
    else:
        scholar_q = title[:200]
    from urllib.parse import quote_plus
    scholar_url = (f"https://scholar.google.com/scholar?q="
                    f"{quote_plus(scholar_q)}")
    try:
        page = StealthyFetcher.fetch(scholar_url, headless=True,
                                       solve_cloudflare=True, wait=2500)
    except Exception:
        return None
    if not page or getattr(page, "status", 0) != 200:
        return None

    m = re.search(
        r"https?://(?:www\.)?researchgate\.net/(?:profile/[A-Za-z0-9-]+/)?"
        r"publication/\d+[A-Za-z0-9_%/-]+",
        page.html_content)
    if not m:
        return None
    landing_url = m.group(0).rstrip(")&\"'")
    # If we landed on a deep PDF URL (.../links/<hex>/<slug>.pdf), strip
    # back to the publication root — that's where the Download button is.
    landing_url = re.sub(r"/links?/.*$", "", landing_url)
    landing_url = re.sub(r"/citation.*$", "", landing_url)

    # Stage 2 + 3: render landing page in Scrapling, then click Download.
    result_holder: dict = {"bytes": None}

    def click_download(page):
        try:
            with page.expect_download(timeout=30000) as dl_info:
                page.locator(
                    '[data-testid="research-header-cta-download-fulltext"]'
                ).first.click()
            download = dl_info.value
            # Read the downloaded bytes into memory (Playwright keeps a
            # temp file; .path() returns a Path object).
            path = download.path()
            if path is not None:
                with open(path, "rb") as f:
                    result_holder["bytes"] = f.read()
        except Exception:
            # No button / no download / timeout / login-wall → just give up.
            pass
        return page

    try:
        StealthyFetcher.fetch(landing_url, headless=True,
                                solve_cloudflare=True, wait=4000,
                                page_action=click_download)
    except Exception:
        return None

    data = result_holder["bytes"]
    if data and _is_pdf_bytes(data):
        return data
    return None
