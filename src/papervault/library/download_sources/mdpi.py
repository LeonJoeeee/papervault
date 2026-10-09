from __future__ import annotations

import re
from typing import Optional

from ..models import Paper
from ._shared import _is_pdf_bytes, _load_stealthy_fetcher


def _try_mdpi_scrapling(paper: Paper) -> Optional[bytes]:
    """MDPI (10.3390/...) papers are CC-BY OA but the canonical URLs are
    fronted by Akamai Bot Manager — direct ``requests`` calls return 403,
    and even Scrapling's solve_cloudflare alone gets a JS-challenge stub
    instead of the PDF. The working pattern (verified live for
    magnetochemistry9040091, magnetochemistry9040096, universe11060174):

      1. ``StealthyFetcher.fetch`` on ``https://doi.org/{doi}`` solves the
         landing-page challenge and lands on ``mdpi.com/{issn}/{vol}/{issue}/{art}``.
      2. Inside ``page_action``: parse ``citation_pdf_url`` meta.
      3. ``page.evaluate('window.location.href = pdf_url')`` triggers the
         Akamai ``bm-verify`` meta-refresh challenge. The headless browser's
         JS engine executes the challenge naturally.
      4. ``page.expect_download()`` captures the resulting PDF stream.

    Slow (~15-25s) so kept near the end of the cascade. No-ops for
    non-MDPI DOIs.
    """
    doi = paper.doi or ""
    if not doi.startswith("10.3390/"):
        return None
    StealthyFetcher = _load_stealthy_fetcher("mdpi_scrapling")
    if StealthyFetcher is None:
        return None
    import tempfile, os as _os, os.path as _osp
    save_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    state = {"pdf_path": None}

    def _action(page):
        html = page.content()
        m = re.search(
            r'name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)',
            html, re.I)
        if not m:
            return page
        pdf_url = m.group(1)
        try:
            with page.expect_download(timeout=45000) as dl_info:
                page.evaluate(f'window.location.href = "{pdf_url}"')
            dl = dl_info.value
            dl.save_as(save_path)
            state["pdf_path"] = save_path
        except Exception:
            pass
        return page

    try:
        StealthyFetcher.fetch(
            f"https://doi.org/{doi}",
            solve_cloudflare=True, network_idle=True, timeout=60000,
            humanize=False, geoip=False, page_action=_action)
    except Exception:
        return None
    if state["pdf_path"] and _osp.exists(state["pdf_path"]):
        try:
            with open(state["pdf_path"], "rb") as f:
                data = f.read()
            _os.unlink(state["pdf_path"])
            if _is_pdf_bytes(data):
                return data
        except Exception:
            return None
    return None
