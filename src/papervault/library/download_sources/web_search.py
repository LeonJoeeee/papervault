from __future__ import annotations

import os
import re
from typing import Optional

import requests

from ..models import Paper
from ._shared import BROWSER_HEADERS, TIMEOUT, USER_AGENT, _is_pdf_bytes


def _try_web_search(paper: Paper) -> Optional[bytes]:
    """Last-ditch generic search: DuckDuckGo HTML for ``"<title>" filetype:pdf``.

    Restricted by ``PAPER_LIBRARY_SEARCH_BLOCKLIST`` (comma-sep host
    suffixes; appended as ``-site:<host>`` to the query). Catches OA
    copies hosted on places no aggregator indexes — journal mirrors,
    institutional repositories, conference sites — that S2 / Unpaywall /
    OpenAlex / CORE all missed.

    Verifies the candidate is a real PDF via ``_is_pdf_bytes``. Title-
    correctness verification happens downstream in
    ``_verify_pdf_matches_metadata`` (so we don't accidentally save a
    paper that merely cites ours).
    """
    if not paper.title or len(paper.title) < 20:
        return None
    import urllib.parse
    blocklist = [d.strip() for d in os.environ.get(
        "PAPER_LIBRARY_SEARCH_BLOCKLIST", "").split(",") if d.strip()]
    q_parts = [f'"{paper.title}"', "filetype:pdf"]
    q_parts.extend(f"-site:{d}" for d in blocklist)
    q = " ".join(q_parts)
    search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(q)}"
    try:
        r = requests.get(search_url, timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
        # status 202 = DDG anti-bot CAPTCHA gate ("complete the challenge").
        # Treat as transient block — return None so cascade continues.
        if r.status_code == 202 or not r.ok:
            return None
        # Defense-in-depth: detect captcha keywords in body.
        if "Unfortunately, bots use DuckDuckGo" in r.text:
            return None
    except Exception:
        return None
    raw_results = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"', r.text)
    candidates: list[str] = []
    seen: set[str] = set()
    for ddg_link in raw_results[:5]:
        m = re.search(r'uddg=([^&]+)', ddg_link)
        if not m:
            continue
        actual = urllib.parse.unquote(m.group(1))
        # Defense in depth: even if blocklist op missed in DDG, drop here
        if any(d in actual for d in blocklist):
            continue
        if actual in seen:
            continue
        seen.add(actual)
        candidates.append(actual)
    for url in candidates:
        try:
            pdf = requests.get(url, timeout=TIMEOUT,
                               headers=BROWSER_HEADERS, allow_redirects=True)
            if pdf.ok and _is_pdf_bytes(pdf.content):
                return pdf.content
        except Exception:
            continue
    return None
