from __future__ import annotations

import os
from typing import Optional

import requests

from ..models import Paper
from ._shared import BROWSER_HEADERS, TIMEOUT, USER_AGENT, _is_pdf_bytes


def _try_unpaywall(paper: Paper) -> Optional[bytes]:
    """Unpaywall: returns multiple ``oa_locations``; try them all, with
    BROWSER_HEADERS as a second pass for publishers that 403 bare UAs.

    The previous version only tried the first location's ``url_for_pdf``,
    which gave up immediately when (a) the field was missing — common for
    repository entries that only have ``url`` — or (b) the URL was blocked
    by a publisher CDN (MDPI's Akamai, RSC's Cloudflare). Iterating + the
    browser header retry recovers some of these.
    """
    if not paper.doi:
        return None
    email = os.environ.get("UNPAYWALL_EMAIL", "research@example.invalid")
    api = f"https://api.unpaywall.org/v2/{paper.doi}?email={email}"
    try:
        meta = requests.get(api, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}).json()
    except Exception:
        return None

    # Collect all candidate URLs across best_oa_location + oa_locations
    candidates: list[str] = []
    seen: set[str] = set()
    def _add(loc: dict | None) -> None:
        if not loc:
            return
        for field in ("url_for_pdf", "url"):
            u = loc.get(field)
            if u and u not in seen:
                seen.add(u)
                candidates.append(u)
    _add(meta.get("best_oa_location"))
    for loc in (meta.get("oa_locations") or []):
        _add(loc)
    if not candidates:
        return None

    # Two-pass: simple UA first, full browser headers second
    for url in candidates:
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    for url in candidates:
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers=BROWSER_HEADERS,
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None
