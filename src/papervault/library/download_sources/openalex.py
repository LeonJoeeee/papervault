from __future__ import annotations

import os
from typing import Optional

import requests

from ..models import Paper
from ._shared import TIMEOUT, USER_AGENT, _is_pdf_bytes


def _try_openalex(paper: Paper) -> Optional[bytes]:
    """Try OpenAlex's oa_locations for alternate PDF URLs.

    Different from Unpaywall: OpenAlex returns multiple `oa_locations`,
    not just one `best_oa_location`, so it often surfaces PDFs Unpaywall
    misses.
    """
    if not paper.doi:
        return None
    mailto = os.environ.get("OPENALEX_MAILTO", "research@example.invalid")
    api = f"https://api.openalex.org/works/doi:{paper.doi}?mailto={mailto}"
    try:
        meta = requests.get(api, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT}).json()
    except Exception:
        return None

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(loc: dict | None) -> None:
        if not loc:
            return
        for field in ("pdf_url", "url_for_pdf"):
            url = loc.get(field)
            if url and url not in seen:
                seen.add(url)
                candidates.append(url)

    _add(meta.get("best_oa_location"))
    for loc in meta.get("oa_locations") or []:
        _add(loc)

    for url in candidates:
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None
