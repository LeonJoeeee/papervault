from __future__ import annotations

import os
from typing import Optional

import requests

from ..models import Paper
from ._shared import _is_pdf_bytes, log


def _try_wiley_tdm(paper: Paper) -> Optional[bytes]:
    """Wiley Text-and-Data-Mining (TDM) API — official publisher API for
    Wiley-hosted journals, free for academic users. Covers ~2,000 Wiley
    journals including Wiley Online Library and AGU titles (which Wiley
    publishes since 2013, prefix 10.1029/10.1002).

    Without this tier, Wiley papers are ~impossible: their CDN does TLS
    fingerprinting + JS challenges that defeat curl-cffi, and sci-hub's
    coverage of 2018+ Wiley titles is patchy.

    Token registration (free, ~1 day approval): see
    https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining
    Set WILEY_TDM_TOKEN env var.

    The API returns the PDF directly. Has a documented rate limit; we add
    a small inline sleep to be polite.
    """
    token = os.environ.get("WILEY_TDM_TOKEN", "").strip()
    if not token:
        log.debug("wiley_tdm: skipped — WILEY_TDM_TOKEN not set")
        return None
    if not paper.doi:
        return None
    # Wiley TDM serves any DOI hosted on onlinelibrary.wiley.com — that
    # includes 10.1002/* (Wiley) and 10.1029/* (AGU, post-2013).
    prefix = paper.doi.split("/", 1)[0]
    if prefix not in ("10.1002", "10.1029", "10.1111", "10.1046"):
        return None
    from urllib.parse import quote
    encoded_doi = quote(paper.doi, safe="")
    api_url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{encoded_doi}"
    try:
        r = requests.get(api_url, timeout=60,
                         headers={"Wiley-TDM-Client-Token": token},
                         allow_redirects=True)
    except Exception as exc:
        log.warning("wiley_tdm[%s]: request exception %r", paper.key, exc)
        return None
    if r.ok and _is_pdf_bytes(r.content):
        return r.content
    # Visible failure — log status + first body bytes so the operator can
    # see WHY the TDM API returned non-PDF (rate limit / not subscribed /
    # bad token / paywalled-without-subscription / etc.).
    body_head = r.content[:200].decode("utf-8", errors="replace")
    log.warning(
        "wiley_tdm[%s]: status=%d ctype=%r len=%d body[:200]=%r",
        paper.key, r.status_code,
        r.headers.get("Content-Type", "")[:60],
        len(r.content), body_head,
    )
    return None
