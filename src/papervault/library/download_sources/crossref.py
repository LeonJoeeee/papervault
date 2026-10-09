from __future__ import annotations

import os
from typing import Optional

import requests

from ..models import Paper
from ._shared import TIMEOUT, USER_AGENT, _is_pdf_bytes


def _try_crossref_link(paper: Paper) -> Optional[bytes]:
    """Try CrossRef's ``link[]`` field for text-mining-licensed PDFs.

    For each ``intended-application = text-mining`` URL, identify
    ourselves as a TDM client (``User-Agent: TextMining/1.0``,
    ``Accept: application/pdf``). Empirically: IOPscience, AAS, AGU/Wiley
    and other CrossRef-listed publishers gate ordinary browser/curl UAs
    behind perfdrive/Cloudflare/Akamai, but honour the TDM intent and
    serve the PDF directly when the request identifies as text-mining.

    This is the documented, legitimate path. CrossRef's ``link[]`` field
    is the publisher's machine-readable opt-in to bulk text/data mining;
    using it as such isn't bypassing anything — it's *using* it.
    """
    if not paper.doi:
        return None
    email = os.environ.get("UNPAYWALL_EMAIL", "research@example.invalid")
    api = f"https://api.crossref.org/works/{paper.doi}"
    try:
        meta = requests.get(
            api, timeout=TIMEOUT,
            headers={"User-Agent": f"paper-library/0.1 (mailto:{email})"},
        ).json()
    except Exception:
        return None
    links = ((meta.get("message") or {}).get("link") or [])
    candidates: list[tuple[str, bool]] = []  # (url, is_tdm)
    seen: set[str] = set()
    for link in links:
        url = link.get("URL")
        if not url or url in seen:
            continue
        intended = (link.get("intended-application") or "").lower()
        ctype = (link.get("content-type") or "").lower()
        is_tdm = intended == "text-mining"
        if (is_tdm or ctype == "application/pdf"
                or url.lower().endswith(".pdf")):
            seen.add(url)
            candidates.append((url, is_tdm))
    tdm_headers = {
        "User-Agent": f"TextMining/1.0 (mailto:{email})",
        "Accept": "application/pdf",
    }
    for url, is_tdm in candidates:
        headers = tdm_headers if is_tdm else {"User-Agent": USER_AGENT}
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers=headers, allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None
