from __future__ import annotations

import re
from typing import Optional

import requests

from ..models import Paper
from ._shared import BROWSER_USER_AGENT, TIMEOUT, _is_pdf_bytes


# SSRN download link pattern (relative to https://papers.ssrn.com).
_SSRN_DELIVERY_RE = re.compile(
    r'href=["\'](/sol3/Delivery\.cfm/[^"\']+\.pdf[^"\']*)["\']',
    re.IGNORECASE,
)


def _try_ssrn(paper: Paper) -> Optional[bytes]:
    """SSRN papers — DOI prefix `10.2139/ssrn.<N>` → fetch via the
    abstract page's "Download This Paper" link.

    SSRN sometimes paywalls papers (private uploads, embargoed); those
    return non-PDF and fail magic-byte check. Public SSRN papers download
    cleanly.
    """
    if not paper.doi or not paper.doi.startswith("10.2139/ssrn."):
        return None
    try:
        abstract_id = paper.doi.split("ssrn.")[-1].strip()
    except Exception:
        return None
    if not abstract_id:
        return None
    landing_url = (
        f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={abstract_id}"
    )
    session = requests.Session()
    session.headers.update({"User-Agent": BROWSER_USER_AGENT})
    try:
        landing = session.get(landing_url, timeout=TIMEOUT, allow_redirects=True)
    except Exception:
        return None
    if not landing.ok:
        return None
    # Look for a Delivery.cfm download link in the page.
    m = _SSRN_DELIVERY_RE.search(landing.text)
    if not m:
        return None
    pdf_path = m.group(1)
    pdf_url = f"https://papers.ssrn.com{pdf_path}"
    try:
        r = session.get(
            pdf_url,
            timeout=TIMEOUT,
            headers={"Referer": landing_url},
            allow_redirects=True,
        )
        if r.ok and _is_pdf_bytes(r.content):
            return r.content
    except Exception:
        return None
    return None
