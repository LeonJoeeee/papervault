from __future__ import annotations

import time
from typing import Optional

import requests

from ..models import Paper
from ._shared import BROWSER_HEADERS, TIMEOUT, _is_pdf_bytes


def _try_iopscience_direct(paper: Paper, max_attempts: int = 3) -> Optional[bytes]:
    """For IOPscience-hosted DOIs (10.3847 = AAS/ApJ; 10.1088 = IOP) try the
    direct PDF endpoint. The article landing page (``/article/{doi}``) is
    routinely blocked by Radware Bot Manager, but the PDF endpoint
    (``/article/{doi}/pdf``) has a separate, weaker policy — discovered
    during R10/R11 of the 82-paper bench: 15 stubborn ApJ papers
    recoverable here when crossref_tm and scihub had missed.

    Some mihomo exit IPs get Radware-redirected to perfdrive.com/captcha
    (response is 200 + ~14KB HTML, not a PDF). Retry with fresh connections
    to give different IPs a chance.
    """
    doi = paper.doi or ""
    if not (doi.startswith("10.3847/") or doi.startswith("10.1088/")):
        return None
    url = f"https://iopscience.iop.org/article/{doi}/pdf"
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, timeout=TIMEOUT, allow_redirects=True,
                             headers={**BROWSER_HEADERS,
                                      "Accept": "application/pdf,*/*;q=0.8"})
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            pass
        if attempt < max_attempts - 1:
            time.sleep(1 + 2 * attempt)
    return None
