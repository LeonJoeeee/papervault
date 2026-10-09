from __future__ import annotations

import os
from typing import Optional

import requests

from ..models import Paper
from ._shared import _is_pdf_bytes


def _try_elsevier_tdm(paper: Paper) -> Optional[bytes]:
    """Elsevier Text-and-Data-Mining API — official Elsevier API for
    institutional subscribers. Returns the article in XML by default; we
    request PDF specifically.

    Requires:
      - ELSEVIER_TDM_API_KEY env var (institutional subscription required)

    Without this, Elsevier (10.1016) papers are heavily paywalled. Sci-Hub
    coverage of 2020+ Elsevier titles has gaps.
    """
    key = os.environ.get("ELSEVIER_TDM_API_KEY", "").strip()
    if not key:
        return None
    if not paper.doi:
        return None
    prefix = paper.doi.split("/", 1)[0]
    if prefix not in ("10.1016",):
        return None
    api_url = (f"https://api.elsevier.com/content/article/doi/{paper.doi}"
               f"?apiKey={key}")
    try:
        r = requests.get(api_url, timeout=60,
                         headers={"Accept": "application/pdf"},
                         allow_redirects=True)
    except Exception:
        return None
    if r.ok and _is_pdf_bytes(r.content):
        return r.content
    return None
