from __future__ import annotations

import re
from typing import Optional

import requests

from ..models import Paper
from ._shared import TIMEOUT, USER_AGENT, _is_pdf_bytes


def _try_inspire(paper: Paper) -> Optional[bytes]:
    """Try Inspire-HEP's documents[] field for fulltext links.

    Especially useful for HEP / particle physics / astrophysics where
    Inspire is the canonical fulltext source.
    """
    if not paper.arxiv_id and not paper.doi:
        return None
    if paper.arxiv_id:
        arxiv_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
        query = f"arxiv:{arxiv_id}"
    else:
        query = f"doi:{paper.doi}"
    api = (
        "https://inspirehep.net/api/literature"
        f"?q={query}&size=1&fields=documents"
    )
    try:
        meta = requests.get(api, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT}).json()
    except Exception:
        return None
    hits = (meta.get("hits") or {}).get("hits") or []
    if not hits:
        return None
    documents = (hits[0].get("metadata") or {}).get("documents") or []
    for doc in documents:
        url = doc.get("url")
        if not url:
            continue
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None
