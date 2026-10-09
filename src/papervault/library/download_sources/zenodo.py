from __future__ import annotations

import re
from typing import Optional

import requests

from ..models import Paper
from ._shared import TIMEOUT, USER_AGENT, _is_pdf_bytes


def _try_zenodo(paper: Paper) -> Optional[bytes]:
    """Try Zenodo (CERN-hosted records API).

    Strong for CS / physics preprints, datasets, and conference papers.
    Pulls `hits.hits[0].files[]` and downloads PDF files via `links.self`.
    Uses `type == "pdf"`, or a `.pdf` key when the type is missing or empty.
    """
    if not paper.doi and not paper.arxiv_id:
        return None
    if paper.doi:
        q = f'doi:"{paper.doi}"'
    else:
        arxiv_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
        q = f'arxiv:"{arxiv_id}"'
    api = "https://zenodo.org/api/records"
    try:
        meta = requests.get(
            api,
            params={"q": q, "size": 1},
            timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        ).json()
    except Exception:
        return None
    hits = ((meta.get("hits") or {}).get("hits") or [])
    if not hits:
        return None
    files = hits[0].get("files") or []
    for f in files:
        ftype = (f.get("type") or "").lower()
        untyped_pdf = not ftype and (f.get("key") or "").lower().endswith(".pdf")
        if ftype != "pdf" and not untyped_pdf:
            continue
        url = ((f.get("links") or {}).get("self"))
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
