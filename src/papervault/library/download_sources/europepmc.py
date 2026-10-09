from __future__ import annotations

import re
from typing import Optional

import requests

from ..models import Paper
from ._shared import TIMEOUT, USER_AGENT, _is_pdf_bytes


def _try_europepmc(paper: Paper) -> Optional[bytes]:
    """Try EuropePMC (EBI-hosted, broader than US PMC).

    Covers life science + adjacent physics + applied research. Pulls from
    the result's `fullTextUrlList[]` (filtered to documentStyle="pdf"),
    plus the PMC PDF render URL when a `pmcid` is available.
    """
    if not paper.doi and not paper.arxiv_id:
        return None
    if paper.doi:
        query = f"DOI:{paper.doi}"
    else:
        arxiv_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
        query = f"arXiv:{arxiv_id}"
    api = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={query}&format=json&resultType=core"
    )
    try:
        meta = requests.get(api, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT}).json()
    except Exception:
        return None
    results = ((meta.get("resultList") or {}).get("result") or [])
    if not results:
        return None
    result = results[0]

    candidates: list[str] = []
    seen: set[str] = set()

    full_text_urls = (
        (result.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    )
    for entry in full_text_urls:
        url = entry.get("url")
        if not url or url in seen:
            continue
        if (entry.get("documentStyle") or "").lower() == "pdf":
            seen.add(url)
            candidates.append(url)

    pmcid = result.get("pmcid")
    if pmcid:
        render_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
        if render_url not in seen:
            seen.add(render_url)
            candidates.append(render_url)

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
