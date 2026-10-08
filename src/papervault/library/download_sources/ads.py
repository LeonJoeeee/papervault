from __future__ import annotations

import os
import re
from typing import Optional

import requests

from ..models import Paper
from ._shared import TIMEOUT, USER_AGENT, _is_pdf_bytes


def _try_ads(paper: Paper) -> Optional[bytes]:
    """Try NASA ADS link_gateway via the search API for bibcode + esources.

    ADS doesn't return direct PDF URLs in search; PDFs go through a
    redirect gateway:
      https://ui.adsabs.harvard.edu/link_gateway/<bibcode>/<source_type>

    Source types like EPRINT_PDF, PUB_PDF, ADS_PDF, AUTHOR_PDF lead to
    PDFs (often via 302 redirect to the publisher; allow_redirects=True
    handles either 200+body or 302+Location).

    Silently skipped if ADS_API_TOKEN is unset.
    """
    token = os.environ.get("ADS_API_TOKEN", "").strip()
    if not token:
        return None
    if not paper.doi and not paper.arxiv_id:
        return None
    if paper.doi:
        query = f"doi:{paper.doi}"
    else:
        arxiv_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
        query = f"identifier:{arxiv_id}"
    api = (
        "https://api.adsabs.harvard.edu/v1/search/query"
        f"?q={query}&fl=bibcode,esources&rows=1"
    )
    try:
        resp = requests.get(
            api, timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT,
                     "Authorization": f"Bearer {token}"},
        )
        meta = resp.json()
    except Exception:
        return None
    docs = (meta.get("response") or {}).get("docs") or []
    if not docs:
        return None
    bibcode = docs[0].get("bibcode")
    esources = docs[0].get("esources") or []
    if not bibcode:
        return None
    for source_type in esources:
        if "PDF" not in source_type.upper():
            continue
        gateway = (
            f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/{source_type}"
        )
        try:
            r = requests.get(gateway, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue

    # ADS Legacy Article Service fallback. For papers that don't have a
    # direct PDF esource (typical of pre-1995 articles whose original
    # publisher metadata never recorded one), NASA ADS still hosts a
    # scanned PDF at:
    #   https://articles.adsabs.harvard.edu/pdf/<bibcode>
    # This is the dataset behind the old "ADS Article Service" — it
    # covers ApJ / ApJL / A&A and several other journals back to the
    # 1950s. Bibcodes from the search API are the canonical key.
    # No auth required (public endpoint).
    if bibcode:
        legacy_url = f"https://articles.adsabs.harvard.edu/pdf/{bibcode}"
        try:
            r = requests.get(legacy_url, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            pass
    return None
