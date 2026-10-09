from __future__ import annotations

import re
from typing import Optional

import requests

from ..models import Paper
from ..sources.inspire import (
    API_URL,
    _document_urls,
    inspire_identity_matches,
    inspire_record_id,
    inspire_title_eligible,
    lookup_inspire_record,
    merge_inspire_fields,
    retained_inspire_fields,
)
from ._shared import TIMEOUT, USER_AGENT, _is_pdf_bytes


def inspire_applicable(paper: Paper) -> bool:
    """Share member prerequisites with domain-group skip/miss accounting."""
    return bool(paper.doi or paper.arxiv_id or inspire_record_id(paper.inspire_record_id)
                or (not paper.inspire_record_id and inspire_title_eligible(paper.model_dump())))


def _try_inspire(paper: Paper) -> Optional[bytes]:
    """Use retained source documents/recid, existing identifiers, or a bounded title lookup.

    Bytes still face the common cascade identity verifier before any save.
    Validated historical source fills persist at DownloadQueue's existing
    save boundary; terminal cards are never reopened here.
    """
    if not inspire_applicable(paper):
        return None
    headers = {"User-Agent": USER_AGENT}
    reference = paper.model_dump()
    pair = retained_inspire_fields(reference)
    urls = pair["inspire_document_urls"]
    recid = pair["inspire_record_id"]
    if recid and not urls:
        candidate = lookup_inspire_record(reference, recid=recid, timeout=TIMEOUT, headers=headers)
        if candidate is None:
            return None
        if (inspire_record_id(paper.inspire_record_id) != recid
                or not inspire_identity_matches(paper.model_dump(), candidate)):
            return None
        fields = merge_inspire_fields(paper.model_dump(), candidate)
        for key, value in fields.items():
            setattr(paper, key, value)
        urls = candidate["inspire_document_urls"]
    elif not recid and (paper.arxiv_id or paper.doi):
        # Preserve the existing identifier query. Older API fixtures/cached
        # responses can omit recids; their bytes remain usable through the PDF
        # verifier, but cannot confer a retained source identity.
        query = ("arxiv:" + re.sub(r"v\d+$", "", paper.arxiv_id.strip())
                 if paper.arxiv_id else f"doi:{paper.doi}")
        try:
            response = requests.get(API_URL, params={"q": query, "size": 1,
                                    "fields": "documents.url"}, timeout=TIMEOUT, headers=headers)
            response.raise_for_status()
            hits = (response.json().get("hits") or {}).get("hits") or []
            urls = _document_urls((hits[0].get("metadata") or {}).get("documents")) if hits else []
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            return None
    elif not recid:
        if paper.inspire_record_id:
            return None  # do not reinterpret a malformed nonblank source ID
        candidate = lookup_inspire_record(reference, timeout=TIMEOUT, headers=headers)
        if candidate is None:
            return None
        fields = merge_inspire_fields(paper.model_dump(), candidate)
        if not fields:
            return None
        for key, value in fields.items():
            setattr(paper, key, value)
        urls = candidate["inspire_document_urls"]
    for url in urls:
        try:
            response = requests.get(url, timeout=TIMEOUT, headers=headers, allow_redirects=True)
            if response.ok and response.status_code != 206 and _is_pdf_bytes(response.content):
                return response.content
        except requests.RequestException:
            continue
    return None
