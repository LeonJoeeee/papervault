"""HAL and HAL-INSU deposited files, looked up by exact DOI."""
from __future__ import annotations

import math
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import urljoin, urlsplit

import requests

from ..models import Paper
from ._shared import TIMEOUT, USER_AGENT, log
from .known_file_url import _fetch_pdf_url

_API = "https://api.archives-ouvertes.fr/search/"
_FIELDS = "halId_s,doiId_s,title_s,fileMain_s,uri_s,submitType_s"
_request_lock = threading.Lock()
_next_request_at = 0.0
_cooldown_until = 0.0


def _retry_after_seconds(value: str) -> float:
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return 60.0
    return max(1.0, delay) if math.isfinite(delay) else 60.0


def _hal_get(url: str, **kwargs) -> requests.Response:
    """Pace API/file/redirect requests across worker threads; never retry a 429.

    Cooldown callers return a miss immediately, leaving other cascade sources
    available. Follow at most five redirects, each consuming its own slot.
    """
    global _next_request_at, _cooldown_until
    follow_redirects = kwargs.pop("allow_redirects", True)
    for _ in range(6):
        with _request_lock:
            now = time.monotonic()
            if now < _cooldown_until:
                raise requests.RequestException("HAL rate-limit cooldown")
            if now < _next_request_at:
                time.sleep(_next_request_at - now)
            _next_request_at = time.monotonic() + 1.0
            response = requests.get(url, allow_redirects=False, **kwargs)
            if response.status_code == 429:
                delay = _retry_after_seconds(response.headers.get("Retry-After", ""))
                _cooldown_until = time.monotonic() + delay
                log.warning("hal_repository: HTTP 429; cooling down for %.1fs", delay)
        if not follow_redirects or not response.is_redirect:
            return response
        url = urljoin(response.url, response.headers["Location"])
        kwargs.pop("params", None)
    raise requests.TooManyRedirects("HAL document exceeded five redirects")


def _try_hal_repository(paper: Paper) -> Optional[bytes]:
    """Fetch fileMain_s from a DOI-matched deposit, then use cascade identity.

    The default HAL portal includes HAL-INSU. Notices and landing-page uri_s
    values are never file candidates. No credentials or title search needed.
    """
    doi = (paper.doi or "").strip()
    if not doi:
        return None
    escaped = doi.replace("\\", "\\\\").replace('"', '\\"')
    try:
        response = _hal_get(
            _API, params={"q": f'doiId_s:("{escaped}")', "wt": "json",
                          "fl": _FIELDS, "rows": 10},
            headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
        )
        if not response.ok:
            return None
        meta = response.json()
    except (requests.RequestException, ValueError):
        return None
    result = meta.get("response") if isinstance(meta, dict) else None
    docs = result.get("docs") if isinstance(result, dict) else None
    if not isinstance(docs, list):
        return None
    seen = set()
    for doc in docs[:10]:
        if not isinstance(doc, dict):
            continue
        record_dois = doc.get("doiId_s")
        if isinstance(record_dois, str):
            record_dois = [record_dois]
        if not isinstance(record_dois, list) or not any(
            isinstance(d, str) and d.strip().casefold() == doi.casefold() for d in record_dois
        ):
            continue
        url = doc.get("fileMain_s")
        if not isinstance(url, str) or not url or url in seen:
            continue
        try:
            parts = urlsplit(url)
        except ValueError:
            continue
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            continue
        seen.add(url)
        data = _fetch_pdf_url(url, get=_hal_get)
        if data:
            return data
    return None
