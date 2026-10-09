"""Bounded CORE metadata resolution and repository-original retrieval.

Stored CORE URLs identify outputs; the CORE search adapter's paper_id identifies
works. Keep this member read-only toward Paper: recovered identifiers travel with
its bytes until the cascade verifies and saves the winning PDF.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from urllib3.exceptions import HTTPError as _HTTPError

from ..models import Paper, canonicalize_author
from ..store import Library
from ._shared import USER_AGENT, _is_pdf_bytes

_API = "https://api.core.ac.uk/v3/"
_API_INTERVAL = 6.0
_MAX_OUTPUTS = 2
_MAX_ROOTS = 3
_MAX_LINKS = 2
_MAX_GETS = 6  # Includes redirects and explicit files, not just root URLs.
_METADATA_LIMIT = 4 * 1024 * 1024
_HTML_LIMIT = 256 * 1024
_PDF_LIMIT = 32 * 1024 * 1024
_api_lock = threading.Lock()
_api_next_at = 0.0
_api_cooldown_until = 0.0


def _explicit_auth(request: requests.PreparedRequest) -> requests.PreparedRequest:
    """Suppress requests' implicit .netrc auth while retaining proxy settings.

    API requests carry only their explicit CORE bearer; public requests carry
    no authentication. A truthy auth callable prevents ambient library or
    publisher credentials from replacing either policy.
    """
    return request


class _CoreUnavailable(Exception):
    """An API error/rate limit stops this attempt, including search fallback."""


class _CoreRateLimited(_CoreUnavailable):
    """Stop metadata discovery; already validated public origins can still run."""


class _CorePDF(bytes):
    def __new__(cls, data: bytes, doi: str, arxiv_id: str):
        result = super().__new__(cls, data)
        result.doi, result.arxiv_id = doi, arxiv_id
        return result


def _commit_core_identifiers(data: bytes, paper: Paper, library: Library) -> None:
    """Called only after the outer verifier and atomic save succeed."""
    if isinstance(data, _CorePDF):
        library.fill_verified_identifiers(paper.key, doi=data.doi, arxiv_id=data.arxiv_id)


def _http_url(value: object) -> Optional[str]:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    if (any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
            or re.search(r"%(?![0-9a-fA-F]{2})", value)):
        return None
    try:
        parts = urlsplit(value)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None or parts.port == 0):
            return None
        requests.PreparedRequest().prepare_url(value, None)
    except (ValueError, requests.RequestException):
        return None
    return urlunsplit(parts._replace(fragment=""))


def _decimal(value: object) -> Optional[str]:
    value = str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else ""
    if re.fullmatch(r"[0-9]+", value) and value.lstrip("0"):
        return value.lstrip("0")
    return None


def _output_id(value: object, *, api_only: bool = False) -> Optional[str]:
    url = _http_url(value)
    if not url:
        return None
    p = urlsplit(url)
    if p.query or p.fragment or p.port not in {None, 80 if p.scheme == "http" else 443}:
        return None
    if p.hostname == "api.core.ac.uk" and p.scheme == "https":
        match = re.fullmatch(r"/v3/outputs/([0-9]+)/?", p.path)
    elif not api_only and p.hostname in {"core.ac.uk", "www.core.ac.uk"}:
        match = re.fullmatch(r"/(?:download/(?:pdf/)?([0-9]+)(?:\.pdf)?|outputs/([0-9]+))/?", p.path)
    else:
        return None
    return _decimal(next((g for g in match.groups() if g), "")) if match else None


def _core_locator(paper: Paper) -> Optional[tuple[str, str]]:
    output = _output_id(paper.url)
    if output:
        return "outputs", output
    # A numeric identifier alone is never a namespace. search_core stores
    # work['id']; don't guess the output namespace if a work lookup misses.
    work = _decimal(paper.paper_id) if paper.source == "core" else None
    return ("works", work) if work else None


def _core_skip_reason(paper: Paper) -> Optional[str]:
    if not os.environ.get("CORE_API_KEY", "").strip():
        return "missing_credentials"
    if not (_core_locator(paper) or paper.doi.strip() or len(paper.title.strip()) >= 20):
        return "missing_identifier"
    return None


def _delay(value: str, *, epoch: bool = False) -> float:
    """CORE emits ISO timestamps; Retry-After also permits seconds/HTTP dates."""
    try:
        number = float(value)
        return max(0.0, number - time.time() if epoch else number)
    except (ValueError, TypeError):
        pass
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
        except (ValueError, TypeError, OverflowError):
            return 0.0
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return max(0.0, date.timestamp() - time.time())


def _read_body(response: requests.Response, limit: int, *, origin: bool = False) -> bytes:
    # Request identity encoding and reject servers that ignore it. Decoders can
    # buffer arbitrarily many slow chunks before returning any decoded bytes.
    if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
        raise _CoreUnavailable("unexpected content encoding")
    chunks, size = [], 0
    prefix = b""
    deadline = time.monotonic() + 40
    while True:
        # read1 yields currently available data instead of waiting to fill a
        # 64 KiB buffer; slow-drip responses must reach the deadline check.
        try:
            chunk = response.raw.read1(64 * 1024, decode_content=False)
        except _HTTPError as exc:
            raise _CoreUnavailable("body read failed") from exc
        if not chunk:
            break
        if origin and len(prefix) < 5:
            prefix = (prefix + chunk)[:5]
            if len(prefix) == 5:
                limit = _PDF_LIMIT if _is_pdf_bytes(prefix) else _HTML_LIMIT
        size += len(chunk)
        if size > limit or time.monotonic() > deadline:
            raise _CoreUnavailable("body limit")
        chunks.append(chunk)
    data = b"".join(chunks)
    length = response.headers.get("Content-Length")
    if length:
        if not length.isdecimal() or int(length) != len(data):
            raise _CoreUnavailable("incomplete body")
    return data


def _api_get(path: str, key: str, *, params: Optional[dict] = None) -> Optional[dict]:
    global _api_next_at, _api_cooldown_until
    # Lock covers transport and response headers, so another downloader cannot
    # race a just-arriving 429. During cooldown return a miss without sleeping.
    with _api_lock:
        if time.monotonic() < _api_cooldown_until:
            raise _CoreRateLimited("cooldown")
        wait = _api_next_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            with requests.get(_API + path, params=params,
                              headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT,
                                       "Accept-Encoding": "identity"},
                              auth=_explicit_auth, timeout=15, stream=True, allow_redirects=False) as r:
                delays = [_delay(r.headers.get("Retry-After", "")),
                          _delay(r.headers.get("X-RateLimit-Retry-After", ""))]
                if r.status_code == 429 or r.headers.get("X-RateLimit-Remaining") == "0":
                    delays.append(_delay(r.headers.get("X-RateLimit-Reset", ""), epoch=True))
                    cooldown = max(delays + ([60.0] if r.status_code == 429 and not any(delays) else []))
                    _api_cooldown_until = max(_api_cooldown_until, time.monotonic() + cooldown)
                elif any(delays):
                    _api_next_at = max(_api_next_at, time.monotonic() + max(delays))
                if r.status_code == 404:
                    return None
                if r.status_code == 429:
                    raise _CoreRateLimited("CORE HTTP 429")
                if r.status_code != 200:
                    raise _CoreUnavailable(f"CORE HTTP {r.status_code}")
                data = json.loads(_read_body(r, _METADATA_LIMIT))
                if not isinstance(data, dict):
                    raise _CoreUnavailable("invalid metadata")
                return data
        finally:
            _api_next_at = max(_api_next_at, time.monotonic() + _API_INTERVAL)


def _doi(value: object) -> str:
    if not isinstance(value, str):
        return ""
    clean = re.sub(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)", "", value.strip(), flags=re.I)
    return clean.casefold() if re.fullmatch(r"10\.[0-9]{4,9}/[^\s<>]+", clean) else ""


def _arxiv(value: object) -> str:
    if not isinstance(value, str):
        return ""
    clean = re.sub(r"^(?:arxiv:|oai:arxiv\.org:)", "", value.strip(), flags=re.I)
    match = re.fullmatch(r"([0-9]{4}\.[0-9]{4,5}|[a-z-]+(?:\.[A-Z]{2})?/[0-9]{7})(?:v[0-9]+)?", clean, re.I)
    if not match:
        return ""
    base = match[1]
    digits = base.split("/")[-1]
    return base if 1 <= int(digits[2:4]) <= 12 else ""


def _url_arxiv(value: object) -> str:
    url = _http_url(value)
    if not url:
        return ""
    p = urlsplit(url)
    if p.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        return ""
    m = re.fullmatch(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?", p.path)
    return _arxiv(m[1]) if m else ""


def _metadata_identifiers(metadata: dict) -> Optional[tuple[str, str]]:
    dois = {_doi(metadata.get("doi"))} - {""}
    arxivs = {_arxiv(metadata.get("arxivId"))} - {""}
    identifiers = metadata.get("identifiers") or {}
    # The work and output endpoints use different identifier shapes.
    if isinstance(identifiers, dict):
        identifiers = [{"type": kind, "identifier": value} for kind, value in identifiers.items()]
    for entry in identifiers if isinstance(identifiers, list) else []:
        if not isinstance(entry, dict):
            continue
        value, kind = entry.get("identifier"), str(entry.get("type", "")).lower()
        if kind == "doi":
            dois.add(_doi(value))
        elif kind in {"arxiv", "arxiv_id"} or (kind in {"oai", "oai_id"}
                and isinstance(value, str) and value.lower().startswith("oai:arxiv.org:")):
            arxivs.add(_arxiv(value))
    for value in [*(metadata.get("sourceFulltextUrls") or []), metadata.get("downloadUrl")]:
        arxivs.add(_url_arxiv(value))
    for value in [metadata.get("oai"), *(metadata.get("oaiIds") or [])]:
        if isinstance(value, str) and value.lower().startswith("oai:arxiv.org:"):
            arxivs.add(_arxiv(value))
    dois.discard("")
    arxivs.discard("")
    if len(dois) > 1 or len(arxivs) > 1:
        return None
    return next(iter(dois), ""), next(iter(arxivs), "")


def _norm(value: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in value.casefold()).split())


def _corroborate(paper: Paper, metadata: dict, parent: tuple[str, str] = ("", "")) -> Optional[tuple[str, str]]:
    ids = _metadata_identifiers(metadata)
    if ids is None:
        return None
    doi, arxiv = ids
    old_doi = re.sub(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)", "", paper.doi.strip(), flags=re.I)
    old_arxiv = _arxiv(paper.arxiv_id) or paper.arxiv_id.strip()
    if ((doi and old_doi and doi.casefold() != old_doi.casefold())
            or (arxiv and old_arxiv and arxiv.casefold() != old_arxiv.casefold())
            or (doi and parent[0] and doi.casefold() != parent[0].casefold())
            or (arxiv and parent[1] and arxiv.casefold() != parent[1].casefold())):
        return None
    title = metadata.get("title")
    our, their = _norm(paper.title), _norm(title) if isinstance(title, str) else ""
    exact_id = bool((doi and old_doi and doi.casefold() == old_doi.casefold())
                    or (arxiv and old_arxiv and arxiv.casefold() == old_arxiv.casefold()))
    if our and their:
        if our != their and (len(our) < 20 or SequenceMatcher(None, our, their, autojunk=False).ratio() < 0.90):
            return None
    elif not exact_id:
        return None
    authors = metadata.get("authors") or []
    ours = {_norm(canonicalize_author(a).split()[-1]) for a in paper.authors if a.strip()}
    theirs = set()
    for author in authors if isinstance(authors, list) else []:
        name = author.get("name", "") if isinstance(author, dict) else author
        if isinstance(name, str) and name.strip():
            theirs.add(_norm(canonicalize_author(name).split()[-1]))
    if ours and theirs and not ours.intersection(theirs):
        return None
    return doi or parent[0], arxiv or parent[1]


def _candidate_url(value: object) -> Optional[str]:
    url = _http_url(value)
    if not url:
        return None
    p = urlsplit(url)
    if p.hostname == "api.core.ac.uk":
        return None  # Metadata endpoints aren't public full-text candidates.
    if p.hostname in {"core.ac.uk", "www.core.ac.uk"}:
        output = _output_id(url)
        if output and p.path.startswith("/download/"):
            return f"https://core.ac.uk/download/{output}.pdf"
        return None  # display, reader and thumbnail URLs
    arxiv = _url_arxiv(url)
    if arxiv:
        return f"https://arxiv.org/pdf/{arxiv}"
    return url


class _CoreLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.citations: list[str] = []
        self.files: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        a = {key.lower(): value or "" for key, value in attrs}
        if tag == "meta" and a.get("name", "").lower() == "citation_pdf_url":
            self.citations.append(a.get("content", ""))
        elif tag == "link" and a.get("type", "").lower() == "application/pdf":
            self.files.append(a.get("href", ""))
        elif tag == "a":
            href = a.get("href", "")
            try:
                path = urlsplit(href).path.lower()
            except ValueError:
                return
            if (path.endswith(".pdf") or re.search(r"/(?:bitstreams?|download)(?:/|$)", path)
                    or path.endswith("/viewcontent.cgi")):
                self.files.append(href)


class _Origins:
    def __init__(self):
        self.gets = 0
        self.seen: set[str] = set()

    def get(self, url: str) -> Optional[tuple[bytes, str]]:
        # Manual redirects count toward the same six-GET budget and revalidate
        # every URL. Credentials are never attached to origins or file links.
        while self.gets < _MAX_GETS and url not in self.seen:
            self.seen.add(url)
            self.gets += 1
            try:
                with requests.get(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}, timeout=15,
                                  auth=_explicit_auth, stream=True, allow_redirects=False) as r:
                    if r.status_code in {301, 302, 303, 307, 308}:
                        url = _candidate_url(urljoin(url, r.headers.get("Location", "")))
                        if not url:
                            return None
                        continue
                    if (not 200 <= r.status_code < 300 or r.status_code == 206
                            or r.headers.get("Content-Range")):
                        return None
                    return _read_body(r, _HTML_LIMIT, origin=True), url
            except (requests.RequestException, _CoreUnavailable, ValueError):
                return None
        return None

    def pdf(self, url: str) -> Optional[bytes]:
        response = self.get(url)
        if response is None:
            return None
        body, final_url = response
        if _is_pdf_bytes(body):
            return body if b"%%EOF" in body[-1024:] else None
        links = _CoreLinks()
        try:
            links.feed(body.decode("utf-8", errors="replace"))
        except ValueError:
            return None
        seen, candidates = set(), []
        for href in links.citations + links.files:
            try:
                file_url = _candidate_url(urljoin(final_url, href))
            except ValueError:
                continue
            if file_url and file_url not in seen and file_url not in self.seen:
                seen.add(file_url)
                candidates.append(file_url)
        for file_url in candidates[:_MAX_LINKS]:
            response = self.get(file_url)
            if response and _is_pdf_bytes(response[0]) and b"%%EOF" in response[0][-1024:]:
                return response[0]
        return None


def _try_core(paper: Paper) -> Optional[bytes]:
    """Resolve a stored output/work locator, then bounded DOI/title discovery.

    Disabled fulltext metadata doesn't forbid source links. Return complete PDF
    bytes through the ordinary cascade; access blocks remain misses, not absence.
    """
    if _core_skip_reason(paper):
        return None
    key = os.environ.get("CORE_API_KEY", "").strip()
    candidates: dict[str, Optional[tuple[str, str]]] = {}
    seen_outputs: set[str] = set()

    def harvest(metadata: dict, ids: tuple[str, str]) -> None:
        urls = list(metadata.get("sourceFulltextUrls") or [])
        urls += [metadata.get("downloadUrl"), metadata.get("fullTextLink")]
        urls += [link.get("url") for link in metadata.get("links") or []
                 if isinstance(link, dict) and link.get("type") == "download"]
        for raw in urls:
            url = _candidate_url(raw)
            if not url:
                continue
            if url not in candidates:
                candidates[url] = ids
                continue
            previous = candidates[url]
            if previous is not None:
                if any(old and new and old.casefold() != new.casefold()
                       for old, new in zip(previous, ids)):
                    candidates[url] = None  # Contradictory metadata isn't a clean miss.
                else:
                    candidates[url] = (previous[0] or ids[0], previous[1] or ids[1])

    def expand(work: dict, ids: tuple[str, str]) -> None:
        harvest(work, ids)
        for raw in work.get("outputs") or []:
            output = _output_id(raw, api_only=True)
            if not output or output in seen_outputs:
                continue
            if len(seen_outputs) >= _MAX_OUTPUTS:
                break
            seen_outputs.add(output)
            metadata = _api_get(f"outputs/{output}", key)
            if metadata:
                output_ids = _corroborate(paper, metadata, ids)
                if output_ids is not None:
                    harvest(metadata, output_ids)

    try:
        locator = _core_locator(paper)
        if locator:
            namespace, identifier = locator
            metadata = _api_get(f"{namespace}/{identifier}", key)
            if metadata:
                ids = _corroborate(paper, metadata)
                if ids is None:
                    return None  # A conflict is not a clean miss for search fallback.
                if namespace == "outputs":
                    seen_outputs.add(identifier)
                    harvest(metadata, ids)
                else:
                    expand(metadata, ids)
        queries = []
        if paper.doi.strip():
            clean_doi = paper.doi.replace('"', '').replace('\\', '')
            queries.append((f'doi:"{clean_doi}"', 3))
        if len(paper.title.strip()) >= 20:
            title = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in paper.title)[:120]
            queries.append((f'title:"{title}"', 5))
        for query, limit in queries:
            if candidates:
                break
            result = _api_get("search/works/", key, params={"q": query, "limit": limit})
            for work in (result or {}).get("results", [])[:limit]:
                if isinstance(work, dict):
                    ids = _corroborate(paper, work)
                    if ids is not None:
                        expand(work, ids)
    except _CoreRateLimited:
        pass  # No further metadata calls; do not discard known public originals.
    except (requests.RequestException, _CoreUnavailable, ValueError, TypeError):
        return None

    # Equivalent CORE downloads deduplicate before the cap. Originals precede
    # every CORE-hosted file so blocked duplicates cannot starve repositories.
    ordered = [(url, ids) for url, ids in candidates.items() if ids is not None]
    ordered.sort(key=lambda candidate: urlsplit(candidate[0]).hostname == "core.ac.uk")
    origins = _Origins()
    for url, ids in ordered[:_MAX_ROOTS]:
        pdf = origins.pdf(url)
        if pdf:
            return _CorePDF(pdf, *ids)
    return None
