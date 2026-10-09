from __future__ import annotations

import json
import io
import os
import re
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import partial
from http.client import HTTPResponse
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import quote, unquote, urljoin, urlsplit

import requests
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import HTTPError

from ..models import Paper
from ..store import Library
from ._shared import USER_AGENT, _is_pdf_bytes, log

_MAX_CANDIDATES = 6
_MAX_LINKS = 2
_MAX_REQUESTS = 16  # Includes metadata, redirects and landing-to-file requests.
_MAX_REDIRECTS = 5
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_HTML_BYTES = 128 * 1024
_MAX_PDF_BYTES = 32 * 1024 * 1024
_TIME_BUDGET = 60
_SOCKET_TIMEOUT = 10
_DNS_SLOTS = threading.BoundedSemaphore(4)


def _resolve_addresses(host: str, port: int, deadline: float) -> list:
    """Wait only within the budget; at most four system resolvers can linger."""
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not _DNS_SLOTS.acquire(blocking=False):
        raise requests.Timeout("OpenAlex DNS budget exhausted")
    result = queue.Queue(maxsize=1)

    def resolve():
        try:
            result.put((True, socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
        except OSError as exc:
            result.put((False, exc))
        finally:
            _DNS_SLOTS.release()

    threading.Thread(target=resolve, daemon=True, name="openalex-dns").start()
    try:
        ok, value = result.get(timeout=remaining)
    except queue.Empty as exc:
        raise requests.Timeout("OpenAlex DNS deadline exhausted") from exc
    if not ok:
        raise requests.ConnectionError(str(value))
    return value


def _connect_socket(connection, *, deadline: float):
    addresses = _resolve_addresses(connection._dns_host, connection.port, deadline)
    error = None
    for family, kind, protocol, _, address in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise requests.Timeout("OpenAlex connect deadline exhausted")
        sock = socket.socket(family, kind, protocol)
        try:
            sock.settimeout(min(connection.timeout, remaining))
            for option in connection.socket_options or []:
                sock.setsockopt(*option)
            if connection.source_address:
                sock.bind(connection.source_address)
            sock.connect(address)
            return sock
        except OSError as exc:
            error = exc
            sock.close()
    raise requests.ConnectionError(str(error or "no resolved addresses"))


class _DeadlineReader(io.RawIOBase):
    """Bound every socket read, including HTTP headers and chunk framing."""

    def __init__(self, raw, sock, deadline):
        self.raw, self.sock, self.deadline = raw, sock, deadline

    def readable(self):
        return True

    def readinto(self, target):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OpenAlex source deadline exhausted")
        self.sock.settimeout(min(_SOCKET_TIMEOUT, remaining))
        return self.raw.readinto(target)

    def close(self):
        self.raw.close()
        super().close()


def _deadline_response(sock, *, deadline, **kwargs):
    response = HTTPResponse(sock, **kwargs)
    raw = response.fp.detach()
    response.fp = io.BufferedReader(_DeadlineReader(raw, sock, deadline))
    return response


class _DeadlineAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, deadline):
        self.deadline = deadline
        super().__init__(max_retries=0)

    def _bind_deadline(self, manager):
        deadline = self.deadline
        factory = staticmethod(partial(_deadline_response, deadline=deadline))

        class HTTP(HTTPConnection):
            response_class = factory

            def _new_conn(self):
                return _connect_socket(self, deadline=deadline)

        class HTTPS(HTTPSConnection):
            response_class = factory

            def _new_conn(self):
                return _connect_socket(self, deadline=deadline)

        class HTTPPool(HTTPConnectionPool):
            ConnectionCls = HTTP

        class HTTPSPool(HTTPSConnectionPool):
            ConnectionCls = HTTPS

        manager.pool_classes_by_scheme = {"http": HTTPPool, "https": HTTPSPool}

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self._bind_deadline(self.poolmanager)

    def proxy_manager_for(self, *args, **kwargs):
        manager = super().proxy_manager_for(*args, **kwargs)
        self._bind_deadline(manager)
        return manager


@contextmanager
def _request(url: str, *, deadline: float):
    # Adapter.send bypasses Session's ambient netrc auth and eager redirect
    # body consumption. Follow redirects only in the budgeted loop below.
    adapter = _DeadlineAdapter(deadline)
    try:
        request = requests.Request("GET", url, headers={
            "User-Agent": USER_AGENT, "Accept-Encoding": "identity",
        }).prepare()
        with adapter.send(request, stream=True,
                          timeout=min(_SOCKET_TIMEOUT, deadline - time.monotonic()),
                          verify=os.environ.get("REQUESTS_CA_BUNDLE") or True,
                          proxies=requests.utils.get_environ_proxies(url)) as response:
            yield response
    finally:
        adapter.close()


def _openalex_work_id(paper: Paper) -> Optional[str]:
    """Only an exact work URL or a source-qualified paper_id is an identity."""
    values = [paper.url]
    if paper.source == "openalex":
        values.append(paper.paper_id)
    for value in values:
        pattern = r"https?://openalex\.org/([Ww][0-9]+)"
        match = re.fullmatch(pattern, value or "")
        if not match and paper.source == "openalex" and value == paper.paper_id:
            match = re.fullmatch(r"([Ww][0-9]+)", value or "")
        if match:
            return match[1].upper()
    return None


def _http_url(value: object, *, metadata: bool = False) -> Optional[str]:
    if (not isinstance(value, str) or not value or "\\" in value
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
            or re.search(r"%(?![0-9a-fA-F]{2})", value)):
        return None
    try:
        parts = urlsplit(value)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.port == 0):
            return None
        prepared = requests.PreparedRequest()
        prepared.prepare_url(value, None)
        hostname = urlsplit(prepared.url).hostname.rstrip(".")
        if metadata:
            # A merged singleton may redirect to another singleton, never to
            # paid content, a search/list endpoint or an arbitrary host.
            if (parts.scheme != "https" or parts.netloc != "api.openalex.org"
                    or not re.fullmatch(r"/works/(?:[Ww][0-9]+|doi:10\.[^?\s]+)", parts.path)
                    or (parts.query and not parts.query.startswith("mailto="))
                    or "&" in parts.query):
                return None
        elif hostname == "openalex.org" or hostname.endswith(".openalex.org"):
            # Cached content is separately authenticated and metered. Deny it
            # even if a location, explicit link or redirect advertises it.
            return None
        return value.split("#", 1)[0]
    except (ValueError, requests.RequestException):
        return None


@dataclass
class _FetchBudget:
    deadline: float = field(default_factory=lambda: time.monotonic() + _TIME_BUDGET)
    requests: int = 0
    seen: set[str] = field(default_factory=set)

    def get(self, url: str, *, metadata: bool = False) -> Optional[tuple[str, bytes]]:
        """Stream bounded bodies; HTML may be capped, PDFs/JSON must be complete."""
        for redirects in range(_MAX_REDIRECTS + 1):
            url = _http_url(url, metadata=metadata)
            remaining = self.deadline - time.monotonic()
            if (not url or url in self.seen or self.requests >= _MAX_REQUESTS
                    or remaining <= 0):
                return None
            self.seen.add(url)
            self.requests += 1
            try:
                with _request(url, deadline=self.deadline) as response:
                    status = response.status_code
                    if status in {301, 302, 303, 307, 308}:
                        if redirects == _MAX_REDIRECTS or not response.headers.get("Location"):
                            return None
                        url = urljoin(url, response.headers["Location"])
                        continue
                    if (not 200 <= status < 300 or status == 206
                            or response.headers.get("Content-Range")):
                        log.debug("openalex: miss HTTP %s at %s", status, url)
                        return None
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        return None
                    body = bytearray()
                    while True:
                        # The adapter enforces the deadline during HTTP framing
                        # too; read1 avoids waiting for a full payload chunk.
                        chunk = response.raw.read1(8192, decode_content=False)
                        if time.monotonic() >= self.deadline:
                            return None
                        if not chunk:
                            break
                        body.extend(chunk)
                        limit = (_MAX_METADATA_BYTES if metadata else
                                 _MAX_PDF_BYTES if _is_pdf_bytes(body) else _MAX_HTML_BYTES)
                        if len(body) > limit:
                            if not metadata and not _is_pdf_bytes(body):
                                return url, bytes(body[:limit])
                            log.debug("openalex: oversized body at %s", url)
                            return None
                    length = response.headers.get("Content-Length")
                    if length and int(length) != len(body):
                        return None
                    return url, bytes(body)
            except (requests.RequestException, HTTPError, ValueError) as exc:
                log.debug("openalex: miss at %s (%s)", url, type(exc).__name__)
                return None
        return None


class _FileLinks(HTMLParser):
    """Only explicit PDF citation metadata and file/download anchors."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.citations: list[str] = []
        self.links: list[str] = []
        self.anchor: Optional[dict] = None
        self.label: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and (attrs.get("name") or "").lower() == "citation_pdf_url":
            if attrs.get("content"):
                self.citations.append(attrs["content"])
        if tag == "a":
            self.anchor, self.label = attrs, []

    def handle_data(self, data):
        if self.anchor is not None:
            self.label.append(data)

    def handle_endtag(self, tag):
        if tag != "a" or self.anchor is None:
            return
        attrs, self.anchor = self.anchor, None
        href = attrs.get("href") or ""
        path = unquote(urlsplit(href).path).lower()
        label = " ".join(self.label).strip().lower()
        if (path.endswith(".pdf") or attrs.get("type") == "application/pdf"
                or "download" in attrs or label in {"pdf", "download", "download pdf", "full text pdf"}):
            self.links.append(href)


def _doi(meta: dict) -> str:
    from ..fetch import looks_like_doi, normalize_doi

    ids = meta.get("ids")
    for value in (meta.get("doi"), ids.get("doi") if isinstance(ids, dict) else None):
        if isinstance(value, str):
            doi = normalize_doi(value).lower()
            if looks_like_doi(doi) and not any(c.isspace() for c in doi):
                return doi
    return ""


def _arxiv_url_id(url: str) -> str:
    """Keep explicit versions, and reject lookalike hosts or unrelated paths."""
    from ..fetch import looks_like_arxiv

    parts = urlsplit(url)
    if parts.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        return ""
    match = re.fullmatch(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?", unquote(parts.path))
    return match[1] if match and looks_like_arxiv(match[1]) else ""


class _OpenAlexPDF(bytes):
    """Speculative identifiers travel with bytes, never on the shared Paper."""

    def __new__(cls, data: bytes, doi: str, arxiv_id: str):
        result = super().__new__(cls, data)
        result.doi, result.arxiv_id = doi, arxiv_id
        return result


def _apply_openalex_identifiers(data: bytes, paper: Paper, library: Library,
                                verify_reason: str) -> None:
    if not isinstance(data, _OpenAlexPDF) or not verify_reason.startswith("llm_match:"):
        return
    incoming = library.fill_verified_identifiers(paper.key, doi=data.doi, arxiv_id=data.arxiv_id)
    if incoming:
        library.log({"event": "openalex_identifiers_recovered", "key": paper.key,
                     **incoming})


def _try_openalex(paper: Paper) -> Optional[bytes]:
    """Resolve a stored work/DOI through bounded public repository locations.

    See https://help.openalex.org/data/locations/ . OA/content flags and field
    names are hints, never successful downloads. Verification stays in the
    cascade; failed/deleted IDs and access/quota failures are ordinary misses.
    """
    work_id = _openalex_work_id(paper)
    if not (work_id or paper.doi):
        return None
    mailto = os.environ.get("OPENALEX_MAILTO", "research@example.invalid")
    # Preserve the existing DOI route for DOI-bearing papers, even if their
    # stored OpenAlex work ID has since been deleted.
    identity = f"doi:{quote(paper.doi, safe='/')}" if paper.doi else work_id
    budget = _FetchBudget()
    result = budget.get(f"https://api.openalex.org/works/{identity}?mailto={quote(mailto)}",
                        metadata=True)
    if result is None:
        return None
    try:
        meta = json.loads(result[1])
    except (ValueError, UnicodeError):
        return None
    if not isinstance(meta, dict):
        return None

    candidates: list[str] = []
    locations = [meta.get("best_oa_location"), meta.get("primary_location")]
    for key in ("locations", "oa_locations"):
        if isinstance(meta.get(key), list):
            locations.extend(meta[key])
    arxiv_ids: set[str] = set()
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        for key in ("pdf_url", "url_for_pdf", "landing_page_url"):
            url = _http_url(loc.get(key))
            if url:
                arxiv = _arxiv_url_id(url)
                if arxiv:
                    arxiv_ids.add(arxiv)
                variants = [url]
                if url.startswith("http://"):
                    https = "https://" + url[len("http://"):]
                    variants.insert(0, https)
                    log.debug("openalex: HTTPS candidate %s; retain advertised %s", https, url)
                for candidate in variants:
                    if candidate not in candidates and len(candidates) < _MAX_CANDIDATES:
                        candidates.append(candidate)
    doi = _doi(meta)
    if doi and len(candidates) < _MAX_CANDIDATES:
        url = f"https://doi.org/{doi}"
        if url not in candidates:
            candidates.append(url)
    arxiv_id = next(iter(arxiv_ids)) if len(arxiv_ids) == 1 else ""

    for url in candidates:
        result = budget.get(url)
        if result is None:
            continue
        final_url, data = result
        if _is_pdf_bytes(data):
            return _OpenAlexPDF(data, doi, _arxiv_url_id(final_url) or arxiv_id)
        try:
            page = _FileLinks()
            page.feed(data.decode("utf-8", errors="replace"))
            links = list(dict.fromkeys(page.citations + page.links))[:_MAX_LINKS]
            for link in links:
                linked = budget.get(urljoin(final_url, link))
                if linked and _is_pdf_bytes(linked[1]):
                    return _OpenAlexPDF(linked[1], doi, _arxiv_url_id(linked[0]) or arxiv_id)
        except ValueError:
            continue
    return None
