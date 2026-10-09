from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Literal, Optional
from urllib.parse import urlsplit

import requests
from urllib3.exceptions import HTTPError

from ..models import ArxivWithdrawal, Paper, canonicalize_author, normalize_title
from ._shared import TIMEOUT, USER_AGENT, _is_pdf_bytes, log


_ID = re.compile(r"(?P<base>(?:[0-9]{4}\.[0-9]{4,5}|[a-z][a-z.-]*/[0-9]{7}))(?:v[1-9][0-9]*)?")
_MAX_METADATA_BYTES = 256 * 1024
_RECORD_BUDGET = 20
_RECORD_TIMEOUT = 12


def _base_id(identifier: str) -> str:
    identifier = re.sub(r"^arxiv:\s*", "", identifier, flags=re.I)
    match = _ID.fullmatch(identifier)
    return match["base"] if match else ""


def _canonical_arxiv_url_id(raw: str, *, allow_pdf: bool = False) -> str:
    """Accept only a complete canonical URL, not embedded prose or lookalike hosts."""
    if not raw or "\\" in raw or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in raw):
        return ""
    try:
        url = urlsplit(raw)
        if (url.scheme not in {"http", "https"} or url.hostname != "arxiv.org"
                or url.username is not None or url.password is not None
                or url.port not in {None, 443 if url.scheme == "https" else 80}
                or url.query or url.fragment):
            return ""
    except ValueError:
        return ""
    prefix = "/abs/"
    if allow_pdf and url.path.startswith("/pdf/"):
        prefix = "/pdf/"
    if not url.path.startswith(prefix):
        return ""
    identifier = url.path[len(prefix):]
    if prefix == "/pdf/":
        identifier = identifier.removesuffix(".pdf")
    return identifier if _ID.fullmatch(identifier) else ""


def _arxiv_input(paper: Paper) -> str:
    if paper.arxiv_id:
        identifier = paper.arxiv_id.strip()
        return identifier if _base_id(identifier) else ""
    # File URLs also need the withdrawal guard before the earlier stored-file tier.
    return _canonical_arxiv_url_id(paper.url, allow_pdf=True)


@dataclass
class _ArxivRecord:
    requested_id: str
    status: Literal["found", "not_found", "unknown"] = "unknown"
    latest_id: str = ""
    withdrawn: bool = False
    evidence: str = ""
    reason: str = "unrecognized_metadata"
    evidence_url: str = ""
    observed_at: str = ""
    title: str = ""
    authors: tuple[str, ...] = ()


class _ArxivPage(HTMLParser):
    """Collect only the source's identity, history and scoped notice fields."""

    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
             "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.meta = {}
        self.canonical = []
        self.text = {key: [] for key in ("banner", "comments", "history", "heading", "missing")}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            key = attrs.get("name") or attrs.get("property")
            self.meta.setdefault(key, []).append(attrs.get("content", ""))
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical.append(attrs.get("href", ""))
        if tag not in self._VOID:
            self.stack.append((tag, attrs))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        ids = {attrs.get("id") for _, attrs in self.stack}
        classes = {c for _, attrs in self.stack for c in (attrs.get("class") or "").split()}
        tags = {tag for tag, _ in self.stack}
        if "abs" in ids and "error" in classes:
            self.text["banner"].append(data)
        if "abs" in ids and "comments" in classes and "td" in tags:
            self.text["comments"].append(data)
        if "submission-history" in classes:
            self.text["history"].append(data)
        if "content" in ids and "h1" in tags:
            self.text["heading"].append(data)
        if "content" in ids and "p" in tags:
            self.text["missing"].append(data)

    def value(self, key):
        return " ".join("".join(self.text[key]).split())


def _lookup_arxiv_record(identifier: str) -> _ArxivRecord:
    """One bounded latest-page lookup; generic 404s/API omissions prove nothing.

    No redirects or retries. Body reads are capped, checked against a 20s budget,
    and individually subject to the 12s socket timeout. Unknown layouts abstain.
    """
    base = _base_id(identifier)
    record = _ArxivRecord(requested_id=identifier,
                          evidence_url=f"https://arxiv.org/abs/{base}",
                          observed_at=datetime.now(timezone.utc).isoformat())
    if not base:
        record.reason = "invalid_identifier"
        return record
    deadline = time.monotonic() + _RECORD_BUDGET
    try:
        with requests.get(record.evidence_url, timeout=_RECORD_TIMEOUT, stream=True,
                          headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
                          allow_redirects=False) as response:
            record.reason = f"metadata_http_{response.status_code}"
            if response.status_code not in {200, 404} or response.headers.get("Content-Range"):
                return record
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) > _MAX_METADATA_BYTES):
                record.reason = "metadata_size_limit"
                return record
            content = bytearray()
            while time.monotonic() < deadline:
                # read1 returns available bytes, so trickled bodies cannot keep
                # an iter_content chunk assembling indefinitely between checks.
                chunk = response.raw.read1(8192, decode_content=True)
                if time.monotonic() >= deadline:
                    break
                if not chunk:
                    break
                content.extend(chunk)
                if len(content) > _MAX_METADATA_BYTES:
                    record.reason = "metadata_size_limit"
                    return record
            else:
                record.reason = "metadata_time_limit"
                return record
            if time.monotonic() >= deadline:
                record.reason = "metadata_time_limit"
                return record
            if (length is not None and not response.headers.get("Content-Encoding")
                    and len(content) != int(length)):
                record.reason = "incomplete_metadata"
                return record
            html = content.decode("utf-8", errors="strict")
            if not re.search(r"</html>\s*$", html, re.I):
                record.reason = "incomplete_metadata"
                return record
            page = _ArxivPage()
            page.feed(html)
            page.close()
            if response.status_code == 404:
                if (page.value("heading") == f"Article {base} not found"
                        and f"There is no record of an article with identifier '{base}'." in page.value("missing")):
                    record.status = "not_found"
                    record.reason = "base_record_not_found"
                return record
    except (requests.RequestException, HTTPError, OSError, ValueError):
        record.reason = "metadata_transport_or_parse_error"
        return record

    record.reason = "metadata_identity_or_version_unknown"
    if (page.meta.get("citation_arxiv_id") != [base]
            or len(page.canonical) != 1 or _canonical_arxiv_url_id(page.canonical[0]) != base
            or len(page.meta.get("og:url", [])) != 1):
        return record
    latest = _canonical_arxiv_url_id(page.meta["og:url"][0])
    version = re.search(r"v([1-9][0-9]*)$", latest)
    entries = re.findall(r"\[v([1-9][0-9]*)\]([^[]*)", page.value("history"))
    if (not version or _base_id(latest) != base or not entries
            or int(version[1]) != max(int(v) for v, _ in entries)
            or sum(v == version[1] for v, _ in entries) != 1):
        return record
    entry = next(text for v, text in entries if v == version[1])
    banner = page.value("banner")
    withdrawn = bool(re.match(r"This paper has been withdrawn\b", banner))
    if withdrawn != ("(withdrawn)" in entry):
        record.reason = "metadata_withdrawal_ambiguous"
        return record
    record.status = "found"
    record.latest_id = latest
    record.withdrawn = withdrawn
    record.title = next(iter(page.meta.get("citation_title", [])), "")
    record.authors = tuple(page.meta.get("citation_author", []))
    record.reason = page.value("comments") or banner or "current_version_available"
    if withdrawn:
        record.evidence = f"{banner}; [v{version[1]}] {entry.strip()}"
    return record


def _confirmed_arxiv_withdrawal(paper: Paper) -> Optional[ArxivWithdrawal]:
    evidence = paper.arxiv_withdrawal
    # Fill-blanks merges can assign serialized nested models directly. Validate
    # that representation here without changing assignment rules for all Paper fields.
    if isinstance(evidence, dict):
        try:
            evidence = ArxivWithdrawal.model_validate(evidence)
        except ValueError:
            return None
        paper.arxiv_withdrawal = evidence
    if not isinstance(evidence, ArxivWithdrawal):
        return None
    if evidence and _base_id(_arxiv_input(paper)) == _base_id(evidence.latest_id) != "":
        return evidence
    return None


def _remember_withdrawal(paper: Paper, record: _ArxivRecord) -> None:
    paper.arxiv_withdrawal = ArxivWithdrawal(
        requested_id=record.requested_id, latest_id=record.latest_id,
        evidence=record.evidence, reason=record.reason,
        evidence_url=record.evidence_url, observed_at=record.observed_at,
    )


def _prepare_arxiv(paper: Paper) -> Optional[_ArxivRecord]:
    identifier = _arxiv_input(paper)
    if not identifier:
        return None
    record = _lookup_arxiv_record(identifier)
    if record.status == "not_found":
        log.warning("arxiv[%s]: confirmed missing base — clearing arxiv_id=%r (keeping doi=%r)",
                    paper.key, paper.arxiv_id, paper.doi)
        paper.arxiv_id = ""
    elif record.status == "found":
        if not paper.arxiv_id:
            paper.arxiv_id = identifier
        if record.withdrawn:
            _remember_withdrawal(paper, record)
    return record


def _try_arxiv_with_record(paper: Paper, record: Optional[_ArxivRecord]) -> Optional[bytes]:
    if record is None or record.status == "not_found" or record.withdrawn:
        return None
    # Explicit historical input is retained as identity evidence, never fetched.
    # With unknown metadata, the unversioned PDF still requests only current.
    selected = record.latest_id or _base_id(record.requested_id)
    try:
        response = requests.get(f"https://arxiv.org/pdf/{selected}", timeout=TIMEOUT,
                                headers={"User-Agent": USER_AGENT}, allow_redirects=False)
        if (200 <= response.status_code < 300 and response.status_code != 206
                and not response.headers.get("Content-Range") and _is_pdf_bytes(response.content)):
            return response.content
    except requests.RequestException:
        pass
    return None


def _try_arxiv(paper: Paper) -> Optional[bytes]:
    if _confirmed_arxiv_withdrawal(paper):
        return None
    return _try_arxiv_with_record(paper, _prepare_arxiv(paper))


def _arxiv_title_has_identity(paper: Paper, record: Optional[_ArxivRecord] = None) -> bool:
    if paper.arxiv_id:
        return True
    identifier = _arxiv_input(paper)
    if not identifier:
        return False
    # Preserve a missing URL as input evidence without treating it as a
    # positively resolved record. Only the matching per-call lookup permits this.
    return not (record is not None and record.status == "not_found"
                and _base_id(identifier) == _base_id(record.requested_id))


def _try_arxiv_by_title(paper: Paper, *, record: Optional[_ArxivRecord] = None) -> Optional[bytes]:
    """When the paper has no arxiv_id but has a title, search arxiv by
    title to discover an arxiv preprint version. Many papers in
    paywalled journals have arxiv preprints whose ID didn't get captured
    in the original metadata fetch.

    On a confident match, persists the discovered arxiv_id back to the
    paper so future cascade attempts skip the search.
    """
    if _arxiv_title_has_identity(paper, record):
        return None  # already had arxiv_id; _try_arxiv would have used it
    title = (paper.title or "").strip()
    if len(title) < 20:  # too short to disambiguate reliably
        return None

    from ..sources.arxiv import search_arxiv
    try:
        # search_arxiv internally prefixes the query with `all:`, which
        # cross-field-searches title + abstract + comments. Don't add an
        # extra `ti:` prefix — `all:ti:"..."` is invalid syntax and
        # arxiv returns 0 hits for it. Just pass the title text; the
        # title-overlap match below filters non-title matches.
        # Strip troublesome punctuation that could confuse arxiv parser.
        clean = re.sub(r'["\\?<>]', '', title)
        # Cap query length — arxiv's URL-encoded query can hit length
        # limits with very long titles. 100 chars is plenty for ranking.
        clean = clean[:100]
        results = search_arxiv(clean, max_results=5)
    except Exception:
        return None

    if not results:
        return None

    # Find best match by title overlap
    paper_title_norm = re.sub(r'[^a-z0-9]+', ' ', title.lower()).strip()
    best = None
    for r in results:
        cand_title = (r.get('title') or '').strip()
        cand_norm = re.sub(r'[^a-z0-9]+', ' ', cand_title.lower()).strip()
        # Strict-enough overlap:
        # - normalized titles must share their first 30+ chars exactly, OR
        # - one is a prefix/suffix of the other after normalization
        first_chars = min(50, len(paper_title_norm), len(cand_norm))
        if first_chars >= 30 and paper_title_norm[:first_chars] == cand_norm[:first_chars]:
            best = r
            break
        if (paper_title_norm in cand_norm) or (cand_norm in paper_title_norm):
            if abs(len(cand_norm) - len(paper_title_norm)) < 30:
                best = r
                break

    if not best:
        return None

    discovered_arxiv_id = best.get('arxiv_id', '')
    if not discovered_arxiv_id:
        return None

    # A loose title candidate must not poison the original on a failed fetch.
    candidate = paper.model_copy(deep=True)
    candidate.arxiv_id = discovered_arxiv_id
    candidate.arxiv_withdrawal = None
    record = _prepare_arxiv(candidate)
    if record and record.withdrawn:
        # No PDF is available for the usual identity gate. Require exact title
        # and a full canonical author name on the confirmed source page before
        # attributing its withdrawal to this paper; otherwise abstain.
        names = {normalize_title(canonicalize_author(a)) for a in paper.authors}
        matched_author = any(len(name) >= 5 and name in names for name in
                             (normalize_title(canonicalize_author(a)) for a in record.authors))
        if normalize_title(record.title) == normalize_title(paper.title) and matched_author:
            paper.arxiv_id = discovered_arxiv_id
            paper.arxiv_withdrawal = candidate.arxiv_withdrawal
        return None
    data = _try_arxiv_with_record(candidate, record)
    if data:
        paper.arxiv_id = discovered_arxiv_id
    return data
