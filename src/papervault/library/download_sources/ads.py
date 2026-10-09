"""ADS retrieval and conservative evidence about the indexed publication."""
from __future__ import annotations

import hashlib
import io
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from html import unescape
from typing import Optional
from urllib.parse import quote, unquote, urlsplit

import requests

from ..models import ADSAvailability, ADSDocumentEvidence, Paper
from ._shared import TIMEOUT, USER_AGENT, _is_pdf_bytes

_API = "https://api.adsabs.harvard.edu/v1/search/query"
_FIELDS = "bibcode,title,author,doi,identifier,doctype,pub,esources"
_BIBCODE = re.compile(r"[0-9]{4}[A-Za-z0-9&.]{14}[A-Za-z.]")
_ADS_HOSTS = {"ui.adsabs.harvard.edu", "adsabs.harvard.edu"}
_PDF_SOURCES = {"EPRINT_PDF", "PUB_PDF", "ADS_PDF", "AUTHOR_PDF"}
_MAX_PDF_BYTES = 32 * 1024 * 1024
_MAX_PAGES = 300


def _ads_bibcode(paper: Paper) -> Optional[str]:
    """Qualify a stored ADS identity without making a request.

    paper_id is shared with Semantic Scholar, so only ADS provenance qualifies
    a standalone ID. Decode a recognized URL's bibcode exactly once.
    """
    raw = paper.url or ""
    if any(ord(c) < 32 or ord(c) == 127 for c in raw):
        return None
    source_ads = paper.source.strip().lower() == "ads"
    code = None
    if raw.strip():
        try:
            url = urlsplit(raw.strip())
            if (url.scheme not in {"https", "http"} or url.hostname not in _ADS_HOSTS
                    or url.username is not None or url.password is not None
                    or url.port not in {None, 443 if url.scheme == "https" else 80}):
                return None
            match = re.fullmatch(r"/abs/([^/]+)(?:/abstract)?/?", url.path)
            if not match:
                return None
            code = unquote(match[1])
        except ValueError:
            return None
        if not _BIBCODE.fullmatch(code):
            return None
    if source_ads and paper.paper_id:
        if not _BIBCODE.fullmatch(paper.paper_id):
            return None
        if code and code != paper.paper_id:
            return None
        code = paper.paper_id
    return code


def _strings(value) -> list[str]:
    return [x for x in value if isinstance(x, str)] if isinstance(value, list) else []


def _arxiv(value: str) -> str:
    return re.sub(r"v\d+$", "", value.removeprefix("arXiv:").strip()).lower()


def _fetch_ads_metadata(paper: Paper) -> tuple[Optional[dict], str]:
    """Fetch a single exact identity; prerequisites/API errors are unknowns."""
    token = os.environ.get("ADS_API_TOKEN", "").strip()
    if not token:
        return None, "missing_credentials"
    code = _ads_bibcode(paper)
    try:
        ads_url = urlsplit(paper.url).hostname in _ADS_HOSTS
    except ValueError:
        ads_url = False
    if not code and (ads_url or (paper.source.strip().lower() == "ads" and paper.paper_id)):
        return None, "invalid_ads_identity"
    if paper.doi:
        query = f'doi:"{paper.doi}"'
    elif paper.arxiv_id:
        query = f'identifier:"{_arxiv(paper.arxiv_id)}"'
    elif code:
        query = f'bibcode:"{code}"'
    else:
        return None, "missing_identifier"
    try:
        resp = requests.get(_API, params={"q": query, "fl": _FIELDS, "rows": 1},
                            timeout=TIMEOUT, headers={"User-Agent": USER_AGENT,
                                                     "Authorization": f"Bearer {token}"})
        if resp.status_code != 200:
            return None, f"api_http_{resp.status_code}"
        payload = resp.json()
        docs = payload["response"]["docs"]
        if not isinstance(docs, list):
            return None, "malformed_metadata"
        if not docs:
            return None, "metadata_not_found"
        doc = docs[0]
        if not isinstance(doc, dict):
            return None, "malformed_metadata"
    except (ValueError, KeyError, TypeError):
        return None, "malformed_metadata"
    except requests.RequestException:
        return None, "api_transport_error"
    returned = doc.get("bibcode")
    identifiers = _strings(doc.get("identifier"))
    if not isinstance(returned, str) or not _BIBCODE.fullmatch(returned):
        return None, "metadata_identity_mismatch"
    # A quoted search can still return a different record. Require agreement
    # with each supplied qualifying identifier, not merely the first search hit.
    if code and code != returned:
        return None, "metadata_identity_mismatch"
    if paper.doi and paper.doi.lower() not in {x.lower() for x in _strings(doc.get("doi"))}:
        return None, "metadata_identity_mismatch"
    if paper.arxiv_id and _arxiv(paper.arxiv_id) not in {_arxiv(x) for x in identifiers}:
        return None, "metadata_identity_mismatch"
    return doc, "metadata_matched"


def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFKD", unescape(text)).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _has_abstract_prose(text: str) -> bool:
    """Require positive prose evidence; lists of titles cannot supply a body.

    These explicit English publication forms are deliberately conservative.
    Unrecognized language/layout or a very short entry remains unknown.
    """
    normalized = _normalized(text)
    sentences = re.findall(r"[^.!?]+[.!?](?=\s|$)", text)
    return (len(normalized.split()) >= 40
            and sum(len(_normalized(s).split()) >= 10 for s in sentences) >= 2
            and bool(re.search(r"\b(?:we|our|is|are|was|were|will|can|could|has|have)\b", normalized)))


def _inspect_ads_document(data: bytes, paper: Paper) -> ADSDocumentEvidence:
    """Recognize explicit publication forms, abstaining on ambiguous documents.

    A short PDF or the word 'Abstract' is insufficient. Meeting confirmation
    needs ADS abstract metadata (checked by the classifier), a matching title
    and author, and explicit venue/submission evidence. Anthologies need their
    cover label plus the target entry, which can be far beyond the first pages.
    """
    evidence = ADSDocumentEvidence(outcome="pdf", size_bytes=len(data),
                                   sha256=hashlib.sha256(data).hexdigest())
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        evidence.pages = len(reader.pages)
        if len(reader.pages) > _MAX_PAGES:
            evidence.reason = "document_page_limit"
            return evidence
        pages = [(p.extract_text() or "")[:20000] for p in reader.pages]
    except Exception:
        evidence.outcome = "invalid_pdf"
        evidence.reason = "pdf_parse_error"
        return evidence
    cover = _normalized("\n".join(pages[:2]))
    anthology = bool(re.search(r"(?im)^\s*(?:abstracts?\s+book|book\s+of\s+abstracts)\s*$",
                              "\n".join(pages[:2])))
    if anthology:
        evidence.reason = "abstract_book_entry_unconfirmed"
    title = _normalized(paper.title)
    surnames = [_normalized(a.split(",")[0] if "," in a else a.split()[-1])
                for a in paper.authors if a.strip()]
    if len(title) < 15 or not surnames or len(surnames[0]) < 3:
        if not anthology:
            evidence.reason = "insufficient_identity"
        return evidence
    # PDF kerning can insert spaces inside words (e.g. ISW AT). Compare the
    # entire long title with whitespace removed, retaining author agreement.
    compact_title = title.replace(" ", "")
    title_pattern = re.compile(r"\s*".join(re.escape(c) for c in compact_title))
    target = None
    substantive = False
    for i, page in enumerate(pages):
        # Keep line boundaries: a complete heading is stronger identity
        # evidence than a title embedded in prose or a reference citation.
        lines = page.splitlines()
        normalized = "\n".join(_normalized(line) for line in lines)
        if re.search(r"(?im)^\s*(?:table\s+of\s+contents|contents)\b", normalized):
            continue
        references = re.search(r"(?im)^\s*(?:references|bibliography)\b", normalized)
        for match in title_pattern.finditer(normalized):
            line_start = normalized.rfind("\n", 0, match.start()) + 1
            line_end = normalized.find("\n", match.end())
            if line_end < 0:
                line_end = len(normalized)
            if (normalized[line_start:match.start()].strip()
                    or normalized[match.end():line_end].strip()
                    or (references and references.start() < match.start())):
                continue
            nearby = normalized[max(0, match.start() - 400):match.end() + 500]
            if not re.search(r"(?<!\w)" + re.escape(surnames[0]) + r"(?!\w)", nearby):
                continue
            # A page reference ends a listing's entry. Other titles farther
            # down a contents page must not supply the requested work's body.
            last_heading_line = normalized[:match.end()].count("\n")
            tail = "\n".join(lines[last_heading_line + 1:])
            page_reference = re.search(r"(?m)^(?:\s*\d{1,3}|.*\.{2,}\s*\d{1,3})\s*$", tail)
            if page_reference:
                tail = tail[:page_reference.start()]
            substantive = _has_abstract_prose(tail)
            if anthology and not substantive:
                continue
            target = i
            break
        if target is not None:
            break
    if target is None:
        if not anthology:
            evidence.reason = "document_identity_unconfirmed"
        return evidence
    evidence.identity_match = True
    evidence.target_page = target + 1
    entry = pages[target]
    # Explicit abstract-book scope applies to the indexed contribution, never
    # to an apparent page count for the whole linked book.
    entry_context = "\n".join(pages[target:target + 2])
    if anthology and len(entry.split()) < 1500 and "introduction" not in _sections(entry_context):
        evidence.document_kind = "abstract_anthology"
        evidence.reason = "abstract_book_cover_and_matching_entry"
    elif substantive and len(pages) <= 2 and (
        ("geophysical research abstracts" in cover and "egu general assembly" in cover)
        or ("cospar" in cover and "poster only" in cover)
    ) and not _sections("\n".join(pages)):
        evidence.document_kind = "meeting_abstract"
        evidence.reason = "meeting_publication_and_matching_entry"
    elif not anthology:
        sections = _sections("\n".join(pages))
        if "introduction" in sections and len(sections) >= 2:
            evidence.document_kind = "full_text"
            evidence.reason = "matching_work_with_body_sections"
    return evidence


def _sections(text: str) -> set[str]:
    return {x.lower() for x in re.findall(
        r"(?im)^\s*(?:\d+[.\s]+)?(introduction|methodology|methods|results|conclusions)\b", text)}


def _classify_ads_availability(metadata: dict, document_observations: list[ADSDocumentEvidence],
                               existing_assets=()) -> ADSAvailability:
    """Pure classification; neither no links nor failed requests prove absence."""
    kind = metadata.get("doctype")
    result = ADSAvailability(publication_kind=kind if isinstance(kind, str) else "unknown",
                             bibcode=metadata.get("bibcode", ""),
                             esources=_strings(metadata.get("esources")),
                             identifiers=_strings(metadata.get("identifier")),
                             documents=document_observations, existing_assets=list(existing_assets),
                             reason="document_availability_unconfirmed")
    matched = [d for d in document_observations if d.outcome == "pdf" and d.identity_match
               and not d.identity_rejected]
    if any(d.document_kind == "full_text" for d in matched):
        result.availability = "retrievable"
        result.reason = "matching_full_text_document"
    elif result.publication_kind in {"abstract", "inproceedings"} and any(
            d.document_kind == "abstract_anthology" for d in matched):
        result.availability = "confirmed_abstract_only"
        result.reason = "published_abstract_in_anthology"
    elif result.publication_kind == "abstract" and any(
            d.document_kind == "meeting_abstract" for d in matched):
        result.availability = "confirmed_abstract_only"
        result.reason = "published_meeting_abstract"
    elif any(d.identity_rejected for d in document_observations):
        result.reason = "document_identity_mismatch"
    elif any(d.outcome == "blocked" for d in document_observations):
        result.availability = "blocked"
        result.reason = "document_access_blocked"
    elif not result.esources:
        result.reason = "no_ads_link_document_unconfirmed"
    elif not any(s in _PDF_SOURCES for s in result.esources):
        result.reason = "html_or_other_source_unconfirmed"
    return result


def _fetch_ads_document(url: str, paper: Paper) -> tuple[Optional[bytes], ADSDocumentEvidence]:
    observation = ADSDocumentEvidence(url=url)
    started = time.monotonic()
    try:
        with requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT},
                          allow_redirects=True, stream=True) as r:
            observation.status_code = r.status_code
            if r.status_code == 206 or r.headers.get("Content-Range"):
                observation.outcome = "incomplete"
                observation.reason = "partial_http_response"
                return None, observation
            if not 200 <= r.status_code < 300:
                observation.outcome = ("blocked" if r.status_code in {401, 403, 429}
                                       else "stale" if r.status_code in {404, 410} else "http_error")
                return None, observation
            chunks = []
            size = 0
            for chunk in r.iter_content(65536):
                chunks.append(chunk)
                size += len(chunk)
                if size > _MAX_PDF_BYTES or time.monotonic() - started > TIMEOUT:
                    observation.outcome = "incomplete"
                    observation.reason = "document_limit"
                    return None, observation
                if size >= 5 and not _is_pdf_bytes(b"".join(chunks)[:5]):
                    observation.outcome = "html"
                    observation.reason = "non_pdf_response"
                    return None, observation
            data = b"".join(chunks)
            length = r.headers.get("Content-Length")
            if length and not r.headers.get("Content-Encoding") and len(data) != int(length):
                observation.outcome = "incomplete"
                observation.reason = "content_length_mismatch"
                return None, observation
            if not _is_pdf_bytes(data):
                observation.outcome = "invalid_pdf"
                return None, observation
            inspected = _inspect_ads_document(data, paper)
            inspected.url = url
            inspected.status_code = r.status_code
            return data, inspected
    except (requests.RequestException, ValueError):
        observation.outcome = "transport_error"
        return None, observation


def _try_ads(paper: Paper) -> Optional[bytes]:
    """Use DOI/arXiv or a stored ADS bibcode in the existing aggregator tier.

    Return bytes for the usual coordinator identity gate; completed, confirmed
    abstract documents instead leave typed evidence for indexed-item retirement.
    Never write files or reset routing here. One legacy fallback, no retries.
    """
    checked_at = datetime.now(timezone.utc).isoformat()
    metadata, reason = _fetch_ads_metadata(paper)
    if metadata is None:
        paper.ads_availability = ADSAvailability(bibcode=_ads_bibcode(paper) or "", reason=reason,
                                                checked_at=checked_at)
        return None
    code = quote(metadata["bibcode"], safe="")
    urls = [f"https://ui.adsabs.harvard.edu/link_gateway/{code}/{source}"
            for source in dict.fromkeys(_strings(metadata.get("esources"))) if source in _PDF_SOURCES]
    urls.append(f"https://articles.adsabs.harvard.edu/pdf/{code}")
    observations = []
    candidate = None
    for url in urls:
        data, observation = _fetch_ads_document(url, paper)
        observations.append(observation)
        if (data and observation.document_kind not in {"meeting_abstract", "abstract_anthology"}
                and observation.reason != "abstract_book_entry_unconfirmed"):
            candidate = data
            break
        if observation.status_code == 429:
            break
    result = _classify_ads_availability(metadata, observations)
    result.checked_at = checked_at
    paper.ads_availability = result
    return candidate


def _record_ads_verification(paper: Paper, data: bytes, ok: bool) -> None:
    """Apply the existing coordinator identity verdict to matching ADS bytes."""
    result = paper.ads_availability
    digest = hashlib.sha256(data).hexdigest()
    rejected = False
    for document in result.documents:
        if document.sha256 == digest and not ok:
            document.identity_match = False
            document.identity_rejected = True
            rejected = True
    if rejected:
        result.availability = "unknown"
        result.reason = "document_identity_mismatch"


def _log_ads_availability(paper: Paper, library) -> None:
    result = paper.ads_availability
    if result.checked_at:
        result.existing_assets = [kind for kind in ("pdf", "md", "txt")
                                  if (library.has_pdf(paper.key) if kind == "pdf"
                                      else library.has_extract(paper.key, kind))]
        library.log({"event": "ads_availability", "key": paper.key,
                     "source": "ads", "availability": result.model_dump()})


def _settle_ads_abstract(paper: Paper, library) -> bool:
    """Stop this indexed item's hunt without deleting or re-gating any asset.

    Normal queue callers persist the Paper via Library.save. The reason is also
    immediately durable in the existing manifest. No terminal state is reset.
    """
    if paper.ads_availability.availability != "confirmed_abstract_only":
        return False
    from ..models import DOWNLOAD_STATUS_METADATA_ONLY, DOWNLOAD_STATUS_OK, DOWNLOAD_STATUS_PENDING
    has_full_text = library.has_pdf(paper.key) or library.has_extract(paper.key, "md")
    if not has_full_text:
        paper.download_status = DOWNLOAD_STATUS_METADATA_ONLY
    elif paper.download_status == DOWNLOAD_STATUS_PENDING:
        paper.download_status = DOWNLOAD_STATUS_OK
    if library.has_extract(paper.key, "md") and library.md_source(paper.key) == "firecrawl":
        paper.firecrawl_pdf_hunt_exhausted = True
    library.log({"event": "download_abstract_only", "key": paper.key,
                 "source": "ads", "bibcode": paper.ads_availability.bibcode,
                 "reason": paper.ads_availability.reason,
                 "scope": "indexed_ads_item", "assets_preserved": True})
    return True


def _ads_availability_report(library) -> dict:
    """Report stored evidence only; no requests, mutations or terminal resets."""
    counts = dict.fromkeys(("confirmed_abstract_only", "retrievable", "blocked", "unknown"), 0)
    records = []
    for paper in library.all_papers():
        result = paper.ads_availability
        if not (result.bibcode or _ads_bibcode(paper)):
            continue
        counts[result.availability] += 1
        records.append({"key": paper.key, **result.model_dump(),
                        "bibcode": result.bibcode or _ads_bibcode(paper)})
    return {"counts": counts, "records": records}
