"""Inspire-HEP API search tool.

Inspire-HEP (CERN-hosted) covers high-energy physics, particle physics,
cosmic ray physics, and adjacent astrophysics. No auth, generous rate limits.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any
from urllib.parse import urlsplit

import requests

API_URL = "https://inspirehep.net/api/literature"
FIELDS = ("titles,authors,publication_info,preprint_date,arxiv_eprints,dois,"
          "abstracts,citation_count,control_number,documents.url")
MAX_RETRIES = 5
RETRY_DELAY = 5.0


def inspire_record_id(value: Any) -> str:
    """Canonical literature recid; never a DOI, arXiv or Semantic Scholar ID."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return ""
    value = str(value).strip()
    if not re.fullmatch(r"[0-9]+", value) or int(value) <= 0:
        return ""
    return str(int(value))


def _document_urls(documents: Any) -> list[str]:
    """Keep only absolute HTTP(S) links, never embedded attachments/full text."""
    urls: list[str] = []
    for doc in documents if isinstance(documents, list) else []:
        url = doc.get("url") if isinstance(doc, dict) else doc
        if not isinstance(url, str):
            continue
        if (not url or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url)
                or "\\" in url or re.search(r"%(?![0-9a-fA-F]{2})", url)):
            continue
        try:
            parsed = urlsplit(url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None):
                continue
            if parsed.port == 0:
                continue
            requests.PreparedRequest().prepare_url(url, None)
        except (ValueError, requests.RequestException):
            continue
        if url not in urls:
            urls.append(url)
    return urls


def retained_inspire_fields(data: dict) -> dict:
    """Validate the retained pair; links without a source recid are unusable."""
    recid = inspire_record_id(data.get("inspire_record_id"))
    return {"inspire_record_id": recid,
            "inspire_document_urls": _document_urls(data.get("inspire_document_urls"))
            if recid else []}


def _record_fields(hit: dict) -> dict:
    meta = hit.get("metadata") or {}
    values = [v for v in (hit.get("id"), meta.get("control_number")) if v is not None]
    ids = [inspire_record_id(v) for v in values]
    recid = ids[0] if ids and all(ids) and len(set(ids)) == 1 else ""
    return {"inspire_record_id": recid,
            "inspire_document_urls": _document_urls(meta.get("documents")) if recid else []}


def _exact_title(title: str) -> str:
    return " ".join(unicodedata.normalize("NFC", title or "").casefold().split())


def _author_signatures(authors: Any) -> dict[str, set[tuple[str, ...]]]:
    signatures: dict[str, set[tuple[str, ...]]] = {}
    if not isinstance(authors, list):
        return signatures
    for name in authors:
        if not isinstance(name, str) or not name.strip():
            return {}
        name = unicodedata.normalize("NFC", name).casefold().strip()
        if "," in name:
            family, given = name.split(",", 1)
        else:
            parts = name.split()
            family, given = parts[-1], " ".join(parts[:-1])
        family = "".join(c for c in family if c.isalnum())
        if len(family) < 2 or family in {"unknown", "al"}:
            return {}
        given_tokens = tuple(re.findall(r"[^\W\d_]+", given))
        signatures.setdefault(family, set()).add(given_tokens)
    return signatures


def _given_names_compatible(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    # A missing middle name is unknown; a supplied full token must agree.
    # Initials corroborate full names, but John and Jane cannot corroborate
    # each other merely because both start with J.
    return all(a == b or (len(a) == 1 and b.startswith(a))
               or (len(b) == 1 and a.startswith(b)) for a, b in zip(left, right))


def _author_group_matches(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> bool:
    """Require a distinct compatible partner for each author sharing a surname."""
    if len(left) != len(right):
        return False
    a_names, b_names = sorted(left), sorted(right)
    assigned: dict[int, int] = {}

    def match(a: int, visited: set[int]) -> bool:
        for b, name in enumerate(b_names):
            if b in visited or not _given_names_compatible(a_names[a], name):
                continue
            visited.add(b)
            if b not in assigned or match(assigned[b], visited):
                assigned[b] = a
                return True
        return False

    return all(match(a, set()) for a in range(len(a_names)))


def inspire_identity_matches(reference: dict, candidate: dict) -> bool:
    """Exact full title, corroborated authors, and no known bibliographic conflict.

    Deliberately stricter than the search fold's loose title dedupe. Unknown
    years remain unknown; initials may corroborate full given names.
    """
    title = _exact_title(reference.get("title") or "")
    if not title or title != _exact_title(candidate.get("title") or ""):
        return False
    left = _author_signatures(reference.get("authors"))
    right = _author_signatures(candidate.get("authors"))
    if not left or left.keys() != right.keys():
        return False
    for family in left:
        if not _author_group_matches(left[family], right[family]):
            return False
    ly, ry = reference.get("year"), candidate.get("year")
    if ly and ry and str(ly) != str(ry):
        return False
    for field in ("doi", "arxiv_id"):
        a, b = reference.get(field) or "", candidate.get(field) or ""
        if field == "arxiv_id":
            a, b = re.sub(r"v\d+$", "", a), re.sub(r"v\d+$", "", b)
        if a and b and a.casefold() != b.casefold():
            return False
    return True


def merge_inspire_fields(target: dict, incoming: dict) -> dict:
    """Return coherent source fills for a verified twin; never mix recids."""
    pair = retained_inspire_fields(incoming)
    recid = pair["inspire_record_id"]
    old_id = target.get("inspire_record_id") or ""
    # Search representatives may omit a donor's year/identifiers. Keep that
    # transient evidence through the final upsert, rather than verifying only
    # the representative's incomplete bibliography. Never persist this tag.
    evidence = [incoming, *(incoming.get("_inspire_metadata") or [])]
    old_evidence = [target, *(target.get("_inspire_metadata") or [])]
    if (not recid or (old_id and inspire_record_id(old_id) != recid)
            or not all(inspire_identity_matches(a, b) for a in old_evidence for b in evidence)):
        return {}
    old_urls = _document_urls(target.get("inspire_document_urls")) if old_id else []
    pair["inspire_document_urls"] = list(dict.fromkeys(old_urls + pair["inspire_document_urls"]))
    return pair  # a later donor may add bibliographic evidence even if links are unchanged


def inspire_title_eligible(data: dict) -> bool:
    """Historical lookup is limited to explicitly INSPIRE-sourced cards."""
    sources = re.split(r"[\s,;+|]+", (data.get("source") or "").casefold())
    return ("inspire" in sources and len((data.get("title") or "").strip()) >= 20
            and bool(_author_signatures(data.get("authors"))))


def lookup_inspire_record(reference: dict, *, recid: str = "", timeout: int = 30,
                          headers: dict | None = None) -> dict | None:
    """One bounded GET: exact record, or complete unique bibliographic title match.

    Empty documents still return the validated identity; an absent link is
    an unknown source-route miss, never a claim that full text does not exist.
    No retries, arbitrary first hit, or DOI/arXiv fabrication.
    """
    if recid:
        recid = inspire_record_id(recid)
        if not recid:
            return None
        url, params = f"{API_URL}/{recid}", None
    else:
        if not inspire_title_eligible(reference):
            return None
        title = reference["title"].strip().replace("\\", "\\\\").replace('"', '\\"')
        url, params = API_URL, {"q": f'title:"{title}"', "size": 3, "fields": FIELDS}
    try:
        response = requests.get(url, params=params, timeout=timeout, headers=headers)
        response.raise_for_status()
        data = response.json()
        if recid:
            hits = [data]
        else:
            envelope = data.get("hits") or {}
            hits = envelope.get("hits") or []
            total = envelope.get("total")
            if isinstance(total, dict):
                if total.get("relation") != "eq":
                    return None
                total = total.get("value")
            if (isinstance(total, bool) or not isinstance(total, int)
                    or total < 0 or total > 3 or total != len(hits)):
                return None  # truncated or unknown total cannot prove uniqueness
        candidates = []
        for hit in hits:
            candidate = _paper_from_hit(hit)
            candidate_id = candidate["inspire_record_id"]
            if recid and candidate_id != recid:
                return None
            if (_exact_title(reference.get("title")) == _exact_title(candidate.get("title"))
                    and not _author_signatures(candidate.get("authors"))):
                return None  # a same-title incomplete twin cannot be ruled out
            if inspire_identity_matches(reference, candidate):
                if not candidate_id:
                    return None  # incomplete identity must not hide an ambiguous twin
                candidates.append(candidate)
        return candidates[0] if len(candidates) == 1 else None
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return None


def search_inspire(query: str, max_results: int = 30) -> list[dict[str, Any]]:
    """Search Inspire-HEP and return structured paper metadata."""
    if not query:
        return []

    # Bare phrase-AND query: pass the raw term string. INSPIRE's default
    # operator AND-joins the tokens of a bare (unquoted, unfielded) phrase, so
    # ``q="magnetic reconnection"`` matches records containing BOTH tokens. We
    # deliberately do NOT quote (would force an exact-phrase match, far too
    # narrow for thin geospace coverage) nor wrap in a field qualifier.
    params = {
        "q": query,
        "size": max_results,
        "fields": FIELDS,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(RETRY_DELAY)
                continue
            resp.raise_for_status()
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.SSLError):
            time.sleep(RETRY_DELAY)
            continue
        except requests.exceptions.HTTPError:
            return []
    else:
        return []

    try:
        data = resp.json()
    except ValueError:
        return []

    papers: list[dict[str, Any]] = []
    for hit in (data.get("hits") or {}).get("hits", []) or []:
        papers.append(_paper_from_hit(hit))

    return papers


def _paper_from_hit(hit: dict) -> dict:
    meta = hit.get("metadata") or {}

    titles = meta.get("titles") or []
    title = (titles[0].get("title") if titles else "") or ""

    authors = []
    for a in (meta.get("authors") or []):
        name = (a.get("full_name") or "").strip()
        if name:
            authors.append(name)

    pub_info = meta.get("publication_info") or []
    year: Any = None
    venue = ""
    if pub_info:
        first = pub_info[0] or {}
        year = first.get("year")
        venue = (first.get("journal_title") or "") or ""
    if not year:
        preprint_date = meta.get("preprint_date") or ""
        if isinstance(preprint_date, str) and len(preprint_date) >= 4:
            prefix = preprint_date[:4]
            if prefix.isdigit():
                year = int(prefix)

    arxiv_eprints = meta.get("arxiv_eprints") or []
    arxiv_id = ""
    if arxiv_eprints:
        arxiv_id = (arxiv_eprints[0].get("value") or "") or ""

    dois = meta.get("dois") or []
    doi = ""
    if dois:
        doi = (dois[0].get("value") or "") or ""

    abstracts = meta.get("abstracts") or []
    abstract = ""
    if abstracts:
        abstract = (abstracts[0].get("value") or "") or ""

    citation_count = int(meta.get("citation_count") or 0)

    url = ""
    if arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
    elif doi:
        url = f"https://doi.org/{doi}"

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "year": year,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "url": url,
        "venue": venue,
        "citation_count": citation_count,
        **_record_fields(hit),
    }
