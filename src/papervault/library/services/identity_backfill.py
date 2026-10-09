"""Bounded URL identity recovery and a recorded pass through the ordinary cascade."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import requests

from .. import fetch
from ..download_sources import ads, arxiv, core, openalex, publisher
from ..models import IdentityRecovery, Paper
from ..sources.inspire import inspire_record_id, lookup_inspire_record
from ..store import Library

MAX_CAP = 50
_DOI_HOSTS = {"doi.org", "dx.doi.org", "onlinelibrary.wiley.com", "link.springer.com",
              "iopscience.iop.org", "journals.aps.org", "academic.oup.com",
              "pubs.acs.org", "journals.sagepub.com", "www.tandfonline.com"}


def read_identity_records(root: Path) -> list[Paper]:
    """Use the store's last-good snapshot rules without its writable constructor."""
    data = (Library._read_index_file(root / "index.json")
            or Library._read_index_file(root / "index.json.bak"))
    if data is None or not isinstance(data.get("papers"), dict):
        raise ValueError(f"no readable library index at {root}")
    return [Paper(**raw) for raw in data["papers"].values()]


def _url_identity(url: str) -> tuple[str, str]:
    valid = core._http_url(url)
    if not valid:
        return "", ""
    identifier = arxiv._canonical_arxiv_url_id(valid, allow_pdf=True)
    if identifier:
        return "", arxiv._base_id(identifier)
    parts = urlsplit(valid)
    if parts.hostname not in _DOI_HOSTS:
        return "", ""
    match = re.search(r"/(10\.[0-9]{4,9}/.+)$", unquote(parts.path))
    value = match[1] if match else ""
    if parts.hostname not in {"doi.org", "dx.doi.org"}:
        value = re.sub(r"/(?:pdf|full|abstract|epdf)$", "", value, flags=re.I)
    return core._doi(value), ""


def _bibliography_matches(paper: Paper, metadata: dict, *, require_authors=False) -> bool:
    title = metadata.get("title")
    if (not isinstance(title, str) or len(fetch.normalize_title(paper.title)) < 25
            or not fetch._safe_title_match(paper.title, title, stub_has_year=paper.year is not None)):
        return False
    authors = metadata.get("authors") or []
    if paper.authors and (authors or require_authors):
        if not fetch._safe_author_match(paper.authors, authors):
            return False
    year = metadata.get("year")
    return year is None or fetch._safe_year_match(paper.year, year)


def _verify(paper: Paper, doi: str, arxiv_id: str, route: str) -> IdentityRecovery | None:
    doi, arxiv_id = core._doi(doi), core._arxiv(arxiv_id)
    if doi:
        metadata = fetch._fetch_crossref_by_doi(doi)
        if (not metadata or core._doi(metadata.get("doi")) != doi
                or not _bibliography_matches(paper, metadata, require_authors=True)):
            return None
        title = metadata["title"]
    elif arxiv_id:
        record = arxiv._lookup_arxiv_record(arxiv_id)
        if (record.status != "found" or not _bibliography_matches(paper, {
                "title": record.title, "authors": list(record.authors)}, require_authors=True)):
            return None
        title = record.title
    else:
        return None
    return IdentityRecovery(doi=doi, arxiv_id="" if doi else arxiv_id, route=route,
                            source_url=paper.url, verified_title=title,
                            observed_at=datetime.now(timezone.utc).isoformat())


def _single(values, normalize) -> str:
    if not isinstance(values, list):
        return ""
    clean = {normalize(value) for value in values} - {""}
    return next(iter(clean)) if len(clean) == 1 else ""


def _source_identity(paper: Paper) -> tuple[str, str, str] | None:
    """One exact source lookup; source title must agree before registry verification."""
    if ads._ads_bibcode(paper):
        metadata, _ = ads._fetch_ads_metadata(paper)
        if metadata and _bibliography_matches(paper, {
                "title": " ".join(metadata.get("title") or []), "authors": metadata.get("author") or []}):
            return (_single(metadata.get("doi"), core._doi),
                    _single(metadata.get("identifier"), core._arxiv), "ads_api")
        return None
    locator = core._core_locator(paper)
    parts = urlsplit(paper.url)
    if parts.hostname in {"core.ac.uk", "www.core.ac.uk", "api.core.ac.uk"}:
        match = re.fullmatch(r"/(?:v3/)?works/([0-9]+)/?", parts.path)
        if match and core._decimal(match[1]):
            work_id = core._decimal(match[1])
            if locator and locator != ("works", work_id):
                return None
            locator = ("works", work_id)
    if locator:
        key = os.environ.get("CORE_API_KEY", "").strip()
        if not key:
            return None
        metadata = core._api_get("/".join(locator), key)
        if not metadata or core._decimal(metadata.get("id")) != locator[1]:
            return None
        ids = core._corroborate(paper, metadata) if metadata else None
        return (*ids, "core_api") if ids else None
    work = openalex._openalex_work_id(paper)
    if work:
        mailto = os.environ.get("OPENALEX_MAILTO", "research@example.invalid")
        result = openalex._FetchBudget().get(
            f"https://api.openalex.org/works/{work}?mailto={quote(mailto)}", metadata=True)
        if result is None:
            return None
        metadata = json.loads(result[1])
        if not isinstance(metadata, dict) or str(metadata.get("id", "")).rstrip("/").lower() != f"https://openalex.org/{work}".lower():
            return None
        authors = [(item.get("author") or {}).get("display_name", "")
                   for item in metadata.get("authorships") or [] if isinstance(item, dict)]
        if not _bibliography_matches(paper, {"title": metadata.get("title") or metadata.get("display_name"),
                                            "authors": authors, "year": metadata.get("publication_year")}):
            return None
        return openalex._doi(metadata), "", "openalex_api"
    recid = inspire_record_id(paper.inspire_record_id)
    if parts.hostname == "inspirehep.net":
        match = re.fullmatch(r"/literature/([0-9]+)/?", parts.path)
        url_recid = inspire_record_id(match[1]) if match else ""
        if recid and url_recid and recid != url_recid:
            return None
        recid = recid or url_recid
    if recid:
        metadata = lookup_inspire_record(paper.model_dump(), recid=recid,
                                         headers={"User-Agent": fetch._UA})
        if metadata:
            return metadata.get("doi", ""), metadata.get("arxiv_id", ""), "inspire_api"
    return None


def eligible(paper: Paper) -> bool:
    return not (paper.doi or paper.arxiv_id or paper.domain_status) and bool(core._http_url(paper.url))


def recover_identity(paper: Paper) -> IdentityRecovery | None:
    """Pattern → one landing page → exact source API. All fills fail closed.

    Pattern parsing requires no discovery fetch. DOI verification uses the
    existing Crossref authority and title/author screen; arXiv verification
    uses its bounded current-page resolver. Never mutates the supplied card.
    """
    if not eligible(paper):
        return None
    tried = set()

    def verify_once(doi, identifier, route):
        for identity in ((core._doi(doi), ""), ("", core._arxiv(identifier))):
            if identity == ("", "") or identity in tried:
                continue
            tried.add(identity)
            result = _verify(paper, *identity, route)
            if result:
                return result
        return None

    result = verify_once(*_url_identity(paper.url), "url_pattern")
    if result:
        return result
    landing = publisher._fetch_citation_landing(paper.url, metadata_only=True)
    if landing is not None and landing.ok and not publisher._is_pdf_bytes(landing.content):
        if landing.url != paper.url:
            result = verify_once(*_url_identity(landing.url), "url_redirect")
            if result:
                return result
        values = publisher._citation_meta(landing.text)
        result = verify_once(_single(values.get("citation_doi", []), core._doi),
                             _single(values.get("citation_arxiv", []) + values.get("citation_arxiv_id", []),
                                     core._arxiv), "citation_meta")
        if result:
            return result
    try:
        candidate = _source_identity(paper)
        return verify_once(*candidate) if candidate else None
    except (requests.RequestException, core._CoreUnavailable, ValueError, TypeError, AttributeError):
        return None


def _selection(papers: list[Paper], cap: int, keys: list[str] | None) -> list[Paper]:
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= MAX_CAP:
        raise ValueError(f"cap must be between 1 and {MAX_CAP}")
    by_key = {p.key: p for p in papers if eligible(p)}
    order = list(dict.fromkeys(keys)) if keys is not None else sorted(by_key)
    return [by_key[key] for key in order if key in by_key][:cap]


def inspect_identities(papers: list[Paper], *, cap: int, keys: list[str] | None = None) -> dict:
    selected = _selection(papers, cap, keys)
    records, routes = [], Counter()
    for paper in selected:
        recovery = recover_identity(paper)
        if recovery:
            routes[recovery.route] += 1
        records.append({"key": paper.key, "outcome": "verified" if recovery else "abstained",
                        "recovery": recovery.model_dump() if recovery else None})
    return {"dry_run": True, "cap": cap, "scanned": len(selected), "recovered": sum(routes.values()),
            "recovered_by_route": dict(routes), "records": records,
            "acquisition": {"attempted": 0, "by_source": {}, "outcomes": {}}}


def backfill_identities(library: Library, *, cap: int, keys: list[str] | None = None,
                        acquire: bool = False) -> dict:
    from ..download import download_paper

    papers = library.all_papers()
    selected = _selection(papers, cap, keys)
    library.log({"event": "identity_backfill_start", "cap": cap,
                 "keys": [p.key for p in selected], "acquire": acquire})
    result = inspect_identities(selected, cap=cap, keys=keys)
    result["dry_run"] = False
    routes, sources, outcomes = Counter(), Counter(), Counter()
    result["recovered"] = 0
    for item in result["records"]:
        if item["recovery"] is None:
            continue
        recovery = IdentityRecovery(**item["recovery"])
        item["outcome"] = library.set_recovered_identity(item["key"], recovery)
        if item["outcome"] != "set":
            continue
        routes[recovery.route] += 1
        result["recovered"] += 1
        library.save(force=True)
        if not acquire:
            continue
        paper = library.get(item["key"])
        if (paper.download_status not in {"pending", "failed", "metadata_only"}
                or library.has_pdf(paper.key) or library.has_extract(paper.key, "md")
                or library.has_extract(paper.key, "txt")):
            outcomes["existing_or_ineligible"] += 1
            continue
        result["acquisition"]["attempted"] += 1
        # Only this newly backfilled row is revived, immediately before the
        # ordinary cascade's current-version and PDF identity checks.
        paper.download_status = "pending"
        ok = download_paper(paper, library)
        outcome = ("withdrawn" if paper.arxiv_withdrawal else "pdf" if ok else
                   "text" if paper.download_status == "ok" and library.has_extract(paper.key, "md") else "miss")
        outcomes[outcome] += 1
        source = (paper.download_source_member if paper.download_source in {"oa_aggregators", "domain_aggregators"}
                  and paper.download_source_member else paper.download_source)
        if outcome in {"pdf", "text"}:
            sources[source] += 1
        item["acquisition"] = outcome
        library.log({"event": "identity_backfill_acquisition", "key": paper.key,
                     "outcome": outcome,
                     "source": source if outcome in {"pdf", "text"} else ""})
        library.save(force=True)
    result["recovered_by_route"] = dict(routes)
    result["acquisition"].update(by_source=dict(sources), outcomes=dict(outcomes))
    library.log({"event": "identity_backfill_pass", **result})
    return result
