"""Bounded URL identity recovery and a recorded pass through the ordinary cascade."""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import requests

from papervault.config import identity_v2_enabled

from .. import fetch
from ..download_sources import ads, arxiv, core, openalex, publisher
from ..models import IdentityRecovery, Paper
from ..identity_v2 import Outcome, RouteRejection, correspondence_decision
from ..sources.inspire import inspire_record_id, lookup_inspire_record
from ..store import Library

MAX_CAP = 50
SNAPSHOT_FIELDS = {"title", "authors", "year", "url", "source", "paper_id", "inspire_record_id",
                   "doi", "arxiv_id", "added_at", "domain_status"}
_DOI_HOSTS = {"doi.org", "dx.doi.org", "onlinelibrary.wiley.com", "link.springer.com",
              "iopscience.iop.org", "journals.aps.org", "academic.oup.com",
              "pubs.acs.org", "journals.sagepub.com", "www.tandfonline.com"}


def read_identity_records(root: Path) -> list[Paper]:
    """Use the store's last-good snapshot rules without its writable constructor."""
    if os.environ.get("PAPERVAULT_CATALOG_AUTHORITY", "json") == "postgres":
        library = Library(root, read_only=True)
        try:
            return library.all_papers(include_quarantined=True)
        finally:
            library.close()
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


def _verify(paper: Paper, doi: str, arxiv_id: str, route: str, *, source_url: str | None = None,
            route_metadata: dict | None = None, registry_cache: dict | None = None) -> IdentityRecovery | None:
    doi, arxiv_id = core._doi(doi), core._arxiv(arxiv_id)
    if doi:
        if registry_cache is not None:
            if doi not in registry_cache:
                registry_cache[doi] = fetch._fetch_crossref_by_doi(doi)
            metadata = registry_cache[doi]
        else:
            metadata = fetch._fetch_crossref_by_doi(doi)
        if identity_v2_enabled():
            observed = {name: (metadata or {}).get(name) for name in ("doi", "title", "authors", "year")}
            url = paper.url if source_url is None else source_url
            decision = correspondence_decision(doi, url, reference=paper, observed=observed,
                                                route_metadata=route_metadata)
            return IdentityRecovery(doi=doi, route=route, source_url=url,
                verified_title=observed["title"] if isinstance(observed["title"], str) else "",
                observed_metadata=observed, route_metadata=route_metadata,
                correspondence=decision, observed_at=datetime.now(timezone.utc).isoformat())
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


def _source_identity(paper: Paper, *, collect_evidence: bool = False) -> tuple | None:
    """One exact source lookup; v2 leaves DOI bibliography decisions to P2."""
    def candidate(doi, identifier, route, evidence, url):
        return (doi, identifier, route, evidence, url) if collect_evidence else (doi, identifier, route)

    bibcode = ads._ads_bibcode(paper)
    if bibcode:
        metadata, _ = ads._fetch_ads_metadata(paper)
        if metadata:
            doi = _single(metadata.get("doi"), core._doi)
            evidence = {"title": " ".join(metadata.get("title") or []),
                        "authors": metadata.get("author") or []}
            if (collect_evidence and doi) or _bibliography_matches(paper, evidence):
                if collect_evidence:
                    evidence.update(doi=doi, year=metadata.get("year"))
                query = quote(f'bibcode:"{bibcode}"', safe="")
                return candidate(doi, _single(metadata.get("identifier"), core._arxiv), "ads_api", evidence,
                                 f"https://api.adsabs.harvard.edu/v1/search/query?q={query}")
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
        ids = core._metadata_identifiers(metadata) if collect_evidence else core._corroborate(paper, metadata)
        if ids and collect_evidence and not ids[0]:
            ids = core._corroborate(paper, metadata)  # preserve non-DOI verification
        if not ids:
            return None
        if not collect_evidence:
            return *ids, "core_api"
        authors = [a.get("name", "") if isinstance(a, dict) else a
                   for a in metadata.get("authors") or []]
        evidence = {"doi": ids[0], "title": metadata.get("title"), "authors": authors,
                    "year": metadata.get("yearPublished", metadata.get("year"))}
        return candidate(*ids, "core_api", evidence, f"https://api.core.ac.uk/v3/{'/'.join(locator)}")
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
        evidence = {"title": metadata.get("title") or metadata.get("display_name"),
                    "authors": authors, "year": metadata.get("publication_year")}
        doi = openalex._doi(metadata)
        if not ((collect_evidence and doi) or _bibliography_matches(paper, evidence)):
            return None
        if collect_evidence:
            evidence["doi"] = doi
        return candidate(doi, "", "openalex_api", evidence, f"https://api.openalex.org/works/{work}")
    recid = inspire_record_id(paper.inspire_record_id)
    if parts.hostname == "inspirehep.net":
        match = re.fullmatch(r"/literature/([0-9]+)/?", parts.path)
        url_recid = inspire_record_id(match[1]) if match else ""
        if recid and url_recid and recid != url_recid:
            return None
        recid = recid or url_recid
    if recid:
        metadata = lookup_inspire_record(paper.model_dump(), recid=recid,
            headers={"User-Agent": fetch._UA}, **({"collect_only": True} if collect_evidence else {}))
        if metadata:
            doi = metadata.get("doi", "")
            if collect_evidence and not doi:
                from ..sources.inspire import inspire_identity_matches
                if not inspire_identity_matches(paper.model_dump(), metadata):
                    return None
            evidence = {name: metadata.get(name) for name in ("doi", "title", "authors", "year")}
            return candidate(doi, metadata.get("arxiv_id", ""), "inspire_api", evidence,
                             f"https://inspirehep.net/api/literature/{recid}")
    return None


def eligible(paper: Paper) -> bool:
    return not (paper.doi or paper.arxiv_id or paper.domain_status) and bool(core._http_url(paper.url))


def _landing_bibliography(values: dict) -> dict | None:
    """Retain supplied citation fields, including ambiguity/invalid field types."""
    evidence = {}
    for field in ("doi", "title"):
        raw = values.get(f"citation_{field}") or []
        if raw:
            value = _single(raw, core._doi if field == "doi" else lambda text: text.strip())
            evidence[field] = value or raw
    if values.get("citation_author"):
        evidence["authors"] = values["citation_author"]
    dates = [value for name in ("citation_year", "citation_date", "citation_publication_date")
             for value in values.get(name, [])]
    if dates:
        def year(value):
            match = re.match(r"^([0-9]{4})(?:[-/]|$)", value)
            return match[1] if match else value
        value = _single(dates, year)
        evidence["year"] = int(value) if re.fullmatch(r"[0-9]{4}", value) else value
    return evidence or None


def recover_identity(paper: Paper) -> IdentityRecovery | None:
    """Pattern → one landing page → exact source API. All fills fail closed.

    Pattern parsing requires no discovery fetch. DOI verification uses the
    existing Crossref authority and title/author screen; arXiv verification
    uses its bounded current-page resolver. V2 retains raw P1 observations and
    evaluates each actual route at P2, sharing registry lookups within this pass.
    Never mutates the supplied card.
    """
    if not eligible(paper):
        return None
    tried = set()
    rejections: list[RouteRejection] = []
    undecided: list[IdentityRecovery] = []
    registry_cache = {} if identity_v2_enabled() else None

    def verify_once(doi, identifier, route, *, source_url=None, route_metadata=None):
        for identity in ((core._doi(doi), ""), ("", core._arxiv(identifier))):
            marker = (identity, route, source_url or paper.url) if identity_v2_enabled() else identity
            if identity == ("", "") or marker in tried:
                continue
            tried.add(marker)
            result = _verify(paper, *identity, route, source_url=source_url,
                             route_metadata=route_metadata, registry_cache=registry_cache)
            if result:
                if identity_v2_enabled() and result.correspondence is not None:
                    if result.correspondence.outcome != Outcome.RESOLVED:
                        undecided.append(result)
                        if result.correspondence.outcome == Outcome.CONTRADICTED:
                            rejection = RouteRejection(result.source_url, result.route,
                                                       result.correspondence.reason)
                            if rejection not in rejections:
                                rejections.append(rejection)
                        continue
                if rejections:
                    result = result.model_copy(update={"route_rejections": list(rejections)})
                return result
        return None

    result = verify_once(*_url_identity(paper.url), "url_pattern")
    if result:
        return result
    landing = publisher._fetch_citation_landing(paper.url, metadata_only=True)
    if landing is not None and landing.ok and not publisher._is_pdf_bytes(landing.content):
        values = publisher._citation_meta(landing.text) if identity_v2_enabled() else None
        evidence = _landing_bibliography(values) if values is not None else None
        if landing.url != paper.url:
            result = verify_once(*_url_identity(landing.url), "url_redirect",
                source_url=landing.url if identity_v2_enabled() else None, route_metadata=evidence)
            if result:
                return result
        values = publisher._citation_meta(landing.text) if values is None else values
        result = verify_once(_single(values.get("citation_doi", []), core._doi),
                             _single(values.get("citation_arxiv", []) + values.get("citation_arxiv_id", []),
                                     core._arxiv), "citation_meta",
            source_url=landing.url if identity_v2_enabled() else None, route_metadata=evidence)
        if result:
            return result
    try:
        candidate = (_source_identity(paper, collect_evidence=True) if identity_v2_enabled()
                     else _source_identity(paper))
        result = (verify_once(*candidate[:3], route_metadata=candidate[3], source_url=candidate[4])
                  if candidate and len(candidate) == 5
                  else verify_once(*candidate) if candidate else None)
        if result:
            return result
    except (requests.RequestException, core._CoreUnavailable, ValueError, TypeError, AttributeError):
        pass
    # Preserve the failed decision for P3, after trying independent identities.
    # Positive contradiction takes precedence in the report; unavailable routes
    # are still retryable and no paper status is written here.
    if undecided:
        result = next((item for item in undecided
                       if item.correspondence.outcome == Outcome.CONTRADICTED), undecided[0])
        return result.model_copy(update={"route_rejections": rejections})
    return None


def _selection(papers: list[Paper], cap: int, keys: list[str] | None) -> list[Paper]:
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= MAX_CAP:
        raise ValueError(f"cap must be between 1 and {MAX_CAP}")
    by_key = {p.key: p for p in papers if eligible(p)}
    order = list(dict.fromkeys(keys)) if keys is not None else sorted(by_key)
    return [by_key[key] for key in order if key in by_key][:cap]


def inspect_identities(papers: list[Paper], *, cap: int, keys: list[str] | None = None,
                       pace_seconds: float = 0) -> dict:
    selected = _selection(papers, cap, keys)
    records, routes = [], Counter()
    for index, paper in enumerate(selected):
        if index and pace_seconds:
            time.sleep(pace_seconds)
        recovery = recover_identity(paper)
        outcome = "verified" if recovery else "abstained"
        if identity_v2_enabled() and recovery and recovery.correspondence is not None:
            if recovery.correspondence.outcome != Outcome.RESOLVED:
                outcome = recovery.correspondence.outcome.value
        if recovery and outcome == "verified":
            routes[recovery.route] += 1
        records.append({"key": paper.key, "outcome": outcome,
                        "recovery": recovery.model_dump() if recovery else None})
    return {"dry_run": True, "cap": cap, "scanned": len(selected), "recovered": sum(routes.values()),
            "recovered_by_route": dict(routes), "records": records,
            "acquisition": {"attempted": 0, "by_source": {}, "outcomes": {}}}


def backfill_identities(library: Library, *, cap: int, keys: list[str] | None = None,
                        acquire: bool = False) -> dict:
    papers = library.all_papers()
    selected = _selection(papers, cap, keys)
    library.log({"event": "identity_backfill_start", "cap": cap,
                 "keys": [p.key for p in selected], "acquire": acquire})
    result = inspect_identities(selected, cap=cap, keys=keys)
    return apply_identity_report(library, result, acquire=acquire)


def apply_identity_report(library: Library, result: dict, *, acquire: bool = False,
                          doi_only: bool = False, snapshots: dict[str, Paper] | None = None) -> dict:
    """Apply the reviewed inspection path; async callers hold lib_write_lock.

    Splitting verification from application lets the correction pipeline inspect
    snapshots off-loop without giving a background thread a writable Library.
    The standing pipeline defers non-DOI claims per #198 Q4.
    """
    from ..download import download_paper

    result["dry_run"] = False
    routes, sources, outcomes = Counter(), Counter(), Counter()
    result["recovered"] = 0
    for item in result["records"]:
        if item["recovery"] is None:
            continue
        recovery = IdentityRecovery(**item["recovery"])
        if snapshots is not None:
            snapshot, current = snapshots.get(item["key"]), library.get(item["key"])
            if (snapshot is None or current is None or current.model_dump(include=SNAPSHOT_FIELDS)
                    != snapshot.model_dump(include=SNAPSHOT_FIELDS)):
                item["outcome"] = "stale_snapshot"
                if current is not None and snapshot is not None:
                    library.corrections.stage_fields(current, snapshot.model_dump(),
                        operation="stale_identity_backfill", provenance={"recovery": recovery.model_dump()},
                        proposals={k: v for k, v in {"doi": recovery.doi, "arxiv_id": recovery.arxiv_id}.items() if v})
                continue
        if identity_v2_enabled():
            paper = library.get(item["key"])
            if paper is None:
                item["outcome"] = "no_paper"
                continue
            rejections = list(recovery.route_rejections)
            if recovery.doi:
                # Recheck the current bibliography at the adoption boundary;
                # never trust a precomputed verdict from an inspection report.
                decision = correspondence_decision(recovery.doi, recovery.source_url,
                    reference=paper, observed=(recovery.observed_metadata if recovery.observed_metadata is not None
                                              else {"title": recovery.verified_title}),
                    route_metadata=recovery.route_metadata)
                recovery = recovery.model_copy(update={"correspondence": decision})
                item["recovery"] = recovery.model_dump()
                if decision.outcome == Outcome.CONTRADICTED:
                    rejections.append(RouteRejection(recovery.source_url, recovery.route, decision.reason))
            changed = False
            for rejection in rejections:
                if rejection not in paper.identity_route_rejections:
                    paper.identity_route_rejections.append(rejection)
                    changed = True
            if changed:
                library.save(force=True)
            if recovery.doi and decision.outcome != Outcome.RESOLVED:
                item["outcome"] = decision.outcome.value
                continue
        if doi_only and not recovery.doi:
            item["outcome"] = "deferred_non_doi"
            continue
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
