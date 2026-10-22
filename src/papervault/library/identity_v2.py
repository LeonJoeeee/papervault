"""Pure identity selection and P2 correspondence (accepted spec, Appendix A4).

Selection never merges records or orders retrieval tiers. Missing evidence is
unresolved; positive contradictory evidence rejects one route, never a paper.
The bibliography thresholds follow the existing conservative DOI resolver.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum, IntEnum
from urllib.parse import unquote, urlsplit


TITLE_RATIO_MIN = 0.92
TITLE_RATIO_MIN_NO_YEAR = 0.97
TITLE_TOKEN_JACCARD_MIN = 0.90
MIN_NORMALIZED_TITLE_LENGTH = 25
AUTHOR_SURNAME_JACCARD_MIN = 0.50  # also require the reference's first surname
YEAR_MAX_DELTA = 1  # online/print publication dates can differ by one year


class Rung(IntEnum):
    DOI = 0
    SOURCE = 1
    URL = 2


class Outcome(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    CONTRADICTED = "contradicted"


class RejectionKind(str, Enum):
    ROUTE_REJECTED = "route_rejected"
    IDENTITY_REJECTED = "identity_rejected"
    EXHAUSTED = "exhausted"


class Reason(str, Enum):
    DOI_EQUIVALENT = "doi_equivalent"
    BIBLIOGRAPHY_MATCH = "bibliography_match"
    INVALID_DOI = "invalid_doi"
    INVALID_URL = "invalid_url"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DOI_MISMATCH = "doi_mismatch"
    TITLE_MISMATCH = "title_mismatch"
    AUTHOR_MISMATCH = "author_mismatch"
    YEAR_MISMATCH = "year_mismatch"


@dataclass(frozen=True)
class CorrespondenceDecision:
    outcome: Outcome
    reason: Reason
    rejection: RejectionKind | None = None


@dataclass(frozen=True)
class RouteRejection:
    url: str
    route: str
    reason: Reason
    kind: RejectionKind = RejectionKind.ROUTE_REJECTED


@dataclass(frozen=True)
class CanonicalIdentity:
    rung: Rung
    key: str


@dataclass(frozen=True)
class NamespacedIdentity:
    namespace: str
    identifier: str

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.identifier}"


def _field(record, name, default=None):
    return record.get(name, default) if isinstance(record, dict) else getattr(record, name, default)


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _values(value) -> tuple:
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _http_url(value) -> str:
    value = _text(value)
    if not value or any(c.isspace() or ord(c) < 32 for c in value) or "\\" in value:
        return ""
    try:
        parts = urlsplit(value)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None or parts.port == 0):
            return ""
    except ValueError:
        return ""
    return value


def canonical_doi(value) -> str:
    """Validate a DOI, folding resolver wrappers and case, never guessing an ID."""
    value = _text(value)
    value = re.sub(r"^doi:\s*", "", value, flags=re.I)
    if re.match(r"^(?:https?://)?(?:dx\.)?doi\.org/", value, re.I):
        url = value if re.match(r"^https?://", value, re.I) else "https://" + value
        if not _http_url(url):
            return ""
        value = unquote(urlsplit(url).path.lstrip("/"))
    return value.casefold() if re.fullmatch(r"10\.[0-9]{4,9}/[^\s<>]+", value) else ""


_DOI_PUBLISHER_HOSTS = {"onlinelibrary.wiley.com", "link.springer.com", "iopscience.iop.org",
                        "journals.aps.org", "academic.oup.com", "pubs.acs.org",
                        "journals.sagepub.com", "www.tandfonline.com"}


def _route_doi(url: str) -> str:
    doi = canonical_doi(url)
    if doi:
        return doi
    parts = urlsplit(url)
    if parts.hostname not in _DOI_PUBLISHER_HOSTS:
        return ""
    match = re.search(r"/(10\.[0-9]{4,9}/.+)$", unquote(parts.path))
    if not match:
        return ""
    return canonical_doi(re.sub(r"/(?:pdf|full|abstract|epdf)$", "", match[1], flags=re.I))


_NAMESPACE_ALIASES = {"arxiv:paper": "arxiv", "inspire:literature": "inspire",
                      "semantic_scholar:paper": "semantic_scholar"}
_PRIMARY_NAMESPACES = {"arxiv": "arxiv", "inspire": "inspire",
                       "semantic_scholar": "semantic_scholar", "core": "core:work",
                       "ads": "ads:bibcode", "openalex": "openalex:work"}


def format_namespace(namespace, identifier) -> str:
    """CORE must explicitly name work or output; numeric equality is insufficient."""
    namespace = _NAMESPACE_ALIASES.get(_text(namespace), _text(namespace))
    identifier = _text(identifier)
    if namespace not in {*_PRIMARY_NAMESPACES.values(), "core:output"}:
        return ""
    if namespace in {"core:work", "core:output", "inspire"}:
        if not re.fullmatch(r"[0-9]+", identifier) or not identifier.lstrip("0"):
            return ""
        identifier = identifier.lstrip("0")
    elif namespace == "arxiv":
        match = re.fullmatch(r"([0-9]{4}\.[0-9]{4,5}|[a-z-]+(?:\.[A-Z]{2})?/[0-9]{7})(?:v[0-9]+)?",
                             identifier, re.I)
        if not match or not 1 <= int(match[1].split("/")[-1][2:4]) <= 12:
            return ""
        identifier = match[1]
    elif not re.fullmatch(r"[A-Za-z0-9._/-]+", identifier):
        return ""
    return f"{namespace}:{identifier}"


def parse_namespace(value) -> NamespacedIdentity | None:
    namespace, _, identifier = _text(value).rpartition(":")
    formatted = format_namespace(namespace, identifier)
    if not formatted:
        return None
    namespace, _, identifier = formatted.rpartition(":")
    return NamespacedIdentity(namespace, identifier)


def select_canonical(record, *, current: CanonicalIdentity | None = None) -> CanonicalIdentity | None:
    """DOI > qualified source ID > route, without changing any stored record.

    A valid current canonical wins same-rung ties. Otherwise ties are lexical,
    independent of observation order; conflicting observations stay available.
    Legacy paper_id is qualified only with an unambiguous known source.
    """
    candidates: set[CanonicalIdentity] = set()

    def doi(value):
        key = canonical_doi(value)
        if key:
            candidates.add(CanonicalIdentity(Rung.DOI, key))

    def source(namespace, value):
        key = format_namespace(namespace, value)
        if key:
            candidates.add(CanonicalIdentity(Rung.SOURCE, key))

    def url(value):
        key = _http_url(value)
        if key:
            candidates.add(CanonicalIdentity(Rung.URL, key))

    doi(_field(record, "doi"))
    source("arxiv", _field(record, "arxiv_id"))
    source("inspire", _field(record, "inspire_record_id"))
    namespace = _PRIMARY_NAMESPACES.get(_text(_field(record, "source")))
    if namespace:
        source(namespace, _field(record, "paper_id"))
    url(_field(record, "url"))
    for observation in _values(_field(record, "source_identities")):
        for value in _values(_field(observation, "dois")):
            doi(value)
        for value in _values(_field(observation, "arxiv_ids")):
            source("arxiv", value)
        ids = _field(observation, "source_ids")
        if isinstance(ids, dict):
            for namespace, values in ids.items():
                for value in _values(values):
                    source(namespace, value)
        for value in _values(_field(observation, "urls")):
            url(value)
    choice = min(candidates, key=lambda item: (item.rung, item.key)) if candidates else None
    if isinstance(current, CanonicalIdentity):
        key = ""
        if current.rung == Rung.DOI:
            key = canonical_doi(current.key)
        elif current.rung == Rung.SOURCE:
            parsed = parse_namespace(current.key)
            key = parsed.key if parsed else ""
        elif current.rung == Rung.URL:
            key = _http_url(current.key)
        if key and (choice is None or current.rung <= choice.rung):
            return CanonicalIdentity(current.rung, key)
    return choice


def classify_rung(record) -> Rung | None:
    canonical = select_canonical(record)
    return canonical.rung if canonical else None


def _title(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(value).lower()).strip()


def serving_bibliographic_key(record) -> tuple[str, str, int] | None:
    """Conservative read-only signature for an identifier-poor serving duplicate.

    This grants no catalog merge. A caller may suppress an identifier-poor row
    only when this signature has exactly one identified representative. Distinct
    identified versions and ambiguous signatures retain separate identities.
    """
    title = re.sub(r"^the ", "", _title(_field(record, "title")))
    authors = _values(_field(record, "authors"))
    author = _title(authors[0]) if authors else ""
    year = _field(record, "year")
    if len(title) < MIN_NORMALIZED_TITLE_LENGTH or not author or type(year) is not int:
        return None
    return title, author, year


def _surnames(values) -> list[str]:
    result = []
    for value in _values(values):
        text = _text(value).split(",", 1)[0] if "," in _text(value) else _text(value)
        words = text.split()
        surname = words[-1].lower().strip(".,;") if words else ""
        if surname:
            result.append(surname)
    return result


def _jaccard(left, right) -> float:
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left and right else 0.0


def _bibliography_reason(reference, observed, *, partial=False) -> Reason | None:
    left, right = _title(_field(reference, "title")), _title(_field(observed, "title"))
    left_authors = _surnames(_field(reference, "authors"))
    right_authors = _surnames(_field(observed, "authors"))
    left_year, right_year = _field(reference, "year"), _field(observed, "year")
    ratio_min = TITLE_RATIO_MIN if type(left_year) is int else TITLE_RATIO_MIN_NO_YEAR
    if min(len(left), len(right)) >= MIN_NORMALIZED_TITLE_LENGTH:
        if (SequenceMatcher(None, left, right).ratio() < ratio_min
                or _jaccard(left.split(), right.split()) < TITLE_TOKEN_JACCARD_MIN):
            return Reason.TITLE_MISMATCH
    if left_authors and right_authors:
        if (left_authors[0] not in right_authors
                or _jaccard(left_authors, right_authors) < AUTHOR_SURNAME_JACCARD_MIN):
            return Reason.AUTHOR_MISMATCH
    if type(left_year) is int and type(right_year) is int:
        if abs(left_year - right_year) > YEAR_MAX_DELTA:
            return Reason.YEAR_MISMATCH
    for metadata in (reference, observed):
        title = _field(metadata, "title")
        authors, year = _field(metadata, "authors"), _field(metadata, "year")
        if ((title is not None and not isinstance(title, str))
                or (authors is not None and (not isinstance(authors, (list, tuple))
                                            or not all(isinstance(a, str) for a in authors)))
                or (year is not None and type(year) is not int)):
            return Reason.INSUFFICIENT_EVIDENCE
    if partial:
        # Missing landing/source fields do not discard corroboration already
        # established by the paper and registry. Supplied short titles cannot
        # establish either agreement or a positive contradiction.
        if right and len(right) < MIN_NORMALIZED_TITLE_LENGTH:
            return Reason.INSUFFICIENT_EVIDENCE
    elif (min(len(left), len(right)) < MIN_NORMALIZED_TITLE_LENGTH
            or (left_authors and not right_authors)
            or (left_year is not None and type(right_year) is not int)):
        return Reason.INSUFFICIENT_EVIDENCE
    return None


def correspondence_decision(doi, url, *, reference=None, observed=None,
                            route_metadata=None) -> CorrespondenceDecision:
    """Resolve one DOI/route pair from supplied evidence, without I/O or mutation.

    A resolver URL can prove equivalence alone. If bibliography is supplied,
    all required fields must corroborate it: short/missing/invalid fields are
    unresolved, while explicit DOI/title/author/year disagreements contradict.
    An opaque route needs matching registry DOI and bibliography evidence.
    Supplemental route metadata is checked when present, without requiring
    landing pages to publish every bibliographic field.
    """
    def unresolved(reason=Reason.INSUFFICIENT_EVIDENCE):
        return CorrespondenceDecision(Outcome.UNRESOLVED, reason)

    def contradicted(reason):
        return CorrespondenceDecision(Outcome.CONTRADICTED, reason, RejectionKind.ROUTE_REJECTED)

    canonical = canonical_doi(doi)
    if not canonical:
        return unresolved(Reason.INVALID_DOI)
    route = _http_url(url)
    if not route:
        return unresolved(Reason.INVALID_URL)
    route_doi = _route_doi(route)
    if route_doi and route_doi != canonical:
        return contradicted(Reason.DOI_MISMATCH)
    observed_raw_doi = _field(observed, "doi")
    observed_doi = canonical_doi(observed_raw_doi)
    if observed_doi and observed_doi != canonical:
        return contradicted(Reason.DOI_MISMATCH)
    if observed_raw_doi and not observed_doi:
        return unresolved()
    if route_metadata is not None:
        if not isinstance(route_metadata, dict):
            return unresolved()
        raw = route_metadata.get("doi")
        route_observed_doi = canonical_doi(raw)
        if route_observed_doi and route_observed_doi != canonical:
            return contradicted(Reason.DOI_MISMATCH)
        if raw and not route_observed_doi:
            return unresolved()
        expected = {name: _field(observed, name) if _field(observed, name) is not None
                    else _field(reference, name) for name in ("title", "authors", "year")}
        reason = _bibliography_reason(expected, route_metadata, partial=True)
        if reason:
            return unresolved() if reason == Reason.INSUFFICIENT_EVIDENCE else contradicted(reason)
    if reference is None and observed is None:
        return (CorrespondenceDecision(Outcome.RESOLVED, Reason.DOI_EQUIVALENT)
                if route_doi == canonical else unresolved())

    reason = _bibliography_reason(reference, observed)
    if reason:
        return unresolved() if reason == Reason.INSUFFICIENT_EVIDENCE else contradicted(reason)
    if route_doi == canonical or observed_doi == canonical:
        return CorrespondenceDecision(Outcome.RESOLVED, Reason.BIBLIOGRAPHY_MATCH)
    return unresolved()
