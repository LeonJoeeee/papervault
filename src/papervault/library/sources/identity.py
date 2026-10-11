"""Pure capture of source-supplied identifiers; no fetching or fusion decisions."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from ..models import SourceIdentity


PRIMARY_NAMESPACES = {
    "arxiv": "arxiv:paper",
    "ads": "ads:bibcode",
    "semantic_scholar": "semantic_scholar:paper",
    "openalex": "openalex:work",
    "inspire": "inspire:literature",
    "core": "core:work",
}


def identity_values(values: Any) -> list[str]:
    """Keep distinct supplied scalar values, with missing values represented by []."""
    if not isinstance(values, (list, tuple)):
        values = [values]
    out: list[str] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            continue
        value = str(value).strip()
        if value and value not in out:
            out.append(value)
    return out


def source_identity(source: str, *, source_ids: dict, dois=(), arxiv_ids=(), urls=()) -> dict:
    """Create an observation, retaining all IDs rather than choosing a winner."""
    clean_urls = []
    for url in identity_values(urls):
        try:
            parsed = urlsplit(url)
            if (parsed.scheme in {"http", "https"} and parsed.hostname
                    and parsed.username is None and parsed.password is None
                    and not any(c.isspace() for c in url)):
                clean_urls.append(url)
        except ValueError:
            continue
    doi_values = [re.sub(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)", "", v, flags=re.I)
                  for v in identity_values(dois)]
    arxiv_values = [re.sub(r"^(?:arxiv:|oai:arxiv\.org:|https?://arxiv\.org/abs/)",
                           "", v, flags=re.I) for v in identity_values(arxiv_ids)]
    return SourceIdentity(
        source=source,
        source_ids={namespace: identity_values(values) for namespace, values in source_ids.items()},
        dois=identity_values(doi_values), arxiv_ids=identity_values(arxiv_values),
        urls=clean_urls,
    ).model_dump()


def normalized_source_identities(data: dict, source: str) -> list[dict]:
    """Preserve adapter observations; accommodate the older normalized result shape.

    Never call this on legacy persisted rows: their unqualified paper_id may
    lack enough provenance to assign a namespace safely.
    """
    captured = data.get("source_identities")
    if captured is not None:
        return [o.model_dump() for o in merge_source_identities(captured)]
    namespace = PRIMARY_NAMESPACES.get(source)
    if namespace is None:
        return []
    identifier = data.get("paper_id")
    if source == "arxiv":
        identifier = data.get("arxiv_id")
    elif source == "inspire":
        identifier = data.get("inspire_record_id")
    return [source_identity(source, source_ids={namespace: identifier},
                            dois=data.get("doi"), arxiv_ids=data.get("arxiv_id"),
                            urls=[data.get("url"), *(data.get("inspire_document_urls") or [])])]


def semantic_scholar_identity(paper: dict) -> dict:
    """Capture the same SS surface for search and identifier-based admission."""
    ext = paper.get("externalIds") or {}
    return source_identity(
        "semantic_scholar",
        source_ids={"semantic_scholar:paper": paper.get("paperId"),
                    **{f"semantic_scholar:external:{kind}": value
                       for kind, value in ext.items() if kind not in {"DOI", "ArXiv"}}},
        dois=ext.get("DOI"), arxiv_ids=ext.get("ArXiv"), urls=paper.get("url"),
    )


def merge_source_identities(*groups) -> list[SourceIdentity]:
    """Append distinct observations, retaining conflicts and observed absences.

    Equality dedupes repeated search hits; it does not conflate namespaces or
    silently replace the identifiers/URLs carried by an earlier observation.
    """
    out: list[SourceIdentity] = []
    for group in groups:
        for value in group or []:
            observation = SourceIdentity.model_validate(value)
            if observation not in out:
                out.append(observation.model_copy(deep=True))
    return out
