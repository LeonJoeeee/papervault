"""Multi-source literature search across arXiv + Semantic Scholar + OpenAlex +
INSPIRE + ADS + CORE. Returns deduped Paper-shaped dicts (not yet inserted
into the library).

Also exposes citation-graph chasing: given a seed paper, pull its references
(papers it cites) and citations (papers that cite it) from Semantic Scholar.

✦ Phase 29 (2026-05-27): query expansion moved upstream (LLM intent parser
in ``services.intent_parser``). This module no longer does rule-based
``expand_queries``; callers pass a list of search-term variants directly.
EuropePMC was dropped (medical bias, off-topic for space physics + AI4Science).
External fetch is now ``async`` with per-backend concurrency.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Optional

import requests

from .sources.arxiv import search_arxiv
from .sources.semantic_scholar import search_semantic_scholar
from .sources.inspire import search_inspire
from .sources.ads import search_ads
from .sources.core import search_core
from .sources.exceptions import BackendDegraded

logger = logging.getLogger(__name__)


_SS_BASE = "https://api.semanticscholar.org/graph/v1/paper"
_SS_FIELDS = ("title,authors,year,abstract,citationCount,externalIds,"
              "publicationTypes,venue,publicationVenue,url")


# Per-(term, backend) fetch cap — the SINGLE source of truth (§0). Replaces the
# old ``_BACKEND_WEIGHTS`` list AND the V5 SPECIALIST_CAP/GENERALIST_CAP. KEYS ARE
# THE FULL ``search_*`` SUFFIXES (``semantic_scholar``, NEVER ``s2``) so that
# ``f"search_{name}"`` resolves to the real backend function via a call-time
# ``globals()`` lookup (the lookup is call-time — NOT a frozen dict — specifically
# so ``monkeypatch.setattr`` in tests intercepts; see ``_fetch_one_backend``).
CAPS = {
    "arxiv":            30,
    "ads":              30,
    "semantic_scholar": 15,
    "openalex":         15,
    "inspire":          15,
    "core":             15,
}


def _normalize_ss_paper(paper: dict, source: str) -> dict:
    """Normalize a Semantic Scholar paper record to our Paper-dict shape."""
    authors = [a.get("name", "") for a in paper.get("authors", [])]
    ext = paper.get("externalIds") or {}
    venue = paper.get("venue") or (paper.get("publicationVenue") or {}).get("name", "")
    pub_types = paper.get("publicationTypes") or []
    title = (paper.get("title") or "").strip()
    return {
        "title": title,
        "authors": authors,
        "abstract": (paper.get("abstract") or "").strip(),
        "year": paper.get("year"),
        "doi": (ext.get("DOI") or "").strip(),
        "arxiv_id": (ext.get("ArXiv") or "").strip(),
        "paper_id": paper.get("paperId", ""),
        "venue": venue,
        "publication_types": pub_types,
        "citation_count": int(paper.get("citationCount") or 0),
        "url": paper.get("url") or "",
        "is_review": _looks_like_review({"title": title, "publication_types": pub_types}),
        "source": source,
    }


def fetch_references(paper_id_or_doi: str, *, limit: int = 200) -> list[dict]:
    """Fetch papers cited by the given paper (its bibliography)."""
    if not paper_id_or_doi:
        return []
    pid = paper_id_or_doi
    url = f"{_SS_BASE}/{pid}/references"
    try:
        r = requests.get(url, params={"fields": _SS_FIELDS, "limit": min(limit, 1000)},
                         timeout=30)
        if r.status_code == 429:
            time.sleep(5.0)
            r = requests.get(url, params={"fields": _SS_FIELDS, "limit": min(limit, 1000)},
                             timeout=30)
        r.raise_for_status()
    except Exception:
        return []
    out: list[dict] = []
    for entry in r.json().get("data", []) or []:
        cited = entry.get("citedPaper") or {}
        if not cited.get("title"):
            continue
        out.append(_normalize_ss_paper(cited, "ss_references"))
    return out


def search_openalex(query: str, *, max_results: int = 30,
                    mailto: Optional[str] = None) -> list[dict]:
    """Search OpenAlex (250M+ academic works, free, no key needed)."""
    import os
    if not query:
        return []
    mailto = mailto or os.environ.get("OPENALEX_MAILTO", "research@example.invalid")
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": min(max_results, 200),
        "select": ("id,title,authorships,publication_year,abstract_inverted_index,"
                   "cited_by_count,doi,type,primary_location,open_access"),
        "mailto": mailto,
    }
    oa_headers = {"User-Agent": f"paper-pipeline (mailto:{mailto})"}
    try:
        r = requests.get(url, params=params, timeout=30, headers=oa_headers)
        # Retry on 429 (rate limit) AND 409 (conflict/transient). RE-SEND the
        # mailto/User-Agent header on the retry (the prior single re-issue
        # dropped it, dropping OpenAlex into the slow anonymous pool).
        if r.status_code in (429, 409):
            time.sleep(5.0)
            r = requests.get(url, params=params, timeout=30, headers=oa_headers)
        r.raise_for_status()
    except Exception:
        return []
    out: list[dict] = []
    for w in r.json().get("results", []) or []:
        title = (w.get("title") or "").strip()
        if not title:
            continue
        # OpenAlex stores abstracts as inverted index — reconstruct
        abs_inv = w.get("abstract_inverted_index") or {}
        abstract = ""
        if abs_inv:
            positions: list[tuple[int, str]] = []
            for word, idxs in abs_inv.items():
                for i in idxs:
                    positions.append((i, word))
            positions.sort()
            abstract = " ".join(w for _, w in positions)
        authors = []
        for a in (w.get("authorships") or []):
            name = ((a.get("author") or {}).get("display_name") or "").strip()
            if name:
                authors.append(name)
        # Real OpenAlex nests the venue at primary_location.source.display_name
        # (the old top-level host_venue field was REMOVED from the API). Read the
        # nested path first; fall back to a legacy host_venue dict for any
        # archived/legacy payloads still carrying it.
        venue = (((w.get("primary_location") or {}).get("source") or {})
                 .get("display_name") or "").strip()
        if not venue:
            venue = ((w.get("host_venue") or {}).get("display_name") or "").strip()
        doi = (w.get("doi") or "").replace("https://doi.org/", "").strip()
        out.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": w.get("publication_year"),
            "doi": doi,
            "arxiv_id": "",
            "paper_id": (w.get("id") or "").rsplit("/", 1)[-1],
            "venue": venue,
            "publication_types": [w.get("type") or ""],
            "citation_count": int(w.get("cited_by_count") or 0),
            "url": w.get("doi") or w.get("id") or "",
            "is_review": _looks_like_review(
                {"title": title, "publication_types": [w.get("type") or ""]}
            ),
            "source": "openalex",
        })
    return out


def fetch_citations(paper_id_or_doi: str, *, limit: int = 200) -> list[dict]:
    """Fetch papers that cite the given paper (forward citation lookup)."""
    if not paper_id_or_doi:
        return []
    url = f"{_SS_BASE}/{paper_id_or_doi}/citations"
    try:
        r = requests.get(url, params={"fields": _SS_FIELDS, "limit": min(limit, 1000)},
                         timeout=30)
        if r.status_code == 429:
            time.sleep(5.0)
            r = requests.get(url, params={"fields": _SS_FIELDS, "limit": min(limit, 1000)},
                             timeout=30)
        r.raise_for_status()
    except Exception:
        return []
    out: list[dict] = []
    for entry in r.json().get("data", []) or []:
        citing = entry.get("citingPaper") or {}
        if not citing.get("title"):
            continue
        out.append(_normalize_ss_paper(citing, "ss_citations"))
    return out


_REVIEW_KEYWORDS = ("review", "survey", "overview", "perspective", "annual")


def _normalize_paper(p: dict, source: str) -> dict:
    """Normalize a search result into the Paper dict shape (sans key)."""
    return {
        "title": (p.get("title") or "").strip(),
        "authors": p.get("authors") or [],
        "year": int(p.get("year")) if p.get("year") and str(p.get("year")).isdigit() else None,
        "venue": p.get("venue") or "",
        "abstract": (p.get("abstract") or "").strip(),
        "doi": (p.get("doi") or "").strip(),
        "arxiv_id": (p.get("arxiv_id") or "").strip(),
        "paper_id": (p.get("paper_id") or "").strip(),
        "url": (p.get("url") or "").strip(),
        "citation_count": int(p.get("citation_count") or 0),
        "publication_types": p.get("publication_types") or [],
        "is_review": _looks_like_review(p),
        "source": source,
    }


def _looks_like_review(p: dict) -> bool:
    title = (p.get("title") or "").lower()
    if any(kw in title for kw in _REVIEW_KEYWORDS):
        return True
    if "Review" in (p.get("publication_types") or []):
        return True
    return False


_BACKEND_TIMEOUT_SECONDS = 30.0

# Per-backend min-interval pacing (mechanisms 2 & 3, §3). arXiv = one call per
# unique term ≥3s apart; CORE ≥6s apart. Process-global so the limiters protect
# the upstream rate limit across all concurrent ``search_papers`` calls (under
# network_sem=4 this serializes arXiv/CORE across calls — accepted; the
# ``_pace`` wait is OUTSIDE the fetch ``wait_for``, so late calls do not
# self-timeout). The lazy ``asyncio.Lock`` binds to the current running loop
# (fine for the single long-lived server loop).
ARXIV_MIN_INTERVAL = 3.0
CORE_MIN_INTERVAL = 6.0

_LAST_CALL: dict[str, float] = {}
_PACE_LOCK: dict[str, asyncio.Lock] = {}


def _interval_for(backend: str) -> float:
    return {"arxiv": ARXIV_MIN_INTERVAL, "core": CORE_MIN_INTERVAL}.get(backend, 0.0)


async def _pace(backend: str, min_interval: float) -> None:
    """Enforce a per-backend minimum inter-call interval. Called OUTSIDE the
    fetch ``wait_for`` (load-bearing — the pacing wait must NOT count against the
    30s fetch timeout)."""
    if min_interval <= 0:
        return
    # check-then-set (NOT setdefault — that EAGERLY constructs+discards a
    # throwaway Lock every call, binding to the current loop).
    lock = _PACE_LOCK.get(backend)
    if lock is None:
        lock = _PACE_LOCK[backend] = asyncio.Lock()
    async with lock:
        wait = min_interval - (time.monotonic() - _LAST_CALL.get(backend, 0.0))
        if wait > 0:
            await asyncio.sleep(wait)
        _LAST_CALL[backend] = time.monotonic()


def _year_drop(paper: dict, year_min, year_max) -> bool:
    """The ONE real filter (§1 YEAR_DROP) — applied client-side here in the
    external arm because no backend signature accepts a year bound.

    Drop iff a window bound is set AND the paper's year is a non-None int
    OUTSIDE ``[year_min, year_max]``. A None-year (or any non-int) paper is
    ALWAYS KEPT (recency unverifiable, not a reason to cut). No-op when both
    bounds are None. Mirrors ``server.YEAR_DROP`` exactly (kept local so
    search.py has no import-cycle back into the MCP server).
    """
    if year_min is None and year_max is None:
        return False
    y = paper.get("year")                      # NEVER paper.year (AttributeError on a dict)
    if not isinstance(y, int):                 # None-year (and any non-int) → KEEP
        return False
    return (year_min is not None and y < year_min) or (year_max is not None and y > year_max)


async def _fetch_one_backend(
    query: str,
    backend_name: str,
    max_n: int,
    *,
    term_idx: int,
    year_min=None,
    year_max=None,
    degraded_map=None,
    sort_by_recency: bool = False,
    timeout: float = _BACKEND_TIMEOUT_SECONDS,
) -> list[dict]:
    """Run one (term, backend) call in a thread; never raises.

    Per-backend timeout (default 30s) prevents one slow backend from blocking
    the whole ``asyncio.gather`` — a dead/slow/blocked backend is one EMPTY
    list for that pair, never a retry-to-abort.

    Three failure classes are recorded as DEGRADED into ``degraded_map`` (a
    ``collections.Counter`` keyed by backend; ``backend -> #DEGRADED pairs``):
    a typed ``BackendDegraded`` raised by the source (retry-exhausted, WAF,
    auth-rejected, …), this coroutine's own ``asyncio.TimeoutError``, and any
    other ``Exception`` (e.g. a raw non-429 HTTPError from an un-hardened path).
    All three still return ``[]`` (failure isolation unchanged). The increment
    happens on the LOOP thread (this async body, no ``await`` mid-increment),
    NOT inside the threaded ``backend_fn`` — race-safe under the concurrent
    ``gather`` (R3-F5).

    Pacing (``_pace``) runs FIRST — OUTSIDE the ``wait_for`` — so the per-backend
    min-interval wait does NOT count against the 30s fetch timeout.

    Tags every surviving result with its provenance at fetch time:
      - ``_source_origin="external"`` (the §4a partition is library-positive;
        an un-stamped external node would mis-classify and the WHOLE external
        arm would silently vanish — ``_normalize_paper`` does NOT stamp it),
      - ``term_idx`` (which search term fired this fetch),
      - ``rank`` = the backend's TRUE NATIVE position, stamped BEFORE the
        client-side year cut so ``term_ranks`` (§4a) holds the genuine native
        rank, not a dense re-rank among year-survivors.

    Backend function is looked up from this module's globals at CALL TIME via a
    subscript (so ``monkeypatch.setattr`` in tests intercepts correctly; the
    import-time wiring check at module bottom makes the ``KeyError`` impossible
    for a wired backend).
    """
    backend_fn = globals()[f"search_{backend_name}"]   # call-time (monkeypatch-friendly)
    # arXiv is the ONLY backend that takes a sort-hint (no citations → never an
    # importance sort); thread the recency hint only to it, keeping the uniform
    # ``(query, max_results=)`` call for the others.
    fn_kwargs = {"max_results": max_n}
    if backend_name == "arxiv":
        fn_kwargs["sort_by_recency"] = sort_by_recency
    await _pace(backend_name, _interval_for(backend_name))   # PACING FIRST — outside wait_for
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(backend_fn, query, **fn_kwargs),
            timeout=timeout,
        )
    except BackendDegraded as e:
        logger.warning(
            "search: backend %s DEGRADED for query %r: %s", backend_name, query, e,
        )
        if degraded_map is not None:
            degraded_map[backend_name] += 1   # increment on the LOOP thread (R3-F5)
        return []
    except asyncio.TimeoutError:
        logger.warning(
            "search: backend %s timeout (>%ds) for query %r",
            backend_name, int(timeout), query,
        )
        if degraded_map is not None:
            degraded_map[backend_name] += 1
        return []
    except Exception as e:
        logger.warning(
            "search: backend %s failed for query %r: %s",
            backend_name, query, e,
        )
        if degraded_map is not None:
            degraded_map[backend_name] += 1   # also a raw non-429 HTTPError
        return []
    results = [_normalize_paper(p, backend_name) for p in raw]   # year int|None (int 0→None already)
    for i, d in enumerate(results):                              # _source_origin + NATIVE rank, FIRST
        d["_source_origin"] = "external"
        d["term_idx"] = term_idx
        d["rank"] = i                          # TRUE native position (pre-year)
    # Post-fetch year cut, AFTER stamping (native-rank fidelity preserved).
    return [d for d in results if not _year_drop(d, year_min, year_max)]


async def search_external_async(
    queries: list[str],
    *,
    year_min=None,
    year_max=None,
    ranking_hint: str = "by_relevance",
) -> tuple[list[dict], dict]:
    """Fan out T×6 concurrent (term, backend) fetches; return the TAGGED,
    UN-deduped, UN-sorted concatenation of every pair's results (``ext_raw``)
    PLUS the ``degraded_map`` (the §3 channel back to §8's ``sources_degraded``).

    Every (term, backend) pair fires unconditionally via ``asyncio.gather`` +
    ``asyncio.to_thread`` (backends are synchronous). Per-(term, backend) caps
    come from the flat ``CAPS`` dict. Each surviving result carries its
    provenance tags ``{_source_origin="external", term_idx, rank (native)}``
    and has passed the client-side year cut (``_year_drop``, stamped BEFORE the
    cut). The §4a fold downstream does identity-dedup and rebuilds the per-term
    ranklists from ``term_idx``/``rank`` — there is NO flatten-sort-cap here.

    Args:
        queries: the parsed search terms (1..8 distinct, non-empty — §1).
            Each becomes one query per backend.
        year_min / year_max: the parsed year window, applied client-side per
            pair (no backend signature accepts a year bound — §3 reality).
        ranking_hint: when ``"by_recency"``, arXiv sorts ``submittedDate desc``
            (threaded to the arXiv backend only); else relevance.

    Returns:
        ``(ext_raw, degraded_map)`` — ``ext_raw`` is the flat concatenation of
        every (term, backend) pair's tagged result dicts (NOT
        flattened-sorted-capped); ``degraded_map`` is a Counter of
        ``backend -> #DEGRADED pairs`` for §8's ``sources_degraded``. BOTH the
        normal AND the empty-queries paths emit the 2-tuple.
    """
    degraded_map: dict[str, int] = collections.Counter()   # init BEFORE the guard
    if not queries:
        return [], degraded_map                            # 2-tuple on the empty path too

    sort_by_recency = ranking_hint == "by_recency"

    # Build (term_idx, query, backend, max_n) task list. Function lookup
    # happens in _fetch_one_backend (call-time globals) so monkeypatch works.
    tasks = [
        _fetch_one_backend(q, name, n, term_idx=ti,
                           year_min=year_min, year_max=year_max,
                           degraded_map=degraded_map,
                           sort_by_recency=sort_by_recency)
        for ti, q in enumerate(queries)
        for (name, n) in CAPS.items()
    ]

    logger.info(
        "search_external_async: %d terms x %d backends = %d concurrent fetches",
        len(queries), len(CAPS), len(tasks),
    )

    # All (term, backend) pairs in parallel. One dead backend = one empty list.
    results = await asyncio.gather(*tasks)
    ext_raw = [p for batch in results for p in batch]
    logger.info("search_external_async: ext_raw=%d tagged nodes, degraded=%s "
                "(no dedup/sort here)", len(ext_raw), dict(degraded_map))
    return ext_raw, degraded_map


# ---------------- Backward-compat alias ----------------
# Phase 29 (2026-05-27) renamed the production entry point to
# ``search_external_async`` (async, no rule-based expansion). Existing
# tests in tests/test_mcp_server.py monkeypatch ``search_all`` to inject
# canned external candidates; we keep this name resolvable so test
# collection doesn't error out. Production code path no longer calls it.
def search_all(*args, **kwargs):
    """Deprecated. Use ``search_external_async`` directly. Kept as a
    resolvable symbol so legacy ``monkeypatch.setattr`` in tests doesn't
    crash at collection time. Calling this in production code is a bug."""
    raise NotImplementedError(
        "search_all() is deprecated; use search_external_async()."
    )


# ---------------- import-time wiring check (-O-safe) ----------------
# Deploy-time guard: MUST survive ``python -O`` (a bare ``assert`` is stripped
# under -O/PYTHONOPTIMIZE; with the §3 rewrite a genuinely-unwired backend would
# then raise a bare ``KeyError`` out of ``_fetch_one_backend`` into the
# ``asyncio.gather`` → hard crash of the whole external fan-out). NAME PRESENCE
# only — not call-signature compatibility. Placed at module bottom so every
# ``search_*`` function (including ``search_openalex`` defined here) is already
# bound when the check runs. CAPS key ``semantic_scholar`` must resolve to
# ``search_semantic_scholar``; a stray ``s2`` key would fire this RuntimeError.
if not all(f"search_{_n}" in globals() for _n in CAPS):   # noqa: E305
    raise RuntimeError("search.py: a backend in CAPS has no search_* function")
