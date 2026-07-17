"""Direct metadata fetchers by DOI / arxiv_id (not keyword search).

Used by AddService when the caller hands us an exact identifier.

**Metadata pipeline (D6', 2026-05)** — for DOI lookups, Crossref is the
authoritative source and goes FIRST: it's the publisher-submitted DOI
registry, so its title / year / venue / authors / canonical DOI are
treated as ground truth. Then Semantic Scholar fills the supplementary
fields (abstract, arxiv_id, full author names, citation_count, url)
that Crossref doesn't carry. SS values do NOT overwrite Crossref's
skeleton fields — only fill the gaps.

If Crossref doesn't have the DOI (rare; mostly pre-1996 papers and
some preprint DOIs), we fall back to SS as the primary, preserving
the old behavior.

For arxiv lookups, arXiv API stays authoritative (it owns its own
metadata) — same as before.
"""

from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from typing import Optional

import requests

from papervault import config
from .models import normalize_title
from .sources.arxiv import search_arxiv
from .sources.semantic_scholar import search_semantic_scholar


_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
# Polite-pool User-Agent for scholarly APIs (Crossref/OpenAlex). The contact e-mail
# is operator-configured (PAPERVAULT_CONTACT_EMAIL); never hardcode a personal address.
_UA = f"papervault/0.1 (mailto:{config.CONTACT_EMAIL})"
_ARXIV_RE = re.compile(r"^(?:arXiv:)?(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(v\d+)?$", re.I)
# Common copy-paste DOI wrappers: the ``doi:`` scheme and the resolver URL
# (with or without the ``dx.`` host and the scheme). Stripped to the bare
# ``10.xxxx/...`` form before the anchored ``_DOI_RE`` runs, so a pasted
# ``doi:10.../`` or ``https://doi.org/10...`` resolves instead of falling
# through to the fuzzy path (which would mis-resolve or not_found).
_DOI_PREFIX_RE = re.compile(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)", re.I)


def normalize_doi(s: str) -> str:
    """Strip a leading ``doi:`` / ``https://doi.org/`` / ``http://dx.doi.org/``
    wrapper so the bare ``10.xxxx/...`` form is returned (for the regex + the
    ``_by_doi`` index lookup). A bare DOI / non-DOI string is returned stripped
    of surrounding whitespace, unchanged otherwise."""
    return _DOI_PREFIX_RE.sub("", (s or "").strip())


def looks_like_doi(s: str) -> bool:
    return bool(_DOI_RE.match(normalize_doi(s)))


def looks_like_arxiv(s: str) -> bool:
    return bool(_ARXIV_RE.match(s.strip()))


def normalize_arxiv(s: str) -> str:
    m = _ARXIV_RE.match(s.strip())
    return m.group(1) if m else s.strip()


# ----------------------------- internal helpers ----------------------------


def _fetch_crossref_by_doi(doi: str) -> Optional[dict]:
    """Crossref skeleton lookup. Returns the trustworthy subset:
    title, year, venue, doi (canonical), authors (truncated), publication_types.
    Returns None on failure / 404."""
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=30,
            headers={"User-Agent": _UA},
        )
        if not r.ok:
            return None
        msg = r.json().get("message", {})
        title = " ".join(msg.get("title") or []) or ""
        authors = []
        for a in msg.get("author", []):
            name = " ".join(filter(None, [a.get("given"), a.get("family")]))
            if name:
                authors.append(name)
        year = None
        for k in ("issued", "published-online", "published-print"):
            parts = (msg.get(k) or {}).get("date-parts") or []
            if parts and parts[0]:
                year = parts[0][0]
                break
        # Crossref returns the canonical (case-corrected) DOI form via msg["DOI"].
        canonical_doi = (msg.get("DOI") or doi).strip()
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "doi": canonical_doi,
            "venue": (msg.get("container-title") or [""])[0],
            # Bibliographic locators. ``page`` is ABSENT for article-number
            # journals (ApJ / PRD / JHEP / most space-physics venues) — they use
            # ``article-number`` — so fall back to it, else pages stays blank.
            "volume": str(msg.get("volume") or "").strip(),
            "pages": str(msg.get("page") or msg.get("article-number") or "").strip(),
            "issue": str(msg.get("issue") or "").strip(),
            "publication_types": [msg.get("type", "")],
            # Crossref also has the resolver URL and citation count;
            # surface them but treat as overrideable by SS richer data.
            "url": msg.get("URL", "") or "",
            "citation_count": msg.get("is-referenced-by-count", 0) or 0,
            "_abstract_xref": msg.get("abstract", "") or "",  # often empty / XML
        }
    except Exception:
        return None


def crossref_enrich_lookup(doi: str) -> tuple[str, dict]:
    """3-state Crossref locator lookup for the enrich sweep (venue/volume/pages/issue).

    Returns one of:
      ("ok", {venue, volume, pages, issue}) — Crossref answered (200); fill + stamp.
      ("miss", {})        — definitive 404 / book / preprint with no journal data;
                            stamp enriched_at so it is NOT re-probed, fill nothing.
      ("transient", {})   — 429 / 5xx / timeout / connection error; do NOT stamp —
                            the next sweep retries it (fail-open).

    The miss-vs-transient split is load-bearing: without it an un-completable DOI
    spins every sweep forever, OR a Crossref outage permanently marks good rows
    un-enrichable. ``page`` falls back to ``article-number`` (ApJ/PRD/JHEP use it).
    """
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=30,
            headers={"User-Agent": _UA},
        )
    except Exception:
        return ("transient", {})
    if r.status_code == 404:
        return ("miss", {})
    if not r.ok:                      # 429 / 5xx / soft-block
        return ("transient", {})
    try:
        msg = r.json().get("message", {})
    except Exception:
        return ("transient", {})
    return ("ok", {
        "venue": (msg.get("container-title") or [""])[0],
        "volume": str(msg.get("volume") or "").strip(),
        "pages": str(msg.get("page") or msg.get("article-number") or "").strip(),
        "issue": str(msg.get("issue") or "").strip(),
    })


def _fetch_ss_by_doi(doi: str) -> Optional[dict]:
    """Semantic Scholar full metadata lookup. Returns rich data:
    abstract, arxiv_id, full author names, citation_count, paper_id.
    Returns None on failure / 404."""
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        params = {"fields": ("title,authors,year,abstract,citationCount,externalIds,"
                             "publicationTypes,venue,publicationVenue,url")}
        r = requests.get(url, params=params, timeout=30)
        if not r.ok:
            return None
        paper = r.json()
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        ext = paper.get("externalIds") or {}
        venue = paper.get("venue") or (paper.get("publicationVenue") or {}).get("name", "")
        return {
            "title": paper.get("title", ""),
            "authors": authors,
            "abstract": paper.get("abstract") or "",
            "year": paper.get("year"),
            "doi": doi,
            "arxiv_id": ext.get("ArXiv", ""),
            "paper_id": paper.get("paperId", ""),
            "venue": venue,
            "publication_types": paper.get("publicationTypes") or [],
            "citation_count": paper.get("citationCount", 0) or 0,
            "url": paper.get("url", "") or "",
        }
    except Exception:
        return None


def _merge_crossref_skeleton_with_ss(xref: dict, ss: Optional[dict]) -> dict:
    """Crossref fields are authoritative; SS fills gaps only.

    Returned dict has the same shape as the legacy ``fetch_by_doi``
    return value (title, authors, abstract, year, doi, arxiv_id,
    paper_id, venue, publication_types, citation_count, url, source).
    """
    out = {
        "title": xref["title"],
        "authors": xref["authors"],          # Crossref authoritative
        "year": xref["year"],
        "doi": xref["doi"],
        "venue": xref["venue"],
        # Crossref-authoritative locators (SS has no volume/pages, never overrides).
        "volume": xref.get("volume", ""),
        "pages": xref.get("pages", ""),
        "issue": xref.get("issue", ""),
        "publication_types": xref["publication_types"],
        # Fields Crossref doesn't reliably have — start blank, fill below
        "abstract": "",
        "arxiv_id": "",
        "paper_id": "",
        "url": xref.get("url", ""),
        "citation_count": xref.get("citation_count", 0),
        "source": "crossref",
    }

    if ss is None:
        # Crossref-only path. Salvage what we can.
        if xref.get("_abstract_xref"):
            out["abstract"] = xref["_abstract_xref"]
        return out

    # SS available — fill missing fields. Crossref skeleton is NOT
    # overwritten on conflict (title, authors, year, doi, venue).
    out["abstract"] = ss.get("abstract") or ""
    out["arxiv_id"] = ss.get("arxiv_id") or ""
    out["paper_id"] = ss.get("paper_id") or ""
    # Prefer SS url if Crossref didn't have one; SS often has the
    # publisher-facing landing page rather than dx.doi.org.
    if not out["url"] and ss.get("url"):
        out["url"] = ss["url"]
    # Citation count: prefer SS (richer, more current).
    if ss.get("citation_count"):
        out["citation_count"] = ss["citation_count"]
    # Prefer SS author names ONLY if last-name overlap is sane —
    # otherwise stick with Crossref. Author NAMES (with full given
    # names) are SS's strength; Crossref truncates to "F. Last".
    if _author_lastnames_match(xref["authors"], ss.get("authors", [])):
        out["authors"] = ss["authors"]
    # Otherwise leave out["authors"] = Crossref's (truncated but correct).
    out["source"] = "crossref+ss"
    return out


def _author_lastnames_match(xref_authors: list[str], ss_authors: list[str],
                            min_overlap: float = 0.7) -> bool:
    """Loose check: do the two author lists overlap on last-name set?

    Crossref returns "F. Lastname" or "Firstname Lastname". SS returns
    "Firstname Middlename Lastname". We compare last tokens, normalized.
    """
    if not xref_authors or not ss_authors:
        return False
    def lastnames(names: list[str]) -> set[str]:
        out = set()
        for n in names:
            toks = (n or "").strip().split()
            if toks:
                out.add(toks[-1].lower().strip(".,;"))
        return out
    x, s = lastnames(xref_authors), lastnames(ss_authors)
    if not x or not s:
        return False
    overlap = len(x & s) / max(len(x), len(s))
    return overlap >= min_overlap


# ----------------------------- public API ----------------------------------


def fetch_by_doi(doi: str) -> Optional[dict]:
    """Crossref-as-skeleton DOI lookup with SS gap-fill (D6').

    Returns the merged metadata dict, or None if neither backend
    has the DOI. See ``_merge_crossref_skeleton_with_ss`` for the
    merge policy.
    """
    doi = doi.strip()
    xref = _fetch_crossref_by_doi(doi)
    ss = _fetch_ss_by_doi(doi)
    if xref is not None:
        return _merge_crossref_skeleton_with_ss(xref, ss)
    if ss is not None:
        # Crossref doesn't have it (rare; mostly very old papers /
        # exotic preprint DOIs). Trust SS but mark the source so
        # downstream code can audit the lower-confidence path.
        ss["source"] = "semantic_scholar"
        return ss
    return None


# (_UA defined at top of module from config.CONTACT_EMAIL)


def fetch_abstract_by_doi(doi: str) -> str:
    """Best-effort ABSTRACT for a DOI. Tries OpenAlex (reconstruct the
    abstract_inverted_index — broadest coverage) then Semantic Scholar. Returns
    "" on miss. Network-bound and blocking: call it OFF the event loop / write
    lock (e.g. via asyncio.to_thread in a reconcile sweep), never inline in the
    ingest gate."""
    doi = (doi or "").strip()
    if not doi:
        return ""
    # OpenAlex by DOI -> reconstruct the inverted index.
    try:
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        params = {"mailto": config.CONTACT_EMAIL}
        r = requests.get(url, params=params, timeout=30, headers={"User-Agent": _UA})
        if r.status_code in (429, 409):
            time.sleep(2.0)
            r = requests.get(url, params=params, timeout=30, headers={"User-Agent": _UA})
        if r.ok:
            inv = (r.json() or {}).get("abstract_inverted_index") or {}
            if inv:
                pos = sorted((i, w) for w, idxs in inv.items() for i in idxs)
                ab = " ".join(w for _, w in pos).strip()
                if len(ab) >= 60:
                    return ab
    except Exception:
        pass
    # Semantic Scholar by DOI.
    try:
        ss = _fetch_ss_by_doi(doi)
        ab = ((ss or {}).get("abstract") or "").strip()
        if len(ab) >= 60:
            return ab
    except Exception:
        pass
    return ""


def fetch_by_arxiv(arxiv_id: str) -> Optional[dict]:
    """Look up arxiv metadata for a given id (with or without version)."""
    base = normalize_arxiv(arxiv_id)
    # search_arxiv supports id-style queries via "id:"
    try:
        results = search_arxiv(f"id:{base}", max_results=1)
        if results:
            p = results[0]
            return {
                "title": p["title"],
                "authors": p["authors"],
                "abstract": p["abstract"],
                "year": int(p["year"]) if str(p["year"]).isdigit() else None,
                "arxiv_id": p["arxiv_id"],
                "doi": "",
                "url": p["url"],
                "venue": "arXiv",
                "publication_types": ["JournalArticle"],
                "citation_count": 0,
                "source": "arxiv",
            }
    except Exception:
        pass

    # Fallback: also try Semantic Scholar's arxiv prefix
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{base}"
        r = requests.get(url, timeout=30,
                         params={"fields": "title,authors,year,abstract,citationCount,externalIds,venue,url"})
        if r.ok:
            paper = r.json()
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            ext = paper.get("externalIds") or {}
            return {
                "title": paper.get("title", ""),
                "authors": authors,
                "abstract": paper.get("abstract") or "",
                "year": paper.get("year"),
                "doi": ext.get("DOI", ""),
                "arxiv_id": ext.get("ArXiv", base),
                "venue": paper.get("venue") or "arXiv",
                "publication_types": [],
                "citation_count": paper.get("citationCount", 0) or 0,
                "url": paper.get("url", "") or "",
                "source": "semantic_scholar",
            }
    except Exception:
        pass
    return None


# ====================== by-title DOI resolution (safe) ======================
#
# A no-DOI in-domain stub (title + authors + year + abstract, no doi/arxiv)
# can never be enriched (reconcile._needs_enrich gates on bool(p.doi)) and can
# never be downloaded (the cascade + Sci-Hub/Anna tiers key on DOI/arxiv). The
# ONLY way to rescue it is to FIND its DOI by bibliographic search.
#
# By-title resolution is HEURISTIC and therefore DANGEROUS: a near-title match
# resolves the WRONG paper's DOI -> the library writes a wrong identity ->
# later fetches/cites the wrong paper -> silent corruption. A live probe of
# the heliophysics stub "The transport of cosmic rays in the heliosheath"
# (Strauss, 2012) confirmed Crossref's TOP-scored hit (32.9) was the WRONG
# paper ("Galactic Cosmic Rays in the Dynamic Heliosphere", Potgieter, 2011) —
# so Crossref score / rank is NOT a safe match signal.
#
# The resolver therefore IGNORES score/rank and instead requires a strict
# triple-agreement predicate on EVERY candidate, accepting a DOI ONLY when
# exactly ONE candidate passes. Otherwise it ABSTAINS (returns None). The
# common outcome for these low-signal stubs is abstention — that is by design;
# over-eager wrong-DOI writes are the failure mode we are engineered against.

# Title near-exact thresholds. SequenceMatcher ratio is the primary gate;
# token-Jaccard is a second, order-insensitive gate (catches a reordered or
# truncated title that happens to score a high SequenceMatcher ratio).
_TITLE_RATIO_MIN = 0.92
_TITLE_JACCARD_MIN = 0.90
# When the stub year is missing (the *nd rows), we cannot use the year gate,
# so demand a stricter title ratio to compensate (or abstain).
_TITLE_RATIO_MIN_NO_YEAR = 0.97
# A title shorter than this (normalized) is too generic to disambiguate by
# itself — abstain rather than risk a coincidental match. Mirrors
# download._try_arxiv_by_title's len<20 raw-title guard, applied to the
# normalized form.
_MIN_NORM_TITLE_LEN = 25
# First-author surname agreement: require the stub's first surname to be in the
# candidate's surname set, AND a sane overall last-name Jaccard.
_AUTHOR_JACCARD_MIN = 0.5


def _title_token_jaccard(a: str, b: str) -> float:
    """Order-insensitive token overlap on normalized titles (0.0..1.0)."""
    ta = set(normalize_title(a or "").split())
    tb = set(normalize_title(b or "").split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _name_to_surname(name: str) -> str:
    """Reduce ONE author name to a comparable surname token, normalized
    (lowercase, no trailing punctuation).

    Handles BOTH orderings so a stub stored surname-first matches a Crossref
    "given family" candidate:
      - "Given Family"   -> last whitespace token  ("A. P. Rouillard" -> rouillard)
      - "Family, Given"  -> last token of the part BEFORE the first comma
                            ("Chiappetta, Federica" -> chiappetta;
                             "De Rosa, M. L."        -> rosa, matching the
                             candidate "M. L. De Rosa" -> rosa)
    """
    s = (name or "").strip()
    if "," in s:
        s = s.split(",", 1)[0].strip()  # surname-first form: drop the given-name tail
    toks = s.split()
    if not toks:
        return ""
    return toks[-1].lower().strip(".,;")


def _first_surname(authors: list[str]) -> str:
    """Surname of the first author, normalized (see _name_to_surname)."""
    for a in authors or []:
        sn = _name_to_surname(a)
        if sn:
            return sn
    return ""


def _surnames(authors: list[str]) -> set[str]:
    out = set()
    for a in authors or []:
        sn = _name_to_surname(a)
        if sn:
            out.add(sn)
    return out


def _safe_title_match(stub_title: str, cand_title: str, *, stub_has_year: bool) -> bool:
    """Near-exact title agreement (NOT "relevant"). Both the SequenceMatcher
    ratio AND the token-Jaccard must clear the bar; when the stub has no year
    the ratio bar is raised to compensate for the missing year gate."""
    ratio = SequenceMatcher(None, normalize_title(stub_title),
                            normalize_title(cand_title)).ratio()
    jac = _title_token_jaccard(stub_title, cand_title)
    ratio_min = _TITLE_RATIO_MIN if stub_has_year else _TITLE_RATIO_MIN_NO_YEAR
    return ratio >= ratio_min and jac >= _TITLE_JACCARD_MIN


def _safe_author_match(stub_authors: list[str], cand_authors: list[str]) -> bool:
    """First-author surname of the stub must appear in the candidate's surname
    set, AND the last-name Jaccard must clear the bar. Reusing the loose
    last-name comparison tolerates the Crossref-vs-source name-format drift."""
    stub_first = _first_surname(stub_authors)
    if not stub_first:
        return False
    cand_surnames = _surnames(cand_authors)
    if stub_first not in cand_surnames:
        return False
    stub_surnames = _surnames(stub_authors)
    if not stub_surnames or not cand_surnames:
        return False
    jac = len(stub_surnames & cand_surnames) / len(stub_surnames | cand_surnames)
    return jac >= _AUTHOR_JACCARD_MIN


def _safe_year_match(stub_year: Optional[int], cand_year: Optional[int]) -> bool:
    """Year within +-1. If the STUB has no year, this gate is skipped (the
    caller compensates with a stricter title ratio); if the stub HAS a year but
    the candidate does not, that is a mismatch (we can't confirm)."""
    if stub_year is None:
        return True  # no-year stub: handled by the stricter title bar
    if cand_year is None:
        return False
    try:
        return abs(int(stub_year) - int(cand_year)) <= 1
    except (TypeError, ValueError):
        return False


def _crossref_title_candidates(title: str, first_surname: str,
                               rows: int = 5) -> Optional[list[dict]]:
    """Query Crossref by bibliographic title + author. Returns a list of
    normalized candidate dicts {doi, title, authors, year}, or None on a
    transient failure (so the caller can abstain WITHOUT stamping, letting a
    later sweep retry). An empty list means a clean answer with no usable hit.
    """
    params = {
        "query.bibliographic": title,
        "rows": str(rows),
        "select": "DOI,title,author,issued,container-title",
    }
    if first_surname:
        params["query.author"] = first_surname
    try:
        r = requests.get(
            "https://api.crossref.org/works",
            params=params,
            timeout=30,
            headers={"User-Agent": _UA},
        )
    except Exception:
        return None  # transient
    if not r.ok:
        return None  # 429 / 5xx / soft-block -> transient, abstain-and-retry
    try:
        items = r.json().get("message", {}).get("items", []) or []
    except Exception:
        return None
    out = []
    for it in items:
        doi = (it.get("DOI") or "").strip()
        if not doi:
            continue
        authors = []
        for a in it.get("author", []) or []:
            name = " ".join(filter(None, [a.get("given"), a.get("family")]))
            if name:
                authors.append(name)
        year = None
        for k in ("issued", "published-online", "published-print"):
            parts = (it.get(k) or {}).get("date-parts") or []
            if parts and parts[0]:
                year = parts[0][0]
                break
        out.append({
            "doi": doi,
            "title": " ".join(it.get("title") or []) or "",
            "authors": authors,
            "year": year,
        })
    return out


def resolve_doi_by_title(title: str, authors: list[str],
                         year: Optional[int]) -> Optional[str]:
    """Resolve a canonical DOI for a no-DOI stub by bibliographic search.

    Returns a canonical DOI string ONLY on a HIGH-confidence match, else None
    (ABSTAIN). The match predicate (ALL must hold for a candidate to be
    accepted, evaluated on every returned candidate — score/rank IGNORED):

      1. TITLE near-exact: normalized SequenceMatcher ratio >= 0.92 AND
         token-Jaccard >= 0.90 (raised to ratio >= 0.97 when the stub has no
         year). The stub's normalized title must be >= 25 chars (too short =
         abstain — too generic to disambiguate).
      2. FIRST-AUTHOR surname present in the candidate's surnames AND overall
         last-name Jaccard >= 0.5.
      3. YEAR within +-1 (skipped only when the stub year is None, compensated
         by the stricter title bar in (1)).

    If ZERO candidates pass, or MORE THAN ONE passes (ambiguous) -> abstain.
    A transient Crossref failure (network / 429 / 5xx) -> abstain (None) so the
    caller does NOT stamp a permanent "attempted" marker and a later sweep
    retries. NEVER decides based on the backend's relevance score.
    """
    title = (title or "").strip()
    if not title:
        return None
    norm = normalize_title(title)
    if len(norm) < _MIN_NORM_TITLE_LEN:
        return None  # too short / degenerate (e.g. session-code prefix) -> abstain
    stub_has_year = year is not None
    first_surname = _first_surname(authors)

    cands = _crossref_title_candidates(title, first_surname)
    if cands is None:
        return None  # transient -> abstain without stamping (caller retries)

    passed: list[str] = []
    for c in cands:
        if not _safe_title_match(title, c["title"], stub_has_year=stub_has_year):
            continue
        if not _safe_author_match(authors, c["authors"]):
            continue
        if not _safe_year_match(year, c.get("year")):
            continue
        passed.append(c["doi"])

    # Dedupe on the canonical (lowercased) DOI — the same DOI appearing twice
    # in the result set is NOT ambiguity. Keep the FIRST-seen casing for a
    # deterministic return.
    unique: dict[str, str] = {}
    for d in passed:
        unique.setdefault(d.lower(), d)
    if len(unique) != 1:
        return None  # zero (no match) OR >1 (ambiguous) -> abstain
    return next(iter(unique.values()))
