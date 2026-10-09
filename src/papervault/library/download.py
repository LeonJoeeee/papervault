"""Fetch PDFs into the library through the ordered ``_STRATEGIES`` cascade.

Operator overrides come first, followed by stored file URLs and arXiv.
Publisher, aggregator, and last-resort tiers follow; every hit passes the
same PDF identity verifier before saving.

Sci-Hub is opt-in via PAPER_PIPELINE_USE_SCIHUB=1.
NASA ADS requires ADS_API_TOKEN; silently skipped otherwise.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import sys as _sys
import time as time
from dataclasses import dataclass as dataclass
from html import unescape as unescape
from html.parser import HTMLParser as HTMLParser
from pathlib import Path
from types import ModuleType as _ModuleType
from typing import Callable, Optional
from urllib.parse import (
    parse_qs as parse_qs,
    unquote as unquote,
    urljoin as urljoin,
    urlsplit as urlsplit,
)

import requests as requests

from papervault.llm_routing import route

from .models import (
    DOWNLOAD_STATUS_FAILED,
    DOWNLOAD_STATUS_METADATA_ONLY,
    DOWNLOAD_STATUS_OK,
    DOWNLOAD_STATUS_PENDING,
    Paper,
)
from .store import Library

log = logging.getLogger(__name__)


from .download_sources import scihub as _scihub
from .download_sources._shared import (
    BROWSER_HEADERS as BROWSER_HEADERS,
    BROWSER_USER_AGENT as BROWSER_USER_AGENT,
    TIMEOUT as TIMEOUT,
    USER_AGENT as USER_AGENT,
    _gate_firecrawl_md as _gate_firecrawl_md,
    _is_pdf_bytes as _is_pdf_bytes,
    _load_stealthy_fetcher as _load_stealthy_fetcher,
    _strip_frontmatter as _strip_frontmatter,
)
from .download_sources.ads import _try_ads as _try_ads
from .download_sources.annas_archive import (
    _AnnasLink as _AnnasLink,
    _AnnasLinks as _AnnasLinks,
    _AnnasOption as _AnnasOption,
    _annas_download_options as _annas_download_options,
    _annas_get as _annas_get,
    _annas_resolve_option as _annas_resolve_option,
    _try_annas_archive_api as _try_annas_archive_api,
)
from .download_sources.arxiv import (
    _try_arxiv as _try_arxiv,
    _try_arxiv_by_title as _try_arxiv_by_title,
)
from .download_sources.core import _try_core as _try_core
from .download_sources.crossref import _try_crossref_link as _try_crossref_link
from .download_sources.elsevier import _try_elsevier_tdm as _try_elsevier_tdm
from .download_sources.europepmc import _try_europepmc as _try_europepmc
from .download_sources.firecrawl import _try_firecrawl_text_fallback as _try_firecrawl_text_fallback
from .download_sources.inspire import _try_inspire as _try_inspire
from .download_sources.iopscience import _try_iopscience_direct as _try_iopscience_direct
from .download_sources.known_file_url import (
    _known_file_url as _known_file_url,
    _try_known_file_url as _try_known_file_url,
)
from .download_sources.mdpi import _try_mdpi_scrapling as _try_mdpi_scrapling
from .download_sources.openalex import _try_openalex as _try_openalex
from .download_sources.publisher import (
    _CITATION_PDF_URL_RE as _CITATION_PDF_URL_RE,
    _CITATION_PDF_URL_RE_REV as _CITATION_PDF_URL_RE_REV,
    _try_citation_pdf_url as _try_citation_pdf_url,
    _try_cloudscraper_publisher as _try_cloudscraper_publisher,
    _try_curl_impersonate as _try_curl_impersonate,
)
from .download_sources.researchgate import (
    _RG_PDF_LINK_RE as _RG_PDF_LINK_RE,
    _RG_PUBLICATION_RE as _RG_PUBLICATION_RE,
    _try_researchgate as _try_researchgate,
)
from .download_sources.scihub import (
    _SCIHUB_CACHE_TTL as _SCIHUB_CACHE_TTL,
    _SCIHUB_DISCOVERY_SOURCES as _SCIHUB_DISCOVERY_SOURCES,
    _SCIHUB_DOMAIN_RE as _SCIHUB_DOMAIN_RE,
    _SCIHUB_FALLBACK_DOMAINS as _SCIHUB_FALLBACK_DOMAINS,
    _SCIHUB_PDF_PATTERNS as _SCIHUB_PDF_PATTERNS,
    _discover_scihub_mirrors as _discover_scihub_mirrors,
    _scihub_one_mirror as _scihub_one_mirror,
    _try_scihub as _try_scihub,
)
from .download_sources.semantic_scholar import _try_semantic_scholar_oa as _try_semantic_scholar_oa
from .download_sources.ssrn import _SSRN_DELIVERY_RE as _SSRN_DELIVERY_RE, _try_ssrn as _try_ssrn
from .download_sources.unpaywall import _try_unpaywall as _try_unpaywall
from .download_sources.url_overrides import _try_url_overrides as _try_url_overrides
from .download_sources.web_search import _try_web_search as _try_web_search
from .download_sources.wiley import _try_wiley_tdm as _try_wiley_tdm
from .download_sources.zenodo import _try_zenodo as _try_zenodo


def _atomic_save(dest: Path, content: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(dest)


def _normalize_for_match(s: str) -> str:
    """Lowercase + strip non-alphanumerics + collapse whitespace.

    Used by the PDF-verification step. Designed so that "k-PINN" and
    "k pinn", or "γ-Ray" rendered as "g-Ray" by some PDF extractors,
    or running-head reformatting, all normalize to comparable strings.
    """
    s = (s or '').lower()
    s = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in s)
    return ' '.join(s.split())


_VERIFY_PROMPT = (
    "You verify whether a downloaded PDF is the EXACT paper that was requested.\n"
    "You are given the requested paper's metadata and the first pages of the "
    "PDF's extracted text (its title / authors / abstract are here).\n"
    "It is a MATCH if the PDF is the SAME WORK as the requested metadata — allow "
    "minor formatting differences, a running-head rephrasing of the title, "
    "abbreviated / transliterated / reordered author names, and preprint-vs-"
    "journal versions of the same paper.\n"
    "It is NOT a match if the PDF is a DIFFERENT paper (its title AND authors are "
    "a different work), a publisher landing / search / recommendation page, a "
    "paywall stub, or a different paper that merely CITES the requested one.\n"
    'Reply with ONLY a JSON object: {"match": true|false, "reason": "<short>"}'
)


def _llm_verify_identity(head: str, paper: Paper, *, llm=None,
                         attempts: int = 3) -> tuple[bool, str]:
    """Ask a cheap LLM (MiMo v2.5) whether ``head`` (the PDF's first pages) is the
    paper described by ``paper``'s metadata.

    LLM-robustness (every LLM caller must handle failure): RETRY up to
    ``attempts`` times on BOTH a call exception (API/transient — on top of the
    KeyPool's own key-failover) AND a malformed / JSON-less reply (a fresh ask
    often returns valid JSON). Only after the retries are exhausted do we
    fail-OPEN (accept) so a persistently-flaky API never discards a real
    download. A clean ``{"match": ...}`` reply short-circuits immediately."""
    import json
    authors = ", ".join((paper.authors or [])[:8])
    user = (f"Requested title: {paper.title}\n"
            f"Requested authors: {authors}\n"
            f"Requested year: {paper.year or '?'}\n\n"
            f"--- first pages of the PDF ---\n{head}")
    if llm is None:
        try:
            from .llm import get_llm

            # PDF metadata verification uses flash (PAPERVAULT_LLM_VERIFY overrides).
            llm = get_llm(model=route("verify"))
        except Exception as exc:
            return True, f"verify_llm_unavailable: {repr(exc)[:60]}"
    msgs = [{"role": "system", "content": _VERIFY_PROMPT},
            {"role": "user", "content": user}]
    last = "no_attempt"
    for _ in range(max(1, attempts)):
        try:
            raw = llm.call(msgs)
        except Exception as exc:
            last = f"call_error: {repr(exc)[:50]}"
            continue                                   # retry on API/transient error
        m = re.search(r"\{.*\}", raw or "", re.S)
        if m:
            try:
                v = json.loads(m.group(0))
            except json.JSONDecodeError:
                v = None
            if isinstance(v, dict) and "match" in v:
                ok = bool(v["match"])
                reason = str(v.get("reason", ""))[:120]
                return ok, (f"llm_match: {reason}" if ok else f"llm_mismatch: {reason}")
        last = "malformed_output"                      # retry on missing/bad JSON
    return True, f"verify_fail_open_after_{max(1, attempts)}_tries ({last})"


def _verify_pdf_matches_metadata(data: bytes, paper: Paper, *, llm=None) -> tuple[bool, str]:
    """Verify the downloaded PDF actually IS the paper we asked for.

    Catches sci-hub serving a different paper at a borderline DOI, a scraper
    returning a publisher landing page, wrong-revision artifacts, etc.

    Accept-on-doubt valves short-circuit FIRST (so we never spend an LLM call on
    an undecidable PDF): title too short to verify, pypdf can't parse, < 50 chars
    of text (scanned/image — downstream OCR handles it), or page-1 custom-font
    garbage. Otherwise a cheap LLM (MiMo v2.5) judges identity on the first 2
    pages. Reject -> caller treats it as a miss and tries the next cascade source.
    """
    title = (paper.title or '').strip()
    if len(title) < 15:
        return True, 'title too short to verify'

    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages_text = []
        for page in reader.pages[:2]:        # first 2 pages — title/authors/abstract
            try:
                pages_text.append(page.extract_text() or '')
            except Exception:
                pages_text.append('')
        head = '\n'.join(pages_text)[:6000]
    except Exception as exc:
        return True, f'pypdf failed: {repr(exc)[:60]}'

    if len(head) < 50:
        return True, 'no extractable text (likely scanned)'

    # Custom-font garbage: very low alpha ratio AND no English stopwords on
    # page 1 (old arxiv preprints in custom Type-1 fonts pypdf can't decode).
    p1 = pages_text[0] if pages_text else ''
    if len(p1) >= 200:
        p1_alpha = sum(1 for c in p1 if c.isalpha()) / len(p1)
        if p1_alpha < 0.55:
            p1_lower = p1.lower()
            stopwords = ('the ', ' of ', ' and ', ' for ', ' with ', ' in ', ' we ')
            if sum(p1_lower.count(sw) for sw in stopwords) < 3:
                return True, (f'page 1 garbage (alpha={p1_alpha:.0%}, custom '
                              f'font) — cannot verify')

    return _llm_verify_identity(head, paper, llm=llm)


# --------------------- concurrent group dispatcher -----------------------
#
# Several aggregator tiers (unpaywall/semantic_scholar_oa/openalex/core)
# walk the same shape: query an API for an OA PDF URL, then GET it. Run
# serially they cost 8-12s each per miss; first-hit-wins concurrency
# collapses that to one API's worth of latency. Same trick for the
# domain-specific indexes (inspire/ads/europepmc/zenodo).
#
# Each member is run in its own thread; exceptions are swallowed (treated
# as miss). Pending futures are not cancelled — Python threads can't be
# preempted — but as soon as one returns a usable PDF we stop waiting.

def _safe_call(fn: Callable[[Paper], Optional[bytes]],
                paper: Paper) -> Optional[bytes]:
    """Wrap a tier callable; convert any exception into a miss."""
    try:
        return fn(paper)
    except Exception:
        return None


def _try_concurrent_first_hit(
    paper: Paper,
    members: list[tuple[str, Callable[[Paper], Optional[bytes]]]],
    timeout: int = 60,
) -> Optional[bytes]:
    """Run multiple tier callables in parallel; return the first non-None
    result. Members that miss or raise are ignored. Verification of the
    returned bytes happens in the main download loop (same as serial)."""
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(2, len(members))) as pool:
        futures = {pool.submit(_safe_call, fn, paper): name
                    for name, fn in members}
        try:
            for fut in concurrent.futures.as_completed(
                    futures, timeout=timeout):
                data = fut.result()
                if data:
                    return data
        except concurrent.futures.TimeoutError:
            pass
    return None


def _try_oa_aggregators(paper: Paper) -> Optional[bytes]:
    """Concurrent first-hit over four open-access metadata aggregators.

    Replaces four serial cascade tiers (unpaywall, semantic_scholar_oa,
    openalex, core) with a single concurrent group. Saves ~8-10 seconds
    on every paper where none of the four has a PDF (the common case in
    Round 1 — only 4/82 hit unpaywall, the others all missed all four)."""
    return _try_concurrent_first_hit(paper, [
        ("unpaywall", _try_unpaywall),
        ("semantic_scholar_oa", _try_semantic_scholar_oa),
        ("openalex", _try_openalex),
        ("core", _try_core),
    ])


def _try_domain_aggregators(paper: Paper) -> Optional[bytes]:
    """Concurrent first-hit over four domain-specific paper indexes.

    inspire (HEP), ads (astrophysics), europepmc (biomed), zenodo (data
    repo). Each indexes papers in a different field, so for any given
    paper at most one will have content; running them serially wastes
    ~6-8s on the misses."""
    return _try_concurrent_first_hit(paper, [
        ("inspire", _try_inspire),
        ("ads", _try_ads),
        ("europepmc", _try_europepmc),
        ("zenodo", _try_zenodo),
    ])


# Ordered cascade. Sorted by Round 1 hit-rate (high first), with
# concurrent groups consolidating low-hit-but-cheap aggregators.
# Each entry = (source_tag, callable). Group dispatchers count as one
# tier in this list but run their members concurrently.
_STRATEGIES = [
    # === Free, fast, operator-controlled / preprint ===
    ("url_override", _try_url_overrides),         # 0% but free, operator escape hatch
    ("known_file_url", _try_known_file_url),        # stored files, including identifierless papers
    ("arxiv", _try_arxiv),                          # 24/82 R1 (preprint primary)

    # === Publisher-direct (mihomo proxy makes these high-hit) ===
    ("iopscience_direct", _try_iopscience_direct),  # 15/82 R11 — bypasses Radware
    ("crossref_tm", _try_crossref_link),            # 9/82 R1 after mihomo
    ("citation_pdf_url", _try_citation_pdf_url),    # publisher Highwire meta tag

    # === Token-gated publisher TDM APIs ===
    ("wiley_tdm", _try_wiley_tdm),                  # free academic registration
    ("elsevier_tdm", _try_elsevier_tdm),            # institutional ELSEVIER_TDM_API_KEY

    # === Concurrent OA aggregators (4 sources, first-hit returns) ===
    ("oa_aggregators", _try_oa_aggregators),        # unpaywall/s2_oa/openalex/core

    # === Gray-area, very high R1 hit rate ===
    ("scihub", _try_scihub),                        # 31/82 R1
    ("annas_archive", _try_annas_archive_api),      # gated by ANNAS_ARCHIVE_API_KEY

    # === Concurrent domain-specific indexes ===
    ("domain_aggregators", _try_domain_aggregators),  # inspire/ads/europepmc/zenodo

    # === Heuristics / fragile last-resort scrapers ===
    ("curl_impersonate", _try_curl_impersonate),    # Akamai TLS fingerprint
    ("ssrn", _try_ssrn),
    ("arxiv_by_title", _try_arxiv_by_title),
    ("cloudscraper", _try_cloudscraper_publisher),  # Cloudflare interstitial
    ("mdpi_scrapling", _try_mdpi_scrapling),        # 3/3 MDPI hits via Akamai bm-verify trick (R11)
    ("researchgate", _try_researchgate),
    ("web_search", _try_web_search),                # DDG title+filetype:pdf, last-ditch
]


def _download_skip_reason(source: str, paper: Paper) -> Optional[str]:
    """Mirror the tiers' no-network prerequisites for manifest bookkeeping.

    Check immediately before each call: tiers can mutate identifiers (arXiv
    clears its ID on a 404), so checking afterwards can mislabel a real miss.
    Keep these checks in sync with tier early returns; concurrent groups skip
    only when every member skips. Unknown tiers retain the miss behavior.
    """
    if source == "known_file_url":
        if not (paper.url or "").strip():
            return "missing_file_url"
        if not _known_file_url(paper):
            return "not_file_url"
    credential = {
        "wiley_tdm": "WILEY_TDM_TOKEN",
        "elsevier_tdm": "ELSEVIER_TDM_API_KEY",
        "annas_archive": "ANNAS_ARCHIVE_API_KEY",
    }.get(source)
    if credential and not os.environ.get(credential, "").strip():
        return "missing_credentials"
    if source == "scihub":
        if os.environ.get("PAPER_PIPELINE_USE_SCIHUB", "").strip() not in {"1", "true", "yes"}:
            return "disabled"
        if not (paper.doi or paper.arxiv_id or paper.url):
            return "missing_identifier"
    if source == "arxiv" and not paper.arxiv_id:
        return "missing_arxiv_id"
    if source in {"crossref_tm", "citation_pdf_url", "wiley_tdm", "elsevier_tdm",
                  "annas_archive", "curl_impersonate", "cloudscraper"} and not paper.doi:
        return "missing_doi"
    prefixes = {
        "iopscience_direct": ("10.3847/", "10.1088/"),
        "mdpi_scrapling": ("10.3390/",),
        "ssrn": ("10.2139/ssrn.",),
    }.get(source)
    if prefixes and not (paper.doi or "").startswith(prefixes):
        return "not_applicable"
    tdm_prefixes = {
        "wiley_tdm": ("10.1002", "10.1029", "10.1111", "10.1046"),
        "elsevier_tdm": ("10.1016",),
    }.get(source)
    if tdm_prefixes and paper.doi.split("/", 1)[0] not in tdm_prefixes:
        return "not_applicable"
    if source == "ssrn" and not paper.doi.split("ssrn.")[-1].strip():
        return "missing_identifier"
    if source == "domain_aggregators" and not (paper.doi or paper.arxiv_id):
        return "missing_identifier"
    if source == "oa_aggregators" and not paper.doi:
        if not (os.environ.get("CORE_API_KEY", "").strip()
                and paper.title and len(paper.title) >= 20):
            return "no_applicable_member"
    if source == "arxiv_by_title":
        if paper.arxiv_id:
            return "already_has_arxiv_id"
        title = (paper.title or "").strip()
        if len(title) < 20 or not re.sub(r'["\\?<>]', '', title)[:100]:
            return "insufficient_title"
    if source == "web_search" and (not paper.title or len(paper.title) < 20):
        return "insufficient_title"
    if source == "researchgate" and len((paper.title or "").strip()) < 20 and not (paper.doi or "").strip():
        return "missing_identifier"
    if source == "url_override":
        if not (paper.doi or paper.arxiv_id):
            return "missing_identifier"
        from .services.concurrency import _vault_path
        try:
            overrides = json.loads((Path(_vault_path()) / "url_overrides.json").read_text())
        except (FileNotFoundError, ValueError):
            return "missing_url_override"
        if not (overrides.get(paper.doi or "") or overrides.get(paper.arxiv_id or "")):
            return "missing_url_override"
    dependency = {
        "curl_impersonate": ("curl_cffi", "requests"),
        "cloudscraper": ("cloudscraper", None),
        "annas_archive": ("scrapling.fetchers", "StealthyFetcher"),
        "mdpi_scrapling": ("scrapling.fetchers", "StealthyFetcher"),
        "researchgate": ("scrapling.fetchers", "StealthyFetcher"),
    }.get(source)
    if dependency:
        module, attribute = dependency
        try:
            imported = __import__(module, fromlist=[attribute] if attribute else [])
            if attribute:
                getattr(imported, attribute)
        except (ImportError, AttributeError):
            return "missing_dependency"
    return None


def download_paper(paper: Paper, library: Library) -> bool:
    """Download a single paper's PDF if missing.

    Returns True iff a PDF binary was successfully obtained and written to
    disk. False can mean either total failure OR a successful firecrawl
    text-only fallback — callers must inspect ``paper.download_status`` +
    disk facts (or just call ``services.classify.classify``). A firecrawl
    fallback win leaves ``download_status="ok"`` +
    ``download_source="firecrawl"`` with an md on disk but NO PDF (so
    ``has_pdf`` is False and classify routes it back to DOWNLOAD to hunt
    the real PDF); ``DOWNLOAD_STATUS_FAILED`` / ``DOWNLOAD_STATUS_METADATA_ONLY``
    are the terminal misses.

    Idempotent: if the file exists, returns True without re-fetching.
    """
    dest = library.pdf_path(paper.key)
    if library.has_pdf(paper.key):
        return True

    # ── Re-gate an un-gated HISTORICAL firecrawl md BEFORE hunting a PDF ──
    # (issue #1, manifestation 1). The ~48 migrated firecrawl md predate the
    # completeness gate, so they sit on disk un-gated. classify routes them
    # here (rule 3: ¬has_pdf ∧ firecrawl-md ∧ ¬exhausted → DOWNLOAD). The
    # design's mechanism to honor the §4.3 invariant "md on disk ⟺ gated" is
    # the firecrawl re-entry at the BOTTOM of this function — but that re-entry
    # is only reached if all PDF tiers MISS. If a tier lands a real PDF first
    # (return True below), the re-gate is skipped: disk then has PDF + un-gated
    # md → classify rule 2 → TERMINAL → serve-safety hands out the never-gated
    # stub as full text PERMANENTLY (rule 2 / extract_md both refuse to re-OCR,
    # D5). So we must re-gate the historical md FIRST, independent of any tier
    # outcome: a later tier-hit can then only ever co-exist with an ALREADY
    # gated firecrawl md, preserving the invariant for rule 2.
    if (
        library.has_extract(paper.key, "md")
        and library.md_source(paper.key) == "firecrawl"
        and not paper.firecrawl_pdf_hunt_exhausted
    ):
        try:
            on_disk = library.md_path(paper.key).read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            on_disk = None
        if on_disk is not None:
            body = _strip_frontmatter(on_disk)
            # PASS → md stays on disk gated + stamped exhausted; FAIL → md
            # deleted/neutralised + paper demoted to a terminal status.
            if not _gate_firecrawl_md(paper, library, body):
                # Gate FAIL: the paper is now terminal (metadata_only/failed)
                # with no md on disk. Do NOT hunt a PDF under a terminal status
                # (classify rule 1 would skip it anyway, and writing status=ok
                # over the terminal demotion would lie). Short-circuit.
                return False
            # Gate PASS: md is now gated + ``firecrawl_pdf_hunt_exhausted`` set.
            # Fall through to hunt the real PDF (the rule-3 DOWNLOAD intent). If
            # a tier lands one, the now-GATED md co-exists with the PDF and
            # classify rule 2 rests it served as the gated md — invariant held.

    for source, strategy in _STRATEGIES:
        try:
            skip_reason = _download_skip_reason(source, paper)
            data = strategy(paper)
        except Exception as exc:
            library.log({"event": "download_error", "key": paper.key,
                         "source": source, "error": repr(exc)[:200]})
            continue
        if data:
            ok, verify_reason = _verify_pdf_matches_metadata(data, paper)
            if not ok:
                library.log({"event": "download_pdf_mismatch", "key": paper.key,
                             "source": source, "size": len(data),
                             "reason": verify_reason})
                continue
            _atomic_save(dest, data)
            paper.pdf_path = str(dest.relative_to(library.root))
            # D7: status routes, source labels — never fuse them into one cell.
            paper.download_status = DOWNLOAD_STATUS_OK
            paper.download_source = source
            library.log({"event": "downloaded", "key": paper.key, "source": source,
                         "size": len(data), "verify": verify_reason})
            return True
        if skip_reason:
            library.log({"event": "download_skip", "key": paper.key,
                         "source": source, "reason": skip_reason})
        else:
            library.log({"event": "download_miss", "key": paper.key, "source": source})

    # All PDF tiers missed. Try the firecrawl text-only fallback as
    # last resort. On success it writes extracts/md/{key}.md directly and
    # sets paper.download_status = "ok" + download_source = "firecrawl"
    # (D7); we still return False because no PDF binary was obtained.
    if _try_firecrawl_text_fallback(paper, library):
        # md on disk, no PDF binary → return False (no PDF) but status is
        # now "ok" + source firecrawl. The download queue inspects status to
        # decide chaining.
        return False

    # _try_firecrawl_text_fallback returned False. Two sub-cases:
    #   (a) it already SETTLED the status itself — the firecrawl re-entry /
    #       fresh gate path reached _gate_firecrawl_md, which either demoted to
    #       a terminal (gate FAIL: metadata_only/failed) or, for a non-firecrawl
    #       marker/dots md, re-asserted ``ok`` (issue #3: don't clobber a paper
    #       that HAS full text on disk). In both, status is already off
    #       ``pending`` and correct — re-deriving here would (i) clobber the
    #       valid ``ok`` from #3, and (ii) for the gate-FAIL path emit a SECOND
    #       redundant audit log of the SAME demotion (issue #4). So short-circuit.
    #   (b) firecrawl never produced/settled anything (no endpoint, no DOI,
    #       HTTP/too-short) — status is still ``pending`` and WE derive the
    #       no-full-text terminal below.
    if paper.download_status != DOWNLOAD_STATUS_PENDING:
        return False

    # Even firecrawl missed. Distinguish a "true zero" (no metadata, just
    # an identifier we couldn't resolve) from a "partial victory" (we have
    # rich metadata — DOI/title/authors/year/abstract — even though the
    # full body is paywalled). For citation purposes the partial-win state
    # is genuinely useful, so flag it with a dedicated status that callers
    # can render differently from outright failure.
    if (paper.abstract or "").strip():
        paper.download_status = DOWNLOAD_STATUS_METADATA_ONLY
        library.log({"event": "download_metadata_only", "key": paper.key,
                     "doi": paper.doi, "arxiv_id": paper.arxiv_id,
                     "abstract_chars": len(paper.abstract or "")})
        return False
    paper.download_status = DOWNLOAD_STATUS_FAILED
    library.log({"event": "download_failed", "key": paper.key,
                 "doi": paper.doi, "arxiv_id": paper.arxiv_id})
    return False


class _DownloadModule(_ModuleType):
    @property
    def _scihub_mirrors_cache(self):
        return _scihub._scihub_mirrors_cache

    @_scihub_mirrors_cache.setter
    def _scihub_mirrors_cache(self, value):
        _scihub._scihub_mirrors_cache = value


_sys.modules[__name__].__class__ = _DownloadModule
