"""Fetch PDFs into the library, trying arXiv → Unpaywall → OpenAlex →
Inspire-HEP → NASA ADS → CrossRef-tm → citation_pdf_url (Highwire meta) →
SSRN → EuropePMC → Zenodo → arxiv-by-title → ResearchGate → Sci-Hub in
order.

Sci-Hub is opt-in via PAPER_PIPELINE_USE_SCIHUB=1.
NASA ADS requires ADS_API_TOKEN; silently skipped otherwise.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

import requests

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


USER_AGENT = "paper-pipeline/0.1 (research; email lib@example.invalid)"
# Browser-like UA for sites that block obvious bots (ResearchGate, some
# publisher CDNs). Used only for those tiers — most APIs prefer the
# library UA above.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
# Full browser-style headers: many publisher CDNs (MDPI, SSRN, Springer)
# return 403 to bare User-Agent strings without Accept / Accept-Language /
# Sec-Fetch-* hints. Empirically MDPI 403s on UA-only Chrome request but
# 200s with the full set.
BROWSER_HEADERS = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}
TIMEOUT = 30


def _atomic_save(dest: Path, content: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(dest)


def _is_pdf_bytes(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


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

            # verify role (issue #8): default = the current verify model
            # (PAPER_PIPELINE_VERIFY_MODEL or openai/mimo-v2.5); operator-overridable
            # via PAPERVAULT_LLM_VERIFY. Model only — the library plane sends no thinking param.
            llm = get_llm(model=route("verify")[0])
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


def _try_web_search(paper: Paper) -> Optional[bytes]:
    """Last-ditch generic search: DuckDuckGo HTML for ``"<title>" filetype:pdf``.

    Restricted by ``PAPER_LIBRARY_SEARCH_BLOCKLIST`` (comma-sep host
    suffixes; appended as ``-site:<host>`` to the query). Catches OA
    copies hosted on places no aggregator indexes — journal mirrors,
    institutional repositories, conference sites — that S2 / Unpaywall /
    OpenAlex / CORE all missed.

    Verifies the candidate is a real PDF via ``_is_pdf_bytes``. Title-
    correctness verification happens downstream in
    ``_verify_pdf_matches_metadata`` (so we don't accidentally save a
    paper that merely cites ours).
    """
    if not paper.title or len(paper.title) < 20:
        return None
    import urllib.parse
    blocklist = [d.strip() for d in os.environ.get(
        "PAPER_LIBRARY_SEARCH_BLOCKLIST", "").split(",") if d.strip()]
    q_parts = [f'"{paper.title}"', "filetype:pdf"]
    q_parts.extend(f"-site:{d}" for d in blocklist)
    q = " ".join(q_parts)
    search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(q)}"
    try:
        r = requests.get(search_url, timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
        # status 202 = DDG anti-bot CAPTCHA gate ("complete the challenge").
        # Treat as transient block — return None so cascade continues.
        if r.status_code == 202 or not r.ok:
            return None
        # Defense-in-depth: detect captcha keywords in body.
        if "Unfortunately, bots use DuckDuckGo" in r.text:
            return None
    except Exception:
        return None
    raw_results = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"', r.text)
    candidates: list[str] = []
    seen: set[str] = set()
    for ddg_link in raw_results[:5]:
        m = re.search(r'uddg=([^&]+)', ddg_link)
        if not m:
            continue
        actual = urllib.parse.unquote(m.group(1))
        # Defense in depth: even if blocklist op missed in DDG, drop here
        if any(d in actual for d in blocklist):
            continue
        if actual in seen:
            continue
        seen.add(actual)
        candidates.append(actual)
    for url in candidates:
        try:
            pdf = requests.get(url, timeout=TIMEOUT,
                               headers=BROWSER_HEADERS, allow_redirects=True)
            if pdf.ok and _is_pdf_bytes(pdf.content):
                return pdf.content
        except Exception:
            continue
    return None


def _try_url_overrides(paper: Paper) -> Optional[bytes]:
    """User-curated DOI/arxiv -> URL overrides. Tried first.

    Format: ``$PAPER_LIBRARY_PATH/url_overrides.json``::

        {"<doi-or-arxiv-id>": "<full-PDF-url>", ...}

    Use case: author lab pages hosting their own PDFs, institutional
    repository links, any case where the operator already knows the
    exact PDF URL. Re-read on every call so manual edits take effect
    without restarting the daemon.
    """
    if not paper.doi and not paper.arxiv_id:
        return None
    # Resolve the vault dir through the SAME live-env + config path as the rest of the
    # library (services.concurrency._vault_path: PAPERVAULT_VAULT → PAPER_LIBRARY_PATH →
    # config.VAULT_PATH, expanduser'd) so a relocated vault's overrides are still read,
    # instead of a stale ~/paper-vault default that diverges from Library.root.
    from papervault.library.services.concurrency import _vault_path
    overrides_path = Path(_vault_path()) / "url_overrides.json"
    try:
        overrides = json.loads(overrides_path.read_text())
    except (FileNotFoundError, ValueError):
        return None
    url = overrides.get(paper.doi or "") or overrides.get(paper.arxiv_id or "")
    if not url:
        return None
    try:
        r = requests.get(url, timeout=TIMEOUT,
                         headers=BROWSER_HEADERS, allow_redirects=True)
        if r.ok and _is_pdf_bytes(r.content):
            return r.content
    except Exception:
        pass
    return None


def _try_arxiv(paper: Paper) -> Optional[bytes]:
    if not paper.arxiv_id:
        return None
    arxiv_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}, allow_redirects=True)
    if r.ok and _is_pdf_bytes(r.content):
        return r.content
    return None


def _try_unpaywall(paper: Paper) -> Optional[bytes]:
    """Unpaywall: returns multiple ``oa_locations``; try them all, with
    BROWSER_HEADERS as a second pass for publishers that 403 bare UAs.

    The previous version only tried the first location's ``url_for_pdf``,
    which gave up immediately when (a) the field was missing — common for
    repository entries that only have ``url`` — or (b) the URL was blocked
    by a publisher CDN (MDPI's Akamai, RSC's Cloudflare). Iterating + the
    browser header retry recovers some of these.
    """
    if not paper.doi:
        return None
    email = os.environ.get("UNPAYWALL_EMAIL", "research@example.invalid")
    api = f"https://api.unpaywall.org/v2/{paper.doi}?email={email}"
    try:
        meta = requests.get(api, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}).json()
    except Exception:
        return None

    # Collect all candidate URLs across best_oa_location + oa_locations
    candidates: list[str] = []
    seen: set[str] = set()
    def _add(loc: dict | None) -> None:
        if not loc:
            return
        for field in ("url_for_pdf", "url"):
            u = loc.get(field)
            if u and u not in seen:
                seen.add(u)
                candidates.append(u)
    _add(meta.get("best_oa_location"))
    for loc in (meta.get("oa_locations") or []):
        _add(loc)
    if not candidates:
        return None

    # Two-pass: simple UA first, full browser headers second
    for url in candidates:
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    for url in candidates:
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers=BROWSER_HEADERS,
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None


# Bootstrap list of discovery sources; each is fetched, scanned for
# sci-hub.<tld> domains, then liveness-checked. Order doesn't matter
# beyond "first response is fastest".
_SCIHUB_DISCOVERY_SOURCES = (
    "https://sci-hub.41610.org/",   # PyPaperBot's source — explicit mirror index
    "https://en.wikipedia.org/wiki/Sci-Hub",
    "https://lovescihub.wordpress.com/",
    "https://sci-hub.ru/",       # any alive mirror's footer lists siblings
    "https://sci-hub.ee/",
)

# If all discovery sources fail, fall back to these historical mirrors.
_SCIHUB_FALLBACK_DOMAINS = (
    "sci-hub.ru", "sci-hub.ee", "sci-hub.ren",
    "sci-hub.wf", "sci-hub.box", "sci-hub.41610.org",
)

# Cache: (timestamp_set_at, list_of_alive_mirror_urls)
_scihub_mirrors_cache: tuple[float, list[str]] = (0.0, [])
_SCIHUB_CACHE_TTL = 6 * 3600   # 6 hours

# Regex: only match valid TLD-like suffixes 2-8 lowercase chars,
# followed by word boundary. Avoids matching accidental text.
_SCIHUB_DOMAIN_RE = re.compile(r"\bsci-hub\.[a-z]{2,8}(?:\.[a-z]{2,8})?\b")


def _discover_scihub_mirrors() -> list[str]:
    """Discover alive Sci-Hub mirrors from multiple sources, cache for TTL.

    Process:
      1. If cache fresh, return cached list.
      2. Fetch each source URL; regex out sci-hub.<tld> domains.
      3. If discovery yielded nothing, use _SCIHUB_FALLBACK_DOMAINS.
      4. Liveness-check each domain via HEAD request (timeout 5s).
      5. Return alive ones (as full https://<domain> URLs).
      6. Update cache. If liveness yielded nothing, return all discovered
         (last-resort: try them anyway in the actual download).
    """
    global _scihub_mirrors_cache
    now = time.time()
    cached_at, cached = _scihub_mirrors_cache
    if cached and (now - cached_at) < _SCIHUB_CACHE_TTL:
        return cached

    domains: set[str] = set()
    for src in _SCIHUB_DISCOVERY_SOURCES:
        try:
            r = requests.get(src, timeout=10,
                             headers={"User-Agent": USER_AGENT})
            if r.ok:
                for m in _SCIHUB_DOMAIN_RE.finditer(r.text):
                    domains.add(m.group(0))
        except Exception:
            continue

    if not domains:
        domains = set(_SCIHUB_FALLBACK_DOMAINS)

    # Liveness check
    alive: list[str] = []
    for domain in sorted(domains):
        url = f"https://{domain}"
        try:
            r = requests.head(url, timeout=5, allow_redirects=True,
                              headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                alive.append(url)
        except Exception:
            continue

    # If liveness check yielded nothing (e.g., HEAD blocked everywhere),
    # fall back to all discovered domains as URLs — let _try_scihub's
    # GET handle it.
    if not alive:
        alive = [f"https://{d}" for d in sorted(domains)]

    _scihub_mirrors_cache = (now, alive)
    return alive


# PDF URL extraction patterns for sci-hub landing pages, in priority
# order. Sci-Hub mirror HTML structure changes occasionally; this list
# covers every layout we've observed:
#   1. <meta name="citation_pdf_url" content="..."> — Highwire-standard
#      meta, present on all current sci-hub.ru pages. Most reliable.
#   2. <object type="application/pdf" data="..."> — current sci-hub.ru
#      embed (replaced the old <iframe>/<embed> in 2024).
#   3. <iframe|embed src="...pdf..."> — legacy mirrors / older pages.
#   4. location.href = "...pdf..." — JS-redirect pattern on some mirrors.
_SCIHUB_PDF_PATTERNS = (
    re.compile(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', re.I),
    re.compile(r'<object[^>]+type=["\']application/pdf["\'][^>]+data=["\']([^"\']+)["\']', re.I),
    re.compile(r'(?:iframe|embed)[^>]+src=["\']([^"\']+\.pdf[^"\']*)["\']', re.I),
    re.compile(r'location\.href\s*=\s*[\'"]([^\'"]+\.pdf[^\'"]*)[\'"]'),
)


def _scihub_one_mirror(mirror: str, target: str, max_attempts: int = 3) -> Optional[bytes]:
    """Probe ONE scihub mirror; return PDF bytes or None. Extracted from
    _try_scihub so the mirror loop can race in parallel rather than
    iterate serially (40+s under racing-cascade concurrency, well past
    the per-paper deadline).

    Some mihomo exit IPs get a STRIPPED 7KB landing page from scihub.ru
    with no citation_pdf_url meta, others get the full 26KB page with
    the PDF link. Retry up to ``max_attempts`` with fresh connections
    (each requests.get opens a new TCP socket → new mihomo round-robin
    exit) when the response is suspiciously short or lacks the PDF link.
    """
    landing_url = f"{mirror}/{target}"
    for attempt in range(max_attempts):
        try:
            r = requests.get(landing_url, timeout=TIMEOUT,
                             headers=BROWSER_HEADERS, allow_redirects=True)
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            continue
        # Stripped/throttled response: too short to contain a PDF link.
        # Retry with a fresh connection (different mihomo exit IP).
        if not r.ok or len(r.text) < 12000:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            continue
        pdf_url: Optional[str] = None
        for pat in _SCIHUB_PDF_PATTERNS:
            m = pat.search(r.text)
            if m:
                pdf_url = m.group(1)
                break
        if not pdf_url:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            continue
        pdf_url = pdf_url.split("#", 1)[0]
        if pdf_url.startswith("//"):
            pdf_url = "https:" + pdf_url
        elif pdf_url.startswith("/"):
            pdf_url = mirror + pdf_url
        pdf_headers = dict(BROWSER_HEADERS)
        pdf_headers["Referer"] = landing_url
        pdf_headers["Accept"] = "application/pdf,*/*;q=0.8"
        try:
            pdf_resp = requests.get(pdf_url, timeout=TIMEOUT,
                                    headers=pdf_headers, allow_redirects=True)
            if pdf_resp.ok and _is_pdf_bytes(pdf_resp.content):
                return pdf_resp.content
        except Exception:
            pass
        if attempt < max_attempts - 1:
            time.sleep(2 ** attempt)
    return None


def _try_scihub(paper: Paper) -> Optional[bytes]:
    """Last-resort fetch via Sci-Hub. Off by default; opt in with
    PAPER_PIPELINE_USE_SCIHUB=1. Be aware this may not be legal in your
    jurisdiction.

    Mirrors race in parallel; first valid PDF wins. Serial iteration was
    the dominant bottleneck under racing-cascade concurrency — 8 mirrors
    × per-mirror timeout = 40+s, exceeding the per-paper deadline.
    """
    if os.environ.get("PAPER_PIPELINE_USE_SCIHUB", "").strip() not in {"1", "true", "yes"}:
        return None
    target = paper.doi or paper.arxiv_id or paper.url
    if not target:
        return None
    mirrors = _discover_scihub_mirrors()
    if not mirrors:
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(mirrors)) as ex:
        futures = [ex.submit(_scihub_one_mirror, m, target) for m in mirrors]
        try:
            for fut in concurrent.futures.as_completed(futures, timeout=TIMEOUT * 2):
                data = fut.result()
                if data:
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    return data
        except concurrent.futures.TimeoutError:
            pass
    return None


def _try_annas_archive_api(paper: Paper) -> Optional[bytes]:
    """Anna's Archive — fetch PDF via the documented members-only JSON API.

    Anna's puts CAPTCHAs on browser download paths but offers a stable
    JSON API for members. The cheapest "Brilliant Bookworm" tier
    ($2-7/month) includes "🧬 SciDB papers unlimited without verification"
    plus JSON API access — perfect for academic paper cascades.

    Set ``ANNAS_ARCHIVE_API_KEY`` to your member secret key
    (https://annas-archive.gl/account → "Secret key").

    Steps:
      1. ``GET /scidb/<DOI>`` to discover the md5 hash for the paper.
         Anna's renders the paper page if found; if not, falls back to a
         "Search" page (``<title>`` contains 'Search - Anna's Archive').
      2. ``GET /dyn/api/fast_download.json?md5=<HASH>&key=<KEY>`` →
         returns ``{download_url: "..."}``.
      3. Fetch the URL and verify it's a real PDF.

    Anna's coverage is roughly sci-hub union LibGen plus their own
    scrapes. For modern paywalled papers that escape both, Anna's is the
    main fallback. No-op when key unset.
    """
    api_key = os.environ.get("ANNAS_ARCHIVE_API_KEY", "").strip()
    if not api_key:
        return None
    if not paper.doi:
        return None
    base = "https://annas-archive.gl"
    # 1. /scidb/<DOI> → md5
    try:
        r = requests.get(f"{base}/scidb/{paper.doi}", timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
    except Exception as exc:
        log.warning("annas_archive[%s]: scidb request failed %r", paper.key, exc)
        return None
    if not r.ok:
        return None
    title_m = re.search(r"<title[^>]*>([^<]+)</title>", r.text)
    if title_m and "Search - Anna" in title_m.group(1):
        return None  # Paper not in Anna's index
    md5_m = re.search(r'href="/md5/([a-f0-9]+)"', r.text)
    if not md5_m:
        return None
    md5 = md5_m.group(1)
    # 2. API → download_url
    try:
        api = requests.get(
            f"{base}/dyn/api/fast_download.json",
            params={"md5": md5, "key": api_key},
            timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        data = api.json()
    except Exception as exc:
        log.warning("annas_archive[%s]: API exception %r", paper.key, exc)
        return None
    url = data.get("download_url")
    if not url:
        log.warning("annas_archive[%s]: API returned no download_url, error=%r",
                    paper.key, data.get("error"))
        return None
    # 3. Fetch PDF
    try:
        pdf = requests.get(url, timeout=60, allow_redirects=True,
                           headers={"User-Agent": USER_AGENT})
        if pdf.ok and _is_pdf_bytes(pdf.content):
            return pdf.content
    except Exception as exc:
        log.warning("annas_archive[%s]: PDF fetch failed %r", paper.key, exc)
    return None


def _try_semantic_scholar_oa(paper: Paper) -> Optional[bytes]:
    """Try Semantic Scholar's ``openAccessPdf`` field.

    S2 maintains its own OA crawl (separate from Unpaywall / OpenAlex)
    and surfaces ``openAccessPdf.url`` for papers it has crawled. Often
    catches BRONZE / HYBRID OA papers that Unpaywall missed.
    """
    if not paper.doi:
        return None
    api = (
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{paper.doi}"
        "?fields=openAccessPdf"
    )
    try:
        r = requests.get(api, timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
        if not r.ok:
            return None
        oa = (r.json().get("openAccessPdf") or {})
        url = oa.get("url")
        if not url:
            return None
        # Try with browser headers — S2 OA URLs are usually publisher
        # pages and need a browser-like UA to avoid 403.
        pdf = requests.get(url, timeout=TIMEOUT,
                           headers=BROWSER_HEADERS, allow_redirects=True)
        if pdf.ok and _is_pdf_bytes(pdf.content):
            return pdf.content
    except Exception:
        return None
    return None


def _try_openalex(paper: Paper) -> Optional[bytes]:
    """Try OpenAlex's oa_locations for alternate PDF URLs.

    Different from Unpaywall: OpenAlex returns multiple `oa_locations`,
    not just one `best_oa_location`, so it often surfaces PDFs Unpaywall
    misses.
    """
    if not paper.doi:
        return None
    mailto = os.environ.get("OPENALEX_MAILTO", "research@example.invalid")
    api = f"https://api.openalex.org/works/doi:{paper.doi}?mailto={mailto}"
    try:
        meta = requests.get(api, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT}).json()
    except Exception:
        return None

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(loc: dict | None) -> None:
        if not loc:
            return
        for field in ("pdf_url", "url_for_pdf"):
            url = loc.get(field)
            if url and url not in seen:
                seen.add(url)
                candidates.append(url)

    _add(meta.get("best_oa_location"))
    for loc in meta.get("oa_locations") or []:
        _add(loc)

    for url in candidates:
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None


def _try_core(paper: Paper) -> Optional[bytes]:
    """CORE — UK-based OA aggregator indexing ~280M papers from institutional
    repositories worldwide. Coverage is largely complementary to Unpaywall
    (which uses BASE/PubMed indexing); CORE often surfaces author-uploaded
    postprints and conference proceedings on university servers that other
    aggregators miss. Requires CORE_API_KEY env var.

    Two-pass: DOI-exact first; if that misses or returns no downloadUrl,
    fall back to title search (fuzzy match for top result).
    """
    api_key = os.environ.get("CORE_API_KEY", "").strip()
    if not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT}
    # Note: trailing slash is required — CORE returns 301 to /works/
    api = "https://api.core.ac.uk/v3/search/works/"

    def _query(q: str, limit: int = 3) -> list[dict]:
        try:
            r = requests.get(api, params={"q": q, "limit": limit},
                             headers=headers, timeout=TIMEOUT,
                             allow_redirects=True)
        except Exception:
            return []
        if not r.ok:
            return []
        try:
            return r.json().get("results") or []
        except Exception:
            return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _harvest(work: dict) -> None:
        for field in ("downloadUrl", "fullTextLink"):
            url = work.get(field)
            if url and url not in seen:
                seen.add(url)
                candidates.append(url)
        for link in (work.get("links") or []):
            if isinstance(link, dict):
                u = link.get("url")
                if u and u not in seen:
                    seen.add(u)
                    candidates.append(u)

    # Pass 1: DOI lookup
    if paper.doi:
        for w in _query(f'doi:"{paper.doi}"', 3):
            _harvest(w)

    # Pass 2: title search fallback (CORE indexes plenty without DOI match)
    if not candidates and paper.title and len(paper.title) >= 20:
        # Sanitize title — keep alnum + spaces, drop anything CORE may
        # interpret as syntax
        clean_title = ''.join(c if c.isalnum() or c.isspace() else ' '
                               for c in paper.title)[:120]
        for w in _query(f'title:"{clean_title}"', 5):
            # require title sim > 0.7 to avoid pulling unrelated papers
            cand_title = (w.get("title") or '').lower()
            if not cand_title:
                continue
            our = paper.title.lower()
            from difflib import SequenceMatcher
            if SequenceMatcher(None, cand_title[:120], our[:120], autojunk=False).ratio() < 0.70:
                continue
            _harvest(w)

    for url in candidates:
        try:
            r2 = requests.get(url, timeout=TIMEOUT,
                              headers={"User-Agent": USER_AGENT},
                              allow_redirects=True)
            if r2.ok and _is_pdf_bytes(r2.content):
                return r2.content
        except Exception:
            continue
    return None


def _try_inspire(paper: Paper) -> Optional[bytes]:
    """Try Inspire-HEP's documents[] field for fulltext links.

    Especially useful for HEP / particle physics / astrophysics where
    Inspire is the canonical fulltext source.
    """
    if not paper.arxiv_id and not paper.doi:
        return None
    if paper.arxiv_id:
        arxiv_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
        query = f"arxiv:{arxiv_id}"
    else:
        query = f"doi:{paper.doi}"
    api = (
        "https://inspirehep.net/api/literature"
        f"?q={query}&size=1&fields=documents"
    )
    try:
        meta = requests.get(api, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT}).json()
    except Exception:
        return None
    hits = (meta.get("hits") or {}).get("hits") or []
    if not hits:
        return None
    documents = (hits[0].get("metadata") or {}).get("documents") or []
    for doc in documents:
        url = doc.get("url")
        if not url:
            continue
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None


def _try_ads(paper: Paper) -> Optional[bytes]:
    """Try NASA ADS link_gateway via the search API for bibcode + esources.

    ADS doesn't return direct PDF URLs in search; PDFs go through a
    redirect gateway:
      https://ui.adsabs.harvard.edu/link_gateway/<bibcode>/<source_type>

    Source types like EPRINT_PDF, PUB_PDF, ADS_PDF, AUTHOR_PDF lead to
    PDFs (often via 302 redirect to the publisher; allow_redirects=True
    handles either 200+body or 302+Location).

    Silently skipped if ADS_API_TOKEN is unset.
    """
    token = os.environ.get("ADS_API_TOKEN", "").strip()
    if not token:
        return None
    if not paper.doi and not paper.arxiv_id:
        return None
    if paper.doi:
        query = f"doi:{paper.doi}"
    else:
        arxiv_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
        query = f"identifier:{arxiv_id}"
    api = (
        "https://api.adsabs.harvard.edu/v1/search/query"
        f"?q={query}&fl=bibcode,esources&rows=1"
    )
    try:
        resp = requests.get(
            api, timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT,
                     "Authorization": f"Bearer {token}"},
        )
        meta = resp.json()
    except Exception:
        return None
    docs = (meta.get("response") or {}).get("docs") or []
    if not docs:
        return None
    bibcode = docs[0].get("bibcode")
    esources = docs[0].get("esources") or []
    if not bibcode:
        return None
    for source_type in esources:
        if "PDF" not in source_type.upper():
            continue
        gateway = (
            f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/{source_type}"
        )
        try:
            r = requests.get(gateway, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue

    # ADS Legacy Article Service fallback. For papers that don't have a
    # direct PDF esource (typical of pre-1995 articles whose original
    # publisher metadata never recorded one), NASA ADS still hosts a
    # scanned PDF at:
    #   https://articles.adsabs.harvard.edu/pdf/<bibcode>
    # This is the dataset behind the old "ADS Article Service" — it
    # covers ApJ / ApJL / A&A and several other journals back to the
    # 1950s. Bibcodes from the search API are the canonical key.
    # No auth required (public endpoint).
    if bibcode:
        legacy_url = f"https://articles.adsabs.harvard.edu/pdf/{bibcode}"
        try:
            r = requests.get(legacy_url, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            pass
    return None


def _try_crossref_link(paper: Paper) -> Optional[bytes]:
    """Try CrossRef's ``link[]`` field for text-mining-licensed PDFs.

    For each ``intended-application = text-mining`` URL, identify
    ourselves as a TDM client (``User-Agent: TextMining/1.0``,
    ``Accept: application/pdf``). Empirically: IOPscience, AAS, AGU/Wiley
    and other CrossRef-listed publishers gate ordinary browser/curl UAs
    behind perfdrive/Cloudflare/Akamai, but honour the TDM intent and
    serve the PDF directly when the request identifies as text-mining.

    This is the documented, legitimate path. CrossRef's ``link[]`` field
    is the publisher's machine-readable opt-in to bulk text/data mining;
    using it as such isn't bypassing anything — it's *using* it.
    """
    if not paper.doi:
        return None
    email = os.environ.get("UNPAYWALL_EMAIL", "research@example.invalid")
    api = f"https://api.crossref.org/works/{paper.doi}"
    try:
        meta = requests.get(
            api, timeout=TIMEOUT,
            headers={"User-Agent": f"paper-library/0.1 (mailto:{email})"},
        ).json()
    except Exception:
        return None
    links = ((meta.get("message") or {}).get("link") or [])
    candidates: list[tuple[str, bool]] = []  # (url, is_tdm)
    seen: set[str] = set()
    for link in links:
        url = link.get("URL")
        if not url or url in seen:
            continue
        intended = (link.get("intended-application") or "").lower()
        ctype = (link.get("content-type") or "").lower()
        is_tdm = intended == "text-mining"
        if (is_tdm or ctype == "application/pdf"
                or url.lower().endswith(".pdf")):
            seen.add(url)
            candidates.append((url, is_tdm))
    tdm_headers = {
        "User-Agent": f"TextMining/1.0 (mailto:{email})",
        "Accept": "application/pdf",
    }
    for url, is_tdm in candidates:
        headers = tdm_headers if is_tdm else {"User-Agent": USER_AGENT}
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers=headers, allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None


def _try_europepmc(paper: Paper) -> Optional[bytes]:
    """Try EuropePMC (EBI-hosted, broader than US PMC).

    Covers life science + adjacent physics + applied research. Pulls from
    the result's `fullTextUrlList[]` (filtered to documentStyle="pdf"),
    plus the PMC PDF render URL when a `pmcid` is available.
    """
    if not paper.doi and not paper.arxiv_id:
        return None
    if paper.doi:
        query = f"DOI:{paper.doi}"
    else:
        arxiv_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
        query = f"arXiv:{arxiv_id}"
    api = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={query}&format=json&resultType=core"
    )
    try:
        meta = requests.get(api, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT}).json()
    except Exception:
        return None
    results = ((meta.get("resultList") or {}).get("result") or [])
    if not results:
        return None
    result = results[0]

    candidates: list[str] = []
    seen: set[str] = set()

    full_text_urls = (
        (result.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    )
    for entry in full_text_urls:
        url = entry.get("url")
        if not url or url in seen:
            continue
        if (entry.get("documentStyle") or "").lower() == "pdf":
            seen.add(url)
            candidates.append(url)

    pmcid = result.get("pmcid")
    if pmcid:
        render_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
        if render_url not in seen:
            seen.add(render_url)
            candidates.append(render_url)

    for url in candidates:
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None


def _try_zenodo(paper: Paper) -> Optional[bytes]:
    """Try Zenodo (CERN-hosted records API).

    Strong for CS / physics preprints, datasets, and conference papers.
    Pulls `hits.hits[0].files[]` and downloads each file with
    `type == "pdf"` via its `links.self` URL.
    """
    if not paper.doi and not paper.arxiv_id:
        return None
    if paper.doi:
        q = f'doi:"{paper.doi}"'
    else:
        arxiv_id = re.sub(r"v\d+$", "", paper.arxiv_id.strip())
        q = f'arxiv:"{arxiv_id}"'
    api = "https://zenodo.org/api/records"
    try:
        meta = requests.get(
            api,
            params={"q": q, "size": 1},
            timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        ).json()
    except Exception:
        return None
    hits = ((meta.get("hits") or {}).get("hits") or [])
    if not hits:
        return None
    files = hits[0].get("files") or []
    for f in files:
        ftype = (f.get("type") or "").lower()
        if ftype != "pdf":
            continue
        url = ((f.get("links") or {}).get("self"))
        if not url:
            continue
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers={"User-Agent": USER_AGENT},
                             allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None


def _try_arxiv_by_title(paper: Paper) -> Optional[bytes]:
    """When the paper has no arxiv_id but has a title, search arxiv by
    title to discover an arxiv preprint version. Many papers in
    paywalled journals have arxiv preprints whose ID didn't get captured
    in the original metadata fetch.

    On a confident match, persists the discovered arxiv_id back to the
    paper so future cascade attempts skip the search.
    """
    if paper.arxiv_id:
        return None  # already had arxiv_id; _try_arxiv would have used it
    title = (paper.title or "").strip()
    if len(title) < 20:  # too short to disambiguate reliably
        return None

    from .sources.arxiv import search_arxiv
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

    # Try downloading via arxiv directly. Do NOT persist the discovered arxiv_id
    # up-front: a title-match can be loose, and the bytes still face
    # _verify_pdf_matches_metadata in the caller. Persisting before any outcome
    # poisoned future sweeps even on a plain download failure (verified drill
    # finding). Persist ONLY once we actually have a PDF in hand.
    arxiv_id_clean = re.sub(r'v\d+$', '', discovered_arxiv_id.strip())
    url = f"https://arxiv.org/pdf/{arxiv_id_clean}"
    try:
        r = requests.get(url, timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT},
                         allow_redirects=True)
        if r.ok and _is_pdf_bytes(r.content):
            paper.arxiv_id = discovered_arxiv_id
            return r.content
    except Exception:
        return None
    return None


# Highwire Press <meta name="citation_pdf_url"> is a near-universal
# convention among scholarly publishers. We match both single- and
# double-quoted variants and don't care about attribute order.
_CITATION_PDF_URL_RE = re.compile(
    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
# Some publishers put `content` before `name`. Match that ordering too.
_CITATION_PDF_URL_RE_REV = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
    re.IGNORECASE,
)


def _try_citation_pdf_url(paper: Paper) -> Optional[bytes]:
    """Generic publisher meta-tag PDF discovery.

    Most academic publishers (MDPI, Springer, Wiley, RSC, ACS, IEEE, AIP,
    APS, IOP, etc.) embed a Highwire Press style <meta name="citation_pdf_url"
    content="..."> tag on their article landing pages. Hitting the DOI
    redirect lands you on the publisher page; we then parse for the tag
    and follow it.

    Two-UA strategy: many publishers (MDPI, Springer) want browser-like
    headers and 403 bare UAs; others (IOPscience for AAS / IOP journals)
    do the opposite — bare UA returns the actual PDF, browser-like
    requests get redirected to a perfdrive bot-challenge. So we try BOTH
    UA strategies for both the landing page fetch and the final PDF
    fetch, taking the first that yields a valid PDF.
    """
    if not paper.doi:
        return None

    landing_url = f"https://doi.org/{paper.doi}"
    # Two-UA landing fetch: take whichever mode yields a citation_pdf_url
    # meta tag (or a direct PDF). Browser-headers can succeed for MDPI but
    # gets redirected to a bot-challenge for IOPscience; bare UA inverts
    # that. We must verify the meta tag is actually present, not just
    # that the response was 200 — perfdrive's bot challenge ALSO returns
    # 200 (with HTML but without the meta tag).
    landing = None
    for ua_kind, ua_headers in (("bare", {"User-Agent": USER_AGENT}),
                                 ("browser", BROWSER_HEADERS)):
        try:
            r = requests.get(landing_url, timeout=TIMEOUT,
                              headers=ua_headers, allow_redirects=True)
        except Exception:
            continue
        if not r.ok:
            continue
        if _is_pdf_bytes(r.content):
            return r.content
        if (_CITATION_PDF_URL_RE.search(r.text)
                or _CITATION_PDF_URL_RE_REV.search(r.text)):
            landing = r
            break
    if landing is None:
        return None

    html = landing.text
    m = _CITATION_PDF_URL_RE.search(html) or _CITATION_PDF_URL_RE_REV.search(html)
    if not m:
        return None
    pdf_url = m.group(1).strip()
    if pdf_url.startswith("//"):
        pdf_url = "https:" + pdf_url
    elif pdf_url.startswith("/"):
        from urllib.parse import urlparse
        base = urlparse(landing.url)
        pdf_url = f"{base.scheme}://{base.netloc}{pdf_url}"

    # Try BOTH UA modes for the PDF fetch. IOP is the canonical example
    # of "bare-UA wins": its `/article/{doi}/pdf` endpoint returns
    # application/pdf to USER_AGENT but redirects browser-like requests
    # to a perfdrive bot-challenge HTML page.
    for ua_kind, base_headers in (("bare", {"User-Agent": USER_AGENT}),
                                   ("browser", BROWSER_HEADERS)):
        headers = dict(base_headers)
        headers["Referer"] = landing.url
        headers["Accept"] = "application/pdf,*/*;q=0.8"
        try:
            r = requests.get(pdf_url, timeout=TIMEOUT,
                              headers=headers, allow_redirects=True)
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            continue
    return None


def _try_wiley_tdm(paper: Paper) -> Optional[bytes]:
    """Wiley Text-and-Data-Mining (TDM) API — official publisher API for
    Wiley-hosted journals, free for academic users. Covers ~2,000 Wiley
    journals including Wiley Online Library and AGU titles (which Wiley
    publishes since 2013, prefix 10.1029/10.1002).

    Without this tier, Wiley papers are ~impossible: their CDN does TLS
    fingerprinting + JS challenges that defeat curl-cffi, and sci-hub's
    coverage of 2018+ Wiley titles is patchy.

    Token registration (free, ~1 day approval): see
    https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining
    Set WILEY_TDM_TOKEN env var.

    The API returns the PDF directly. Has a documented rate limit; we add
    a small inline sleep to be polite.
    """
    token = os.environ.get("WILEY_TDM_TOKEN", "").strip()
    if not token:
        log.debug("wiley_tdm: skipped — WILEY_TDM_TOKEN not set")
        return None
    if not paper.doi:
        return None
    # Wiley TDM serves any DOI hosted on onlinelibrary.wiley.com — that
    # includes 10.1002/* (Wiley) and 10.1029/* (AGU, post-2013).
    prefix = paper.doi.split("/", 1)[0]
    if prefix not in ("10.1002", "10.1029", "10.1111", "10.1046"):
        return None
    from urllib.parse import quote
    encoded_doi = quote(paper.doi, safe="")
    api_url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{encoded_doi}"
    try:
        r = requests.get(api_url, timeout=60,
                         headers={"Wiley-TDM-Client-Token": token},
                         allow_redirects=True)
    except Exception as exc:
        log.warning("wiley_tdm[%s]: request exception %r", paper.key, exc)
        return None
    if r.ok and _is_pdf_bytes(r.content):
        return r.content
    # Visible failure — log status + first body bytes so the operator can
    # see WHY the TDM API returned non-PDF (rate limit / not subscribed /
    # bad token / paywalled-without-subscription / etc.).
    body_head = r.content[:200].decode("utf-8", errors="replace")
    log.warning(
        "wiley_tdm[%s]: status=%d ctype=%r len=%d body[:200]=%r",
        paper.key, r.status_code,
        r.headers.get("Content-Type", "")[:60],
        len(r.content), body_head,
    )
    return None


def _try_elsevier_tdm(paper: Paper) -> Optional[bytes]:
    """Elsevier Text-and-Data-Mining API — official Elsevier API for
    institutional subscribers. Returns the article in XML by default; we
    request PDF specifically.

    Requires:
      - ELSEVIER_TDM_API_KEY env var (institutional subscription required)

    Without this, Elsevier (10.1016) papers are heavily paywalled. Sci-Hub
    coverage of 2020+ Elsevier titles has gaps.
    """
    key = os.environ.get("ELSEVIER_TDM_API_KEY", "").strip()
    if not key:
        return None
    if not paper.doi:
        return None
    prefix = paper.doi.split("/", 1)[0]
    if prefix not in ("10.1016",):
        return None
    api_url = (f"https://api.elsevier.com/content/article/doi/{paper.doi}"
               f"?apiKey={key}")
    try:
        r = requests.get(api_url, timeout=60,
                         headers={"Accept": "application/pdf"},
                         allow_redirects=True)
    except Exception:
        return None
    if r.ok and _is_pdf_bytes(r.content):
        return r.content
    return None


def _try_curl_impersonate(paper: Paper) -> Optional[bytes]:
    """Variant of `_try_citation_pdf_url` using curl-cffi to impersonate
    a real browser's TLS fingerprint. Defeats Akamai EdgeSuite and similar
    CDN gates that classify our `requests`-library traffic as bot traffic
    and 403 us — even when the request looks browser-like at the HTTP
    layer (User-Agent, Accept-*, Sec-Fetch-*).

    Validated empirically: `requests` 403s on
    https://www.mdpi.com/2504-2289/6/4/140/pdf, but curl-cffi with
    chrome120 impersonation returns the actual 2.9MB PDF. Same code path,
    same headers — only the TLS fingerprint differs.

    Doesn't help against publisher walls that need JavaScript execution
    (Wiley/onlinelibrary, ACS, ASME) — those still 403 because the wall
    is a JS challenge not a TLS-fingerprint check. For those we'd need
    a real headless browser (Playwright).
    """
    if not paper.doi:
        return None
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return None
    try:
        landing = curl_requests.get(
            f"https://doi.org/{paper.doi}",
            impersonate="chrome120",
            timeout=TIMEOUT,
            allow_redirects=True,
        )
    except Exception:
        return None
    if not landing.ok:
        return None
    if _is_pdf_bytes(landing.content):
        return landing.content
    html = landing.text
    m = _CITATION_PDF_URL_RE.search(html) or _CITATION_PDF_URL_RE_REV.search(html)
    if not m:
        return None
    pdf_url = m.group(1).strip()
    if pdf_url.startswith("//"):
        pdf_url = "https:" + pdf_url
    elif pdf_url.startswith("/"):
        from urllib.parse import urlparse
        base = urlparse(landing.url)
        pdf_url = f"{base.scheme}://{base.netloc}{pdf_url}"
    try:
        r = curl_requests.get(
            pdf_url, impersonate="chrome120", timeout=TIMEOUT,
            headers={"Referer": landing.url,
                     "Accept": "application/pdf,*/*;q=0.8"},
            allow_redirects=True,
        )
        if r.ok and _is_pdf_bytes(r.content):
            return r.content
    except Exception:
        return None
    return None


# SSRN download link pattern (relative to https://papers.ssrn.com).
_SSRN_DELIVERY_RE = re.compile(
    r'href=["\'](/sol3/Delivery\.cfm/[^"\']+\.pdf[^"\']*)["\']',
    re.IGNORECASE,
)


def _try_ssrn(paper: Paper) -> Optional[bytes]:
    """SSRN papers — DOI prefix `10.2139/ssrn.<N>` → fetch via the
    abstract page's "Download This Paper" link.

    SSRN sometimes paywalls papers (private uploads, embargoed); those
    return non-PDF and fail magic-byte check. Public SSRN papers download
    cleanly.
    """
    if not paper.doi or not paper.doi.startswith("10.2139/ssrn."):
        return None
    try:
        abstract_id = paper.doi.split("ssrn.")[-1].strip()
    except Exception:
        return None
    if not abstract_id:
        return None
    landing_url = (
        f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={abstract_id}"
    )
    session = requests.Session()
    session.headers.update({"User-Agent": BROWSER_USER_AGENT})
    try:
        landing = session.get(landing_url, timeout=TIMEOUT, allow_redirects=True)
    except Exception:
        return None
    if not landing.ok:
        return None
    # Look for a Delivery.cfm download link in the page.
    m = _SSRN_DELIVERY_RE.search(landing.text)
    if not m:
        return None
    pdf_path = m.group(1)
    pdf_url = f"https://papers.ssrn.com{pdf_path}"
    try:
        r = session.get(
            pdf_url,
            timeout=TIMEOUT,
            headers={"Referer": landing_url},
            allow_redirects=True,
        )
        if r.ok and _is_pdf_bytes(r.content):
            return r.content
    except Exception:
        return None
    return None


# ResearchGate publication-detail page link from search results. The
# canonical URL shape is `/publication/<numeric_id>_<slug>` — we accept
# absolute or root-relative.
_RG_PUBLICATION_RE = re.compile(
    r'href=["\'](?:https?://(?:www\.)?researchgate\.net)?(/publication/\d+[^"\']*)["\']',
    re.IGNORECASE,
)
# PDF link inside a publication detail page. RG uses several patterns;
# we look for any `.pdf` inside an href, especially `full-text.pdf`,
# `publicationDownloadFile`, or `documentDownload`.
_RG_PDF_LINK_RE = re.compile(
    r'href=["\']([^"\']+(?:\.pdf|publicationDownloadFile[^"\']*|'
    r'documentDownload[^"\']*))["\']',
    re.IGNORECASE,
)


def _try_researchgate(paper: Paper) -> Optional[bytes]:
    """ResearchGate via Scrapling + real-browser click simulation.

    RG's anti-bot is multi-layered: Cloudflare interstitial in front,
    login wall behind, and a click-event check on the actual Download
    button. Plain ``requests`` (the historical implementation here) was
    blocked at layer one; even cookie replay was blocked at layer three.
    The only working path so far is:

      1. Find the RG publication URL — search-engine result (Google
         Scholar via Scrapling) usually exposes it as
         ``/profile/<author>/publication/<id>_<slug>``. This is more
         reliable than RG's own /search/ endpoint, which often hides
         recent uploads behind a login wall.
      2. Drive Scrapling's StealthyFetcher (Playwright + Cloudflare
         Turnstile solver) to the publication's landing page.
      3. Locate the ``[data-testid="research-header-cta-download-fulltext"]``
         button and trigger ``page.expect_download() / page.click()`` —
         a real browser click event, which is what RG validates. RG
         then issues the actual PDF binary in response.

    Slow (~30-60 s per call: ~20 s Cloudflare solve + ~10 s page render
    + ~5 s download). Costs Playwright launch + ~300 MB browser. Position
    as a near-last tier in the cascade. Returns None on any failure;
    safe under repeat invocations.

    All Scrapling imports are lazy so the daemon doesn't pull patchright
    + Playwright until this tier actually runs.
    """
    title = (paper.title or "").strip()
    doi = (paper.doi or "").strip()
    if len(title) < 20 and not doi:
        return None

    try:
        from scrapling.fetchers import StealthyFetcher
    except ImportError:
        return None

    # Stage 1: find an RG publication URL via Google Scholar.
    # Scholar surfaces the canonical /profile/<author>/publication/<id>/...
    # form that RG itself sometimes hides behind a login wall.
    if doi:
        scholar_q = f"{title[:120]} {doi}"
    else:
        scholar_q = title[:200]
    from urllib.parse import quote_plus
    scholar_url = (f"https://scholar.google.com/scholar?q="
                    f"{quote_plus(scholar_q)}")
    try:
        page = StealthyFetcher.fetch(scholar_url, headless=True,
                                       solve_cloudflare=True, wait=2500)
    except Exception:
        return None
    if not page or getattr(page, "status", 0) != 200:
        return None

    m = re.search(
        r"https?://(?:www\.)?researchgate\.net/(?:profile/[A-Za-z0-9-]+/)?"
        r"publication/\d+[A-Za-z0-9_%/-]+",
        page.html_content)
    if not m:
        return None
    landing_url = m.group(0).rstrip(")&\"'")
    # If we landed on a deep PDF URL (.../links/<hex>/<slug>.pdf), strip
    # back to the publication root — that's where the Download button is.
    landing_url = re.sub(r"/links?/.*$", "", landing_url)
    landing_url = re.sub(r"/citation.*$", "", landing_url)

    # Stage 2 + 3: render landing page in Scrapling, then click Download.
    result_holder: dict = {"bytes": None}

    def click_download(page):
        try:
            with page.expect_download(timeout=30000) as dl_info:
                page.locator(
                    '[data-testid="research-header-cta-download-fulltext"]'
                ).first.click()
            download = dl_info.value
            # Read the downloaded bytes into memory (Playwright keeps a
            # temp file; .path() returns a Path object).
            path = download.path()
            if path is not None:
                with open(path, "rb") as f:
                    result_holder["bytes"] = f.read()
        except Exception:
            # No button / no download / timeout / login-wall → just give up.
            pass
        return page

    try:
        StealthyFetcher.fetch(landing_url, headless=True,
                                solve_cloudflare=True, wait=4000,
                                page_action=click_download)
    except Exception:
        return None

    data = result_holder["bytes"]
    if data and _is_pdf_bytes(data):
        return data
    return None


def _try_cloudscraper_publisher(paper: Paper) -> Optional[bytes]:
    """Same shape as _try_citation_pdf_url, but uses cloudscraper to bypass
    Cloudflare JS challenges. Helps with publishers fronted by Cloudflare
    (RSC, parts of Springer, etc.).

    Does NOT help with Akamai (MDPI) or PerimeterX (Wiley/ACS) — those use
    different anti-bot stacks. For those we'd need Playwright; out of
    scope here.

    Imports cloudscraper lazily so the optional dep doesn't break a
    minimal install.
    """
    if not paper.doi:
        return None
    try:
        import cloudscraper
    except ImportError:
        return None
    scraper = cloudscraper.create_scraper()
    try:
        landing = scraper.get(
            f"https://doi.org/{paper.doi}",
            timeout=TIMEOUT,
            allow_redirects=True,
        )
    except Exception:
        return None
    if not landing.ok:
        return None
    if _is_pdf_bytes(landing.content):
        return landing.content
    html = landing.text
    m = _CITATION_PDF_URL_RE.search(html) or _CITATION_PDF_URL_RE_REV.search(html)
    if not m:
        return None
    pdf_url = m.group(1).strip()
    if pdf_url.startswith("//"):
        pdf_url = "https:" + pdf_url
    elif pdf_url.startswith("/"):
        from urllib.parse import urlparse
        base = urlparse(landing.url)
        pdf_url = f"{base.scheme}://{base.netloc}{pdf_url}"
    try:
        r = scraper.get(
            pdf_url,
            timeout=TIMEOUT,
            headers={"Referer": landing.url},
            allow_redirects=True,
        )
        if r.ok and _is_pdf_bytes(r.content):
            return r.content
    except Exception:
        return None
    return None


# Order rationale:
#   1-2: official free sources (arxiv preprint, unpaywall OA copy) — cheapest
#        and most legitimate.
#   3:   sci-hub — high hit rate across paywalled publishers (Elsevier / MDPI /
#        Wiley / Springer). Promoted from last-position in 2026-05 after fixing
#        the regex; running it early avoids ~5 min of timeouts on tiers 4-13
#        for the typical paywalled paper.
#   4-10: OA aggregators (openalex/inspire/ads/europepmc/zenodo) and link
#         heuristics (crossref_tm, citation_pdf_url, ssrn). Cover papers
#         sci-hub doesn't have.
#   11-14: heuristics + scrapers (arxiv_by_title, cloudscraper, researchgate)
#          — last-resort, fragile.
def _strip_frontmatter(text: str) -> str:
    """Drop a leading ``---\\n … \\n---\\n`` YAML frontmatter block, if present.

    The firecrawl md on disk is ``frontmatter + body``; the completeness
    gate must judge the BODY only (the YAML header — source/url/fetched_at —
    is metadata that would confuse a "is this a complete article?" judge).
    Returns ``text`` unchanged when there is no frontmatter.
    """
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end < 0:
        return text
    return text[end + len("\n---\n"):].lstrip("\n")


def _gate_firecrawl_md(paper: Paper, library: Library, body: str,
                       *, target_url: str = "") -> bool:
    """Run the D5 completeness gate on a firecrawl md body and act on it.

    Single decision point shared by both firecrawl-md paths — the fresh
    ``/v1/scrape`` write below AND the idempotent re-entry on an md already
    on disk (incl. the ~48 historical firecrawl papers the migration adopted
    as ``ok``; reconcile routes them back through ``download_paper`` so they
    get gated here — SDD §6.3 / D5 "resolve to full-text or no-full-text").

    ``body`` is the markdown WITHOUT YAML frontmatter (gate judges the body
    only). On PASS: status ``ok`` + source ``firecrawl``, md left on disk,
    returns True. On FAIL: the md is DELETED (no ``text-only:firecrawl``
    limbo, no retain-for-retry, no degraded serve — D5), the md fields are
    cleared, and the paper is demoted to a terminal status — ``metadata_only``
    if an abstract is citable, else ``failed`` — returns False. Gate call /
    LLM / parse error → fail-open (``complete=True``) so a genuinely-good
    rendering isn't discarded over a flaky API (mirrors completeness_gate).

    D3 hardening (2026-06-10): goes through ``confirmed_completeness_gate`` —
    a reject here is DESTRUCTIVE (md unlinked + terminal demote, never
    re-routed), so a single-pass false-reject (~0.8%/pass measured) would
    permanently destroy a good rendering; the reject must be confirmed by a
    second agreeing pass.
    """
    try:
        from .extract import confirmed_completeness_gate
        gate = confirmed_completeness_gate(body)  # body only, frontmatter must not confuse LLM
    except Exception as exc:
        gate = {"complete": True, "reason": "gate_call_error"}
        library.log({"event": "firecrawl_gate_error",
                     "key": paper.key, "exc": str(exc)})

    if not gate.get("complete", True):
        # Reject: no full text from firecrawl either. Remove the md + clear the
        # md fields, then demote — abstract present → metadata_only (citable),
        # else failed (a true zero). This is a TERMINAL status (D5: firecrawl is
        # the last resort, "tried and failed = failed"), so classify rule 1
        # returns TERMINAL — the paper RESTS, it does NOT re-route to DOWNLOAD
        # (no re-hunt). serve-safety has no md to hand out; abstract still served.
        #
        # FAIL-CLOSED on a failed unlink (issue #1, manifestation 2): if the
        # rejected stub cannot be removed (transient OSError), it MUST NOT stay
        # serveable. serve-safety (server._attach_text_reference) hands out ANY
        # md on disk as text_path with NO gate re-check, and classify rule 1
        # never re-routes a now-terminal paper, so a leftover rejected stub
        # would be served as "real + complete" full text forever (violates §4.3
        # "text_path ⟺ gated" + D6). Both ``has_extract(md)`` and ``md_source``
        # key on ``st_size > 0``, so truncating the file to empty makes
        # serve-safety AND classify treat it as absent. Try unlink (clean)
        # first; on OSError fall back to truncate-to-empty; only if BOTH fail
        # does the stub remain (logged for audit).
        md_p = library.md_path(paper.key)
        try:
            md_p.unlink()
        except FileNotFoundError:
            # No file on disk (the fresh-scrape gate-before-write path: md was
            # never written — the unlink is a harmless no-op, the desired
            # end-state already holds). Do NOT create a stray empty file.
            pass
        except OSError:
            # File EXISTS but cannot be removed (transient OSError). It MUST NOT
            # stay serveable: serve-safety hands out any md on disk as text_path
            # with NO gate re-check, and classify rule 1 never re-routes a
            # now-terminal paper. Fall back to truncate-to-empty (st_size==0 →
            # has_extract False → serve/classify treat the md as absent). Only
            # if truncation ALSO fails does the stub remain (logged for audit).
            try:
                md_p.write_text("", encoding="utf-8")
            except OSError:
                library.log({"event": "firecrawl_gate_reject_unlink_failed",
                             "key": paper.key, "url": target_url,
                             "warning": "rejected firecrawl md still on disk — "
                                        "could neither unlink nor truncate it"})
        paper.md_path = None
        paper.md_engine = ""
        paper.md_engine_version = ""
        if (paper.abstract or "").strip():
            paper.download_status = DOWNLOAD_STATUS_METADATA_ONLY
        else:
            paper.download_status = DOWNLOAD_STATUS_FAILED
        library.log({"event": "firecrawl_gate_reject", "key": paper.key,
                     "url": target_url, "reason": gate.get("reason", ""),
                     "demoted_to": paper.download_status})
        return False

    # D7: firecrawl markdown on disk, no PDF binary → status ok + source label,
    # md served as text_path. firecrawl only runs AFTER all 18 PDF tiers missed,
    # so by reaching a PASS here the real-PDF hunt is EXHAUSTED for this paper.
    # Stamp it so classify() routes this paper to its RESTING state (rule 5
    # TERMINAL/skip) instead of re-routing it to DOWNLOAD on every reconcile
    # sweep — which would re-run 18 net strategies + this LLM gate forever (the
    # hot-loop) and, over infinite re-gating, eventually let one spurious
    # incomplete verdict DELETE this genuinely-good md (issue #1/#2, S3). The
    # hunt + re-gate therefore run AT MOST ONCE per firecrawl md. A real PDF
    # arriving later flips has_pdf so classify's PDF rules take over regardless.
    paper.download_status = DOWNLOAD_STATUS_OK
    paper.download_source = "firecrawl"
    paper.firecrawl_pdf_hunt_exhausted = True
    return True


def _try_firecrawl_text_fallback(paper: Paper, library: Library) -> bool:
    """Last-resort text-only fallback when all PDF tiers failed.

    Calls Firecrawl /v1/scrape to obtain a markdown rendering of the
    publisher landing page or PDF endpoint, and writes that markdown
    directly to extracts/md/{key}.md with a YAML frontmatter block
    recording source/url/timestamp. PDF binary remains absent.

    Endpoint selection: prefer ``FIRECRAWL_API_URL`` (e.g.,
    ``http://localhost:3002`` for self-hosted). Cloud is the default.
    Self-hosted firecrawl in this deployment runs alongside mihomo and
    routes outbound through it, so each /scrape call gets a fresh
    SS-pool exit IP — defeats per-IP fingerprinting that would
    otherwise cache a captcha response. Self-hosted needs no API key;
    cloud requires FIRECRAWL_API_KEY.

    Triggered only when:
      * a usable endpoint is configured (cloud key OR self-hosted URL)
      * 18-tier PDF cascade has fully missed
      * No md extract already exists (don't overwrite higher-quality
        marker output). A firecrawl md, once it passes the completeness
        gate, is terminal: there is NO PDF-upgrade re-OCR even if a real
        PDF later appears (D5 "if firecrawl failed, it failed — final"). classify()
        routes such a paper to rule-2 TERMINAL and serve-safety keeps
        serving the firecrawl md via text_path.

    On success (D7): paper.download_status = "ok",
    paper.download_source = "firecrawl", paper.md_path is set,
    md_engine="firecrawl" — an md on disk with no PDF. The download queue
    sees ``not ok`` from download_paper but ``status==ok ∧ has md`` and
    chains to the extract queue, which short-circuits the OCR step (md
    already on disk; no PDF to feed the cascade anyway).

    Returns True iff markdown was written; False otherwise (no key,
    no DOI, http error, too-short response).
    """
    api_url = os.environ.get(
        "FIRECRAWL_API_URL", "https://api.firecrawl.dev").rstrip("/")
    api_key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    is_self_hosted = "localhost" in api_url or "127.0.0.1" in api_url \
        or api_url.startswith("http://")
    if not is_self_hosted and not api_key:
        # Cloud requires a key; self-hosted doesn't.
        return False

    if library.has_extract(paper.key, "md"):
        # Idempotent re-entry on an md already on disk. If it's firecrawl-
        # sourced, the paper is in text-only state — but D5 says firecrawl md
        # must PASS the completeness gate to be served as full text, with no
        # "text-only:firecrawl" limbo. So re-gate the on-disk body (this is
        # how the ~48 historical firecrawl papers the migration adopted as
        # ``ok`` actually resolve to full-text or no-full-text: reconcile
        # routes them DOWNLOAD → all 18 PDF tiers miss → here; SDD §6.3/§4.1).
        # PASS → ok+firecrawl + firecrawl_pdf_hunt_exhausted stamped (S3: the
        # hunt + re-gate run AT MOST ONCE, never re-entered every sweep);
        # FAIL → md deleted + terminal demotion inside _gate_firecrawl_md.
        if library.md_source(paper.key) == "firecrawl":
            # Already gated once (stamp set by the early re-gate in
            # download_paper, or by a prior sweep): the md on disk is a
            # PASSED firecrawl extract. Do NOT re-gate it — a second gate
            # call on a borderline body could spuriously FAIL and DELETE a
            # genuinely-good md (the accumulated-deletion hazard the
            # exhausted stamp exists to prevent, issue #1/#2). Re-assert ok
            # and return True (md stays serveable).
            if paper.firecrawl_pdf_hunt_exhausted:
                paper.download_status = DOWNLOAD_STATUS_OK
                paper.download_source = "firecrawl"
                return True
            try:
                on_disk = library.md_path(paper.key).read_text(
                    encoding="utf-8", errors="replace")
            except OSError:
                return False
            body = _strip_frontmatter(on_disk)
            return _gate_firecrawl_md(paper, library, body)
        # md is from a higher-fidelity engine (marker/dots) — only possible if a
        # PDF was once present and then deleted. We leave the md ALONE (a real
        # OCR extract already passed completeness_gate at write time, §6.1), but
        # we must RE-ASSERT status=ok before returning: if we returned False
        # bare, download_paper's fallthrough (no firecrawl win) would clobber
        # this paper to metadata_only/failed — a terminal status that LIES about
        # a paper that HAS full text on disk. Re-asserting ok keeps the §4.3
        # invariant (status matches the served md) — issue #3, S3.
        #
        # NOTE (G2 fix): after classify rule (3) was tightened so a
        # non-firecrawl md-without-PDF RESTS at TERMINAL (md_source != firecrawl
        # → no DOWNLOAD), classify no longer re-routes such a paper here and
        # download_queue.start() recovery only picks up pending ∧ ¬has_pdf, so
        # in steady state NO path feeds this branch. It survives as a purely
        # DEFENSIVE re-assert (guards the status if the paper is ever explicitly
        # enqueued), NOT as a hot-loop step that "keeps hunting the real PDF".
        paper.download_status = DOWNLOAD_STATUS_OK
        return False

    if paper.doi:
        target_url = f"https://doi.org/{paper.doi}"
    elif paper.url:
        target_url = paper.url
    elif paper.arxiv_id:
        target_url = f"https://arxiv.org/abs/{paper.arxiv_id}"
    else:
        return False

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Self-hosted firecrawl is reachable on the host network. Bypass
    # the daemon's HTTP_PROXY (which routes through mihomo) when calling
    # local services — the firecrawl container itself routes its OWN
    # outbound through mihomo, no need to double-proxy the localhost hop.
    proxies = ({"http": None, "https": None}
               if is_self_hosted else None)
    try:
        r = requests.post(
            f"{api_url}/v1/scrape",
            headers=headers,
            proxies=proxies,
            json={
                "url": target_url,
                "formats": ["markdown"],
                "timeout": 90000,
            },
            timeout=120,
        )
    except Exception as exc:
        library.log({"event": "firecrawl_error", "key": paper.key,
                      "url": target_url, "error": repr(exc)[:200]})
        return False

    if not r.ok:
        library.log({"event": "firecrawl_http_error", "key": paper.key,
                      "url": target_url, "status": r.status_code,
                      "body": r.text[:200]})
        return False

    try:
        body = r.json()
    except Exception:
        return False

    if not body.get("success"):
        library.log({"event": "firecrawl_unsuccessful", "key": paper.key,
                      "url": target_url, "error": str(body.get("error"))[:200]})
        return False

    md = ((body.get("data") or {}).get("markdown") or "").strip()
    if len(md) < 1000:
        library.log({"event": "firecrawl_too_short", "key": paper.key,
                      "url": target_url, "len": len(md)})
        return False

    # Anti-bot challenge detection: publishers gate paywalled / CDN-protected
    # PDFs behind hCaptcha / Cloudflare / Radware perfdrive challenges, which
    # firecrawl will happily render and return as ~2-3 KB of "are you a human"
    # markdown. Length check alone won't catch these (they're past 1000 chars
    # because of i18n language lists / hCaptcha boilerplate). Bail on any
    # known marker; let the paper stay in `failed` state so the operator
    # knows we DON'T have content for it (better than poisoning the library
    # with a captcha page masquerading as a paper).
    md_lower = md.lower()
    _ANTI_BOT_MARKERS = (
        "we apologize for the inconvenience",   # IOP / Radware perfdrive
        "validate.perfdrive.com",
        "incident id:",                          # perfdrive challenge stub
        "hcaptcha",
        "recaptcha",
        "are you a human",
        "i am human",
        "just a moment",                         # Cloudflare interstitial
        "checking your browser",                 # Cloudflare older
        "access denied",
        "edgesuite.net",                         # Akamai block page
    )
    for marker in _ANTI_BOT_MARKERS:
        if marker in md_lower:
            library.log({"event": "firecrawl_anti_bot_detected",
                          "key": paper.key, "url": target_url,
                          "marker": marker, "len": len(md)})
            return False

    from datetime import datetime, timezone, timedelta
    fetched_at = datetime.now(
        timezone(timedelta(hours=8))).isoformat(timespec="seconds")

    frontmatter = (
        "---\n"
        "source: firecrawl\n"
        f"url: {target_url}\n"
        f"fetched_at: {fetched_at}\n"
        "firecrawl_endpoint: /v1/scrape\n"
        "note: PDF binary unavailable; markdown is firecrawl rendering of publisher page.\n"
        "---\n\n"
    )

    # ---- Completeness gate (D5, SDD §6.3) — firecrawl text through the SAME
    # whole-document gate as the OCR spine. The 18-tier cascade already missed,
    # so this rendering is the last shot; if it's a paywall stub / truncated /
    # mid-sentence body it must NOT linger on disk as a serveable text_path
    # (serve-safety treats any md on disk as "real + complete"). FAIL → the
    # shared gate helper demotes to a terminal status — D5: firecrawl is the
    # final answer, "fail the gate → no full text at all, no retain-on-disk retry". Fail-open (LLM/parse error) →
    # keeps a good rendering serveable through API flakiness.
    #
    # Gate the in-memory body BEFORE touching disk — mirrors the extract_md
    # spine (gate final_md in memory, _save_md only after PASS). This closes
    # the transient "md on disk ⟺ gated" window: the old order wrote the md to
    # disk and only THEN gated, so a daemon crash between the write and the gate
    # decision left an UNGATED firecrawl stub on disk that serve-safety would
    # hand out as a text_path for up to one reconcile interval (rule 1: any md
    # on disk ⇒ text_path) before the idempotent re-entry re-gated it. Gating
    # first means a fresh stub is NEVER on disk, even transiently. On FAIL the
    # helper's unlink is a harmless no-op (no file was written yet) and the
    # md-field clears are no-ops (still unset). ``md`` is already the body (no
    # frontmatter) — pass it straight through.
    # IDENTITY gate (2026-06-03): the completeness gate below only judges
    # truncated / paywall / mid-sentence — a COMPLETE but WRONG article (a
    # borderline/wrong DOI, or a stale paper.url rendering a different paper's
    # full body) would pass it and be served as THIS paper's full text. The PDF
    # tiers are identity-checked by _verify_pdf_matches_metadata; this path had
    # none. A firecrawl rendering is the article BODY and often lacks a clean
    # title/author header (it can start mid-Introduction), so we CANNOT key on
    # title/author. Instead require the body to be topically consistent with the
    # paper's OWN abstract (which summarizes it): a wrong article shares almost
    # none of the abstract's content words. Conservative — fires only on a CLEAR
    # mismatch, and only when there's a substantial abstract to compare against
    # (no abstract → skip → accept, so this never false-rejects a real rendering
    # whose abstract we simply don't hold).
    _abs = (paper.abstract or "").strip()
    if len(_abs) >= 120:
        _aw = set(re.findall(r"[a-z]{5,}", _abs.lower()))
        _bw = set(re.findall(r"[a-z]{5,}", md[:8000].lower()))
        _ov = (len(_aw & _bw) / len(_aw)) if _aw else 1.0
        if _ov < 0.30:
            library.log({"event": "firecrawl_identity_reject", "key": paper.key,
                         "abstract_body_overlap": round(_ov, 2),
                         "title": (paper.title or "")[:80]})
            return False

    if not _gate_firecrawl_md(paper, library, md, target_url=target_url):
        return False

    # PASS: persist the rendering. _gate_firecrawl_md already stamped status
    # ok + source firecrawl + firecrawl_pdf_hunt_exhausted; record the md path.
    md_path = library.md_path(paper.key)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = md_path.with_suffix(md_path.suffix + ".tmp")
    tmp.write_text(frontmatter + md)
    tmp.replace(md_path)

    paper.md_path = str(md_path.relative_to(library.root))
    paper.md_engine = "firecrawl"
    paper.md_engine_version = "v1"

    # ✦ Phase 12.3 (#94): pre-curator quality gate (Phase 28 route B
    # reframe — the gate originally protected the 5-Q insight worker
    # from junk input; the worker is gone, but the same flag now
    # excludes the paper from list_undistilled so the Librarian-side
    # paper-curator doesn't bounce on stubs.)
    # The md firecrawl scraped back may be a publisher landing page / paywall
    # stub / CAPTCHA, not a real paper. Run 1 LLM judge to intercept. Reuse
    # extract.py: review_extract (MiMo via get_llm() default). On failure → set
    # insight_invalid_reason, and MCP list_undistilled auto-excludes this paper.
    #
    # S4 (D4): review_extract is now CLARITY-ONLY — the paywall-stub /
    # landing-page judgment (the old ``broken_pdf_suspected`` axis) moved
    # entirely to completeness_gate, which already ran in
    # ``_gate_firecrawl_md`` above and deletes a stub before we ever reach
    # here. So this pre-curator read is just the clarity verdict (``ok``); an
    # unreadable firecrawl rendering still excludes the paper from the
    # curator's worklist.
    try:
        from .extract import review_extract
        verdict = review_extract(md)  # body only, frontmatter must not confuse LLM
        if not verdict.get("ok", True):
            paper.insight_invalid_reason = "firecrawl_stub_suspected"
            library.log({
                "event": "firecrawl_pre_insight_review_failed",
                "key": paper.key,
                "verdict": verdict,
            })
    except Exception as exc:
        # Do not fail download — when the review LLM is unavailable, the
        # Phase 12.2 worker self-check still acts as final defense
        library.log({
            "event": "firecrawl_pre_insight_review_error",
            "key": paper.key,
            "exc": str(exc),
        })

    library.log({
        "event": "firecrawl_text_only_fallback",
        "key": paper.key,
        "url": target_url,
        "md_len": len(md),
    })
    return True


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


def _try_iopscience_direct(paper: Paper, max_attempts: int = 3) -> Optional[bytes]:
    """For IOPscience-hosted DOIs (10.3847 = AAS/ApJ; 10.1088 = IOP) try the
    direct PDF endpoint. The article landing page (``/article/{doi}``) is
    routinely blocked by Radware Bot Manager, but the PDF endpoint
    (``/article/{doi}/pdf``) has a separate, weaker policy — discovered
    during R10/R11 of the 82-paper bench: 15 stubborn ApJ papers
    recoverable here when crossref_tm and scihub had missed.

    Some mihomo exit IPs get Radware-redirected to perfdrive.com/captcha
    (response is 200 + ~14KB HTML, not a PDF). Retry with fresh connections
    to give different IPs a chance.
    """
    doi = paper.doi or ""
    if not (doi.startswith("10.3847/") or doi.startswith("10.1088/")):
        return None
    url = f"https://iopscience.iop.org/article/{doi}/pdf"
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, timeout=TIMEOUT, allow_redirects=True,
                             headers={**BROWSER_HEADERS,
                                      "Accept": "application/pdf,*/*;q=0.8"})
            if r.ok and _is_pdf_bytes(r.content):
                return r.content
        except Exception:
            pass
        if attempt < max_attempts - 1:
            time.sleep(1 + 2 * attempt)
    return None


def _try_mdpi_scrapling(paper: Paper) -> Optional[bytes]:
    """MDPI (10.3390/...) papers are CC-BY OA but the canonical URLs are
    fronted by Akamai Bot Manager — direct ``requests`` calls return 403,
    and even Scrapling's solve_cloudflare alone gets a JS-challenge stub
    instead of the PDF. The working pattern (verified live for
    magnetochemistry9040091, magnetochemistry9040096, universe11060174):

      1. ``StealthyFetcher.fetch`` on ``https://doi.org/{doi}`` solves the
         landing-page challenge and lands on ``mdpi.com/{issn}/{vol}/{issue}/{art}``.
      2. Inside ``page_action``: parse ``citation_pdf_url`` meta.
      3. ``page.evaluate('window.location.href = pdf_url')`` triggers the
         Akamai ``bm-verify`` meta-refresh challenge. The headless browser's
         JS engine executes the challenge naturally.
      4. ``page.expect_download()`` captures the resulting PDF stream.

    Slow (~15-25s) so kept near the end of the cascade. No-ops for
    non-MDPI DOIs.
    """
    doi = paper.doi or ""
    if not doi.startswith("10.3390/"):
        return None
    try:
        from scrapling.fetchers import StealthyFetcher  # type: ignore
    except ImportError:
        return None
    import tempfile, os as _os, os.path as _osp
    save_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    state = {"pdf_path": None}

    def _action(page):
        html = page.content()
        m = re.search(
            r'name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)',
            html, re.I)
        if not m:
            return page
        pdf_url = m.group(1)
        try:
            with page.expect_download(timeout=45000) as dl_info:
                page.evaluate(f'window.location.href = "{pdf_url}"')
            dl = dl_info.value
            dl.save_as(save_path)
            state["pdf_path"] = save_path
        except Exception:
            pass
        return page

    try:
        StealthyFetcher.fetch(
            f"https://doi.org/{doi}",
            solve_cloudflare=True, network_idle=True, timeout=60000,
            humanize=False, geoip=False, page_action=_action)
    except Exception:
        return None
    if state["pdf_path"] and _osp.exists(state["pdf_path"]):
        try:
            with open(state["pdf_path"], "rb") as f:
                data = f.read()
            _os.unlink(state["pdf_path"])
            if _is_pdf_bytes(data):
                return data
        except Exception:
            return None
    return None


# Ordered cascade. Sorted by Round 1 hit-rate (high first), with
# concurrent groups consolidating low-hit-but-cheap aggregators.
# Each entry = (source_tag, callable). Group dispatchers count as one
# tier in this list but run their members concurrently.
_STRATEGIES = [
    # === Free, fast, operator-controlled / preprint ===
    ("url_override", _try_url_overrides),         # 0% but free, operator escape hatch
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
    # is only reached if all 18 tiers MISS. If a tier lands a real PDF first
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
