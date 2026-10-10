from __future__ import annotations

import os
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Callable, Optional
from urllib.parse import unquote, urljoin, urlsplit

import requests

from ..models import Paper
from ..download_telemetry import browser_fetch
from ._shared import (
    BROWSER_HEADERS,
    TIMEOUT,
    USER_AGENT,
    _is_pdf_bytes,
    _load_stealthy_fetcher,
    log,
)
from .scihub import _SCIHUB_FALLBACK_DOMAINS, _SCIHUB_PDF_PATTERNS


@dataclass
class _AnnasLink:
    url: str
    label: str
    section: str = ""
    note: str = ""


class _AnnasLinks(HTMLParser):
    """Read anchors and the primary download link/notes in each section's list item."""

    def __init__(self):
        super().__init__()
        self.links: list[_AnnasLink] = []
        self.downloads: list[_AnnasLink] = []
        self.section = ""
        self.heading = None
        self.anchor = None
        self.item = None
        self.item_links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "h3":
            self.heading = []
            self.section = ""
        elif tag == "li":
            self.item, self.item_links = [], []
        elif tag == "a":
            self.anchor = (attrs.get("href", ""), attrs.get("class", ""), [])

    def handle_data(self, data):
        if self.heading is not None:
            self.heading.append(data)
        if self.anchor is not None:
            self.anchor[2].append(data)
        if self.item is not None:
            self.item.append(data)

    def handle_endtag(self, tag):
        if tag == "h3" and self.heading is not None:
            title = " ".join(self.heading).lower()
            self.section = next((s for s in ("fast", "slow", "external") if s in title), "")
            self.heading = None
        elif tag == "a" and self.anchor is not None:
            href, classes, text = self.anchor
            link = _AnnasLink(href, " ".join(text).strip(), self.section)
            self.links.append(link)
            if self.item is not None:
                self.item_links.append((link, "js-download-link" in classes.split()))
            self.anchor = None
        elif tag == "li" and self.item is not None:
            if self.item_links and self.section:
                link = next((link for link, primary in self.item_links if primary),
                            self.item_links[0][0])
                link.note = " ".join(self.item)
                self.downloads.append(link)
            self.item, self.item_links = None, []


@dataclass
class _AnnasOption:
    url: str
    source: str
    browser: bool
    waitlist: bool


def _annas_download_options(html: str, base: str) -> list[_AnnasOption]:
    """Port the SDK's section/source filtering, with requirement cost sorted first.

    Fast partners consume the membership quota and belong to the explicit API
    fallback. SciDB's JS-only record links are lookup links, not file options.
    """
    page = _AnnasLinks()
    page.feed(html)
    priority = {"libgen": 0, "scihub": 1, "scidb": 2, "slow_partner": 3,
                "nexus": 4, "ipfs": 5, "unknown": 6}
    options = []
    seen = set()
    for link in page.downloads:
        try:
            url = urljoin(base, link.url)
            parts = urlsplit(url)
        except ValueError:
            continue
        label, note = link.label.lower(), link.note.lower()
        host = (parts.hostname or "").lower()
        if (parts.scheme not in {"http", "https"} or not host or url in seen
                or any(s in host for s in (".onion", "libstc.cc", "libgen.is"))
                or "torrent" in label or "torrent" in parts.path
                or any(s in label for s in ("motrix", "cloudconvert", "send to kindle",
                                            "amazon", "download manager", "gopeed"))
                or link.section == "fast" or "/fast_download/" in parts.path
                or "/dyn/api/" in parts.path):
            continue
        if "scidb" in parts.path or "scidb" in label:
            # The SDK classifies these as JS_REQUIRED, outside NONE/BROWSER_AUTO.
            if not unquote(parts.path).lower().endswith(".pdf"):
                continue
            source = "scidb"
        elif "sci-hub" in host or "sci-hub" in label:
            source = "scihub"
        elif "libgen" in host or "libgen" in label:
            source = "libgen"
        elif "/slow_download/" in parts.path:
            source = "slow_partner"
        elif "nexus" in label or "nexusstc" in url:
            source = "nexus"
        elif "ipfs" in url.lower() or "ipfs" in label:
            source = "ipfs"
        else:
            source = "unknown"
        browser = (link.section == "slow" or "browser verification" in note)
        browser = browser and "no browser verification" not in note
        waitlist = "waitlist" in note and "no waitlist" not in note
        options.append(_AnnasOption(url, source, browser, waitlist))
        seen.add(url)
    return sorted(options, key=lambda o: (o.browser, o.waitlist, priority[o.source]))


def _annas_get(url: str):
    """Lightweight browser TLS transport; never send the member cookie to partners."""
    headers = {**BROWSER_HEADERS, "Referer": url}
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        # requests need not have its optional Brotli decoder installed.
        headers["Accept-Encoding"] = "gzip, deflate"
        return requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    return curl_requests.get(url, headers=headers, timeout=60,
                             allow_redirects=True, impersonate="chrome")


def _annas_resolve_option(option: _AnnasOption, render: Callable) -> tuple[Optional[bytes], str]:
    """Resolve at most three source-specific hops, accepting only actual PDF files."""
    url = option.url
    parts = urlsplit(url)
    siblings = iter(f"https://{host}{parts.path}" for host in _SCIHUB_FALLBACK_DOMAINS
                    if host != parts.hostname) if option.source == "scihub" else iter(())
    visited = set()
    for _ in range(3):
        if url in visited:
            break
        visited.add(url)
        try:
            response = _annas_get(url)
        except Exception as exc:
            sibling = next(siblings, None)
            if sibling:
                url = sibling
                continue
            return None, f"download refused: {option.source} {type(exc).__name__}"
        if (response.status_code == 403
                and b"link expired or invalid" in response.content[:2000].lower()):
            return None, "expired"
        content_type = response.headers.get("content-type", "").lower()
        file_type = "html" not in content_type and any(
            t in content_type for t in ("pdf", "octet", "epub", "djvu"))
        if response.status_code == 200 and file_type:
            if _is_pdf_bytes(response.content):
                return response.content, ""
            return None, f"download refused: {option.source} non-PDF file"
        html = response.text
        # Protected slow/IPFS pages can still need a browser after a successful
        # HTTP response carrying the challenge. External ordinary refusals do not.
        if (response.status_code == 403 and option.browser) or "ddos-guard" in html.lower():
            html = render(url, 'a[href*=".pdf" i], a:has-text("Download now"), '
                               'a:has-text("GET")')
            if html is None:
                return None, f"download refused: {option.source} browser challenge"
        elif response.status_code != 200:
            sibling = next(siblings, None)
            if sibling:
                url = sibling
                continue
            return None, f"download refused: {option.source} HTTP {response.status_code}"
        page = _AnnasLinks()
        page.feed(html)
        next_href = None
        if option.source == "scihub":
            for pattern in _SCIHUB_PDF_PATTERNS:
                match = pattern.search(html)
                if match:
                    next_href = match.group(1).split("#", 1)[0]
                    break
        elif option.source == "libgen":
            next_href = next((link.url for link in page.links if "get.php" in link.url), None)
            if not next_href:
                next_href = next((link.url for link in page.links if "ads.php" in link.url), None)
        if not next_href:
            for link in page.links:
                try:
                    path = unquote(urlsplit(link.url).path).lower()
                except ValueError:
                    continue
                if (path.endswith(".pdf") or "/ipfs/" in path
                        or link.label.lower().strip() in {"download now", "download", "get"}):
                    next_href = link.url
                    break
        if not next_href:
            sibling = next(siblings, None)
            if sibling:
                log.info("annas_archive: Sci-Hub option has no file link; trying a sibling mirror")
                url = sibling
                continue
            return None, f"download refused: {option.source} no file link"
        try:
            url = urljoin(str(response.url), unescape(next_href))
            parts = urlsplit(url)
        except ValueError:
            break
        if (parts.scheme not in {"http", "https"} or not parts.netloc
                or "/fast_download/" in parts.path or "/dyn/api/" in parts.path):
            break
    return None, f"download refused: {option.source} resolver budget exhausted"


def _try_annas_archive_api(paper: Paper) -> Optional[bytes]:
    """DOI lookup → md5 detail options → source resolver → PDF bytes.

    Ported approach: eduresser/annas-archive-sdk (fetch.py), not vendored code.
    Direct, no-waitlist options precede browser/waitlist options. Within each
    group: LibGen, Sci-Hub, SciDB files, slow partners, Nexus, IPFS, unknown.
    Established scientific mirrors lead; less predictable gateways come later.
    Sci-Hub options may use the existing sibling-mirror list within the same
    three-hop budget when the offered mirror returns a robot wall or outage.
    Fast partner links are excluded: the 25/day fast_download.json API remains
    an explicitly logged fallback. The primary options do not spend that quota.

    A member key (ANNAS_ARCHIVE_API_KEY) remains necessary for browser lookup.
    /scidb/<DOI> is only an md5 lookup; its signed PDF anchors are never reused.
    Try at most eight options (three resolver hops each), with one fresh detail
    load on an expired signature. An API signature can instead use that one
    expiry retry if the primary route did not spend it. Each browser wait has
    a 60-second cap.
    """
    api_key = os.environ.get("ANNAS_ARCHIVE_API_KEY", "").strip()
    if not api_key or not paper.doi:
        log.info("annas_archive[%s]: skipped: missing member key or DOI", paper.key)
        return None
    StealthyFetcher = _load_stealthy_fetcher("annas_archive")
    if StealthyFetcher is None:
        return None
    base = "https://annas-archive.gl"

    def render(url, selector):
        rendered_html = None

        def capture(page):
            nonlocal rendered_html
            page.wait_for_selector(selector, state="attached", timeout=60000)
            page.wait_for_load_state("domcontentloaded", timeout=60000)
            rendered_html = page.content()

        try:
            response = browser_fetch(StealthyFetcher,
                url, headless=True, timeout=60000,
                cookies=[{"name": "aa_account_id2", "value": api_key, "url": base}],
                page_action=capture,
            )
        except Exception as exc:
            log.warning("annas_archive[%s]: download refused: browser %s",
                        paper.key, type(exc).__name__)
            return None
        # Scrapling swallows callback failures, so validate capture and status.
        if rendered_html is None or response.status != 200:
            log.warning("annas_archive[%s]: download refused: page did not finish rendering "
                        "(HTTP %s)", paper.key, response.status)
            return None
        return rendered_html

    record = render(f"{base}/scidb/{paper.doi}",
                    'a[href^="/md5/"], a[href*=".pdf" i], a[href*="%2epdf" i], '
                    'title:has-text("Search - Anna")')
    if record is None:
        return None
    if re.search(r"<title[^>]*>[^<]*Search - Anna", record, re.I):
        log.warning("annas_archive[%s]: not in Anna's index: DOI lookup returned search results",
                    paper.key)
        return None
    lookup = _AnnasLinks()
    lookup.feed(record)
    md5 = None
    for link in lookup.links:
        match = re.search(r"/md5/([a-f0-9]{32})(?:[/?#]|$)", link.url, re.I)
        if match:
            md5 = match.group(1).lower()
            break
    if not md5:
        log.warning("annas_archive[%s]: download refused: DOI record has no md5", paper.key)
        return None

    detail_url = f"{base}/md5/{md5}"
    selector = ('h3:has-text("Fast downloads"), h3:has-text("Slow downloads"), '
                'h3:has-text("External downloads"), title:has-text("Anna")')
    remaining = 8
    expired = False
    refresh_used = False
    refused_urls = set()
    reason = "no viable option"
    for refresh in range(2):
        detail = render(detail_url, selector)
        if detail is None:
            reason = "download refused: md5 detail page unavailable"
            break
        options = _annas_download_options(detail, detail_url)
        if not options:
            reason = "expired and no retry left: no viable option" if expired else "no viable option"
            break
        refresh_needed = False
        for option in options:
            # Refresh signatures without spending the budget on unchanged refusals.
            if option.url in refused_urls:
                continue
            if remaining == 0:
                break
            remaining -= 1
            pdf, failure = _annas_resolve_option(option, render)
            if pdf is not None:
                log.info("annas_archive[%s]: retrieved md5=%s via %s (no fast-download quota)",
                         paper.key, md5, option.source)
                return pdf
            if failure == "expired":
                expired = True
                if refresh == 0 and remaining:
                    log.warning("annas_archive[%s]: option expired; refreshing md5 detail page",
                                paper.key)
                    refresh_used = True
                    refresh_needed = True
                    break
                reason = "expired and no retry left"
            else:
                refused_urls.add(option.url)
                reason = failure
                log.warning("annas_archive[%s]: %s", paper.key, reason)
        if not refresh_needed:
            if expired:
                reason = "expired and no retry left"
            break
    log.warning("annas_archive[%s]: %s (md5=%s)", paper.key, reason, md5)
    log.warning("annas_archive[%s]: trying fast_download API fallback (25/day quota)", paper.key)
    for api_attempt in range(1 if refresh_used else 2):
        try:
            api = requests.get(
                f"{base}/dyn/api/fast_download.json", params={"md5": md5, "key": api_key},
                timeout=TIMEOUT, headers={"User-Agent": USER_AGENT},
            )
            data = api.json()
            url = data.get("download_url") if isinstance(data, dict) else None
        except Exception as exc:
            # Never log exceptions' URLs: the API query carries the member secret.
            log.warning("annas_archive[%s]: download refused: fast_download API %s",
                        paper.key, type(exc).__name__)
            return None
        if not api.ok or not url:
            log.warning("annas_archive[%s]: download refused: fast_download API HTTP %s, "
                        "no usable download_url", paper.key, api.status_code)
            return None
        try:
            pdf = requests.get(url, timeout=60, allow_redirects=True,
                               headers={"User-Agent": USER_AGENT})
            if pdf.ok and _is_pdf_bytes(pdf.content):
                return pdf.content
            if (pdf.status_code == 403
                    and b"link expired or invalid" in pdf.content[:2000].lower()):
                if api_attempt == 0 and not refresh_used:
                    log.warning("annas_archive[%s]: API option expired; resolving fresh option "
                                "(25/day quota)", paper.key)
                    continue
                log.warning("annas_archive[%s]: expired and no retry left: API option", paper.key)
            else:
                log.warning("annas_archive[%s]: download refused: API option HTTP %s or non-PDF",
                            paper.key, pdf.status_code)
        except Exception as exc:
            log.warning("annas_archive[%s]: download refused: API option %s",
                        paper.key, type(exc).__name__)
        return None
    return None
