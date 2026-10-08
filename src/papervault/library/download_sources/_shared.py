from __future__ import annotations

import logging
from typing import Optional

from ..models import (
    DOWNLOAD_STATUS_FAILED,
    DOWNLOAD_STATUS_METADATA_ONLY,
    DOWNLOAD_STATUS_OK,
    Paper,
)
from ..store import Library


log = logging.getLogger("papervault.library.download")


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


def _is_pdf_bytes(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def _load_stealthy_fetcher(source: str) -> Optional[type]:
    """Lazy browser import, with a diagnostic for direct tier callers too."""
    try:
        from scrapling.fetchers import StealthyFetcher
    except ImportError:
        log.warning("%s: browser dependency unavailable; install the grey extra "
                    "(scrapling[fetchers])", source)
        return None
    return StealthyFetcher


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
        from ..extract import confirmed_completeness_gate
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
