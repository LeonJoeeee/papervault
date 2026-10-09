from __future__ import annotations

import os
import re

import requests

from ..models import DOWNLOAD_STATUS_OK, Paper
from ..store import Library
from ._shared import _gate_firecrawl_md, _strip_frontmatter


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
      * the PDF cascade has fully missed
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
    # whole-document gate as the OCR spine. The PDF cascade already missed,
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
        from ..extract import review_extract
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
