"""The reconcile sweep (D8 / SDD §6.4).

``reconcile_once(library, download_queue, extract_queue)`` scans the whole
library, asks :func:`services.classify.classify` for each paper's one next
Action, and re-enqueues the papers that the two queues' own recovery scans
miss — most importantly the ~48 firecrawl-md papers that have an md on disk
but no PDF (``classify`` routes them DOWNLOAD to hunt the real PDF, while the
download queue's recovery scan only re-enqueues ``status==pending``).

It is the automatic safety net for "a paper fell between the two conveyor
belts": run it once at startup and then on a timer. It is **idempotent**, but
the convergence mechanism differs by population (low fix #3 — the old "every
add is idempotent via the worker's has_pdf/has_extract short-circuit" claim
was false for exactly the population reconcile exists to rescue):

  * **PDF-bearing papers** (EXTRACT route, or a DOWNLOAD that lands a real
    PDF): the worker's ``has_pdf`` / ``has_extract(md)`` short-circuit IS the
    final guard — a re-add of a paper already moving through is a no-op.
  * **firecrawl-md papers** (``has_md`` ∧ ``¬has_pdf``, classify rule 3): these
    NEVER hit the download worker's ``has_pdf`` short-circuit (it only fires
    when has_pdf is True). A re-enqueue actually re-runs the full PDF
    ``download_paper`` + the firecrawl re-gate (an LLM call). Convergence to
    "PDF hunt + re-gate at most once per paper" rests on the persistent
    ``firecrawl_pdf_hunt_exhausted`` stamp (set on the gate PASS in
    ``_gate_firecrawl_md``) flipping classify rule (3) off — NOT on the worker
    guard. F7: the queues now ALSO carry a lightweight in-flight dedup (``add``
    skips a key already sitting in the queue, cleared on dequeue), so a second
    sweep at +600s while the key still WAITS in the queue is suppressed and
    cannot re-enqueue + re-run the gate within one queue-residency window. The
    narrow residual (a re-add arriving AFTER the worker dequeues but before the
    download saves the stamp) stays bounded + safe by the stamp (it converges
    within one download cycle). Both layers hold: the stamp bounds the
    cross-cycle case, the in-flight dedup closes the same-window double sweep.

Reconcile reads only disk facts + ``download_status`` via ``classify`` and
mutates NOTHING on the papers (the queues do the enqueuing). Terminal papers
(``extract_failed`` / ``failed`` / ``metadata_only``) are skipped — they are
audit-only resets, never auto-revived (D7/D9).

Scanning is batched with a yield between batches so a multi-thousand-paper
sweep doesn't monopolize the event loop, and a single bad record is logged and
skipped without aborting the sweep (the routing of one paper never blocks the
rest, SDD §6.4).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import os
import re
import time
from typing import Optional

from .. import fetch
from ..store import Library
from .classify import DOWNLOAD, EXTRACT, classify
from . import concurrency
from . import extract_defer

log = logging.getLogger("papervault.library.reconcile")

# Papers scanned per batch before yielding control back to the event loop.
_BATCH_SIZE = 200
# Rotating cursor across transport-deferred extracts so a few corrupt PDFs at
# the head of the list cannot starve healthy deferred papers from canary probes.
_defer_canary_cursor = 0
# Maximum wall-clock time spent politely pacing a single metadata sweep.
_POLITE_SWEEP_BUDGET_S = 120.0


async def _wait_polite_gate(service: str, *, max_wait: float = 15.0) -> bool:
    """Wait for politeness-v2 headroom before issuing a background sweep call.

    Returns True when permitted (or when politeness-v2 is disabled), False when
    the scope is disabled or cooling down longer than ``max_wait``.
    """
    from papervault.config import polite_v2_enabled
    if not polite_v2_enabled():
        return True
    from ..polite_http import get_ledger
    from ..politeness_v2 import Scope

    scope = Scope(service)
    deadline = time.monotonic() + max_wait
    while True:
        try:
            decision = get_ledger().gate(scope)
        except Exception:
            return False
        if decision.allowed:
            return True
        if decision.reason == "disabled":
            return False
        wait = decision.retry_after if decision.retry_after > 0 else 0.25
        remaining = deadline - time.monotonic()
        if remaining <= 0 or (decision.reason == "cooldown" and wait > remaining):
            return False
        await asyncio.sleep(min(wait, remaining, 5.0))


async def _prio_by_pages(library: Library, key: str) -> int:
    """PRIORITY_LOW for a long paper (>90 pages), PRIORITY_NORMAL otherwise.

    Long papers go to the slow lane (D10) so a 200-page review doesn't block
    the short-paper backlog. Page count comes from the isolated ``pdf_probe``
    subprocess (D10: never an in-thread parse). A bad / unreadable probe falls
    back to the normal lane — ``extract_md`` re-probes and routes it to the
    terminal ``extract_failed`` there, so misrouting it to normal here is
    harmless.

    **Off-loop (med fix)**: ``pdf_probe`` is a synchronous ``subprocess.run``
    with a 30s hard timeout. ``reconcile_once`` is async and runs on the live
    server's 600s timer *while it serves MCP requests*, so a probe spawned
    in-line would block the whole event loop for the full probe duration —
    worst case 30s on a wedged PDF, serially per EXTRACT-routed paper. We run
    it via ``asyncio.to_thread`` so the loop can interleave request handling;
    the probe's own subprocess + timeout still bound the actual work.
    """
    from ..extract import pdf_probe, _LONG_PAPER_PAGE_THRESHOLD
    long_threshold = _LONG_PAPER_PAGE_THRESHOLD
    pdf_path = library.pdf_path(key)
    if not pdf_path.exists():
        return concurrency.PRIORITY_NORMAL
    probe = await asyncio.to_thread(pdf_probe, str(pdf_path))
    if probe.n_pages > long_threshold:
        return concurrency.PRIORITY_LOW
    return concurrency.PRIORITY_NORMAL


# ── metadata enrichment (root-cause fix for missing venue/volume/pages) ──
# Per-sweep cap on Crossref probes (polite pool) + a DEDICATED semaphore so
# enrich never starves the shared PDF-download network pool.
_ENRICH_CAP = 150
_enrich_sem = asyncio.Semaphore(2)


def _needs_enrich(p) -> bool:
    """A DOI-bearing row still missing a journal locator, not yet probed."""
    return (bool((p.doi or "").strip())
            and (not p.venue or not p.volume or not p.pages)
            and p.enriched_at is None)


async def _enrich_one(doi: str) -> tuple[str, dict]:
    from ..fetch import crossref_enrich_lookup
    async with _enrich_sem:                       # <=2 concurrent Crossref GETs
        return await asyncio.to_thread(crossref_enrich_lookup, doi)


# ── by-title DOI resolution (root-cause fix for no-DOI in-domain stubs) ──
# A no-DOI in-domain row (title + authors + year + abstract, no doi, no usable
# arxiv) is STRUCTURALLY un-rescuable by the enrich sweep: _needs_enrich gates
# on bool(p.doi), so it is never a candidate, and the DOI-keyed download cascade
# can never fetch its full text. This sweep gives such a row a DOI BY TITLE
# (conservative, abstain-by-default — see fetch.resolve_doi_by_title) so it
# becomes _needs_enrich-eligible in the SAME sweep's enrich pass and the cascade
# can later fetch it. Own (smaller) cap + semaphore so it never starves the
# enrich/download pools.
_RESOLVE_CAP = 50
_resolve_sem = asyncio.Semaphore(2)


def _needs_doi_resolve(p) -> bool:
    """An in-domain row with NO doi, NO usable arxiv, but an abstract (so it is
    a real searchable stub), not yet probed by the resolver. ``all_papers()``
    already excludes domain-quarantined rows, so in-domain is implied."""
    from ..fetch import looks_like_arxiv
    if (p.doi or "").strip():
        return False
    arxiv = (p.arxiv_id or "").strip()
    if arxiv and looks_like_arxiv(arxiv):
        return False
    if not (p.abstract or "").strip():
        return False
    return getattr(p, "resolve_attempted_at", None) is None


async def _resolve_one(title: str, authors: list, year) -> tuple[str, str]:
    """Off-loop by-title resolve. Returns ("ok", doi) | ("miss", "") |
    ("transient", ""). The resolver returns None for BOTH a clean no-match and
    a transient failure; we can't tell them apart from its return alone, so we
    re-derive the transient case from the Crossref call inside it being a
    network/HTTP failure. To keep the miss-vs-transient split (load-bearing for
    the termination guard) we wrap the candidate fetch: a None from a SUCCESSFUL
    query is a miss (stamp), a None from a FAILED query is transient (retry)."""
    async with _resolve_sem:
        def _call():
            # Re-run the conservative resolver, but split miss vs transient by
            # inspecting the raw candidate fetch first (so a 429/outage doesn't
            # permanently mark a resolvable stub as attempted).
            t = (title or "").strip()
            from ..models import normalize_title
            if not t or len(normalize_title(t)) < fetch._MIN_NORM_TITLE_LEN:
                return ("miss", "")  # degenerate stub -> stamp, don't re-probe
            first = fetch._first_surname(authors)
            cands = fetch._crossref_title_candidates(t, first)
            if cands is None:
                return ("transient", "")  # network/429/5xx -> retry next sweep
            doi = fetch.resolve_doi_by_title(t, authors, year)
            return ("ok", doi) if doi else ("miss", "")
        return await asyncio.to_thread(_call)


async def _resolve_doi_sweep(library: Library, *, cap: int) -> int:
    """Resolve DOIs by title for up to ``cap`` no-DOI in-domain stubs. Fetch OFF
    the event loop; apply (Library.set_resolved_doi) + ``save`` ONCE under
    ``lib_write_lock``. ``resolve_attempted_at`` stamps every NON-transient
    outcome (resolved OR clean-miss/abstain) so the sweep is self-terminating;
    transient errors stay unstamped and retry next sweep (fail-open). Mirrors
    ``_enrich_sweep``."""
    cands = [p for p in library.all_papers() if _needs_doi_resolve(p)][:cap]
    if not cands:
        return 0
    keys = [p.key for p in cands]
    results = await asyncio.gather(*(
        _resolve_one(p.title, list(p.authors or []), p.year) for p in cands))
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    resolved = 0
    mutated = 0
    async with concurrency.lib_write_lock:
        for key, (status, doi) in zip(keys, results):
            if status == "transient":
                continue                          # fail-open: do NOT stamp, retry later
            p = library.get(key)
            if p is None:
                continue
            if status == "ok" and doi:
                outcome, _detail = library.set_resolved_doi(
                    key, doi, provenance="doi_resolved", resolved_at=now)
                if outcome in ("set", "collision"):
                    resolved += 1
                    mutated += 1
                # On "collision" the stub was merged+purged (no row to stamp).
                if outcome in ("set", "has_doi"):
                    p.resolve_attempted_at = now
                    mutated += 1
            else:  # clean miss / abstain -> stamp so it isn't re-probed
                p.resolve_attempted_at = now
                mutated += 1
        if mutated:
            await asyncio.to_thread(library.save)
    log.info("reconcile: doi-resolve probed %d -> resolved %d", len(cands), resolved)
    return resolved


async def _enrich_sweep(library: Library, *, cap: int) -> int:
    """Backfill missing venue/volume/pages from Crossref-by-DOI for up to ``cap``
    DOI-bearing rows. Fetch OFF the event loop; apply (fill-blanks via
    ``Library.enrich``) + ``save`` ONCE under ``lib_write_lock``. ``enriched_at``
    stamps every non-transient outcome (ok OR clean-miss) so the sweep is
    self-terminating; transient errors stay unstamped and retry next sweep. This
    drains the existing no-venue backlog over sweeps AND completes new
    DOI-bearing search-ingest rows — the single root-cause + backfill mechanism.
    """
    from papervault.config import polite_v2_enabled

    cands = [p for p in library.all_papers() if _needs_enrich(p)][:cap]
    if not cands:
        return 0
    targets = [(p.key, (p.doi or "").strip()) for p in cands]
    processed_targets: list[tuple[str, str]] = []
    results: list[tuple[str, dict]] = []
    if polite_v2_enabled():
        # Respect politeness-v2 concurrency=1 & window pacing instead of bursting
        # all candidates concurrently and dropping 119/120 with PolitenessDenied.
        deadline = time.monotonic() + _POLITE_SWEEP_BUDGET_S
        for key, doi in targets:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not await _wait_polite_gate("crossref", max_wait=min(15.0, remaining)):
                break
            processed_targets.append((key, doi))
            results.append(await _enrich_one(doi))
    else:
        processed_targets = targets
        results = await asyncio.gather(*(_enrich_one(doi) for _, doi in targets))
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    filled = 0
    mutated = 0
    async with concurrency.lib_write_lock:
        for (key, doi), (status, fields) in zip(processed_targets, results):
            if status == "transient":
                continue                          # fail-open: do NOT stamp, retry later
            if library.get(key) is None:
                continue
            mutated += 1
            if library.enrich(key, fields, enriched_at=now, provenance={"incoming_source": "crossref",
                    "lookup_doi": doi, "checked_at": now}):
                filled += 1
        if mutated:
            await asyncio.to_thread(library.save)
    log.info("reconcile: enrich probed %d -> filled %d", len(processed_targets), filled)
    return filled


# ── abstract backfill (consistency invariant: every kept row must
#    carry an abstract; see docs/history/2026-06-03-consistency-mechanism.md, repo root).
#    The sweep only ADDS an abstract (never overwrites). ────────────────────────
_ABSTRACT_CAP = 100
_abstract_sem = asyncio.Semaphore(3)
_ABSTRACT_FLOOR = 40
_ABSTRACT_RETRY_DAYS = 30


def _parse_iso_utc(stamp: Optional[str]) -> Optional[_dt.datetime]:
    if not stamp:
        return None
    try:
        dt = _dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=_dt.timezone.utc)
    except ValueError:
        return None


def _has_newer_extract(library: Library, p, attempted_dt: _dt.datetime) -> bool:
    """True iff an on-disk extract for ``p`` is newer than ``attempted_dt``."""
    cutoff = attempted_dt.timestamp()
    checked = []
    for rel in (p.md_path, p.txt_path):
        if rel:
            checked.append(library.root / rel)
    if not checked:
        checked.extend((library.md_path(p.key), library.txt_path(p.key)))
    for path in checked:
        try:
            if path.stat().st_mtime > cutoff:
                return True
        except OSError:
            continue
    return False


def _needs_abstract(p, library: Optional[Library] = None, *,
                    now: Optional[_dt.datetime] = None) -> bool:
    if len((p.abstract or "").strip()) >= _ABSTRACT_FLOOR:
        return False
    attempted = getattr(p, "abstract_attempted_at", None)
    if not attempted:
        return True
    attempted_dt = _parse_iso_utc(attempted)
    if attempted_dt is None:
        return True
    if library is not None and _has_newer_extract(library, p, attempted_dt):
        return True
    current = now or _dt.datetime.now(_dt.timezone.utc)
    return (current - attempted_dt) >= _dt.timedelta(days=_ABSTRACT_RETRY_DAYS)


def _clean_extract_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[*#`]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _read_extract_head(library: Library, p, limit: int) -> str:
    rel = (p.md_path or p.txt_path or "").strip()
    if not rel:
        return ""
    try:
        with (library.root / rel).open("r", encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except Exception:
        return ""


def _abstract_section(library: Library, p) -> str:
    """The row's OWN VERBATIM labelled Abstract section from its extracted full
    text. '' if no extract / no labelled abstract. Local, no network, no
    generation — the safest source."""
    txt = _read_extract_head(library, p, 30000)
    if not txt:
        return ""
    m = re.search(
        r"(?:^|\n)\s*(?:\*\*|#+\s*)?abstract\b[\s.:*]*\n?(.{120,2500}?)"
        r"(?:\n\s*(?:\*\*|#+\s*)?(?:1\.?\s+)?(?:introduction|keywords|key words|"
        r"contents)\b|\n\s*\n\s*\n)", txt, re.IGNORECASE | re.DOTALL)
    if m:
        ab = _clean_extract_text(m.group(1))
        if 120 <= len(ab) <= 2600:
            return ab
    return ""


def _opening_paragraph(library: Library, p) -> str:
    """LAST-RESORT abstract: the FIRST substantive PROSE paragraph in the head of
    the row's own extract (an old paper / letter with no labelled Abstract — its
    lede serves as the abstract). Tightened to skip headers / author-affiliation
    blocks / reference lists / figure-table captions, so it does not store a
    non-abstract paragraph as the abstract (verified drill finding)."""
    txt = _read_extract_head(library, p, 12000)
    if not txt:
        return ""
    for para in re.split(r"\n\s*\n", txt)[:25]:        # document head only
        c = _clean_extract_text(para)
        if len(c) < 200:
            continue
        low = " " + c.lower() + " "
        stop = sum(low.count(w) for w in
                   (" the ", " of ", " and ", " we ", " in ", " that ", " is "))
        if stop < 5:                                   # must read like prose
            continue
        if re.match(r"^\s*(\[\d|\d+\.\s|fig(\.|ure)\b|table\b|references\b|"
                    r"acknowledg|doi[:\s]|http)", c, re.IGNORECASE):
            continue                                   # caption / header / ref start
        if len(re.findall(r"\(\d{4}\)|\bet al\b", c, re.IGNORECASE)) >= 3:
            continue                                   # reference-dense block
        return c[:2000]
    return ""


async def _abstract_probe(library: Library, p, *, polite_deadline: Optional[float] = None) -> tuple[str, str]:
    """3-state abstract backfill probe: ("ok", ab) | ("miss", "") | ("transient", "")."""
    from papervault.config import polite_v2_enabled

    # Priority: the paper's OWN labelled Abstract section (verbatim, safest) ->
    # an authoritative source by DOI (OpenAlex/SS) -> the opening prose paragraph
    # (riskiest, so it runs ONLY after the others miss).
    section = await asyncio.to_thread(_abstract_section, library, p)
    if len(section) >= 60:
        return ("ok", section)
    doi = (p.doi or "").strip()
    doi_transient = False
    if doi:
        can_probe_doi = True
        if polite_v2_enabled():
            max_wait = 15.0 if polite_deadline is None else max(0.0, min(15.0, polite_deadline - time.monotonic()))
            oa_ok = await _wait_polite_gate("openalex", max_wait=max_wait)
            if not oa_ok:
                ss_wait = 15.0 if polite_deadline is None else max(0.0, min(15.0, polite_deadline - time.monotonic()))
                ss_ok = await _wait_polite_gate("semantic_scholar", max_wait=ss_wait)
                if not ss_ok:
                    can_probe_doi = False
                    doi_transient = True
        if can_probe_doi:
            from ..fetch import _fetch_abstract_by_doi_3state
            async with _abstract_sem:
                status, got = await asyncio.to_thread(_fetch_abstract_by_doi_3state, doi)
            if status == "ok" and len((got or "").strip()) >= 60:
                return ("ok", got.strip())
            if status == "transient":
                doi_transient = True
    opening = await asyncio.to_thread(_opening_paragraph, library, p)
    if len(opening) >= 60:
        return ("ok", opening)
    return ("transient", "") if doi_transient else ("miss", "")


async def _abstract_one(library: Library, p) -> str:
    status, ab = await _abstract_probe(library, p)
    return ab if status == "ok" else ""


def _select_abstract_candidates(papers, library: Library, *, cap: int,
                                now: _dt.datetime) -> list:
    """Select up to ``cap`` papers needing abstract backfill (off-loop stat safe)."""
    if cap <= 0:
        return []
    selected = []
    for p in papers:
        if _needs_abstract(p, library, now=now):
            selected.append(p)
            if len(selected) >= cap:
                break
    return selected


async def _abstract_sweep(library: Library, *, cap: int) -> int:
    """Backfill an abstract on up to ``cap`` no-abstract rows: verbatim from the
    row's own extract, else OpenAlex/SS by DOI. Fetch OFF the loop; fill-blanks
    (never overwrite) + save ONCE under the write lock. Non-transient misses
    stamp ``abstract_attempted_at`` so unfillable rows do not re-probe forever."""
    from papervault.config import polite_v2_enabled

    now_dt = _dt.datetime.now(_dt.timezone.utc)
    papers_snapshot = library.all_papers()
    cands = await asyncio.to_thread(_select_abstract_candidates, papers_snapshot, library, cap=cap, now=now_dt)
    if not cands:
        return 0
    processed_keys: list[str] = []
    results: list[tuple[str, str]] = []
    if polite_v2_enabled():
        deadline = time.monotonic() + _POLITE_SWEEP_BUDGET_S
        for p in cands:
            if time.monotonic() >= deadline and (p.doi or "").strip():
                break
            processed_keys.append(p.key)
            results.append(await _abstract_probe(library, p, polite_deadline=deadline))
    else:
        processed_keys = [p.key for p in cands]
        results = await asyncio.gather(*(_abstract_probe(library, p) for p in cands))
    now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
    filled = 0
    mutated = 0
    async with concurrency.lib_write_lock:
        for key, (status, ab) in zip(processed_keys, results):
            if status == "transient":
                continue
            row = library.get(key)
            if row is None or len((row.abstract or "").strip()) >= _ABSTRACT_FLOOR:
                continue
            ab = (ab or "").strip()
            if status == "ok" and len(ab) >= 60:
                row.abstract = ab
                row.abstract_attempted_at = None
                filled += 1
                mutated += 1
            else:
                row.abstract_attempted_at = now_iso
                mutated += 1
        if mutated:
            await asyncio.to_thread(library.save)
    log.info("reconcile: abstract sweep probed %d -> filled %d", len(processed_keys), filled)
    return filled


async def _path_backfill_sweep(library: Library) -> int:
    """Consistency backfill (add-only): a Paper whose md/txt EXISTS on disk but
    whose ``md_path``/``txt_path`` field is null — a torn write between the
    extract write and the record save, or a manually-placed extract. Serving
    already keys off ``has_extract`` (disk), so this is a cosmetic index lag, but
    it keeps the persisted record self-consistent (metadata⟷extract). The on-disk
    canonical location is the truth; set the relative path. NEVER overwrites a
    non-null path. Save ONCE under the write lock."""
    fixes: list[tuple[str, str]] = []           # (key, fmt)
    for scanned, p in enumerate(library.all_papers(), 1):
        if not (p.md_path or "").strip() and library.has_extract(p.key, "md"):
            fixes.append((p.key, "md"))
        if not (p.txt_path or "").strip() and library.has_extract(p.key, "txt"):
            fixes.append((p.key, "txt"))
        if scanned % _BATCH_SIZE == 0:
            await asyncio.sleep(0)
    if not fixes:
        return 0
    n = 0
    async with concurrency.lib_write_lock:
        for key, fmt in fixes:
            row = library.get(key)
            if row is None:
                continue
            field = "md_path" if fmt == "md" else "txt_path"
            if (getattr(row, field) or "").strip():
                continue                        # filled since the scan
            path = library.md_path(key) if fmt == "md" else library.txt_path(key)
            setattr(row, field, str(path.relative_to(library.root)))
            n += 1
        if n:
            await asyncio.to_thread(library.save)
    log.info("reconcile: path-backfill set %d md/txt path fields", n)
    return n


async def reconcile_once(
    library: Library,
    download_queue,
    extract_queue,
    *,
    batch_size: int = _BATCH_SIZE,
) -> dict:
    """Scan the library once and re-route every non-terminal paper (D8).

    For each paper, ``classify`` returns the one next Action:

      * ``DOWNLOAD`` → ``download_queue.add(key, NORMAL)``. Includes the
        firecrawl-md papers (md on disk, no PDF) that the download queue's
        own recovery scan forgets — this is the bug D8 fixes.
      * ``EXTRACT``  → ``extract_queue.add(key, prio_by_pages(key))``
        (>90 pages → PRIORITY_LOW, D10) — UNLESS the paper is transport-DEFERRED
        against its current PDF artifact (issue #43: the extraction backend was
        unreachable and has not recovered), in which case it is SKIPPED this
        sweep (counted ``extract_deferred``) rather than re-enqueued forever. A
        bounded canary slice of the deferred set is still promoted each sweep to
        detect backend recovery (counted ``extract_canary``).
      * ``TERMINAL`` → skipped (audit-only).

    Idempotent + safe to call on a timer. Returns a counts dict
    ``{"scanned", "download", "extract", "extract_deferred", "extract_canary",
    "terminal", "errors", ...}`` for logging.
    """
    from papervault.config import stoprules_v2_enabled
    if stoprules_v2_enabled():
        from ..stop_rules import evaluate_round
        await evaluate_round(library)
    counts = {"scanned": 0, "download": 0, "extract": 0,
              "terminal": 0, "errors": 0, "enriched": 0, "doi_resolved": 0,
              "extract_deferred": 0, "extract_canary": 0}

    # Issue #43: keys that classify routes EXTRACT but that are transport-DEFERRED
    # against their current PDF artifact (backend unreachable, no success since).
    # We do NOT re-enqueue these — that is the busy-loop fix — but we keep a
    # bounded canary slice (below) to detect backend recovery.
    deferred_extract: list[str] = []

    papers = library.all_papers()
    for start in range(0, len(papers), batch_size):
        batch = papers[start:start + batch_size]
        for paper in batch:
            counts["scanned"] += 1
            try:
                route = classify(paper, library)
                if route == DOWNLOAD:
                    download_queue.add(paper.key, concurrency.PRIORITY_NORMAL)
                    counts["download"] += 1
                elif route == EXTRACT:
                    if extract_defer.is_extract_deferred(paper, library):
                        # Skip: extraction was transport-deferred against this
                        # exact PDF and the backend has not recovered. Auto-lifts
                        # on artifact change or a sibling extract success (#43).
                        deferred_extract.append(paper.key)
                        counts["extract_deferred"] += 1
                    else:
                        prio = await _prio_by_pages(library, paper.key)
                        extract_queue.add(paper.key, prio)
                        counts["extract"] += 1
                else:  # TERMINAL — audit-only, never auto-revive.
                    counts["terminal"] += 1
            except Exception as exc:
                # One bad record (corrupt row, transient disk error) must not
                # abort the whole sweep — log it and move on (SDD §6.4).
                counts["errors"] += 1
                library.log({"event": "reconcile_item_error",
                             "key": getattr(paper, "key", "?"),
                             "error": repr(exc)[:200]})
        # Yield between batches so a large sweep doesn't monopolize the loop.
        await asyncio.sleep(0)

    # Issue #43 recovery probe: while papers are transport-deferred, still promote
    # a small bounded canary slice each sweep. If the backend has recovered, a
    # canary extraction succeeds → the process-global success epoch advances →
    # every remaining deferred paper goes stale (is_extract_deferred → False) and
    # is re-enqueued next sweep. During a stable outage this costs O(DEFER_CANARY)
    # attempts/sweep instead of O(pending-set) — the busy-loop is bounded, not
    # perpetual.
    global _defer_canary_cursor
    canary_count = min(extract_defer.DEFER_CANARY, len(deferred_extract))
    if canary_count > 0:
        start_idx = _defer_canary_cursor % len(deferred_extract)
        canary_keys = [
            deferred_extract[(start_idx + offset) % len(deferred_extract)]
            for offset in range(canary_count)
        ]
        _defer_canary_cursor = (start_idx + canary_count) % len(deferred_extract)
    else:
        canary_keys = []
    for key in canary_keys:
        try:
            prio = await _prio_by_pages(library, key)
            extract_queue.add(key, prio)
            counts["extract_canary"] += 1
            counts["extract_deferred"] -= 1
        except Exception:
            log.exception("reconcile: extract canary promotion failed for %s", key)

    # By-title DOI resolution pass — give no-DOI in-domain stubs a DOI so they
    # become enrich-eligible + download-able. Runs BEFORE the enrich pass so a
    # DOI resolved this sweep flows straight into the same sweep's enrich. The
    # resolver is conservative (abstains on ambiguity), so a wrong-DOI write is
    # near-impossible. Isolated so a Crossref hiccup never aborts routing.
    #
    # DORMANT BY DEFAULT (env-gated off). The auto-plug has a narrow residual
    # identical-title false-positive that wants human dry-run review (run the
    # `audit --resolve-stub-dois` batch first), and this path double-fetches
    # Crossref (refactor _resolve_one to pass already-fetched candidates before
    # enabling). Turn on only after the batch proves the real false-positive rate:
    #   env PAPER_PIPELINE_RESOLVE_STUB_DOIS=1
    if os.environ.get("PAPER_PIPELINE_RESOLVE_STUB_DOIS", "").strip().lower() in {"1", "true", "yes"}:
        try:
            counts["doi_resolved"] = await _resolve_doi_sweep(library, cap=_RESOLVE_CAP)
        except Exception:
            log.exception("reconcile doi-resolve pass failed; continuing")

    # Metadata enrichment pass — Crossref-by-DOI backfill of missing
    # venue/volume/pages (drains the existing-row backlog over sweeps + completes
    # new DOI-bearing search-ingest rows). Isolated so a Crossref hiccup never
    # aborts the routing sweep.
    try:
        counts["enriched"] = await _enrich_sweep(library, cap=_ENRICH_CAP)
    except Exception:
        log.exception("reconcile enrich pass failed; continuing")

    # Abstract backfill pass (consistency invariant) — ACTIVE, add-only. Gives a
    # no-abstract row an abstract from its own extract, else OpenAlex/SS by DOI.
    try:
        counts["abstract_filled"] = await _abstract_sweep(library, cap=_ABSTRACT_CAP)
    except Exception:
        log.exception("reconcile abstract pass failed; continuing")

    # Extract-path backfill (consistency invariant) — set md_path/txt_path when
    # the extract is on disk but the field is null (torn write). Add-only, no
    # network; cheap pure-disk scan. Isolated so it never aborts the sweep.
    try:
        counts["path_backfilled"] = await _path_backfill_sweep(library)
    except Exception:
        log.exception("reconcile path-backfill pass failed; continuing")

    library.log({"event": "reconcile_once_done", **counts})
    log.info(
        "reconcile: scanned %d → download %d, extract %d "
        "(deferred %d, canary %d), terminal %d, errors %d",
        counts["scanned"], counts["download"], counts["extract"],
        counts["extract_deferred"], counts["extract_canary"],
        counts["terminal"], counts["errors"],
    )
    await asyncio.to_thread(concurrency.trim_process_memory)
    return counts


async def reconcile_loop(
    library: Library,
    download_queue,
    extract_queue,
    *,
    interval_seconds: float = 600.0,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """Run :func:`reconcile_once` on a timer until cancelled (D8).

    Started by the daemon after the queues are up. Sleeps ``interval_seconds``
    between sweeps; a single sweep raising is caught + logged so the loop
    survives. ``stop_event`` (optional) lets a graceful shutdown break the
    sleep early.
    """
    while True:
        try:
            await reconcile_once(library, download_queue, extract_queue)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("reconcile sweep crashed; continuing")
        if stop_event is not None:
            try:
                await asyncio.wait_for(stop_event.wait(),
                                       timeout=interval_seconds)
                return  # stop_event set → graceful exit
            except asyncio.TimeoutError:
                continue
        else:
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                raise
