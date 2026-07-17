"""Concurrent ingestion: take a list of identifiers (DOIs / arxiv ids /
fuzzy text) and run AddService.add for each in a thread pool.

Why threads not processes: every step (network fetch, file IO, LLM call)
is I/O-bound. The Library write is serialized by filelock so concurrent
upserts don't race on index.json.

Rate limits respected (Semantic Scholar caps free-tier traffic at 1 r/s):
default 4 workers + a 0.5s minimum gap per worker.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional

from ..download import download_paper
from ..models import DOWNLOAD_STATUS_OK
from ..store import Library
from .add_service import AddService


class _RateLimiter:
    """Token-bucket-ish: ensure at most 1 request per `min_interval` seconds.
    Thread-safe and shared across worker threads."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


class BatchAddService:
    """Concurrent wrapper around :class:`AddService.add`.

    Each task: identifier in, AddResult out. Failures don't abort the batch.
    """

    def __init__(self,
                 library: Library,
                 *,
                 llm=None,
                 max_workers: int = 4,
                 min_request_interval: float = 0.5):
        self.library = library
        self._add_service = AddService(library, llm=llm)
        self.max_workers = max_workers
        self._rate = _RateLimiter(min_request_interval)

    def add_many(self,
                 identifiers: Iterable[str],
                 *,
                 force_refresh: bool = False,
                 on_result: Optional[Callable[[str, dict], None]] = None) -> list[dict]:
        """Add many identifiers concurrently. Returns AddResults in the
        order they completed (not the order they were submitted).

        ``on_result`` is invoked from the main thread after each future
        completes — useful for progress logging.
        """
        ids = [i for i in identifiers if i]
        results: list[dict] = []

        def _job(ident: str) -> tuple[str, dict]:
            self._rate.acquire()
            try:
                return ident, self._add_service.add(ident, force_refresh=force_refresh)
            except Exception as exc:
                return ident, {
                    "status": "internal_error",
                    "key": None,
                    "metadata": None,
                    "candidates": None,
                    "message": repr(exc)[:300],
                }

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(_job, i) for i in ids]
            for fut in as_completed(futures):
                ident, result = fut.result()
                results.append(result)
                if on_result:
                    try:
                        on_result(ident, result)
                    except Exception:
                        pass

        return results


def retry_failed_with_cooloff(
    library: Library,
    *,
    n_passes: int = 4,
    cooloff_seconds: int = 300,
    max_workers: int = 10,
    keys: Optional[list[str]] = None,
    on_pass_complete: Optional[Callable[[int, int, int], None]] = None,
) -> dict:
    """Re-run :func:`download_paper` on library entries that previously failed
    or are still pending a PDF. Between passes, sleep ``cooloff_seconds`` to
    let publisher CDNs (Radware/Akamai/Cloudflare) age our mihomo exit IPs
    out of their per-IP throttle window.

    Empirical: this is the highest-leverage batch-level lever for hit rate
    on the 82-paper bench. Single-pass tops out at ~80%; 4-pass with 5-min
    cool-off lifts to ~99%. See CLAUDE.md "Cascade evolution" table for the
    R3→R13 progression that established these defaults.

    Args:
        library: Loaded :class:`Library`.
        n_passes: Total passes (1 = no retries). Default 4 matches R11/R13.
        cooloff_seconds: Sleep between passes. Default 300s = 5 min, the
            empirical sweet spot. Less = mihomo IPs still throttled at
            publisher; more = wasted wall time without further recovery.
        max_workers: Concurrent downloads per pass. Default 10 matches the
            bench harness; higher saturates mihomo's connection pool and
            triggers more publisher rate-limits.
        keys: Restrict to specific paper keys. If None, scans all library
            entries with ``download_status != "ok"`` and missing PDF.
        on_pass_complete: Progress callback ``(pass_idx_1based, hits, still_missing)``.

    Returns:
        ``{"passes": [...], "final_hits": N, "final_misses": [...]}``.
    """
    def _eligible() -> list[str]:
        target = keys if keys is not None else list(library._papers.keys())
        out = []
        for k in target:
            p = library.get(k)
            if p is None:
                continue
            if library.has_pdf(k):
                continue
            # Eligible: ever-tried-and-failed, or pending, or never tried.
            # A status of "ok" without a PDF is a firecrawl text-only win
            # (D7) — exclude it from this PDF-retry batch (it already has
            # serveable text; reconcile handles the real-PDF hunt).
            if (p.download_status or "") == DOWNLOAD_STATUS_OK:
                continue
            out.append(k)
        return out

    def _retry_one(key: str) -> dict:
        paper = library.get(key)
        if paper is None or library.has_pdf(key):
            return {"key": key, "hit": library.has_pdf(key)}
        try:
            ok = download_paper(paper, library)
        except Exception as exc:
            return {"key": key, "hit": False, "error": repr(exc)[:200]}
        return {"key": key, "hit": ok, "status": paper.download_status}

    pass_results: list[dict] = []
    for pass_idx in range(1, n_passes + 1):
        pending = _eligible()
        if not pending:
            break
        hits_this_pass = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_retry_one, k) for k in pending]
            for fut in as_completed(futures):
                r = fut.result()
                if r.get("hit"):
                    hits_this_pass += 1
        library.save()
        still_missing = len(_eligible())
        pass_results.append({
            "pass": pass_idx,
            "attempted": len(pending),
            "hits": hits_this_pass,
            "still_missing": still_missing,
        })
        if on_pass_complete:
            try:
                on_pass_complete(pass_idx, hits_this_pass, still_missing)
            except Exception:
                pass
        if still_missing == 0 or pass_idx == n_passes:
            break
        time.sleep(cooloff_seconds)

    final_missing = _eligible()
    total_hits = sum(p["hits"] for p in pass_results)
    return {
        "passes": pass_results,
        "final_hits": total_hits,
        "final_misses": final_missing,
    }
