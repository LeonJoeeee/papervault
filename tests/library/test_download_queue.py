"""Unit tests for ``papervault.library.services.download_queue.DownloadQueue``.

Uses stub ``download_paper`` to avoid real network calls. Verifies:
- ``start()`` recovery enqueues exactly papers with ``pending`` + no PDF
- ``add()`` triggers a worker pass through ``_process_one``
- Per-task ``has_pdf`` self-check guard short-circuits cleanly
- a firecrawl text-only win (D7: status=ok + md on disk, no PDF) chains to
  on_success (extract queue self-detects md presence and short-circuits) —
  pre-route-B it had a dedicated on_text_only path into the insight queue,
  gone in Phase 28
- ``stop()`` cancels workers cleanly
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from papervault.library.models import (
    DOWNLOAD_STATUS_FAILED,
    DOWNLOAD_STATUS_OK,
    DOWNLOAD_STATUS_PENDING,
)
from papervault.library.services import download_queue as dq_mod
from papervault.library.services.download_queue import DownloadQueue
from papervault.library.store import Library


class _Recorder:
    """Test double for downstream queue.add — captures calls."""

    def __init__(self):
        self.added: list[str] = []

    def __call__(self, key: str) -> None:
        self.added.append(key)


def _write_pdf(lib: Library, key: str) -> None:
    """Create a fake PDF on disk so ``library.has_pdf(key)`` returns True."""
    pdf_path = lib.pdf_path(key)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\nfake")


def _write_md(lib: Library, key: str) -> None:
    md_path = lib.md_path(key)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("# fake md\n", encoding="utf-8")


@pytest.fixture
def lib_three_papers(tmp_path: Path) -> tuple[Library, str, str, str]:
    """Library with three papers:
       - paper_a: status=pending, no PDF → eligible for recovery
       - paper_b: status=pending, has PDF → skipped (already done)
       - paper_c: status=failed → skipped (terminal)
    Returns (library, key_a, key_b, key_c).
    """
    lib = Library(tmp_path)
    pa, _ = lib.upsert({
        "title": "Paper A pending no PDF for queue recovery test",
        "authors": ["AuthorOne"],
        "year": 2020,
    })
    pb, _ = lib.upsert({
        "title": "Paper B pending has PDF skip recovery test case",
        "authors": ["AuthorTwo"],
        "year": 2021,
    })
    pc, _ = lib.upsert({
        "title": "Paper C terminal failed skip recovery test case",
        "authors": ["AuthorThree"],
        "year": 2022,
    })
    pa.download_status = DOWNLOAD_STATUS_PENDING
    pb.download_status = DOWNLOAD_STATUS_PENDING
    _write_pdf(lib, pb.key)
    pc.download_status = DOWNLOAD_STATUS_FAILED
    lib.save()
    return lib, pa.key, pb.key, pc.key


def test_start_recovery_enqueues_pending_no_pdf(
    lib_three_papers, monkeypatch
):
    """Recovery should enqueue paper_a only (b has PDF, c is failed)."""
    lib, key_a, key_b, key_c = lib_three_papers
    monkeypatch.setattr(
        dq_mod, "download_paper",
        lambda paper, library: True,
    )

    async def run():
        q = DownloadQueue(lib, num_workers=0)
        await q.start()
        try:
            return q.qsize()
        finally:
            await q.stop()

    n = asyncio.run(run())
    assert n == 1


def test_add_processes_one_then_chains_extract(
    lib_three_papers, monkeypatch
):
    """A successful download (no md on disk) should call on_success
    (extract chain) with the key."""
    lib, key_a, _, _ = lib_three_papers
    extract_chain = _Recorder()
    insight_chain = _Recorder()

    def fake_download(paper, library):
        # Simulate cascade success: write PDF + D7 status/source split
        _write_pdf(library, paper.key)
        paper.download_status = DOWNLOAD_STATUS_OK
        paper.download_source = "fake_source"
        return True

    monkeypatch.setattr(dq_mod, "download_paper", fake_download)

    async def run():
        q = DownloadQueue(
            lib, num_workers=1,
            on_success=extract_chain, on_text_only=insight_chain,
        )
        # We deliberately don't .start() (recovery would also enqueue);
        # drive _process_one directly to test the post-download routing.
        await q._process_one(key_a)

    asyncio.run(run())
    assert extract_chain.added == [key_a]
    assert insight_chain.added == []


def test_process_one_skips_when_has_pdf_forwards_extract(
    lib_three_papers, monkeypatch
):
    """A paper that already has a PDF but no md should bypass the
    download_paper call and forward straight to extract."""
    lib, _, key_b, _ = lib_three_papers
    called = []
    monkeypatch.setattr(
        dq_mod, "download_paper",
        lambda paper, library: called.append(paper.key) or True,
    )
    extract_chain = _Recorder()
    insight_chain = _Recorder()

    async def run():
        q = DownloadQueue(
            lib, num_workers=0,
            on_success=extract_chain, on_text_only=insight_chain,
        )
        await q._process_one(key_b)

    asyncio.run(run())
    assert called == []                # download_paper NOT called
    assert extract_chain.added == [key_b]  # forwarded to extract
    assert insight_chain.added == []


def test_text_only_firecrawl_chains_to_on_success(
    lib_three_papers, monkeypatch
):
    """``download_paper`` returning False + status=ok with md on disk
    (D7: the firecrawl text-only win) means md is on disk; route to
    ``on_success`` (the extract queue, which self-detects md presence
    and short-circuits).

    ✦ D7 reshape: the dead ``text-only:firecrawl`` status is gone — a
    firecrawl text-only win is now ``download_status="ok"`` +
    ``download_source="firecrawl"`` with md on disk and no PDF. The
    download queue detects it via (status==ok ∧ has md) rather than a
    dedicated status string.

    ✦ Phase 28 (2026-05-24, route B) reframe: pre-route-B this case
    had a dedicated ``on_text_only`` callback that fed the insight
    queue directly (bypassing extract). The insight queue is gone;
    a firecrawl win now uses the same ``on_success`` chain as a
    normal download. The ``on_text_only=`` kwarg is accepted for
    back-compat but ignored.
    """
    lib, key_a, _, _ = lib_three_papers
    extract_chain = _Recorder()
    legacy_chain = _Recorder()  # passed via on_text_only; should be ignored

    def fake_download(paper, library):
        # Simulate firecrawl text-only fallback: md written, no PDF, D7 status
        _write_md(library, paper.key)
        paper.download_status = DOWNLOAD_STATUS_OK
        paper.download_source = "firecrawl"
        return False

    monkeypatch.setattr(dq_mod, "download_paper", fake_download)

    async def run():
        q = DownloadQueue(
            lib, num_workers=0,
            on_success=extract_chain, on_text_only=legacy_chain,
        )
        await q._process_one(key_a)

    asyncio.run(run())
    # text-only path now routes through on_success (extract queue
    # self-detects md presence and short-circuits)
    assert extract_chain.added == [key_a]
    # on_text_only is now a no-op; the legacy chain must not fire
    assert legacy_chain.added == []


def test_stop_cancels_workers_clean(lib_three_papers, monkeypatch):
    """stop() must cancel worker tasks without raising."""
    lib, _, _, _ = lib_three_papers
    monkeypatch.setattr(
        dq_mod, "download_paper",
        lambda paper, library: True,
    )

    async def run():
        q = DownloadQueue(lib, num_workers=2)
        await q.start()
        await asyncio.sleep(0.05)
        await q.stop()
        return len(q._workers)

    n = asyncio.run(run())
    assert n == 0


def test_add_dedups_key_already_in_queue(lib_three_papers):
    """F7: a +600s reconcile sweep that re-adds a key STILL WAITING in the queue
    is suppressed — only ONE item is enqueued, so download_paper + the firecrawl
    re-gate can't run twice within one queue-residency window. The mark clears
    on dequeue, so a re-add after processing starts is allowed (stamp-bounded)."""
    lib, key_a, _, _ = lib_three_papers

    async def run():
        q = DownloadQueue(lib, num_workers=0)  # no workers → key stays queued
        q.add(key_a)
        q.add(key_a)            # duplicate while it waits → deduped
        q.add(key_a)
        size_after_dups = q.qsize()
        # Drain it (simulate a worker dequeue clearing the dedup mark).
        _prio, _seq, k = await q._queue.get()
        q._pending.pop(k, None)   # what the worker loop does on dequeue
        # Now a fresh add is allowed again (the key no longer waits in queue).
        q.add(key_a)
        return size_after_dups, q.qsize()

    size_after_dups, size_after_redrain = asyncio.run(run())
    assert size_after_dups == 1, "duplicate enqueues of a waiting key not deduped"
    assert size_after_redrain == 1, "re-add after dequeue must be allowed again"


def test_urgent_readd_of_queued_key_reaches_worker_at_urgent(lib_three_papers):
    """R1 (priority-inversion regression): a key queued at NORMAL by a reconcile
    sweep, then re-added at URGENT by a foreground get_paper, must reach the
    worker at URGENT — the F7 dedup must NOT silently drop the urgent re-add.

    The priority-aware dedup re-enqueues the more-urgent copy (it jumps ahead in
    the PriorityQueue); the stale NORMAL copy is later dropped by the worker's
    POP-TIME ``_pending`` dedup (T1) — the URGENT tuple surfaces first, claims
    the ``_pending`` mark, and the stale NORMAL duplicate then finds the mark
    gone and is discarded without re-running download_paper. What matters here is
    that the FIRST tuple a worker dequeues carries the URGENT priority, so the
    urgent request actually jumps the backlog."""
    from papervault.library.services import concurrency

    lib, key_a, _, _ = lib_three_papers

    async def run():
        q = DownloadQueue(lib, num_workers=0)  # no workers → keys stay queued
        q.add(key_a, priority=concurrency.PRIORITY_NORMAL)
        q.add(key_a, priority=concurrency.PRIORITY_URGENT)  # strictly more urgent
        # The first item a worker would dequeue (lowest priority number wins).
        first_prio, _seq, first_key = await q._queue.get()
        return first_prio, first_key

    first_prio, first_key = asyncio.run(run())
    assert first_key == key_a
    assert first_prio == concurrency.PRIORITY_URGENT, (
        "urgent re-add of a still-queued NORMAL key was dropped/inverted — "
        "the urgent request never jumps the queue"
    )


def test_same_or_lower_prio_readd_of_queued_key_is_deduped(lib_three_papers):
    """R1 companion: a same-or-lower-priority re-add of a still-waiting key stays
    deduped (the F7 anti-double-sweep guarantee is preserved). Only a
    strictly-more-urgent re-add is allowed to re-enqueue."""
    from papervault.library.services import concurrency

    lib, key_a, _, _ = lib_three_papers

    async def run():
        q = DownloadQueue(lib, num_workers=0)
        q.add(key_a, priority=concurrency.PRIORITY_NORMAL)
        q.add(key_a, priority=concurrency.PRIORITY_NORMAL)  # same prio → deduped
        q.add(key_a, priority=concurrency.PRIORITY_LOW)     # lower prio → deduped
        return q.qsize()

    assert asyncio.run(run()) == 1


def _write_firecrawl_md(lib: Library, key: str) -> None:
    """Create a firecrawl-sourced md on disk (frontmatter source: firecrawl),
    so md_source(key)=='firecrawl' and has_extract(key,'md') is True."""
    md_path = lib.md_path(key)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        "---\nsource: firecrawl\n---\n# real body text\n", encoding="utf-8"
    )


def test_stale_lower_prio_tuple_of_firecrawl_md_is_dropped_not_rehunted(
    lib_three_papers, monkeypatch
):
    """T1 (pop-time dedup, no-op true by construction): a firecrawl-md key
    (``has_md`` ∧ ``¬has_pdf``, ``firecrawl_pdf_hunt_exhausted=True`` — exactly
    the F7-protected population) re-added at URGENT while already queued at
    NORMAL leaves a STALE NORMAL tuple behind. When a worker later pops that
    stale tuple it must be DROPPED — NOT re-run through ``download_paper`` /
    the 18-tier PDF hunt a second time.

    Pre-fix this was false for ¬has_pdf keys: the worker's ``has_pdf``
    short-circuit never fires for a firecrawl-md key, so the stale tuple
    re-entered ``download_paper`` and re-ran all 18 tiers. The pop-time dedup
    makes the "no-op" claim true by construction: the first tuple to surface
    claims the ``_pending`` mark; the stale duplicate finds it gone and is
    discarded without calling ``download_paper``.
    """
    from papervault.library.services import concurrency

    lib, key_a, _, _ = lib_three_papers
    # Make key_a the F7-protected firecrawl-md population: md on disk
    # (source=firecrawl), no PDF, exhausted stamp set, status ok.
    pa = lib.get(key_a)
    pa.download_status = DOWNLOAD_STATUS_OK
    pa.download_source = "firecrawl"
    pa.firecrawl_pdf_hunt_exhausted = True
    lib.save()
    _write_firecrawl_md(lib, key_a)
    assert lib.has_extract(key_a, "md")
    assert not lib.has_pdf(key_a)
    assert lib.md_source(key_a) == "firecrawl"

    download_calls: list[str] = []

    def spy_download(paper, library):
        download_calls.append(paper.key)
        return False  # firecrawl-md key: no real PDF found this pass either

    monkeypatch.setattr(dq_mod, "download_paper", spy_download)

    async def run():
        # num_workers=0 → no auto workers; we drive one worker_loop manually so
        # both the URGENT and the stale NORMAL tuple are dequeued and routed
        # through the real pop-time dedup in _worker_loop.
        q = DownloadQueue(lib, num_workers=0)
        # Reconcile queues it NORMAL; foreground get_paper re-adds URGENT
        # (¬has_pdf → strictly-more-urgent re-add jumps ahead, R1). This leaves
        # the stale NORMAL tuple behind in the PriorityQueue.
        q.add(key_a, priority=concurrency.PRIORITY_NORMAL)
        q.add(key_a, priority=concurrency.PRIORITY_URGENT)
        assert q.qsize() == 2, "expected URGENT + stale NORMAL tuples queued"

        worker = asyncio.create_task(q._worker_loop(0))
        # Both tuples must drain (the URGENT pass processes; the stale NORMAL
        # tuple is the no-op we are pinning).
        await asyncio.wait_for(q._queue.join(), timeout=5.0)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(run())

    assert download_calls == [key_a], (
        "the stale lower-prio firecrawl-md tuple re-ran download_paper / the "
        f"18-tier hunt a second time (calls={download_calls!r}); the pop-time "
        "dedup must drop it as a true no-op"
    )


def test_concurrent_same_key_processed_once(lib_three_papers, monkeypatch):
    """with_dedup wiring (verified concurrency finding): two concurrent
    _process_one calls for the SAME key run download_paper only ONCE — the second
    subscribes to the in-flight run instead of mutating the same Paper again."""
    import time
    lib, key_a, _b, _c = lib_three_papers
    calls = {"n": 0}

    def slow_download(paper, library):
        time.sleep(0.05)            # hold the to_thread so both overlap
        calls["n"] += 1
        return False                # no PDF → no chain

    monkeypatch.setattr(dq_mod, "download_paper", slow_download)

    async def run():
        q = DownloadQueue(lib, num_workers=0)
        await asyncio.gather(q._process_one(key_a), q._process_one(key_a))

    asyncio.run(run())
    assert calls["n"] == 1, f"download_paper ran {calls['n']}x; with_dedup should collapse to 1"
