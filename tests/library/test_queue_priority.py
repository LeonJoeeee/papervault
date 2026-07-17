"""D11: priority-queue tests for the 2 remaining stage queues.

Verifies that urgent items pop before normal, and that FIFO holds
within a single priority level. Each queue's internal
asyncio.PriorityQueue is a (priority, seq, key) tuple — we just check
the order keys come out via the worker loop.

✦ Phase 28 (2026-05-24, route B): the InsightQueue was removed; the
corresponding test was deleted along with the import. Only download
and extract queues remain.
"""
from __future__ import annotations

import asyncio

import pytest

from papervault.library.services import concurrency
from papervault.library.services.download_queue import DownloadQueue
from papervault.library.services.extract_queue import ExtractQueue
from papervault.library.store import Library


def _make_lib(tmp_path):
    return Library(tmp_path / "lib")


@pytest.mark.asyncio
async def test_download_queue_urgent_pops_before_normal(tmp_path):
    lib = _make_lib(tmp_path)
    q = DownloadQueue(lib, num_workers=0)
    # Don't start workers — just inspect the PriorityQueue directly via add.
    q.add("n1", priority=concurrency.PRIORITY_NORMAL)
    q.add("n2", priority=concurrency.PRIORITY_NORMAL)
    q.add("u1", priority=concurrency.PRIORITY_URGENT)
    # PriorityQueue.get returns the smallest first
    _, _, first = await q._queue.get()
    _, _, second = await q._queue.get()
    _, _, third = await q._queue.get()
    assert first == "u1", "urgent should pop first"
    assert second == "n1", "FIFO within priority — n1 before n2"
    assert third == "n2"


@pytest.mark.asyncio
async def test_extract_queue_urgent_pops_before_normal(tmp_path):
    lib = _make_lib(tmp_path)
    q = ExtractQueue(lib, num_workers=0)
    q.add("normal_a", priority=concurrency.PRIORITY_NORMAL)
    q.add("urgent_x", priority=concurrency.PRIORITY_URGENT)
    q.add("normal_b", priority=concurrency.PRIORITY_NORMAL)
    q.add("urgent_y", priority=concurrency.PRIORITY_URGENT)
    order = []
    for _ in range(4):
        _, _, key = await q._queue.get()
        order.append(key)
    # urgent_x and urgent_y first (in insertion order), then normal_a, normal_b
    assert order == ["urgent_x", "urgent_y", "normal_a", "normal_b"]


@pytest.mark.asyncio
async def test_add_defaults_to_normal_priority(tmp_path):
    """Calling add(key) without explicit priority defaults to NORMAL.
    Existing callers that don't know about priorities keep working."""
    lib = _make_lib(tmp_path)
    q = ExtractQueue(lib, num_workers=0)
    q.add("k1")  # no priority arg
    q.add("k2", priority=concurrency.PRIORITY_URGENT)
    _, _, first = await q._queue.get()
    _, _, second = await q._queue.get()
    # k2 was urgent → pops first even though enqueued second
    assert first == "k2"
    assert second == "k1"


# ---------- D12: extract_md on_progress callback ----------


