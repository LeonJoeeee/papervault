"""Tests for services/concurrency.py — resource semaphores + per-key dedup registry."""

from __future__ import annotations

import asyncio

import pytest

from papervault.library.services.concurrency import (
    in_flight_keys,
    is_in_flight,
    lib_write_lock,
    llm_sem,
    network_sem,
    with_dedup,
)


# ---------- with_dedup ----------


@pytest.mark.asyncio
async def test_with_dedup_runs_work_once_for_unique_key():
    counter = [0]

    async def work():
        counter[0] += 1
        return "done"

    result = await with_dedup("k1", work)
    assert result == "done"
    assert counter[0] == 1


@pytest.mark.asyncio
async def test_with_dedup_concurrent_callers_share_result():
    counter = [0]

    async def work():
        counter[0] += 1
        await asyncio.sleep(0.05)
        return "shared"

    results = await asyncio.gather(
        with_dedup("kshared", work),
        with_dedup("kshared", work),
        with_dedup("kshared", work),
    )
    assert results == ["shared", "shared", "shared"]
    assert counter[0] == 1  # work() invoked exactly once


@pytest.mark.asyncio
async def test_with_dedup_different_keys_dont_block_each_other():
    started = []

    async def make_work(tag):
        async def inner():
            started.append(tag)
            await asyncio.sleep(0.05)
            return tag
        return await with_dedup(tag, inner)

    t0 = asyncio.get_running_loop().time()
    results = await asyncio.gather(make_work("a"), make_work("b"), make_work("c"))
    elapsed = asyncio.get_running_loop().time() - t0

    assert sorted(results) == ["a", "b", "c"]
    # parallel: total time should be ~0.05s, NOT 0.15s
    assert elapsed < 0.12, f"calls serialized unexpectedly (took {elapsed:.3f}s)"


@pytest.mark.asyncio
async def test_with_dedup_exception_propagates_to_all_waiters():
    async def work():
        await asyncio.sleep(0.05)
        raise RuntimeError("boom")

    async def caller():
        try:
            await with_dedup("kerr", work)
        except RuntimeError as e:
            return str(e)
        return "no_exception"

    results = await asyncio.gather(caller(), caller(), caller())
    assert results == ["boom", "boom", "boom"]


@pytest.mark.asyncio
async def test_with_dedup_cleans_up_after_completion():
    async def work():
        return None

    assert "kclean" not in in_flight_keys()
    await with_dedup("kclean", work)
    assert "kclean" not in in_flight_keys()
    assert not is_in_flight("kclean")


@pytest.mark.asyncio
async def test_with_dedup_cleans_up_after_exception():
    async def work():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await with_dedup("kfail", work)
    assert "kfail" not in in_flight_keys()


@pytest.mark.asyncio
async def test_with_dedup_subscriber_factory_not_invoked():
    """When a second caller arrives during in-flight work, its factory
    must NOT be invoked — it just awaits the existing Future."""

    async def first_work():
        await asyncio.sleep(0.1)
        return "first-result"

    factory_invocations = [0]

    async def second_work():
        factory_invocations[0] += 1
        return "second-result"

    task = asyncio.create_task(with_dedup("kfactory", first_work))
    await asyncio.sleep(0.02)  # let first caller register the Future

    result = await with_dedup("kfactory", second_work)
    assert result == "first-result"
    assert factory_invocations[0] == 0

    await task  # ensure first task drains


# ---------- module-level resources sanity ----------


def test_resource_primitives_have_expected_types():
    assert isinstance(network_sem, asyncio.Semaphore)
    assert isinstance(llm_sem, asyncio.Semaphore)
    assert isinstance(lib_write_lock, asyncio.Lock)


# NOTE: ``marker_lock`` and ``mimo_vision_sem`` were removed, and the whole
# dots/chandra/marker OCR cascade + its subprocess EnginePool + per-paper GPU
# pin were deleted in the 2026-06-06 MinerU migration. Extract concurrency is
# now a single explicit ``extract_slots`` asyncio.Semaphore
# (``concurrency.acquire_extract_slot`` / ``release_extract_slot``) decoupled
# from GPU count — the persistent MinerU2.5-Pro vLLM server batches the per-page
# sub-requests server-side via ``--max-num-seqs``. See
# ``test_extract_slot_admission_*`` below for that primitive's coverage.


@pytest.mark.asyncio
async def test_network_sem_caps_concurrency():
    """network_sem limits concurrent acquirers to 4."""
    in_flight = [0]
    peak = [0]

    async def hold():
        async with network_sem:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
            await asyncio.sleep(0.02)
            in_flight[0] -= 1

    await asyncio.gather(*(hold() for _ in range(10)))
    assert peak[0] <= 4


# ---------- extract-slot admission (SDD §4.1, MinerU migration) ----------


@pytest.mark.asyncio
async def test_extract_slot_admission_caps_in_flight(monkeypatch):
    """``acquire_extract_slot`` permits at most ``_EXTRACT_CONCURRENCY``
    extractions in flight; the surplus block until a slot is released."""
    from papervault.library.services import concurrency as conc

    # Shrink the cap to 2 for a deterministic test (rebind the live semaphore).
    monkeypatch.setattr(conc, "extract_slots", asyncio.Semaphore(2))

    in_flight = [0]
    peak = [0]

    async def hold():
        await conc.acquire_extract_slot()
        try:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
            await asyncio.sleep(0.02)
            in_flight[0] -= 1
        finally:
            conc.release_extract_slot()

    await asyncio.gather(*(hold() for _ in range(8)))
    assert peak[0] <= 2
    # All permits returned → the semaphore is back at full capacity.
    assert conc.extract_slots._value == 2


@pytest.mark.asyncio
async def test_extract_slot_returns_round_robin_endpoint(monkeypatch):
    """With a 2-endpoint ``MINERU_URL`` the slot hands out endpoints in
    round-robin order so multi-endpoint work spreads evenly (SDD §4.1)."""
    from papervault.library.services import concurrency as conc

    monkeypatch.setenv("MINERU_URL", "http://a:30000,http://b:30001")
    monkeypatch.setattr(conc, "extract_slots", asyncio.Semaphore(8))

    urls = []
    for _ in range(4):
        ep = await conc.acquire_extract_slot()
        urls.append(ep.url)
        conc.release_extract_slot()

    # Round-robin over the two endpoints (cursor is process-wide, so assert the
    # two URLs simply alternate rather than pinning an absolute start offset).
    assert set(urls) == {"http://a:30000", "http://b:30001"}
    assert urls[0] != urls[1]
    assert urls[0] == urls[2] and urls[1] == urls[3]


def test_extract_dispatch_toggle(tmp_path, monkeypatch):
    """``extract_dispatch_enabled`` reads the redefined pause-dispatch
    ``ocr-pool.conf`` toggle: empty file → paused, non-empty / absent → run."""
    from papervault.library.services import concurrency as conc

    monkeypatch.setenv("PAPER_LIBRARY_PATH", str(tmp_path))
    conf = tmp_path / "ocr-pool.conf"

    # Absent file → run (default).
    assert conc.extract_dispatch_enabled() is True
    # Non-empty → run.
    conf.write_text("run")
    assert conc.extract_dispatch_enabled() is True
    # Empty → paused.
    conf.write_text("")
    assert conc.extract_dispatch_enabled() is False
