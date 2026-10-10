"""Recording contract: exercise JSONL, real queue/thread boundaries, no live I/O."""

import asyncio
import json
import threading
from datetime import datetime

import pytest

from papervault.library import download
from papervault.library.services import concurrency, download_queue
from papervault.library.store import Library


def events(lib):
    return ([json.loads(line) for line in lib.manifest_path.read_text().splitlines()]
            if lib.manifest_path.exists() else [])


def setup_paper(tmp_path):
    lib = Library(tmp_path)
    paper, _ = lib.upsert({"title": "Telemetry cohort paper", "doi": "10.1234/test"})
    return lib, paper


def completed_spans(lib, phase):
    return [e for e in events(lib) if e["event"] == "download_telemetry"
            and e["phase"] == phase and e["mark"] == "end"]


@pytest.mark.parametrize("outcome", ["miss", "error", "mismatch", "win"])
def test_tier_outcome_has_timing_and_balanced_slot(tmp_path, monkeypatch, outcome):
    lib, paper = setup_paper(tmp_path)

    def strategy(_):
        if outcome == "error":
            raise RuntimeError("tier failure")
        return b"%PDF-test" if outcome in {"mismatch", "win"} else None

    monkeypatch.setattr(download, "_STRATEGIES", [("sample", strategy)])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: False)
    monkeypatch.setattr(download, "_verify_pdf_matches_metadata",
                        lambda *_: (outcome == "win", "fixture"))
    assert download.download_paper(paper, lib) is (outcome == "win")
    name = {"miss": "download_miss", "error": "download_error",
            "mismatch": "download_pdf_mismatch", "win": "downloaded"}[outcome]
    event = next(e for e in events(lib) if e["event"] == name)
    timing = event["timing"]
    assert len(timing["run_id"]) == 32
    assert datetime.fromisoformat(timing["slot_started_at"]) <= datetime.fromisoformat(
        timing["slot_ended_at"])
    assert timing["slot_duration_s"] >= 0
    slot, = completed_spans(lib, "tier")
    assert slot["source"] == "sample"
    assert slot["duration_s"] >= timing["slot_duration_s"]
    terminal, = completed_spans(lib, "cascade")
    assert terminal["pdf_returned"] is (outcome == "win")
    assert terminal["paper_status"] == paper.download_status
    assert terminal["run_id"] == timing["run_id"]


def test_skip_and_firecrawl_early_return_are_timed(tmp_path, monkeypatch):
    lib, paper = setup_paper(tmp_path)
    monkeypatch.setattr(download, "_STRATEGIES", [("arxiv", lambda _: None)])
    monkeypatch.setattr(download, "_try_firecrawl_text_fallback", lambda *_: True)
    assert download.download_paper(paper, lib) is False
    skip = next(e for e in events(lib) if e["event"] == "download_skip")
    assert skip["reason"] == "missing_arxiv_id"
    assert skip["timing"]["slot_duration_s"] >= 0
    assert len(completed_spans(lib, "firecrawl")) == 1
    assert len(completed_spans(lib, "cascade")) == 1


def test_queue_admission_wait_and_actual_executor(tmp_path, monkeypatch):
    lib, paper = setup_paper(tmp_path)
    monkeypatch.setattr(download_queue, "download_paper", lambda *_: False)

    async def run():
        sem = asyncio.Semaphore(1)
        monkeypatch.setattr(concurrency, "network_sem", sem)
        await sem.acquire()
        queue = download_queue.DownloadQueue(lib, num_workers=1)
        await queue.start()
        for _ in range(100):
            if any(e.get("phase") == "network_wait" for e in events(lib)):
                break
            await asyncio.sleep(0.001)
        assert not completed_spans(lib, "executor")
        await asyncio.sleep(0.01)
        sem.release()
        await asyncio.wait_for(queue._queue.join(), 2)
        await queue.stop()

    asyncio.run(run())
    wait, = completed_spans(lib, "network_wait")
    assert wait["duration_s"] >= 0.01
    hold, = completed_spans(lib, "network_hold")
    thread, = completed_spans(lib, "executor")
    assert thread["thread_id"] != threading.get_ident()
    assert wait["ended_at"] <= thread["started_at"] <= thread["ended_at"] <= hold["ended_at"]
    enqueue = next(e for e in events(lib) if e.get("phase") == "enqueue")
    dequeue = next(e for e in events(lib) if e.get("phase") == "dequeue")
    assert enqueue["run_id"] == dequeue["run_id"] == thread["run_id"]
    assert dequeue["queue_wait_s"] >= 0
    assert dequeue["enqueued_at"] == enqueue["at"]


def test_group_winner_and_losing_thread_drain_are_distinct(tmp_path, monkeypatch):
    from papervault.library import download_telemetry as telemetry

    lib, paper = setup_paper(tmp_path)
    loser_started = threading.Event()
    selected = threading.Event()
    original_log = lib.log

    def record(event):
        original_log(event)
        if event.get("phase") == "winner":
            selected.set()

    monkeypatch.setattr(lib, "log", record)

    def winner(_):
        assert loser_started.wait(1)
        return b"%PDF-winner"

    def loser(_):
        loser_started.set()
        assert selected.wait(2)
        return None

    with telemetry.observation(lib, paper.key):
        result = download._try_concurrent_first_hit(paper, [("win", winner), ("slow", loser)])
    assert result == b"%PDF-winner"
    members = completed_spans(lib, "member")
    assert {e["member"] for e in members} == {"win", "slow"}
    selected = next(e for e in events(lib) if e.get("phase") == "winner")
    drained, = completed_spans(lib, "group")
    slow = next(e for e in members if e["member"] == "slow")
    assert selected["at"] < slow["ended_at"] <= drained["ended_at"]
    assert len({e["run_id"] for e in members}) == 1


def test_browser_failure_closes_call_without_recording_arguments(tmp_path):
    from papervault.library import download_telemetry as telemetry

    lib, paper = setup_paper(tmp_path)

    class Browser:
        @staticmethod
        def fetch(url, **kwargs):
            assert url == "https://example.invalid/private"
            assert kwargs == {"cookies": [{"value": "secret"}]}
            raise ValueError("secret")

    with telemetry.observation(lib, paper.key):
        with pytest.raises(ValueError):
            telemetry.browser_fetch(Browser, "https://example.invalid/private",
                                    cookies=[{"value": "secret"}])
    span, = completed_spans(lib, "browser_call")
    assert span["status"] == "error"
    assert "secret" not in lib.manifest_path.read_text()
    assert "example.invalid" not in lib.manifest_path.read_text()


def test_recording_failures_and_long_labels_do_not_change_result(tmp_path, monkeypatch):
    from papervault.library import download_telemetry as telemetry

    lib, paper = setup_paper(tmp_path)
    with telemetry.observation(lib, paper.key):
        with telemetry.span("tier", source="x" * 10000):
            lib.log({"event": "download_miss", "key": paper.key, "source": "fixture"})
    span, = completed_spans(lib, "tier")
    assert len(span["source"]) <= 64
    assert len(json.dumps(span)) < 1500
    legacy = next(e for e in events(lib) if e["event"] == "download_miss")
    assert len(json.dumps(legacy["timing"])) < 600

    def broken_sink(_):
        raise OSError("disk unavailable")

    monkeypatch.setattr(lib, "log", broken_sink)
    with telemetry.observation(lib, paper.key):
        with telemetry.span("executor"):
            result = 42
    assert result == 42


def test_cancellation_while_waiting_never_acquires_or_releases(tmp_path):
    from papervault.library import download_telemetry as telemetry

    lib, paper = setup_paper(tmp_path)

    async def run():
        sem = asyncio.Semaphore(0)

        async def waiter():
            async with telemetry.network_slot(sem, lib, paper.key):
                pytest.fail("cancelled waiter entered hold")

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sem.locked()

    asyncio.run(run())
    wait, = completed_spans(lib, "network_wait")
    assert wait["status"] == "cancelled"
    assert not completed_spans(lib, "network_hold")


def test_cancelled_worker_release_precedes_actual_thread_end(tmp_path, monkeypatch):
    lib, paper = setup_paper(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def download_stub(*_):
        started.set()
        assert release.wait(2)
        return False

    monkeypatch.setattr(download_queue, "download_paper", download_stub)

    async def run():
        sem = asyncio.Semaphore(1)
        monkeypatch.setattr(concurrency, "network_sem", sem)
        queue = download_queue.DownloadQueue(lib, num_workers=1)
        await queue.start()
        try:
            for _ in range(1000):
                if started.is_set():
                    break
                await asyncio.sleep(0.001)
            assert started.is_set()
            await queue.stop()
            hold, = completed_spans(lib, "network_hold")
            assert hold["status"] == "cancelled"
            assert not completed_spans(lib, "executor")
            assert not sem.locked()
        finally:
            release.set()

    asyncio.run(run())  # asyncio joins its default executor before returning.
    hold, = completed_spans(lib, "network_hold")
    thread, = completed_spans(lib, "executor")
    assert hold["ended_at"] < thread["ended_at"]
    assert hold["run_id"] == thread["run_id"]


def test_priority_upgrade_pairs_the_urgent_enqueue_and_prunes_stale_stamp(tmp_path, monkeypatch):
    lib, paper = setup_paper(tmp_path)
    monkeypatch.setattr(download_queue, "download_paper", lambda *_: False)

    async def run():
        monkeypatch.setattr(concurrency, "network_sem", asyncio.Semaphore(4))
        queue = download_queue.DownloadQueue(lib, num_workers=1)
        queue.add(paper.key)
        queue.add(paper.key, priority=concurrency.PRIORITY_URGENT)
        await queue.start()
        await asyncio.wait_for(queue._queue.join(), 2)
        await queue.stop()
        assert not queue._queued_timing

    asyncio.run(run())
    dequeue, = [e for e in events(lib) if e.get("phase") == "dequeue"]
    enqueue = next(e for e in events(lib) if e.get("phase") == "enqueue"
                   and e["priority"] == concurrency.PRIORITY_URGENT)
    assert dequeue["sequence"] == enqueue["sequence"]
    assert dequeue["run_id"] == enqueue["run_id"]
    assert len(completed_spans(lib, "executor")) == 1
    drop, = [e for e in events(lib) if e.get("phase") == "queue_drop"]
    stale = next(e for e in events(lib) if e.get("phase") == "enqueue"
                 and e["priority"] == concurrency.PRIORITY_NORMAL)
    assert drop["sequence"] == stale["sequence"]
    assert drop["run_id"] == stale["run_id"]
    assert drop["reason"] == "stale_priority_tuple"
    assert drop["dequeued_at"] >= stale["at"]


def test_concurrent_papers_and_search_have_separate_contexts(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from papervault.library import download_telemetry as telemetry

    lib, paper = setup_paper(tmp_path)
    barrier = threading.Barrier(2)

    def record(key):
        with telemetry.observation(lib, key):
            with telemetry.span("tier", source=key):
                barrier.wait(timeout=2)
                lib.log({"event": "download_miss", "key": key, "source": key})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(record, key) for key in [paper.key, "second"]]
        for future in futures:
            future.result(timeout=2)
    tiers = completed_spans(lib, "tier")
    assert len({e["run_id"] for e in tiers}) == 2
    for event in [e for e in events(lib) if e["event"] == "download_miss"]:
        matching = next(e for e in tiers if e["key"] == event["key"])
        assert event["timing"]["run_id"] == matching["run_id"]
        assert event["timing"]["slot_id"] == matching["span_id"]
    lib.log({"event": "outside", "key": paper.key})
    assert "timing" not in events(lib)[-1]

    async def search():
        async with telemetry.network_slot(asyncio.Semaphore(1), lib, actor="search"):
            pass

    asyncio.run(search())
    hold, = completed_spans(lib, "network_hold")
    assert hold["actor"] == "search"
    assert hold["key"] == ""


def test_winner_point_names_its_group_and_source(tmp_path):
    from papervault.library import download_telemetry as telemetry

    lib, paper = setup_paper(tmp_path)
    with telemetry.observation(lib, paper.key):
        with telemetry.span("tier", source="oa_aggregators"):
            download._try_concurrent_first_hit(paper, [("one", lambda _: b"%PDF")])
    group, = completed_spans(lib, "group")
    winner = next(e for e in events(lib) if e.get("phase") == "winner")
    assert winner["span_id"] == group["span_id"]
    assert winner["source"] == "oa_aggregators"


@pytest.mark.parametrize("tier", ["annas_archive", "mdpi_scrapling", "researchgate"])
def test_real_browser_tier_call_is_observed_without_argument_changes(tmp_path, monkeypatch, tier):
    from papervault.library import download_telemetry as telemetry
    from papervault.library.download_sources import annas_archive, mdpi, researchgate

    lib, paper = setup_paper(tmp_path)
    paper.doi = "10.3390/example"
    monkeypatch.setenv("ANNAS_ARCHIVE_API_KEY", "test-member-secret")
    module, strategy = {
        "annas_archive": (annas_archive, annas_archive._try_annas_archive_api),
        "mdpi_scrapling": (mdpi, mdpi._try_mdpi_scrapling),
        "researchgate": (researchgate, researchgate._try_researchgate),
    }[tier]
    called = []

    class Browser:
        @staticmethod
        def fetch(url, **kwargs):
            called.append((url, kwargs))
            raise RuntimeError("browser fixture refusal")

    monkeypatch.setattr(module, "_load_stealthy_fetcher", lambda _: Browser)
    with telemetry.observation(lib, paper.key):
        with telemetry.span("tier", source=tier):
            assert strategy(paper) is None
    spans = completed_spans(lib, "browser_call")
    assert len(spans) == len(called) >= 1
    assert all(e["source"] == tier and e["status"] == "error" for e in spans)
    assert "test-member-secret" not in lib.manifest_path.read_text()
    if tier == "annas_archive":
        assert called[0][1]["cookies"][0]["value"] == "test-member-secret"
    elif tier == "mdpi_scrapling":
        assert called[0][1]["timeout"] == 60000
        assert callable(called[0][1]["page_action"])
    else:
        assert called[0][1]["wait"] == 2500


def test_scihub_mirror_threads_keep_trace_and_do_not_publish_urls(tmp_path, monkeypatch):
    from papervault.library import download_telemetry as telemetry
    from papervault.library.download_sources import scihub

    lib, paper = setup_paper(tmp_path)
    monkeypatch.setenv("PAPER_PIPELINE_USE_SCIHUB", "1")
    monkeypatch.setattr(scihub, "_discover_scihub_mirrors", lambda: ["https://private.invalid"])
    monkeypatch.setattr(scihub, "_scihub_one_mirror", lambda *_: b"%PDF-fixture")
    with telemetry.observation(lib, paper.key):
        with telemetry.span("tier", source="scihub"):
            assert scihub._try_scihub(paper) == b"%PDF-fixture"
    member, = completed_spans(lib, "member")
    group, = completed_spans(lib, "group")
    assert member["member"] == "mirror_0"
    assert member["thread_id"] != threading.get_ident()
    assert member["parent_span_id"] == group["span_id"]
    assert "private.invalid" not in lib.manifest_path.read_text()


def test_confirmed_withdrawal_with_existing_pdf_still_records_terminal(tmp_path):
    from papervault.library.models import ArxivWithdrawal

    lib, paper = setup_paper(tmp_path)
    paper.arxiv_id = "2603.20546"
    paper.arxiv_withdrawal = ArxivWithdrawal(
        requested_id="2603.20546", latest_id="2603.20546v2",
        observed_at="2026-10-09T00:00:00+00:00", evidence="Latest version withdrawn",
        reason="withdrawn", evidence_url="https://arxiv.org/abs/2603.20546")
    lib.pdf_path(paper.key).write_bytes(b"%PDF-existing")
    assert download.download_paper(paper, lib) is False
    terminal, = completed_spans(lib, "cascade")
    assert terminal["paper_status"] == "failed"
    assert terminal["pdf_returned"] is False
    outcome = next(e for e in events(lib) if e["event"] == "download_failed")
    assert outcome["timing"]["run_id"] == terminal["run_id"]
    assert lib.pdf_path(paper.key).read_bytes() == b"%PDF-existing"
