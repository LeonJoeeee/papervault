"""Regression tests for Batch 2 idle CPU/SSD stop-bleeding, off-loop saves, and concurrency."""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
import threading
from types import SimpleNamespace

import pytest

from papervault.library import Library, fetch
from papervault.library.claims_v2 import BudgetLedger
from papervault.library.merge_v2 import MergeStore
from papervault.library.services import concurrency, extract_defer, reconcile
from papervault.library.services.registration import ServingTrace
from papervault.library.state_v2 import Merge, Policy, Retrieval, TerminalState, check_index_readable


class _Resp:
    def __init__(
        self,
        status: int,
        payload=None,
        *,
        bad_json: bool = False,
        headers: dict | None = None,
        text: str = "",
        url: str = "",
    ):
        self.status_code = status
        self._payload = payload
        self._bad_json = bad_json
        self.headers = headers or {}
        self.text = text
        self.url = url
        self.ok = 200 <= status < 300

    def json(self):
        if self._bad_json:
            raise json.JSONDecodeError("Expecting value", "<xml/>", 0)
        return self._payload


@pytest.mark.asyncio
async def test_with_dedup_leader_cancellation_wakes_subscribers():
    """Leader cancellation cancels the shared future so subscribers never hang forever."""
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(60)

    leader = asyncio.create_task(concurrency.with_dedup("cancel-leader", slow))
    await started.wait()
    subscriber = asyncio.create_task(concurrency.with_dedup("cancel-leader", slow))
    await asyncio.sleep(0)

    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(leader, timeout=2.0)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(subscriber, timeout=2.0)
    assert not concurrency.is_in_flight("cancel-leader")


@pytest.mark.asyncio
async def test_with_dedup_subscriber_cancellation_preserves_leader_and_peers():
    """Cancelling one subscriber must not corrupt the shared future for leader or peer subscribers."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def work():
        started.set()
        await release.wait()
        return "shared-value"

    leader = asyncio.create_task(concurrency.with_dedup("cancel-sub", work))
    await started.wait()
    sub1 = asyncio.create_task(concurrency.with_dedup("cancel-sub", work))
    sub2 = asyncio.create_task(concurrency.with_dedup("cancel-sub", work))
    await asyncio.sleep(0)

    sub1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sub1

    release.set()
    assert await asyncio.wait_for(leader, timeout=2.0) == "shared-value"
    assert await asyncio.wait_for(sub2, timeout=2.0) == "shared-value"


@pytest.mark.parametrize("resp", [
    _Resp(400),
    _Resp(410),
    _Resp(200, bad_json=True, headers={"Content-Type": "application/xml"}),
    _Resp(200, bad_json=True, headers={"Content-Type": "text/xml; charset=utf-8"}),
    _Resp(
        200,
        bad_json=True,
        headers={},
        text='<?xml version="1.0" encoding="UTF-8"?>\n<crossref_result xmlns="http://www.crossref.org/qrschema/3.0">',
    ),
    _Resp(
        200,
        bad_json=True,
        headers={},
        url="https://api.crossref.org/works/10.5194/angeo-44-263-2026/transform/application/vnd.crossref.unixsd+xml",
    ),
    _Resp(200, payload="not-a-dict"),
], ids=["400", "410", "app-xml-200", "text-xml-200", "headerless-crossref-xml-200", "unixsd-url-200", "non-dict-200"])
def test_crossref_enrich_poison_pill_is_miss_not_transient(monkeypatch, resp):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: resp)
    assert fetch.crossref_enrich_lookup("10.5194/angeo-44-263-2026.xml") == ("miss", {})


@pytest.mark.parametrize("resp", [
    _Resp(200, bad_json=True, headers={"Content-Type": "text/html; charset=utf-8"}),
    _Resp(200, bad_json=True, headers={"Content-Type": "application/json"}),
    _Resp(200, bad_json=True),
    _Resp(200, bad_json=True, headers={}, text='<?xml version="1.0"?><Error><Code>ServiceUnavailable</Code></Error>'),
    _Resp(
        200,
        bad_json=True,
        headers={"Content-Type": "text/html; charset=utf-8"},
        text="<html><title>Temporary service outage</title></html>",
        url="https://api.crossref.org/works/10.5194/angeo-44-263-2026/transform/application/vnd.crossref.unixsd+xml",
    ),
], ids=[
    "html-gateway-200",
    "truncated-json-200",
    "unknown-content-type-200",
    "generic-xml-error-200",
    "html-outage-after-unixsd-redirect",
])
def test_crossref_enrich_html_or_truncated_200_stays_transient(monkeypatch, resp):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: resp)
    assert fetch.crossref_enrich_lookup("10.3847/1538-4357/good") == ("transient", {})


@pytest.mark.asyncio
async def test_enrich_sweep_skips_save_when_all_transient_and_runs_save_off_loop(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    try:
        lib.upsert({"title": "A sufficiently long title for enrich save guard test",
                    "authors": ["A. Author"], "year": 2024, "doi": "10.1/transient"})
        save_threads = []
        loop_thread = threading.get_ident()
        orig_save = lib.save

        def spy_save(*args, **kwargs):
            save_threads.append(threading.get_ident())
            return orig_save(*args, **kwargs)

        monkeypatch.setattr(lib, "save", spy_save)
        monkeypatch.setattr(fetch, "crossref_enrich_lookup", lambda doi: ("transient", {}))
        assert await reconcile._enrich_sweep(lib, cap=10) == 0
        assert save_threads == []  # zero save calls when all transient!

        monkeypatch.setattr(fetch, "crossref_enrich_lookup", lambda doi: ("miss", {}))
        assert await reconcile._enrich_sweep(lib, cap=10) == 0
        assert len(save_threads) == 1
        assert save_threads[0] != loop_thread  # off the event loop!
    finally:
        lib.close()


@pytest.mark.asyncio
async def test_abstract_sweep_stamps_miss_skips_reprobe_and_lifts_on_new_extract_or_cooldown(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    try:
        p, _ = lib.upsert({"title": "Unfillable paper without abstract or extract",
                           "authors": ["Smith"], "year": 2024, "doi": "10.1/noabs"})
        p.abstract = ""
        # Untouched Paper omits abstract_attempted_at from JSON serialization.
        assert "abstract_attempted_at" not in p.model_dump(mode="json")
        lib.save()

        # Transient DOI failure must NOT stamp abstract_attempted_at.
        monkeypatch.setattr(fetch, "_fetch_abstract_by_doi_3state", lambda doi: ("transient", ""))
        assert await reconcile._abstract_sweep(lib, cap=10) == 0
        assert lib.get(p.key).abstract_attempted_at is None

        # Clean miss stamps abstract_attempted_at and prevents re-probing next sweep.
        probes = []

        def clean_miss(doi):
            probes.append(doi)
            return ("miss", "")

        monkeypatch.setattr(fetch, "_fetch_abstract_by_doi_3state", clean_miss)
        assert await reconcile._abstract_sweep(lib, cap=10) == 0
        assert probes == ["10.1/noabs"]
        assert lib.get(p.key).abstract_attempted_at is not None
        assert await reconcile._abstract_sweep(lib, cap=10) == 0
        assert probes == ["10.1/noabs"]  # NOT re-probed!

        # New extract arriving after abstract_attempted_at lifts the guard automatically.
        md = lib.md_path(p.key)
        md.write_text(
            "# Title\n\n**Abstract.** We measure energetic particles in the inner heliosphere "
            "and derive new transport coefficients across multiple solar wind regimes.\n\n## Introduction\nBody."
        )
        future_ts = _dt.datetime.now(_dt.timezone.utc).timestamp() + 60
        os.utime(md, (future_ts, future_ts))
        p.md_path = str(md.relative_to(lib.root))
        assert reconcile._needs_abstract(p, lib) is True
        assert await reconcile._abstract_sweep(lib, cap=10) == 1
        assert len(lib.get(p.key).abstract) >= 60

        # Cooldown expiry (30+ days) also lifts the guard.
        p2, _ = lib.upsert({"title": "Another unfillable paper for cooldown test",
                            "authors": ["Jones"], "year": 2024, "doi": "10.1/cooldown"})
        p2.abstract = ""
        old_stamp = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=31)).isoformat()
        p2.abstract_attempted_at = old_stamp
        assert reconcile._needs_abstract(p2, lib) is True
    finally:
        lib.close()


@pytest.mark.asyncio
async def test_defer_canary_rotates_cursor_across_sweeps(tmp_path, monkeypatch):
    extract_defer.reset_for_test()
    monkeypatch.setattr(extract_defer, "DEFER_CANARY", 2)
    monkeypatch.setattr(reconcile, "_defer_canary_cursor", 0)
    monkeypatch.setattr(reconcile, "_enrich_sweep", lambda *a, **k: asyncio.sleep(0, result=0))
    monkeypatch.setattr(reconcile, "_abstract_sweep", lambda *a, **k: asyncio.sleep(0, result=0))

    lib = Library(tmp_path)
    try:
        keys = []
        for i in range(5):
            p, _ = lib.upsert({"title": f"Deferred canary rotation paper number {i:02d}",
                               "authors": [f"Author{i}"], "year": 2024, "doi": f"10.1/rot{i}"})
            p.download_status = "ok"
            lib.pdf_path(p.key).write_bytes(b"%PDF-1.4 fake bytes")
            extract_defer.mark_extract_deferred(p, lib)
            keys.append(p.key)
        lib.save()

        seen = []
        for _ in range(3):
            added = []
            dq = SimpleNamespace(add=lambda k, p=5: None)
            eq = SimpleNamespace(add=lambda k, p=5, _a=added: _a.append(k))
            await reconcile.reconcile_once(lib, dq, eq)
            seen.append(added)

        assert seen[0] == keys[0:2]
        assert seen[1] == keys[2:4]
        assert seen[2] == [keys[4], keys[0]]
    finally:
        lib.close()


def test_budget_ledger_caches_reads_and_absent_clean_state(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    try:
        lib.upsert({"title": "Budget ledger caching fixture paper title",
                    "authors": ["Author"], "year": 2024, "doi": "10.1/cache"})
        lib.save()
        ledger = BudgetLedger(tmp_path, clock=lambda: 100.0)

        # Absent ledger: first read inspects index.json, subsequent reads hit absent cache!
        loads = []
        orig_loads = json.loads

        def spy_loads(s, *a, **kw):
            loads.append(True)
            return orig_loads(s, *a, **kw)

        monkeypatch.setattr("papervault.library.claims_v2.json.loads", spy_loads)
        assert ledger.events() == ()
        first_count = len(loads)
        assert first_count >= 1
        assert ledger.events() == ()
        assert ledger.events() == ()
        assert len(loads) == first_count  # cached absent-clean!

        # After appending a campaign, first read parses it and subsequent reads hit read cache!
        campaign = ledger.start_campaign(cap=1, keys=lib.keys())
        loads.clear()
        assert ledger.spent(campaign) == 0
        parsed_once = len(loads)
        assert parsed_once == 1
        assert ledger.spent(campaign) == 0
        assert ledger.events()[0]["event"] == "campaign_start"
        assert len(loads) == parsed_once  # cached parsed rows!
    finally:
        lib.close()


def test_check_index_readable_skips_index_file_under_pg_authority(tmp_path, monkeypatch):
    (tmp_path / "index.json").write_text(json.dumps({"version": 999, "papers": {}}))
    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")
    # Under PG authority, stale/unreadable index.json is not opened or parsed.
    check_index_readable(tmp_path)


def test_manifest_rotation_and_serving_trace_gating(tmp_path, monkeypatch):
    lib = Library(tmp_path)
    try:
        # Gate verbose per-candidate traces when PAPERVAULT_SERVING_TRACE=0.
        monkeypatch.setenv("PAPERVAULT_SERVING_TRACE", "0")
        trace = ServingTrace(lib, "gate-test")
        with trace.batch():
            trace.record("intent", model="test-model", latency_ms=12)
            trace.record("fusion", key="PaperA", reason="pool_selection")
            trace.record("registration", candidate={"title": "Candidate", "doi": "10.1/c"}, decision="submitted")
            trace.record("registration_summary", registered=1, duplicate=0, rejected=0, failed=0)
        events = [json.loads(line) for line in lib.manifest_path.read_text().splitlines()]
        assert [e["stage"] for e in events] == ["intent", "registration_summary"]

        # Size-based in-place tail compaction keeps manifest.log bounded without creating
        # extra vault files (such as manifest.log.1) that would pin coordinated recovery backups.
        monkeypatch.setenv("PAPERVAULT_MANIFEST_MAX_BYTES", "300")
        for i in range(10):
            lib.log({"event": "rotation_test", "seq": i, "pad": "x" * 60})
        assert not lib.manifest_path.with_name("manifest.log.1").exists()
        assert lib.manifest_path.exists()
        assert lib.manifest_path.stat().st_size < 450
        lines = lib.manifest_path.read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(line) for line in lines]
        assert parsed[-1]["seq"] == 9

        from papervault.store.recovery_source import _MUTABLE_VAULT_STATE, _files
        unlisted = [
            p.relative_to(tmp_path).as_posix()
            for p in _files(tmp_path)
            if p.relative_to(tmp_path).as_posix() not in _MUTABLE_VAULT_STATE
        ]
        assert unlisted == []
    finally:
        lib.close()


def test_consolidate_identity_cleans_up_ghost_twin_on_quiesce_error(tmp_path, monkeypatch):
    """If _quiesce raises during _consolidate_identity, the synthetic twin must not leak into _papers."""
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    lib = Library(tmp_path)
    try:
        existing, _ = lib.upsert({
            "title": "Magnetohydrodynamic turbulence in the solar wind",
            "authors": ["Parker, E."], "year": 2024, "arxiv_id": "2401.12345v1",
        })
        lib.save()
        before_keys = set(lib._papers)

        def boom(*args, **kwargs):
            raise ValueError("conflicting resolved arxiv_id identities; adjudication required")

        monkeypatch.setattr(MergeStore, "_quiesce", boom)
        with pytest.raises(ValueError, match="conflicting resolved"):
            lib.upsert({
                "title": "Magnetohydrodynamic turbulence in the solar wind",
                "authors": ["Parker, E."], "year": 2024, "doi": "10.1234/parker",
            })
        assert set(lib._papers) == before_keys
        assert "10.1234/parker" not in lib._by_doi
    finally:
        lib.close()


@pytest.mark.asyncio
async def test_merge_records_restores_fields_and_indexes_when_save_fails(tmp_path, monkeypatch):
    """A failed _save('identity_merge') restores left/right fields and indexes alongside uncertain status."""
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    lib = Library(tmp_path)
    try:
        left, _ = lib.upsert({"title": "Canonical paper for rollback test",
                              "authors": ["Curie"], "doi": "10.1234/left"})
        right, _ = lib.upsert({"title": "Absorbed paper for rollback test",
                               "authors": ["Einstein"], "arxiv_id": "2401.99999",
                               "venue": "ApJ"})
        for p in (left, right):
            p.download_status = "failed"
            p.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
        lib.save()
        ledger = BudgetLedger(tmp_path, clock=lambda: 100.0)
        merge = MergeStore(lib, ledger=ledger, clock=lambda: 100.0)
        await merge.quiesce(left.key, right.key)

        orig_save = lib.save

        def fail_on_merge(**kwargs):
            if left.merge_budget is not None:
                raise OSError("injected save failure")
            return orig_save(**kwargs)

        monkeypatch.setattr(lib, "save", fail_on_merge)
        with pytest.raises(OSError, match="injected save failure"):
            await merge.merge_records(left.key, right.key)

        assert left.arxiv_id == ""
        assert left.venue == ""
        assert left.merge_budget is None
        assert left.merge_intent.status == "uncertain"
        assert left.acquisition_state.quarantine is True
        assert right.acquisition_state.merge == Merge.CANONICAL
        assert right.acquisition_state.alias_key == ""
        assert right.merge_budget is None
        assert right.merge_intent.status == "uncertain"
        assert lib._by_arxiv["2401.99999"] == right.key
    finally:
        lib.close()


def test_trim_process_memory_runs_cleanly():
    result = concurrency.trim_process_memory()
    assert isinstance(result, bool)


def test_budget_ledger_events_returns_defensive_copies(tmp_path):
    """Mutating rows returned by BudgetLedger.events() must never corrupt cached ledger state."""
    lib = Library(tmp_path)
    try:
        p, _ = lib.upsert({"title": "Defensive copy ledger fixture paper",
                           "authors": ["Author"], "year": 2024, "doi": "10.1/defcopy"})
        lib.save()
        ledger = BudgetLedger(tmp_path, clock=lambda: 100.0)
        ledger.start_campaign(cap=2, keys=[p.key])

        ev1 = ledger.events()
        ev1[0]["event"] = "corrupted"
        ev1[0]["keys"].append("INJECTED")

        ev2 = ledger.events()
        assert ev2[0]["event"] == "campaign_start"
        assert ev2[0]["keys"] == [p.key]
    finally:
        lib.close()


def test_budget_ledger_pg_absent_cache_invalidates_on_catalog_revision_bump(tmp_path, monkeypatch):
    """PG absent-ledger cache skips JSONB scan when (max_rev, cnt) is unchanged and rechecks on bump."""
    from contextlib import contextmanager
    from papervault.library import claims_v2
    from papervault.library.claims_v2 import LedgerError
    from papervault.store.persistence import PostgresPersistence

    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")
    monkeypatch.setenv("PAPERVAULT_CATALOG_DSN", "postgresql://unused/absent-cache")
    monkeypatch.setenv("PAPERVAULT_CATALOG_SCHEMA", "catalog")

    state = {"max_rev": 5, "cnt": 10, "has_claim": False, "claim_scans": 0, "stamp_queries": 0}

    class FakeResult:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class FakeConn:
        def execute(self, stmt, params=None):
            text = str(stmt)
            if "MAX(revision)" in text:
                state["stamp_queries"] += 1
                return FakeResult((state["max_rev"], state["cnt"]))
            state["claim_scans"] += 1
            return FakeResult((1,) if state["has_claim"] else None)

    @contextmanager
    def fake_pg_conn(dsn):
        yield FakeConn()

    monkeypatch.setattr(claims_v2, "_pg_absent_connection", fake_pg_conn)
    monkeypatch.setattr(PostgresPersistence, "from_environment", lambda: PostgresPersistence("postgresql://unused/absent-cache"))

    ledger = BudgetLedger(tmp_path, clock=lambda: 100.0)
    assert ledger.events() == ()
    assert state["claim_scans"] == 1
    # Unchanged (max_rev, cnt) skips the JSONB claim scan completely!
    assert ledger.events() == ()
    assert state["claim_scans"] == 1
    assert state["stamp_queries"] == 2

    # External writer commits a claim (bumping max_rev): cache invalidates and fails closed!
    state["max_rev"] = 6
    state["has_claim"] = True
    with pytest.raises(LedgerError, match="persisted claims"):
        ledger.events()
    assert state["claim_scans"] == 2


@pytest.mark.asyncio
async def test_bibtex_failure_does_not_revert_committed_quarantine_or_merge_quiesce(tmp_path, monkeypatch):
    """A BibTeX projection OSError must not revert committed quarantine or merge_intent fences."""
    from papervault.library.claims_v2 import ClaimStore

    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    monkeypatch.setenv("PAPERVAULT_MERGE_V2", "1")
    lib = Library(tmp_path)
    try:
        p1, _ = lib.upsert({"title": "Paper one for bibtex fault isolation test",
                            "authors": ["Curie"], "year": 2024, "doi": "10.1/bib1"})
        p2, _ = lib.upsert({"title": "Paper two for bibtex fault isolation test",
                            "authors": ["Einstein"], "year": 2024, "arxiv_id": "2401.11111"})
        for p in (p1, p2):
            p.download_status = "failed"
            p.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
        lib.save()

        ledger = BudgetLedger(tmp_path, clock=lambda: 100.0)
        claims = ClaimStore(lib, ledger=ledger, clock=lambda: 100.0)
        campaign = ledger.start_campaign(cap=1, keys=[p1.key])
        token = await claims.claim(
            p1.key, expected_state=p1.acquisition_state, expected_generation=0, campaign_id=campaign, ttl=60.0
        )

        # Inject BibTeX failure while making bib interval look elapsed.
        def failing_bib():
            raise OSError("simulated disk full on library.bib")

        monkeypatch.setattr(lib, "_save_bib", failing_bib)
        lib._last_bib_save = 0.0

        await claims.quarantine(token, reason="test quarantine with broken bib")
        assert p1.acquisition_state.quarantine is True
        assert lib._bib_dirty is True

        p3, _ = lib.upsert({"title": "Paper three for merge quiesce bib isolation",
                            "authors": ["Bohr"], "year": 2024, "doi": "10.1/bib3"})
        p3.download_status = "failed"
        p3.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
        lib.save(force=True, force_bib=False)

        merge = MergeStore(lib, ledger=ledger, clock=lambda: 100.0)
        intent = await merge.quiesce(p2.key, p3.key)
        assert intent is not None
        assert p2.merge_intent == intent and p3.merge_intent == intent
        assert lib._bib_dirty is True
    finally:
        lib.close()


@pytest.mark.asyncio
async def test_cancelling_not_sent_waits_for_thread_before_releasing_lock(tmp_path, monkeypatch):
    """Cancelling ClaimStore.not_sent() must keep lib_write_lock held until _not_sent thread drains."""
    from papervault.library.claims_v2 import ClaimStore

    monkeypatch.setenv("PAPERVAULT_REOPEN_V2", "1")
    lib = Library(tmp_path)
    try:
        p, _ = lib.upsert({"title": "Paper for not_sent cancellation lock shielding",
                           "authors": ["Fermi"], "year": 2024, "doi": "10.1/shield"})
        p.download_status = "failed"
        p.terminal_state = TerminalState(retrieval=Retrieval.EXHAUSTED, policy=Policy.CLEAR)
        lib.save()

        ledger = BudgetLedger(tmp_path, clock=lambda: 100.0)
        claims = ClaimStore(lib, ledger=ledger, clock=lambda: 100.0)
        campaign = ledger.start_campaign(cap=1, keys=[p.key])
        token = await claims.claim(
            p.key, expected_state=p.acquisition_state, expected_generation=0, campaign_id=campaign, ttl=60.0
        )

        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        orig_not_sent = claims._not_sent

        def slow_not_sent(paper, tok):
            entered.set()
            assert release.wait(5.0)
            res = orig_not_sent(paper, tok)
            finished.set()
            return res

        monkeypatch.setattr(claims, "_not_sent", slow_not_sent)
        task = asyncio.create_task(claims.not_sent(token))
        assert await asyncio.to_thread(entered.wait, 5.0)

        task.cancel()
        await asyncio.sleep(0.02)
        # Lock MUST stay held while the _not_sent worker thread is still running!
        assert concurrency.lib_write_lock.locked() is True
        assert not task.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)
        assert finished.is_set()
        assert concurrency.lib_write_lock.locked() is False
    finally:
        lib.close()


@pytest.mark.asyncio
async def test_abstract_sweep_candidate_selection_off_loop_and_stops_at_cap(tmp_path, monkeypatch):
    """_abstract_sweep selects candidates off the event loop and stops scanning once cap is reached."""
    lib = Library(tmp_path)
    try:
        old_stamp = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=1)).isoformat()
        for i in range(15):
            p, _ = lib.upsert({"title": f"Abstract sweep off-loop selection paper {i:02d}",
                               "authors": [f"Author{i}"], "year": 2024, "doi": f"10.1/sel{i}"})
            p.abstract = ""
            # First 5 have no attempt stamp (immediately eligible); rest have recent stamp triggering stat().
            if i < 2:
                p.abstract_attempted_at = old_stamp
                md = lib.md_path(p.key)
                md.write_text("# Title\n\nBody without abstract.")
                future_ts = _dt.datetime.now(_dt.timezone.utc).timestamp() + 60
                os.utime(md, (future_ts, future_ts))
                p.md_path = str(md.relative_to(lib.root))
            else:
                p.abstract_attempted_at = old_stamp
        lib.save()

        loop_tid = threading.get_ident()
        stat_threads = []
        orig_has_newer = reconcile._has_newer_extract

        def spy_has_newer(library, paper, attempted_dt):
            stat_threads.append(threading.get_ident())
            return orig_has_newer(library, paper, attempted_dt)

        monkeypatch.setattr(reconcile, "_has_newer_extract", spy_has_newer)
        monkeypatch.setattr(fetch, "_fetch_abstract_by_doi_3state", lambda doi: ("transient", ""))

        await reconcile._abstract_sweep(lib, cap=2)
        # Candidate stat() ran off the event loop and stopped immediately at cap=2 (never scanned all 15)!
        assert len(stat_threads) == 2
        assert all(tid != loop_tid for tid in stat_threads)
    finally:
        lib.close()
