"""Regression tests for Batch 2 knowledge scheduler idle CPU & split-authority fixes."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager

import pytest

from papervault.knowledge.ingest import paper_library_client as client
from papervault.knowledge.ledger.store import LedgerRecord
from papervault.knowledge.scheduler import round as rnd
from papervault.knowledge.store import graph as graph_mod


def test_assert_safe_workspace_logs_info_once_then_debug(monkeypatch, caplog):
    monkeypatch.setenv("NEO4J_WORKSPACE", "l0_probe")
    monkeypatch.setenv("POSTGRES_WORKSPACE", "l0_probe")
    monkeypatch.delenv("KS_ALLOW_PROD_WORKSPACE", raising=False)
    graph_mod._logged_workspaces.clear()

    with caplog.at_level(logging.DEBUG, logger="ks.store.graph"):
        assert graph_mod.assert_safe_workspace() == "l0_probe"
        assert graph_mod.assert_safe_workspace() == "l0_probe"
        assert graph_mod.assert_safe_workspace() == "l0_probe"

    info_records = [r for r in caplog.records if r.levelno == logging.INFO and "workspace=l0_probe" in r.message]
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG and "workspace=l0_probe" in r.message]
    assert len(info_records) == 1
    assert len(debug_records) == 2


def test_paper_library_client_cache_maxsize_and_garbage_intake_fence(tmp_path):
    from papervault.knowledge.scheduler.reconcile import diff

    assert client._load_vault_index.cache_info().maxsize == 1

    entries = {
        "Clean2024": {"key": "Clean2024", "title": "Solar wind turbulence", "authors": ["Smith, J."],
                      "venue": "ApJ", "doi": "10.3847/1234", "publication_types": ["JournalArticle"]},
        "AguAbs": {"key": "AguAbs", "title": "AGU talk", "authors": ["Smith, J."],
                   "venue": "AGU Fall Meeting Abstracts"},
        "EguAbs": {"key": "EguAbs", "title": "EGU talk", "authors": ["Smith, J."],
                   "doi": "10.5194/egusphere-egu24-1234"},
        "Dataset": {"key": "Dataset", "title": "Magnetometer data", "authors": ["Smith, J."],
                    "publication_types": ["Dataset"]},
        "Anon": {"key": "Anon", "title": "Anonymous note", "authors": ["Anonymous"]},
        "Quarantined": {"key": "Quarantined", "title": "Off domain", "authors": ["Smith, J."],
                        "domain_status": "off_domain"},
    }
    (tmp_path / "index.json").write_text(json.dumps({"version": 1, "papers": entries}))
    clean = client.load_vault_index(tmp_path)
    # Quarantined records remain excluded from clean view; pre-fence garbage records remain in clean
    # view so already-distilled docs are NOT retroactively mass-purged, while flagged for intake skip.
    assert "Quarantined" not in clean
    assert clean["Clean2024"].is_deterministic_garbage is False
    for junk_key in ("AguAbs", "EguAbs", "Dataset", "Anon"):
        assert clean[junk_key].is_deterministic_garbage is True

    # New deterministic garbage is skipped by diff().to_distill, while already-distilled legacy
    # records stay untouched (NOT added to to_remove!).
    existing_led = {
        "AguAbs": LedgerRecord("l0_probe", "paper", "AguAbs", "fp_agu", "paper:AguAbs", "done"),
    }
    d = diff(clean, existing_led, fp_of=lambda r: "fp_agu" if r.key == "AguAbs" else f"fp_{r.key}")
    assert [k for k, _ in d.to_distill] == ["Clean2024"]
    assert d.to_remove == []
    assert d.to_redistill == []

    all_recs = client.load_vault_index(tmp_path, include_quarantined=True)
    assert set(all_recs) == set(entries)


def test_paper_library_client_reads_pg_directly_when_authoritative(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")
    monkeypatch.setenv("PAPERVAULT_CATALOG_DSN", "postgresql://unused/catalog")
    monkeypatch.setenv("PAPERVAULT_CATALOG_SCHEMA", "catalog")
    monkeypatch.setattr(client, "DEFAULT_VAULT", tmp_path)
    client._load_pg_vault_index.cache_clear()

    state = {
        "rows": [
            {"key": "P1", "revision": 1, "payload": {"key": "P1", "title": "Real PG paper", "authors": ["A. B."]}},
            {"key": "OffDom", "revision": 1, "payload": {"key": "OffDom", "title": "Off domain", "domain_status": "off_domain"}},
        ],
        "full_scans": 0,
    }

    class FakeResult:
        def __init__(self, row=None, rows=None):
            self._row = row
            self._rows = rows

        def fetchone(self):
            return self._row

        def fetchall(self):
            return self._rows

    class FakeConn:
        def execute(self, stmt, params=None):
            text = str(stmt)
            if "COUNT(*)" in text:
                revs = [r["revision"] for r in state["rows"]]
                return FakeResult(row={"sum_rev": sum(revs), "max_rev": max(revs, default=0), "cnt": len(revs)})
            state["full_scans"] += 1
            return FakeResult(rows=list(state["rows"]))

    @contextmanager
    def fake_pg_conn(dsn):
        yield FakeConn()

    monkeypatch.setattr(client, "_pg_connection", fake_pg_conn)

    # First read queries PG directly (index.json does not even exist!) and filters garbage.
    first = client.load_vault_index(tmp_path)
    assert set(first) == {"P1"}
    assert state["full_scans"] == 1

    # Second read with unchanged (sum_rev, max_rev, cnt) hits cache (0 extra full scans).
    second = client.load_vault_index(tmp_path)
    assert set(second) == {"P1"}
    assert state["full_scans"] == 1

    # Updating a record revision immediately invalidates the cache without waiting for index.json export.
    state["rows"].append(
        {"key": "P2", "revision": 1, "payload": {"key": "P2", "title": "Fresh paper", "authors": ["C. D."]}}
    )
    third = client.load_vault_index(tmp_path)
    assert set(third) == {"P1", "P2"}
    assert state["full_scans"] == 2


@pytest.mark.asyncio
async def test_reconcile_terminal_uses_load_by_status_only(monkeypatch):
    """reconcile_terminal must query processing rows via load_by_status, never full ledger.load."""
    calls = []

    class NarrowLedger:
        async def load(self, source):
            raise AssertionError("reconcile_terminal must not call full ledger.load()")

        async def load_by_status(self, statuses):
            calls.append(tuple(statuses))
            return [
                LedgerRecord("l0_probe", "paper", "P1", "fp1", "paper:P1", "processing"),
                LedgerRecord("l0_probe", "operator", "Op1", "fp2", "op:Op1", "processing"),
            ]

        async def upsert(self, ingest_source, source_id, *, doc_id, status, **kw):
            calls.append(("upsert", ingest_source, source_id, status))

    class StubRag:
        async def aget_docs_by_ids(self, ids):
            return {i: {"status": "processed", "doc_id": i} for i in ids}

    monkeypatch.setattr(rnd, "ledger", NarrowLedger())
    out = await rnd.reconcile_terminal(StubRag())
    assert out["done"] == 1
    assert calls[0] == ("processing",)
    assert ("upsert", "paper", "P1", "done") in calls


@pytest.mark.asyncio
async def test_run_round_idle_fast_path_skips_heavy_work_and_wakes_on_change(monkeypatch):
    """When _idle_state is supplied, unchanged idle ticks skip heavy scans until state changes or interval expires."""
    from papervault.knowledge.ingest.paper_library_client import PaperRecord

    rows = {"A": LedgerRecord("l0_probe", "paper", "A", "fpA", "paper:A", "done")}
    rag_rows = {"paper:A": {"status": "processed", "file_path": "A", "metadata": {}}}
    cat_token = [("json", "/vault/index.json", 100, 500, 10)]
    load_calls = []
    process_calls = []

    class FakeLedger:
        def __init__(self):
            self.rows = rows

        async def load(self, source):
            return {k: v for k, v in self.rows.items() if v.ingest_source == source}

        async def load_by_status(self, statuses):
            ss = set(statuses)
            return [r for r in self.rows.values() if r.status in ss]

    class FakeDocStatus:
        async def get_by_ids(self, ids):
            return [rag_rows.get(i) for i in ids]

        async def get_by_id(self, i):
            return rag_rows.get(i)

        async def get_status_counts(self):
            out = {}
            for r in rag_rows.values():
                out[r["status"]] = out.get(r["status"], 0) + 1
            return out

    class FakeRag:
        rows = rag_rows
        doc_status = FakeDocStatus()

        async def apipeline_process_enqueue_documents(self):
            process_calls.append(True)

    fl = FakeLedger()
    monkeypatch.setattr(rnd, "ledger", fl)
    from papervault.knowledge.scheduler import reconcile as rec_mod
    monkeypatch.setattr(rec_mod, "ledger", fl)
    monkeypatch.setattr(rnd, "catalog_state_token", lambda: cat_token[0])
    monkeypatch.setattr(rnd, "fingerprint", lambda rec: "fpA")

    def fake_load_index():
        load_calls.append(True)
        return {"A": PaperRecord(key="A", md_path="extracts/md/A.md")}

    monkeypatch.setattr(rnd, "load_clean_index", fake_load_index)

    idle_state = {"token": None, "last_full_at": 0.0}
    rag = FakeRag()

    # Round 1: cold start runs full round and records clean idle token.
    r1 = await rnd.run_round(rag, _idle_state=idle_state)
    assert not r1.get("skipped_idle")
    assert len(load_calls) == 1
    assert len(process_calls) == 1
    assert idle_state["token"] is not None

    # Round 2: unchanged state skips load_clean_index, reconcile_library, diff, and process!
    r2 = await rnd.run_round(rag, _idle_state=idle_state)
    assert r2 == {"skipped_idle": True}
    assert len(load_calls) == 1
    assert len(process_calls) == 1

    # Round 3: catalog token changes -> wakes up and runs full round!
    cat_token[0] = ("json", "/vault/index.json", 200, 600, 10)
    r3 = await rnd.run_round(rag, _idle_state=idle_state)
    assert not r3.get("skipped_idle")
    assert len(load_calls) == 2

    # Round 4: periodic full reconcile interval expires -> runs full round even if token unchanged!
    idle_state["last_full_at"] = time.monotonic() - 2000.0
    r4 = await rnd.run_round(rag, _idle_state=idle_state)
    assert not r4.get("skipped_idle")
    assert len(load_calls) == 3

    # Direct run_round(rag) without _idle_state always runs full round.
    r5 = await rnd.run_round(rag)
    assert not r5.get("skipped_idle")
    assert len(load_calls) == 4


def test_paper_library_client_pg_missing_dsn_fails_closed_on_default_vault_only(tmp_path, monkeypatch):
    """PG authority with missing DSN must fail closed on DEFAULT_VAULT, while allowing custom test vaults."""
    default_dir = tmp_path / "default_vault"
    custom_dir = tmp_path / "custom_vault"
    default_dir.mkdir()
    custom_dir.mkdir()
    payload = json.dumps({
        "version": 1,
        "papers": {"P1": {"key": "P1", "title": "Paper", "authors": ["Smith"]}},
    })
    (default_dir / "index.json").write_text(payload)
    (custom_dir / "index.json").write_text(payload)

    monkeypatch.setenv("PAPERVAULT_CATALOG_AUTHORITY", "postgres")
    monkeypatch.delenv("PAPERVAULT_CATALOG_DSN", raising=False)
    monkeypatch.setattr(client, "DEFAULT_VAULT", default_dir)

    with pytest.raises(ValueError, match="PAPERVAULT_CATALOG_DSN"):
        client.load_vault_index(default_dir)
    with pytest.raises(ValueError, match="PAPERVAULT_CATALOG_DSN"):
        client.catalog_state_token(default_dir)

    # Custom isolated vault still reads its own index.json cleanly.
    recs = client.load_vault_index(custom_dir)
    assert set(recs) == {"P1"}


@pytest.mark.asyncio
async def test_main_loop_stays_idle_when_drain_pending_returns_zero_counts(monkeypatch):
    """main_loop must NOT reset idle_state['token'] when drain_pending returns an all-zero count dict."""
    import asyncio
    from papervault.knowledge.ingest.paper_library_client import PaperRecord

    rows = {"A": LedgerRecord("l0_probe", "paper", "A", "fpA", "paper:A", "done")}
    rag_rows = {"paper:A": {"status": "processed", "file_path": "A", "metadata": {}}}
    load_calls = []
    drain_returns = [
        {"ingested": 0, "partial": 0, "already": 0, "disabled": 0, "deferred": 0, "failed": 0},
        {"ingested": 1, "partial": 0, "already": 0, "disabled": 0, "deferred": 0, "failed": 0},
        {"ingested": 0, "partial": 0, "already": 0, "disabled": 0, "deferred": 0, "failed": 0},
    ]

    class FakeLedger:
        def __init__(self):
            self.rows = rows

        async def ensure_schema(self):
            return None

        async def load(self, source):
            return {k: v for k, v in self.rows.items() if v.ingest_source == source}

        async def load_by_status(self, statuses):
            ss = set(statuses)
            return [r for r in self.rows.values() if r.status in ss]

    class FakeDocStatus:
        async def get_by_ids(self, ids):
            return [rag_rows.get(i) for i in ids]

        async def get_by_id(self, i):
            return rag_rows.get(i)

        async def get_status_counts(self):
            return {"processed": 1}

    class FakeRag:
        rows = rag_rows
        doc_status = FakeDocStatus()

        async def apipeline_process_enqueue_documents(self):
            return None

    fl = FakeLedger()
    monkeypatch.setattr(rnd, "ledger", fl)
    from papervault.knowledge.scheduler import reconcile as rec_mod
    monkeypatch.setattr(rec_mod, "ledger", fl)
    monkeypatch.setattr(rnd, "catalog_state_token", lambda: ("json", "/vault/index.json", 100, 500, 10))
    monkeypatch.setattr(rnd, "fingerprint", lambda rec: "fpA")

    def fake_load_index():
        load_calls.append(True)
        return {"A": PaperRecord(key="A", md_path="extracts/md/A.md")}

    monkeypatch.setattr(rnd, "load_clean_index", fake_load_index)

    tick = [0]

    async def fake_drain(rag):
        res = drain_returns[tick[0]]
        tick[0] += 1
        return res

    async def fake_sleep(_interval):
        if tick[0] >= 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(rnd, "drain_pending", fake_drain)
    monkeypatch.setattr(rnd.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await rnd.main_loop(FakeRag(), interval=0.01)

    # Tick 1 ran full round (load_calls=1); drain_pending returned all zeros ->
    # Tick 2 stayed IDLE (load_calls still 1!); then drain_pending returned ingested=1 ->
    # Tick 3 woke up and ran full round (load_calls=2)!
    assert len(load_calls) == 2


@pytest.mark.asyncio
async def test_run_round_idle_fast_path_tolerates_stable_failed_docs_and_wakes_on_change(monkeypatch):
    """Historical terminal failed rows (dup-*, error_parked) must not disable the idle fast-path,
    while any failed-count change or in-flight status must immediately wake a full round."""
    from papervault.knowledge.ingest.paper_library_client import PaperRecord

    rows = {
        "A": LedgerRecord("l0_probe", "paper", "A", "fpA", "paper:A", "done"),
        "P": LedgerRecord("l0_probe", "paper", "P", "fpP", "paper:P", "error_parked"),
    }
    doc_counts = {"processed": 1, "failed": 2}  # e.g. 1 error_parked + 1 dup-* audit row
    rag_rows = {
        "paper:A": {"status": "processed", "file_path": "A", "metadata": {}},
        "paper:P": {"status": "failed", "file_path": "P", "metadata": {}},
    }
    load_calls = []

    class FakeLedger:
        def __init__(self):
            self.rows = rows

        async def load(self, source):
            return {k: v for k, v in self.rows.items() if v.ingest_source == source}

        async def load_by_status(self, statuses):
            ss = set(statuses)
            return [r for r in self.rows.values() if r.status in ss]

    class FakeDocStatus:
        async def get_by_ids(self, ids):
            return [rag_rows.get(i) for i in ids]

        async def get_by_id(self, i):
            return rag_rows.get(i)

        async def get_status_counts(self):
            return dict(doc_counts)

    class FakeRag:
        rows = rag_rows
        doc_status = FakeDocStatus()

        async def apipeline_process_enqueue_documents(self):
            return None

    fl = FakeLedger()
    monkeypatch.setattr(rnd, "ledger", fl)
    from papervault.knowledge.scheduler import reconcile as rec_mod
    monkeypatch.setattr(rec_mod, "ledger", fl)
    monkeypatch.setattr(rnd, "catalog_state_token", lambda: ("pg", 10, 2, "2026-11-21"))
    monkeypatch.setattr(rnd, "fingerprint", lambda rec: f"fp{rec.key}")

    def fake_load_index():
        load_calls.append(True)
        return {
            "A": PaperRecord(key="A", md_path="extracts/md/A.md"),
            "P": PaperRecord(key="P", md_path="extracts/md/P.md"),
        }

    monkeypatch.setattr(rnd, "load_clean_index", fake_load_index)

    idle_state = {"token": None, "last_full_at": 0.0}
    rag = FakeRag()

    # Round 1: full round finishes clean with stable failed rows -> records idle token.
    r1 = await rnd.run_round(rag, _idle_state=idle_state)
    assert not r1.get("skipped_idle")
    assert idle_state["token"] is not None
    assert len(load_calls) == 1

    # Round 2: unchanged counts (including failed=2 and pending=0) -> hits idle fast-path!
    doc_counts["pending"] = 0
    r2 = await rnd.run_round(rag, _idle_state=idle_state)
    assert r2 == {"skipped_idle": True}
    assert len(load_calls) == 1

    # Round 3: failed count changes (e.g. a new doc failed or one healed) -> wakes up!
    doc_counts["failed"] = 3
    r3 = await rnd.run_round(rag, _idle_state=idle_state)
    assert not r3.get("skipped_idle")
    assert len(load_calls) == 2

    # Round 4: in-flight pending doc appears -> wakes up immediately!
    doc_counts["pending"] = 1
    r4 = await rnd.run_round(rag, _idle_state=idle_state)
    assert not r4.get("skipped_idle")
    assert len(load_calls) == 3
