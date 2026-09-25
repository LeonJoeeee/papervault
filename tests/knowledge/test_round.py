"""S4 scheduler round + reconcile_terminal + distill_batch logic (SDD §6.1/§6.6).

Pure logic with a fake in-memory ledger + a fake LightRAG (records calls / returns
canned doc_status). No real DB, no real LightRAG, no network — runs offline.
Covers: F16 content-dedup, F17 batch-exception, REDISTILL_DELETE feedback (F2),
delete-then-insert ordering (§6.6), terminal writeback mapping (F9/F10) + stuck_guard.
"""
from __future__ import annotations

import pytest

import papervault.knowledge.ingest.distill as distill
import papervault.knowledge.scheduler.round as rnd
from papervault.knowledge.ingest.paper_library_client import PaperRecord
from papervault.knowledge.ledger.store import (
    LedgerRecord,
    _max_attempts,
    _next_attempts_status,
)


# ---------------------------------------------------------------- fakes


class FakeLedger:
    """In-memory stand-in for papervault.knowledge.ledger.store (async surface)."""

    def __init__(self, rows: dict[str, LedgerRecord] | None = None):
        self.rows: dict[str, LedgerRecord] = dict(rows or {})

    async def upsert(self, ingest_source, source_id, *, doc_id, status, fingerprint=None):
        prev = self.rows.get(source_id)
        prev_attempts = prev.attempts if prev else 0
        prev_fp = prev.fingerprint if prev else None
        # Drive the REAL #84 circuit-breaker transition (attempts + parking) and the §4.1
        # fingerprint-COALESCE, so these round tests exercise production logic — not a re-impl.
        eff_attempts, eff_status = _next_attempts_status(
            prev_attempts, prev_fp, status, fingerprint, _max_attempts()
        )
        fp = fingerprint if fingerprint is not None else prev_fp  # mirror COALESCE
        self.rows[source_id] = LedgerRecord(
            "l0_probe", ingest_source, source_id, fp, doc_id, eff_status, eff_attempts
        )

    async def delete(self, ingest_source, source_id):
        self.rows.pop(source_id, None)

    async def load(self, ingest_source):
        return {k: v for k, v in self.rows.items() if v.ingest_source == ingest_source}

    async def load_by_status(self, statuses):
        return [r for r in self.rows.values() if r.status in set(statuses)]

    async def ensure_schema(self):
        pass


class FakeDocStatus:
    def __init__(self, status):
        self.status = status


def _basename(file_path: str) -> str:
    # LightRAG 1.5 stores/matches the canonical BASENAME: "paper/<key>" -> "<key>".
    return file_path.rsplit("/", 1)[-1]


class FakeDocStatusStore:
    """The slice of LightRAG's doc_status storage the insert guard touches.

    Rows are {doc_id: {"status": <bare str>, "file_path": str, "metadata": dict}} in insertion
    (== created_at) order, so `get_doc_by_file_basename` returns the OLDEST exact match — the
    PG backend's `ORDER BY created_at ASC, id ASC LIMIT 1` (postgres_impl.py)."""

    def __init__(self, rag: "FakeRag"):
        self._rag = rag

    async def get_doc_by_file_basename(self, basename):
        self._rag.calls.append(("get_doc_by_file_basename", basename))
        for did, row in self._rag.rows.items():
            if row["file_path"] == basename:
                return did, dict(row)
        return None

    async def delete(self, ids):
        self._rag.calls.append(("doc_status_delete", list(ids)))
        for did in ids:
            self._rag.rows.pop(did, None)


class FakeFullDocs:
    def __init__(self, rag: "FakeRag"):
        self._rag = rag

    async def get_by_id(self, did):
        return {"content": "..."} if did in self._rag.full_docs_ids else None


class FakeRag:
    """A doc_status-backed fake LightRAG that records calls.

    `doc_statuses` seeds real docs ({doc_id: DocStatus}); `markers` seeds extra doc_status rows
    (e.g. `dup-*` duplicate-rejection markers) verbatim. Enqueue mirrors LightRAG 1.5's two
    enqueue-time dedups (pipeline.py): an id doc_status already holds, or a canonical file_path
    basename any row already carries, is REJECTED and leaves a new FAILED `dup-*` marker
    (recorded in `created_markers`) instead of a doc — the exact mechanism behind #131."""

    def __init__(self, *, doc_statuses=None, markers=None, full_docs_ids=None,
                 enqueue_raises=False, delete_results=None):
        self.calls: list[tuple] = []
        self.rows: dict[str, dict] = {}
        for did, st in (doc_statuses or {}).items():
            self.rows[did] = {"status": getattr(st, "value", st),
                              "file_path": _basename(did.split(":", 1)[-1]), "metadata": {}}
        for did, row in (markers or {}).items():
            self.rows[did] = dict(row)
        self.full_docs_ids = set(full_docs_ids or ())
        self.enqueue_raises = enqueue_raises
        self.delete_results = delete_results or {}  # {doc_id: status str}
        self.created_markers: list[str] = []
        self.doc_status = FakeDocStatusStore(self)
        self.full_docs = FakeFullDocs(self)

    def set_status(self, did, status):
        """Test hook: the LightRAG pipeline moved `did` to `status` (e.g. a build finished)."""
        self.rows.setdefault(did, {"file_path": _basename(did.split(":", 1)[-1]), "metadata": {}})
        self.rows[did]["status"] = getattr(status, "value", status)

    async def apipeline_enqueue_documents(self, *, input, ids, file_paths):
        self.calls.append(("enqueue", list(ids)))
        if self.enqueue_raises:
            raise ValueError("simulated batch-level enqueue failure")
        for did, fp in zip(ids, file_paths):
            base = _basename(fp)
            if did in self.rows or any(r["file_path"] == base for r in self.rows.values()):
                marker = f"dup-{len(self.created_markers)}-{did}"
                self.rows[marker] = {"status": "failed", "file_path": base,
                                     "metadata": {"is_duplicate": True}}
                self.created_markers.append(marker)
                continue
            self.rows[did] = {"status": "pending", "file_path": base, "metadata": {}}
            self.full_docs_ids.add(did)

    async def apipeline_process_enqueue_documents(self):
        self.calls.append(("process", None))

    async def adelete_by_doc_id(self, doc_id):
        self.calls.append(("delete", doc_id))
        status = self.delete_results.get(doc_id, "success")
        if status == "success":
            self.rows.pop(doc_id, None)
            self.full_docs_ids.discard(doc_id)
        return FakeDocStatus(status)

    async def aget_docs_by_ids(self, ids):
        self.calls.append(("get_docs_by_ids", list(ids)))
        # Mirror LightRAG's RUNTIME shape: {doc_id: plain dict} with `status` a bare
        # string — NOT a DocProcessingStatus object (SDD §6.6 ★). Returning objects here used
        # to mask the getattr-on-dict bug in reconcile_terminal / ks get.
        return {i: {"status": self.rows[i]["status"], "doc_id": i} for i in ids if i in self.rows}


@pytest.fixture
def fake_ledger(monkeypatch):
    fl = FakeLedger()
    monkeypatch.setattr(distill, "ledger", fl)
    monkeypatch.setattr(rnd, "ledger", fl)
    return fl


def _rec(key: str) -> PaperRecord:
    return PaperRecord(key=key, md_path=f"extracts/md/{key}.md")


# ---------------------------------------------------------------- distill_batch


@pytest.mark.asyncio
async def test_distill_batch_content_dedup_F16(fake_ledger, monkeypatch):
    # two papers with identical body → only the first is enqueued; the second is
    # marked done_meta(dup), NOT left at processing (which would stick forever).
    texts = {"A": "Identical body text.", "B": "Identical body text.", "C": "Different."}
    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: texts[rec.key])

    rag = FakeRag()
    items = [(_rec("A"), "fpA"), (_rec("B"), "fpB"), (_rec("C"), "fpC")]
    counters = await distill.distill_batch(rag, items)

    # exactly one enqueue call carrying A and C (B deduped out)
    enqueued = [ids for kind, ids in rag.calls if kind == "enqueue"][0]
    assert "paper:A" in enqueued and "paper:C" in enqueued and "paper:B" not in enqueued
    assert counters["queued"] == 2 and counters["dup"] == 1
    assert fake_ledger.rows["A"].status == "processing"
    assert fake_ledger.rows["C"].status == "processing"
    assert fake_ledger.rows["B"].status == "done_meta"   # dup → terminal, not stuck
    # B carries its REAL fingerprint (not META) so next round's diff sees fpB == fpB and
    # does NOT re-classify it to_redistill → no per-round churn (SDD §6.1 line 221).
    assert fake_ledger.rows["B"].fingerprint == "fpB"


@pytest.mark.asyncio
async def test_distill_batch_batch_exception_writes_error_F17(fake_ledger, monkeypatch):
    texts = {"A": "body A", "B": "body B"}
    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: texts[rec.key])

    rag = FakeRag(enqueue_raises=True)
    counters = await distill.distill_batch(rag, [(_rec("A"), "fpA"), (_rec("B"), "fpB")])

    # batch-level raise must NOT escape; queued keys flip processing → error (fp preserved)
    assert counters["errored"] == 2 and counters["queued"] == 0
    assert fake_ledger.rows["A"].status == "error" and fake_ledger.rows["A"].fingerprint == "fpA"
    assert fake_ledger.rows["B"].status == "error" and fake_ledger.rows["B"].fingerprint == "fpB"


@pytest.mark.asyncio
async def test_distill_batch_meta_and_empty(fake_ledger, monkeypatch):
    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: None)  # no text
    rag = FakeRag()
    counters = await distill.distill_batch(
        rag, [(_rec("M"), distill.META), (_rec("E"), "fpE")]
    )
    assert counters["meta"] == 1 and counters["no_text"] == 1
    assert fake_ledger.rows["M"].status == "done_meta"
    assert fake_ledger.rows["E"].status == "done_meta"
    assert not any(k == "enqueue" for k, _ in rag.calls)  # nothing to enqueue


# ---------------------------------------------------------------- reconcile_terminal


@pytest.mark.asyncio
async def test_reconcile_terminal_maps_processed_failed(fake_ledger, monkeypatch):
    from lightrag.base import DocStatus

    fake_ledger.rows = {
        "A": LedgerRecord("l0_probe", "paper", "A", "h", "paper:A", "processing"),
        "B": LedgerRecord("l0_probe", "paper", "B", "h", "paper:B", "processing"),
        "C": LedgerRecord("l0_probe", "paper", "C", "h", "paper:C", "processing"),
        "D": LedgerRecord("l0_probe", "paper", "D", "h", "paper:D", "done"),  # not processing → ignored
    }
    rag = FakeRag(doc_statuses={
        "paper:A": DocStatus.PROCESSED,
        "paper:B": DocStatus.FAILED,
        "paper:C": DocStatus.PROCESSING,  # 中途态 → 留下轮
    })
    c = await rnd.reconcile_terminal(rag)
    assert fake_ledger.rows["A"].status == "done"
    assert fake_ledger.rows["B"].status == "error"
    assert fake_ledger.rows["C"].status == "processing"  # unchanged
    assert c["done"] == 1 and c["error"] == 1 and c["pending"] == 1


@pytest.mark.asyncio
async def test_reconcile_terminal_stuck_guard(fake_ledger):
    rnd._stuck.clear()
    fake_ledger.rows = {"X": LedgerRecord("l0_probe", "paper", "X", "h", "paper:X", "processing")}
    rag = FakeRag(doc_statuses={})  # X never appears in doc_status (F16/enqueue-drop orphan)

    # rounds 1..N-1: still processing, just counting
    for _ in range(rnd.DEFAULT_STUCK_LIMIT - 1):
        await rnd.reconcile_terminal(rag)
        assert fake_ledger.rows["X"].status == "processing"
    # round N: flips to error
    c = await rnd.reconcile_terminal(rag)
    assert fake_ledger.rows["X"].status == "error"
    assert c["stuck_error"] == 1
    rnd._stuck.clear()


# ---------------------------------------------------------------- run_round


@pytest.mark.asyncio
async def test_run_round_delete_before_insert_and_redistill_feedback(fake_ledger, monkeypatch):
    # index: A (new), B (changed → redistill, delete succeeds), C (changed → redistill,
    # delete FAILS → must NOT be enqueued, F2). ledger has B,C,GONE.
    idx = {"A": _rec("A"), "B": _rec("B"), "C": _rec("C")}
    monkeypatch.setattr(rnd, "load_clean_index", lambda *a, **k: idx)
    monkeypatch.setattr(rnd, "fingerprint", lambda rec: {"A": "fA", "B": "fB_new", "C": "fC_new"}[rec.key])
    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: f"body of {rec.key}")

    fake_ledger.rows = {
        "B": LedgerRecord("l0_probe", "paper", "B", "fB_old", "paper:B", "done"),
        "C": LedgerRecord("l0_probe", "paper", "C", "fC_old", "paper:C", "done"),
        "GONE": LedgerRecord("l0_probe", "paper", "GONE", "fG", "paper:GONE", "done"),
    }
    rag = FakeRag(delete_results={"paper:C": "not_allowed"})  # C delete 403 → skip enqueue

    summary = await rnd.run_round(rag)

    order = [k for k, _ in rag.calls]
    # all deletes happen before any enqueue (§6.6 delete-then-insert mutex)
    # (the #131 insert guard's read-only doc_status lookups may interleave; no delete may follow)
    first_enqueue = order.index("enqueue")
    assert "delete" not in order[first_enqueue:]
    assert set(order[:first_enqueue]) <= {"delete", "get_docs_by_ids", "get_doc_by_file_basename"}
    # enqueue carries A (new) + B (redistill deleted ok), NOT C (delete failed, F2)
    enqueued = [ids for kind, ids in rag.calls if kind == "enqueue"][0]
    assert set(enqueued) == {"paper:A", "paper:B"}
    assert "paper:C" not in enqueued
    # GONE removed from ledger; C stays (kept its old fp for retry next round)
    assert "GONE" not in fake_ledger.rows
    assert summary["to_remove"] == 1 and summary["redistill_removed"] == 1
    # process always called at the tail (heals orphans even if batch empty)
    assert ("process", None) in rag.calls


@pytest.mark.asyncio
async def test_run_round_tail_process_when_nothing_to_do(fake_ledger, monkeypatch):
    idx = {"A": _rec("A")}
    monkeypatch.setattr(rnd, "load_clean_index", lambda *a, **k: idx)
    monkeypatch.setattr(rnd, "fingerprint", lambda rec: "fA")
    fake_ledger.rows = {"A": LedgerRecord("l0_probe", "paper", "A", "fA", "paper:A", "done")}  # in sync
    rag = FakeRag()

    summary = await rnd.run_round(rag)
    assert summary["to_distill"] == 0 and summary["to_redistill"] == 0 and summary["to_remove"] == 0
    assert not any(k == "enqueue" for k, _ in rag.calls)   # no batch
    assert ("process", None) in rag.calls                  # but tail process still runs


# ---------------------------------------------------------------- #84 circuit-breaker


def test_attempts_helper_increments_and_parks_at_threshold():
    # PURE (DB-free) test of the real transition helper store._next_attempts_status.
    assert _next_attempts_status(0, "fp", "error", "fp", 3) == (1, "error")
    assert _next_attempts_status(1, "fp", "error", "fp", 3) == (2, "error")
    assert _next_attempts_status(2, "fp", "error", "fp", 3) == (3, "error_parked")  # threshold → park
    assert _next_attempts_status(3, "fp", "error", "fp", 3) == (4, "error_parked")  # stays parked


def test_attempts_helper_success_resets_streak():
    assert _next_attempts_status(2, "fp", "done", "fp", 3) == (0, "done")
    assert _next_attempts_status(5, "fp", "done_meta", "META", 3) == (0, "done_meta")


def test_attempts_helper_fingerprint_change_resets_but_same_fp_preserves():
    # a genuinely NEW fingerprint (content changed) resets → revives a parked key with a fresh budget
    assert _next_attempts_status(3, "old", "processing", "new", 3) == (0, "processing")
    # same fingerprint on a non-terminal write PRESERVES the streak (spans the redistill 'processing')
    assert _next_attempts_status(2, "fp", "processing", "fp", 3) == (2, "processing")
    # status-only transition (fingerprint None) also preserves the count
    assert _next_attempts_status(2, "fp", "pending_remove", None, 3) == (2, "pending_remove")


def test_attempts_helper_disabled_when_threshold_non_positive():
    # KS_REDISTILL_MAX_ATTEMPTS<=0 disables parking: an error keeps counting but never parks (pre-#84)
    assert _next_attempts_status(99, "fp", "error", "fp", 0) == (100, "error")


@pytest.mark.asyncio
async def test_run_round_parks_key_after_max_consecutive_build_failures(fake_ledger, monkeypatch):
    # THE headline #84 property: a paper whose build fails every round (systemic break) is PARKED
    # after KS_REDISTILL_MAX_ATTEMPTS consecutive failures, and the round then STOPS re-distilling it
    # — no more delete-then-reinsert graph churn. We set the threshold to 2 to prove it is env-driven
    # (not hardcoded) and keep the loop short.
    monkeypatch.setenv("KS_REDISTILL_MAX_ATTEMPTS", "2")
    rnd._stuck.clear()
    idx = {"A": _rec("A")}
    monkeypatch.setattr(rnd, "load_clean_index", lambda *a, **k: idx)
    monkeypatch.setattr(rnd, "fingerprint", lambda rec: "fA")          # content NEVER changes
    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: "body of A")

    rag = FakeRag(enqueue_raises=True)  # every distill_batch F17-errors the queued key

    # Round 1: A is new → to_distill → enqueue raises → error (attempts 1, < 2, NOT parked yet)
    s1 = await rnd.run_round(rag)
    assert s1["to_distill"] == 1
    assert fake_ledger.rows["A"].status == "error" and fake_ledger.rows["A"].attempts == 1

    # Round 2: error → to_redistill → delete ok → enqueue raises → error (attempts 2 == threshold → PARK)
    s2 = await rnd.run_round(rag)
    assert s2["to_redistill"] == 1
    assert fake_ledger.rows["A"].status == "error_parked"
    assert fake_ledger.rows["A"].attempts == 2

    # Round 3: parked → diff drops it → the round does NOTHING to A (no delete, no enqueue = no churn)
    calls_before = len(rag.calls)
    s3 = await rnd.run_round(rag)
    assert s3["to_redistill"] == 0 and s3["to_distill"] == 0 and s3["redistill_removed"] == 0
    new_calls = rag.calls[calls_before:]
    assert not any(kind == "enqueue" for kind, _ in new_calls)  # never re-enqueued
    assert not any(kind == "delete" for kind, _ in new_calls)   # never re-deleted (graph churn stopped)
    assert fake_ledger.rows["A"].status == "error_parked"       # stays parked


@pytest.mark.asyncio
async def test_run_round_successful_build_resets_the_streak(fake_ledger, monkeypatch):
    # #84: a successful build (reconcile PROCESSED → done) clears the consecutive-failure streak, so a
    # later transient failure does NOT immediately re-park a key that was one step from the threshold.
    from lightrag.base import DocStatus

    monkeypatch.setenv("KS_REDISTILL_MAX_ATTEMPTS", "3")
    rnd._stuck.clear()
    idx = {"A": _rec("A")}
    monkeypatch.setattr(rnd, "load_clean_index", lambda *a, **k: idx)
    monkeypatch.setattr(rnd, "fingerprint", lambda rec: "fA")
    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: "body of A")

    # A starts one failure away from parking (attempts=2, status=error).
    fake_ledger.rows = {
        "A": LedgerRecord("l0_probe", "paper", "A", "fA", "paper:A", "error", 2),
    }
    # The failed build left no processed doc (a processed doc would be healed straight to done, #131).
    rag = FakeRag()

    # Round 1: error → redistill → delete ok → enqueue OK → status=processing (streak preserved at 2).
    await rnd.run_round(rag)
    assert fake_ledger.rows["A"].status == "processing" and fake_ledger.rows["A"].attempts == 2

    # The LightRAG pipeline finishes this build between rounds.
    rag.set_status("paper:A", DocStatus.PROCESSED)

    # Round 2: reconcile reads PROCESSED → done → streak RESET to 0.
    await rnd.run_round(rag)
    assert fake_ledger.rows["A"].status == "done" and fake_ledger.rows["A"].attempts == 0

    # A subsequent single failure lands at attempts=1 (NOT parked) — proving the streak really reset.
    await fake_ledger.upsert("paper", "A", doc_id="paper:A", status="error", fingerprint="fA")
    assert fake_ledger.rows["A"].status == "error" and fake_ledger.rows["A"].attempts == 1


# ---------------------------------------------------------------- #131 ledger ↔ graph drift


def _marker(file_path: str, original: str) -> dict:
    """A LightRAG enqueue-time duplicate-rejection row (pipeline.py): FAILED, no full_docs content."""
    return {
        "status": "failed",
        "file_path": file_path,
        "metadata": {"is_duplicate": True, "duplicate_kind": "filename", "original_doc_id": original},
    }


@pytest.mark.asyncio
async def test_reconcile_healed_marks_error_and_parked_rows_done_when_doc_processed(fake_ledger):
    # LightRAG's own retry heals FAILED-with-content docs, but the ledger parked them first
    # (1,831 such rows in #131). A row whose doc is PROCESSED is done; the rest stay put.
    from lightrag.base import DocStatus

    fake_ledger.rows = {
        "A": LedgerRecord("l0_probe", "paper", "A", "fA", "paper:A", "error_parked", 3),
        "B": LedgerRecord("l0_probe", "paper", "B", "fB", "paper:B", "error", 1),
        "C": LedgerRecord("l0_probe", "paper", "C", "fC", "paper:C", "error_parked", 3),
        "D": LedgerRecord("l0_probe", "paper", "D", "fD", "paper:D", "error_parked", 3),
        "E": LedgerRecord("l0_probe", "paper", "E", "fE", "paper:E", "done"),
    }
    rag = FakeRag(doc_statuses={
        "paper:A": DocStatus.PROCESSED,
        "paper:B": DocStatus.PROCESSED,
        "paper:C": DocStatus.FAILED,       # still failed → stays parked
        "paper:E": DocStatus.PROCESSED,
    })                                     # paper:D has no doc at all → stays parked

    c = await rnd.reconcile_healed(rag)

    assert (fake_ledger.rows["A"].status, fake_ledger.rows["A"].attempts) == ("done", 0)
    assert (fake_ledger.rows["B"].status, fake_ledger.rows["B"].attempts) == ("done", 0)
    assert fake_ledger.rows["A"].fingerprint == "fA"  # the built content's fp is kept
    assert fake_ledger.rows["C"].status == "error_parked"
    assert fake_ledger.rows["D"].status == "error_parked"
    assert c == {"healed": 2}


@pytest.mark.asyncio
async def test_reconcile_healed_covers_operator_doc_rows(fake_ledger):
    # 13 textbook rows sat at `error` since 07-24 although 10 of their sections are processed.
    from lightrag.base import DocStatus

    fake_ledger.rows = {
        "Kaipio2005#s75": LedgerRecord(
            "l0_probe", "textbook", "Kaipio2005#s75", None, "textbook:Kaipio2005#s75", "error", 1),
        "Zienkiewicz2013#s340": LedgerRecord(
            "l0_probe", "textbook", "Zienkiewicz2013#s340", None,
            "textbook:Zienkiewicz2013#s340", "error", 1),
    }
    rag = FakeRag(doc_statuses={"textbook:Kaipio2005#s75": DocStatus.PROCESSED})

    await rnd.reconcile_healed(rag)

    assert fake_ledger.rows["Kaipio2005#s75"].status == "done"
    assert fake_ledger.rows["Kaipio2005#s75"].ingest_source == "textbook"
    assert fake_ledger.rows["Zienkiewicz2013#s340"].status == "error"


@pytest.mark.asyncio
async def test_reconcile_healed_respects_subset(fake_ledger):
    from lightrag.base import DocStatus

    fake_ledger.rows = {
        "A": LedgerRecord("l0_probe", "paper", "A", "fA", "paper:A", "error_parked", 3),
        "B": LedgerRecord("l0_probe", "paper", "B", "fB", "paper:B", "error_parked", 3),
        "T#s1": LedgerRecord("l0_probe", "textbook", "T#s1", None, "textbook:T#s1", "error", 1),
    }
    rag = FakeRag(doc_statuses={
        "paper:A": DocStatus.PROCESSED, "paper:B": DocStatus.PROCESSED,
        "textbook:T#s1": DocStatus.PROCESSED,
    })

    await rnd.reconcile_healed(rag, only_keys={"A"})

    assert fake_ledger.rows["A"].status == "done"
    assert fake_ledger.rows["B"].status == "error_parked"   # outside the subset: untouched
    assert fake_ledger.rows["T#s1"].status == "error"       # a subset round touches paper keys only


@pytest.mark.asyncio
async def test_run_round_heals_error_row_instead_of_redistilling_processed_doc(fake_ledger, monkeypatch):
    # Before #131 an `error` row was re-distilled unconditionally: its PROCESSED doc was deleted
    # from the graph and rebuilt. A processed doc is the truth — the row becomes done, no churn.
    from lightrag.base import DocStatus

    rnd._stuck.clear()
    idx = {"A": _rec("A")}
    monkeypatch.setattr(rnd, "load_clean_index", lambda *a, **k: idx)
    monkeypatch.setattr(rnd, "fingerprint", lambda rec: "fA")
    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: "body of A")
    fake_ledger.rows = {"A": LedgerRecord("l0_probe", "paper", "A", "fA", "paper:A", "error", 1)}
    rag = FakeRag(doc_statuses={"paper:A": DocStatus.PROCESSED})

    summary = await rnd.run_round(rag)

    assert fake_ledger.rows["A"].status == "done"
    assert summary["healed"] == {"healed": 1} and summary["to_redistill"] == 0
    assert not any(kind in ("delete", "enqueue") for kind, _ in rag.calls)
    assert rag.rows["paper:A"]["status"] == "processed"


@pytest.mark.asyncio
async def test_run_round_adopts_processed_doc_when_ledger_row_missing(fake_ledger, monkeypatch):
    # The #131 caution: a ledger row deleted while its doc exists made the next round re-insert the
    # doc, which LightRAG rejected as a duplicate and recorded as a new `dup-*` marker. The round
    # must adopt the processed doc instead of re-inserting it.
    from lightrag.base import DocStatus

    rnd._stuck.clear()
    idx = {"A": _rec("A")}
    monkeypatch.setattr(rnd, "load_clean_index", lambda *a, **k: idx)
    monkeypatch.setattr(rnd, "fingerprint", lambda rec: "fA")
    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: "body of A")
    rag = FakeRag(doc_statuses={"paper:A": DocStatus.PROCESSED})   # ledger is empty

    await rnd.run_round(rag)

    assert fake_ledger.rows["A"].status == "done" and fake_ledger.rows["A"].fingerprint == "fA"
    assert not any(kind == "enqueue" and "paper:A" in ids for kind, ids in rag.calls)
    assert rag.created_markers == []


@pytest.mark.asyncio
async def test_distill_batch_tracks_in_flight_existing_doc_instead_of_enqueueing(fake_ledger, monkeypatch):
    # An existing doc that LightRAG is still building (PENDING) is tracked as processing, so
    # reconcile_terminal records its outcome; enqueueing it again would only mint a `dup-*` marker.
    from lightrag.base import DocStatus

    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: "body of A")
    rag = FakeRag(doc_statuses={"paper:A": DocStatus.PENDING})

    counters = await distill.distill_batch(rag, [(_rec("A"), "fA")])

    assert fake_ledger.rows["A"].status == "processing" and fake_ledger.rows["A"].fingerprint == "fA"
    assert not any(kind == "enqueue" for kind, _ in rag.calls)
    assert rag.created_markers == []
    assert counters["existing"] == 1 and counters["queued"] == 0


@pytest.mark.asyncio
async def test_distill_batch_purges_stale_dup_markers_that_block_reinsert(fake_ledger, monkeypatch):
    # The 12 blocked papers of #131: FAILED `dup-*` rows without content share the paper's
    # file_path, and LightRAG's filename dedup matches them regardless of status — so every
    # re-insert was rejected and left yet another marker. The insert path clears them first.
    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: f"body of {rec.key}")
    rag = FakeRag(markers={
        "dup-old": _marker("Aslam2020a", "paper:Aslam2020a"),
        "dup-chain": _marker("Aslam2020a", "dup-old"),
        "dup-other": _marker("McDonald2007", "paper:McDonald2007"),  # another paper's marker
    })

    counters = await distill.distill_batch(rag, [(_rec("Aslam2020a"), "fp")])

    assert "dup-old" not in rag.rows and "dup-chain" not in rag.rows
    assert "dup-other" in rag.rows                       # only markers on THIS file_path go
    assert rag.created_markers == []                      # the insert was accepted, not rejected
    assert rag.rows["paper:Aslam2020a"]["status"] == "pending"
    assert fake_ledger.rows["Aslam2020a"].status == "processing"
    assert counters["queued"] == 1 and counters["purged_markers"] == 2


@pytest.mark.asyncio
async def test_distill_batch_never_purges_a_live_doc_on_the_same_file_path(fake_ledger, monkeypatch):
    # Only a stale marker (dup- id, FAILED, no content) may be deleted. A live doc or a failed doc
    # that still holds content on the same file_path is kept, and the paper is not enqueued over it.
    monkeypatch.setattr(distill, "read_extract_raw", lambda rec, *a, **k: "body of A")
    rag = FakeRag(
        markers={
            "legacy:A": {"status": "processed", "file_path": "A", "metadata": {}},
            "dup-with-content": _marker("B", "paper:B"),
        },
        full_docs_ids={"dup-with-content"},
    )

    counters = await distill.distill_batch(rag, [(_rec("A"), "fA"), (_rec("B"), "fB")])

    assert "legacy:A" in rag.rows and "dup-with-content" in rag.rows
    assert not any(kind == "enqueue" for kind, _ in rag.calls)
    assert rag.created_markers == []
    assert fake_ledger.rows["A"].status == "error" and fake_ledger.rows["B"].status == "error"
    assert counters["blocked"] == 2
