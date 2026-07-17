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
from papervault.knowledge.ledger.store import LedgerRecord


# ---------------------------------------------------------------- fakes


class FakeLedger:
    """In-memory stand-in for papervault.knowledge.ledger.store (async surface)."""

    def __init__(self, rows: dict[str, LedgerRecord] | None = None):
        self.rows: dict[str, LedgerRecord] = dict(rows or {})

    async def upsert(self, ingest_source, source_id, *, doc_id, status, fingerprint=None):
        prev = self.rows.get(source_id)
        # mirror COALESCE: keep existing fingerprint when None is passed
        fp = fingerprint if fingerprint is not None else (prev.fingerprint if prev else None)
        self.rows[source_id] = LedgerRecord("l0_probe", ingest_source, source_id, fp, doc_id, status)

    async def delete(self, ingest_source, source_id):
        self.rows.pop(source_id, None)

    async def load(self, ingest_source):
        return {k: v for k, v in self.rows.items()}

    async def ensure_schema(self):
        pass


class FakeDocStatus:
    def __init__(self, status):
        self.status = status


class FakeRag:
    """Records enqueue/process/delete calls; returns canned aget_docs_by_ids."""

    def __init__(self, *, doc_statuses=None, enqueue_raises=False, delete_results=None):
        self.calls: list[tuple] = []
        self.doc_statuses = doc_statuses or {}      # {doc_id: DocStatus}
        self.enqueue_raises = enqueue_raises
        self.delete_results = delete_results or {}  # {doc_id: status str}

    async def apipeline_enqueue_documents(self, *, input, ids, file_paths):
        self.calls.append(("enqueue", list(ids)))
        if self.enqueue_raises:
            raise ValueError("simulated batch-level enqueue failure")

    async def apipeline_process_enqueue_documents(self):
        self.calls.append(("process", None))

    async def adelete_by_doc_id(self, doc_id):
        self.calls.append(("delete", doc_id))
        return FakeDocStatus(self.delete_results.get(doc_id, "success"))

    async def aget_docs_by_ids(self, ids):
        self.calls.append(("get_docs_by_ids", list(ids)))
        # Mirror LightRAG 1.4.16's RUNTIME shape: {doc_id: plain dict} with `status` a bare
        # string — NOT a DocProcessingStatus object (SDD §6.6 ★). Returning objects here used
        # to mask the getattr-on-dict bug in reconcile_terminal / ks get.
        out = {}
        for i in ids:
            if i not in self.doc_statuses:
                continue
            s = self.doc_statuses[i]
            out[i] = {"status": getattr(s, "value", s), "doc_id": i}  # enum → bare string value
        return out


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
    first_enqueue = order.index("enqueue")
    assert all(order[i] == "delete" or order[i] == "get_docs_by_ids" for i in range(first_enqueue))
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
