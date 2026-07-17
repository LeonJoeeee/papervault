"""Regression net for the aget_docs_by_ids return-shape contract (SDD §6.6 ★ / §6.8).

LightRAG 1.4.16's `aget_docs_by_ids` is type-hinted `dict[str, DocProcessingStatus]` but at
RUNTIME returns `{doc_id: plain dict}` — it passes `doc_status.get_by_id()` straight through
(lightrag.py:3192/3209), and BOTH configured backends return a plain dict:
  - PGDocStatusStorage.get_by_id  → `return dict(...)`  (postgres_impl.py:3818)
  - JsonDocStatusStorage.get_by_id → `return self._data.get(id)` (json_doc_status_impl.py:238)
whose `status` is a BARE STRING ('processed'/'failed'/…), not a DocProcessingStatus object.

The pre-fix code did `getattr(st, "status", None)` on that dict → always None → every PROCESSED
doc was misclassified, stuck_guard tripped, and successful docs were flipped to error (breaking
the §6.5 step5 rebuild closure). `ks get` likewise reported every field as null. These tests pin
the real dict shape so a regression to getattr-on-dict fails loudly. No LightRAG / no DB / no LLM.
"""
from __future__ import annotations

import pytest

import papervault.knowledge.scheduler.round as rnd
from papervault.knowledge.ledger.store import LedgerRecord


# ---------------------------------------------------------------- fakes


class _FakeLedger:
    def __init__(self, rows):
        self.rows = dict(rows)

    async def upsert(self, ingest_source, source_id, *, doc_id, status, fingerprint=None):
        prev = self.rows.get(source_id)
        fp = fingerprint if fingerprint is not None else (prev.fingerprint if prev else None)
        self.rows[source_id] = LedgerRecord("l0_probe", ingest_source, source_id, fp, doc_id, status)

    async def load(self, ingest_source):
        return dict(self.rows)


class _DictRag:
    """aget_docs_by_ids returns the REAL runtime shape: {doc_id: plain dict, status bare str}."""

    def __init__(self, status_by_doc_id):
        self._statuses = status_by_doc_id  # {doc_id: bare status string}

    async def aget_docs_by_ids(self, ids):
        return {
            i: {"status": self._statuses[i], "doc_id": i}
            for i in ids
            if i in self._statuses
        }


# ---------------------------------------------------------------- _doc_status unit


def test_doc_status_reads_bare_string_from_dict():
    from lightrag.base import DocStatus

    # DocStatus is a str-Enum, so the bare string the backend hands back compares equal.
    assert "processed" == DocStatus.PROCESSED
    assert rnd._doc_status({"status": "processed"}) == DocStatus.PROCESSED
    assert rnd._doc_status({"status": "failed"}) == DocStatus.FAILED
    assert rnd._doc_status({"status": "processing"}) == DocStatus.PROCESSING
    # dict with no status key, and absent row, both → None (→ stuck_guard, not a crash).
    assert rnd._doc_status({"doc_id": "x"}) is None
    assert rnd._doc_status(None) is None


def test_getattr_on_dict_is_the_bug_we_fixed():
    # The exact anti-pattern that was in round.py:63 — getattr on the dict yields None,
    # which is why PROCESSED docs were misclassified. Pinned so nobody reintroduces it.
    real_shape = {"status": "processed"}
    assert getattr(real_shape, "status", None) is None          # the latent bug
    assert rnd._doc_status(real_shape) == "processed"           # the fix


# ---------------------------------------------------------------- reconcile_terminal on dict shape


@pytest.mark.asyncio
async def test_reconcile_terminal_maps_processed_on_real_dict_shape(monkeypatch):
    rnd._stuck.clear()
    fl = _FakeLedger({
        "A": LedgerRecord("l0_probe", "paper", "A", "h", "paper:A", "processing"),
        "B": LedgerRecord("l0_probe", "paper", "B", "h", "paper:B", "processing"),
    })
    monkeypatch.setattr(rnd, "ledger", fl)
    # Backend hands back PLAIN DICTS with bare-string status (the real LightRAG shape).
    rag = _DictRag({"paper:A": "processed", "paper:B": "failed"})

    c = await rnd.reconcile_terminal(rag)

    assert fl.rows["A"].status == "done"     # PROCESSED → done (was wrongly 'processing'/error)
    assert fl.rows["B"].status == "error"    # FAILED → error
    assert c["done"] == 1 and c["error"] == 1
    rnd._stuck.clear()


@pytest.mark.asyncio
async def test_reconcile_terminal_processed_dict_never_trips_stuck_guard(monkeypatch):
    # Pre-fix, a PROCESSED dict was unrecognized → after stuck_limit rounds it flipped to error.
    # Now the very first round must settle it to done and never increment _stuck.
    rnd._stuck.clear()
    fl = _FakeLedger({"A": LedgerRecord("l0_probe", "paper", "A", "h", "paper:A", "processing")})
    monkeypatch.setattr(rnd, "ledger", fl)
    rag = _DictRag({"paper:A": "processed"})

    for _ in range(rnd.DEFAULT_STUCK_LIMIT + 1):
        await rnd.reconcile_terminal(rag)

    assert fl.rows["A"].status == "done"
    assert "A" not in rnd._stuck            # never counted as stuck
    rnd._stuck.clear()
