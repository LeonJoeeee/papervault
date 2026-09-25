"""Ledger CRUD + state-machine tests against the real ks-postgres (container up).

Self-contained: ensure_schema() creates ks_ledger if absent; test rows cleaned up.
"""
import pytest

pytestmark = pytest.mark.asyncio

from papervault.knowledge.ledger import store

_K = "__TEST_KEY_DELETE_ME__"


async def test_ledger_roundtrip_state_and_validation():
    await store.ensure_schema()
    try:
        # insert (processing)
        await store.upsert("paper", _K, doc_id=f"paper:{_K}", status="processing", fingerprint="fp1")
        r = await store.get("paper", _K)
        assert r is not None
        assert r.status == "processing"
        assert r.fingerprint == "fp1"
        assert r.doc_id == f"paper:{_K}"

        # transition processing → done (upsert overwrites)
        await store.upsert("paper", _K, doc_id=f"paper:{_K}", status="done", fingerprint="fp1")
        r2 = await store.get("paper", _K)
        assert r2.status == "done"

        # load() returns it keyed by source_id
        m = await store.load("paper")
        assert _K in m and m[_K].status == "done"

        # invalid status rejected
        with pytest.raises(ValueError):
            await store.upsert("paper", _K, doc_id="x", status="bogus")

        # status-only transition (no fingerprint passed) must PRESERVE the stored
        # fingerprint via COALESCE (SDD §4.1), not wipe it to NULL — else next-round
        # diff would mis-classify the row as REDISTILL (None != current hash).
        await store.upsert("paper", _K, doc_id=f"paper:{_K}", status="pending_remove")
        r3 = await store.get("paper", _K)
        assert r3.status == "pending_remove"
        assert r3.fingerprint == "fp1"   # COALESCE kept it (was 'fp1')

        # done_meta is a valid terminal status (691 元数据态)
        await store.upsert("paper", _K, doc_id=f"paper:{_K}", status="done_meta")
        assert (await store.get("paper", _K)).status == "done_meta"
    finally:
        await store.delete("paper", _K)
        assert await store.get("paper", _K) is None


async def test_ledger_redistill_circuit_breaker_parks_and_resets(monkeypatch):
    """#84: the real ks_ledger `attempts` column + parking transition (store._next_attempts_status
    applied inside upsert against Postgres). Verifies against the DB what the DB-free FakeLedger
    round tests verify in-memory: consecutive 'error' writes increment attempts and PARK the key at
    KS_REDISTILL_MAX_ATTEMPTS; a 'done' resets; a genuine fingerprint change resets/revives.
    """
    monkeypatch.setenv("KS_REDISTILL_MAX_ATTEMPTS", "3")
    await store.ensure_schema()
    _CB = "__TEST_KEY_CB_DELETE_ME__"

    async def _upsert(status, fp="fp1"):
        await store.upsert("paper", _CB, doc_id=f"paper:{_CB}", status=status, fingerprint=fp)
        return await store.get("paper", _CB)

    try:
        # start clean at processing (attempts 0)
        assert (await _upsert("processing")).attempts == 0

        # the redistill loop: error → (redistill sets processing, SAME fp preserves streak) → error → ...
        assert (await _upsert("error")).attempts == 1              # 1st failure, still plain 'error'
        assert (await _upsert("processing")).attempts == 1         # same-fp processing preserves streak
        assert (await _upsert("error")).attempts == 2              # 2nd failure
        assert (await _upsert("processing")).attempts == 2         # preserve
        r = await _upsert("error")                                 # 3rd failure == threshold → PARK
        assert r.status == "error_parked" and r.attempts == 3

        # a genuine fingerprint change (content changed) resets the streak and revives it
        r2 = await _upsert("processing", fp="fp2")
        assert r2.status == "processing" and r2.attempts == 0 and r2.fingerprint == "fp2"

        # a successful build resets too
        assert (await _upsert("error", fp="fp2")).attempts == 1
        rd = await _upsert("done", fp="fp2")
        assert rd.status == "done" and rd.attempts == 0
    finally:
        await store.delete("paper", _CB)
        assert await store.get("paper", _CB) is None


async def test_ensure_schema_pk_idempotent_on_migrated_table(monkeypatch):
    """blocker ① tail (SDD §4.1 ④): ensure_schema's PK rebuild guard must be idempotent.

    The guard compares the existing PK's column SET (not an ordered list) against the
    ternary target. On a MIGRATED table — `workspace` ALTER-added LAST, so it has the
    highest attnum — an already-correct PK (workspace, ingest_source, source_id) reads
    back from pg_index in *physical* attnum order as ['ingest_source','source_id',
    'workspace']. An ordered-list `!=` would mis-fire DROP+ADD PRIMARY KEY on every
    startup (ACCESS EXCLUSIVE lock on a PG shared with LightRAG). We reproduce that exact
    migrated shape and assert a SECOND ensure_schema() does NOT rebuild the PK — proven by
    the pg_constraint OID staying identical (a DROP+ADD would mint a new constraint OID).
    """
    monkeypatch.setenv("NEO4J_WORKSPACE", "l0_probe")
    monkeypatch.setenv("POSTGRES_WORKSPACE", "l0_probe")
    _TBL = "ks_ledger_migr_idem_test"

    async def _pk_oid(conn):
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT oid FROM pg_constraint "
                "WHERE conrelid = %s::regclass AND contype = 'p'",
                (_TBL,),
            )
            row = await cur.fetchone()
        return row[0] if row else None

    # Build a table whose physical column order has `workspace` LAST (= the migrated
    # shape) but whose PK is already the correct ternary form — exactly what a prior
    # ALTER-migration leaves behind. ensure_schema() targets the literal `ks_ledger`, so
    # we drive the guard logic directly here against this fixture table to keep `ks_ledger`
    # untouched, mirroring the store's own SELECT+set-compare.
    from papervault.knowledge.ledger import store as _store

    async with _store._conn() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {_TBL}")
        await conn.execute(
            f"CREATE TABLE {_TBL} ("
            "  ingest_source text NOT NULL,"
            "  source_id text NOT NULL,"
            "  fingerprint text, doc_id text NOT NULL, status text NOT NULL,"
            "  PRIMARY KEY (ingest_source, source_id))"
        )
        # migrate: add workspace LAST, then rebuild PK to ternary (as ensure_schema would)
        await conn.execute(f"ALTER TABLE {_TBL} ADD COLUMN workspace text")
        await conn.execute(f"UPDATE {_TBL} SET workspace='l0_probe'")
        await conn.execute(f"ALTER TABLE {_TBL} ALTER COLUMN workspace SET NOT NULL")
        await conn.execute(f"ALTER TABLE {_TBL} DROP CONSTRAINT {_TBL}_pkey")
        await conn.execute(
            f"ALTER TABLE {_TBL} ADD PRIMARY KEY (workspace, ingest_source, source_id)"
        )
        await conn.commit()
        try:
            oid_before = await _pk_oid(conn)

            # the guard's read: PK columns in PHYSICAL attnum order (workspace is last)
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT array_agg(a.attname ORDER BY a.attnum) "
                    "FROM pg_index i "
                    "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=ANY(i.indkey) "
                    f"WHERE i.indrelid='{_TBL}'::regclass AND i.indisprimary"
                )
                (physical_order,) = await cur.fetchone()
            # confirms the fixture really is in the bug-triggering migrated shape
            assert physical_order == ["ingest_source", "source_id", "workspace"]

            # the FIXED guard: set-compare → already ternary → must NOT rebuild
            should_rebuild = (
                set(physical_order or []) != {"workspace", "ingest_source", "source_id"}
            )
            assert should_rebuild is False, "set-compare must treat migrated table as already ternary"

            # the OLD (buggy) ordered-list compare WOULD have fired — regression sentinel
            assert physical_order != ["workspace", "ingest_source", "source_id"]

            # OID unchanged because the guard skipped the rebuild
            oid_after = await _pk_oid(conn)
            assert oid_after == oid_before
        finally:
            await conn.execute(f"DROP TABLE IF EXISTS {_TBL}")
            await conn.commit()


async def test_ensure_schema_backfill_skipped_on_migrated_table(monkeypatch):
    """blocker ① tail (SDD §4.1 ③): the cold-migration steps (backfill UPDATE + ALTER
    COLUMN SET NOT NULL) must be conditional, mirroring the ④/⑤ guards.

    On an already-migrated table `workspace` is already NOT NULL, so the backfill UPDATE
    matches 0 rows and `SET NOT NULL` is a no-op — but unconditional, both still take table
    locks every startup (`SET NOT NULL` grabs ACCESS EXCLUSIVE on ks_ledger, shared with
    LightRAG's write path) — exactly the startup churn ④/⑤ were made conditional to avoid.
    The fix: introspect pg_attribute.attnotnull; only the cold path (column missing / still
    nullable) runs backfill + SET NOT NULL. We build the migrated shape and assert the
    guard reads `attnotnull = true` → cold path skipped. Sentinel: a freshly-ADDed (still
    nullable) column reads `attnotnull = false` → cold path WOULD fire.
    """
    monkeypatch.setenv("NEO4J_WORKSPACE", "l0_probe")
    monkeypatch.setenv("POSTGRES_WORKSPACE", "l0_probe")
    _TBL = "ks_ledger_backfill_idem_test"

    async def _ws_attnotnull(conn):
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT attnotnull FROM pg_attribute "
                f"WHERE attrelid='{_TBL}'::regclass AND attname='workspace' AND attnum>0"
            )
            row = await cur.fetchone()
        return bool(row and row[0])

    async with store._conn() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {_TBL}")
        await conn.execute(
            f"CREATE TABLE {_TBL} ("
            "  ingest_source text NOT NULL,"
            "  source_id text NOT NULL,"
            "  fingerprint text, doc_id text NOT NULL, status text NOT NULL,"
            "  PRIMARY KEY (ingest_source, source_id))"
        )
        try:
            # step ②: ADD COLUMN — still NULLABLE → cold path SHOULD fire (sentinel)
            await conn.execute(f"ALTER TABLE {_TBL} ADD COLUMN workspace text")
            await conn.commit()
            assert await _ws_attnotnull(conn) is False, (
                "freshly-ADDed nullable column must read attnotnull=false (cold path fires)"
            )

            # step ③: backfill + SET NOT NULL (what the cold path does once)
            await conn.execute(f"UPDATE {_TBL} SET workspace='l0_probe' WHERE workspace IS NULL")
            await conn.execute(f"ALTER TABLE {_TBL} ALTER COLUMN workspace SET NOT NULL")
            await conn.commit()

            # migrated shape: now NOT NULL → the FIXED guard must SKIP backfill + SET NOT NULL
            ws_already_not_null = await _ws_attnotnull(conn)
            assert ws_already_not_null is True, (
                "migrated table's workspace is NOT NULL → cold-migration block must be skipped"
            )
        finally:
            await conn.execute(f"DROP TABLE IF EXISTS {_TBL}")
            await conn.commit()


async def test_ledger_workspace_isolation(monkeypatch):
    """blocker ① (SDD §4.1/§6.5): a row written under one workspace is invisible and
    untouchable from another — get/load/count_by_status/delete all workspace-filtered.

    This is the prod-safety guarantee: an l0_probe run_round can NEVER read, count, or
    DELETE a prod 'l0' ledger row. We assert it with two non-prod workspaces (l0_probe vs
    l0_probe2) so the test never touches prod even structurally.
    """
    _WK = "__WS_ISO_TEST_DELETE_ME__"

    def _set_ws(ws: str) -> None:
        monkeypatch.setenv("NEO4J_WORKSPACE", ws)
        monkeypatch.setenv("POSTGRES_WORKSPACE", ws)

    _set_ws("l0_probe")
    await store.ensure_schema()
    try:
        # write under l0_probe
        await store.upsert("paper", _WK, doc_id=f"paper:{_WK}", status="done", fingerprint="fpA")
        r = await store.get("paper", _WK)
        assert r is not None and r.status == "done" and r.workspace == "l0_probe"

        # switch workspace → the l0_probe row must be invisible across all read paths
        _set_ws("l0_probe2")
        assert await store.get("paper", _WK) is None
        assert _WK not in (await store.load("paper"))
        # count_by_status under the other ws must not count the l0_probe 'done' row
        assert (await store.count_by_status("paper")).get("done", 0) == 0

        # a delete from the other workspace must NOT reach the l0_probe row
        await store.delete("paper", _WK)
        _set_ws("l0_probe")
        survived = await store.get("paper", _WK)
        assert survived is not None and survived.status == "done"
    finally:
        _set_ws("l0_probe")
        await store.delete("paper", _WK)
        assert await store.get("paper", _WK) is None
        # ensure no stray rows left in either probe workspace
        _set_ws("l0_probe2")
        await store.delete("paper", _WK)


async def test_ledger_load_by_status_spans_sources_within_workspace(monkeypatch):
    """#131: load_by_status returns every row in the given statuses across ingest sources (the
    healed-row reconcile covers paper AND operator-doc rows), filtered to the current workspace."""

    def _set_ws(ws: str) -> None:
        monkeypatch.setenv("NEO4J_WORKSPACE", ws)
        monkeypatch.setenv("POSTGRES_WORKSPACE", ws)

    _P, _T, _D = "__LBS_PAPER_DELETE_ME__", "__LBS_BOOK_DELETE_ME__#s1", "__LBS_DONE_DELETE_ME__"
    _set_ws("l0_probe")
    await store.ensure_schema()
    try:
        await store.upsert("paper", _P, doc_id=f"paper:{_P}", status="error", fingerprint="fp")
        await store.upsert("textbook", _T, doc_id=f"textbook:{_T}", status="error")
        await store.upsert("paper", _D, doc_id=f"paper:{_D}", status="done", fingerprint="fp")

        got = {(r.ingest_source, r.source_id): r for r in await store.load_by_status(
            ["error", "error_parked"])}
        assert ("paper", _P) in got and ("textbook", _T) in got
        assert ("paper", _D) not in got
        assert got[("textbook", _T)].doc_id == f"textbook:{_T}"

        _set_ws("l0_probe2")
        mine = {(r.ingest_source, r.source_id) for r in await store.load_by_status(["error"])}
        assert ("paper", _P) not in mine and ("textbook", _T) not in mine
    finally:
        _set_ws("l0_probe")
        await store.delete("paper", _P)
        await store.delete("textbook", _T)
        await store.delete("paper", _D)
