"""KS ledger — SoT for "what KS has processed" (SDD §4.1).

Replaces the dead v2 table `paper_attempts`. Per (workspace, ingest_source, source_id):
fingerprint + doc_id + status. async psycopg pool (mirrors sidecar/crud.py).

status 权威枚举(SDD §13):processing | done | done_meta | error | pending_remove
(absent = 无行)。

WORKSPACE 隔离(SDD §4.1/§6.5,blocker ① 落地):`ks_ledger` 与 LightRAG 表同库,过去
共用单表无 workspace 维度 → l0_probe 跑 run_round 会写/删生产 l0 账本行(铁律(1))。
现加 `workspace` 列,PK = (workspace, ingest_source, source_id),五个入口(upsert/get/
load/count_by_status/delete)全部按当前 workspace 过滤。workspace 不由 caller 各传一遍
(易漏、也可被绕过),而是模块内从单一权威 `store.graph.assert_safe_workspace()` 解析
(= POSTGRES_WORKSPACE,与 cli._count_doc_status 同源),令 round.py/distill.py 的调用
签名无需改、也无法绕过隔离。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from papervault.knowledge.config import CONFIG
from papervault.knowledge.store.graph import assert_safe_workspace

log = logging.getLogger(__name__)

VALID_STATUS = {"processing", "done", "done_meta", "error", "pending_remove"}


def _workspace() -> str:
    """当前 ledger workspace = `assert_safe_workspace()` 解析值(SDD §4.1/§6.5).

    与 cli._count_doc_status 读 lightrag_doc_status 用的 workspace 同源,保证 `ks stats`
    两侧计数(ledger vs doc_status)落在同一 namespace,§6.5 step5 验收闸不跨 workspace
    误判。同时拒生产 'l0'(无 KS_ALLOW_PROD_WORKSPACE=1)、拒两 env 不等/空 — 与图侧同闸。
    """
    return assert_safe_workspace()


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ks_ledger (
    workspace         text NOT NULL,
    ingest_source     text NOT NULL,
    source_id         text NOT NULL,
    fingerprint       text,
    doc_id            text NOT NULL,
    status            text NOT NULL,
    last_processed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace, ingest_source, source_id)
)
"""
_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS ks_ledger_status ON ks_ledger(workspace, status)"


@dataclass
class LedgerRecord:
    workspace: str
    ingest_source: str
    source_id: str
    fingerprint: Optional[str]
    doc_id: str
    status: str


_pool: Optional[AsyncConnectionPool] = None


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(CONFIG.postgres.dsn, min_size=1, max_size=5, open=False)
        await _pool.open()
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def _conn() -> AsyncIterator[psycopg.AsyncConnection]:
    pool = await get_pool()
    async with pool.connection() as conn:
        yield conn


async def ensure_schema() -> None:
    """Idempotent schema + workspace migration (SDD §4.1/§8/§13, blocker ①).

    `CREATE TABLE IF NOT EXISTS` is a no-op against an already-existing old-form table
    (no `workspace` column, binary PK) — so a bare DDL swap would silently leave the
    upgrade inert and the isolation a no-op. This runs a re-runnable migration instead:
      1. CREATE TABLE IF NOT EXISTS (new form, for a fresh DB).
      2. ALTER TABLE ADD COLUMN IF NOT EXISTS workspace  (existing old table gains it).
      3. Backfill any NULL-workspace rows to the current resolved workspace, then NOT NULL.
      4. Rebuild PK from (ingest_source, source_id) → (workspace, ingest_source, source_id).
      5. DROP old status index, CREATE new (workspace, status) index.

    PROD-SAFETY (铁律(1)/(3)): the migration only touches whatever DB the pool points at;
    backfill assigns NULL-workspace rows to `_workspace()` (= assert_safe_workspace(),
    which refuses prod 'l0' without explicit opt-in). It is verified only on l0_probe; a
    prod migration is deferred to the user. All steps are idempotent / re-runnable.
    """
    ws = _workspace()  # gate first (refuses prod 'l0' without opt-in) + value for backfill
    async with _conn() as conn:
        await conn.execute(_CREATE_TABLE)
        # idempotent column add (cheap catalog no-op if already present — keep unconditional).
        await conn.execute("ALTER TABLE ks_ledger ADD COLUMN IF NOT EXISTS workspace text")
        # ★ condition the cold-migration steps (backfill UPDATE + SET NOT NULL) on a one-shot
        # introspect, mirroring the ④/⑤ guards below. The backfill UPDATE + `ALTER COLUMN
        # SET NOT NULL` were previously UNCONDITIONAL every startup: on an already-migrated
        # table the UPDATE matches 0 rows but still takes RowExclusiveLock + scans for NULLs,
        # and `SET NOT NULL` is catalog-mutating DDL that grabs ACCESS EXCLUSIVE on ks_ledger
        # (shared PG with LightRAG's write path) before catalog inspection — exactly the
        # startup-time table-lock churn the conditional PK(④)/index(⑤) guards were built to
        # eliminate. So introspect pg_attribute.attnotnull: if the column already exists and
        # is NOT NULL, this whole block is a pure no-op (no table lock). Only the cold path
        # (column just ADDed / still nullable) runs the backfill + SET NOT NULL (SDD §4.1 ③).
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT attnotnull FROM pg_attribute "
                "WHERE attrelid = 'ks_ledger'::regclass AND attname = 'workspace' AND attnum > 0",
            )
            row = await cur.fetchone()
        ws_already_not_null = bool(row and row[0])
        if not ws_already_not_null:
            # cold migration: backfill rows that predate the column, then enforce NOT NULL.
            # ★ loud-log (铁律(3) 一致): backfill 的『安全』全建立在『WHERE workspace IS NULL 命中的
            # 行必然都属于当前 _workspace()』这个隐含前提上。在 l0_probe 流里这恒是首次冷迁移(本
            # 分支只在列缺失/仍可空时进入);但若未来有人在生产 l0(KS_ALLOW_PROD_WORKSPACE=1)对一
            # 张曾被 l0_probe 跑过迁移、却仍遗留 NULL 行的同库表跑 ensure_schema,这些 NULL 行会被无
            # 声归到当前 workspace。故把『静默归并』升级为『有痕迹』:rowcount>0 必 WARNING,让意外
            # 回填可见,而非静默(§6.5 铁律(3)『绝不在生产静默跑迁移』)。无需改 DDL。
            cur = await conn.execute(
                "UPDATE ks_ledger SET workspace=%s WHERE workspace IS NULL", (ws,)
            )
            if cur.rowcount and cur.rowcount > 0:
                log.warning(
                    "ks_ledger backfill: stamped %d workspace-less row(s) with workspace=%r "
                    "(expected only on the FIRST cold migration; a non-zero count on a routine "
                    "startup means stray NULL rows were silently merged into this workspace)",
                    cur.rowcount, ws,
                )
            await conn.execute("ALTER TABLE ks_ledger ALTER COLUMN workspace SET NOT NULL")
        # rebuild PK to the ternary form only if it isn't already (re-runnable).
        # ★ compare the PK column SET, not an ordered list: a composite PK's identity is
        # its member set, and the query's `ORDER BY a.attnum` returns *physical* column
        # order, not PK-definition order. On a MIGRATED table (workspace ALTER-added last,
        # attnum=max) an already-correct ternary PK reads back as
        # ['ingest_source','source_id','workspace'] — an ordered-list `!=` would be a
        # false-positive every startup, firing a needless DROP+ADD PRIMARY KEY (ACCESS
        # EXCLUSIVE lock on a PG shared with LightRAG's write path). Set-compare is
        # order-independent → idempotent on both fresh and migrated tables (SDD §4.1 ④).
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT array_agg(a.attname)
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = 'ks_ledger'::regclass AND i.indisprimary
                """
            )
            (pk_cols,) = await cur.fetchone()
        if set(pk_cols or []) != {"workspace", "ingest_source", "source_id"}:
            await conn.execute("ALTER TABLE ks_ledger DROP CONSTRAINT IF EXISTS ks_ledger_pkey")
            await conn.execute(
                "ALTER TABLE ks_ledger ADD PRIMARY KEY (workspace, ingest_source, source_id)"
            )
        # status index must lead with workspace (reconcile scans pending_remove/error per ws).
        # `_CREATE_INDEX` is already `CREATE INDEX IF NOT EXISTS ks_ledger_status ON
        # (workspace, status)` — a true no-op once that index exists. So an unconditional
        # leading DROP would make startup NON-idempotent (probe / scheduler main_loop each
        # take a needless ACCESS EXCLUSIVE on ks_ledger, the same shared-DB startup churn the
        # PK guard above was made conditional to avoid). Mirror that guard: introspect the
        # existing ks_ledger_status columns and only DROP+rebuild when they aren't already
        # (workspace, status) — i.e. only to replace a hypothetical legacy single-column
        # (status) index. On fresh/migrated correct tables this is a pure no-op (SDD §4.1 ⑤).
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT array_agg(a.attname ORDER BY x.ord)
                FROM pg_index i
                JOIN pg_class c ON c.oid = i.indexrelid
                JOIN unnest(i.indkey) WITH ORDINALITY AS x(attnum, ord) ON true
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = x.attnum
                WHERE i.indrelid = 'ks_ledger'::regclass AND c.relname = 'ks_ledger_status'
                """
            )
            (idx_cols,) = await cur.fetchone()
        if idx_cols != ["workspace", "status"]:
            await conn.execute("DROP INDEX IF EXISTS ks_ledger_status")
            await conn.execute(_CREATE_INDEX)
        await conn.commit()


async def upsert(
    ingest_source: str,
    source_id: str,
    *,
    doc_id: str,
    status: str,
    fingerprint: Optional[str] = None,
) -> None:
    if status not in VALID_STATUS:
        raise ValueError(f"invalid ledger status: {status!r} (allowed: {sorted(VALID_STATUS)})")
    ws = _workspace()
    async with _conn() as conn:
        await conn.execute(
            """
            INSERT INTO ks_ledger
                (workspace, ingest_source, source_id, fingerprint, doc_id, status, last_processed_at)
            VALUES (%s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (workspace, ingest_source, source_id) DO UPDATE SET
                -- 指纹更新语义(SDD §4.1):仅显式传新指纹才更新;不传(None)的"只改 status"转移
                -- (pending_remove/error)保留已存指纹,否则被覆成 NULL → 下轮 diff 误判 REDISTILL。
                fingerprint       = COALESCE(EXCLUDED.fingerprint, ks_ledger.fingerprint),
                doc_id            = EXCLUDED.doc_id,
                status            = EXCLUDED.status,
                last_processed_at = now()
            """,
            (ws, ingest_source, source_id, fingerprint, doc_id, status),
        )
        await conn.commit()


async def get(ingest_source: str, source_id: str) -> Optional[LedgerRecord]:
    ws = _workspace()
    async with _conn() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM ks_ledger WHERE workspace=%s AND ingest_source=%s AND source_id=%s",
                (ws, ingest_source, source_id),
            )
            row = await cur.fetchone()
    return _to_record(row) if row else None


async def load(ingest_source: str) -> dict[str, LedgerRecord]:
    """All rows for a source in the current workspace → {source_id: LedgerRecord}.

    For reconcile diff (§6.1). workspace-filtered (SDD §4.1) — l0_probe diff never sees
    prod 'l0' rows, so run_round's to_remove can't reach across the isolation boundary.
    """
    ws = _workspace()
    async with _conn() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM ks_ledger WHERE workspace=%s AND ingest_source=%s",
                (ws, ingest_source),
            )
            rows = await cur.fetchall()
    return {r["source_id"]: _to_record(r) for r in rows}


async def count_by_status(ingest_source: str) -> dict[str, int]:
    """Per-status row counts for one ingest_source in the current workspace (SDD §6.8).

    → {status: n, ...}; statuses with zero rows are omitted. The §6.5 step5 acceptance
    gate (`count(status ∈ {done, done_meta}) == |idx|`) reads from this. workspace is the
    SAME assert_safe_workspace() value that cli._count_doc_status filters lightrag_doc_status
    by, so `ks stats`'s two sides compare counts within ONE namespace (no cross-ws串数).
    """
    ws = _workspace()
    async with _conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT status, count(*) FROM ks_ledger "
                "WHERE workspace=%s AND ingest_source=%s GROUP BY status",
                (ws, ingest_source),
            )
            rows = await cur.fetchall()
    return {status: n for status, n in rows}


async def delete(ingest_source: str, source_id: str) -> None:
    ws = _workspace()
    async with _conn() as conn:
        await conn.execute(
            "DELETE FROM ks_ledger WHERE workspace=%s AND ingest_source=%s AND source_id=%s",
            (ws, ingest_source, source_id),
        )
        await conn.commit()


def _to_record(row: dict) -> LedgerRecord:
    return LedgerRecord(
        workspace=row["workspace"],
        ingest_source=row["ingest_source"],
        source_id=row["source_id"],
        fingerprint=row["fingerprint"],
        doc_id=row["doc_id"],
        status=row["status"],
    )
