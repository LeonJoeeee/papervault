"""KS ledger — SoT for "what KS has processed" (SDD §4.1).

Replaces the dead v2 table `paper_attempts`. Per (workspace, ingest_source, source_id):
fingerprint + doc_id + status. async psycopg pool (mirrors sidecar/crud.py).

status 权威枚举(SDD §13):processing | done | done_meta | done_abstract | error | error_parked | pending_remove
(done_abstract, #144: a metadata-state paper whose one abstract-only doc is processed.)
(absent = 无行)。

REDISTILL 熔断(#84):`attempts` 列记录该 key **连续** build 失败次数。每次写 status='error'
自增;写成功终态(done/done_meta)或**指纹真变**(新内容)清零。累计到 KS_REDISTILL_MAX_ATTEMPTS
(默认 3)时,该次 'error' 写入被改写成终态 **'error_parked'**(泊车)—— reconcile.diff 不再把
error_parked 并入 to_redistill,于是 systemic build 失败(如 #84 embedding stack 断)不会每 60s
一轮无限 redistill+remove(删图)+重投,把 churn 有界收住。泊车 key 仅由(a)显式 force re-ingest
(ledger.delete 清行,attempts 归零)或(b)指纹变化(内容真变)复活。
(c) #131: a parked/error row whose LightRAG doc later reaches PROCESSED (LightRAG retries
FAILED-with-content docs itself) is written back `done` by round.reconcile_healed. After (a),
the round never re-enqueues a doc that doc_status still holds: distill_batch adopts it instead.

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
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from papervault.knowledge.config import CONFIG
from papervault.knowledge.store.graph import assert_safe_workspace

log = logging.getLogger(__name__)

VALID_STATUS = {"processing", "done", "done_meta", "done_abstract", "error", "error_parked",
                "pending_remove"}
# Terminal success statuses: a write of one of these resets the #84 failure streak.
_SUCCESS_STATUSES = ("done", "done_meta", "done_abstract")

# REDISTILL circuit-breaker (#84): after this many CONSECUTIVE build failures a key is PARKED
# (status→error_parked, terminal — reconcile.diff no longer re-distills it), instead of being
# re-distilled + re-removed (graph churn) every round forever on a systemic build break.
DEFAULT_MAX_ATTEMPTS = 3


def _max_attempts() -> int:
    """Consecutive-failure threshold before a key is parked (env KS_REDISTILL_MAX_ATTEMPTS).

    Read at call time (not import) so an operator can retune without a restart and tests can
    monkeypatch the env. A value <= 0 disables parking (a key can never reach the threshold —
    it stays 'error' and keeps retrying, the pre-#84 behavior)."""
    try:
        return int(os.getenv("KS_REDISTILL_MAX_ATTEMPTS", str(DEFAULT_MAX_ATTEMPTS)))
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS


def _next_attempts_status(
    prev_attempts: int,
    prev_fingerprint: Optional[str],
    new_status: str,
    new_fingerprint: Optional[str],
    max_attempts: int,
) -> tuple[int, str]:
    """PURE circuit-breaker transition (#84) — the single source of truth for attempts + parking.

    Given the row's current (attempts, fingerprint) and the incoming (status, fingerprint),
    return the (attempts, effective_status) to persist. `upsert` reads the prior row and applies
    this; the in-memory FakeLedger in the round tests calls the SAME function, so the DB-free
    round/diff tests exercise the real decision (not a re-implemented fake).

      - new_status == 'error'                → attempts+1; PARK (→'error_parked') once it reaches
                                               max_attempts (>0). This is the loop that ran away
                                               in #84 (error → to_redistill → build fails → error).
      - new_status in _SUCCESS_STATUSES      → success ⇒ reset attempts to 0 (a good build clears
                                               the streak, so a later transient failure gets a
                                               fresh budget).
      - a genuinely NEW fingerprint          → content actually changed ⇒ reset to 0 (revives a
                                               parked key with a fresh attempt budget; this is the
                                               fingerprint-change revival path).
      - anything else (processing/pending_remove, or a same-fp write) → PRESERVE the count, so the
        streak spans the intermediate 'processing' a redistill sets between two failures.
    """
    if new_status == "error":
        n = prev_attempts + 1
        if max_attempts > 0 and n >= max_attempts:
            return n, "error_parked"
        return n, "error"
    if new_status in _SUCCESS_STATUSES:
        return 0, new_status
    if new_fingerprint is not None and new_fingerprint != prev_fingerprint:
        return 0, new_status
    return prev_attempts, new_status


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
    attempts          integer NOT NULL DEFAULT 0,
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
    attempts: int = 0  # consecutive build-failure count (#84 circuit-breaker); 0 = clean


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
        # #84 circuit-breaker: consecutive build-failure counter. NOT NULL DEFAULT 0 backfills
        # every existing row to 0 in the same catalog op (cheap; no table rewrite for a constant
        # default on PG ≥ 11), so pre-#84 rows start with a clean streak. Idempotent no-op once present.
        await conn.execute(
            "ALTER TABLE ks_ledger ADD COLUMN IF NOT EXISTS attempts integer NOT NULL DEFAULT 0"
        )
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
        # Read the prior row (attempts + fingerprint) so the #84 circuit-breaker transition and
        # the §4.1 fingerprint-COALESCE are computed together in Python via the pure
        # `_next_attempts_status` helper. The scheduler is a single serial writer per workspace
        # (SDD §6.6 进程内串行), so this read-then-upsert on one pooled connection has no
        # concurrent same-key writer to race — and it keeps ONE source of truth for the parking
        # rule (the same helper the FakeLedger tests drive), instead of a duplicated SQL CASE.
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT attempts, fingerprint FROM ks_ledger "
                "WHERE workspace=%s AND ingest_source=%s AND source_id=%s",
                (ws, ingest_source, source_id),
            )
            prev = await cur.fetchone()
        prev_attempts = int(prev["attempts"]) if prev and prev["attempts"] is not None else 0
        prev_fp = prev["fingerprint"] if prev else None
        eff_attempts, eff_status = _next_attempts_status(
            prev_attempts, prev_fp, status, fingerprint, _max_attempts()
        )
        # 指纹更新语义(SDD §4.1):仅显式传新指纹才更新;不传(None)的"只改 status"转移
        # (pending_remove/error)保留已存指纹,否则被覆成 NULL → 下轮 diff 误判 REDISTILL。
        eff_fp = fingerprint if fingerprint is not None else prev_fp
        if eff_status == "error_parked" and status == "error":
            log.warning(
                "ks_ledger: %s/%s PARKED after %d consecutive build failures (status→error_parked, "
                "no longer re-distilled; revive via force re-ingest or a content/fingerprint change) — #84",
                ingest_source, source_id, eff_attempts,
            )
        await conn.execute(
            """
            INSERT INTO ks_ledger
                (workspace, ingest_source, source_id, fingerprint, doc_id, status, attempts, last_processed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (workspace, ingest_source, source_id) DO UPDATE SET
                fingerprint       = EXCLUDED.fingerprint,
                doc_id            = EXCLUDED.doc_id,
                status            = EXCLUDED.status,
                attempts          = EXCLUDED.attempts,
                last_processed_at = now()
            """,
            (ws, ingest_source, source_id, eff_fp, doc_id, eff_status, eff_attempts),
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


async def load_by_status(statuses: Iterable[str]) -> list[LedgerRecord]:
    """Every row in the current workspace whose status is one of `statuses`, ACROSS ingest
    sources (#131: the healed-row reconcile covers paper and operator-doc rows alike).

    workspace-filtered like every other entry (SDD §4.1); served by the (workspace, status) index.
    """
    ws = _workspace()
    async with _conn() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM ks_ledger WHERE workspace=%s AND status = ANY(%s)",
                (ws, list(statuses)),
            )
            rows = await cur.fetchall()
    return [_to_record(r) for r in rows]


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
        attempts=int(row.get("attempts") or 0),
    )
