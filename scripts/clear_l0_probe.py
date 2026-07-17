"""Clear ONLY the l0_probe dev workspace (Neo4j label + Postgres workspace-column rows + ks_ledger),
KEEPING llm_cache. SAFETY = every delete is HARDCODED to workspace/label 'l0_probe' (never l0); we
count l0_probe AND l0 before+after and ASSERT l0 is unchanged. Connection params from .env (values
not printed). Run:  uv run python scripts/clear_l0_probe.py
"""
from __future__ import annotations

import asyncio
from pathlib import Path

WS = "l0_probe"               # hardcoded scope — NEVER l0
PROD = "l0"                   # safety baseline only (must stay unchanged)
assert WS == "l0_probe" and PROD == "l0"

_env: dict[str, str] = {}
for _line in Path(__file__).resolve().parent.parent.joinpath(".env").read_text().splitlines():
    _line = _line.strip()
    if not _line or _line.startswith("#") or "=" not in _line:
        continue
    _k, _v = _line.split("=", 1)
    _env[_k.strip()] = _v.strip().strip('"').strip("'")


def clear_neo4j() -> None:
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(_env["NEO4J_URI"], auth=(_env["NEO4J_USERNAME"], _env["NEO4J_PASSWORD"]))
    dbname = _env.get("NEO4J_DATABASE") or "neo4j"
    try:
        with drv.session(database=dbname) as s:
            before_p = s.run("MATCH (n:`l0_probe`) RETURN count(n) AS c").single()["c"]
            before_l0 = s.run("MATCH (n:`l0`) RETURN count(n) AS c").single()["c"]
            print(f"[neo4j] BEFORE  l0_probe={before_p}  l0={before_l0}", flush=True)
            if before_p > 200_000:
                raise SystemExit(f"ABORT: l0_probe node count {before_p} is prod-scale — refusing.")
            # batched scoped delete (label-scoped; never a bare MATCH (n))
            while True:
                deleted = s.run(
                    "MATCH (n:`l0_probe`) WITH n LIMIT 20000 DETACH DELETE n RETURN count(n) AS c"
                ).single()["c"]
                if deleted == 0:
                    break
            after_p = s.run("MATCH (n:`l0_probe`) RETURN count(n) AS c").single()["c"]
            after_l0 = s.run("MATCH (n:`l0`) RETURN count(n) AS c").single()["c"]
            print(f"[neo4j] AFTER   l0_probe={after_p}  l0={after_l0}", flush=True)
            assert after_p == 0, f"l0_probe not empty: {after_p}"
            assert after_l0 == before_l0, f"CRITICAL: l0 changed {before_l0}->{after_l0}"
    finally:
        drv.close()


async def clear_pg() -> None:
    import asyncpg
    conn = await asyncpg.connect(
        host=_env["POSTGRES_HOST"], port=int(_env["POSTGRES_PORT"]),
        user=_env["POSTGRES_USER"], password=_env["POSTGRES_PASSWORD"],
        database=_env.get("POSTGRES_DATABASE") or _env.get("POSTGRES_DB"),
    )
    try:
        # find every table that HAS a workspace column (the lightrag_* tables + ks_ledger)
        rows = await conn.fetch(
            "SELECT table_name FROM information_schema.columns "
            "WHERE column_name='workspace' AND table_schema='public'"
        )
        tables = sorted({r["table_name"] for r in rows})
        print(f"[pg] tables with a workspace column: {tables}", flush=True)
        keep = [t for t in tables if "llm_cache" in t.lower()]   # keep cached extractions
        wipe = [t for t in tables if t not in keep]
        print(f"[pg] will WIPE l0_probe rows from {wipe}; KEEP {keep}", flush=True)

        async def counts(label: str) -> None:
            for t in tables:
                cp = await conn.fetchval(f'SELECT count(*) FROM "{t}" WHERE workspace=$1', WS)
                cl = await conn.fetchval(f'SELECT count(*) FROM "{t}" WHERE workspace=$1', PROD)
                print(f"  [pg {label}] {t:28} l0_probe={cp:<7} l0={cl}", flush=True)

        print("[pg] BEFORE:", flush=True)
        before_l0 = {t: await conn.fetchval(f'SELECT count(*) FROM "{t}" WHERE workspace=$1', PROD) for t in tables}
        await counts("before")
        for t in wipe:
            await conn.execute(f'DELETE FROM "{t}" WHERE workspace=$1', WS)
        print("[pg] AFTER:", flush=True)
        await counts("after")
        # verify l0 untouched + l0_probe wiped (except kept tables)
        for t in tables:
            cl = await conn.fetchval(f'SELECT count(*) FROM "{t}" WHERE workspace=$1', PROD)
            assert cl == before_l0[t], f"CRITICAL: l0 changed in {t}: {before_l0[t]}->{cl}"
        for t in wipe:
            cp = await conn.fetchval(f'SELECT count(*) FROM "{t}" WHERE workspace=$1', WS)
            assert cp == 0, f"l0_probe not empty in {t}: {cp}"
    finally:
        await conn.close()


if __name__ == "__main__":
    print("=== CLEAR l0_probe (Neo4j + Postgres), keep llm_cache, verify l0 untouched ===", flush=True)
    clear_neo4j()
    asyncio.run(clear_pg())
    print("=== DONE: l0_probe cleared; l0 verified unchanged ===", flush=True)
