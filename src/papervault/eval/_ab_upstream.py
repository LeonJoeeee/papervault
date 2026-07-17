"""A/B upstream measurement (extraction-prompt validation) — parametrized by WORKSPACE.

Reuses upstream_metrics' constants but works on l0 OR l0_probe (the prod sampler hardcodes l0).
  guardrails:  density + U3 off-ontology + U4 collision + U5 isolated for a workspace (Cypher only).
  sample-u1:   write N U1 edges (relationship_description + source chunks) for the MiMo faithfulness judge.

Usage:
  python -m papervault.eval._ab_upstream guardrails l0_probe
  python -m papervault.eval._ab_upstream sample-u1 l0_probe /tmp/u1_new.jsonl 120
"""
import asyncio, collections, json, re, sys

SEP = "<SEP>"
ONTOLOGY = {"application", "concept", "dataset", "finding", "instrument", "material",
            "method", "mission", "model", "phenomenon", "quantity"}
norm = lambda x: re.sub(r"[\s\-_]+", "", x.lower())


def _env():
    env = {}
    for line in open(".env"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env.setdefault(k, v)
    return env


def _driver(env):
    from neo4j import GraphDatabase
    return GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"]))


def _dsn(env):
    return (f"postgresql://{env['POSTGRES_USER']}:{env['POSTGRES_PASSWORD']}@"
            f"{env.get('POSTGRES_HOST', 'localhost')}:{env.get('POSTGRES_PORT', '5432')}/{env['POSTGRES_DATABASE']}")


def guardrails(env, ws: str) -> dict:
    assert ws in ("l0", "l0_probe")
    d = _driver(env)
    with d.session() as s:
        tot = s.run(f"MATCH (n:`{ws}`) RETURN count(n) AS c").single()["c"]
        rels = s.run(f"MATCH (:`{ws}`)-[r]->(:`{ws}`) RETURN count(r) AS c").single()["c"]
        rows = [(r["n"], r["t"], r["deg"]) for r in
                s.run(f"MATCH (n:`{ws}`) RETURN n.entity_id AS n, n.entity_type AS t, COUNT{{(n)--()}} AS deg")]
    d.close()
    offont = sum(1 for _, t, _ in rows if (t or "UNKNOWN") not in ONTOLOGY)
    g = collections.defaultdict(int)
    for n, _, _ in rows:
        if n:
            g[norm(n)] += 1
    coll = sum(v for v in g.values() if v > 1)
    iso = sum(1 for _, _, dg in rows if dg == 0)
    maxdeg = max((dg for _, _, dg in rows), default=0)
    return {"workspace": ws, "entities": tot, "relations": rels,
            "avg_degree": round(2 * rels / max(tot, 1), 2), "max_degree": maxdeg,
            "U3_off_ontology_pct": round(offont / max(tot, 1) * 100, 2),
            "U4_collision_pct": round(coll / max(tot, 1) * 100, 2),
            "U5_isolated_pct": round(iso / max(tot, 1) * 100, 2)}


async def sample_u1(env, ws: str, out: str, n: int = 120):
    assert ws in ("l0", "l0_probe")
    import asyncpg
    c = await asyncpg.connect(_dsn(env))
    d = _driver(env)

    def chunks_of(cell):
        try:
            return json.loads(cell) if isinstance(cell, str) else list(cell)
        except Exception:
            return []

    rels = []
    for clause, lim in [("count=1", n // 2), ("count>1", n - n // 2)]:
        rels += await c.fetch(f"SELECT id, chunk_ids, count FROM lightrag_relation_chunks "
                              f"WHERE workspace='{ws}' AND {clause} ORDER BY random() LIMIT {lim}")
    u1 = []
    with d.session() as s:
        for r in rels:
            if SEP not in r["id"]:
                continue
            head, tail = r["id"].split(SEP, 1)
            er = s.run(f"MATCH (a:`{ws}` {{entity_id:$h}})-[e]-(b:`{ws}` {{entity_id:$t}}) "
                       "RETURN e.description AS d, e.keywords AS k LIMIT 1", h=head, t=tail).single()
            if not er:
                continue
            src = []
            for cid in chunks_of(r["chunk_ids"])[:3]:
                row = await c.fetchrow(f"SELECT content FROM lightrag_doc_chunks WHERE workspace='{ws}' AND id=$1", cid)
                if row:
                    src.append(row["content"][:2200])
            if src:
                u1.append({"head": head, "tail": tail, "description": er["d"], "keywords": er["k"],
                           "count": r["count"], "source_chunks": src})
    with open(out, "w") as f:
        for e in u1:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    await c.close()
    d.close()
    return len(u1)


def main():
    env = _env()
    cmd = sys.argv[1]
    if cmd == "guardrails":
        print(json.dumps(guardrails(env, sys.argv[2]), indent=2))
    elif cmd == "sample-u1":
        ws, out = sys.argv[2], sys.argv[3]
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 120
        got = asyncio.run(sample_u1(env, ws, out, n))
        print(f"sampled U1={got} edges from {ws} -> {out}")
    else:
        raise SystemExit("usage: _ab_upstream.py guardrails <ws> | sample-u1 <ws> <out> [n]")


if __name__ == "__main__":
    main()
