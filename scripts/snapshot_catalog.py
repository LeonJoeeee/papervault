"""F0 baseline snapshot (read-only): catalog counts, v2 state, identity, R2 coverage, DB counts."""
import argparse
import json, os, time
from collections import Counter
from pathlib import Path

ROOT = Path("/data/paper-vault")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "snapshots/catalog-snapshot.json",
                    help="snapshot destination (default: /data/paper-vault/snapshots/catalog-snapshot.json)")
OUT = parser.parse_args().output.expanduser().resolve()

# STRICTLY READ-ONLY: parse index.json directly; never construct Library (its __init__
# creates vault directories / touches manifest state).
from papervault.library.models import Paper
INDEX = json.loads((ROOT / "index.json").read_text())
papers = []
for _rec in (INDEX.get("papers") or {}).values():
    try:
        papers.append(Paper(**{k: v for k, v in _rec.items() if k in Paper.model_fields}))
    except Exception:
        pass

def c(seq): return dict(Counter(seq))
snap = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "records": len(papers)}
snap["download_status"] = c(p.download_status or "" for p in papers)
snap["sources"] = c((p.source or "").split("+")[0].lower() for p in papers)
snap["v2_retrieval"] = c(str(p.acquisition_state.retrieval) for p in papers)
snap["v2_policy"] = c(str(p.acquisition_state.policy) for p in papers)
snap["v2_merge"] = c(str(p.acquisition_state.merge) for p in papers)
snap["v2_quarantine"] = c(str(p.acquisition_state.quarantine) for p in papers)
snap["has_doi"] = sum(1 for p in papers if (p.doi or "").strip())
snap["has_arxiv"] = sum(1 for p in papers if (p.arxiv_id or "").strip())
snap["years"] = c(p.year for p in papers if isinstance(p.year, int))

# identity duplicates: normalized DOI and arXiv
doi = Counter((p.doi or "").strip().lower() for p in papers if (p.doi or "").strip())
arx = Counter((p.arxiv_id or "").strip().lower() for p in papers if (p.arxiv_id or "").strip())
snap["duplicate_dois"] = {k: v for k, v in doi.items() if v > 1}
snap["duplicate_arxivs"] = {k: v for k, v in arx.items() if v > 1}

# files
pdfs = list((ROOT / "pdfs").glob("*.pdf"))
snap["pdf_files"] = len(pdfs)
snap["pdf_bytes"] = sum(f.stat().st_size for f in pdfs)
md = list((ROOT / "extracts" / "md").glob("*.md"))
txt = list((ROOT / "extracts" / "txt").glob("*.txt"))
snap["md_files"] = len(md); snap["txt_files"] = len(txt)
snap["md_bytes"] = sum(f.stat().st_size for f in md)
snap["txt_bytes"] = sum(f.stat().st_size for f in txt)

# R2 coverage over the frozen landmark pools (metadata / readable / indexed)
pools = []
for line in (Path("/home/leon/lab/papervault/src/papervault/eval/gold_search_pools.jsonl")).read_text().splitlines():
    line = line.strip()
    if line: pools.append(json.loads(line))
by_key = {p.key: p for p in papers}
def readable(p):
    return (ROOT / "extracts" / "md" / f"{p.key}.md").exists() or (ROOT / "extracts" / "txt" / f"{p.key}.txt").exists()

# PG: LightRAG doc_status processed flags + ks_ledger
pg = {}
try:
    import psycopg
    dsn = os.environ.get("PAPERVAULT_POSTGRES_DSN") or os.environ.get("POSTGRES_DSN") or ""
    if not dsn:
        pw = os.environ.get("POSTGRES_PASSWORD", "")
        dsn = f"postgresql://{os.environ.get('POSTGRES_USER','ks')}:{pw}@{os.environ.get('POSTGRES_HOST','127.0.0.1')}:{os.environ.get('POSTGRES_PORT','5432')}/{os.environ.get('POSTGRES_DB','knowledge_system')}"
    with psycopg.connect(dsn, connect_timeout=5, options="-c default_transaction_read_only=on") as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("select status, count(*) from lightrag_doc_status group by status")
                pg["doc_status"] = {r[0]: r[1] for r in cur.fetchall()}
            except Exception as e:
                pg["doc_status_error"] = str(e)[:200]
            try:
                cur.execute("select status, count(*) from ks_ledger group by status")
                pg["ks_ledger"] = {str(r[0]): r[1] for r in cur.fetchall()}
            except Exception as e:
                pg["ks_ledger_error"] = str(e)[:200]
except Exception as e:
    pg["error"] = str(e)[:200]
snap["postgres"] = pg

# R2 compute (metadata = in library; readable = extract file exists; indexed = doc_status processed for that file)
def file_key(fp):
    fp = fp or ""
    base = fp.rsplit("/", 1)[-1]
    for suf in (".md", ".txt"):
        if base.endswith(suf): base = base[: -len(suf)]
    return base

indexed_keys = set()
try:
    import psycopg
    pw = os.environ.get("POSTGRES_PASSWORD", "")
    dsn = f"postgresql://{os.environ.get('POSTGRES_USER','ks')}:{pw}@{os.environ.get('POSTGRES_HOST','127.0.0.1')}:{os.environ.get('POSTGRES_PORT','5432')}/{os.environ.get('POSTGRES_DB','knowledge_system')}"
    with psycopg.connect(dsn, connect_timeout=5, options="-c default_transaction_read_only=on") as conn:
        with conn.cursor() as cur:
            cur.execute("select file_path from lightrag_doc_status where status = 'processed'")
            for (fp,) in cur.fetchall():
                indexed_keys.add(file_key(fp))
except Exception:
    pass

# pool entries are {title, doi, ...}; match by normalized DOI first, then normalized title
def norm_title(t):
    import re
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())
by_doi = {((p.doi or "").strip().lower()): p for p in papers if (p.doi or "").strip()}
by_title = {}
for p in papers:
    k = norm_title(p.title)
    if k: by_title.setdefault(k, p)
def match(entry):
    doi = (entry.get("doi") or "").strip().lower()
    if doi and doi in by_doi: return by_doi[doi]
    ttl = norm_title(entry.get("title"))
    return by_title.get(ttl) if ttl else None
r2 = {"pools": len(pools), "entries_total": 0, "matched": 0, "readable": 0, "indexed": 0, "per_pool": []}
for pool in pools:
    entries = pool.get("pool") or []
    if not entries: continue
    matched, rd, ix = [], [], []
    for e in entries:
        p2 = match(e)
        if p2 is not None:
            matched.append(p2)
            if readable(p2): rd.append(p2)
            if p2.key in indexed_keys: ix.append(p2)
    r2["entries_total"] += len(entries); r2["matched"] += len(matched)
    r2["readable"] += len(rd); r2["indexed"] += len(ix)
    r2["per_pool"].append({"qid": pool.get("qid"), "n": len(entries),
                            "matched": len(matched), "readable": len(rd), "indexed": len(ix)})
snap["r2_coverage"] = r2
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(snap, indent=1, ensure_ascii=False))
print("wrote", OUT)
print("records:", snap["records"], "| status:", snap["download_status"])
print("R2 entries:", r2["entries_total"], "matched:", r2["matched"], "readable:", r2["readable"], "indexed:", r2["indexed"])
