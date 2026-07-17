"""Downstream eval runner — drives the REAL KS query path over the fixed gold questions and
saves a per-question record (the raw retrieval + the synth answer) for scoring.

This is the 'ruler' producer: one run = one variant's answers over the FIXED 25 gold questions.
It does NOT score against a baseline (that's stats.py) and does NOT call the judge (that's the
judge subagent + judge_aggregate.py). It captures EXACTLY what the deterministic backbone
(backbone.py) and the judge both need, so a run can be re-scored offline forever without
re-querying the graph.

PROD-SAFETY (铁律 1): this connects to a LIVE graph. It HARD-REFUSES unless
NEO4J_WORKSPACE == POSTGRES_WORKSPACE == 'l0_probe' (the isolated probe workspace). It never
touches prod 'l0'. The existing ~59-paper partial graph on l0_probe is fine for a SMOKE run;
this script is read-only against the graph (query only — no ingest, no clear).

Run:
  NEO4J_WORKSPACE=l0_probe POSTGRES_WORKSPACE=l0_probe \
    uv run python experiments/eval/run_eval.py --gold experiments/eval/gold.jsonl --tag baseline

  # smoke against the partial graph with the bundled tiny gold fixture, no real gold needed:
  NEO4J_WORKSPACE=l0_probe POSTGRES_WORKSPACE=l0_probe \
    uv run python experiments/eval/run_eval.py --gold experiments/eval/gold.sample.jsonl --tag smoke --limit 3

Output: experiments/eval/results/<tag>.jsonl — one JSON object per gold question, in the
EXACT shape backbone.py + judge_prompt.md consume (see backbone.py module docstring):
  {
    "qid": "...",
    "intent": "...",                              # carried for the judge
    "data": {"chunks":[{file_path,paper_key,content,reference_id}], "references":[{file_path,reference_id}],
             "entities":[...], "relationships":[...]},   # the raw aquery_data 'data' block
                                                           # (paper_key = bare cite key for the judge)
    "metadata": {"processing_info": {"total_entities_found": N}},   # graph signal
    "cited_papers": [...],                        # aquery._cited_papers output
    "kb_coverage": "empty"|"thin"|"strong",
    "answer": "<synth prose>"|"(KB 无相关知识)"|"(synthesis LLM failed...)",
    "meta": {tag, workspace, ts, query_mode, top_k, chunk_top_k, gold_file, error?}
  }
If a query raises, the record still lands with meta.error set so one bad question doesn't lose
the whole run. After the run, if every gold entry carries gold_keys, the deterministic backbone
vector is printed (a convenience; the canonical scoring is stats.py + judge_aggregate.py).

NOTE: the JUDGE inputs are all in this file (intent, answer, data.chunks[].content, gold) — the
judge subagent reads results/<tag>.jsonl + gold.jsonl, it does NOT need the live graph.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
REPO_ROOT = EVAL_DIR.parent.parent  # so `from papervault.eval import backbone` resolves

# Reuse the backbone's exact 'paper/<key>' -> '<key>' rule for the judge's bare paper_key, so
# the chunk's paper_key matches the prose's inline [<bare key>] cites (drill 2026-06-02e, tail 8).
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from backbone import strip_paper_key  # noqa: E402  (sibling module, EVAL_DIR on path)


def _rerank_fail_count() -> int:
    """Process-global terminal-rerank-failure count (0 if module not importable)."""
    try:
        from papervault.knowledge.store.lightrag_init import rerank_failure_count
        return rerank_failure_count()
    except Exception:  # noqa: BLE001
        return 0


def _reset_rerank_fails() -> None:
    """Zero the rerank-failure counter so this run's count reflects only this run."""
    try:
        from papervault.knowledge.store.lightrag_init import reset_rerank_failures
        reset_rerank_failures()
    except Exception:  # noqa: BLE001
        pass


def _require_probe_workspace() -> str:
    """Prod-safety gate. Default: 'l0_probe' only. EXCEPTION (user-approved 2026-06-06): allow a
    READ-ONLY eval against prod 'l0' when KS_ALLOW_PROD_WORKSPACE=1 — run_eval is query-only (no
    ingest, no clear, no graph writes), so a baseline on the production graph is safe. Both
    workspace envs must match.
    """
    neo = os.environ.get("NEO4J_WORKSPACE", "").strip()
    pg = os.environ.get("POSTGRES_WORKSPACE", "").strip()
    if neo != pg:
        sys.stderr.write(f"ABORT: workspace mismatch NEO4J_WORKSPACE={neo!r} POSTGRES_WORKSPACE={pg!r}\n")
        raise SystemExit(2)
    if neo == "l0_probe":
        return "l0_probe"
    if neo == "l0" and os.environ.get("KS_ALLOW_PROD_WORKSPACE") == "1":
        sys.stderr.write("NOTE: eval on PROD l0 — READ-ONLY baseline (KS_ALLOW_PROD_WORKSPACE=1).\n")
        return "l0"
    sys.stderr.write(
        "ABORT: eval requires NEO4J_WORKSPACE=POSTGRES_WORKSPACE='l0_probe', OR ='l0' with "
        f"KS_ALLOW_PROD_WORKSPACE=1 (read-only prod baseline). Got {neo!r}/{pg!r}.\n"
    )
    raise SystemExit(2)


def load_gold(path: Path) -> list[dict[str, Any]]:
    """Read gold.jsonl → list of gold entries. Validates the required keys per entry.

    `#`-prefixed and blank lines are skipped (comments allowed, per gold.schema.md).
    """
    gold: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            g = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"gold {path} line {i}: invalid JSON: {e}")
        for req in ("qid", "intent", "gold_keys"):
            if req not in g:
                raise SystemExit(f"gold {path} line {i} (qid={g.get('qid')!r}): missing key {req!r}")
        gold.append(g)
    if not gold:
        raise SystemExit(f"no gold questions found in {path}")
    return gold


def _slim_chunk(c: dict[str, Any]) -> dict[str, Any]:
    """Keep only what backbone scoring + the judge need from an aquery_data chunk.

    Emits BOTH the raw 'file_path' ('paper/<key>', what the backbone strips) AND a bare
    'paper_key' ('<key>', None for non-paper chunks). The judge harness fills the prompt's
    {{paper_key}} from THIS field so it matches the prose's inline [<bare key>] cites — never
    from the raw file_path, which would carry the 'paper/' prefix and make every citation check
    a spurious missing_chunk (SDD §6.10 B, tail 8).
    """
    return {
        "file_path": c.get("file_path"),
        "paper_key": strip_paper_key(c.get("file_path")),
        "content": c.get("content"),
        "reference_id": c.get("reference_id"),
    }


def _slim_ref(r: dict[str, Any]) -> dict[str, Any]:
    return {"file_path": r.get("file_path"), "reference_id": r.get("reference_id")}


async def run_one(rag: Any, g: dict[str, Any], base_meta: dict[str, Any], variant: str = "baseline",
                  no_synth: bool = False) -> dict[str, Any]:
    """Query ONE gold question via the real path and capture the backbone result shape.

    SINGLE retrieval (SDD §6.10 D, drill 2026-06-02d): call rag.aquery_data() ONCE with the
    exact QueryParam aquery pins in its module constants, then run KS's three pure out-feed
    functions — synth_answer(data, intent) / aquery._cited_papers(data) /
    aquery._assess_coverage(metadata, data) — on THAT one captured data dict. This is what
    aquery.query() does internally, but inlining it guarantees the answer/cited_papers/
    kb_coverage AND the raw data.chunks the backbone scores all come from the SAME retrieval —
    so the 'allowed' set the backbone uses for hallucinated_rate is exactly what synth saw.
    (The old code double-queried: aquery.query() re-retrieved, then a second aquery_data
    re-retrieved again, leaving a soundness seam if retrieval were ever non-deterministic.)
    On failure aquery.query()'s own guards are reproduced: a non-success / empty response
    yields the empty out-feed. Any raised exception is caught → meta.error.
    """
    from lightrag import QueryParam
    from papervault.knowledge.query import aquery as aq
    from papervault.knowledge.query.synth import synth_answer

    intent = g["intent"]
    rec: dict[str, Any] = {
        "qid": g["qid"], "intent": intent,
        "data": {"chunks": [], "references": [], "entities": [], "relationships": []},
        "metadata": {},
        "cited_papers": [], "kb_coverage": "empty", "answer": None,
        "meta": dict(base_meta),
    }
    _rr0 = _rerank_fail_count()  # per-question rerank-degradation snapshot (exact at concurrency=1)
    try:
        if variant == "multiquery":
            # V-MQ: multi-query+RRF retrieval (multiquery.py). kb_coverage stays on the original
            # single-intent signal; fusion fires only on strong-coverage Qs (trap path unchanged).
            from papervault.knowledge.query.multiquery import retrieve_fused

            data, metadata, fused_applied = await retrieve_fused(intent, rag)
            rec["meta"]["fused_applied"] = fused_applied
        elif variant == "standard":
            # STANDARD "modern RAG" LOCKED baseline (multiquery.py): anchor + orthogonal facets,
            # reranker OFF, RRF⊕round-robin merge → top-80 chunks → synth. NO coverage gate (always
            # runs the full multi-query+merge). kb_coverage stays on the ANCHOR (full-intent) signal.
            from papervault.knowledge.query.multiquery import retrieve_standard

            data, metadata = await retrieve_standard(intent, rag)
        else:
            # baseline: ONE retrieval, exactly the QueryParam aquery.query() uses.
            res = await rag.aquery_data(
                intent,
                QueryParam(
                    mode=aq._QUERY_MODE, top_k=aq._TOP_K,
                    chunk_top_k=aq._CHUNK_TOP_K, enable_rerank=aq._ENABLE_RERANK,
                ),
            )
            ok_res = isinstance(res, dict) and res.get("status") == "success" and res.get("data")
            data = res["data"] if ok_res else {"chunks": [], "references": [], "entities": [], "relationships": []}
            metadata = res.get("metadata") if isinstance(res, dict) else None

        # Reproduce aquery.query()'s guards: empty data → empty out-feed sentinel.
        if not (data.get("entities") or data.get("relationships") or data.get("chunks")):
            rec["answer"] = aq._EMPTY["answer"]
            rec["cited_papers"] = list(aq._EMPTY["cited_papers"])
            rec["kb_coverage"] = aq._EMPTY["kb_coverage"]
            rec["meta"]["rerank_degraded"] = _rerank_fail_count() > _rr0
            return rec

        # The three pure out-feed functions, all over the SAME captured data dict.
        # --no-synth (FAST tier): skip ONLY the synth LLM round-trip (the ~60-chunk MiMo call that
        # dominates wall-clock). cited_papers + kb_coverage are pure functions over the retrieval
        # data (no answer text), so they + the deterministic backbone (@served / gold_citation_recall
        # / hallucinated_rate, all from data.chunks + cited_papers vs gold) stay fully intact — only
        # the judge-dependent metrics (nugget/cit_sp/faith/trap) are unavailable without an answer.
        rec["answer"] = None if no_synth else await synth_answer(data, intent)
        rec["cited_papers"] = aq._cited_papers(data)
        rec["kb_coverage"] = aq._assess_coverage(metadata, data)

        rec["data"] = {
            "chunks": [_slim_chunk(c) for c in (data.get("chunks") or [])],
            "references": [_slim_ref(r) for r in (data.get("references") or [])],
            "entities": [],        # unused by the backbone; dropped to keep the dump small
            "relationships": [],
        }
        pinfo = (metadata or {}).get("processing_info")
        if pinfo is not None:
            rec["metadata"] = {"processing_info": {"total_entities_found": pinfo.get("total_entities_found")}}
    except Exception as e:  # noqa: BLE001 — one bad question must not lose the run
        rec["meta"]["error"] = f"{type(e).__name__}: {e}"
    rec["meta"]["rerank_degraded"] = _rerank_fail_count() > _rr0
    return rec


async def main_async(args: argparse.Namespace) -> None:
    from papervault.knowledge.query import aquery as aq
    from papervault.knowledge.store.graph import assert_safe_workspace, get_graph

    ws = _require_probe_workspace()
    assert_safe_workspace()  # belt-and-suspenders: also run the production gate

    gold = load_gold(Path(args.gold))
    if args.limit:
        gold = gold[: args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.tag}.jsonl"

    # Self-describing dump: stamp the RESOLVED query-path knobs (not just top_k/chunk_top_k) so a
    # stale-default A/B is caught at read time — the fcap/diversity/refusal knobs default-ON in code,
    # and a variant run that forgets to pin one would otherwise be silently mis-attributed.
    import os as _os
    from papervault.knowledge.query import multiquery as _mq
    base_meta = {
        "tag": args.tag, "workspace": ws, "ts": int(time.time()),
        "query_mode": aq._QUERY_MODE, "top_k": aq._TOP_K, "chunk_top_k": aq._CHUNK_TOP_K,
        "gold_file": str(args.gold), "variant": args.variant, "no_synth": bool(args.no_synth),
        "knobs": {
            "KS_MQ_FANOUT": _os.getenv("KS_MQ_FANOUT", "1"),
            "KS_MQ_N_SUBQ": _mq._N_SUBQ, "KS_MQ_SUB_CHUNK_TOP_K": _mq._SUB_CHUNK_TOP_K,
            "KS_MQ_MAX_CHUNKS_PER_PAPER": _mq._MQ_MAX_CHUNKS_PER_PAPER,
            "KS_MQ_ENABLE_RERANK": _mq._MQ_ENABLE_RERANK, "KS_MAX_TOTAL_TOKENS": _mq._MAX_TOTAL_TOKENS,
            "KS_SYNTH_STRICT_REFUSAL": _os.getenv("KS_SYNTH_STRICT_REFUSAL", "1"),
            "KS_RERANK_MAX_LENGTH": _os.getenv("KS_RERANK_MAX_LENGTH", "4096"),
            "KS_MQ_CITATION_PRIOR": _os.getenv("KS_MQ_CITATION_PRIOR", "0"),
            "KS_MQ_CITATION_LAMBDA": _os.getenv("KS_MQ_CITATION_LAMBDA", "0.5"),
        },
    }
    print(f"EVAL RUN | tag={args.tag} | ws={ws} | variant={args.variant} | {len(gold)} questions -> {out_path}", flush=True)

    rag = await get_graph()
    _reset_rerank_fails()  # this run's rerank-failure count must reflect only this run (not a stale process global)

    # Concurrency: with the V-MQ rework sub-queries no longer call the LLM (pre-supplied keywords),
    # so questions can run concurrently without bursting the MiMo pool. --concurrency 1 = the
    # original sequential behaviour (baseline). Results are written back in gold order.
    sem = asyncio.Semaphore(max(1, args.concurrency))
    done = 0
    n = len(gold)

    async def _do(idx: int, g: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        nonlocal done
        async with sem:
            rec = await run_one(rag, g, base_meta, variant=args.variant, no_synth=args.no_synth)
        done += 1
        err = rec["meta"].get("error")
        print(
            f"  [{done}/{n}] {g['qid']} | cov={rec['kb_coverage']} | "
            f"chunks={len(rec['data']['chunks'])} | cited={len(rec['cited_papers'])}"
            + (f" | fused={rec['meta']['fused_applied']}" if "fused_applied" in rec["meta"] else "")
            + (f" | ERROR {err}" if err else ""),
            flush=True,
        )
        return idx, rec

    pairs = await asyncio.gather(*[_do(i, g) for i, g in enumerate(gold)])
    pairs.sort(key=lambda t: t[0])
    records = [rec for _, rec in pairs]
    with out_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"EVAL RUN DONE | wrote {len(records)} records -> {out_path}", flush=True)

    # Integrity gate (2026-06-07): the reranker can OOM on a contended GPU; LightRAG SWALLOWS
    # that and serves UNRANKED chunks silently. lightrag_init now counts terminal rerank
    # failures — if any fired, this run's ranking-sensitive metrics are DEGRADED, so surface it
    # LOUD rather than let a silent-degraded baseline be trusted. Per-question degradation is
    # also stamped into each rec["meta"]["rerank_degraded"] inside run_one (exact at
    # concurrency=1; approximate under concurrency>1 since the counter is process-global).
    nfail = _rerank_fail_count()
    if nfail:
        n_q_degraded = sum(1 for r in records if r.get("meta", {}).get("rerank_degraded"))
        print(
            f"⚠️  RERANK DEGRADED: {nfail} rerank call(s) failed (CUDA OOM / error) → "
            f"{n_q_degraded} question(s) served UNRANKED chunks. This run is NOT trustworthy for "
            f"ranking-sensitive metrics — free GPU0 (or lower KS_RERANK_MAX_ASYNC) and re-run.",
            flush=True,
        )

    # Convenience deterministic vector (only if every gold entry has gold_keys present).
    if all("gold_keys" in g for g in gold):
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from papervault.eval import backbone  # type: ignore

        gold_by_qid = {g["qid"]: g for g in gold}
        out = backbone.evaluate_run(records, gold_by_qid)
        print("DETERMINISTIC BACKBONE VECTOR (convenience; canonical scoring = stats.py):", flush=True)
        print(json.dumps(out["aggregate"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Downstream eval runner (l0_probe only).")
    p.add_argument("--gold", required=True, help="path to gold.jsonl")
    p.add_argument("--tag", required=True, help="run tag → results/<tag>.jsonl")
    p.add_argument("--limit", type=int, default=0, help="only the first N gold questions (smoke)")
    p.add_argument("--variant", default="baseline", choices=["baseline", "multiquery", "standard"],
                   help="retrieval variant: 'baseline' (single mix query), 'multiquery' (V-MQ "
                        "RAG-Fusion, gated), or 'standard' (locked modern-RAG baseline: anchor+facets, "
                        "no reranker, RRF+round-robin merge to top-80, no coverage gate)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="how many gold questions to run concurrently (V-MQ sub-queries are LLM-free, so >1 is safe)")
    p.add_argument("--no-synth", action="store_true",
                   help="FAST tier: skip the synth LLM call (answer=None). Keeps the deterministic "
                        "backbone (@served / gold_citation_recall / hallucinated_rate) + cited_papers "
                        "+ kb_coverage; ~4x faster retrieval-lever screening. No judge metrics.")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
