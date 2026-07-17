# Root-cause: why @served is stuck at 0.78 (the recall ceiling)

Date: 2026-06-18 · Owner: KS · Status: DIAGNOSIS COMPLETE (data-backed) — fix direction pending user GO.
Frame: gold_v4 (47 answerable Qs / 232 gold (qid,key) pairs), live baseline lctx@60 (multiquery + V-SR +
60 served chunks), prod `l0` read-only. Supersedes the survey's "recall is graph-limited" framing with
a sharper, measured split.

## Question
The headline `harm(@served, nugget)=0.7825` is bottlenecked on @served=0.784 — **22% of gold papers
never reach synth**. The optimization survey concluded "recall is graph-limited, needs a rebuild
(human-gated)". Before spending a rebuild, this drill establishes WHY those papers are absent.

## Method (all deterministic / read-only)
1. From the 3 baseline dumps (`lctx_v4_r{1,2,3}`), computed the gold papers NEVER served across all 3
   runs → **38 unique papers** (43 (Q,paper) miss-pairs over 24 of 47 answerable Qs). `/tmp/never_served_gold.json`.
2. Ingestion + density check (`aget_docs_by_ids` over prod l0): status + chunks_count for all 38.
3. Stub check (PG `lightrag_doc_chunks`): total chunk chars per paper.
4. **Reachability probe**: re-ran retrieval for every needing-Q with a WIDE net (top_k=200,
   chunk_top_k=400, rerank OFF) to see the RAW recall pool (~160-278 papers / 400 chunks per Q), and
   checked whether each missed paper appears ANYWHERE in it. ⚠️ First pass was wrong — I forgot to lift
   `max_total_tokens`, so it defaulted to 30000 and truncated the returned chunks to ~6 (the round-2
   bug), making everything look "not recalled". Corrected with `max_total_tokens=2_000_000`.

## Findings

### F1 — It is NOT an ingestion or extraction problem. A content rebuild would not help.
- **38/38 are ingested** (`status=processed` in doc_status). ZERO never-ingested.
- **0/38 are stubs** — all have ≥6000 chars of real chunk content. The low-chunk papers (Adriani2011=2,
  Parker1966=3, Lagaris1997=4) are genuinely SHORT papers (PRL letters / ICRC proceedings), not
  truncated extractions. Re-extraction / re-chunking has nothing to recover.
- ⇒ The earlier "needs a graph rebuild" conclusion is **wrong for the content**: the content is all
  present, indexed, and embedded. The miss is entirely on the RETRIEVAL side.

### F2 — The 22% splits into two distinct retrieval failures (measured, not hypothesized)
Reachability probe over all 24 needing-Qs, per unique paper (reachable = recalled for ≥1 needing-Q in
the 400-chunk wide net):

| class | n papers | what it means | lever class |
|---|---|---|---|
| **IN-POOL but cut** | **18 / 38 (47%)** | recalled into the 200-400 wide pool, but cut before the 60 served | RANKING — no rebuild |
| **NEVER recalled even wide** | **20 / 38 (53%)** | absent even at a 6.6× wider net (400 vs 60 chunks) | RECALL MODALITY — no content rebuild |

(By (Q,paper) pair: 20 in-pool-cut / 23 never-recalled of 43.)

- **IN-POOL (18):** Adriani2011, Aguilar2015, Aslam2019, Cai2021, Castellina2019, Corti2018, Ferreira2000,
  Florinski2009, Florinski2013a, Galaris2022, Karniadakis2021, Malkov2011, Moskalenko1998, Sciascio2016,
  Tomassetti2023, Tomassetti2025, Torre2012, Wang2019. These are recalled but lose the RRF→60-cut race.
- **NEVER RECALLED (20):** Abdollahi2017, Aguilar2018a, Aloisio2015, Blandford1987, Chen2020, Chen2023,
  Dembinski2017, Dong2020, Goswami2022, Lagaris1997, Martucci2023, Maurin2019, Nayek2021, Parker1958,
  Parker1965, Parker1966, Raissi2019, Rathore2024, Richardson1999, Stone2013.

### F3 — The never-recalled set is dominated by FOUNDATIONAL papers, independent of chunk count
- Classics that DEFINE a concept: Parker1958/1965/1966 (transport eq.), Lagaris1997 + Raissi2019 +
  Karniadakis2021 (PINN foundations), Blandford1987 (DSA). They are never recalled for the modern,
  SPECIFIC queries that need them.
- It is NOT a density issue: chunk-rich papers are also never recalled (Dong2020=28, Blandford1987=25,
  Rathore2024=17, Chen2020=17, Raissi2019=12). Plenty of indexed content, still invisible to retrieval.
- Mechanism: **the paper that DEFINES a concept is semantically/lexically distant from the modern query
  that USES it.** A specific 2024 query ("PINN training failure modes + remedies") embeds far from the
  general 1997/2019 foundational text; and in a 3671-paper corpus, hundreds of recent papers that USE
  the concept out-rank the one that introduced it. Vector + entity-graph retrieval both miss it.

## Implications
The recall ceiling is a **retrieval problem in two flavors, BOTH fixable without a content rebuild**:
- 47% (in-pool) is a RANKING problem — cheap, no rebuild.
- 53% (never-recalled) is a RECALL-MODALITY gap — needs a retrieval arm the current vector+entity-graph
  stack lacks (lexical/exact-name, or citation-graph expansion), still no content rebuild.
A full graph rebuild (the human-gated "headline lever") is **confirmed off-target** — it would re-process
content that retrieval already fails to reach.

## Candidate fixes (menu — none built yet)
- **A. Ranking recovery for the 18 in-pool (cheapest, no rebuild, test FIRST).**
  - Per-paper chunk-diversity cap (`_MQ_MAX_CHUNKS_PER_PAPER`, currently OFF): if the 60 served are
    dominated by a few chunk-rich papers, capping per-paper frees slots for the cut-but-present papers.
  - Rerank/RRF/cut tuning. ⚠️ NOT "serve more than 60" — round 2 showed 60 is the synth unimodal peak
    (headline turns down past 60). The fix is to RANK these into the 60, not widen it.
  - NEXT CHECK before building: the per-Q RANK DEPTH of the 18 (recoverable only if they sit ~rank
    60-150, not ~rank 380).
- **B. New retrieval modality for the 20 never-recalled (moderate code, no content rebuild).**
  - Lexical / BM25 / exact-name fallback arm (KS has NO lexical arm today — only vector+entity-graph).
    Directly catches canonical papers named (or whose concept is named) in the query.
  - Citation-graph expansion: pull papers CITED by the already-recalled modern papers — foundational
    papers are exactly the ones the recalled papers cite. (Needs a small spike: do we have citation
    edges in the graph / pl metadata to traverse?)
- **C. (heavier, only if A+B insufficient) re-embed with a stronger / instruction-tuned embedding** —
  this IS a partial rebuild (re-embed, no re-extract). Defer until A+B measured.

## Recommended next step
A/B the cheap ranking lever (A: per-paper diversity cap) on the FAST tier first (it targets the 18
in-pool papers, measurable as @served on gold_v4), and in parallel spike whether a lexical-name arm
recovers the 20 classics. Both are no-content-rebuild. Decide B vs C only after A is measured.

Artifacts: `/tmp/never_served_gold.json`, `/tmp/absent_diag.json`, `/tmp/reach_full.json`.
