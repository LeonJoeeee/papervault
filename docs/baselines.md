# papervault — Baselines

The measured floors and envelopes the PRD's definition-of-done points at ("published
floor", "published budget"). Every number here is a real measurement with a date and
conditions — never an aspiration. Organized by the caller-facing businesses; component
instruments and arbitration rules at the end.

**Reference conditions** (all numbers below unless stated): lab deployment #1 — RTX 3090
24 GB (`CUDA_VISIBLE_DEVICES=1`), fp16 reranker @ max_length 4096, rerank 2×4
(`KS_RERANK_MAX_ASYNC=2 × KS_RERANK_BATCH_SIZE=4`), `RERANK_TIMEOUT=480`, LightRAG 1.5.4,
corpus ≈ 5.2k papers (≈ 4.4k full-text distilled), models mimo-v2.5-pro (synth slot) /
mimo-v2.5 (build slot) via the LiteLLM gateway (12-key pool).

## Business: `query` (ask the knowledge base)

| metric | value | measured |
| --- | --- | --- |
| recall@served (distinct papers, FAST gold, --no-synth) | **0.8568** | 2026-07-18, frozen corpus, tag `frozen154_07180011` |
| — same-window 1.4.16 reference | 0.8046 | same frozen window (papervault **+5.2 pp**) |
| noise floor (run-level, historical) | 0.0132 | the flip/no-flip bar for retrieval changes |
| hallucinated citation rate | **0.0** | every eval arm to date |
| single-query end-to-end latency (with synthesis) | **172 s** (pre-cap); pool-cap 120 halves the retrieval stage (arbitrated 1.90×, issue #21) — live number to be re-measured | 2026-07-18 |
| synthesis stage alone | 55–79 s | prompt ≈ 200k tokens |
| long-call survival | ≥ 485 s proven | S17 progress heartbeat (45 s ticks) kept a deliberately-strict 120 s-SSE client alive |

Honest note: at the *retrieval* level trap questions still serve content
(trap_violation 1.0 in --no-synth arms, historical constant); refusal is enforced at the
synthesis layer. The e2e honesty gate lives there, not in retrieval.

## Business: `search_papers` (discover literature)

| metric | value | measured |
| --- | --- | --- |
| end-to-end latency (incl. ingest triage of new finds) | 319 s mean, 222–525 s range | 2026-07-18, 12-intent baseline run (supersedes the 216–251 s 2-smoke figure) |
| must-find recall (coverage: fraction of the frozen landmark pool the search surfaces; `scripts/mustfind_recall.py` vs `gold_search_pools.jsonl`) | **0.1806 mean / 0.00 min** | 2026-07-18 night, single-shot v1 (serving varies run-to-run — treat as ±0.1 until a 3-rep baseline lands); results `mfr_v1_0718b` |
| result relevance (P@served: fraction of served papers the canonical judge scores ≥ 0.7 against the stated need, title+abstract) | **0.9111 mean / 0.7333 min** | 2026-07-18; **canonical judge = Claude opus blind panel, reasoning effort MAX** (12 fresh-context agents, one intent each, rubric-only, no access to other judges' scores); results `srb_v1_0718_claudejudge_max.json` |

The coverage/precision split is the search business's honest portrait: 91 % of what it
serves is relevant, but it surfaces fewer than 1 in 5 of the landmark papers a review
would demand — it serves the topical frontier and misses the canon (validated by hand:
an intent NAMING PAMELA/AMS got 15 modern analysis papers, zero of the original
measurement papers). Improvement lever tracked separately.

Judge triangulation (all on the same 180 served papers, `srb_v1_0718`): MiMo strong-slot
0.9056, opus default-effort 0.9444, opus max-effort 0.9111 — spread ±0.03, opus
default-vs-max self-agreement 170/180. Treat ±0.03 as judge-side measurement noise on this
instrument. Regime (user decision 2026-07-18): the opus max-effort blind panel is the
CANONICAL judge for baselines and arbitration; the MiMo judge (in
`scripts/search_relevance_baseline.py`) remains a cheap dev-time smoke that never decides.
Known boundary behavior: adjacent-subtopic serving is where judges disagree (UHECR intent:
0.60 MiMo / 0.73 canonical / 1.00 opus-default).

## Business: `get_paper` (lookup + full text)

| metric | value | measured |
| --- | --- | --- |
| latency (batch of 2, key + DOI forms) | **≈ 0 s** | every smoke |
| identifier forms | key / DOI / arXiv / fuzzy — all resolve | 2026-07-16 caller test |

This business is frozen: no optimization budget; regression watch only.

## Business: freshness pipeline (search → queryable)

| metric | value | measured |
| --- | --- | --- |
| search → full text on disk (download + OCR) | ≈ 10 min | 2026-07-16 live probe (5 papers) |
| search → queryable knowledge (distilled into graph) | ≈ 20–30 min | scheduler rounds every 60 s; distill on arrival |
| pipeline health under load | 342 consecutive clean rounds, 0 errors | post-cutover observation window, 2026-07-18 |

## Build plane (not caller-facing; cost envelope)

| metric | value | measured |
| --- | --- | --- |
| extraction yield (sample: 218k-char paper) | 25 chunks → 434 entities + 547 relations | 2026-07-18 probe, LightRAG 1.5.4 |
| ontology purity (fresh 1.5.4 build) | 434/437 in the closed 11-type set, **zero `Other`** | l0_probe; 3 UNKNOWN = framework relation-endpoint placeholders (0.56 % pre-exists in prod) |
| distill throughput | ≈ 1 paper/min under the 2-key era throttle | 2026-07-17 night; **remeasure with the 12-key pool** |

## Resource envelope

| metric | value | measured |
| --- | --- | --- |
| co-active VRAM peak (4-way OCR + concurrent queries) | **18.8 / 24 GB** | 2026-07-16 stress test |
| held-in-reserve levers | MinerU gmu 0.40→0.30; rerank batch 4→smaller | never needed |

## Arbitration rules (how these numbers may change)

1. **Quality is a hard constraint**: a retrieval-affecting change flips its default only
   if the paired recall delta stays within the 0.0132 noise floor (or improves).
2. **Arbitration window = corpus-freeze window**: pause ingestion (stop the service or
   disable auto-ingest) for the duration of any paired eval — live distills contaminated
   a verdict once (2026-07-18; see the papervault issue #4 thread).
3. **Same-day paired, identical knobs both arms** (`KS_RERANK_MAX_LENGTH=4096` explicit —
   historical drivers defaulted to a wrong 1024).
4. **Component instruments** (the eval harness in `src/papervault/eval/`: gold sets,
   backbone, judges, FULL arm) are the sensitive tools for small-delta arbitration; this
   card is the product-level truth. See `docs/eval.md` for the two-tier regime.
5. **Paired AND interleaved, ≥ 3 reps/arm** for FAST-gold arbitration: within-arm rep
   spread (LLM-decompose non-determinism) measured ~3× the historical run-level noise
   floor (2026-07-18 rerank pair: one arm spanned 0.8169–0.8537 across reps), so a
   single-shot A/B is not decisive; per-rep paired deltas are.
