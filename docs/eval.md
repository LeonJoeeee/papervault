# Evaluation

> Measured floors live in [`docs/baselines.md`](baselines.md); this file is HOW they are measured.

papervault ships its own eval harness (`src/papervault/eval/`) so retrieval quality is
**reproducible, not claimed**. It measures the knowledge `query` path against a gold set:
recall@served, citation recall, hallucination rate, and trap/refusal gates.

## Two tiers, deliberately

Retrieval quality has a hard dependency on the GPU stack (embedding + reranking) and a live
corpus, so the harness splits into what CI can prove and what only reference hardware can:

- **CI (every push):** the metric/aggregation **logic** — backbone parsing, `metric_reffree`,
  `stats`, `headline`, `judge_aggregate` — runs as unit tests with fixed inputs. This guards
  the scoring code from regressions. It does **not** run retrieval (no GPU on CI runners).
- **Reference machine (manual gate):** the end-to-end recall run against a real corpus:

  ```bash
  # The harness runs only against the reserved eval workspace 'l0_probe' (a prod-safety
  # gate). Point BOTH workspace envs at it for the run, or run_eval aborts.
  export NEO4J_WORKSPACE=l0_probe POSTGRES_WORKSPACE=l0_probe
  python -m papervault.eval.run_eval --gold src/papervault/eval/gold_fast.jsonl \
      --tag mytag --variant multiquery --concurrency 4
  ```

  This needs the full stack up (`papervault serve` + GPU + databases + a populated vault) and
  an LLM endpoint. It is the gate you run before shipping a change that could move retrieval
  quality (reranker dtype, chunking, decompose prompts, model routing).

  **Workspace gate.** `run_eval` hard-aborts unless `NEO4J_WORKSPACE == POSTGRES_WORKSPACE ==
  'l0_probe'` (or `'l0'` with `KS_ALLOW_PROD_WORKSPACE=1`, a read-only baseline against a
  production graph). These names are fixed in the harness — your serving workspace (e.g. the
  `main` shipped in `.env.example`) is **not** accepted, so set the eval workspace explicitly as
  above and populate it with the corpus you want scored.

## Gold sets

The bundled gold sets (`gold_fast.jsonl` = FAST arm, `gold_v4.jsonl` = FULL arm, plus trap
sets) are **space physics** — they match the factory domain pack. Adapting papervault to another
domain means authoring a gold set for that field; the harness itself is domain-agnostic.

## Arbitration discipline

Retrieval-affecting changes are flag-gated and benchmark-arbitrated: run the same gold + config
before and after, same day (the corpus drifts as auto-ingest runs), and only flip the default
if the recall delta clears the measured noise floor. This is how the reranker fp16 flip and the
build-model switch were decided upstream.

**Evidence convention.** Eval result files stay **local and untracked** — they embed verbatim
paper chunks, which the repo's no-paper-data gate forbids (PRD DoD #7), so the results dir
(`src/papervault/eval/results/`, plus the judge scratch dir) is gitignored by design. Arbitration
verdicts therefore live on GitHub: quote the summary numbers on the relevant issue thread — never
commit the raw results file.

## Retrieval tuning knobs (benchmark-arbitrated)

The query path is tuned by a family of environment variables — exactly the levers the
arbitration discipline above governs. Change one, re-run the same gold + config before and
after, and keep it only if the recall delta clears the measured noise floor. Defaults below are
the shipped values.

| Env var | Default | Effect |
| --- | --- | --- |
| `KS_QUERY_VARIANT` | `multiquery` | Retrieval variant selector (the shipped path). |
| `KS_TOP_K` | `100` | Entity/relation top-k pulled from the graph. |
| `KS_CHUNK_TOP_K` | `60` | Chunk top-k pulled for context. |
| `KS_MAX_TOTAL_TOKENS` | `300000` | Hard context-token budget for the assembled prompt. |
| `KS_MQ_N_SUBQ` | `5` | Sub-queries the decomposer fans a question into. |
| `KS_MQ_RRF_K` | `60` | Reciprocal-rank-fusion constant when merging sub-query hits. |
| `KS_MQ_SUB_CHUNK_TOP_K` | `60` | Chunk top-k per sub-query before fusion. |
| `KS_MQ_ENABLE_RERANK` | `true` | Rerank the fused chunk set on the GPU. |
| `KS_RERANK_POOL_CAP` | `120` | Prefix-cap the candidate pool fed to the cross-encoder (`0` = off; never caps below the requested top_n). Arbitrated 2026-07-18: 120 is recall-neutral at 1.90× retrieval speed (issue #21). |
| `KS_MQ_MAX_CHUNKS_PER_PAPER` | `0` | Per-paper chunk cap (`0` = no cap; raise to diversify served papers). |
| `KS_MQ_MIN_COVERAGE` | `thin` | Coverage-gate floor before synthesis. |
| `KS_MQ_CITATION_PRIOR` | `0` | Citation-count prior on fused ranking (`0` = off). |
| `KS_MQ_CITATION_LAMBDA` | `0.5` | Weight of the citation prior when it is on. |
| `KS_MQ_FANOUT` | `1` | Adaptive fan-out on/off (`KS_MQ_FANOUT_MAX`=100, `KS_MQ_FANOUT_SLOPE`=7 shape it). |
| `KS_SYNTH_STRICT_REFUSAL` | `1` | Refuse to answer on thin/absent evidence (`0` = permissive). |
| `KS_SYNTH_COVERAGE` | `0` | Coverage-oriented synthesis variant (`1` = on). |
| `PAPERVAULT_SEARCH_AUTHORITY_PRIOR` | `0` | `search_papers` return ordering: soft RRF rank-blend of relevance with an age-normalized citation **authority** rank (`0` = off). **Arbitration pending (issue #46/#37) — default stays off until the paired coverage+precision run clears the noise floor** (P@served must hold the 0.9111 floor; must-find recall should rise). RANK-based, pool-only, never raw counts. |
| `PAPERVAULT_SEARCH_AUTHORITY_LAMBDA` | `0.5` | Weight λ of the authority prior when it is on (parsed defensively: non-numeric → 0.5, out-of-range → clamped to `[0,1]`, both with a warning). Arbitration runs must record the flag+λ per arm MANUALLY — the `pl` search path has no auto meta-stamp. |
| `PAPERVAULT_CANON_RESERVED_SLOTS` | `0` | `search_papers` return: reserve N of the served-15 for the highest RAW-citation, intent-relevant papers (issue #37 canon lever). A HARD floor complementary to the #46 soft prior — surfaces buried OLD high-cite landmarks (RAW citations, not age-normalized). Pool-only, minimal displacement (one swap per buried classic). `0` = off. **Arbitration pending — default stays off until the paired coverage+precision run shows coverage RISES and P@served HOLDS.** Parsed defensively (non-int/negative → 0). |
| `PAPERVAULT_SEARCH_NO_INGEST` | `0` | Frozen-corpus eval switch: `search_papers` skips its Stage-3 write path (no upsert / no download enqueue / no `library.save()`), so search is READ-ONLY and `return_pool` collapses to the in-library candidates. NOT a production path (it drops fresh external discoveries) — its ONLY use is a byte-frozen library during a paired lever arbitration. `0` = off (normal ingest-on behavior). |

Changing any of these is a retrieval-affecting change: it must clear the eval noise floor before
you flip the default.
