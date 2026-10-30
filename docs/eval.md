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
  .venv/bin/python -m papervault.eval.run_eval --gold src/papervault/eval/gold_fast.jsonl \
      --tag mytag --variant multiquery --concurrency 4
  ```

  The harness runs its own process against the GPU, databases, populated vault and an
  LLM endpoint. Run it in a stopped-service maintenance window: it refuses when
  `papervault.service`, `mineru.service`, or `papervault-mineru.service` is active or
  activating. The databases stay available. `KS_EVAL_ALLOW_COTENANCY=1` deliberately
  overrides that guard; it does not establish a safe GPU budget or freeze the corpus.
  It is the gate you run before shipping a change that could move retrieval
  quality (reranker dtype, chunking, decompose prompts, model routing).

  **Workspace gate.** `run_eval` hard-aborts unless `NEO4J_WORKSPACE == POSTGRES_WORKSPACE ==
  'l0_probe'` (or `'l0'` with `KS_ALLOW_PROD_WORKSPACE=1`, a read-only baseline against a
  production graph). These names are fixed in the harness — your serving workspace (e.g. the
  `main` shipped in `.env.example`) is **not** accepted, so set the eval workspace explicitly as
  above and populate it with the corpus you want scored.

Production `KS_QUERY_VARIANT` defaults to `multiquery`, whose decomposition uses
`PAPERVAULT_LLM_DECOMPOSE` (default `gpt-6.1-terra@high`), as does eval's `--variant multiquery`.
Eval selects its variant with `--variant` (`baseline` by default, also `multiquery`
and `standard`); `KS_QUERY_VARIANT` does not override that CLI choice. Specify the
variant explicitly when comparing production retrieval. Most query settings below
are read at module import, so restart the eval or serving process after changes.

`eval/run_eval.py` shares production's post-retrieval synthesis and response assembly
in `knowledge/query/aquery.py` (`synthesize_outfeed` and the pure `build_outfeed`).
Synthesized results separate `retrieved_papers` from `cited_papers`: the latter
contains only exact inline `[paper_key]` citations verified against retrieved evidence.
In FAST `--no-synth` mode, `cited_papers` explicitly falls back to `retrieved_papers`
for recall scoring. Raw retrieved chunks and references remain in eval output even
when coverage is empty or the answer cites only a subset. Empty coverage skips
synthesis; exclusively `abstract_only` evidence is capped at `"weak"`. The shared
`query_receipt` records the generation (when catalog-pinned), variant, coverage,
retrieved/cited counts and synthesis observations.

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
paper chunks, which the repo's no-paper-data gate forbids (PRD DoD #7). Generated artifacts live
under `PAPERVAULT_EVAL_DIR`, defaulting to `/data/paper-vault/eval-results/`: result JSONL files
are `<root>/<tag>.jsonl`, and filled prompts/judge seeds are `<root>/judge/<tag>/`. This matches
the existing external archive. Bundled gold sets and prompt templates remain in
`src/papervault/eval/`. The shared path resolver rejects artifact paths inside any `src/`
directory, including symlink targets, and paths escaping the configured artifact root.
Set `PAPERVAULT_EVAL_DIR` in the process environment before running the harness to select another
external location; resolution alone creates no directories. Arbitration verdicts live on GitHub:
quote the summary numbers on the relevant issue thread — never commit the raw results file.

## Operator hygiene

Keep environment backups outside the checkout in `/home/leon/backups/papervault-env/`, with
directory mode `700` and file mode `600`. `papervault doctor` warns when `.env.bak*` entries
remain at the repository root, without opening or printing their contents. Retain backups until
their owner decides their disposition; do not put credentials in issues or PRs.

`python scripts/sweep_worktrees.py --repo /path/to/papervault` reports eligible worktrees without
changing anything. After the workers have stopped and their disposable worktrees have been
reviewed, add `--execute` to remove clean, registered direct children of the primary checkout's
`.claude/worktrees/` and `.agents/worktrees/`. The sweeper checks tracked, untracked, and ignored
files, skips locked/current/detached/unknown paths, and requires ancestry in the locally fetched
`origin/main` or a merged PR targeting `main` for the exact current branch head in the same
repository. Fetch `origin/main` before the audit; unavailable PR evidence preserves the worktree.
It uses non-forced `git worktree remove` and `git worktree prune`, retaining all local and remote
branch refs. A removed clean worktree can be recreated with `git worktree add <path> <branch>`.

`python scripts/snapshot_catalog.py --output /path/to/catalog-snapshot.json` selects the snapshot
destination explicitly. The default is `/data/paper-vault/snapshots/catalog-snapshot.json`, outside
agent job scratch and the checkout; repeated runs replace that snapshot, so choose a dated
`--output` when retaining multiple baselines. Snapshot collection still reads the catalog directly
and opens database connections read-only. MDPI browser downloads remove their temporary PDF on
every exit, and test-owned temporary vaults use pytest's cleanup lifecycle.

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
| `KS_MQ_MIN_COVERAGE` | `thin` | Minimum anchor coverage required for multi-query fusion; below it, retain anchor evidence. Empty coverage always skips synthesis. |
| `KS_MQ_CITATION_PRIOR` | `0` | Citation-count prior on fused ranking (`0` = off). |
| `KS_MQ_CITATION_LAMBDA` | `0.5` | Weight of the citation prior when it is on. |
| `KS_CITATION_MAP` | unset | Override the citation-map path; otherwise use `PAPERVAULT_CITATION_MAP`, default `<PAPERVAULT_DATA>/citation_map.json`. Used by the optional citation prior. |
| `KS_MQ_FANOUT` | `1` | Adaptive fan-out on/off (`KS_MQ_FANOUT_MAX`=100, `KS_MQ_FANOUT_SLOPE`=7 shape it). |
| `KS_SYNTH_STRICT_REFUSAL` | `1` | Refuse to answer on thin/absent evidence (`0` = permissive). |
| `KS_SYNTH_COVERAGE` | `0` | Coverage-oriented synthesis variant (`1` = on). |
| `KS_RERANK_MAX_LENGTH` | `4096` | Per-chunk cross-encoder token limit; shortening can change ranking quality. |
| `KS_KW_PROMPT_COMPAT` | `0` | Restore the historical LightRAG keyword prompt for experiments; stock prompt is the shipped path. |
| `PAPERVAULT_SEARCH_AUTHORITY_PRIOR` | `1` | Enable authority in both `SearchPipeline.fusion` local BM25/RRF pool selection (45 local + up to 45 additional externally vouched records) and final `by_relevance` RRF ordering. Authority is citation velocity (citations/year of age, raw citations when year is unavailable) multiplied by a three-layer domain-aware venue classifier: normalized exact/fuzzy names, DOI prefixes, then fallback evidence, with 1.5x / 1.2x / 1.0x / 0.5x tiers. Exact `1` enables; `0` disables. Explicit recency/importance bypasses both priors; no prior can admit a gate reject. |
| `PAPERVAULT_SEARCH_AUTHORITY_LAMBDA` | `0.25` | Shared λ / `authority_weight` for local fusion and the final score `1/(60 + relevance_rank) + λ/(60 + authority_rank)`. Zero removes the added authority RRF terms at both boundaries; the base relevance ordering still uses authority after score and external-evidence ties. Non-numeric → `0.25`; out-of-range → clamped to `[0,1]`, both with a warning. Record the flag and λ explicitly for each arbitration arm; see [ADR-0008](adr/0008-search-authority-venue-and-garbage-fencing.md). |

Changing any of these is a retrieval-affecting change: it must clear the eval noise floor before
you flip the default.

Operational deadlines, concurrency and harness-specific settings also need to be
held constant between arms:

| Env var | Default | Effect |
| --- | --- | --- |
| `KS_DECOMPOSE_TIMEOUT_S` | `300` | Production/eval decomposition deadline in seconds. |
| `KS_SYNTH_TIMEOUT_S` | `840` | Shared synthesis deadline in seconds. |
| `KS_SYNTH_MAX_RETRIES` / `KS_SYNTH_BACKOFF_BASE` | `2` / `2.0` | Extra synthesis attempts and exponential-backoff base in seconds. |
| `KS_RERANK_DTYPE` / `KS_RERANK_BATCH_SIZE` | `float32` / `32` when unset; template `float16` / `8` | Reranker precision and GPU batch size. Precision changes need arbitration. |
| `KS_RERANK_MAX_ASYNC` / `RERANK_TIMEOUT` | `1` / `480` | Concurrent rerank calls and LightRAG worker timeout in seconds. |
| `PAPERVAULT_EVAL_DIR` | `/data/paper-vault/eval-results` | Root for raw runs and judge artifacts; never a source directory. |
| `KS_EVAL_ALLOW_COTENANCY` | unset | Exact `1` bypasses the stopped-service guard; workspace checks still apply. |
| `PAPERVAULT_LLM_SYNTH` | `gpt-6-astra@xhigh` | Evidence-grounded synthesis route, shared by production and eval. |
| `PAPERVAULT_LLM_DECOMPOSE` / `PAPERVAULT_LLM_EVAL_JUDGE` | `gpt-6.1-terra@high` / `glm-5.3,gpt-6.1-terra@high` | Decomposition and eval-judge routes (`eval_judge` fails over from `glm-5.3` to `gpt-6.1-terra@high` on transient upstream errors with `PAPERVAULT_LLM_FALLBACK_COOLDOWN_S=120`). |

The retained standalone `judge_mimo` utility uses `KS_GATEWAY_URL` (default
`http://127.0.0.1:4000/v1`) and requires `KS_VIRTUAL_KEY`; these utility names are
separate from the production `PAPERVAULT_LLM_GATEWAY_*` transport settings. It also
reads `JUDGE_GOLD` (`gold_v2.jsonl`), `JUDGE_MAX_TOKENS` (`32000`),
`JUDGE_MAX_ATTEMPTS` (`2`) and `JUDGE_CALL_TIMEOUT` (`300` seconds). Its
`--concurrency` CLI option controls concurrency. `_ab_u1_judge` shares those gateway
settings and reads `AB_JUDGE_CONC` (`20`). `recall_curve` accepts
`RECALL_CURVE_GOLD` (`gold_v4.jsonl`); `verdict_lctx` accepts `VERDICT_GOLD`
(`gold_v4.jsonl`), `VERDICT_BASE_TAGS` (`lctx_v4_r1,lctx_v4_r2,lctx_v4_r3`) and
`VERDICT_VAR_SEEDS` (`1,2,3`). Set those selectors explicitly for reproducible reruns
of historical experiments. Do not confuse their historical defaults with the gold
file or tags selected for a new arbitration.
