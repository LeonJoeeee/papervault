# Evaluation

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
  python -m papervault.eval.run_eval --gold src/papervault/eval/gold_fast.jsonl \
      --tag mytag --variant multiquery --concurrency 4
  ```

  This needs the full stack up (`papervault serve` + GPU + databases + a populated vault) and
  an LLM endpoint. It is the gate you run before shipping a change that could move retrieval
  quality (reranker dtype, chunking, decompose prompts, model routing).

## Gold sets

The bundled gold sets (`gold_fast.jsonl` = FAST arm, `gold_v4.jsonl` = FULL arm, plus trap
sets) are **space physics** — they match the factory domain pack. Adapting papervault to another
domain means authoring a gold set for that field; the harness itself is domain-agnostic.

## Arbitration discipline

Retrieval-affecting changes are flag-gated and benchmark-arbitrated: run the same gold + config
before and after, same day (the corpus drifts as auto-ingest runs), and only flip the default
if the recall delta clears the measured noise floor. This is how the reranker fp16 flip and the
build-model switch were decided upstream.
