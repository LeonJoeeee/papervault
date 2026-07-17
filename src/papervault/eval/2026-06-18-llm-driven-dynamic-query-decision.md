# Decision: LLM-driven DYNAMIC query (one `n_sources` signal drives budget + decomposition)

Date: 2026-06-18 · Owner: KS · Status: v2 (budget) benchmarking; v3 (facets) queued.

## Context / problem
The downstream headline `harm(@served_distinct, nugget)` is bottlenecked on @served, and the @served
misses concentrate on BROAD / high-fanout questions (corr(gold_size, recall) = -0.61): their fixed
60-chunk / 5-facet budget can't surface the 8-14 distinct papers a thorough answer needs — a
single-facet gold paper gets outvoted out of the global RRF 60-cut. A graph rebuild does NOT fix this
(@served is query-budget-bound, not graph-bound — validated 2026-06-16, see NIGHT_LOG).

First lever (`fcap` v1, word-count gate): widen the served + sub-pool budget 60->100 for intents
>75 words. 3-run gold_v4: headline 0.814 -> 0.837 = **+0.023, PASSES the 0.0132 noise floor** (the
only above-floor gain of the optimization push). But the DETECTOR was dumb (word count, corr~0.42
with true fanout) and binary.

## User's design (2026-06-18): make it LLM-driven, continuous, and whole-query-dynamic
"宽问题加预算显而易见,但'是否宽'该由 LLM 判,而且不该离散——根据任务动态调,一个光谱 + few-shots,
定一个标准。甚至整个都可以动态,包括查询的拆分。"

→ Replace the heuristic with ONE task-breadth signal estimated by an LLM, and let it drive BOTH the
retrieval budget AND the decomposition granularity.

## Architecture: one `n_sources` estimate → budget + facet count
A cheap, SEPARATE MiMo call (`_estimate_n_sources`, isolated from `_decompose` so it can never perturb
the regression-prone facet prompt) returns:

  n_sources = "how many DISTINCT library papers a THOROUGH, COMPLETE answer would need to draw on"

This is the task-driven breadth signal. Off-domain / unanswerable-from-this-library questions estimate
~1 (the corpus has nothing) — which is the built-in trap safety.

### The standard (rubric, in the prompt) + few-shots
- one fact / one method / one phenomenon            -> 2-4
- a mechanism spanning a few works                  -> 4-7
- broad synthesis / multi-way comparison / review /
  several competing hypotheses, species, regimes    -> 8-15
- off-domain (condensed-matter, chemistry, pure math
  — outside cosmic-ray/heliophysics/solar/space-wx + AI4Science) -> 1
Few-shots: "force-field single parameter" -> 3 ; "synthesize modulation/drift/diffusion across model
families" -> 12 ; "single-bubble sonoluminescence mechanism" -> 1. (Smoke-verified 2026-06-18: the LLM
returned exactly 3 / 12 / 1.)

### Continuous mappings (the "spectrum")
- BUDGET (v2, DONE): served_cap = sub_pool = `clamp(n_sources * 7, 60, 100)`.
  n_sources<=8 -> 60 (floor, unchanged); 12 -> 84; 14 -> 98; 15+ -> 100 (cap at the validated point,
  not 120 = the earlier dilution zone). Env: KS_MQ_FANOUT_SLOPE (7), KS_MQ_FANOUT_MAX (100).
- DECOMPOSITION (v3, QUEUED): n_facets = `clamp(round(n_sources * 0.6), 2, 8)`, trap (n_sources<=1)
  -> 1. narrow(3)->2, mid(5)->3, broad(12)->7, cap 8. Implemented by passing the variable n to the
  UNCHANGED `_decompose(intent, n=n_facets)` (decompose already takes n; "exactly {n}" produces it).

## Why v3 should NOT reproduce the earlier variable-facet regression
The 2026-06-14 decompose rework ("up to N, FEWER is correct, anti-under-split") REGRESSED trap -0.074:
it split by the question's SURFACE structure, so a trap naming two off-domain phenomena got extra
off-domain facets whose keywords matched real graph entities -> pushed the trap empty->thin -> fusion
fired -> tangential context -> synth answered. v3 ties facet count to n_sources (in-corpus breadth),
so a trap -> n_sources 1 -> 1 facet (anchor only) -> NO off-domain facets -> no leak. AND the decompose
PROMPT is untouched (only the n cap varies) — the earlier regression came from changing the prompt's
splitting instructions, which v3 does not do.

## Safety / reversibility
- `_estimate_n_sources` is a separate isolated call; `_decompose` prompt + `_parse_facets` unchanged.
- Flag-gated by KS_MQ_FANOUT. **PROMOTED to default "1" (ON) 2026-06-18** after the +0.0324 validation
  (was "0" while benchmarking); set KS_MQ_FANOUT=0 for the byte-identical 60/5 baseline. All knobs env-tunable.
- The downstream eval runs reranker max_length=1024 (KS_RERANK_MAX_LENGTH) to MATCH the fsdef baseline
  + stay feasible; the live reranker default is 4096 (separate live-quality choice).

## Validation plan (isolate each lever — the benchmark is the arbiter)
1. v2 (budget dynamic, facets fixed 5): 3-run gold_v4 vs fsdef. [RUNNING]
2. v3 (budget + facets both dynamic): 3-run vs fsdef AND vs v2 → attribute the facet-count effect
   (it regressed before, so it MUST be isolated). Build only AFTER v2 (cannot edit multiquery.py while
   v2's benchmark runs — fresh `uv` per seed would pick up v3 and corrupt v2).
Promote per the usual gate: headline > baseline by > k*0.0132, jackknife, all red-line gates
(esp. trap_correct_refusal) non-regressed. Keep flag-gated unless a clean win; enable by default
(KS_MQ_FANOUT=1) per the user's "有提升的部分直接进默认" once a 5-seed confirms past jackknife/trap noise.

## Status
- fcap v1 (word-count): committed `ab28a96`, flag-gated default OFF, validated +0.023.
- fcap v2 (LLM `n_sources` budget): **DEPLOYED, default-ON 2026-06-18** (KS_MQ_FANOUT default "1"),
  validated +0.0324 (passes noise floor + jackknife). The session's one robust quality win.
- fcap v3 (+ dynamic facets): designed (this doc), queued behind v2.

## Two-tier benchmark (user 2026-06-18: "应该有两个级别,一个十几分钟快速验证,一个全量")
Each full benchmark is ~3h (56 Qs x 3 seeds x synth + judge) — too slow to iterate. Split it:

- **FAST tier (~5-15 min) — directional filter for RETRIEVAL levers.** Fixed curated subset
  `gold_fast.jsonl` (14 Qs = 7 highest-fanout [fcap's target, where @served varies] + 4 narrow
  [saturation/sanity] + 3 traps), 1 seed, DETERMINISTIC backbone only (paper_recall_at_served_distinct,
  hallucinated_rate, gold_citation_recall — all computed from the retrieval dump vs gold, NO LLM judge).
  fcap/v3 effect IS @served (deterministic) so this measures the target directly. Run:
  `run_eval --gold gold_fast.jsonl --tag fast_<v>` then compute @served from results/fast_<v>.jsonl.
  Add `--no-synth` (TODO, after the in-flight v2) to skip the synth call -> ~5 min retrieval-only.
- **FULL tier (~3h) — promotion decision.** 56 Qs, 3 seeds, full MiMo judge -> headline + ALL red-line
  gates (trap_correct_refusal / nugget / cit_sp / faithfulness). REQUIRED before promoting, because a
  lever can lift @served yet break trap (e.g. fcap word-count's trap noise) — only the judge catches it.

Protocol: iterate variants on FAST (@served filter) -> only the survivors get a FULL run -> promote on
the full gates. Reranker max_length=1024 on both (match the fsdef baseline + feasible); live = 4096.

## v3 (dynamic facet count) — FAST-tier filtered OUT, NOT adopted (2026-06-18)
Fast tier (gold_fast, 1 seed, deterministic @served): v3 (dynamic facets + budget) @served = 0.830 vs
v2 (budget only) 0.858 vs fsdef 0.756. v3 does NOT beat v2 (~within 1-seed noise, slightly lower):
scaling facet COUNT by n_sources gives broad Qs more facets but mid Qs FEWER than v2's fixed 5, netting
no @served gain. Decision: KEEP v2 (simpler, robust, validated +0.0324); do NOT adopt v3's dynamic
facets (more complexity + the earlier variable-facet regression history, for zero fast-tier gain).
The two-tier protocol paid off: v3 filtered in ~14 min, no 3h full run wasted. The dynamic-query idea
nets out as: dynamic BUDGET (v2) is the win; dynamic facet COUNT (v3) is not.
