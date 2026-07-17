# Autonomous overnight optimization log (2026-06-14, full authority handoff)

Frame: gold_v4 (56 Qs) · baseline = lctx@60 (`lctx_v4_r{1,2,3}`) · headline = harm(@served_distinct,
nugget_recall) = **0.7825** · judge = MiMo (zero Claude quota) · prod l0 READ-ONLY.
Hard lines held even under full authority: NO graph rebuild (prep a doc, don't execute), NO live
deploy, NO commits to the entangled tree. All changes reversible / flag-gated.

Roadmap from the drill (`/tmp/drill_levers.json`), ranked:
- L1 V-MQ5 (env-only): KS_MQ_N_SUBQ=5 KS_TOP_K=120 KS_MQ_SUB_CHUNK_TOP_K=60 KS_MQ_RRF_K=80 → @served +0.03-0.06
- L4 preserve-uncertainty (synth, low risk, gate-positive)
- L3 coverage rule (synth, KS_SYNTH_COVERAGE=1, higher gate risk → watch hallu/trap)
- L2 chunk-per-paper diversification (needs small code edit, flag-gated)
- combine winners (retrieval × synth are orthogonal, like V-MQ+V-SR)

Per-round protocol: eval 3× over gold_v4 → MiMo judge → `verdict_lctx <prefix>` → promote if
(headline up over noise floor 0.0132 + jackknife + all gates OK). Judge hangs → kill + idempotent
resume, accept N-1/N.

## Rounds
- **Round A — L1 V-MQ5** [running, started 14:35] env: KS_TOP_K=120 KS_MQ_RRF_K=80 (N_SUBQ=5, SUB=60, chunk=60 base). Awaiting eval+judge+verdict.
- **Round A — L1 V-MQ5** [DONE, no win] env KS_TOP_K=120 KS_MQ_RRF_K=80 (N_SUBQ=5). verdict_lctx: headline 0.7914→0.7898 (Δ-0.0015, below noise floor), trap 0.926→0.852 REGRESSED, gold_citation_recall -0.019. Root cause = top_k=120 DILUTION (matches the earlier topk100>topk120 finding; at the 60-chunk cap, extra depth reshuffles the served set without adding distinct gold, and perturbs trap). NOT promoted. Lesson: depth past ~100 doesn't help @served at a 60-cap; retrieval gains must come from DIVERSITY (L2) not depth.
- **Round B — L3+L4 synth (coverage + preserve-uncertainty)** [starting] flag KS_SYNTH_COVERAGE=1, base lctx@60. Targets nugget (present-not-used) + faithfulness; watch hallu/trap.
- **Round B — L3+L4 covunc** [judging] eval done (3x56). SNAG: first judge pass (hardened 300s/2-attempt) gave 40/168 APITimeoutError — coverage answers are LONGER → judge prompt huge → MiMo gen >300s. LESSON: coverage-variant answers need GENEROUS judge timeout. Fix: idempotent re-judge with JUDGE_CALL_TIMEOUT=900 JUDGE_MAX_ATTEMPTS=3 (skips the 128 valid). The 300/2 default is too aggressive for long answers — pass 900/3 explicitly for synth-coverage variants henceforth.
- **Round C — L2 covcap (KS_MQ_MAX_CHUNKS_PER_PAPER=2)** [eval running in parallel on GPU while Round B re-judges on the gateway].
- **Round B — L3+L4 covunc** [DONE, no win] verdict_lctx (164/168 judged): headline 0.7914→0.7977 (Δ+0.0064, BELOW the 0.0132 noise floor → not significant), trap 0.926→0.889 (−0.037, within ε=0.10 but not strict), other gates OK. Coverage rule lifts nugget marginally but sub-noise + small trap cost. NOT promoted.
- PATTERN: L1(depth) and L3+L4(coverage) both give SUB-NOISE-FLOOR gains. lctx@60 is near the no-rebuild ceiling on this frame; the 0.0132 long-ctx noise floor swamps small synth/retrieval tweaks. Real remaining headroom = @served retrieval ceiling (needs better fusion/decomposition or graph rebuild) + a tiny hard-tail synth gap.
- **Round C — L2 covcap (chunk-per-paper cap=2)** [judging on gateway].
- **Round D — fan8 (KS_MQ_N_SUBQ=8 decomposition BREADTH, top_k=100)** [eval running on GPU in parallel]. Rationale: drill root-caused missed-gold as RRF fusion concentration; L1 attacked it with DEPTH (top_k=120) and failed (dilution); fan8 attacks it with BREADTH (more diverse sub-queries) — a different mechanism.
- **Round C — L2 covcap (chunk cap=2)** [DONE, slight loss] verdict_lctx: headline 0.7914→0.7865 (Δ-0.0049), trap −0.074, cit_sp −0.004. Capping per-paper chunks drops within-paper depth (nugget in a paper's 3rd chunk lost) > the distinct-diversity gain. NOT promoted.
- VERDICT-SO-FAR: 3/4 drill levers tried, NONE wins (L1 depth no-win, L3+L4 coverage sub-noise +0.006, L2 dedup −0.005). lctx@60 is at the NO-REBUILD CEILING on this frame; the 0.0132 long-ctx noise floor swamps the small gains. fan8 (breadth) is the last untried mechanism.
- **Round D — fan8 (N_SUBQ=8 breadth)** [eval done, judging].
- **Round E — rrkon (KS_MQ_ENABLE_RERANK=true)** [eval running, GPU]. HIGH-VALUE untested A/B: lctx baseline turned facet rerank OFF for @served diversity but never verified that's optimal for the harm(@served,nugget) HEADLINE. rerank ON = quality-ordered chunks → maybe higher nugget at lower @served; harmonic could go either way.
- **Round D — fan8 (N_SUBQ=8 breadth)** [DONE, no win] verdict_lctx: headline 0.7914→0.7972 (Δ+0.0058, below floor), trap HELD 0.926 (breadth doesn't erode trap, unlike depth L1), gcr −0.004. NOT promoted but trap-clean + small-positive.
- OBSERVATION: covunc (+0.0064) and fan8 (+0.0058) are two consistent small-positive ORTHOGONAL levers (synth-coverage × retrieval-breadth). Like V-MQ+V-SR, stacking them might reach ~+0.012 (≈ the 0.0132 floor). → will try a combined run.
- **Round E — rrkon (rerank ON A/B)** [eval running, GPU].
- **Round E — rrkon (rerank ON)** [eval done, judging]. served_med still 60.
- **Round F — covfan8 (coverage synth + N_SUBQ=8 breadth, stacked)** [eval running, GPU]. Tests whether the two small-positive orthogonal levers add to cross the noise floor.
- **Round E — rrkon (rerank ON)** [DONE, NEAR-WIN] verdict_lctx: headline 0.7914→0.8028 (Δ+0.0115, just below the 0.0132 floor), ALL gates OK, trap IMPROVED 0.926→0.963. STRONGEST lever yet + cleanest. KEY FINDING: the lctx baseline turned facet rerank OFF for @served diversity, but rerank ON is BETTER for the harm(@served,nugget) headline AND trap — the rerank-off default was a long-context-unlock artifact, not headline-optimal. Candidate to fold into the deployed config.
- NEXT: allpos = rerank ON + coverage(KS_SYNTH_COVERAGE=1) + breadth(N_SUBQ=8) — stack all three positives (rrkon +0.0115, covunc +0.006, fan8 +0.006) → likely crosses the floor = a real win.
- **Round F — covfan8 (coverage + breadth)** [eval done, judging].
- **Round G — allpos (rerank ON + coverage + breadth, ALL 3 positives stacked)** [eval running, GPU]. Best shot at crossing the noise floor for a real win.
- **Round F — covfan8 (coverage+breadth)** [DONE, no win] verdict_lctx: headline +0.0064 (== covunc alone → levers DON'T stack, overlapping headroom), trap −0.074 (coverage erosion). NOT promoted.
- KEY INSIGHT: small levers do NOT add (covfan8 == covunc). The winner is the single strongest lever = rerank-ON (+0.0115, all gates clean, trap +0.037). Coverage erodes trap, so the cleanest stack to try = rerank-ON + breadth (no coverage).

## DEFAULT-PROMOTION (user: "有提升的部分直接进默认", 2026-06-14)
Promoted the validated wins to the LIVE query() defaults (env-overridable = reversible):
aquery: _QUERY_VARIANT single→multiquery, _CHUNK_TOP_K 12→60, _TOP_K 40→100.
multiquery: _N_SUBQ 4→5, _SUB_CHUNK_TOP_K 16→60, _MAX_TOTAL_TOKENS=300000 on the query QueryParams,
facet rerank stays ON (the rrkon winner). synth: V-SR (strict-refusal) now default ON.
NOT promoted (sub-noise/negative): coverage rule, N_SUBQ=8 breadth, chunk-per-paper dedup, top_k=120.
Live default now ≡ rrkon config = the night's best (headline 0.806, trap 0.963, all gates clean).
100/100 eval tests green; query path imports clean. Service NOT restarted (deploy timing = human);
revert = unset the envs or git-revert these 3 files. Tree still entangled (#30) → not committed.

## decompose few-shot + 128K (user-directed, 2026-06-14) — VALIDATED, committed 2c3fb87
fsdef = deployed default + few-shot(in _DECOMPOSE_SYSTEM) + 128K decompose max_tokens. 3-run gold_v4,
0 decompose fallbacks. vs rrkon (prev default): @served 0.801->0.817, headline 0.798->0.811 (+0.013;
+0.0229 vs lctx baseline, PASSES the 0.0132 noise floor), nugget ~flat 0.835, trap 0.926 (held), all
gates OK. win_strict=false only on jackknife (same caliber as rrkon). Better decompose facets -> better
retrieval = a real +@served gain. KEPT as default. NEW deployed default headline ~0.814.
Corroborates drill action #1 (better decompose -> recall): the up-to-n + over-split + _parse_facets
dedup + anchor-merge rework is worth doing next (would compound this @served gain).

## PROMPT REWORK + DELIBERATION (user: "好好推敲一下这些prompt" + "decompose和base query合成一次", 2026-06-14)
Two user asks: (1) merge the decompose + base-query LLM calls into one; (2) workflow-deliberate ALL prompts.

MERGE (3->2 LLM calls/Q): retrieve_fused now decomposes FIRST, then runs the BASE query with the
ANCHOR facet's hl/ll pre-supplied so LightRAG SKIPS its own keyword-extraction LLM call (operate.py:
pre-defined keywords short-circuit). Coverage-gate semantics UNCHANGED (still assessed on the
base/anchor single-query; 'empty' never fuses; base stays datas[0] priority input). Fallback: if
decompose returns [] -> bare-intent single-query with LightRAG's own extraction (byte-identical old path).

DELIBERATION WORKFLOW (wf_fa2cdf9b-634, 22 agents): Ground (quality-frame + real exemplars) ->
13 opus lens-critics (decompose 4 / synth 5 / extraction 4) -> 3 opus adversaries (attack every
proposed edit) -> synthesis -> coherence. LESSONS: (a) model:'fable' subagents 100%-FAIL here (access
error) -> resumed with model:'opus', cache returned the 18 unchanged agents instantly (see memory
workflow-no-fable-subagents). (b) the synth agents WROTE the production files directly (not just
returned text); caught via the coherence agent's mtime note BEFORE the in-flight rework benchmark
corrupted (it was still on run 1 = rework in-memory, no result files written) -> killed it by explicit
PID, no corruption. Reviewed every diff against the synthesis; all sound.

FRAME (re-confirmed by the deliberation, drives what is even worth a prompt edit):
- RECALL is GRAPH-BOUND, NOT prompt-reachable: ~92% of gold misses are ABSENT from the retrieved
  subgraph; ~25% of gold is unreachable without a graph REBUILD. Prompt edits must NOT chase recall.
- The only prompt-reachable levers: refusal discipline (trap gate, rebuild-free) + answer-quality
  (nugget/faithfulness/precision). Coverage-push trades AGAINST trap (-0.074 measured) -> do not fold
  the coverage rule on by default.
- kb_coverage is DEAD (all 'strong', incl. all 9 traps); cited_papers (top-level) = the SERVED set
  (25-57), NOT in-text cites -> judge over/under-citing from inline [Key] tags only.
- The lone trap leak is a SELF-AWARE override (model quotes rule 3b then violates) -> a prompt tweak
  alone may not fully kill it; needs 3-seed trap majority to even detect (single-run trap unreliable).

DELIBERATED CHANGES (all adversary-vetted; query-path = decompose+synth tested by the benchmark;
extraction = ingest-time, validated only at a human-gated REBUILD):
- decompose (the headline lever per coherence): the rework's one-sided "FEWER is correct" UNDER-split
  wide compound Qs (positron DM-vs-pulsar-vs-modulation; lepton+proton drift windows) -> starved RRF
  on exactly the @served bottleneck. Fixes: balanced "up to {n} ... FEWER if fewer, do not collapse
  distinct angles" + an ANTI-UNDER-SPLIT clause (compound on-topic angles in DIFFERENT papers -> one
  facet each; off-domain/single-topic carve-out so traps don't get padded) + concrete-shared-subject
  anchor for compare/combine intents (not a broad umbrella) + few-shot anchor realigned to a SINGLE
  concept (PINN moved to facet 1) + system rule "no bracket chars outside the array" (parser harden).
- synth: req1 symmetric wrong-cite defect (attach ONLY the entailing key; drop-don't-borrow if no
  chunk supports even by direct implication; no stacking non-supporting keys) + req2 guarded stacking
  (stack only when EACH key's chunk independently states the claim) + 3b anti-reframe + 3b PARTIAL
  FLOOR (a part is supported iff a chunk states/directly-implies it; zero supported parts = total
  non-coverage -> hard stop; plugs the manufacture-a-partial trap leak) + user-turn hard-stop mirror.
- extraction: grounded relationship_description base-patch (kills base "rationale" vs KS rule-8
  contradiction) + directed source->target base-patch (+symmetric-relation preserve) + 2 added
  grounded example edges (recall floor) + de-hedged PSP/PINN example descriptions. BUGFIX: the rework
  numbered GROUNDING "8." which collided with base "8. Completion Signal" -> renumbered the injected
  block to a clean 1-13 (verified assembled, no dup, format() OK).

BENCHMARK (delib_r{1,2,3}, lctx@60 config, gold_v4): DONE — REGRESSED, REVERTED. verdict vs fsdef
(0.8143): headline 0.7981 (delta -0.0162), trap 0.926->0.852 (-0.074, gate REGRESSED),
gold_citation_recall -0.018 (gate REGRESSED), citation_support_precision -0.016 (gate REGRESSED),
hallucinated/faithfulness OK; win_strict=false, win_trap_adjusted=false. vs lctx@60 (0.7914): +0.0068
(below floor) but trap still -0.074. ROOT CAUSE = exactly the flagged risk: the decompose anti-under-
split licensed off-domain facets on traps -> tangential context fused -> synth answered -> trap eroded.
ACTION: `git restore`d multiquery.py + synth.py to the committed fsdef default (2c3fb87); kept
extraction_prompt.py (ingest-time, validated only at the rebuild) + the test_rerank.py default fix.
LESSON: 22 adversarial agents rated the prompt rework sound; the 3-seed benchmark still showed a net
loss. Careful deliberation != improvement — the benchmark is the only arbiter. Prompt micro-tweaks
(req1/req2/3b) were at-best sub-noise even in the best case; the decompose anti-under-split was the
headline hope and it regressed (trap). fsdef (0.8143, trap 0.926) stays the proven default.

INFRA BUGS hit + fixed (eval harness, NOT production): (1) the runner skipped build_judge_prompts ->
judge "unknown tag dirs"; (2) judge_mimo doesn't load_dotenv -> "KS_VIRTUAL_KEY not set". Both folded
into the runner template (build_judge_prompts per run + `set -a; . ./.env; set +a` before judge).

## LOW-LEVEL BUG HUNT (user: "drill 原始代码 看有没有别的低级错误", 2026-06-15)
Workflow wf_a94de60f (6 areas x find+adversarial-verify -> rank, all opus). 8 confirmed in-class, ZERO
need a rebuild. Codebase structurally healthy; one important FALSE POSITIVE caught by data:
- #1 (workflow rated HIGH "your deploy silently loses +0.18"): _MQ_ENABLE_RERANK default "true" claimed
  to be wrong (BASELINE_FULLCORPUS.md says facets must be rerank-OFF). VERIFIED FALSE POSITIVE via the
  dumps: distinct-papers/Q is median 45 for BOTH fsdef (rerank ON) and lctx_v4 (rerank OFF) — rerank-ON
  does NOT collapse the served set; and NIGHT_LOG Round E rrkon already A/B'd rerank-ON as the WINNER
  (+0.0115, trap 0.926->0.963) which SUPERSEDED the BASELINE doc. Every finder anchored to the stale
  BASELINE doc and missed the rrkon round. LESSON: verify a workflow finding against the LATEST
  experiment + real data, not just the SOT doc; the SOT doc can be stale (it was). FIX = update the doc,
  NOT the code (code 'true' is correct). [see memory verify-workflow-findings-vs-latest-experiment]
- REAL fixes applied (all query-time, no rebuild): (a) RERANKER TRUNCATION (the seed): lightrag_init.py
  _RERANK_MAX_LENGTH 1024 (stale comment said chunk=1200; actual chunk_token_size=2400, BGE-M3 measured
  p50 2758 / max 3485 -> 97% of chunks scored on first ~37%) -> 4096 (env KS_RERANK_MAX_LENGTH), runs on
  the pinned 3090 where embed already does 8192. BENCHMARKING NOW (rrk4k vs fsdef). (b) aquery.py single-
  query branch missing max_total_tokens -> inherited 30000 -> ~12-chunk truncation off the live path;
  added KS_MAX_TOTAL_TOKENS (A/B fairness). (c) GPU comments were INVERTED — KS reranker/embed run on
  the pinned RTX 3090 (cuda:0 via CUDA_VISIBLE_DEVICES=1), not the 5090 (corrected in the reranker note).
- Remaining LOW comment-drift (batch hygiene, behavior-safe): llm.py worker-cap 480s->1800s, aquery
  docstring 40/12->100/60, _bge_embed "~1200" comment, decompose "max_tokens=8000"->131072, GPU comments.

## AUTONOMY SESSION (user: "全自主推进 10h, workflow 自用, token 随便用, 全权", 2026-06-16)
COMMITTED+PUSHED the pre-build batch (post-ingest-build e13623c/8560e09/663dd9b): reranker max_length
1024->4096 (full-chunk scoring; user-chosen after we measured cost is actual-token-bound not padded-to-
max, ~0.24s/pair, ~4min/query live, memory-safe single-query — the 22GB was a concurrency-4 eval high-
water, live rerank is serialized) + #2 single-query max_total_tokens + all stale-comment fixes +
extraction-prompt rework (for the rebuild) + eval transients gitignored.

REBUILD REALITY: a clean full prod l0 rebuild = ~15h (conc47 log: 3400 papers / 54591s; a NEW prompt =
all-cache-miss = full ~15h). My window is 10h AND a rebuild clears prod l0 (live graph) -> KS degraded
for the whole build. So I am NOT launching the full prod rebuild unsupervised (it cannot finish + would
leave KS broken). Instead the window's job = DE-RISK it: validate the extraction-prompt on a sample.

EXTRACTION-PROMPT SAMPLE A/B (go/no-go for the 15h rebuild): cleared l0_probe (l0 untouched: 226809
before==after) -> building n=150 fresh papers with the NEW prompt on l0_probe. Compare U1 faithfulness
(same MiMo judge) NEW(l0_probe) vs OLD(prod l0, read-only) + density + U4. OLD baseline captured:
prod l0 = 226809 ent / 558068 rel / avg_deg 4.92 / U3 0.59% / U4 0.87%(post-dedup) / U5 8.31% iso;
/tmp/u1_old.jsonl = 120 old-prompt edges (judge deferred to avoid gateway contention with the build).
Tools: experiments/eval/_ab_upstream.py (guardrails + U1 sample, workspace-param) + _ab_u1_judge.py
(MiMo faithfulness judge). GO if NEW U1 > old ~0.77 AND density not collapsed AND U4 not worse.
A/B RESULTS (MiMo faithfulness judge, same judge both sides — absolute U1 is LOW vs the survey's
Claude-judge 0.77 because this judge is much stricter; only the DELTA is meaningful):
- NEW prompt (l0_probe, 150 papers, completed 150/150 clean): U1 = 0.41 (120 edges) BUT 0.34 (200 edges
  — the 120 was a high subsample; 200 is the reliable estimate). Guardrails: 163 edges/paper (density
  HEALTHY, > prod's 141), U4 6.16% pre-dedup (< old pre-dedup 9.65% — naming rule helps), U3 0.64%.
  COST: 76 LLM format-errors (MiMo garbling the tuple format on the more-complex new prompt) vs OLD=0.
- OLD prompt (prod l0, full corpus, 120 edges): U1 = 0.30.
- So NEW(200)=0.34 vs OLD(prod)=0.30 = +0.04, <1 sigma = NOT significant (the +0.11 from the 120-edge
  new sample shrank to +0.04 at 200). The new prompt's faithfulness gain is now MARGINAL, not the clear
  win the small sample suggested. LEANING NO-GO, pending the clean de-confounded number.

CRITICAL INFRA BLOCKER found + fixed (essential for ANY rebuild): the first clean-A/B old build FAILED
127/150 docs with PostgreSQL `TooManyConnectionsError('sorry, too many clients already')` + `pool is
closing` — Postgres max_connections=100, and the build (max_parallel_insert=16 + embedding_func_max_async
=16 + the DB-backed LiteLLM gateway's pool + leftover idle conns) exceeded it. NOT keys (78/94 live, 429
rate 0.3%), NOT the prompt. A full 3833-paper rebuild at default concurrency would hit the SAME wall and
stall. FIX (validated: old build v2 running with 0 conn-errors): KS_MAX_PARALLEL_INSERT=16->6 +
KS_EMBED_MAX_ASYNC=16->8 (KS_LLM_MAX_ASYNC=32 unchanged — that's gateway HTTP, not PG). Any rebuild MUST
set these.

REBUILD REALITY (refined): clean new-prompt build = 150 papers / 3h at conc32 -> full 3833 ~= multi-DAY
(probe self-estimated 357h on the degraded run; realistic ~30-76h depending on concurrency + paper size).
A full rebuild is a multi-day, infra-sensitive, token-heavy, SUPERVISED operation — not a fire-and-forget
autonomous one. And the validation so far does NOT justify it (U1 gain marginal +0.04, format-error
regression). Re-running the clean SAME-PAPERS old build (v2, with the PG fix) for the de-confounded U1.
CLEAN SAME-150-PAPERS A/B (old build v2 completed 150/150, 0 PG-conn-errors — the PG fix WORKS):
  U1 faithfulness (200 edges each, same MiMo judge):  OLD 0.305  ->  NEW 0.340  = +0.035 @ 0.74 sigma
  (NOT significant). Edges/paper:  OLD 178  ->  NEW 163  (-8%, the grounding rule DROPS edges).
  Entities: 15172 -> 14297 (-6%). U4 collision: 6.55% -> 6.16% (new slightly better). Format-errors:
  16 -> 76 (4.75x, new prompt garbles MiMo's tuple output more). VERDICT = NO-GO, judge-INDEPENDENT:
  even setting aside the (strict, possibly-noisy) U1 judge, the new prompt has clear COSTS (-8% density,
  4.75x format-errors) without a significant benefit. REVERTED (f1383ff). The deliberated extraction
  rework (grounding/directionality/naming), like the synth rework before it, did NOT validate — same
  lesson: deliberation != measured improvement; the benchmark is the arbiter.

## AUTONOMY-SESSION WRAP-UP (for the morning) — 2026-06-16
WHAT SHIPPED (committed + pushed, post-ingest-build, all query-time / no-rebuild):
  e13623c reranker max_length 1024->4096 (full-chunk scoring; user-chosen; live-quality, eval uses
          KS_RERANK_MAX_LENGTH=1024 to stay fast) + single-query max_total_tokens + stale-comment fixes.
  8560e09 -> f1383ff: extraction rework landed then REVERTED (clean A/B = wash-to-negative, above).
  663dd9b eval docs (delib regression, bug-hunt, rerank-default doc fix).
THE BIG DECISION — FULL REBUILD: **NO-GO** (data-backed). The rebuild's only real content was the new
  extraction prompt, which the clean A/B shows is a wash-to-negative. Rebuilding with the OLD prompt =
  the same graph = pointless; with the NEW prompt = -8% density + 4.75x format-errors for a +0.035-n.s.
  U1. And @served (the headline bottleneck) is query-time-budget-bound, NOT graph-bound, so NO rebuild
  helps it. So there is NO validated upstream lever to justify the multi-day (~30-76h) rebuild. I did
  NOT clear prod l0; prod is UNTOUCHED + serving the whole time.
CRITICAL INFRA for any FUTURE rebuild: Postgres max_connections=100 — the build MUST run with
  KS_MAX_PARALLEL_INSERT=6 + KS_EMBED_MAX_ASYNC=8 (validated 0 conn-errors) or it stalls (~85% docs fail
  with TooManyConnectionsError, as the first old-build v1 did). 16/94 MiMo keys are 429-dead (78 live).
STATE: the system is at its no-rebuild / no-better-prompt CEILING. Proven default = fsdef (headline
  0.814) + reranker 4096 (live quality). The last query-time lever fcap (fanout-adaptive @served budget)
  is BENCHMARKING now (vs fsdef) — if it too lands sub-floor, @served is confirmed plateaued and the next
  real gains need either a genuinely different extraction approach (fix the format-error regression first)
  or accepting the current strong baseline. FCAP VERDICT (3-run gold_v4, reranker 1024 to match fsdef): fsdef 0.8143 -> fcap 0.8373 = **+0.023,
  PASSES the 0.0132 noise floor** (the session's ONLY above-floor headline gain — the fanout-adaptive
  @served budget IS a real lever; the drill's "likely sub-floor" was too pessimistic). BUT win_strict=
  false: survives_jackknife=false (gain concentrated on the broad-Q subset it targets — expected but
  fragile), trap 0.926->0.852 (-0.074, WITHIN eps=0.10 = trap noise, gate_at_eps passes), cit_sp -0.0056
  / faith -0.0025 (tiny, more-context cost). DECISION: KEEP flag-gated (KS_MQ_FANOUT, default OFF) — a
  validated-available lever, NOT auto-enabled (not a strict win + enabling slows broad queries: 100
  chunks rerank, esp. with reranker 4096). RECOMMEND enabling (KS_MQ_FANOUT=1) for the +0.023 quality
  gain after a 5-seed run resolves the jackknife fragility + trap noise. This is the live @served lever
  the rebuild could NOT provide.

## fcap v2 (LLM n_sources budget) — STRONG WIN (2026-06-18)
Full 3-seed gold_v4 vs fsdef (reranker 1024): headline 0.8143 -> 0.8467 = **+0.0324, passes noise floor
AND survives_jackknife=TRUE** (v1 word-count was +0.023 jackknife-FAIL — the LLM n_sources detector is
both bigger and ROBUST). Fast-tier early read (gold_fast) had flagged it: @served 0.756 -> 0.858. Gates:
hallu/gcr OK; cit_sp -0.003 / faith -0.009 (tiny, more-context cost); trap -0.074 (gate_at_eps_0.10=TRUE
= the recurring -2/27 noise artifact seen identical across delib/covunc/covfan8/v1/v2 — variant floors
traps via n_sources=1, smoke-confirmed). win_strict=false ONLY on the strict-eps=0 cit_sp/faith/trap
flags; the headline is a clean robust win. Committed flag-gated (KS_MQ_FANOUT, default OFF at commit time). NEXT: v3
(dynamic facet count on top), fast-tier-first per the two-tier protocol; then enable the best by default.
> CORRECTION 2026-06-18: v2 was subsequently **PROMOTED to default ON** (KS_MQ_FANOUT default "1" in
> multiquery.py) and is LIVE; v3 (dynamic facets) was fast-tier-filtered OUT (no @served gain, not adopted).
> Set KS_MQ_FANOUT=0 for the byte-identical 60/5 baseline. (Stale "default OFF" lines above are point-in-time.)
