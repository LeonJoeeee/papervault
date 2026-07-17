# FULL-CORPUS downstream baseline (gold_v2, prod l0) — the CURRENT frame

Date: 2026-06-08 · Graph: prod `l0`, 226,809 entities / 558,068 relations (3,671 papers) ·
Gold: `gold_v2.jsonl` (39 answerable, gold_keys 166 after alias-dedup + 9 off-domain traps) ·
Runner: `run_eval.py --gold gold_v2.jsonl --concurrency 6` on the 5090, hardened reranker
(0 rerank-degraded), 1 judge seed (`judge/fullcorpus_baseline/`). Aggregator:
`headline_fullcorpus.py`.

## THE NUMBERS — final (2026-06-11: 3 runs × tightened contract × Opus judge, 1 seed/run)

- **HEADLINE = 0.4295** (3-run mean; per-run [0.4261, 0.4339, 0.4284])
- **Noise floor: run_mean_sd = 0.0040** (range 0.0078) — pinned as
  `headline.FULLCORPUS_NOISE_FLOOR_RUN_MEAN_SD`. Much tighter than the retired test100 floor
  (0.0127): mid-range questions swing less run-to-run than the old saturated landscape.
- **trap_correct_refusal: [4/9, 4/9, 2/9] ≈ 0.37 mean** — ⚠️ the honest number. The old 7/9 was
  an artifact of the lenient contract: most of the system's trap "refusals" are
  REFUSE-THEN-EXPLAIN hybrids (synth says "not covered" then answers from outside knowledge
  anyway), which the tightened contract correctly scores as violations. Real refusal
  discipline ≈ 1/3 — a genuine synth-prompt improvement target. Per-run swing (4/4/2 of 9)
  confirms the drill's fragility warning → variant comparisons MUST use 3-seed majority on
  traps or gate epsilon ≥ 0.10.
- Vector (3-run mean, tightened): nugget_recall 0.662 (was 0.685 lenient — −0.023, in the
  predicted band) · citation_support_precision 0.672 (was 0.731 — −0.059, in band) ·
  faithfulness 0.806 · hallucinated 0.020 · hit@12 0.769 · citation_recall 0.819 ·
  relevance 0.875 · phantom 0.179 · paper_recall@12_distinct 0.4259 · paper_recall@5 0.411 ·
  over_confidence_rate 0.2308 (DASHBOARD-ONLY, dead signal — see drill)
- Judge integrity: 144/144 validator-clean (2 derived-scalar arithmetic slips mechanically
  recomputed from the per-check lists, audit-trail preserved). Old-contract r1 seeds archived
  at `judge/fullcorpus_baseline_oldcontract/`.

## ⚠️ Comparability — the old numbers are a RETIRED FRAME

The HEADLINE is a coordinate on a (corpus, gold, retrieval-config) frame, NOT an absolute system
property. **Never compare 0.4371 against test100-frame numbers (0.864 / 0.7403).** Four
simultaneous frame shifts: (1) corpus 89 → 3,671 papers (vastly more distractors per query);
(2) gold_keys re-anchored + expanded for the full corpus (denominators grew); (3) traps replaced
(10 dead heliophysics-adjacent → 9 off-domain); (4) effective retrieval depth differs (~6-8
distinct chunks reach scoring, not 12 — see drill D3). Legitimate cross-frame signals are the
frame-free rates only: hallucinated_rate (0.05→0.04 ✓), faithfulness, citation_support_precision.

## What the 2026-06-08 adversarial drill (#24, 4 reviewers) established

- **D1 (trap design)**: kb_coverage saturates at 'strong' on 227k entities (all 9 off-domain
  traps included) → backbone `trap_violation_rate` pinned 1.0 (dead) and `over_confidence_rate`
  ≡ 1−hit@12 (dead calibration signal, gameable via coverage-bin recalibration). FIXED:
  over_confidence_rate removed from `headline.ABSOLUTE_GATES`; both marked dashboard-only in
  backbone.py. The judge's `trap_correct_refusal_rate` is the sole trap gate (mandatory,
  fail-closed). Trap-gate fragility: 9 traps × 1 seed → 1 flip = ±11pp; mitigation = 3 judge
  seeds on traps (majority machinery already exists) and/or gate epsilon ~0.10.
- **D2 (gold validity)**: recall 0.4259 is REAL headroom, not gold artifact — 83% of audited
  added keys are sound primary sources; expansion DEFLATED recall by 1.4pt (anti-circular).
  Fixed the only 2 bad additions (same-paper alias dups, removed). Known residual:
  under-expansion in the 15 zero-addition Qs (PINN family worst — retrieved Wang2021 NTK is
  genuinely-answering but unlisted); re-expansion pass queued.
- **D3 (protocol)**: old noise floor 0.0127 measured on the retired frame is INVALID here →
  re-measuring (3 repeats). Variant protocol stays the old one: 3 runs per side, verdict fed
  3-run-mean tables, k=1. Effective chunk depth is ~6-8 (not 12): the "@12" cap is applied to
  CHUNKS pre-dedup, and merged retrieval yields 6-8 chunks → H4 anti-dedup currently inert;
  consider persisting the pre-truncation top-40 file_path list in future runs.
- **D4 (judge quality)**: good enough to run the loop — arithmetic perfect 48/48, evidence
  discipline clean. Contract tightened (compound-claim entail rule; multi-part nugget rule;
  refuse-then-explain = NOT a clean refusal) — expect nugget_recall ~0.05-0.10 and
  citation_support_precision ~0.03-0.08 LOWER on future runs (stricter, more honest scoring);
  re-baseline before comparing. Validator now accepts the failed-trap identity (nugget_recall
  0.0 with empty list when trap_correct_refusal=false).

## Protocol for the #5 variant loop on this frame

1. ✅ Noise floor measured + pinned: `headline.FULLCORPUS_NOISE_FLOOR_RUN_MEAN_SD = 0.0040`.
2. Variants: 3 runs × 1 seed each side (tightened contract, SAME judge model = Opus);
   verdict() on 3-run-mean tables, k=1, noise_floor_run_mean_sd=0.0040.
3. Traps: 3 judge seeds (majority) or epsilon ≥ 0.10 on the trap gate (per-run swing 4/4/2
   of 9 measured — single-seed trap scalars are NOT stable).
4. Gates: hallucinated_rate + gold_citation_recall (absolute), citation_support_precision +
   faithfulness (paired), trap_correct_refusal_rate (mandatory scalar). over_confidence_rate /
   trap_violation_rate are dashboard-only — never optimize them.

## Identified improvement levers (from this baseline, ranked)

1. **Retrieval recall 0.426** — the headline bottleneck (answer quality 0.66-0.81 is fine on
   what IS retrieved). Levers: query rewriting/decomposition (multiquery/standard variants
   already implemented in `multiquery.py`), top_k/chunk_top_k, fusion.
2. **Trap/refusal discipline ≈ 1/3** — synth answers from outside knowledge after
   acknowledging the gap; a synth-prompt rule ("if the material doesn't cover it, STOP after
   saying so") is a cheap, rebuild-free lever measured by the trap gate.
3. citation_support_precision 0.672 — compound sentences over-claim vs their cited chunk;
   synth-prompt citation discipline.

---

# #5 VARIANT LOOP — OUTCOME (2026-06-14): V-MQ+V-SR is a STRICT WIN

## Judge cutover to MiMo (ruler change, 2026-06-14)
The Opus-subagent judge could not finish the variant loop (Claude session/rate limits killed the
220-wide fan-out repeatedly). Per user directive the judge was cut over to **MiMo via the LiteLLM
gateway** (`experiments/eval/judge_mimo.py`, 94/94 keys live, async conc 60, zero Claude quota).
RULER INVARIANT honored: ALL 9 frame tags (3 baseline + V-MQ + V-SR) were **re-judged with MiMo**
so the instrument is identical on both sides of every verdict. The old Opus seeds are archived at
`judge/_opusjudge_archive_20260614-mimocutover/` (372 seeds, audit only — never mixed).

**Ruler-drift check (MiMo vs archived Opus on the SAME baseline seeds)** — MiMo is trustworthy for
headline/quality, lenient on traps:
- HEADLINE: Opus 0.4306 vs MiMo 0.4242 — **within 0.006** (headline verdicts are judge-robust).
- nugget_recall 0.662/0.627, faithfulness 0.806/0.768, citation_support 0.672/0.745 — all within ~0.07.
- **trap_correct_refusal: Opus 0.370 vs MiMo 0.741** — MiMo runs a ~+0.33 CONSTANT lenient offset
  (does not penalize refuse-then-explain hybrids as hard as Opus). So MiMo trap ABSOLUTES are not
  comparable to the Opus "honest 1/3"; but RELATIVE trap deltas under one MiMo ruler are valid and
  were CROSS-CONFIRMED on the strict Opus ruler (V-SR: Opus baseline 0.370 → Opus V-SR 0.833, +0.46).

## The four-way result (3 runs/side, MiMo judge, gold_v2, prod l0 read-only, 5090)

```
                HEAD    pRec@12*  hit@12*  hallu*   nugget  cit_sp   faith   trap     verdict
BASELINE       0.4242    0.4259   0.7692  0.0191   0.6272  0.7450  0.7680  0.741     —
V-MQ           0.5589    0.5439   0.9145  0.0229   0.6879  0.7893  0.8614  0.667     win_strict=FALSE
V-SR           0.4207    0.4259   0.7692  0.0159   0.6077  0.7559  0.7703  0.926     win_strict=FALSE
V-MQ+V-SR      0.5514    0.5401   0.9145  0.0140   0.6980  0.7742  0.8545  0.963     win_strict=TRUE ✅
                (* = deterministic, judge-independent: gold_keys match in results, not the judge)
```

- **V-MQ (multiquery retrieval, `--variant multiquery`)**: a large, REAL, judge-independent
  retrieval win — paper_recall@12_distinct 0.426→0.544 (+0.118, deterministic), hit@12 0.769→0.914,
  headline +0.135, passes noise floor + jackknife + 34 movers. NOT a strict win ALONE: extra
  retrieved context regresses trap refusal (0.74→0.67, synth more willing to answer a trap) and
  nudges hallucinated_rate up a hair (0.019→0.023). Both are the expected cost of retrieving more.
- **V-SR (strict-refusal synth, `KS_SYNTH_STRICT_REFUSAL=1`)**: a clean, judge-INDEPENDENT
  refusal-discipline win — trap 0.741→0.926 (MiMo) / 0.370→0.833 (Opus cross-check), all quality
  gates clean, ZERO retrieval cost. NOT a headline win ALONE (it doesn't touch retrieval → recall
  flat → headline flat).
- **V-MQ+V-SR (compose both — orthogonal levers, retrieval-side × synth-side)**: **STRICT WIN.**
  headline 0.4242→**0.5514 (+0.1272)**, passes noise floor + jackknife + 34 movers; trap
  0.741→**0.963**; ALL gates OK (`win_strict=true`, `win_trap_adjusted=true`). Best-or-tied on
  EVERY axis: recall +0.114, hit@12 +0.145, **hallucinated 0.0191→0.0140 (BELOW baseline)**,
  nugget +0.071, faithfulness +0.087, trap +0.222. Synergy bonus: V-SR's "don't assert what the
  chunks don't support" discipline cancels the small hallucinated regression V-MQ introduced alone.

## Robustness (why this win holds)
Triangulated four ways with NO extra Claude quota: (1) the headline gain is dominated by the
DETERMINISTIC recall metrics (judge-proof); (2) MiMo≈Opus on headline/quality (within 0.006);
(3) the trap mechanism is cross-confirmed on the strict Opus ruler (+0.46); (4) verdict()'s own
noise-floor + jackknife + movers-subset significance all pass.

## Recommendation (human-gated deploy)
Make the live MCP query() path default to **multiquery retrieval + strict-refusal synth**,
flag-gated (`--variant multiquery` semantics on the live path + `KS_SYNTH_STRICT_REFUSAL=1`),
branch + test, never auto-deploy. This is the #5 loop's shippable improvement.

---

# OPTIMIZATION ROUND 1 (2026-06-14): cap/pool sweep + a fixed metric bug + recall is graph-limited

Loop architecture: a design WORKFLOW (opus ideate + haiku scout, no fable) proposes the next cheap
(no-rebuild) variant batch; SCRIPTS run 3× eval (prod l0 read-only) + MiMo-judge + verdict; then a
pure-python drill decomposes the result. Judge stays MiMo (constant ruler). Four query-path
constants were env-wired (defaults byte-identical to baseline): `KS_CHUNK_TOP_K` (aquery, default
12), `KS_TOP_K` (40), `KS_MQ_N_SUBQ` (4), `KS_MQ_RRF_K` (60), `KS_MQ_SUB_CHUNK_TOP_K` (16) — so a
variant is an env-var set, no per-round code edit.

## ⚠️ Metric bug FIXED (backbone.py hallucinated_rate) — affects all chunk_top_k>12 variants
`hallucinated_rate`'s "allowed" set was `chunk_papers_12` (papers in the first 12 CHUNKS). A variant
serving chunk_top_k>12 hands synth 18-20 chunks; synth legitimately citing a paper from served chunk
#15 was mislabeled HALLUCINATED. Empirically (capwide_r1): 128 prose-keys flagged @12 but only 22
real — **106/128 were a pure artifact** (the cited paper WAS in the served chunks). This produced a
false `hallucinated_rate worse_by≈0.20` that vetoed two real wins. FIX: `allowed =
retrieved_papers_from_chunks(data, top_n=None)` = ALL papers the synth actually saw (hallucination =
citing a paper never retrieved). **No-op for the 12-chunk baseline/V-MQ/V-SR/V-MQ+V-SR runs**
(served==12 → identical), so all prior numbers stand. 85/85 eval tests green after the fix.

## Round-1 variants (all = V-MQ+V-SR + one lever; MiMo judge, 3 runs, gold_v2, prod l0, post-fix)
```
                    HEAD    pRec@12*  hit@12*  hallu*  nugget  cit_sp   faith   trap    verdict
V-MQ+V-SR (incumb) 0.5514    0.5401   0.9145  0.0140  0.6980  0.7742  0.8545  0.963   win (prior)
capwide cap18+pool28 0.5571  0.5370   0.9060  0.0170  0.7141  0.7890  0.8866  0.926   win_strict=TRUE ✅
cap20 cap20-only   0.5550    0.5283   0.9060  0.0230  0.7286  0.8051  0.9033  0.889   FALSE (hallu +0.004 over strict ε=0)
                    (* deterministic, judge-independent)
```
- **capwide = new marginal frontier** (headline +0.0057 over V-MQ+V-SR ≈ 1.4× noise floor, all
  gates clean). KEY SURPRISE: the cap-raise did NOT move recall (pRec@12_distinct 0.540→0.537, flat/
  down — pool-widen 16→28 reshuffled RRF). The headline gain is entirely JUDGE-side: feeding synth
  ~16-18 chunks instead of 12 lifts nugget_recall (+0.016), citation_support_precision (+0.015),
  faithfulness (+0.032) at near-zero gate cost. So the real round-1 lesson = "serve synth ~16-18
  chunks, not 12" (a free quality lever); cap=20 overshoots (real hallu +0.004 trips the strict gate).

## ⛔ Recall is GRAPH-LIMITED, not rank-limited — the no-rebuild ceiling on recall
Diagnostic over capwide_r1 (widest retrieval): of 98 gold misses, **90 (91.8%) are ABSENT from the
retrieved subgraph entirely** (not in chunks, entities, OR relationships) — only 8 (8.2%) sit deeper
in the pool (cheap-reachable). So paper_recall ~0.54 is bounded by WHAT GETS RETRIEVED, not by the
cut/rerank/fusion order. No no-rebuild knob (cap, pool, fusion, rerank) can recover the 92%. Last
cheap recall lever to test = retrieval DEPTH (`KS_TOP_K` 40→100) [running: tag `topk100`]. If depth
doesn't move recall, further headline gains require GRAPH-side work (extraction/embedding/chunking
rebuild) = out of the no-rebuild loop, human-gated. (Depth update: KS_TOP_K 40→100 did move
recall +0.020 (2-run, judge-indep); 120 dips it. But round 2 found the REAL unlock below.)

---

# OPTIMIZATION ROUND 2 (2026-06-14): the 1M context was never used — LightRAG MAX_TOTAL_TOKENS=30000

## Root cause (longctx-investigate workflow, verified in LightRAG 1.4.16 source)
Synth never saw more than ~6-12 chunks REGARDLESS of chunk_top_k: LightRAG's process_chunks_unified
truncates chunks to available_chunk_tokens = max_total_tokens − sys − kg_context − query. KS never
set max_total_tokens → inherited the dataclass default **MAX_TOTAL_TOKENS=30000**. At ~2.4k tok/chunk
and ~14-20k chunk budget = ~6-8 chunks/sub-query → fused ~26 distinct (gobig's "160 slots" never
bound; the 30k ceiling did). **MiMo's 1M context sat unused the whole time.**

## Unlock (env-only + a 6-line facet-rerank gate; NO rebuild, NO graph change)
- `MAX_TOTAL_TOKENS=300000` (QueryParam default, read at import) → ~286k chunk budget. THE master knob.
- ~~`KS_MQ_ENABLE_RERANK=false`~~ **SUPERSEDED 2026-06-14 — facet rerank is now ON by default (code
  default "true", the rrkon winner).** The original theory here (rerank-ON collapses RRF distinctness)
  was an *unlock-era hypothesis* that the later Round-E `rrkon` A/B EMPIRICALLY REFUTED: rerank-ON kept
  served at 60 chunks / **median 45 distinct papers — IDENTICAL to rerank-OFF** (verified from the
  fsdef vs lctx_v4 dumps), and won headline +0.0115 with trap 0.926→0.963. So facet rerank-ON is the
  deployed default; this "=false" line is retained only as the historical unlock note. (A 2026-06-15
  bug-hunt flagged the code default "true" as a bug by trusting THIS stale line — it is not a bug; the
  code is correct and this doc was stale.)
- `KS_CHUNK_TOP_K=60 KS_MQ_SUB_CHUNK_TOP_K=60 KS_MQ_N_SUBQ=5 KS_TOP_K=100` + V-SR.
- Served **60 distinct chunks** (was 12), **0 synth failures** (~150k-tok prompts fine; MiMo 1M confirmed).

## `lctx` = NEW FRONTIER — headline 0.608 (3-run; +0.18 vs baseline, +0.05 vs V-MQ+V-SR)
```
group   served  @12     @24     @36    @served  nugget  hallu   trap   HEADLINE
lctx      60   0.543   0.657   0.720   0.747   0.834   0.022   0.926   0.6080
capwide   18   0.537   0.586   0.586   0.586   0.714   0.017   0.926   0.5571
vmqsr     12   0.540   0.540   0.540   0.540   0.698   0.014   0.963   0.5514
baseline  ~6   0.426   —       —       —       0.426   0.019   0.741   0.4242
```
- @12 (locked headline retrieval term) FLAT (~0.54) — win is NOT from @12.
- @served 0.54→0.747 (+0.21): 75% of available gold now reaches synth (was ~53%).
- **nugget_recall 0.698→0.834 (+0.12-0.14, robust 3-run)** — synth USES the recovered gold; headline
  rises through the harmonic's nugget half. cit_sp/faith/gold_citation_recall also UP (more RELEVANT
  chunks → better-supported claims; the predicted precision drop did NOT happen). trap 0.926 (>baseline).
- Verdict: passes noise_floor + jackknife + 35 movers; trap/cit_sp/faith/gold_citation_recall gates OK.

## ⚠️ win_strict=FALSE here is a GATE FALSE-NEGATIVE (metric brittleness, not an lctx defect)
Only blocker: `hallucinated_rate worse_by=0.0026`. But that metric's run-noise dwarfs it — baseline
per-run [0.042, 0.004, 0.013] (run-sd ≈0.018, 10× swing); lctx [0.014, 0.032, 0.019] mean 0.022 sits
INSIDE baseline's range. The ABSOLUTE_GATES compare mean-vs-mean at epsilon=0 → any variant fails ~half
the time on noise. FIX (pair with the @served/@K secondaries): give absolute gates an epsilon = each
metric's measured run-sd (or a paired test), like the trap gate's epsilon=0.10. Under any noise-aware
gate, lctx is a STRICT WIN.

## Validated user intuitions
1. "Use the 1M context" — the single biggest lever (+0.05 over the prior frontier) once 30k was lifted.
2. "Chunks are unimodal, not monotone" — true, but the apparent ~18 peak (round 1) was an ARTIFACT of
   the 30k cap. With it lifted, nugget is still climbing at 60 (0.83) → real peak ≥60; @served 0.747
   (75% of gold) suggests diminishing returns soon.

## Unimodal peak located: ~60 chunks (chunk sweep at the lctx config, judge-independent + judged)
```
served  HEAD    nugget  cit_sp  faith  hallu   trap     (1-run pts; lctx@60 also 3-run)
  40   0.5618  0.821   0.826   0.928  0.029   1.000   ← rising
  60   0.5868  0.847   0.815   0.916  0.014   0.889   ← PEAK (3-run: HEAD 0.608, nugget 0.834, trap 0.926)
 100   0.5731  0.848   0.841   0.926  0.024   0.889   ← past peak (HEAD turns DOWN; nugget plateaus 0.847→0.848)
```
The user's unimodal hypothesis is confirmed: nugget_recall saturates by ~60 and HEAD turns down by 100
(too many chunks → the synth LLM uses them less well, even with 1M context). Trap takes a small hit
40→60 (1.0→0.89) then stabilizes. **Deployable sweet spot = 60 served chunks.** FINAL frontier config:
`--variant multiquery` + `KS_SYNTH_STRICT_REFUSAL=1` + `MAX_TOTAL_TOKENS=300000` +
`KS_MQ_ENABLE_RERANK=false` + `KS_CHUNK_TOP_K=60 KS_MQ_SUB_CHUNK_TOP_K=60 KS_MQ_N_SUBQ=5 KS_TOP_K=100`.

## Metric decision (pending user GO — discuss→doc→go)
Keep paper_recall@12_distinct as the LOCKED headline term (H4 + comparability with 0.42→0.55→0.61).
ADD dashboard secondaries (judge-indep, NOT in headline/verdict): recall@served_distinct + fixed-K
recall@24/@36 (`experiments/eval/recall_curve.py`). FIX the absolute-gate epsilon (noise-aware). These
together are the "metric adjusts when served-chunk count changes" the user asked for, without moving
the locked ruler.
