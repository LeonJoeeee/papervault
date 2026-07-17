# Metric-adjustment decision — long-context regime (2026-06-14)

Status: **IMPLEMENTED 2026-06-14** (user GO: "以当前版本为最新baseline，跑最新benchmark，再优化"). The
long-context config (lctx@60) is the NEW baseline; the new-frame verdict harness is `verdict_lctx.py`.
- A (dashboard secondaries): `recall_curve.py` — DONE.
- B (headline @12→@served): `backbone.paper_recall_at_served_distinct` + `headline.headline_table(recall_key=)`
  + `verdict(recall_key=)`; default stays @12 (short-context arc unchanged) — DONE, 100/100 eval tests green.
- C (noise-aware gates + regime noise floor): `headline.FULLCORPUS_LONGCTX_NOISE_FLOOR_RUN_MEAN_SD=0.0132`
  + `verdict(absolute_epsilons=)`; verdict_lctx passes hallucinated ε=0.016, gold_citation_recall ε=0 — DONE.
NEW baseline numbers (lctx@60, harm(@served,nugget)): headline ≈ 0.755. Future variants: `uv run python
experiments/eval/verdict_lctx.py <prefix>`. SOT for numbers: `BASELINE_FULLCORPUS.md`.

## Why this doc exists
The #5 optimization loop moved the system from serving the synth ~6-12 chunks to **60 chunks**
(long-context unlock: LightRAG `MAX_TOTAL_TOKENS` 30000→300000 + facet-rerank-off). That regime
shift broke two assumptions baked into the locked downstream ruler:

1. **The headline recall term `paper_recall@12_distinct` now under-credits the system.** Gold
   surfaced at served-ranks 13-60 is genuinely handed to synth (and lifts nugget_recall) but scores
   0 on @12. The headline still RISES (via the nugget half of the harmonic: 0.551→0.608) but @12
   itself is flat (~0.54) and hides the real retrieval gain (@served 0.54→0.75).
2. **The noise characterization was measured in the OLD regime and is wrong for the new one.**
   Headline run-to-run sd: OLD single-query/6-chunk regime = **0.0026** (pinned floor 0.0040);
   LCTX multiquery/60-chunk regime = **0.0132** — **5× noisier** (longer answers over more chunks =
   more synth run-to-run swing). The absolute red-line gates also assumed "deterministic rate,
   noise≈0", but `hallucinated_rate` / `gold_citation_recall` are SYNTH-derived (parsed from synth
   prose / cited_papers) and carry real run-noise: baseline hallucinated_rate per-run
   [0.042, 0.004, 0.013], **run_sd 0.0162**. The ε=0 absolute gate over-fires on that noise.

Concrete damage already seen: `lctx` (headline 0.608, +0.18 vs baseline, all paired/trap gates OK)
was scored `win_strict=FALSE` ONLY because `hallucinated_rate` "regressed" by **0.0022** — an order
of magnitude below that metric's own 0.0162 run-sd. A genuine, large win vetoed by pure noise.

## What does NOT need redoing (answering "对应的 baseline 也要重新做?")
- The baseline **headline number** (0.4242, @12 ruler) stands — @12 is unchanged.
- The **@served / @24 / @36 secondaries** are computed from the EXISTING baseline result dumps
  (`recall_curve.py`), no re-evaluation.
- The lctx-vs-baseline **verdict outcome** is unaffected by the noise fix: delta 0.172 ≫ even the
  larger 0.0132 floor (×13). The win is robust either way.

## What DOES need redoing
- The **noise floor** and the **per-gate run-sd** must be measured in the REGIME UNDER TEST, not
  borrowed from the old baseline regime. The long-context regime floor = **0.0132** (from
  lctx_r1/r2/r3, identical-config repeats), not 0.0040. This matters for FUTURE close calls
  (e.g. 60 vs 80 chunks, delta ~0.01) where the wrong floor would falsely declare a win.

## Decision

### A. Add dashboard secondaries (DONE — additive, NOT in headline/verdict)
`experiments/eval/recall_curve.py`: reports `recall@12_distinct` (= locked headline term) plus
fixed-K distinct shadows `@24`, `@36`, and `@served_distinct` (top_n=None), per tag/3-run-mean,
judge-independent. Interpretation rule: @12 flat + (@served AND nugget_recall) rising = a genuine
long-context win the headline structurally under-credits; @served rising + nugget flat = the deep
chunks are noise. K stays FIXED (not =n_served) so the shadow cannot be gamed by serving more (H4).

### B. REDEFINE the headline retrieval term: @12_distinct → @served_distinct (needs GO) — REVISED
Original plan was "keep @12 locked". On reflection (user, 2026-06-14): once the system serves 60
chunks, `recall@12_distinct` is the WRONG optimization target — it's slack (gold at served-rank
13-60 is used by synth but scores 0), flat (~0.54), and a holdover from the 12-chunk regime. The
objective should track what now matters: COVERAGE (did gold reach synth) + ANSWER QUALITY.

**Proposed headline = harmonic(paper_recall@served_distinct, nugget_recall).** Arc (3-run, recomputed
from existing dumps — no re-eval):
```
variant       harm(@12,nug)   harm(@served,nug)   nugget-only
baseline         0.422            0.422             0.627
V-MQ+V-SR        0.545            0.545             0.698
capwide(18)      0.541            0.586             0.714
lctx(60)         0.594            0.755            0.834
```
Why @served is the right cut now (and H4-safe in this regime):
- **Backward-compatible**: for variants serving ≤12 chunks, @served == @12, so the entire
  0.422→0.545 history is preserved BYTE-IDENTICAL; the two headlines diverge ONLY for served>12 —
  exactly where @12 becomes unfaithful. No retro-rewrite of the short-context arc.
- **Keeps a judge-independent anchor** (@served is deterministic gold-key match) — the headline does
  not become pure nugget (judge-only).
- **H4 (anti-gaming) still holds in the long-context regime**: @served is NOT free to inflate —
  (a) it is capped by the retrieval ceiling (~0.75; 25% of gold is absent from the subgraph, a
  graph-side limit); (b) over-serving past the unimodal peak (~60) is penalized by the trap/
  faithfulness gates and by nugget_recall plateauing (the harmonic stops rising); (c) distinct-paper
  counting kills dup-padding. The original H4 concern (a de-dup variant inflating recall@12 at a
  FIXED 12-cut) does not apply to an honestly-served-and-used deeper set.
- Keep `recall@12_distinct` + @24/@36 as DASHBOARD/legacy (recall_curve.py) for cross-frame continuity.

Consequence for the optimization roadmap (answers "召回很难优化了, 看覆盖和别的?"): chunk-serving has
hit its unimodal peak (~60), so COVERAGE (@served) is now ceiling-bound at the retrieval limit ~0.75
→ further coverage gains require GRAPH-side work (rebuild: extraction/embedding/chunking to retrieve
the absent 25%) = the human-gated big lever. ANSWER-QUALITY gains (nugget_recall / faithfulness /
citation_support_precision) remain reachable via synth-prompt + rerank levers (no rebuild).

### C. Make the absolute red-line gates + noise floor NOISE-AWARE and REGIME-MATCHED (needs GO)
1. **Noise floor**: `verdict()` must be fed the run-mean-sd measured from identical repeats of the
   VARIANT's own regime (or max(baseline-regime, variant-regime) to be conservative). For the
   long-context frame pin **`FULLCORPUS_LONGCTX_NOISE_FLOOR_RUN_MEAN_SD = 0.0132`**; keep 0.0040 for
   the short-context frame. The verdict helper picks the floor by regime.
2. **Absolute-gate epsilon**: replace the single `epsilon=0.0` with a PER-METRIC epsilon = that
   metric's baseline run-sd (measured, not guessed). Measured values (gold_v2, 3 baseline repeats):
   - `hallucinated_rate`: run_sd **0.0162** → ε ≈ 0.016
   - `gold_citation_recall`: run_sd **0.0000** (deterministic — gold-key match; keep ε=0) 
   - (paired gates citation_support_precision/faithfulness already use the paired test + floor;
     report their run_sd 0.0176 / 0.0135 for transparency.)
   Implementation: `verdict()` gains an `absolute_epsilons: dict[str,float]` (default the measured
   map); `verdict_fullcorpus.py` passes it. Mean-vs-mean still fail-closed on empty/partial tables.
3. **Re-run the existing verdicts** under the noise-aware gates: `lctx` flips to `win_strict=TRUE`
   (hallucinated worse_by 0.0022 < ε 0.016); `cap20` (worse_by 0.0039 < 0.016) also clears its
   gate — re-check whether its other axes still qualify; `capwide`/`vmqsr`/V-MQ+V-SR unaffected.

## Caveat to keep honest
gold_citation_recall is genuinely deterministic (run_sd 0.0000) — its gate stays ε=0 (a real drop is
real). Only the synth-derived rates get a noise band. This keeps the fix from becoming a blanket
gate-loosener: each ε is the metric's MEASURED noise, nothing more.

## Optional (not required for the lctx win)
A "baseline at the new regime" (single-query + MAX_TOTAL_TOKENS=300000 + chunk_top_k=60) would
isolate the token-unlock's effect on the NON-multiquery path. Not needed for the total-improvement
claim (lctx vs original) or the lever isolation (lctx vs V-MQ+V-SR = +0.05). Run only if we want
that specific decomposition.
