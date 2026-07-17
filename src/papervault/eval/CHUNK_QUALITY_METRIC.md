# KS material-quality optimization metric — v2 (for re-drill, 2026-06-03)

> **What this is:** the number that says how good the MATERIAL is that retrieval hands the final
> write-up step (synth). It will be the primary target for retrieval-side KS work (multi-query,
> chunking, merge, reranker). v1 was drilled and failed (5/5 reviewers `fix_first`, 16 high holes);
> v2 rebuilds it around ONE principle the user settled, which dissolves most of those holes. To be
> re-drilled before any build.

## The real goal it must faithfully track
**The retrieved material is the best possible input for synth to fully and accurately answer the
intent** — every thing the answer truly needs is present with real substance, with minimal useless
padding, and enough of it. The number must rise *if and only if* that genuinely happened.

## The ONE principle (settled with the user)
**Score every piece of material by ONE question: does it actually help ANSWER this intent?**
Useful → high; useless → low **even if it is on-topic**. Utility, never topic-overlap. This single
rule is what makes the metric hard to game (see "why this resists gaming").

## What gets scored (settled)
The metric scores **everything synth actually reads** — not more, not less:
- the **chunks** synth receives (target 20–50; = what we feed synth, see Quantity),
- the **entities** synth receives (up to ~40),
- the **relationships** synth receives (up to ~40).
**No deduplication by paper** (settled): if one paper supplies ten genuinely useful pieces, all ten
count — throwing them away would punish a paper that really is the best source. (The narrow case
where this could mislead is handled by Coverage being answer-aware, below — not by a dedup rule.)

## The two views (kept independent) + a view-only summary

### Per-piece usefulness (the precision-shaped view)
A judge scores EACH material piece (chunk / entity / relationship) 0–100 on its real contribution to
answering THIS intent, writing the counter-case first:
- **90–100**: directly supplies substance the answer needs — a passage/fact synth would quote, with
  the actual mechanism / number / equation **present in the piece itself**.
- **70–89**: genuinely useful supporting context or a real building block.
- **40–69**: on-topic but contributes nothing to THIS intent (cap 55).
- **0–39**: off-topic / keyword-only / "sounds related" with no usable substance (cap 39).
Score = average over all pieces, /100.

**Anti-bluff = the user's principle, stated as a cap:** a judge can always *argue* a piece "could be
useful." That does not count. A piece reaches 70+ only if it names the **specific substance** the
intent needs; "sounds related" caps at 39; on-topic-but-useless caps at 55. Two consequences fall
out for free (they were separate v1 holes):
- a chunk that only *points* ("see Section 4", "as shown in Fig. 10") without the substance in it →
  synth can't answer from it → **useless → low**. (Pointer-not-payload solved by the principle.)
- a cross-domain piece is judged the same way — by real usable substance, not by "sounds
  transferable" — so we do NOT need pl's cross-domain transfer ladder (KS is mostly single-domain).

### Coverage of what the answer needs (the recall-shaped view)
1. Fix the **things a complete answer to this intent truly needs** — its distinct required angles.
   On the gold set these come from the nuggets we already wrote; in general a judge derives them.
   **Answer-aware:** for a "synthesize / compare across the literature" intent, *having several
   independent sources corroborate* is ITSELF one of the needs — so a set that is all one paper
   genuinely fails to cover it. For an intent a single paper can fully answer, one paper covers it
   and that is correct.
2. A judge reads ALL the material and scores each required angle 0–3: 3 = fully answerable from the
   material; 2 = substantively supported; 1 = only touched/mentioned; 0 = absent.
3. Coverage = Σ band / (3·N). Depth bands stop "mention everything shallowly" from passing.

### Summary number (look only, never decides)
Harmonic mean of (coverage, per-piece-usefulness) — collapses toward the worse view so imbalance is
visible at a glance. It NEVER decides on its own.

## Why this resists gaming (the v1 holes, and why the principle closes them)
- **Pad with on-topic junk** → junk is useless → drags the per-piece average DOWN. Closed.
- **Pointer chunks / fragments that name a result but don't contain it** → useless to synth → low. Closed.
- **"Sounds relevant" bluff** → caps at 39 unless real substance named. Closed.
- **One paper smeared across many pieces** → each piece can be genuinely useful (so usefulness stays
  high, and that is FAIR — we don't dedup), BUT for a multi-source intent, Coverage's "needs several
  sources" angle is unmet → Coverage drops. Closed at the Coverage view, not by a dedup hammer.
- **Score 30 but synth reads 12** → we score EXACTLY what synth reads (Quantity). Closed.
- **Entities/relationships invisible** → now scored too. Closed.

## Decision rule (variant vs baseline)
- Better ⟺ both views improve, OR one improves and the other does not fall beyond the measured
  judge-noise band. The summary harmonic never decides alone (it has a cheap park-point; the pair
  does not).
- **No final-answer backstop (settled, D3):** we do NOT also require the downstream answer score to
  hold. This metric stands ALONE as the optimization target — which RAISES the bar on it: it must be
  trustworthy enough on its own, so the re-drill must hit proxy-validity hardest.

## Traps (questions with no real answer)
Never scored on the two views above (they have no required angles). Judged ONLY by a separate
should-refuse check: a trap's correct material-side outcome is "no piece earns a real-usefulness
score ≥ the substantive threshold for any plausible angle." Kept off the quality average entirely.

## Quantity
The scored set = synth's full input (chunks + entities + relationships). Chunk target 20–50; this is
also what synth reads (we will raise the synth chunk count to match). Pending: confirm the
`mimo-v2.5-pro` context window actually holds 20–50 chunks without quality loss — the design is robust
to the exact number (we score "what synth reads"), but the number itself must be set + measured, not
assumed. Below a floor of useful material = too little (also the honest "should refuse" signal).

## Still to settle / measure before trusting a verdict
- **Judge-noise band**: measure run-to-run noise of each view (multi-seed judge replays) BEFORE any
  verdict; reuse the answer-level paired-comparison machinery. (Without a backstop this is critical.)
- **Angle source**: fixed nuggets (trustworthy, test-set only) vs judge-derived (general). Leaning both.
- **Per-piece usefulness rubric** for entities/relationships (a fact/edge, not a passage) — calibrate.

## What the re-drill must answer (D3 raised the bar)
1. With NO final-answer backstop, is "better usefulness + coverage" genuinely the same as "synth
   writes a better answer", or can they still diverge (e.g. material has it but the fixed synth can't
   synthesize across pieces)? This is now the make-or-break question.
2. Does the utility principle REALLY close the v1 holes, or did any survive / mutate?
3. New holes introduced by v2 (scoring entities+relations; answer-aware coverage; no dedup; no backstop)?
4. What real material failure does usefulness + coverage + the trap check still NOT catch?
