# Gold re-validation for the FULL corpus (downstream drill, 2026-06-06)

The downstream eval gold (`gold.jsonl`: 26 strong + 13 thin + 10 traps = 49 Q) was authored over the
fixed 89/100-paper `test100` subset. The full prod `l0` graph is now **3,671 papers / 226,809 entities**.
This note records why the gold is INVALID on the full corpus and what re-authoring is needed.

## Finding 1 — all 10 traps are DEAD on the full corpus

Traps assert "the corpus does NOT cover this topic → the system must refuse". They were grep-verified
absent in `test100`. On the full vault (`/data/paper-vault/extracts/md`, 3,668 files) every trap topic is
now substantially covered, so a correct system should now ANSWER them — the refuse-expected logic and the
over-confidence guardrail are inverted:

| trap | full-vault files |
|---|---|
| substorm onset / magnetotail trigger | 30 |
| CR ionization in protoplanetary disks | 4 |
| GLE #5 1956 spectrum | 103* |
| solar-flare neutrino detection | 3 |
| reconnection rate (Petschek/Sweet-Parker) | 44 |
| tokamak runaway electrons | 53 |
| exoplanet atmospheric escape | 84 |
| auroral kilometric radiation (AKR) | 13 |
| ionospheric plasma bubbles / scintillation | 31 |
| DM direct detection (WIMP/xenon) | 60 |

(*loose regex; topic clearly present either way.) → **The 10 traps must be removed/replaced.**

## Finding 2 — the trap METHODOLOGY does not transfer to the full corpus

The original traps were the valuable kind: **plausible heliophysics-ADJACENT topics** the corpus happened
to miss (so a naive RAG is tempted to hallucinate → discriminating). On the full corpus, **every adjacent
topic is now covered** — the "hard adjacent trap" category is gone. A grep sweep for genuinely-absent
topics returns only FAR off-domain ones (condensed-matter / AMO / chemistry): sonoluminescence (0),
optical tweezers (0), negative-index metamaterial (0), spin ice (0), time crystal (1), superhydrophobic
(1), quantum Hall (2), NMR (2), Feshbach (2). These are **easy** traps (the system obviously has nothing →
trivial refusal → low discrimination). **Decision needed (human):** keep easy off-domain traps as a basic
over-confidence guardrail, OR reduce trap reliance and test refusal a different way (e.g. answerable-but-
out-of-scope, or in-corpus questions whose specific sub-claim is absent).

## Finding 3 — the 39 answerable Qs have INCOMPLETE gold_keys

`gold_keys` ⊆ `test100` (89 papers). On 3,671 papers, more papers now answer each intent, so
`paper_recall` / `gold_citation_recall` are capped/unreliable (a perfect system retrieving a newer, equally-
valid paper is wrongly penalised). The intents + nuggets are still sound; the gold_keys need EXPANSION
against the full corpus (retrieve candidates → Claude-judge "directly-answering?" → add). This is an
eval-pipeline-dependent task (retrieval + judge), and ground-truth authoring is human-in-the-loop.

## Deliverable produced this pass

- `experiments/eval/gold_traps_v2.jsonl` — 9 DRAFT replacement traps (off-domain, grep-verified absent),
  schema-valid (`gold_keys=[]`, `nuggets=[]`, `band=empty`). Honest caveat baked into each rationale:
  easier than the originals (see Finding 2). For review, not yet merged into `gold.jsonl`.

## What remains (human-gated)

1. Decide the trap methodology (Finding 2) — easy off-domain traps vs a different refusal test.
2. Expand the 39 answerable Qs' `gold_keys` for the full corpus (retrieval + judge; HITL review).
3. THEN run the downstream baseline (#23) on the re-validated gold, and the adversarial metric drill (#24).
