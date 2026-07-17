# Research Domain — Space Physics + AI4Science

## Core criterion

For each candidate paper, decide: does the paper's **method OR physics**
have reusable value for a researcher working on **space physics +
AI4Science**?

- If the paper is in our physics domain → keep
- If the paper's method can be reused on our physics problems → keep
- If the paper is a methodological advance (no specific application) → keep
- If the paper applies methods to a non-physics domain (medical,
  finance, NLP, vision, etc.) — REJECT, unless the methodological
  contribution is the main point (rare; applications usually are not
  method-novel)

## Tiers

### Tier 1 — Core (always keep)

- **1A — Space physics**: cosmic ray transport (GCR/SEP/UHECR/ACR),
  heliospheric physics, magnetospheric physics, ionosphere, solar wind,
  solar physics, space weather, plasma in space context, particle
  acceleration in space.
- **1B — AI4Science methods**: PINN/XPINN/PINO, neural operators
  (FNO/DeepONet/etc.), Bayesian inversion, Gaussian process, Neural
  Process, symbolic regression, equation discovery, surrogate models,
  scientific ML, physics-informed methods, inverse problem ML.
- **1C — Cross (space physics × AI)** — HIGHEST PRIORITY: PINN for
  solar/SEP/GCR, ML for cosmic ray forecasting, neural inversion of
  space-plasma observations, GNN-PINN for multi-spacecraft etc.

### Tier 2 — Adjacent (usually keep)

- **2A — Adjacent physics × AI**: lab plasma + ML, geophysics + ML,
  climate/ocean + ML, fluid dynamics + ML, astrophysics + ML, particle
  physics + ML. Methods transfer to our physics.
- **2B — General AI methodology**: new neural architectures, optimizers,
  training techniques, no specific physics application (still useful as
  method input).
- **2C — Adjacent physics (no AI)**: only when methods or physics
  directly transferable to space physics (e.g., shock acceleration in
  lab plasma).

### Tier 3 — Off-domain (reject)

- AI applied to medical / biology / agriculture / finance / business /
  education / NLP / vision / social science / theology / law — REJECT
  even if the method is novel (the application is the focus, not the
  method).
- Non-physics without AI4Science methodological angle.

## Decision flow

1. Filter stage — Tier 3? → keep=false, score=0
2. Rerank stage — within Tier 1-2, score 0-1 by query relevance
   (ABSOLUTE scale, not relative to other candidates in batch):
   - 0.9+: directly addresses the query with high specificity
   - 0.7-0.9: clearly relevant, secondary topic but useful
   - 0.4-0.7: partial overlap, tangential
   - 0.2-0.4: same field but doesn't address query
   - 0.0-0.2: off-topic / off-domain

## Few-shot examples

### Example 1 — clear Tier 1C (accept)

```
Title: "XPINN-based inversion of Parker transport in the outer heliosphere"
Abstract: We use a physics-informed neural network to invert solar
wind parameters from Voyager 1/2 cosmic ray flux measurements...
```
→ `{"keep": true, "score": 0.95, "tier": "1C", "rank_reason": "XPINN inversion on Voyager CR, core cross"}`

### Example 2 — clear Tier 3 (reject)

```
Title: "Deep learning prediction of breast cancer recurrence"
Abstract: A CNN model trained on mammography images predicts breast
cancer recurrence with 89% accuracy...
```
→ `{"keep": false, "score": 0.0, "tier": "3", "rank_reason": "Medical ML application"}`

### Example 3 — Tier 1B method paper (accept)

```
Title: "Improving PINN training stability via causal sampling"
Abstract: We propose a new sampling strategy for PINN training that
improves convergence on stiff PDEs...
```
→ `{"keep": true, "score": 0.85, "tier": "1B", "rank_reason": "PINN training method advance"}`

### Example 4 — Tier 2B pure AI method (accept)

```
Title: "Adaptive learning rate via gradient noise estimation"
Abstract: We propose AdaNoise, a new optimizer that estimates gradient
noise to adapt learning rate per layer...
```
→ `{"keep": true, "score": 0.5, "tier": "2B", "rank_reason": "General optimizer, methodology input"}`

### Example 5 — tricky: PINN applied to non-physics (reject)

```
Title: "Physics-informed neural network for blood flow prediction"
Abstract: We apply PINN with Navier-Stokes loss to predict blood flow
in patient-specific arteries from CT scans...
```
→ `{"keep": false, "score": 0.0, "tier": "3", "rank_reason": "PINN application to medical, no method novelty"}`

(Note: if same paper instead proposed a *new* Navier-Stokes PINN loss
formulation, it would be Tier 1B → keep.)

### Example 6 — Tier 2A adjacent (accept)

```
Title: "Neural operator for global ocean circulation prediction"
Abstract: We train a FNO surrogate on CMIP6 simulations to predict
multi-decade ocean state...
```
→ `{"keep": true, "score": 0.55, "tier": "2A", "rank_reason": "Neural operator for ocean, methods transfer to MHD/space"}`

### Example 7 — Tier 2C borderline (accept)

```
Title: "Particle acceleration in tokamak runaway electron events"
Abstract: We model runaway electron generation in tokamak disruptions
using kinetic simulations...
```
→ `{"keep": true, "score": 0.5, "tier": "2C", "rank_reason": "Particle acceleration physics, transfers to space-physics shock acceleration"}`

### Example 8 — clear Tier 3 noise (reject)

```
Title: "Topic modeling of news articles using BERT embeddings"
Abstract: We apply BERT to cluster news articles by topic...
```
→ `{"keep": false, "score": 0.0, "tier": "3", "rank_reason": "NLP application, no physics or methods relevance"}`

### Example 9 — Tier 1A pure space physics (accept)

```
Title: "Voyager 1 observations of cosmic ray anisotropy beyond the heliopause"
Abstract: We report GCR anisotropy measurements from V1 LECP between
2014-2024...
```
→ `{"keep": true, "score": 0.9, "tier": "1A", "rank_reason": "Direct Voyager outer-heliosphere GCR observation"}`

### Example 10 — query-dependent score (accept, lower score)

```
Query intent: "PINN for solar energetic particle transport"
Title: "Bayesian neural network for galactic cosmic ray modulation"
Abstract: We use a Bayesian NN to model GCR modulation by solar activity...
```
→ `{"keep": true, "score": 0.4, "tier": "1C", "rank_reason": "AI for CR but Bayesian (not PINN) and GCR (not SEP); tangential"}`

(Note: keep because it's Tier 1C, but score reflects partial query match.)
