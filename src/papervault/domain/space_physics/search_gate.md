# Research Domain — Space Physics, Space Computing, and AI4Science

## Core criterion

For each candidate paper, decide: does the paper's **method, physics, OR
space-computing system/hardware/networking contribution** have reusable value
for a researcher working on **space physics, space computing, and AI4Science**?

- If the paper is in our space-physics or space-computing domain → keep
- If the paper's method can be reused on our physics or space-computing problems → keep
- If the paper is a methodological advance (no specific application) → keep
- If the paper applies existing methods to an unrelated domain (medical,
  finance, generic NLP/vision, terrestrial commercial cloud/web/mobile, etc.)
  with no reusable scientific or constrained-compute contribution → reject.
  A transferable methodological contribution that is the main point can qualify
  under Tier 1B or Tier 2. Spaceborne remote-sensing/vision on-board processing
  and orbital networking/computing are core Tier 1C contributions.

## Tiers

### Tier 1 — Core (always keep)

- **1A — Space physics**: cosmic ray transport (GCR/SEP/UHECR/ACR),
  heliospheric physics, magnetospheric physics, ionosphere, solar wind,
  solar physics, space weather, plasma in space context, particle
  acceleration in space.
- **1B — AI4Science & scientific/efficient ML methods**: PINN/XPINN/PINO,
  neural operators (FNO/DeepONet/etc.), Bayesian inversion, Gaussian process,
  Neural Process, symbolic regression, equation discovery, surrogate models,
  scientific ML, physics-informed methods, inverse problem ML; model
  compression, quantization, pruning, and acceleration methods transferable
  to resource-constrained scientific or orbital inference.
- **1C — Cross & Space Computing (space physics × AI, OR orbital/spaceborne
  computing)** — HIGHEST PRIORITY: PINN for solar/SEP/GCR, ML for cosmic ray
  forecasting, neural inversion of space-plasma observations, GNN-PINN for
  multi-spacecraft, plus all five space-computing sub-directions:
  1. Orbital/satellite edge computing and spaceborne computing constellations:
     LEO/MEO/GEO computing constellations and in-orbit cloud/edge computing.
  2. On-board AI inference/training and real-time satellite payload,
     remote-sensing, or space-weather data processing.
  3. Space computing hardware architectures, FPGA/SoC/ASIC/NPU acceleration,
     and radiation-tolerant / fault-tolerant computing: SEE/SEU/SEL/TID
     mitigation, triple modular redundancy (TMR), radiation-hardened processors.
  4. Space-air-ground integrated computing (SAGIN) and inter-satellite link
     (ISL) task offloading, routing, and collaborative scheduling.
  5. Space data centers, orbital compute energy management, and thermal
     control systems for spaceborne compute.

### Tier 2 — Adjacent (usually keep)

- **2A — Adjacent physics or constrained edge/fault-tolerant systems × AI**:
  lab plasma + ML, geophysics + ML, climate/ocean + ML, fluid dynamics + ML,
  astrophysics + ML, particle physics + ML; embedded/edge AI, fault-tolerant
  distributed computing, and low-power hardware acceleration whose methods
  directly transfer to spaceborne resource/radiation constraints or our physics.
- **2B — General AI & systems methodology**: new neural architectures,
  optimizers, training techniques, and general efficient inference/systems
  methodology, with no specific physics application required.
- **2C — Adjacent physics or aerospace/radiation engineering (no AI)**:
  methods or physics directly transferable to space physics (e.g., shock
  acceleration in lab plasma); spacecraft electronics, semiconductor radiation
  effects, satellite communications, and spacecraft thermal/power engineering
  transferable to space computing.

### Tier 3 — Off-domain (reject)

- Application-focused medical / biology / agriculture / finance / business /
  education / generic NLP / generic vision / social science / theology / law
  papers without a transferable constrained-compute/AI4Science methodological
  contribution. Applying an existing PINN or neural architecture alone is not
  method novelty.
- Purely terrestrial commercial cloud/web/mobile applications with no
  orbital/spaceborne context or transferable constrained-compute/AI4Science
  methodology; other non-physics work without such a contribution.
- Spaceborne remote-sensing/vision on-board processing and orbital
  networking/computing are **Tier 1C**, NOT rejected as generic vision, NLP,
  or business applications.

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

### Example 11 — Tier 1C orbital edge computing (accept)

```
Title: "Collaborative task offloading in LEO satellite edge computing constellations"
Abstract: We schedule on-board compute and inter-satellite link routing in a
space-air-ground integrated network under orbital contact and power constraints...
```
→ `{"keep": true, "score": 0.9, "tier": "1C", "rank_reason": "Orbital edge computing with ISL offloading and collaborative scheduling"}`

### Example 12 — Tier 1C radiation-tolerant on-board AI (accept)

```
Title: "Radiation-tolerant FPGA acceleration for on-board remote-sensing neural inference"
Abstract: Quantized inference and triple modular redundancy mitigate single-event
upsets while processing satellite imagery within payload power and latency budgets...
```
→ `{"keep": true, "score": 0.95, "tier": "1C", "rank_reason": "Spaceborne FPGA neural inference and SEU mitigation; core computing, not generic vision"}`

### Example 13 — Tier 1C orbital data center engineering (accept)

```
Title: "Joint energy management and thermal control for orbital data centers"
Abstract: We coordinate compute loads, solar power, and radiator capacity across
orbital eclipse cycles to maintain safe processor temperatures...
```
→ `{"keep": true, "score": 0.9, "tier": "1C", "rank_reason": "Power and thermal systems directly support spaceborne compute"}`
