# 0005 — Track the latest LightRAG minor line, behind a port-map + gate discipline

Status: Accepted (2026-07-17)

## Context
LightRAG is papervault's framework spine and moves fast. Two failure modes were both
observed in one day: an UNBOUNDED floor (`>=1.4`) let a clean install drift to an
unvalidated 1.5.4 where the domain ontology and noise-suppression silently no-op'd
(retro-review blocker); and a hard pin to 1.4.x would freeze papervault off the
framework's bugfixes and features indefinitely. The user's call: stay on the latest
line — but never blindly.

## Decision
Pin to a CEILING-BOUNDED minor line (currently `lightrag-hku>=1.5,<1.6`). Raising the
ceiling to a new line is a gated port, never a version bump:
1. a fresh **port map** over the known touchpoint checklist (extraction-prompt channels,
   addon_params consumption, llm kwargs, ctor args, rerank interface, doc-pipeline APIs +
   DocStatus states, workspace envs, QueryParam fields, storage/schema migration compat);
2. the **machine-checkable prompt-content gates** green (`tests/knowledge/test_lightrag_port.py`
   — guidance completeness, escape-hatch kill, fail-loud anchors);
3. a **probe build** on a throwaway workspace asserting zero off-ontology nodes, plus the
   **FAST recall gate** vs the previous line's baseline (delta within the noise floor).
Rejected: floating latest (proven silent regression); freezing on a proven version
(compounding staleness, and the eventual forced jump only gets bigger).

## Consequences
papervault stays current at the cost of one gated port per minor line. The fail-loud
guards in `extraction_prompt.py` convert any missed touchpoint from silent degradation
into a boot-time error. The 1.4.16-era behavior remains the recall reference until a
new baseline supersedes it.
