# 0003 — Domain ships as an editable config pack; space physics is the factory default

Status: Accepted (2026-07-17)

## Context
The search ingest gate (domain rubric + few-shot judgments), the knowledge-graph entity ontology, the extraction few-shots, MCP instruction text, and the eval gold sets are all space-physics-specific and were hardcoded in the source services. The first release targets space-physics researchers (the originating group's beta); other fields are welcome but not accommodated.

## Decision
Externalize the domain texts into operator-editable config files — the "domain pack" — shipped with space-physics values as the factory default and worked sample; setup docs explain each file's role. No preset registry, no plugin system. Rejected: a multi-preset mechanism (complexity with no second validated domain to justify it); keeping the texts hardcoded (locks the product to one field and blocks even willing adapters).

## Consequences
Swapping domains = editing files, zero code changes. Quality outside space physics is the adapting operator's responsibility, stated plainly in docs. Gold sets remain space-physics; the eval harness documents how to build one for another field.
