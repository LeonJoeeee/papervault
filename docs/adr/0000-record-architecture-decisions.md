# 0000 — Record architecture decisions as ADRs

Status: Accepted (2026-07-17)

## Context
papervault is developed by coding agents across parallel sessions. Without a decision log, every session re-derives "why is it this way" at real cost.

## Decision
Significant decisions are recorded as ADRs in `docs/adr/NNNN-kebab-title.md`. A decision earns an ADR when either axis fires: it touches top-level design (structure, interfaces, dependencies, a key quality goal), or it is costly to reverse. Trivial choices, spikes, and anything already covered do not. Supersede, never edit; dated amendments are legal for partial supersession or factual correction. Numbers are assigned at merge; in-flight branches draft as `DRAFT-kebab-title.md`.

## Consequences
`ls docs/adr/` is the index of why the project is shaped as it is. A log that records everything protects nothing — admission discipline is part of the rule.
