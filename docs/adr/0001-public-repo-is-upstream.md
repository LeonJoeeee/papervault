# 0001 — This repo is upstream; the originating lab is deployment #1

Status: Accepted (2026-07-17)

## Context
papervault's code originates in a private research monorepo, where the two source services were developed and still run live. The code needs exactly one development home; the alternative was keeping the private monorepo as the source of truth and exporting snapshots here.

## Decision
papervault is the sole development home for this code. The originating monorepo deletes its copies and consumes papervault releases like any dependency, becoming deployment #1. Rejected: private-truth + snapshot export — a permanent sync tax, a second-class public repo, and outside issues/PRs that would need hand-carrying back into a private tree.

## Consequences
One-time migration cost, including a live-service cutover for the lab instance. After that: one place to fix bugs, one place for issues and PRs. The lab loses in-tree hackability — its changes go through papervault like everyone else's.
