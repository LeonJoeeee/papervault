# 0006 — Acquisition pipeline v2: contract-first redesign (identity ladder, guarded re-open, durable budgets)

Status: Accepted (2026-10-12)

## Context
The per-source repair program (#176–#208) completed tier-level repair, but the residual failures are contract-level, not tier-level: identity fusion (CORE's work/output namespace collisions — ~2% recovery ceiling in the #203 pilot; 3,465 abstract-tier docs; 1,384 permanently-missing of the #157 cohort), re-open semantics (the #157 campaign had to be hand-run; ADS's 642 recoverable routes have no path back in), politeness accounting (no single request-boundary ledger across callers; the #173 probe lacks causal attribution), merge concurrency (CORE-era ghost states), and rollback across durable formats. The #174 design spec survived four adversarial review rounds (`gpt-6-astra` @ max effort) and nine owner rulings before this ADR.

## Decision
Adopt the contract-first redesign recorded in `docs/specs/2026-10-12-acquisition-pipeline-v2.md`: a three-dimension terminal model (retrieval / policy / merge) with a single admission predicate; a DOI-first identity ladder with four-phase correspondence staging (route-rejected ≠ identity-rejected ≠ exhausted); atomic-claim/lease/generation re-open with durable, pre-dispatch budget reservation and stale-result fencing; all-applicable-window politeness accounting at one request boundary across all callers; single-writer drained merges with conservative budget combination; and a ship-to-prod / fix-forward rollout posture with exposure as success-measurement (not a deployment gate).

## Consequences
- Static hybrid ordering; dynamic ranking stays OFF until an off-policy evaluation design exists (owner ruling 1).
- Scoped autonomous re-opens under a standing cap (≤200 papers/campaign, ≤3 campaigns/week; ruling 4); larger campaigns go to the owner.
- Merges/promotions become operationally heavier (quiesce → drain → single writer) — deliberate.
- The compatibility matrix + drain barrier gate format-changing deploys; batch irreversible operations get a tens-scale dry-run first.
- No maintained forward-recovery build; post-irreversible incidents are handled ad hoc, short downtime accepted (ruling 8).
- Implementation proceeds as scoped issues under the Q5 ordering; numeric limiter values, thresholds, and cutover-gate numbers land there (non-blocking for this decision, per the challenge's closure condition).
