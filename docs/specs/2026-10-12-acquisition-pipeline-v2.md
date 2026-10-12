# Acquisition pipeline v2 — identity ladder, guarded re-open, durable budgets, terminal taxonomy

Status: accepted

Challenge record: four adversarial review rounds (astra, `gpt-6-astra` @ max effort) over v0→v1→v2→v3→v4; convergence declared under the R4 closure condition (see Appendix C). Owner rulings recorded 2026-10-12 (Appendix B). Issue: #174.

## Problem & context

The per-source repair program (#176–#208) made every download tier work in isolation, but the pipeline's decision layer — identity resolution, terminal states, re-open, politeness accounting, merge concurrency, rollback — remains ad hoc. The residual failures are contract-level, not tier-level:

- **Identity**: CORE's stored `paper_id` is a *work* id; work/output namespaces collide numerically (#203 pilot: ~2% recovery ceiling, abstains correct). DOI↔URL disagreements have no defined resolution path. 3,465 documents sit at abstract tier and 1,384 of the #157 cohort are permanently missing — partly a metadata-fusion failure, not a download failure.
- **Re-open**: `retry` was a state without semantics; #157's 2,993 recovered papers needed a hand-run campaign. ADS's 642 recoverable routes (#205) have no defined path back into the pipeline.
- **Politeness**: rate windows were enforced per-tier in places, never at a single request boundary with a single ledger; the #173 probe could not even run because attribution ("which cap binds?") is undefined without controlled variation.
- **Merging**: concurrent writers can produce ghost states (the CORE-era lesson); identity promotion/merge has no drain protocol.
- **Rollback**: no compatibility matrix across scheduler/schema/identity-version/budget-ledger; irreversible operations (promotion, merge, budget decrement) have no stated post-incident posture.

This spec settles the contracts **before** implementation, so the v2 work is a migration against frozen interfaces rather than another round of ad-hoc repair.

## Options considered

1. **Keep patching the current cascade.** Rejected: every repair round since #157 has re-manifested the same three contract gaps (identity, re-open, accounting); patching leaves the state model conflated and the next incident recreates the campaign.
2. **Full adaptive redesign** (learned ordering, per-request dynamic caps, calendar-based staged rollout). Rejected: the feedback data is censored (only attempted tiers are observed), so learning would drift; and our operating style is ship-to-prod with fast forward-fixes, not calendar gates.
3. **Contract-first redesign on the existing skeleton** (chosen): keep the tier functions, freeze the contracts below, migrate incrementally behind them. Rejected alternatives within this frame are recorded per-contract in Appendix A.

## Decision

The pipeline keeps its tiers; the following contracts become normative (full text in Appendix A).

- **Identity ladder (D-A)**: DOI-first waterfall — "有 doi 的先用 doi，没有 doi 的用 a，连 a 都没有的用 b" — with correspondence staged into four phases (evidence collection → correspondence resolution → identity adoption → retrieval+verification). Rejections are three-grained: route-rejected ≠ identity-rejected ≠ exhausted. Routes are rejected individually; a paper never permanently fails because one route contradicted. Full namespaces (`core:work:123` ≠ `core:output:123`); `unresolved ≠ absent`.
- **Terminal model (D-B)**: three orthogonal dimensions per record — retrieval / policy / merge — with a guarded transition table; `terminal-reason` is only a projection. One admission predicate (v4): `acquire ⟺ ¬quarantine ∧ retrieval=eligible ∧ canonical ∧ identity=resolved ∧ effective_policy=allowed ∧ lease=valid ∧ budget=admitted`, applied at pre-dispatch recheck. Re-open (selector) flips retrieval-exhausted → eligible only; it never touches policy. Policy overrides are separate, revisioned operations. Legacy terminal records (cause unrecorded) are treated as policy-unknown.
- **Re-open correctness (D-C)**: atomic claim on (state, generation) + lease TTL + durable budget reservation *before* dispatch + stale-result fencing on both state writes and artifact publication + idempotent replay. Lease expiry ≠ execution termination: expired-uncertain moves the paper to quarantine until drain evidence. **Authorization (owner ruling 4)**: the orchestrator runs scoped, capped, recorded re-opens autonomously under predeclared stop rules; default standing cap ≤200 papers per campaign, ≤3 campaigns/week; anything larger goes to the owner.
- **Politeness & accounting (D-D)**: every upstream's *full set of applicable windows* is enforced at the real request boundary (e.g. 3/s **and** 60/10min are parallel constraints, enforced with sliding-window accounting, not token-bucket approximations); *all* request paths (workers, canary, backfill, route resolvers, redirects/retries, browser requests at the send point) post to one per-scope ledger; accounting unavailable ⇒ fail-closed. Documented limits start conservative (owner ruling 2); undocumented upstreams run a bounded envelope (≤50% of the nearest documented analogue, concurrency 1, +≤25%/24h only on positive-completion evidence, stop at cap or first 429/timeout cluster).
- **Merge & identity promotion (D-E)**: single-writer, drained operations — stop new claims on the affected keys, bounded drain wait, then write. Merges redirect aliases and take the *union of restrictions* / *intersection of grants*; budgets combine conservatively (spent = sum, budget = min). Promotion/merge are irreversible operations.
- **Telemetry & measurement (D-F)**: events carry (identity-version, attempt-generation, lease-id, request-scope-id); wait/service/drain durations separated; incomplete attempts are censored and never imputed. Stop rules are executable (detector/threshold/actuator/resulting state): duplicate in-flight identity > 0 ⇒ halt campaign; quota-window violation > 0 ⇒ scope cooldown + halt; identity-mismatched publication > 0 ⇒ halt + quarantine. The scenario matrix (reopen-failure / lower-rung / conflicting-identity / active-merge / timeout-drain / policy-excluded) is the **success-measurement set** (≥20 attempts/scenario default), **not a deployment gate** (owner ruling 3).
- **Ordering (D-G)**: static hybrid order + route-first as the operating mode; **dynamic ranking stays off** until an off-policy evaluation design exists (owner ruling 1).
- **Rollout posture (D-H)**: ship to production, fix forward (owner ruling 3); the compatibility matrix + drain barrier (below) still gate *deploys that change on-disk/durable formats*. Batch **irreversible** operations (thousand-scale merges, mass terminal marking) get a bounded dry-run (tens of items) before the full batch. Dynamic ranking remains off. No maintained forward-recovery build (owner ruling 8): post-irreversible incidents are handled ad hoc, with short downtime accepted.

## Out of scope

- Tier-level download mechanics (per-source fixes already landed in #186–#196; new tiers are separate issues).
- The knowledge-store / graph side (KS build, LightRAG) — governed elsewhere; this spec only fixes the *library* acquisition pipeline.
- Dashboards/UI for any of this; telemetry stays queryable (`cli stats`), no visual layer.
- Numeric cutover gates (exact limiter values, threshold numbers) — those land as implementation-time issues; the challenge declared them non-blocking for admission.
- Serving/query behavior of the KS store.

## Verification

Machine-judgeable end-to-end checks, all in CI:

1. **Contract fixtures** (shared by both the metadata writer and the scheduler): CORE namespace collision; DOI↔URL contradiction (route rejected, paper retained); merge-with-in-flight-work (both records carrying queued/active work); re-open after drain; four rollback scenarios (after reopen / after promotion / after success / mid-flight timeout) with failure injection — expected-result table in Appendix A (A3).
2. **Admission predicate tests**: every one of the six conditions (plus `¬quarantine`) blocks its acquisition case; selector never mutates policy.
3. **Accounting tests**: 61-request counterexample (0.35s spacing across 21s satisfies 3/s but must violate strict 60/600s — sliding-window implementation must reject it); two tiers sharing one upstream share one ledger; accounting-down ⇒ no request is emitted.
4. **Production instrumentation**: exposure measurements per scenario class are queryable from telemetry within one campaign; stop rules fire on injected conditions in staging fixtures.
5. **#173-style bounded observation** stays available for future causal questions (owner ruling 5 deferred the one-shot cap-variation experiment; the machinery is the existing probe plan of PR #202).

## Failure detection & rollback

This change touches shared interfaces, the on-disk ledger formats, and irreversible operations, so:

- **Detection**: the executable stop rules above (duplicate in-flight, quota violation, identity-mismatch publication) plus per-scope cooldowns; all events carry generation/lease/scope ids so a partial failure is attributable.
- **Compatibility matrix** (normative): scheduler binary ↔ schema ↔ identity-definition version ↔ budget-ledger format ↔ queue/lease formats; νN reads νN−1..νN; an old binary meeting new-format data fails safe (stops, does not corrupt).
- **Drain barrier**: deploys that change formats are performed as stop-intake → verify-drain → flip; the deploy mechanism must prevent old workers restarting after the barrier (versioned start path + verification step as a pre-flight check item).
- **Irreversible operations** (promotion, merge, append-only budget decrement): allowed only with validity fencing (fresh `budget=admitted` and lease at dispatch; publication only for the current generation). Post-incident posture: ad hoc repair (owner ruling 8) — rollback to νN−1 is supported only *before* an irreversible point in that window; after it, the supported move is a forward fix, with short downtime accepted as needed.
- **Kill switches**: the v2 path ships behind flags (consistent with ADR 0004's tier gating discipline), so a bad rollout can be disabled without format rollback.

## Appendix A — Normative contracts (astra-challenged v1–v4; authoritative for implementation)

Conflict rule: this appendix overrides any summary above on conflict.

### A1. Terminal transitions (v2-D1, v3-A1, v4)

- Records carry three independent dimensions: ① retrieval ∈ {ineligible, eligible, in-progress, exhausted, acquired, verified}; ② policy ∈ {clear, excluded(reason), override(scope, revision), unknown}; ③ merge ∈ {canonical, alias→key}. `quarantine` is an independent marker, not part of the enum. The selector must evaluate all three.
- Normative transitions: retrieval allowed ⟺ policy-clear-or-validly-overridden ∧ valid lease ∧ identity-resolved; `reopen(selector)`: retrieval-exhausted → eligible only, never policy; policy override = separate operation bound to (reason-scope, revision), double-checked at dispatch and at publication; on merge: alias redirect + **union of restrictions (strictest wins)**, grants intersect; "successfully acquired, later policy-excluded" is a legal coexistence (servable, never re-acquirable).
- Publication gate: only if current generation ∧ policy recheck passes ∧ identity recheck passes.
- **Single admission predicate (final, v4)**: `acquire ⟺ ¬quarantine ∧ retrieval=eligible ∧ canonical (not alias) ∧ identity=resolved ∧ effective_policy=allowed ∧ lease=valid ∧ budget=admitted`. Applied at the pre-dispatch recheck. Quarantine release still requires drain/fencing evidence (A2/D-C).
- `effective_policy`: `allowed ⟺ policy=clear (whose reason is not overridden by any exclusion) ∨ override_valid`; `override_valid ⟺ scope covers the action ∧ revision ≥ required ∧ merges never amplify a grant`; `policy=unknown ⇒ not allowed`; resolution of policy-unknown requires an explicit owner/orchestrator ruling action (cross-referenced with #198); an ordinary re-open selector never touches policy.
- Legacy terminal (cause unrecorded): treated conservatively as policy-unknown until re-evaluation.
- New evidence permits **identity re-evaluation**; after successful re-evaluation, retrieval must still pass the full admission guard.
- Ruling 9 (owner): policy-excluded papers' existing artifacts remain served, read-only, marked with the exclusion reason; all further retrieval stays forbidden. This supersedes the earlier unconditional "servable" wording in v2-D1.

### A2. Undocumented-upstream envelope (v3-A2, v2-D3)

- No nearest-analogue limit, or initial/hard cap unconfigured ⇒ that scope is **disabled (zero requests)** until the envelope is configured and authorized.
- Two separate "stop" executors: ① cap-bound ⇒ freeze at the cap; raising it needs re-authorization (executor = config; authority = owner); ② 429/timeout cluster ⇒ scope cooldown; after the quiet window, resume in place *without* speed-up (automatic, recorded).
- Positive completion is the *speed-up* criterion (under the bounded hypothesis), never proof of upstream permission. Undocumented envelopes are always accounted as explicit bounded assumptions.
- Envelope start values (owner ruling 2): ≤50% of the nearest documented analogue, concurrency 1, +≤25%/24h on positive-completion evidence only.

### A3. Merges, leases, budgets (v2-D2, v2-D3, v4-microfix)

- Claim: atomic CAS on (state, generation) + lease TTL; pre-dispatch recheck; **durable budget reservation before dispatch**; on crash-uncertainty, default is *no refund* (refund only on proven-not-sent).
- Consumption survives restart/replay/rollback: the budget ledger is independent of process state, and is an element of the compatibility matrix.
- Generation fencing covers state writes *and* artifact publication (stale results must not publish, not merely not mutate state).
- Lease expiry ≠ execution termination: expired-uncertain ⇒ quarantine (no new retrieval), pending drain evidence or human escalation; resources remain accounted during the uncertainty.
- Merges require the quiescence protocol: stop new claims on the affected keys, bounded drain wait, then the single writer merges. Budgets combine conservatively (spent = sum; budget = min; independent generation counters re-sequenced and persisted at merge). Fixtures must exercise merge-with-in-flight/queued work.
- Expiry of a merge-wait does **not** authorize the merge — the paper stays quarantined, resources keep counting; promotion is an operational action, not evidence of termination. (v3-A4)
- **Write-compatibility**: writers = scheduler + metadata writer (#198 process) + correction pipeline; all writers are inside the drain barrier; mixed writing is forbidden; the deploy mechanism must prevent an old worker from restarting after the barrier (versioned start path + verification step as a cutover pre-flight item). (v3-A3)
- **Four-scenario expected-result table** (fixtures, failure-injected, no production): identity = retained / canonical after promotion; policy = union of restrictions; queue = drained-verified at the barrier; generation = monotone, re-sequenced and persisted after merge; artifacts = publication fenced; budgets = decrement retained. (v3-A3)

### A4. Correspondence staging (v2-D4)

- Four phases with distinct evidence/authority: **P1 route-evidence collection** (URL reachable? carries meta? network allowed, spends budget, accounted as its own read-only request class with its own scope+budget, explicit policy permission — collection must never smuggle in unauthorized retrieval, v3-A4) → **P2 correspondence decision** (DOI↔URL consistency: canonical-DOI equivalence, title/author/year thresholds) producing resolved / unresolved (retryable) / **contradicted (the *route* is rejected, not the paper)**; rejected routes record their reason, other routes remain usable → **P3 identity adoption** (record-level, through the #198 contract) → **P4 retrieval + verification** (artifact verified against identity/version before publication).
- Rejection granularity: route-rejected ≠ identity-rejected ≠ exhausted; identity-rejected needs positive contradicting evidence and keeps a re-evaluation trigger (new evidence re-opens).
- Legacy-terminal causes fold into route/identity/exhausted with the conservative default.

### A5. Rollout, gates, and measurement (v2-D5, v2-D6, with owner rulings 3/5/8 applied)

- **Success measurement** (not a deployment gate): each scenario class accumulates ≥20 real attempts (default) before the owner treats the run as "proven"; scenario matrix = reopen-failure / lower-rung / conflicting-identity / active-merge / timeout-drain / policy-excluded; policy-excluded counts "correctly-refused admissions" (target: 0 forbidden retrievals). (v3-A4)
- **Event semantics**: every event carries (identity-version, attempt-generation, lease-id, request-scope-id); wait/service/drain separated; admission-denied event; verified-success event; censored attempts never imputed.
- **Executable stop rules** with named thresholds (numbers land at implementation): duplicate in-flight identity > 0 → stop campaign; rate-window violation > 0 → scope cooldown + stop; identity-mismatched publication > 0 → stop + quarantine. "Significant regression" must be a named threshold, not an adjective.
- **Compatibility & rollback** (v2-D5): matrix as in Failure-detection section; irreversible list = identity promotion, merge, budget decrement (append-only); rollback drills = the four fixture scenarios.
- **Ship-to-prod, fix-forward** (owner ruling 3): no calendar or exposure deployment gate; versioning + fast deploys are the safety mechanism; the exposure measurements exist to *judge* success. Carve-out: batch irreversible ops get a tens-scale dry-run first.
- **No forward-recovery build** (owner ruling 8): incidents after an irreversible point are handled ad hoc; short downtime is acceptable; this is a non-blocking operational stance.
- **Deferred**: the one-shot cap-variation experiment (owner ruling 5) is not exercised — downloads are not the active bottleneck (backend OCR is); authorization lapses and is re-requested only if download speed becomes a real question.
- **Unfolding measurement plan**: the #173 probe (PR #202) remains the mechanism for bounded, naturally-dequeued observation windows (≤200 papers, ≤20 min, no manufactured work).

## Appendix B — Owner rulings (2026-10-12, issue #174 comment 6127580827)

1. Dynamic ranking exploration — adopted: OFF in the first release.
2. Limiter initial values — adopted: documented-conservative starts; bounded envelope for undocumented upstreams.
3. Shadow period & canary — **amended**: ship straight to production and fix forward; exposure ≥20/scenario retained as success-measurement only, not a deployment gate; batch irreversible operations get a bounded dry-run (orchestrator carve-out).
4. Re-open authorization — adopted: orchestrator runs scoped/capped/recorded re-opens; default ≤200 papers, ≤3 campaigns/week.
5. Bounded cap-variation experiment — **deferred** (downloads not the bottleneck; OCR is); authorization lapses, re-request when needed.
6. Terminal-reason taxonomy & transitions — adopted as written.
7. Identity promotion/merge single-writer, drained — adopted.
8. Rollback posture after irreversible boundaries — **owner's call**: no maintained forward-recovery build; handle incidents ad hoc (short downtime / manual repair acceptable); non-blocking.
9. Serving policy for policy-excluded artifacts — adopted: serve existing artifacts read-only and marked; retrieval stays forbidden.

## Appendix C — Challenge record

Four adversarial rounds by `gpt-6-astra` @ max effort (a clean reviewer that did not write the spec), archived on #174:

- **R1** (comment 6110173617): 15 findings — identity ladder formalization (CORE namespace collisions; DOI/URL conflict rules), politeness = all applicable windows at the request boundary across all callers, terminal taxonomy + transitions, re-open correctness (atomic claim/lease, attempt generations, stale-result fencing, idempotent replay), serial-per-paper = executor termination/drain + identity-level exclusion, rollback contract, probe causal limits, ordering objective, concurrency dimensioning, route-layer accounting, shadow limits, rollout coverage, dynamic ranking, telemetry semantics, ledger balance. → absorbed into v1 (6110177561).
- **R2** (6110257728): audit of v1 — caught a summary-count slip (13 vs 15) and the #157 ledger discrepancy (subsequently reconciled exactly: 2,959 ok + 34 extract-failed = 2,993 recovered; 1,384 missing; sums to 4,377) — and seven new fracture points. → D1–D7 (6110280419).
- **R3** (6110352943): verdict "not converged, conditions listed" (P0-1..P0-4). → v3 A1–A4 closed them (6110353154).
- **R4** (6110401442/6110401660): one narrow blocker — `quarantine` missing from the admission guard; supplied the normative fix `acquire ⟺ ¬quarantine ∧ [six conditions]`; declared R3's conditions closed with that amendment = **converged for owner review**.

Finding ledger (15 rows R1 → closures): #1 identity → A4 staging; #2 politeness → A2/D-D; #3 terminal → A1; #4 re-open → A3; #5 serial → A3 termination semantics; #6 rollback → A5; #7 probe causality → corrected (ruling 5 defers the experiment); #8 ordering → objective defined (static hybrid; ruling 1); #9 concurrency → dimensioned (D-D + A2); #10 route accounting → A4 + D-D; #11 shadow → corrected (A5 measurement); #12 rollout coverage → A5 measurement; #13 dynamic ranking → default off; #14 telemetry → A5 semantics; #15 ledger balance → resolved exactly on #157.

Numeric implementation gates (limiter values, threshold numbers, fixture thresholds) are explicitly non-blocking for this spec's admission (R4 closure condition); they land as implementation-time issues under the Q5 ordering.
