# 0007 — Ratify the papervault constitution and refactor plan

Status: Accepted; ratified by the owner (2026-10-18).

## Context

Four failures motivated a governing contract for the redesign, as recorded in the
[owner's source material on issue #260](https://github.com/LeonJoeeee/papervault/issues/260#issuecomment-6249378204):

1. **Serving-stage loss.** A request explicitly naming PAMELA and AMS-02 served no
   original measurement papers. The query-construction defect was fixed in #248/#256,
   but all three originals had already been in the library since May–August 2026;
   fusion/return-gate losses still kept them out of the answer. A duplicate record
   pair also existed. Useful retrieval must be a primary contract alongside storage
   integrity.
2. **Metric confusion.** Measuring must-find recall on one served list conflated
   answering the current need with coverage of the corpus's canon. The constitution
   separates named-need recall, current canon coverage, precision with recall, and
   mechanical integrity as R1–R4.
3. **Structural tangle.** Search wrote records through several paths, ingest feedback
   reinforced frontier bias, and source adapters exposed different error shapes.
   Lifecycle and retrieval were entangled: legacy `policy=UNKNOWN` records prevented
   re-acquisition without a ruling path until #245/#257 supplied one. The operator
   command `audit --retry-metadata-only` also wrote status outside the state machine.
4. **Hygiene drift.** Evaluation artifacts accumulated under `src/` (about 259 MB),
   worktrees grew without bounds (about 10 GB), secret-bearing `.env.bak-*` files
   accumulated in the repository root, and scratch accumulated. Explicit retention,
   bounded operation, and protected credentials are governing concerns.

## Decision

Adopt the owner-ratified [constitution v2.1-rc2](../specs/2026-10-18-papervault-constitution.md)
and [refactor plan v0.3](../specs/2026-10-18-papervault-refactor-plan.md).
The constitution governs; it takes precedence over the plan and earlier architecture
descriptions. Durable collection integrity and useful, evidence-backed retrieval are
joint primary contracts. The plan implements them through reviewable migration phases
inside this repository, preserving the MCP surface and incident-hardened capabilities.
These are target contracts, not a claim that the current runtime already meets them.

The rewrite defines three **logical persistence roles**, not a requirement for three
separate database engines:

- **Source:** immutable, content-addressed paper artifacts and operator documents,
  retained as research assets; notebook-class documents remain private-instance-only.
- **Catalog:** the transactional authority over identity, policy, registration,
  lineage, and processing/publication state, including metadata-only records that
  precede acquisition.
- **Derived:** extracted text, indexes, embeddings, and graph data, with catalog/input
  revisions and versioned recipes, directly or through lineage manifests. Rebuilds
  create new versions; expensive derivatives are retained when recovery objectives
  require them.

There is one authoritative home per catalog fact. The authority store must provide
atomic mutation/audit/pending-work commits, constraints, multi-writer concurrency,
and crash recovery. The engine is selected in the F1 ADR; **Postgres is the current
selection**, already operated for the knowledge layer. The JSON index remains a
canonical, human-readable **export**, never catalog authority. Artifact bytes are
published durably before references commit; catalog publication records identify
persisted derived versions and covered inputs, since job completion alone does not
prove indexed coverage.

One-way derivation constrains dependencies: computation does not make a derived
claim authoritative, while validated registration and completion commands may update
the catalog. Bibliographic records use work@version identities; evolving operator
documents use stable document identities linked to immutable evidence revisions.
Reads declare scope, compatible evidence versions, freshness, and partial or
unavailable materialization while enforcing applicable policy and evidence availability.

**Serving is a read.** Search may emit bounded operational telemetry and serving
traces and submit registration requests, but registration executes as a separately
admitted command selected independently of the served list. Serving neither waits
for nor depends on registration admission, execution, success, or acquisition.
External results can carry stable source identities before receiving internal IDs.
Execution judgments belong to pipelines; domain integrity judgments and transactional
changes belong to the domain/persistence block.

## Review trail and owner ratification

The [original owner record](https://github.com/LeonJoeeee/papervault/issues/260#issuecomment-6249378204)
and [ratified-pair archive on #249](https://github.com/LeonJoeeee/papervault/issues/249#issuecomment-6249785828)
are the durable sources for the review history. The reviews were independent,
clean-context, read-only Codex `gpt-6-astra` rounds at max effort.

| Reviewed revision | Verdict | Recorded findings or disposition |
| --- | --- | --- |
| v1.x, round 1 (v0.1) | Reject | 11 findings and 7 amendments. |
| v1.x, round 2 | Substantially repaired; 3 blockers | Identity unit, projection serving contract, configuration-mismatch breadth; also R3 recall wording. |
| v1.x, round 3 | ACCEPT | Revision 3 accepted, with implementation mechanisms deferred to ADRs. |
| v2.0 systematic rewrite, round 4 | Coherent; NOT YET | Minimal blocking set 1–3; v2.1 applies all five amendments: layer/authority/lineage precision, serving effects, operator-document identity, publication versus coverage, and read-operator contract. |
| Joint review: constitution v2.1 + plan v0.1 | Constitution ACCEPT AS-IS; plan NOT YET | Plan had 4 blockers. |
| Plan v0.2 confirmation | One residual blocker | Amendments 1–5 applied; §5 risk table still needed alignment with the P1/P2 handoff protocol. |
| Plan v0.3 final confirmation | PLAN ACCEPT | §5 aligned; remaining blocking set: none. |

The owner's clarifications developed in two rounds after the v1.x reviews: first,
papers and the graph were identified as two databases with explicit relationships;
then the owner expanded this to **three data domains** — papers, operator-supplied
documents (textbooks/notebooks), and knowledge (the graph). The owner explicitly
endorsed Postgres authority for the papers catalog and declared the constitution in
force. The systematic rewrite realizes these corpus distinctions through the
Source/Catalog/Derived roles above, with transactional engine selection left to F1.

The [final owner instruction on #260](https://github.com/LeonJoeeee/papervault/issues/260#issuecomment-6250212187)
supersedes v1.1 with the ratified constitution v2.1 and plan v0.3. Both specs preserve
the supplied text verbatim, including the constitution's `v2.1-rc2` heading and
"candidate for ratification" status; the acceptance record here establishes their
ratified standing without editing that historical wording.

The owner's authorization boundary is **"up to before the shadow cutover"**:
the authority cutover is excluded and requires further authorization. Recording the
full plan does not authorize that cutover; this lane changes documentation only.

## Consequences

The constitution and slice plan precede code. Baseline instruments and relevant
authority/registration contracts precede behavior-changing slices. Each slice needs
proof on real material and an independent verdict.

- **F0 — Baseline freeze (P0):** capture legacy measurements before F1 can change
  behavior; version instruments, workloads, and judge configuration; archive the
  catalog snapshot.
- **F1 — Foundations (P1–P5):** transactional authority and its engine ADR,
  registration, schema versioning/migration, backup/recovery, unified coverage, and
  derived-layer discipline. Prerequisite contracts apply at each domain cutover;
  P2's authority cutover remains outside current authorization.
- **F3 — Search plane (P3 registration separation; P6 redesign):** separate serving
  and acquisition, diagnose losses by serving stage, and measure precision/recall.
- **F4 — Query plane (P7):** evidence-bound citations, indexed scope/freshness,
  evidence availability, and canon handling.
- **F5 — Lifecycle completion (P8):** generalize rulings/campaigns and operator
  visibility; #245/#257 supplied the first installment.
- **F6 — Structural cleanup (P8 and independent preparatory slices):** may start at
  any time with zero behavior change.

Five required mechanism ADRs carry the deferred details:

1. **Identity resolution and merge:** version-aware identity, operator-document
   revisions, ambiguity adjudication, aliases, and merge mechanisms.
2. **Migration/cutover:** one authority per fact, restartable imports and writer
   fencing, reconciliation, rehearsed handoff and failed-cutover recovery. A frozen
   snapshot is a rollback only before authoritative writes resume; afterward accepted
   mutations, audit, pending work, and attempted external effects must be preserved
   and reconciled, or recovery proceeds forward.
3. **File integrity and recovery:** restorable sets, bytes-before-references,
   garbage-collection versus publication races, and hash verification.
4. **Operator visibility:** observable stuck jobs, materialization lag, exhausted
   budgets, disk pressure, and divergence from publication records.
5. **Trust boundaries:** retrieved documents and model outputs cannot authorize
   commands, policy changes, or merges.

Mechanisms belong in ADRs; numerical acceptance thresholds and justified exceptions
belong in `docs/baselines.md`, separately from frozen measurements. Phase gates cover
integrity, reconciliation of actual materialization, behavioral non-regression and
serving independence, reversibility, and recovery from outside the host failure
domain. This decision lands the governing pair; runtime changes, baseline capture,
and the follow-up ADRs belong to their own slices.
