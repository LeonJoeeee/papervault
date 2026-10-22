# papervault refactor plan v0.3 (strangler → three-layer constitution)

Status (2026-10-22): **P0–P8 implemented, reviewed and merged on `main`**: F0,
F1/P1–P5 (including P5a–P5c), S1 search plane
[#251](https://github.com/LeonJoeeee/papervault/issues/251), S2 query plane
[#252](https://github.com/LeonJoeeee/papervault/issues/252), S3 lifecycle completion
[#253](https://github.com/LeonJoeeee/papervault/issues/253), and S5 hygiene
[#255](https://github.com/LeonJoeeee/papervault/issues/255). Umbrella
[#249](https://github.com/LeonJoeeee/papervault/issues/249) is closed; the AGY plugin
adaptation [#290](https://github.com/LeonJoeeee/papervault/issues/290) is also merged.

The phase descriptions and acceptance gates below retain the ratified plan's design
requirements. Implementation completion does not activate every operational surface:
deployment #1's library catalog is live on PostgreSQL, while catalog-owned knowledge
jobs/publication require their separate authority handoff and recovery/maintenance
tools retain their explicit operator gates. Current behavior and operational limits
are described in [architecture](../architecture.md).

Review history: constitution ACCEPT AS-IS; plan amendments 1–5 applied; §5 aligned
with the P1/P2 handoff protocol. This plan executes constitution v2.1-rc2; where they
disagree, the constitution wins.

## 1. Scope: what stays, what is replaced

**Preserved (incident-hardened, working):**
- MCP contracts: one FastMCP process, 3 tools (search_papers, get_paper, query) + 4 resources; admission control; access log.
- Capabilities: source adapters, LLM gates (intent parse, ingest judge, return judge), MinerU gate client, download tiers (26-tier cascade, ADS-native first).
- Pipelines: DownloadQueue (4 workers, recovery scan), ExtractQueue (GPU-scaled), library reconcile loop.
- Knowledge plane: LightRAG integration (PG KV/vector/doc-status + Neo4j), 60s distill scheduler, build breaker, off-peak gate.
- The test suite (2,800+) as the behavioral contract while refactoring.

**Replaced (the wrong assumptions):**
- Catalog authority: `index.json` full-rewrite → transactional store (Postgres) behind a repository interface; JSON becomes a canonical export.
- Registration: search-inline upsert → a separate registration command with its own admission.
- Coverage truths: library status / ks_ledger / LightRAG doc_status → one authoritative catalog state + derivative reports carrying revisions.
- Correction ledger: JSON file → catalog tables (append-only rulings/events).
- CLI write discipline: "stop the service" → multi-writer concurrency under the authority.

**New code lives as a new package inside the repo** (e.g. `papervault/store/` for the repository + schema; adapters keep the old `Library` facade delegating to it). No second repository, no forked service.

## 2. Per-domain activation contract (prerequisite for every cutover)

Before a domain becomes authoritative, its contracts must enforce: version-aware identity; operator-document revisions; action-scoped policy; atomic mutation/audit/pending-work commit; operation idempotency; atomic lease/budget admission; rejection of stale completions; **bytes-before-references** with protection from concurrent GC; **publication of persisted versions and covered inputs (job completion alone does not establish coverage)**; and read contracts that enforce policy, quarantine and revocation, bind evidence versions, and disclose unavailable/stale/incomplete materialization. Imported ambiguity and provenance gaps remain explicit. Preserved components must conform to these contracts. The required ADRs and their phase prerequisites are listed before implementation. Foundations at the relevant cutover are mandatory; the full query redesign or a global snapshot are not.

## 3. Phases, each independently reviewable and reversible

### P0 — Baseline freeze (F0)
Capture legacy measurements BEFORE any change: R1/R2/R3 instruments + workloads + judge configs versioned; snapshot the current `index.json` (counts: records, states, identities, duplicates); record the current service topology and timings. Exit: baselines committed; snapshot archived.

### P1 — New catalog, shadow (F1a)
- Postgres schema: records, identities (unique), aliases/merges, acquisition state, jobs/attempts, audit events, rulings, derivations registry.
- Repository interface + adapter; old `Library` delegates reads to it in shadow mode.
- Import tool: `index.json` → catalog with a reconciliation report (counts per status, identity uniqueness, reference integrity).
- Import uses either a **consistent snapshot with durable, restartable, idempotent change capture**, or a **rehearsed write pause**. Shadow comparisons identify the source revisions compared; differences are logged, never served.
- Each migration stage names **one authority per fact** and fences all service, CLI, and background writers during handoff.
- Exit: reconciliation validates identity mappings, references, hashes and revision compatibility; shadow-read difference rate 0 on the acceptance workload.

### P2 — Cutover the catalog (F1b)
- Flip authority to PG behind a flag; all writers go through the repository; JSON export job runs on every mutation batch (debounced) and remains byte-comparable.
- Handoff protocol: one authority per fact; all service, CLI, and background writers fenced during the handoff; downtime and recovery guarantees established by rehearsal.
- **Frozen-snapshot rollback is permitted only before authoritative writes resume.** Thereafter rollback requires verified preservation of accepted mutations, audit, pending work, and reconciliation of attempted external effects; otherwise recovery proceeds **forward**.
- Exit: live operation with R4 = 0 violations; export parity verified daily; concurrent service/CLI writes exercised.

### P3 — Registration and the search write path (F1c/F3a)
- Search serving performs **no catalog business-state mutation**. It may emit bounded telemetry and submit **separately admitted registration requests selected independently of the final served list**. Serving neither waits for registration execution nor depends on its admission or success.
- Acquisition selection decoupled from serving selection; lightweight registration independent of download/OCR admission.
- Exit: acceptance covers empty and nonempty served lists with registration independently accepted or rejected, plus stalled and failed registration; an empty served list does not prohibit eligible registration; R1/R3 unchanged.

### P4 — One coverage truth (F1d)
- Fold `ks_ledger` states into catalog jobs/state; treat LightRAG `doc_status` as a derivative report with source revisions.
- Reconcile loops simplified to: derivative-vs-authority comparison + repair.
- Exit: the three-truth reconciliation code paths removed; a single query returns coverage; R2 measured directly from authority + derivative reports.

### P5 — Derived layer discipline (F1e)
- Extracts (txt/md) registered as derivations with recipes (engine versions); graph/vector rebuild tooling with versioned snapshots; retention/GC for derivatives. Ordinary GC does not remove retained research assets; explicit authorized purge follows retention policy and preserves identity tombstones and required audit evidence.
- Exit: rebuild drill from authority + source layer reproduces a labeled derivative version; GC-vs-publication race test passes.

### P6 — Search plane redesign (F3)
Per constitution: retrieval/serving decomposition with stage-level loss diagnosis (local → external → fusion → gate → ranking); R1 fixtures (PAMELA/AMS-02 + held-out requests); precision/recall reporting. (Separate design memo.)

### P7 — Query plane redesign (F4)
Read contracts, citation support, evidence availability, canon handling. (Separate design memo.)

### P8 — Lifecycle completion (F5) and cleanup (F6)
Rulings/campaigns generalized (#245/#257 first installment), operator visibility report, structural cleanup slices.

## 4. Acceptance gates (per phase)

- **G1 Integrity**: mechanically defined violations = 0 (duplicate resolved identifiers, dangling references, alias cycles, missing mutation events, inconsistent committed revisions); semantic identity ambiguity is reported separately.
- **G2 Reconciliation**: identity mappings, references, hashes, revision compatibility and **actual materialization against publication records** validated; legitimate missing/retained material classified explicitly. Aggregate count equality is not a general invariant.
- **G3 Behavioral non-regression and serving independence**: suite green; served-result contracts unchanged; **serving independence measured** (registration accepted/rejected independently of the served list); citation support, indexed scope/freshness, read latency and availability under ingestion failure/load, queue/retry/spend bounds measured where applicable.
- **G4 Reversibility**: rollback rehearsed before any live cutover, honoring the handoff protocol (frozen-snapshot rollback only before authoritative writes resume; thereafter forward recovery with preserved mutations/audit/pending work).
- **G5 Recovery**: drills restore from outside the host failure domain against declared recovery objectives, including retained expensive derivatives; phase gates exercise concurrent service/CLI writes, crash/retry boundaries, stale-worker completion, publication/GC races, and policy changes against lagging derivatives.
- **G6 Measurements vs targets**: frozen measurements remain unchanged; versioned acceptance targets and justified exceptions are recorded separately; legacy assertions conflicting with the constitution are explicitly superseded.
- Every slice requires evidence on real material and an independent verdict.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Data loss during migration | Import via a **consistent snapshot with durable, restartable, idempotent change capture**, or a **rehearsed write pause**; **fence all writers during handoff**; any secondary-store writes are **recoverable, nonauthoritative projections**; frozen-snapshot rollback only before authoritative writes resume — thereafter recovery proceeds **forward** with verified preservation of accepted mutations, audit, pending work, and reconciliation of attempted external effects |
| Long downtime | Downtime claims are tied to the **selected, rehearsed import mode**: live shadow phases apply only in change-capture mode; a write-pause import uses its rehearsed window (P1/P2 protocol) |
| Behavioral drift while refactoring | Preserve tool contracts; suite + frozen rulers; shadow reads |
| Scope creep into capabilities | Capabilities are explicitly out of scope; only data/registry/write-path changes |
| Hidden coupling to Library | Adapter-first refactor; grep-based dependency inventory per phase; tests must pass against the adapter |
| LLM nondeterminism in knowledge rebuilds | Derivatives are versioned assets; rebuild = new version, evaluated, not overwritten |

## 6. Out of scope

Microservices; distributed consensus; full event sourcing; multi-tenant auth; rewriting capabilities or the knowledge-plane internals; changing the MCP tool surface.

## 7. Definition of done (whole refactor)

- Constitution invariants enforced by tests and, where possible, database constraints.
- One authoritative home per fact; JSON is an export; no stop-service discipline for maintenance writes.
- R1/R2/R3 at or above the frozen baseline; recovery objectives drilled.
- The derived layer is rebuildable-by-contract; expensive derivatives retained.
