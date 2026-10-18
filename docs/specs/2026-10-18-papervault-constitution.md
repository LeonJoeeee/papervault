# papervault constitution v2.1-rc2 (systematic rewrite)

Status: **candidate for ratification** — systematic rewrite incorporating the settled macro architecture and the reviews of v1.x and v2.0; v2.1 applies the five review amendments (layer/authority/lineage precision; serving effects; operator-document identity; publication-vs-coverage; read-operator contract). On acceptance this becomes `docs/specs/<date>-papervault-constitution.md` + an ADR. Governance: every rule states an integrity, retrieval, or operating rationale, supported by a real incident or established failure analysis; mechanisms and numerical thresholds live in ADRs and `docs/baselines.md`. The document stays readable in one sitting.

## Article 0 — Essence and needs

Papervault is **a single-tenant research data system serving AI agents**: a durable research library plus a derived knowledge base. It is the researcher's externalized long-term memory — the corpus is the memory, the catalog is the index of memory, the derived knowledge is the digested understanding; search is recall, query is reflection.

Four actual needs define every requirement:
1. **Find what should be read** — the relevant frontier AND the named classics; when a need names a work or an experiment, it must be retrievable and served.
2. **Read it** — full text must be reliably obtainable, openable, and citable.
3. **Ask** — synthesis over the whole corpus (papers, textbooks, the researcher's own notes) with citations that resolve to retained evidence.
4. **Accrete unattended** — the corpus grows by itself with bounded cost, latency, and operational burden.

Two primary contracts follow, neither secondary: **durable collection integrity** (records, identity, content, provenance, append-only audit) and **useful, evidence-backed retrieval**. The motivating failure — originals stored for months, never served — happened after acquisition, in retrieval.

## Article 1 — Three data domains, realized as three layers

Source, catalog, and derived are **logical persistence roles**:

1. **Source** — immutable raw artifacts: paper PDFs and operator-supplied documents (textbooks, notebooks, instrument references). Content-addressed, never mutated, retained as research assets (operator documents are often irreplaceable sole copies; notebook-class documents are private-instance-only).
2. **Catalog** — the transactional authority over **identity, policy, registration, lineage, and processing/publication state** (records, states, jobs, rulings, audit events). See Article 2. Records may exist **before** artifact acquisition (metadata-only registration).
3. **Derived** — everything computed from the layers above: extracted text (txt/md), search indexes, embeddings, the knowledge graph. Persisted derivatives identify their **catalog revision and artifact inputs**, plus their versioned recipe (model, prompt, chunking, configuration) — directly or through a lineage manifest.

One-way derivation constrains **dependencies**, not all flows: derived data never becomes authoritative catalog fact merely by being computed, but **validated registration and completion commands may update the catalog**. Each role has its own retention class: source artifacts are immutable research assets; catalog rows are transactionally mutated with audit; derived objects are versioned and rebuildable as NEW versions (never silently overwritten), with expensive derivatives retained when regeneration would violate recovery objectives.

**One authoritative home per fact**: for each authoritative **catalog fact** (identity, policy, registration, lineage, state) there is exactly one home; derived copies and computed claims state the revision they represent and are never consulted as authority. There are no competing truths (graph coverage is a derived report, not a second state).

## Article 2 — Authority and commit semantics

The catalog is held by **a transactional authority store** — one that provides atomic commit of a mutation with its audit event and pending downstream work, uniqueness and referential-integrity constraints, multi-writer concurrency, and crash recovery. The engine is selected in the F1 ADR (current selection: Postgres, already operated for the knowledge layer); the choice is a mechanism, not a constitutional principle.

- **Commit**: each authoritative mutation commits atomically with its audit event and any pending downstream work. Retried commands are idempotent in their **committed effects** (stable operation IDs, conditional completion rejecting stale workers). External effects already attempted (OCR/LLM calls) may execute twice after a timeout; unknown-outcome charges are accounted conservatively — no universal exactly-once execution is promised.
- **Bytes before references**: artifact bytes are published durably before their references commit; garbage collection never races an in-flight publication.
- **Projections and derivatives** are restartable, reconciled, and state the source revision they represent. The catalog records **publication** of persisted derived versions and their covered input revisions; **job completion alone does not establish indexed coverage**. Read contracts identify the compatible version actually served and disclose missing or stale materialization; reconciliation detects divergence from catalog publication records. Compatibility need not require a global snapshot across unrelated records.
- **Read contracts**: every read operator carries a consistency/freshness contract — acceptable staleness, how degraded or partial results are distinguished from complete ones, and that applicable authority (policy, quarantine, revocation) and evidence availability are enforced despite lag. A derivative that falsely claims completeness, or mixes incompatible source revisions, is an integrity violation; disclosed lag is not.
- **Export**: the JSON index is retained as a canonical, human-readable export of the catalog — never the authority.

## Article 3 — Three blocks (logical responsibilities)

### 3.1 Capabilities
I/O adapters and transformations (source adapters, LLM gates, OCR, download tiers, extractors) that own **no authoritative business state**. Operational state (caches, pools, model residency) is allowed. One capability, one interface; not a service per integration (MinerU is a service for resource isolation, not a pattern). Input lineage rides alongside outputs (extraction yields new bytes and a new ref; search starts without a ref).

### 3.2 Pipeline
Composition, scheduling, retries, deadlines, delivery. Judgment about **execution** lives here (retry, requeue, degrade, circuit-break, escalate); **domain integrity judgments** (identity resolution, policy validity, legal state transitions) belong to the domain block. Pipelines never rewrite content. Each run has one identity, child attempts, a recoverable unfinished state, and a shared run record; execution history is append-only.

### 3.3 Domain & persistence
Owns identity, policy, validation, registration, and transactional changes. Registration is the only entrance for new records; dedup happens at registration. Records have **stable internal IDs**. **Bibliographic Records** represent **work@version** units; preprint/published/edition relationships are explicit links between Records, never silent merges. **Non-bibliographic operator documents** (notebooks, evolving notes) use stable internal document identities with explicitly linked immutable revisions; external identifiers are optional. Citations identify the **evidence revision and location** used; later edits preserve prior evidence subject to retention policy. Citation keys and external identifiers resolve through version-aware identity rules (normalization in the identity ADR); content hashes identify bytes; sections retain parent identity, location, and the evidence version they cite. Alias resolution terminates without cycles; ambiguous matches stay explicit for adjudication. **Merging identities confers no authorization beyond the rulings applicable to each merged record.** Retained artifacts are immutable; corrections and merges are new rows preserving prior identities and observations. Retention distinguishes research assets, audit history, derivatives, telemetry, and scratch; GC and explicit purge preserve tombstones and audit evidence. **A tombstone preserves identity, not evidence**: historical citations may resolve to an explicit "evidence purged/unavailable" record; new synthesis must not treat it as supporting content; dangling references are defined accordingly.

## Article 4 — Data model and invariants

Entities: Record (internal ID + resolvable identifiers + metadata + artifact refs + state + provenance); Artifact (bytes, hash); Derivation (inputs, outputs, versioned recipe); Alias/Merge; Ruling; Job (run identity, attempts, recoverable state); Ledger (append-only audit events); Telemetry.

States: retrieval ∈ {eligible, verified, exhausted}; policy ∈ {clear, excluded, override, unknown}; merge ∈ {canonical, alias}; quarantine flag. The admission predicate states **permission to start** an acquisition (not an obligation); cancellation, scheduling, and capacity remain; conditions are checked at one atomic enforcement point that reserves lease and budget together.

Invariants:
1. **One authoritative home per fact**; derived copies declare their source revision and are never authority.
2. Resolved-identifier mappings are unique under the version-aware identity rules; aliases terminate; ambiguous matches are held for adjudication. False merges are integrity violations on par with duplicates.
3. Artifact bytes are immutable and content-addressed; every derivation is a new row carrying inputs, outputs, and its versioned recipe.
4. Every **authoritative business-state** transition commits atomically with its audit event; no operator path bypasses this (heartbeats, caches, telemetry counters are out of scope).
5. `policy=unknown` denies the specified action; a ruling names its action, scope, precedence, expiry, and revocation, and must be applicable to unlock — an inapplicable ruling unlocks nothing.
6. Every served result carries a stable identity and provenance appropriate to its origin (external attempt, operator document, local record), with a serving trace; legacy gaps are marked, never back-filled with invented attempts.
7. History is queryable and exportable; deletion is governed by retention rules, not forbidden.

## Article 5 — Operators

- **5.1 Search (read + separate registration)**: **serving does not mutate catalog business state as part of answering.** Read operators may emit bounded operational telemetry and serving traces. Search may submit a **registration request executed as a separately admitted command**; serving neither waits for nor depends on registration, acquisition, or their success. External candidates may carry stable source identities and provenance **before** receiving an internal Record ID. Acquisition selection is independent of final serving selection. Serving records enough stage decisions to locate losses (local retrieval → external retrieval → fusion → gate → ranking).
- **5.2 Ingest / registration**: identity resolution, dedup, state initialization.
- **5.3 Download / Extract**: capabilities under pipeline orchestration; outputs are artifacts and derivations.
- **5.4 Rulings & corrections**: scoped, append-only operator acts.
- **5.5 Query**: read path over the derived knowledge layer; synthesis cites records whose evidence supports the claim; nothing is written back implicitly. **Query declares its indexed scope and freshness; absence from indexed evidence does not establish absence from the corpus.** `get_paper` resolves identities and aliases to metadata and version-bound readable evidence, or an explicit unavailable/purged status.
- **5.6 Distillation**: derivations registered with source refs and recipes.
- Background processing failures never disable otherwise-available reads.

## Article 6 — Cross-block principles

1. Service/caller full decoupling (MinerU, #235–#238).
2. Gates split by failure boundary; **unify the contract, not the file**.
3. Observation semantics: receipts record what actually served (provider, model, version — explicitly unknown when undisclosed); harmless configuration discrepancies are observed, not refused; observation never overrides validation or integrity requirements.
4. Unified error taxonomy: `code`, `scope` (item | service), `retryability` (permanent | transient-with-delay | transient-no-delay), with defined semantics for an absent delay; input errors and provider malformations never share a fallback path.
5. Bounded operation: queues, concurrency, retries, deadlines, disk, spend; admission atomically reserves authority and budget; completion verifies authority is still current.
6. Credentials: supplied through protected configuration, disclosed only to their intended authentication endpoints; repositories, receipts, and logs contain none.
7. Receipts are assets: source, hashes, refs, recipe versions, attempt records.
8. Backups cover authoritative data and retained artifacts, include a copy outside the host's failure domain, and are tested by restoration; recovery objectives state acceptable loss and restoration time including derivative rebuilds.

## Article 7 — Acceptance (measured, not asserted)

Versioned workloads, labels, result budgets, evaluator configurations; thresholds and known exceptions live in `docs/baselines.md`.
- **R1 Named-need recall**: annotated requests with required relevant works/evidence categories and a fixed result budget; recall measured on final results at that budget, with loss diagnosis by serving stage (the PAMELA/AMS-02 incident is a mandatory fixture; passing it alone proves nothing general). Named-need retrieval obeys applicable policy and declared result budgets; unmet needs remain explicit and are scored under versioned baseline rules.
- **R2 Canon coverage over time**: over a frozen landmark pool, measured as CURRENT metadata coverage, readable-content coverage, and indexed coverage — separately; historical ingest counts do not count.
- **R3 Precision with recall**: precision and recall (same workload and budget as R1), return counts, and abstention reported together; the judge is calibrated against human labels and versioned.
- **R4 Mechanical integrity**: zero duplicate resolved identifiers, dangling references, alias cycles, missing mutation events, inconsistent committed revisions; semantic ambiguity (incl. false-merge risk) is reported, not auto-certified.
- Also acceptance: query citation support, read latency under ingestion load, retry/spend boundedness, and recovery drills.

## Article 8 — Governance and migration

- Rules require a stated integrity, retrieval, or operating rationale, supported by incidents or established failure analysis; mechanisms and numbers live in ADRs and baselines.
- The constitution and the slice plan precede code; authority/registration contracts and baseline instruments precede the behavior-changing slices that depend on them.
- Old code is read-only reference for rewritten parts; **staged migration through the new contracts is permitted** where safer than wholesale replacement, each stage reviewable.
- No slice ships without proof on real material and an independent verdict.

## Slice plan

- **F0** Baseline freeze: capture legacy measurements BEFORE any F1 migration can change behavior (instruments, workloads, judge configuration versioned).
- **F1** Foundations: transactional authority (engine ADR), registration contract, schema versioning/migration, backup/recovery. Prerequisite for behavior-changing slices.
- **F3** Search plane. **F4** Query plane. **F5** Lifecycle operators completion. **F6** Structural cleanup (any time, zero behavior change).

Required ADRs: identity resolution & merge; migration/cutover; file integrity & recovery; operator visibility; trust boundaries (retrieved documents and model outputs never authorize commands, policy changes, or merges).

## Appendix A — Macro-architecture mapping (settled 2026-10-18)

| Today | Target layer | Change |
|---|---|---|
| `index.json` (112 MB, full rewrite) | Catalog (export) | Postgres authority; JSON becomes export |
| Paper records + v2 states in JSON | Catalog | Transactional tables, constraints, audit |
| `pdfs/` (72 GB), operator documents | Source | Immutable, content-addressed; textbooks/notebooks registered as a source class |
| `extracts/{txt,md}` (1.8 GB) | Derived | Receipted with recipes; rebuildable as new versions |
| Postgres LightRAG tables + ks_ledger | Catalog + Derived | KS ledger folds into catalog jobs/state; LightRAG tables are derivatives |
| Neo4j graph | Derived | Versioned derivative; never authority |
| `correction-ledger.json` | Catalog | Transactional rulings/corrections |
| Three graph-coverage truths (library status / ks_ledger / doc_status) | Catalog | One authoritative state; derived reports carry revisions |
| Search upserts during serving | Operator 5.1 | Serving read; registration separate |
| CLI writers vs service (stop-service discipline) | Catalog | Multi-writer concurrency under the authority store |
