# P4 knowledge authority rehearsal

This procedure implements [issue #278](https://github.com/LeonJoeeee/papervault/issues/278) and [the publication decision](../adr/DRAFT-catalog-knowledge-publication.md). It is an operator procedure, not a record of a completed cutover. The feature defaults to legacy; installation, a restart and a flag change do not fence a writer. Heavy rebuilds and drills belong in the owner's monthly server-reboot window, at most two hours.

## Connections and evidence

Use explicit connections and operator-selected durable evidence destinations. `scripts/import_knowledge.py` does not discover credentials or load `.env`. Its mutating rehearsal actions reject the live `papervault_catalog` and `knowledge_system` database names. The orchestrator must authorize any live administrative operation separately. Do not point the dedicated test fixture at the live catalog.

Run the unit suite and lint in the lane:

```sh
.venv/bin/pytest -m 'not integration' -q
.venv/bin/ruff check src tests
```

With `PAPERVAULT_CATALOG_TEST_DSN` set by the orchestrator to a dedicated disposable database, the same suite executes catalog interleavings, capture, import and privilege tests instead of skipping them. The fixture creates unique schemas in that database. Confirm duplicate claims, lease expiry, stale completion, input changes, wrong recipes, succeeded-but-unpublished jobs, publication loss, interrupted imports, final-delta continuity, column/table privileges and forward-only rollback all execute. Verify actual Neo4j and vector backend behavior separately: unpublished/stale namespaces, graph paths and aggregates, deletion tombstones, caches, connection initialization and corpus-scale latency. Storage doubles cannot establish backend enforcement.

Backend schema preparation is explicit and separate from runtime. Review its additive plan without connecting:

```sh
.venv/bin/python scripts/prepare_knowledge_backend.py --embedding-dim 1024
```

The pinned runtime uses unsuffixed vector tables, so omit `--embedding-model-name`. `--execute` requires explicit `--dsn` and/or Neo4j connection arguments; inspect `--help` for the complete interface. The default execution boundary requires test/rehearsal database names. `--allow-live` is only an explicit tool option, never authorization to administer production. Prepared tables must retain the exact `(workspace,id)` ownership key and required columns. The tool performs additive schema/index creation, with no legacy row migration or deletion. Runtime readers and writers refuse an unprepared schema and perform no implicit preparation.

The real-store harness uses `PAPERVAULT_BACKEND_TEST_DSN` and
`PAPERVAULT_BACKEND_TEST_NEO4J_URI`, `PAPERVAULT_BACKEND_TEST_NEO4J_USER`,
`PAPERVAULT_BACKEND_TEST_NEO4J_PASSWORD`, `PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE`.
The Neo4j test database must already exist. Both database names must identify a dedicated
test/rehearsal target. Run:

```sh
.venv/bin/pytest -m integration tests/knowledge/test_catalog_backend_integration.py -q
```

Its three real-store cases exercise original merger summaries and shared-entity traversal,
vector visibility before top-k, cache isolation, and sealed/stale graph/vector/cache writes.
Fixture embeddings and a fixture summary provider avoid GPU/LLM calls. The fixture prepares
model-specific test tables additively and cleans only its generated namespaces and catalog
schema; shared tables/indexes remain. Missing target variables skip these cases and provide
no backend acceptance evidence.

For a read-only source inspection, invoke the importer with an explicit source DSN and source identity, or supply a previously authorized snapshot:

```sh
.venv/bin/python scripts/import_knowledge.py --action dry-run --source-dsn "$P4_SOURCE_READONLY_DSN" --source-id legacy-ledger --source-schema public --output "$P4_DRY_RUN_REPORT"
```

This action opens a read-only transaction and installs no capture objects. Add explicit `--mappings`, `--docstatus` and non-secret `--configuration` evidence when available. A plain read-only snapshot is not an aligned migration checkpoint and cannot authorize handoff. Retain its keyed dispositions and raw provenance limitations in the operator's established evidence store; publish only non-secret summaries and approved links on the issue/PR.

## Rehearsal sequence

1. Initialize a dedicated catalog using `--action initialize-catalog --catalog-dsn ... --operation-id ...`. This installs the shared schema and importer explicitly; runtime startup performs no schema migration. Initialize exactly one active generation with `KnowledgeRepository.command('create_generation', stable_operation_id, {'generation_id': generation_id, 'candidate': False})`. No legacy output is selected automatically.
2. On a dedicated source clone, `--action install-capture --source-dsn ... --source-id ...` installs durable inserts/updates/deletes capture and a transactionally ordered counter. `--action capture-snapshot` takes the aligned snapshot and C0 under a source-table lock. Retain source configuration and schema evidence. A rollback consumes no sequence number. TRUNCATE and capture-history edits are rejected.
3. Use `--action apply --snapshot ... --catalog-dsn ... --operation-id ...` to import stable source-row/version identities. Reapply, interrupt and resume. Verify that newer catalog-owned projections are not overwritten. Compare keyed matches, ledger-only and document-status-only units, duplicates, conflicts and provenance gaps; every item needs a disposition. Unknown legacy identities remain quarantined, non-executable and excluded from conforming coverage. Imported failure streaks and parking transfer once without replaying success/failure events or inventing historical attempts.
4. Shadow-compare compatible checkpoints with the catalog-only coverage and import reports. Comparison must not dispatch, repair, publish or change serving. The importer's audited migration writes are distinct from comparison. Preserve original attempt/counter checkpoints and independently inventory external work with uncertain outcomes.
5. Stop dispatch and pipeline admission. Drain in-flight work, or fence both catalog completions and backend writes. Stop/isolate LightRAG internal workers that cannot honor attempt ownership. Record actual evidence for all five drain fields: `dispatch_stopped`, `pipeline_admission_stopped`, `inflight_drained`, `backend_writes_fenced`, `internal_workers_stopped`, plus a durable receipt. Setting fields true is an attestation, not a substitute for performing the operations.
6. `--action freeze-source --legacy-role ... --evidence ...` takes the final writer lock, installs the durable DML fence, revokes table and column DML from named legacy roles and PUBLIC, and verifies effective privileges, including inherited grants. An inherited privilege or owner/superuser capability that remains writable prevents a successful fence. Capture final C1; `--action capture-delta --after C0` returns changes through the final checkpoint. Apply every change and prove the contiguous sequence, exact final row set and digest.
7. Perform key-level validation of dispositions, references, quarantine, accounting and externally attempted work. Record each of `dispositions`, `references`, `quarantine`, `accounting`, `external_attempts` and its validation receipt. The handoff evidence combines source/capture IDs, final sequence, source fence, drain evidence and validation. The importer validates its recorded final projections and rejects a mismatch.
8. Commit `--action handoff --catalog-dsn ... --expected-epoch ... --operation-id ... --evidence ...` only on the authorized rehearsal target. The transaction commits the validated checkpoint, audit and new catalog authority epoch. Test old-epoch completions and legacy DML rejection. Confirm `PAPERVAULT_KNOWLEDGE_JOBS=legacy` and JSON library authority now fail startup rather than silently taking control.
9. After an authorized handoff, use `PAPERVAULT_KNOWLEDGE_JOBS=postgres`, `PAPERVAULT_CATALOG_AUTHORITY=postgres`, and explicit catalog DSN/schema. The scheduler captures exact current inputs, records desired inventories and logical jobs, delivers submissions and publishes verified receipts. No legacy processed status can satisfy publication. Verify the scheduler and operator routes never perform ledger DML, including runtime privilege checks and dependency inspection.

## Rebuilds, serving and recovery

Routine ingestion advances active manifests. A forced operator rebuild creates a fresh candidate containing the complete desired inventory. Its durable inventory intent allows recovery after capture/requirement/fork interruptions. Candidate claims require a committed `begin_maintenance_window` command with a unique `window_id`, `kind='monthly_reboot'`, timezone `Asia/Shanghai`, `starts_at`, `ends_at` no more than 7200 seconds apart, an operator nonce and the expected maintenance revision. Expiry stops new candidate admission. A missed window retains pending work.

Candidate publication does not promote it. `promote_generation` requires exact candidate manifest/current-requirements revisions and positive compatible coverage. New registrations or input changes must be incorporated in a new explicit manifest or make promotion fail. Query serving pins the exact selected manifest and required namespaces; an ineligible contribution makes its aggregate unavailable. Durable reader/recovery pins and committed references protect outputs; this slice provides no garbage collection command.

Inject crashes before/after external acceptance, after durable outputs but before catalog registration, after catalog commit but before its response, and before outbox acknowledgement. Recover by stable submission/operation IDs and persisted receipts. An ambiguous outcome retains conservative cost and pending work; delivery retries do not become attempts. A stale fence cannot register its outputs. Lost published material retains publication history and queues a deduplicated isolated repair, admitted only during the monthly maintenance window.

Rollback after handoff keeps the catalog database, authority epoch, audit, outbox, accepted receipts and external-attempt accounting. Deploy only a catalog-compatible binary. Do not restore a frozen pre-cutover database or select the legacy flag. Returning authority to the ledger needs a separately designed and rehearsed reverse migration.

## Coverage evidence

Use `scripts/coverage_report.py --help` to supply explicit frozen pools, F0 snapshot and either catalog rows or a read-only catalog connection. The report retains the frozen 12 pools/111 entries and DOI/title matcher. It displays the unchanged legacy measurement separately from current version-qualified metadata, readable full-text evidence and indexed coverage. The headline is full text only; abstract/metadata tiers remain separately labeled. The report records selectors, evaluation time and the 300-second policy. Unknown legacy provenance and stale observations stay in the denominator without conforming credit.

Inspect every keyed difference, including absent required section IDs, requested/selected versions, inventory/parent evidence, recipe, tier, publication, materialization and freshness. Attach receipts or discrepancies. F0 supplies aggregate historical results rather than a consistent recovery snapshot or complete historical per-entry receipts; the tool reports that limitation instead of inventing them. Aggregate agreement is not acceptance. Any accepted exception or alternate policy must be separately versioned and cannot rewrite F0.
