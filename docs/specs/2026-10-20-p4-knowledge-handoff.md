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
test/rehearsal target, with the explicitly guarded disposable Neo4j Community exception
described below. Run:

```sh
.venv/bin/pytest -m integration tests/knowledge/test_catalog_backend_integration.py -q
```

Its nineteen real-store cases exercise original merger summaries and shared-entity traversal,
vector visibility before top-k, cache isolation, sealed/stale graph/vector/cache writes,
and retirement through the original deletion/rebuild reducer, including partial failures.
Fixture embeddings and a fixture summary provider avoid GPU/LLM calls. The fixture prepares
model-specific test tables additively and cleans only its generated namespaces and catalog
schema; shared tables/indexes remain. Missing target variables skip these cases and provide
no backend acceptance evidence.

### Staging backend acceptance commands

Run the harness in a fresh lane-local Python process against an operator-provisioned,
dedicated database on each **staging** backend. For example, name both databases
`p4_backend_rehearsal`. Names must contain a delimited `test` or `rehearsal` token.
PostgreSQL live/default names, including `papervault_catalog`, `knowledge_system` and
`postgres`, remain rejected before connecting. The Neo4j database must already exist.
Neo4j Community's default `neo4j` database is accepted only when the operator explicitly
sets `PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE=1` **and** its target URI has a different
normalized host/port from the service's configured live `NEO4J_URI`. The operator must
supply that live URI through the existing configuration/credential channel before pytest
starts; tests never load `.env`. Missing or invalid URIs are refused. URI schemes and
credentials do not distinguish endpoints; host case, trailing dots, implicit Bolt port
7687 and equivalent loopback addresses are normalized. Numeric IPv4 aliases (including
shortened, decimal, hexadecimal and octal forms), all of `127.0.0.0/8`, IPv6 loopback
and `localhost` share one loopback identity. Empty, whitespace-containing and malformed
hostnames are invalid, including in the supplied live URI. This exception applies only to
the exact database name `neo4j`; it does not relax PostgreSQL or production preparation
guards. Never use the CLI's `--allow-live` for this acceptance run.

The orchestrator supplies the five connection variables and, for the default-database
exception, the two guard inputs through its existing configuration/credential channel;
do not load the service's `.env` or publish their values:

| Variable | Required value |
| --- | --- |
| `PAPERVAULT_BACKEND_TEST_DSN` | Explicit staging PostgreSQL host, port, dbname, user and password, e.g. libpq `host=STAGING_HOST port=5432 dbname=p4_backend_rehearsal user=TEST_USER password=TEST_PASSWORD`. URI DSNs are also supported; TLS options follow the DSN. |
| `PAPERVAULT_BACKEND_TEST_NEO4J_URI` | Explicit staging Bolt URI, including the port and applicable TLS scheme. |
| `PAPERVAULT_BACKEND_TEST_NEO4J_USER` | Dedicated staging test user. |
| `PAPERVAULT_BACKEND_TEST_NEO4J_PASSWORD` | That user's password. |
| `PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE` | Existing dedicated Neo4j database, e.g. `p4_backend_rehearsal`, or `neo4j` on the guarded disposable target. |
| `PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE` | Optional; exactly `1` is required to permit the disposable default `neo4j` database. Other values do not opt in. |
| `NEO4J_URI` | Required for that exception: the service's configured live URI, used only to reject a matching target host/port. Keep the staging URI in `PAPERVAULT_BACKEND_TEST_NEO4J_URI`. |

Pytest captures these five required inputs and the two optional guard inputs before
scrubbing application environment variables. Only the integration-marked
`backend_test_targets` fixture exposes that
snapshot to the harness, which validates both database names before any connection.
The variables stay absent from the runtime environment; dotenv loading remains disabled.

The PostgreSQL role needs schema/table/index creation, fixture DML and cleanup permissions;
pgvector must be installed or the role must be allowed to create its extension. The Neo4j
role needs index creation, graph reads and fixture writes/deletions in its dedicated
database. The fixture rejects an already initialized legacy PostgreSQL pool; use a fresh
pytest process instead of a service interpreter. No runtime `POSTGRES_*` fallback supplies
the backend connection.

Review the exact fixture schema plan without any connection or write:

```sh
.venv/bin/python scripts/prepare_knowledge_backend.py --embedding-dim 3 --embedding-model-name pv_acceptance_fixture
```

Expected stdout is JSON with `dependency: lightrag-hku==1.5.7`, `embedding_dim: 3`,
the `postgres`/`neo4j` additive statements and vector tables ending in
`_pv_acceptance_fixture_3d`. Exit **0** only establishes that the plan was generated.
The harness prepares this plan itself; no separate `--execute` command is required.
The fixture's three-dimensional embedder and summary provider make no GPU or LLM calls.

For Neo4j 5 Community, the sanctioned target is a throwaway `neo4j:5-community`
container with a **fresh volume** on an alternate port. The operator provisions it;
the worker does not start or reconfigure any live service. For example, after supplying
the staging password and the service's configured live `NEO4J_URI` through the existing
channel, and confirming loopback port 17687 differs from the live endpoint and is free:

```sh
export NEO4J_URI="${NEO4J_URI:?Supply the configured live Neo4j URI before starting pytest}"
export PAPERVAULT_BACKEND_TEST_NEO4J_USER=neo4j
export PAPERVAULT_BACKEND_TEST_NEO4J_DATABASE=neo4j
export PAPERVAULT_BACKEND_TEST_NEO4J_URI=bolt://127.0.0.1:17687
export PAPERVAULT_BACKEND_TEST_NEO4J_DISPOSABLE=1
export NEO4J_AUTH="neo4j/${PAPERVAULT_BACKEND_TEST_NEO4J_PASSWORD:?Supply a fresh staging password}"
P4_NEO4J_VOLUME="$(docker volume create)"
P4_NEO4J_CONTAINER="$(docker run -d --publish 127.0.0.1:17687:7687 \
  --mount "type=volume,source=$P4_NEO4J_VOLUME,target=/data" \
  --env NEO4J_AUTH neo4j:5-community)"
unset NEO4J_AUTH
```

Wait for that container's Bolt service to become ready before running the harness.
Use only this newly created volume, never the live `ks-neo4jdata` volume or an existing
data directory. PostgreSQL still needs an explicitly named test/rehearsal database.
Retain the container/volume identifiers with the run evidence for inspection and
operator cleanup after acceptance; this procedure performs no automatic deletion.

Check collection and the target refusal cases, then execute the actual backend cases:

```sh
.venv/bin/pytest -m integration tests/knowledge/test_catalog_backend_integration.py --collect-only -q
.venv/bin/pytest -m integration tests/knowledge/test_catalog_backend_integration.py -k refuses -q
.venv/bin/pytest -m integration tests/knowledge/test_catalog_backend_integration.py -k 'refuses or accepts' -q
.venv/bin/pytest -m integration tests/knowledge/test_catalog_backend_integration.py -q -rs
```

Expected results: **131 tests collected**, **72 passed, 59 deselected** for the refusal
check, **96 passed, 35 deselected** for all target checks (no connections), and
**131 passed**, exit **0**, for acceptance. Thirty-five skipped
real-store tests mean missing explicit credentials and do not constitute acceptance.
The real-store test names identify original merge summaries/shared-entity traversal,
latest-version vector visibility/top-k/tombstones, cache/sealed/stale-write isolation,
and retirement with clean rebuild, backend write failure, missing/partial-cache and
unattributable-description variants, plus document-index verification during build and
metadata retirement (missing/partial entity or relationship indexes and empty status chunks),
interruption between graph and vector writes, same-lease refusal, fresh-namespace lease recovery,
and entity/relationship vector content, identity, source and reverse-orientation mismatches.
The original relationship reducer's full entity-vector content after source appends
must also verify alongside the original truncated form.
Any setup, assertion or cleanup error is a failed acceptance run.

There is no operator namespace parameter: each fixture creates a random
`catalog_backend_acceptance_<uuid>` catalog schema and catalog-admitted `pv_<48 hex digits>`
attempt namespaces. PostgreSQL stores ownership in `workspace`; Neo4j stores it in
`_pv_namespace`. Readers pin selected layers before graph traversal/vector ranking/cache
access and reject writes through the read-only adapter. The **harness itself is mutating**:
it writes fixture outputs, seals/publishes test receipts, reclaims an expired test attempt,
and creates additive schema/index objects. Cleanup deletes rows/nodes only for its recorded
attempt namespaces and drops its catalog schema. Shared backend tables, extension and
indexes remain; interruption can leave test-owned material for operator inspection.
These commands neither inspect nor modify live stores. Read-only guarantees apply to
the schema-plan command, collection/refusal checks, serving adapters and the source
`dry-run` below, not to the backend acceptance fixture.

For the separate dedicated-Postgres acceptance suite, supply
`PAPERVAULT_CATALOG_TEST_DSN` explicitly and run `.venv/bin/pytest tests/store -q`.
That fixture creates/drops unique `catalog_test_<uuid>` schemas; it does not use
`PAPERVAULT_BACKEND_TEST_DSN`. Record the head SHA, commands, exit codes, case counts and
non-secret output on the PR. Corpus-scale query plans/latency and real-material R2
dispositions require their own evidence; these fixture results do not establish them.

For a read-only source inspection, invoke the importer with an explicit source DSN and source identity, or supply a previously authorized snapshot:

```sh
.venv/bin/python scripts/import_knowledge.py --action dry-run --source-dsn "$P4_SOURCE_READONLY_DSN" --source-id legacy-ledger --source-schema public --output "$P4_DRY_RUN_REPORT"
```

This action opens a read-only transaction and installs no capture objects. Add explicit `--mappings`, `--docstatus` and non-secret `--configuration` evidence when available. A plain read-only snapshot is not an aligned migration checkpoint and cannot authorize handoff. Retain its keyed dispositions and raw provenance limitations in the operator's established evidence store; publish only non-secret summaries and approved links on the issue/PR.

## Rehearsal sequence

1. Initialize a dedicated catalog using `--action initialize-catalog --catalog-dsn ... --operation-id ...`. This installs the shared schema and importer explicitly; runtime startup performs no schema migration. Initialize exactly one active generation with `KnowledgeRepository.command('create_generation', stable_operation_id, {'generation_id': generation_id, 'candidate': False})`. No legacy output is selected automatically.
2. On a dedicated source clone, `--action install-capture --source-dsn ... --source-id ...` installs durable inserts/updates/deletes capture and a transactionally ordered counter. `--action capture-snapshot` takes the aligned snapshot and C0 under a source-table lock. Retain source configuration and schema evidence. A rollback consumes no sequence number. TRUNCATE and capture-history edits are rejected.
3. Use `--action apply --snapshot ... --catalog-dsn ... --operation-id ...` to import stable source-row/version identities. Reapply, interrupt and resume. When mappings or catalog units become resolved, reapply with a new operation ID even if source versions are unchanged. The importer validates the entire plan before writing and appends accepted projection lineage separately from immutable source events. Equal-version updates must retain verified mappings and accepted disposition/observation evidence; conflicting verified mappings fail explicitly and preserve quarantine. Rehearse an interrupted unmapped plan, a committed enriched plan, and refusal of the older plan's resumption without changing jobs, heads, checkpoint or report. Identical replay remains idempotent, and genuinely newer source versions may change projections. Handoff compares projections to the final report and current unit mappings and refuses unprojected resolutions. Verify that newer catalog-owned projections are not overwritten. Compare keyed matches, ledger-only and document-status-only units, duplicates, conflicts and provenance gaps; every item needs a disposition. Unknown legacy identities remain quarantined, non-executable and excluded from conforming coverage. Imported failure streaks and parking transfer once without replaying success/failure events or inventing historical attempts.
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

Exercise peak-window and breaker pauses during an admitted build. A valid deferral settles and fences the attempt, acknowledges delivery and queues the job without a failure-streak event or parking change; the round stops further admission. Uncertain charges and recovery pins remain. Actual lease loss retains the abandoned-attempt behavior. For retirement, inject rebuild failures despite deletion success and verify that no receipt is sealed or publication selected, with a discrepancy and repair recorded. A failed retirement settles and fences its attempt, acknowledges delivery, preserves uncertain charges/recovery pins and requires an isolated rebuild; delivery retries do not spend another failure event. `not_found` is unverified retirement and cannot be accepted as proof that an inherited contribution is absent.

Include partial extraction-cache loss, caches without descriptions for an affected entity or relationship, and empty normalized summaries from both callbacks and cache hits. All must refuse retirement before sealing, even when LightRAG reports successful deletion. Exercise typed peak/breaker pauses inside the original retirement summary callback: they must reach the catalog deferral command without a failure-streak event, discrepancy or repair. Also advance a migration checkpoint through another row, then attempt enrichment from the older checkpoint; it must leave all accepted projections and the newer report unchanged. Reissue enrichment against the current checkpoint before handoff.

Inject absent and incomplete document-to-entity and document-to-relationship indexes after graph material exists. Build verification must refuse sealing; retirement must check immutable contribution evidence before the original deletion discovers targets, record a discrepancy/repair, and preserve production selection. Include source-limit cases with new chunk memberships but no private graph write, and an empty/partial document-status chunk list. A genuinely empty contribution remains valid. Historical receipts without recorded contribution evidence need an isolated rebuild before retirement can be verified.

Replay physical receipts sealed before the document-index verification protocol, both before and after accepted execution completion. The shared catalog validator must reject further completion/publication, retain the historical receipt/job state, and record a discrepancy without moving selections. This includes metadata downgrades: recovery must not bypass the versioned retirement attestation. Do not backfill an attestation into an immutable historical receipt.

Interrupt retirement after a shared graph/tracker has been rebuilt but before its entity or relationship vector is replaced. Exercise both stale and absent vectors, then redeliver within the same lease: the durable retirement-start record must refuse reuse and request an isolated rebuild. Expire/reclaim the attempt and prove the fresh namespace rebuilds from the selected parent and can publish matching graph/vector content. Inject content, endpoint, source-chunk and stale reverse-orientation vector mismatches; none may seal. Confirm failed private graph/vector rows never enter selected traversal or ranking. Replay an earlier retirement receipt without graph/vector attestation and retain its history while refusing new completion/publication. Lose the seal response after commit and prove the persisted receipt is recovered without treating it as failed execution.

Capture timestamp-bearing INSERT, UPDATE and DELETE operations through C1 with trailing-zero and nonzero fractional seconds and different session timezones. Snapshot, exported old/new images, final import row count/digest and freeze must agree semantically. Changed timestamp instants, source identities, fingerprints, counters or extra fields must still fail strict old-image alignment. Never rewrite the immutable capture journal to obtain agreement.

Resume a C0 import accepted before timestamp normalization, including same-checkpoint replay/enrichment and C1 INSERT/UPDATE/DELETE. Retain an unchanged C0 row through the new final freeze and handoff. Verify that original source events, hashes and reports remain byte-for-byte unchanged; canonical report/checkpoint digests must agree with the final freeze. Invalid literal hashes, changed instants or counters, independent projection edits and enrichment from an older checkpoint must still fail.

Complete a receipt, advance its generation manifest with another publication, and run default single-slot scheduler rounds. Receipt recovery must leave capacity for a newly fenced rebase attempt, its publication against the current aggregate, and later queued jobs. For capture replay, import duplicate A/B at C0, delete B at C1, and prove C1 applies and permits validated handoff while retaining A's historical duplicate disposition and immutable C0 events.

Rollback after handoff keeps the catalog database, authority epoch, audit, outbox, accepted receipts and external-attempt accounting. Deploy only a catalog-compatible binary. Do not restore a frozen pre-cutover database or select the legacy flag. Returning authority to the ledger needs a separately designed and rehearsed reverse migration.

## Coverage evidence

Use `scripts/coverage_report.py --help` to supply explicit frozen pools, F0 snapshot and either catalog rows or a read-only catalog connection. The report retains the frozen 12 pools/111 entries and DOI/title matcher. It displays the unchanged legacy measurement separately from current version-qualified metadata, readable full-text evidence and indexed coverage. The headline is full text only; abstract/metadata tiers remain separately labeled. The report records selectors, evaluation time and the 300-second policy. Unknown legacy provenance and stale observations stay in the denominator without conforming credit.

Inspect every keyed difference, including absent required section IDs, requested/selected versions, inventory/parent evidence, recipe, tier, publication, materialization and freshness. Attach receipts or discrepancies. F0 supplies aggregate historical results rather than a consistent recovery snapshot or complete historical per-entry receipts; the tool reports that limitation instead of inventing them. Aggregate agreement is not acceptance. Any accepted exception or alternate policy must be separately versioned and cannot rewrite F0.
