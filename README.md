# papervault

A self-hosted **literature + knowledge layer for coding agents**, delivered as one MCP
server. Point any MCP-capable agent (Claude Code, and others) at it and the agent gains
three tools:

- **`search_papers`** — discover literature by describing your research intent in prose;
  relevant finds are auto-ingested (downloaded, OCR'd, full text on disk).
- **`get_paper`** — look up papers by citation key / DOI / arXiv id / fuzzy title; read
  verbatim full text from a returned path.
- **`query`** — ask the knowledge base a question, get a cited prose answer synthesized
  from a knowledge graph distilled from everything the library has ingested.

The corpus compounds without babysitting: search → queryable knowledge in tens of minutes,
self-healing ingest queues, cited answers grounded in real sources.

> **Status:** pre-release (closed beta). See `docs/PRD.md` for scope and `docs/architecture.md`
> for structure. This is industrial-grade infrastructure, not a toy — read the hardware floor
> before installing.

## Hardware floor (not optional)

papervault runs its own embedding, reranking, and OCR models locally. There is **no GPU-less
or degraded mode** — below the floor, this product is not for you.

- **1× NVIDIA GPU, 24 GB** (RTX 3090 / 4090 class). A single card runs the full stack
  (embed + rerank + OCR co-tenant) inside a measured VRAM budget; a second GPU is comfort,
  never a requirement.
- **Linux + CUDA.**
- **An OpenAI-compatible LLM endpoint + key**, supplied by you (graph build, search
  triage, and answer synthesis all route to it). papervault bundles no keys.
- **Docker** (for the Neo4j + Postgres backing stores).

## Quick start

See [`docs/INSTALL.md`](docs/INSTALL.md) for the full walkthrough. In brief:

```bash
cp .env.example .env          # then fill in your LLM endpoint + key + data dir
docker volume create ks-pgdata      # external volumes hold the production data (create once)
docker volume create ks-neo4jdata
docker compose --env-file .env -f deploy/docker-compose.yml up -d   # Neo4j + Postgres
uv venv && uv pip install -e ".[grey]"              # core HTTP OCR client + grey download tiers
papervault doctor             # verify GPU, databases, LLM config
papervault-mcp                # start the MCP server (== papervault serve)
```

Provision the standalone `mineru-service` separately at `http://127.0.0.1:30000`
(override with `MINERU_URL`). papervault submits PDFs to its official `mineru-api`
task API over HTTP, waits in the shared FIFO queue, and reads the completed markdown.
The service owns model preload and GPU unloading after 20 idle minutes. papervault's
`mineru` extra is retained as an empty compatibility extra: `uv sync --extra mineru`
remains valid without installing `mineru` or `torchvision` in the caller's environment.
`papervault doctor` checks the configured endpoint's `/health`.

Then register the server with your agent (Claude Code: install the bundled plugin; other
clients: point them at the MCP endpoint — see `docs/INSTALL.md`).

## Domain

papervault ships tuned for **space physics + AI4Science** as the factory default and worked
sample. The domain layer — search-gate rubric, entity ontology, extraction few-shots,
instruction text — lives in editable config files (`src/papervault/domain/`). Adapting to
another field means editing those files; quality outside the shipped domain is the adapting
operator's responsibility.

## Acquisition state v2

`PAPERVAULT_TERMINAL_V2=1` opts into the first acquisition-pipeline v2 slice
(default off). Acquisition checks retrieval eligibility, canonical identity,
policy permission, and the independent quarantine marker immediately before
download dispatch. Ordinary downloads keep their in-memory coordination;
campaign re-opens use the durable lease and budget providers described below.

Legacy dimensions are derived without rewriting the index: `metadata_only` and
`failed` map to exhausted retrieval, while `ok` and `extract_failed` map to
acquired retrieval (`ok` becomes verified only with a recorded, labelled extract).
All cause-unrecorded terminals retain unknown policy and cannot acquire again
until a separate policy ruling. A status reset or identity backfill alone cannot
clear that policy. New status writes record dimensions and cause; legacy dimensions
are persisted only when their record is next changed. Existing artifacts remain
available through the ordinary serving path.

`PAPERVAULT_IDENTITY_V2=1` opts into slice 2 (default off). Its pure identity
ladder selects DOI, then a namespaced source ID, then a URL; CORE work and output
IDs stay distinct. The ladder selects a canonical identity without merging
records or changing download order. URL recovery checks correspondence again
before adopting a DOI: unresolved evidence stays retryable, and a contradiction
records a rejection of that route while retaining the paper and other routes.
Redirects retain the final evidence-bearing URL; supplied landing/source
bibliography participates in the decision, and raw evidence types survive adoption.
Existing identity records remain readable; optional evidence is omitted when absent.

Correspondence uses a minimum normalized title length of 25, title sequence
similarity ≥0.92 (≥0.97 without a reference year), title token Jaccard ≥0.90,
author surname Jaccard ≥0.50 plus reference first-surname agreement, and a year
difference ≤1. Missing or malformed required evidence stays unresolved. DOI
resolver forms and supported publisher DOI paths are compared canonically;
supplied contradictory bibliography rejects the route even when DOI forms agree.

With the identity flag on, `domain_status` markers `off_domain`, `non_paper`,
and `bad_extract` project to excluded policy with the same reason; unknown markers
project to unknown policy. Retrieval state and existing artifacts are retained.
Existing scoped policy overrides are retained and must cover the domain reason.
This applies to legacy and stored dimensions at read time. Enable terminal-v2
to enforce this projection at ordinary download dispatch. Re-open v2 independently
enforces acquisition admission and domain restrictions for campaigns.

`PAPERVAULT_REOPEN_V2=1` opts into slice 3 (default off). In a maintenance window
with `papervault.service` stopped, run an explicit bounded campaign:

```bash
python -m papervault.library.cli reopen-campaign --cap 20 --dry-run --json
python -m papervault.library.cli reopen-campaign --cap 20 --json
python -m papervault.library.cli reopen-campaign --keys approved.keys --cap 50 --dry-run --json
```

The selector takes exhausted retrieval with allowed policy, canonical resolved
identity and no quarantine. It changes only retrieval eligibility; unknown or
excluded policy needs a separate ruling. Dry-run reads the index and validates
authorization without creating files or sending requests. Apply claims and
enqueues the selected cohort through the existing download queue, waits for its
results, and refuses a live service, unavailable service probe, or active campaign.
An exclusive `.reopen-campaign.lock` prevents concurrent standalone writers.
It does not recover unrelated pending work. There are no automatic campaigns.
Optional `--keys FILE` restricts selection to citation keys in file order (one
per line, with `#` comments and duplicates ignored). Every listed key is checked,
including those after the cap: missing keys, records refused by `reopenable()`,
exhausted budgets, and eligible keys deferred by the cap appear in the accounting.
An empty key file selects nothing.

For the owner-approved ADS recovery in [#245](https://github.com/LeonJoeeee/papervault/issues/245),
resolve UNKNOWN policy with an explicit ruling before the scoped campaign:

```bash
python -m papervault.library.cli policy-ruling --keys approved.keys --action acquire --reason 'Owner approved ADS recovery #245'
# In a maintenance window with papervault.service stopped:
python -m papervault.library.cli policy-ruling --keys approved.keys --action acquire --reason 'Owner approved ADS recovery #245' --apply
```

The default is a read-only JSON report of matched keys and current policies;
missing keys and cohort mismatches are counted and cause exit 2. Apply refuses
the entire request if any key is invalid. The cohort requires source `ads`, no
stored DOI/arXiv ID, an ADS abstract URL agreeing with `paper_id`, and exhausted
`metadata_only` retrieval. Existing exclusions, quarantine, aliases, merge intents,
and unsettled claims cannot be cleared by this ruling. UNKNOWN remains deny by
default everywhere else.

Apply appends one ruling (UTC timestamp, action, reason, keys, revision) to
`correction-ledger.json` before durably saving grants. Each grant covers only
`acquire`, has an empty reason set, is bound to its citation key, and meets the
selected records' required revisions. Retrieval and download status stay intact;
the campaign owns the subsequent re-open. Repeating a recorded key/action/revision
does not add evidence rows, including recovery after an interrupted index save.
Rulings and standalone campaigns share the writer lock and refuse a running
service or unavailable service probe. The orchestrator owns live application.

Standing authorization permits at most 200 reservations per campaign and three
campaign starts in any rolling seven days (UTC wall clock), including interrupted
starts. Larger campaigns require the owner and are refused by this CLI. The
append-only `reopen-budget.jsonl`, beside `index.json`, records version-1 JSON
events for authorization, reservation, dispatch, completion, proven-not-sent
refunds and fencing. Appends and claim index saves require successful `fsync`;
unreadable accounting refuses acquisition. A reservation precedes dispatch and
survives process restart and index rollback. A restored record missing its claim
cannot acquire through the ordinary path when the ledger records its generation.
Refunds are compensating events,
never removal of consumption; uncertainty never refunds or resends.

The optional worker-owned `reopen_claim` record stores generation, lease ID,
expiry, campaign, scope and phase. Every result uses that exact capability;
state and PDF/text publication are fenced together under the library write-lock.
Attempts download into `.reopen-staging/<lease-id>/` inside the vault, with copied
assets and a detached record. Successful publication checks current policy,
identity and changed-artifact baselines; it preserves concurrent metadata and
applies artifact deletions as well as writes. Completed generations apply once.
Normal legacy records keep their serialization when the flag is off. A publication
fence failure halts further campaign dispatch; queued attempts refund only when
the runner proves they were never sent.

Expiry or uncertain execution sets the independent quarantine marker; expiry
never proves thread termination. Dispatched attempts are quarantined on restart,
and their resources remain charged. Cancellation quarantines promptly and retains
the download permit until the executor actually drains. Turning the flag off stops
new campaigns and does not turn persisted claims into ordinary downloads. Uncertain staging must be
retained until recovery establishes drain/fencing. The maintenance API
`ClaimStore.release_quarantine(key, drained_lease_id=...)` requires evidence that
the exact executor drained, or an explicit `human_reason=...`; it advances the
durable generation and records the evidence without refunding. After every
campaign executor is drained/fenced, `BudgetLedger.finish_campaign(...,
drained=True)` records closure. These are single-writer operations, matching the
existing process model in `services/concurrency.py`; they are not scheduler tasks.

`PAPERVAULT_MERGE_V2=1` opts into slice 5 (default off). Duplicate identity
consolidation and preprint/journal promotion persist a paired merge intent,
refuse new claims on both keys, and wait for queued work, live executors and
durable campaign claims to drain. Each checkpoint uses the existing library
write lock; it releases the lock between checkpoints so executors can finish.
The merge intent projects quarantine for new admission without marking existing
execution as uncertain. Dispatched campaign results still pass the normal
generation, lease, policy, identity and budget publication checks and settle their
claims. Lease-expiry or execution-uncertainty quarantine continues to drop results.
The default wait is 60 seconds (`MergeStore.quiesce(..., wait_seconds=...)`).
An expired or uncertain wait stays quarantined and blocked after restart;
expiry never permits a merge or refunds uncertain consumption. The operator
or orchestrator can explicitly call `await MergeStore.restart(expired_intent,
wait_seconds=60)` to persist a fresh paired wait without releasing either fence.
The returned intent must still pass the normal drain checks before `merge_records`;
ordinary retries do not restart expired waits. Uncertain execution/commit intents
remain blocked and require separate operator recovery.

The canonical record retains the union of restrictions, intersection of grants,
and source observations. The absorbed record remains an alias; `resolve_alias`
returns its canonical record. Existing artifacts remain in this vault and are
read through retained paths. `reopen-budget.jsonl` receives an explicit `merge`
event with both keys, generations before/after and drain evidence, before the
index is saved. Spent budgets sum, ceilings take the minimum of the affected
keys' campaign caps (the standing 200-paper cap for keys with no history), and
generations re-sequence to their sum plus one. A ledger/index disagreement fails
closed and requires operator repair; an uncertain commit is never replayed.
Disabling the flag retains fences on persisted intents and merged records.
Plain metadata fills use their existing path; untouched legacy rows gain no
merge fields. Standalone synchronous consolidations use the same write lock;
runtime callers must own it on the server's event loop.

`PAPERVAULT_POLITE_V2=1` opts into slice 4 (default off). API sends for Wiley,
Crossref, Unpaywall, Semantic Scholar, OpenAlex and arXiv share one process ledger
per service, combining credentials, IPs and endpoints conservatively. Every
configured sliding window must admit the send, with concurrency accounted
separately. A full window, occupied permit, cooldown or unavailable accounting
refuses the send; the wrapper does not add retries or waits. With the flag off,
the original transport receives the same arguments and returns the same response.

Wiley starts at 1 request/second **and** 30/600 seconds, below the issue's
documented 3/second and 60/600 seconds. Semantic Scholar starts at 1/2 seconds
(the batch module documents 1/second); arXiv's API retains 1/3 seconds. Each
service has concurrency 1. Crossref, Unpaywall and OpenAlex have no numeric
provider limit documented in the current code and remain **disabled** until an
explicit envelope is configured and authorized. Retry delays, sweep caps and
the existing network semaphore are not provider windows.

`PAPERVAULT_POLITE_ENVELOPES` accepts a JSON object keyed by `crossref`,
`unpaywall` or `openalex`. Each entry needs `analogue` (`wiley`,
`semantic_scholar` or `arxiv`), `initial` and `hard_cap` arrays of
`[requests, seconds]`, and a nonempty `authorization` reference. An initial
window must correspond to each analogue window, in increasing duration order;
its duration may grow, its burst cannot exceed the analogue, and its rate must
be at most half the analogue. Caps specify maximum counts and minimum durations
for those corresponding windows. Missing
fields disable that scope; malformed or changed live configuration fails closed.
For example, the shape for a separately authorized envelope based on Wiley is:

```json
{"crossref":{"analogue":"wiley","initial":[[1,1],[30,600]],"hard_cap":[[1,1],[60,600]],"authorization":"owner approval reference"}}
```

This example records an assumption, not a claim about Crossref's permission.
The pure `Ledger.increase(scope, windows)` API takes explicitly chosen values:
at most +25% in rate per window after 24 hours and fresh, completed 2xx evidence.
Counts may increase or durations shorten, within the configured caps. There
is no automatic ramp. Configurations, increases, cap freezes, cooldowns and
resumes are inspectable through `Ledger.snapshot()`. A 429, or three timeouts
in a rolling 60 seconds, starts a 60-second scope cooldown and stops further
speed-up. A late timeout extends the quiet window. After it expires, requests
resume at the same limits. Unread/partially closed streams and unproven raw
streams without a content length earn no ramp evidence. The ledger is in memory
under the existing single-process model; envelope state and request windows do
not survive a restart, so restart is not a supported way to clear a live quota
or cooldown. Browser/stealth, publisher content, redirects and arXiv page/PDF
paths remain for later slices; existing semaphore and cascade behavior is retained.

`PAPERVAULT_STOPRULES_V2=1` opts into slice 6 (default off). Reconcile evaluates
the rules before routing; download queue rounds and campaign settlement also
evaluate them. The detectors consume durable outstanding claims and merge
redirects, actual scope send timestamps, and explicit publication evidence.
They never interpret an admission refusal or a rejected PDF tier as a violation.
Named thresholds in `library/stop_rules.py` are zero: duplicate in-flight
identity halts the campaign; an exceeded quota window cools the scope and halts
active campaigns; identity-mismatched publication halts its campaign (or active
campaigns for unclaimed work) and quarantines the current record.
Publication comparison uses the adopted DOI-first canonical identity and resolves
merge aliases. Clearing or repairing a secondary identifier or route leaves a
matching canonical identity intact and does not trigger a stop.
Verified blank identifier fills keep their existing promotion behavior; the
comparison advances only for fills acknowledged by the existing writers.
`SIGNIFICANT_REGRESSION_FRACTION=0.20` defines
the measurement term; it does not enforce a fourth rule without measurements.

Campaign halt is an idempotent append-only ledger event, distinct from verified
finish/drain. Replay denies new reservations and dispatches, while existing
dispatched work keeps its completion and accounting semantics. Uncertain work
stays charged; proven-unsent queued work can settle normally. Scope cooldown
uses the same lock and configured quiet window as 429/timeout cooldown; repeated
rounds do not extend an active cooldown, and resume never increases the limits.
Disabling the flag stops evaluation and new measurement fields; persisted
campaign halts continue to fence admission.

Enabled manifest events add `identity_version` (v2 identity definition),
`attempt_generation`, `lease_id` and an opaque `request_scope_id`, preserving
existing fields. Unknown correlation values are null; request scope is distinct
from the campaign's logical scope. Wait, executor/request service, and actual
executor drain have separate duration fields. Incomplete attempts are recorded
as censored, with null unobserved durations, and never back-filled with an
estimated latency or outcome. `admission-denied` records refused sends;
`verified-success` requires positive identity verification and live publication,
so verification fail-open and temporary campaign stages do not count as success.

## URL identity backfill

For records with a URL but no DOI or arXiv ID, inspect a bounded sample without
writing to the vault:

```bash
python -m papervault.library.cli identity-backfill --cap 20 --json
```

Recovery checks DOI/publisher and arXiv URL patterns, then one citation-metadata
landing fetch (including its redirect target), then exact ADS, CORE, OpenAlex or
INSPIRE metadata. DOI registry metadata or the current arXiv page must pass the
existing title/author consistency screen before an identifier is filled. Missing,
blocked, ambiguous or conflicting evidence causes abstention. Existing identifiers
and collisions are left intact; successful fills retain route, URL, verified title
and UTC observation time in `identity_recovery`.

In an authorized maintenance window after deployment, add `--apply --acquire` to
persist identities and immediately run the ordinary current-only cascade on only
the newly recovered rows without existing assets. `--keys FILE` restricts selection
to listed citation keys (one per line). `--cap` is required, accepts 1–50, and bounds
both inspected and acquired records; unrelated terminal sets are never reset.
Writes refuse while `papervault.service` is running. The JSON result and library
manifest record recovery counts by route, acquisition outcomes, and successful
sources (including the winning member of aggregator groups). Report that post-deploy
pass's numbers on [issue #197](https://github.com/LeonJoeeee/papervault/issues/197).

The standing correction pipeline uses that same verification/application path with
acquisition disabled. Notify it explicitly after a credential becomes available or
an affected source is repaired/deployed, in an authorized maintenance window:

```bash
python -m papervault.library.cli correction-event credential_arrived --source ads --cap 20 --apply
python -m papervault.library.cli correction-event source_repaired --source core --cap 10 --apply
python -m papervault.library.cli correction-event identifiers_arrived --source ads --keys affected.keys --cap 10 --apply
```

These commands are the operator notification points; they do not watch credentials
or poll source health. Source adapters can use the same async `run_event` interface
with the running library. `identifiers_arrived` requires explicit stored citation
keys whose source identifiers or locator URLs gained evidence. Each event intersects
keys with its source cohort, then caps selection at the required `--cap` (1–50).
Without keys, only identifierless rows and unrecorded verified routes consume the cap.
Known identifiers remain intact. The standing pipeline applies DOI recoveries;
non-DOI claims are recorded as deferred because #198's lower identity ladder is TBD.
Requests are serial, with at least one second between inspected records and sweep
cohorts; the reviewed source helpers retain their timeouts, limits and cooldowns.

Library background boot owns a monthly correction task. Its durable due time has
a 30-day floor, survives restart, and starts 30 days after first activation. An
operator may run the first sweep earlier with `correction-sweep --apply`; subsequent
sweeps respect that floor. Each cohort defaults to 20 selected papers (hard maximum
50); the whole sweep defaults to and cannot exceed 100. Both citation-key and cohort
cursors rotate so capped misses and later cohorts progress. CORE starts last per
its measured ceiling. Override the defaults downward with `--per-cohort-cap` and
`--total-cap`. The constants and validation live in `services/correction_pipeline.py`;
the reviewed backfill's record ceiling remains in `services/identity_backfill.py`.

`correction-ledger.json`, beside the vault's `index.json`, keeps conflicts, immutable
batch manifests, adjudications, route findings, scheduling and monthly aggregates.
The library manifest records every pass and monthly report. Merge/enrichment and
verified-identity claim hooks stage conflicting fields with both original sides and
provenance; they retain existing fill-blanks behavior. The backfill still never
merges or purges a collision (the older title-based DOI resolver keeps its existing
merge behavior, with field conflicts staged before merging).

At ten distinct pending subject papers, one atomic ledger write emits and assigns
a manifest for an **orchestrator-dispatched Codex CLI `gpt-6-luna` / `max`** agent.
Each manifest contains ten papers and at most fifty conflicts. No agent is launched
by the library. Export its actual evidence with `correction-report --kind manifests
--manifest ID`. Import an investigation using `correction-adjudicate results.json
--apply`, where JSON contains `manifest_id` and `results` entries with `conflict_id`,
`status` (`resolved` or `needs_human`), `decision`, and `evidence`. Resolved answers
require evidence; omitted/unadjudicable answers become retained `needs_human`.
Identical replay is idempotent; foreign/duplicate answers fail atomically. Adjudication
records a decision and does not apply field changes. Those require separate scoped
authorization. `correction-report --month YYYY-MM` reports aggregate counts and current
unresolved dispositions; `--kind conflicts` exports the retained evidence. All report
exports are read-only; maintenance writes refuse while the service is running.

Verified stored ADS full-text observations are copied into the route ledger. A #205
survey packet may also be supplied to a key-scoped ADS event using `--routes FILE`:
a JSON list of `{key, url, source_identifier, provenance}` objects, with the exact
stored ADS bibcode and provenance containing timezone-aware `checked_at`,
`verification: "remote_pdf"`, and the survey's evidence (for example `issue: 205` and
`status_code: 200`). Imports require selected keys and at most three routes per paper.
`correction-report --kind routes` is the #174 scoped re-open's selection interface.
The correction pipeline downloads no assets and changes no terminal states.

Download telemetry is appended to the same manifest by ordinary acquisitions. Existing
outcomes retain their fields and gain `timing`: a 32-character `run_id`, UTC `at`,
and, inside a tier slot, `slot_id`, `slot_started_at`, `slot_ended_at`, and monotonic
`slot_duration_s`. These slot endpoints describe elapsed time **at outcome emission**;
use the corresponding completed `tier` span for the full slot including final bookkeeping.

Additional `download_telemetry` records share `run_id`, paper `key`, `actor`
(`download` or competing `search`), UTC `at`, process/thread IDs, and `phase`/`mark`.
Spans have `start` and `end` marks with `span_id`, `parent_span_id`, `started_at`,
`ended_at`, monotonic `duration_s`, and control-flow `status` (`ok`, `error`,
`cancelled`). Status `ok` means the operation returned; tier success remains the
existing outcome event. Phases cover `queue_worker`, `network_wait`, `network_hold`,
actual `executor` work, `cascade`, arXiv `preflight`, `tier`, `strategy`,
`verification`, `pdf_save`, `firecrawl`, parallel `group`/`member`, and `browser_call`.
`enqueue`/`dequeue` points pair by queue sequence and run ID; dequeue includes
`enqueued_at`, `queue_wait_s`, worker ID and priority. `queue_drop` points close
stale priority-promoted tuples with their sequence and dequeue time. Cascade end carries
`paper_status`, `outcome_source` and `pdf_returned`; group `winner` points precede
executor drain. Mirror members use indices, never target URLs. Telemetry stores
scalar fields only, bounds labels to 64 characters and keys to 128, and retains
only one admission stamp per queued tuple. New-record failures emit a warning and
do not change acquisition results. Manifest writes serialize complete JSONL records.

The semaphore spans describe **whole cascade/search admission**, including browser,
verification and executor drain; member fan-out can issue multiple requests per
permit. Browser-call spans observe the exposed fetch API, not process creation/exit,
PSS/RSS or isolated remote latency. Cancellation can release a permit before the
executor thread ends; measure both timelines. Partial spans at capture boundaries
must be censored. Request counts/status/rate headers and same-window host process
memory measurements remain necessary to validate upstream limits and resource costs.
This instrumentation changes neither the four workers nor the four semaphore permits.

## Catalog persistence preparation

The default authority remains `index.json`. The optional
`PAPERVAULT_CATALOG_AUTHORITY=postgres` backend requires an explicit
`PAPERVAULT_CATALOG_DSN` and a migrated catalog. Library saves commit guarded delta
batches with audit and pending work; stale conflicts surface as `catalog_conflict`.
PG saves leave JSON untouched; run `scripts/export_catalog.py --output <path> --verify`
for the canonical export. PG purge/deletion is refused during P2.

The [readiness checklist and rehearsal runbook](docs/specs/2026-10-18-p2-readiness.md)
cover the copy-only cutover/rollback tools and remaining owner gates. Live activation
is not authorized by this release.

## Coordinated recovery

P5b provides `scripts/snapshot_derivatives.py` and `scripts/rebuild_derivatives.py`.
Both require explicit roots and connection identities and default to dry-run. They
disable dotenv discovery. `PAPERVAULT_KNOWLEDGE_JOBS=legacy` remains the default;
these tools do not activate catalog scheduling, change the serving generation,
promote a rebuild, or delete research assets.

The operational roots approved in [#283](https://github.com/LeonJoeeee/papervault/issues/283)
are `/data/p5b-recovery` and `/data/p5b-drills/<drill-id>` on the owner's RAID.
The worker's staging runs use explicit isolated `/tmp` roots, recorded in
[the baseline evidence](docs/baselines.md). The orchestrator owns deployment,
weekly operational snapshots and production drills in the monthly reboot window
(at most two hours). No timer or live backup configuration is activated by this PR.

A complete set contains the catalog schema, installed index definitions (including
importer-installed DOI/arXiv uniqueness), exact row encodings, sequence state,
checkpoint/epoch/audit boundary; the entire vault including source bytes, extraction
inputs/versions and operational ledgers; knowledge inputs/local state; recipes and
receipts; all PostgreSQL KV/cache/doc-status/vector tables including retained extra
tables and indexes; and Neo4j nodes, relationships, properties, indexes and
constraints. Every component binds the same checkpoint and has file lengths and
SHA-256 hashes. Verification also checks sealed receipt ancestry and recomputes
the hashes of retained PostgreSQL/graph materializations; empty tables cannot stand
in for missing outputs. Pending extraction completions must retain their source,
input, recipe and artifact dependency closure even before catalog registration.
Catalog registration happens after the exported checkpoint and
cannot appear in its own dump. The existing nightly catalog-only dump and hourly
JSON export remain separate operations; neither alone covers these dependencies.

Drain all library and graph clients before capture. The coordinator holds exclusive
vault/knowledge-root leases, catalog writer/backend-access advisory locks, and
PostgreSQL table locks through export, verification and registration. Graph clients
hold their root lease through successful storage finalization; failed flushes fail
closed. Neo4j's transaction relies on this drained application boundary, so direct
administrative/backend writers must also be stopped by the operator. A busy barrier
refuses capture. An independently timed collection of database dumps is not accepted
as a coordinated set.

The source JSON configuration has this shape (credentials are named environment
variables, never values in a committed configuration):

```json
{
  "vault": "/data/paper-vault",
  "knowledge_root": "<explicit existing knowledge working directory>",
  "catalog": {"dsn_env": "RECOVERY_CATALOG_DSN", "schema": "<catalog schema>"},
  "postgres": {
    "dsn_env": "RECOVERY_BACKEND_DSN", "schema": "<backend schema>",
    "roles": {
      "full_docs": "lightrag_doc_full", "text_chunks": "lightrag_doc_chunks",
      "full_entities": "lightrag_full_entities", "full_relations": "lightrag_full_relations",
      "entity_chunks": "lightrag_entity_chunks", "relation_chunks": "lightrag_relation_chunks",
      "llm_response_cache": "lightrag_llm_cache", "doc_status": "lightrag_doc_status",
      "chunks_vdb": "lightrag_vdb_chunks", "entities_vdb": "lightrag_vdb_entity",
      "relationships_vdb": "lightrag_vdb_relation"
    }
  },
  "neo4j": {"uri": "<explicit source URI>", "database": "neo4j",
    "user": "neo4j", "password_env": "RECOVERY_NEO4J_PASSWORD"}
}
```

Use the actual installed table names, including any embedding-model suffixes. A
missing role or incompatible schema refuses capture. The source catalog must have
the explicit knowledge-schema migration (version 7 with P5c); capture does not migrate
the source. Run the preview first; the
orchestrator can schedule the applied `--due` command weekly after arranging drain:

```bash
.venv/bin/python scripts/snapshot_derivatives.py --config source.json --root /data/p5b-recovery
.venv/bin/python scripts/snapshot_derivatives.py --config source.json --root /data/p5b-recovery --due --apply
.venv/bin/python scripts/snapshot_derivatives.py --verify /data/p5b-recovery/<snapshot-id>
```

`--due` requires a new complete set when the newest verified retained set is at
least seven days old. UTC microseconds plus a UUID distinguish same-day snapshots.
Four verified complete sets roll; failed and sample exports cannot replace them.
Older packages remain protected if they still hold the only retained copy of
research/input bytes, so an unresolved protection can temporarily exceed four.
Mutable projections and ledgers are included in every set without pinning each
historical revision indefinitely. Removal is limited to verified owned recovery
package copies; no source tree or live backend is pruned.

The operational derivative soft cap is **2 TB (2,000,000,000,000 bytes)**, with an
**80% alert (1.6 TB)**. Plans account for derivatives, retained sets, a staged export
and a held capacity reservation. Insufficient filesystem headroom refuses the run;
crossing the soft cap emits alert evidence and never triggers asset deletion.
Blocking snapshot/restore work has a two-hour deadline. Logical exports use memory
proportional to the exported database data; full-corpus resource qualification is
still required before operational scheduling.

Restore and replay use separate target configurations. Catalog targets must be
absent `catalog_recovery_*` schemas in an explicitly named `test`/`rehearsal`
database; PostgreSQL derivative targets use absent `recovery_backend_*` schemas
with matching extensions. Neo4j requires a distinct empty endpoint/database and
`"disposable": true`; source aliases and populated unowned targets are rejected.
Reruns resume only a target bound to the same operation and package. The packaged
DDL is trusted operator material, not an import format for untrusted third parties.
Catalog export format 2 binds the full installed index inventory to the checkpoint.
Restore recreates additional indexes and compares every definition with the source,
including on resume; invalid/unready indexes and older packages without index state
are refused. An index definition incompatible with the bundled table constraints
rolls back the empty catalog target rather than silently weakening its integrity.

```bash
.venv/bin/python scripts/rebuild_derivatives.py --mode restore --package <set> --root <empty-target> --generation restored
.venv/bin/python scripts/rebuild_derivatives.py --mode restore --package <set> --root <empty-target> --generation restored --config target.json --apply
.venv/bin/python scripts/rebuild_derivatives.py --mode replay --package <set> --root <other-empty-target> --generation replayed --source-generation <source-generation> --config replay-target.json --apply
```

Restore preserves historical content recipes/receipts, installs a new durable writer
fence and epoch, and records a separate materialization receipt. Copied dispatch
remains disabled; that receipt cannot supply fresh live acceptance. Replay imports
only selected original inputs/recipes into a fresh candidate catalog and runs the
normal admission, execution, readback, sealing and publication path, producing new
attempts and receipts. Isolated replay explicitly resets unit-level execution
quarantine on fresh units; its plan and receipt record each affected unit's original
revision and reason under `unit_quarantine_disposition`. Record-level policy and
quarantine remain enforced, and the source is unchanged. This reset is scoped to
non-promotable sample execution, not a live quarantine release. It never imports
historical runnable jobs. Physical replay
requires `"physical_replay": true`, explicit isolated PostgreSQL/Neo4j targets, the
exact supported recipe, and its installed models/providers. It does not substitute
a different transform when the original recipe is unavailable. Metadata replay
needs no graph/model calls. Physical replay's PostgreSQL DSN must name one IP
address (with the same `hostaddr`, if present); DNS, multi-host and alternate
address routing are rejected. Runtime tables must resolve to the owned target
schema, including on resume, before any backend write.

Both paths recheck completed materializations on resume. Restore verifies its added
candidate/catalog rows against an atomically recorded digest and rejects extra
Neo4j indexes or constraints. Replay preserves completed
admissions, stops on errors or deadline, and may resume in a later permitted monthly
window; it cannot extend a consumed window in the same month. All drill/replay
candidate generations created by the P5b replay tool are permanently marked sample
and cannot promote. P5c's separate candidate fork/promotion path is described below.

Weekly cadence is a scheduling policy, not a committed RPO; acceptable loss and
full-system recovery time remain unqualified. Sample elapsed time is not full RTO,
especially where source strata or rebuild work were unavailable. RAID and these
local copies share the host failure domain. Off-host replication and host-loss/
disaster recovery remain outside P5b and are not satisfied by this evidence.

## Disposable extracts and isolated maintenance

P5c adds `scripts/maintain_derivatives.py` for catalog inspection, explicit extract
disposition, monthly windows, resumable GC, candidate forks and promotion. It disables
dotenv discovery, requires an explicit DSN environment name, schema and vault, and
defaults to a read-only plan. Catalog scheduling remains opt-in; the default is
`PAPERVAULT_KNOWLEDGE_JOBS=legacy`. The orchestrator owns production activation and
the monthly reboot drill. Worker rehearsals use disposable stores and `/tmp` roots.

Every registered extract starts **protected**. An operator may mark only its
immutable markdown output disposable using a reason, nonce and exact retention
revision. Eligibility begins **90 days after accepted supersession**, regardless of
file age: the clock starts when the last extraction head or record selection ceases.
Reselecting an older output cancels that clock; removing its last selection starts
a new ninety-day interval. Record selections include physical path aliases.
Pre-P5c receipts start protected. Their migration timestamp is provisional: first
verified vault binding establishes a conservative clock from current selections,
since earlier absolute-path and alias selections cannot be reconstructed without
that root. An unselected historical output waits ninety days from that binding.
Source inputs, receipts, current heads, selected records, live generations, queued
jobs, reader pins and unresolved recovery/job pins remain protected. Reader pins
expire within two hours; job/recovery pins require an explicit reviewed disposition.

GC commits a permanent deletion intent before filesystem IO and checks reference
admission under the same catalog writer lock. A tombstoned identity cannot acquire
new references. It verifies the exact path, hash, size and single-link regular file
through directory descriptors, rejects symlinks, and unlinks only that `output.md`.
A failed unlink or lost catalog response retains its intent; retry resumes the same
deletion under a fresh valid lease. Filesystem/catalog writers must use the normal
fences; administrative writers must be drained during maintenance. Recovery sets
accept valid disposal evidence while still verifying every source and receipt.
Pre-GC snapshots conservatively keep their existing protected-byte retention; GC
does not change snapshot cadence or automatically prune those older packages.

Use `inspect` to obtain the current authority, window, retention revisions and
eligibility reasons. Supply a JSON payload for every other action:

```bash
.venv/bin/python scripts/maintain_derivatives.py inspect --dsn-env MAINTENANCE_CATALOG_DSN --catalog-schema <schema> --vault <vault> --operation-id inspect
.venv/bin/python scripts/maintain_derivatives.py gc --dsn-env MAINTENANCE_CATALOG_DSN --catalog-schema <schema> --vault <vault> --payload gc.json --operation-id gc-batch
# Apply only in the explicitly recorded window; one batch names at most 100 completions.
.venv/bin/python scripts/maintain_derivatives.py gc --dsn-env MAINTENANCE_CATALOG_DSN --catalog-schema <schema> --vault <vault> --payload gc.json --operation-id gc-batch --owner <operator> --window-id <window> --apply
```

| Action | Required JSON payload |
| --- | --- |
| `window` | `window_id`, `kind: "monthly_reboot"`, timezone-qualified `starts_at`/`ends_at` (at most 7,200 seconds), `operator_nonce`, `expected_revision` from authority's `maintenance_revision` |
| `disposition` | `completion_id`, boolean `disposable`, `reason`, `operator_nonce`, `expected_revision` from `retention_revision` |
| `gc` | `completion_ids`: 1–100 distinct explicit completion IDs |
| `fork` | new `generation_id`, active `source_generation_id`, `expected_manifest_revision`, `expected_requirements_revision` |
| `promote` | candidate `generation_id`, `expected_manifest_revision`, `expected_requirements_revision`, `expected_serving_revision` |

For `inspect`, an optional payload containing `generation_id` includes that generation's
revision. Fork and promotion require `--owner`/`--window-id` on apply, just as GC does.
The tool holds a fenced maintenance lease and rechecks it before committing. Fork
queues work in a fresh candidate with no selected output. The ordinary scheduler
builds it only inside the monthly window with funded capacity. Candidate publication
never changes production selection. Both scheduler awaits and guarded backend
operations cancel at the window deadline. Already-sent remote operations can have
unknown outcomes; their private versions and capacity remain protected until verified.
Promotion requires a complete compatible
manifest, fresh output observations and exact current revisions; a sample generation
cannot promote. Preserve the operation ID and payload when retrying a lost response.

Catalog intake and new execution reserve physical space under a shared
`<knowledge_root>/maintenance/capacity` root. The **2 TB** cap and **80%** alert include
filesystem derivatives, outstanding reservations and the external usage census.
Insufficient capacity blocks new work and preserves current/protected bytes; there
is no emergency deletion. `PAPERVAULT_CATALOG_EXECUTION_BYTES` is a positive per-job
estimate (default 1 GiB), including scratch and output, not a hard quota. Admission
requires headroom for outstanding jobs' output plus twice the new estimate for
its reservation and output.
Unknown execution outcomes keep their reservation across restart until verified
receipt recovery; unknown files are never automatically cleared. Completed external
growth remains charged until a qualified census started after settlement covers it.
Admission samples the census and counts reservations inside the same serialization
boundary used by settlement. A controlled build pause settles the exact deferred
attempt, including after restart, while keeping its partial output charged until
a later census covers it. Recovery pins and uncertain cost records remain intact.
Receipt recovery
and publication of already sealed work remain available when new admission stops.

`PAPERVAULT_DERIVATIVE_USAGE_FILE` must point to an operator-owned mode-0600 JSON
census refreshed within 300 seconds, with `scope: {vault_root, knowledge_root}`,
timezone-qualified `measured_at`, and nonnegative `backend_bytes`. Include external
PostgreSQL/Neo4j derivatives and retained sets outside the two filesystem roots;
the runtime also measures catalog relation bytes. Missing, stale or mismatched
measurements block new work, including metadata intake. An explicitly measured
zero is valid for a metadata-only fixture. This is conservative admission
accounting; per-job estimates and Neo4j size estimates do not establish an exact
disk quota or corpus-scale capacity qualification. The 80% threshold logs a capacity
alert. A census marked `metadata_sample: true` cannot fund physical execution.

Generate the census with `scripts/measure_derivative_usage.py`. Reuse the explicit
snapshot source configuration above and add `extra_derivative_roots`, listing retained
recovery sets and every other derivative filesystem root outside `vault/extracts`
and `knowledge_root` (an explicit empty list affirms none). The output's parent must
already exist. Both PostgreSQL and Neo4j adapters are required unless
`metadata_sample: true` proves that the catalog has only metadata dependencies.
The preview only prints measurements; `--apply` atomically publishes the protected
census. Refresh it at least every 300 seconds while new work is being admitted.

```bash
.venv/bin/python scripts/measure_derivative_usage.py --config source-with-capacity-roots.json --output <usage.json>
.venv/bin/python scripts/measure_derivative_usage.py --config source-with-capacity-roots.json --output <usage.json> --apply
```

`scripts/drill_derivative_maintenance.py` exercises the protocol in a new disposable
PostgreSQL fixture and small isolated filesystem roots. The recorded P5c rehearsal
in [baseline evidence](docs/baselines.md) covers deletion retries, preserved source
hashes, capacity refusal, isolated candidate publication and guarded promotion.
Its metadata-only timings do not qualify full-text rebuilds or production RTO.
The pre-existing P5b isolated recovery-replay caller constructs its scheduler without
the runtime capacity gate. Its staging reservation does not establish P5c admission
accounting for replay jobs; this remains a separate recovery-replay limitation.

## Releases

papervault is consumed **by pin** (ADR-0001): deployments track an annotated git tag
(`vX.Y.Z[-stage]`), never `main`.

The repo carries **one version**, and every change PR bumps it per the semver call stated in
its description. The declared version fields move together in that PR: `pyproject.toml`
`[project].version` plus the `papervault` entry in `uv.lock` (refresh with `uv lock`),
`plugin/.claude-plugin/plugin.json` `version`, and the `papervault` entry's `version` in
`.claude-plugin/marketplace.json`; `tests/test_plugin_version_sync.py` fails when they drift.

A release IS a tag — CI's build job produces the wheel/sdist for every commit, so tagging is
the whole ceremony: the tag names the version already on `main` (`vX.Y.Z`, plus an optional
`-stage` suffix), and a release commit changes no version field. Latest tag: `v0.1.6-beta`
(closed beta), cut before the one-version rule, so its number predates the package version.

## License

MIT — see [`LICENSE`](LICENSE).
