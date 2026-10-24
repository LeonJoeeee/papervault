# papervault

A self-hosted literature and knowledge layer for coding agents, delivered as one MCP
server. Claude Code, Codex CLI, Antigravity CLI/IDE and other MCP clients share three tools:

| Tool | Purpose |
| --- | --- |
| `search_papers` | Discover papers from a natural-language research intent; submit independently admitted registration and background acquisition. |
| `get_paper` | Resolve a citation key, DOI, arXiv ID or fuzzy title and return metadata plus a serve-safe full-text path when available. |
| `query` | Retrieve evidence from the knowledge graph and synthesize an answer with verified paper citations. |

Version **0.10.13** adds dual-track local retrieval and calibrated LLM planning and gates.
The product is pre-release (closed beta). Read the [PRD](docs/PRD.md) for scope,
[architecture](docs/architecture.md) for boundaries, [installation guide](docs/INSTALL.md)
for deployment, and [baselines](docs/baselines.md#post-v01010-retrieval-coverage-hygiene-and-lightrag-157-baseline--2026-10-24)
for dated measurements. The [constitution and refactor plan](docs/adr/0007-papervault-constitution.md)
govern collection integrity and useful retrieval together.

## Hardware floor (not optional)

The supported deployment uses Linux and CUDA, one **24 GB-class NVIDIA GPU** (RTX
3090/4090 class), Docker for Neo4j and PostgreSQL, and an operator-supplied
OpenAI-compatible LLM endpoint and key. Embedding and reranking run locally; the
independent MinerU OCR service shares the GPU. The hardware floor is part of the
product contract. The factory domain pack is **space physics and AI4Science**;
its search rubric, ontology, examples and instructions live in
[`src/papervault/domain/`](src/papervault/domain/). Other domains need their own
configuration and evaluation gold sets.

## Quick start

Follow [INSTALL.md](docs/INSTALL.md) for the complete setup:

```bash
cp .env.example .env          # fill in the LLM endpoint, key and data roots
# External Docker volumes hold the backing stores; create once.
docker volume create ks-pgdata
docker volume create ks-neo4jdata
docker compose --env-file .env -f deploy/docker-compose.yml up -d
uv venv && uv pip install -e ".[grey]"
.venv/bin/papervault doctor
.venv/bin/papervault serve
```

Provision `mineru-service` separately at `http://127.0.0.1:30000` (override with
`MINERU_URL`). papervault uses the official `mineru-api` HTTP task API through core
`httpx`: submit, poll, then read completed markdown. MinerU owns FIFO processing,
model preload, recovery and GPU unloading after 20 idle minutes. The `mineru` extra
is empty for compatibility; `uv sync --extra mineru` installs no OCR runtime in the
caller. `papervault doctor` checks `/health`.

Knowledge auto-ingest is opt-in: `.env.example` sets `KS_AUTO_INGEST_ENABLED=false`.
Enable it after configuring and populating the vault. With ingestion enabled,
search → acquisition → extraction → distillation compounds the corpus in the
background; returning a search hit does not promise completed acquisition.
Grey download tiers also require explicit opt-in; see [ADR-0004](docs/adr/0004-grey-download-tiers-flag-gated.md).

### Connect an agent

The bundled `plugin/` is a thin MCP registration shell for Claude Code and
Antigravity. Codex CLI and other clients connect directly to
`http://127.0.0.1:8080/mcp` by default. See
[Connecting an agent](docs/INSTALL.md#connecting-an-agent) for client configuration,
custom endpoints and remote tailnet access.

```bash
agy plugin validate ./plugin --json
agy plugin install ./plugin
# Alternative: direct MCP registration
agy mcp add papervault http://127.0.0.1:8080/mcp
```

Start a new Antigravity session after installing. Agent shells contain no retrieval
or lifecycle logic.

### LLM stages and routing

Every text LLM call selects one of nine roles. Set `PAPERVAULT_LLM_<ROLE>` to
`flash`, `standard` or `pro`; unset or blank uses the defaults below. Both direct
endpoints and the optional gateway must serve these level aliases. A supplied
test/client handle bypasses role selection for that call.

| Plane and stage | Role | Default |
| --- | --- | --- |
| S1 search intent: API terms and local KG concepts in one call | `intent` | `pro` |
| S1 ingest/domain and return relevance judges; fuzzy `get_paper` resolution | `judge` | `standard` |
| S1 downloaded PDF identity verification | `verify` | `flash` |
| S1 extracted-text chunk clarity review and document completeness gate | `gate` | `flash` |
| S2 knowledge build: entity/relation extraction and merge summaries | `build` | `standard` |
| S2 query high/low-level keyword extraction when not supplied by decomposition | `keyword` | `flash` |
| S2 multiquery decomposition and source-count/fan-out planning, including eval | `decompose` | `standard` |
| S2 evidence-grounded answer synthesis, including eval | `synth` | `pro` |
| Evaluation answer and refusal judgments | `eval_judge` | `standard` |

Embedding (BGE-M3), reranking and OCR use their existing local/service paths and
do not select text LLM roles. The prompt examples calibrate the planner, both
search judges, PDF identity, clarity/completeness, decomposition and fan-out.

## Search and catalog quality

`search_papers` combines arXiv, ADS, Semantic Scholar, OpenAlex, INSPIRE and CORE
with the local library. The coordinator in
[`SearchPipeline`](src/papervault/library/services/search_pipeline.py) has six
method boundaries: **intent → plan → retrieval → fusion → return gate → serving**.
The operational stages below include the gates and independently admitted command
path within those boundaries; they are not six separate services.

1. **Stage 1 — intent and plan.** The `intent` role parses API/BM25 `search_terms`
   (short phrases), independent canonical scientific `kg_concepts`, result count,
   year bounds and ranking hints in one call. Missing/empty concepts default to
   the normalized terms; the API terms keep their existing expansion rules.
   The post-parse `year_min` guard protects canon: for a lower-bound-only
   plan, canonical/foundational/classical/seminal/historical wording clears the lower
   bound, and survey/overview/current-understanding wording clears it when no explicit
   recency cue exists. A cleared bound also resets `by_recency` to `by_relevance`.
   Bounded date intervals remain intact; unknown years survive year filtering.
2. **Stage 2 — retrieval.** Snapshot the local library, then fetch external terms
   alongside optional KG/vector candidates off the event loop.
   Under polite v2, arXiv, Semantic Scholar and OpenAlex serialize admission/fetch
   through a per-`(event_loop, backend)` lock. Backend pacing and fetch-thread queue
   waits precede timed execution. `PolitenessDenied`, including wrapped denials, and
   queue starvation remain observable but do not trip the circuit breaker. Service
   failures do: the defaults are three consecutive failures and a **3,600-second**
   cooldown; successful zero-hit fetches reset the breaker too. Configure
   `PAPERVAULT_SEARCH_BREAKER_TRIPS` / `PAPERVAULT_SEARCH_BREAKER_COOLDOWN_S` and
   `PAPERVAULT_SEARCH_FETCH_WORKERS` / `PAPERVAULT_SEARCH_FETCH_QUEUE_TIMEOUT_S`.
   Unconfigured, degraded and genuine zero-hit sources remain distinguishable.

   **Stage 2.5 — fusion.** `SearchPipeline.fusion` resolves aliases and compatible
   bibliographic duplicates, retaining external rank evidence on local records.
   Dual-track local retrieval combines **BM25 + PGVector chunks/entities/relations
   + Neo4j cross-facet topology + authority**. BGE-M3 embeds the full user query
   and canonical concepts; chunk vectors use the full query, while entity and
   relation vectors also use concepts. RRF merges their paper ranks. Edges between
   different concept facets and high-weight hub edges add topology evidence.
   `file_path` values split on `<SEP>` resolve to retained canonical paper keys;
   serving exclusions and year bounds still apply. KG candidates can reach the
   local pool without a BM25 keyword match. KG ranks contribute to fusion beside
   per-term local ranks; external ranks retain their existing evidence and pool.
   Term-wise BM25/KG/RRF pooling admits up to `LIB_CAP=45` local candidates plus
   `ALSO_INCLUDE_CAP=45` additional externally vouched local records; the external
   pool cap is 60. The local `authority_weight` prior enters RRF **before the pool
   cut**, helping landmark papers reach judgment. It also weights the vouched
   supplement. Explicit recency/importance ordering bypasses this prior.

   `PAPERVAULT_SEARCH_KG=0` disables this optional arm. If unset, it is enabled
   only when `PAPERVAULT_STORAGE_BACKEND=postgres`; `PAPERVAULT_SEARCH_KG=1`
   explicitly enables it. Configure the knowledge database with `POSTGRES_*`
   (not the catalog DSN), `POSTGRES_WORKSPACE` and `NEO4J_*`; any explicit Neo4j
   workspace must match. Queries use PostgreSQL read-only transactions with
   `hnsw.ef_search=120` and Neo4j read sessions, without schema initialization.
   When `NEO4J_DATABASE` is unset, the reader tries LightRAG's
   `chunk-entity-relation` database and falls back to the home database only if
   the preferred database does not exist. Encoding shares S2's GPU admission
   semaphore; database reads do not hold its permit.
   Missing configuration, embedding/backend failures or unavailable stores yield
   no KG candidates and leave the BM25/external search available.
3. **Stage 3 — domain/ingest gate (the “flash gate”).** Judge external candidates
   against the domain-pack tiers and paper classification. Source publication types
   take precedence when conclusive; an LLM paper/non-paper judgment fills missing
   metadata. Production search passes a shared judge handle to its gates, defaulting
   to `PAPERVAULT_LLM_JUDGE=standard`; the separate extraction `gate` role defaults
   to `flash`. The stage name does not select a model.
4. **Stage 4 — relevance and quality gates.** `registration_rejection` screens the
   external pool for conference/meeting abstracts, non-papers, contentless stubs and
   invalid new-record metadata before preview. Newly allocated records need named
   authors and either a DOI/arXiv/INSPIRE identity or sufficient citation evidence;
   an identifierless record also needs an abstract. Known identities have narrower
   legacy-quality exceptions. `judge_return` then scores local and eligible external
   candidates against the actual research need; only scores **≥0.4** survive.
   Failed/unjudged batches are dropped and counted, never accepted by default.
5. **Stage 5 — decoupled registration and acquisition.** The command set is selected
   independently of the served result cap. After constructing the response, search
   submits bounded registration requests without awaiting admission, catalog writes
   or acquisition. The command rechecks `registration_rejection`, current identity,
   policy and catalog revisions, then commits admitted metadata and pending work.
   Background queues acquire and extract accepted papers. Queue refusal, shutdown
   or command failure is traced and does not change the search response.
6. **Stage 6 — final ordering and serving.** Within the return gate, default
   `by_relevance` ordering blends dense relevance rank with authority rank using RRF
   (`k=60`); it can cross relevance-score bands but cannot resurrect a rejected paper.
   Named-entity slot reservation retains accepted witnesses within the result cap.
   Serving rechecks current exclusions and years before returning. This ordering
   precedes the asynchronous Stage 5 command execution; the stage numbers describe
   responsibilities, not a registration-before-return dependency.

### Authority at both ranking boundaries

`PAPERVAULT_SEARCH_AUTHORITY_PRIOR=1` enables both the local fusion prior and final
relevance blend; set it to `0` to disable. `PAPERVAULT_SEARCH_AUTHORITY_LAMBDA=0.25`
is passed as `authority_weight` to fusion and as λ to final ordering. The final
score is `1/(60 + relevance_rank) + λ/(60 + authority_rank)`.
Disabling the prior or setting λ to zero removes the added RRF terms. The base
relevance ordering still uses authority to break ties after score and external
rank evidence.

Authority combines **citation velocity** and a **three-layer, domain-aware venue/DOI
classifier**. Citation velocity is `citation_count / max(1, current_year - year)`;
missing/invalid years use raw citation counts. Classification combines normalized
exact/fuzzy venue names, recognized DOI prefixes, and fallback identifier/citation
evidence. Its multipliers are:

| Multiplier | Evidence |
| --- | --- |
| **1.5x** | Recognized leading/review venues, such as PRL, Nature, Space Science Reviews or Reviews of Geophysics, including known DOI routes. |
| **1.2x** | Recognized domain and methods venues, such as ApJ, JGR, A&A, MNRAS, JCP and major ML conferences. |
| **1.0x** | Neutral fallback for other identified or unclassified papers. |
| **0.5x** | Explicit low-tier/conference-abstract evidence, or a venue-only record with no DOI/arXiv/INSPIRE identity and fewer than five citations. |

Exact names across tiers precede fuzzy matching (threshold 0.88); low-tier evidence
wins before positive tiers, while a broad publisher DOI does not demote a recognized
1.5x venue. This is a ranking signal, not a claim that citation counts establish
canon or permission to bypass quality gates. Explicit `by_recency` / `by_importance`
hints bypass both authority blends and retain relevance-score-first ordering.
See [ADR-0008](docs/adr/0008-search-authority-venue-and-garbage-fencing.md) and
[evaluation knobs](docs/eval.md#retrieval-tuning-knobs-benchmark-arbitrated).

### Two-way quality fence

`registration_rejection` prevents new catalog pollution; `serving_excluded` protects
reads and acquisition from positively identified bad legacy rows. The latter checks
domain status, acquisition quarantine, explicitly excluded policy, and
`deterministic_garbage_reason`: conference abstracts, positively classified non-papers
or anonymous first authors. Missing legacy metadata alone is insufficient to exclude.
Aliases and identity rows remain durable, preventing rediscovery from allocating the
same junk again. Routine `JANITOR_PURGE` / `Library.purge()` deletion is retired.

Serving eligibility and acquisition eligibility are separate. For a `Paper`, eligible
retrieval alone does not authorize download: `state_v2.admission` also requires a
canonical nonalias identity, resolved correspondence, applicable policy, no quarantine,
and valid lease/budget evidence. `Paper` has no `eligible` property. A terminal
acquisition status alone does not hide valid existing evidence from serving.

Official ADS `ADS_SCAN` PDFs may defer identity verification to OCR, bound to the
inspected PDF hash. Publishing their text requires a positive post-OCR title/author
identity verdict; mismatch or unavailable verification retains evidence without
publishing it. Cancelling a fetch can withdraw queued executor work but cannot prove
a running synchronous thread has stopped. Registration shields and drains its writer
before propagating cancellation; campaign cancellation retains its permit until
executor drain and quarantines uncertain work.

## Knowledge engine and query evidence

The framework spine is **`lightrag-hku==1.5.7`**, with Neo4j graph storage and
PostgreSQL KV, vector and document-status stores. BGE-M3 embedding and the BGE
reranker run in process on the GPU. A domain pack supplies a closed entity ontology,
extraction guidance and examples. The exact patch pin and fail-loud private-hook
adapter guards follow [ADR-0005](docs/adr/0005-track-latest-lightrag-line-gated.md):
an upgrade needs a fresh port map, prompt gates, pause/deferral and recovery tests,
a throwaway ontology probe and a FAST recall comparison.

Adapters preserve LightRAG's extraction, merge, deletion and query reducers. The
1.5.7 port forwards batch ingress/owner arguments and deferred run context, handles
retirement rebuild reports and extraction-cache tuples, and materializes inherited
status updates under the namespace write fence. Raw-workspace conflict repair is
refused for catalog storage; it requires an isolated rebuild.

The query plane (S2) pins one catalog `serving_manifest` for the whole request:
anchor and facet retrieval, traversal, caches, aggregation and synthesis share one
generation. Reader pins protect dependencies; policy, input compatibility and fresh
evidence are rechecked against that same view before returning. Promotion cannot
switch an in-flight query to another generation. Multiquery overlaps decomposition
with anchor retrieval, drains cancelled decomposition and gates fan-out on coverage.
Evidence prioritizes `canon`, then `fulltext`, then `abstract_only`; textbooks and
explicit canon annotations establish canon status.

Production and eval share synthesis and the pure `build_outfeed` response builder.
`retrieved_papers` lists evidence candidates; `cited_papers` contains only exact
inline `[paper_key]` citations verified against that pool. Operator documents are
reported separately. Empty coverage skips synthesis; exclusively abstract evidence
can reach at most `"weak"` coverage. `query_receipt` exposes generation, variant,
coverage, retrieved/cited counts, reranker degradation and synthesis observations.
FAST `--no-synth` eval explicitly uses retrieved papers as its citation fallback.

## Catalog authority (P2)

`PAPERVAULT_CATALOG_AUTHORITY` selects `json` (the unset default) or `postgres`.
PostgreSQL requires an explicit `PAPERVAULT_CATALOG_DSN` and migrated schema
(`PAPERVAULT_CATALOG_SCHEMA=catalog` by default); it never falls back to JSON.
Library commits revision-guarded record deltas, audit and pending work atomically.
Stale conflicts surface as `catalog_conflict`; catalog deletion is refused.
Artifact bytes become durable before their catalog references.

Deployment #1 uses **`papervault_catalog`** with `PAPERVAULT_CATALOG_AUTHORITY=postgres`,
following the [authorized 2026-10-19 cutover](https://github.com/LeonJoeeee/papervault/issues/268#issuecomment-6257668738).
Ordinary PG saves do not rewrite `index.json`. The verified canonical JSON export
is a projection, generated by `papervault-export.timer` hourly at **:17** (host CST,
`OnCalendar=*:17`), or explicitly:

```bash
.venv/bin/python scripts/export_catalog.py --output <path> --verify
```

The [persistence decision](docs/adr/0010-catalog-persistence-cutover.md) and
[cutover record](docs/specs/2026-10-18-p2-readiness.md) define writer fencing and
rehearsal. After authoritative writes resume, recovery is forward-only; restoring an
old JSON export cannot roll back accepted catalog mutations. Deployment #1 retains
external Docker volumes `ks-pgdata` / `ks-neo4jdata` and container names `ks-postgres`
/ `ks-neo4j`; changing them requires a separately authorized migration that preserves
and verifies the original data.

## Acquisition v2

The six flags below default **off** and accept `1`, `true` or `yes`. Their contracts
are recorded in [ADR-0006](docs/adr/0006-acquisition-pipeline-v2-contracts.md) and the
[acquisition design](docs/specs/2026-10-12-acquisition-pipeline-v2.md).

| Flag | Contract |
| --- | --- |
| `PAPERVAULT_TERMINAL_V2` | Enforce retrieval/policy/merge/identity/quarantine admission immediately before ordinary dispatch. Legacy dimensions are projected on read and persisted on a state write; a status reset alone grants no policy permission. |
| `PAPERVAULT_IDENTITY_V2` | DOI → namespaced source ID → URL identity ladder, preserving CORE work/output namespaces. Correspondence may accept, abstain or reject one route; identity selection alone neither merges nor reorders download tiers. Domain markers project into policy; terminal v2 enforces this at ordinary dispatch. |
| `PAPERVAULT_REOPEN_V2` | Explicit bounded campaigns with durable pre-dispatch reservations, generation/lease claims, isolated staging and guarded publication. Campaign admission independently enforces identity, policy and quarantine. |
| `PAPERVAULT_POLITE_V2` | Process-shared API request windows, concurrency and cooldowns at the migrated request boundaries; every configured window must admit a send. |
| `PAPERVAULT_MERGE_V2` | Paired durable merge intent, stopped admission and verified executor/claim drain before canonical/alias consolidation; conservative restrictions and budgets survive the merge. |
| `PAPERVAULT_STOPRULES_V2` | Detect duplicate in-flight identity, exceeded quota windows and identity-mismatched publication; halt campaigns/cool scopes/quarantine as applicable. Admission refusals and rejected PDF tiers are not violations. |

Unknown policy denies new acquisition without an applicable scoped ruling. Existing
artifacts remain retained. Disabling a flag does not erase persisted claims, quarantine,
merge intents, budget consumption or campaign halts. Stop rules do not enforce the
unmeasured significant-regression term (`SIGNIFICANT_REGRESSION_FRACTION=0.20`).

### Campaigns, policy and recovery

Use a maintenance window with `papervault.service` stopped for write operations.
Standalone campaigns and rulings share an exclusive writer lock and refuse a running
service, unavailable service probe or incompatible active work. The orchestrator
owns live application. `reopen-campaign` requires `--dry-run` for a read-only
preview; `policy-ruling` previews by default and writes only with `--apply`:

```bash
python -m papervault.library.cli reopen-campaign --cap 20 --dry-run --json
# Apply the bounded campaign in the stopped-service window.
python -m papervault.library.cli reopen-campaign --cap 20 --json
python -m papervault.library.cli reopen-campaign --keys approved.keys --cap 50 --dry-run --json
python -m papervault.library.cli policy-ruling --keys approved.keys --action acquire --reason 'Owner approved recovery' --actor owner
# Apply the previewed ruling in the stopped-service window.
python -m papervault.library.cli policy-ruling --keys approved.keys --action acquire --reason 'Owner approved recovery' --actor owner --apply
```

Key files contain one citation key per line, with comments/duplicates ignored; an
empty file selects nothing. Accounting includes refused and cap-deferred keys.
Rulings are key-bound, revisioned and append-only. `acquire` (`override`) grants
permission for covered policy reasons; `exclude` (`restrict`) denies acquisition;
`clear` clears the ruling. None releases quarantine. PG commits record, ruling,
audit and pending work together; JSON-mode crash-intent replay requires the latest
key-bound revision, original record snapshot and matching covered reasons. Spent or
superseded grants cannot reopen work. `audit` download retries and `add --force-refresh`
record `operator_reopen` without granting policy permission; extraction retries retain
their attempt/defer reset behavior through named operator services.
`audit --retry-low-quality` and `audit --retry-extract-keys` use those extraction
services; download audit retries accept `--reason` and `--actor`.

Standing limits are **200 reservations per campaign** and **three starts per rolling
seven UTC days**, including interrupted starts. The fsynced append-only
`reopen-budget.jsonl` remains authoritative for consumption across restart/index
rollback. Reservation precedes dispatch; uncertainty never refunds or resends.
Results publish from `.reopen-staging/<lease-id>/` only under the exact generation,
lease, identity, policy, budget and changed-artifact guards. Expiry does not prove
termination: uncertain/dispatched restart work stays quarantined and charged.
Retain staging until drain/fencing is established.

The lifecycle CLI exposes:

| Command | Guarded operation |
| --- | --- |
| `claim-recover KEY [--lease-id ID]` | Refund only proven-unsent work; dispatched attempts need `--executor-proof`. |
| `release-quarantine KEY [--lease-id ID]` | Record explicit human disposition and fence the old generation without refund. |
| `campaign-halt ID [--drained]` | Stop new dispatch; finish only after outstanding reservations and claims settle. |
| `merge-consolidate CANONICAL ABSORBED` | Require merge v2, quiesce/drain both records, retain alias and artifacts. |
| `merge-restart CANONICAL` | Restart an expired paired wait under the existing fences; uncertain commits need separate recovery. |

All five require `--reason REASON --actor ACTOR` and a stopped service. Merge budgets
sum spent consumption, take the minimum applicable ceiling and advance generations;
a ledger/catalog disagreement fails closed. Ordinary retries never release expired
or uncertain merge fences.

### API politeness and envelopes

Built-in limits are Wiley **1/second and 30/600 seconds**, Semantic Scholar
**1/2 seconds**, arXiv API **1/3 seconds**, each with concurrency **1**. Crossref,
Unpaywall and OpenAlex are disabled under polite v2 until an explicitly authorized
`PAPERVAULT_POLITE_ENVELOPES` entry exists. This JSON accepts those three scopes only:

```json
{"crossref":{"analogue":"wiley","initial":[[1,1],[30,600]],"hard_cap":[[1,1],[60,600]],"authorization":"owner approval reference"}}
```

Each initial window corresponds to the chosen `wiley`, `semantic_scholar` or `arxiv`
analogue, with no larger burst, no shorter duration and at most half its rate. Caps
bound counts and durations. The example is an operator assumption requiring separate
authorization, not a statement of provider permission. Missing fields disable a
scope; malformed or changed live configuration fails closed.

The wrapper refuses a full window, occupied permit or cooldown without adding
waits/retries. Explicit `Ledger.increase` is bounded to +25% rate after 24 hours
and fresh completed 2xx evidence; there is no automatic ramp. A 429 or three timeouts
in 60 seconds starts a 60-second cooldown. Restart is not a supported quota reset.
Browser/content/redirect and arXiv page/PDF paths are outside this API migration.

## Identity correction and acquisition observability

Inspect URL-only records without modifying the vault:

```bash
python -m papervault.library.cli identity-backfill --cap 20 --json
```

Backfill checks URL patterns, a citation-metadata landing fetch (including redirects),
then exact source metadata. DOI registry/arXiv evidence must pass title/author
correspondence before adoption. Missing, blocked or contradictory evidence causes
abstention; existing identifiers/collisions are retained. In an authorized stopped-service
window, `--apply --acquire` persists verified identities and runs the ordinary cascade
only for newly recovered rows without assets. `--keys FILE` restricts selection;
required `--cap` is 1–50 and bounds inspection/acquisition.

The standing correction pipeline uses the same verification path with acquisition
**disabled**. Notify it after credentials or source evidence arrive:

```bash
python -m papervault.library.cli correction-event credential_arrived --source ads --cap 20 --apply
python -m papervault.library.cli correction-event source_repaired --source core --cap 10 --apply
python -m papervault.library.cli correction-event identifiers_arrived --source ads --keys affected.keys --cap 10 --apply
python -m papervault.library.cli correction-sweep --apply
python -m papervault.library.cli correction-report --kind conflicts
python -m papervault.library.cli correction-report --kind manifests --manifest ID
python -m papervault.library.cli correction-adjudicate results.json --apply
python -m papervault.library.cli correction-report --month YYYY-MM
python -m papervault.library.cli correction-report --kind routes
```

Events intersect explicit keys with the source cohort; `identifiers_arrived` requires
keys. Each pass requires a 1–50 cap. The monthly task has a durable 30-day floor,
rotating key/cohort cursors, default 20 per cohort (maximum 50), and a total maximum
of 100; `--per-cohort-cap` / `--total-cap` may lower these values. The pipeline
records DOI recoveries and defers lower-ladder claims; it neither downloads assets
nor resets terminal states. DOI collisions wait for V2 alias consolidation.

`correction-ledger.json` retains conflicts, route evidence, scheduling, investigations
and monthly reports. At ten pending subject papers it creates a manifest for an
orchestrator-dispatched Codex `gpt-6-luna` / `max` investigation; the library launches
no agent. Adjudication imports `manifest_id` and `results` entries (`conflict_id`,
`status: resolved|needs_human`, `decision`, `evidence`). Resolved entries need evidence;
replay is idempotent. Decisions do not authorize field changes. Reports are read-only;
maintenance writes refuse a running service.

Key-scoped ADS events may import `--routes FILE`: at most three `{key, url,
source_identifier, provenance}` routes per paper, bound to the stored bibcode with
timezone-aware `checked_at`, `verification: "remote_pdf"` and survey evidence.
Route reports supply scoped reopening evidence.

The manifest also retains download outcomes and scalar `download_telemetry` spans:
queue wait, network admission, executor/cascade, tier/group/member, verification,
publication and actual drain. Correlate `run_id`, key, UTC times and span IDs;
partial attempts remain censored. `status=ok` means control flow returned, not that
acquisition succeeded. Whole-cascade permits can contain multiple member requests;
browser-call spans do not measure process memory. Validate provider limits with
request counts/status/rate headers and matching host measurements, not semaphore
counts alone. Stop-rule correlation fields preserve unknown values as null.

## Catalog knowledge publication (P4)

`PAPERVAULT_KNOWLEDGE_JOBS=legacy` is the default. Selecting `postgres` requires
PostgreSQL library authority plus a **rehearsed, committed authority handoff**;
a configuration edit cannot transfer ownership. Follow the
[publication decision](docs/adr/0011-catalog-knowledge-publication.md) and
[P4 procedure](docs/specs/2026-10-20-p4-knowledge-handoff.md). This documentation
records implementation contracts, not a live P4 activation.

Catalog intake captures exact immutable input bytes, metadata and versioned recipes.
Jobs, attempts, submissions, expected inventories, sealed receipts and generation
manifests have separate identities. Private LightRAG deltas overlay one pinned
aggregate; adapters resolve latest replacement rows and tombstones before retrieval
or traversal. Unpublished outputs cannot alter the serving view. Publication
atomically checks current inputs, eligibility, aggregate base and output evidence
before recording lineage, selections, audit and downstream work. Candidate publication
does not promote production. Unknown outcomes retain charges and recovery pins.

P5a extends this boundary to extraction: `extracts/inputs/<sha256>` and
`extracts/versions/<completion_id>/{output.md,completion.json}` retain consumed bytes
and immutable receipts. Conventional markdown paths are compatibility projections.
Legacy/JSON extraction may retain a durable pending bundle without catalog coverage;
committed catalog extraction uses `publish_derivation` to register receipt,
derivation, audit and distillation outbox together. Recovery delivers the original
evidence without re-extracting or inventing historical recipes. Staged reopening
still needs its guarded claim/writeback transaction.

Coverage requires selected compatible publications and fresh positive verification
of every required output (default observation age **300 seconds**). Backend
`processed` status alone is insufficient. The read-only
`scripts/coverage_report.py` separates this P4 ruler from legacy coverage. Physical
retirement verifies document discovery/contribution indexes, reducer rebuilds and
graph/vector consistency before sealing; missing historical evidence needs an
isolated rebuild. Corpus-scale layered-query latency and cumulative manifest
metadata growth remain activation qualification concerns.

## Coordinated recovery

P5b uses `scripts/snapshot_derivatives.py` and `scripts/rebuild_derivatives.py` with
explicit roots/connection identities, dotenv discovery disabled and dry-run defaults.
Operational roots are `/data/p5b-recovery` and `/data/p5b-drills/<drill-id>` on the
owner's RAID. The orchestrator owns weekly scheduling and production drills in the
monthly reboot window (at most two hours); [staging evidence](docs/baselines.md)
does not establish live scheduling or full-system recovery qualification.

Drain library, graph and administrative backend writers before capture. Exclusive
vault/knowledge leases, catalog/backend writer barriers and PostgreSQL table locks
bind one checkpoint. A complete set contains the catalog schema/index inventory,
rows/sequences/epoch/audit boundary, entire vault and operational ledgers, immutable
extraction/knowledge inputs, recipes/receipts, PostgreSQL KV/cache/status/vector
materializations, and Neo4j graph/schema. Hashes, lengths and dependency closure are
verified before registration. Independently timed dumps, the nightly catalog-only
dump and hourly JSON export do not constitute this set.

Use an explicit source JSON configuration; credentials name environment variables:

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

Supply actual installed table names, including embedding-model suffixes. Missing
roles or incompatible schema refuse capture; source migration is explicit (knowledge
schema version 7 for P5c).

```bash
.venv/bin/python scripts/snapshot_derivatives.py --config source.json --root /data/p5b-recovery
.venv/bin/python scripts/snapshot_derivatives.py --config source.json --root /data/p5b-recovery --due --apply
.venv/bin/python scripts/snapshot_derivatives.py --verify /data/p5b-recovery/<snapshot-id>
```

`--due` requires a new complete set after seven days. Retain four verified complete
sets; failed/sample exports cannot displace them, and only-copy protections may
retain more. Pruning is limited to verified owned package copies. Plans include
retained sets, staging and reservations; insufficient physical headroom refuses
capture. The **2 TB** derivative soft cap / **80%** threshold produces alert evidence
for snapshots, never emergency source deletion. Capture/restore has a two-hour
deadline; export memory cost and full-corpus timings require separate qualification.

Restore/replay use distinct isolated targets. Catalog schemas must be absent
`catalog_recovery_*` in an explicitly named test/rehearsal database; derivative
schemas must be absent `recovery_backend_*` with matching extensions. Neo4j must be
an empty distinct endpoint/database marked `"disposable": true`. Resume requires the
same operation/package binding and verifies completed materializations and indexes.

```bash
.venv/bin/python scripts/rebuild_derivatives.py --mode restore --package <set> --root <empty-target> --generation restored
.venv/bin/python scripts/rebuild_derivatives.py --mode restore --package <set> --root <empty-target> --generation restored --config target.json --apply
.venv/bin/python scripts/rebuild_derivatives.py --mode replay --package <set> --root <other-empty-target> --generation replayed --source-generation <source-generation> --config replay-target.json --apply
```

Restore preserves original content recipes/receipts while installing a new epoch
and writer fence; copied dispatch remains disabled. Replay executes selected original
inputs/recipes through new attempts and receipts without importing runnable history.
Its unit-level execution-quarantine reset is recorded and limited to isolated samples;
record policy/quarantine still applies. Physical replay needs `"physical_replay": true`,
exact recipes/providers/models and explicit isolated backends. Its PG DSN names one
IP (matching `hostaddr` if present), with runtime tables verified in the owned schema.
Metadata replay needs no graph/model calls. P5b replay generations are permanently
sample/non-promotable. These tools neither change live authority nor select serving.

Weekly cadence is not a measured RPO; sample timings are not full RTO. Same-host
RAID copies share the host failure domain and do not provide off-host disaster recovery.

## Disposable extracts and isolated maintenance

P5c's `scripts/maintain_derivatives.py` handles retention, bounded GC and isolated
candidate rebuild/promotion. It requires explicit DSN/schema/vault identities,
disables dotenv discovery and defaults to a read-only plan.

Every extract starts **protected**. Only an explicitly disposable immutable
`output.md` may be collected, **90 days after losing its last accepted head/record
selection**. Reselection cancels the clock; file age is irrelevant. Pre-P5c rows
start protected and acquire a conservative clock on verified vault binding. Source
inputs, receipts, current selections, live generations, jobs and reader/recovery pins
remain protected. Reader pins expire within two hours; job/recovery pins need a
reviewed disposition.

GC commits a permanent deletion intent under the same writer lock as reference
admission before filesystem IO. It verifies exact path/hash/size and a regular,
single-link file through directory descriptors, rejects symlinks and unlinks only
the named output. Interrupted deletion resumes the same intent under a fresh lease.
Sources/receipts and pre-GC snapshot protections remain intact.

```bash
.venv/bin/python scripts/maintain_derivatives.py inspect --dsn-env MAINTENANCE_CATALOG_DSN --catalog-schema <schema> --vault <vault> --operation-id inspect
.venv/bin/python scripts/maintain_derivatives.py gc --dsn-env MAINTENANCE_CATALOG_DSN --catalog-schema <schema> --vault <vault> --payload gc.json --operation-id gc-batch
.venv/bin/python scripts/maintain_derivatives.py gc --dsn-env MAINTENANCE_CATALOG_DSN --catalog-schema <schema> --vault <vault> --payload gc.json --operation-id gc-batch --owner <operator> --window-id <window> --apply
```

`inspect` exposes authority, retention revisions and eligibility. Other actions
consume explicit JSON payloads:

| Action | Required payload |
| --- | --- |
| `window` | `window_id`, `kind: "monthly_reboot"`, timezone-qualified `starts_at`/`ends_at` (≤7,200 seconds), `operator_nonce`, authority `expected_revision` |
| `disposition` | `completion_id`, boolean `disposable`, `reason`, `operator_nonce`, retention `expected_revision` |
| `gc` | `completion_ids`: 1–100 distinct explicit IDs |
| `fork` | New `generation_id`, active `source_generation_id`, `expected_manifest_revision`, `expected_requirements_revision` |
| `promote` | Candidate `generation_id`, `expected_manifest_revision`, `expected_requirements_revision`, `expected_serving_revision` |

`inspect` may take `generation_id`. GC/fork/promote apply needs `--owner` and
`--window-id`; a fenced lease and exact revisions are rechecked before commit.
The module CLI also exposes `extract-retention`, `extract-pin`, `release-extract-pin`
and `extract-gc` with the same explicit target/operation/apply boundary; use `--help`
for revision, nonce and disposition arguments. `extract-gc --completion-id ID` may
repeat explicit identities. Preserve operation IDs/payloads across lost responses.

Fork queues a fresh unselected candidate. Execution is funded and bounded to the
recorded monthly reboot window; deadline cancellation retains uncertain remote
outputs and charges. Promotion separately requires a complete compatible manifest,
fresh evidence and exact current revisions. Sample generations cannot promote.

### Capacity accounting and usage census

New catalog intake/execution reserves physical capacity under
`<knowledge_root>/maintenance/capacity`. The **2,000,000,000,000-byte (2 TB)** cap and
**80% (1.6 TB)** alert include filesystem derivatives, reservations and external
backend/retained-set usage. Unfunded work stops without deleting protected assets;
sealed-receipt recovery/publication remains available. The default estimate is
`PAPERVAULT_CATALOG_EXECUTION_BYTES=1073741824` (1 GiB per job), including scratch
and output. Headroom covers outstanding output plus twice the new estimate.
Unknown outcomes remain charged across restart until verified settlement; completed
external growth stays charged until a census begun after settlement covers it.

`PAPERVAULT_DERIVATIVE_USAGE_FILE` names operator-owned mode-0600 JSON refreshed
within **300 seconds**: `scope: {vault_root, knowledge_root}`, timezone-qualified
`measured_at` and nonnegative `backend_bytes`. Include external PG/Neo4j and retained
sets outside those roots; runtime adds catalog relation sizes. Missing, stale or
mismatched evidence blocks even metadata intake. A `metadata_sample: true` census
cannot fund physical execution. Estimates do not establish an exact disk quota or
corpus-scale performance qualification.

Reuse the snapshot configuration with `extra_derivative_roots` listing every
external derivative/recovery root (an explicit empty list affirms none). Both PG
and Neo4j measurement adapters are required unless a metadata-only fixture is
proved. The output parent must exist; preview prints measurements, apply atomically
publishes the protected census:

```bash
.venv/bin/python scripts/measure_derivative_usage.py --config source-with-capacity-roots.json --output <usage.json>
.venv/bin/python scripts/measure_derivative_usage.py --config source-with-capacity-roots.json --output <usage.json> --apply
```

Isolated physical replay uses the same capacity gate with a census scoped to
`<target>/vault` and `<target>/knowledge`; metadata replay measures its local files
and catalog synchronously. `scripts/drill_derivative_maintenance.py` exercises the
protocol in disposable stores. The [P5c baseline](docs/baselines.md#p5c-isolated-maintenance-rehearsal--2026-10-21)
records synthetic metadata timings, not production full-text rebuild cost or RTO.

## Development and evaluation

Use a worktree-local environment; the primary checkout's editable install tests
the primary tree. Repository records are English. Install and run the required checks:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev
.venv/bin/pytest -m 'not integration' -q
.venv/bin/ruff check src tests
```

[Evaluation](docs/eval.md) explains gold sets, reference-machine runs and benchmark
arbitration. Raw eval outputs contain paper text and stay local/untracked under
`PAPERVAULT_EVAL_DIR`; publish summary evidence on the issue or PR. CI tests scoring
logic without GPU/DB/network and does not replace reference-hardware retrieval gates.
[Baselines](docs/baselines.md) separates dated legacy rulers, post-refactor measurements
and isolated recovery/maintenance rehearsals.

## Releases

papervault is consumed **by pin** ([ADR-0001](docs/adr/0001-public-repo-is-upstream.md)):
deployments track an annotated git tag (`vX.Y.Z[-stage]`), never `main`.

The repo carries **one version**, and every change PR bumps it per the semver call
stated in its description. The declared fields move together: `pyproject.toml`
`[project].version`, the `papervault` entry in `uv.lock` (refresh with `uv lock`),
`plugin/.claude-plugin/plugin.json`, `plugin/plugin.json`, and the `papervault`
entry in `.claude-plugin/marketplace.json`. `tests/test_plugin_version_sync.py`
fails when they drift. This retrieval/routing/prompt change uses the issue-requested
patch version **0.10.13**.

A release is a tag naming the version already on `main` (`vX.Y.Z`, optionally
`-stage`); the release commit changes no version field. CI builds wheel/sdist on
every commit. The historical `v0.1.6-beta` tag predates the one-version rule.

## License

MIT — see [LICENSE](LICENSE).
