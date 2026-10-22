# papervault — Architecture

> Shared baseline for all parallel work. Read before any task.
> Changing anything here = public merge + human approval + an ADR.

The owner-ratified [papervault constitution v2.1-rc2](specs/2026-10-18-papervault-constitution.md) and [refactor plan v0.3](specs/2026-10-18-papervault-refactor-plan.md) (2026-10-18) govern the redesign, with the constitution taking precedence. [ADR-0007](adr/0007-papervault-constitution.md) records their rationale, independent review trail, owner clarifications, authorization boundary, and deferred mechanisms. This overview describes the existing implementation; future changes and implementation slices must conform to the governing documents.

## Context
papervault is a self-hosted literature + knowledge layer for coding agents. It runs as ONE MCP server process exposing three tools — `search_papers` (discover + auto-ingest literature), `get_paper` (metadata + full-text lookup), `query` (cited synthesis from a knowledge graph) — with an independently managed MinerU OCR service plus Neo4j and Postgres. Agents connect over MCP (Claude Code and Antigravity CLI/IDE via the bundled plugin; Codex CLI and any MCP client via documented config). All LLM calls route to one operator-supplied OpenAI-compatible endpoint. Hardware floor: one 24 GB-class NVIDIA GPU on Linux (PRD constraints).

## Structure
One Python package (`papervault`), one server process, subsystems as modules:

- **library/** — search serving and literature acquisition. `library/mcp/server.py` delegates to the six-stage `services/search_pipeline.py`; `services/registration.py` separately admits registration and acquisition work. Source adapters cover arXiv, ADS, Semantic Scholar, OpenAlex, INSPIRE and CORE. The LLM ingest gate (domain-pack rubric), tiered download cascade (shadow-library and anti-bot tiers flag-gated, default off — ADR-0004), extract queue driving the OCR server, catalog persistence (Postgres authority + canonical `index.json` export) and per-paper extracts remain here. Policy, recovery and extraction maintenance use named operator services. Library owns all writes to the vault.
- **knowledge/** — knowledge distillation + serving. `ingest/` captures and distills evidence, `scheduler/` admits work, `store/` owns backend adapters and catalog serving, and `query/` retrieves and synthesizes. Legacy rounds read the vault and reconcile the ingest ledger; the P4 catalog implementation instead captures immutable inputs, admits catalog jobs and publishes verified isolated LightRAG outputs. LightRAG uses Neo4j graph and Postgres KV/vector/doc-status stores; in-process GPU embed + rerank (BGE-M3, bge-reranker). `query/aquery.py` pins catalog serving across a whole query and shares response assembly with eval.
- **mcp/** — the single endpoint, three tools, progress heartbeat (long queries survive client idle timeouts).
- **domain pack** (config, not code) — search-gate rubric, entity ontology, extraction few-shots, instruction fragments. Space physics is the factory default and worked sample (ADR-0003).
- **eval/** — the bundled harness (recall@served, hallucination rate, trap gates) + method notes. Retrieval/pipeline quality changes are benchmark-arbitrated.
- **plugin shell** — the minimal installable `plugin/` subdir: Claude Code uses `plugin/.claude-plugin/plugin.json` + `plugin/.mcp.json`, routed to by the repo-root `.claude-plugin/marketplace.json`; Antigravity (`agy`) CLI/IDE uses native `plugin/plugin.json` + `plugin/mcp_config.json`. Both register the same HTTP MCP endpoint (`http://127.0.0.1:8080/mcp` by default); only `plugin/` ships on install. [Agent setup](INSTALL.md#connecting-an-agent) covers native AGY installation (`agy plugin install ./plugin`) and direct MCP configuration, also used by Codex CLI. No skills shipped in v1. Thin by rule; per-agent adapters never hold logic.

Boundaries: library owns vault artifacts and registration. The legacy library→knowledge handoff reads `index.json` and paper artifacts. P4 reads catalog records and captures exact immutable artifact/metadata revisions before execution; the hourly JSON remains a projection. Reader and writer are versioned together inside the package. External processes: shared MinerU OCR service (official `mineru-api` task HTTP API via core `httpx`; service-owned FIFO queue, model preload, 20-minute idle GPU unload and recovery; no library pre-flight, with submit/poll/result requests determining reachability), Neo4j + Postgres (docker-compose), the operator LLM endpoint (OpenAI-compatible).

## Search, query and lifecycle contracts

**S1 search plane (#251).** `library/services/search_pipeline.py` separates intent,
plan, retrieval, fusion, return gate and serving into six stages with request-local
snapshots and stage traces. Serving submits independently selected registration
requests without waiting for admission, execution or acquisition success. Fusion
and `preview_candidates` share canonical serving deduplication, fold aliases and
bibliographic duplicates, and retain external rank evidence on local records.
Named-entity slot reservation preserves accepted witnesses within the result cap.
In `by_relevance` ordering, `PAPERVAULT_SEARCH_AUTHORITY_PRIOR=1` is the default:
a pool-only, age-normalized citation-authority RRF blend with
`PAPERVAULT_SEARCH_AUTHORITY_LAMBDA=0.25`. Set the prior to `0` to disable it;
explicit recency/importance hints bypass the blend, and it never admits a gate reject.

All six search sources use structured `BackendDegraded` failures under constitution
Art. 6.4: `code`, `scope` (`item` or `service`), `retryability` (`permanent`,
`transient_with_delay`, `transient_no_delay`) and `retry_after_s`. Missing or invalid
delays default to five seconds only for delayed transient failures; the other classes
carry no delay. Genuine zero-hit responses remain distinct from unavailable,
unconfigured or malformed sources, and receipts omit provider bodies and error text.
Official ADS `ADS_SCAN` (and validated official scanned-PDF routes) may pass through
download with identity verification deferred to OCR, bound to the inspected PDF hash.
`library/extract.py` requires a positive post-OCR title/author identity verdict before
publishing the completion; mismatch or unavailable/malformed verification retains the
rejected evidence and withholds publication.

**S2 query plane (#252).** Catalog serving pins one single-generation
`serving_manifest` for the full `query()` execution: anchor and facet retrieval,
graph traversal, caches, aggregation and synthesis all share that view. Reader pins
protect its dependencies; eligibility, compatible inputs and fresh evidence are
revalidated against that exact manifest before returning, without switching to a
newly promoted generation. `query/multiquery.py` overlaps `_decompose()` with anchor
retrieval, cancels and drains decomposition if the anchor rejects fusion, and admits
fan-out estimation only after the coverage gate. Fused evidence prioritizes
`canon`, then `fulltext`, then `abstract_only` before the served-window cut; textbooks
and explicit canon annotations establish canon status, never citation counts alone.

Production and `eval/run_eval.py` share `query/aquery.py`'s synthesis and pure
`build_outfeed` response builder. `retrieved_papers` records the evidence candidates;
`cited_papers` contains only exact inline `[paper_key]` citations verified against
that pool, with operator documents reported separately. FAST eval with `--no-synth`
uses retrieved candidates as its explicit citation fallback. Empty coverage skips
synthesis; exclusively `abstract_only` evidence can reach at most `"weak"`, even with
a rich graph signal. `query_receipt` reports generation, variant, coverage, retrieved
and cited counts, reranker degradation and synthesis observations.

**S3 lifecycle completion (#253).** Scoped policy rulings commit record changes,
append-only ruling history, audit and pending work in one PostgreSQL transaction;
the JSON correction ledger is a compatibility projection. JSON-mode crash-intent
replay checks the latest key-bound revision, original record snapshot and covered
policy reasons, so spent or superseded grants cannot reopen acquisition. Library CLI
operators `claim-recover`, `release-quarantine`, `campaign-halt`, `merge-consolidate`
and `merge-restart` preserve lease, budget, policy and drain fences and require actor
and reason in a stopped-service maintenance window. Extraction maintenance
(`audit --retry-low-quality`, `audit --retry-extract-keys`) routes through named
operator services and retains immutable extraction evidence. The routine
`JANITOR_PURGE`/`purge()` route is retired: reconciliation does not delete research
assets; `Library.purge()` refuses destructive record/artifact deletion in both
PostgreSQL and standalone JSON modes.

## Catalog, publication and maintenance

Since the [owner-approved live cutover on 2026-10-19](https://github.com/LeonJoeeee/papervault/issues/268#issuecomment-6257668738),
deployment #1 uses PostgreSQL database **`papervault_catalog` as catalog authority**
with `PAPERVAULT_CATALOG_AUTHORITY=postgres`. Library's existing working set reads
catalog records and commits revision-guarded delta batches with audit/pending work.
`index.json` is the **canonical export**, generated separately with verification by
the user unit `papervault-export.timer` **hourly at :17** (`OnCalendar=*:17`, host CST).
It is not a second authority and is not rewritten by ordinary PG saves. Configuration
still defaults to JSON when the authority setting is unset; the live deployment
explicitly selects Postgres. PG deletion is refused in P2. The
[catalog persistence ADR](adr/DRAFT-catalog-persistence-cutover.md) records commit
semantics, completion guards and rehearsal fencing; the
[execution record and historical rehearsal runbook](specs/2026-10-18-p2-readiness.md)
record the live sequence, operational timers and closed rollback boundary: recovery
after the writes-resumed marker is forward-only.

The [P4 knowledge publication decision](adr/DRAFT-catalog-knowledge-publication.md) adds catalog-owned jobs, attempts, receipts, expected inventories, generation manifests, selections and nonauthoritative observations. `PAPERVAULT_KNOWLEDGE_JOBS` defaults to `legacy`; selecting `postgres` requires a separately rehearsed and committed authority handoff, not just configuration. Isolated attempt deltas preserve LightRAG's existing reducers while serving pins a catalog-selected logical aggregate before retrieval and traversal. The [P4 procedure](specs/2026-10-20-p4-knowledge-handoff.md) defines the capture, writer fences, dedicated database/backend checks and forward-only rollback. This code change does not record a live activation.

P5a adds extraction evidence to that publication boundary. MinerU and Firecrawl completions retain exact consumed inputs under `extracts/inputs/` and immutable output/receipt bundles under `extracts/versions/<completion_id>/`. The library's conventional markdown file is a compatibility projection. In legacy or standalone JSON mode, the bundle is durable pending registration and grants no catalog coverage. Under committed catalog authority, `publish_derivation` atomically registers the extraction receipt, derivation, audit and distillation outbox before success. Recovery delivers the original evidence without another transformation or execution attempt. Reopen staging retains refused outputs and requires its existing guarded writeback before selection; merge adoption retains the original extraction lineage.

P5b coordinates verified recovery sets without changing live authority. P5c adds explicitly disposable extract retention and isolated maintenance. Supersession starts a ninety-day clock; default protection, current selections and reader/job/recovery references prevent deletion. A catalog writer lock serializes new references with permanent deletion tombstones, and fenced monthly leases bound resumable filesystem deletion. Source inputs and receipts remain immutable. Rebuilds fork into unselected candidates; complete compatible manifests and exact revision checks gate promotion. Shared physical reservations and fresh external usage accounting block unfunded intake/execution at the 2 TB derivative soft cap, with an 80% alert and no emergency asset deletion. These tools do not activate catalog scheduling or claim recovery from host loss. The [operator workflow](../README.md#disposable-extracts-and-isolated-maintenance) specifies the explicit targets and retained evidence.

Deployment #1 retains the legacy container names `ks-postgres` and `ks-neo4j`; production data lives in the existing Docker volumes `ks-pgdata` and `ks-neo4jdata`, declared external in `deploy/docker-compose.yml`. A future move off these names requires a maintenance window: stop PaperVault and both stores, copy each volume to its new name, update Compose, bring the stores up, and verify data and connectivity before resuming PaperVault. Preserve the original volumes for rollback: stop the replacement stores, restore the original Compose names and volumes, and bring the original stores up before resuming PaperVault.

## Key quality goals
1. **Measured, not claimed** — the eval harness is a product component; risky quality changes merge only with benchmark arbitration.
2. **Single-GPU budget** — resident models + OCR co-tenant fit a published 24 GB VRAM envelope under full co-load, zero OOM.
3. **Zero-babysit ingestion** — queues and scheduler rounds are self-healing; the corpus compounds unattended (search → queryable in ~tens of minutes).
4. **One way to run** — a single profile; no fallbacks or degraded modes; failures are explicit and diagnosable.
5. **MCP is the only compatibility surface** — everything of value lives at or below the tool contract; agent-specific shells stay thin.

Decisions and their reasons: `docs/adr/`, including [independent MinerU ownership](adr/DRAFT-independent-mineru-service.md).
