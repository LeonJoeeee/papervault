# papervault — Architecture

> Shared baseline for all parallel work. Read before any task.
> Changing anything here = public merge + human approval + an ADR.

The owner-ratified [papervault constitution v2.1-rc2](specs/2026-10-18-papervault-constitution.md) and [refactor plan v0.3](specs/2026-10-18-papervault-refactor-plan.md) (2026-10-18) govern the redesign, with the constitution taking precedence. [ADR-0007](adr/0007-papervault-constitution.md) records their rationale, independent review trail, owner clarifications, authorization boundary, and deferred mechanisms. This overview describes the existing implementation; future changes and implementation slices must conform to the governing documents.

## Context
papervault is a self-hosted literature + knowledge layer for coding agents. It runs as ONE MCP server process exposing three tools — `search_papers` (discover + auto-ingest literature), `get_paper` (metadata + full-text lookup), `query` (cited synthesis from a knowledge graph) — with an independently managed MinerU OCR service plus Neo4j and Postgres. Agents connect over MCP (Claude Code via the bundled plugin; any MCP client via documented config). All LLM calls route to one operator-supplied OpenAI-compatible endpoint. Hardware floor: one 24 GB-class NVIDIA GPU on Linux (PRD constraints).

## Structure
One Python package (`papervault`), one server process, subsystems as modules:

- **library/** — literature acquisition. Intent-parsed multi-source search (arXiv, ADS, Semantic Scholar, OpenAlex, INSPIRE, CORE), LLM ingest gate (domain-pack rubric), tiered download cascade (shadow-library and anti-bot tiers flag-gated, default off — ADR-0004), extract queue driving the OCR server, catalog persistence (Postgres authority + canonical `index.json` export) and per-paper extracts on disk. Owns all writes to the vault.
- **knowledge/** — knowledge distillation + serving. Legacy rounds read the vault and reconcile the ingest ledger; the P4 catalog implementation instead captures immutable inputs, admits catalog jobs and publishes verified isolated LightRAG outputs. LightRAG uses Neo4j graph and Postgres KV/vector/doc-status stores; in-process GPU embed + rerank (BGE-M3, bge-reranker); the query pipeline decomposes, retrieves, reranks and produces cited synthesis.
- **mcp/** — the single endpoint, three tools, progress heartbeat (long queries survive client idle timeouts).
- **domain pack** (config, not code) — search-gate rubric, entity ontology, extraction few-shots, instruction fragments. Space physics is the factory default and worked sample (ADR-0003).
- **eval/** — the bundled harness (recall@served, hallucination rate, trap gates) + method notes. Retrieval/pipeline quality changes are benchmark-arbitrated.
- **plugin shell** — the minimal installable `plugin/` subdir (`plugin/.claude-plugin/plugin.json` + `plugin/.mcp.json` MCP wiring), routed to by the repo-root `.claude-plugin/marketplace.json`; only `plugin/` ships on install. No skills shipped in v1. Thin by rule; per-agent adapters never hold logic.

Boundaries: library owns vault artifacts and registration. The legacy library→knowledge handoff reads `index.json` and paper artifacts. P4 reads catalog records and captures exact immutable artifact/metadata revisions before execution; the hourly JSON remains a projection. Reader and writer are versioned together inside the package. External processes: MinerU OCR service (HTTP through its gate; service-owned stop/start and recovery; no library pre-flight, with workload waking a stopped backend and client discovery/parse errors determining reachability), Neo4j + Postgres (docker-compose), the operator LLM endpoint (OpenAI-compatible).

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

P5a adds extraction evidence to that publication boundary. MinerU and Firecrawl completions retain exact consumed inputs under `extracts/inputs/` and immutable output/receipt bundles under `extracts/versions/<completion_id>/`. The library's conventional markdown file is a compatibility projection. In legacy or standalone JSON mode, the bundle is durable pending registration and grants no catalog coverage. Under committed catalog authority, `publish_derivation` atomically registers the extraction receipt, derivation, audit and distillation outbox before success. Recovery delivers the original evidence without another transformation or execution attempt. Reopen staging retains refused outputs and requires its existing guarded writeback before selection; merge adoption retains the original extraction lineage. Snapshot/restore, GC and retirement remain outside this slice.

Deployment #1 retains the legacy container names `ks-postgres` and `ks-neo4j`; production data lives in the existing Docker volumes `ks-pgdata` and `ks-neo4jdata`, declared external in `deploy/docker-compose.yml`. A future move off these names requires a maintenance window: stop PaperVault and both stores, copy each volume to its new name, update Compose, bring the stores up, and verify data and connectivity before resuming PaperVault. Preserve the original volumes for rollback: stop the replacement stores, restore the original Compose names and volumes, and bring the original stores up before resuming PaperVault.

## Key quality goals
1. **Measured, not claimed** — the eval harness is a product component; risky quality changes merge only with benchmark arbitration.
2. **Single-GPU budget** — resident models + OCR co-tenant fit a published 24 GB VRAM envelope under full co-load, zero OOM.
3. **Zero-babysit ingestion** — queues and scheduler rounds are self-healing; the corpus compounds unattended (search → queryable in ~tens of minutes).
4. **One way to run** — a single profile; no fallbacks or degraded modes; failures are explicit and diagnosable.
5. **MCP is the only compatibility surface** — everything of value lives at or below the tool contract; agent-specific shells stay thin.

Decisions and their reasons: `docs/adr/`, including [independent MinerU ownership](adr/DRAFT-independent-mineru-service.md).
