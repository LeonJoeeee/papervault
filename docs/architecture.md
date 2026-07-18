# papervault — Architecture

> Shared baseline for all parallel work. Read before any task.
> Changing anything here = public merge + human approval + an ADR.

## Context
papervault is a self-hosted literature + knowledge layer for coding agents. It runs as ONE MCP server process exposing three tools — `search_papers` (discover + auto-ingest literature), `get_paper` (metadata + full-text lookup), `query` (cited synthesis from a knowledge graph) — plus the infrastructure it owns: a MinerU vLLM OCR server, Neo4j, and Postgres. Agents connect over MCP (Claude Code via the bundled plugin; any MCP client via documented config). All LLM calls route to one operator-supplied OpenAI-compatible endpoint. Hardware floor: one 24 GB-class NVIDIA GPU on Linux (PRD constraints).

## Structure
One Python package (`papervault`), one server process, subsystems as modules:

- **library/** — literature acquisition. Intent-parsed multi-source search (arXiv, ADS, Semantic Scholar, OpenAlex, INSPIRE, CORE), LLM ingest gate (domain-pack rubric), tiered download cascade (shadow-library and anti-bot tiers flag-gated, default off — ADR-0004), extract queue driving the OCR server, vault persistence (`index.json` + per-paper extracts on disk). Owns all writes to the vault.
- **knowledge/** — knowledge distillation + serving. Scheduler rounds (reconcile → diff → distill) reading the vault; LightRAG graph store (Neo4j graph, Postgres KV/vector/doc-status); in-process GPU embed + rerank (BGE-M3, bge-reranker); query pipeline (decompose → retrieve + rerank → cited synthesis); ingest ledger.
- **mcp/** — the single endpoint, three tools, progress heartbeat (long queries survive client idle timeouts).
- **domain pack** (config, not code) — search-gate rubric, entity ontology, extraction few-shots, instruction fragments. Space physics is the factory default and worked sample (ADR-0003).
- **eval/** — the bundled harness (recall@served, hallucination rate, trap gates) + method notes. Retrieval/pipeline quality changes are benchmark-arbitrated.
- **plugin shell** — the minimal installable `plugin/` subdir (`plugin/.claude-plugin/plugin.json` + `plugin/.mcp.json` MCP wiring), routed to by the repo-root `.claude-plugin/marketplace.json`; only `plugin/` ships on install. No skills shipped in v1. Thin by rule; per-agent adapters never hold logic.

Boundaries: the library→knowledge hand-off is the on-disk vault — single writer (library), one reader (knowledge scheduler) — with reader and writer versioned together inside the package. External processes: OCR server (HTTP; lifecycle managed by library), Neo4j + Postgres (docker-compose), the operator LLM endpoint (OpenAI-compatible).

## Key quality goals
1. **Measured, not claimed** — the eval harness is a product component; risky quality changes merge only with benchmark arbitration.
2. **Single-GPU budget** — resident models + OCR co-tenant fit a published 24 GB VRAM envelope under full co-load, zero OOM.
3. **Zero-babysit ingestion** — queues and scheduler rounds are self-healing; the corpus compounds unattended (search → queryable in ~tens of minutes).
4. **One way to run** — a single profile; no fallbacks or degraded modes; failures are explicit and diagnosable.
5. **MCP is the only compatibility surface** — everything of value lives at or below the tool contract; agent-specific shells stay thin.

Decisions and their reasons: `docs/adr/`.
