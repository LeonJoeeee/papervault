# papervault — PRD

## One sentence
A self-hosted, single-host research literature and knowledge infrastructure for coding and research agents — exposed as one MCP endpoint (`search_papers`, `get_paper`, `query`) so Antigravity, Claude Code, Codex, and other MCP clients can discover papers, read verbatim full text, and synthesize strictly cited answers across tens of thousands of papers and operator-supplied textbooks and notes.

## Why build it
- Research agents guess when left without grounded full text; generic RAG wrappers ship unmeasured retrieval quality and silently drop landmark papers or fabricate citations.
- This stack runs in production on a ~30k-paper personal/group research corpus spanning **space physics, space computing, and AI4Science** alongside operator-supplied textbooks and private research notes — backed by transactional PostgreSQL catalog authority, measured recall/precision floors, self-healing acquisition and distillation queues, single-GPU resource budgets, and a reproducible evaluation harness.
- The pipeline keeps clean domain boundaries via an editable domain pack while avoiding multi-tenant or SaaS bloat; packaging and governing it cleanly beats every researcher rebuilding a brittle literature stack from scratch.

## Design Philosophy

These five principles bridge the researcher's black-box caller needs (omniscient-mentor accuracy/completeness and fast, resource-disciplined execution on one Linux/CUDA workstation) to concrete engineering tradeoffs.

### 1. Caller accuracy and canon recall over blind heuristics
- **What we prioritize:** Giving the calling agent the exact literature and evidence it actually needs — both frontier work and landmark canon, including indirect knowledge-graph-linked papers that lack literal keyword overlap. Search separates intent parsing, multi-track retrieval (external scholarly APIs + local BM25 + PGVector/Neo4j KG topology + citation/venue authority prior), quality/relevance gates, and serving; query pins one consistent serving manifest and separates `retrieved_papers` from strictly verified inline `cited_papers`.
- **Temptations we reject:** Unmeasured "vibes" prompt or weight tweaks without eval arbitration; letting recency filters or BM25 literalism hide foundational works; accepting unjudged or hallucinated citations just to produce fluent prose.

### 2. Automatic compounding with strict quality and token discipline
- **What we prioritize:** An unattended corpus loop (`search -> decoupled registration -> acquisition -> OCR extraction -> graph distillation`) that turns discovery into queryable knowledge without babysitting, while keeping catalog quality high and LLM spend bounded. A two-way quality fence blocks conference abstracts, stubs, and non-papers; PDF identity and post-OCR clarity/completeness gates protect the vault; tiered role routing (`PAPERVAULT_LLM_<ROLE>`) drains subscription quota before paid relays, uses per-model failover cooldowns, trips build breakers on upstream outages, and supports optional off-peak build windows.
- **Temptations we reject:** Blindly ingesting every search hit into the catalog or graph; burning thousands of failed build calls during upstream outages; hardcoding one frontier model across trivial triage and heavy synthesis alike.

### 3. Pay for work, never for idleness
- **What we prioritize:** Fast active responsiveness with minimal steady-state host footprint (CPU, RAM, VRAM, SSD). Blocking embedding, reranking, and external fetches stay off the MCP event loop; heavy MCP calls use bounded-wait resumable task handles so client timeouts never cancel in-flight work; process allocator (`MALLOC_MMAP_THRESHOLD_`, `MALLOC_ARENA_MAX`) and BLAS/Rayon thread caps prevent memory bloat and oversubscription; BGE idle unload and standalone `mineru-service`'s 20-minute idle GPU release return VRAM when the workstation is quiet.
- **Temptations we reject:** Keeping OCR models resident inside the MCP server process; unbounded worker pools or polling spin-loops; letting background bursts starve interactive `get_paper` / `query` calls or leak RSS indefinitely.

### 4. One transactional catalog authority and fail-closed integrity
- **What we prioritize:** Clear logical persistence roles — **Source** (immutable raw PDFs and operator documents), **Catalog** (single transactional PostgreSQL authority over identity, policy, registration, lineage, and state; JSON exports are projections only), and **Derived** (versioned extracts, embeddings, and graph views bound to inputs and recipes). Serving is a read that never mutates business state synchronously; unknown policy, stale revisions, missing verification, or unavailable catalog authority fail closed; coordinated recovery sets and explicit retention policies protect irreplaceable research assets.
- **Temptations we reject:** Dual-truth stores that silently fall back from PostgreSQL to JSON; conflating backend `processed` flags with verified catalog coverage; silent in-place overwrites, routine destructive purges, or speculative auto-merges that destroy identity history.

### 5. Build for single-host reality while keeping clean domain boundaries
- **What we prioritize:** Designing honestly for one 24 GB-class NVIDIA GPU workstation running Linux/CUDA, Docker (PostgreSQL + Neo4j), standalone `mineru-service`, and one Python MCP server — while keeping the existing clean internal boundaries (`mcp/`, `library/`, `store/`, `knowledge/`, `domain/`, `eval/`) and the externalized domain pack (`space_physics`, covering space physics, space computing, and AI4Science) intact.
- **Temptations we reject:** Paying engineering and maintenance tax for hypothetical multi-tenant SaaS, cloud auto-scaling, GPU-less degraded modes, or multi-preset plugin frameworks with no second validated domain; duplicating retrieval or lifecycle logic inside client-specific plugins.

## Features (as user value)
- Agent can **discover** literature by describing research intent in prose (`search_papers`: multi-source external search fused with local BM25 + PGVector/Neo4j KG topology + authority prior; admitted finds are registered and acquired asynchronously without blocking serving).
- Agent can **read** any library paper's verbatim full text (`get_paper`: citation key / DOI / arXiv ID / fuzzy title resolution returning metadata plus a serve-safe full-text path when available).
- Agent can **ask** the knowledge base a question (`query`: multiquery graph-RAG synthesis with strictly verified `[paper_key]` inline citations, separate operator-document attribution, and explicit coverage/refusal honesty).
- Operator can **incorporate canonical textbooks and private notes** (`papervault ingest-doc` and live operator-drop drain, with private notebooks strictly gated to private instances).
- The corpus **compounds without operator action**: search → acquisition → standalone MinerU OCR → graph distillation with self-healing queues, two-way quality fencing, and bounded cost.
- The **domain layer is operator-editable configuration** (entity ontology, extraction few-shots, search domain gate, instructions); the factory `space_physics` pack covers space physics, space computing, and AI4Science — other fields adapt the pack files without code edits.
- Operator can **measure their own instance** with the bundled eval harness (recall@served, must-find search recall, precision, hallucination rate, trap/refusal gates) — quality bars are reproducible, not claimed.
- Works across **Antigravity (`agy`), Claude Code, Codex, and any MCP-capable client** via thin plugin shells or direct streamable-HTTP/stdio MCP configuration. The only external LLM requirement is an operator-supplied OpenAI-compatible endpoint or gateway + key.

### What we are NOT doing
- **No GPU-less / lite / degraded mode.** The floor is one 24 GB-class NVIDIA GPU on Linux. Below the floor, this product is not for you.
- **No hosted/SaaS or multi-tenant offering** — single-host, self-hosted only.
- **No paper-corpus distribution** (copyright: every instance builds and retains its own library).
- **No onboarding hand-holding** beyond honest docs, `papervault doctor`, and `papervault smoke` — no wizards, no silent fallback paths.
- **No bespoke per-agent integrations** — MCP is the only compatibility surface; per-agent plugins stay thin registration shells.
- **Not open-sourcing the research-agent orchestration layer** (paper pipeline, reviewers, cycles). Only the literature + knowledge infrastructure lives here.
- **Not shipping multiple domain presets** — the `space_physics` pack (space physics + space computing + AI4Science) is the shipped default; adapting to another field is the operator's work (ADR-0003).

## Quality & operational bars
1. **Clean-machine install & preflight:** A documented Linux + 1× 24 GB GPU install reaches green on `papervault doctor` and `papervault smoke`, with standalone `mineru-service` serving official `mineru-api` over HTTP.
2. **Reproducible evaluation:** CI proves deterministic scoring/aggregation logic, routing invariants, and catalog/lifecycle contracts on fixed inputs; reference-machine GPU eval runs (`docs/eval.md`, `docs/baselines.md`) gate retrieval-affecting changes against measured noise floors.
3. **Externalized domain configuration:** Search-gate rubric, ontology, extraction examples, and instructions live in operator-editable pack files (`src/papervault/domain/space_physics/` or `PAPERVAULT_DOMAIN_PATH`) with no code edits required to adapt domains.
4. **Bounded single-GPU & host envelope:** Co-loaded embedding/reranking and standalone MinerU OCR stay inside the 24 GB GPU budget with zero OOM; idle BGE/MinerU release returns VRAM when quiet; allocator and thread caps keep host RSS and CPU bounded.
5. **Timeout-resilient MCP serving:** Long `search_papers` and `query` runs survive client wall-clock timeouts via bounded-wait resumable task IDs and progress heartbeats without dropping admission permits or cancelling in-flight work.
6. **Multi-client MCP parity:** Antigravity (`agy`), Claude Code, and Codex connect cleanly to the same MCP endpoint (local loopback or tailnet HTTPS with `PAPERVAULT_MCP_ALLOWED_HOSTS` / optional `PAPERVAULT_MCP_TOKEN`).
7. **Repository hygiene & legal posture:** Public repository (since 2026-08-02) with zero committed credentials, `.env` backups, PII, or paper text/eval outputs inside the tree; grey-zone download tiers ship flag-gated and disabled by default (ADR-0004).

## Constraints
- Hardware floor: 1× NVIDIA 24 GB (RTX 3090/4090 class), Linux + CUDA. A second GPU is comfort, never a requirement.
- Operator supplies an OpenAI-compatible LLM endpoint or gateway; the product bundles no keys.
- Stack profile: pinned `lightrag-hku==1.5.7` + Neo4j + PostgreSQL (knowledge store & transactional catalog authority) + standalone `mineru-service` (official `mineru-api` HTTP task service, ADR-0009) + local BGE-M3 embed/rerank.
- This public repository is upstream (ADR-0001): deployments track annotated release tags, not moving branches.
- Grey-zone download tiers (Sci-Hub, Anna's Archive, anti-bot scraping) ship flag-gated and disabled by default (ADR-0004).

## Release & governance posture
- **Public upstream (since 2026-08-02):** The July 2026 closed-beta phase is complete. The repo is public under MIT with default-OFF grey download tiers in `.env.example` (ADR-0004).
- **Release-time versioning:** Ordinary feature/fix slices do not bump the global version; authorized releases synchronize all five version manifests (`pyproject.toml`, `uv.lock`, `plugin/.claude-plugin/plugin.json`, `plugin/plugin.json`, `.claude-plugin/marketplace.json`) and tag `vX.Y.Z`.
