# papervault — PRD

## One sentence
An industrial-grade, self-hosted literature + knowledge layer for coding agents — installed as a Claude Code plugin (MCP underneath), it lets any research agent discover papers, read their full text, and query a graph-RAG knowledge base distilled from everything it has ingested, with cited answers.

## Why build it
- Research agents guess; grounded answers require a corpus the agent has actually read. Generic RAG tooling ships unmeasured retrieval quality — the claims are vibes.
- This stack already exists and runs in production on a 5,165-paper space-physics corpus: measured recall floors, self-healing ingest queues, a single-GPU VRAM budget, and a benchmark harness. Packaging it costs less than everyone else rebuilding it badly.
- The pipeline is domain-general by design; space physics is only its first instance. Nothing installable today gives coding agents this layer.

## Features (as user value)
- Agent can **discover** literature by describing research intent in prose (multi-source search; relevant finds are auto-ingested: downloaded, OCR'd, full text on disk).
- Agent can **read** any library paper's verbatim full text via a returned file path; batch metadata lookup by citation key / DOI / arXiv id / fuzzy title.
- Agent can **ask** the knowledge base a question and get cited prose synthesis ([paper_key] inline cites + an honesty signal on coverage), served from a knowledge graph distilled from every ingested paper.
- The corpus **compounds without operator action**: search → queryable knowledge in ~tens of minutes; self-healing queues; no babysitting.
- The **domain layer is operator-editable configuration** (entity ontology, extraction few-shots, search domain gate); space physics ships as the factory default and worked sample — other fields adapt the sample files at their own effort.
- Operator can **measure their own instance** with the bundled eval harness (recall@served, hallucination rate, trap questions) — the quality bar is reproducible, not claimed.
- Works from **Claude Code via plugin install**; works from **any MCP-capable agent** (Codex, …) via documented config. The only LLM requirement: an operator-supplied OpenAI-compatible endpoint + key.

### What we are NOT doing
- **No GPU-less / lite / degraded mode.** The floor is one 24 GB-class NVIDIA GPU on Linux. Below the floor, this product is not for you.
- **No hosted/SaaS offering** — self-hosted only.
- **No paper-corpus distribution** (copyright: every instance builds its own library).
- **No onboarding hand-holding** beyond honest docs and a smoke test — no wizards, no fallback paths.
- **No bespoke per-agent integrations** — MCP is the only compatibility surface; per-agent packaging stays a thin shell.
- **Not open-sourcing the research-agent orchestration layer** (paper pipeline, reviewers, cycles). Only the literature + knowledge services.
- **Not shipping multiple domain presets** — space physics is the shipped domain; adapting to another field is the operator's work (ADR-0003).

## Definition of done (v1 release)
1. A documented clean-machine install (Linux, 1× 24 GB GPU) reaches green on a shipped smoke test — search ingests new papers, full text is readable, knowledge query returns a cited answer — in ≤ 8 operator commands.
2. The bundled eval harness runs against an operator corpus and emits the full metric set (recall@served, hallucinated_rate, trap gates); CI proves it end-to-end on a small open-access fixture corpus with a pinned floor.
3. The domain layer (search-gate rubric, entity ontology, extraction few-shots, instruction text) is externalized to operator-editable config files with the space-physics factory default; swapping domains requires no code edits, and each file's role is documented.
4. Full co-load (concurrent OCR + concurrent queries) stays inside the published single-GPU VRAM budget with zero OOM on reference hardware; the budget ships as the config defaults.
5. Queries longer than default MCP client timeouts survive end-to-end (progress heartbeat), covered by the smoke test.
6. The Claude Code plugin installs from the marketplace repo (private during beta); at least one non-Claude MCP client is verified against the same server with documented config.
7. The public repo has fresh history, a clean secrets audit (zero credentials, zero .env, zero key backups), no paper data, a license file, and green CI on every merge.

## Constraints
- Hardware floor: 1× NVIDIA 24 GB (RTX 3090/4090 class), Linux + CUDA. A second GPU is comfort, never a requirement.
- Operator supplies an OpenAI-compatible LLM endpoint; the product bundles no keys and no gateway.
- Stack is fixed in v1: LightRAG (pinned) + Neo4j + Postgres + MinerU vLLM OCR + BGE embed/rerank. One profile, no alternates.
- The repo starts from zero history; nothing migrates from the private monorepo except curated code.
- This repo is upstream (ADR-0001): the lab instance is deployment #1, consuming releases.
- Grey-zone download tiers (Sci-Hub, Anna's Archive, anti-bot scraping) ship flag-gated and disabled by default (ADR-0004).

## Release plan
- **v1 = closed beta** for the originating research group (all space physics); repo stays private, group members install from it directly.
- **Open testing** follows: the repo flips public. The flip is gated by the secrets/legal/PII audit in DoD #7 — it is a release step, not a formality.
