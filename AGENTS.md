# papervault

## Core need

A single researcher runs `papervault` on one Linux/CUDA workstation as the shared literature and knowledge memory behind their coding and research agents (Antigravity, Claude Code, Codex). Every caller interacts through one MCP black box (`search_papers`, `get_paper`, `query`) and relies on two non-negotiable promises:

1. **Omniscient-mentor accuracy and completeness:** Return what was actually asked for — never drop named landmark canon, classic references, or indirect knowledge-graph-linked papers behind fragile keyword matching, and never fabricate citations or pass unverified evidence off as grounded prose.
2. **Fast, resource-disciplined execution:** Deliver answers without wasted caller waits, burn zero idle CPU, RAM, VRAM, or SSD when the host is quiet, and keep LLM token spend disciplined while the corpus compounds automatically across ~30k papers and operator-supplied textbooks and notes.

## Design Philosophy

1. **Caller accuracy and canon recall over blind heuristics:** Separate search intent, multi-track retrieval (external APIs + local BM25 + vector/KG topology + authority prior), quality/relevance gates, and evidence-grounded synthesis so named works and foundational canon survive instead of being drowned by recency bias or keyword misses. Reject unmeasured "vibes" tuning — retrieval changes must clear measured eval noise floors.
2. **Automatic compounding with strict quality and token discipline:** Background acquisition, OCR extraction, and graph distillation accrete knowledge unattended, but every stage is gated against catalog pollution (two-way garbage fence, PDF identity verification, clarity/completeness gates) and wasteful spend (tiered role routing, cooldown failovers, build breakers, optional off-peak scheduling). Reject fire-and-forget ingest that floods the corpus with abstracts, stubs, or runaway retry loops.
3. **Pay for work, never for idleness:** Bound concurrency, memory arenas, CPU threads, and GPU residency to the single-GPU envelope; offload blocking work off the MCP event loop and release idle models (BGE idle unload, standalone MinerU 20-minute GPU unload) so the workstation stays quiet between bursts. Reject always-resident bloat, polling spin-loops, and caller-side OCR lifecycle coupling.
4. **One transactional catalog authority and fail-closed integrity:** Keep Source (immutable raw bytes), Catalog (authoritative identity, policy, registration, lineage, state), and Derived (extracts, vectors, graph) logically separate. Serving is a read that never silently mutates business state; unknown policy, stale revisions, missing verification, or unreachable authority fail closed rather than falling back silently. Reject dual-truth stores, silent overwrites, and destructive routine purges.
5. **Build for single-host reality while keeping clean domain boundaries:** Optimize directly for one 24 GB-class Linux/CUDA host without hypothetical multi-tenant, SaaS, or GPU-less degraded-mode tax, while preserving the existing editable domain pack (`space_physics` covering space physics, space computing, and AI4Science) and thin MCP client shells. Reject premature abstraction layers and bespoke per-agent business logic.

## Commands

Run inside a worktree-local virtual environment:

- Install / sync dev venv: `UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev`
- Lint: `.venv/bin/ruff check src tests`
- Unit tests (no GPU/DB/network): `.venv/bin/pytest -m 'not integration' -q`
- Preflight check: `.venv/bin/papervault doctor`

## Gotchas

- **Read-only cache sandboxes:** Prefix `uv` commands with `UV_CACHE_DIR=/tmp/uv-cache`.
- **Live service boundary:** The live user unit `papervault.service` runs `.venv/bin/papervault-mcp` from the primary checkout on `main`. Never edit the primary checkout, its `.venv`, its `.env`, or `/data/paper-vault` from a worktree lane, and never restart, stop, or reconfigure `papervault.service` without explicit operator authorization.
- **Worktree venv isolation:** Every worktree needs its own `.venv` (`uv sync --extra dev` inside the worktree); the primary `.venv` is an editable install of the primary checkout and will test the wrong tree.
- **Release-time versioning:** Ordinary feature/fix commits do **not** bump the global package version. Version bumps happen only on authorized releases and must synchronize all five version fields guarded by `tests/test_plugin_version_sync.py` (`pyproject.toml`, `uv.lock`, `plugin/.claude-plugin/plugin.json`, `plugin/plugin.json`, `.claude-plugin/marketplace.json`).
- **New worktree untracked files:** None — everything load-bearing is tracked.
- **No paper text or secrets in Git:** Never commit `.env`, `.env.bak*`, private hostnames/emails, or raw eval artifacts into the repo.
- **Record language:** English.

## Where to look

- When aligning on product scope, caller promises, or detailed Design Philosophy -> read `docs/PRD.md`.
- When changing search, query, LLM routing, catalog authority, or recovery contracts -> read `docs/architecture.md` and `docs/specs/2026-10-18-papervault-constitution.md`.
- When revisiting costly-to-reverse architectural choices -> read `docs/adr/`.
- When checking measured retrieval/recovery floors or running evals -> read `docs/baselines.md` and `docs/eval.md`.
- When deploying, tuning runtime env vars, or configuring MCP clients/systemd -> read `docs/INSTALL.md` and `.env.example`.
