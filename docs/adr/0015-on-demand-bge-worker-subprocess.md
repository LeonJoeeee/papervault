# 0015 — On-demand BGE worker subprocess for zero-idle-VRAM embedding and reranking

Status: Accepted (2026-11-21); delivered in `v0.11.0` (`c1cc27e`, `24b2756`) and verified through `v0.11.2` (`c24293e`). Amends the in-process local model placement assumed under [ADR-0002](0002-single-package-single-process.md) while preserving the single operator-managed MCP service boundary.

## Context

`papervault` runs `BAAI/bge-m3` embedding and `BAAI/bge-reranker-v2-m3` reranking locally on a single 24 GB-class NVIDIA GPU co-tenanted with the independent MinerU OCR service ([ADR-0009](0009-independent-mineru-service.md)). Historically, both models loaded in-process inside the long-lived `papervault-mcp` server process.

In production on lab deployment #1 (`v0.10.31`), importing `torch` and initializing CUDA inside `papervault-mcp` created an irreducible steady-state footprint: even after `KS_BGE_IDLE_UNLOAD=1` deleted Python model references and called `torch.cuda.empty_cache()`, the persistent CUDA primary context and PyTorch/glibc allocations kept **~432 MiB of GPU VRAM** and **~6.4 GiB of idle RSS** (with `17.0 GiB` peak and `278 MB` swap) pinned indefinitely between bursts. Because CUDA contexts cannot be fully destroyed within a running Python process without terminating it, in-process cleanup alone could not achieve zero idle GPU residency.

## Decision

Isolate local BGE embedding and reranking behind an on-demand worker subprocess owned by `papervault-mcp`, controlled by `KS_BGE_WORKER_MODE` (default `subprocess`; `inprocess` retained as a rollback escape hatch):

1. **Zero PyTorch/CUDA imports in the MCP server process.** Under `KS_BGE_WORKER_MODE=subprocess`, `papervault-mcp` spawns `sys.executable -m papervault.knowledge.store.bge_worker` lazily on the first embedding or reranking request. Top-level imports in `bge_worker.py` remain lightweight stdlib so importing `_BGEWorkerClient` never loads `torch`, `transformers`, `sentence_transformers`, or `FlagEmbedding` in the parent process.
2. **Strictly zero-pickle binary-framed RPC with stdout isolation.** Requests and responses cross `stdin`/`stdout` using `[4-byte big-endian uint32 header_len][UTF-8 JSON header][optional raw float32 C-contiguous ndarray payload]`. Before importing ML libraries, the worker redirects OS file descriptor `1` to `2` (`stderr`) and communicates exclusively on a duplicated private protocol descriptor so third-party library logging cannot corrupt framing.
3. **Orphan prevention and bounded RPC recovery.** The parent passes a dedicated read-only kernel pipe (`_KS_BGE_WORKER_DEATH_FD` via `pass_fds`); a daemon watchdog thread in the worker blocks on that pipe and terminates immediately on EOF so an abrupt parent crash never leaves a CUDA-blocked orphan. Per-request `select.select` deadlines (`KS_BGE_WORKER_EMBED_TIMEOUT_S=180`, `KS_BGE_WORKER_RERANK_TIMEOUT_S=600`, `<= 0` disables) kill and reap hung workers so subsequent requests respawn cleanly. An explicit MCP shutdown hook terminates active workers on graceful server exit.
4. **True zero-idle-VRAM release.** With `KS_BGE_IDLE_UNLOAD=1` and `KS_BGE_IDLE_TIMEOUT=600`, idle timeout exits the worker subprocess cleanly (`rc=0`), returning 100% of its PyTorch CPU RAM, glibc arenas, and CUDA primary context (`0 MiB` idle VRAM) to the OS. The next `query` or `search_papers` respawns the worker on demand.

Rejected alternatives:

- **In-process unload + `torch.cuda.empty_cache()` + `malloc_trim(0)` only:** cannot release the CUDA primary context (~432 MiB VRAM) or PyTorch's imported runtime footprint inside the long-lived server process.
- **Standalone always-on systemd BGE microservice (analogous to MinerU):** unlike MinerU, local BGE serves only `papervault`; adding another operator-managed daemon, port, and unit would erode [ADR-0002](0002-single-package-single-process.md)'s single-service deployment model without multi-caller benefit.

## Consequences

- Operators still manage one `papervault.service` unit and one MCP endpoint; no new port or external daemon is introduced.
- On lab deployment #1 (`v0.11.2`, 5.5 h steady-state alongside Batch 2–4 memory fixes), idle GPU VRAM fell from **432 MiB → 0 MiB** and main-process idle RSS fell from **6.4 GiB → 1.44 GiB** (`1.89 GiB` HWM, `0 kB` swap); see [baselines](../baselines.md#post-v0112-resource-envelope-and-live-verification--2026-11-21).
- The first `query` or `search_papers` after a 10-minute idle exit incurs a one-time cold-spawn cost (measured on RTX 3090: **3.4 s** embed load + **6.4 s** rerank load), absorbed well within normal call budgets.
- Operators who prefer zero cold-spawn latency over idle VRAM release can set `KS_BGE_IDLE_UNLOAD=0` (keeping the subprocess warm) or `KS_BGE_WORKER_MODE=inprocess` (restoring legacy in-process loading).
