# Independent MinerU service ownership

Status: Implements the owner rulings in [issue #235](https://github.com/LeonJoeeee/papervault/issues/235) and [issue #237](https://github.com/LeonJoeeee/papervault/issues/237) (2026-10-16), with the HTTP task client update accepted in [issue #288](https://github.com/LeonJoeeee/papervault/issues/288) (2026-10-22); number assigned at merge.

## Context

MinerU has multiple callers. A library-owned start/stop watchdog or restart can
interrupt another caller's work. The original #235 host deployment provided an independent backend
(`mineru.service`, private `:30001`) and an always-on gate (`mineru-gate.service`,
consumer `:30000`) that owns idle stop, workload-triggered start and bounded recovery.

## Decision

Keep papervault's existing HTTP consumer endpoint and endpoint resolution.
The library performs no caller-side pre-flight; `ensure_ready` is a compatibility
no-op. It never manages the server's lifecycle or inspects backend load state.
The service handles workload after idle GPU unload, including any backend wakeup
it needs. The client's submit/poll/result requests determine reachability;
their transport errors preserve extraction's uncharged deferral and operator
PDF pickup's pending state. Pending queue-wait timeouts remain transport errors;
read or wall-clock timeouts during observed processing charge an extraction attempt.

Under #288, the shared `mineru-service` exposes unmodified official `mineru-api`.
Its `AsyncTaskManager` owns FIFO queuing, with `--enable-vlm-preload true` and
GPU model unload after 20 idle minutes. papervault submits whole PDFs with
`backend=vlm-engine` over core `httpx`, polls task status and fetches markdown.
It imports no MinerU parser and owns no rendering, singleton or model-discovery
cache. Each endpoint attempt owns an `httpx.AsyncClient` and closes it on its
own event loop, including cancellation. This replaces the prior thread-offloaded
MinerU singleton/discovery and cached-client cleanup. Extract-worker placement
remains unchanged.

## Consequences

Lifecycle environment settings, watchdog tasks, transition locks, start
cooldowns and parse-triggered self-healing leave the caller. The independent
service absorbs backend start latency and owns recovery for every consumer. Operators
provision it separately; the library cannot repair a down service. Maintenance
co-tenancy guards retain their unit-presence checks against `mineru.service`
and the retained standalone template's `papervault-mineru.service`.
The orchestrator verifies a submit/poll/result round-trip after idle GPU unload,
including any service-owned backend wakeup, after deployment.
