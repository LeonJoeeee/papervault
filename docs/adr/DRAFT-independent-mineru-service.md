# Independent MinerU service ownership

Status: Implements the owner rulings in [issue #235](https://github.com/LeonJoeeee/papervault/issues/235) and [issue #237](https://github.com/LeonJoeeee/papervault/issues/237) (2026-10-16); number assigned at merge.

## Context

MinerU has multiple callers. A library-owned start/stop watchdog or restart can
interrupt another caller's work. The host now provides an independent backend
(`mineru.service`, private `:30001`) and an always-on gate (`mineru-gate.service`,
consumer `:30000`) that owns idle stop, workload-triggered start and bounded recovery.

## Decision

Keep papervault's existing HTTP consumer endpoint and endpoint resolution.
The library performs no caller-side pre-flight; `ensure_ready` is a compatibility
no-op. It never manages the server's lifecycle or inspects backend load state.
A stopped backend is idle and ready through the gate: the workload POST starts
the backend. The real client's discovery/parse calls determine reachability;
their transport errors preserve extraction's uncharged deferral and operator
PDF pickup's pending state.

MinerU singleton construction, discovery and cleanup lookups run in thread
offloads, so contention on the singleton's lock cannot block the MCP event loop
(the hazard identified in #76). The prepared predictor passes into the parser,
avoiding its otherwise synchronous singleton lookup. Socket cleanup stays on
the client's own event loop. Extract-worker placement remains unchanged.

## Consequences

Lifecycle environment settings, watchdog tasks, transition locks, start
cooldowns and parse-triggered self-healing leave the caller. The independent
service absorbs backend start latency and owns recovery for every consumer. Operators
provision it separately; the library cannot repair a down service. Maintenance
co-tenancy guards retain their unit-presence checks against `mineru.service`
and the retained standalone template's `papervault-mineru.service`.
The orchestrator verifies stopped-backend extraction after deployment.
