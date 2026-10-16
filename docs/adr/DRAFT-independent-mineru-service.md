# Independent MinerU service ownership

Status: Implements the owner ruling in [issue #235](https://github.com/LeonJoeeee/papervault/issues/235) (2026-10-16); number assigned at merge.

## Context

MinerU has multiple callers. A library-owned start/stop watchdog or restart can
interrupt another caller's work. The host now provides an independent backend
(`mineru.service`, private `:30001`) and an always-on gate (`mineru-gate.service`,
consumer `:30000`) that owns idle stop, workload-triggered start and bounded recovery.

## Decision

Keep papervault's existing HTTP consumer endpoint and endpoint resolution.
The library probes only gate `/health` once within a bounded HTTP budget;
it never manages the server's lifecycle or inspects backend load state. Health
200 admits workload even with a stopped backend and no model served yet:
stopped is idle and ready through the gate. The subsequent workload POST starts
the backend; requiring a model-list probe first would prevent that start.
Gate health failure remains a transport error, preserving extraction's uncharged
deferral and operator PDF pickup's pending state.

## Consequences

Lifecycle environment settings, watchdog tasks, transition locks, start
cooldowns and parse-triggered self-healing leave the caller. The independent
service absorbs backend start latency and owns recovery for every consumer. Operators
provision it separately; the library cannot repair a down service. Maintenance
co-tenancy guards retain their unit-presence checks against `mineru.service`
and the retained standalone template's `papervault-mineru.service`.
The orchestrator verifies stopped-backend extraction after deployment.
