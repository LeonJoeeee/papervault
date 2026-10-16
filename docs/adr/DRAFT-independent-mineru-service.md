# Independent MinerU service ownership

Status: Implements the owner ruling in [issue #235](https://github.com/LeonJoeeee/papervault/issues/235) (2026-10-16); number assigned at merge.

## Context

MinerU has multiple callers. A library-owned start/stop watchdog or restart can
interrupt another caller's work. The host now provides an always-on backend
(`mineru.service`, private `:30001`) and a gate (`mineru-gate.service`, consumer
`:30000`) that owns idle sleep, request-triggered wake and bounded recovery.

## Decision

Keep papervault's existing HTTP consumer endpoint and endpoint resolution.
The library probes `/health` and `/v1/models` once within a bounded HTTP budget;
it never manages the server's lifecycle or inspects sleep state. A sleeping
backend is ready when health is 200 and the model list contains a model.
Readiness failure remains a transport error, preserving extraction's uncharged
deferral and operator PDF pickup's pending state.

## Consequences

Lifecycle environment settings, watchdog tasks, transition locks, start
cooldowns and parse-triggered self-healing leave the caller. The independent
service absorbs wake latency and owns recovery for every consumer. Operators
provision it separately; the library cannot repair a down service. Maintenance
co-tenancy guards retain their unit-presence checks against `mineru.service`.
The orchestrator verifies asleep-to-extraction behavior after deployment.
