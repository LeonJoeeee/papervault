# P2 readiness — executed live cutover

Live authority flip: **EXECUTED**, with owner approval, by the orchestrator on
**2026-10-19 09:18–09:23 CST (UTC+08:00)**. The
[live cutover decision and execution record](https://github.com/LeonJoeeee/papervault/issues/268#issuecomment-6257668738)
supersedes the preparation-only status. PostgreSQL database `papervault_catalog` is now
the authoritative catalog; `/data/paper-vault/index.json` is its canonical export.
[The persistence/cutover ADR](../adr/0010-catalog-persistence-cutover.md) defines the
mechanics accepted on [#268](https://github.com/LeonJoeeee/papervault/issues/268).

## Executed sequence

The orchestrator executed the live sequence manually. The rehearsal CLI deliberately
refuses the live vault; the copy-only commands below were not used to activate it.
The overall window and the import/export durations below are wall-clock timings.

| Step | Recorded result |
| --- | --- |
| Backup | Retained `index.json`, correction ledger, reopen-budget ledger and `.env` in `/home/leon/backups/papervault-cutover-20261019` (109 MB). |
| Stop writers | Stopped `papervault.service`; 8 idle loopback sessions were left to reconnect automatically. |
| Fresh catalog database | Recreated `papervault_catalog` before the final import. |
| Import | `import_catalog --apply`: **36 s**, **26,696 records**, difference rate **0.0**, integrity errors **0**; 21 identity conflicts reported and preserved without merging. |
| Compare | **26,696 compared / 0 different / difference rate 0.0**. |
| Authority flip | Set `.env` to `PAPERVAULT_CATALOG_AUTHORITY=postgres` with an explicit `PAPERVAULT_CATALOG_DSN`; retained `.env.bak-20261019-cutover`. Credentials remain in protected configuration. |
| Export | PostgreSQL → `/data/paper-vault/index.json`: **8 s**, verification difference rate **0.0**. |
| Writes-resumed marker | Durably recorded `/data/paper-vault/.catalog-cutover.json` with `phase: flipped_live` and `writes_resumed: true`, before starting the service. |
| Start and health | `papervault.service` active; MCP initialization and `get_paper("Adriani2009")` succeeded. |
| First PostgreSQL writes | `catalog.audit` advanced **26,701 → 26,714**: **+13 audit rows** from recovered-queue state flushes. The JSON index SHA remained unchanged. |

The frozen-snapshot rollback window **closed at the writes-resumed marker**, before
service start. Recovery is now **forward-only**: preserve accepted PostgreSQL writes,
audit and pending work. A restart or health failure cannot reopen that window; do not
restore an older JSON snapshot over accepted catalog mutations.

## Post-cutover operations

The orchestrator installed the following user units after the flip. Times use the
deployment host's CST (UTC+08:00) timezone.

| Unit | Schedule | Operation |
| --- | --- | --- |
| `papervault-export.timer` | Hourly at **:17**, `OnCalendar=*:17` | Export PostgreSQL to the canonical `index.json` with `--verify`. The operational test refreshed the file and used 7 s CPU; this is separate from the 8 s wall-clock cutover export above. |
| `papervault-db-backup.timer` | Nightly at **03:23** | `pg_dump -Fc papervault_catalog` to `/home/leon/backups/papervault-db/`, with **14-day retention**. The first dump was 28 MB. |

The JSON index is an hourly projection, not a second authority; ordinary PostgreSQL
saves do not rewrite it. Deferred items remain: adjudication of the 21 identity
conflicts, the twin-creation defect tracked in
[#267](https://github.com/LeonJoeeee/papervault/issues/267), and off-host backup
replication under the F1 backup ADR. The local dump schedule does not close that
off-host recovery item.

P5b adds the separate [coordinated recovery procedure](../../README.md#coordinated-recovery)
and [isolated drill evidence](../baselines.md). Its weekly sets include source and
derivative stores under `/data/p5b-recovery`; the orchestrator owns operational
activation after merge. The nightly catalog-only timer above remains unchanged and
does not establish recovery-set completeness. Same-host P5b evidence does not close
the off-host recovery item.

## Historical rehearsal evidence

Preparation first imported a read-only JSON/ledger copy into isolated
`papervault_catalog_stage`: 26,696 records, comparator difference rate 0 and no import
integrity errors. The bounded restart fixture exercised the real Library load/save
path, writes-resumed marker and post-start export parity, rather than the full MCP
topology. Opt-in PostgreSQL tests covered independent-process writers, overlapping
write conflicts, stale completion rejection and append-only audit.

The orchestrator subsequently recorded the
[full-scale rehearsal](https://github.com/LeonJoeeee/papervault/issues/268#issuecomment-6252595278):
26,696 records, read-only symlinks to retained paper artifacts, a staging MCP service
serving `get_paper("Adriani2009")`, and zero-difference import/flip comparisons.
Wall-clock timings were 29 s for import, 25 s for preflight/apply, 30 s for flip and
3.5 s for rollback. Identity conflicts had risen to 21 and were reported without
merging. The live sequence above records the subsequent production comparison and
activation separately.

## Historical fenced rehearsal runbook

Retained as guidance for isolated copies. These commands do not authorize another live
flip. The rehearsal phase file is `.catalog-handoff.json`; the manual live cutover used
the `.catalog-cutover.json` marker recorded above.

Provision a vault **copy**, retain its JSON, correction ledger and reopen-budget ledger,
and provision a staging database named `papervault_catalog_stage`. For a full service
rehearsal, copy the retained artifacts and isolate knowledge-store dependencies too.
Set `PAPERVAULT_CATALOG_DSN` explicitly in protected configuration; never place credentials
in a PR, shell history or a published command. Use a new catalog schema for a new copy.
Every writer must run the new cooperative fencing code, or be stopped and drained.

The following paths are placeholders for orchestrator-selected copy destinations. The
CLI refuses the live vault and requires the env file to remain inside the copy.

```bash
.venv/bin/python scripts/cutover_catalog.py --rehearsal --vault "$COPY" --env-file "$COPY/.env" --schema "$SCHEMA" --preflight
.venv/bin/python scripts/cutover_catalog.py --rehearsal --vault "$COPY" --env-file "$COPY/.env" --schema "$SCHEMA" --preflight --apply
.venv/bin/python scripts/cutover_catalog.py --rehearsal --vault "$COPY" --env-file "$COPY/.env" --schema "$SCHEMA" --flip
# Exercise this branch while the copy remains fenced; no service restart yet.
.venv/bin/python scripts/cutover_catalog.py --rehearsal --vault "$COPY" --env-file "$COPY/.env" --schema "$SCHEMA" --rollback
```

For the resume rehearsal, repeat preflight/apply and flip after rollback, then run:

```bash
.venv/bin/python scripts/cutover_catalog.py --rehearsal --vault "$COPY" --env-file "$COPY/.env" --schema "$SCHEMA" --resume --restart-command "$COPY_RESTART_ARGV" --health-command "$COPY_HEALTH_ARGV" --reason 'authorized isolated copy rehearsal'
.venv/bin/python scripts/export_catalog.py --schema "$SCHEMA" --output "$COPY/index.json" --verify
```

Commands are parsed as argv, without a shell. The restart command must bind the service
to the copy, load its env file, and return; health must fail if the copy service is not
ready. The tool runs neither a default systemd action nor a live command. Reserve 15
minutes for stop/drain and source capture, 15 for import/parity, 15 for probe/verification,
and 15 for restart/health or recovery. Measure the full topology before treating that
budget as achieved. If a command fails, read `.catalog-handoff.json` before recovery;
its `writes_resumed` value is the rollback boundary. Preserve frozen JSON, audit, job
markers and any uncertain `.reopen-staging` results. No scratch copy is a backup.

If `--preflight --apply` fails with phase `preparing` and `writes_resumed=false`,
or succeeds with phase `preflighted` and the operator decides not to flip, cancel
preparation with the same copy, env file, schema and staging database:

```bash
.venv/bin/python scripts/cutover_catalog.py --rehearsal --vault "$COPY" --env-file "$COPY/.env" --schema "$SCHEMA" --rollback
```

This releases the repository fence and records `rolled_back` without changing the
vault JSON or env, even if import only partially committed or schema creation failed.
For `preflighted`, cancellation first checks that JSON still matches the recorded
SHA256. Changed JSON is refused with forward-recovery guidance and fences retained.
No frozen snapshot is needed or restored. It does not undo imported catalog rows.
Do not delete the handoff file manually.
Retry the same command if interrupted during recovery. Correct integrity errors or
reconcile parity differences, then rerun `--preflight --apply`. An explicit
`export_catalog.py --verify` is permitted after recovery; it replaces JSON from PG,
so run it only when that is the intended reconciliation. Once `writes_resumed=true`,
rollback is refused regardless of phase; preserve accepted writes and recover forward.

## Verification

The [preparation PR](https://github.com/LeonJoeeee/papervault/pull/269) contains rehearsal
commands, snapshot hashes, counts, comparator output and timings; the linked
orchestrator records above supply full-scale rehearsal and live execution evidence.
The documentation update PR records final commands, exit codes, version alignment and
suite/lint/CI results. PG tests skip only without
`PAPERVAULT_CATALOG_TEST_DSN`; an explicitly configured but unavailable DSN fails.
The exact merge-result CI and independent acceptance verdict remain orchestrator-owned.
