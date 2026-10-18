# P2 readiness — preparation only

Live authority flip: **NO GO / not authorized**. This checklist implements
[#268](https://github.com/LeonJoeeee/papervault/issues/268) §6 and its accepted amendment;
[the persistence/cutover ADR](../adr/DRAFT-catalog-persistence-cutover.md) defines mechanics.

| Owner checklist item | Current evidence / remaining action |
| --- | --- |
| Live shadow is in sync | A read-only copy of the current live JSON/ledger was imported into an isolated `papervault_catalog_stage`: 26,696 records, comparator difference rate 0, no import integrity errors. This establishes copy/stage parity. The production shadow DSN was not supplied or discovered; the orchestrator must repeat the live read-only comparison. |
| Copy cutover and rollback | Fenced title probe, verified canonical export and frozen rollback exercised on the metadata copy. Final command timings and restart-fixture evidence are recorded in the PR. No production service or config was changed. |
| Concurrency | Opt-in PostgreSQL tests run two OS processes with independent connections/working sets: unrelated rows survive, one overlapping write reports a conflict, one completion wins and its stale peer is rejected, audit remains append-only, bounded joins detect deadlocks. Orchestrator must rerun with its explicit test DSN. |
| Window and recovery | Target: stop/drain writers → final delta import/comparison → frozen export/flip/fenced probe → mark writes resumed → restart/health/export verification, within 60 minutes. Metadata-only timings do not establish a full deployment downtime guarantee. Frozen rollback ends before the restart command; afterward recover forward. |
| Identity conflicts | The copy still reports 20 normalized identity conflict groups. Owner adjudication of aliases and the two suspected mislabels is pending; this change preserves every record and performs no adjudication. |

The restart rehearsal uses a disposable HTTP health fixture backed by the real Library
load/save path and the staging catalog. It performs an authoritative save after the
writes-resumed marker and verifies post-start export parity. It does **not** start the
complete MCP/knowledge/queue topology or copy the 72 GB source corpus. A full isolated
MCP deployment rehearsal, including retained artifacts and its dependencies, remains an
orchestrator prerequisite. Do not point its knowledge plane at production stores.

## Fenced rehearsal runbook

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

The PR contains final commands, exit codes, suite/lint/CI results, snapshot hashes,
counts, comparator output and timings. PG tests skip only without
`PAPERVAULT_CATALOG_TEST_DSN`; an explicitly configured but unavailable DSN fails.
The exact merge-result CI and independent acceptance verdict remain orchestrator-owned.
