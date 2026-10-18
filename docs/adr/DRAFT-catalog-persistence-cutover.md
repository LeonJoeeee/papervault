# Catalog persistence and fenced cutover

Status: design accepted on [#268](https://github.com/LeonJoeeee/papervault/issues/268),
including its guarded-completion amendment of 2026-10-18. Number assigned at integration.
This implements P2 preparation; live activation requires separate owner authorization.

## Context

The working set has many direct Paper mutations and debounced Library saves. Rewriting
all callers would enlarge the migration. P1 supplies a lossless PostgreSQL shadow
catalog, but its advisory lock alone cannot reject observations from stale writers.
JSON remains the default authority during preparation.

## Decision

Keep the Library facade. `PAPERVAULT_CATALOG_AUTHORITY` selects `json` (default) or
`postgres`; PG requires an explicit `PAPERVAULT_CATALOG_DSN`, with optional
`PAPERVAULT_CATALOG_SCHEMA` (`catalog`). There is no credential discovery or JSON
fallback on PG failure. Schema migration belongs to the explicit import/preflight job.

The PG backend remembers each raw payload, its normalized Paper snapshot and revision.
A save diffs the working set and sends only changed records. One transaction commits
that delta, one `state_flush` audit event per changed key with its payload SHA256, alias
projections, and pending export/reopen job markers. Unchanged records are untouched.
The save batch (the debounce interval) is the commit unit, rather than each attribute
assignment. A crash loses the uncommitted in-memory batch, as in JSON mode, and cannot
commit half of a catalog batch. Acknowledgment preserves changes made to the working
object while the save ran; committed snapshots advance only after the outer commit.
Metadata-only PG edits preserve absent legacy defaults and unknown payload fields,
including with acquisition V2 flags enabled. Derived acquisition state stays a read
projection until an ingress/status writer or an explicit state command records an
observation; saving metadata alone does not materialize it.

Every record upsert/status write requires `expected_revision`. New writes start at
revision 1; existing P1 rows migrate with revision 0. A stale expectation raises
`StaleWrite` before audit or pending work commits. The backend reloads and retries
once only for disjoint bibliographic metadata changes. Overlaps, identity/policy/state
changes, unknown changes and a second race raise `CatalogConflict`, leave the rejected
batch dirty, and emit `catalog_conflict`. The rejected snapshot is not advanced, so a
later save cannot silently overwrite the winner. Operator resolution requires a fresh
working set and deliberate reapplication of the edit.

Reopen jobs use stable `reopen:<key>` IDs, generation as attempt, and the claim lease.
The existing file ledger and in-memory admission checks remain the pre-dispatch gate.
Conditional `complete_job` accepts only the expected attempt/lease in queued/dispatched
state, with its audit in the same transaction as the record batch. Ordinary saves of
terminal claim transitions use that predicate too. Imported active claims materialize
matching jobs before preflight can report readiness. Stale job snapshots cannot regress
newer attempts or reopen settled attempts. Staged publication takes file lock then PG
lock, checks revision and lease before touching artifacts, and retains its stage on a
failed/unknown publication commit for forward recovery. Artifact bytes precede their
catalog references; a failed catalog commit may leave unpublished bytes and must not
be treated as successful publication. This does not implement P5 versioned derivatives.

In PG mode, purge and DOI-resolution collision merges refuse before changing the
working set or unlinking files, including before dispatch to the V2 merge path.
Missing keys in a save also refuse.
Retention, tombstones and record deletion await P5.

JSON load/save/rotation/BibTeX behavior stays in its existing path. PG save never writes
`index.json` or its backup. `scripts/export_catalog.py --output <path> --verify` reads
one repeatable-read catalog snapshot, writes the current version-1 papers format to a
unique temporary file, runs the shadow comparator, and publishes by rename plus fsync.
It preserves unknown payload fields. Export is a separate job, not a second authority;
the `catalog-export:<key>` markers name pending projection work and the source revision.
Scheduling and acknowledgement of that job remain an operational responsibility before
live activation; P2 preparation provides the executable export job.

## Handoff and recovery

Every Library holds a shared lifetime lock for its vault; close releases it and prevents
further writes. Handoff takes the exclusive lock and a durable PG writer fence, covering
service/CLI Libraries and direct repository writers. A durable vault phase record keeps
writers fenced between commands and after a process failure. All old writer versions
must first be stopped/drained by the orchestrator; the cooperative lock cannot attest to
an unmodified older binary. The probe uses an explicit internal fence token.

The preparation CLI requires `--rehearsal`, an explicit vault copy and env file inside
it, and the `papervault_catalog_stage` database. It refuses the live vault and its
subdirectories. It deliberately provides no live activation escape hatch.

1. `--preflight --apply`: fence, import the JSON delta idempotently, verify counts,
   reference integrity and comparator zero; record source hash and catalog audit head.
2. `--flip`: verify those revisions, retain a frozen canonical export, change the single
   authority env setting, perform and restore a real Library title probe while fenced,
   then export/compare again. Its final phase is `flipped_fenced`.
3. `--rollback`: a failed preflight in `preparing` or a successful preflight in
   `preflighted` can be canceled before writes resume. For `preflighted`, first
   verify the current JSON SHA256 matches the recorded `json_sha256`; a mismatch
   refuses cancellation, retains the fences and requires forward recovery.
   Release its database fence (if schema creation reached that point) and
   durably record `rolled_back`, leaving JSON and env untouched. Partial imports
   remain nonauthoritative history; cancellation requires neither a frozen export
   nor count/parity equality. An interruption after DB fence release can be retried.
   Correct the import inputs before another preflight; an explicit verified export
   is allowed after recovery, while Library authority checks remain enforced.
   After flip, restore frozen JSON and JSON authority, release the database fence,
   and verify JSON loads. Interrupted probe writes carry a durable
   handoff token in audit, so they can be distinguished from unaccounted writes. An
   interrupted rollback can be resumed. Catalog probe/audit rows remain intact as
   nonauthoritative rehearsal history; a subsequent preflight imports JSON again.
4. `--resume`: verify parity, durably set `writes_resumed=true`, release fences, then
   invoke the explicit orchestrator restart command and health command with a recorded
   reason. Export parity is checked after restart. A restart/health failure cannot reopen
   the frozen rollback window. Unaccounted writes or a changed frozen artifact require
   forward recovery; never restore an older JSON snapshot over accepted writes.

The narrow rollback path restores the catalog authority, not arbitrary external side
effects. Rehearsal evidence and remaining live prerequisites are maintained in the
[readiness checklist](../specs/2026-10-18-p2-readiness.md). Source/artifact backup and the
other per-domain activation contracts in ADR-0007 remain prerequisites to a live flip.
