# papervault — Baselines

The measured floors and envelopes the PRD's definition-of-done points at ("published
floor", "published budget"). Every number here is a real measurement with a date and
conditions — never an aspiration. Organized by the caller-facing businesses; component
instruments and arbitration rules at the end.

> Result tags cited below (e.g. `frozen154_*`, `srb_v1_*`, `mfr_v2_*`) name LOCAL,
> untracked evidence files (they embed verbatim paper text — see the Evidence convention
> in `docs/eval.md`). They are provenance labels, not repo paths.

**Reference conditions** (all numbers below unless stated): lab deployment #1 — RTX 3090
24 GB (`CUDA_VISIBLE_DEVICES=1`), fp16 reranker @ max_length 4096, rerank 2×4
(`KS_RERANK_MAX_ASYNC=2 × KS_RERANK_BATCH_SIZE=4`), `RERANK_TIMEOUT=480`, LightRAG 1.5.4,
corpus ≈ 5.2k papers (≈ 4.4k full-text distilled), models mimo-v2.5-pro (synth slot) /
mimo-v2.5 (build slot) via the LiteLLM gateway (12-key pool).

## Business: `query` (ask the knowledge base)

| metric | value | measured |
| --- | --- | --- |
| recall@served (distinct papers, FAST gold, --no-synth) | **0.8568** | 2026-07-18, frozen corpus, tag `frozen154_07180011` |
| — same-window 1.4.16 reference | 0.8046 | same frozen window (papervault **+5.2 pp**) |
| noise floor (run-level, historical) | 0.0132 | the flip/no-flip bar for retrieval changes |
| hallucinated citation rate | **0.0** | every eval arm to date |
| single-query end-to-end latency (with synthesis) | **187 s** (quiet-GPU single-shot, 41 cited, synth thinking ON); pool-cap 120's retrieval halving (arbitrated 1.90×, issue #21) absorbed the corpus growth + the thinking cost — supersedes the 172 s pre-cap / to-be-re-measured note | 2026-07-19, corpus ≈ 6.5k |
| synthesis stage alone | 55–79 s | prompt ≈ 200k tokens |
| long-call survival | ≥ 485 s proven | S17 progress heartbeat (45 s ticks) kept a deliberately-strict 120 s-SSE client alive |

Honest note: at the *retrieval* level trap questions still serve content
(trap_violation 1.0 in --no-synth arms, historical constant); refusal is enforced at the
synthesis layer. The e2e honesty gate lives there, not in retrieval.

## Business: `search_papers` (discover literature)

| metric | value | measured |
| --- | --- | --- |
| end-to-end latency (incl. ingest triage of new finds) | 319 s mean, 222–525 s range | 2026-07-18, 12-intent baseline run (supersedes the 216–251 s 2-smoke figure) |
| must-find recall (coverage: fraction of the frozen landmark pool the search surfaces; `scripts/mustfind_recall.py` vs `gold_search_pools.jsonl`) | **0.2361 mean / 0.00 min** | 2026-07-19, instrument v2 (audited served lists, lenient matcher, empty-serve retry+quarantine); single-shot — per-intent recall swings between runs, 3-rep protocol tracked on #37; results `mfr_v2_0719` |
| result relevance (P@served: fraction of served papers the canonical judge scores ≥ 0.7 against the stated need, title+abstract) | **0.9111 mean / 0.7333 min** | 2026-07-18; **canonical judge = Claude opus blind panel, reasoning effort MAX** (12 fresh-context agents, one intent each, rubric-only, no access to other judges' scores); results `srb_v1_0718_claudejudge_max.json` |
| ranking quality (order within the served list: P@5 / nDCG@15, canonical-judge relevance scores as graded truth over the `srb_v1_0718` served order) | **P@5 0.9667 mean / 0.60 min · nDCG@15 0.9828 mean** | 2026-07-18, same canonical judge; ranking is NOT the weak axis — coverage is |

The coverage/precision split is the search business's honest portrait: 91 % of what it
serves is relevant, but it surfaces fewer than 1 in 5 of the landmark papers a review
would demand — it serves the topical frontier and misses the canon (validated by hand:
an intent NAMING PAMELA/AMS got 15 modern analysis papers, zero of the original
measurement papers). Improvement lever tracked separately.

Judge triangulation (all on the same 180 served papers, `srb_v1_0718`): MiMo strong-slot
0.9056, opus default-effort 0.9444, opus max-effort 0.9111 — spread ±0.03, opus
default-vs-max self-agreement 170/180. Treat ±0.03 as judge-side measurement noise on this
instrument. Regime (user decision 2026-07-18): the opus max-effort blind panel is the
CANONICAL judge for baselines and arbitration; the MiMo judge (in
`scripts/search_relevance_baseline.py`) remains a cheap dev-time smoke that never decides.
Known boundary behavior: adjacent-subtopic serving is where judges disagree (UHECR intent:
0.60 MiMo / 0.73 canonical / 1.00 opus-default).

## Business: `get_paper` (lookup + full text)

| metric | value | measured |
| --- | --- | --- |
| latency (batch of 2, key + DOI forms) | **≈ 0 s** | every smoke |
| identifier forms | key / DOI / arXiv / fuzzy — all resolve | 2026-07-16 caller test |

This business is frozen: no optimization budget; regression watch only.

## Business: freshness pipeline (search → queryable)

| metric | value | measured |
| --- | --- | --- |
| search → full text on disk (download + OCR) | ≈ 10 min | 2026-07-16 live probe (5 papers) |
| search → queryable knowledge (distilled into graph) | ≈ 20–30 min | scheduler rounds every 60 s; distill on arrival |
| pipeline health under load | 342 consecutive clean rounds, 0 errors | post-cutover observation window, 2026-07-18 |

## Build plane (not caller-facing; cost envelope)

| metric | value | measured |
| --- | --- | --- |
| extraction yield (sample: 218k-char paper) | 25 chunks → 434 entities + 547 relations | 2026-07-18 probe, LightRAG 1.5.4 |
| ontology purity (fresh 1.5.4 build) | 434/437 in the closed 11-type set, **zero `Other`** | l0_probe; 3 UNKNOWN = framework relation-endpoint placeholders (0.56 % pre-exists in prod) |
| distill throughput | ≈ 1 paper/min under the 2-key era throttle | 2026-07-17 night; **remeasure with the 12-key pool** |

## Resource envelope

| metric | value | measured |
| --- | --- | --- |
| co-active VRAM peak (4-way OCR + concurrent queries) | **18.8 / 24 GB** | 2026-07-16 stress test |
| held-in-reserve levers | MinerU gmu 0.40→0.30; rerank batch 4→smaller | never needed |

## F0 frozen baseline (2026-10-18)

This is the P0 freeze required by the [ratified refactor plan](specs/2026-10-18-papervault-refactor-plan.md),
before F1 changes behavior. The [corrected orchestrator artifacts on #263](https://github.com/LeonJoeeee/papervault/issues/263#issuecomment-6250459625)
are preserved verbatim: [snapshot tool](../scripts/snapshot_catalog.py) and
[frozen JSON](baselines/f0-snapshot-2026-10-18.json). The orchestrator measured the snapshot at
**2026-10-18T21:25:55+0800** against lab deployment #1. These are sequential observations of a
running system, not an atomic database/filesystem snapshot or a migration backup. Live counts
drift; later runs must produce new evidence rather than overwrite this frozen measurement.

The legacy source/configuration reference is commit
[`f872ec097f488356f4547b3c8447712e3263db71`](https://github.com/LeonJoeeee/papervault/tree/f872ec097f488356f4547b3c8447712e3263db71)
(package 0.7.37). The following instrument, workload, and judge-configuration identities freeze
what is being compared; the historical July results were not remeasured in October.

| ruler | instrument version | workload version | judge/configuration version |
| --- | --- | --- | --- |
| R1, historical must-find recall proxy | [`scripts/mustfind_recall.py`](../scripts/mustfind_recall.py), instrument v2; Git blob `fddb3fbb5727ca3955710955456d2e63b5964dff` | `intents_v1.jsonl`: first 12 broad intents; `gold_search_pools.jsonl`: 12 frozen pools, 111 entries | v2 matcher: normalized DOI first, then normalized title containment (shorter title ≥ 4 words) or token Jaccard ≥ 0.8; no LLM judge; sequential default `search_papers` calls, empty serve retried once then quarantined/excluded |
| R2, current canon coverage | `scripts/snapshot_catalog.py`, F0 corrected artifact; SHA-256 `01b328e596f0ce5b445bd53e99977d6bc9e29fcb95e9d9523a86ac03b8b1da57` | Same frozen landmark pools as R1 | F0 deterministic matcher: stripped/lowercase DOI, then exact title after lowercasing/removing non-ASCII-alphanumerics; no LLM judge; coverage definitions below |
| R3, historical precision/ranking | [`scripts/search_relevance_baseline.py`](../scripts/search_relevance_baseline.py) rubric/served-list collector; Git blob `79e592440eb4dc3ea47e02bc26ac6dd37d74ce0b` | First 12 broad intents from `intents_v1.jsonl`; July 18 served order `srb_v1_0718`, 180 papers | Frozen canonical config `opus-max-blind-2026-07-18`: Claude opus, reasoning effort MAX, 12 fresh contexts (one per intent), rubric-only blind panel; title+abstract, relevant ≥ 0.7. The script's MiMo judge is a dev smoke, not this canonical judge |

Workload Git blobs at the reference commit:
[`intents_v1.jsonl`](../src/papervault/eval/intents_v1.jsonl) =
`6744ad5d4f64247986830202645d4b91c89ce867` (24 intents total, first 12 broad used here);
[`gold_search_pools.jsonl`](../src/papervault/eval/gold_search_pools.jsonl) =
`e204cf641bbe4b5bf889791b963b55d093817988`. The legacy search default budget was 15 unless
the intent parser suggested a count (`PAPER_SEARCH_DEFAULT_LIMIT=15`); the R3 panel contains
180 served papers (15 per intent on average). The drivers do not pin a caller-supplied result
count. The exact opus model release was not recorded in the historical baseline; the configuration above
preserves the recorded model family and effort, without inventing a release identifier.

| frozen historical measurement | value | measurement/configuration |
| --- | --- | --- |
| R1 proxy: must-find recall | **0.2361 mean / 0.00 min** | 2026-07-19, instrument v2, `mfr_v2_0719`; audited served lists, lenient matcher, empty-serve retry/quarantine; single shot |
| R3: P@served | **0.9111 mean / 0.7333 min** | 2026-07-18, `srb_v1_0718_claudejudge_max.json`, canonical opus MAX blind panel; 180 served papers over 12 intents |
| R3: ranking | **P@5 0.9667 mean / 0.60 min; nDCG@15 0.9828 mean** | Same July 18 served order and canonical relevance scores |
| R3: judge-side spread | **±0.03** | Same 180 papers: MiMo 0.9056, opus default effort 0.9444, opus MAX 0.9111; default/MAX agreement 170/180 |

These preserve the legacy rulers. R1's historical pool recall is a proxy for the constitution's
named-need recall; R1 and R3 came from different dates/served lists, so they are not a paired
precision/recall run. The historical record does not establish a fixed resolved budget for every
R1 intent, human-label calibration, or an abstention rate. A future constitution-aligned ruler
must version those additions and the mandatory PAMELA/AMS-02 fixture separately; none is claimed
as newly measured here. The earlier sections retain their historical run conditions and tags.

For R2, **metadata** means a pool entry matches a parsed `Paper` record; **readable** means that
record has an existing `extracts/md/<key>.md` or `extracts/txt/<key>.txt`; **indexed** means its
key appears as a `.md`/`.txt` file-path basename in a LightRAG `doc_status` row marked `processed`.
This legacy instrument checks file existence and processed-path membership, not text quality,
content hashes, or revision-compatible publication. It counts pool entries, not deduplicated
works, and its match rule differs from the historical R1 lenient title matcher.

P4 adds `scripts/coverage_report.py` as a separate version-qualified instrument. It preserves
the frozen pools and denominator, reports the legacy comparison separately, and reads current
catalog-selected inputs, recipes, inventories, publications and observations. The owner-approved
headline counts **full-text publications only** with exact readable evidence and fresh verification
of every required output; abstract and metadata tiers are labeled separately. Observation age is
at most **300 seconds**, measured from `observed_at`. Unverified legacy material stays unknown and
retained in the denominator until verification/rebuild. The report preserves exact pool-entry and
unit/version differences, including missing required sections. The frozen artifact does not supply
complete historical per-entry identities or receipts; unavailable historical provenance is explicit.
Neither this instrument nor numerical agreement rewrites the F0 ruler or proves the P4 handoff.
See the [P4 evidence procedure](specs/2026-10-20-p4-knowledge-handoff.md).

| frozen landmark pool | entries | metadata | readable | indexed |
| --- | --- | --- | --- | --- |
| broad-cr-spectra | 8 | 8 | 8 | 8 |
| broad-equation-discovery | 10 | 7 | 7 | 7 |
| broad-galactic-cr-transport | 9 | 7 | 7 | 7 |
| broad-heliosheath-acr | 10 | 9 | 9 | 9 |
| broad-heliospheric-obs | 9 | 9 | 9 | 9 |
| broad-imf-turbulence | 10 | 9 | 9 | 9 |
| broad-pinn-solvers | 10 | 8 | 8 | 8 |
| broad-positron-dm-excess | 10 | 8 | 8 | 8 |
| broad-sde-solvers | 6 | 5 | 5 | 5 |
| broad-sep-gle | 10 | 8 | 8 | 8 |
| broad-solar-modulation | 9 | 9 | 9 | 9 |
| broad-uhecr-air-showers | 10 | 5 | 5 | 5 |
| **Total (entry-weighted)** | **111** | **92 (82.88%)** | **92 (82.88%)** | **92 (82.88%)** |

The **19 unmatched landmark entries are a canon-seeding follow-up, not a P0 failure**. They
are unmatched by this frozen DOI/title rule; the count alone does not prove absence from the
vault. P0 records the measurement, without seeding papers or changing serving behavior.

The catalog observation contains **26,684** parsed records: 22,417 `ok`, 3,941 `metadata_only`,
151 `extract_failed`, 161 `failed`, and 14 `pending`. It includes 26,674 canonical records and
10 aliases, with 41 quarantined records; identifier counts are 19,951 with DOI and 10,512 with
arXiv ID. The raw duplicate groups (10 DOI, 10 arXiv) include all parsed records, without alias
resolution. The JSON also preserves every source/year/state count. Files: 22,560 PDFs
(76,595,455,518 bytes), 22,329 Markdown extracts (1,564,032,680 bytes), and 3,666 text extracts
(291,732,310 bytes). PostgreSQL reports 43,151 processed / 11 failed doc-status rows and ledger
counts `done=38955`, `done_abstract=4194`, `done_meta=168`, `processing=1`; these are separate
legacy observations, not interchangeable paper-coverage counts.

Topology remains the repository-described deployment: one `papervault.service` MCP process,
independent MinerU OCR service, and `ks-postgres` / `ks-neo4j` stores (see
[architecture](architecture.md)). Timing references remain the dated measurements above:
search 319 s mean (222–525 s), query 187 s with synthesis, search-to-full-text ≈ 10 min, and
search-to-queryable ≈ 20–30 min. No new October latency or topology probe was performed.

Snapshot usage (operator only, with the project's dependencies installed):

```bash
.venv/bin/python scripts/snapshot_catalog.py
```

The verbatim tool is specific to deployment #1: it reads `/data/paper-vault/index.json`,
PDF/extract directory entries and sizes, and the pool file at
`/home/leon/lab/papervault/src/papervault/eval/gold_search_pools.jsonl`. Its sole file write is
the report selected by `--output`, defaulting to
`/data/paper-vault/snapshots/catalog-snapshot.json`. Repeated default runs replace that
file; choose a dated `--output` to retain separate baselines. The output parent must
already exist and be writable. It never constructs `Library`: records are parsed
with `Paper` (invalid records are skipped). Both PostgreSQL connections use
`default_transaction_read_only=on` and issue only `SELECT`s. Supply credentials through the
existing operator environment, never committed files. The aggregate query accepts
`PAPERVAULT_POSTGRES_DSN`, then `POSTGRES_DSN`, then `POSTGRES_*` components; the indexed-key
query uses only components (`POSTGRES_USER=ks`, `POSTGRES_HOST=127.0.0.1`, `POSTGRES_PORT=5432`,
`POSTGRES_DB=knowledge_system`, plus `POSTGRES_PASSWORD`). Ensure those components identify the
same store even when a DSN is set. Aggregate-query errors are recorded; indexed-key query
failures are silently ignored by this frozen script, so a successful process exit alone does
not validate indexed coverage. The orchestrator reported a successful live run for the frozen
artifact; reruns require these same input paths and database settings.

## P5b isolated recovery drill — 2026-10-21

The final staging run [p5b-20261021-staging-04](baselines/p5b-20261021-staging-04/results.json)
restored and replayed **100 distinct parent papers** in empty isolated targets.
It used the census of distinct titles in the tracked 12 landmark pools, preserving
111 pool memberships, 91 papers with DOI and 9 without, and decade strata from
the 1940s through the 2020s. The [sample inventory](baselines/p5b-20261021-staging-04/sample.json)
records every parent, source hash, stratum and omission. This is metadata evidence:
authentic abstracts/full text, operator documents/notebooks/textbooks, section-built
parents, extraction-engine/OCR strata and real graph/vector derivations were
unavailable. One synthetic 3D vector row and two graph nodes/one edge tested physical
store restoration; they do not count toward the 100 parents or qualify those missing
strata.

| Measurement | Final staging result |
| --- | ---: |
| Source admission through normal claims and publication | 92.966 s |
| Coordinated snapshot and verification | 3.324 s |
| Restoration, including physical backend readback | 7.619 s |
| Replay, including new receipts and final readback | 110.815 s |
| Whole staging instrument | 220.187 s |
| Peak worker RSS | 1,223,016 KiB |
| Sampled peak allocated staging bytes, including reservation | 646,701,056 |
| Owned database bytes at completion | 318,423,040 |
| External model/GPU calls and external model cost | 0 / 0 / $0 |

Filesystem usage was sampled every 0.1 seconds; database-server peak resources and
host energy cost were not measured. These figures are **not a full-system RTO**.
The explicit staging root was `/tmp/papervault-p5b-cont.arVV5T/staging-04`. The dedicated
`p5b_catalog_test` database and task-owned loopback Neo4j containers supplied isolated
stores. The [environment record](baselines/p5b-20261021-staging-04/environment.json)
names every schema, endpoint, code hash and command. Staging source, package and
target directories are disposable reproducible fixtures; their durable manifests,
receipts and measurements are committed here.

```bash
.venv/bin/python /tmp/p5b-test-env.py .venv/bin/python scripts/drill_recovery.py \
  --root /tmp/papervault-p5b-cont.arVV5T/staging-04 \
  --evidence docs/baselines/p5b-20261021-staging-04 \
  --drill-id p5b-20261021-staging-04 \
  --neo4j-source bolt://127.0.0.1:32782 --neo4j-target bolt://127.0.0.1:32783 --apply
```

The [snapshot manifest](baselines/p5b-20261021-staging-04/snapshot-manifest.json),
[backup inclusion](baselines/p5b-20261021-staging-04/backup-inclusion.json),
[recipes](baselines/p5b-20261021-staging-04/recipes.json),
[restore receipt](baselines/p5b-20261021-staging-04/restore-receipt.json),
[replay receipt](baselines/p5b-20261021-staging-04/replay-receipt.json),
[verification log](baselines/p5b-20261021-staging-04/verification.log) and
[hashes](baselines/p5b-20261021-staging-04/hashes.json) establish inclusion of all seven
components, all 11 PostgreSQL derivative roles and the graph fixture. Replay used
100 original metadata inputs/recipes and generated 100 new normal receipts in a
permanently non-promotable sample generation. Source serving selection was unchanged.
No failures occurred in this final run. The [first attempt](baselines/p5b-20261021-staging-01/failure.json)
stopped on a missing-DOI payload validation error before capture; its traceback and
hashes remain recorded. The [second run](baselines/p5b-20261021-staging-02/results.json)
succeeded before subsequent verifier/restore hardening and is historical evidence,
superseded by the [third run](baselines/p5b-20261021-staging-03/results.json).
Run 04 repeats the drill after the catalog index and resume corrections; earlier
runs remain historical evidence and their packages lack format-2 index capture.

[Catalog index evidence](baselines/p5b-20261021-staging-04/catalog-indexes.json)
records all **67 installed index definitions** at the checkpoint and their restored
counterparts. The sample now enters through the normal importer, which installs
DOI/arXiv partial unique indexes. Restore verifies the exact set/definitions; the
dedicated PostgreSQL regression also proves duplicate DOI/arXiv insert rejection.
Nonconstraint index removals or redefinitions are restored before loading rows that
would violate a stronger bundled default. Constraint-index incompatibility fails
atomically without accepting a weaker target.

[Validation evidence](baselines/p5b-20261021-staging-04/validation.json) records the
final commands, exit codes, counts, code identities and logs for the full
non-integration suite, dedicated PostgreSQL tests, separate disposable Neo4j tests
and Ruff. The [regression log](baselines/p5b-20261021-staging-04/regression-evidence.log)
retains failing-then-passing results for installed indexes, checkpoint binding,
resume row/schema changes, and the recorded isolated quarantine-reset policy.
The [earlier suite failure](baselines/p5b-20261021-staging-04/suite-before-deadline-test-fix.log)
records an existing 90 ms whole-operation deadline assertion exceeded by setup and
cleanup. The test now asserts export interruption directly, retaining the real
timer and retention guards; a disabled-timer mutation fails the amended assertion.
The [scoped code review](baselines/p5b-20261021-staging-04/scoped-review.log) found no
remaining correctness findings after two additional regressions were fixed; it
also accepted the deadline-test correction. It assessed code and assertions, not
the later drill/validation evidence. Current-head
CI and the independent integration verdict remain separate evidence owned by the
orchestrator.

Operational storage remains `/data/p5b-recovery` and `/data/p5b-drills`, with the
2 TB derivative soft cap and 80% alert. The orchestrator owns the weekly operational
snapshot activation and production drill in the monthly reboot window (at most two
hours). Weekly snapshots with four rolling complete sets do not establish a measured
RPO. Same-host RAID copies provide no off-host or host-failure/disaster recovery
guarantee. The [operational procedure](../README.md#coordinated-recovery) records
drain requirements, retention protection and these limits; this run made no live
activation or production-window claim.

## P5c isolated maintenance rehearsal — 2026-10-21

The [P5c rehearsal](baselines/p5c-20261021-staging-01/results.json) exercised
disposable-extract GC and candidate rebuild/promotion in a task-owned PostgreSQL
container and new `/tmp` roots. It used four synthetic extraction receipts sharing
one source and three metadata parents. No live catalog, live vault, Neo4j store,
external model or GPU was used.

| Measurement | Staging result |
| --- | ---: |
| Whole instrument | 1.445 s |
| GC window work, including injected failure and retry | 0.076 s |
| Candidate rebuild and guarded promotion | 0.482 s |
| Recorded window allowance, per catalog | 120 s |
| Observed serving interruption | 0 s |
| Preserved extraction receipts | 4 |

Serving availability was checked by reading an eligible serving manifest after
each candidate publication and refused promotion; it was not continuously probed.
The [GC evidence](baselines/p5c-20261021-staging-01/gc.json) records the real
ninety-day refusal, protected/current refusals, a committed deletion intent after
an injected unlink failure, and successful retry under the same deletion identity.
Only the disposable markdown disappeared; source, protected/current output hashes
and all receipts remained. Superseded fixture rows were explicitly aged by a
test-only database update after proving the real retention refusal.

The [rebuild evidence](baselines/p5c-20261021-staging-01/rebuild.json) records
capacity exhaustion with zero new claims and unchanged file inventories. Three
funded metadata jobs then rebuilt an isolated candidate. Candidate publications
kept the prior serving manifest, incomplete/stale promotion requests were refused,
and exact current revisions promoted the complete compatible candidate. All
refusals and applied commands appear in the
[action record](baselines/p5c-20261021-staging-01/actions.json).

```bash
P5C_DRILL_DSN='host=127.0.0.1 port=32786 dbname=p5c_drill user=p5c_drill' \
  .venv/bin/python scripts/drill_derivative_maintenance.py \
  --dsn-env P5C_DRILL_DSN \
  --root /tmp/papervault-284.9HQAzZ/rehearsal/final/source-root \
  --evidence /tmp/papervault-284.9HQAzZ/rehearsal/final/evidence \
  --drill-id p5c-20261021-staging-01 --disposable
```

The [environment record](baselines/p5c-20261021-staging-01/environment.json)
identifies the disposable endpoint, schemas, package version and code hashes.
The [verification log](baselines/p5c-20261021-staging-01/verification.log) and
[artifact hashes](baselines/p5c-20261021-staging-01/hashes.json) retain the checked
result independently of the disposable container and staging files.
The initial [validation record](baselines/p5c-20261021-staging-01/validation.json) binds
commands, exit codes and source hashes: **3,932 passed** in the full non-integration
suite, **693 passed** in the dedicated PostgreSQL/store run, and Ruff passed.
The suite retains two existing skips and excludes 167 integration cases; the
broader store run skips four existing Neo4j cases without supplied endpoints.
The [regression record](baselines/p5c-20261021-staging-01/regression-evidence.log)
explains the repaired unit fixture, with its original assertions retained.
Both initial [scoped reviews](baselines/p5c-20261021-staging-01/scoped-review.log) found no
remaining blockers. The task-owned container was removed after verification;
[fixture disposition](baselines/p5c-20261021-staging-01/fixture-disposition.json)
identifies the preserved evidence and disposable staging files.

A [second rehearsal](baselines/p5c-20261021-staging-02/results.json) exercised the
same protocol with serialized capacity accounting and selection-based retention
clocks. The instrument took **1.296 s**, with **0.034 s** of GC window work and
**0.348 s** for candidate rebuild/promotion, within each **120 s** window. Observed
manifest-read interruption remained **0 s**, using the same sampling method.
Its [environment](baselines/p5c-20261021-staging-02/environment.json),
[actions](baselines/p5c-20261021-staging-02/actions.json),
[GC](baselines/p5c-20261021-staging-02/gc.json) and
[rebuild](baselines/p5c-20261021-staging-02/rebuild.json) records retain exact code
hashes, refusals and preservation evidence independently of the removed fixture.
The [regressions](baselines/p5c-20261021-staging-02/regression-evidence.log) show
failing-first admission/settlement interleavings, deferred-job restart recovery,
foreign reservation settlement, excluded-entry reader admission and last-selection
retention, including initially unbound migrated vaults. The
[scoped source review](baselines/p5c-20261021-staging-02/scoped-review.log) found no
remaining critical or important findings; it did not provide an integration verdict.
The second [validation record](baselines/p5c-20261021-staging-02/validation.json)
binds the corrected code to **3,950 passed**, two skipped and 167 deselected in the
non-integration suite, **703 passed** and four Neo4j skips in the dedicated
PostgreSQL/store run, and a passing Ruff check. The four dependency deprecation
warnings match the baseline. [Validation hashes](baselines/p5c-20261021-staging-02/validation-hashes.json)
cover the retained evidence; the first rehearsal's measurements remain unchanged.

This synthetic metadata rehearsal does not qualify full-text/graph/vector rebuild
cost, production downtime, full-system RTO or host-failure recovery. Defaults remain
legacy.

## Arbitration rules (how these numbers may change)

1. **Quality is a hard constraint**: a retrieval-affecting change flips its default only
   if the paired recall delta stays within the 0.0132 noise floor (or improves).
2. **Arbitration window = corpus-freeze window**: pause ingestion (stop the service or
   disable auto-ingest) for the duration of any paired eval — live distills contaminated
   a verdict once (2026-07-18; see the papervault issue #4 thread).
3. **Same-day paired, identical knobs both arms** (`KS_RERANK_MAX_LENGTH=4096` explicit —
   historical drivers defaulted to a wrong 1024).
4. **Component instruments** (the eval harness in `src/papervault/eval/`: gold sets,
   backbone, judges, FULL arm) are the sensitive tools for small-delta arbitration; this
   card is the product-level truth. See `docs/eval.md` for the two-tier regime.
5. **Paired AND interleaved, ≥ 3 reps/arm** for FAST-gold arbitration: within-arm rep
   spread (LLM-decompose non-determinism) measured ~3× the historical run-level noise
   floor (2026-07-18 rerank pair: one arm spanned 0.8169–0.8537 across reps), so a
   single-shot A/B is not decisive; per-rep paired deltas are.
