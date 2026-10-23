# 0005 — Track the latest LightRAG minor line, behind a port-map + gate discipline

Status: Accepted (2026-07-17)

## Context
LightRAG is papervault's framework spine and moves fast. Two failure modes were both
observed in one day: an UNBOUNDED floor (`>=1.4`) let a clean install drift to an
unvalidated 1.5.4 where the domain ontology and noise-suppression silently no-op'd
(retro-review blocker); and a hard pin to 1.4.x would freeze papervault off the
framework's bugfixes and features indefinitely. The user's call: stay on the latest
line — but never blindly.

## Decision
Pin to a CEILING-BOUNDED minor line (currently `lightrag-hku>=1.5,<1.6`). Raising the
ceiling to a new line is a gated port, never a version bump:
1. a fresh **port map** over the known touchpoint checklist (extraction-prompt channels,
   addon_params consumption, llm kwargs, ctor args, rerank interface, doc-pipeline APIs +
   DocStatus states, workspace envs, QueryParam fields, storage/schema migration compat);
2. the **machine-checkable prompt-content gates** green (`tests/knowledge/test_lightrag_port.py`
   — guidance completeness, escape-hatch kill, fail-loud anchors);
3. a **probe build** on a throwaway workspace asserting zero off-ontology nodes, plus the
   **FAST recall gate** vs the previous line's baseline (delta within the noise floor).
Rejected: floating latest (proven silent regression); freezing on a proven version
(compounding staleness, and the eventual forced jump only gets bigger).

## Consequences
papervault stays current at the cost of one gated port per minor line. The fail-loud
guards in `extraction_prompt.py` convert any missed touchpoint from silent degradation
into a boot-time error. The 1.4.16-era behavior remains the recall reference until a
new baseline supersedes it.

## Amendment — breaker deferral adapter (#148, 2026-10-07)

The maintainer-approved pause adapter binds to private document-pipeline hooks, now ported
to 1.5.7 under #313 (originally 1.5.4).
The package requirement now pins that exact patch, matching the lockfile and runtime version
guard; clean installs must not resolve a newer patch that the adapter has not been ported to.
A patch upgrade must update the adapter guard and pass its queue-deferral/recovery tests as
well as the existing port gates. The bounded minor-line policy remains the upgrade direction.

The 1.5.7 patch port forwards batch ingress/owner arguments and preserves the run context
on deferred status transitions. Catalog retirement preserves the rebuild report and unpacks
the extraction-cache tuple. Catalog status-field updates materialize inherited rows under
the namespace write fence, and layered reads accept the scheduler's single-status pages.
Raw-workspace source-conflict repair is refused for catalog storage; it requires an isolated
rebuild. The preparation schema literals remain unchanged; its reviewed dependency pin moves
with the runtime. Historical 1.5.4 probe/recall evidence remains historical, not a 1.5.7 result.

## Verification — 1.5.7 patch port (#313, 2026-10-23)

The orchestrator ran both live gates using the task lane's `lightrag-hku==1.5.7`
environment. The [host measurement record](https://github.com/LeonJoeeee/papervault/issues/313#issuecomment-6369432590)
contains the exact commands and artifact paths; [PR #315](https://github.com/LeonJoeeee/papervault/pull/315)
preserves the report outputs alongside the port map and machine-checkable gates.

The throwaway probe used `Bonomi2020` in workspace `l0_probe_157` (218,263 raw
characters; 203,829 after reference stripping). It reached `processed` in 94.67 seconds:
**22 chunks produced 579 entities and 582 relations**. Of those entities, **578/579
(99.83%)** belong to the closed 11-type ontology; one is an `UNKNOWN` endpoint
placeholder. There were **zero `Other` entities and zero off-ontology nodes**
(`other_count=0`, `off_ontology={}`); the placeholder is accounted for separately.
The host record confirms that the throwaway workspace was cleaned in both Neo4j
and PostgreSQL after verification.

The FAST comparison used `src/papervault/eval/gold_fast.jsonl` (14 questions:
11 answerable, three traps), `variant=multiquery`, and `--no-synth`, against the
same-corpus 1.5.4 reference:

| Metric | 1.5.4 (`post144_10072338`) | 1.5.7 (`port157_1024`) | Delta |
| --- | --- | --- | --- |
| `gold_citation_recall` / recall@served | 0.733170 | 0.747140 | +0.013970 (+1.40 pp) |
| `paper_recall_at_12_distinct` | 0.503153 | 0.488223 | -0.014931 (-1.49 pp) |

In `--no-synth` mode, `gold_citation_recall` measures recall over distinct served
papers using the retrieved-paper citation fallback. The recall@served gate improved;
the orchestrator assessed the separate recall@12 movement as run-level noise.
All 14 candidate receipts report `rerank_degraded=False`. The retained candidate
rows are `/data/paper-vault/eval-results/port157_1024.jsonl`; offline recalculation
from those rows and the baseline reproduces every aggregate in the host comparison
report. These retrieval measurements do not establish synthesis quality.
