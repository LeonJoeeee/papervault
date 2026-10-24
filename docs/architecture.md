# papervault — Architecture

> Shared baseline for all parallel work. Read before any task.
> Changing anything here = touching the core: public merge + human approval + an ADR.

## Context

papervault is a self-hosted literature and knowledge layer exposing one MCP endpoint
with `search_papers`, `get_paper` and `query`. It connects coding agents to scholarly
sources, a retained paper vault and a cited graph-RAG knowledge base. One Python
server uses an operator-supplied OpenAI-compatible LLM endpoint, local GPU embedding
and reranking, an independently managed MinerU HTTP OCR service, PostgreSQL and
Neo4j. Linux/CUDA and one 24 GB-class NVIDIA GPU define the hardware floor. The
[ratified constitution and refactor plan](adr/0007-papervault-constitution.md) govern
collection integrity and useful retrieval as joint primary contracts.

## Structure

One package, one server process, with these ownership boundaries:

| Component | Responsibility and boundary |
| --- | --- |
| `mcp/` | The three-tool public interface, progress heartbeats and request admission. Agent plugins register the endpoint and contain no business logic. |
| `library/` | Search coordination, bibliographic identity, registration, acquisition and extraction. Library owns vault writes; registration is a separately admitted command. |
| `store/` | Transactional catalog authority, revisions, audit/outbox, publication lineage, recovery and derivative lifecycle/capacity. JSON exports are projections under PostgreSQL authority. |
| `knowledge/` | Immutable knowledge intake, distillation scheduling, LightRAG backend adapters, evidence retrieval and synthesis. Catalog commands accept verified outputs; backend computation alone grants no publication authority. |
| `domain/` | Configured search rubric, entity ontology, extraction examples and instructions; space physics/AI4Science is the factory domain. |
| `eval/` | Gold workloads, scoring and shared response assembly; reference-machine retrieval evidence complements CI's deterministic scoring tests. |

Persistence has three logical roles: **Source** retains research/input bytes;
**Catalog** owns identity, policy, registration and processing/publication facts;
**Derived** holds extracts, vectors, graph data and indexes bound to exact inputs
and recipes. These are authority roles, not three required database engines.
Library and knowledge ship together; their internal interfaces are versioned as
one package. The MCP contract is the compatibility surface.

### Search, query and lifecycle contracts

**S1 — search and acquisition.** `SearchPipeline` separates intent, plan, retrieval,
fusion, return gate and serving using request-local snapshots and bounded traces.
The post-parse guard clears a lone `year_min` for canonical wording, or survey wording
without explicit recency, preserving foundational candidates. External adapters
(arXiv, ADS, Semantic Scholar, OpenAlex, INSPIRE and CORE) distinguish genuine zero
hits from structured `BackendDegraded` outcomes. Polite-v2 search admission/fetch
uses per-`(event_loop, backend)` locks for migrated arXiv/Semantic Scholar/OpenAlex
calls; politeness denials and executor queue waits do not spend backend-breaker
failures. Cancellation cannot establish that a synchronous executor has drained.

Fusion resolves aliases and compatible bibliographic duplicates while retaining
external rank evidence. Local BM25/RRF selection and final `by_relevance` ordering
both receive the configured authority prior: citation velocity multiplied by a
domain-aware venue classifier combining venue names, DOI prefixes and fallback
evidence (1.5x / 1.2x / 1.0x / 0.5x). Explicit recency/importance bypasses it.
Authority cannot admit a gate reject; accepted named-entity witnesses receive
reserved result slots. Registration candidates are selected independently of the
served cap, and serving never waits for their command admission or acquisition.
The [operator reference](../README.md#search-and-catalog-quality) describes the
operational stages, pool limits, gates and knobs; [ADR-0008](adr/0008-search-authority-venue-and-garbage-fencing.md)
records the decision.

The two-way quality fence rejects new garbage through `registration_rejection`
and excludes bad retained rows through `serving_excluded` / positive
`deterministic_garbage_reason` evidence. Domain exclusions, policy and quarantine
also fence serving; missing legacy metadata alone does not. Identity/alias rows
and research assets remain retained. A `Paper`'s acquisition eligibility is enforced
by `state_v2.admission`: eligible retrieval, canonical resolved identity, applicable
policy, no quarantine and valid lease/budget evidence. Scoped policy rulings cannot
silently release quarantine. Campaign claims reserve durable budgets before dispatch;
generation/lease fences guard staged publication. Expiry and cancellation never
prove executor termination or refund uncertain work. Merge intents stop admission
and require drain before canonical/alias consolidation. Persisted fences survive
flag changes. Routine purge is refused. Official ADS scans defer identity checking
to a hash-bound post-OCR verdict before extraction publication.

**S2 — knowledge and query.** The framework spine is exactly
`lightrag-hku==1.5.7`, with Neo4j graph and PostgreSQL KV/vector/doc-status stores.
BGE-M3 embedding and BGE reranking run in process. [ADR-0005](adr/0005-track-latest-lightrag-line-gated.md)
requires exact-patch adapter guards, a port map, prompt-content gates, queue-deferral/
recovery tests, ontology probe and FAST recall evidence for upgrades. Adapters
preserve LightRAG reducers, ingress/owner arguments, deferred context, retirement
reports and extraction-cache contracts. Catalog status updates obey namespace
fences; raw-workspace repair is refused in favor of isolated rebuilds.

A catalog query pins one single-generation `serving_manifest` across retrieval,
traversal, caches, aggregation and synthesis. Reader pins protect that view; policy,
compatible inputs and fresh evidence are revalidated against it before return,
without switching to a newly promoted generation. Multiquery overlaps decomposition
with anchor retrieval, drains cancellation and gates fan-out on coverage. Evidence
prioritizes canon, full text, then abstracts; canon comes from textbooks or explicit
annotations. Production and eval share synthesis and the pure response builder.
`retrieved_papers` identifies candidates; `cited_papers` contains only verified inline
`[paper_key]` citations, with operator documents separate. Empty coverage skips
synthesis; abstract-only evidence is capped at `"weak"`. FAST `--no-synth` uses the
retrieved-paper citation fallback, not a synthesis-quality measurement.

### Authority, publication and derivatives

**P2 — catalog authority.** PostgreSQL library saves commit revision-guarded record
deltas, audit and pending work atomically, with durable bytes before references.
Conflicts fail explicitly; record deletion is refused. Configuration defaults to
JSON, while deployment #1 explicitly selects `papervault_catalog` as authority.
Its hourly verified `index.json` export is a projection. PostgreSQL failures never
fall back to JSON. Authority handoff fences writers; recovery after writes resume
preserves accepted mutations and proceeds forward. See the
[persistence decision](adr/DRAFT-catalog-persistence-cutover.md).

**P4/P5a — verified publication.** Knowledge implementation selection defaults to
`legacy`; `postgres` requires its own rehearsed, committed authority handoff, not
just a flag. Catalog intake captures immutable inputs and recipes; jobs, attempts,
submissions, inventories, receipts and selections have separate identities. Attempts
write private LightRAG deltas over one pinned aggregate, using latest-row replacement
and tombstones. Verified sealed receipts precede atomic lineage/publication/audit/
outbox acceptance. Candidates do not move production selection. Immutable extraction
input/output/receipt bundles precede registration; legacy pending bundles grant no
catalog coverage. Recovery delivers original evidence without re-execution or
fabricated provenance. Complete coverage requires compatible selected publications
and fresh positive output observations, not backend `processed` alone. See the
[publication decision](adr/DRAFT-catalog-knowledge-publication.md).

**P5b/P5c — recovery and derivative lifecycle.** Coordinated sets bind catalog,
retained sources, inputs, receipts and physical PostgreSQL/Neo4j outputs to one
verified checkpoint under drained writer barriers. Isolated restore/replay creates
new fences; sample generations cannot promote. Retention protects source bytes,
receipts, selections and reader/job/recovery dependencies. Only explicitly disposable
extract outputs superseded for ninety days can enter fenced, resumable GC; permanent
deletion intents serialize with new references. Rebuilds fork unselected candidates
and promote only complete compatible freshly verified manifests with exact revisions.
Shared reservations and fresh external usage accounting stop unfunded new intake/
execution at the 2 TB derivative cap, with an 80% alert and no emergency asset
purge. Sealed-receipt recovery remains available. Applied maintenance uses bounded
monthly windows. Same-host recovery sets do not establish host-loss recovery or
full-system RPO/RTO. [Operator procedures](../README.md#coordinated-recovery) and
[dated baselines](baselines.md) carry commands, qualification limits and measurements.

## Key quality goals

1. **Collection integrity and useful retrieval** — preserve durable identity/evidence while measuring recall, precision and availability with separate rulers.
2. **One authority per fact** — atomic catalog mutations and verified publication prevent exports or derived computation from becoming competing authorities.
3. **Fail-closed, recoverable execution** — explicit degradation, durable fences and retained uncertainty prevent silent acceptance, duplicate dispatch and destructive repair.
4. **Single-GPU operation** — bounded admission, concurrency and capacity preserve the published hardware envelope.
5. **Stable agent interface** — one MCP surface and thin client shells keep behavior inside the shared package.

Decisions and their reasons: `docs/adr/`.
