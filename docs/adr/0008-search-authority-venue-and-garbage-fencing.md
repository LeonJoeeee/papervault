# 0008 — Preserve canon and catalog identity with authority ranking and quality fences

Status: Accepted (2026-10-24); records decisions delivered in v0.10.5–v0.10.9.

## Context

Broad survey searches could infer a recent `year_min` and lose foundational papers.
Concurrent polite-v2 admission could turn a local rate-window refusal into a
one-hour backend outage. Local BM25 pool cuts could discard landmark papers before
return judgment; raw citation counts and sparse venue metadata also favored noisy
records. New-registration screening alone left deterministic legacy garbage visible
and eligible for repeated acquisition. These behaviors undermine the joint integrity
and retrieval contract of [ADR-0007](0007-papervault-constitution.md).

## Decision

Preserve useful canon and durable identity through coordinated selection and
admission fences, as delivered in [#307](https://github.com/LeonJoeeee/papervault/pull/307),
[#309](https://github.com/LeonJoeeee/papervault/pull/309),
[#311](https://github.com/LeonJoeeee/papervault/pull/311) and
[#314](https://github.com/LeonJoeeee/papervault/pull/314):

- Guard intent after parsing: clear a lower-bound-only `year_min` for canonical
  wording, or survey wording without explicit recency, and reset inferred recency
  ordering when clearing. Preserve bounded date intervals.
- Serialize migrated search API admission/fetch per event loop and backend. Keep
  `PolitenessDenied` and executor queue waits observable but outside backend-breaker
  accounting; local contention does not establish provider failure.
- Apply authority at **two ranking boundaries**: local BM25/RRF pool selection
  (45 local plus up to 45 additional externally vouched records), then final
  `by_relevance` ordering among accepted candidates. Authority is citation velocity
  times a three-layer venue classifier (normalized exact/fuzzy names, DOI prefixes,
  fallback evidence) with 1.5x / 1.2x / 1.0x / 0.5x multipliers. Default λ is 0.25;
  explicit recency/importance bypasses both priors. Ranking never overrides gates.
- Use `registration_rejection` for newly admitted records and `serving_excluded`
  for retained records. Positive `deterministic_garbage_reason` evidence fences
  conference abstracts, classified non-papers and anonymous authors alongside
  domain/policy/quarantine exclusions. Missing legacy metadata alone is insufficient.
  Keep identity/alias rows and artifacts so rediscovery cannot recreate the same
  excluded identity. Routine purge is refused.

These choices replace breaker accounting that conflates local denial with backend
failure, relevance-only local pool cuts, and admission-only garbage screening.
They do not introduce a new authority store or change the MCP surface.

## Consequences

Landmarks can reach judgment before the result cap, while authority remains a
configurable, domain-specific prior rather than an eligibility or canon verdict.
Quality rejection reasons and source degradation remain inspectable. Retaining
fenced rows costs catalog space and requires serving/acquisition callers to honor
shared predicates. Evaluation records coverage, served recall and hygiene separately;
corpus availability does not guarantee a landmark will be returned for an intent.
See the [operator reference](../../README.md#search-and-catalog-quality) and
[dated baseline](../baselines.md#post-v01010-retrieval-coverage-hygiene-and-lightrag-157-baseline--2026-10-24).
