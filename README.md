# papervault

A self-hosted **literature + knowledge layer for coding agents**, delivered as one MCP
server. Point any MCP-capable agent (Claude Code, and others) at it and the agent gains
three tools:

- **`search_papers`** — discover literature by describing your research intent in prose;
  relevant finds are auto-ingested (downloaded, OCR'd, full text on disk).
- **`get_paper`** — look up papers by citation key / DOI / arXiv id / fuzzy title; read
  verbatim full text from a returned path.
- **`query`** — ask the knowledge base a question, get a cited prose answer synthesized
  from a knowledge graph distilled from everything the library has ingested.

The corpus compounds without babysitting: search → queryable knowledge in tens of minutes,
self-healing ingest queues, cited answers grounded in real sources.

> **Status:** pre-release (closed beta). See `docs/PRD.md` for scope and `docs/architecture.md`
> for structure. This is industrial-grade infrastructure, not a toy — read the hardware floor
> before installing.

## Hardware floor (not optional)

papervault runs its own embedding, reranking, and OCR models locally. There is **no GPU-less
or degraded mode** — below the floor, this product is not for you.

- **1× NVIDIA GPU, 24 GB** (RTX 3090 / 4090 class). A single card runs the full stack
  (embed + rerank + OCR co-tenant) inside a measured VRAM budget; a second GPU is comfort,
  never a requirement.
- **Linux + CUDA.**
- **An OpenAI-compatible LLM endpoint + key**, supplied by you (graph build, search
  triage, and answer synthesis all route to it). papervault bundles no keys.
- **Docker** (for the Neo4j + Postgres backing stores).

## Quick start

See [`docs/INSTALL.md`](docs/INSTALL.md) for the full walkthrough. In brief:

```bash
cp .env.example .env          # then fill in your LLM endpoint + key + data dir
docker compose --env-file .env -f deploy/docker-compose.yml up -d   # Neo4j + Postgres
uv venv && uv pip install -e ".[mineru,grey]"       # beta standard: + local OCR + grey download tiers
papervault doctor             # verify GPU, databases, LLM config
papervault-mcp                # start the MCP server (== papervault serve)
```

Then register the server with your agent (Claude Code: install the bundled plugin; other
clients: point them at the MCP endpoint — see `docs/INSTALL.md`).

## Domain

papervault ships tuned for **space physics + AI4Science** as the factory default and worked
sample. The domain layer — search-gate rubric, entity ontology, extraction few-shots,
instruction text — lives in editable config files (`src/papervault/domain/`). Adapting to
another field means editing those files; quality outside the shipped domain is the adapting
operator's responsibility.

## URL identity backfill

For records with a URL but no DOI or arXiv ID, inspect a bounded sample without
writing to the vault:

```bash
python -m papervault.library.cli identity-backfill --cap 20 --json
```

Recovery checks DOI/publisher and arXiv URL patterns, then one citation-metadata
landing fetch (including its redirect target), then exact ADS, CORE, OpenAlex or
INSPIRE metadata. DOI registry metadata or the current arXiv page must pass the
existing title/author consistency screen before an identifier is filled. Missing,
blocked, ambiguous or conflicting evidence causes abstention. Existing identifiers
and collisions are left intact; successful fills retain route, URL, verified title
and UTC observation time in `identity_recovery`.

In an authorized maintenance window after deployment, add `--apply --acquire` to
persist identities and immediately run the ordinary current-only cascade on only
the newly recovered rows without existing assets. `--keys FILE` restricts selection
to listed citation keys (one per line). `--cap` is required, accepts 1–50, and bounds
both inspected and acquired records; unrelated terminal sets are never reset.
Writes refuse while `papervault.service` is running. The JSON result and library
manifest record recovery counts by route, acquisition outcomes, and successful
sources (including the winning member of aggregator groups). Report that post-deploy
pass's numbers on [issue #197](https://github.com/LeonJoeeee/papervault/issues/197).

Download telemetry is appended to the same manifest by ordinary acquisitions. Existing
outcomes retain their fields and gain `timing`: a 32-character `run_id`, UTC `at`,
and, inside a tier slot, `slot_id`, `slot_started_at`, `slot_ended_at`, and monotonic
`slot_duration_s`. These slot endpoints describe elapsed time **at outcome emission**;
use the corresponding completed `tier` span for the full slot including final bookkeeping.

Additional `download_telemetry` records share `run_id`, paper `key`, `actor`
(`download` or competing `search`), UTC `at`, process/thread IDs, and `phase`/`mark`.
Spans have `start` and `end` marks with `span_id`, `parent_span_id`, `started_at`,
`ended_at`, monotonic `duration_s`, and control-flow `status` (`ok`, `error`,
`cancelled`). Status `ok` means the operation returned; tier success remains the
existing outcome event. Phases cover `queue_worker`, `network_wait`, `network_hold`,
actual `executor` work, `cascade`, arXiv `preflight`, `tier`, `strategy`,
`verification`, `pdf_save`, `firecrawl`, parallel `group`/`member`, and `browser_call`.
`enqueue`/`dequeue` points pair by queue sequence and run ID; dequeue includes
`enqueued_at`, `queue_wait_s`, worker ID and priority. `queue_drop` points close
stale priority-promoted tuples with their sequence and dequeue time. Cascade end carries
`paper_status`, `outcome_source` and `pdf_returned`; group `winner` points precede
executor drain. Mirror members use indices, never target URLs. Telemetry stores
scalar fields only, bounds labels to 64 characters and keys to 128, and retains
only one admission stamp per queued tuple. New-record failures emit a warning and
do not change acquisition results. Manifest writes serialize complete JSONL records.

The semaphore spans describe **whole cascade/search admission**, including browser,
verification and executor drain; member fan-out can issue multiple requests per
permit. Browser-call spans observe the exposed fetch API, not process creation/exit,
PSS/RSS or isolated remote latency. Cancellation can release a permit before the
executor thread ends; measure both timelines. Partial spans at capture boundaries
must be censored. Request counts/status/rate headers and same-window host process
memory measurements remain necessary to validate upstream limits and resource costs.
This instrumentation changes neither the four workers nor the four semaphore permits.

## Releases

papervault is consumed **by pin** (ADR-0001): deployments track an annotated git tag
(`vX.Y.Z[-stage]`), never `main`.

The repo carries **one version**, and every change PR bumps it per the semver call stated in
its description. The declared version fields move together in that PR: `pyproject.toml`
`[project].version` plus the `papervault` entry in `uv.lock` (refresh with `uv lock`),
`plugin/.claude-plugin/plugin.json` `version`, and the `papervault` entry's `version` in
`.claude-plugin/marketplace.json`; `tests/test_plugin_version_sync.py` fails when they drift.

A release IS a tag — CI's build job produces the wheel/sdist for every commit, so tagging is
the whole ceremony: the tag names the version already on `main` (`vX.Y.Z`, plus an optional
`-stage` suffix), and a release commit changes no version field. Latest tag: `v0.1.6-beta`
(closed beta), cut before the one-version rule, so its number predates the package version.

## License

MIT — see [`LICENSE`](LICENSE).
