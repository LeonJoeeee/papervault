# Installing papervault

papervault is industrial-grade infrastructure with a hard hardware floor. Read
[the README's hardware section](../README.md#hardware-floor-not-optional) first —
below the floor, this product is not for you.

## Prerequisites

- **1× NVIDIA GPU, 24 GB** (RTX 3090 / 4090 class), Linux + CUDA.
- **Docker** (for Postgres + Neo4j).
- **[uv](https://docs.astral.sh/uv/)** (Python env + install).
- **An OpenAI-compatible LLM endpoint + key** you supply.

## Steps

```bash
# 1. Clone
git clone <repo-url> papervault && cd papervault

# 2. Configure — copy the template and fill in your LLM endpoint/key + data dir
cp .env.example .env
$EDITOR .env          # set PAPERVAULT_LLM_*, POSTGRES/NEO4J passwords,
                      # PAPERVAULT_CONTACT_EMAIL

# 3. Backing stores (Postgres + Neo4j). --env-file is REQUIRED: with `-f
#    deploy/...`, compose reads deploy/.env, NOT the repo-root .env — pass it
#    explicitly so your POSTGRES/NEO4J passwords reach the containers.
docker compose --env-file .env -f deploy/docker-compose.yml up -d

# 4. Install papervault (beta-standard: core + local OCR + grey download tiers)
uv venv && uv pip install -e ".[mineru,grey]"
#   uv pip install -e .                 # minimal core (no OCR, OA-only downloads)

# 5. Provision MinerU independently before enabling extraction.
#    The [mineru] extra supplies HTTP client libraries, not the vLLM server.
#    On the shared host, mineru.service runs the backend privately on :30001;
#    mineru-gate.service exposes :30000 and owns backend stop/start and recovery.
#    Those services are provisioned outside this repository. papervault does not
#    start, stop or restart them. Do not install a second listener on :30000.
#    The default MINERU_URL stays http://127.0.0.1:30000; set it for another host.
curl -fsS http://127.0.0.1:30000/health
#    Health 200 is ready even with backend=stopped; a workload POST starts it.

# 6. Preflight — verify GPU, databases, LLM config, domain pack
papervault doctor

# 7. Run the server
papervault serve                         # streamable-http on 127.0.0.1:8080
#   papervault serve --stdio             # subprocess transport
```

> Note: `doctor` checks that an LLM key is PRESENT, not that it authenticates — a wrong
> key surfaces only at the first real `query()` (as an auth error).

`papervault doctor` must be green before serving. It checks the .env, data dirs,
the workspace vars, the domain pack, GPU/VRAM, the OCR endpoint + local model
weights, and Postgres + Neo4j connectivity:

- **workspace (FAIL):** `NEO4J_WORKSPACE` / `POSTGRES_WORKSPACE` must both be
  non-empty AND equal (or `papervault serve` aborts at boot); the reserved prod
  name `l0` fails unless `KS_ALLOW_PROD_WORKSPACE=1` is set on purpose.
- **MinerU (WARN):** probes the configured OCR endpoint; if it's down, extracts
  just queue until it comes up — not fatal, so a warning, not a failure.
- **BGE weights (WARN):** the embed + reranker default to `BAAI/bge-m3` +
  `BAAI/bge-reranker-v2-m3` (~4.5 GB total), pulled lazily from Hugging Face on
  first use. `doctor` detects whether they're already local (a directory, or the
  HF cache) WITHOUT downloading; absent ⇒ a warning that the first build/query
  downloads them. Pre-seed or redirect the HF cache with `HF_HOME`, or point
  `BGE_M3_MODEL_PATH` / `BGE_RERANKER_MODEL_PATH` at local weights.

For a deeper check, `papervault smoke` runs doctor, boots the MCP server over
stdio, and round-trips its tools (`list_tools` + a read-only `get_paper`) — no
LLM required (`papervault smoke --skip-db` skips the DB connectivity checks).

## LLM routing and migration

Every text LLM request sends a gateway level: `flash`, `standard`, or `pro`.
The endpoint resolves its model and effort per key. Defaults are:

| Role | Level | Override |
| --- | --- | --- |
| Graph extraction/merge | standard | `PAPERVAULT_LLM_BUILD` |
| Query keyword extraction | flash | `PAPERVAULT_LLM_KEYWORD` |
| Answer synthesis | pro | `PAPERVAULT_LLM_SYNTH` |
| Extract clarity/completeness gates | flash | `PAPERVAULT_LLM_GATE` |
| PDF metadata verification | flash | `PAPERVAULT_LLM_VERIFY` |
| Library intent/judges/resolver/re-rank | standard | `PAPERVAULT_LLM_JUDGE` |
| Decomposition (eval-only variant) | standard | `PAPERVAULT_LLM_DECOMPOSE` |
| Eval judges | standard | `PAPERVAULT_LLM_EVAL_JUDGE` |

For example, `PAPERVAULT_LLM_SYNTH=standard` overrides synthesis. Values must be
bare levels; unset/blank uses the table. Provider model names and suffixes are
rejected with the variable name in the error. `doctor` validates and reports these
routes. Speech (`mimo-v2.5-asr`) and local BGE embedding/rerank are unchanged.

`PAPERVAULT_MODEL`, `PAPERVAULT_BUILD_MODEL`, `KS_BUILD_MODEL`,
`PAPER_PIPELINE_VERIFY_MODEL`, and the eval scripts' `MIMO_MODEL` are retired.
Replace old model/mode settings with the role variables above. Caller effort
settings have been removed; configure effort on the endpoint instead.

The optional gateway transport (`PAPERVAULT_LLM_GATEWAY=1`) forwards each level
unchanged to `PAPERVAULT_LLM_GATEWAY_URL` using `PAPERVAULT_LLM_GATEWAY_KEY`.
Without that flag, the single endpoint/key or key pool also receives the same
bare levels: the direct endpoint must serve these aliases and own their model
and effort. Pool entries supply credentials/endpoints; their model field does
not override a call site's role level. There is no caller-side provider-model
mapping. Bare `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` model fallbacks are retired;
use the documented `PAPERVAULT_LLM_*` endpoint/key configuration.

## MinerU service boundary

MinerU is a shared, independently managed service ([issue #235](https://github.com/LeonJoeeee/papervault/issues/235)).
On the shared host, `mineru.service` is boot-enabled with a private backend at
`:30001`. The always-on `mineru-gate.service` owns the consumer endpoint
`http://127.0.0.1:30000`, starts the backend for workload, stops it after 600
seconds without workload, and owns bounded backend recovery. The legacy
`paper-library-mineru.service` is a transition alias of `mineru.service`. The retained
`deploy/papervault-mineru.service` is a standalone direct-listener template;
it is not the shared host's backend or gate configuration.

papervault requires only gate `/health` 200. It probes once within a three-second
total budget; no unit start, warmup polling, load-state inspection, or lifecycle
cooldown occurs in the caller. A stopped backend is **idle and ready**: the gate
answers health 200 with `{"status":"ok","backend":"stopped"}`. The caller does not
inspect that body or require `/v1/models`, which cannot serve a model while the
backend is stopped. The next OCR workload POST starts the backend inside the
service boundary; the gate waits for it to become ready and forwards the request
(measured cold start: about 20–25 seconds). Unavailable gate health raises a
transport error, so library extraction defers without charging an attempt and
operator PDF pickup leaves the PDF pending.

`MINERU_URL` and `PAPER_LIBRARY_MINERU_*` HTTP-client settings remain supported.
Remove obsolete `PAPER_LIBRARY_MINERU_ONDEMAND`, `_WATCHDOG`, `_UNIT`,
`_IDLE_TIMEOUT`, `_IDLE_CHECK`, `_READY_TIMEOUT`, `_HEALTH_POLL`, and
`_IMAGE_DECODE_RESTART_AFTER` settings (each suffix uses the
`PAPER_LIBRARY_MINERU` prefix). There are no caller-side lifecycle settings.
Maintenance-window co-tenancy guards still check `papervault.service`,
`mineru.service`, and the retained standalone template's `papervault-mineru.service`;
they do not inspect backend load state.

After deployment, the orchestrator verifies an extraction round-trip with the
backend stopped. The request must complete through the same `:30000` endpoint
without caller-side intervention. Host measurements and service deployment
remain outside the lane's unit-test evidence.

## BGE idle release

The in-process BGE embedder and reranker have a separate opt-in idle release:
`KS_BGE_IDLE_UNLOAD=1` and `KS_BGE_IDLE_TIMEOUT=600`. After both unload, a
normal `query` or `search_papers` reloads the model it needs. Measure GPU memory
and the `bge embedder loaded in ...s` / `bge reranker loaded in ...s` log entries
on the target GPU. To roll back BGE idle release, set `KS_BGE_IDLE_UNLOAD=0`
and restart papervault under the operator's deployment procedure; connected
MCP clients must reconnect. These BGE settings do not control MinerU.

## Acquisition-v2 format deploys: drain barrier

For a durable-format change, the barrier is **stop-intake → verify-drain →
flip** (accepted spec A3/A5). All writers participate: scheduler, metadata
writer, correction pipeline and standalone campaigns. Mixed old/new writing
is forbidden. Disabling a v2 flag alone does not stop these producers.

1. **Pre-flight:** prepare the pinned release and its versioned start path
   (`<release-root>/v0.7.25/.venv/bin/papervault serve`, for example). Verify
   the supervisor's executable path and installed package version against that
   pin. Check index and ledger headers before starting a replacement writer.
   Verify executor lifetime tracking covered **all writers from admission**,
   including standalone work; enabling flags only now is insufficient. For
   acquisition workers, keep merge-v2 lifetime tracking enabled from startup.
   Without that history, the helper cannot certify drain. Current formats
   are index/budget/claims/lease/queued-claim **1**, identity definition **2**;
   `compat_v2.COMPATIBILITY_MATRIX` declares each reader's current and preceding
   format. Future format writers must stamp bumped surfaces in the index or
   JSONL event's `acquisition_versions` header; bump the index/ledger `version`
   too when its container changes. Unmarked surfaces retain the slices 1–6
   versions. Checks are always-on and read-only for current formats. Download
   and extraction executors check before entry independently of merge tracking;
   index saves check before backup rotation. Turning off v2 flags cannot bypass
   newer-format refusal; the legacy index reader is unchanged. Feature flags
   keep their existing defaults. Politeness accounting is process-local,
   with no independent durable format.
2. **Stop intake:** disable old writer restart policies, timers and launchers;
   seal MCP acquisition, reconcile/correction producers, metadata jobs and new
   campaigns. Let already admitted work settle. The deployment's seal callback
   must confirm every producer is stopped and keep intake closed through flip.
   Pausing extraction alone leaves queued work and does not establish a barrier.
3. **Verify drain in the owning writer process**, using its live `Library` and
   registered download/extract queues (a fresh process cannot see its executors):

   ```python
   from dataclasses import asdict
   from papervault.library.drain_v2 import verify_drained

   evidence = await verify_drained(
       library, stop_intake=seal_all_writer_intake,
       execution_tracking_verified=tracking_verified_from_preflight)
   # seal_all_writer_intake is the deployment's async callback; return exactly
   # True only after every producer is sealed, keeping the seal in place.
   # The tracking argument must be exactly True, backed by the pre-flight above.
   assert evidence.drained, asdict(evidence)
   ```

   Record the timestamp and evidence in the deployment record. The helper
   checks durable claims (including ledger-only reservations), the in-flight
   registry, merge and ordinary-thread lifetime registries, merge intents and
   queued/pending/unfinished work.
   False evidence or unreadable accounting stops cutover. It does not dequeue,
   refund or release quarantine. Lease expiry and cancellation do not prove
   execution ended; resolve uncertainty with existing drain/fencing recovery.
4. **Flip:** stop the drained old writer, select the versioned new start path,
   then verify path/version and format readability again before admitting work.
   Keep every old launcher disabled; the supervisor must reject an old-version
   restart after the barrier. Verify that rejection as a cutover pre-flight item
   before reopening intake. If the deployment cannot enforce it, do not flip.

Rollback before an irreversible point requires the same barrier and a readable
matrix. After promotion, merge or an append-only budget decrement, retain
canonical identities, aliases, restrictions, generations, artifacts and charges;
use an ad hoc forward fix (short downtime is supported), never restore an old
index or ledger to erase the operation. The four failure-injected fixture drills
in `tests/library/test_rollback_drills_v2.py` exercise this retained-data posture;
they perform no production operations.

## Connecting an agent

**Claude Code** — install the bundled plugin (points Claude Code at the running
MCP server):

```bash
# from the papervault repo
/plugin marketplace add ./        # the marketplace manifest routes to the minimal plugin/ subdir
/plugin install papervault        # installs ONLY plugin/ (manifest + MCP registration), not the repo
```

**Any other MCP client** — register the streamable-http endpoint
`http://127.0.0.1:8080/mcp` (or your host/port) as an MCP server. The three tools
(`search_papers`, `get_paper`, `query`) appear automatically.

The plugin defaults to `http://127.0.0.1:8080/mcp`. To use a different address,
set `PAPERVAULT_MCP_URL` in the **client's environment before starting Claude Code**:

```bash
export PAPERVAULT_MCP_URL="http://127.0.0.1:9090/mcp"
claude
```

Use the full endpoint URL, including `/mcp`. Claude Code expands the variable in
`plugin/.mcp.json`; no installed-plugin edit is needed. Unset the variable to use
the loopback default. The URL is not derived from the server's `serve` config or
`.env`; if you change its host/port, update the client environment and any direct
MCP registrations to match. Start a new Claude Code session after changing it.

> **Bind loopback only (or set a token).** `papervault serve` binds `127.0.0.1`
> by default and has no auth unless you set one — passing `--host 0.0.0.0` (or any
> non-loopback host) with no token exposes all three tools unauthenticated. Keep it
> on loopback, a Tailscale/VPN address, or a trusted private LAN. To require auth,
> set `PAPERVAULT_MCP_TOKEN=<secret>`: the HTTP transport then rejects any request
> without `Authorization: Bearer <secret>` (401 JSON). Unset ⇒ current behavior,
> unchanged; the `--stdio` transport is unaffected either way.

### From another device on your tailnet (Claude Code, Codex CLI, Codex app)

On the **server box**, leave papervault listening on `127.0.0.1:8080` and
run [Tailscale Serve](https://tailscale.com/docs/reference/tailscale-cli/serve):

```bash
tailscale serve --bg --https=8080 http://127.0.0.1:8080
```

Replace `<node>.<tailnet>.ts.net` below with the server's tailnet DNS name. Add
this line to the server checkout's `.env`, then **restart the service**:

```dotenv
PAPERVAULT_MCP_ALLOWED_HOSTS=<node>.<tailnet>.ts.net:8080
```

To remove this proxy mapping, run `tailscale serve --https=8080 off`.
The setting accepts comma-separated `Host` values, ignores surrounding whitespace
and empty entries, and always retains loopback access. An exact `host:port` allows
only that port; a bare `host` allows only a port-less `Host`; `host:*` allows any
explicit port but does **not** match a port-less `Host`. On the default HTTPS port
443, write the bare hostname. Configured entries also allow their corresponding
HTTP and HTTPS origins; unrelated hosts and origins remain refused.

On **each client device**, connect to the same tailnet and configure the HTTPS URL:

**Claude Code with the bundled plugin** — install it as above, then start Claude
Code with the tailnet endpoint in its environment:

```bash
export PAPERVAULT_MCP_URL="https://<node>.<tailnet>.ts.net:8080/mcp"
claude mcp list
claude
```

Confirm `claude mcp list` shows `plugin:papervault:papervault` at that HTTPS URL
and reports it connected. For subsequent sessions, keep the variable in the
environment used to launch Claude Code (for example, export it in your shell
startup file). Setting it only in the server's `.env` does not configure clients.

**Claude Code without the plugin** — direct registration is also available;
user scope makes it available in every project on that device:

```bash
claude mcp add --transport http --scope user papervault https://<node>.<tailnet>.ts.net:8080/mcp
```

Claude Code's handshake timeout is `MCP_TIMEOUT` (milliseconds, default 30000);
its per-call timeout is `MCP_TOOL_TIMEOUT`. The lab registration works with the
defaults; adjust these only if needed. See the
[Claude Code MCP documentation](https://code.claude.com/docs/en/mcp).

**Codex CLI** — add this block to `~/.codex/config.toml` on the client device
(configuration used with codex-cli 0.153.4):

```toml
[mcp_servers.papervault]
url = "https://<node>.<tailnet>.ts.net:8080/mcp"
startup_timeout_sec = 120
tool_timeout_sec = 900
```

Papervault's handshake can take 30–60 seconds under load. The default roughly
10-second Codex startup timeout can silently drop the tools; keep the explicit
timeouts above (see [#29](https://github.com/LeonJoeeee/papervault/issues/29) and
[Codex MCP settings](https://developers.openai.com/codex/mcp)). For a
**non-interactive Codex agent**, also add the following line inside that same
`[mcp_servers.papervault]` block:

```toml
default_tools_approval_mode = "approve"
```

This approves calls to this server without prompting; otherwise non-interactive
agents can have every tool call auto-rejected
([#116](https://github.com/LeonJoeeee/papervault/issues/116)).

**Codex app** — the desktop app is expected to read the same
`~/.codex/config.toml` on that device. Use the same block:

```toml
[mcp_servers.papervault]
url = "https://<node>.<tailnet>.ts.net:8080/mcp"
startup_timeout_sec = 120
tool_timeout_sec = 900
```

For both Codex clients, verify on the device with `codex mcp list`, then start a
new client session and confirm the three tools are available.

Follow the **“Bind loopback only (or set a token)”** security blockquote above.
Without `PAPERVAULT_MCP_ALLOWED_HOSTS`, a non-loopback `Host` is refused with 421
even when the server is bound to `0.0.0.0`. With the variable, papervault still has
**no authentication unless `PAPERVAULT_MCP_TOKEN` is set**, so the tailnet's device
identity is the whole perimeter. `PAPERVAULT_MCP_TOKEN` gates **every** HTTP
request, including loopback ones: enabling it means reconfiguring every local
client too (Claude Code, Codex reviewers, and any script).

## Choosing a domain

papervault ships tuned for space physics + AI4Science. To adapt it to another
field, edit the domain pack (`src/papervault/domain/space_physics/`) or point
`PAPERVAULT_DOMAIN_PATH` at your own pack directory with the same files:
`domain.toml`, `ontology.json`, `extraction_examples.txt`,
`extraction_exclusions.txt`, `search_gate.md`. Quality outside the shipped domain
is your responsibility.

## Grey-zone download tiers

Shadow libraries (Sci-Hub, Anna's Archive) and anti-bot scraping tiers are
**enabled in the beta posture**: the standard install includes the `grey` extra and
`.env.example` ships `PAPER_PIPELINE_USE_SCIHUB=1`. Running them is still your own
legal decision — comment the flag out (and skip the extra) to opt out. Anna's
Archive additionally requires a personal member key (`ANNAS_ARCHIVE_API_KEY`).
Before the repo flips public, this shipped default reverts to OFF — see
[ADR-0004](adr/0004-grey-download-tiers-flag-gated.md) and its amendment.


## ADS availability evidence

The existing ADS member of `domain_aggregators` accepts DOI/arXiv identifiers and
validated stored ADS record URLs or ADS-qualified bibcodes. It requires
`ADS_API_TOKEN`. Every returned PDF still passes the cascade's identity check.

ADS attempts record `Paper.ads_availability`: publication kind, availability,
checked time, stable request URLs and document evidence. Inspect stored outcomes
with `python -m papervault.library.cli audit --ads-availability --json`. This report
makes no source requests and does not reset terminal records; older, unaudited
records remain `unknown`.

Availability separates `retrievable`, `blocked`, `unknown`, and
`confirmed_abstract_only`. Confirmation requires matching document evidence for
a published meeting abstract or an abstract-book contribution, including entries
beyond the book's first pages. It settles the indexed item's full-paper hunt with
a durable reason and preserves existing PDFs/extracts. The conclusion applies
to that publication, not to whether a related full paper exists elsewhere.
Missing ADS links, HTML-only metadata, access blocks, API errors and incomplete
documents do not establish full-text absence. Ambiguous forms remain unknown.
The document rules recognize conservative English publication forms; very short
entries and unfamiliar languages or layouts remain unknown.

## Run papervault as a service (optional)

`deploy/papervault.service` is a user-level systemd unit for the server. Its
`WorkingDirectory`/`ExecStart` assume the clone lives at `~/projects/dev/papervault` —
edit both paths to your clone location before `systemctl --user enable --now papervault`.
Keep its `Environment=MALLOC_MMAP_THRESHOLD_=1048576` line (or the same setting in a
drop-in) when you adapt it: glibc reads it only at process start, so it cannot live in
`.env`, and without it memory freed after an OCR burst stays in the process.
