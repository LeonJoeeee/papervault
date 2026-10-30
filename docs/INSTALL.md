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

# 3. Backing stores (Postgres + Neo4j). The data volumes are declared external and
#    hold your production data, so they must exist before the first start — a
#    missing volume fails the start instead of silently creating an empty database.
#    --env-file is REQUIRED: with `-f deploy/...`, compose reads deploy/.env, NOT
#    the repo-root .env — pass it explicitly so your POSTGRES/NEO4J passwords reach
#    the containers.
docker volume create ks-pgdata
docker volume create ks-neo4jdata
docker compose --env-file .env -f deploy/docker-compose.yml up -d

# 4. Install papervault (core HTTP OCR client + grey download tiers)
uv venv && uv pip install -e ".[grey]"
#   uv pip install -e .                 # core HTTP OCR client, OA-only downloads

# 5. Provision MinerU independently before enabling extraction.
#    Core httpx supplies the client; [mineru] is an empty compatibility extra.
#    The shared mineru-service exposes official mineru-api at :30000 and owns
#    FIFO queuing, model preload and GPU unloading after 20 idle minutes.
#    Provision the service outside papervault's venv. Do not install a second
#    listener on :30000 or change a shared service from the caller.
#    The default MINERU_URL stays http://127.0.0.1:30000; set it for another host.
curl -fsS http://127.0.0.1:30000/health
#    Health 200 confirms endpoint health; model loading is service-owned.

# 6. Preflight — verify GPU, databases, LLM config, domain pack
.venv/bin/papervault doctor

# 7. Run the server
.venv/bin/papervault serve               # streamable-http on 127.0.0.1:8080
#   .venv/bin/papervault serve --stdio    # subprocess transport
```

> Note: `doctor` checks that a key for the selected direct or gateway transport is PRESENT,
> not that it authenticates — a wrong
> key surfaces only at the first real `query()` (as an auth error).

`papervault doctor` must be green before serving. It checks the .env, data dirs,
the workspace vars, the domain pack, GPU/VRAM, the OCR endpoint + local model
weights, and Postgres + Neo4j connectivity. With `PAPERVAULT_CATALOG_AUTHORITY=postgres`,
it also checks the explicit catalog DSN and base P2 schema in a read-only connection;
it never installs or migrates a schema:

- **workspace (FAIL):** `NEO4J_WORKSPACE` / `POSTGRES_WORKSPACE` must both be
  non-empty AND equal (or `papervault serve` aborts at boot); the reserved prod
  name `l0` fails unless `KS_ALLOW_PROD_WORKSPACE=1` is set on purpose.
- **MinerU (WARN):** probes `GET {base}/health` on the first configured OCR
  endpoint, without importing MinerU client libraries. If it's down, extracts
  queue until it comes up — not fatal, so a warning, not a failure.
- **BGE weights (WARN):** the embed + reranker default to `BAAI/bge-m3` +
  `BAAI/bge-reranker-v2-m3` (~4.5 GB total), pulled lazily from Hugging Face on
  first use. `doctor` detects whether they're already local (a directory, or the
  HF cache) WITHOUT downloading; absent ⇒ a warning that the first build/query
  downloads them. Pre-seed or redirect the HF cache with `HF_HOME`, or point
  `BGE_M3_MODEL_PATH` / `BGE_RERANKER_MODEL_PATH` at local weights.

For a deeper check, `papervault smoke` runs doctor, boots the MCP server over
stdio, and round-trips its tools (`list_tools` + a read-only `get_paper`) — no
LLM required (`papervault smoke --skip-db` skips the DB connectivity checks).
The child uses a temporary JSON vault with legacy knowledge jobs and auto-ingest off;
it removes the catalog DSN and prevents the operator's `.env` from restoring it.

The install commands create a project-local `.venv`; invoke its binaries as shown,
or activate it before using bare `papervault` commands elsewhere in this guide.

`KS_AUTO_INGEST_ENABLED=false` is the shipped default. Set it to `true` (also
accepts `1` or `yes`) after the vault and stores are ready to start background
distillation. Without this opt-in, the advertised search-to-queryable loop does not
run. `KS_AUTO_INGEST_INTERVAL_SEC` defaults to `60` seconds. Query retrieval,
synthesis, evaluation controls and their defaults are documented in
[Evaluation](eval.md#retrieval-tuning-knobs-benchmark-arbitrated); the template
includes commented examples. Changes affecting retrieval quality require arbitration.

## LLM routing and migration

Every text LLM request resolves a concrete `primary[,fallback...]` target chain
from its role (`<model>[@<modifier>][,<fallback>...]`). Defaults are:

| Role | Default route | Override |
| --- | --- | --- |
| Graph extraction/merge | `glm-5.3,gpt-6.1-terra@high` | `PAPERVAULT_LLM_BUILD` |
| Query keyword extraction | `gpt-6.1-terra@high` | `PAPERVAULT_LLM_KEYWORD` |
| Answer synthesis | `gpt-6-astra@xhigh` | `PAPERVAULT_LLM_SYNTH` |
| Extract clarity/completeness gates | `glm-5.3-flash,gpt-6.1-terra@high` | `PAPERVAULT_LLM_GATE` |
| PDF metadata verification | `glm-5.3-flash,gpt-6.1-terra@high` | `PAPERVAULT_LLM_VERIFY` |
| Library search intent | `gpt-6.1-terra@high` | `PAPERVAULT_LLM_INTENT` |
| Library judges/resolver/re-rank | `gpt-6.1-terra@high` | `PAPERVAULT_LLM_JUDGE` |
| Decomposition (production multiquery and eval) | `gpt-6.1-terra@high` | `PAPERVAULT_LLM_DECOMPOSE` |
| Eval judges | `glm-5.3,gpt-6.1-terra@high` | `PAPERVAULT_LLM_EVAL_JUDGE` |

For `glm-*` models, adaptive thinking is enabled by default (`@nothinking` or
`@none` disables it; effort tiers on `glm-*` raise `ValueError`). Non-`glm-*`
models accept `@none`, `@low`, `@medium`, `@high`, `@xhigh`, `@max`, `@thinking`,
or `@nothinking`, or omit the modifier for plain OpenAI-compatible models (e.g.
`PAPERVAULT_LLM_JUDGE=gpt-4.1-mini`). Multi-target routes fail over immediately
from a throttled or timed-out primary (`429`, `408`, `5xx`, timeout,
`StreamTruncated`) to the next target and cool down the primary for
`PAPERVAULT_LLM_FALLBACK_COOLDOWN_S` seconds (default `120`, `<= 0` disables)
before probing it again; single-target GPT roles never cross-model fallback.

The retired symbolic level names (`flash`, `standard`, `pro`, `max`, `ultra` and
`papervault-*`) are rejected with `ValueError` naming `PAPERVAULT_LLM_<ROLE>`.
`papervault doctor` validates and reports all nine routes. Speech (`mimo-v2.5-asr`)
and local BGE embedding/rerank are unchanged.

`PAPERVAULT_MODEL`, `PAPERVAULT_BUILD_MODEL`, `KS_BUILD_MODEL`,
`PAPER_PIPELINE_VERIFY_MODEL`, and the eval scripts' `MIMO_MODEL` are retired.
Replace old model/level settings with concrete role routes above (or remove them
from `.env` to use the defaults).

The optional gateway transport (`PAPERVAULT_LLM_GATEWAY=1`) sends concrete model
IDs and thinking/effort parameters directly to `PAPERVAULT_LLM_GATEWAY_URL` using
`PAPERVAULT_LLM_GATEWAY_KEY` (matching the transparent LLM Gateway `v0.7.0`).
Without that flag, the single endpoint/key or key pool (`PAPERVAULT_LLM_KEYS`)
sends the same concrete routes; when no role override is forced, key-pool entries
rotate across their configured `base_url`/`api_key`/`model` groups and auto-disable
permanent `401`/`403` credentials. Bare `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`
model fallbacks remain retired; use the documented `PAPERVAULT_LLM_*` settings.

`doctor` accepts a configured gateway key without a direct endpoint key or key-pool
file when gateway mode is on; a missing gateway key fails even if direct credentials
exist. To select an environment file, set `PAPERVAULT_ENV` to an existing file;
otherwise configuration looks for `.env` in the current directory, then the checkout.
Existing process variables take precedence over file values. `POSTGRES_DB` is bridged
to `POSTGRES_DATABASE` only when the latter is unset; keep them equal when explicitly
setting both so the ledger and LightRAG use the same database.

## Advanced runtime settings

The template covers the usual deployment controls. These additional active settings
are useful for diagnosis or capacity tuning; defaults come from the current runtime:

| Env var | Default | Effect |
| --- | --- | --- |
| `KS_STARTUP_AWAIT_SEC` | `3` | Maximum startup wait in seconds; initialization/retries continue in the background after it expires. `0` skips waiting. |
| `KS_SKIP_EMBED_SELFCHECK` | `0` | `1`, `true` or `yes` bypasses the embedding boot probe; use only for diagnosis or a separately verified stack. |
| `KS_EMBED_MAX_ASYNC` / `KS_EMBED_GPU_MAX_ASYNC` | `16` / `1` | LightRAG embedding-call queue concurrency versus actual GPU forward-pass concurrency. |
| `KS_EMBED_OFFLOAD` | `1` | Offload embedding to a worker thread; `0` restores blocking inference on the server loop. |
| `KS_LLM_MAX_TOKENS` | `131000` | Knowledge LLM output ceiling unless the caller sets its own limit. |
| `KS_LLM_BACKOFF_JITTER` | `0.75` | Jitter added to direct key-pool retry backoff in seconds. |
| `KS_GATEWAY_CONN_RETRIES` / `KS_GATEWAY_CONN_BACKOFF_CAP` | `6` / `15` | Unreachable-gateway connection retries and backoff cap in seconds, still bounded by the configured retry budget. |
| `KS_SLOW_CALL_LOG_S` | `60` | Log successful knowledge LLM calls exceeding this duration in seconds. |
| `PAPERVAULT_LLM_FALLBACK_COOLDOWN_S` | `120` | Process-local primary-model cooldown in seconds after multi-target failover (`<= 0` disables). |
| `PAPERVAULT_LIB_FIRST_CHUNK_S` / `PAPERVAULT_LIB_RETRY_BUDGET_S` | `0` / `480` | Library-side pre-first-chunk cutoff (`0` disables; positive seconds opt in) and total retry-admission budget in seconds; separate from the `KS_*` knowledge settings. |
| `PAPERVAULT_SAVE_MIN_INTERVAL_S` / `PAPERVAULT_BIB_MIN_INTERVAL_S` | `0` / `60` | Non-forced library save debounce and BibTeX refresh cadence in seconds. Forced durable lifecycle saves bypass debounce. |
| `PAPERVAULT_EXTRACT_DEFER_CANARY` | `8` | Maximum deferred extraction recovery canaries promoted per reconcile sweep; `0` disables canaries. |
| `PAPERVAULT_LOOP_LAG_MONITOR_MS` | `0` | Event-loop stall warning threshold in milliseconds; `0` disables monitoring. |
| `PAPERVAULT_LOOP_LAG_INTERVAL_MS` / `PAPERVAULT_LOOP_LAG_ROLLUP_S` | `50` / `30` | Monitor sample interval in milliseconds and summary cadence in seconds. |
| `NEO4J_DATABASE` | sanitized storage namespace | Catalog reader's preferred Neo4j database; it also tries the default database when unavailable. |
| `NEO4J_MAX_CONNECTION_POOL_SIZE` | `100` | Catalog Neo4j reader connection-pool ceiling. |
| `NEO4J_CONNECTION_TIMEOUT` / `NEO4J_CONNECTION_ACQUISITION_TIMEOUT` / `NEO4J_MAX_TRANSACTION_RETRY_TIME` | `30` each | Neo4j reader connect, pool acquisition and transaction-retry limits in seconds. |
| `NEO4J_MAX_CONNECTION_LIFETIME` / `NEO4J_LIVENESS_CHECK_TIMEOUT` / `NEO4J_KEEP_ALIVE` | `300` / `300` / `true` | Neo4j reader connection lifetime/liveness thresholds in seconds and TCP keepalive. |
| `REQUESTS_CA_BUNDLE` | unset | Custom CA bundle for the bounded OpenAlex downloader; unset uses normal certificate verification. |

`ENTITY_EXTRACTION_USE_JSON` must stay unset/false: domain examples currently support
text extraction only, and graph initialization refuses JSON mode. `LLM_KEYS_FILE` is
a legacy fallback name for `PAPERVAULT_LLM_KEYS`; use the latter in new deployments.
The retained `MIMO_API_KEY_1`…`MIMO_API_KEY_6` and `MIMO_BASE_URL_1`…`MIMO_BASE_URL_6`
fields populate a legacy config view; current completion credentials come from the
documented direct key/pool or gateway transport. `PAPER_LIBRARY_MCP_TOKEN` only
suppresses a warning in the legacy standalone library launcher; use the unified
server's `PAPERVAULT_MCP_TOKEN` for HTTP authentication.

## MinerU service boundary

MinerU is a shared, independently managed service ([issue #235](https://github.com/LeonJoeeee/papervault/issues/235)).
The shared `mineru-service` exposes unmodified official `mineru-api` at
`http://127.0.0.1:30000` ([issue #288](https://github.com/LeonJoeeee/papervault/issues/288)).
Its `AsyncTaskManager` owns FIFO queuing across callers. The service runs with
`--enable-vlm-preload true` and unloads GPU models after 20 idle minutes;
model loading, idle release and recovery stay inside the service boundary.

papervault's core `httpx` dependency supplies the entire OCR client. The
`mineru` extra is empty and retained for compatibility: `uv sync --extra mineru`
and installs using `.[mineru]` remain valid without adding `mineru` or
`torchvision` to papervault's environment. Install the OCR service separately.

For each PDF, the client submits multipart `POST {base}/tasks` with
`backend=vlm-engine`, polls `GET {base}/tasks/{task_id}`, then reads markdown
from `GET {base}/tasks/{task_id}/result`. Rendering and inference happen in
the service. Pending queue-wait timeouts defer extraction without charging
an attempt; read or wall-clock timeouts observed during active processing
charge an extraction attempt. Connection failures and service outages remain
uncharged transport failures in every phase.

papervault performs no caller-side pre-flight; `ensure_ready` is an immediate
no-op. The service handles workload when its models are unloaded, including
any backend wakeup it needs. Submit/poll/result requests determine reachability;
transport failures leave operator PDF pickup pending. Each endpoint attempt
owns an `httpx.AsyncClient` and closes it on its own event loop, including
cancellation. papervault holds no MinerU singleton, parser or model-discovery
cache and performs no unit start, warmup polling or load-state inspection.

`MINERU_URL` and the active `PAPER_LIBRARY_MINERU_*` HTTP-client settings remain supported.
Remove obsolete `PAPER_LIBRARY_MINERU_ONDEMAND`, `_WATCHDOG`, `_UNIT`,
`_IDLE_TIMEOUT`, `_IDLE_CHECK`, `_READY_TIMEOUT`, `_HEALTH_POLL`,
`_IMAGE_DECODE_RESTART_AFTER`, `_PRECHECK`, and `_PRECHECK_TIMEOUT` settings
(each suffix uses the `PAPER_LIBRARY_MINERU` prefix). Also remove
`MINERU_PROCESSING_WINDOW_SIZE`, `PAPER_LIBRARY_MINERU_CLIENT_CLOSE`,
`PAPER_LIBRARY_MINERU_NATIVE_RETRIES` and `PAPER_LIBRARY_MINERU_NATIVE_BACKOFF`
from the caller's configuration. Rendering windows belong to the service;
the client always closes its own sockets. Native retry constants remain
import-compatible but do not affect HTTP task requests. Active request
timeouts, endpoint retries/backoff and per-attempt connection limits are
documented in [`.env.example`](../.env.example).

The retained `deploy/papervault-mineru.service` template starts the legacy
`mineru-vllm-server`, which does not supply the required task API; it is not
a setup path for current papervault. `scripts/probe_mineru_socket_leak.py`
retains the historical pre-0.10.1 parser-client reproduction, whose
`CLIENT_CLOSE` before/after switch no longer applies to the HTTP task client.
Maintenance-window co-tenancy guards still check `papervault.service`,
`mineru.service`, and the retained standalone template's `papervault-mineru.service`;
they do not inspect backend load state.

After deployment, the orchestrator verifies a submit/poll/result round-trip
through the same `:30000` endpoint after idle GPU unload, including any
service-owned backend wakeup. Host measurements and service deployment remain
outside the lane's unit-test evidence.

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

The Claude Code plugin defaults to `http://127.0.0.1:8080/mcp`. To use a different address,
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

**Antigravity (`agy`) CLI/IDE** — install the native plugin shell after starting
the papervault server. From the repository root (commands checked with AGY 1.4.5):

```bash
agy plugin validate ./plugin --json
agy plugin install ./plugin
agy plugin list
```

AGY reads `plugin/plugin.json` and `plugin/mcp_config.json`; the latter registers
`papervault` at `http://127.0.0.1:8080/mcp` using `serverUrl`. Start a new CLI or
IDE session after installation. In the IDE, open **Additional Options (...) >
MCP Servers** and confirm the three tools (`search_papers`, `get_paper`, `query`)
are available. The plugin contains client wiring; the server runs separately.

**AGY direct MCP connection** — as an alternative to the plugin, register the
HTTP endpoint directly. Use this route for a different host/port: the native
plugin has a fixed URL, and `PAPERVAULT_MCP_URL` only configures Claude Code.

```bash
agy mcp add papervault http://127.0.0.1:8080/mcp
agy mcp list
```

For IDE setup through JSON, merge this entry into
`~/.gemini/config/mcp_config.json`, preserving any existing `mcpServers` entries:

```json
{
  "mcpServers": {
    "papervault": {
      "serverUrl": "http://127.0.0.1:8080/mcp"
    }
  }
}
```

Replace the URL with your server's full endpoint, including `/mcp`, and start a
new Antigravity session after changing the configuration. Choose either plugin
installation or direct MCP registration to avoid duplicate connections.

**Any other MCP client** — register the streamable-http endpoint
`http://127.0.0.1:8080/mcp` (or your host/port) as an MCP server. The three tools
(`search_papers`, `get_paper`, `query`) appear automatically.

> **Bind loopback only (or set a token).** `papervault serve` binds `127.0.0.1`
> by default and has no auth unless you set one — passing `--host 0.0.0.0` (or any
> non-loopback host) with no token exposes all three tools unauthenticated. Keep it
> on loopback, a Tailscale/VPN address, or a trusted private LAN. To require auth,
> set `PAPERVAULT_MCP_TOKEN=<secret>`: the HTTP transport then rejects any request
> without `Authorization: Bearer <secret>` (401 JSON). Unset ⇒ current behavior,
> unchanged; the `--stdio` transport is unaffected either way.

### From another device on your tailnet (Claude Code, Antigravity, Codex CLI, Codex app)

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

**Antigravity (`agy`) CLI/IDE** — register the tailnet endpoint directly instead
of using the plugin's loopback default:

```bash
agy mcp add papervault https://<node>.<tailnet>.ts.net:8080/mcp
agy mcp list
```

For IDE JSON setup, use this HTTPS URL as `mcpServers.papervault.serverUrl` in
`~/.gemini/config/mcp_config.json` (see the local example above). Start a new
session and inspect **Additional Options (...) > MCP Servers** for the tools.

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
client too (Claude Code, Antigravity, Codex reviewers, and any script).

## Choosing a domain

papervault ships tuned for space physics + AI4Science. To adapt it to another
field, edit the domain pack (`src/papervault/domain/space_physics/`) or point
`PAPERVAULT_DOMAIN_PATH` at your own pack directory with the same files:
`domain.toml`, `ontology.json`, `extraction_examples.txt`,
`extraction_exclusions.txt`, `search_gate.md`. Quality outside the shipped domain
is your responsibility.

## Grey-zone download tiers

Shadow libraries (Sci-Hub, Anna's Archive) and anti-bot scraping tiers are
**off by default**. Installing the `grey` extra supplies dependencies but does not
enable these tiers. To opt in, uncomment `PAPER_PIPELINE_USE_SCIHUB=1` in `.env`;
that decision remains the operator's responsibility. Anna's
Archive additionally requires a personal member key (`ANNAS_ARCHIVE_API_KEY`).
See
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
