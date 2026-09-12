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
$EDITOR .env          # set PAPERVAULT_LLM_*, PAPERVAULT_MODEL, POSTGRES/NEO4J passwords,
                      # PAPERVAULT_CONTACT_EMAIL

# 3. Backing stores (Postgres + Neo4j). --env-file is REQUIRED: with `-f
#    deploy/...`, compose reads deploy/.env, NOT the repo-root .env — pass it
#    explicitly so your POSTGRES/NEO4J passwords reach the containers.
docker compose --env-file .env -f deploy/docker-compose.yml up -d

# 4. Install papervault (beta-standard: core + local OCR + grey download tiers)
uv venv && uv pip install -e ".[mineru,grey]"
#   uv pip install -e .                 # minimal core (no OCR, OA-only downloads)

# 5. Start the MinerU OCR server (ONLY if you installed the [mineru] extra).
#    extract talks to a MinerU vLLM server over HTTP on :30000; the [mineru]
#    client libs do NOT start it, and nothing else will — without it every
#    extract fails and papers sit at text_status="pending" forever.
#
#    RECOMMENDED — run it as a systemd --user service (restarts on failure, and
#    papervault's on-demand lifecycle can start/stop it by name):
cp deploy/papervault-mineru.service ~/.config/systemd/user/
#    Edit the copy if `mineru-vllm-server` isn't on your login PATH or to pick a GPU.
systemctl --user daemon-reload
systemctl --user enable --now papervault-mineru.service
#
#    ALTERNATIVE — run it in the foreground (no restart-on-failure):
#    mineru-vllm-server --host 127.0.0.1 --port 30000 &
#
#    First run pulls the MinerU2.5 weights from ModelScope (MINERU_MODEL_SOURCE,
#    default `modelscope`). Point papervault at another host/port via MINERU_URL.

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


## Run papervault as a service (optional)

`deploy/papervault.service` is a user-level systemd unit for the server. Its
`WorkingDirectory`/`ExecStart` assume the clone lives at `~/projects/dev/papervault` —
edit both paths to your clone location before `systemctl --user enable --now papervault`.
