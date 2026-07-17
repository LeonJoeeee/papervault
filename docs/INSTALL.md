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
#    extract fails and papers sit at text_status="pending" forever. Leave it up:
mineru-vllm-server --host 127.0.0.1 --port 30000 &
#    First run pulls the MinerU2.5 weights from ModelScope (MINERU_MODEL_SOURCE,
#    default `modelscope`). Point papervault at another host/port via MINERU_URL.

# 6. Preflight — verify GPU, databases, LLM config, domain pack
papervault doctor

# 7. Run the server
papervault serve                         # streamable-http on 127.0.0.1:8080
#   papervault serve --stdio             # subprocess transport
```

`papervault doctor` must be green before serving. It checks the .env, data dirs,
the domain pack, GPU/VRAM, and Postgres + Neo4j connectivity. It does NOT cover
everything a boot needs: it does not validate the workspace vars
(`NEO4J_WORKSPACE` / `POSTGRES_WORKSPACE` must both be non-empty AND equal, or
`papervault serve` aborts at boot), and it does not probe the MinerU endpoint —
so a green doctor can still fail to serve on empty/mismatched workspace vars, or
serve but never build a corpus if MinerU is down.

**First ingest/query downloads model weights.** The embed + reranker default to
`BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3` (~4.5 GB total), pulled lazily from
Hugging Face on first use — `doctor` does NOT download or probe them, so it can
be green while the first build/query then blocks on (or, air-gapped, fails at)
the download. Pre-seed or redirect the HF cache with `HF_HOME`, or point
`BGE_M3_MODEL_PATH` / `BGE_RERANKER_MODEL_PATH` at local weights.

## Connecting an agent

**Claude Code** — install the bundled plugin (points Claude Code at the running
MCP server):

```bash
# from the papervault repo
/plugin marketplace add ./
/plugin install papervault
```

**Any other MCP client** — register the streamable-http endpoint
`http://127.0.0.1:8080/mcp` (or your host/port) as an MCP server. The three tools
(`search_papers`, `get_paper`, `query`) appear automatically.

> Note: this endpoint URL is hardcoded (see `.mcp.json` at the repo root), not derived
> from your `serve` config. If you run `papervault serve --port`/`--host` on a different
> address, update the URL in `.mcp.json` and in any MCP client registration to match, or
> clients will point at the wrong address.

> **Bind loopback only.** `papervault serve` binds `127.0.0.1` by default and has
> NO built-in auth — passing `--host 0.0.0.0` (or any non-loopback host) exposes
> all three tools unauthenticated. Keep it on loopback, a Tailscale/VPN address,
> or a trusted private LAN. (`PAPER_LIBRARY_MCP_TOKEN` guards only the *standalone*
> library HTTP server; the unified `papervault serve` has no token equivalent.)

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
