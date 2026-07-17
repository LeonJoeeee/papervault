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

# 3. Backing stores (Postgres + Neo4j)
docker compose -f deploy/docker-compose.yml up -d

# 4. Install papervault (beta-standard: core + local OCR + grey download tiers)
uv venv && uv pip install -e ".[mineru,grey]"
#   uv pip install -e .                 # minimal core (no OCR, OA-only downloads)

# 5. Preflight — verify GPU, databases, LLM config, domain pack
papervault doctor

# 6. Run the server
papervault serve                         # streamable-http on 127.0.0.1:8080
#   papervault serve --stdio             # subprocess transport
```

`papervault doctor` must be green before serving. It checks the .env, data dirs,
the domain pack, GPU/VRAM, and Postgres + Neo4j connectivity.

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
