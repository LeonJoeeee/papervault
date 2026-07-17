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
docker compose -f deploy/docker-compose.yml up -d   # Neo4j + Postgres
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

## License

MIT — see [`LICENSE`](LICENSE).
