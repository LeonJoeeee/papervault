"""papervault CLI — operator entrypoint.

    papervault doctor     # preflight: config, data dirs, GPU, databases, LLM endpoint
    papervault smoke      # end-to-end sanity: doctor + boot the MCP server + tool round-trip
    papervault serve      # run the MCP server (same as `papervault-mcp`)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from papervault import config


@click.group()
def main() -> None:
    """papervault — self-hosted literature + knowledge layer for coding agents."""


# --------------------------------------------------------------------------- #
#  doctor                                                                      #
# --------------------------------------------------------------------------- #

class _Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, name: str, detail: str = "") -> None:
        click.echo(f"  {click.style('✓', fg='green')} {name}" + (f" — {detail}" if detail else ""))

    def warn(self, name: str, detail: str = "") -> None:
        self.warnings += 1
        click.echo(f"  {click.style('!', fg='yellow')} {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name: str, detail: str = "") -> None:
        self.failures += 1
        click.echo(f"  {click.style('✗', fg='red')} {name}" + (f" — {detail}" if detail else ""))


def _check_config(r: _Report) -> None:
    click.echo(click.style("config", bold=True))
    if config.ENV_FILE:
        r.ok(".env", str(config.ENV_FILE))
    else:
        r.warn(".env", "no .env found — relying on process environment")
    has_pool = config.LLM_KEYS_FILE.exists()
    if has_pool:
        r.ok("LLM key-pool file", str(config.LLM_KEYS_FILE))
    elif config.LLM_API_KEY:
        r.ok("LLM single key", "PAPERVAULT_LLM_API_KEY set")
    else:
        r.fail("LLM credentials", "set PAPERVAULT_LLM_API_KEY or PAPERVAULT_LLM_KEYS")
    if config.SYNTH_MODEL:
        r.ok("model slots", f"synth={config.SYNTH_MODEL} build={config.BUILD_MODEL}")
    else:
        r.fail("model slots", "set PAPERVAULT_MODEL (and optionally PAPERVAULT_BUILD_MODEL)")
    if not config.LLM_BASE_URL and not has_pool:
        r.warn("LLM base URL", "PAPERVAULT_LLM_BASE_URL empty (ok if the key-pool file sets base_url)")


def _check_paths(r: _Report) -> None:
    click.echo(click.style("data layout", bold=True))
    for name, path in [("data dir", config.DATA_DIR), ("vault", config.VAULT_PATH),
                       ("knowledge store", config.STORAGE_DIR)]:
        try:
            path.mkdir(parents=True, exist_ok=True)
            r.ok(name, str(path))
        except OSError as e:
            r.fail(name, f"{path}: {e}")


def _check_gpu(r: _Report) -> None:
    click.echo(click.style("gpu", bold=True))
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        r.fail("torch", f"import failed: {e}")
        return
    if not torch.cuda.is_available():
        r.fail("cuda", "torch.cuda.is_available() is False — papervault requires a GPU")
        return
    n = torch.cuda.device_count()
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1024**3
        line = f"{props.name} ({gb:.0f} GB)"
        (r.ok if gb >= 22 else r.warn)(f"cuda:{i}", line + ("" if gb >= 22 else " — below the 24 GB floor"))


def _check_databases(r: _Report) -> None:
    click.echo(click.style("databases", bold=True))
    # Postgres — resolve the DSN through the SAME PostgresConfig the ledger + LightRAG
    # store use (knowledge.config), so doctor can never pass against a different database
    # than the runtime targets. One resolution, no divergence.
    try:
        import psycopg
        from papervault.knowledge.config import PostgresConfig
        pg = PostgresConfig.from_env()
        with psycopg.connect(pg.dsn, connect_timeout=5):
            r.ok("postgres", f"{pg.host}:{pg.port}/{pg.db} (user={pg.user})")
        # LightRAG's PG storage reads POSTGRES_DATABASE (config.py bridges it FROM
        # POSTGRES_DB when unset). If the two diverge — explicit POSTGRES_DATABASE, or a
        # bare env with neither set (LightRAG then falls back to 'postgres') — the graph
        # KV/vector/doc_status store silently targets a DIFFERENT database than the
        # ledger doctor just verified. Surface it (review F2).
        store_db = os.environ.get("POSTGRES_DATABASE", "")
        if not store_db:
            r.warn("postgres (graph store)",
                   "POSTGRES_DATABASE unset and POSTGRES_DB missing → LightRAG will use "
                   "db 'postgres', splitting the graph store from the ledger db")
        elif store_db != pg.db:
            r.warn("postgres (graph store)",
                   f"POSTGRES_DATABASE={store_db!r} != POSTGRES_DB={pg.db!r} — graph "
                   "store and ledger live in DIFFERENT databases (ok only if intentional)")
    except Exception as e:  # noqa: BLE001
        r.fail("postgres", str(e).splitlines()[0] if str(e) else type(e).__name__)
    # Neo4j
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        auth = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
        drv = GraphDatabase.driver(uri, auth=auth)
        try:
            drv.verify_connectivity()
            r.ok("neo4j", uri)
        finally:
            drv.close()
    except Exception as e:  # noqa: BLE001
        r.fail("neo4j", str(e).splitlines()[0] if str(e) else type(e).__name__)


def _check_domain(r: _Report) -> None:
    click.echo(click.style("domain pack", bold=True))
    try:
        from papervault.domain import get_domain
        d = get_domain()
        r.ok("domain", f"{d.label} — {len(d.entity_types)} entity types, "
                       f"{len(d.extraction_examples)} extraction examples")
    except Exception as e:  # noqa: BLE001
        r.fail("domain", str(e))


# Kept in sync with knowledge.store.graph (PROD_WORKSPACE / assert_safe_workspace); read
# the env directly here so doctor stays light (no lightrag import just to preflight).
_PROD_WORKSPACE = "l0"


def _check_workspace(r: _Report) -> None:
    """FAIL-level: the two workspace vars gate `papervault serve` at boot
    (knowledge.store.graph.assert_safe_workspace). Catch a bad config here so doctor
    is green ⟺ serve can boot, instead of a green doctor then a boot-time abort."""
    click.echo(click.style("workspace", bold=True))
    neo = os.environ.get("NEO4J_WORKSPACE", "").strip()
    pg = os.environ.get("POSTGRES_WORKSPACE", "").strip()
    allow_prod = os.environ.get("KS_ALLOW_PROD_WORKSPACE", "").strip() == "1"
    if not neo or not pg:
        r.fail("workspace vars",
               f"NEO4J_WORKSPACE={neo!r} POSTGRES_WORKSPACE={pg!r} — both must be set "
               "and non-empty (serve aborts at boot otherwise)")
        return
    if neo != pg:
        r.fail("workspace vars",
               f"NEO4J_WORKSPACE={neo!r} != POSTGRES_WORKSPACE={pg!r} — graph and "
               "KV/vector/doc_status would split across namespaces")
        return
    if neo == _PROD_WORKSPACE and not allow_prod:
        r.fail("workspace vars",
               f"reserved prod workspace {_PROD_WORKSPACE!r} — set KS_ALLOW_PROD_WORKSPACE=1 "
               "to opt in on purpose, or use another name for dev/test")
        return
    r.ok("workspace vars", f"{neo}" + (" (prod opt-in)" if allow_prod and neo == _PROD_WORKSPACE else ""))


def _check_ocr_and_models(r: _Report) -> None:
    """WARN-level: OCR + local model weights. Neither is fatal at boot — OCR is optional
    at runtime and the BGE weights download lazily on first use — but a green doctor that
    hides "OCR down" / "4.5 GB pending download" is a foot-gun, so surface both."""
    click.echo(click.style("ocr + local models", bold=True))

    # --- MinerU OCR endpoint (optional at runtime) ---
    try:
        from papervault.library.mineru_client import endpoints_from_env
        eps = endpoints_from_env()
        base = eps[0].url.rstrip("/") if eps else "http://127.0.0.1:30000"
    except Exception:  # noqa: BLE001
        base = "http://127.0.0.1:30000"
    try:
        import urllib.request
        with urllib.request.urlopen(f"{base}/health", timeout=3) as resp:  # noqa: S310
            if getattr(resp, "status", 200) == 200:
                r.ok("mineru", base)
            else:
                r.warn("mineru", f"{base}/health returned {resp.status} — extracts will "
                                 "queue until the OCR server is up")
    except Exception as e:  # noqa: BLE001
        detail = str(e).splitlines()[0] if str(e) else type(e).__name__
        r.warn("mineru", f"{base} unreachable ({detail}) — extracts will queue until the "
                         "OCR server is up")

    # --- MinerU CLIENT parsing libs (import-level, NOT the server) ---
    # #56: the serving venv can have a healthy /health (server up) yet lack the
    # client-side parsing libs, so EVERY extract fails with mineru_import_failed
    # while doctor stays green — the exact silent-death that hid the 07-18 cutover
    # breakage. Probe the same lazy imports the extract path uses
    # (mineru_client.py :256-257) so the check catches a client/server split.
    try:
        from mineru.cli.common import aio_do_parse  # noqa: F401,WPS433
        from mineru_vl_utils.vlm_client.base_client import (  # noqa: F401,WPS433
            RequestError,
            ServerError,
        )
        r.ok("mineru client", "parsing libs importable")
    except Exception as e:  # noqa: BLE001
        detail = str(e).splitlines()[0] if str(e) else type(e).__name__
        r.warn("mineru client",
               f"import failed ({detail}) — the OCR server may be up but EVERY extract "
               "will return mineru_import_failed; install the mineru client libs into the "
               "serving venv (not the vllm extra)")

    # --- BGE embed + reranker checkpoints (download ~4.5 GB on first use) ---
    embed = os.environ.get("BGE_M3_MODEL_PATH", "BAAI/bge-m3")
    rerank = os.environ.get("BGE_RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
    missing = [name for name, spec in (("bge-m3", embed), ("bge-reranker", rerank))
               if not _model_available(spec)]
    if not missing:
        r.ok("bge weights", "embed + reranker present locally")
    else:
        r.warn("bge weights",
                f"{', '.join(missing)} not found locally — first build/query downloads "
                "~4.5 GB from Hugging Face")


def _model_available(spec: str) -> bool:
    """True iff a BGE checkpoint is locally available WITHOUT downloading anything.

    A local directory (with contents) counts as present; otherwise ``spec`` is treated as
    a Hugging Face repo id and probed against the local HF cache (a cached ``config.json``
    ⇒ the snapshot is on disk). Never triggers a network fetch."""
    p = Path(os.path.expanduser(spec))
    if p.is_dir() and any(p.iterdir()):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        for fname in ("config.json", "model.safetensors", "pytorch_model.bin"):
            if isinstance(try_to_load_from_cache(spec, fname), str):
                return True
    except Exception:  # noqa: BLE001 — hub absent / cache unreadable ⇒ treat as not-present
        return False
    return False


def _run_doctor_checks(skip_db: bool) -> _Report:
    """Run every preflight check and return the populated report (no process exit).
    Shared by the ``doctor`` and ``smoke`` commands."""
    r = _Report()
    _check_config(r)
    _check_paths(r)
    _check_workspace(r)
    _check_domain(r)
    _check_gpu(r)
    _check_ocr_and_models(r)
    if not skip_db:
        _check_databases(r)
    return r


@main.command()
@click.option("--skip-db", is_flag=True, help="skip Postgres/Neo4j connectivity checks")
def doctor(skip_db: bool) -> None:
    """Preflight: verify config, data dirs, workspace, GPU, OCR/models, databases, domain."""
    r = _run_doctor_checks(skip_db)
    click.echo()
    if r.failures:
        click.echo(click.style(f"FAIL — {r.failures} problem(s), {r.warnings} warning(s)", fg="red", bold=True))
        sys.exit(1)
    click.echo(click.style(f"OK — {r.warnings} warning(s)", fg="green", bold=True))


# --------------------------------------------------------------------------- #
#  smoke                                                                        #
# --------------------------------------------------------------------------- #

_SMOKE_EXPECTED_TOOLS = {"search_papers", "get_paper", "query"}
# DOI-shaped so get_paper resolves it read-only (no fuzzy LLM); guaranteed absent.
_SMOKE_MISSING_ID = "papervault-smoke-nonexistent-doi-10.0000/xyz"


async def _smoke_server_phase(full: bool) -> list[tuple[bool, str]]:
    """Boot ``papervault-mcp --stdio`` as a subprocess, connect over the MCP stdio
    client, and exercise the tool surface. Returns a list of (ok, message) steps.

    The subprocess is owned by the ``stdio_client`` context manager, which terminates
    and reaps it on exit (see the outer ``finally`` guarantee). The whole phase is
    time-boxed so a server that can't boot (e.g. auto-ingest ON with no reachable DB)
    degrades to a reported failure instead of hanging.

    The child server is pointed at a throwaway temp data dir (vault + knowledge store)
    so booting it — which runs the library's status migration + save — never writes into
    the operator's real vault. get_paper on a guaranteed-absent id returns not_found
    against the empty temp vault just the same."""
    import asyncio
    import tempfile

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    steps: list[tuple[bool, str]] = []

    async def _drive(session: "ClientSession") -> None:
        await asyncio.wait_for(session.initialize(), timeout=60)
        steps.append((True, "stdio server booted + session initialized"))

        # (c) exactly the three advertised tools
        listed = await asyncio.wait_for(session.list_tools(), timeout=30)
        names = {t.name for t in listed.tools}
        if names == _SMOKE_EXPECTED_TOOLS:
            steps.append((True, f"list_tools == {sorted(names)}"))
        else:
            steps.append((False, f"list_tools mismatch: got {sorted(names)}, "
                                  f"expected {sorted(_SMOKE_EXPECTED_TOOLS)}"))

        # (d) get_paper on a guaranteed-absent id → structured not_found/error (no LLM)
        res = await asyncio.wait_for(
            session.call_tool("get_paper", {"identifiers": _SMOKE_MISSING_ID}), timeout=30)
        status = _smoke_get_paper_status(res)
        if status in {"not_found", "error"}:
            steps.append((True, f"get_paper(absent) → status={status}"))
        else:
            steps.append((False, f"get_paper(absent) → unexpected status={status!r}"))

        # (--full) best-effort LLM-requiring calls; never gate the smoke result.
        if full:
            for tool, arg in (("search_papers", {"query": "smoke test probe query"}),
                              ("query", {"intent": "smoke test probe intent"})):
                try:
                    await asyncio.wait_for(session.call_tool(tool, arg), timeout=120)
                    steps.append((True, f"[--full, best-effort] {tool} returned"))
                except Exception as e:  # noqa: BLE001
                    steps.append((True, f"[--full, best-effort] {tool} did not complete "
                                        f"({type(e).__name__}) — informational only"))

    try:
        with tempfile.TemporaryDirectory(prefix="papervault-smoke-") as tmp:
            vault = str(Path(tmp) / "vault")
            child_env = os.environ.copy()   # realistic boot, but isolated data dir
            child_env["PAPERVAULT_DATA"] = tmp
            child_env["PAPERVAULT_VAULT"] = vault
            child_env["PAPERVAULT_STORAGE"] = str(Path(tmp) / "knowledge_store")
            child_env.pop("PAPER_LIBRARY_PATH", None)   # legacy real-vault fallback
            # Isolate the GRAPH STORE too, not just the filesystem vault (review F1): the
            # smoke child must never run a second scheduler against the host's live
            # Neo4j/PG workspace. The tool phase needs no scheduler and no graph.
            child_env["KS_AUTO_INGEST_ENABLED"] = "false"
            child_env["NEO4J_WORKSPACE"] = "papervault_smoke"
            child_env["POSTGRES_WORKSPACE"] = "papervault_smoke"
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "papervault.mcp", "--stdio", "--library-path", vault],
                env=child_env,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await _drive(session)
    except Exception as e:  # noqa: BLE001
        steps.append((False, f"server phase failed ({type(e).__name__}: {e}) — the stdio "
                             "server may need Postgres/Neo4j (e.g. KS_AUTO_INGEST_ENABLED=true "
                             "forces a graph/DB connection at boot); retry with the DBs up or "
                             "auto-ingest off"))
    return steps


def _smoke_get_paper_status(result) -> str:
    """Pull the effective status out of a get_paper CallToolResult (top-level status,
    else the single result item's status)."""
    import json
    try:
        text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        payload = json.loads(text)
    except Exception:  # noqa: BLE001
        return "error" if getattr(result, "isError", False) else "unparseable"
    if payload.get("status") == "error":
        return "error"
    results = payload.get("results") or []
    if results and isinstance(results[0], dict):
        return results[0].get("status", "unknown")
    return payload.get("status", "unknown")


@main.command()
@click.option("--skip-db", is_flag=True, help="skip Postgres/Neo4j connectivity checks in the doctor phase")
@click.option("--full", is_flag=True, help="also run best-effort LLM-requiring search_papers + query (labeled)")
def smoke(skip_db: bool, full: bool) -> None:
    """End-to-end sanity: run doctor, boot the MCP server over stdio, and round-trip its tools.

    Exits non-zero if doctor fails or any core tool step fails. LLM-requiring calls are
    kept OUT of the default run (use --full for a best-effort, non-gating pass)."""
    import asyncio

    click.echo(click.style("== doctor phase ==", bold=True))
    r = _run_doctor_checks(skip_db)
    if r.failures:
        click.echo()
        click.echo(click.style(
            f"SMOKE FAIL — doctor reported {r.failures} problem(s); aborting before the "
            "server phase", fg="red", bold=True))
        sys.exit(1)
    click.echo(click.style(f"doctor OK — {r.warnings} warning(s)", fg="green"))

    click.echo()
    click.echo(click.style("== server phase ==", bold=True))
    steps = asyncio.run(_smoke_server_phase(full))
    for ok, msg in steps:
        (r.ok if ok else r.fail)("smoke", msg)

    failures = sum(1 for ok, _ in steps if not ok)
    click.echo()
    if failures:
        click.echo(click.style(f"SMOKE FAIL — {failures} server-phase step(s) failed",
                               fg="red", bold=True))
        sys.exit(1)
    click.echo(click.style("SMOKE OK — doctor green + tool round-trip passed", fg="green", bold=True))


@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("mcp_args", nargs=-1, type=click.UNPROCESSED)
def serve(mcp_args: tuple[str, ...]) -> None:
    """Run the MCP server (passes through flags, e.g. `papervault serve --port 9000`)."""
    from papervault.mcp.__main__ import main as mcp_main
    sys.argv = ["papervault-mcp", *mcp_args]
    sys.exit(mcp_main())


# --------------------------------------------------------------------------- #
#  ingest-doc  (operator-supplied textbook / notebook → knowledge graph)       #
# --------------------------------------------------------------------------- #

@main.command("ingest-doc")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--kind", required=True, type=click.Choice(["textbook", "notebook"]),
              help="source class: textbook (canonical, #45) or notebook (private, #47)")
@click.option("--key", required=True,
              help="provenance key: textbook:AuthorYear or notebook:<idea>-<scope>")
@click.option("--max-tokens", type=int, default=None,
              help="max tokens per ingested section (default PAPERVAULT_DOC_MAX_TOKENS=2400)")
def ingest_doc(path: Path, kind: str, key: str, max_tokens: int | None) -> None:
    """Ingest an OPERATOR-supplied document into the knowledge graph with typed provenance.

    A markdown (.md) or plain-text (.txt) file the operator brings by hand — a canonical
    textbook / major review (--kind textbook, issue #45) or one of the lab's own executor
    notebooks (--kind notebook, issue #47) — enters the SAME LightRAG graph papers use, but
    OUTSIDE the paper-library pipeline. Markdown is split heading-aware (chapter/section
    boundaries) with a max-token cap; plain text uses the sentence-boundary chunker.

    Guards (both default OFF — a stock/shared/beta instance refuses):
      textbook → set PAPERVAULT_OPERATOR_SOURCES=1 (beta safety).
      notebook → set PAPERVAULT_PRIVATE_SOURCES=1  (UNPUBLISHED research; shared/beta
                 instances must NEVER ingest notebooks).

    CO-WRITE GATE: this is a MAINTENANCE-WINDOW operation — it writes the ledger + LightRAG
    graph the live service is concurrently serving, so it REFUSES to run while
    papervault.service (or the MinerU unit) is systemd-active (DB co-write corruption risk).
    Stop the service first, or set KS_INGEST_ALLOW_COTENANCY=1 to override deliberately.

    Out of scope for v1: retrieval weighting for canonical sources (#46), domain-pack
    textbook lists, PDF OCR (supply extracted md/txt), and re-ingest of appended notebooks
    (v1 is push-once — re-ingesting an already-ingested key is refused; --force/supersede is
    future work).
    """
    import asyncio

    from papervault.knowledge.ingest.operator_docs import (
        AlreadyIngestedError,
        SourceDisabledError,
        check_source_enabled,
        ingest_document,
        validate_key,
    )
    from papervault.ops_guards import require_services_stopped

    # Fail fast on key format + guard BEFORE booting the (heavy, workspace-gated) graph.
    try:
        validate_key(kind, key)
    except ValueError as e:
        raise click.ClickException(str(e))
    try:
        check_source_enabled(kind)
    except SourceDisabledError as e:
        raise click.ClickException(str(e))
    # Co-write gate (blocker, PR #54): refuse while the live service is active — an operator
    # doc ingest writing the same ledger + graph next to a serving instance risks DB co-write
    # corruption. Reuses the shared systemd gate (eval uses the same one, issue #32).
    require_services_stopped(
        override_env="KS_INGEST_ALLOW_COTENANCY",
        reason=("operator-doc ingest is a maintenance-window operation that writes the SAME "
                "ledger + graph the live service is concurrently serving — a co-writer risks "
                "DB co-write corruption."),
    )

    async def _go() -> dict:
        from papervault.knowledge.ledger.store import close_pool
        from papervault.knowledge.store.graph import close_graph, get_graph

        rag = await get_graph()  # workspace-gated (refuses prod 'l0' without opt-in)
        try:
            return await ingest_document(rag, kind, key, str(path), max_tokens=max_tokens)
        finally:
            # Flush LightRAG storage backends BEFORE dropping the singleton (finalize_storages
            # is the counterpart to initialize_storages) — a maintenance-window write must not
            # leave buffered graph/vector/KV state unpersisted. close_graph only nulls the ref.
            await rag.finalize_storages()
            await close_graph()
            await close_pool()

    try:
        result = asyncio.run(_go())
    except AlreadyIngestedError as e:
        raise click.ClickException(str(e))
    click.echo(
        f"ingested {result['key']} ({result['kind']}): {result['sections']} section(s) — "
        f"done={result.get('done', 0)} error={result.get('error', 0)} "
        f"pending={result.get('pending', 0)}"
    )


if __name__ == "__main__":
    main()
