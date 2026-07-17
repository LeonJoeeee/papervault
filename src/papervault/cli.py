"""papervault CLI — operator entrypoint.

    papervault doctor     # preflight: config, data dirs, GPU, databases, LLM endpoint
    papervault serve      # run the MCP server (same as `papervault-mcp`)
"""
from __future__ import annotations

import os
import sys

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
    # Postgres
    try:
        import psycopg
        dsn = (f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
               f"port={os.getenv('POSTGRES_PORT', '5432')} "
               f"user={os.getenv('POSTGRES_USER', 'papervault')} "
               f"password={os.getenv('POSTGRES_PASSWORD', '')} "
               f"dbname={os.getenv('POSTGRES_DB', 'papervault')}")
        with psycopg.connect(dsn, connect_timeout=5):
            r.ok("postgres", f"{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}")
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


@main.command()
@click.option("--skip-db", is_flag=True, help="skip Postgres/Neo4j connectivity checks")
def doctor(skip_db: bool) -> None:
    """Preflight: verify config, data dirs, GPU, databases, and the domain pack."""
    r = _Report()
    _check_config(r)
    _check_paths(r)
    _check_domain(r)
    _check_gpu(r)
    if not skip_db:
        _check_databases(r)
    click.echo()
    if r.failures:
        click.echo(click.style(f"FAIL — {r.failures} problem(s), {r.warnings} warning(s)", fg="red", bold=True))
        sys.exit(1)
    click.echo(click.style(f"OK — {r.warnings} warning(s)", fg="green", bold=True))


@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("mcp_args", nargs=-1, type=click.UNPROCESSED)
def serve(mcp_args: tuple[str, ...]) -> None:
    """Run the MCP server (passes through flags, e.g. `papervault serve --port 9000`)."""
    from papervault.mcp.__main__ import main as mcp_main
    sys.argv = ["papervault-mcp", *mcp_args]
    sys.exit(mcp_main())


if __name__ == "__main__":
    main()
