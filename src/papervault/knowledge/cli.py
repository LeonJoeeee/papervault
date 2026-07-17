"""KS CLI — `ks` entry point = the v1 OPERATOR surface (SDD §6.8).

The MCP surface is just query(intent) for the executor (mcp/server.py). Operator-only
ops — diagnostics, ledger/doc_status status views, by-id doc lookup — live here, NOT on
the MCP surface:
    uv run ks query "PINN 在 SEP 现状"            # same §6.4 pipeline the MCP tool wraps
    uv run ks query "STEREO SEPT 能量" --json
    uv run ks stats                                # §6.5 step5 acceptance + ops truth view
    uv run ks get paper:Reames2023                 # by-id LightRAG doc_status row

All reads are workspace-gated via assert_safe_workspace() (SDD §6.0): hitting prod 'l0'
without KS_ALLOW_PROD_WORKSPACE=1 refuses to run = explicit failure, never a silent
cross-workspace read. Nothing here writes the DB.
"""
from __future__ import annotations

import asyncio
import json
import sys

import click

from papervault.knowledge.config import CONFIG


async def _count_doc_status(workspace: str) -> dict[str, int]:
    """Per-status counts from lightrag_doc_status for one workspace (SDD §6.8).

    Reads the LightRAG doc_status table directly over the ledger's PG DSN — no need to
    spin up the full LightRAG/BGE/Neo4j stack just to count rows. workspace is the
    POSTGRES_WORKSPACE-resolved value (assert_safe_workspace()); the table is keyed
    (workspace, id) so the filter is mandatory or we'd mix prod + probe (postgres_impl.py).
    """
    from papervault.knowledge.ledger.store import _conn  # same PG pool / DSN

    async with _conn() as conn:
        async with conn.cursor() as cur:
            # Table may be absent on a never-initialized DB → treat as empty.
            await cur.execute(
                "SELECT to_regclass('public.lightrag_doc_status') IS NOT NULL"
            )
            (exists,) = await cur.fetchone()
            if not exists:
                return {}
            await cur.execute(
                "SELECT status, count(*) FROM lightrag_doc_status "
                "WHERE workspace=%s GROUP BY status",
                (workspace,),
            )
            rows = await cur.fetchall()
    return {status: n for status, n in rows}


@click.group()
def cli() -> None:
    """Knowledge System operator CLI (SDD §6.8)."""


@cli.command()
@click.argument("intent")
@click.option("--json", "json_out", is_flag=True, help="Output raw JSON")
def query(intent: str, json_out: bool) -> None:
    """Query KS in plain language (same §6.4 pipeline as the MCP tool).

    Returns {answer (prose with [paper_key] cites), cited_papers, kb_coverage}.
    """
    async def _go() -> dict:
        from papervault.knowledge.ledger.store import close_pool
        from papervault.knowledge.query.aquery import query as _query
        from papervault.knowledge.store.graph import close_graph

        try:
            return await _query(intent=intent)
        finally:
            await close_graph()
            await close_pool()

    result = asyncio.run(_go())

    if json_out:
        click.echo(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        return

    click.echo(f"\n=== Query: {intent!r} ===")
    click.echo(f"kb_coverage: {result.get('kb_coverage')}")
    click.echo(f"\n{result.get('answer', '')}\n")
    cited = result.get("cited_papers") or []
    if cited:
        click.echo(f"--- cited_papers ({len(cited)}) ---")
        click.echo(", ".join(cited))


@cli.command()
@click.option("--json", "json_out", is_flag=True, help="Output raw JSON")
def stats(json_out: bool) -> None:
    """KS status truth view (SDD §6.8): KS ledger + LightRAG doc_status counts.

    This is the §6.5 step5 full-rebuild acceptance view: the rebuild is done when
    ledger(done + done_meta) == |idx| and doc_status has no processing/failed残留.
    """
    async def _go() -> dict:
        from papervault.knowledge.ledger.store import close_pool, count_by_status
        from papervault.knowledge.store.graph import assert_safe_workspace

        # Same gate the ingest/query paths use — also resolves the workspace to filter
        # doc_status by (refuses prod 'l0' without explicit opt-in).
        workspace = assert_safe_workspace()
        try:
            ledger_counts = await count_by_status("paper")
            doc_status_counts = await _count_doc_status(workspace)
        finally:
            await close_pool()
        ledger_total = sum(ledger_counts.values())
        done = ledger_counts.get("done", 0) + ledger_counts.get("done_meta", 0)
        return {
            "workspace": workspace,
            "ledger": {
                "by_status": ledger_counts,
                "total": ledger_total,
                "done_plus_done_meta": done,  # §6.5 step5: == |idx| when rebuild complete
            },
            "doc_status": {"by_status": doc_status_counts},
        }

    result = asyncio.run(_go())

    if json_out:
        click.echo(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        return

    click.echo(f"\n=== KS stats (workspace={result['workspace']}) ===")
    led = result["ledger"]
    click.echo(f"\nledger (ingest_source=paper), total={led['total']}:")
    for status, n in sorted(led["by_status"].items()):
        click.echo(f"  {status:<14} {n}")
    click.echo(f"  -> done+done_meta = {led['done_plus_done_meta']} (== |idx| when rebuilt)")
    ds = result["doc_status"]["by_status"]
    click.echo("\nlightrag_doc_status:")
    if not ds:
        click.echo("  (none — table absent or empty for this workspace)")
    for status, n in sorted(ds.items()):
        click.echo(f"  {status:<14} {n}")


@cli.command()
@click.argument("doc_id")
@click.option("--json", "json_out", is_flag=True, help="Output raw JSON")
def get(doc_id: str, json_out: bool) -> None:
    """Look up one LightRAG doc_status row by doc_id (SDD §6.8 / §13).

    doc_id = paper:{key} | textbook:{isbn}#ch{n} | web:{url}. Shows
    status / content_summary / chunks_count / error_msg / track_id.
    """
    async def _go() -> dict | None:
        from papervault.knowledge.ledger.store import close_pool
        from papervault.knowledge.store.graph import close_graph, get_graph

        rag = await get_graph()  # workspace-gated
        try:
            statuses = await rag.aget_docs_by_ids([doc_id])
            st = statuses.get(doc_id)
            if st is None:
                return None
            # aget_docs_by_ids returns {doc_id: plain dict} at runtime (NOT DocProcessingStatus
            # objects — LightRAG passes get_by_id() straight through; PG/JSON backends both
            # return dicts). So read by dict-subscript; getattr on a dict yields None for every
            # field, reporting an existing doc as all-null. See SDD §6.6 ★ / §6.8.
            def _f(key):
                return st.get(key) if isinstance(st, dict) else getattr(st, key, None)

            status_val = _f("status")
            return {
                "doc_id": doc_id,
                "status": getattr(status_val, "value", status_val),  # enum → str if ever an obj
                "content_summary": _f("content_summary"),
                "content_length": _f("content_length"),
                "chunks_count": _f("chunks_count"),
                "file_path": _f("file_path"),
                "track_id": _f("track_id"),
                "error_msg": _f("error_msg"),
                "created_at": _f("created_at"),
                "updated_at": _f("updated_at"),
            }
        finally:
            await close_graph()
            await close_pool()

    row = asyncio.run(_go())
    if row is None:
        click.echo(f"Not found in doc_status: {doc_id}", err=True)
        sys.exit(1)
    click.echo(json.dumps(row, indent=2, default=str, ensure_ascii=False))


def main() -> None:
    """Entry point for `ks` script."""
    cli()


if __name__ == "__main__":
    main()
