"""KS operator CLI (`python -m papervault.knowledge.cli`) = the v1 OPERATOR surface.

The MCP surface is just query(intent) for the executor (mcp/server.py). Operator-only
ops — diagnostics, ledger/doc_status status views, by-id doc lookup — live here, NOT on
the MCP surface:
    python -m papervault.knowledge.cli query "PINN 在 SEP 现状"     # same §6.4 pipeline the MCP tool wraps
    python -m papervault.knowledge.cli query "STEREO SEPT 能量" --json
    python -m papervault.knowledge.cli stats                        # §6.5 step5 acceptance + ops truth view
    python -m papervault.knowledge.cli get paper:Reames2023         # by-id LightRAG doc_status row
    python -m papervault.knowledge.cli rollback-abstracts           # #144 undo, dry-run by default

All reads are workspace-gated via assert_safe_workspace() (SDD §6.0): hitting prod 'l0'
without KS_ALLOW_PROD_WORKSPACE=1 refuses to run = explicit failure, never a silent
cross-workspace read. The only writer is `rollback-abstracts --apply`, which refuses while
papervault.service is running (the live scheduler writes the same ledger + graph).
"""
from __future__ import annotations

import asyncio
import json
import sys

import click



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
    sources = result.get("cited_sources") or []
    if sources:
        click.echo(f"--- cited_sources ({len(sources)}) ---")
        click.echo(", ".join(sources))
    abstract_only = result.get("abstract_only_papers") or []
    if abstract_only:
        click.echo(f"--- abstract_only_papers ({len(abstract_only)}; abstract only, no full text) ---")
        click.echo(", ".join(abstract_only))


@cli.command()
@click.option("--json", "json_out", is_flag=True, help="Output raw JSON")
def stats(json_out: bool) -> None:
    """KS status truth view (SDD §6.8): KS ledger + LightRAG doc_status counts.

    This is the §6.5 step5 full-rebuild acceptance view: the rebuild is done when
    ledger(done + done_meta + done_abstract) == |idx| and doc_status has no processing/failed残留.
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
                "done_plus_done_meta": done,
                # §6.5 step5: == |idx| when rebuild complete (#144 adds the done_abstract class)
                "done_terminal": done + ledger_counts.get("done_abstract", 0),
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
    click.echo(f"  -> done+done_meta+done_abstract = {led['done_terminal']} (== |idx| when rebuilt)")
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


# `is-active` answers during which papervault.service may still run a scheduler round.
_SERVICE_RUNNING_STATES = ("active", "activating", "deactivating", "reloading")


def _active_service() -> list[str]:
    """papervault.service when systemd reports it running, else []."""
    from papervault.ops_guards import active_service_units

    return active_service_units(("papervault.service",), states=_SERVICE_RUNNING_STATES)


@cli.command("rollback-abstracts")
@click.option("--apply", is_flag=True,
              help="Delete the docs and rewrite the ledger (default: dry run, nothing written)")
@click.option("--json", "json_out", is_flag=True, help="Output raw JSON")
def rollback_abstracts(apply: bool, json_out: bool) -> None:
    """Undo #144: delete every abstract-only doc and return its ledger row to done_meta.

    Dry run by default (counts the class from the ledger; no graph, no writes). --apply deletes
    each abstract-only `paper:<key>` doc and writes its row done_meta/META. It refuses while
    papervault.service is running: the live scheduler writes the same ledger + graph (a race), and
    LightRAG refuses deletes while its pipeline is busy.

    To keep the rollback, restart the service with KS_ABSTRACT_DOCS=0 — otherwise the next round
    fingerprints these papers as abstract docs again and re-ingests them. (KS_ABSTRACT_DOCS=0 alone
    also rolls back, gradually, through the normal scheduler rounds.)
    """
    from papervault.knowledge.ingest.abstract_doc import abstract_docs_enabled

    if apply:
        running = _active_service()
        if running:
            click.echo(f"ABORT: {', '.join(running)} is running — it writes the same ledger + graph "
                       "and would race this rollback. Stop it first "
                       "(systemctl --user stop papervault.service), then re-run with --apply.",
                       err=True)
            sys.exit(2)

    async def _go() -> dict:
        from papervault.knowledge.ingest.distill import rollback_abstract_docs
        from papervault.knowledge.ledger.store import close_pool
        from papervault.knowledge.store.graph import close_graph, get_graph

        if not apply:
            try:
                return await rollback_abstract_docs(None, apply=False)
            finally:
                await close_pool()
        rag = await get_graph()  # workspace-gated (refuses prod 'l0' without opt-in)
        try:
            return await rollback_abstract_docs(rag, apply=True)
        finally:
            await rag.finalize_storages()
            await close_graph()
            await close_pool()

    result = asyncio.run(_go())
    result["dry_run"] = not apply
    result["abstract_docs_enabled"] = abstract_docs_enabled()
    if json_out:
        click.echo(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        return
    mode = "WRITE" if apply else "DRY-RUN (nothing written)"
    click.echo(f"rollback-abstracts [{mode}]: {result['abstract_rows']} abstract-only ledger row(s)")
    for status, n in sorted(result["by_status"].items()):
        click.echo(f"  {status:<14} {n}")
    if apply:
        click.echo(f"reverted to done_meta: {result['reverted']}; delete failed: "
                   f"{len(result['failed'])}; skipped (doc is full text): "
                   f"{len(result['skipped_not_abstract'])}")
        for key in (result["failed"] + result["skipped_not_abstract"])[:20]:
            click.echo(f"    kept {key}")
    else:
        click.echo("dry-run: stop papervault.service, then re-run with --apply to write.")
    click.echo("Set KS_ABSTRACT_DOCS=0 in the service env before restarting papervault.service, "
               "or the scheduler re-ingests these papers as abstract docs."
               + ("" if result["abstract_docs_enabled"] else " (KS_ABSTRACT_DOCS=0 is set here.)"))


def main() -> None:
    """Entry point for the operator CLI (`python -m papervault.knowledge.cli`)."""
    cli()


if __name__ == "__main__":
    main()
