"""Frozen-corpus eval server for the issue #37 canon-lever arbitration.

The two search rulers (``mustfind_recall.py`` coverage, ``search_relevance_baseline.py``
precision) are MCP HTTP clients — they need a live ``search_papers`` endpoint. But the
production entrypoint (``papervault.mcp.__main__``) also starts the ingest/download/extract
queues + the knowledge graph-build scheduler at boot, so a live prod server MUTATES the
corpus on every call. A paired arbitration on a drifting corpus is invalid (2026-07-18).

This launcher stands up ONLY the ``search_papers`` / ``get_paper`` tools (``build_server``)
and serves them — WITHOUT the background-plane lifespan wrapper the prod entrypoint adds.
No queues, no reconcile, no MinerU monitor, no graph scheduler start → nothing drifts. Pair
it with ``PAPERVAULT_SEARCH_NO_INGEST=1`` so ``search_papers`` also skips its own Stage-3
write path: the library index stays byte-frozen for the whole ON-vs-OFF run.

Usage (from the worktree, with the frozen switch + the lever knob for the ON arm):
    PAPERVAULT_SEARCH_NO_INGEST=1 PAPERVAULT_CANON_RESERVED_SLOTS=0 \\
        .venv/bin/python scripts/frozen_eval_server.py --port 8080
"""
from __future__ import annotations

import argparse
import logging
import os

import uvicorn

# LIBRARY-ONLY factory (get_paper + search_papers) — NOT the unified server. The unified
# build also wires the knowledge (KS graph) plane, which opens Postgres/Neo4j pools we do
# not need for a search-path eval and which contend with the LIVE service's pool ("too many
# clients"). The library plane reads only the file-based vault snapshot → zero DB contention.
from papervault.library.mcp.server import build_server

log = logging.getLogger("papervault.frozen_eval")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--library-path", default=None,
                    help="vault root — point at a FROZEN snapshot so the eval reads a "
                         "byte-stable corpus while the live service keeps serving on 8080")
    ap.add_argument("--log-level", default="warning")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Fail-loud guard: NEVER let this harness mutate the PRODUCTION vault. Two safe modes:
    #   (a) NO_INGEST=1 → search is read-only (no writes anywhere), OR
    #   (b) --library-path points at an ISOLATED snapshot (≠ production vault) → ingest-on is
    #       fine because every write lands in the throwaway copy, real corpus untouched.
    # Refuse ONLY the unsafe combo: ingest-on AND aimed at the production vault.
    from papervault import config as _cfg
    prod_vault = os.path.realpath(os.path.expanduser(str(_cfg.VAULT_PATH)))
    target_vault = os.path.realpath(os.path.expanduser(str(args.library_path or _cfg.VAULT_PATH)))
    read_only = os.environ.get("PAPERVAULT_SEARCH_NO_INGEST") == "1"
    if not read_only and target_vault == prod_vault:
        raise SystemExit(
            f"REFUSING: ingest-on against the PRODUCTION vault ({target_vault}) would drift the "
            "live corpus. Set PAPERVAULT_SEARCH_NO_INGEST=1, or point --library-path at an "
            "isolated snapshot.")

    if args.library_path:
        os.environ["PAPERVAULT_VAULT"] = args.library_path   # Library reads this live at construction
    server = build_server(library_path=args.library_path)   # tools only — no bg planes
    server.settings.host = args.host
    server.settings.port = args.port
    app = server.streamable_http_app()   # FastMCP's own session lifespan; NO bg-plane wrapper
    log.warning("FROZEN eval server on %s:%d (vault=%s, no_ingest=%s, no queues, no scheduler, "
                "CANON_RESERVED_SLOTS=%s)", args.host, args.port, target_vault, read_only,
                os.environ.get("PAPERVAULT_CANON_RESERVED_SLOTS", "0"))
    uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port,
                                  log_level=args.log_level.lower())).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
