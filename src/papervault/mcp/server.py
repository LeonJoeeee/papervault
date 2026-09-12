"""The unified papervault MCP server — ONE FastMCP process, three tools.

ADR-0002: the library plane (``search_papers`` + ``get_paper``) and the knowledge
plane (``query``) are exposed by a single server. This module composes them:

  * library tools come from :func:`papervault.library.mcp.server.build_server`,
    called with our own FastMCP instance so its tools register onto it (and the
    Library + stage queues get stashed on the instance for the boot sequence);
  * the knowledge ``query`` tool is registered directly from the knowledge server
    module (its heartbeat + pipeline live there).

Background lifecycle (library queues + reconcile + MinerU monitor, and the
knowledge scheduler) is owned by the process entrypoint (:mod:`papervault.mcp.__main__`),
which starts it once at server boot. The per-session lifespan below only
idempotently ensures the knowledge scheduler is up (streamable-http runs a
FastMCP lifespan once per client session, not once at boot).
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from papervault import config
from papervault.knowledge.mcp.server import query as _knowledge_query
from papervault.knowledge.mcp.server import start_background as _start_knowledge_bg
from papervault.library.mcp.server import build_server as _build_library
from papervault.mcp.access_log import install_access_log
from papervault.mcp.admission import install_admission

logger = logging.getLogger(__name__)

# FastMCP's default loopback lists; the SDK does not export these constants.
LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
LOOPBACK_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def build_transport_security(hosts: list[str]) -> TransportSecuritySettings | None:
    """Extend SDK loopback protection with parsed PAPERVAULT_MCP_ALLOWED_HOSTS."""
    if not hosts:
        return None
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=LOOPBACK_HOSTS + hosts,
        allowed_origins=LOOPBACK_ORIGINS + [
            f"{scheme}://{host}" for host in hosts for scheme in ("https", "http")
        ],
    )


INSTRUCTIONS = """
papervault is your literature + knowledge layer. It gives a research agent two things:
the PAPERS themselves and DIGESTED knowledge distilled from them. Three tools:

- search_papers(query): DISCOVER literature. Describe (in prose, not keywords) what you're
    working on and what you want to find; a backend LLM parses the intent, searches multiple
    sources, and returns papers ranked by relevance. Relevant finds are auto-ingested
    (downloaded + OCR'd) in the background so they become readable and, later, queryable.
- get_paper(identifiers): look up ONE paper — or a LIST in one call — by citation key / DOI /
    arXiv id / fuzzy text. Returns a minimal record per identifier plus EXACTLY ONE of
    ``text_path`` (Read it for verbatim full text) or ``text_status`` (pending / metadata_only /
    download_failed / extract_failed). READ-ONLY: a paper not in the library returns not_found.
- query(intent): ASK the knowledge base. Pose ONE full natural-language question (what you're
    doing + what you want to know), like briefing a senior colleague. You get a cited prose
    answer (inline [paper_key] cites) + a cited_papers list + a coverage honesty signal.

Typical flow: query() for what's already known → search_papers() to pull fresher literature →
get_paper() to open specific full texts. Knowledge lags the library by minutes (ingest +
distill), so a freshly-found paper is readable via get_paper before query() can cite it.
""".strip()


@asynccontextmanager
async def _lifespan(_app: "FastMCP") -> AsyncIterator[None]:
    # Per-session (streamable-http) hook: idempotently ensure the knowledge scheduler is up.
    # Server-lifetime background is started once at boot by __main__; teardown is at process exit.
    await _start_knowledge_bg()
    yield


def build_server(library_path: str | None = None) -> FastMCP:
    """Build the single papervault FastMCP with all three tools registered."""
    security = build_transport_security(config.MCP_ALLOWED_HOSTS)
    if security is None:
        mcp = FastMCP("papervault", instructions=INSTRUCTIONS, lifespan=_lifespan)
    else:
        mcp = FastMCP("papervault", instructions=INSTRUCTIONS, lifespan=_lifespan,
                      transport_security=security)
        logger.info("PAPERVAULT_MCP_ALLOWED_HOSTS extra hosts: %s", config.MCP_ALLOWED_HOSTS)
    # Library plane: registers search_papers + get_paper onto `mcp`, stashes Library + queues.
    _build_library(library_path=library_path, mcp=mcp)
    # Knowledge plane: register the query tool (its module owns the heartbeat + pipeline).
    mcp.tool()(_knowledge_query)
    # Load control first (issue #28), THEN observability — the access log wraps the
    # admission layer, so MCPCALL durations stay the caller-experienced truth (queue
    # wait included) and busy answers still produce an MCPCALL line.
    install_admission(mcp)
    install_access_log(mcp)
    return mcp
