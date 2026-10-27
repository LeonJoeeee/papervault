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
from uuid import uuid4

import anyio
from anyio.abc import TaskStatus
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import (
    MCP_SESSION_ID_HEADER,
    SESSION_ID_PATTERN,
    StreamableHTTPServerTransport,
)
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecurityMiddleware, TransportSecuritySettings
from mcp.types import INVALID_REQUEST, ErrorData, JSONRPCError
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

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


class ResilientHTTPSessionManager(StreamableHTTPSessionManager):
    """Keep stateful transports while recovering client IDs after session loss.

    Each transport still owns one ServerSession, preserving admission fairness
    and GET notification streams. Only the protocol initialization gate is
    stateless: a client remembering an ID can call tools after a server restart.
    This restores connectivity, not in-flight calls or missed SSE events.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # SDK 1.27 has no owner map; keep this lifecycle local across supported
        # SDK releases. None records anonymous ownership, including after loss.
        self._session_principals: dict[str, tuple[str, str | None, str | None] | None] = {}

    @asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        async with super().run():
            try:
                yield
            finally:
                self._session_principals.clear()

    async def _handle_stateful_request(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        # Reject untrusted requests before allocating a transport or recording an
        # owner. The transport also retains the SDK's ordinary validation.
        error = await TransportSecurityMiddleware(self.security_settings).validate_request(
            request, is_post=request.method == "POST",
        )
        if error is not None:
            await error(scope, receive, send)
            return
        session_id = request.headers.get(MCP_SESSION_ID_HEADER)
        if session_id is not None and not SESSION_ID_PATTERN.fullmatch(session_id):
            await self._session_error(scope, receive, send, 400, "Invalid session ID")
            return
        user = scope.get("user")
        owner = None
        if isinstance(user, AuthenticatedUser):
            token = user.access_token
            issuer = (getattr(token, "claims", None) or {}).get("iss")
            # Match the SDK 1.28 principal identity while supporting 1.27 tokens
            # that do not yet expose claims or subject.
            owner = (token.client_id, str(issuer) if issuer is not None else None,
                     getattr(token, "subject", None))

        async with self._session_creation_lock:
            transport = self._server_instances.get(session_id) if session_id is not None else None
            if session_id in self._session_principals and self._session_principals[session_id] != owner:
                await self._session_error(scope, receive, send, 404, "Session not found")
                return
            if transport is None:
                if session_id is not None and request.method not in ("POST", "GET"):
                    await self._session_error(scope, receive, send, 404, "Session not found")
                    return
                session_id = session_id if session_id is not None else uuid4().hex
                transport = StreamableHTTPServerTransport(
                    mcp_session_id=session_id,
                    is_json_response_enabled=self.json_response,
                    event_store=self.event_store,
                    security_settings=self.security_settings,
                    retry_interval=self.retry_interval,
                )
                self._server_instances[session_id] = transport
                # Retain owners across crashes, including anonymous ownership.
                self._session_principals[session_id] = owner
                assert self._task_group is not None
                await self._task_group.start(self._run_session, session_id, transport)

        # Never hold the creation lock during HTTP: a recovered GET can stay
        # open indefinitely and unrelated clients must still create sessions.
        if transport.idle_scope is not None and self.session_idle_timeout is not None:
            transport.idle_scope.deadline = anyio.current_time() + self.session_idle_timeout
        await transport.handle_request(scope, receive, send)

    async def _run_session(
        self,
        session_id: str,
        transport: StreamableHTTPServerTransport,
        *,
        task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        try:
            async with transport.connect() as (read_stream, write_stream):
                task_status.started()
                idle_scope = anyio.CancelScope()
                if self.session_idle_timeout is not None:
                    idle_scope.deadline = anyio.current_time() + self.session_idle_timeout
                    transport.idle_scope = idle_scope
                with idle_scope:
                    await self.app.run(
                        read_stream, write_stream, self.app.create_initialization_options(),
                        stateless=True,
                    )
                if idle_scope.cancelled_caught and not transport.is_terminated:
                    # Idle eviction is recoverable; remove before terminating so
                    # it is not confused with an explicit DELETE tombstone.
                    if self._server_instances.get(session_id) is transport:
                        del self._server_instances[session_id]
                    await transport.terminate()
        except Exception:
            logger.exception("MCP session %.64s crashed", session_id)
        finally:
            # Explicit DELETE marks the transport terminated before yielding.
            # Keep that instance as a tombstone for this manager's lifetime.
            # Identity guards keep an old runner from removing its replacement.
            if self._server_instances.get(session_id) is transport and not transport.is_terminated:
                del self._server_instances[session_id]

    @staticmethod
    async def _session_error(
        scope: Scope, receive: Receive, send: Send, status: int, message: str,
    ) -> None:
        body = JSONRPCError(
            jsonrpc="2.0", id="server-error", error=ErrorData(code=INVALID_REQUEST, message=message),
        )
        await Response(body.model_dump_json(by_alias=True, exclude_none=True),
                       status_code=status, media_type="application/json")(scope, receive, send)


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
    A long query returns status="running" and a query_id after a bounded wait. Retry with
    the same intent or intent="<query_id>" to collect its answer without restarting work.

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
    # Install before FastMCP lazily creates its HTTP app. Stdio remains unchanged.
    mcp._session_manager = ResilientHTTPSessionManager(
        app=mcp._mcp_server,
        event_store=mcp._event_store,
        retry_interval=mcp._retry_interval,
        json_response=mcp.settings.json_response,
        security_settings=mcp.settings.transport_security,
    )
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
