"""In-process async client for the persistent MinerU2.5-Pro vLLM server.

The daemon talks to a plain ``vllm serve`` of the MinerU2.5-Pro weights
(path A, §3.1 of the L2 SDD) via mineru's own in-process VLM http-client
(``aio_do_parse`` with ``backend="vlm-http-client"`` + ``server_url``). The
model loads ONCE in the server; this module sends one whole-PDF parse per
call and reads back the markdown that mineru writes to
``<out_dir>/<stem>/vlm/<stem>.md``.

**The transport-vs-extraction error split (C1, §2.2).** The single failure
bucket of the old cascade is split into TWO typed exceptions so the caller
(``extract.extract_md``) branches on ``type``, never on string-sniffing:

  - ``MineruTransportError`` — the server is unreachable / a connection died /
    the request never produced a usable response, OR an AMBIGUOUS bare-500 we
    choose not to charge (§2.3), OR a 400 ``Failed to load image`` (the server
    cannot decode the PNGs the client itself rendered — issue #134). The PAPER
    is fine; the SERVER (or net) is the problem. The caller MUST NOT charge an
    ``extract_attempt`` and MUST NOT terminalize — non-mutation lets classify
    re-route the paper next sweep.
  - ``MineruExtractionError`` — the server WAS reached and RESPONDED with a
    DISCRIMINABLE per-doc error verdict (400/422/structured-500/truncation),
    OR produced output the client judges unusable (thin/missing md). The PAPER
    is the problem. The caller charges an attempt; terminal at budget only.

These wrap what the mineru http-client actually raises —
``mineru_vl_utils.vlm_client.base_client.ServerError`` (a ``RuntimeError``)
and ``RequestError`` (a ``ValueError``) — plus the daemon wall-clock cap.

**Inline retry / failover (§2.3).** mineru's http-client already wraps a
native ``httpx_retries.RetryTransport`` (``Retry`` total=``max_retries``,
``backoff_factor=retry_backoff_factor``) that absorbs brief blips before any
exception escapes — that is wired via the ``max_retries`` /
``retry_backoff_factor`` kwargs, NOT hand-rolled. The loop in
``extract_mineru`` is the OUTER, endpoint-failover layer ON TOP of that: a
transport error on endpoint A retries on B (round-robin across the
comma-separated ``MINERU_URL`` list), with capped exponential backoff,
bounded by the daemon wall-clock deadline. An extraction-class error raises
immediately (no retry — a retry hits the same wall).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import resource
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# --------------------------- typed exceptions (C1) -------------------------


class MineruTransportError(Exception):
    """Server unreachable / connection-level failure / request never produced a
    usable response, OR an AMBIGUOUS bare-500 we choose not to charge (§2.3).

    The PAPER is fine; the SERVER (or net) is the problem. The caller MUST NOT
    charge an attempt, MUST NOT terminalize — non-mutation is the C1 mechanism
    that lets classify re-route the paper to EXTRACT on the next sweep.
    """


class MineruImageDecodeError(MineruTransportError):
    """The server answered HTTP 400 ``Failed to load image`` (issue #134).

    The client renders every page PNG itself, so a server that cannot decode them
    is broken, never the paper. The observed cause was a vLLM server process left
    running on a deleted venv: Pillow loads its format plugins lazily, the import
    failed, and every ``Image.open`` from then on raised. A ``MineruTransportError``
    subclass so ``extract_md``'s C1 transport arm parks the paper with NO attempt
    charged; the distinct type lets the caller count consecutive occurrences and
    trigger the server self-heal (``services.mineru_server``).
    """


class MineruExtractionError(Exception):
    """Server WAS reached and RESPONDED with a DISCRIMINABLE per-doc error
    verdict, OR produced output the client/gate judges unusable.

    The PAPER is the problem. Charges an ``extract_attempt``; terminal at the
    retry budget only.
    """


# ------------------------------- endpoints ---------------------------------


@dataclass(frozen=True)
class Endpoint:
    """One MinerU vLLM server URL (e.g. the 3090 at :30000 or the 5090 at
    :30001). ``label`` is for logging only."""

    label: str
    url: str


def endpoints_from_env() -> list[Endpoint]:
    """Parse the comma-separated ``MINERU_URL`` env into an ordered endpoint
    list. Steady-state has ONE endpoint; a 2-endpoint list enables the
    round-robin failover inside ``extract_mineru``.

    Defaults to the single steady-state 3090 server at ``127.0.0.1:30000``
    when ``MINERU_URL`` is unset, so the daemon has a sane default with no
    unit edit (the engine env-vars default IN-CODE, SDD §10).
    """
    raw = (os.environ.get("MINERU_URL", "") or "").strip()
    if not raw:
        raw = "http://127.0.0.1:30000"
    out: list[Endpoint] = []
    for i, part in enumerate(p.strip() for p in raw.split(",")):
        if part:
            out.append(Endpoint(label=f"ep{i}", url=part))
    return out


# ------------------------------- tunables ----------------------------------

# Per-request HTTP read+write cap, threaded into the mineru http-client as
# ``http_timeout`` (§4.2 D10-part-1). Bounds ONE HTTP call; a zero-byte read
# timeout classifies as transport (§2.2).
_MINERU_HTTP_TIMEOUT = float(
    os.environ.get("PAPER_LIBRARY_MINERU_HTTP_TIMEOUT", "600"))

# Native httpx_retries.Retry total inside the mineru http-client — absorbs
# brief blips WITHIN a single call before any exception escapes.
_MINERU_INLINE_NATIVE_RETRIES = int(
    os.environ.get("PAPER_LIBRARY_MINERU_NATIVE_RETRIES", "3"))
_MINERU_NATIVE_BACKOFF_FACTOR = float(
    os.environ.get("PAPER_LIBRARY_MINERU_NATIVE_BACKOFF", "0.5"))

# ── Request fan-out + socket ceiling (issue #96) ────────────────────────────
# mineru's http-client defaults to max_concurrency=100 requests in flight PER
# PARSE and an UNCAPPED httpx pool, so 8 concurrent OCR slots reached ~800
# sockets to the server (747 observed). ``max_concurrency`` bounds the requests
# one parse keeps in flight; the default 32 equals the MinerU unit's
# ``--max-num-seqs 32`` — the server never serves more at once, so more only
# queue there. ``max_connections`` caps the per-event-loop pool that ALL
# concurrent parses against one endpoint share; httpx queues requests beyond it
# (no error). Read at call time so an env override needs no reload.
_DEFAULT_MAX_CONCURRENCY = 32
_DEFAULT_MAX_CONNECTIONS = 64


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int((os.environ.get(name, "") or "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def client_limits() -> tuple[int, int]:
    """``(max_concurrency, max_connections)`` for the mineru http-client, from
    ``PAPER_LIBRARY_MINERU_MAX_CONCURRENCY`` (default 32) and
    ``PAPER_LIBRARY_MINERU_MAX_CONNECTIONS`` (default 64). A missing, invalid or
    non-positive value falls back to the default."""
    return (
        _positive_int_env("PAPER_LIBRARY_MINERU_MAX_CONCURRENCY",
                          _DEFAULT_MAX_CONCURRENCY),
        _positive_int_env("PAPER_LIBRARY_MINERU_MAX_CONNECTIONS",
                          _DEFAULT_MAX_CONNECTIONS),
    )


# OUTER endpoint-failover retry budget (in-MEMORY loop counter, NOT
# extract_attempts). Each iteration may switch endpoints round-robin.
_INLINE_RETRIES = int(
    os.environ.get("PAPER_LIBRARY_MINERU_INLINE_RETRIES", "3"))
_BACKOFF_BASE = float(
    os.environ.get("PAPER_LIBRARY_MINERU_BACKOFF_BASE", "1.0"))
_BACKOFF_CAP = float(
    os.environ.get("PAPER_LIBRARY_MINERU_BACKOFF_CAP", "30.0"))

# ── Reachability pre-check (issue #76 — MCP event-loop robustness) ──────────
# ``aio_do_parse``'s VLM-http-client CONSTRUCTION (``vlm_analyze._get_model_async``
# / ``__new__``) runs SYNCHRONOUS work on the S4 single event loop before any
# awaitable I/O — py-spy repeatedly caught MainThread stuck there when the MinerU
# server is unreachable (a fresh cold construction blocks ~1.3s; a hanging /
# modelscope-cold-fetch endpoint blocks far longer — 10-40s observed). On a
# restart with an extract backlog while MinerU is DOWN, that on-loop block plus
# the ~30s outer async retry churn per paper starves the MCP
# ``initialize``/``tools/list`` handshake (issue #76 / ai-research-lab#31 Cause B).
# The same event-loop-starvation class PR #75 fixed for the KS query path.
#
# Fix (mirrors PR #75's flag-gated, behaviour-preserving shape): before the
# blocking construction, run a FULLY-ASYNC ``/health`` probe with a SHORT timeout.
# A down/hanging server resolves to unreachable in ≤``_PRECHECK_TIMEOUT`` seconds
# WITHOUT ever blocking the loop, and we raise ``MineruTransportError`` —
# ``extract_md``'s existing C1 transport arm then defers the paper with NO charge
# (identical OUTCOME to the pre-#76 path, reached fast + off-block instead of via
# the on-loop construction stall). A reachable server (``/health`` 200) proceeds
# exactly as before, so an up MinerU is byte-unchanged. Health-up-but-model-not-
# loaded transients still fall through to ``aio_do_parse``'s bare-500→transport
# handling (unchanged). ``PAPER_LIBRARY_MINERU_PRECHECK=0`` restores the pre-#76
# straight-to-``aio_do_parse`` behaviour (diagnostic/revert lever).
_PRECHECK_ENABLED = os.environ.get(
    "PAPER_LIBRARY_MINERU_PRECHECK", "1").strip().lower() in ("1", "true", "yes")
_PRECHECK_TIMEOUT = float(
    os.environ.get("PAPER_LIBRARY_MINERU_PRECHECK_TIMEOUT", "2.5"))


async def _endpoint_health_ok(url: str, timeout: float) -> bool:
    """True iff ``url`` answers ``GET /health`` 200 within ``timeout`` seconds.

    Fully async (``httpx.AsyncClient``) so a DOWN (connection-refused) or HANGING
    (black-holed connect / slow ``/health``) server resolves to ``False`` WITHOUT
    blocking the event loop — the whole point of the #76 pre-check. Any error /
    timeout ⇒ ``False`` (treat as unreachable, conservative: a false-negative only
    costs a transport-defer + reconcile retry, never a wrong per-doc charge)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url.rstrip("/") + "/health")
        return r.status_code == 200
    except Exception:  # noqa: BLE001 — any failure ⇒ not reachable
        return False


async def _any_endpoint_ready(endpoints: list[Endpoint], timeout: float) -> bool:
    """True iff ANY endpoint answers ``/health`` 200 (probed concurrently).

    ``any`` — not ``all`` — preserves the round-robin failover contract: proceed
    to ``aio_do_parse`` while at least ONE card is up, so a single dead endpoint
    in a 2-endpoint set never blocks the healthy one. Steady state has ONE
    endpoint, so this degenerates to a single probe."""
    results = await asyncio.gather(
        *(_endpoint_health_ok(ep.url, timeout) for ep in endpoints),
        return_exceptions=True,
    )
    return any(r is True for r in results)


# ------------------------- ServerError discriminator -----------------------


def _status_of(exc: Exception) -> Optional[int]:
    """Best-effort parse of the HTTP status code off a ``ServerError`` message.

    The mineru http-client embeds the code in two shapes:
      - ``"Unexpected status code: [{code}], response body: ..."`` (parse loop)
      - ``"... Status code: {code}, response body: ..."`` (model-name probe)
    Returns the int code, or ``None`` if the message carries no parseable code.
    """
    msg = str(exc)
    m = re.search(r"status code:\s*\[(\d{3})\]", msg, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"status code:\s*(\d{3})\b", msg, re.I)
    if m:
        return int(m.group(1))
    return None


def _is_connect_fail(exc: Exception) -> bool:
    """A ``ServerError("Failed to connect to server ...")`` — the server unit is
    down / starting (the ``systemctl restart`` window, the case that motivates
    C1). Also catches generic connect/refused wording."""
    msg = str(exc).lower()
    return ("failed to connect to server" in msg
            or "connection refused" in msg
            or "name or service not known" in msg
            or "failed to establish a new connection" in msg)


def _is_zero_byte_timeout(exc: Exception) -> bool:
    """A read/connect timeout with no usable response — transport (§2.2). The
    mineru client surfaces these as ``ServerError`` (non-200/parse-fail) or the
    httpx timeout types leak through; treat any timeout wording as transport
    UNLESS the daemon wall-clock cap fired (that is handled separately as
    extraction-class while the server is alive)."""
    msg = str(exc).lower()
    return ("timeout" in msg or "timed out" in msg
            or "read timed out" in msg)


def _is_server_image_decode_fail(exc: Exception) -> bool:
    """A 400 whose body says the SERVER could not decode an image (issue #134).

    vLLM's ``load_bytes`` answers ``400 Failed to load image: cannot identify image
    file`` when its Pillow cannot open a PNG. The client generated that PNG, so the
    fault is server-side: transport-class, not a per-doc 400 verdict."""
    return _status_of(exc) == 400 and "failed to load image" in str(exc).lower()


def _is_structured_500(exc: Exception) -> bool:
    """A 500-ish error that carries a STRUCTURED per-doc error verdict — the
    ``ServerError("Error from server: {object:error ...}")`` case (a
    discriminable "server responds with a real error on THIS file"). This is
    EXTRACTION-class (§2.2), distinct from a bare 500 (transport by default)."""
    msg = str(exc).lower()
    return ("error from server:" in msg
            or "'object': 'error'" in msg
            or '"object": "error"' in msg
            or "object': 'error" in msg)


def alternate(endpoints: list[Endpoint], i: int) -> Endpoint:
    """Round-robin pick: iteration ``i`` selects ``endpoints[i % len]`` so a
    single dead card on attempt ``i`` fails over to the next on ``i+1``. With
    one endpoint this degenerates to retrying the same URL (steady state)."""
    return endpoints[i % len(endpoints)]


# ------------------------------- md read-back ------------------------------


def _read_md_from_outdir(outdir: Path, stem: str) -> str:
    """Read the markdown mineru wrote at ``<outdir>/<stem>/vlm/<stem>.md``.

    ``_async_process_vlm`` hardcodes ``parse_method="vlm"`` and ``prepare_env``
    builds ``<output_dir>/<pdf_file_name>/<parse_method>/`` then writes
    ``f"{pdf_file_name}.md"`` (verified common.py:180,309-310,437). A missing
    or thin file means the server produced nothing usable → EXTRACTION-class.
    """
    md_path = outdir / stem / "vlm" / f"{stem}.md"
    if not md_path.exists():
        raise MineruExtractionError(
            f"thin_or_missing_md: expected {md_path} not written by server")
    text = md_path.read_text()
    if not text or not text.strip():
        raise MineruExtractionError("thin_or_missing_md: empty md body")
    return text


# ---------------- transport lifecycle: shared MinerU client (#93) ----------
#
# The OCR path delegates the actual HTTP to mineru's ``aio_do_parse``, which
# drives a PROCESS-GLOBAL singleton ``HttpVlmClient``
# (``mineru.backend.vlm.vlm_analyze.ModelSingleton``). That client caches ONE
# ``httpx.AsyncClient`` PER EVENT LOOP and, on every loop switch, DROPS the
# other loop's client WITHOUT ``aclose()``
# (``mineru_vl_utils/vlm_client/http_client.py`` ``_aio_client`` ->
# ``self._aio_client_cache.clear()``). Each dropped client keeps its pooled
# keepalive sockets to the MinerU server OPEN, and is later GC'd against a
# now-closed loop.
#
# On a SINGLE persistent loop this is harmless (one client, reused — verified
# flat). But whenever OCR touches a SECOND event loop in the same process
# (e.g. a transient ``asyncio.run`` turn beside the daemon's persistent loop),
# every switch leaks the previous loop's pooled sockets: issue #93's outage
# (~306 ESTABLISHED sockets to the MinerU port, ~2 leaked per OCR'd book,
# 1024-fd cliff) and the ``'_UnixSelectorEventLoop' object has no attribute
# '_ssock'`` teardown error (a dropped client finalized on its dead loop).
#
# Fix (transport-lifecycle only — call semantics/timeouts/retry/error model are
# untouched): papervault owns the per-loop client's lifecycle. ``extract_mineru``
# brackets each parse with a per-loop reference count; when the LAST in-flight
# parse on a loop finishes (refcount hits 0), we ``aclose()`` THAT loop's cached
# mineru client and evict it from mineru's cache — ON that loop, while it is
# still alive, BEFORE it is torn down. Reference-counted so the up-to
# ``_EXTRACT_CONCURRENCY`` concurrent parses on the daemon loop never close a
# client another parse is still using. ALL access to mineru internals is GUARDED
# (getattr / try) so a mineru build without these attributes — or a unit test
# with a stubbed ``aio_do_parse`` — degrades to today's behaviour, never crashes.
# ``PAPER_LIBRARY_MINERU_CLIENT_CLOSE=0`` disables the close (revert lever).

_CLIENT_CLOSE_ENABLED = os.environ.get(
    "PAPER_LIBRARY_MINERU_CLIENT_CLOSE", "1").strip().lower() in ("1", "true", "yes")

# Per-event-loop count of extract_mineru calls currently in flight, keyed by the
# loop object (matches mineru's own per-loop cache key). Only ever touched from
# ON the loop it counts, so no lock is needed (asyncio is single-threaded per
# loop). Entries are popped at refcount 0, so a transient loop is not pinned.
_loop_parse_refcounts: "dict[object, int]" = {}


def _incref_current_loop() -> None:
    loop = asyncio.get_running_loop()
    _loop_parse_refcounts[loop] = _loop_parse_refcounts.get(loop, 0) + 1


async def _decref_current_loop_and_maybe_close() -> None:
    """Decrement the current loop's in-flight parse count; when it reaches 0,
    aclose + evict this loop's cached mineru async client (the #93 fix).

    Under a SERIAL drain (one book at a time on the daemon loop) this closes +
    rebuilds the client once per book. That is INTENTIONAL and negligible: the
    MinerU endpoint is loopback plain HTTP (no TLS handshake), so a fresh
    httpx.AsyncClient + TCP connect costs microseconds against a multi-second
    GPU parse — and the singleton predictor (model-name, etc.) is untouched, only
    its per-loop httpx client is rebuilt. When parses overlap (the up-to
    _EXTRACT_CONCURRENCY daemon path) the client is reused until the burst drains
    to 0, so the hot path keeps its pooled connections."""
    loop = asyncio.get_running_loop()
    n = _loop_parse_refcounts.get(loop, 0) - 1
    if n > 0:
        _loop_parse_refcounts[loop] = n
        return
    _loop_parse_refcounts.pop(loop, None)
    if _CLIENT_CLOSE_ENABLED:
        await _close_mineru_client_for_loop(loop)


async def _close_mineru_client_for_loop(loop) -> None:
    """aclose() + evict the mineru singleton's cached ``httpx.AsyncClient`` for
    ``loop`` (must be the running loop, so aclose runs on the client's own loop).

    Fully guarded/best-effort: a mineru without the expected internals, or any
    aclose error, is swallowed — closing the client must never fail an extract.
    """
    # Only touch mineru's singleton if its VLM backend is ALREADY imported in
    # this process (a real parse ran). Never TRIGGER the heavy torch/vllm import
    # just to clean up — a unit test with a stubbed aio_do_parse (mineru's VLM
    # backend never loaded) must stay light and hit the no-op path here.
    mod = sys.modules.get("mineru.backend.vlm.vlm_analyze")
    ModelSingleton = getattr(mod, "ModelSingleton", None) if mod is not None else None
    if ModelSingleton is None:  # pragma: no cover - no real OCR in this process
        return
    try:
        predictors = list(getattr(ModelSingleton(), "_models", {}).values())
    except Exception:  # noqa: BLE001 - never break the extract on cleanup
        return
    for predictor in predictors:
        client = getattr(predictor, "client", None)
        cache = getattr(client, "_aio_client_cache", None)
        if not isinstance(cache, dict):
            continue
        aio = cache.pop(loop, None)
        if aio is None:
            continue
        try:
            await aio.aclose()
        except Exception:  # noqa: BLE001 - best-effort transport teardown
            pass


# ------------------ client limits on mineru's singleton (#96) ---------------
#
# ``aio_do_parse`` forwards its kwargs to ``ModelSingleton().get_model``, which
# builds ONE predictor per (backend, model_path, server_url) the first time and
# reuses it. mineru 3.4.4 reads ``max_concurrency`` from those kwargs but never
# passes ``max_connections`` to the client, so the socket ceiling would be
# silently lost. Before each parse we therefore obtain the SAME singleton
# predictor (same key, same kwargs, off the loop exactly as mineru's own
# ``_get_model_async`` does) and set both limits on it and its HTTP client. The
# per-loop ``httpx.AsyncClient`` is built lazily from ``client.max_connections``
# during the parse, so it is created capped. Guarded like the #93 close hook: a
# mineru without these internals (or a stubbed test) just skips this step.


def _apply_client_limits(predictor, max_concurrency: int, max_connections: int) -> None:
    client = getattr(predictor, "client", None)
    if client is not None:
        if hasattr(client, "max_connections"):
            client.max_connections = max_connections
        if hasattr(client, "max_concurrency"):
            client.max_concurrency = max_concurrency
    if hasattr(predictor, "max_concurrency"):
        predictor.max_concurrency = max_concurrency


async def _prepare_capped_predictor(server_url: str, client_kwargs: dict) -> None:
    """Build (or fetch) mineru's singleton predictor for ``server_url`` and cap it.

    Errors from the construction itself (e.g. the model-name probe against a
    down server) propagate: ``aio_do_parse`` would have raised the same error
    from the same call, and the caller classifies it identically."""
    mod = sys.modules.get("mineru.backend.vlm.vlm_analyze")
    singleton_cls = getattr(mod, "ModelSingleton", None) if mod is not None else None
    if singleton_cls is None:
        return
    get_model = getattr(singleton_cls(), "get_model", None)
    if get_model is None:  # pragma: no cover - defensive, unknown mineru build
        return
    predictor = await asyncio.to_thread(
        get_model, "http-client", None, server_url, **client_kwargs)
    _apply_client_limits(predictor, client_kwargs["max_concurrency"],
                         client_kwargs["max_connections"])


# ------------------------- fd-watermark guard (#93) ------------------------

_FD_WARN_FRACTION = float(os.environ.get("PAPER_LIBRARY_FD_WARN_FRACTION", "0.6"))
_FD_WARN_INTERVAL = 60.0     # at most one warn line per this many seconds
_fd_warn_last_log = 0.0


def check_fd_watermark(logger: Optional[logging.Logger] = None) -> None:
    """Cheap, throttled open-fd watermark check (issue #93).

    Warn (at most once per ``_FD_WARN_INTERVAL`` s) when the process's open-fd
    count crosses ``_FD_WARN_FRACTION`` of the ``RLIMIT_NOFILE`` SOFT limit, so
    the next fd/socket leak fails LOUD early instead of silently at the
    'Too many open files' cliff. Best-effort: any error is swallowed.

    Placement/throttle note: the throttle timestamp advances ONLY when a warn
    actually fires, so in the healthy case (fds below the threshold) the early
    ``return`` never triggers and the ``resource.getrlimit`` + ``os.listdir``
    run on EVERY call — i.e. once per paper at the intended extract-worker host.
    That is deliberate and cheap (an fd-dir listing of a few hundred entries is
    microseconds); the 60 s throttle exists only to de-dupe the WARN LINE during
    a sustained high-fd condition, not to gate the (negligible) probe cost."""
    global _fd_warn_last_log
    try:
        now = time.monotonic()
        if now - _fd_warn_last_log < _FD_WARN_INTERVAL:
            return
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if not soft or soft <= 0:
            return
        n = len(os.listdir("/proc/self/fd"))
        if n >= _FD_WARN_FRACTION * soft:
            _fd_warn_last_log = now
            (logger or log).warning(
                "open-fd watermark: %d/%d fds (%.0f%% of the soft limit) — "
                "possible fd/socket leak (see issue #93)",
                n, soft, 100.0 * n / soft)
    except Exception:  # noqa: BLE001 - a health probe must never raise
        pass


# ------------------------------- main entry --------------------------------


async def extract_mineru(
    pdf_bytes: bytes,
    endpoints: list[Endpoint],
    *,
    stem: str = "doc",
    http_timeout: float = _MINERU_HTTP_TIMEOUT,
    wall_clock_deadline: Optional[float] = None,
) -> str:
    """One whole-PDF MinerU parse → returns the markdown string.

    Thin transport-lifecycle wrapper (issue #93) around :func:`_extract_mineru_impl`:
    it reference-counts in-flight parses PER EVENT LOOP and, when the last parse
    on this loop finishes, aclose()s + evicts this loop's cached mineru async
    client so its pooled sockets never leak on a later loop switch. The parse
    logic, error model, timeouts and retries are entirely in the impl below —
    UNCHANGED. See the "transport lifecycle" block above for the mechanism.
    """
    if not endpoints:
        raise MineruTransportError("no_endpoints_configured")
    _incref_current_loop()
    try:
        return await _extract_mineru_impl(
            pdf_bytes,
            endpoints,
            stem=stem,
            http_timeout=http_timeout,
            wall_clock_deadline=wall_clock_deadline,
        )
    finally:
        await _decref_current_loop_and_maybe_close()


async def _extract_mineru_impl(
    pdf_bytes: bytes,
    endpoints: list[Endpoint],
    *,
    stem: str = "doc",
    http_timeout: float = _MINERU_HTTP_TIMEOUT,
    wall_clock_deadline: Optional[float] = None,
) -> str:
    """One whole-PDF MinerU parse → returns the markdown string.

    Sends the WHOLE PDF (no page cap, no chunking) to a MinerU vLLM endpoint
    via ``aio_do_parse(backend="vlm-http-client")``, reads back the md mineru
    writes, and returns it. Implements the §2.3 outer endpoint-failover loop on
    top of the http-client's native per-call retry.

    Raises:
      ``MineruTransportError`` — server unreachable / connect-fail / bare-500 /
        502/503/504/429 / zero-byte timeout / inline budget exhausted, or its
        ``MineruImageDecodeError`` subclass (400 ``Failed to load image``: the
        server cannot decode the client's own PNGs, issue #134). The caller does
        NOT charge an attempt and does NOT terminalize (C1).
      ``MineruExtractionError`` — 400/422 / structured-500 / truncation /
        unexpected finish_reason / thin-or-missing md / the daemon wall-clock
        cap firing while the server is alive. The caller charges toward budget.

    Args:
      pdf_bytes: the whole PDF as bytes.
      endpoints: ordered endpoint list (round-robin failover target set).
      stem: output file stem (drives the read-back path); the caller passes the
        paper key so concurrent extractions never collide on the out_dir.
      http_timeout: per-request HTTP read+write cap (mineru ``http_timeout``).
      wall_clock_deadline: absolute ``time.time()`` deadline; crossing it while
        the server is alive and answering is a per-doc pathology → extraction.
    """
    if not endpoints:
        raise MineruTransportError("no_endpoints_configured")

    # Reachability pre-check (issue #76): fast-fail a DOWN/unreachable MinerU
    # here — on the loop but FULLY ASYNC (never blocks it) — BEFORE the
    # synchronous ``aio_do_parse`` client construction that py-spy caught
    # starving the MCP handshake, and before the heavy lazy import + ~30s retry
    # churn below. A down server ⇒ MineruTransportError ⇒ ``extract_md`` C1
    # transport arm ⇒ deferred, NO charge (same outcome as pre-#76, reached in
    # ≤_PRECHECK_TIMEOUT s async instead of via the on-loop stall). Off-flag
    # (PAPER_LIBRARY_MINERU_PRECHECK=0) restores the straight-to-parse path.
    if _PRECHECK_ENABLED and not await _any_endpoint_ready(
            endpoints, _PRECHECK_TIMEOUT):
        raise MineruTransportError(
            "mineru_unreachable_precheck: no endpoint answered GET /health 200 "
            f"within {_PRECHECK_TIMEOUT:.1f}s "
            f"({', '.join(e.url for e in endpoints)})")

    # Lazy import: mineru is a heavy, optionally-absent dep (it is installed in
    # the server unit's venv and the daemon's import path per SDD §3.0, but
    # importing papervault.library.extract must never require it). An import failure
    # here is a deploy/transport-class problem, not a per-doc defect.
    try:
        from mineru.cli.common import aio_do_parse  # noqa: WPS433
        from mineru_vl_utils.vlm_client.base_client import (  # noqa: WPS433
            RequestError,
            ServerError,
        )
    except Exception as exc:  # pragma: no cover - exercised only without mineru
        raise MineruTransportError(f"mineru_import_failed: {exc!r}") from exc

    os.environ.setdefault("MINERU_MODEL_SOURCE", "modelscope")

    last_transport: Optional[Exception] = None
    # Did the server ANSWER but the body was too slow (a read-timeout)? If the
    # wall-clock cap later fires, that means THIS doc is pathologically slow =
    # a per-doc property → EXTRACTION (charge → budget-terminal, preserves old
    # D9). A cap that fires while the server is DEAD (connect-fail / 5xx outage)
    # = TRANSPORT (don't charge; reconcile retries). slow_doc is set ONLY by a
    # read-timeout against a reachable server, never by 5xx/connect.
    slow_doc = False
    with tempfile.TemporaryDirectory(prefix="mineru_out_") as tmp:
        outdir = Path(tmp)
        for i in range(_INLINE_RETRIES + 1):
            if wall_clock_deadline is not None and time.time() >= wall_clock_deadline:
                # Wall-clock cap reached. If the server ANSWERED but the doc was
                # too slow (read-timeout) → per-doc pathology → EXTRACTION (charge,
                # preserves D9). If the server was DEAD/outaged → TRANSPORT.
                if slow_doc:
                    raise MineruExtractionError("wall_clock_cap_server_alive")
                raise MineruTransportError("wall_clock_cap_dead_server")
            ep = alternate(endpoints, i)
            # Per-call remaining budget so one slow call can't overshoot the
            # daemon wall-clock cap; if the deadline is sooner than the
            # per-request timeout, shrink the request timeout to fit.
            call_timeout = http_timeout
            if wall_clock_deadline is not None:
                remaining = wall_clock_deadline - time.time()
                if remaining <= 0:
                    if slow_doc:
                        raise MineruExtractionError("wall_clock_cap_server_alive")
                    raise MineruTransportError("wall_clock_cap_dead_server")
                call_timeout = max(1.0, min(http_timeout, remaining))
            max_concurrency, max_connections = client_limits()
            client_kwargs = {
                "http_timeout": call_timeout,
                "max_retries": _MINERU_INLINE_NATIVE_RETRIES,
                "retry_backoff_factor": _MINERU_NATIVE_BACKOFF_FACTOR,
                "max_concurrency": max_concurrency,
                "max_connections": max_connections,
            }
            try:
                await _prepare_capped_predictor(ep.url, client_kwargs)
                await aio_do_parse(
                    output_dir=str(outdir),
                    pdf_file_names=[stem],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=["en"],          # REQUIRED positional; VLM ignores
                    backend="vlm-http-client",   # do_parse strips "vlm-" → "http-client"
                    server_url=ep.url,
                    formula_enable=True,
                    table_enable=True,
                    image_analysis=False,
                    f_dump_md=True,
                    f_dump_middle_json=False,
                    f_dump_model_output=False,
                    f_dump_content_list=False,
                    f_draw_layout_bbox=False,
                    f_draw_span_bbox=False,
                    f_dump_orig_pdf=False,
                    start_page_id=0,
                    end_page_id=None,
                    # ── http-client kwargs (consumed via **kwargs, §3.3) ──
                    # http_timeout / max_retries / retry_backoff_factor, plus the
                    # #96 max_concurrency / max_connections caps.
                    # NOTE: NO connect_timeout — not plumbed through aio_do_parse.
                    **client_kwargs,
                )
                return _read_md_from_outdir(outdir, stem)

            except ServerError as exc:
                code = _status_of(exc)
                # ── server cannot decode the client's own PNGs (issue #134) ──
                # Checked BEFORE the 400 → extraction test: the server is broken,
                # not the paper. Raise at once (every page fails the same way, so an
                # inline retry only burns time); the caller parks it uncharged.
                if _is_server_image_decode_fail(exc):
                    raise MineruImageDecodeError(str(exc)) from exc
                # ── EXTRACTION-class (per-doc defect) → NO retry, charge ──
                if code in (400, 422) or _is_structured_500(exc):
                    raise MineruExtractionError(str(exc)) from exc
                # ── TRANSPORT-class → inline retry only ──
                if (code in (502, 503, 504, 429)
                        or _is_connect_fail(exc)
                        or _is_zero_byte_timeout(exc)):
                    # A read-timeout = the server CONNECTED but the body was too
                    # slow → mark slow_doc so a later wall-clock cap charges this
                    # doc. 5xx / connect-fail are server outages → leave it clear.
                    if _is_zero_byte_timeout(exc) and not _is_connect_fail(exc):
                        slow_doc = True
                    last_transport = exc
                    await _async_sleep_backoff(i, wall_clock_deadline)
                    continue
                if code == 500:
                    # BARE 500 = AMBIGUOUS → default TRANSPORT (§2.3). Do NOT
                    # charge, do NOT terminalize; budget + reconcile catch a
                    # genuinely-500ing doc. A systemctl restart returns bare
                    # 500 while weights load — must never condemn a paper.
                    last_transport = exc
                    await _async_sleep_backoff(i, wall_clock_deadline)
                    continue
                # Unknown ServerError (no parseable code, not connect/timeout):
                # treat as transport (conservative — never wrongly charge).
                last_transport = exc
                await _async_sleep_backoff(i, wall_clock_deadline)
                continue

            except RequestError as exc:
                # truncation / unexpected finish_reason / thin output → per-doc
                # extraction failure. No retry (a retry hits the same wall).
                raise MineruExtractionError(str(exc)) from exc

            except MineruExtractionError:
                # _read_md_from_outdir thin/missing → extraction-class. Re-raise.
                raise

            except asyncio.CancelledError:
                raise

            except Exception as exc:  # noqa: BLE001 - transport catch-all
                # httpx ConnectError / ReadTimeout / RemoteProtocolError / reset
                # etc. leak through as their own types — all connection-level →
                # transport (§2.2). Inline-retry; never charge on these.
                # A ReadTimeout means the server CONNECTED but the body was too
                # slow → slow_doc (a later wall-clock cap then charges this doc).
                if (type(exc).__name__ in ("ReadTimeout", "PoolTimeout", "WriteTimeout")
                        and not _is_connect_fail(exc)):
                    slow_doc = True
                last_transport = exc
                await _async_sleep_backoff(i, wall_clock_deadline)
                continue

    # Inline budget exhausted. If a reachable server kept read-timing-out
    # (slow_doc) → per-doc slowness → EXTRACTION (charge → budget-terminal,
    # preventing a forever hot-loop on a pathologically slow PDF). If by
    # 5xx/connect (server outage) → TRANSPORT → reconcile re-enqueues next sweep
    # (clean hand-off: inline = seconds-scale blips, reconcile = minutes+).
    if slow_doc:
        raise MineruExtractionError(
            f"slow_doc_inline_retries_exhausted: {last_transport!r}")
    raise MineruTransportError(
        f"inline_retries_exhausted: {last_transport!r}")


async def _async_sleep_backoff(i: int, wall_clock_deadline: Optional[float]) -> None:
    """Capped exponential backoff between outer failover iterations, clamped to
    the daemon wall-clock deadline so backoff never overshoots the cap.

    Non-blocking (``asyncio.sleep``) — a synchronous ``time.sleep`` here would
    freeze the whole event loop and every concurrent extraction.
    """
    delay = min(_BACKOFF_BASE * (2 ** i), _BACKOFF_CAP)
    if wall_clock_deadline is not None:
        remaining = wall_clock_deadline - time.time()
        delay = max(0.0, min(delay, remaining))
    if delay > 0:
        await asyncio.sleep(delay)
