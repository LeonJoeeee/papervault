"""HTTP task client for the independently managed MinerU OCR service.

Submit a whole PDF to official ``mineru-api``, poll its FIFO task queue, then
read the completed markdown. No MinerU libraries, page rendering, model state
or service lifecycle belong in this process.

C1 keeps service/connection failures and queue-wait timeouts uncharged through
``MineruTransportError``. Per-document verdicts and read/wall-clock timeouts
observed during active processing raise ``MineruExtractionError``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)


class MineruTransportError(Exception):
    """Service unavailable or transport failed; the caller must not charge an attempt."""


class MineruImageDecodeError(MineruTransportError):
    """HTTP 400 says the service could not decode an image; preserve uncharged deferral."""


class MineruExtractionError(Exception):
    """A per-document failure or unusable output; the caller charges its attempt budget."""


@dataclass(frozen=True)
class Endpoint:
    """One MinerU consumer URL (default gate: ``http://127.0.0.1:30000``).

    ``label`` is for logging only.
    """

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


# Per-request read/write timeout; the caller's absolute deadline also bounds
# the complete submit/poll/result flow and every retry/backoff.
_MINERU_HTTP_TIMEOUT = float(os.environ.get("PAPER_LIBRARY_MINERU_HTTP_TIMEOUT", "600"))

_DEFAULT_MAX_CONCURRENCY = 32
_DEFAULT_MAX_CONNECTIONS = 64


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int((os.environ.get(name, "") or "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def client_limits() -> tuple[int, int]:
    """``(max_concurrency, max_connections)`` for the HTTP task client, from
    ``PAPER_LIBRARY_MINERU_MAX_CONCURRENCY`` (default 32) and
    ``PAPER_LIBRARY_MINERU_MAX_CONNECTIONS`` (default 64). A missing, invalid or
    non-positive value falls back to the default."""
    return (
        _positive_int_env("PAPER_LIBRARY_MINERU_MAX_CONCURRENCY",
                          _DEFAULT_MAX_CONCURRENCY),
        _positive_int_env("PAPER_LIBRARY_MINERU_MAX_CONNECTIONS",
                          _DEFAULT_MAX_CONNECTIONS),
    )


_INLINE_RETRIES = int(
    os.environ.get("PAPER_LIBRARY_MINERU_INLINE_RETRIES", "3"))
_BACKOFF_BASE = float(
    os.environ.get("PAPER_LIBRARY_MINERU_BACKOFF_BASE", "1.0"))
_BACKOFF_CAP = float(
    os.environ.get("PAPER_LIBRARY_MINERU_BACKOFF_CAP", "30.0"))

_TASK_POLL_INTERVAL = 1.0


def alternate(endpoints: list[Endpoint], i: int) -> Endpoint:
    """Round-robin pick: iteration ``i`` selects ``endpoints[i % len]`` so a
    single dead card on attempt ``i`` fails over to the next on ``i+1``. With
    one endpoint this degenerates to retrying the same URL (steady state)."""
    return endpoints[i % len(endpoints)]


_MALLOC_TRIM_ENABLED = os.environ.get(
    "PAPER_LIBRARY_MINERU_MALLOC_TRIM", "1").strip().lower() in ("1", "true", "yes")
_UNLOADED = object()
_malloc_trim_fn = _UNLOADED   # resolved on first use; None = unavailable here


def _load_malloc_trim():
    """glibc's ``malloc_trim`` as a ctypes function, or None off glibc."""
    try:
        if not os.confstr("CS_GNU_LIBC_VERSION"):
            return None
    except (AttributeError, ValueError, OSError):
        return None
    try:
        import ctypes
        fn = ctypes.CDLL(None).malloc_trim   # the libc already in this process
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_size_t]
    fn.restype = ctypes.c_int
    return fn


def trim_heap() -> bool:
    """Return freed heap memory to the OS with ``malloc_trim(0)``.

    True when the trim ran; False when disabled or not on glibc. Never raises
    for an unavailable symbol; the caller still guards against surprises."""
    global _malloc_trim_fn
    if not _MALLOC_TRIM_ENABLED:
        return False
    if _malloc_trim_fn is _UNLOADED:
        _malloc_trim_fn = _load_malloc_trim()
    if _malloc_trim_fn is None:
        return False
    _malloc_trim_fn(0)
    return True


async def _trim_after_parse() -> None:
    """Run ``trim_heap`` off the event loop (a trim walks every arena and can
    take a noticeable time on a multi-GB heap). Best-effort: never fails a parse."""
    try:
        await asyncio.to_thread(trim_heap)
    except Exception:  # noqa: BLE001 - memory hygiene must never break an extract
        log.debug("malloc_trim after MinerU parse failed", exc_info=True)


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


@dataclass
class _TaskState:
    status: str | None = None
    error: str | None = None

    def update(self, payload: dict) -> None:
        status = payload.get("status")
        if status not in ("pending", "processing", "completed", "failed"):
            raise MineruTransportError(f"invalid_task_status: {status!r}")
        self.status = status
        self.error = payload.get("error")


def _remaining_time(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("wall_clock_cap")
    return remaining


async def _request_json(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> dict:
    response = await client.request(method, url, **kwargs)
    code = response.status_code
    if not response.is_success:
        detail = f"HTTP {code}: {response.text}"
        if code == 400 and "failed to load image" in response.text.lower():
            raise MineruImageDecodeError(detail)
        if code in (400, 409, 422):
            raise MineruExtractionError(detail)
        # Gateway/overload responses are service failures even if their body
        # contains JSON. A bare 500 is ambiguous and remains uncharged.
        if code == 500:
            try:
                verdict = response.json()
            except ValueError:
                verdict = None
            if isinstance(verdict, dict) and verdict:
                raise MineruExtractionError(detail)
        raise MineruTransportError(detail)
    try:
        payload = response.json()
    except ValueError as exc:
        raise MineruTransportError("invalid_api_json") from exc
    if not isinstance(payload, dict):
        raise MineruTransportError("invalid_api_payload")
    return payload


def _markdown_from_result(payload: dict, stem: str) -> str:
    results = payload.get("results")
    entry = None
    if isinstance(results, dict):
        entry = results.get(stem)
        if stem not in results and len(results) == 1:
            entry = next(iter(results.values()))
    text = entry.get("md_content") if isinstance(entry, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise MineruExtractionError("thin_or_missing_md: empty or missing md_content")
    return text


async def _run_task(
    client: httpx.AsyncClient, endpoint: Endpoint, pdf_bytes: bytes, stem: str,
    state: _TaskState, wall_clock_deadline: float | None,
) -> str:
    base = endpoint.url.rstrip("/")
    _remaining_time(wall_clock_deadline)
    payload = await _request_json(
        client, "POST", f"{base}/tasks",
        files={"files": (f"{stem}.pdf", pdf_bytes, "application/pdf")},
        data={"backend": "vlm-engine", "lang_list": "en", "formula_enable": "true",
              "table_enable": "true", "image_analysis": "false", "return_md": "true",
              "return_images": "false"},
    )
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise MineruTransportError("missing_task_id")
    state.update(payload)
    task_url = f"{base}/tasks/{quote(task_id, safe='')}"
    while state.status not in ("completed", "failed"):
        _remaining_time(wall_clock_deadline)
        state.update(await _request_json(client, "GET", task_url))
        if state.status in ("pending", "processing"):
            remaining = _remaining_time(wall_clock_deadline)
            delay = _TASK_POLL_INTERVAL if remaining is None else min(_TASK_POLL_INTERVAL, remaining)
            # Yield even with a zero polling interval, so cancellation and other
            # extractions can run when responses arrive immediately.
            await asyncio.sleep(delay)
    _remaining_time(wall_clock_deadline)
    result = await _request_json(client, "GET", f"{task_url}/result")
    if state.status == "failed":
        raise MineruExtractionError(f"task_failed: {state.error}")
    return _markdown_from_result(result, stem)


async def extract_mineru(
    pdf_bytes: bytes,
    endpoints: list[Endpoint],
    *,
    stem: str = "doc",
    http_timeout: float = _MINERU_HTTP_TIMEOUT,
    wall_clock_deadline: Optional[float] = None,
) -> str:
    """Extract one whole PDF via the shared service's submit/poll/result API.

    ``wall_clock_deadline`` is an absolute ``time.time()`` deadline. A queue
    wait (pending, including queued_ahead > 0) never charges a PDF attempt;
    read/wall-clock timeouts after observing processing do. Connection errors
    and HTTP service outages remain transport failures in every phase.

    Each endpoint attempt owns its HTTP client and closes it on this loop,
    including cancellation. No process-global clients survive loop teardown.
    """
    if not endpoints:
        raise MineruTransportError("no_endpoints_configured")
    try:
        return await _extract_mineru_impl(
            pdf_bytes, endpoints, stem=stem, http_timeout=http_timeout,
            wall_clock_deadline=wall_clock_deadline,
        )
    finally:
        await _trim_after_parse()


async def _extract_mineru_impl(
    pdf_bytes: bytes,
    endpoints: list[Endpoint],
    *,
    stem: str = "doc",
    http_timeout: float = _MINERU_HTTP_TIMEOUT,
    wall_clock_deadline: Optional[float] = None,
) -> str:
    last_transport: Exception | None = None
    for i in range(_INLINE_RETRIES + 1):
        # A new endpoint gets a new task state: processing on a previous,
        # unreachable service cannot turn this endpoint's queue wait into a charge.
        state = _TaskState()
        try:
            _remaining_time(wall_clock_deadline)
            max_concurrency, max_connections = client_limits()
            async with httpx.AsyncClient(
                timeout=http_timeout,
                limits=httpx.Limits(max_connections=max_connections,
                                    max_keepalive_connections=min(max_concurrency, max_connections)),
            ) as client:
                remaining = _remaining_time(wall_clock_deadline)
                operation = _run_task(client, alternate(endpoints, i), pdf_bytes, stem,
                                      state, wall_clock_deadline)
                return await asyncio.wait_for(operation, timeout=remaining)
        except MineruImageDecodeError:
            raise
        except (httpx.ReadTimeout, TimeoutError) as exc:
            if state.status in ("processing", "failed"):
                raise MineruExtractionError(
                    f"task_{state.status}_timeout: {state.error or str(exc)}") from exc
            last_transport = exc
        except (MineruTransportError, httpx.HTTPError, httpx.InvalidURL) as exc:
            if state.status == "failed":
                raise MineruExtractionError(f"task_failed: {state.error}") from exc
            last_transport = exc
        if i < _INLINE_RETRIES:
            await _async_sleep_backoff(i, wall_clock_deadline)
    raise MineruTransportError(f"inline_retries_exhausted: {last_transport!r}") from last_transport


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
