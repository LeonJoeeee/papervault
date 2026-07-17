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
    choose not to charge (§2.3). The PAPER is fine; the SERVER (or net) is the
    problem. The caller MUST NOT charge an ``extract_attempt`` and MUST NOT
    terminalize — non-mutation lets classify re-route the paper next sweep.
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
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# --------------------------- typed exceptions (C1) -------------------------


class MineruTransportError(Exception):
    """Server unreachable / connection-level failure / request never produced a
    usable response, OR an AMBIGUOUS bare-500 we choose not to charge (§2.3).

    The PAPER is fine; the SERVER (or net) is the problem. The caller MUST NOT
    charge an attempt, MUST NOT terminalize — non-mutation is the C1 mechanism
    that lets classify re-route the paper to EXTRACT on the next sweep.
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

# OUTER endpoint-failover retry budget (in-MEMORY loop counter, NOT
# extract_attempts). Each iteration may switch endpoints round-robin.
_INLINE_RETRIES = int(
    os.environ.get("PAPER_LIBRARY_MINERU_INLINE_RETRIES", "3"))
_BACKOFF_BASE = float(
    os.environ.get("PAPER_LIBRARY_MINERU_BACKOFF_BASE", "1.0"))
_BACKOFF_CAP = float(
    os.environ.get("PAPER_LIBRARY_MINERU_BACKOFF_CAP", "30.0"))


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

    Sends the WHOLE PDF (no page cap, no chunking) to a MinerU vLLM endpoint
    via ``aio_do_parse(backend="vlm-http-client")``, reads back the md mineru
    writes, and returns it. Implements the §2.3 outer endpoint-failover loop on
    top of the http-client's native per-call retry.

    Raises:
      ``MineruTransportError`` — server unreachable / connect-fail / bare-500 /
        502/503/504/429 / zero-byte timeout / inline budget exhausted. The
        caller does NOT charge an attempt and does NOT terminalize (C1).
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
            try:
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
                    http_timeout=call_timeout,
                    max_retries=_MINERU_INLINE_NATIVE_RETRIES,
                    retry_backoff_factor=_MINERU_NATIVE_BACKOFF_FACTOR,
                    # NOTE: NO connect_timeout — not plumbed through aio_do_parse.
                )
                return _read_md_from_outdir(outdir, stem)

            except ServerError as exc:
                code = _status_of(exc)
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
