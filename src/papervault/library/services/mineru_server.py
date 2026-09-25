"""On-demand lifecycle for the persistent MinerU vLLM server unit.

The MinerU server (``papervault-mineru.service``; legacy ``paper-library-mineru.service``
still selectable via ``PAPER_LIBRARY_MINERU_UNIT``) holds ~9 GB on the 3090 for
its whole lifetime (vLLM pre-allocates the KV-cache pool and never releases it
while running). In steady state the vault is fully extracted and the extract
queue is idle almost always, so a 24/7-resident server wastes the card. This
controller makes the server **on-demand**: started on the worker critical path
right before an extract, and stopped after the queue has been idle for
``IDLE_TIMEOUT``, freeing the GPU (e.g. for a future KS embedding co-tenant).

**OPT-IN** via ``PAPER_LIBRARY_MINERU_ONDEMAND=1``. Default OFF ⇒ every method is
a strict no-op and the persistent always-on deploy is byte-unchanged.

Key review-driven invariants baked in here:
  * ``ensure_ready`` raises :class:`MineruServerUnavailable` — a subclass of
    ``MineruTransportError`` — so ``extract_md``'s existing C1 handler catches it,
    charges NO attempt, and lets reconcile retry (review fix #1).
  * readiness = ``/health`` 200 **AND** ``/v1/models`` lists a loaded model — a
    bare ``/health`` 200 can precede the engine being parse-ready (review fix #3).
  * ``READY_TIMEOUT`` ≥ the unit's ``TimeoutStartSec`` (300 s) so a legitimately
    slow cold start isn't abandoned mid-flight (review fix #4).
  * idle = ``queue.qsize()==0`` AND no ``ex:``-prefixed key in
    ``concurrency.in_flight_keys()`` (a function, ``ex:``-prefixed — review fix #2).
  * a lockless ``ensure_ready`` fast-path checks ``_stopping`` after its
    readiness await, so a concurrent teardown cannot admit OCR.
  * a failed/timeout start enters an exponential cooldown so OOM (e.g. a GPU
    co-tenant grabbed the VRAM) doesn't burn ``READY_TIMEOUT`` every sweep
    forever (review fix #7).
  * if ``systemctl --user`` has no D-Bus (non-systemd host), on-demand force-OFFs
    to persistent rather than stalling every extract (review fix #8).
  * self-heal (issue #134): N consecutive ``MineruImageDecodeError`` parses (the
    server answers 400 ``Failed to load image`` for the client's own PNGs — a
    stale server process) trigger ONE ``systemctl --user restart`` under the same
    lock / ``_stopping`` / cooldown rules as start and stop. It is not repeated
    until a parse succeeds again. With on-demand OFF the controller never touches
    systemd: it logs one ERROR naming the cause and the unit to restart.
The in-process ``asyncio.Lock`` serializes only this daemon's transitions; it
does not guard a manual ``systemctl`` or a separate OCR process. Keep on-demand
OFF for independent OCR scripts that use the same server.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import time
from typing import Optional

from ..mineru_client import MineruTransportError, endpoints_from_env

log = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "") == "1"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


# Readiness wait ≥ unit TimeoutStartSec=300 + margin (review fix #4).
_READY_TIMEOUT = _env_float("PAPER_LIBRARY_MINERU_READY_TIMEOUT", 330.0)
_CHECK_INTERVAL = _env_float("PAPER_LIBRARY_MINERU_IDLE_CHECK", 60.0)
_IDLE_TIMEOUT = _env_float("PAPER_LIBRARY_MINERU_IDLE_TIMEOUT", 600.0)
_HEALTH_POLL = _env_float("PAPER_LIBRARY_MINERU_HEALTH_POLL", 2.0)
_HEALTH_HTTP_TIMEOUT = 3.0
_COOLDOWN_MIN = 30.0
_COOLDOWN_CAP = 600.0
_START_TIMEOUT = 320.0   # ≥ unit TimeoutStartSec=300
_STOP_TIMEOUT = 70.0     # ≥ unit TimeoutStopSec=60
_EXTRACT_PREFIX = "ex:"  # concurrency.with_dedup stage key for extraction
# Consecutive server-side image-decode failures (no success between them) that
# trigger the self-heal restart (on-demand ON) or the one ERROR (OFF). Issue #134.
_IMAGE_DECODE_RESTART_AFTER = max(1, _env_int(
    "PAPER_LIBRARY_MINERU_IMAGE_DECODE_RESTART_AFTER", 3))


class MineruServerUnavailable(MineruTransportError):
    """``ensure_ready`` could not bring the server to model-ready within budget.

    Subclasses ``MineruTransportError`` so ``extract_md``'s ``except
    MineruTransportError`` arm catches it → no attempt charged, reconcile retries
    (review fix #1). Never reaches the EXTRACTION (charge/terminal) arm.
    """


class MineruServerController:
    """Process-wide controller for the MinerU server unit's on-demand lifecycle."""

    def __init__(
        self,
        *,
        ondemand: Optional[bool] = None,
        unit: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._ondemand = _env_flag("PAPER_LIBRARY_MINERU_ONDEMAND") if ondemand is None else ondemand
        self._unit = unit or os.environ.get(
            "PAPER_LIBRARY_MINERU_UNIT", "papervault-mineru.service")
        if base_url is None:
            eps = endpoints_from_env()
            base_url = eps[0].url if eps else "http://127.0.0.1:30000"
        self._base = base_url.rstrip("/")
        self._lock = asyncio.Lock()
        self._last_activity = time.monotonic()
        self._stopping = False
        self._cooldown = 0.0          # current backoff length (doubles on each failed start)
        self._cooldown_until = 0.0    # monotonic deadline; ensure_ready short-circuits before it
        self._bus_ok: Optional[bool] = None  # None=unprobed
        self._active_ocr = 0
        # Issue #134 self-heal state. The streak counts image-decode failures
        # since the last successful parse; ``_heal_spent`` allows ONE restart per
        # unhealthy stretch; ``_heal_alerted`` keeps the ERROR to one line.
        self._image_decode_streak = 0
        self._heal_spent = False
        self._heal_alerted = False
        self._heal_task: Optional[asyncio.Task] = None

    @property
    def ondemand(self) -> bool:
        return self._ondemand

    # ----------------------------- activity ----------------------------------
    def note_activity(self) -> None:
        """Stamp 'extract activity happened now'. Cheap, sync. Called on enqueue,
        on worker pull, and on ensure_ready entry so the idle timer measures time
        since real activity, not time since enqueue (review fix #5)."""
        self._last_activity = time.monotonic()

    @asynccontextmanager
    async def ocr_session(self):
        """Keep MinerU resident for OCR outside the library extract queue."""
        if not self._ondemand:
            yield
            return
        async with self._lock:
            self._active_ocr += 1
            self.note_activity()
        try:
            await self.ensure_ready()
            yield
        finally:
            self._active_ocr -= 1
            self.note_activity()

    # --------------------- readiness gate (worker path) -----------------------
    async def ensure_ready(self) -> None:
        """Block until the MinerU server is model-ready, starting it if needed.

        No-op when on-demand is OFF. On the worker critical path (called from
        ``extract_md`` INSIDE its C1 try-block). Raises
        :class:`MineruServerUnavailable` (a transport error → no charge) if the
        server can't be made ready within ``READY_TIMEOUT`` or while in
        start-failure cooldown.
        """
        if not self._ondemand:
            return
        self.note_activity()
        # The probe awaits I/O: a stop can begin while it is suspended. Check
        # _stopping again after the await before admitting OCR without the lock.
        ready = not self._stopping and await self._ready()
        if ready and not self._stopping:
            return
        async with self._lock:
            if await self._ready():
                return
            now = time.monotonic()
            if now < self._cooldown_until:
                raise MineruServerUnavailable(
                    f"mineru server in start-failure cooldown "
                    f"({self._cooldown_until - now:.0f}s left)")
            if not await self._bus_available():
                raise MineruServerUnavailable(
                    "systemctl --user unavailable (no D-Bus); cannot start mineru server")
            started = time.perf_counter()
            start_rc = await self._systemctl("start")
            if start_rc != 0:
                self._arm_cooldown()
                raise MineruServerUnavailable(
                    f"mineru systemctl start failed (rc={start_rc}; "
                    f"start-failure cooldown {self._cooldown:.0f}s)")
            deadline = time.monotonic() + _READY_TIMEOUT
            while time.monotonic() < deadline:
                if await self._ready():
                    self._cooldown = 0.0          # success resets the backoff
                    self._cooldown_until = 0.0
                    log.info("mineru model-ready after %.3fs", time.perf_counter() - started)
                    return
                await asyncio.sleep(_HEALTH_POLL)
            # Timed out → exponential cooldown, then transport (no charge → reconcile retries).
            self._arm_cooldown()
            raise MineruServerUnavailable(
                f"mineru server not model-ready within {_READY_TIMEOUT:.0f}s "
                f"(start-failure cooldown {self._cooldown:.0f}s)")

    # ------------------ self-heal: stale server (issue #134) -------------------
    def note_parse_success(self) -> None:
        """A MinerU parse returned output: the server decodes images again. Clears
        the streak and re-arms the one-restart budget and the one-line alert."""
        self._image_decode_streak = 0
        self._heal_spent = False
        self._heal_alerted = False

    def note_image_decode_failure(self) -> Optional[asyncio.Task]:
        """Count one ``MineruImageDecodeError`` parse (a 400 ``Failed to load
        image`` the server returned for the client's own PNG).

        After ``_IMAGE_DECODE_RESTART_AFTER`` consecutive ones: with on-demand ON,
        schedule ONE background restart of the unit (returned, so a caller or test
        may await it; the extract worker does not wait on it); with on-demand OFF,
        or when the restart was already spent this stretch, log one ERROR. Must be
        called from a running event loop (``extract_md`` is async)."""
        self._image_decode_streak += 1
        if self._image_decode_streak < _IMAGE_DECODE_RESTART_AFTER:
            return None
        if self._heal_task is not None and not self._heal_task.done():
            return self._heal_task
        if not self._ondemand:
            self._alert_stale_server("on-demand is OFF, so papervault will not restart it")
            return None
        if self._heal_spent:
            self._alert_stale_server("the one self-heal restart did not help")
            return None
        self._heal_spent = True
        self._heal_task = asyncio.get_running_loop().create_task(
            self._restart_for_image_decode())
        return self._heal_task

    def _alert_stale_server(self, why: str) -> None:
        if self._heal_alerted:
            return
        self._heal_alerted = True
        log.error(
            "MinerU server %s answered %d consecutive parses with HTTP 400 "
            "'Failed to load image' for PNGs the client rendered itself. The server "
            "process is likely stale (e.g. still running from a moved or deleted "
            "venv whose Pillow image plugins can no longer load). Papers are parked "
            "without an attempt charged; %s. Restart the unit: "
            "systemctl --user restart %s",
            self._base, self._image_decode_streak, why, self._unit)

    async def _restart_for_image_decode(self) -> None:
        """Restart the unit once and wait for model-readiness, under the lock.

        Same rules as ``ensure_ready``'s start: skipped during the start-failure
        cooldown or without a user bus; a failed restart or a ready timeout arms
        the cooldown. ``_stopping`` is held across the ``systemctl restart`` call
        so the lockless ``ensure_ready`` fast path cannot admit OCR against a
        server that is going down; later callers wait on the lock until ready."""
        async with self._lock:
            now = time.monotonic()
            if now < self._cooldown_until:
                # Not spent: a later failure may retry once the cooldown lapses.
                self._heal_spent = False
                log.warning(
                    "mineru self-heal: restart of %s skipped — start-failure "
                    "cooldown (%.0fs left)", self._unit, self._cooldown_until - now)
                return
            if not await self._bus_available():
                self._alert_stale_server("systemctl --user is unavailable (no D-Bus)")
                return
            log.warning(
                "mineru self-heal: %d consecutive 'Failed to load image' parses — "
                "restarting %s (likely a stale server process)",
                self._image_decode_streak, self._unit)
            started = time.perf_counter()
            self._stopping = True
            try:
                rc = await self._systemctl("restart")
            finally:
                self._stopping = False
            self._image_decode_streak = 0
            if rc != 0:
                self._arm_cooldown()
                log.error("mineru self-heal: systemctl restart %s failed (rc=%s; "
                          "cooldown %.0fs)", self._unit, rc, self._cooldown)
                return
            deadline = time.monotonic() + _READY_TIMEOUT
            while time.monotonic() < deadline:
                if await self._ready():
                    self._cooldown = 0.0
                    self._cooldown_until = 0.0
                    log.info("mineru self-heal: %s model-ready after %.3fs",
                             self._unit, time.perf_counter() - started)
                    return
                await asyncio.sleep(_HEALTH_POLL)
            self._arm_cooldown()
            log.error("mineru self-heal: %s not model-ready within %.0fs after "
                      "restart (cooldown %.0fs)", self._unit, _READY_TIMEOUT,
                      self._cooldown)

    def _arm_cooldown(self) -> None:
        self._cooldown = min(max(self._cooldown * 2.0, _COOLDOWN_MIN), _COOLDOWN_CAP)
        self._cooldown_until = time.monotonic() + self._cooldown

    # ------------------------ idle monitor (background) -----------------------
    async def monitor_loop(self, extract_queue) -> None:
        """Stop the server after a sustained idle window. No-op when OFF; if the
        systemd user bus is absent, force-OFF (degrade to persistent, fix #8)."""
        if not self._ondemand:
            return
        if not await self._bus_available():
            log.warning(
                "mineru on-demand: systemctl --user unavailable (no D-Bus) — "
                "forcing on-demand OFF; the MinerU server stays persistent")
            self._ondemand = False
            return
        log.info("mineru on-demand monitor started (idle_timeout=%.0fs, check=%.0fs)",
                 _IDLE_TIMEOUT, _CHECK_INTERVAL)
        while True:
            try:
                await asyncio.sleep(_CHECK_INTERVAL)
                if not self._past_idle(extract_queue):
                    continue
                async with self._lock:
                    # Re-check everything UNDER the lock (the fast outer check may
                    # be stale by the time we acquired it).
                    if self._past_idle(extract_queue) and await self._ready():
                        # _ready awaited I/O, during which a worker may have
                        # entered. No await separates this final check from
                        # setting _stopping, while the controller lock is held.
                        if self._past_idle(extract_queue):
                            self._stopping = True
                            try:
                                await self._systemctl("stop")
                            finally:
                                self._stopping = False
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a monitor hiccup must never kill the loop
                log.exception("mineru on-demand monitor iteration failed")

    def _past_idle(self, extract_queue) -> bool:
        return (
            self._idle(extract_queue)
            and (time.monotonic() - self._last_activity) > _IDLE_TIMEOUT
        )

    def _idle(self, extract_queue) -> bool:
        """True when no library or operator OCR is queued or in flight."""
        from . import concurrency
        if extract_queue.qsize() != 0 or self._active_ocr:
            return False
        return not any(k.startswith(_EXTRACT_PREFIX) for k in concurrency.in_flight_keys())

    # ------------------------------- probes -----------------------------------
    async def _ready(self) -> bool:
        """Model-ready iff ``/health`` 200 AND ``/v1/models`` lists ≥1 model
        (review fix #3: a bare ``/health`` 200 can precede engine readiness)."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=_HEALTH_HTTP_TIMEOUT) as client:
                h = await client.get(f"{self._base}/health")
                if h.status_code != 200:
                    return False
                m = await client.get(f"{self._base}/v1/models")
                if m.status_code != 200:
                    return False
                data = m.json()
            return bool((data or {}).get("data"))
        except Exception:  # noqa: BLE001 — any error ⇒ not ready
            return False

    async def _bus_available(self) -> bool:
        """One-shot probe (cached): can we talk to the systemd --user bus?"""
        if self._bus_ok is not None:
            return self._bus_ok
        rc, err = await self._run_systemctl("is-active")
        # rc is None only on FileNotFoundError/timeout; a bus failure prints
        # 'Failed to connect to bus' on stderr regardless of rc.
        self._bus_ok = rc is not None and "failed to connect to bus" not in (err or "").lower()
        return self._bus_ok

    async def _systemctl(self, verb: str) -> Optional[int]:
        rc, _ = await self._run_systemctl(verb)
        return rc

    async def _run_systemctl(self, verb: str) -> tuple[Optional[int], str]:
        """Run ``systemctl --user <verb> <unit>`` off the event loop with a
        timeout. Returns (rc, stderr); rc None ⇒ systemctl missing or timed out."""
        timeout = {"start": _START_TIMEOUT,
                   "restart": _STOP_TIMEOUT + _START_TIMEOUT}.get(verb, _STOP_TIMEOUT)
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "--user", verb, self._unit,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except FileNotFoundError:
            return None, "systemctl not found"
        try:
            _out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            log.error("systemctl --user %s %s timed out after %.0fs", verb, self._unit, timeout)
            return None, "timeout"
        err_s = (err or b"").decode("utf-8", "replace")
        if verb != "is-active":
            log.info("systemctl --user %s %s → rc=%s", verb, self._unit, proc.returncode)
        return proc.returncode, err_s


_controller: Optional[MineruServerController] = None


def get_server_controller() -> MineruServerController:
    """Process-wide singleton (lazy from env)."""
    global _controller
    if _controller is None:
        _controller = MineruServerController()
    return _controller


def reset_server_controller_for_test() -> None:
    global _controller
    _controller = None
