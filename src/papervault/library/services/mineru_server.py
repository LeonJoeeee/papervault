"""Consumer-side readiness probes for the independent MinerU service.

Consumers use the gate at ``http://127.0.0.1:30000`` (or the first endpoint
resolved by ``MINERU_URL``). The gate owns backend stop/start, idle release and
self-healing. Readiness requires only gate ``/health`` 200. A stopped backend
is idle and ready to accept workload through the gate, even with no model
served yet. The gate starts the backend on the next workload POST and waits
until it can serve the request; the health probe does not start the backend.

``ensure_ready`` performs one HTTP probe within a fixed three-second budget.
Failure raises ``MineruServerUnavailable``, a ``MineruTransportError``, so
extraction defers without charging an attempt. There is no warmup wait or
caller-side transition state. ``ocr_session`` applies the same readiness check
to operator OCR.

Environment:
  | Variable | Default | Meaning |
  | --- | --- | --- |
  | ``MINERU_URL`` | ``http://127.0.0.1:30000`` | Comma-separated consumer endpoints; readiness probes the first. |

Endpoint and HTTP-client settings remain owned by ``mineru_client``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from ..mineru_client import MineruTransportError, endpoints_from_env

_HEALTH_HTTP_TIMEOUT = 3.0


class MineruServerUnavailable(MineruTransportError):
    """Gate health failed; extraction defers with no attempt charged."""


class MineruServerController:
    """Readiness checks shared by extraction workers and operator OCR."""

    def __init__(self, *, base_url: Optional[str] = None) -> None:
        if base_url is None:
            eps = endpoints_from_env()
            base_url = eps[0].url if eps else "http://127.0.0.1:30000"
        self._base = base_url.rstrip("/")

    @asynccontextmanager
    async def ocr_session(self):
        """Confirm readiness before operator OCR."""
        await self.ensure_ready()
        yield

    async def ensure_ready(self) -> None:
        """Probe gate health within the HTTP budget; stopped is idle and ready."""
        try:
            async with asyncio.timeout(_HEALTH_HTTP_TIMEOUT):
                ready = await self._ready()
        except TimeoutError:
            ready = False
        if not ready:
            raise MineruServerUnavailable(
                f"mineru gate unavailable at {self._base}: /health must return 200")

    async def _ready(self) -> bool:
        """Gate health 200 admits workload regardless of backend state."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=_HEALTH_HTTP_TIMEOUT) as client:
                health = await client.get(f"{self._base}/health")
            return health.status_code == 200
        except Exception:  # noqa: BLE001 — failed HTTP observations mean unavailable
            return False


_controller: Optional[MineruServerController] = None


def get_server_controller() -> MineruServerController:
    """Process-wide readiness probe (lazy from env)."""
    global _controller
    if _controller is None:
        _controller = MineruServerController()
    return _controller


def reset_server_controller_for_test() -> None:
    global _controller
    _controller = None
