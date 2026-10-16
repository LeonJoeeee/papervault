"""Consumer-side readiness probes for the independent MinerU service.

Consumers use the gate at ``http://127.0.0.1:30000`` (or the first endpoint
resolved by ``MINERU_URL``). The service owns backend lifecycle, idle sleep,
wake-up and self-healing. Readiness requires ``/health`` 200 AND ``/v1/models``
listing a model. A sleeping backend satisfies both; these probes do not wake
it or count as workload. The gate absorbs wake latency on the next OCR request.

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
    """Readiness failed; extraction defers with no attempt charged."""


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
        """Probe once within the HTTP budget; unavailable is a transport error."""
        try:
            async with asyncio.timeout(_HEALTH_HTTP_TIMEOUT):
                ready = await self._ready()
        except TimeoutError:
            ready = False
        if not ready:
            raise MineruServerUnavailable(
                f"mineru server not ready at {self._base}: "
                "/health must return 200 and /v1/models must list a model")

    async def _ready(self) -> bool:
        """A sleeping backend is ready when health and model-list probes pass."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=_HEALTH_HTTP_TIMEOUT) as client:
                health = await client.get(f"{self._base}/health")
                if health.status_code != 200:
                    return False
                models = await client.get(f"{self._base}/v1/models")
                if models.status_code != 200:
                    return False
                payload = models.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            return isinstance(data, list) and any(
                isinstance(model, dict) and isinstance(model.get("id"), str) and model["id"]
                for model in data
            )
        except Exception:  # noqa: BLE001 — failed HTTP/JSON observations mean unavailable
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
