"""Compatibility hooks for the independent MinerU service.

There is no caller-side pre-flight. The service owns backend/model wakeup;
the HTTP client's submit/poll/result errors determine reachability
and preserve extraction's uncharged transport deferral. Endpoint and
HTTP-client settings remain owned by ``mineru_client``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from ..mineru_client import MineruTransportError, endpoints_from_env


class MineruServerUnavailable(MineruTransportError):
    """Service unavailability is a transport error; retained for compatibility."""


class MineruServerController:
    """Compatibility API shared by extraction workers and operator OCR."""

    def __init__(self, *, base_url: Optional[str] = None) -> None:
        if base_url is None:
            eps = endpoints_from_env()
            base_url = eps[0].url if eps else "http://127.0.0.1:30000"
        self._base = base_url.rstrip("/")

    @asynccontextmanager
    async def ocr_session(self):
        """Admit operator OCR; workload reaches the service without pre-flight."""
        await self.ensure_ready()
        yield

    async def ensure_ready(self) -> None:
        """No caller-side pre-flight: backend/model wakeup belongs to the service."""
        return None


_controller: Optional[MineruServerController] = None


def get_server_controller() -> MineruServerController:
    """Process-wide compatibility controller (lazy from env)."""
    global _controller
    if _controller is None:
        _controller = MineruServerController()
    return _controller


def reset_server_controller_for_test() -> None:
    global _controller
    _controller = None
