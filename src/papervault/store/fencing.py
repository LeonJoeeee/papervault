"""Cooperative lifetime fencing for service and CLI Library instances (Linux)."""
from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path


class WriterFenced(RuntimeError):
    pass


def handoff_state(root: Path) -> dict:
    path = root / ".catalog-handoff.json"
    return json.loads(path.read_bytes()) if path.exists() else {}


def writer_lease(root: Path, authority: str | None, *, handoff_token: str | None = None):
    """Fence all writers; authority-independent exports pass authority=None."""
    root.mkdir(parents=True, exist_ok=True)
    if handoff_token:
        state = handoff_state(root)
        if state.get("fence_id") != handoff_token or state.get("writes_resumed"):
            raise WriterFenced("invalid fenced-probe token")
        return None
    lease = (root / ".catalog-writers.lock").open("a+")
    try:
        try:
            fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WriterFenced("catalog handoff currently holds the writer fence") from None
        state = handoff_state(root)
        if state and (state.get("phase") not in {"resumed", "rolled_back"}
                      or (authority is not None
                          and authority != ("postgres" if state["phase"] == "resumed" else "json"))):
            raise WriterFenced("catalog handoff fences writers or authority disagrees with durable phase")
        return lease
    except BaseException:
        lease.close()
        raise


@contextmanager
def exclusive_fence(root: Path):
    lease = (root / ".catalog-writers.lock").open("a+")
    try:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WriterFenced("service/CLI writers active; stop and drain all writers first") from None
        yield
    finally:
        lease.close()
