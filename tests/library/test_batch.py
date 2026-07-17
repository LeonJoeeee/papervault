"""Tests for services/batch.py — concurrent ingestion."""

from __future__ import annotations

import threading
import time

import pytest

from papervault.library import Library
from papervault.library.services.batch import BatchAddService, _RateLimiter


@pytest.fixture
def lib(tmp_path):
    return Library(tmp_path)


class _FakeAdd:
    """Records each call. Optional per-id behavior dict for mixed outcomes."""

    def __init__(self, behavior=None):
        self.calls: list[str] = []
        self._lock = threading.Lock()
        self.behavior = behavior or {}

    def add(self, ident, *, force_refresh=False):
        with self._lock:
            self.calls.append(ident)
        action = self.behavior.get(ident, "ok")
        if action == "raise":
            raise RuntimeError(f"boom-{ident}")
        return {"status": "added" if action == "ok" else action,
                "key": f"key-{ident}", "metadata": None,
                "candidates": None, "message": "ok"}


def test_add_many_returns_one_result_per_input(lib):
    svc = BatchAddService(lib, max_workers=2, min_request_interval=0)
    svc._add_service = _FakeAdd()
    out = svc.add_many(["a", "b", "c"])
    assert len(out) == 3
    assert {r["key"] for r in out} == {"key-a", "key-b", "key-c"}


def test_add_many_skips_blank_identifiers(lib):
    svc = BatchAddService(lib, max_workers=2, min_request_interval=0)
    svc._add_service = _FakeAdd()
    out = svc.add_many(["x", "", None, "y"])
    assert {r["key"] for r in out} == {"key-x", "key-y"}


def test_add_many_failures_do_not_abort(lib):
    fake = _FakeAdd(behavior={"b": "raise"})
    svc = BatchAddService(lib, max_workers=2, min_request_interval=0)
    svc._add_service = fake
    out = svc.add_many(["a", "b", "c"])
    by_status = {r["status"] for r in out}
    assert "added" in by_status
    assert "internal_error" in by_status
    assert sorted(fake.calls) == ["a", "b", "c"]


def test_on_result_callback_fires_per_completion(lib):
    svc = BatchAddService(lib, max_workers=2, min_request_interval=0)
    svc._add_service = _FakeAdd()
    seen: list[tuple[str, str]] = []
    svc.add_many(["a", "b"], on_result=lambda i, r: seen.append((i, r["status"])))
    assert sorted(seen) == [("a", "added"), ("b", "added")]


def test_on_result_exceptions_are_swallowed(lib):
    svc = BatchAddService(lib, max_workers=1, min_request_interval=0)
    svc._add_service = _FakeAdd()

    def bad_callback(_i, _r):
        raise ValueError("user code is buggy")

    # Should still complete without raising.
    out = svc.add_many(["a", "b"], on_result=bad_callback)
    assert len(out) == 2


def test_empty_input_returns_empty(lib):
    svc = BatchAddService(lib)
    svc._add_service = _FakeAdd()
    assert svc.add_many([]) == []


# ----------- _RateLimiter ---------------------------------------------------


def test_rate_limiter_paces_calls():
    rl = _RateLimiter(min_interval=0.1)
    t0 = time.monotonic()
    rl.acquire()
    rl.acquire()
    rl.acquire()
    # Three back-to-back calls; the second and third must each wait ~0.1s.
    assert time.monotonic() - t0 >= 0.18  # allow slack for flakiness


def test_rate_limiter_zero_interval_does_not_sleep():
    rl = _RateLimiter(min_interval=0)
    t0 = time.monotonic()
    for _ in range(50):
        rl.acquire()
    assert time.monotonic() - t0 < 0.1


def test_rate_limiter_thread_safety():
    rl = _RateLimiter(min_interval=0.02)
    results: list[float] = []

    def worker():
        rl.acquire()
        results.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # All five spaced ≥ ~0.02s apart, total ≥ ~0.08s.
    assert time.monotonic() - t0 >= 0.07
