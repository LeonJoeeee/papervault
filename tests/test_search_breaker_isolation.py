"""Breaker state must be isolated outside tests/library/test_search.py too."""

from papervault.library import search


def test_open_circuit_outside_the_search_test_module(monkeypatch):
    """Leave an open circuit for the shared fixture to clean after this test."""
    monkeypatch.setattr(search, "BREAKER_TRIPS", 3)
    monkeypatch.setattr(search, "BREAKER_COOLDOWN_S", 3600.0)
    monkeypatch.setattr(search, "_breaker_now", lambda: 1000.0)

    for _ in range(3):
        search._breaker_record_degraded("semantic_scholar")

    assert search._breaker_open_backends() == ["semantic_scholar"]


def test_previous_test_leaves_no_breaker_state():
    assert search._BREAKER_OPEN_UNTIL == {}
    assert search._BREAKER_FAILS == {}
