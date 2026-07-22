"""Tests for the reserved canonical slots lever (issue #37).

The lever RESERVES N of the served slots for the highest raw-citation, intent-relevant
papers in ``search_papers`` (server.py Stage 4→5), layered AFTER the relevance / #46
authority order and BEFORE the top-N cut. DEFAULT OFF and byte-identical when off. These
exercise the pure ``_reserve_canon_slots`` / ``_parse_canon_reserved_slots`` helpers with
an injected N (env-independent) so the promotion math is deterministic.

Companion switch ``SEARCH_NO_INGEST`` (frozen-corpus eval) is also asserted DEFAULT OFF.
"""

from __future__ import annotations

from papervault.library.mcp.server import (
    CANON_RESERVED_SLOTS,
    SEARCH_NO_INGEST,
    _parse_canon_reserved_slots,
    _reserve_canon_slots,
)


def _cand(key, *, citation_count):
    return {"key": key, "citation_count": citation_count}


def _keys(scored):
    return [c["key"] for _s, c in scored]


# ----------- defaults ship OFF ----------------------------------------------


def test_defaults_are_off():
    """Both the lever and the frozen-eval switch ship DEFAULT OFF."""
    assert CANON_RESERVED_SLOTS == 0
    assert SEARCH_NO_INGEST is False


def test_off_is_byte_identical():
    """n=0 → the SAME list object is returned, order untouched."""
    scored = [(0.9, _cand("A", citation_count=1)),
              (0.8, _cand("B", citation_count=9999))]
    out = _reserve_canon_slots(scored, limit=1, n=0)
    assert out is scored


def test_pool_fits_slate_is_noop():
    """Pool already fits the slate (len <= limit) → nothing is buried, same list."""
    scored = [(0.9, _cand("A", citation_count=1)),
              (0.8, _cand("B", citation_count=9999))]
    out = _reserve_canon_slots(scored, limit=5, n=2)
    assert out is scored


def test_reserved_already_served_is_noop():
    """The most-cited paper is already inside the served head → no promotion needed,
    same list returned (no wasted slot)."""
    scored = [(0.9, _cand("A", citation_count=9999)),   # most-cited AND top relevance
              (0.8, _cand("B", citation_count=1)),
              (0.7, _cand("C", citation_count=2)),
              (0.6, _cand("D", citation_count=3))]
    out = _reserve_canon_slots(scored, limit=2, n=1)
    assert out is scored


# ----------- the core behavior: promote a buried classic --------------------


def test_buried_classic_is_promoted_into_the_slate():
    """A high-citation classic ranked BELOW the served cut is pulled into the served
    top-N, displacing exactly the lowest-relevance served paper."""
    scored = [
        (0.90, _cand("A", citation_count=10)),
        (0.80, _cand("B", citation_count=20)),
        (0.70, _cand("C", citation_count=5)),     # would be served at the cut
        (0.60, _cand("D", citation_count=9999)),  # BURIED classic (below cut)
        (0.50, _cand("E", citation_count=1)),
    ]
    out = _reserve_canon_slots(scored, limit=3, n=1)
    served = _keys(out[:3])
    assert "D" in served                       # the classic reached the slate
    assert served == ["A", "B", "D"]           # C (lowest-relevance served) displaced
    assert _keys(out) == ["A", "B", "D", "C", "E"]  # C/E keep their relative order below


def test_one_swap_per_buried_classic_already_served_costs_nothing():
    """With N=2 where one reserved paper is already served and one is buried, only the
    buried classic displaces a served paper — the already-served one costs no slot."""
    scored = [
        (0.90, _cand("A", citation_count=5)),
        (0.80, _cand("B", citation_count=9999)),  # most-cited, ALREADY served
        (0.70, _cand("C", citation_count=3)),     # served at the cut
        (0.60, _cand("D", citation_count=8888)),  # 2nd most-cited, BURIED
        (0.50, _cand("E", citation_count=1)),
    ]
    out = _reserve_canon_slots(scored, limit=3, n=2)
    served = _keys(out[:3])
    assert set(served) == {"A", "B", "D"}      # only C displaced (one buried classic)
    assert "C" not in served


def test_off_topic_giant_cannot_enter():
    """Pool-only: a paper NOT in ``scored`` (dropped by the relevance gate) can never be
    reserved — the lever only ever reorders the gate-passing pool."""
    scored = [
        (0.90, _cand("A", citation_count=10)),
        (0.80, _cand("B", citation_count=20)),
        (0.70, _cand("C", citation_count=5)),
    ]
    out = _reserve_canon_slots(scored, limit=2, n=1)
    assert set(_keys(out)) == {"A", "B", "C"}  # no key materialized from thin air


def test_n_clamped_to_limit():
    """n >= limit reserves the whole slate for the most-cited (clamped, no overflow)."""
    scored = [
        (0.90, _cand("A", citation_count=1)),
        (0.80, _cand("B", citation_count=2)),
        (0.70, _cand("C", citation_count=9999)),  # most-cited, buried
        (0.60, _cand("D", citation_count=8888)),  # 2nd, buried
    ]
    out = _reserve_canon_slots(scored, limit=2, n=99)
    served = set(_keys(out[:2]))
    assert served == {"C", "D"}                 # top-2 by citation fill the whole slate


# ----------- env parsing is defensive ---------------------------------------


def test_parse_defensive():
    assert _parse_canon_reserved_slots("3") == 3
    assert _parse_canon_reserved_slots("0") == 0
    assert _parse_canon_reserved_slots("-4") == 0      # negative clamps to off
    assert _parse_canon_reserved_slots("banana") == 0  # non-int falls back to off
    assert _parse_canon_reserved_slots("") == 0
