"""Contract tests for A2/D-D; clocks and completions never use the network."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from papervault.library.politeness_v2 import (
    AdmissionDenied, Envelope, Ledger, Policy, Scope, Window,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


SCOPE = Scope("wiley")


def sent(ledger, scope=SCOPE, status=200, *, timed_out=False, completed=True):
    ticket = ledger.note_sent(scope)
    ledger.note_result(ticket, status_code=status, timed_out=timed_out, completed=completed)


def envelope(**kwargs):
    return Envelope(analogue=(Window(100, 60),), initial=(Window(40, 60),),
                    hard_cap=(Window(60, 60),), authorization="issue:221 test fixture", **kwargs)


def test_strict_long_window_rejects_61st_despite_refilling_token_bucket():
    clock = Clock()
    ledger = Ledger({SCOPE: Policy((Window(3, 1), Window(60, 600)))}, clock=clock)
    # Even a slow-window token bucket starting with 60 tokens would pass:
    # after 60 sends at .35s, 60 - 60 + 21*(60/600) = 2.1 tokens.
    for index in range(60):
        clock.now = index * 0.35
        sent(ledger)
    clock.now = 21.0
    decision = ledger.gate(SCOPE)
    assert not decision.allowed
    assert decision.reason == "window"
    assert decision.retry_after == pytest.approx(579.0)
    with pytest.raises(AdmissionDenied):
        ledger.note_sent(SCOPE)
    assert ledger.snapshot(SCOPE).sent == 60
    clock.now = 600
    sent(ledger)
    assert ledger.snapshot(SCOPE).sent == 61


def test_all_windows_bind_in_parallel_and_expire_at_exact_boundary():
    clock = Clock()
    ledger = Ledger({SCOPE: Policy((Window(2, 1), Window(3, 10)))}, clock=clock)
    sent(ledger)
    sent(ledger)
    assert ledger.gate(SCOPE).retry_after == 1
    clock.now = 1
    sent(ledger)
    assert ledger.gate(SCOPE).retry_after == 9
    clock.now = 9.999
    assert not ledger.gate(SCOPE).allowed
    clock.now = 10
    assert ledger.gate(SCOPE).allowed


def test_tier_names_do_not_split_a_shared_upstream_scope():
    clock = Clock()
    ledger = Ledger({SCOPE: Policy((Window(1, 10),))}, clock=clock)
    # Different call names share the same service key, including reconstructed scopes.
    def tdm_tier():
        sent(ledger, Scope("wiley"))

    def resolver_tier():
        return ledger.note_sent(Scope("wiley"))

    tdm_tier()
    with pytest.raises(AdmissionDenied):
        resolver_tier()
    assert ledger.gate(Scope("other")).reason == "disabled"


def test_concurrency_is_separate_from_request_windows():
    clock = Clock()
    ledger = Ledger({SCOPE: Policy((Window(1, 1),))}, clock=clock)
    ticket = ledger.note_sent(SCOPE)
    clock.now = 2
    assert ledger.gate(SCOPE).reason == "concurrency"
    ledger.note_result(ticket, status_code=500)
    assert ledger.gate(SCOPE).allowed
    sent(ledger)
    assert ledger.gate(SCOPE).reason == "window"


def test_atomic_note_sent_does_not_admit_competing_threads():
    ledger = Ledger({SCOPE: Policy((Window(1, 60),), concurrency=10)}, clock=Clock())
    def attempt(_):
        try:
            return ledger.note_sent(SCOPE)
        except AdmissionDenied:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        tickets = list(pool.map(attempt, range(40)))
    assert sum(ticket is not None for ticket in tickets) == 1
    assert ledger.snapshot(SCOPE).in_flight == 1


@pytest.mark.parametrize("configuration", [None, Envelope(), Envelope(
    analogue=(Window(100, 60),), initial=(Window(40, 60),), hard_cap=(Window(60, 60),))])
def test_missing_limit_or_authorization_disables_scope(configuration):
    ledger = Ledger({SCOPE: configuration}, clock=Clock())
    assert ledger.gate(SCOPE).reason == "disabled"
    with pytest.raises(AdmissionDenied):
        ledger.note_sent(SCOPE)


def test_envelope_requires_fresh_positive_completion_and_24_hours():
    clock = Clock()
    ledger = Ledger({SCOPE: envelope()}, clock=clock)
    clock.now = 86400
    assert not ledger.increase(SCOPE, (Window(50, 60),))
    sent(ledger, status=500)
    sent(ledger, completed=False)
    assert not ledger.increase(SCOPE, (Window(50, 60),))
    sent(ledger)
    assert ledger.increase(SCOPE, (Window(50, 60),))
    assert ledger.snapshot(SCOPE).evidence == 0
    assert ledger.snapshot(SCOPE).events[-1].kind == "increase"
    sent(ledger)
    assert not ledger.increase(SCOPE, (Window(60, 60),))
    clock.now += 86400
    assert ledger.increase(SCOPE, (Window(60, 60),))
    assert ledger.snapshot(SCOPE).ramp_stopped
    clock.now += 86400
    sent(ledger)
    assert not ledger.increase(SCOPE, (Window(61, 60),))


def test_envelope_rejects_more_than_25_percent_and_does_not_choose_numbers():
    clock = Clock()
    ledger = Ledger({SCOPE: envelope()}, clock=clock)
    sent(ledger)
    clock.now = 86400
    assert not ledger.increase(SCOPE, (Window(51, 60),))
    assert not ledger.increase(SCOPE, (Window(50, 59),))
    assert ledger.snapshot(SCOPE).windows == (Window(40, 60),)
    assert ledger.increase(SCOPE, (Window(50, 60),))


def test_429_cooldown_resumes_in_place_and_permanently_stops_ramp():
    clock = Clock()
    ledger = Ledger({SCOPE: envelope(quiet_seconds=60)}, clock=clock)
    sent(ledger)
    sent(ledger, status=429)
    assert ledger.gate(SCOPE).reason == "cooldown"
    assert ledger.snapshot(SCOPE).evidence == 0
    clock.now = 60
    assert ledger.gate(SCOPE).allowed
    assert ledger.snapshot(SCOPE).events[-1].kind == "resume"
    sent(ledger)
    clock.now = 86400
    assert not ledger.increase(SCOPE, (Window(50, 60),))
    assert ledger.snapshot(SCOPE).windows == (Window(40, 60),)


def test_timeout_cluster_propagates_cooldown_and_late_failure_extends_quiet_window():
    clock = Clock()
    ledger = Ledger({SCOPE: Policy((Window(20, 1),), concurrency=2,
                                  quiet_seconds=60, timeout_cluster=3,
                                  timeout_window=10)}, clock=clock)
    sent(ledger, status=None, timed_out=True)
    clock.now = 11
    sent(ledger, status=None, timed_out=True)  # first timeout has expired
    assert ledger.gate(SCOPE).allowed
    clock.now = 12
    late = ledger.note_sent(SCOPE)
    sent(ledger, status=None, timed_out=True)
    clock.now = 13
    sent(ledger, status=None, timed_out=True)
    assert ledger.gate(SCOPE).reason == "cooldown"
    clock.now = 20
    ledger.note_result(late, timed_out=True)
    clock.now = 79
    assert ledger.gate(SCOPE).reason == "cooldown"
    clock.now = 80
    assert ledger.gate(SCOPE).allowed


def test_duplicate_or_foreign_completion_cannot_release_another_permit():
    ledger = Ledger({SCOPE: Policy((Window(20, 1),), concurrency=2)}, clock=Clock())
    other = Ledger({SCOPE: Policy((Window(20, 1),), concurrency=2)}, clock=Clock())
    first = ledger.note_sent(SCOPE)
    second = ledger.note_sent(SCOPE)
    foreign = other.note_sent(SCOPE)
    with pytest.raises(ValueError):
        ledger.note_result(foreign, status_code=200)
    assert ledger.snapshot(SCOPE).in_flight == 2
    ledger.note_result(first, status_code=200)
    with pytest.raises(ValueError):
        ledger.note_result(first, status_code=200)
    assert ledger.snapshot(SCOPE).in_flight == 1
    ledger.note_result(second, status_code=200)


def test_single_request_analogue_can_start_at_half_rate_by_lengthening_window():
    clock = Clock()
    config = Envelope(analogue=(Window(1, 1),), initial=(Window(1, 2),),
                      hard_cap=(Window(1, 2),), authorization="explicit")
    ledger = Ledger({SCOPE: config}, clock=clock)
    sent(ledger)
    clock.now = 1
    assert not ledger.gate(SCOPE).allowed
    clock.now = 2
    assert ledger.gate(SCOPE).allowed


def test_single_request_envelope_ramps_explicit_period_at_most_25_percent():
    clock = Clock()
    config = Envelope(analogue=(Window(1, 1),), initial=(Window(1, 2),),
                      hard_cap=(Window(1, 1),), authorization="explicit")
    ledger = Ledger({SCOPE: config}, clock=clock)
    sent(ledger)
    clock.now = 86400
    assert not ledger.increase(SCOPE, (Window(1, 1.5),))  # +33.3%
    assert ledger.increase(SCOPE, (Window(1, 1.6),))  # exactly +25%
    sent(ledger)
    assert ledger.gate(SCOPE).retry_after == pytest.approx(1.6)
    clock.now += 86400
    assert ledger.increase(SCOPE, (Window(1, 1.28),))
    sent(ledger)
    clock.now += 86400
    assert ledger.increase(SCOPE, (Window(1, 1.024),))
    sent(ledger)
    clock.now += 86400
    assert not ledger.increase(SCOPE, (Window(1, 0.99),))  # explicit hard cap
    assert ledger.increase(SCOPE, (Window(1, 1),))
    assert ledger.snapshot(SCOPE).ramp_stopped


def test_response_headers_cool_shared_scope_without_releasing_send_ticket():
    ledger = Ledger({SCOPE: Policy((Window(10, 1),), concurrency=2)}, clock=Clock())
    ticket = ledger.note_sent(SCOPE)
    ledger.note_response(ticket, status_code=429)
    assert ledger.gate(SCOPE).reason == "cooldown"
    assert ledger.snapshot(SCOPE).in_flight == 1
    ledger.note_result(ticket, status_code=429, completed=False, timed_out=True)
    assert ledger.snapshot(SCOPE).in_flight == 0


@pytest.mark.parametrize("scope", [Scope("wiley", credential="other"),
                                    Scope("wiley", ip="other"), Scope("wiley", endpoint="other")])
def test_scope_dimensions_are_explicit_and_unconfigured_scopes_do_not_inherit_limits(scope):
    ledger = Ledger({SCOPE: Policy((Window(1, 1),))}, clock=Clock())
    assert ledger.gate(scope).reason == "disabled"


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_window_cannot_create_an_unbounded_policy(bad):
    with pytest.raises(ValueError):
        Window(1, bad)


def test_invalid_envelope_rejects_half_analogue_violation_and_missing_parallel_window():
    with pytest.raises(ValueError):
        Envelope(analogue=(Window(100, 60),), initial=(Window(51, 60),),
                 hard_cap=(Window(60, 60),), authorization="explicit")
    with pytest.raises(ValueError):
        Envelope(analogue=(Window(3, 1), Window(60, 600)), initial=(Window(1, 1),),
                 hard_cap=(Window(2, 1),), authorization="explicit")


def test_clock_reversal_and_nonfinite_time_fail_closed():
    clock = Clock()
    ledger = Ledger({SCOPE: Policy((Window(1, 1),))}, clock=clock)
    clock.now = 10
    sent(ledger)
    for bad in [9, float("nan"), float("inf")]:
        clock.now = bad
        with pytest.raises(ValueError):
            ledger.note_sent(SCOPE)


def test_explicit_cooldown_keeps_tickets_windows_and_stops_resume_ramp():
    from papervault.library.politeness_v2 import Envelope, Ledger, Scope, Window
    now = [10.0]
    scope = Scope("fixture")
    envelope = Envelope((Window(10, 1),), (Window(2, 1),), (Window(5, 1),),
                        "fixture authority", quiet_seconds=30)
    ledger = Ledger({scope: envelope}, clock=lambda: now[0])
    ticket = ledger.note_sent(scope)
    ledger.enter_cooldown(scope, "quota_window_violation")
    snapshot = ledger.snapshot(scope)
    assert snapshot.sent == 1 and snapshot.in_flight == 1
    assert snapshot.cooldown_until == 40.0 and snapshot.windows == (Window(2, 1),)
    assert snapshot.events[-1].kind == "quota_window_violation"
    assert ledger.gate(scope).reason == "cooldown"
    ledger.note_result(ticket, status_code=200)
    now[0] = 40.0
    assert ledger.gate(scope).allowed
    assert ledger.snapshot(scope).events[-1].kind == "resume"
    now[0] += 86400
    assert not ledger.increase(scope, (Window(3, 1),))
