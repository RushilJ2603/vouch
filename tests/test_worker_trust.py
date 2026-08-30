"""Section 9.5 -- the circuit breaker, and 3.4 criterion 2's pass condition.

Trust rises slowly and falls fast. Going up needs volume and sustained clean
performance; going down does not, because a confirmed failure in a high-trust
row hits hard -- the whole premise was that this configuration was safe.
"""
import pytest

from vouch import config, exposure, ledger
from vouch.worker import trust
from vouch.worker.trust import BreakerState

CEILING = 720.0            # band 10k-50k, the band Demo 2 runs on
TS = 1_700_000_000.0


def decisions(pattern: str) -> list[dict]:
    """'c' clean, 'w' wrong, oldest first."""
    return [{"outcome": "clean" if ch == "c" else "wrong"} for ch in pattern]


# ── Tripping ───────────────────────────────────────────────────────────────

def test_untripped_row_stays_untripped_below_the_threshold():
    state = trust.evaluate_breaker(BreakerState(), decisions("c" * 97 + "ww"), TS)
    assert not state.tripped


def test_three_confirmed_failures_in_the_window_trips_it():
    """3.4 criterion 2: the breaker returns budget to 0 within 3 confirmed
    failures in a 100-decision window."""
    state = trust.evaluate_breaker(BreakerState(), decisions("c" * 97 + "www"), TS)
    assert state.tripped and state.tripped_at == TS


def test_failures_outside_the_window_do_not_count():
    """The breaker reads the most recent window, not the lifetime. Averages are
    slow, and slow is exactly wrong when something has started going bad."""
    old_failures = decisions("www" + "c" * 150)
    assert not trust.evaluate_breaker(BreakerState(), old_failures, TS).tripped


def test_a_long_clean_history_does_not_dilute_recent_failures():
    """40,000 clean decisions must not buy immunity from three bad ones today."""
    history = decisions("c" * 40_000 + "www")
    assert trust.evaluate_breaker(BreakerState(), history, TS).tripped


# ── Recovery, and why it is asymmetric ─────────────────────────────────────

def test_recovery_requires_clean_decisions_not_elapsed_time():
    tripped = BreakerState(tripped_at=TS)
    later = trust.evaluate_breaker(tripped, decisions("www"), TS + 86_400 * 30)
    assert later.tripped, "a month of wall clock must not clear the breaker"


def test_fifty_clean_decisions_clears_it():
    tripped = BreakerState(tripped_at=TS)
    cleared = trust.evaluate_breaker(tripped, decisions("www" + "c" * 50), TS)
    assert not cleared.tripped


def test_forty_nine_clean_decisions_does_not():
    tripped = BreakerState(tripped_at=TS)
    state = trust.evaluate_breaker(tripped, decisions("www" + "c" * 49), TS)
    assert state.tripped and state.clean_since_trip == 49


def test_a_single_failure_resets_the_clean_run():
    """The bar to re-enter autonomy is deliberately higher than the bar to stay
    in it, or a row on the boundary flaps state on every decision."""
    tripped = BreakerState(tripped_at=TS)
    state = trust.evaluate_breaker(tripped, decisions("www" + "c" * 40 + "w" + "c" * 5), TS)
    assert state.tripped and state.clean_since_trip == 5


def test_recovery_bar_is_higher_than_the_trip_bar():
    assert config.RECOVER_CLEAN > config.TRIP_FAILURES


# ── The breaker overrides the record ───────────────────────────────────────

def test_a_tripped_breaker_forces_budget_to_zero():
    """The record says this configuration was safe. The breaker is the evidence
    that it has stopped being, so the record does not get a vote."""
    earned = trust.recompute_row("10k-50k", CEILING, 400, 400, 400, BreakerState())
    collapsed = trust.recompute_row("10k-50k", CEILING, 400, 400, 400,
                                    BreakerState(tripped_at=TS))
    assert earned["budget"] == pytest.approx(681.95, abs=0.01)
    assert collapsed["budget"] == 0.0
    assert collapsed["state"] == "supervised"


def test_demo_2_pass_condition_end_to_end():
    """3.4 criterion 2, exactly as stated: budget is 0 below 30 clean
    decisions, >= 90% of ceiling by 400, and the breaker returns it to 0
    within 3 confirmed failures in a 100-decision window."""
    assert trust.recompute_row("10k-50k", CEILING, 29, 29, 29, BreakerState())["budget"] == 0.0
    at_400 = trust.recompute_row("10k-50k", CEILING, 400, 400, 400, BreakerState())["budget"]
    assert at_400 / CEILING >= 0.90

    state = trust.evaluate_breaker(BreakerState(), decisions("c" * 397 + "www"), TS)
    after = trust.recompute_row("10k-50k", CEILING, 400, 400, 400, state)
    assert after["budget"] == 0.0


def test_the_breaker_cannot_raise_a_budget():
    """16's invariant under everything: an unavailable or degraded component
    reduces autonomy, never increases it."""
    for n in (0, 50, 73, 400, 5000):
        clean = trust.recompute_row("10k-50k", CEILING, n, n, n, BreakerState())
        tripped = trust.recompute_row("10k-50k", CEILING, n, n, n, BreakerState(tripped_at=TS))
        assert tripped["budget"] <= clean["budget"]


def test_recompute_matches_the_ledger_when_untripped():
    """The breaker is a veto layered on top of section 9, not a second
    scoring path that could drift away from it."""
    row = trust.recompute_row("2k-10k", exposure.ceiling("issue_refund", "2k-10k"),
                              200, 200, 200, BreakerState())
    assert row["p_lo"] == pytest.approx(ledger.wilson_lower(200, 200))
    assert row["budget"] == pytest.approx(128.92, abs=0.01)
