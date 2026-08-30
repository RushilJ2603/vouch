"""Step 2 of the nightly flow -- recompute trust, and the circuit breaker (13, 9.5).

Trust rises slowly and falls fast. Going up requires volume and sustained
clean performance; going down does not, because a confirmed failure in a
high-trust row hits hard -- the premise was that this was safe.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import config, ledger

TRIP_FAILURES = config.TRIP_FAILURES        # confirmed failures...
TRIP_WINDOW = config.TRIP_WINDOW            # ...within this many recent decisions
RECOVER_CLEAN = config.RECOVER_CLEAN        # clean decisions to leave supervised


@dataclass
class BreakerState:
    tripped_at: float | None = None
    clean_since_trip: int = 0

    @property
    def tripped(self) -> bool:
        return self.tripped_at is not None


def evaluate_breaker(state: BreakerState, recent: list[dict],
                     run_reference_ts: float) -> BreakerState:
    """Averages are slow, and slow is exactly wrong when something has started
    going bad. The breaker reads the most recent window, not the lifetime.

    Recovery requires CLEAN DECISIONS rather than elapsed time, and the bar to
    re-enter autonomy is higher than the bar to stay in it -- otherwise a row
    on the boundary flaps between states on every decision.
    """
    window = recent[-TRIP_WINDOW:]
    failures = sum(1 for r in window if r.get("outcome") == "wrong")

    if not state.tripped:
        if failures >= TRIP_FAILURES:
            return BreakerState(tripped_at=run_reference_ts, clean_since_trip=0)
        return state

    clean_run = 0
    for row in reversed(recent):
        if row.get("outcome") == "clean":
            clean_run += 1
        elif row.get("outcome") == "wrong":
            break
    if clean_run >= RECOVER_CLEAN:
        return BreakerState(tripped_at=None, clean_since_trip=0)
    return BreakerState(tripped_at=state.tripped_at, clean_since_trip=clean_run)


def recompute_row(band: str, ceiling: float, n_clean: float, n_total: float,
                  n_own_raw: int, breaker: BreakerState,
                  neighbours: list[tuple[str, float, float]] | None = None) -> dict:
    """Full recomputation for one trust row.

    A tripped breaker forces budget to zero regardless of the record. That is
    the point: the record says this configuration was safe, and the breaker is
    the evidence that it has stopped being.
    """
    row = ledger.evaluate(band, ceiling, n_clean, n_total, n_own_raw, neighbours or [])
    if breaker.tripped:
        row["budget"] = 0.0
        row["state"] = "supervised"
        row["tripped_at"] = breaker.tripped_at
    row["clean_since_trip"] = breaker.clean_since_trip
    return row
