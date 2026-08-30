"""Population Stability Index and the dual-baseline scheme (9.7).

One baseline cannot answer both questions the system needs answered: "has the
input shifted since we last looked" and "has it shifted since the calibrator
was fitted". A single baseline that re-anchors forgets the second; one that
never moves alarms forever on a legitimate permanent shift.
"""
from __future__ import annotations

import math
from typing import Sequence

DEFAULT_BINS = 10
MIN_OBSERVATIONS = 100      # below this: insufficient_data, no alarm (19)
_EPS = 1e-6


def quantile_edges(reference: Sequence[float], bins: int = DEFAULT_BINS) -> list[float]:
    """Bin edges from the REFERENCE distribution, held fixed afterwards.

    Re-deriving edges from the current window each night compares two moving
    targets and can report perfect stability while the distribution walks away.
    """
    if not reference:
        return []
    ordered = sorted(reference)
    n = len(ordered)
    return [ordered[min(n - 1, int(round(i * n / bins)))] for i in range(1, bins)]


def _histogram(values: Sequence[float], edges: Sequence[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for v in values:
        slot = 0
        while slot < len(edges) and v > edges[slot]:
            slot += 1
        counts[slot] += 1
    total = len(values) or 1
    return [c / total for c in counts]


def psi(reference: Sequence[float], current: Sequence[float],
        bins: int = DEFAULT_BINS) -> float:
    """sum (curr - ref) * ln(curr / ref) over fixed reference bins.

    Rule of thumb: < 0.10 stable, 0.10-0.25 moderate, > 0.25 material.
    """
    if not reference or not current:
        return 0.0
    edges = quantile_edges(reference, bins)
    ref = _histogram(reference, edges)
    cur = _histogram(current, edges)
    total = 0.0
    for r, c in zip(ref, cur):
        r, c = max(r, _EPS), max(c, _EPS)
        total += (c - r) * math.log(c / r)
    return total


def has_power(n: int) -> bool:
    """Every alarm carries a minimum-sample floor. A quiet Tuesday is not
    drift, and a system that cries wolf on 40 observations gets muted."""
    return n >= MIN_OBSERVATIONS
