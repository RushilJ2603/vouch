"""The trust ledger — Wilson bound, earned budget, borrowing, fingerprint (§9).

Nothing here decides anything. It answers one question: given this row's
record, how much is it allowed to be wrong about, in rupees.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

from . import config

# ── 9.1 Wilson lower bound ─────────────────────────────────────────────────
#
#         p + z^2/2n - z*sqrt( p(1-p)/n + z^2/4n^2 )
# p_lo = --------------------------------------------
#                     1 + z^2/n
#
# Two-sided 95%, z = 1.96, pinned in config. The pessimistic bound, never the
# raw rate: 3 clean out of 3 is a perfect record that proves almost nothing.


def wilson_lower(clean: float, total: float, z: float | None = None) -> float:
    """Lower bound of the two-sided confidence interval on the clean rate."""
    z = config.Z if z is None else z
    if total <= 0:
        return 0.0                      # defined, not an error
    p = clean / total
    d = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, (centre - margin) / d)


def wilson_interval(clean: float, total: float, z: float | None = None) -> tuple[float, float]:
    """Both bounds. Used for the per-bin intervals on the reliability diagram
    (7.5) -- with ~150 rows per bin those intervals are wide, and the design
    plots them rather than drawing a smooth line through sparse data."""
    z = config.Z if z is None else z
    if total <= 0:
        return 0.0, 0.0
    p = clean / total
    d = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, (centre - margin) / d), min(1.0, (centre + margin) / d)


# ── 9.2 Budget ─────────────────────────────────────────────────────────────


def budget(p_lo: float, ceiling: float, n_own_raw: int) -> float:
    """What this row has earned. Zero until the record justifies more.

    Reaches EXACTLY zero, and three other parts of the design depend on that:
    the collapsing close-call band (10.1), the golden fixtures (22), and
    worked case B. The day-one floor is NOT here -- it lives in the gate as an
    exposure test (10.1), where it belongs.
    """
    if n_own_raw < config.MIN_OWN_OBSERVATIONS:      # supervised (9.3)
        return 0.0
    earned = ceiling * (p_lo - config.P_MIN) / (1.0 - config.P_MIN)
    return max(0.0, min(ceiling, earned))


# ── 9.3 Borrowing strength across bands ────────────────────────────────────
# Borrow via EFFECTIVE COUNTS so the Wilson formula stays unchanged and rows
# that borrow nothing reproduce the published numbers exactly.


def _decay(distance: int) -> float:
    return 0.5 ** distance


def effective_counts(
    band: str,
    own_clean: float,
    own_total: float,
    neighbours: Iterable[tuple[str, float, float]] = (),
) -> tuple[float, float]:
    """(k_eff, n_eff) after borrowing from neighbours at lambda = borrow_weight."""
    names = [b[0] for b in config.BANDS]
    i = names.index(band)
    lam = config.BORROW_WEIGHT
    k_eff, n_eff = own_clean, own_total
    for other, clean_j, total_j in neighbours:
        w = lam * _decay(abs(i - names.index(other)))
        k_eff += clean_j * w
        n_eff += total_j * w
    return k_eff, n_eff


def evaluate(
    band: str,
    ceiling: float,
    own_clean: float,
    own_total: float,
    n_own_raw: int,
    neighbours: Iterable[tuple[str, float, float]] = (),
) -> dict[str, Any]:
    """Full row evaluation: borrow, bound, price.

    The min-observations gate counts `n_own_raw` -- unweighted and un-borrowed --
    because a gate that constrains borrowing must not itself be satisfiable by
    borrowing. Without it, 40,000 clean small refunds would buy a 2,880 budget
    on a 62,000 refund the agent has attempted twice.
    """
    k_eff, n_eff = effective_counts(band, own_clean, own_total, neighbours)
    p_lo = wilson_lower(k_eff, n_eff)
    b = budget(p_lo, ceiling, n_own_raw)
    return {
        "band": band,
        "n_eff": n_eff,
        "k_eff": k_eff,
        "n_own_raw": n_own_raw,
        "p_lo": p_lo,
        "ceiling": ceiling,
        "budget": b,
        "state": "autonomous" if b > 0.0 else "supervised",
    }


# ── 9.4 Configuration fingerprint ──────────────────────────────────────────


def fingerprint(
    model_id: str,
    system_prompt: str,
    tool_schemas: list[dict[str, Any]],
    sensor_version: str,
    calibrator_version: str,
) -> str:
    """Trust is keyed on the configuration that earned it.

    Change the configuration and no row is found, so the reset is structural
    rather than a policy anyone has to remember to enforce.
    """
    payload = json.dumps(
        {
            "model": model_id,
            "prompt": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "tools": sorted(tool_schemas, key=lambda t: t["name"]),
            "sensors": sensor_version,
            "calibrator": calibrator_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
