"""Feature assembly (Appendix A), with point-in-time integrity enforced by the
signature itself (11).

`assemble()` takes `as_of` as a REQUIRED POSITIONAL parameter. There is no
default. A caller without a reference time cannot compile, which is the whole
enforcement mechanism -- a post-hoc statistical check can only tell you
afterwards that something leaked, without telling you where.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any, Iterable, Sequence

# ── The feature order is the contract (Appendix A) ─────────────────────────
# The calibrator consumes exactly these 15, in this order. `feature_order` in
# baselines.json must match, and the 21.3 gate asserts it. A silently
# reordered vector is a wrong prediction that never errors.

FEATURE_ORDER: tuple[str, ...] = (
    "verify_fail_frac",        # 1
    "verify_n_claims",         # 2
    "retrieval_support_min",   # 3
    "retrieval_support_mean",  # 4
    "tool_retry_count",        # 5
    "tool_error_count",        # 6
    "hedge_density",           # 7
    "length_z",                # 8
    "logprob_mean",            # 9
    "logprob_min",             # 10
    "pii_score",               # 11
    "injection_score",         # 12
    "policy_score",            # 13
    "secret_score",            # 14
    "judge_p_wrong",           # 15
)

# Which features can be `unavailable` and therefore carry a companion
# indicator column. Features 5, 6, 7 and 14 cannot: there is no model to
# saturate and no external system to fail.
CAN_BE_UNAVAILABLE = frozenset({
    "verify_fail_frac", "verify_n_claims",
    "retrieval_support_min", "retrieval_support_mean",
    "length_z",
    "logprob_mean", "logprob_min",
    "pii_score", "injection_score", "policy_score",
    "judge_p_wrong",
})

# ── Capability tiers (7.2) ─────────────────────────────────────────────────
# Several providers do not expose log-probabilities. Rather than assume the
# capability, the calibrator is fit PER TIER, so a deployment without logprobs
# gets a model fit on the features it actually has instead of one leaning on
# imputed medians.

MODEL_A_FEATURES = FEATURE_ORDER[:14]      # Tier 2 has not fired
MODEL_B_FEATURES = FEATURE_ORDER           # Tier 2 has fired

CAPABILITY_TIERS: dict[str, tuple[str, ...]] = {
    "full": FEATURE_ORDER,
    "topk": FEATURE_ORDER,                                  # 9, 10 filled by an entropy proxy
    "none": tuple(f for f in FEATURE_ORDER if not f.startswith("logprob")),
}


class Missing(str, Enum):
    """7.6. The distinction is between absence of RISK and absence of
    INFORMATION, and only the second should widen the estimate."""
    VALUE = "value"
    NOT_SUPPORTED = "not_supported"   # capability absent for this deployment; no effect
    UNAVAILABLE = "unavailable"       # should have been measured and was not; widens


# ── Feature 8, the point-in-time trap ──────────────────────────────────────

MIN_ROWS_FOR_LENGTH_Z = 30


def length_z(
    length: int,
    history: Iterable[tuple[float, int]],
    as_of: float,
) -> tuple[float | None, Missing]:
    """(len(answer) - mu_action) / sigma_action, over THIS ACTION's rows
    strictly before `as_of`.

    `history` is an iterable of (ts, length) for the same action. Computing mu
    and sigma over the whole corpus is the silent leak of 11: a response
    written on day one that was unusually long FOR DAY ONE scores as typical,
    because days two and three shifted the mean. The calibrator then learns
    that length carries no signal and gives it a coefficient near zero. In
    production, where the distribution is only knowable up to the present, the
    feature carries real signal -- and the model has been taught to ignore it.
    Nothing errors. The feature is simply dead.
    """
    past = [n for ts, n in history if ts < as_of]
    if len(past) < MIN_ROWS_FOR_LENGTH_Z:
        return None, Missing.UNAVAILABLE
    mu = sum(past) / len(past)
    var = sum((n - mu) ** 2 for n in past) / len(past)
    sigma = math.sqrt(var)
    if sigma == 0.0:
        return 0.0, Missing.VALUE
    return (length - mu) / sigma, Missing.VALUE


# ── Assembly ───────────────────────────────────────────────────────────────

def assemble(
    as_of: float,
    signals: dict[str, tuple[Any, Missing]],
    capability_tier: str = "full",
) -> dict[str, Any]:
    """Build the feature dict for one decision.

    `as_of` is required and positional on purpose (11). It is unused inside
    this function by design -- the caller must already have applied it when
    computing `length_z` and the retrieval features -- but it is carried into
    the returned record so a replay can assert what reference time produced it.

    `signals` maps a feature name to (value, Missing). Sentinels are assigned
    HERE, at assembly time, and never patched afterwards.
    """
    if capability_tier not in CAPABILITY_TIERS:
        raise ValueError(f"unknown capability tier {capability_tier!r}")

    out: dict[str, Any] = {"_as_of": as_of, "_capability_tier": capability_tier}
    active = CAPABILITY_TIERS[capability_tier]

    for name in FEATURE_ORDER:
        value, reason = signals.get(name, (None, Missing.UNAVAILABLE))

        if name not in active:
            # Constant across every request for this deployment. Absent from
            # the model entirely rather than flagged as degraded.
            out[name] = None
            continue

        if reason is Missing.NOT_SUPPORTED:
            out[name] = None
        elif reason is Missing.UNAVAILABLE:
            out[name] = None
        else:
            out[name] = value

        if name in CAN_BE_UNAVAILABLE:
            out[f"{name}_unavailable"] = 1 if reason is Missing.UNAVAILABLE else 0

    return out


def to_vector(features: dict[str, Any], names: Sequence[str]) -> list[float]:
    """Ordered numeric vector for fitting or inference.

    An `unavailable` feature contributes 0.0 in the value slot and 1 in its
    companion indicator, so the calibrator learns from the INDICATOR that
    information was missing, rather than from an imputed value pretending it
    was not.
    """
    vector: list[float] = []
    for name in names:
        value = features.get(name)
        vector.append(0.0 if value is None else float(value))
        if name in CAN_BE_UNAVAILABLE:
            vector.append(float(features.get(f"{name}_unavailable", 0)))
    return vector


def vector_names(names: Sequence[str]) -> list[str]:
    """Column names matching `to_vector`, for coefficient inspection (21.3)."""
    out: list[str] = []
    for name in names:
        out.append(name)
        if name in CAN_BE_UNAVAILABLE:
            out.append(f"{name}_unavailable")
    return out
