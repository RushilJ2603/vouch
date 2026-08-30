"""The gate -- the only component that decides (10).

Three numbers meet here and one comparison settles it. About thirty lines of
actual decision logic, and that is the point: one auditable place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from . import config


class Verdict(str, Enum):
    ACT = "act"
    CHECK_HARDER = "check_harder"
    ESCALATE = "escalate"
    BLOCK = "block"


class Region(str, Enum):
    """Stamped at decision time, not derived later (11). Calibration is
    reported per region because a global ECE hides the close-call band -- the
    only region where a better estimate changes the outcome."""
    ACT = "act"
    CLOSE_CALL = "close_call"
    ESCALATE = "escalate"


@dataclass
class InvariantResult:
    violated: bool = False
    failed: list[str] = field(default_factory=list)


# ── 10.2 Invariants -- the fixed path ──────────────────────────────────────
# Run on every request regardless of score. No track record buys past them.
# An invariant whose sensor is `unavailable` FAILS CLOSED: absence of evidence
# is not evidence.

def check_invariants(signals: dict[str, float | None], ctx: dict[str, float]) -> InvariantResult:
    failed: list[str] = []

    def below(name: str, key: str, threshold: float) -> None:
        value = signals.get(key)
        if value is None or value >= threshold:   # unavailable -> violated
            failed.append(name)

    below("no_pii_egress", "pii_score", 0.5)
    below("no_prompt_injection", "injection_score", 0.5)
    below("no_secrets_in_output", "secret_score", 0.5)

    if ctx.get("retrieval_scope", 0) > ctx.get("user_scope", 0):
        failed.append("within_authorisation_scope")
    if ctx.get("amount", 0) > ctx.get("hard_limit", float("inf")):
        failed.append("within_hard_amount_limit")

    return InvariantResult(violated=bool(failed), failed=failed)


# ── 10.1 decide() ──────────────────────────────────────────────────────────


def decide(
    p_wrong: float,
    exposure: float,
    budget: float,
    invariants: InvariantResult,
    k: float | None = None,
    review_cost: float | None = None,
) -> Verdict:
    k = config.K if k is None else k
    review_cost = config.REVIEW_COST if review_cost is None else review_cost

    if invariants.violated:                    # fixed path -- never earnable
        return Verdict.BLOCK

    if exposure <= review_cost:                # day-one floor -- see below
        return Verdict.ACT

    expected_loss = p_wrong * exposure
    if expected_loss <= budget:
        return Verdict.ACT
    if expected_loss <= budget * k:            # the close-call band
        return Verdict.CHECK_HARDER
    return Verdict.ESCALATE


# The day-one floor tests EXPOSURE, not EXPECTED LOSS, and that distinction is
# load-bearing. Under the old formulation a 62,000 refund with exposure 22,320
# and a confident p_wrong = 0.0015 gives an expected loss of 33.48 -- below the
# 40 review cost -- so the gate would have executed a 62,000 payout from an
# agent with a three-decision record. Testing exposure says something
# unconditional instead: this entire action, even if completely wrong, costs
# less than having a person look at it. True of an 80 refund; never true of a
# 62,000 one, no matter how confident the model is.
#
# When budget = 0 the close-call band collapses to nothing, because
# budget * k = 0. Everything escalates and Tier 2 never fires. That falls out
# of the arithmetic rather than needing a special case, and it is reachable
# precisely because budget() is not floored (5.3).


def region_of(verdict: Verdict) -> Region:
    if verdict is Verdict.CHECK_HARDER:
        return Region.CLOSE_CALL
    if verdict is Verdict.ACT:
        return Region.ACT
    return Region.ESCALATE


def redecide_after_tier2(
    p_wrong: float, exposure: float, budget: float, invariants: InvariantResult
) -> Verdict:
    """Tier 2 re-entry (10.1): re-run at k = 1.0 so the second pass has no
    close-call band and must resolve. Capped at one re-entry. No loops."""
    return decide(p_wrong, exposure, budget, invariants, k=1.0)
