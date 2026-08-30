"""The response ladder -- seven rungs measured in human attention (10.3).

Above the line the system does not simply block. Human attention is the
scarcest thing here, so the ladder spends as little of it as the failure
allows.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .gate import Verdict


class Rung(IntEnum):
    PASS = 0        # goes out, action executes, logged            none
    ANNOTATE = 1    # confidence marker or citation attached       none
    REDACT = 2      # one span masked, the rest ships              none
    REGENERATE = 3  # retried with the reason, capped at 2         none
    ADVISORY = 4    # agent drafts, a human clicks approve         ~15 s
    ESCALATE = 5    # held, a human decides with evidence          minutes
    BLOCK = 6       # refused; no human offered                    none


HUMAN_TIME = {
    Rung.PASS: "none", Rung.ANNOTATE: "none", Rung.REDACT: "none",
    Rung.REGENERATE: "none", Rung.ADVISORY: "~15 s",
    Rung.ESCALATE: "minutes", Rung.BLOCK: "none",
}

MAX_REGENERATE_ATTEMPTS = 2


@dataclass
class Signals:
    """Only what the ladder reads. An `unavailable` sensor arrives as None and
    is treated as present-and-dangerous, matching the invariants (10.2)."""
    pii_score: float | None = 0.0
    secret_score: float | None = 0.0
    injection_score: float | None = 0.0
    verify_fail_frac: float | None = 0.0
    span_is_localised: bool = False

    def _at_least(self, value: float | None) -> float:
        return 1.0 if value is None else value      # unavailable fails closed

    @property
    def pii(self) -> float: return self._at_least(self.pii_score)

    @property
    def secret(self) -> float: return self._at_least(self.secret_score)

    @property
    def injection(self) -> float: return self._at_least(self.injection_score)

    @property
    def verify_fail(self) -> float: return self._at_least(self.verify_fail_frac)


def select_rung(signals: Signals, expected_loss: float, budget: float,
                has_side_effect: bool) -> Rung:
    """The failure TYPE picks the family; the DISTANCE above budget picks how
    far up that family the response goes.

    Rung 4 is the one most systems skip and the one that decides whether the
    review queue survives. The agent does all the work and stops one inch
    short of executing, so a human approves a finished thing in fifteen
    seconds instead of doing it themselves in ten minutes.
    """
    severity = expected_loss / max(budget, 1e-9)

    if signals.pii > 0.5 or signals.secret > 0.5:
        return Rung.REDACT if signals.span_is_localised else Rung.ESCALATE
    if signals.injection > 0.5:
        return Rung.BLOCK
    if signals.verify_fail > 0:                     # a claim contradicts the record
        return Rung.REGENERATE if severity < 3 else Rung.ESCALATE
    if has_side_effect:
        return Rung.ADVISORY if severity < 3 else Rung.ESCALATE
    return Rung.ANNOTATE if severity < 2 else Rung.ESCALATE


def rung_for_verdict(verdict: Verdict, signals: Signals, expected_loss: float,
                     budget: float, has_side_effect: bool) -> Rung:
    """A BLOCK verdict is rung 6 whatever the ladder would otherwise pick, and
    an ACT verdict never climbs above rung 1: the gate has already decided the
    action is affordable, so spending human attention on it is the waste the
    whole design exists to remove."""
    if verdict is Verdict.BLOCK:
        return Rung.BLOCK
    if verdict is Verdict.ACT:
        return Rung.PASS
    return select_rung(signals, expected_loss, budget, has_side_effect)


def next_after_failed_regenerate(attempts: int) -> Rung:
    """The cap prevents retry loops, which are otherwise the cheapest way to
    burn money and latency at once."""
    return Rung.REGENERATE if attempts < MAX_REGENERATE_ATTEMPTS else Rung.ESCALATE
