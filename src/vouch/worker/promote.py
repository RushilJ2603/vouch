"""Calibrator governance (18).

`calibrator_version` sits inside the fingerprint, which is correct -- a refit
changes what `P(wrong) = 0.02` means. Taken naively that means a routine refit
resets every trust row in the system to supervised: months of evidence gone
because somebody improved the model. This module exists so that cannot happen
by accident.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .. import config

_CAL = config.load()["calibration"]
MIGRATE_CARRY_MAX = _CAL["migrate_carry_max"]      # 0.02
MIGRATE_HALVE_MAX = _CAL["migrate_halve_max"]      # 0.05
MIN_REGION_N = _CAL["promotion_min_region_n"]      # 100

SHADOW_MIN_DAYS = 30
SHADOW_MIN_OUTCOMES = 500


class Path(str, Enum):
    A = "A"          # held-out gate, auto-promote
    B = "B"          # shadow scoring, human promotes
    BLOCKED = "blocked"


class Migration(str, Enum):
    CARRY = "carry"          # evidence intact
    HALVE = "halve"          # evidence halved, recompute
    RESET = "reset"          # meanings diverged too far


@dataclass
class Candidate:
    version: str
    ece_global: float
    ece_close_call: float
    n_global: int
    n_close_call: int
    sklearn_version: str
    artifact_sha256: str


def path_a_passes(candidate: Candidate, production: Candidate) -> bool:
    """Better than production BOTH globally and in the close-call region, with
    at least 100 samples in each. A candidate that improves the average while
    degrading close calls is worse where it matters."""
    if candidate.n_global < MIN_REGION_N or candidate.n_close_call < MIN_REGION_N:
        return False
    return (candidate.ece_global < production.ece_global
            and candidate.ece_close_call < production.ece_close_call)


def decide_path(candidate: Candidate, production: Candidate,
                running_sklearn: str) -> Path:
    """A version mismatch blocks outright. A joblib fitted under a different
    scikit-learn loads WITHOUT ERROR and scores differently -- silent
    train/serve skew that would corrupt every subsequent trust update (2.8).
    """
    if candidate.sklearn_version != running_sklearn:
        return Path.BLOCKED
    return Path.A if path_a_passes(candidate, production) else Path.B


def shadow_promotion_ready(days_elapsed: float, closed_outcomes: int,
                           challenger_better: bool) -> bool:
    """30 days OR 500 closed outcomes, WHICHEVER COMES LATER -- and then a
    human decides. Never automatic.

    Path B exists because a calibrator refitted after a genuine distribution
    shift often scores WORSE on a frozen test set drawn from before the shift
    while being materially better on live traffic. Single-gate promotion
    blocks exactly the model you need most, and the cost of not having this
    path is highest precisely when the system is under stress.
    """
    return (challenger_better
            and days_elapsed >= SHADOW_MIN_DAYS
            and closed_outcomes >= SHADOW_MIN_OUTCOMES)


def migration_for(ece_delta_close_call: float) -> Migration:
    """The fingerprint's job is to prevent trusting a track record earned by a
    DIFFERENT system. If the new calibrator agrees with the old to within 0.02
    ECE in the region that decides things, it is not a different system in any
    way the arithmetic cares about."""
    if ece_delta_close_call <= MIGRATE_CARRY_MAX:
        return Migration.CARRY
    if ece_delta_close_call <= MIGRATE_HALVE_MAX:
        return Migration.HALVE
    return Migration.RESET


def migrate(row: dict, ece_delta_close_call: float) -> dict:
    action = migration_for(ece_delta_close_call)
    out = dict(row)
    if action is Migration.CARRY:
        out["migration"] = "carry"
        return out
    if action is Migration.HALVE:
        out["n_total"] = row.get("n_total", 0.0) * 0.5
        out["n_clean"] = row.get("n_clean", 0.0) * 0.5
        out["migration"] = "halve"
        return out
    out.update(n_total=0.0, n_clean=0.0, n_own_raw=0, p_lo=0.0,
               budget=0.0, state="supervised", migration="reset")
    return out
