"""Configuration loading and the typed constants everything else descends from.

Loaded once at startup. §5 is the numeric spine: change a number here and it
changes everywhere, which is the entire point of the file existing.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────

CONFIG_DIR = Path(os.environ.get("VOUCH_CONFIG_DIR", Path(__file__).resolve().parents[2] / "config"))
VOUCH_YAML = CONFIG_DIR / "vouch.yaml"
ACTIONS_YAML = CONFIG_DIR / "actions.yaml"
POLICIES_YAML = CONFIG_DIR / "policies.yaml"

# ── Size bands (§5.2) ──────────────────────────────────────────────────────
# Fixed, not learned. Rupees. The top band is CLOSED by hard_limit — an
# open-ended band has no maximum exposure and therefore no derivable ceiling.

BANDS: list[tuple[str, float, float]] = [
    ("0-2k",    0.0,       2_000.0),
    ("2k-10k",  2_000.0,  10_000.0),
    ("10k-50k", 10_000.0, 50_000.0),
    ("50k+",    50_000.0, 200_000.0),
]


def band_for(amount: float) -> str:
    """Which band an amount falls in. Upper-exclusive, except the closed top."""
    for name, lo, hi in BANDS:
        if lo <= amount < hi:
            return name
    if amount == BANDS[-1][2]:
        return BANDS[-1][0]
    raise ValueError(f"amount {amount} is outside every band; hard_limit should have blocked it")


def band_max(band: str) -> float:
    for name, _lo, hi in BANDS:
        if name == band:
            return hi
    raise KeyError(band)


# ── Loading ────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    with VOUCH_YAML.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def load_actions() -> dict[str, Any]:
    with ACTIONS_YAML.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def load_policies() -> dict[str, Any]:
    with POLICIES_YAML.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def policy_profile(name: str) -> dict[str, Any]:
    profile = load_policies().get(name)
    if profile is None:
        raise KeyError(name)
    return profile


def action_spec(action: str) -> dict[str, Any]:
    """An action with no entry has no price, so the caller must BLOCK (§8)."""
    spec = load_actions().get(action)
    if spec is None:
        raise KeyError(action)
    return spec


# ── The numeric spine (§5.1) ───────────────────────────────────────────────
# Module-level so callers read `config.Z`, not `config.load()["ledger"]["z"]`.

_cfg = load()

Z: float = _cfg["ledger"]["z"]
P_MIN: float = _cfg["ledger"]["p_min"]
RISK_APPETITE: float = _cfg["ledger"]["risk_appetite"]
BORROW_WEIGHT: float = _cfg["ledger"]["borrow_weight"]
MIN_OWN_OBSERVATIONS: int = _cfg["ledger"]["min_own_observations"]
AGE_HALFLIFE_DAYS: float = _cfg["ledger"]["age_halflife_days"]
TRIP_FAILURES: int = _cfg["ledger"]["trip_failures"]
TRIP_WINDOW: int = _cfg["ledger"]["trip_window"]
RECOVER_CLEAN: int = _cfg["ledger"]["recover_clean"]

FORCED_REVIEW_RATE: float = _cfg["sampling"]["forced_review_rate"]

K: float = _cfg["gate"]["k"]
REVIEW_COST: float = _cfg["gate"]["review_cost"]

TIER2_MODEL: str = _cfg["tier2"]["model"]
TIER2_THINKING: str = _cfg["tier2"]["thinking"]
TIER2_TIMEOUT_SEC: float = _cfg["tier2"]["timeout_sec"]
TIER2_MAX_REENTRY: int = _cfg["tier2"]["max_reentry"]

AGENT_MODEL: str = _cfg["agent"]["model"]


# ── Sensor version (§9.4) ──────────────────────────────────────────────────

def sensor_version() -> str:
    """Version string for the whole sensor stack, folded into the fingerprint.

    The Tier 2 model AND its thinking mode are part of this deliberately.
    Feature 15 (`judge_p_wrong`) is a probability produced by that exact
    configuration; flipping `thinking` from disabled to enabled changes its
    distribution while leaving `model_id` untouched. Without this, the
    fingerprint would not move and every trust row would silently carry
    evidence earned under a different judge.
    """
    payload = json.dumps(
        {
            "tier0": _cfg["sensors"]["tier0_version"],
            "tier1": _cfg["sensors"]["tier1_version"],
            "tier2_model": TIER2_MODEL,
            "tier2_thinking": TIER2_THINKING,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sv-" + hashlib.sha256(payload.encode()).hexdigest()[:12]
