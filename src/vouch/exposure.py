"""Exposure -- pricing the action (8), and the derived per-band ceiling (5.2).

Exposure is not the rupee amount. Every action declares its own risk surface,
and the ceiling for a band falls out of `risk_appetite` rather than being four
unexplained constants.
"""
from __future__ import annotations

from typing import Any

from . import config


def _resolve(spec: dict[str, Any], key: str, payload: dict[str, Any], default: float) -> float:
    """A multiplier is either a number or the name of a field in the payload."""
    value = spec.get(key, default)
    if isinstance(value, str):
        return float(payload[value])
    return float(value)


def multipliers(action: str, payload: dict[str, Any] | None = None) -> float:
    """reversibility * visibility * blast_radius, combined."""
    payload = payload or {}
    spec = config.action_spec(action)
    return (
        _resolve(spec, "reversibility", payload, 1.0)
        * _resolve(spec, "visibility", payload, 1.0)
        * _resolve(spec, "blast_radius", payload, 1.0)
    )


def exposure(action: str, amount: float, payload: dict[str, Any] | None = None) -> float:
    """(amount or base_amount) * reversibility * visibility * blast_radius.

    1,240 refund:  1240 * 0.30 * 1.20 * 1.00 = 446.40
    """
    spec = config.action_spec(action)
    base = amount if spec.get("amount_field") is not None else float(spec["base_amount"])
    return base * multipliers(action, payload)


def ceiling(action: str, band: str, payload: dict[str, Any] | None = None) -> float:
    """ceiling(band) = risk_appetite * band_max * reversibility * visibility * blast_radius

    Reads in English as: at full proven reliability, we tolerate up to a 4%
    chance of being wrong on the largest action in this band. That is a
    sentence a business owner can accept or reject. "The ceiling is 720" is not.
    """
    return config.RISK_APPETITE * config.band_max(band) * multipliers(action, payload)


def hard_limit(action: str) -> float:
    return float(config.action_spec(action)["hard_limit"])
