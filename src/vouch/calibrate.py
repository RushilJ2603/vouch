"""Calibration and its measurement (7.5).

Raw logistic output is close to calibrated but not reliably so, so a
single-parameter Platt scaler is fitted on a held-out split. Everything else
here exists to MEASURE that, because calibration is the thing the whole system
is claiming.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .ledger import wilson_interval

# ── Splits (7.5) ───────────────────────────────────────────────────────────
# train / calibrate / test = 60 / 20 / 20, stratified on the outcome label.
# Platt, not isotonic: isotonic is better known and non-parametric, but it
# overfits at ~300 calibration rows.

DEFAULT_BINS = 10


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ── Platt scaling ──────────────────────────────────────────────────────────

@dataclass
class Platt:
    """A single-parameter logistic on the model's logit (plus intercept)."""
    a: float = 1.0
    b: float = 0.0

    def apply(self, p: float) -> float:
        return _sigmoid(self.a * _logit(p) + self.b)

    def apply_all(self, probs: Sequence[float]) -> list[float]:
        return [self.apply(p) for p in probs]


def fit_platt(probs: Sequence[float], labels: Sequence[int],
              iterations: int = 200, lr: float = 0.1) -> Platt:
    """Newton-free gradient fit. Two parameters and a few hundred rows -- a
    dependency on an optimiser would be more surface than the thing it solves.
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels differ in length")
    if not probs:
        raise ValueError("cannot fit a calibrator on an empty split")

    xs = [_logit(p) for p in probs]
    a, b = 1.0, 0.0
    n = len(xs)
    for _ in range(iterations):
        ga = gb = 0.0
        for x, y in zip(xs, labels):
            err = _sigmoid(a * x + b) - y
            ga += err * x
            gb += err
        a -= lr * ga / n
        b -= lr * gb / n
    return Platt(a=a, b=b)


# ── Reliability, in equal-frequency bins ───────────────────────────────────

@dataclass
class Bin:
    n: int
    predicted: float
    observed: float
    lo: float
    hi: float


@dataclass
class Report:
    n: int
    ece: float
    mce: float
    brier: float
    bins: list[Bin] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"n": self.n, "ece": self.ece, "mce": self.mce, "brier": self.brier,
                "bins": [b.__dict__ for b in self.bins]}


def quantile_bins(probs: Sequence[float], n_bins: int = DEFAULT_BINS) -> list[list[int]]:
    """Equal-FREQUENCY bins, not equal-width. Predictions cluster near zero and
    equal-width bins leave most of the range empty."""
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    if not order:
        return []
    n_bins = max(1, min(n_bins, len(order)))
    size, extra = divmod(len(order), n_bins)
    out, start = [], 0
    for i in range(n_bins):
        stop = start + size + (1 if i < extra else 0)
        if stop > start:
            out.append(order[start:stop])
        start = stop
    return out


def evaluate(probs: Sequence[float], labels: Sequence[int],
             n_bins: int = DEFAULT_BINS) -> Report:
    """ECE, MCE and Brier, with a Wilson interval on every bin's observed rate.

    ECE   = sum_b (n_b / N) * |observed_b - predicted_b|
    MCE   = max_b |observed_b - predicted_b|
    Brier = (1/N) sum (p_i - y_i)^2
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels differ in length")
    n = len(probs)
    if n == 0:
        return Report(n=0, ece=0.0, mce=0.0, brier=0.0, bins=[])

    bins: list[Bin] = []
    ece = mce = 0.0
    for idx in quantile_bins(probs, n_bins):
        nb = len(idx)
        predicted = sum(probs[i] for i in idx) / nb
        wrong = sum(labels[i] for i in idx)
        observed = wrong / nb
        lo, hi = wilson_interval(wrong, nb)
        bins.append(Bin(n=nb, predicted=predicted, observed=observed, lo=lo, hi=hi))
        gap = abs(observed - predicted)
        ece += (nb / n) * gap
        mce = max(mce, gap)

    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n
    return Report(n=n, ece=ece, mce=mce, brier=brier, bins=bins)


# ── Per-region reporting is MANDATORY (7.5) ────────────────────────────────

REGIONS = ("act", "close_call", "escalate")
MIN_REGION_N = 100          # calibration.promotion_min_region_n


def evaluate_by_region(
    probs: Sequence[float],
    labels: Sequence[int],
    regions: Sequence[str],
    n_bins: int = DEFAULT_BINS,
) -> dict[str, Report]:
    """ECE is an average, and an average hides the tail.

    A calibrator can score 1,380 `act` turns at ECE 0.011 and 95 `close_call`
    turns at 0.21 -- saying 12% where reality is 34% -- and still show a global
    ECE around 0.026, comfortably inside the 0.05 pass condition. It would ship
    systematically under-escalating exactly the decisions that were too close
    to call. `close_call` is the ONLY region where a better estimate changes a
    decision, so it is reported separately and gated separately.
    """
    if not (len(probs) == len(labels) == len(regions)):
        raise ValueError("probs, labels and regions differ in length")

    out = {"global": evaluate(probs, labels, n_bins)}
    for region in REGIONS:
        idx = [i for i, r in enumerate(regions) if r == region]
        out[region] = evaluate([probs[i] for i in idx], [labels[i] for i in idx], n_bins)
    return out


def region_has_power(report: Report) -> bool:
    """Below the floor a region reports `insufficient_data` and raises no
    alarm. Every alarm in this system carries a minimum-sample floor (19)."""
    return report.n >= MIN_REGION_N


# ── Judge calibration (21.3, added 2026-08-24) ─────────────────────────────


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation. Used to answer the one question about the judge that
    a pricing page cannot: does `judge_p_wrong` track reality at all.

    Feature 15 is the judge's PROBABILITY, not its verdict. A judge that is
    smarter than the agent but emits a flat 0.1 or 0.9 makes calibrator B
    collapse to calibrator A, and Tier 2 becomes latency that changes no
    decision. Nothing errors -- which is exactly why this is a hard gate.
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys differ in length")
    n = len(xs)
    if n < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def distinct_deciles(values: Sequence[float]) -> int:
    """How much of [0, 1] the judge's probabilities actually span. A judge
    pinned to two values carries almost no information however well it ranks."""
    return len({min(9, int(v * 10)) for v in values})


# ── 7.7 Inverse-probability weighting ──────────────────────────────────────


def ipw_rate(forced_rows: Sequence[int], rate: float) -> float:
    """Unbiased error rate over the population the gate would have LET THROUGH.

    Human labels arrive only on escalated requests, which are by construction
    the ones the system already thought were risky. Fitting on those describes
    the escalated population, not the autonomous one -- learn only from them and
    the system becomes confident about the path it stopped looking at.

    Each forced-review row stands for `1 / rate` rows that were never reviewed:

        w_i     = 1 / FORCED_REVIEW_RATE
        err_hat = sum(w_i * y_i) / sum(w_i)      over forced-review rows only

    With a single uniform rate this reduces to the plain mean, and that is
    correct rather than a shortcut -- the weights only start to matter when the
    sampling rate varies by stratum, which is the extension this signature
    leaves room for. Borrowed from card-fraud operations, where it has been
    standard for decades.
    """
    if not forced_rows:
        return 0.0
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"forced-review rate must be in (0, 1], got {rate}")
    w = 1.0 / rate
    return sum(w * y for y in forced_rows) / (w * len(forced_rows))


def sampling_bias(all_labelled: Sequence[int], forced_rows: Sequence[int],
                  rate: float) -> dict[str, float]:
    """Both curves, and the gap between them.

    Demo 1 reports the naive rate and the IPW-corrected rate side by side,
    because the gap IS the sampling bias, made visible. If the two sit on top
    of each other the bias was small, and we can say so with evidence instead
    of assertion.
    """
    naive = (sum(all_labelled) / len(all_labelled)) if all_labelled else 0.0
    corrected = ipw_rate(forced_rows, rate)
    return {"naive_rate": naive, "ipw_rate": corrected,
            "bias": naive - corrected, "n_forced": len(forced_rows)}
