"""Section 7.5 -- calibration, and the average that hides the failure."""
import random

import pytest

from vouch import calibrate


@pytest.fixture
def separable():
    """A scorer with real discrimination but a miscalibrated level."""
    rng = random.Random(7)
    probs, labels = [], []
    for _ in range(1200):
        wrong = rng.random() < 0.20
        base = 0.55 if wrong else 0.12
        probs.append(min(0.99, max(0.01, rng.gauss(base, 0.10))))
        labels.append(1 if wrong else 0)
    return probs, labels


# ── Binning ────────────────────────────────────────────────────────────────

def test_bins_are_equal_frequency_not_equal_width():
    """Predictions cluster near zero; equal-width bins leave most of the range
    empty and the diagram says nothing."""
    probs = [0.01] * 90 + [0.9] * 10
    sizes = [len(b) for b in calibrate.quantile_bins(probs, 10)]
    assert sizes == [10] * 10


def test_binning_handles_fewer_rows_than_bins():
    assert len(calibrate.quantile_bins([0.1, 0.2, 0.3], 10)) == 3
    assert calibrate.quantile_bins([], 10) == []


# ── Metrics ────────────────────────────────────────────────────────────────

def test_a_perfectly_calibrated_scorer_has_near_zero_ece():
    rng = random.Random(3)
    probs, labels = [], []
    for _ in range(4000):
        p = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        probs.append(p)
        labels.append(1 if rng.random() < p else 0)
    report = calibrate.evaluate(probs, labels)
    assert report.ece < 0.03
    assert report.brier < 0.25


def test_an_overconfident_scorer_is_caught():
    """Says 5%, reality is 40%."""
    probs = [0.05] * 500
    labels = [1 if i % 10 < 4 else 0 for i in range(500)]
    report = calibrate.evaluate(probs, labels)
    assert report.ece == pytest.approx(0.35, abs=0.02)
    assert report.mce >= report.ece - 1e-9   # equal here; ECE sums, so it drifts


def test_every_bin_carries_a_wilson_interval():
    """With ~150 rows per bin the intervals are wide, and the design plots
    them rather than drawing a smooth line through sparse data."""
    rng = random.Random(11)
    probs = [rng.random() for _ in range(1500)]
    labels = [1 if rng.random() < p else 0 for p in probs]
    for b in calibrate.evaluate(probs, labels).bins:
        assert 0.0 <= b.lo <= b.observed <= b.hi <= 1.0
        assert b.hi > b.lo


def test_empty_input_does_not_explode():
    report = calibrate.evaluate([], [])
    assert report.n == 0 and report.ece == 0.0 and report.bins == []


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        calibrate.evaluate([0.1, 0.2], [1])


# ── Platt ──────────────────────────────────────────────────────────────────

def test_platt_improves_calibration_without_destroying_discrimination(separable):
    probs, labels = separable
    split = 800
    before = calibrate.evaluate(probs[split:], labels[split:])
    platt = calibrate.fit_platt(probs[:split], labels[:split])
    after = calibrate.evaluate(platt.apply_all(probs[split:]), labels[split:])
    assert after.ece < before.ece
    assert after.brier <= before.brier + 1e-9


def test_platt_is_monotonic():
    """It rescales confidence; it must not reorder responses."""
    platt = calibrate.Platt(a=1.4, b=-0.3)
    scores = [platt.apply(p) for p in [0.01, 0.1, 0.3, 0.6, 0.9, 0.99]]
    assert all(b > a for a, b in zip(scores, scores[1:]))


def test_platt_refuses_an_empty_split():
    with pytest.raises(ValueError):
        calibrate.fit_platt([], [])


# ── The reason per-region reporting is mandatory ───────────────────────────

def test_a_good_global_ece_can_hide_a_broken_close_call_band():
    """1,380 act turns at ECE ~0.011 and 95 close_call turns saying 12% where
    reality is 34%. The global number lands inside the 0.05 pass condition and
    the system ships under-escalating exactly the decisions that were too
    close to call."""
    probs = [0.011] * 1380 + [0.12] * 95
    labels = [1 if i % 91 == 0 else 0 for i in range(1380)] + \
             [1 if i % 3 == 0 else 0 for i in range(95)]
    regions = ["act"] * 1380 + ["close_call"] * 95

    reports = calibrate.evaluate_by_region(probs, labels, regions)

    assert reports["global"].ece < 0.05                 # passes the global gate
    assert reports["close_call"].ece > 0.15             # and is badly broken here
    assert reports["act"].ece < 0.02


def test_a_thin_region_reports_insufficient_data_rather_than_an_alarm():
    """Every alarm in this system carries a minimum-sample floor."""
    reports = calibrate.evaluate_by_region([0.1] * 20, [0] * 20, ["close_call"] * 20)
    assert not calibrate.region_has_power(reports["close_call"])
    assert calibrate.region_has_power(calibrate.evaluate([0.1] * 150, [0] * 150))


def test_all_three_regions_are_always_reported():
    reports = calibrate.evaluate_by_region([0.1], [0], ["act"])
    assert set(reports) == {"global", "act", "close_call", "escalate"}
