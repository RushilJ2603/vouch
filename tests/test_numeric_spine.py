"""Section 5.4 -- the verified fixtures. Computed, not quoted.

If any of these fail, something in the numeric spine moved and every worked
example, demo and published figure is now wrong.
"""
import pytest

from vouch import config, exposure, ledger

# ── 5.4 Wilson lower bound, two-sided, z = 1.96 ────────────────────────────

WILSON_FIXTURES = [
    (0, 0, 0.0000, "defined, not an error"),
    (0, 50, 0.0000, "fifty failures"),
    (3, 3, 0.4385, "perfect record, proves nothing"),
    (4, 4, 0.5101, "one human approval changes essentially nothing"),
    (20, 20, 0.8389, "above p_min, below the ownership gate"),
    (50, 50, 0.9286, "autonomous"),
    (73, 73, 0.9500, ""),
    (75, 75, 0.9513, ""),
    (150, 150, 0.9750, ""),
    (200, 200, 0.9812, "case C"),
    (400, 400, 0.9905, ""),
    (1000, 1000, 0.9962, ""),
    (3098, 3104, 0.9958, "corrected from a published 0.9971"),
    (11398, 11412, 0.9979, "case A; corrected from a published 0.9986"),
]


@pytest.mark.parametrize("clean,total,expected,note", WILSON_FIXTURES)
def test_wilson_lower(clean, total, expected, note):
    assert ledger.wilson_lower(clean, total) == pytest.approx(expected, abs=5e-5)


def test_ownership_gate_is_the_first_autonomy_point():
    """The reliability floor is already clear before 30 observations, so the
    unborrowed ownership gate determines the first eligible row."""
    assert ledger.wilson_lower(20, 20) >= config.P_MIN
    assert ledger.budget(ledger.wilson_lower(29, 29), 720.0, 29) == 0.0
    assert ledger.budget(ledger.wilson_lower(30, 30), 720.0, 30) > 0.0


def test_wilson_is_monotonic_in_n_for_perfect_records():
    values = [ledger.wilson_lower(n, n) for n in range(1, 500)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_wilson_never_exceeds_raw_rate():
    for clean, total in [(3, 3), (50, 50), (99, 100), (11398, 11412)]:
        assert ledger.wilson_lower(clean, total) <= clean / total


# ── 5.2 Derived ceilings ───────────────────────────────────────────────────

def test_combined_multiplier_for_issue_refund():
    assert exposure.multipliers("issue_refund") == pytest.approx(0.36)


@pytest.mark.parametrize("band,expected", [
    ("0-2k", 28.80),
    ("2k-10k", 144.00),
    ("10k-50k", 720.00),
    ("50k+", 2880.00),
])
def test_derived_ceiling(band, expected):
    assert exposure.ceiling("issue_refund", band) == pytest.approx(expected)


def test_small_band_ceiling_sits_below_review_cost():
    """Band 0-2k has ceiling 28.80 < review_cost 40, so every action in it
    passes the day-one exposure test and the ledger never binds. Demo 2 must
    therefore run on 10k-50k or above."""
    assert exposure.ceiling("issue_refund", "0-2k") < config.REVIEW_COST


# ── 5.4 Budget curve, band 10k-50k, ceiling 720.00 -- this is Demo 2 ───────

CEILING_10K_50K = 720.00

# (clean, total, p_lo, budget, note). Every row is a PERFECT record except the
# last, which is worked case A -- 11,398 clean of 11,412, i.e. 14 real failures.
# An earlier benchmark labelled that row "11,412" under a "Clean decisions"
# heading, which reads as a perfect record and is wrong: wilson(11412, 11412) is
# 0.999663, giving budget 718.65 (99.8% of ceiling), not 711.77 (98.9%).
BUDGET_CURVE = [
    (4, 4, 0.5101, 0.00, "supervised (n_own_raw < 30)"),
    (20, 20, 0.8389, 0.00, "supervised (n_own_raw < 30)"),
    (30, 30, 0.8865, 265.93, "first eligible row"),
    (50, 50, 0.9286, 434.60, "60.4% of ceiling"),
    (73, 73, 0.9500, 520.02, "72.2% of ceiling"),
    (75, 75, 0.9513, 525.10, "72.9% of ceiling"),
    (100, 100, 0.9630, 572.02, "79.4% of ceiling"),
    (150, 150, 0.9750, 620.12, "86.1% of ceiling"),
    (200, 200, 0.9812, 644.62, "89.5% of ceiling"),
    (381, 381, 0.9900, 680.07, "94.5% of ceiling"),
    (400, 400, 0.9905, 681.95, "94.7% of ceiling"),
    (1000, 1000, 0.9962, 704.69, "97.9% of ceiling"),
    (11398, 11412, 0.9979, 711.77, "case A -- 14 failures, NOT a perfect record"),
]


@pytest.mark.parametrize("clean,total,p_lo,expected_budget,state", BUDGET_CURVE)
def test_budget_curve(clean, total, p_lo, expected_budget, state):
    actual_p_lo = ledger.wilson_lower(clean, total)
    assert actual_p_lo == pytest.approx(p_lo, abs=5e-5)
    assert ledger.budget(actual_p_lo, CEILING_10K_50K, total) == pytest.approx(expected_budget, abs=0.01)


def test_budget_curve_perfect_rows_really_are_perfect():
    """Regression for the mislabelled row. Deriving p_lo from n is what catches
    a table whose p_lo column was pasted from a different record; checking the
    budget against the STATED p_lo would not have."""
    for clean, total, p_lo, _budget, _note in BUDGET_CURVE[:-1]:
        assert clean == total
        assert ledger.wilson_lower(total, total) == pytest.approx(p_lo, abs=5e-5)


def test_budget_crosses_zero_at_30():
    assert ledger.budget(ledger.wilson_lower(29, 29), CEILING_10K_50K, 29) == 0.0
    assert ledger.budget(ledger.wilson_lower(30, 30), CEILING_10K_50K, 30) > 0.0


def test_budget_reaches_exactly_zero():
    """Three other parts of the design depend on budget = 0 being reachable:
    the collapsing close-call band, the golden fixtures, and worked case B.
    The prior design floored this at review_cost, making zero unreachable."""
    assert ledger.budget(0.99, CEILING_10K_50K, 29) == 0.0     # min-observations gate
    assert ledger.budget(0.80, CEILING_10K_50K, 5000) == 0.0   # below p_min


def test_budget_never_exceeds_ceiling():
    assert ledger.budget(1.0, CEILING_10K_50K, 100_000) == pytest.approx(CEILING_10K_50K)


def test_evidence_has_diminishing_returns():
    """400 decisions buys 94.7% of the ceiling; the next 11,000 buy four points
    more. The curve should flatten, and the design should not pretend otherwise."""
    at_400 = ledger.budget(ledger.wilson_lower(400, 400), CEILING_10K_50K, 400)
    at_11412 = ledger.budget(ledger.wilson_lower(11398, 11412), CEILING_10K_50K, 11412)
    assert at_400 / CEILING_10K_50K == pytest.approx(0.947, abs=0.005)
    assert at_11412 / CEILING_10K_50K == pytest.approx(0.989, abs=0.005)


# ── 9.3 Borrowing, and the safety stop ─────────────────────────────────────

def test_borrowing_changes_nothing_when_there_are_no_neighbours():
    """Rows that borrow nothing must reproduce the published numbers exactly."""
    row = ledger.evaluate("10k-50k", CEILING_10K_50K, 200, 200, 200)
    assert row["p_lo"] == pytest.approx(0.9812, abs=5e-5)


def test_min_observations_gate_defeats_borrowing():
    """40,000 clean small refunds must NOT buy autonomy on a 62,000 refund the
    agent has attempted twice. A gate that constrains borrowing must not itself
    be satisfiable by borrowing, so it counts n_own_raw."""
    row = ledger.evaluate(
        "50k+", 2880.0, own_clean=2, own_total=2, n_own_raw=2,
        neighbours=[("0-2k", 40_000, 40_000)],
    )
    assert row["p_lo"] > config.P_MIN      # borrowing really does lift the bound
    assert row["budget"] == 0.0            # and the gate stops it anyway
    assert row["state"] == "supervised"
