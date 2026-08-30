"""Section 10 -- the gate, and the five worked cases of 5.5.

These are the decision table. If one changes, a worked example in the
product surface, proposal or video is now telling a different story.
"""
import pytest

from vouch import exposure, gate, ledger
from vouch.gate import Verdict

CLEAN = gate.InvariantResult(violated=False)
VIOLATED = gate.InvariantResult(violated=True, failed=["no_pii_egress"])


def _row(band, clean, total, n_own_raw=None):
    c = exposure.ceiling("issue_refund", band)
    p_lo = ledger.wilson_lower(clean, total)
    return ledger.budget(p_lo, c, total if n_own_raw is None else n_own_raw)


# ── 5.5 The five worked cases ──────────────────────────────────────────────

def test_case_a_small_refund_long_record_acts():
    """1,240 refund, band 0-2k, 11,398 clean of 11,412."""
    budget = _row("0-2k", 11398, 11412)
    exp = exposure.exposure("issue_refund", 1240)
    assert budget == pytest.approx(28.47, abs=0.01)
    assert exp == pytest.approx(446.40)
    assert 0.008 * exp == pytest.approx(3.57, abs=0.01)
    assert gate.decide(0.008, exp, budget, CLEAN) is Verdict.ACT


def test_case_b_large_refund_no_record_escalates():
    """62,000 refund, band 50k+, 3 clean of 3. Budget is zero because
    n_own_raw = 3 < 30, and a confident p_wrong cannot buy past that."""
    budget = _row("50k+", 3, 3)
    exp = exposure.exposure("issue_refund", 62000)
    assert budget == 0.0
    assert exp == pytest.approx(22_320.00)
    assert 0.008 * exp == pytest.approx(178.56, abs=0.01)
    assert gate.decide(0.008, exp, budget, CLEAN) is Verdict.ESCALATE


def test_case_c_close_call_checks_harder_then_acts():
    """7,000 refund, band 2k-10k, 200 clean of 200. Lands in the close-call
    band, Tier 2 sharpens the estimate, re-decide at k = 1.0 resolves to ACT."""
    budget = _row("2k-10k", 200, 200)
    exp = exposure.exposure("issue_refund", 7000)
    assert budget == pytest.approx(128.92, abs=0.01)
    assert exp == pytest.approx(2_520.00)

    assert 0.055 * exp == pytest.approx(138.60, abs=0.01)
    assert budget * 2.0 == pytest.approx(257.85, abs=0.01)
    assert gate.decide(0.055, exp, budget, CLEAN) is Verdict.CHECK_HARDER

    # C' -- after Tier 2
    assert 0.015 * exp == pytest.approx(37.80, abs=0.01)
    assert gate.redecide_after_tier2(0.015, exp, budget, CLEAN) is Verdict.ACT


def test_case_d_below_review_cost_acts_on_day_one():
    """80 refund, no track record at all. Exposure 28.80 is below the 40 it
    costs a person to look at it, so it goes through regardless."""
    exp = exposure.exposure("issue_refund", 80)
    assert exp == pytest.approx(28.80)
    for p_wrong in (0.0, 0.5, 1.0):
        assert gate.decide(p_wrong, exp, 0.0, CLEAN) is Verdict.ACT


# ── 10.1 The day-one floor tests exposure, not expected loss ───────────────

def test_day_one_floor_rejects_the_confident_large_payout():
    """The pathology the exposure test removes: a 62,000 refund with a
    confident p_wrong = 0.0015 has an expected loss of 33.48, below the 40
    review cost. Testing expected loss would ACT. Testing exposure escalates."""
    exp = exposure.exposure("issue_refund", 62000)
    assert 0.0015 * exp == pytest.approx(33.48, abs=0.01)
    assert 0.0015 * exp < 40.0
    assert gate.decide(0.0015, exp, 0.0, CLEAN) is Verdict.ESCALATE


# ── 10.1 The close-call band collapses at budget = 0 ───────────────────────

def test_close_call_band_collapses_when_budget_is_zero():
    """budget * k = 0, so nothing can land in the band and Tier 2 never fires.
    Falls out of the arithmetic; no special case."""
    for p_wrong in (1e-9, 0.001, 0.5):
        assert gate.decide(p_wrong, 5000.0, 0.0, CLEAN) is Verdict.ESCALATE


# ── 10.2 Invariants -- the fixed path ──────────────────────────────────────

def test_invariants_block_regardless_of_record():
    assert gate.decide(0.0, 1.0, 1e9, VIOLATED) is Verdict.BLOCK


def test_unavailable_sensor_fails_closed():
    """If pii_score could not be computed, no_pii_egress is treated as
    violated, not as passed. Absence of evidence is not evidence."""
    result = gate.check_invariants(
        {"pii_score": None, "injection_score": 0.1, "secret_score": 0.1},
        {"retrieval_scope": 1, "user_scope": 1, "amount": 100, "hard_limit": 200000},
    )
    assert result.violated and "no_pii_egress" in result.failed


def test_amount_above_hard_limit_is_blocked():
    result = gate.check_invariants(
        {"pii_score": 0.0, "injection_score": 0.0, "secret_score": 0.0},
        {"retrieval_scope": 1, "user_scope": 1, "amount": 250000, "hard_limit": 200000},
    )
    assert result.violated and "within_hard_amount_limit" in result.failed


def test_clean_signals_pass():
    result = gate.check_invariants(
        {"pii_score": 0.1, "injection_score": 0.1, "secret_score": 0.1},
        {"retrieval_scope": 1, "user_scope": 2, "amount": 1240, "hard_limit": 200000},
    )
    assert not result.violated and result.failed == []


# ── 8 Fail closed on an unregistered action ────────────────────────────────

def test_unregistered_action_has_no_price():
    """An action not in actions.yaml has no exposure and therefore no price,
    so the caller must BLOCK. This is the single most likely cause of
    'Vouch is refusing everything' during integration."""
    with pytest.raises(KeyError):
        exposure.exposure("wire_transfer", 5000)


# ── 10.1 Ordering of the verdict ladder ────────────────────────────────────

@pytest.mark.parametrize("p_wrong,expected", [
    (0.001, Verdict.ACT),           # EL 2.52  <= budget 128.92
    (0.050, Verdict.ACT),           # EL 126.00 <= budget
    (0.052, Verdict.CHECK_HARDER),  # EL 131.04 in (128.92, 257.84]
    (0.102, Verdict.CHECK_HARDER),  # EL 257.04 still inside
    (0.103, Verdict.ESCALATE),      # EL 259.56 past budget * k
])
def test_decision_boundaries(p_wrong, expected):
    assert gate.decide(p_wrong, 2520.0, 128.92, CLEAN) is expected
