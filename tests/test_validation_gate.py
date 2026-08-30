"""Section 21.3 -- proving each hard block actually blocks.

A gate nobody has watched fail is a gate nobody knows works. Every check here
is fed a corpus that should be rejected, and asserted to reject it.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_corpus.py"
spec = importlib.util.spec_from_file_location("validate_corpus", SCRIPT)
gate = importlib.util.module_from_spec(spec)
sys.modules["validate_corpus"] = gate
spec.loader.exec_module(gate)


@pytest.fixture(autouse=True)
def clear_results():
    gate.results.clear()
    yield
    gate.results.clear()


def status_of(check: str) -> str:
    return next(s for c, s, _ in gate.results if c == check)


def _turn(turn_id, amount_paise, outcome="clean", region="act"):
    return {"turn_id": turn_id, "outcome": outcome, "region": region,
            "action": {"type": "issue_refund", "amount_paise": amount_paise}}


# ── 4. Label balance is the agent acceptance test ──────────────────────────

def test_an_agent_that_never_errs_is_rejected():
    """Below 3% there is nothing to calibrate against."""
    gate.check_label_balance([_turn(i, 100000) for i in range(500)])
    assert status_of("label-balance") == gate.FAIL


def test_an_agent_that_errs_constantly_is_rejected():
    """Above 25% it is too weak to be representative."""
    rows = [_turn(i, 100000, "wrong" if i % 2 else "clean") for i in range(500)]
    gate.check_label_balance(rows)
    assert status_of("label-balance") == gate.FAIL


def test_an_error_rate_inside_the_band_passes():
    rows = [_turn(i, 100000, "wrong" if i % 10 == 0 else "clean") for i in range(500)]
    gate.check_label_balance(rows)
    assert status_of("label-balance") == gate.PASS


# ── 5. Band coverage ───────────────────────────────────────────────────────

def test_a_corpus_concentrated_in_one_band_is_rejected():
    gate.check_band_coverage([_turn(i, 50_000) for i in range(1000)])   # all Rs500
    assert status_of("band-coverage") == gate.FAIL


def test_all_four_bands_populated_passes():
    rows = []
    for i in range(160):
        for paise in (100_000, 500_000, 2_500_000, 10_000_000):
            rows.append(_turn(f"{i}-{paise}", paise))
    gate.check_band_coverage(rows)
    assert status_of("band-coverage") == gate.PASS


# ── 6. Region coverage ─────────────────────────────────────────────────────

def test_too_few_close_calls_is_rejected():
    """Per-region ECE needs the samples to exist."""
    rows = [_turn(i, 100000, region="act") for i in range(1000)]
    rows += [_turn(f"c{i}", 100000, region="close_call") for i in range(40)]
    gate.check_region_coverage(rows)
    assert status_of("region-coverage") == gate.FAIL


# ── 8. Judge calibration -- the silent killer ──────────────────────────────

def _judged(probs, truths):
    corpus = [_turn(i, 100000, outcome=t) for i, t in enumerate(truths)]
    judged = [{"turn_id": i, "p_wrong": p, "verdict": "ok", "model": "glm-4.7",
               "thinking": "disabled"} for i, p in enumerate(probs)]
    return judged, corpus


def test_a_flat_judge_is_rejected():
    """A judge that always says 0.1 makes calibrator B collapse to calibrator
    A. Tier 2 becomes 400 ms of latency that changes no decision, and nothing
    anywhere errors."""
    truths = ["wrong" if i % 5 == 0 else "clean" for i in range(300)]
    judged, corpus = _judged([0.1] * 300, truths)
    gate.check_judge_calibration(judged, corpus)
    assert status_of("judge-calibration") == gate.FAIL


def test_a_judge_that_ranks_backwards_is_rejected():
    truths = ["wrong" if i % 5 == 0 else "clean" for i in range(300)]
    probs = [0.05 if t == "wrong" else 0.9 for t in truths]
    judged, corpus = _judged(probs, truths)
    gate.check_judge_calibration(judged, corpus)
    assert status_of("judge-calibration") == gate.FAIL


def test_a_judge_with_real_signal_passes():
    truths = ["wrong" if i % 5 == 0 else "clean" for i in range(300)]
    probs = [0.1 + 0.7 * (t == "wrong") + 0.001 * i for i, t in enumerate(truths)]
    judged, corpus = _judged(probs, truths)
    gate.check_judge_calibration(judged, corpus)
    assert status_of("judge-calibration") == gate.PASS


# ── 9. Schema adherence ────────────────────────────────────────────────────

def test_malformed_judge_output_is_rejected():
    judged = [{"turn_id": i, "p_wrong": "high", "verdict": "ok"} for i in range(50)]
    gate.check_judge_schema(judged)
    assert status_of("judge-schema") == gate.FAIL


def test_a_probability_outside_zero_one_is_a_violation():
    judged = [{"turn_id": i, "p_wrong": 0.2, "verdict": "ok"} for i in range(48)]
    judged += [{"turn_id": 98, "p_wrong": 1.4, "verdict": "ok"},
               {"turn_id": 99, "p_wrong": -0.2, "verdict": "ok"}]
    gate.check_judge_schema(judged)
    assert status_of("judge-schema") == gate.FAIL


def test_one_violation_is_tolerated():
    judged = [{"turn_id": i, "p_wrong": 0.2, "verdict": "ok"} for i in range(49)]
    judged.append({"turn_id": 49, "p_wrong": None, "verdict": "ok"})
    gate.check_judge_schema(judged)
    assert status_of("judge-schema") == gate.PASS


# ── 10. Mode consistency guards the one-model rule ─────────────────────────

def test_a_corpus_that_mixes_judge_modes_is_rejected():
    """Feature 15 fitted from thinking=disabled and served from
    thinking=enabled is train/serve skew wearing a disguise."""
    judged = [{"turn_id": i, "p_wrong": 0.2, "verdict": "ok", "model": "glm-4.7",
               "thinking": "disabled" if i % 2 else "enabled"} for i in range(50)]
    gate.check_judge_mode_consistency(judged)
    assert status_of("judge-mode") == gate.FAIL


def test_a_corpus_from_one_judge_configuration_passes():
    judged = [{"turn_id": i, "p_wrong": 0.2, "verdict": "ok", "model": "glm-4.7",
               "thinking": "disabled"} for i in range(50)]
    gate.check_judge_mode_consistency(judged)
    assert status_of("judge-mode") == gate.PASS


# ── 1. Point-in-time ───────────────────────────────────────────────────────

def test_point_in_time_passes_on_the_current_tree():
    gate.check_point_in_time()
    assert status_of("point-in-time") == gate.PASS
