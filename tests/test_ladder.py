"""Section 10.3 -- the seven rungs, measured in human attention."""

from vouch.gate import Verdict
from vouch.ladder import Rung, Signals, next_after_failed_regenerate, rung_for_verdict, select_rung

CLEAN = Signals(pii_score=0.0, secret_score=0.0, injection_score=0.0, verify_fail_frac=0.0)


def sig(**kw) -> Signals:
    base = dict(pii_score=0.0, secret_score=0.0, injection_score=0.0,
                verify_fail_frac=0.0, span_is_localised=False)
    base.update(kw)
    return Signals(**base)


# ── The failure type picks the family ──────────────────────────────────────

def test_localised_pii_is_redacted_not_escalated():
    """One span masked, the rest ships. No human time at all."""
    assert select_rung(sig(pii_score=0.9, span_is_localised=True), 10, 100, False) is Rung.REDACT


def test_diffuse_pii_escalates():
    assert select_rung(sig(pii_score=0.9, span_is_localised=False), 10, 100, False) is Rung.ESCALATE


def test_injection_blocks_and_offers_no_human():
    """Rung 6 refuses. There is nothing for a person to decide."""
    assert select_rung(sig(injection_score=0.9), 10, 100, False) is Rung.BLOCK


def test_pii_is_checked_before_injection_as_10_3_specifies():
    """This encodes the document's ORDER, which is worth questioning.

    10.3 tests pii/secret first and injection second, so a response that is
    both prompt-injected AND carries localised PII is REDACTED and shipped,
    not blocked. Redaction removes the leaked span; it does nothing about the
    injection that produced the response. Reversing the two lines would block
    it instead.

    The implementation follows the document. Raised as an open question rather
    than silently changed -- see 27.
    """
    both = sig(pii_score=0.9, span_is_localised=True, injection_score=0.9)
    assert select_rung(both, 10, 100, False) is Rung.REDACT


def test_a_contradicted_claim_regenerates_when_severity_is_low():
    assert select_rung(sig(verify_fail_frac=0.5), 100, 100, False) is Rung.REGENERATE


def test_a_contradicted_claim_escalates_when_severity_is_high():
    assert select_rung(sig(verify_fail_frac=0.5), 400, 100, False) is Rung.ESCALATE


# ── Rung 4 is the one most systems skip ────────────────────────────────────

def test_a_side_effect_gets_advisory_not_escalation():
    """The agent does all the work and stops one inch short of executing. A
    human approves a finished thing in ~15 s instead of doing it in ten
    minutes. This rung is what decides whether the review queue survives."""
    assert select_rung(CLEAN, 100, 100, has_side_effect=True) is Rung.ADVISORY


def test_a_costly_side_effect_still_escalates():
    assert select_rung(CLEAN, 400, 100, has_side_effect=True) is Rung.ESCALATE


def test_a_clean_read_only_answer_is_merely_annotated():
    assert select_rung(CLEAN, 100, 100, has_side_effect=False) is Rung.ANNOTATE


# ── Unavailable sensors fail closed here too ───────────────────────────────

def test_an_unavailable_sensor_is_treated_as_present_and_dangerous():
    """Matching 10.2. A None score must never read as 0.0."""
    assert select_rung(sig(pii_score=None, span_is_localised=False), 1, 100, False) is Rung.ESCALATE
    assert select_rung(sig(injection_score=None), 1, 100, False) is Rung.BLOCK


# ── The verdict outranks the ladder at both ends ───────────────────────────

def test_a_block_verdict_is_rung_six_whatever_the_ladder_thinks():
    assert rung_for_verdict(Verdict.BLOCK, CLEAN, 1, 1e9, False) is Rung.BLOCK


def test_an_act_verdict_never_spends_human_attention():
    """The gate has already decided the action is affordable. Spending a
    person on it is exactly the waste the whole design exists to remove."""
    assert rung_for_verdict(Verdict.ACT, CLEAN, 1, 1e9, True) is Rung.PASS


def test_escalate_verdict_defers_to_the_ladder():
    rung = rung_for_verdict(Verdict.ESCALATE, sig(pii_score=0.9, span_is_localised=True),
                            10, 100, False)
    assert rung is Rung.REDACT


# ── The regenerate cap ─────────────────────────────────────────────────────

def test_regeneration_is_capped_then_climbs():
    """Uncapped retries are the cheapest way to burn money and latency at once."""
    assert next_after_failed_regenerate(0) is Rung.REGENERATE
    assert next_after_failed_regenerate(1) is Rung.REGENERATE
    assert next_after_failed_regenerate(2) is Rung.ESCALATE


# ── Rung ordering is the human-attention ordering ──────────────────────────

def test_rungs_are_ordered_and_only_two_cost_human_time():
    from vouch.ladder import HUMAN_TIME
    assert [r.value for r in Rung] == list(range(7))
    costly = [r for r, t in HUMAN_TIME.items() if t != "none"]
    assert set(costly) == {Rung.ADVISORY, Rung.ESCALATE}


def test_zero_budget_does_not_divide_by_zero():
    assert select_rung(CLEAN, 100.0, 0.0, True) is Rung.ESCALATE
