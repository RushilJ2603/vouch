"""Appendix A -- the feature contract, and the section 11 leak it can hide."""
import pytest

from vouch import features
from vouch.features import Missing


def test_feature_order_is_exactly_fifteen_in_order():
    assert len(features.FEATURE_ORDER) == 15
    assert features.FEATURE_ORDER[0] == "verify_fail_frac"
    assert features.FEATURE_ORDER[13] == "secret_score"
    assert features.FEATURE_ORDER[14] == "judge_p_wrong"


def test_model_a_never_sees_the_judge():
    """Model A is used when Tier 2 has NOT fired, so feature 15 must be absent
    from it entirely -- not present and imputed."""
    assert "judge_p_wrong" not in features.MODEL_A_FEATURES
    assert "judge_p_wrong" in features.MODEL_B_FEATURES
    assert len(features.MODEL_A_FEATURES) == 14


def test_as_of_is_required_and_positional():
    """The enforcement mechanism for section 11 is the signature itself: a
    caller with no reference time cannot compile."""
    with pytest.raises(TypeError):
        features.assemble(signals={})           # no as_of


# ── Feature 8, the point-in-time trap ──────────────────────────────────────

def test_length_z_uses_only_rows_before_as_of():
    """A response that was long FOR ITS OWN DAY must score as long, even if
    later days shift the mean past it."""
    # Early window centred near 100 with real spread; later window is far
    # longer, which is exactly what drags the mean past the row being scored.
    early = [(float(t), 100 + (t % 7) * 5) for t in range(40)]
    later = [(float(t), 900 + (t % 7) * 5) for t in range(100, 140)]

    honest, reason = features.length_z(300, early + later, as_of=50.0)
    leaked, _ = features.length_z(300, early + later, as_of=1000.0)

    assert reason is Missing.VALUE
    assert honest > 0          # unusually long against its own past
    assert leaked < 0          # and typical-to-short once the future leaks in
    assert honest != leaked


def test_length_z_is_unavailable_below_thirty_rows():
    value, reason = features.length_z(300, [(float(t), 100) for t in range(29)], as_of=50.0)
    assert value is None and reason is Missing.UNAVAILABLE


def test_length_z_survives_zero_variance():
    history = [(float(t), 100) for t in range(40)]
    value, reason = features.length_z(100, history, as_of=50.0)
    assert value == 0.0 and reason is Missing.VALUE


# ── 7.6 missing semantics ──────────────────────────────────────────────────

def test_unavailable_sets_the_companion_indicator():
    out = features.assemble(1.0, {"pii_score": (None, Missing.UNAVAILABLE)})
    assert out["pii_score"] is None
    assert out["pii_score_unavailable"] == 1


def test_measured_value_clears_the_indicator():
    out = features.assemble(1.0, {"pii_score": (0.2, Missing.VALUE)})
    assert out["pii_score"] == 0.2
    assert out["pii_score_unavailable"] == 0


def test_not_supported_is_not_the_same_as_unavailable():
    """A deployment with no logprobs must NOT look like one whose encoders are
    failing on every request. Collapsing both into one _missing flag would
    widen every estimate for a month and push traffic up the ladder for no
    reason at all."""
    absent = features.assemble(1.0, {"logprob_mean": (None, Missing.NOT_SUPPORTED)})
    broken = features.assemble(1.0, {"logprob_mean": (None, Missing.UNAVAILABLE)})
    assert absent["logprob_mean"] is broken["logprob_mean"] is None
    assert absent["logprob_mean_unavailable"] == 0
    assert broken["logprob_mean_unavailable"] == 1


def test_features_that_cannot_be_unavailable_have_no_indicator():
    out = features.assemble(1.0, {"secret_score": (0.0, Missing.VALUE)})
    assert "secret_score_unavailable" not in out
    assert "hedge_density_unavailable" not in out


def test_capability_tier_none_drops_logprobs_entirely():
    out = features.assemble(1.0, {"logprob_mean": (-0.5, Missing.VALUE)}, capability_tier="none")
    assert out["logprob_mean"] is None
    assert "logprob_mean_unavailable" not in out


def test_unknown_capability_tier_is_rejected():
    with pytest.raises(ValueError):
        features.assemble(1.0, {}, capability_tier="guess")


def test_missing_signal_defaults_to_unavailable_not_to_zero():
    """Section 16: a signal that failed must never silently default to 0.0,
    which reads as 'measured, and clean'."""
    out = features.assemble(1.0, {})
    assert out["verify_fail_frac"] is None
    assert out["verify_fail_frac_unavailable"] == 1


# ── Vectorisation ──────────────────────────────────────────────────────────

def test_vector_and_names_line_up():
    out = features.assemble(1.0, {n: (0.5, Missing.VALUE) for n in features.FEATURE_ORDER})
    vec = features.to_vector(out, features.MODEL_B_FEATURES)
    names = features.vector_names(features.MODEL_B_FEATURES)
    assert len(vec) == len(names)
    assert len(vec) == 15 + len(features.CAN_BE_UNAVAILABLE)


def test_as_of_is_carried_into_the_record():
    """So a replay can assert which reference time produced the row."""
    assert features.assemble(1234.5, {})["_as_of"] == 1234.5
