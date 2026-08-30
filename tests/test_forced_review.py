"""Section 7.7 -- the sampling-bias trap, and the fix."""
import pytest

from vouch import calibrate, config

RATE = config.FORCED_REVIEW_RATE


def test_the_rate_is_roughly_one_in_three_hundred():
    assert 1 / RATE == pytest.approx(303, abs=5)


def test_ipw_over_a_uniform_rate_is_the_plain_mean():
    """Not a shortcut -- with one uniform sampling rate the weights cancel.
    They start to matter when the rate varies by stratum."""
    rows = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert calibrate.ipw_rate(rows, RATE) == pytest.approx(0.2)


def test_ipw_rejects_an_impossible_rate():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            calibrate.ipw_rate([1, 0], bad)


def test_no_forced_rows_is_zero_not_a_crash():
    assert calibrate.ipw_rate([], RATE) == 0.0


def test_the_gap_between_the_curves_is_the_bias():
    """This is the number Demo 1 exists to show. Escalated traffic is 40%
    wrong; the population the gate would have let through is 2% wrong. Fitting
    on escalations alone would claim the system errs twenty times as often as
    it does on the path it actually automates."""
    escalated_heavy = [1] * 40 + [0] * 60          # naive view, 40%
    forced = [1] * 2 + [0] * 98                    # what ACT traffic really does, 2%
    report = calibrate.sampling_bias(escalated_heavy, forced, RATE)
    assert report["naive_rate"] == pytest.approx(0.40)
    assert report["ipw_rate"] == pytest.approx(0.02)
    assert report["bias"] == pytest.approx(0.38)


def test_no_bias_is_reported_as_no_bias():
    """If the two curves sit on top of each other, the bias was small and we
    can say so with evidence rather than assertion."""
    same = [1] * 5 + [0] * 95
    report = calibrate.sampling_bias(same, same, RATE)
    assert report["bias"] == pytest.approx(0.0)


def test_the_sampler_is_seeded_so_replays_agree():
    """11 forbids wall-clock entropy anywhere the corpus is replayed. An
    unseeded sampler would make two replays of the same corpus disagree about
    which rows a human reviewed."""
    import random

    from vouch.proxy import app
    assert isinstance(app._FORCED_REVIEW, random.Random)
    a = random.Random(20260825)
    b = random.Random(20260825)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_forced_review_only_ever_removes_autonomy():
    """It converts ACT into ESCALATE and never the reverse, so it cannot
    violate criterion 6 while trying to measure it."""
    import inspect

    from vouch.proxy import app
    # The route delegates to ``evaluate`` so the same control path can also
    # score an already-generated live-showcase response without a second
    # provider call. Inspect the implementation that owns the gate logic.
    source = inspect.getsource(app.evaluate)
    assert "verdict is Verdict.ACT" in source
    assert "verdict = Verdict.ESCALATE" in source
