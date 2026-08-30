"""Section 18 -- calibrator governance.

`calibrator_version` sits inside the fingerprint, which is correct: a refit
changes what `P(wrong) = 0.02` MEANS. Taken naively that means a routine refit
resets every trust row in the system -- months of evidence gone because
somebody improved the model. This module exists so that cannot happen by
accident, and these tests exist so it cannot start happening again.
"""

from vouch.worker import promote
from vouch.worker.promote import Candidate, Migration, Path

RUNNING_SKLEARN = "1.9.0"


def cand(ece_g=0.030, ece_cc=0.050, n_g=1000, n_cc=200, sk=RUNNING_SKLEARN, v="cal-v2"):
    return Candidate(version=v, ece_global=ece_g, ece_close_call=ece_cc,
                     n_global=n_g, n_close_call=n_cc, sklearn_version=sk,
                     artifact_sha256="deadbeef")


PRODUCTION = cand(ece_g=0.040, ece_cc=0.070, v="cal-v1")


# ── Path A: the held-out gate ──────────────────────────────────────────────

def test_better_on_both_metrics_auto_promotes():
    assert promote.decide_path(cand(), PRODUCTION, RUNNING_SKLEARN) is Path.A


def test_better_globally_but_worse_on_close_calls_does_not():
    """A candidate that improves the average while degrading the close-call
    band is worse exactly where a better estimate changes a decision."""
    worse_where_it_counts = cand(ece_g=0.010, ece_cc=0.090)
    assert promote.decide_path(worse_where_it_counts, PRODUCTION, RUNNING_SKLEARN) is Path.B


def test_a_thin_close_call_region_cannot_auto_promote():
    """Every gate in this system carries a minimum-sample floor."""
    assert not promote.path_a_passes(cand(n_cc=99), PRODUCTION)
    assert promote.path_a_passes(cand(n_cc=100), PRODUCTION)


def test_a_thin_global_sample_cannot_auto_promote():
    assert not promote.path_a_passes(cand(n_g=99), PRODUCTION)


# ── The hard block ─────────────────────────────────────────────────────────

def test_an_sklearn_mismatch_blocks_outright():
    """A joblib fitted under a different scikit-learn loads WITHOUT ERROR and
    scores differently. That is silent train/serve skew, and it would corrupt
    every subsequent trust update."""
    assert promote.decide_path(cand(sk="1.8.2"), PRODUCTION, RUNNING_SKLEARN) is Path.BLOCKED


def test_the_block_outranks_a_perfect_candidate():
    perfect = cand(ece_g=0.0, ece_cc=0.0, sk="1.7.0")
    assert promote.decide_path(perfect, PRODUCTION, RUNNING_SKLEARN) is Path.BLOCKED


# ── Path B: shadow scoring, and why it exists ──────────────────────────────

def test_shadow_needs_both_time_and_outcomes():
    """30 days OR 500 closed outcomes, whichever comes LATER."""
    assert not promote.shadow_promotion_ready(29, 5000, True)
    assert not promote.shadow_promotion_ready(365, 499, True)
    assert promote.shadow_promotion_ready(30, 500, True)


def test_shadow_never_promotes_a_worse_challenger():
    assert not promote.shadow_promotion_ready(365, 5000, challenger_better=False)


def test_path_b_is_the_route_for_a_post_shift_refit():
    """A calibrator refitted after a genuine distribution shift often scores
    WORSE on a frozen test set drawn from before the shift, while being
    materially better on live traffic. A single gate blocks exactly the model
    you need most, and it does so when the system is already under stress."""
    post_shift = cand(ece_g=0.055, ece_cc=0.075)      # worse on the old test set
    assert promote.decide_path(post_shift, PRODUCTION, RUNNING_SKLEARN) is Path.B


# ── Migration: the footgun this section exists to disarm ───────────────────

def test_close_agreement_carries_evidence_intact():
    """If the new calibrator agrees with the old to within 0.02 ECE in the
    region that decides things, it is not a different system in any way the
    arithmetic cares about."""
    assert promote.migration_for(0.019) is Migration.CARRY


def test_moderate_divergence_halves_evidence():
    assert promote.migration_for(0.021) is Migration.HALVE
    assert promote.migration_for(0.050) is Migration.HALVE


def test_wide_divergence_resets():
    assert promote.migration_for(0.051) is Migration.RESET


def test_carry_leaves_the_row_untouched():
    row = {"n_total": 400.0, "n_clean": 400.0, "n_own_raw": 400, "budget": 681.95}
    out = promote.migrate(row, 0.01)
    assert out["n_total"] == 400.0 and out["n_clean"] == 400.0
    assert out["migration"] == "carry"


def test_halve_halves_the_counts_but_keeps_the_row():
    row = {"n_total": 400.0, "n_clean": 400.0, "n_own_raw": 400}
    out = promote.migrate(row, 0.03)
    assert out["n_total"] == 200.0 and out["n_clean"] == 200.0
    assert out["migration"] == "halve"


def test_reset_returns_the_row_to_supervised():
    row = {"n_total": 40_000.0, "n_clean": 39_990.0, "n_own_raw": 40_000,
           "budget": 700.0, "state": "autonomous"}
    out = promote.migrate(row, 0.20)
    assert out["budget"] == 0.0 and out["state"] == "supervised"
    assert out["n_total"] == 0.0 and out["n_own_raw"] == 0


def test_a_routine_refit_does_not_wipe_the_system():
    """The whole point of 18. Without it, improving the model costs every trust
    row in the system its entire history."""
    row = {"n_total": 400.0, "n_clean": 400.0, "n_own_raw": 400}
    assert promote.migrate(row, 0.005)["n_total"] == 400.0


def test_migration_never_increases_evidence():
    """16's invariant: nothing here may raise autonomy."""
    row = {"n_total": 400.0, "n_clean": 400.0, "n_own_raw": 400}
    for delta in (0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 1.0):
        assert promote.migrate(row, delta)["n_total"] <= 400.0
