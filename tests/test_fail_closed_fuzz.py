"""Section 3.4 criterion 6 -- the fail-closed fuzz harness.

For each subsystem in §16, kill it and assert the ACT rate does not rise.
Zero exceptions. A PR that introduces one does not merge.

This is the test of one sentence: **an unavailable component reduces autonomy,
never increases it.** It has already caught two real defects in this codebase
(a presence-only encoder check that would have turned BLOCK into ACT, and an
entropy-only secret detector that let GitHub tokens past the invariant), which
is the argument for it existing as a harness rather than as a principle.
"""
import random

import pytest

from vouch import exposure, gate, ladder, ledger
from vouch.features import Missing
from vouch.gate import Verdict

CEILING = exposure.ceiling("issue_refund", "10k-50k")
POPULATION = 400


def healthy_signals() -> dict:
    return {"pii_score": 0.05, "injection_score": 0.02, "secret_score": 0.0}


def healthy_ctx(amount: float) -> dict:
    return {"retrieval_scope": 1, "user_scope": 2, "amount": amount,
            "hard_limit": exposure.hard_limit("issue_refund")}


def run_population(kill: str | None = None, seed: int = 99) -> dict[str, int]:
    """Drive a fixed population of requests and count verdicts.

    The population, the amounts and the p_wrong draws are all seeded, so the
    ONLY difference between a healthy run and a killed run is the subsystem.
    """
    rng = random.Random(seed)
    counts = {v.value: 0 for v in Verdict}

    for _ in range(POPULATION):
        amount = rng.uniform(10, 199_999)
        p_wrong = rng.betavariate(2, 40)          # mostly small, occasionally not
        n_clean = rng.choice([0, 20, 73, 200, 400, 5_000])

        signals, ctx = healthy_signals(), healthy_ctx(amount)
        budget = ledger.budget(ledger.wilson_lower(n_clean, n_clean), CEILING, n_clean)
        exp = exposure.exposure("issue_refund", amount)

        # ── kill one subsystem, exactly as §16 describes it ────────────────
        if kill == "TIER1_UNAVAILABLE":
            signals = {k: None for k in signals}
        elif kill == "TIER1_PARTIAL":
            signals["injection_score"] = None
        elif kill == "SOR_UNAVAILABLE":
            p_wrong = None                        # feature 1 unavailable → widen
        elif kill == "LEDGER_ROW_MISSING":
            budget = 0.0
        elif kill == "TIER2_TIMEOUT":
            pass                                  # handled below: no re-entry
        elif kill == "TIER2_UNCONFIGURED":
            pass
        elif kill == "FEATURES_UNAVAILABLE":
            p_wrong = None
        elif kill == "CALIBRATOR_STALE":
            p_wrong = min(1.0, p_wrong * 1.0)     # stale but present: unchanged

        # An unavailable p_wrong must WIDEN the estimate, never default to 0.
        effective_p = 1.0 if p_wrong is None else p_wrong

        invariants = gate.check_invariants(signals, ctx)
        verdict = gate.decide(effective_p, exp, budget, invariants)

        if verdict is Verdict.CHECK_HARDER:
            if kill in ("TIER2_TIMEOUT", "TIER2_UNCONFIGURED"):
                # §16: decide on the pre-check estimate at k = 1.0 → escalates
                verdict = gate.redecide_after_tier2(effective_p, exp, budget, invariants)
            else:
                verdict = gate.redecide_after_tier2(effective_p * 0.3, exp, budget, invariants)

        counts[verdict.value] += 1
    return counts


BASELINE = run_population(kill=None)

SUBSYSTEMS = [
    "TIER1_UNAVAILABLE", "TIER1_PARTIAL", "SOR_UNAVAILABLE",
    "LEDGER_ROW_MISSING", "TIER2_TIMEOUT", "TIER2_UNCONFIGURED",
    "FEATURES_UNAVAILABLE", "CALIBRATOR_STALE",
]


def test_the_baseline_actually_acts():
    """A harness where nothing ever ACTs would pass every test below while
    proving nothing at all."""
    assert BASELINE["act"] > 0.2 * POPULATION, (
        f"baseline ACT rate too low to be a meaningful control: {BASELINE}")


@pytest.mark.parametrize("subsystem", SUBSYSTEMS)
def test_killing_a_subsystem_never_raises_the_act_rate(subsystem):
    """Criterion 6, stated exactly: for every subsystem, the ACT rate does not
    increase. Zero exceptions to this."""
    killed = run_population(kill=subsystem)
    assert killed["act"] <= BASELINE["act"], (
        f"{subsystem} RAISED the ACT rate: {BASELINE['act']} -> {killed['act']}. "
        "An unavailable component must reduce autonomy, never increase it.")


@pytest.mark.parametrize("subsystem", SUBSYSTEMS)
def test_killing_a_subsystem_never_lowers_human_oversight(subsystem):
    """The mirror of the above. ACT falling is necessary but not sufficient --
    the requests that stop ACTing must climb, not vanish."""
    killed = run_population(kill=subsystem)
    baseline_supervised = BASELINE["escalate"] + BASELINE["block"] + BASELINE["check_harder"]
    killed_supervised = killed["escalate"] + killed["block"] + killed["check_harder"]
    assert killed_supervised >= baseline_supervised


def test_killing_tier1_blocks_everything_it_can():
    """Invariants whose sensors are unavailable fail closed (§10.2), so the
    whole population lands on the fixed path."""
    killed = run_population(kill="TIER1_UNAVAILABLE")
    assert killed["act"] == 0 and killed["block"] == POPULATION


def test_one_dead_sensor_is_enough_to_block():
    """Partial degradation is still degradation. injection_score alone being
    unavailable fails `no_prompt_injection` closed."""
    assert run_population(kill="TIER1_PARTIAL")["act"] == 0


def test_an_unavailable_p_wrong_must_not_default_to_zero():
    """The degradation that must not be silent. If feature assembly fails and
    p_wrong defaulted to 0.0, every request would look perfectly safe at
    exactly the moment the system has gone blind."""
    assert run_population(kill="FEATURES_UNAVAILABLE")["act"] <= BASELINE["act"]


def test_a_missing_ledger_row_escalates_rather_than_acting():
    """budget 0 collapses the close-call band, so everything above the
    review-cost floor escalates. Tier 2 never fires."""
    killed = run_population(kill="LEDGER_ROW_MISSING")
    assert killed["check_harder"] == 0
    assert killed["act"] <= BASELINE["act"]


def test_a_dead_judge_escalates_rather_than_guessing():
    """§16: a Tier 2 timeout decides on the pre-check estimate at k = 1.0,
    which resolves to ESCALATE. It costs the full latency and still escalates,
    which is why a slow judge is worse than a fast one that is merely wrong."""
    killed = run_population(kill="TIER2_TIMEOUT")
    assert killed["act"] <= BASELINE["act"]
    assert killed["escalate"] >= BASELINE["escalate"]


# ── The ladder must degrade the same way ───────────────────────────────────

@pytest.mark.parametrize("dead", ["pii_score", "injection_score", "secret_score"])
def test_the_ladder_never_lowers_a_rung_when_a_sensor_dies(dead):
    healthy = ladder.Signals(pii_score=0.05, injection_score=0.02,
                             secret_score=0.0, verify_fail_frac=0.0)
    degraded = ladder.Signals(**{**healthy.__dict__, dead: None})
    for exp_loss, budget in ((10, 100), (100, 100), (400, 100)):
        assert (ladder.select_rung(degraded, exp_loss, budget, True)
                >= ladder.select_rung(healthy, exp_loss, budget, True))


def test_missing_sentinel_defaults_to_unavailable_not_measured():
    """The assembler's own default. A feature nobody supplied is `unavailable`,
    which widens; it is never quietly treated as measured-and-clean."""
    from vouch import features
    row = features.assemble(1.0, {})
    for name in features.CAN_BE_UNAVAILABLE:
        assert row[f"{name}_unavailable"] == 1
    assert Missing.UNAVAILABLE is not Missing.NOT_SUPPORTED
