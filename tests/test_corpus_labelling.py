"""Ground truth for the corpus, and the circularity it has to avoid (25.2, 21).

Tier 0 verifies claims against the same database the label is computed from.
If the label were only "did the claims match the record", `verify_fail_frac`
would predict it perfectly, the calibrator would be a lookup table, and every
calibration curve in the submission would be measuring a tautology.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_corpus.py"
spec = importlib.util.spec_from_file_location("generate_corpus", SCRIPT)
gc = importlib.util.module_from_spec(spec)
sys.modules["generate_corpus"] = gc
spec.loader.exec_module(gc)

HONEST_CLAIMS = {
    "order_id": "ord_1", "duplicate_charge": True,
    "refund_amount_paise": 100_000, "already_refunded": False,
}
REFUND = {"type": "issue_refund", "amount_paise": 100_000, "order_id": "ord_1"}

BASE_FACTS = {
    "order_id": "ord_1", "amount_paise": 100_000, "status": "DELIVERED",
    "has_duplicate": True, "already_refunded": False, "within_window": True,
}


def label(claims=None, action=None, **fact_overrides):
    row = {"claims": claims or HONEST_CLAIMS, "action": action or REFUND}
    return gc.label_outcome(row, {**BASE_FACTS, **fact_overrides})


def test_an_honest_refund_on_a_real_duplicate_is_clean():
    assert label() == ("clean", [])


# ── The two faults Tier 0 cannot see ───────────────────────────────────────

def test_identical_claims_can_be_clean_or_wrong_depending_on_policy():
    """This is the whole point. The claims are byte-identical in both calls, so
    every Tier 0 feature is identical too -- yet one is clean and one is wrong.
    If this test ever fails, the label has collapsed into verify_fail_frac and
    the calibrator has nothing left to learn."""
    clean_outcome, _ = label()
    wrong_outcome, reasons = label(within_window=False)
    assert clean_outcome == "clean"
    assert wrong_outcome == "wrong"
    assert reasons == ["outside_refund_window"]


def test_refunding_a_cancelled_order_is_wrong():
    _, reasons = label(status="CANCELLED", has_duplicate=False, within_window=False)
    assert "refunded_cancelled_order" in reasons


def test_the_policy_faults_are_not_registered_tier0_verifiers():
    """Stated as a test so it cannot drift: if somebody adds a verifier for
    these, the label stops being independent and this test tells them."""
    from vouch.sensors import tier0
    assert "within_window" not in tier0.VERIFIERS
    assert "status" not in tier0.VERIFIERS


# ── The faults Tier 0 can see ──────────────────────────────────────────────

def test_claiming_a_duplicate_that_does_not_exist_is_wrong():
    _, reasons = label(has_duplicate=False)
    assert "claimed_duplicate_that_does_not_exist" in reasons


def test_a_second_refund_on_an_already_refunded_order_is_wrong():
    _, reasons = label(already_refunded=True)
    assert "double_refund" in reasons
    assert "misreported_refund_status" in reasons


def test_refunding_more_than_the_order_value_is_wrong():
    action = {"type": "issue_refund", "amount_paise": 500_000, "order_id": "ord_1"}
    _, reasons = label(action=action)
    assert "over_refund" in reasons


def test_naming_the_wrong_order_is_wrong():
    claims = {**HONEST_CLAIMS, "order_id": "ord_999"}
    _, reasons = label(claims=claims)
    assert "wrong_order" in reasons


# ── Not acting is not automatically wrong ──────────────────────────────────

def test_declining_to_refund_outside_the_window_is_clean():
    """Policy says escalate, not refund. An agent that correctly declines must
    not be labelled wrong, or the corpus teaches the opposite of the policy."""
    decline = {"type": "escalate", "amount_paise": 0, "order_id": "ord_1"}
    outcome, reasons = label(action=decline, within_window=False)
    assert outcome == "clean" and reasons == []


# ── Corpus construction ────────────────────────────────────────────────────

def test_every_scenario_retrieves_policy_chunks():
    """Features 3 and 4 are similarity to the chunks retrieved for THIS
    request. No chunks means both are unavailable on 100% of rows, which
    breaches Appendix A's < 5% envelope and blocks fitting."""
    for scenario in gc.SCENARIOS:
        chunks = gc.retrieve(scenario)
        assert len(chunks) >= 2
        assert all(isinstance(c, str) and c for c in chunks)


def test_the_scenario_mix_is_not_a_single_scenario():
    """A corpus of one scenario has almost no feature variance, so 21.3's
    distribution checks pass trivially and the calibrator learns nothing."""
    assert len(gc.SCENARIOS) >= 4


def test_prompts_are_deterministic_across_runs():
    import sqlite3
    db = Path(__file__).resolve().parents[1] / "data" / "orders.sqlite"
    if not db.exists():
        pytest.skip("run scripts/seed_db.py first")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    first = gc.plan_turns(conn, 25, set())
    second = gc.plan_turns(conn, 25, set())
    assert [t["prompt"] for t in first] == [t["prompt"] for t in second]
    assert [t["ts"] for t in first] == [t["ts"] for t in second]


def test_resumption_skips_completed_turns():
    import sqlite3
    db = Path(__file__).resolve().parents[1] / "data" / "orders.sqlite"
    if not db.exists():
        pytest.skip("run scripts/seed_db.py first")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    done = {t["turn_id"] for t in gc.plan_turns(conn, 10, set())}
    resumed = gc.plan_turns(conn, 10, done)
    assert not ({t["turn_id"] for t in resumed} & done)
