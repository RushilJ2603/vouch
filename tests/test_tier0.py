"""Section 7.1 -- deterministic verification against the system of record."""
import sqlite3
from pathlib import Path

import pytest

from vouch.sensors import tier0

DB = Path(__file__).resolve().parents[1] / "data" / "orders.sqlite"

pytestmark = pytest.mark.skipif(not DB.exists(), reason="run scripts/seed_db.py first")


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(DB)
    yield c
    c.close()


@pytest.fixture(scope="module")
def duplicate_order(conn):
    return conn.execute(
        "SELECT order_id FROM charge GROUP BY order_id, amount_paise HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone()[0]


@pytest.fixture(scope="module")
def single_charge_order(conn):
    return conn.execute(
        "SELECT order_id FROM charge GROUP BY order_id HAVING COUNT(*) = 1 LIMIT 1"
    ).fetchone()[0]


def _amount(conn, order_id):
    return conn.execute('SELECT amount_paise FROM "order" WHERE id = ?', (order_id,)).fetchone()[0]


# ── Truthful claims verify clean ───────────────────────────────────────────

def test_a_truthful_claim_set_passes(conn, duplicate_order):
    frac, n = tier0.verify_claims(conn, {
        "order_id": duplicate_order,
        "duplicate_charge": True,
        "refund_amount_paise": _amount(conn, duplicate_order),
    })
    assert (frac, n) == (0.0, 3)


# ── Each verifier catches its own lie ──────────────────────────────────────

def test_nonexistent_order_is_caught(conn):
    frac, n = tier0.verify_claims(conn, {"order_id": "ord_does_not_exist"})
    assert (frac, n) == (1.0, 1)


def test_false_duplicate_charge_claim_is_caught(conn, single_charge_order):
    """The agent claims a duplicate charge on an order charged exactly once."""
    frac, n = tier0.verify_claims(conn, {
        "order_id": single_charge_order,
        "duplicate_charge": True,
    })
    assert n == 2 and frac == pytest.approx(0.5)


def test_wrong_refund_amount_is_caught(conn, duplicate_order):
    frac, n = tier0.verify_claims(conn, {
        "order_id": duplicate_order,
        "refund_amount_paise": _amount(conn, duplicate_order) + 50_000,
    })
    assert n == 2 and frac == pytest.approx(0.5)


# ── 7.1 the uncheckable-claim rule ─────────────────────────────────────────

def test_unregistered_claims_are_uncheckable_not_passing(conn, duplicate_order):
    """A claim with no verifier contributes to NEITHER the numerator nor the
    denominator. It lowers verify_n_claims instead."""
    frac, n = tier0.verify_claims(conn, {
        "order_id": duplicate_order,
        "customer_was_polite": True,
        "vibes": "good",
    })
    assert (frac, n) == (0.0, 1)


def test_empty_claims_are_distinguishable_from_verified_ones(conn, duplicate_order):
    """'I reviewed your account and everything looks correct' with claims: {}
    produces the same fail fraction as a fully-verified reply. Only
    verify_n_claims separates them, which is why it is a separate feature."""
    content_free = tier0.verify_claims(conn, {})
    verified = tier0.verify_claims(conn, {
        "order_id": duplicate_order,
        "refund_amount_paise": _amount(conn, duplicate_order),
    })
    assert content_free[0] == verified[0] == 0.0     # identical fail fraction
    assert content_free[1] == 0 and verified[1] == 2  # and yet distinguishable


def test_a_claim_that_cannot_be_evaluated_counts_as_failed(conn):
    frac, n = tier0.verify_claims(conn, {"order_id": "x", "refund_amount_paise": "not a number"})
    assert n == 2 and frac == 1.0


# ── Free textual signals ───────────────────────────────────────────────────

def test_hedge_density():
    assert tier0.hedge_density("The refund was issued.") == 0.0
    assert tier0.hedge_density("It might possibly be around the right amount") > 0.2
    assert tier0.hedge_density("") == 0.0


def test_retrieval_support_rewards_grounded_sentences():
    chunks = ["refunds are issued within 30 days of a duplicate charge"]
    grounded = tier0.retrieval_support("Refunds are issued within 30 days.", chunks)
    invented = tier0.retrieval_support("Your warranty covers accidental screen damage.", chunks)
    assert grounded[0] > invented[0]


def test_retrieval_support_with_no_chunks_is_zero_not_one():
    assert tier0.retrieval_support("Anything at all.", []) == (0.0, 0.0)


def test_tool_counts():
    calls = [{"is_retry": False}, {"is_retry": True}, {"error": "timeout"}]
    assert tier0.tool_counts(calls) == (1, 1)
    assert tier0.tool_counts([]) == (0, 0)
