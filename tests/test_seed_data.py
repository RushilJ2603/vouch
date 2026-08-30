"""The seeded system of record, and the contract between hard limits and bands.

Section 21.3 requires >= 150 turns in each of the four bands before any fitting
happens. The corpus is drawn from this database, so the coverage has to exist
here first or the gate blocks Week 2.
"""
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from vouch import config, exposure, gate

DB = Path(__file__).resolve().parents[1] / "data" / "orders.sqlite"

pytestmark = pytest.mark.skipif(not DB.exists(), reason="run scripts/seed_db.py first")


@pytest.fixture(scope="module")
def amounts() -> list[float]:
    conn = sqlite3.connect(DB)
    try:
        return [r[0] / 100 for r in conn.execute('SELECT amount_paise FROM "order"')]
    finally:
        conn.close()


def test_amounts_are_stored_in_paise(amounts):
    """Rupees would put the whole corpus in band 0-2k and quietly delete three
    quarters of the design."""
    assert max(amounts) > 2_000


def test_every_band_clears_the_bootstrap_gate(amounts):
    hard_limit = exposure.hard_limit("issue_refund")
    counts = Counter(config.band_for(a) for a in amounts if a <= hard_limit)
    for band, _lo, _hi in config.BANDS:
        assert counts.get(band, 0) >= 150, f"band {band} has {counts.get(band, 0)}, 21.3 wants >= 150"


def test_the_block_path_has_rows_to_exercise_it(amounts):
    """A system of record contains orders the agent may never refund. Without
    over-limit rows the hard-limit invariant is never exercised by real data."""
    hard_limit = exposure.hard_limit("issue_refund")
    assert sum(1 for a in amounts if a > hard_limit) > 0


# ── The ordering contract ──────────────────────────────────────────────────

def test_hard_limit_is_checked_before_the_band_is_resolved(amounts):
    """`band_for` raises above the hard limit, because the top band is CLOSED
    by that limit and an open-ended band has no derivable ceiling. That is the
    correct behaviour, and it makes the call order load-bearing: invariants
    first, band second. Getting it backwards turns a BLOCK into a 500."""
    hard_limit = exposure.hard_limit("issue_refund")
    over = next(a for a in amounts if a > hard_limit)

    result = gate.check_invariants(
        {"pii_score": 0.0, "injection_score": 0.0, "secret_score": 0.0},
        {"retrieval_scope": 1, "user_scope": 1, "amount": over, "hard_limit": hard_limit},
    )
    assert gate.decide(0.0, 1.0, 1e9, result) is gate.Verdict.BLOCK

    with pytest.raises(ValueError, match="hard_limit"):
        config.band_for(over)
