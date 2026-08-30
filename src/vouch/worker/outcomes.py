"""Step 1 of the nightly flow -- close outcomes (13, 9.6).

The labels are already being generated and thrown away. Every deployment
running an agent produces free training signal: record re-verification, human
overrides, reversals, and users re-asking the same question three ways. That
last one is the one everybody discards, and it is enormous.

Nothing here touches wall clock. `run_reference_ts` is bound once by the
caller and passed down, so replaying last month's run produces last month's
numbers (11).
"""
from __future__ import annotations

import sqlite3

from .. import config

# ── 9.6 Evidence weighting ─────────────────────────────────────────────────

EVIDENCE_WEIGHT = {
    "record_verification": 1.00,    # fact, not opinion
    "human_override": 1.00,         # an expert labelled it
    "reversal": 0.90,               # the business corrected it
    "user_rework": 0.30,            # medium quality, enormous volume
    "offline_judge": 0.50,          # paid, sampled
}

AGE_HALFLIFE_SECONDS = config.AGE_HALFLIFE_DAYS * 86400


def age_weight(age_seconds: float) -> float:
    """Half-life decay. A clean record from two years ago is weaker evidence
    about today's configuration than one from last week."""
    return 0.5 ** (age_seconds / AGE_HALFLIFE_SECONDS)


def evidence_weight(source: str) -> float:
    """An unknown source weighs nothing. Silently defaulting to 1.0 would let
    any new signal source mint trust without anyone deciding it should."""
    return EVIDENCE_WEIGHT.get(source, 0.0)


def due_decisions(conn: sqlite3.Connection, run_reference_ts: float) -> list[sqlite3.Row]:
    """Decisions whose horizon has closed and that carry no outcome yet.

    An outcome recorded BEFORE its horizon closes is provisional, and treating
    it as final biases the corpus toward whatever resolves fastest -- which is
    exactly the fast, cheap, wrong answers.
    """
    return conn.execute(
        "SELECT * FROM decision WHERE outcome IS NULL AND horizon_ends <= ? ORDER BY ts",
        (run_reference_ts,),
    ).fetchall()


def process_events(conn: sqlite3.Connection, run_reference_ts: float) -> int:
    """Resolve due raw events into final decision outcomes.

    Highest registered evidence weight wins; newest breaks a tie. All events
    for that decision are then marked processed so a weaker late duplicate
    cannot rewrite a closed outcome on the next run.
    """
    events = conn.execute(
        "SELECT e.* FROM outcome_event e "
        "JOIN decision d ON d.id = e.decision_id "
        "WHERE e.processed_at IS NULL AND d.outcome IS NULL "
        "AND d.horizon_ends <= ? "
        "ORDER BY e.decision_id, e.weight DESC, e.reported_ts DESC, e.id DESC",
        (run_reference_ts,),
    ).fetchall()
    winners: dict[str, sqlite3.Row] = {}
    for event in events:
        winners.setdefault(event["decision_id"], event)
    for decision_id, event in winners.items():
        close_outcome(
            conn, decision_id, event["outcome"], event["source"],
            float(event["reported_ts"]),
        )
        conn.execute(
            "UPDATE outcome_event SET processed_at = ? "
            "WHERE decision_id = ? AND processed_at IS NULL",
            (run_reference_ts, decision_id),
        )
    return len(winners)


def close_outcome(conn: sqlite3.Connection, decision_id: str, outcome: str,
                  source: str, outcome_ts: float) -> None:
    if outcome not in ("clean", "wrong"):
        raise ValueError(f"outcome must be clean or wrong, got {outcome!r}")
    if source not in EVIDENCE_WEIGHT:
        raise ValueError(f"unregistered outcome source {source!r}")
    conn.execute(
        "UPDATE decision SET outcome = ?, outcome_source = ?, outcome_ts = ? WHERE id = ?",
        (outcome, source, outcome_ts, decision_id),
    )


def weighted_counts(rows: list[dict], run_reference_ts: float) -> tuple[float, float, int]:
    """(n_clean, n_total, n_own_raw) for one trust row.

    `n_own_raw` counts UNWEIGHTED, UN-BORROWED observations, because the gate
    that constrains borrowing must not itself be satisfiable by borrowing or
    by generous evidence weights (9.3).
    """
    n_clean = n_total = 0.0
    n_own_raw = 0
    for row in rows:
        if row.get("outcome") not in ("clean", "wrong"):
            continue
        n_own_raw += 1
        weight = evidence_weight(row.get("outcome_source", "")) * age_weight(
            max(0.0, run_reference_ts - float(row.get("outcome_ts") or row["ts"]))
        )
        n_total += weight
        if row["outcome"] == "clean":
            n_clean += weight
    return n_clean, n_total, n_own_raw
