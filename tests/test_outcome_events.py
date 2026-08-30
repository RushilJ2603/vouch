"""Outcome ingestion cannot mint weight and closes only after its horizon."""
from __future__ import annotations

from vouch import store
from vouch.worker import outcomes


def _decision(conn, decision_id: str = "d1", horizon: float = 200.0) -> None:
    store.append_decision(conn, {
        "id": decision_id, "ts": 100.0, "agent_id": "a", "action": "issue_refund",
        "band": "0-2k", "fingerprint": "fp", "calibrator_version": "cal-v1",
        "capability_tier": "none", "features": "{}", "p_wrong": 0.1,
        "tier_reached": 1, "exposure": 100.0, "budget": 20.0,
        "expected_loss": 10.0, "verdict": "act", "region": "act", "rung": 0,
        "horizon_ends": horizon, "latency_ms": 20.0, "upstream_ms": 15.0,
    })


def test_strongest_due_event_wins_and_all_are_marked_processed():
    conn = store.connect(":memory:")
    store.init_schema(conn)
    _decision(conn)
    conn.executemany(
        "INSERT INTO outcome_event "
        "(decision_id, reported_ts, outcome, source, weight) VALUES (?,?,?,?,?)",
        [
            ("d1", 150.0, "wrong", "user_rework", 0.30),
            ("d1", 160.0, "clean", "human_override", 1.00),
        ],
    )
    assert outcomes.process_events(conn, 199.0) == 0
    assert outcomes.process_events(conn, 250.0) == 1
    row = conn.execute("SELECT outcome, outcome_source FROM decision WHERE id='d1'").fetchone()
    assert tuple(row) == ("clean", "human_override")
    assert conn.execute(
        "SELECT COUNT(*) FROM outcome_event WHERE processed_at=250.0"
    ).fetchone()[0] == 2
