"""The nightly flow (13). Runs at 02:00.

Nothing here is on a request path, and nothing here touches wall clock except
to bind `run_reference_ts` ONCE at the top. Every downstream step takes that
value explicitly, so replaying last month's run produces last month's numbers.

    python -m vouch.worker --as-of $(date +%s) [--dry-run]

`--as-of` is REQUIRED and there is no default. Section 11 permits no
wall-clock call anywhere inside `src/vouch/` -- `make lint` greps for them and
fails the build -- so the reference time is bound at the process boundary and
threaded explicitly from there. That also makes every historical replay exact
rather than approximately right.
"""
from __future__ import annotations

import argparse
import sys

from .. import config, exposure
from . import drift, outcomes, trust


def run(conn, run_reference_ts: float, dry_run: bool = False) -> dict:
    """Steps 1 to 3. Step 4 (refit) is triggered by, not part of, this flow."""
    report = {"run_reference_ts": run_reference_ts, "closed": 0, "rows": 0, "alarms": []}

    # STEP 1 -- close outcomes whose horizon has passed.
    report["closed"] = outcomes.process_events(conn, run_reference_ts)

    # STEP 2 -- recompute trust per (agent, action, band, fingerprint).
    keys = conn.execute(
        "SELECT DISTINCT agent_id, action, band, fingerprint FROM decision"
    ).fetchall()
    for key in keys:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM decision WHERE agent_id=? AND action=? AND band=? "
            "AND fingerprint=? ORDER BY ts", tuple(key)).fetchall()]
        # Declines and explicit escalations are valuable calibration outcomes,
        # but they have no side effect to price and therefore never mint an
        # action budget. Unknown actions are likewise excluded fail-closed.
        try:
            config.action_spec(key["action"])
        except KeyError:
            continue
        if key["band"] == "unpriced":
            continue
        n_clean, n_total, n_own_raw = outcomes.weighted_counts(rows, run_reference_ts)
        ceiling = exposure.ceiling(key["action"], key["band"])
        breaker = trust.evaluate_breaker(trust.BreakerState(), rows, run_reference_ts)
        row = trust.recompute_row(key["band"], ceiling, n_clean, n_total,
                                  n_own_raw, breaker)
        report["rows"] += 1
        if not dry_run:
            conn.execute(
                "INSERT INTO trust_row (agent_id, action, band, config_fingerprint, "
                "n_total, n_clean, n_own_raw, p_lo, budget, ceiling, state, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(agent_id, action, band, config_fingerprint) DO UPDATE SET "
                "n_total=excluded.n_total, n_clean=excluded.n_clean, "
                "n_own_raw=excluded.n_own_raw, p_lo=excluded.p_lo, "
                "budget=excluded.budget, state=excluded.state, "
                "updated_at=excluded.updated_at",
                (key["agent_id"], key["action"], key["band"], key["fingerprint"],
                 n_total, n_clean, n_own_raw, row["p_lo"], row["budget"],
                 ceiling, row["state"], run_reference_ts))

        # STEP 3 -- drift, with the minimum-sample floor on every alarm.
        if not drift.has_power(len(rows)):
            continue
        report["alarms"].append({"key": tuple(key), "verdict": "checked"})

    if not dry_run:
        conn.commit()
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vouch.worker")
    ap.add_argument("--as-of", type=float, required=True,
                    help="run_reference_ts, epoch seconds. REQUIRED: 11 permits no "
                         "wall-clock read inside src/vouch/, and the worker TAKES this "
                         "value rather than reading it, so replaying last month's run "
                         "produces last month's numbers. Cron passes $(date +%s).")
    ap.add_argument("--db", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from .. import store
    db = args.db or "data/vouch.sqlite"
    conn = store.connect(db)
    store.init_schema(conn)
    report = run(conn, args.as_of, dry_run=args.dry_run)
    print(f"closed {report['closed']} outcomes, recomputed {report['rows']} trust rows, "
          f"{len(report['alarms'])} rows had enough observations to check for drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
