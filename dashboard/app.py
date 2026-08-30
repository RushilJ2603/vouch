"""Live trust rows and the four charts (6.3).

Deliberately read-only and deliberately plain. The dashboard is a window onto
the decision log, not a control surface: anything that can change a trust row
belongs in the worker, where it is auditable.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "vouch.sqlite"
REPORTS = ROOT / "reports"


def trust_rows() -> list[dict]:
    if not DB.exists():
        return []
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(
        "SELECT agent_id, action, band, p_lo, budget, ceiling, state, n_own_raw "
        "FROM trust_row ORDER BY agent_id, action, band")]


def render() -> str:
    rows = trust_rows()
    if not rows:
        return ("No trust rows yet. Serve traffic through the proxy, then run\n"
                "  make worker\n"
                "which closes outcomes and recomputes the ledger.")
    out = [f"{'agent':<16}{'action':<16}{'band':<10}{'p_lo':>8}{'budget':>10}"
           f"{'ceiling':>10}{'n':>7}  state", "-" * 84]
    for r in rows:
        out.append(f"{r['agent_id']:<16}{r['action']:<16}{r['band']:<10}"
                   f"{r['p_lo']:>8.4f}{r['budget']:>10.2f}{r['ceiling']:>10.2f}"
                   f"{r['n_own_raw']:>7}  {r['state']}")
    charts = sorted(p.name for p in REPORTS.glob("*.png")) if REPORTS.exists() else []
    out.append("")
    out.append(f"charts in reports/: {', '.join(charts) if charts else 'none generated yet'}")
    return "\n".join(out)


def main() -> int:
    print(render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
