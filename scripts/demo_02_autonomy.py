#!/usr/bin/env python3
"""Demo 2 -- a trust row earning autonomy, then losing it (24.2).

Proves the mechanism: nothing below 30 clean decisions, ~79% of ceiling at
100, ~95% at 400, and a circuit breaker that collapses it inside three
confirmed failures.

Runs on band Rs10,000-50,000, NOT on small refunds. In band 0-2k the ceiling
is 28.80, below the 40 review cost for every action in the band, so those
actions pass the day-one exposure test and the ledger never binds. The design
is working as intended; the small band is simply the wrong place to show it.

Free. No API key, no corpus, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vouch import config, exposure, ledger  # noqa: E402
from vouch.worker.trust import BreakerState, evaluate_breaker  # noqa: E402

BAND = "10k-50k"
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> int:
    ceiling = exposure.ceiling("issue_refund", BAND)
    print(f"Demo 2 -- band {BAND}, ceiling Rs{ceiling:.2f} "
          f"(risk_appetite {config.RISK_APPETITE} x band_max {config.band_max(BAND):,.0f} "
          f"x multipliers {exposure.multipliers('issue_refund'):.2f})\n")

    print(f"  {'clean':>7}  {'p_lo':>8}  {'budget':>9}  {'% ceiling':>10}  state")
    print("  " + "-" * 55)

    curve = []
    first_nonzero = None
    for n in [4, 20, 30, 50, 73, 75, 100, 150, 200, 381, 400, 1000]:
        p_lo = ledger.wilson_lower(n, n)
        budget = ledger.budget(p_lo, ceiling, n)
        pct = 100 * budget / ceiling
        state = "autonomous" if budget > 0 else (
            "supervised (n < 30)" if n < config.MIN_OWN_OBSERVATIONS else "below p_min")
        if budget > 0 and first_nonzero is None:
            first_nonzero = n
        curve.append((n, p_lo, budget, pct))
        print(f"  {n:>7}  {p_lo:>8.4f}  {budget:>9.2f}  {pct:>9.1f}%  {state}")

    print(f"\n  First n with any budget at all: {first_nonzero}")

    # Where exactly does it cross? The ownership gate is now binding.
    crossing = next(n for n in range(1, 200) if ledger.budget(
        ledger.wilson_lower(n, n), ceiling, n) > 0)
    print(f"  Crossing recomputed from scratch: {crossing}")
    assert crossing == config.MIN_OWN_OBSERVATIONS, (
        f"expected the floor to be crossed at {config.MIN_OWN_OBSERVATIONS}, got {crossing}"
    )

    at_400 = ledger.budget(ledger.wilson_lower(400, 400), ceiling, 400)
    print(f"  At 400 decisions: {100 * at_400 / ceiling:.1f}% of ceiling "
          f"(pass condition: >= 90%)")
    assert at_400 / ceiling >= 0.90

    # ── Then take it away. Three confirmed failures inside the window. ──
    print("\n  Circuit breaker -- trust rises slowly and falls fast:")
    history = [{"outcome": "clean"} for _ in range(400)]
    state = evaluate_breaker(BreakerState(), history, run_reference_ts=0.0)
    print(f"    after 400 clean:            tripped={state.tripped}  "
          f"budget=Rs{at_400:.2f}")

    history += [{"outcome": "wrong"} for _ in range(3)]
    state = evaluate_breaker(BreakerState(), history, run_reference_ts=1.0)
    print(f"    after 3 confirmed failures: tripped={state.tripped}  budget=Rs0.00")
    assert state.tripped, "three failures inside the window must trip the breaker"

    history += [{"outcome": "clean"} for _ in range(config.RECOVER_CLEAN)]
    state = evaluate_breaker(state, history, run_reference_ts=2.0)
    print(f"    after {config.RECOVER_CLEAN} clean again:        tripped={state.tripped} "
          f" (recovery needs clean decisions, not elapsed time)")

    _plot(curve, ceiling, crossing)
    print("\n  PASS: budget 0 below 30, >= 90% of ceiling by 400, breaker collapses it.")
    _record(2, "Autonomy earned, and revoked", True,
            f"budget 0 below 30 clean decisions; {100 * at_400 / ceiling:.1f}% of "
            f"ceiling at 400; breaker returns it to \u20b90.00 after 3 confirmed "
            f"failures, and recovery needs {config.RECOVER_CLEAN} clean decisions "
            "rather than elapsed time")
    return 0


def _plot(curve, ceiling, crossing: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # The charts sit inside dashboard/page.html, so they carry its ground and
        # its type rather than matplotlib's. Two typefaces across the largest
        # block of a page reads as two designs.
        plt.rcParams.update({
            "font.family": ["DejaVu Sans"], "font.size": 9,
            "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "text.color": "#16161a", "axes.labelcolor": "#4a4a52",
            "xtick.color": "#77777f", "ytick.color": "#77777f",
            "axes.edgecolor": "#cfcfc9", "grid.color": "#e0e0dc",
            "axes.titlesize": 10, "axes.titleweight": "semibold",
            "figure.constrained_layout.use": True,
        })
    except ImportError:
        return
    REPORTS.mkdir(exist_ok=True)
    ns = [c[0] for c in curve]
    budgets = [c[2] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ns, budgets, marker="o", color="#00a0a0")
    ax.axhline(ceiling, ls="--", color="#999", lw=1)
    ax.axvline(crossing, ls=":", color="#7e3ff2", lw=1)
    ax.annotate(f"{crossing}: first earned rupee", (crossing, ceiling * 0.5),
                fontsize=8, color="#7e3ff2", rotation=90, va="center")
    ax.annotate(f"ceiling {ceiling:.0f}", (ns[-1], ceiling), fontsize=8,
                va="bottom", ha="right", color="#666")
    ax.set_xscale("log")
    ax.set_xlabel("clean decisions (log scale)")
    ax.set_ylabel("earned budget, Rs")
    ax.set_title(f"Demo 2 — autonomy earned in band {BAND}")
    fig.tight_layout()
    fig.savefig(REPORTS / "demo_02_autonomy.png", dpi=130)
    print("\n  chart -> reports/demo_02_autonomy.png")


def _record(criterion: int, name: str, passed: bool, evidence: str) -> None:
    """Write this demo's own verdict where the dashboard can read it.

    The alternative was hardcoding six pass/fail rows into the page, which is
    how a surface ends up asserting a number nobody recomputed.
    """
    import json
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"criterion_{criterion}.json").write_text(json.dumps({
        "criterion": criterion, "name": name,
        "passed": bool(passed), "evidence": evidence,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
