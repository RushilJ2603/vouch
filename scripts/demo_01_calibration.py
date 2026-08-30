#!/usr/bin/env python3
"""Demo 1 -- calibration (24.1).

Proves that `expected_loss = p_wrong x exposure` means something. Every other
claim in the submission is downstream of this one: if p_wrong is not
calibrated, the budget arithmetic is dressing.

Pass: ECE < 0.05 global AND < 0.08 close-call, with >= 100 samples per region.
Both numbers are gated, because a global ECE is an average and an average
hides the close-call band -- the only region where a better estimate changes
a decision.

Free to run, but needs the corpus. Replays; never calls a model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vouch import calibrate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "corpus_v1.jsonl"
REPORTS = ROOT / "reports"
ECE_GLOBAL_MAX, ECE_CLOSE_CALL_MAX, MIN_REGION_N = 0.05, 0.08, 100


def main() -> int:
    if not CORPUS.exists():
        print("Demo 1 -- calibration: NOT YET RUNNABLE\n")
        print(f"  needs {CORPUS.relative_to(ROOT)}, which does not exist.")
        print("  Generate it first (this is the step that spends money):\n")
        print("    export DEEPSEEK_API_KEY=...   # see .env.example")
        print("    python3 scripts/generate_corpus.py --limit 150   # ~Rs3 pilot")
        print("    python3 scripts/validate_corpus.py               # 21.3 gate\n")
        print("  Nothing is fitted on a corpus that has not cleared the 21.3 gate.")
        return 0

    rows = [json.loads(line) for line in CORPUS.open(encoding="utf-8") if line.strip()]
    scored = [r for r in rows if isinstance(r.get("p_wrong"), (int, float))
              and r.get("outcome") in ("clean", "wrong")]
    if not scored:
        print("Corpus exists but carries no scored decisions yet. Run the proxy "
              "replay first, then this demo.")
        return 0

    probs = [r["p_wrong"] for r in scored]
    labels = [1 if r["outcome"] == "wrong" else 0 for r in scored]
    regions = [r.get("region", "act") for r in scored]
    reports = calibrate.evaluate_by_region(probs, labels, regions)

    print(f"Demo 1 -- calibration over {len(scored)} scored turns\n")
    print(f"  {'region':<12} {'n':>6} {'ECE':>8} {'MCE':>8} {'Brier':>8}   power")
    print("  " + "-" * 56)
    for name in ("global", "act", "close_call", "escalate"):
        r = reports[name]
        power = "ok" if calibrate.region_has_power(r) or name == "global" else "insufficient_data"
        print(f"  {name:<12} {r.n:>6} {r.ece:>8.4f} {r.mce:>8.4f} {r.brier:>8.4f}   {power}")

    ok = (reports["global"].ece < ECE_GLOBAL_MAX
          and reports["close_call"].ece < ECE_CLOSE_CALL_MAX
          and reports["close_call"].n >= MIN_REGION_N)
    print(f"\n  {'PASS' if ok else 'FAIL'}: global < {ECE_GLOBAL_MAX}, "
          f"close-call < {ECE_CLOSE_CALL_MAX}, >= {MIN_REGION_N} per region")
    if not ok and reports["close_call"].n < MIN_REGION_N:
        print("  (close-call region is too thin to judge; that is insufficient_data, "
              "not a pass)")
    _plot(reports)
    _record(1, "Calibrated probability", ok,
            f"global ECE {reports['global'].ece:.4f} (gate < {ECE_GLOBAL_MAX}); "
            f"close_call n={reports['close_call'].n} (needs \u2265 {MIN_REGION_N})")
    return 0 if ok else 1


def _plot(reports) -> None:
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
    r = reports["global"]
    if not r.bins:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    xs = [b.predicted for b in r.bins]
    ys = [b.observed for b in r.bins]
    lo = [b.observed - b.lo for b in r.bins]
    hi = [b.hi - b.observed for b in r.bins]
    # The intervals are wide at ~150 rows per bin. We plot them rather than
    # drawing a smooth line through sparse data.
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color="#00a0a0", capsize=3)
    ax.set_xlabel("predicted P(wrong)")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Demo 1 — reliability, ECE {r.ece:.4f}")
    fig.tight_layout()
    fig.savefig(REPORTS / "demo_01_calibration.png", dpi=130)
    print("  chart -> reports/demo_01_calibration.png")


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
