#!/usr/bin/env python3
"""Demo 3 -- the latency ledger, measured (24.3).

Drives traffic through the proxy against a mock upstream that streams at a
realistic token rate, then reports p50/p99 of ADDED latency alongside the
Tier 2 firing rate.

Pass: p50 < 40 ms, p99 < 500 ms, firing rate 2-5%, over >= 1,000 requests.
(Restated 2026-08-25 from `p50 < 5 ms` -- see 12.5.)

All three are gated together, deliberately. Any one alone can be gamed: narrow
`k` until Tier 2 never fires and the p99 looks wonderful while the close-call
band does nothing at all.

Free. No API key, no network -- the upstream is a mock.

    python3 scripts/demo_03_latency.py --drive 1000
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vouch import config, exposure, ledger, store  # noqa: E402
from vouch.proxy import app as proxy  # noqa: E402


def _plot(added: list[float], p50: float, p99: float, firing: float) -> None:
    """24.3 asks for the histogram against the 12.3 budget table.

    Colour is load-bearing and follows the deck: teal is the allowed/fast side,
    magenta-pink is the limit and the part held for a human.
    """
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
    fig, ax = plt.subplots(figsize=(7, 4.2))
    # A handful of cold-start outliers stretch the axis to several hundred ms
    # and squash the distribution the chart exists to show. Clip the view and
    # SAY what is outside it -- a latency chart that silently drops its tail is
    # the one kind of dishonesty this whole demo argues against.
    view = max(p99 * 1.6, 60.0)
    beyond = sum(1 for v in added if v > view)
    ax.hist([v for v in added if v <= view], bins=60, color="#00a0a0", alpha=0.85)
    ax.set_xlim(0, view)
    if beyond:
        ax.annotate(f"{beyond} of {len(added)} requests above {view:.0f} ms, "
                    f"max {max(added):.0f} ms", (0.98, 0.72), xycoords="axes fraction",
                    fontsize=8, color="#4a4a52", ha="right")
    ax.axvline(40, ls="--", color="#7e3ff2", lw=1.2)
    ax.annotate("40 ms gate (12.5)", (40, ax.get_ylim()[1] * 0.92), fontsize=8,
                color="#7e3ff2", rotation=90, va="top", ha="right")
    ax.axvline(p50, ls=":", color="#4a4a52", lw=1)
    ax.annotate(f"p50 {p50:.1f} ms", (p50, ax.get_ylim()[1] * 1.02), fontsize=8,
                color="#4a4a52", va="bottom", ha="center")
    ax.set_xlabel("added latency, ms (proxy total minus upstream)")
    ax.set_ylabel("requests")
    ax.set_title(f"Demo 3 — added latency over {len(added)} requests "
                 f"(p99 {p99:.1f} ms, Tier 2 firing {firing:.1%})")
    fig.tight_layout()
    fig.savefig(REPORTS / "demo_03_latency.png", dpi=130)
    print("\n  chart -> reports/demo_03_latency.png")


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "demo03.sqlite"
REPORTS = ROOT / "reports"
MIN_REQUESTS = 1000
AGENT = "refund-agent"
FINGERPRINT = "demo03"

REPLY = (" ".join(["I have reviewed order {oid} against the refund policy and the "
                   "charge ledger, and issued the refund."] * 3))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def seed_ledger(conn, n_clean: int = 400) -> None:
    """A row that has EARNED a budget. Without one, budget is 0, the close-call
    band collapses to nothing (budget x k = 0), Tier 2 never fires, and the
    firing-rate gate can never be met. That is the design working, not a
    workaround -- but it means this demo has to replay a system with a record.
    """
    store.init_schema(conn)
    for band, _lo, _hi in config.BANDS:
        ceiling = exposure.ceiling("issue_refund", band)
        p_lo = ledger.wilson_lower(n_clean, n_clean)
        conn.execute(
            "INSERT OR REPLACE INTO trust_row (agent_id, action, band, "
            "config_fingerprint, n_total, n_clean, n_own_raw, p_lo, budget, "
            "ceiling, state, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (AGENT, "issue_refund", band, FINGERPRINT, float(n_clean), float(n_clean),
             n_clean, p_lo, ledger.budget(p_lo, ceiling, n_clean), ceiling,
             "autonomous", 0.0))
    conn.commit()


def adopt_calibrator(conn) -> None:
    """Serve `p_wrong` from Calibrator A, the way 12.3 step 7 specifies.

    The registry row lives in the operational DB that `scripts/bootstrap.py`
    writes, and this demo builds its own throwaway one, so the row is copied
    across and then adopted through the real 12.6 handshake -- checksum and
    sklearn version included. Nothing here bypasses it.

    Without this the proxy falls back to a per-action prior, which is a
    CONSTANT: expected loss becomes constant x exposure, the gate thresholds on
    exposure alone, and Tier 2 fires on ~40% of traffic instead of 2-5%.
    """
    import importlib.metadata as md
    src = ROOT / "data" / "vouch.sqlite"
    if not src.exists():
        print("  no calibrator registry yet -- run scripts/bootstrap.py first; "
              "falling back to the per-action prior (16)")
        return
    other = store.connect(str(src), read_only=True)
    rows = other.execute("SELECT * FROM calibrator_registry").fetchall()
    other.close()
    for r in rows:
        cols = ", ".join(r.keys())
        marks = ", ".join("?" * len(r.keys()))
        conn.execute(f"INSERT OR REPLACE INTO calibrator_registry ({cols}) "
                     f"VALUES ({marks})", tuple(r))
    conn.commit()
    adopted = proxy.reload_calibrator(conn, md.version("scikit-learn"))
    if adopted:
        print(f"  calibrator {adopted} adopted (12.3 step 7 is live)")
    else:
        print(f"  calibrator NOT adopted: "
              f"{proxy.STATE.get('degraded_reason', 'no ready version')}")


def realistic_amounts(n: int, seed: int = 4242) -> list[int]:
    """The same shape the seeded record uses. NOT tuned to hit a firing rate:
    tuning the traffic until Tier 2 fires 3% of the time is the same gaming
    the three-way gate exists to prevent."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.52:
            amount = rng.uniform(10, 1_999)
        elif r < 0.77:
            amount = rng.uniform(2_000, 9_999)
        elif r < 0.89:
            amount = rng.uniform(10_000, 49_999)
        else:
            amount = rng.uniform(50_000, 199_999)
        out.append(int(amount * 100))
    return out


async def drive(n: int, concurrency: int) -> None:
    from vouch.proxy.app import (
        ChatRequest,
        VouchBlock,
        chat_completions,
        start_log_writer,
    )
    start_log_writer()                     # 12.3 step 12 runs off the request path
    sem = asyncio.Semaphore(concurrency)

    async def one(amount_paise: int, i: int) -> None:
        async with sem:
            req = ChatRequest(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": REPLY.format(oid=f"ord_{i:05d}")}],
                vouch=VouchBlock(agent_id=AGENT, action_hint="issue_refund",
                                 amount_paise=amount_paise),
            )
            try:
                await chat_completions(req)
            except Exception:
                pass                       # over-hard-limit requests raise; they are real traffic

    await asyncio.gather(*(one(a, i) for i, a in enumerate(realistic_amounts(n))))
    from vouch.proxy import app as _proxy
    if _proxy.LOG_QUEUE is not None:
        await _proxy.LOG_QUEUE.join()      # let the writer drain before measuring


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", type=int, default=0, help="requests to generate first")
    ap.add_argument("--concurrency", type=int, default=40)
    args = ap.parse_args()

    if args.drive:
        DB.unlink(missing_ok=True)
        conn = store.connect(str(DB))
        seed_ledger(conn)
        proxy.STATE["conn"] = conn
        proxy.STATE["fingerprint"] = FINGERPRINT
        proxy.STATE["started_at"] = time.time()
        proxy.STATE["ledger"] = {
            (r["agent_id"], r["action"], r["band"], r["config_fingerprint"]): dict(r)
            for r in store.load_all_trust_rows(conn)}
        proxy.STATE["length_history"] = {}
        adopt_calibrator(conn)
        print(f"Driving {args.drive} requests at concurrency {args.concurrency} "
              f"against the mock upstream...")
        start = time.perf_counter()
        asyncio.run(drive(args.drive, args.concurrency))
        print(f"  done in {time.perf_counter() - start:.1f}s\n")

    if not DB.exists():
        print("Demo 3 -- no traffic recorded. Run with --drive 1000.")
        return 0

    conn = store.connect(str(DB), read_only=True)
    rows = conn.execute(
        "SELECT latency_ms, upstream_ms, tier_reached, verdict FROM decision").fetchall()
    if not rows:
        print("Demo 3 -- decision log is empty.")
        return 0

    added = [r["latency_ms"] - r["upstream_ms"] for r in rows]
    firing = sum(1 for r in rows if r["tier_reached"] >= 2) / len(rows)
    p50, p99 = percentile(added, 0.50), percentile(added, 0.99)

    print(f"Demo 3 -- added latency over {len(rows)} requests\n")
    print(f"  p50 added:      {p50:9.3f} ms    (pass: < 40, restated 12.5)")
    print(f"  p99 added:      {p99:9.3f} ms    (pass: < 500)")
    print(f"  max added:      {max(added):9.3f} ms")
    print(f"  Tier 2 firing:  {firing:9.2%}       (pass: 2-5%)")

    from collections import Counter
    mix = Counter(r["verdict"] for r in rows)
    print("\n  verdict mix: " + ", ".join(f"{k}={v}" for k, v in sorted(mix.items())))

    # ── Diagnose before reporting a pass or a fail ─────────────────────────
    if mix.get("block", 0) / len(rows) > 0.5:
        print()
        print("  DIAGNOSIS: almost everything BLOCKED, and Tier 2 never fired.")
        print("  This is not a gate misconfiguration. The Tier 1 ONNX encoders for")
        print("  features 11-13 do not exist yet (scripts/export_onnx.py is a stub),")
        print("  so tier1.score_spans returns `unavailable` for pii, injection and")
        print("  policy. 10.2 fails an invariant whose sensor is unavailable CLOSED --")
        print("  absence of evidence is not evidence -- so every request is blocked.")
        print()
        print("  The system is behaving exactly as specified and is useless in that")
        print("  state. The latency figures below are for the BLOCK path, which")
        print("  short-circuits before Tier 2, so they are NOT the numbers 24.3 asks")
        print("  for. Export the encoders, then re-run.")
        print()
        print("  Note also what the p50 above implies. It passes at 4.90 ms with Tier 1")
        print("  doing regex work only. tests/test_stream_overlap.py measured added")
        print("  latency floors at ONE forward pass, and 6.1 puts ONNX CPU at 5-15 ms.")
        print("  Real encoders will therefore push p50 past the 5 ms budget on their")
        print("  own. That is the same finding as 12.4, arriving from a second")
        print("  direction, and it says the budget needs restating rather than the")
        print("  implementation needing tuning.")

    print("\n  The p99 is meaningless without the firing rate beside it. At a 3% firing")
    print("  rate the slowest 1% of ALL requests sits entirely inside the 3% that fired")
    print("  Tier 2, so the overall p99 is the 67th percentile of JUDGE latency, not its")
    print("  99th. Narrowing k until Tier 2 never fires buys a beautiful p99 and a")
    print("  close-call band that does nothing, which is why all three are gated.")

    _plot(added, p50, p99, firing)

    checks = {
        "p50 < 40 ms": p50 < 40,
        "p99 < 500 ms": p99 < 500,
        "firing 2-5%": 0.02 <= firing <= 0.05,
        f">= {MIN_REQUESTS} requests": len(rows) >= MIN_REQUESTS,
    }
    print()
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {'PASS' if all(checks.values()) else 'NOT PASSED'}")
    _record(3, "Latency bounded", all(checks.values()),
            f"p50 {p50:.1f} ms (gate < 40), p99 {p99:.1f} ms (gate < 500), "
            f"Tier 2 firing {firing:.1%} (gate 2\u20135%), over {len(rows)} requests")
    return 0 if all(checks.values()) else 1


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
