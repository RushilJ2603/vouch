"""Section 21.3 -- the bootstrap validation gate.

Hard blocks that run BEFORE any fitting. A failure names the check, the
statistic and the offending rows, then exits non-zero. Nothing is fitted on a
corpus that has not cleared this.

    python3 scripts/validate_corpus.py

Every check degrades to SKIP when its input does not exist yet, so this is
runnable from the first day of Week 2 rather than only at the end of it.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vouch import calibrate, config, exposure, features  # noqa: E402
from vouch.worker import drift  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "corpus_v1.jsonl"
JUDGE = ROOT / "data" / "corpus" / "judge_v1.jsonl"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(check: str, status: str, detail: str = "") -> None:
    results.append((check, status, detail))


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ── 1. Point-in-time integrity (11) ────────────────────────────────────────

def check_point_in_time() -> None:
    wall_clock = re.compile(r"time\.time\(\)|datetime\.now\(|utcnow")
    source = ROOT / "src" / "vouch"
    offenders = []
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source).as_posix()
        if relative == "proxy/app.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if wall_clock.search(line):
                offenders.append(f"{relative}:{number}: {line.strip()}")
    if offenders:
        record("point-in-time", FAIL, f"wall-clock outside proxy/app.py: {offenders[:3]}")
    else:
        record("point-in-time", PASS, "no wall-clock calls outside proxy/app.py")


# ── 2. Feature order (Appendix A) ──────────────────────────────────────────

def check_feature_order() -> None:
    baselines = ROOT / "artifacts" / "baselines.json"
    if not baselines.exists():
        record("feature-order", SKIP, "artifacts/baselines.json not built yet")
        return
    order = json.loads(baselines.read_text(encoding="utf-8")).get("feature_order", [])
    if tuple(order) != features.FEATURE_ORDER:
        record("feature-order", FAIL, "baselines.json does not match Appendix A")
    else:
        record("feature-order", PASS, f"{len(order)} features, in order")


# ── 3. Null semantics (Appendix A envelopes) ───────────────────────────────

def check_null_semantics(rows: list[dict]) -> None:
    if not rows or "features" not in rows[0]:
        record("null-semantics", SKIP, "corpus carries no assembled features yet")
        return
    n = len(rows)
    bad = []
    for name in features.FEATURE_ORDER:
        flag = f"{name}_unavailable"
        rate = sum(r.get("features", {}).get(flag, 0) for r in rows) / n
        if name in features.CAN_BE_UNAVAILABLE and rate >= 0.05:
            bad.append(f"{name} unavailable on {rate:.1%}")
    if bad:
        record("null-semantics", FAIL, "; ".join(bad) + " (envelope is < 5%)")
    else:
        record("null-semantics", PASS, "every feature inside its envelope")


# ── 4. Label balance -- the agent acceptance test ──────────────────────────

def check_label_balance(rows: list[dict]) -> None:
    labelled = [r for r in rows if r.get("outcome") in ("clean", "wrong")]
    if not labelled:
        record("label-balance", SKIP, "corpus not labelled yet")
        return
    rate = sum(1 for r in labelled if r["outcome"] == "wrong") / len(labelled)
    detail = f"error rate {rate:.1%} over {len(labelled)} turns"
    if 0.03 <= rate <= 0.25:
        record("label-balance", PASS, detail)
    else:
        record("label-balance", FAIL, detail + " -- outside 3-25%. "
               + ("Agent too strong: nothing to calibrate against."
                  if rate < 0.03 else "Agent too weak to be representative."))


# ── 5. Band coverage ───────────────────────────────────────────────────────

def check_band_coverage(rows: list[dict]) -> None:
    amounts = [r["action"]["amount_paise"] / 100 for r in rows
               if isinstance(r.get("action"), dict) and "amount_paise" in r["action"]]
    if not amounts:
        record("band-coverage", SKIP, "corpus has no priced actions yet")
        return
    limit = exposure.hard_limit("issue_refund")
    counts = Counter(config.band_for(a) for a in amounts if a <= limit)
    thin = [f"{b}={counts.get(b, 0)}" for b, _, _ in config.BANDS if counts.get(b, 0) < 150]
    if thin:
        record("band-coverage", FAIL, "below 150: " + ", ".join(thin))
    else:
        record("band-coverage", PASS, ", ".join(f"{b}={counts[b]}" for b, _, _ in config.BANDS))


# ── 6. Region coverage ─────────────────────────────────────────────────────

def check_region_coverage(rows: list[dict]) -> None:
    regions = [r.get("region") for r in rows if r.get("region")]
    if not regions:
        record("region-coverage", SKIP, "no decisions replayed yet")
        return
    n = Counter(regions).get("close_call", 0)
    if n >= 100:
        record("region-coverage", PASS, f"{n} close_call turns")
    else:
        record("region-coverage", FAIL,
               f"only {n} close_call turns; per-region ECE needs >= 100")


# ── 7. Distribution stability inside the corpus ────────────────────────────

def check_distribution_stability(rows: list[dict]) -> None:
    if len(rows) < 200 or "features" not in rows[0]:
        record("distribution-stability", SKIP, "needs >= 200 assembled rows")
        return
    half = len(rows) // 2
    breaches = []
    for name in features.FEATURE_ORDER:
        a = [r["features"][name] for r in rows[:half]
             if isinstance(r.get("features", {}).get(name), (int, float))]
        b = [r["features"][name] for r in rows[half:]
             if isinstance(r.get("features", {}).get(name), (int, float))]
        if len(a) < 50 or len(b) < 50:
            continue
        value = drift.psi(a, b)
        if value > 0.10:
            breaches.append(f"{name} PSI={value:.3f}")
    if breaches:
        record("distribution-stability", FAIL, "; ".join(breaches))
    else:
        record("distribution-stability", PASS, "all features PSI <= 0.10 across corpus halves")


# ── 8. Judge calibration -- NEW, and the one that fails silently ───────────

def check_judge_calibration(judged: list[dict], corpus: list[dict]) -> None:
    if not judged:
        record("judge-calibration", SKIP, "judge_v1.jsonl not generated yet")
        return
    truth = {r.get("turn_id"): r.get("outcome") for r in corpus}
    pairs = [(r["p_wrong"], 1 if truth.get(r.get("turn_id")) == "wrong" else 0)
             for r in judged
             if isinstance(r.get("p_wrong"), (int, float))
             and truth.get(r.get("turn_id")) in ("clean", "wrong")]
    if len(pairs) < 50:
        record("judge-calibration", SKIP, f"only {len(pairs)} judged rows with ground truth")
        return
    probs = [p for p, _ in pairs]
    rho = calibrate.spearman(probs, [y for _, y in pairs])
    deciles = calibrate.distinct_deciles(probs)
    detail = f"rho={rho:.3f} over {len(pairs)} rows, spans {deciles} deciles"
    if rho >= 0.3 and deciles >= 3:
        record("judge-calibration", PASS, detail)
    else:
        record("judge-calibration", FAIL, detail +
               " -- feature 15 carries no usable signal, so calibrator B collapses "
               "to calibrator A and Tier 2 is pure latency")


# ── 9. Judge schema adherence -- NEW ───────────────────────────────────────

def check_judge_schema(judged: list[dict]) -> None:
    if not judged:
        record("judge-schema", SKIP, "judge_v1.jsonl not generated yet")
        return
    sample = judged[:50]
    violations = [r.get("turn_id") for r in sample
                  if not isinstance(r.get("p_wrong"), (int, float))
                  or not 0.0 <= r.get("p_wrong", -1) <= 1.0
                  or not isinstance(r.get("verdict"), str)]
    if len(violations) <= 1:
        record("judge-schema", PASS, f"{len(violations)} violation(s) in {len(sample)} calls")
    else:
        record("judge-schema", FAIL,
               f"{len(violations)} violations in {len(sample)}: {violations[:5]}")


# ── 10. Judge mode consistency -- guards constraint 3 ──────────────────────

def check_judge_mode_consistency(judged: list[dict]) -> None:
    if not judged:
        record("judge-mode", SKIP, "judge_v1.jsonl not generated yet")
        return
    seen = {(r.get("model"), r.get("thinking")) for r in judged}
    if len(seen) > 1:
        record("judge-mode", FAIL,
               f"corpus mixes judge configurations {seen} -- feature 15 is fitted "
               "from one distribution and served from another")
    else:
        record("judge-mode", PASS, f"single configuration {seen.pop()}")


def main() -> int:
    corpus, judged = load(CORPUS), load(JUDGE)

    check_point_in_time()
    check_feature_order()
    check_null_semantics(corpus)
    check_label_balance(corpus)
    check_band_coverage(corpus)
    check_region_coverage(corpus)
    check_distribution_stability(corpus)
    check_judge_calibration(judged, corpus)
    check_judge_schema(judged)
    check_judge_mode_consistency(judged)

    width = max(len(c) for c, _, _ in results)
    print(f"\n21.3 bootstrap validation gate  --  corpus {len(corpus)} turns, "
          f"judged {len(judged)}\n" + "-" * (width + 60))
    for check, status, detail in results:
        print(f"  {status:4}  {check:<{width}}  {detail}")

    failed = [c for c, s, _ in results if s == FAIL]
    skipped = [c for c, s, _ in results if s == SKIP]
    print("-" * (width + 60))
    if failed:
        print(f"BLOCKED: {len(failed)} check(s) failed -- {', '.join(failed)}")
        print("Nothing is fitted until these pass.")
        return 1
    print(f"{len(results) - len(skipped)} passed, {len(skipped)} not yet runnable.")
    print("\nOne check is human and deliberately so: COEFFICIENT SANITY. Every fitted")
    print("coefficient's sign must be explicable by hand against 7.3. It is the")
    print("cheapest defence against a model that fits well and means nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
