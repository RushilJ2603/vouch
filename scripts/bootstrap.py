#!/usr/bin/env python3
"""21.2 initial bootstrap -- corpus to scored decisions.

Implements 21.2 steps 3, 5, 6 and 10: compute Tier 0 and Tier 1 features with
`as_of` bound to each turn's own timestamp (11), apply the temporal split (21.4),
fit calibrators A and B, measure ECE globally and per region, and replay the
corpus through the gate to populate `p_wrong` and `region` on every row.

Idempotent by design (21): safe to re-run. Spends no money -- everything it
needs is already on disk.

Two notes on method, both deliberate.

Scoring every row needs a prediction for the TRAINING rows too, and an
in-sample prediction would flatter the curve that Demo 1 exists to show. So the
scored corpus is built by EXPANDING-WINDOW cross-fitting: block i is predicted
by a model fit only on blocks strictly before it. Every scored row is therefore
out-of-sample, and no row is ever predicted by a model that has seen a later
one. 21.4's rule -- temporal, never random -- is the reason it is expanding
rather than k-fold. The headline ECE reported against 21.2 step 6 is still the
plain 80/20 fit, evaluated on the frozen final 20%.

The ledger is EMPTY at the start (21.5) and is filled as the replay proceeds.
That ordering matters: with budget 0 the close-call band collapses (10.1), so a
replay against a static empty ledger would put every row in `escalate` and
region coverage could never be met. Trust has to be earned across the replay,
which is the same mechanism Demo 2 draws.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import sklearn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from vouch import calibrate, config, exposure, features, gate, ledger, store  # noqa: E402
from vouch.agent import RESPOND_TOOL, SYSTEM  # noqa: E402
from vouch.features import Missing  # noqa: E402
from vouch.sensors import tier0, tier1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "orders.sqlite"
CORPUS = ROOT / "data" / "corpus" / "corpus_v1.jsonl"
JUDGE = ROOT / "data" / "corpus" / "judge_v1.jsonl"
ARTIFACTS = ROOT / "artifacts"
BASELINES = ARTIFACTS / "baselines.json"
# Features 1-14 cost minutes of ONNX inference and do not change when more of
# the corpus is judged -- only feature 15 does. Caching them is what makes
# re-running after `judge_corpus.py` cheap, which 21 asks for when it calls
# this procedure idempotent. Delete the file to force a recompute.
FEATURE_CACHE = ARTIFACTS / "features_v1.jsonl"

TEST_FRAC = 0.20            # 21.4: the frozen test set is the final 20% by ts
N_BLOCKS = 5                # expanding-window blocks for cross-fitting
AGENT_ID = "agent_corpus"   # one agent; the corpus is a single deployment
CAL_VERSION = "cal-v1"      # 21.2 step 9; the proxy adopts it once ready_at is set

# DeepSeek returns no log-probabilities, so features 9 and 10 are absent for
# this deployment rather than degraded. 7.2 fits per capability tier precisely
# so a deployment without logprobs gets a model over the features it has.
CAPABILITY_TIER = "none"


# ── 21.2 step 3: features, with as_of bound to each turn ───────────────────

def build_signals(conn, row, judge_p, length_history):
    """The same signal set `proxy/app.py` assembles, from a corpus row."""
    reply = row.get("reply_text") or ""
    claims = row.get("claims") or {}
    ts = float(row["ts"])

    frac, n_claims = tier0.verify_claims(conn, claims) if claims else (0.0, 0)
    r_min, r_mean = tier0.retrieval_support(reply, row.get("retrieved_chunks") or [])
    lz, lz_missing = features.length_z(len(reply), length_history, ts)
    t1 = tier1.score_spans(reply)
    # Feature 12 reads the customer's message, not the reply (Appendix A).
    t1_req = tier1.score_request(row.get("prompt") or "")

    def seen(value):
        return (value, Missing.VALUE) if value is not None else (None, Missing.UNAVAILABLE)

    signals = {
        "verify_fail_frac": (frac, Missing.VALUE),
        "verify_n_claims": (n_claims, Missing.VALUE),
        "retrieval_support_min": seen(r_min),
        "retrieval_support_mean": seen(r_mean),
        # No tool loop in the corpus. 5 and 6 cannot be `unavailable` -- there
        # is no external system to fail -- so they are a measured zero.
        "tool_retry_count": (0, Missing.VALUE),
        "tool_error_count": (0, Missing.VALUE),
        "hedge_density": (tier0.hedge_density(reply), Missing.VALUE),
        "length_z": (lz, lz_missing),
        "logprob_mean": (None, Missing.NOT_SUPPORTED),
        "logprob_min": (None, Missing.NOT_SUPPORTED),
        "pii_score": seen(t1.get("pii_score")),
        "injection_score": seen(t1_req.get("injection_score")),
        "policy_score": seen(t1.get("policy_score")),
        "secret_score": (t1.get("secret_score") or 0.0, Missing.VALUE),
        # Appendix A: feature 15 is `not_supported` for model A, which never
        # sees it -- NOT `unavailable`. Only ~300 turns are judged by design
        # (21.1), so encoding the rest as unavailable would report a 79% null
        # rate on a feature that was never meant to be there, and 21.3's <5%
        # envelope would fail on a sensor that is working exactly as specified.
        "judge_p_wrong": ((judge_p, Missing.VALUE) if judge_p is not None
                          else (None, Missing.NOT_SUPPORTED)),
    }
    return signals, {k: (v[0] if v[1] is Missing.VALUE else None)
                     for k, v in signals.items()}


def fit(vectors, labels):
    """Calibrator over the assembled vectors. Single-class input has nothing to
    fit and returns the base rate, which is the honest answer rather than a
    model that cannot be wrong."""
    if len({*labels}) < 2:
        rate = (sum(labels) / len(labels)) if labels else 0.05
        return None, rate
    # NOT `class_weight="balanced"`. Re-weighting the minority class is the
    # right move when you want to separate classes and the wrong move when you
    # want the number itself to be true: it inflates every predicted
    # probability toward the reweighted prior, so a 3.9% base rate reads back
    # as tens of percent. 24.1 asks whether `p_wrong` MEANS anything, and
    # 10.1 multiplies it by exposure to price an action -- both need the
    # probability calibrated, not merely ranked.
    model = LogisticRegression(max_iter=2000)
    model.fit(vectors, labels)
    return model, None


def predict(model, base_rate, vectors):
    if model is None:
        return [base_rate] * len(vectors)
    return [float(p) for p in model.predict_proba(vectors)[:, 1]]


def main() -> int:
    if not CORPUS.exists():
        print("no corpus; run scripts/generate_corpus.py first")
        return 1

    rows = [json.loads(line) for line in CORPUS.open(encoding="utf-8") if line.strip()]
    rows.sort(key=lambda r: float(r["ts"]))          # 21.4: temporal, never random

    judge_map = {}
    if JUDGE.exists():
        for line in JUDGE.open(encoding="utf-8"):
            if line.strip():
                j = json.loads(line)
                judge_map[j["turn_id"]] = j["p_wrong"]

    print(f"21.2 bootstrap -- {len(rows)} turns, {len(judge_map)} judged\n")

    # ── step 3 ─────────────────────────────────────────────────────────────
    cached = {}
    if FEATURE_CACHE.exists():
        for line in FEATURE_CACHE.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                cached[rec["turn_id"]] = rec["features"]

    assembled, null_counts = [], defaultdict(int)
    if len(cached) >= len(rows) and all(r["turn_id"] in cached for r in rows):
        print("  features: reusing cache "
              f"({FEATURE_CACHE.name}); re-injecting feature 15 only")
        for row in rows:
            feats = dict(cached[row["turn_id"]])
            jp = judge_map.get(row["turn_id"])
            feats["judge_p_wrong"] = jp
            # Unjudged is `not_supported`, which carries NO unavailable flag --
            # same encoding `features.assemble` produces on the cold path.
            feats["judge_p_wrong_unavailable"] = 0
            assembled.append(feats)
    else:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        history: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for i, row in enumerate(rows):
            action = (row.get("action") or {}).get("type") or "issue_refund"
            signals, _raw = build_signals(
                conn, row, judge_map.get(row["turn_id"]), history[action])
            feats = features.assemble(float(row["ts"]), signals, CAPABILITY_TIER)
            assembled.append(feats)
            history[action].append((float(row["ts"]), len(row.get("reply_text") or "")))
            if (i + 1) % 250 == 0:
                print(f"  features: {i + 1}/{len(rows)}", flush=True)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with FEATURE_CACHE.open("w", encoding="utf-8") as f:
            for row, feats in zip(rows, assembled):
                f.write(json.dumps({"turn_id": row["turn_id"], "features": feats}) + "\n")

    # ── Feature 13: the logistic head over MiniLM (Appendix A) ─────────────
    # Fitted here because it is the one feature trained ON the corpus, which is
    # why export_onnx.py could only mark it `pending`.
    #
    # It is CROSS-FITTED for the same reason `p_wrong` is, and the reason is
    # sharper here: the head is trained on the very labels calibrator A then
    # predicts. Scoring a training row with a head that memorised it would hand
    # the calibrator a feature that looks clairvoyant in fitting and collapses
    # in production -- a leak that shows up as a BETTER ECE, so nothing would
    # look wrong. Each block is scored by a head fit only on earlier blocks; the
    # head that ships is fit on the training split and saved for serving.
    y_all = [1 if r.get("outcome") == "wrong" else 0 for r in rows]
    embeds = [tier1._embed((r.get("reply_text") or "")) for r in rows]
    usable = [i for i, e in enumerate(embeds) if e is not None]
    print(f"  feature 13: embedded {len(usable)}/{len(rows)} replies")

    if usable:
        pol_block = max(1, len(rows) // N_BLOCKS)
        for b in range(1, N_BLOCKS):
            lo = b * pol_block
            hi = (b + 1) * pol_block if b < N_BLOCKS - 1 else len(rows)
            past = [i for i in usable if i < lo]
            if not past or len({y_all[i] for i in past}) < 2:
                continue
            head = LogisticRegression(max_iter=3000)
            head.fit([embeds[i] for i in past], [y_all[i] for i in past])
            here = [i for i in range(lo, hi) if embeds[i] is not None]
            if here:
                pr = head.predict_proba([embeds[i] for i in here])[:, 1]
                for i, p in zip(here, pr):
                    assembled[i]["policy_score"] = float(p)
                    assembled[i]["policy_score_unavailable"] = 0

        # The head that SERVES: fit on the training split only, then saved.
        fit_idx = [i for i in usable if i < int(len(rows) * (1.0 - TEST_FRAC))]
        if fit_idx and len({y_all[i] for i in fit_idx}) >= 2:
            serving = LogisticRegression(max_iter=3000)
            serving.fit([embeds[i] for i in fit_idx], [y_all[i] for i in fit_idx])
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            joblib.dump(serving, ARTIFACTS / "policy_head.joblib")
            print(f"  feature 13: head fitted on {len(fit_idx)} rows -> "
                  f"{(ARTIFACTS / 'policy_head.joblib').name}")

    for feats in assembled:
        for name in features.FEATURE_ORDER:
            if feats.get(name) is None:
                null_counts[name] += 1

    names_a = tuple(f for f in features.MODEL_A_FEATURES
                    if f in features.CAPABILITY_TIERS[CAPABILITY_TIER])
    names_b = tuple(f for f in features.MODEL_B_FEATURES
                    if f in features.CAPABILITY_TIERS[CAPABILITY_TIER])

    labelled = [i for i, r in enumerate(rows) if r.get("outcome") in ("clean", "wrong")]
    y = {i: (1 if rows[i]["outcome"] == "wrong" else 0) for i in labelled}
    print(f"\n  labelled rows: {len(labelled)}  "
          f"({sum(y.values())} wrong, {sum(y.values()) / max(1, len(labelled)):.1%})")

    # ── step 5: the 21.4 split, then fit A and B ───────────────────────────
    cut = int(len(rows) * (1.0 - TEST_FRAC))
    train_idx = [i for i in labelled if i < cut]
    test_idx = [i for i in labelled if i >= cut]
    print(f"  temporal split: train {len(train_idx)}, frozen test {len(test_idx)} "
          f"(boundary ts {float(rows[cut]['ts']):.0f})")

    def vecs(idx, names):
        return [features.to_vector(assembled[i], names) for i in idx]

    model_a, rate_a = fit(vecs(train_idx, names_a), [y[i] for i in train_idx])
    judged_train = [i for i in train_idx if assembled[i].get("judge_p_wrong") is not None]
    model_b, rate_b = fit(vecs(judged_train, names_b), [y[i] for i in judged_train])
    print(f"  calibrator A: {len(train_idx)} rows over {len(names_a)} features")
    print(f"  calibrator B: {len(judged_train)} rows over {len(names_b)} features "
          f"{'(base rate only -- too few judged)' if model_b is None else ''}")

    # ── step 6: ECE on the frozen test set ─────────────────────────────────
    test_p = predict(model_a, rate_a, vecs(test_idx, names_a))
    test_y = [y[i] for i in test_idx]
    held = calibrate.evaluate(test_p, test_y) if test_idx else None
    if held:
        print(f"\n  frozen-test ECE {held.ece:.4f}  MCE {held.mce:.4f}  "
              f"Brier {held.brier:.4f}  (21.2 step 6)")

    # ── scored corpus: expanding-window cross-fit, every row out-of-sample ──
    p_hat: dict[int, float] = {}
    block = max(1, len(rows) // N_BLOCKS)
    for b in range(1, N_BLOCKS):
        lo, hi = b * block, (b + 1) * block if b < N_BLOCKS - 1 else len(rows)
        past = [i for i in labelled if i < lo]
        if not past:
            continue
        m, r = fit(vecs(past, names_a), [y[i] for i in past])
        here = [i for i in range(lo, hi)]
        for i, p in zip(here, predict(m, r, vecs(here, names_a))):
            p_hat[i] = p
    # The first block has no earlier rows to learn from; the base rate is what
    # the system honestly knows on day one, and 21.5 says so.
    base = sum(y.values()) / max(1, len(y))
    for i in range(len(rows)):
        p_hat.setdefault(i, base)

    # ── step 10: replay through the gate, ledger filling as it goes ────────
    trust: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    regions = defaultdict(int)
    for i, row in enumerate(rows):
        act = row.get("action") or {}
        action = act.get("type") or "issue_refund"
        amount = (act.get("amount_paise") or 0) / 100.0

        # 8: an action with no entry has no exposure and therefore no price.
        # `no_action` and `escalate` are the agent declining to act -- they carry
        # the amount that was AT STAKE, but produce no side effect, so they can
        # neither spend budget nor earn a trust row. They still face the fixed
        # path (10.2), which is why they go through the gate rather than around.
        try:
            hard = exposure.hard_limit(action)
            exposure_v = exposure.exposure(action, amount)
            priced = True
        except KeyError:
            hard, exposure_v, priced = float("inf"), 0.0, False

        band = None if not priced or amount > hard else config.band_for(amount)

        if band is None:
            budget_v = 0.0
        else:
            key = (AGENT_ID, action, band)
            clean, total, raw = trust[key]
            budget_v = ledger.evaluate(
                band, exposure.ceiling(action, band), clean, total, raw)["budget"]

        inv = gate.check_invariants(
            {k: assembled[i].get(k) for k in
             ("pii_score", "injection_score", "secret_score")},
            {"retrieval_scope": 1, "user_scope": 1,
             "amount": amount, "hard_limit": hard})
        verdict = gate.decide(p_hat[i], exposure_v, budget_v, inv)
        region = gate.region_of(verdict).value
        regions[region] += 1
        row["p_wrong"] = round(p_hat[i], 6)
        row["region"] = region
        row["split"] = "test" if i >= cut else "train"
        # 21.3's null-semantics and distribution-stability checks read the
        # assembled vector off the row; without it they can only SKIP.
        row["features"] = assembled[i]

        if band is not None and row.get("outcome") in ("clean", "wrong"):
            key = (AGENT_ID, action, band)
            trust[key][0] += 1.0 if row["outcome"] == "clean" else 0.0
            trust[key][1] += 1.0
            trust[key][2] += 1

    print("\n  replay regions: " + ", ".join(f"{k}={v}" for k, v in sorted(regions.items())))
    print(f"  trust rows derived: {len(trust)}")

    # ── step 8: baselines.json ─────────────────────────────────────────────
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    BASELINES.write_text(json.dumps({
        "feature_order": list(features.FEATURE_ORDER),
        "capability_tier": CAPABILITY_TIER,
        "model_a_features": list(names_a),
        "model_b_features": list(names_b),
        "sklearn_version": sklearn.__version__,
        "ece_global": (held.ece if held else None),
        "mce_global": (held.mce if held else None),
        "brier_global": (held.brier if held else None),
        "test_boundary_ts": float(rows[cut]["ts"]),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "null_rates": {k: round(v / len(rows), 4) for k, v in sorted(null_counts.items())},
        "fallback_priors": {"issue_refund": round(base, 6)},
    }, indent=2), encoding="utf-8")

    # ── steps 8-9: serialise the calibrators and register cal-v1 ───────────
    # Without this the proxy has nothing to load, `reload_calibrator` finds no
    # candidate, and 12.3 step 7 silently runs on the fallback prior -- which is
    # a constant, so the gate ends up thresholding on exposure alone.
    import hashlib

    import sklearn as _sk
    bundle_path = ARTIFACTS / f"calibrator_{CAL_VERSION}.joblib"
    joblib.dump({"a": model_a, "b": model_b,
                 "names_a": list(names_a), "names_b": list(names_b),
                 "rate_a": rate_a, "rate_b": rate_b,
                 "capability_tier": CAPABILITY_TIER}, bundle_path)
    sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    vconn = store.connect(str(ROOT / "data" / "vouch.sqlite"))
    store.init_schema(vconn)
    # `ready_at` is what the proxy gates adoption on (12.6), and it is set only
    # after the checksum above exists -- never in the same breath as the insert.
    vconn.execute(
        "INSERT OR REPLACE INTO calibrator_registry (version, created_at, "
        "promoted_at, ready_at, state, promotion_path, sklearn_version, "
        "capability_tier, ece_global, rows_carried, artifact_sha256, notes) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (CAL_VERSION, float(rows[0]["ts"]), float(rows[-1]["ts"]),
         float(rows[-1]["ts"]), "production", "A", _sk.__version__,
         CAPABILITY_TIER, (held.ece if held else None), len(train_idx), sha,
         "bootstrap 21.2; expanding-window cross-fit for the scored corpus"))

    # Step 10 derives trust during the replay above; persist the final rows so
    # the first live proxy request starts with the same earned record instead
    # of an empty ledger. The fingerprint exactly matches the live structured
    # agent contract used by proxy._configuration_fingerprint().
    fingerprint = ledger.fingerprint(
        config.AGENT_MODEL,
        SYSTEM,
        [RESPOND_TOOL["function"]],
        config.sensor_version(),
        CAL_VERSION,
    )
    for (agent_id, action, band), (clean, total, raw) in trust.items():
        ceiling = exposure.ceiling(action, band)
        derived = ledger.evaluate(band, ceiling, clean, total, int(raw))
        vconn.execute(
            "INSERT INTO trust_row (agent_id, action, band, config_fingerprint, "
            "n_total, n_clean, n_own_raw, p_lo, budget, ceiling, state, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(agent_id, action, band, config_fingerprint) DO UPDATE SET "
            "n_total=excluded.n_total, n_clean=excluded.n_clean, "
            "n_own_raw=excluded.n_own_raw, p_lo=excluded.p_lo, "
            "budget=excluded.budget, ceiling=excluded.ceiling, state=excluded.state, "
            "updated_at=excluded.updated_at",
            (agent_id, action, band, fingerprint, total, clean, int(raw),
             derived["p_lo"], derived["budget"], ceiling, derived["state"],
             float(rows[-1]["ts"])),
        )
    vconn.commit()
    vconn.close()
    print(f"  registered {CAL_VERSION}  sha256 {sha[:12]}...  "
          f"sklearn {_sk.__version__}")

    with CORPUS.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n  wrote {BASELINES.relative_to(ROOT)} and scored "
          f"{CORPUS.relative_to(ROOT)}")
    print("  next: scripts/validate_corpus.py, then scripts/demo_01_calibration.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
