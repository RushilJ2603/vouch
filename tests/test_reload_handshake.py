"""Section 12.6 -- the calibrator reload handshake.

A model swap must never leave the proxy serving a half-written artifact, and
must never produce a window in which some decisions used the old calibrator
and some the new one without that being recorded.
"""
import importlib.metadata as md

import pytest

from vouch import store
from vouch.proxy import app

RUNNING = md.version("scikit-learn")


@pytest.fixture
def conn(tmp_path, monkeypatch):
    # 12.6 now verifies the ARTIFACT, not just the registry row: the file must
    # exist and its sha256 must match what was registered. So the tests write a
    # real one instead of a placeholder hash, which is the contract the proxy
    # actually enforces.
    monkeypatch.setattr(app, "_ARTIFACTS", tmp_path)
    c = store.connect(":memory:")
    store.init_schema(c)
    for k in ("calibrator_version", "degraded_reason", "calibrator_a", "calibrator_b"):
        app.STATE.pop(k, None)
    app.STATE["conn"] = c
    yield c
    c.close()


def write_artifact(version, *, corrupt=False):
    """A real joblib bundle plus its true digest. Returns the sha to register."""
    import hashlib

    import joblib
    path = app._ARTIFACTS / f"calibrator_{version}.joblib"
    joblib.dump({"a": None, "b": None, "names_a": [], "names_b": [],
                 "rate_a": 0.04, "rate_b": None, "capability_tier": "none"}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha-does-not-match" if corrupt else digest


def register(conn, version, *, ready, state="production", sklearn=RUNNING,
             artifact=True, corrupt=False):
    sha = write_artifact(version, corrupt=corrupt) if artifact else f"sha-{version}"
    conn.execute(
        "INSERT INTO calibrator_registry (version, created_at, ready_at, state, "
        "sklearn_version, capability_tier, artifact_sha256) VALUES (?,?,?,?,?,?,?)",
        (version, 0.0, ready, state, sklearn, "none", sha))
    conn.commit()


# ── ready_at is the gate ───────────────────────────────────────────────────

def test_a_calibrator_without_ready_at_is_never_adopted():
    """`ready_at` is set only AFTER the worker's integrity check. Reloading
    before it means serving a half-written artifact."""
    c = store.connect(":memory:")
    store.init_schema(c)
    app.STATE["conn"] = c
    # artifact=False: this test never reaches the artifact check, and writing a
    # real one would land in the repo's artifacts/ -- this test does not take
    # the fixture that redirects it to tmp_path.
    register(c, "cal-v2", ready=None, artifact=False)
    assert app.find_ready_calibrator(c) is None
    assert app.reload_calibrator(c, RUNNING) is None


def test_a_ready_calibrator_is_adopted(conn):
    register(conn, "cal-v2", ready=100.0)
    assert app.reload_calibrator(conn, RUNNING) == "cal-v2"
    assert app.STATE["calibrator_version"] == "cal-v2"


def test_the_newest_ready_version_wins(conn):
    register(conn, "cal-v2", ready=100.0)
    register(conn, "cal-v3", ready=200.0)
    assert app.reload_calibrator(conn, RUNNING) == "cal-v3"


def test_a_non_production_row_is_ignored(conn):
    register(conn, "cal-v9", ready=500.0, state="shadow")
    assert app.reload_calibrator(conn, RUNNING) is None


# ── The hard refusal ───────────────────────────────────────────────────────

def test_an_sklearn_mismatch_refuses_and_keeps_the_previous(conn):
    """A joblib fitted under a different scikit-learn loads WITHOUT ERROR and
    scores differently. Serving it would corrupt every subsequent trust update,
    so the swap is refused and the cached calibrator stays."""
    register(conn, "cal-v2", ready=100.0)
    app.reload_calibrator(conn, RUNNING)
    register(conn, "cal-v3", ready=200.0, sklearn="0.24.1")

    assert app.reload_calibrator(conn, RUNNING) is None
    assert app.STATE["calibrator_version"] == "cal-v2", "must keep the previous cache"
    assert app.STATE["degraded_reason"] == "SKLEARN_MISMATCH"


def test_a_good_swap_clears_the_degraded_flag(conn):
    register(conn, "cal-v2", ready=100.0, sklearn="0.24.1")
    app.reload_calibrator(conn, RUNNING)
    assert app.STATE.get("degraded_reason") == "SKLEARN_MISMATCH"

    register(conn, "cal-v3", ready=200.0)
    assert app.reload_calibrator(conn, RUNNING) == "cal-v3"
    assert "degraded_reason" not in app.STATE


# ── No blended window ──────────────────────────────────────────────────────

def test_reloading_the_same_version_is_a_no_op(conn):
    register(conn, "cal-v2", ready=100.0)
    assert app.reload_calibrator(conn, RUNNING) == "cal-v2"
    assert app.reload_calibrator(conn, RUNNING) is None, "must not churn on every poll"


def test_the_swap_reloads_the_ledger_for_the_new_fingerprint(conn):
    conn.execute(
        "INSERT INTO trust_row (agent_id, action, band, config_fingerprint, "
        "ceiling, updated_at) VALUES (?,?,?,?,?,?)",
        ("refund-agent", "issue_refund", "10k-50k", "fp-new", 720.0, 0.0))
    conn.commit()
    register(conn, "cal-v2", ready=100.0)
    app.reload_calibrator(conn, RUNNING)
    assert ("refund-agent", "issue_refund", "10k-50k", "fp-new") in app.STATE["ledger"]


def test_every_decision_records_its_calibrator_version():
    """That is what makes the transition window identifiable and excludable
    from that night's calibration analysis, rather than silently blended."""
    assert "calibrator_version" in app._DECISION_SQL


def test_the_poll_is_an_interval_not_a_ttl():
    """A one-hour TTL means a calibrator promoted at 02:47 is picked up
    somewhere before 03:47, and that night's ECE averages two models."""
    from vouch import config
    assert config.load()["proxy"]["calibrator_poll_sec"] == 60


# ── The artifact itself, added 2026-08-28 ──────────────────────────────────
# Until this date `reload_calibrator` recorded the registry metadata, reloaded
# the ledger, and never loaded the models. `STATE["calibrator_a"]` stayed None,
# nothing on the request path noticed, and 12.3 step 7 ran on the fallback
# prior -- a CONSTANT -- for every request. The gate then thresholded on
# exposure alone and fired Tier 2 on 40% of traffic against a 2-5% target.

def test_a_ready_row_with_no_artifact_on_disk_is_refused(conn):
    """A registry row is a promise about a file. If the file is not there the
    promise is broken, and adopting the version would leave the proxy claiming
    a calibrator it does not have."""
    register(conn, "cal-v2", ready=100.0, artifact=False)
    assert app.reload_calibrator(conn, RUNNING) is None
    assert app.STATE.get("degraded_reason") == "CALIBRATOR_ARTIFACT_MISSING"
    assert app.STATE.get("calibrator_version") != "cal-v2"


def test_a_checksum_mismatch_is_refused(conn):
    """Half-written or altered. Scoring with it is worse than not scoring at
    all, because it produces numbers and they are wrong -- and nothing errors."""
    register(conn, "cal-v2", ready=100.0, corrupt=True)
    assert app.reload_calibrator(conn, RUNNING) is None
    assert app.STATE.get("degraded_reason") == "CALIBRATOR_CHECKSUM_MISMATCH"


def test_adoption_actually_loads_the_models(conn):
    """The bug this whole block exists for: the version was recorded without
    the models ever being loaded."""
    register(conn, "cal-v2", ready=100.0)
    assert app.reload_calibrator(conn, RUNNING) == "cal-v2"
    assert "calibrator_names_a" in app.STATE
    assert app.STATE["calibrator_rate_a"] == 0.04
