"""The proxy -- the only integration surface (12).

**This module is the ONLY place in `src/vouch/` permitted to read wall clock.**
Everything downstream takes `as_of` explicitly. `make lint` enforces that with
a grep, and the exemption lives here because this is where the present moment
genuinely is the answer: a request is happening now (11).

One uvicorn worker, deliberately. The ledger is a dict in this process and the
sub-microsecond read is the entire latency argument; multi-worker needs a
shared cache and a different, slower claim (20.2).
"""
from __future__ import annotations

import asyncio
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import config, exposure, features, gate, ladder, ledger, store
from ..gate import Verdict
from ..sensors import tier0, tier1, tier2
from .stream import StreamingScorer, UpstreamUnavailable, consume, provider_upstream


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await startup()
    try:
        yield
    finally:
        await shutdown()


app = FastAPI(title="Vouch", version="0.1.0", lifespan=lifespan)

NON_EXECUTING_ACTIONS = frozenset({"no_action", "escalate"})

# `/artifacts` inside the container; the repo copy outside it -- the same
# convention `sensors/tier1.py` uses for the encoders.
from pathlib import Path as _Path  # noqa: E402

_ARTIFACTS = (_Path("/artifacts") if _Path("/artifacts").is_dir()
              else _Path(__file__).resolve().parents[3] / "artifacts")

# ── 12.2 Startup cache ─────────────────────────────────────────────────────
# Seeded so a replay reproduces the same hold-out. 11 forbids wall-clock
# entropy anywhere the corpus is replayed, and an unseeded sampler would make
# two replays of the same corpus disagree about which rows a human reviewed.
_FORCED_REVIEW = random.Random(20260825)

STATE: dict[str, Any] = {
    "ledger": {},              # (agent, action, band, fp) -> row; in-memory dict
    "calibrator_a": None,
    "calibrator_b": None,
    "baselines": None,
    "fallback_priors": {},     # median p_wrong per action -- the 16 degradation path
    "started_at": None,
    "conn": None,
    "orders_conn": None,
    # Feature 8 needs THIS ACTION's reply lengths strictly before as_of (11).
    # Held in memory beside the ledger: a wall-clock-free running history, so a
    # replay of the same traffic produces the same length_z.
    "length_history": {},      # action -> list[(ts, length)]
    "capability_tier": "none",
}


class VouchBlock(BaseModel):
    agent_id: str
    user_scope: list[str] = Field(default_factory=list)
    policy_mode: str = "adaptive"
    action_hint: str | None = None
    amount_paise: int | None = None

    # Everything below feeds the feature vector Calibrator A is fitted on
    # (Appendix A). Until 2026-08-28 none of it existed on the request, so
    # `_run_tier0` read `claims` off a model that had no such field and got
    # `{}` every time: features 1 and 2 were not degraded, they were pinned to
    # zero, and features 3-6 were never computed at all. A calibrator fitted
    # with those present cannot be served against a request without them --
    # that is the train/serve skew 2.8's version check exists to prevent, one
    # level up, so step 7 could not be wired until this was carried.
    claims: dict | None = None              # 7.1: structured TOOL ARGUMENTS
    retrieved_chunks: list[str] | None = None   # features 3, 4
    tool_calls: list[dict] | None = None        # features 5, 6


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    tools: list[dict] = Field(default_factory=list)
    vouch: VouchBlock | None = None
    stream: bool = False


class OutcomeReport(BaseModel):
    decision_id: str
    outcome: str = Field(pattern="^(clean|wrong)$")
    source: str


# ── 12.6 Calibrator reload handshake ───────────────────────────────────────
# A model swap must never leave the proxy serving a half-written artifact, and
# must never produce a window in which some decisions used the old calibrator
# and some the new one without that being recorded.
#
# `ready_at` is set by the worker ONLY after its integrity check passes, and
# the proxy reloads ONLY when `ready_at IS NOT NULL`. Every decision row
# records the calibrator_version that produced it, so the transition window is
# identifiable and excludable from that night's analysis rather than silently
# blended into it.
#
# Without this a naive one-hour TTL means a calibrator promoted at 02:47 is
# picked up somewhere before 03:47. In those minutes some decisions are scored
# by v4 and some by v5, that night's ECE is a meaningless average of two
# different models, and if it looks bad the team debugs the wrong one.


def find_ready_calibrator(conn) -> dict | None:
    """The newest version that has PASSED integrity and is not already live."""
    row = conn.execute(
        "SELECT version, artifact_sha256, sklearn_version, capability_tier "
        "FROM calibrator_registry "
        "WHERE ready_at IS NOT NULL AND state = 'production' "
        "ORDER BY ready_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    candidate = dict(row)
    if candidate["version"] == STATE.get("calibrator_version"):
        return None
    return candidate


def reload_calibrator(conn, running_sklearn: str) -> str | None:
    """Verify, then swap atomically. Returns the version adopted, or None.

    A version mismatch REFUSES the swap and keeps the previously cached
    calibrator (16, `SKLEARN_MISMATCH`). A joblib fitted under a different
    scikit-learn loads without error and scores differently, which would
    corrupt every subsequent trust update.
    """
    candidate = find_ready_calibrator(conn)
    if candidate is None:
        return None

    if candidate["sklearn_version"] != running_sklearn:
        STATE["degraded_reason"] = "SKLEARN_MISMATCH"
        STATE["log_errors"] = STATE.get("log_errors", 0)
        return None

    # Load and verify the artifact BEFORE any rebind. Until 2026-08-28 this
    # function recorded the metadata and reloaded the ledger without ever
    # loading the models, so `calibrator_a` stayed None and step 7 quietly ran
    # on the fallback prior -- a constant -- for every request.
    import hashlib

    import joblib
    path = _ARTIFACTS / f"calibrator_{candidate['version']}.joblib"
    if not path.exists():
        STATE["degraded_reason"] = "CALIBRATOR_ARTIFACT_MISSING"
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != candidate["artifact_sha256"]:
        # A half-written or altered artifact is refused outright. Scoring with
        # one is worse than not scoring: it produces numbers, and they are wrong.
        STATE["degraded_reason"] = "CALIBRATOR_CHECKSUM_MISMATCH"
        return None
    try:
        bundle = joblib.load(path)
    except Exception:
        STATE["degraded_reason"] = "CALIBRATOR_UNREADABLE"
        return None

    # Atomic from the request path's point of view: a single rebind, so no
    # request can observe A from one version and B from another.
    STATE["calibrator_a"] = bundle.get("a")
    STATE["calibrator_b"] = bundle.get("b")
    STATE["calibrator_names_a"] = tuple(bundle.get("names_a") or ())
    STATE["calibrator_names_b"] = tuple(bundle.get("names_b") or ())
    STATE["calibrator_rate_a"] = bundle.get("rate_a")
    STATE["calibrator_rate_b"] = bundle.get("rate_b")
    STATE["calibrator_version"] = candidate["version"]
    STATE["capability_tier"] = candidate["capability_tier"]
    STATE["calibrator_sha256"] = candidate["artifact_sha256"]
    STATE["ledger"] = {
        (r["agent_id"], r["action"], r["band"], r["config_fingerprint"]): dict(r)
        for r in conn.execute("SELECT * FROM trust_row")
    }
    STATE.pop("degraded_reason", None)
    return candidate["version"]


async def _calibrator_poller() -> None:
    """60-second poll (12.2 item 2). Deliberately not a TTL."""
    interval = float(config.load()["proxy"]["calibrator_poll_sec"])
    import importlib.metadata as md
    running = md.version("scikit-learn")
    while True:
        await asyncio.sleep(interval)
        try:
            reload_calibrator(STATE["conn"], running)
        except Exception as exc:
            STATE["log_last_error"] = f"reload: {type(exc).__name__}: {exc}"


def start_log_writer() -> None:
    """Called at startup, and by any harness that drives the handler directly."""
    global LOG_QUEUE
    LOG_QUEUE = asyncio.Queue(maxsize=config.load()["proxy"]["log_queue_max"])
    STATE["log_writer_task"] = asyncio.get_running_loop().create_task(_log_writer())


async def startup() -> None:
    STATE["started_at"] = time.time()
    start_log_writer()
    db_path = os.environ.get("VOUCH_DB_PATH", "data/vouch.sqlite")
    orders_path = os.environ.get("VOUCH_ORDERS_DB_PATH", "data/orders.sqlite")
    conn = store.connect(db_path)
    store.init_schema(conn)
    STATE["conn"] = conn
    # Tier 0 is dispatched to a worker thread and reads the system of record,
    # not the operational Vouch database. The connection is read-only and
    # permits that deliberate thread hand-off; SQLite's serialized mode guards
    # concurrent readers while no request can mutate the seeded record.
    STATE["orders_conn"] = store.connect(
        orders_path, read_only=True, check_same_thread=False
    )
    STATE["ledger"] = {
        (r["agent_id"], r["action"], r["band"], r["config_fingerprint"]): dict(r)
        for r in store.load_all_trust_rows(conn)
    }
    import importlib.metadata as md
    reload_calibrator(conn, md.version("scikit-learn"))
    STATE["calibrator_poller_task"] = asyncio.create_task(_calibrator_poller())


async def shutdown() -> None:
    task = STATE.pop("calibrator_poller_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    if LOG_QUEUE is not None:
        await LOG_QUEUE.join()
    writer = STATE.pop("log_writer_task", None)
    if writer is not None:
        writer.cancel()
        try:
            await writer
        except asyncio.CancelledError:
            pass
    for name in ("orders_conn", "conn"):
        conn = STATE.get(name)
        if conn is not None:
            conn.close()
            STATE[name] = None


def ledger_lookup(key: tuple) -> float:
    """A missing row means budget 0, not an error (12.3 step 3)."""
    row = STATE["ledger"].get(key)
    return float(row["budget"]) if row else 0.0


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "uptime_sec": time.time() - (STATE["started_at"] or time.time()),
        "ledger_rows": len(STATE["ledger"]),
        "calibrator_a": STATE["calibrator_a"] is not None,
        "calibrator_b": STATE["calibrator_b"] is not None,
        "calibrator_version": STATE.get("calibrator_version"),
        "tier2_model": config.TIER2_MODEL,
        "tier2_thinking": config.TIER2_THINKING,
        "sensor_version": config.sensor_version(),
        "log_dropped": STATE.get("log_dropped", 0),
        "log_errors": STATE.get("log_errors", 0),
        "log_queue_depth": LOG_QUEUE.qsize() if LOG_QUEUE is not None else 0,
        "log_spill_count": STATE.get("log_spill_count", 0),
        "log_last_error": STATE.get("log_last_error"),
        "tier1_errors": STATE.get("tier1_errors", 0),
        "tier1_last_error": STATE.get("tier1_last_error"),
        "tier2_unavailable": STATE.get("tier2_unavailable", 0),
        "tier2_last_degraded_reason": STATE.get("tier2_last_degraded_reason"),
    }


@app.get("/v1/vouch/ledger")
async def read_ledger(agent_id: str | None = None, action: str | None = None) -> list[dict]:
    return [r for k, r in STATE["ledger"].items()
            if (agent_id is None or k[0] == agent_id) and (action is None or k[1] == action)]


@app.get("/v1/vouch/decisions")
async def read_decisions(since: float = 0.0, verdict: str | None = None,
                         limit: int = 100) -> list[dict]:
    sql = "SELECT * FROM decision WHERE ts >= ?"
    args: list = [since]
    if verdict:
        sql += " AND verdict = ?"
        args.append(verdict)
    sql += " ORDER BY ts DESC LIMIT ?"
    args.append(min(limit, 1000))
    return [dict(r) for r in STATE["conn"].execute(sql, args).fetchall()]


@app.post("/v1/vouch/outcome")
async def report_outcome(report: OutcomeReport) -> dict:
    from ..worker.outcomes import EVIDENCE_WEIGHT
    if report.source not in EVIDENCE_WEIGHT:
        raise HTTPException(422, f"unregistered outcome source {report.source!r}")
    # The proxy NEVER writes decision.outcome directly -- it enqueues an event
    # and worker step 1 resolves it against the horizon (Appendix C).
    STATE["conn"].execute(
        "INSERT INTO outcome_event "
        "(decision_id, reported_ts, outcome, source, weight) VALUES (?,?,?,?,?)",
        (report.decision_id, time.time(), report.outcome, report.source,
         EVIDENCE_WEIGHT[report.source]),
    )
    STATE["conn"].commit()
    return {"accepted": True}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> dict:
    return await evaluate(req)


async def evaluate(req: ChatRequest, upstream_override=None) -> dict:
    """12.3, steps 1 to 13. The ~2 ms is the sum of the per-step budgets."""
    t_start = time.perf_counter()
    ts = time.time()                       # THE reference time for this decision (11)
    shadow_mode = req.vouch is None or bool(config.load()["sampling"]["shadow_mode"])
    if req.vouch is None:
        req.vouch = VouchBlock(agent_id="shadow-default")

    action = req.vouch.action_hint or "issue_refund"
    try:
        policy = config.policy_profile(req.vouch.policy_mode)
    except KeyError as exc:
        raise HTTPException(422, f"unknown policy mode {req.vouch.policy_mode!r}") from exc
    amount = (req.vouch.amount_paise or 0) / 100
    has_side_effect = action not in NON_EXECUTING_ACTIONS
    if has_side_effect:
        try:
            spec = config.action_spec(action)
        except KeyError:
            # An action with no entry has no exposure and therefore no price (8).
            raise HTTPException(422, f"action {action!r} is not registered; fail closed")
        hard_limit = float(spec["hard_limit"])
        over_limit = amount > hard_limit
        band = None if over_limit else config.band_for(amount)
    else:
        hard_limit = float("inf")
        over_limit = False
        band = None

    fingerprint = STATE.get("fingerprint") or _configuration_fingerprint(req)
    earned_budget = 0.0 if band is None else ledger_lookup(
        (req.vouch.agent_id, action, band, fingerprint))
    budget = earned_budget * float(policy["budget_multiplier"])

    # Step 4: dispatch Tier 0, do NOT await it.
    t0_task = asyncio.create_task(asyncio.to_thread(_run_tier0, req))

    # Feature 12 reads the CUSTOMER'S MESSAGE (Appendix A), so it can start now
    # rather than waiting for the reply. Dispatched alongside Tier 0 for the
    # same reason: it is a forward pass, and 12.4 overlaps it with generation
    # instead of paying for it serially.
    inj_task = asyncio.create_task(
        asyncio.to_thread(tier1.score_request, _extract_last_user_text(req)))

    # Step 5: stream upstream while Tier 1 reads what has already arrived.
    # `upstream_ms` is measured separately and subtracted, because the claim is
    # about ADDED latency -- the model's own generation was never ours to count.
    scorer = StreamingScorer(score_fn=tier1.score_spans)
    reply_text = _extract_last_user_text(req)
    upstream = upstream_override or STATE.get("upstream")
    try:
        stream = upstream(reply_text) if upstream is not None else provider_upstream(req)
        _body, upstream_s = await consume(stream, scorer)
    except UpstreamUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc

    t0 = await t0_task                     # already finished during the stream
    t1 = await scorer.finalize()           # drains at most one in-flight pass
    if scorer.last_error is not None:
        STATE["tier1_errors"] = STATE.get("tier1_errors", 0) + 1
        STATE["tier1_last_error"] = scorer.last_error

    signals = {**t0, **t1, **(await inj_task)}
    exposure_value = exposure.exposure(action, amount) if has_side_effect else 0.0
    invariants = gate.check_invariants(signals, {
        "retrieval_scope": 1, "user_scope": 1,
        "amount": amount, "hard_limit": hard_limit,
    })

    # Step 7: assemble with as_of = ctx.ts, then run Calibrator A.
    # Feature 8 is the length of the ANSWER, so it measures `_body` -- the text
    # the model produced -- not `reply_text`, which despite its name is the
    # incoming user turn that was sent upstream.
    length_signal = _length_z(action, len(_body or ""), ts)
    feats = _assemble(ts, signals, None, length_signal)
    p_wrong = _p_wrong(action, feats)
    verdict = gate.decide(
        p_wrong, exposure_value, budget, invariants, k=float(policy["k"])
    )
    region = gate.region_of(verdict)

    # Step 10: policy chooses between no judge, close-call-only, and deliberate
    # deep verification of every meaningful side effect.
    tier_reached = 1
    tier2_trigger = None
    degraded_reason = None
    forced_deep_check = (
        policy["tier2_mode"] == "side_effects"
        and has_side_effect
        and exposure_value > config.REVIEW_COST
        and not invariants.violated
    )
    should_run_tier2 = (
        policy["tier2_mode"] != "disabled"
        and (verdict is Verdict.CHECK_HARDER or forced_deep_check)
    )
    if should_run_tier2:
        tier_reached = 2
        tier2_trigger = "deep_policy" if forced_deep_check else "close_call"
        # Awaiting the judge is the only place this handler spends real time.
        # A timeout returns unavailable and re-decides at k = 1.0, escalating.
        t2 = await tier2.judge(
            _extract_last_user_text(req),
            _body or "",
            {"type": action, "amount_paise": req.vouch.amount_paise or 0},
            req.vouch.retrieved_chunks or [],
            timeout_sec=float(policy["tier2_timeout_sec"]),
        )
        if t2.available:
            # Model B was fitted on this exact judge configuration. Reassemble
            # from the same point-in-time signals; feature 8 is reused rather
            # than recomputed, so this request cannot enter its own history.
            feats = _assemble(
                ts, signals, (t2.p_wrong, features.Missing.VALUE), length_signal
            )
            p_wrong = _p_wrong(action, feats, model_b=True)
        else:
            # Appendix A distinguishes a judge that was never part of model A
            # (`not_supported`) from one that should have answered and did not
            # (`unavailable`). Keep A's pre-check estimate, as §16 requires,
            # but log the unavailable indicator for the null-rate gate.
            degraded_reason = t2.degraded_reason or "TIER2_TIMEOUT"
            feats = _assemble(
                ts, signals, (None, features.Missing.UNAVAILABLE), length_signal
            )
            STATE["tier2_unavailable"] = STATE.get("tier2_unavailable", 0) + 1
            STATE["tier2_last_degraded_reason"] = degraded_reason
        if forced_deep_check and degraded_reason is not None:
            verdict = Verdict.ESCALATE
        else:
            verdict = gate.redecide_after_tier2(
                p_wrong, exposure_value, budget, invariants
            )

    # ── 7.7 forced-review hold-out ─────────────────────────────────────────
    # Human labels arrive only on ESCALATED requests, which are by construction
    # the ones the system already thought were risky. A calibration curve fitted
    # on those describes the escalated population, not the autonomous one --
    # learn only from them and the system grows confident about the path it
    # stopped looking at. So a small random slice of would-have-passed requests
    # is escalated anyway and flagged, and anything estimated about the
    # autonomous population is inverse-probability weighted (calibrate.ipw_rate).
    #
    # The `region` stamped above is the region the gate ACTUALLY chose, and it
    # is deliberately NOT overwritten here: this row belongs to the `act`
    # population for calibration purposes even though a human will look at it.
    forced_review = False
    if verdict is Verdict.ACT and _FORCED_REVIEW.random() < config.FORCED_REVIEW_RATE:
        forced_review = True
        verdict = Verdict.ESCALATE

    rung = ladder.rung_for_verdict(
        verdict, ladder.Signals(**{k: signals.get(k) for k in
                                   ("pii_score", "secret_score", "injection_score",
                                    "verify_fail_frac") if k in signals}),
        p_wrong * exposure_value, budget, has_side_effect=has_side_effect)

    latency_ms = (time.perf_counter() - t_start) * 1000.0
    upstream_ms = upstream_s * 1000.0
    decision_id = _enqueue_decision(
        ts, req, action, band, fingerprint, feats, p_wrong,
        exposure_value, budget, verdict, region, rung,
        tier_reached, latency_ms, upstream_ms, forced_review,
        degraded_reason,
    )

    decision = {
        "decision_id": decision_id,
        "verdict": verdict.value,
        "region": region.value,
        "rung": int(rung),
        "tier_reached": tier_reached,
        "tier2_trigger": tier2_trigger,
        "p_wrong": p_wrong,
        "exposure": exposure_value,
        "budget": budget,
        "expected_loss": p_wrong * exposure_value,
        "forced_review": forced_review,
        "degraded_mode": degraded_reason is not None,
        "degraded_reason": degraded_reason,
        "latency_ms": latency_ms - upstream_ms,
        "upstream_ms": upstream_ms,
        "calibrator_version": STATE.get("calibrator_version", "bootstrap"),
        "shadow_mode": shadow_mode,
        "policy_mode": req.vouch.policy_mode,
        "earned_budget": earned_budget,
        "budget_multiplier": float(policy["budget_multiplier"]),
        "reason": _decision_reason(verdict, action, band, p_wrong, exposure_value, budget),
    }
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:20]}",
        "object": "chat.completion",
        "created": int(ts),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": _body or ""},
            "finish_reason": "stop",
        }],
        "vouch": decision,
    }


async def evaluate_response(req: ChatRequest, reply_text: str) -> dict:
    """Run an already-generated response through the exact production control path.

    The judge surface uses this after DeepSeek returns typed tool arguments. It
    avoids a second provider call and, unlike mutating ``STATE['upstream']``, is
    safe when ordinary requests are being served concurrently.
    """
    async def supplied_response(_prompt: str):
        yield reply_text

    return await evaluate(req, upstream_override=supplied_response)


# ── helpers ────────────────────────────────────────────────────────────────

def _decision_reason(verdict: Verdict, action: str, band: str | None,
                     p_wrong: float, exposure_value: float, budget: float) -> str:
    expected_loss = p_wrong * exposure_value
    if verdict is Verdict.ACT:
        if exposure_value <= config.REVIEW_COST:
            return "action exposure is below the cost of human review"
        return (f"expected loss {expected_loss:.2f} is within earned budget "
                f"{budget:.2f} for {action} in {band}")
    if verdict is Verdict.CHECK_HARDER:
        return (f"expected loss {expected_loss:.2f} is close enough to earned budget "
                f"{budget:.2f} for deeper verification")
    if verdict is Verdict.BLOCK:
        return "a fixed safety invariant failed; earned trust cannot override it"
    return (f"expected loss {expected_loss:.2f} exceeds earned budget "
            f"{budget:.2f}; human control is retained")

def _extract_last_user_text(req: ChatRequest) -> str:
    for message in reversed(req.messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _configuration_fingerprint(req: ChatRequest) -> str:
    """§9.4 identity from the configuration that is serving this request."""
    system_prompt = "\n".join(
        str(message.get("content", ""))
        for message in req.messages if message.get("role") == "system"
    )
    schemas = []
    for tool in req.tools:
        schema = tool.get("function", tool)
        if not isinstance(schema, dict) or not isinstance(schema.get("name"), str):
            raise HTTPException(422, "each tool must carry a named function schema")
        schemas.append(schema)
    return ledger.fingerprint(
        req.model,
        system_prompt,
        schemas,
        config.sensor_version(),
        STATE.get("calibrator_version", "bootstrap"),
    )


def _run_tier0(req: ChatRequest) -> dict:
    """Features 1-8, all free and all deterministic (7.1).

    `claims` empty is NOT the same as claims that all verified: feature 1 reads
    0.0 either way, which is exactly why feature 2 exists beside it as the count.
    """
    conn = STATE.get("orders_conn") or STATE["conn"]
    claims = req.vouch.claims or {}
    frac, n = tier0.verify_claims(conn, claims) if claims else (0.0, 0)
    text = _extract_last_user_text(req)

    chunks = req.vouch.retrieved_chunks
    if chunks:
        r_min, r_mean = tier0.retrieval_support(text, chunks)
    else:
        r_min = r_mean = None            # unavailable, never a defaulted 0.0

    retries, errors = tier0.tool_counts(req.vouch.tool_calls or [])
    return {"verify_fail_frac": frac, "verify_n_claims": n,
            "hedge_density": tier0.hedge_density(text),
            "retrieval_support_min": r_min, "retrieval_support_mean": r_mean,
            "tool_retry_count": retries, "tool_error_count": errors}


def _length_z(action: str, length: int, as_of: float) -> tuple[float | None, Any]:
    """Feature 8, over THIS action's history strictly before `as_of` (11).

    The history is appended AFTER the feature is computed, so a request never
    contributes to its own mean -- which is the leak 11 describes, and it would
    make the feature look informative in fitting and inert in production.
    """
    history = STATE["length_history"].setdefault(action, [])
    value, missing = features.length_z(length, history, as_of)
    history.append((as_of, length))
    return value, missing


def _assemble(as_of: float, signals: dict,
              judge_signal: tuple[float | None, features.Missing] | None,
              length_z: tuple) -> dict:
    """12.3 step 7. Sentinels are assigned HERE and never patched afterwards."""
    lz_value, lz_missing = length_z
    sig: dict = {}
    for name in features.FEATURE_ORDER:
        if name in ("logprob_mean", "logprob_min"):
            # Capability tier `none`: absent for this deployment rather than
            # degraded, so no companion indicator is set (7.2).
            sig[name] = (None, features.Missing.NOT_SUPPORTED)
        elif name == "judge_p_wrong":
            sig[name] = (judge_signal if judge_signal is not None
                         else (None, features.Missing.NOT_SUPPORTED))
        elif name == "length_z":
            sig[name] = (lz_value, lz_missing)
        else:
            v = signals.get(name)
            sig[name] = ((v, features.Missing.VALUE) if v is not None
                         else (None, features.Missing.UNAVAILABLE))
    return features.assemble(as_of, sig, STATE.get("capability_tier", "none"))


def _p_wrong(action: str, feats: dict, *, model_b: bool = False) -> float:
    """Run Calibrator A, or B once Tier 2 has fired (12.3 steps 7 and 10).

    16's degradation path is the per-action fallback prior. That is a defensible
    number when no calibrator is loaded -- it is NOT a silent 0.0, and it is not
    a substitute for the calibrator, which is what this system ran on until
    2026-08-28: a constant, multiplied by exposure, thresholding on exposure
    alone and firing Tier 2 on 40% of traffic.
    """
    suffix = "b" if model_b else "a"
    cal = STATE.get(f"calibrator_{suffix}")
    names = STATE.get(f"calibrator_names_{suffix}")
    if cal is None or not names:
        rate = STATE.get(f"calibrator_rate_{suffix}")
        if rate is not None:
            return float(rate)
        return float(STATE["fallback_priors"].get(action, 0.05))
    try:
        return float(cal.predict_proba([features.to_vector(feats, names)])[0][1])
    except Exception as exc:
        STATE["log_last_error"] = f"calibrator_{suffix}: {type(exc).__name__}"
        return float(STATE["fallback_priors"].get(action, 0.05))


# ── 12.3 step 12 -- the decision log ───────────────────────────────────────
# Fire and forget, BOUNDED, never awaited. An in-process queue rather than a
# broker, because a broker adds a failure mode to a path that must never fail
# a request (6.2).
#
# The first implementation of this did a synchronous INSERT and commit inside
# the handler and measured a p50 of 42.8 ms against a 5 ms budget -- one fsync
# per decision, serialised across every concurrent request. "Never awaited" is
# not a stylistic preference in step 12; it is most of the latency claim.

LOG_QUEUE: "asyncio.Queue | None" = None


async def _log_writer() -> None:
    """Drains the queue in batches and commits once per batch, off the request
    path entirely."""
    assert LOG_QUEUE is not None
    while True:
        row = await LOG_QUEUE.get()
        batch = [row]
        while not LOG_QUEUE.empty() and len(batch) < 200:
            batch.append(LOG_QUEUE.get_nowait())
        try:
            STATE["conn"].executemany(_DECISION_SQL, batch)
            STATE["conn"].commit()
        except Exception as exc:
            # The log must never fail a REQUEST -- but a silently discarded
            # decision log means tomorrow's calibration is fitted on a hole,
            # so it is counted and surfaced on /healthz rather than swallowed.
            unspilled = 0
            for spilled_row in batch:
                try:
                    _spill_decision(spilled_row)
                except OSError:
                    unspilled += 1
            STATE["log_errors"] = STATE.get("log_errors", 0) + unspilled
            STATE["log_last_error"] = f"{type(exc).__name__}: {exc}"
        for _ in batch:
            LOG_QUEUE.task_done()


_DECISION_SQL = (
    "INSERT INTO decision (id, ts, agent_id, action, band, fingerprint, "
    "calibrator_version, capability_tier, features, p_wrong, tier_reached, "
    "exposure, budget, expected_loss, verdict, region, rung, horizon_ends, "
    "latency_ms, upstream_ms, forced_review, degraded_mode, degraded_reason) "
    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)

_DECISION_COLUMNS = (
    "id", "ts", "agent_id", "action", "band", "fingerprint",
    "calibrator_version", "capability_tier", "features", "p_wrong",
    "tier_reached", "exposure", "budget", "expected_loss", "verdict",
    "region", "rung", "horizon_ends", "latency_ms", "upstream_ms",
    "forced_review", "degraded_mode", "degraded_reason",
)


def _spill_root() -> _Path:
    configured = _Path(os.environ.get(
        "VOUCH_SPILL_PATH", config.load()["proxy"]["spill_path"]
    ))
    if configured.is_absolute() and configured.parent.exists():
        return configured
    return _Path(__file__).resolve().parents[3] / "data" / "spill"


def _spill_decision(row: tuple) -> None:
    """Persist one queue-overflow row for deterministic replay."""
    import json

    root = _spill_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "decisions.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(zip(_DECISION_COLUMNS, row))) + "\n")
    STATE["log_spill_count"] = STATE.get("log_spill_count", 0) + 1


def _enqueue_decision(ts, req, action, band, fingerprint, feats, p_wrong,
                      exposure_value, budget, verdict, region, rung,
                      tier_reached, latency_ms, upstream_ms, forced_review,
                      degraded_reason) -> str:
    """Never blocks. A full queue spills the row to replayable JSONL."""
    import json
    horizon = (float(config.action_spec(action)["horizon_sec"])
               if action not in NON_EXECUTING_ACTIONS else 0.0)
    decision_id = uuid.uuid4().hex
    row = (
        decision_id, ts, req.vouch.agent_id, action,
        band or ("unpriced" if action in NON_EXECUTING_ACTIONS else "over_limit"),
        fingerprint, STATE.get("calibrator_version", "bootstrap"),
        STATE.get("capability_tier", "none"), json.dumps(feats), p_wrong,
        tier_reached, exposure_value, budget, p_wrong * exposure_value,
        verdict.value, region.value, int(rung), ts + horizon, latency_ms, upstream_ms,
        int(forced_review), int(degraded_reason is not None), degraded_reason,
    )
    if LOG_QUEUE is None:                   # no loop running (tests, replays)
        STATE["conn"].execute(_DECISION_SQL, row)
        return decision_id
    try:
        LOG_QUEUE.put_nowait(row)
    except asyncio.QueueFull:
        try:
            _spill_decision(row)
        except OSError as exc:
            STATE["log_errors"] = STATE.get("log_errors", 0) + 1
            STATE["log_last_error"] = f"spill: {type(exc).__name__}: {exc}"
            raise HTTPException(503, "decision log queue and spill are unavailable") from exc
    return decision_id
