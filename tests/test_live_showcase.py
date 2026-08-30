"""Live judge surface: typed generation, bounded policies and real control path."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from vouch import agent, config, store
from vouch.proxy import app
from vouch.sensors import tier2


def _tool_completion(payload: dict):
    function = SimpleNamespace(arguments=json.dumps(payload))
    call = SimpleNamespace(function=function)
    message = SimpleNamespace(tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_structured_agent_uses_the_locked_tool_and_model():
    seen = {}

    class Completions:
        async def create(self, **kwargs):
            seen.update(kwargs)
            return _tool_completion({
                "reply_text": "I verified the duplicate and issued the refund.",
                "claims": {
                    "order_id": "ord_001",
                    "duplicate_charge": True,
                    "refund_amount_paise": 70000,
                    "already_refunded": False,
                },
                "action": {
                    "type": "issue_refund",
                    "amount_paise": 70000,
                    "order_id": "ord_001",
                },
            })

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = asyncio.run(agent.generate("prompt", client=client))
    assert result.action.type == "issue_refund"
    assert seen["model"] == config.AGENT_MODEL
    assert seen["tools"] == [agent.RESPOND_TOOL]
    assert seen["tool_choice"]["function"]["name"] == "respond_to_customer"


def test_policy_profiles_are_bounded_and_deep_forces_the_judge(tmp_path, monkeypatch):
    conn = store.connect(str(tmp_path / "vouch.sqlite"))
    store.init_schema(conn)
    previous = dict(app.STATE)
    previous_queue = app.LOG_QUEUE
    app.STATE.update({
        "conn": conn,
        "orders_conn": None,
        "ledger": {("agent", "issue_refund", "0-2k", "fp"): {"budget": 500.0}},
        "fingerprint": "fp",
        "calibrator_a": object(),
        "calibrator_b": object(),
        "calibrator_names_a": ("verify_fail_frac",),
        "calibrator_names_b": ("verify_fail_frac", "judge_p_wrong"),
        "calibrator_version": "cal-v1",
        "capability_tier": "none",
        "length_history": {},
    })
    app.LOG_QUEUE = None
    monkeypatch.setattr(app, "_FORCED_REVIEW", SimpleNamespace(random=lambda: 1.0))
    monkeypatch.setattr(app, "_run_tier0", lambda _req: {
        "verify_fail_frac": 0.0,
        "verify_n_claims": 1,
        "retrieval_support_min": 0.9,
        "retrieval_support_mean": 0.9,
        "tool_retry_count": 0,
        "tool_error_count": 0,
        "hedge_density": 0.0,
    })
    monkeypatch.setattr(app.tier1, "score_request", lambda _text: {"injection_score": 0.0})
    monkeypatch.setattr(app.tier1, "score_spans", lambda _text: {
        "pii_score": 0.0, "policy_score": 0.0, "secret_score": 0.0,
    })
    monkeypatch.setattr(app, "_p_wrong", lambda *_args, model_b=False: 0.01)
    calls = []

    async def judge(*_args, **kwargs):
        calls.append(kwargs["timeout_sec"])
        return tier2.Tier2Result(0.01, "correct", ("verified",))

    monkeypatch.setattr(app.tier2, "judge", judge)
    req = app.ChatRequest(
        model=config.AGENT_MODEL,
        messages=[{"role": "user", "content": "refund"}],
        tools=[agent.RESPOND_TOOL],
        vouch=app.VouchBlock(
            agent_id="agent", policy_mode="deep", action_hint="issue_refund",
            amount_paise=70_000, claims={"order_id": "ord_001"},
            retrieved_chunks=["Duplicate charges may be refunded."],
        ),
    )
    try:
        response = asyncio.run(app.evaluate_response(req, "Refund issued."))
        decision = response["vouch"]
        assert calls == [15.0]
        assert decision["tier_reached"] == 2
        assert decision["tier2_trigger"] == "deep_policy"
        assert decision["policy_mode"] == "deep"
        assert decision["budget"] == pytest.approx(325.0)
        assert response["choices"][0]["message"]["content"] == "Refund issued."
    finally:
        conn.close()
        app.STATE.clear()
        app.STATE.update(previous)
        app.LOG_QUEUE = previous_queue


def test_the_three_modes_trade_budget_for_verification():
    fast = config.policy_profile("fast")
    adaptive = config.policy_profile("adaptive")
    deep = config.policy_profile("deep")
    assert fast["budget_multiplier"] > adaptive["budget_multiplier"] > deep["budget_multiplier"]
    assert fast["tier2_mode"] == "disabled"
    assert adaptive["tier2_mode"] == "close_calls"
    assert deep["tier2_mode"] == "side_effects"
    assert fast["tier2_timeout_sec"] < adaptive["tier2_timeout_sec"] < deep["tier2_timeout_sec"]
