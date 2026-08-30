"""Section 12.3 step 10 -- live Tier 2 and its fail-closed path."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from vouch import store
from vouch.features import Missing
from vouch.proxy import app
from vouch.sensors import tier2


class FakeCompletions:
    def __init__(self, completion=None, error: Exception | None = None):
        self.completion = completion
        self.error = error
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.completion


def fake_client(*, arguments: str | None = None, error: Exception | None = None):
    tool_calls = [] if arguments is None else [
        SimpleNamespace(function=SimpleNamespace(arguments=arguments))
    ]
    completion = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(tool_calls=tool_calls)
    )])
    completions = FakeCompletions(completion, error)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_the_live_judge_reads_structured_tool_arguments():
    payload = json.dumps({
        "p_wrong": 0.17,
        "verdict": "questionable",
        "reasons": ["refund window is unclear"],
    })
    client, completions = fake_client(arguments=payload)

    result = asyncio.run(tier2.judge(
        "customer request", "agent reply", {"type": "issue_refund"},
        ["refunds are permitted within 30 days"], client=client,
    ))

    assert result.available
    assert result.p_wrong == pytest.approx(0.17)
    assert result.verdict == "questionable"
    assert result.reasons == ("refund window is unclear",)
    assert completions.kwargs["tools"] == [tier2.JUDGE_TOOL]
    assert completions.kwargs["tool_choice"] == "auto"


def test_a_missing_key_is_unavailable_not_an_exception(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    result = asyncio.run(tier2.judge(
        "customer request", "agent reply", {"type": "issue_refund"}, [],
    ))

    assert not result.available
    assert result.degraded_reason == "TIER2_UNCONFIGURED"


def test_a_timeout_is_reported_as_unavailable():
    client, _ = fake_client(error=asyncio.TimeoutError())

    result = asyncio.run(tier2.judge(
        "customer request", "agent reply", {"type": "issue_refund"}, [], client=client,
    ))

    assert not result.available
    assert result.degraded_reason == "TIER2_TIMEOUT"


def test_malformed_tool_arguments_are_unavailable():
    client, _ = fake_client(arguments="not-json")

    result = asyncio.run(tier2.judge(
        "customer request", "agent reply", {"type": "issue_refund"}, [], client=client,
    ))

    assert not result.available
    assert result.degraded_reason == "TIER2_SCHEMA_VIOLATION"


@pytest.mark.parametrize("payload", [
    {"p_wrong": "0.2", "verdict": "correct", "reasons": []},
    {"p_wrong": 0.2, "verdict": "maybe", "reasons": []},
    {"p_wrong": 0.2, "verdict": "correct", "reasons": "because"},
    {"p_wrong": 0.2, "verdict": "correct", "reasons": [], "extra": True},
])
def test_every_part_of_the_judge_schema_is_enforced(payload):
    client, _ = fake_client(arguments=json.dumps(payload))

    result = asyncio.run(tier2.judge(
        "customer request", "agent reply", {"type": "issue_refund"}, [], client=client,
    ))

    assert not result.available
    assert result.degraded_reason == "TIER2_SCHEMA_VIOLATION"


@pytest.fixture
def request_path(monkeypatch):
    conn = store.connect(":memory:")
    store.init_schema(conn)
    previous = dict(app.STATE)
    previous_queue = app.LOG_QUEUE
    app.STATE.update({
        "conn": conn,
        "ledger": {("refund-agent", "issue_refund", "0-2k", "fp-test"): {"budget": 10.0}},
        "fingerprint": "fp-test",
        "calibrator_version": "cal-v1",
        "capability_tier": "none",
        "length_history": {},
        "tier2_unavailable": 0,
    })
    app.LOG_QUEUE = None

    monkeypatch.setattr(app, "_FORCED_REVIEW", SimpleNamespace(random=lambda: 1.0))
    monkeypatch.setattr(app, "_run_tier0", lambda _req: {
        "verify_fail_frac": 0.0,
        "verify_n_claims": 3,
        "retrieval_support_min": 0.8,
        "retrieval_support_mean": 0.9,
        "tool_retry_count": 0,
        "tool_error_count": 0,
        "hedge_density": 0.0,
    })
    monkeypatch.setattr(app.tier1, "score_request", lambda _text: {"injection_score": 0.0})
    monkeypatch.setattr(app.tier1, "score_spans", lambda _text: {
        "pii_score": 0.0,
        "policy_score": 0.1,
        "secret_score": 0.0,
    })

    async def upstream(_text):
        yield "The duplicate charge has been refunded."

    app.STATE["upstream"] = upstream
    req = app.ChatRequest(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "Please refund the duplicate charge."}],
        vouch=app.VouchBlock(
            agent_id="refund-agent",
            action_hint="issue_refund",
            amount_paise=70_000,
            claims={"order_id": "ord_001", "duplicate_charge": True},
            retrieved_chunks=["Duplicate charges may be refunded."],
            tool_calls=[],
        ),
    )
    yield conn, req

    conn.close()
    app.STATE.clear()
    app.STATE.update(previous)
    app.LOG_QUEUE = previous_queue


def test_close_call_runs_calibrator_b_and_resolves_to_act(request_path, monkeypatch):
    conn, req = request_path
    calls = []

    def predict(_action, feats, *, model_b=False):
        calls.append((model_b, feats))
        return 0.01 if model_b else 0.05

    seen = {}

    async def judge(prompt, reply_text, action, chunks, client=None, timeout_sec=None):
        seen.update(
            prompt=prompt,
            reply_text=reply_text,
            action=action,
            chunks=chunks,
            timeout_sec=timeout_sec,
        )
        return tier2.Tier2Result(0.01, "correct", ("policy permits the refund",))

    monkeypatch.setattr(app, "_p_wrong", predict)
    monkeypatch.setattr(app.tier2, "judge", judge)

    response = asyncio.run(app.chat_completions(req))

    assert [model_b for model_b, _ in calls] == [False, True]
    assert calls[1][1]["judge_p_wrong"] == pytest.approx(0.01)
    assert calls[1][1]["judge_p_wrong_unavailable"] == 0
    assert response["choices"][0]["message"]["content"] == (
        "The duplicate charge has been refunded."
    )
    assert response["vouch"]["verdict"] == "act"
    assert response["vouch"]["p_wrong"] == pytest.approx(0.01)
    assert response["vouch"]["tier_reached"] == 2
    assert response["vouch"]["degraded_mode"] is False
    assert seen == {
        "prompt": "Please refund the duplicate charge.",
        "reply_text": "The duplicate charge has been refunded.",
        "action": {"type": "issue_refund", "amount_paise": 70_000},
        "chunks": ["Duplicate charges may be refunded."],
        "timeout_sec": 2.0,
    }

    row = conn.execute("SELECT * FROM decision").fetchone()
    logged = json.loads(row["features"])
    assert logged["judge_p_wrong"] == pytest.approx(0.01)
    assert row["tier_reached"] == 2
    assert row["calibrator_version"] == "cal-v1"
    assert row["degraded_mode"] == 0


def test_unavailable_tier2_keeps_a_estimate_and_escalates(request_path, monkeypatch):
    conn, req = request_path
    calls = []

    def predict(_action, feats, *, model_b=False):
        calls.append(model_b)
        return 0.05

    async def unavailable(*_args, **_kwargs):
        return tier2.Tier2Result(None, degraded_reason="TIER2_UNCONFIGURED")

    monkeypatch.setattr(app, "_p_wrong", predict)
    monkeypatch.setattr(app.tier2, "judge", unavailable)

    response = asyncio.run(app.chat_completions(req))

    assert calls == [False], "an unavailable judge reuses the pre-check estimate"
    assert response["vouch"]["verdict"] == "escalate"
    assert response["vouch"]["p_wrong"] == pytest.approx(0.05)
    assert response["vouch"]["tier_reached"] == 2
    assert response["vouch"]["degraded_mode"] is True
    assert response["vouch"]["degraded_reason"] == "TIER2_UNCONFIGURED"
    assert app.STATE["tier2_unavailable"] == 1

    row = conn.execute("SELECT * FROM decision").fetchone()
    logged = json.loads(row["features"])
    assert logged["judge_p_wrong"] is None
    assert logged["judge_p_wrong_unavailable"] == 1
    assert row["degraded_mode"] == 1
    assert row["degraded_reason"] == "TIER2_UNCONFIGURED"


def test_model_a_and_a_dead_judge_have_different_missing_encodings():
    signals = {"secret_score": 0.0}
    model_a = app._assemble(1.0, signals, None, (None, Missing.UNAVAILABLE))
    dead_judge = app._assemble(
        1.0, signals, (None, Missing.UNAVAILABLE), (None, Missing.UNAVAILABLE)
    )

    assert model_a["judge_p_wrong_unavailable"] == 0
    assert dead_judge["judge_p_wrong_unavailable"] == 1
