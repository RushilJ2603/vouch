"""Tier 2 -- the policy-selected deep check (7.3).

A larger model reviews the answer against the retrieved policy with a
structured prompt. The adaptive default fires on roughly 3% of traffic; an
explicit deep policy can run it for every meaningful side effect. It is the
ONLY sensor on the request path that costs real provider latency.

The model and its `thinking` mode both come from config and both feed
`sensor_version()`. One model serves this and the offline judge: feature 15 is
fitted from the offline pass and served from here, so two models would be
train/serve skew.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .. import config

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "p_wrong": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "verdict": {"type": "string", "enum": ["correct", "questionable", "wrong"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["p_wrong", "verdict", "reasons"],
    "additionalProperties": False,
}

# Measured 2026-08-28 (Appendix E.8): this provider IGNORES `response_format`.
# `json_schema` returns prose and nothing errors, `json_object` times out, and a
# FORCED tool_choice hangs -- only `auto` returns arguments. 7.1 makes structured
# tool arguments the hard requirement of the design in any case, and the agent
# path was moved to them on 2026-08-25 for the same reason.
#
# `scripts/judge_corpus.py` imports this tool and SYSTEM rather than restating
# either. Feature 15 is fitted there and served here; a second copy of the schema
# or the prompt would fit calibrator B on one distribution and serve another.
JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "record_judgement",
        "description": "Record the structured judgement of the agent's action.",
        "parameters": JUDGE_SCHEMA,
    },
}

SYSTEM = (
    "You review a support agent's response against the retrieved policy and the "
    "order record. Report `p_wrong` as your own calibrated probability that the "
    "response or its action is wrong. Use the full range: reserve values below "
    "0.05 for responses you are confident are correct and above 0.8 for clear "
    "violations. Do not anchor on 0.5."
)


@dataclass
class Tier2Result:
    p_wrong: float | None           # None = unavailable (7.6)
    verdict: str | None = None
    reasons: tuple[str, ...] = ()
    degraded_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.p_wrong is not None


UNAVAILABLE = Tier2Result(p_wrong=None, degraded_reason="TIER2_TIMEOUT")
UNCONFIGURED = Tier2Result(p_wrong=None, degraded_reason="TIER2_UNCONFIGURED")


def build_prompt(prompt: str, reply_text: str, action: dict, chunks: list[str]) -> str:
    policy = "\n".join(f"- {c}" for c in chunks)
    return (
        f"Retrieved policy:\n{policy}\n\n"
        f"Request:\n{prompt}\n\n"
        f"Agent reply:\n{reply_text}\n\n"
        f"Action taken: {json.dumps(action)}\n\n"
        "Is this response and action correct under the policy above?"
    )


def _client():
    key = os.environ.get("ZAI_API_KEY")
    if not key:
        raise RuntimeError("ZAI_API_KEY is not set")
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=key, base_url=config.load()["tier2"]["base_url"])


async def judge(prompt: str, reply_text: str, action: dict,
                chunks: list[str], client=None,
                timeout_sec: float | None = None) -> Tier2Result:
    """Returns UNAVAILABLE rather than raising.

    A timeout is not an error the request should fail on -- 16 decides on the
    pre-check estimate at k = 1.0, which escalates. That is deliberately worse
    for the caller than a fast answer and deliberately better than acting on a
    number nobody produced.
    """
    import asyncio

    try:
        client = client or _client()
    except RuntimeError:
        # A missing key is an expected deployment state for the offline demos.
        # It is still a failed sensor on a live close call, so the proxy must
        # resolve conservatively rather than letting the exception escape.
        return UNCONFIGURED

    try:
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=config.TIER2_MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": build_prompt(
                              prompt, reply_text, action, chunks)}],
                tools=[JUDGE_TOOL],
                tool_choice="auto",
                extra_body={"thinking": {"type": config.TIER2_THINKING}},
            ),
            timeout=(config.TIER2_TIMEOUT_SEC if timeout_sec is None else timeout_sec),
        )
    except asyncio.TimeoutError:
        return UNAVAILABLE
    except Exception:
        # Endpoint and SDK failures have the same decision semantics as a
        # timeout: no probability was produced, so the pre-check estimate is
        # re-decided at k=1.0. The caller surfaces the degraded reason.
        return UNAVAILABLE

    try:
        payload = json.loads(
            completion.choices[0].message.tool_calls[0].function.arguments)
        p = payload["p_wrong"]
        verdict = payload["verdict"]
        reasons = payload["reasons"]
    except (KeyError, ValueError, TypeError, IndexError, AttributeError,
            json.JSONDecodeError):
        # A schema violation is `unavailable`, never a defaulted 0.0. A judge
        # that returns prose must not read as "reviewed, and clean".
        return Tier2Result(p_wrong=None, degraded_reason="TIER2_SCHEMA_VIOLATION")

    if (set(payload) != {"p_wrong", "verdict", "reasons"}
            or isinstance(p, bool) or not isinstance(p, (int, float))
            or not 0.0 <= p <= 1.0
            or verdict not in {"correct", "questionable", "wrong"}
            or not isinstance(reasons, list)
            or not all(isinstance(reason, str) for reason in reasons)):
        return Tier2Result(p_wrong=None, degraded_reason="TIER2_SCHEMA_VIOLATION")

    return Tier2Result(p_wrong=float(p), verdict=verdict, reasons=tuple(reasons))
