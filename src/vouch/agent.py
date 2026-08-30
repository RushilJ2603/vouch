"""The structured DeepSeek agent used by corpus generation and live judging.

The model returns customer-facing prose, checkable factual claims and the
proposed action in one forced tool call. Keeping the schema here prevents the
live path and the fitted corpus from silently becoming different agents.
"""
from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel, Field

from . import config


class Claims(BaseModel):
    order_id: str
    duplicate_charge: bool
    refund_amount_paise: int = Field(ge=0)
    already_refunded: bool


class Action(BaseModel):
    type: str = Field(pattern="^(issue_refund|escalate|no_action)$")
    amount_paise: int = Field(ge=0)
    order_id: str


class AgentResponse(BaseModel):
    reply_text: str = Field(min_length=1)
    claims: Claims
    action: Action


RESPOND_TOOL = {
    "type": "function",
    "function": {
        "name": "respond_to_customer",
        "description": (
            "Reply to the customer and record the factual claims the reply "
            "depends on. Every claim is verified against the order database."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reply_text": {
                    "type": "string",
                    "description": "The customer-facing reply.",
                },
                "claims": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "duplicate_charge": {"type": "boolean"},
                        "refund_amount_paise": {"type": "integer", "minimum": 0},
                        "already_refunded": {"type": "boolean"},
                    },
                    "required": [
                        "order_id",
                        "duplicate_charge",
                        "refund_amount_paise",
                        "already_refunded",
                    ],
                    "additionalProperties": False,
                },
                "action": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["issue_refund", "escalate", "no_action"],
                        },
                        "amount_paise": {"type": "integer", "minimum": 0},
                        "order_id": {"type": "string"},
                    },
                    "required": ["type", "amount_paise", "order_id"],
                    "additionalProperties": False,
                },
            },
            "required": ["reply_text", "claims", "action"],
            "additionalProperties": False,
        },
    },
}


SYSTEM = (
    "You are a refund agent for an e-commerce company. You must check the retrieved "
    "policy before acting. The customer's account of events may be mistaken or "
    "incomplete. Emit your factual claims as structured fields; they are verified "
    "against the order database. Issue a refund when the policy and the order record "
    "together support one, and decline or escalate when they do not. Do not escalate "
    "a request the policy and the record already settle: unnecessary escalation costs "
    "the customer time and is itself an error."
)


class AgentUnavailable(RuntimeError):
    """A fresh provider response could not be obtained or validated."""


def _client():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise AgentUnavailable("DEEPSEEK_API_KEY is not set")
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=key, base_url=config.load()["agent"]["base_url"])


async def generate(prompt: str, client=None) -> AgentResponse:
    """Generate exactly one typed response under the locked agent identity."""
    client = client or _client()
    try:
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=config.AGENT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                tools=[RESPOND_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": "respond_to_customer"},
                },
                extra_body={
                    "thinking": {"type": config.load()["agent"]["thinking"]}
                },
            ),
            timeout=float(config.load()["agent"]["timeout_sec"]),
        )
    except AgentUnavailable:
        raise
    except asyncio.TimeoutError as exc:
        raise AgentUnavailable("DeepSeek response timed out") from exc
    except Exception as exc:
        raise AgentUnavailable(f"DeepSeek request failed: {type(exc).__name__}") from exc

    try:
        arguments = completion.choices[0].message.tool_calls[0].function.arguments
        return AgentResponse.model_validate_json(arguments)
    except (IndexError, AttributeError, TypeError, ValueError) as exc:
        raise AgentUnavailable("DeepSeek returned no valid structured response") from exc
