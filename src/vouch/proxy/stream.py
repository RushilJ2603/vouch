"""Tier 1 overlapped with token generation (12.4).

**The highest-risk part of the build.** The ~2 ms claim holds only if the
cheap sensors genuinely overlap generation. A sequential implementation
measures ~200 ms and the number has to be restated everywhere it appears.

A model does not produce its answer all at once -- it writes it over two or
three seconds. During that window the beginning exists and the end does not,
and that time is otherwise wasted. `feed()` must NEVER block the loop that is
passing tokens through to the caller; it buffers, and dispatches a scoring
pass only when a boundary is reached and no pass is already in flight.
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Callable

from .. import config

FLUSH_TOKENS = config.load()["tier1"]["flush_tokens"]        # 32
_SENTENCE_END = re.compile(r"[.!?]\s*$")


@dataclass
class StreamingScorer:
    """Accumulates spans and scores them as they become available.

    The per-response value for each feature is the MAX over spans: one leaked
    key in one sentence is a leak, however clean the rest of the answer is.
    """
    score_fn: Callable[[str], dict[str, float | None]]
    flush_tokens: int = FLUSH_TOKENS

    _buffer: str = ""
    _pending: asyncio.Task | None = None
    _scores: dict[str, float | None] = field(default_factory=dict)
    _spans_scored: int = 0
    _last_error: str | None = None

    def feed(self, chunk: str) -> None:
        """Called once per streamed chunk. Never awaits, never blocks.

        If a scoring pass is already in flight the text simply keeps
        accumulating -- dropping it would silently unscore part of the answer,
        and awaiting it here would serialise the very thing this exists to
        overlap.
        """
        self._buffer += chunk
        if not self._ready_to_flush():
            return
        if self._pending is not None:
            if not self._pending.done():
                return
            # A completed task's callback may still be waiting for the event
            # loop. Absorb it before replacing `_pending`, or the next span can
            # overwrite the only direct reference to the previous result.
            self._absorb(self._pending)
            self._pending = None
        self._dispatch()

    def _ready_to_flush(self) -> bool:
        return (_SENTENCE_END.search(self._buffer) is not None
                or len(self._buffer.split()) >= self.flush_tokens)

    def _dispatch(self) -> None:
        span, self._buffer = self._buffer, ""
        if not span.strip():
            return
        self._pending = asyncio.create_task(asyncio.to_thread(self.score_fn, span))
        self._pending.add_done_callback(self._absorb)
        self._spans_scored += 1

    def _absorb(self, task: asyncio.Task) -> None:
        if task.cancelled():
            self._last_error = "Tier 1 scoring task was cancelled"
            return
        error = task.exception()
        if error is not None:
            self._last_error = f"{type(error).__name__}: {error}"
            return
        self._merge(task.result())

    def _merge(self, scores: dict[str, float | None]) -> None:
        for name, value in scores.items():
            if value is None:
                self._scores.setdefault(name, None)
                continue
            current = self._scores.get(name)
            self._scores[name] = value if current is None else max(current, value)

    async def finalize(self) -> dict[str, float | None]:
        """Drains at most ONE in-flight pass, plus whatever is left in the
        buffer. Bounded by construction, which is what keeps step 6 of 12.3
        inside its ~0.1 ms budget."""
        if self._pending is not None:
            try:
                self._merge(await self._pending)
            except Exception as exc:
                # Fail closed through absent feature values, but retain the
                # failure for health reporting instead of swallowing it.
                self._last_error = f"{type(exc).__name__}: {exc}"
            self._pending = None
        if self._buffer.strip():
            self._merge(await asyncio.to_thread(self.score_fn, self._buffer))
            self._buffer = ""
            self._spans_scored += 1
        return dict(self._scores)

    @property
    def spans_scored(self) -> int:
        return self._spans_scored

    @property
    def last_error(self) -> str | None:
        return self._last_error


# ── Mock upstream (24.3) ───────────────────────────────────────────────────
# Demo 3 drives real traffic through the proxy against a mock upstream that
# streams at a REALISTIC token rate. The rate is the whole point: the ~2 ms
# claim is that Tier 1 finishes inside the window the model is already
# spending. An upstream that returns instantly proves nothing, and would in
# fact make a sequential implementation look perfectly fine.

DEFAULT_TOKENS_PER_SEC = 45.0        # a typical streamed completion


class UpstreamUnavailable(RuntimeError):
    """The configured agent provider could not produce a response."""


async def provider_upstream(req):
    """Stream the configured DeepSeek completion without forwarding `vouch`.

    The upstream receives only the standard OpenAI fields. Vouch-specific
    identity, policy and action context remain inside the control plane.
    """
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise UpstreamUnavailable("DEEPSEEK_API_KEY is not set")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=key, base_url=config.load()["agent"]["base_url"])
    kwargs = {
        "model": req.model,
        "messages": req.messages,
        "stream": True,
        "extra_body": {
            "thinking": {"type": config.load()["agent"]["thinking"]}
        },
    }
    if req.tools:
        kwargs["tools"] = req.tools
    try:
        response = await client.chat.completions.create(**kwargs)
        async for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            content = getattr(choices[0].delta, "content", None)
            if content:
                yield content
    except UpstreamUnavailable:
        raise
    except Exception as exc:
        raise UpstreamUnavailable(
            f"DeepSeek request failed: {type(exc).__name__}"
        ) from exc


async def mock_upstream(text: str, tokens_per_sec: float = DEFAULT_TOKENS_PER_SEC,
                        chunk_tokens: int = 4):
    """Yield `text` in chunks, paced the way a real model streams it."""
    words = text.split()
    delay = chunk_tokens / tokens_per_sec
    for i in range(0, len(words), chunk_tokens):
        await asyncio.sleep(delay)
        yield " ".join(words[i:i + chunk_tokens]) + " "


async def consume(stream, scorer: "StreamingScorer") -> tuple[str, float]:
    """Drive a stream through the scorer, passing every chunk straight through.

    Returns (full_text, upstream_seconds). `upstream_ms` is measured here and
    subtracted downstream, because the claim is about ADDED latency -- the
    model's own generation time was never ours to count (12.5).
    """
    import time as _clock
    chunks: list[str] = []
    start = _clock.perf_counter()
    async for chunk in stream:
        chunks.append(chunk)
        scorer.feed(chunk)               # must never block the passthrough
    return "".join(chunks), _clock.perf_counter() - start
