"""Section 12.4 -- the concurrency claim, measured rather than asserted.

The highest-risk part of the build. The ~2 ms figure holds only if the cheap
sensors genuinely overlap token generation; a sequential implementation
measures ~200 ms and the number has to be restated everywhere it appears.
"""
import asyncio
import time

import pytest

from vouch.proxy.stream import StreamingScorer, consume, mock_upstream

TEXT = " ".join(["Your refund of 1240 rupees has been processed against order 4471."] * 14)


def encoder(ms: float):
    """Stand-in for the four Tier 1 encoders. 6.1 puts ONNX on CPU at 5-15 ms
    per forward pass and PyTorch eager at ~50, which is why the stack pins
    ONNX -- the difference decides whether the overlap is worth anything."""
    def run(span: str) -> dict:
        time.sleep(ms / 1000.0)
        return {"pii_score": 0.1, "injection_score": 0.0,
                "policy_score": 0.0, "secret_score": 0.0}
    return run


async def _added_ms(score_fn) -> tuple[float, int]:
    scorer = StreamingScorer(score_fn=score_fn)
    start = time.perf_counter()
    _body, upstream_s = await consume(mock_upstream(TEXT), scorer)
    await scorer.finalize()
    return (time.perf_counter() - start - upstream_s) * 1000, scorer.spans_scored


# ── The overlap is real ────────────────────────────────────────────────────

def test_scoring_happens_during_the_stream_not_after():
    _added, spans = asyncio.run(_added_ms(encoder(12)))
    assert spans >= 4, "spans must be scored while tokens are still arriving"


def test_overlapped_beats_sequential_by_a_wide_margin():
    overlapped, _ = asyncio.run(_added_ms(encoder(12)))

    async def naive() -> float:
        start = time.perf_counter()
        chunks = []
        up_start = time.perf_counter()
        async for chunk in mock_upstream(TEXT):
            chunks.append(chunk)
        upstream_s = time.perf_counter() - up_start
        words = "".join(chunks).split()
        for i in range(0, len(words), 32):        # score AFTER the stream closed
            encoder(12)(" ".join(words[i:i + 32]))
        return (time.perf_counter() - start - upstream_s) * 1000

    sequential = asyncio.run(naive())

    assert sequential > overlapped * 2, (
        f"overlap bought nothing: {overlapped:.1f} ms vs {sequential:.1f} ms sequential")


def test_feed_never_blocks_the_passthrough_loop():
    """A slow encoder must not slow the stream reaching the caller."""
    async def run() -> tuple[float, float]:
        scorer = StreamingScorer(score_fn=encoder(40))
        start = time.perf_counter()
        _body, upstream_s = await consume(mock_upstream(TEXT), scorer)
        during = time.perf_counter() - start
        await scorer.finalize()
        return during, upstream_s

    during, upstream_s = asyncio.run(run())
    assert during == pytest.approx(upstream_s, abs=0.05), (
        "the passthrough loop stalled waiting on a scoring pass")


# ── The floor, and why it is a floor ───────────────────────────────────────

@pytest.mark.parametrize("encoder_ms", [5, 12, 25])
def test_added_latency_floors_at_one_forward_pass(encoder_ms):
    """MEASURED FINDING, 2026-08-24. Overlap removes every forward pass except
    the last one, and the last one cannot be removed at all: the final span of
    the answer only exists once the stream has closed, so there is no
    generation time left to hide its scoring behind.

    Added latency is therefore approximately ONE forward pass, whatever the
    payload length. 12.3 step 6 budgets ~0.1 ms for `scorer.finalize()` while
    describing it as draining one in-flight pass -- those two statements are
    only compatible if a pass is free, and on ONNX CPU it is 5-15 ms.
    """
    added, _ = asyncio.run(_added_ms(encoder(encoder_ms)))
    assert added >= encoder_ms * 0.9, "cannot beat one forward pass"
    assert added <= encoder_ms * 1.6, (
        f"added {added:.1f} ms is more than one pass ({encoder_ms} ms); "
        "the overlap has regressed")


def test_the_two_millisecond_budget_needs_a_sub_two_millisecond_encoder():
    """The consequence, stated as a test so it cannot be forgotten: the ~2 ms
    obsolete claim was only reachable if a Tier
    1 forward pass was itself under ~2 ms. Compare the two simulated encoders
    rather than treating scheduler overhead as a product latency gate; §24.3's
    measured 40 ms gate is the absolute check that now matters."""
    fast, _ = asyncio.run(_added_ms(encoder(0.5)))
    realistic, _ = asyncio.run(_added_ms(encoder(12)))
    assert realistic > fast * 2
    assert realistic > 5, (
        "a realistic ONNX encoder puts added latency above the ~2 ms budget")


# ── finalize() is bounded ──────────────────────────────────────────────────

def test_finalize_drains_at_most_one_in_flight_pass():
    async def run() -> float:
        scorer = StreamingScorer(score_fn=encoder(20))
        await consume(mock_upstream(TEXT), scorer)
        start = time.perf_counter()
        await scorer.finalize()
        return (time.perf_counter() - start) * 1000

    drain_ms = asyncio.run(run())
    assert drain_ms < 20 * 2.5, "finalize drained more than one pass plus the tail"


def test_finalize_cannot_lose_a_task_that_already_finished():
    """A done task's callback may still be queued behind the current coroutine.

    `finalize()` must read the task itself even when `done()` is already true;
    otherwise a fast encoder can disappear between those two states and every
    Tier 1 signal reaches the gate as unavailable.
    """
    async def run() -> dict:
        scorer = StreamingScorer(score_fn=lambda _span: {})
        task = asyncio.create_task(asyncio.sleep(0, result={"secret_score": 1.0}))
        await task
        scorer._pending = task
        return await scorer.finalize()

    assert asyncio.run(run())["secret_score"] == 1.0


def test_feed_cannot_overwrite_a_finished_unabsorbed_task():
    async def run() -> dict:
        scorer = StreamingScorer(score_fn=lambda _span: {"pii_score": 0.0})
        task = asyncio.create_task(asyncio.sleep(0, result={"secret_score": 1.0}))
        await task
        scorer._pending = task
        scorer.feed("The next complete sentence.")
        return await scorer.finalize()

    scores = asyncio.run(run())
    assert scores["secret_score"] == 1.0
    assert scores["pii_score"] == 0.0


def test_max_over_spans_survives_the_overlap():
    """One leaked key in one sentence is a leak, even if it arrived in the
    middle of the stream and later spans were clean."""
    def spiky(span: str) -> dict:
        return {"secret_score": 1.0 if "AKIA" in span else 0.0}
    async def run() -> dict:
        scorer = StreamingScorer(score_fn=spiky)
        text = ("All fine here. " * 8 + "Debug AKIAIOSFODNN7EXAMPLE here. "
                + "All fine again. " * 8)
        await consume(mock_upstream(text), scorer)
        return await scorer.finalize()

    assert asyncio.run(run())["secret_score"] == 1.0
