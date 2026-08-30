"""Tier 0 -- free signals and deterministic verification against the record (7.1).

The hardest part of verification is normally extracting claims from prose. We
sidestep it: the agent emits its factual claims as structured tool arguments
alongside the reply, so verification is a typed comparison against SQLite
rather than an NLP problem.

Nothing here decides anything. It produces features 1-7 of Appendix A.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any, Callable

# ── The verifier registry (7.1) ────────────────────────────────────────────
# One verifier per claim key. A claim with NO registered verifier is
# uncheckable: it contributes to neither the numerator nor the denominator of
# verify_fail_frac, and lowers verify_n_claims instead. The calibrator learns
# to treat a low claim count as a WEAK signal, not a safe one.

Verifier = Callable[[sqlite3.Connection, Any, dict[str, Any]], bool]


def _order_exists(conn, value, claims) -> bool:
    row = conn.execute('SELECT 1 FROM "order" WHERE id = ?', (str(value),)).fetchone()
    return row is not None


def _duplicate_charge(conn, value, claims) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM charge WHERE order_id = ? AND amount_paise = ("
        '  SELECT amount_paise FROM "order" WHERE id = ?)',
        (str(claims.get("order_id")), str(claims.get("order_id"))),
    ).fetchone()
    return bool(value) is (row[0] > 1)


def _refund_amount_matches(conn, value, claims) -> bool:
    row = conn.execute(
        'SELECT amount_paise FROM "order" WHERE id = ?', (str(claims.get("order_id")),)
    ).fetchone()
    return row is not None and int(value) == int(row[0])


def _already_refunded(conn, value, claims) -> bool:
    row = conn.execute(
        "SELECT EXISTS(SELECT 1 FROM refund WHERE order_id = ?)",
        (str(claims.get("order_id")),),
    ).fetchone()
    return bool(value) is bool(row[0])


VERIFIERS: dict[str, Verifier] = {
    "order_id": _order_exists,
    "duplicate_charge": _duplicate_charge,
    "refund_amount_paise": _refund_amount_matches,
    "already_refunded": _already_refunded,
}


def verify_claims(conn: sqlite3.Connection, claims: dict[str, Any]) -> tuple[float, int]:
    """(verify_fail_frac, verify_n_claims) -- features 1 and 2.

    "No failed claims" is NOT the same as "verified". An agent that replies
    "everything looks correct" with claims: {} produces verify_fail_frac = 0.0,
    exactly like a fully-verified four-claim response. verify_n_claims is a
    separate feature precisely so the calibrator can tell them apart.
    """
    checkable = 0
    failed = 0
    for key, value in claims.items():
        verifier = VERIFIERS.get(key)
        if verifier is None:
            continue                        # uncheckable: neither numerator nor denominator
        checkable += 1
        try:
            if not verifier(conn, value, claims):
                failed += 1
        except (ValueError, TypeError, sqlite3.Error):
            failed += 1                     # a claim that cannot be evaluated is a failed claim
    if checkable == 0:
        return 0.0, 0
    return failed / checkable, checkable


# ── Free textual signals (features 3-7) ────────────────────────────────────

HEDGES = frozenset({
    "might", "maybe", "possibly", "perhaps", "probably", "seems", "appears",
    "likely", "unlikely", "could", "roughly", "approximately", "around",
    "believe", "think", "unclear", "unsure", "generally", "typically", "usually",
})

_WORD = re.compile(r"[a-z']+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def hedge_density(text: str) -> float:
    """Feature 7. Hedging lexicon matches over tokens."""
    tokens = _WORD.findall(text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in HEDGES) / len(tokens)


def _overlap(sentence: str, chunk: str) -> float:
    a = set(_WORD.findall(sentence.lower()))
    b = set(_WORD.findall(chunk.lower()))
    if not a:
        return 1.0                          # nothing asserted, nothing unsupported
    return len(a & b) / len(a)


def retrieval_support(text: str, chunks: list[str]) -> tuple[float, float]:
    """(min, mean) support across sentences -- features 3 and 4.

    Computed over the chunks retrieved for THIS request. If the index is
    rebuilt before scoring, the chunks are not the chunks the model saw, and
    the feature is measuring something that never happened (11).
    """
    sentences = [s for s in _SENTENCE.split(text.strip()) if s.strip()]
    if not sentences or not chunks:
        return 0.0, 0.0
    scores = [max(_overlap(s, c) for c in chunks) for s in sentences]
    return min(scores), sum(scores) / len(scores)


def tool_counts(tool_calls: list[dict[str, Any]]) -> tuple[int, int]:
    """(tool_retry_count, tool_error_count) -- features 5 and 6."""
    retries = sum(1 for c in tool_calls if c.get("is_retry"))
    errors = sum(1 for c in tool_calls if c.get("error"))
    return retries, errors
