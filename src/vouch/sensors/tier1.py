"""Tier 1 -- streaming encoders for features 11 to 14 (Appendix A).

These sensors measure PII (11), prompt injection (12), policy violations (13),
and secrets (14). When an ONNX model is unavailable, the sensor returns None (7.6).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from vouch import config

# Features 11 to 14
# pii_score, injection_score, policy_score, secret_score

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD_LIKE = re.compile(r"[A-Za-z0-9+/=_-]{16,}")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    ent = 0.0
    for count in freq.values():
        p = count / len(s)
        ent -= p * math.log2(p)
    return ent


# Known credential formats. These are matched by SHAPE, not by entropy,
# because the shape is the evidence. A GitHub token is mostly lowercase
# hex-ish text with entropy around 3.9 -- comfortably under any entropy
# threshold -- so an entropy-only detector scores it ~0.36 and the
# `no_secrets_in_output` invariant (threshold 0.5) lets it through. An
# invariant that fails OPEN on the most common credential format on earth is
# worse than no invariant, because it reads as coverage.
_KNOWN_SECRETS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),  # OpenAI / DeepSeek / Z.ai style
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),  # Google API
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}"),  # Slack
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),  # JWT
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),  # PEM
    re.compile(r"(?i)\b(api[_-]?key|secret|token|passwd|password)\b\s*[=:]\s*\S{8,}"),
)


def _score_secret(span: str) -> float:
    """Feature 14. Regex patterns AND Shannon entropy over candidate spans.

    Deterministic, no model, so this one is never `unavailable` (Appendix A).
    """
    for pattern in _KNOWN_SECRETS:
        if pattern.search(span):
            return 1.0  # shape is conclusive; no need to weigh entropy

    max_score = 0.0
    for match in _WORD_LIKE.finditer(span):
        ent = _shannon_entropy(match.group(0))
        if ent > 4.5:  # random base64 territory
            max_score = max(max_score, min(1.0, (ent - 4.5) / 1.5))
    return max_score


# ── Feature 11: "Presidio-style regex + NER over MiniLM embeddings" ────────
# Appendix A defines pii_score as regex AND a model. The regex half needs no
# model at all, and treating the whole feature as `unavailable` when only the
# NER half is missing throws away a signal we can compute for free.
#
# The asymmetry is the point, and it is the same asymmetry as 16's:
#   a regex HIT  is conclusive  -> the value is measured, and it is 1.0
#   a regex MISS is NOT evidence of absence -> without NER we cannot conclude
#     "no PII", so the feature is `unavailable` and the invariant fails closed
#
# That keeps the guarantee in 16: an unavailable component reduces autonomy,
# never increases it. A regex-only detector reporting 0.0 would claim the
# opposite -- "checked, and clean" -- on the strength of a check it never ran.

_PII_PATTERNS = (
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email
    re.compile(r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)"),  # India mobile
    re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),  # card-like
    re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"),  # Aadhaar
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),  # PAN
    re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),  # IFSC
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),  # IPv4
)


def _score_pii_regex(span: str) -> float | None:
    """1.0 on a conclusive hit, else None -- never 0.0. See the note above."""
    for pattern in _PII_PATTERNS:
        if pattern.search(span):
            return 1.0
    return None


# ── Session pool (12.2 item 4) ─────────────────────────────────────────────
# Loaded once at startup and held for the process lifetime. int8 is preferred
# over fp32 wherever both exist: measured 2026-08-25, quantizing MiniLM took a
# forward pass from 7.4 ms to 3.4 ms and the injection encoder from 55.2 ms to
# 27.7 ms, and cut 739 MB to 244 MB. 12.4 showed added latency floors at one
# forward pass, so the slowest encoder sets the floor for the whole layer.

_SESSIONS: dict[str, object] = {}


def _onnx_root() -> Path:
    """`/artifacts/onnx` inside the container; the repo copy outside it."""
    configured = Path(config.load()["tier1"]["onnx_root"])
    if configured.exists():
        return configured
    return Path(__file__).resolve().parents[3] / "artifacts" / "onnx"


# Encoders that MUST NOT use the int8 build, measured 2026-08-27.
# Dynamic int8 leaves the injection encoder inert: fp32 scores P(injection)
# 1.0000 on every injection probe, int8 scores a near-constant ~0.07 and calls
# them all SAFE. Feature 12 is on the FIXED path (10.2), where a false negative
# is not recoverable by any track record, so this one pays the extra ~27 ms.
# 27 decision 8's own argument -- "buying 25 ms by weakening an invariant is
# the wrong trade" -- applies with more force to zeroing it outright.
_FP32_ONLY = frozenset({"prompt_injection"})


def _model_path(name: str) -> Path | None:
    """int8 first, fp32 second -- except where int8 destroys the signal."""
    folder = _onnx_root() / name
    order = ("model.onnx",) if name in _FP32_ONLY else ("model_int8.onnx", "model.onnx")
    for candidate in order:
        if (folder / candidate).exists():
            return folder / candidate
    return None


def _tokenizer_available() -> bool:
    """A transformer encoder cannot run without one, and 6.1 keeps
    `transformers` out of serving to keep torch out of it."""
    import importlib.util

    return importlib.util.find_spec("tokenizers") is not None


def _is_onnx_model_available(name: str) -> bool:
    """Available means RUNNABLE, not merely present on disk.

    This distinction nearly shipped as a silent autonomy increase. Once the
    export produced the files, a presence-only check reported the encoders
    available while inference was still unwired -- so the scores defaulted to
    0.0, the invariants PASSED, and requests that had been correctly BLOCKing
    would have started to ACT. That is precisely the failure 16 forbids: an
    unavailable component must reduce autonomy, never increase it, and 0.0
    reads as "checked, and clean" from a check that never ran.
    """
    return _model_path(name) is not None and _tokenizer_available()


def _session(name: str):
    """One session per encoder, created on first use and cached."""
    if name not in _SESSIONS:
        path = _model_path(name)
        if path is None:
            return None
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(config.load()["tier1"]["pool_size"])
        _SESSIONS[name] = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
    return _SESSIONS[name]


_TOKENIZERS: dict[str, object] = {}
_LABELS: dict[str, dict] = {}


def _tokenizer(name: str):
    """The Rust `tokenizers` file saved beside the ONNX graph at export time."""
    if name not in _TOKENIZERS:
        path = _model_path(name)
        if path is None:
            return None
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(path.parent / "tokenizer.json"))
        tok.enable_truncation(max_length=256)
        _TOKENIZERS[name] = tok
    return _TOKENIZERS[name]


def _softmax(row: list[float]) -> list[float]:
    hi = max(row)
    exps = [math.exp(v - hi) for v in row]
    total = sum(exps)
    return [e / total for e in exps]


def _run(name: str, span: str):
    """Feed one span through an encoder. Returns raw logits, or None.

    **This is the half that did not exist.** `_is_onnx_model_available` was
    already correct, but nothing ever called the session: features 11 to 13
    returned the 0.0 they were initialised with. So the moment an encoder was
    exported, its score read "measured, and clean" from a forward pass that
    never happened -- invariants would PASS and requests that had correctly
    BLOCKed would start to ACT, which is exactly the 16 failure the
    availability check above was written to prevent, arriving one layer down.
    """
    session, tok = _session(name), _tokenizer(name)
    if session is None or tok is None or not span.strip():
        return None
    enc = tok.encode(span)
    feed = {}
    for inp in session.get_inputs():
        if inp.name == "input_ids":
            feed[inp.name] = [enc.ids]
        elif inp.name == "attention_mask":
            feed[inp.name] = [enc.attention_mask]
        elif inp.name == "token_type_ids":
            feed[inp.name] = [enc.type_ids]
    if "input_ids" not in feed:
        return None
    import numpy as np

    feed = {k: np.array(v, dtype=np.int64) for k, v in feed.items()}
    return session.run(None, feed)[0]


def _embed(span: str) -> list[float] | None:
    """Mean-pooled MiniLM sentence embedding -- the input to feature 13's head.

    The ONNX export's first output is `last_hidden_state` [1, seq, hidden], so
    pooling over tokens is what turns it into one sentence vector.
    """
    out = _run("minilm", span)
    if out is None:
        return None
    import numpy as np

    arr = np.asarray(out)
    vec = arr[0].mean(axis=0) if arr.ndim == 3 else arr[0]
    return [float(x) for x in vec]


_POLICY_HEAD: list = []  # one-slot cache; [] unloaded, [None] absent


def _policy_head():
    """Feature 13's logistic head (Appendix A), fitted by scripts/bootstrap.py.

    It is NOT an ONNX model, which is why `_is_onnx_model_available("policy")`
    never found it and the feature sat at `unavailable` on every request.
    """
    if not _POLICY_HEAD:
        path = _onnx_root().parent / "policy_head.joblib"
        try:
            import joblib

            _POLICY_HEAD.append(joblib.load(path) if path.exists() else None)
        except Exception:
            _POLICY_HEAD.append(None)
    return _POLICY_HEAD[0]


def _score_policy(span: str) -> float | None:
    """Feature 13. P(policy breach) from the head over a MiniLM embedding."""
    head = _policy_head()
    if head is None:
        return None
    emb = _embed(span)
    if emb is None:
        return None
    return float(head.predict_proba([emb])[0][1])


def _score_injection(span: str) -> float | None:
    """Feature 12. Sequence classification: P(injection) over the whole span."""
    logits = _run("prompt_injection", span)
    if logits is None:
        return None
    probs = _softmax([float(v) for v in logits[0]])
    # protectai's label 1 is INJECTION. A binary head with the positive class
    # last is the convention; guard anyway so a relabelled export cannot
    # silently invert the meaning of the score.
    return float(probs[-1]) if len(probs) >= 2 else None


def _pii_label_ids(name: str = "pii") -> list[int]:
    """Which NER labels count as PERSONAL data, read from the export.

    Only PER and LOC. `dslim/bert-base-NER` also emits ORG and MISC, and
    counting those would fire `no_pii_egress` on every reply that names the
    company or quotes an order id -- blocking more traffic than having no
    encoder at all, for text containing no personal data whatsoever. An
    organisation is not a person.

    Labels are read from config.json rather than hard-coded, because a
    re-export with a different label order would otherwise silently change
    which entities count as PII.
    """
    if name not in _LABELS:
        path = _model_path(name)
        if path is None:
            return []
        import json

        cfg = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        id2label = cfg.get("id2label", {})
        _LABELS[name] = {int(k): v for k, v in id2label.items()}
    return [i for i, lab in _LABELS.get(name, {}).items() if lab.split("-")[-1] in ("PER", "LOC")]


def _score_pii_ner(span: str) -> float | None:
    """Feature 11, the NER half. Token classification over `dslim/bert-base-NER`.

    The span score is the strongest personal-entity token, matching the `max`
    over spans Appendix A specifies for the response as a whole.
    """
    wanted = _pii_label_ids()
    if not wanted:
        return None
    logits = _run("pii", span)
    if logits is None:
        return None
    best = 0.0
    for token_logits in logits[0]:
        probs = _softmax([float(v) for v in token_logits])
        best = max(best, sum(probs[i] for i in wanted if i < len(probs)))
    return best


def score_spans(text: str) -> dict[str, float | None]:
    """Score the full text by breaking it into spans (12.4).

    Spans are flushed at sentence boundaries or at flush_tokens tokens.
    Per-response value is the max over spans.
    """
    # Simple span splitting: sentences, then chunks of flush_tokens words.
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    if not sentences:
        sentences = [text]

    flush_tokens = config._cfg["tier1"]["flush_tokens"]
    spans = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(0, max(1, len(words)), flush_tokens):
            spans.append(" ".join(words[i : i + flush_tokens]))

    if not spans:
        spans = [""]

    # If ONNX models exist, initialize max_scores with 0.0 instead of None (7.6)
    pii_avail = _is_onnx_model_available("pii")
    # Feature 13 needs BOTH halves runnable: the head, and the MiniLM encoder
    # that produces its input. Either missing is `unavailable`, never 0.0.
    pol_avail = _policy_head() is not None and _is_onnx_model_available("minilm")

    # `injection_score` is NOT here: it is scored over the customer's message,
    # whole text, by `score_request` below.
    max_scores: dict[str, float | None] = {
        "pii_score": 0.0 if pii_avail else None,
        "policy_score": 0.0 if pol_avail else None,
        "secret_score": 0.0,
    }

    # Features 11-13 need the ONNX sessions (scripts/export_onnx.py). While the
    # files are absent, _is_onnx_model_available is False and each returns
    # None = `unavailable`, which is the CORRECT sentinel: the invariant then
    # fails closed (10.2) rather than reading a missing encoder as clean.
    for span in spans:
        max_scores["secret_score"] = max(max_scores["secret_score"] or 0.0, _score_secret(span))
        # The regex half of feature 11 runs whether or not the NER model exists.
        hit = _score_pii_regex(span)
        if hit is not None:
            max_scores["pii_score"] = max(max_scores["pii_score"] or 0.0, hit)

        # And the model halves actually run. An `avail` flag that does not lead
        # to a forward pass is a score of 0.0 dressed as a measurement.
        if pii_avail:
            ner = _score_pii_ner(span)
            if ner is None:  # session died mid-flight
                max_scores["pii_score"] = None
            elif max_scores["pii_score"] is not None:
                max_scores["pii_score"] = max(max_scores["pii_score"], ner)

    # Feature 13 is scored ONCE over the whole response, not per span (App. A).
    # The head is fitted on whole-response embeddings, so per-span serving would
    # be train/serve skew -- and it cost ~23 ms of p50, about half the layer's
    # total, by running MiniLM N times where once is enough.
    if pol_avail:
        max_scores["policy_score"] = _score_policy(text)

    return max_scores


def score_request(text: str) -> dict[str, float | None]:
    """Feature 12, over the CUSTOMER'S MESSAGE, as whole text (Appendix A).

    Deliberately not part of `score_spans`, and deliberately not span-max.

    Features 11 and 14 ask "does this OUTPUT leak something", so they scan the
    reply span by span as it streams. Feature 12 asks "was this agent TOLD to
    misbehave", and 23's adversarial scenario puts the injection in the
    customer's message. The encoder is trained on inputs, so scoring an
    assistant reply is out-of-distribution -- a reply stating the action it took
    reads like instruction text and scores 0.93.

    Span-max made it far worse. Max over N spans compounds a per-span
    false-positive rate into a near-certainty, so on long text it detects
    LENGTH. Measured over the 1,466-turn corpus: reply span-max caught 99% of
    injections and blocked 79% of honest traffic; message whole-text blocks 0%.
    On the 10.2 fixed path a false positive is unrecoverable by any record, so
    the 79% was an outage wearing a guard's uniform.
    """
    if not _is_onnx_model_available("prompt_injection"):
        return {"injection_score": None}
    return {"injection_score": _score_injection(text)}
