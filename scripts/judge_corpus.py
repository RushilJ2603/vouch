#!/usr/bin/env python3
"""Judge the corpus against policy using the Tier 2 model."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from vouch import config  # noqa: E402

# The SAME schema and system prompt the live judge serves from. Feature 15 is
# FITTED here and SERVED from sensors/tier2.py, so a second copy of either would
# fit calibrator B on one distribution and serve it from another (dead_ends.md).
from vouch.sensors.tier2 import JUDGE_TOOL, SYSTEM  # noqa: E402


class JudgeResponse(BaseModel):
    p_wrong: float = Field(..., ge=0.0, le=1.0)
    verdict: str
    reasons: list[str]

GLM_IN_PER_MTOK, GLM_OUT_PER_MTOK = 0.60, 2.20      # Z.ai list price

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus" / "corpus_v1.jsonl"
JUDGE_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus" / "judge_v1.jsonl"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-spend", type=float, default=0.50,
                        help="hard ceiling in USD; refuses to start above it and "
                             "stops mid-run if the running total reaches it")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        print("No corpus to judge.")
        return

    # Load corpus
    corpus = []
    with open(CORPUS_PATH) as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))

    # Resumability
    seen_ids = set()
    if JUDGE_PATH.exists():
        with open(JUDGE_PATH) as f:
            for line in f:
                if line.strip():
                    seen_ids.add(json.loads(line)["turn_id"])

    # Stratified sample
    bands = {"0-2k": [], "2k-10k": [], "10k-50k": [], "50k+": []}
    for row in corpus:
        if row["turn_id"] in seen_ids:
            continue
        amt = row.get("action", {}).get("amount_paise", 0) / 100.0
        try:
            band = config.band_for(amt)
            bands[band].append(row)
        except ValueError:
            pass # blocked

    to_judge = []
    # Try to pick limit/4 from each band
    target_per_band = args.limit // 4
    rng = random.Random(42)                 # seeded once; the sample is reproducible
    for _band_name, rows in bands.items():
        to_judge.extend(rng.sample(rows, min(target_per_band, len(rows))))

    # Top up from whatever is left if a band could not fill its quota.
    chosen = {r["turn_id"] for r in to_judge}
    shortfall = args.limit - len(to_judge)
    if shortfall > 0:
        remaining = [r for r in corpus
                     if r["turn_id"] not in seen_ids and r["turn_id"] not in chosen]
        to_judge.extend(rng.sample(remaining, min(shortfall, len(remaining))))

    if not to_judge:
        print("Nothing to judge.")
        return

    # GLM-4.7 list price, NOT DeepSeek's. Using the agent's rates here
    # understates the judge line by roughly 3x.
    global GLM_IN_PER_MTOK, GLM_OUT_PER_MTOK
    est_input = sum(len(r.get("prompt", "")) + len(r.get("reply_text", ""))
                    for r in to_judge) // 4 + len(to_judge) * 120
    est_output = len(to_judge) * 110
    total_cost = (est_input / 1e6) * GLM_IN_PER_MTOK + (est_output / 1e6) * GLM_OUT_PER_MTOK

    print(f"Plan:       {len(to_judge)} turns to judge with {config.TIER2_MODEL} "
          f"(thinking={config.TIER2_THINKING})")
    print(f"Est tokens: {est_input:,} in / {est_output:,} out")
    print(f"Est cost:   ${total_cost:.4f}  (~Rs{total_cost * 88:.2f})")

    # ── Hard spend cap ─────────────────────────────────────────────────────
    # An estimate printed before the run is not a guardrail. This refuses to
    # start above the ceiling and re-checks after every call from REPORTED
    # usage, so a prompt that turns out longer than estimated stops the run
    # rather than quietly overspending.
    if total_cost > args.max_spend:
        print(f"\nREFUSING: estimated ${total_cost:.3f} exceeds --max-spend "
              f"${args.max_spend:.2f}. Lower --limit or raise the cap deliberately.")
        return 1

    if args.dry_run:
        print("Dry run, exiting.")
        if to_judge:
            print("\nSample turn to judge:")
            print(to_judge[0]["prompt"])
        return

    if not args.yes:
        ans = input("Proceed? [y/N] ")
        if ans.lower() != "y":
            return

    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("ZAI_API_KEY", "dummy"),
        base_url=config._cfg["tier2"]["base_url"],
    )

    JUDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    violations = 0
    JUDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_PATH, "a") as f:
        for i, row in enumerate(to_judge):
            if spent >= args.max_spend:
                print(f"\nSTOPPED at {i}/{len(to_judge)}: spend cap "
                      f"${args.max_spend:.2f} reached (${spent:.4f} actual). "
                      f"Re-run to continue -- the file is resumable.")
                break
            prompt = (
                f"Review this agent response against policy.\n"
                f"Prompt: {row['prompt']}\n"
                f"Reply: {row['reply_text']}\n"
                f"Action: {row['action']}\n"
            )
            response = client.chat.completions.create(
                model=config.TIER2_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                tools=[JUDGE_TOOL],
                tool_choice="auto",
                extra_body={"thinking": {"type": config.TIER2_THINKING}}
            )
            usage = getattr(response, "usage", None)
            if usage:
                spent += (usage.prompt_tokens / 1e6) * GLM_IN_PER_MTOK
                spent += (usage.completion_tokens / 1e6) * GLM_OUT_PER_MTOK
            # 21.3 counts schema violations rather than crashing on one. A judge
            # that answers in prose is `unavailable`, never a defaulted number.
            calls = getattr(response.choices[0].message, "tool_calls", None)
            if not calls:
                violations += 1
                continue
            try:
                parsed = JudgeResponse.model_validate_json(calls[0].function.arguments)
            except ValidationError:
                violations += 1
                continue
            out = {
                "turn_id": row["turn_id"],
                "model": config.TIER2_MODEL,
                "thinking": config.TIER2_THINKING,
                "p_wrong": parsed.p_wrong,
                "verdict": parsed.verdict,
                "reasons": parsed.reasons,
            }
            f.write(json.dumps(out) + "\n")
            f.flush()

    print(f"\nActual spend this run: ${spent:.4f}")
    judged = len(to_judge) - violations
    rate = violations / len(to_judge) if to_judge else 0.0
    print(f"Schema violations: {violations}/{len(to_judge)} ({rate:.1%}) "
          f"-- 21.3 allows <= 1 in 50; judged {judged}")

if __name__ == "__main__":
    main()
