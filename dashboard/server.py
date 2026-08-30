"""The judge-facing Vouch surface (6.4).

Evidence routes are read-only. An explicit live run generates one response and
appends its result through the same proxy log path; policies, outcomes, prior
decisions and trust rows cannot be edited here.

No new dependency. FastAPI and uvicorn are already pinned for the proxy (2.5),
so this costs the serving image nothing, and 6.1's rule about keeping that image
lean stays intact.

    python scripts/run_showcase.py         # http://127.0.0.1:8501

The evidence page needs no key. Provider calls occur only after a live click.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
sys.path.insert(0, str(ROOT / "src"))

import data  # noqa: E402
from vouch import agent, config  # noqa: E402
from vouch.proxy import app as control  # noqa: E402

PAGE = Path(__file__).resolve().parent / "page.html"
REPORTS = ROOT / "reports"

LIVE_LOCK = asyncio.Lock()
START_LOCK = asyncio.Lock()
CONTROL_STARTED = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global CONTROL_STARTED
    try:
        yield
    finally:
        if CONTROL_STARTED:
            await control.shutdown()
            CONTROL_STARTED = False


app = FastAPI(title="Vouch dashboard", version="0.1.0", lifespan=lifespan)


class LiveRequest(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=64)
    policy_mode: str = Field(default="adaptive", pattern="^(fast|adaptive|deep)$")


async def _ensure_control_started() -> None:
    global CONTROL_STARTED
    if CONTROL_STARTED:
        return
    async with START_LOCK:
        if not CONTROL_STARTED:
            await control.startup()
            CONTROL_STARTED = True


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(PAGE.read_text(encoding="utf-8"))


@app.get("/api/state")
async def state() -> JSONResponse:
    return JSONResponse(data.payload())


@app.get("/api/live/scenarios")
async def scenarios() -> JSONResponse:
    return JSONResponse(data.live_scenarios())


@app.get("/api/live/status")
async def live_status() -> JSONResponse:
    return JSONResponse({
        "ready": bool(os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("ZAI_API_KEY")),
        "agent": {"model": config.AGENT_MODEL,
                  "configured": bool(os.environ.get("DEEPSEEK_API_KEY"))},
        "judge": {"model": config.TIER2_MODEL,
                  "configured": bool(os.environ.get("ZAI_API_KEY"))},
        "policy_modes": list(config.load_policies()),
    })


@app.post("/api/live/evaluate")
async def live_evaluate(request: LiveRequest) -> JSONResponse:
    row = data.resolve_live_scenario(request.scenario_id)
    if row is None:
        raise HTTPException(404, "scenario is not in the curated live set")

    missing = [name for name in ("DEEPSEEK_API_KEY", "ZAI_API_KEY")
               if not os.environ.get(name)]
    if missing:
        raise HTTPException(503, f"server is missing {', '.join(missing)}")

    try:
        await asyncio.wait_for(LIVE_LOCK.acquire(), timeout=0.01)
    except asyncio.TimeoutError as exc:
        raise HTTPException(409, "a live evaluation is already running") from exc

    try:
        started = time.perf_counter()
        try:
            generated = await agent.generate(row["prompt"])
        except agent.AgentUnavailable as exc:
            raise HTTPException(502, str(exc)) from exc
        generation_ms = (time.perf_counter() - started) * 1000.0

        action = generated.action.model_dump()
        await _ensure_control_started()
        req = control.ChatRequest(
            model=config.AGENT_MODEL,
            messages=[
                {"role": "system", "content": agent.SYSTEM},
                {"role": "user", "content": row["prompt"]},
            ],
            tools=[agent.RESPOND_TOOL],
            vouch=control.VouchBlock(
                agent_id="agent_corpus",
                user_scope=[f"orders:{action['order_id']}"],
                policy_mode=request.policy_mode,
                action_hint=action["type"],
                amount_paise=action["amount_paise"],
                claims=generated.claims.model_dump(),
                retrieved_chunks=row.get("retrieved_chunks") or [],
                tool_calls=[],
            ),
        )
        completion = await control.evaluate_response(req, generated.reply_text)
        decision = completion["vouch"]
        return JSONResponse({
            "scenario_id": request.scenario_id,
            "reply_text": generated.reply_text,
            "claims": generated.claims.model_dump(),
            "action": action,
            "vouch": decision,
            "providers": {
                "agent": config.AGENT_MODEL,
                "judge": (
                    config.TIER2_MODEL
                    if decision["tier_reached"] == 2 and not decision["degraded_mode"]
                    else None
                ),
                "judge_attempted": (
                    config.TIER2_MODEL if decision["tier_reached"] == 2 else None
                ),
            },
            "generation_ms": generation_ms,
        })
    finally:
        LIVE_LOCK.release()


@app.get("/charts/{name}")
async def chart(name: str) -> FileResponse:
    # Resolved and confined: the chart directory is the only readable root, so
    # a crafted name cannot walk out of it.
    path = (REPORTS / name).resolve()
    if path.suffix != ".png" or REPORTS.resolve() not in path.parents:
        raise HTTPException(404, "no such chart")
    if not path.exists():
        raise HTTPException(404, "chart not generated yet -- run make demo")
    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="warning")
