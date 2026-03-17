"""
Tommy OS Agent - resume-worthy starter project

What this is:
- A small FastAPI service that exposes a personal AI chief-of-staff agent
- Uses OpenAI Responses API via the official OpenAI Python SDK
- Supports four modes:
  1) daily_brief
  2) weekly_reset
  3) decision_mode
  4) overwhelm_mode
- Saves lightweight run history locally for evaluation/debugging

How to run:
1) python -m venv .venv
2) source .venv/bin/activate   # On Windows: .venv\Scripts\activate
3) pip install fastapi uvicorn openai pydantic python-dotenv
4) export OPENAI_API_KEY=your_key_here
5) uvicorn tommy_os_agent:app --reload
6) Open http://127.0.0.1:8000/docs

Notes:
- This is intentionally simple so you can understand it and extend it.
- Good next steps are adding auth, calendar integrations, Notion/Google Tasks sync,
  observability, evaluation, and a small frontend.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI


load_dotenv()

app = FastAPI(
    title="Tommy OS Agent",
    description="A personal chief-of-staff AI agent for planning, prioritization, and decision support.",
    version="0.1.0",
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
DATA_DIR = Path(os.getenv("TOMMY_OS_DATA_DIR", ".tommy_os_data"))
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "runs.jsonl"


Mode = Literal["daily_brief", "weekly_reset", "decision_mode", "overwhelm_mode"]


class AgentRequest(BaseModel):
    mode: Mode = Field(description="Which workflow the agent should run")
    today_date: Optional[str] = Field(default=None, description="Date string for context")
    energy_level: Optional[int] = Field(default=None, ge=1, le=10)
    mood: Optional[str] = None
    main_schedule: Optional[str] = None
    top_current_tasks: Optional[list[str]] = None
    stressors: Optional[str] = None
    priority_now: Optional[str] = None
    time_sensitive: Optional[str] = None
    avoiding: Optional[str] = None
    major_events: Optional[str] = None
    main_deadlines: Optional[str] = None
    work_goals: Optional[str] = None
    creative_goals: Optional[str] = None
    relationship_goals: Optional[str] = None
    financial_priorities: Optional[str] = None
    health_priorities: Optional[str] = None
    messy_right_now: Optional[str] = None
    desired_week_feel: Optional[str] = None
    decision_to_make: Optional[str] = None
    options: Optional[list[str]] = None
    desired_outcome: Optional[str] = None
    concerns: Optional[str] = None
    short_term_impact: Optional[str] = None
    long_term_impact: Optional[str] = None
    gut_feeling: Optional[str] = None
    brain_dump: Optional[str] = None
    user_context: Optional[str] = Field(
        default=None,
        description="Persistent personal context, goals, constraints, and preferences",
    )


class AgentResponse(BaseModel):
    mode: Mode
    output: str
    created_at_utc: str


SYSTEM_PROMPT = """
You are Tommy OS, a personal chief-of-staff AI agent.

Your job:
Help the user manage life across these categories:
1. Career / work
2. Creative / music / brand
3. Relationship / family / wedding / personal logistics
4. Finances / bills / debt / major spending decisions
5. Health / energy / fitness
6. Emotional clarity / reflection / mental load

Operating principles:
- Be practical, structured, concise, and high-signal.
- Reduce overwhelm; do not create more of it.
- Separate urgent from important.
- Call out tradeoffs, bottlenecks, and hidden risks.
- Favor realistic progress over idealized plans.
- If the user is overloaded, reduce scope.
- If the user is avoiding something important, say so directly but respectfully.
- Do not use generic motivational fluff.

Formatting rules:
- Use short headers.
- Use bullets sparingly and only when useful.
- Make recommendations specific and actionable.
- Never recommend more than 5 action items unless explicitly asked.

For each mode, return the following:
- daily_brief: Top 3 Priorities, Watch-Outs, Game Plan, Can Wait, Simplify This, Mindset
- weekly_reset: Weekly Priorities, Bottlenecks, Focus by Category, Ignore for Now, Theme of the Week
- decision_mode: Best Option, Why, Tradeoffs, Blind Spots, Recommended Next Move
- overwhelm_mode: What Actually Matters, First 3 Moves, Can Wait, What You're Carrying That Isn't Urgent, Reset Thought
""".strip()


def _build_user_prompt(req: AgentRequest) -> str:
    base_context = req.user_context or "No additional saved user context provided."

    if req.mode == "daily_brief":
        return f"""
MODE: daily_brief

User context:
{base_context}

Daily inputs:
- Today/Date: {req.today_date}
- Energy level: {req.energy_level}
- Mood: {req.mood}
- Main schedule: {req.main_schedule}
- Top current tasks: {req.top_current_tasks}
- Stressors: {req.stressors}
- What matters most right now: {req.priority_now}
- Anything time-sensitive: {req.time_sensitive}
- Anything being avoided: {req.avoiding}
""".strip()

    if req.mode == "weekly_reset":
        return f"""
MODE: weekly_reset

User context:
{base_context}

Weekly inputs:
- Week of: {req.today_date}
- Major events: {req.major_events}
- Main deadlines: {req.main_deadlines}
- Work goals: {req.work_goals}
- Creative goals: {req.creative_goals}
- Relationship/personal goals: {req.relationship_goals}
- Financial priorities: {req.financial_priorities}
- Health priorities: {req.health_priorities}
- What feels messy right now: {req.messy_right_now}
- What I want this week to feel like: {req.desired_week_feel}
""".strip()

    if req.mode == "decision_mode":
        return f"""
MODE: decision_mode

User context:
{base_context}

Decision inputs:
- Decision: {req.decision_to_make}
- Options: {req.options}
- Desired outcome: {req.desired_outcome}
- Concerns: {req.concerns}
- Short-term impact: {req.short_term_impact}
- Long-term impact: {req.long_term_impact}
- Gut feeling: {req.gut_feeling}
""".strip()

    if req.mode == "overwhelm_mode":
        return f"""
MODE: overwhelm_mode

User context:
{base_context}

Brain dump:
{req.brain_dump}
""".strip()

    raise ValueError(f"Unsupported mode: {req.mode}")


def _save_run(request_payload: dict, response_text: str) -> None:
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": request_payload,
        "response": response_text,
    }
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": "tommy-os-agent",
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "history_file": str(HISTORY_FILE),
    }


@app.post("/agent/run", response_model=AgentResponse)
def run_agent(req: AgentRequest) -> AgentResponse:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set.")

    user_prompt = _build_user_prompt(req)

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI API call failed: {exc}") from exc

    # The SDK's unified text accessor is convenient when available.
    output_text = getattr(response, "output_text", None)
    if not output_text:
        try:
            output_text = str(response)
        except Exception:
            output_text = "No text output returned."

    _save_run(req.model_dump(), output_text)

    return AgentResponse(
        mode=req.mode,
        output=output_text,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/runs/recent")
def recent_runs(limit: int = 5) -> dict:
    if not HISTORY_FILE.exists():
        return {"runs": []}

    lines = HISTORY_FILE.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines[-max(1, limit):]]
    return {"runs": records}


if __name__ == "__main__":
    # Optional local dev entry point.
    import uvicorn

    uvicorn.run("tommy_os_agent:app", host="127.0.0.1", port=8000, reload=True)
