from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agentguard.storage import Storage

_DIR = Path(__file__).parent
app = FastAPI(title="AgentGuard Dashboard")
app.mount("/static", StaticFiles(directory=_DIR / "static"), name="static")
templates = Jinja2Templates(directory=_DIR / "templates")

storage = Storage()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, agent: str | None = None, status: str | None = None):
    sessions = storage.list_sessions(agent_name=agent, status=status, limit=100)
    stats = storage.get_stats(agent_name=agent)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sessions": sessions,
        "stats": stats,
        "filter_agent": agent or "",
        "filter_status": status or "",
    })


@app.get("/session/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: str):
    record = storage.get_session(session_id)
    if not record:
        return HTMLResponse("<h1>Session not found</h1>", status_code=404)
    return templates.TemplateResponse("session.html", {
        "request": request,
        "session": record,
    })


@app.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    stats = storage.get_stats()
    sessions = storage.list_sessions(limit=500)

    cost_by_agent: dict[str, float] = {}
    status_counts = {"completed": 0, "tripped": 0, "error": 0}
    daily_costs: dict[str, float] = {}

    for s in sessions:
        agent = s["agent_name"]
        cost_by_agent[agent] = cost_by_agent.get(agent, 0) + (s["total_cost_usd"] or 0)

        st = s["status"]
        if st in status_counts:
            status_counts[st] += 1

        day = s["started_at"][:10]
        daily_costs[day] = daily_costs.get(day, 0) + (s["total_cost_usd"] or 0)

    sorted_days = sorted(daily_costs.keys())
    daily_labels = sorted_days
    daily_values = [daily_costs[d] for d in sorted_days]

    return templates.TemplateResponse("stats.html", {
        "request": request,
        "stats": stats,
        "cost_by_agent": cost_by_agent,
        "status_counts": status_counts,
        "daily_labels": daily_labels,
        "daily_values": daily_values,
    })
