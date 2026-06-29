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
async def index(
    request: Request,
    agent: str | None = None,
    status: str | None = None,
    meta_key: str | None = None,
    meta_value: str | None = None,
):
    metadata = None
    if meta_key and meta_value:
        metadata = {meta_key: meta_value}

    sessions = storage.list_sessions(
        agent_name=agent, status=status, metadata=metadata, limit=100
    )
    stats = storage.get_stats(agent_name=agent)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sessions": sessions,
        "stats": stats,
        "filter_agent": agent or "",
        "filter_status": status or "",
        "filter_meta_key": meta_key or "",
        "filter_meta_value": meta_value or "",
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
    cost_by_agent = storage.get_cost_by_agent()
    daily_labels, daily_values = storage.get_daily_costs()
    status_counts = {
        "completed": stats["completed"],
        "tripped": stats["trips"],
        "error": stats["errors"],
    }

    return templates.TemplateResponse("stats.html", {
        "request": request,
        "stats": stats,
        "cost_by_agent": cost_by_agent,
        "status_counts": status_counts,
        "daily_labels": daily_labels,
        "daily_values": daily_values,
    })
