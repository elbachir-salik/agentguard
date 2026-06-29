"""FastAPI agent example (no API key — mock LLM in route).

Run:
    pip install agentguard[fastapi]
    uvicorn examples.fastapi_agent:app --reload
"""

from fastapi import Depends, FastAPI

from agentguard import Guard
from agentguard.integrations.fastapi import create_session_dependency, setup_agentguard
from agentguard.session import Session

guard = Guard(agent_name="fastapi-agent", max_turns=20, max_cost=1.00)

app = FastAPI(title="AgentGuard FastAPI Example")
setup_agentguard(app, guard)
SessionDep = create_session_dependency()


class _MockUsage:
    prompt_tokens = 10
    completion_tokens = 5


class _MockMessage:
    content = "Hello from the FastAPI agent!"
    tool_calls = None


class _MockChoice:
    message = _MockMessage()


class _MockResponse:
    __module__ = "openai.types.chat"
    usage = _MockUsage()
    model = "gpt-4o"
    choices = [_MockChoice()]


@app.post("/chat")
async def chat(session: Session = Depends(SessionDep)):
    session.call(lambda: _MockResponse())
    summary = session.summary()
    return {
        "reply": "Hello from the FastAPI agent!",
        "session_id": summary["session_id"],
        "turns": summary["turns"],
        "cost_usd": summary["total_cost_usd"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
