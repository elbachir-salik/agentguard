# AgentGuard

[![CI](https://github.com/salikelbachir/agentguard/actions/workflows/ci.yml/badge.svg)](https://github.com/salikelbachir/agentguard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**The black box + circuit breaker for AI agents.**

Wrap any LLM call in 3 lines. Record every turn. Catch loops, budget overruns, and timeouts before they cause damage. Replay any session. Browse it all in a local dashboard. No cloud. No API keys. Just `pip install agentguard`.

```python
from agentguard import Guard

guard = Guard(agent_name="my-agent", max_cost=5.00, max_turns=20, max_tool_retries=3, timeout=60)

with guard.session() as session:
    response = session.call(client.chat.completions.create, model="gpt-4o", messages=messages)
```

That's it. Every call is recorded, costs are tracked, and if your agent loops or burns through budget — the circuit breaker stops it.

---

## Why

Agents in production fail silently. They don't crash — they loop, hallucinate, burn tokens, and go off-scope. By the time you notice, the damage is done.

Existing tools are either enterprise SaaS (Datadog, AgentOps — cloud, API keys, $$$) or narrowly focused (just a breaker, no recording). Nobody built a self-hosted SDK that combines **recording + protection + dashboard** in one `pip install`.

---

## Install

```bash
pip install agentguard
```

For the local dashboard:

```bash
pip install agentguard[dashboard]
```

Or from source:

```bash
git clone https://github.com/salikelbachir/agentguard.git
cd agentguard
pip install -e ".[dashboard]"
```

For LangChain integration:

```bash
pip install agentguard[langchain]
```

For the OpenAI Python SDK wrapper:

```bash
pip install agentguard[openai]
```

For FastAPI integration:

```bash
pip install agentguard[fastapi]
```

---

## OpenAI Python SDK

Wrap your existing `OpenAI` client — `chat.completions.create` is guarded automatically:

```python
from openai import OpenAI
from agentguard import Guard
from agentguard.integrations.openai import guard_openai

guard = Guard(agent_name="my-agent", max_cost=1.00, max_turns=15)
client = OpenAI()

with guard_openai(guard, client) as (session, guarded):
    response = guarded.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(session.summary())
```

Works with sync and async clients (`AsyncOpenAI`), including streaming (`stream=True`).

See [`examples/openai_guarded.py`](examples/openai_guarded.py) for a no-API-key mock demo.

---

## FastAPI

One Guard session per HTTP request via a dependency. Trips return **HTTP 429** automatically.

```python
from fastapi import Depends, FastAPI
from agentguard import Guard
from agentguard.integrations.fastapi import create_session_dependency, setup_agentguard
from agentguard.session import Session

guard = Guard(agent_name="api-agent", max_turns=20, max_cost=1.00)

app = FastAPI()
setup_agentguard(app, guard)
SessionDep = create_session_dependency()

@app.post("/chat")
async def chat(session: Session = Depends(SessionDep)):
    response = session.call(client.chat.completions.create, model="gpt-4o", messages=[...])
    return {"session_id": session.record.session_id, "turns": session.summary()["turns"]}
```

Install: `pip install agentguard[fastapi]`

Each request gets metadata (`path`, `method`, `client_host`) for filtering in the dashboard/CLI.

See [`examples/fastapi_agent.py`](examples/fastapi_agent.py) — run with `uvicorn examples.fastapi_agent:app --reload`.

---

## Docker

Build and run the dashboard in a container:

```bash
docker build -t agentguard .
docker run -p 8585:8585 -v agentguard-data:/data agentguard
```

Open [http://localhost:8585](http://localhost:8585). Session data is persisted in the `agentguard-data` volume at `/data/agentguard.db`.

---

## LangChain / LangGraph

Use AgentGuard with LangChain via a callback handler — pass it to any chat model:

```python
from agentguard import Guard
from agentguard.integrations.langchain import guard_session
from langchain_openai import ChatOpenAI

guard = Guard(agent_name="my-agent", max_turns=20, max_tool_retries=3)

with guard_session(guard) as (session, callbacks):
    llm = ChatOpenAI(model="gpt-4o", callbacks=callbacks)
    response = llm.invoke("Hello!")
    print(session.summary())
```

For LangGraph, pass the same callback list to the chat model inside your graph node.

See [`examples/langchain_basic.py`](examples/langchain_basic.py) for a no-API-key demo using `FakeListChatModel`.

---

## What You Get

### 1. The Black Box (Flight Recorder)

Every LLM call is recorded: input, output, tokens, cost, tool calls, latency. Stored locally in SQLite at `~/.agentguard/agentguard.db`.

### 2. The Circuit Breaker

Real-time protection with composable rules:

| Rule | What it catches | Config |
|------|----------------|--------|
| **Budget** | Cost or token limit exceeded | `max_cost=5.00`, `max_tokens=100000` |
| **Budget warning** | Soft alert before hard trip (no stop) | `warn_cost=4.00`, `warn_pct=0.8`, `on_warn=...` |
| **Turns** | Too many LLM calls in one session | `max_turns=20` |
| **Loop Detection** | Same tool called repeatedly with similar input | `max_tool_retries=3` |
| **Timeout** | Session running too long | `timeout=60` |
| **Scope** | Unauthorized tool usage | `allowed_tools=[...]`, `blocked_tools=[...]` |

When a rule trips, AgentGuard raises `CircuitBreakerTripped` and saves the session with full context.

Budget and turn limits allow the current LLM call to finish before tripping — the trip fires on the post-call check.

### Streaming (OpenAI-compatible)

Wrap streaming completions with `session.stream()`. Chunks pass through transparently; the turn is recorded after the stream finishes:

```python
with guard.session() as session:
    stream = session.stream(
        client.chat.completions.create,
        model="gpt-4o",
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="")
```

Use `stream_options={"include_usage": True}` so token counts are captured on the final chunk.

### Async

For async clients (e.g. `AsyncOpenAI`), use `acall` and `astream` inside a normal `with guard.session()` block:

```python
with guard.session() as session:
    response = await session.acall(
        client.chat.completions.create,
        model="gpt-4o",
        messages=messages,
    )

    stream = await session.astream(
        client.chat.completions.create,
        model="gpt-4o",
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="")
```

### 3. CLI

```bash
agentguard sessions                        # list all recorded sessions
agentguard sessions --meta env=staging     # filter by metadata
agentguard replay <session_id>             # replay a session turn by turn
agentguard export <session_id>             # export session as JSON
agentguard stats                           # cost and usage summary
agentguard dashboard                       # local web UI at localhost:8585
```

### 4. Local Dashboard

A dark-themed web UI to browse sessions, replay turns, and view cost/usage charts. FastAPI + Jinja2 + Chart.js. No React, no build step.

```bash
agentguard dashboard
# open http://localhost:8585
```

---

## Usage

### Basic

```python
from agentguard import Guard

guard = Guard(agent_name="support-bot", max_cost=1.00, max_turns=15)

with guard.session() as session:
    response = session.call(
        client.chat.completions.create,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Refund order #4521"}],
    )
    print(session.summary())
```

### Catching a Trip

```python
from agentguard import Guard, CircuitBreakerTripped

guard = Guard(agent_name="my-agent", max_cost=0.10, max_tool_retries=3)

try:
    with guard.session() as session:
        while True:
            response = session.call(client.chat.completions.create, ...)
            # ... handle tool calls, loop back ...
except CircuitBreakerTripped as e:
    print(f"Stopped: {e}")
    # Session is saved automatically with status="tripped"
```

### Custom Rules

```python
from agentguard.rules.base import BaseRule, RuleResult, SessionState

class NoPIIRule(BaseRule):
    name = "no_pii"

    def check(self, state: SessionState) -> RuleResult:
        if state.turns:
            content = str(state.turns[-1].output.get("content", ""))
            if "SSN" in content or "social security" in content.lower():
                return RuleResult.trip("PII detected in output")
        return RuleResult.passed()

guard = Guard(agent_name="safe-bot", rules=[NoPIIRule()])
```

### Session Metadata

Attach labels to sessions for filtering and debugging:

```python
with guard.session(metadata={"customer_id": "4521", "env": "staging", "ticket": "1234"}) as session:
    response = session.call(client.chat.completions.create, ...)
```

Filter in CLI:

```bash
agentguard sessions --meta env=staging --meta customer_id=4521
```

### Multi-Agent Linking

Link child sessions to a parent when one agent spawns another. The dashboard shows the full chain (parent → children):

```python
orchestrator = Guard(agent_name="orchestrator")
worker = Guard(agent_name="worker")

with orchestrator.session() as parent:
    parent.call(plan_task, ...)

    with worker.session(parent_session_id=parent.record.session_id) as child:
        child.call(run_subtask, ...)
```

List child sessions in the CLI:

```bash
agentguard sessions --parent <parent_session_id>
```

Exported JSON includes `parent_session_id` for each session.

### Callbacks

Register hooks that fire on every turn or when the circuit breaker trips. Use them for logging, Slack alerts, metrics, or custom recovery logic:

```python
import logging
from agentguard import Guard
from agentguard.models import BreakerEvent, SessionRecord, Turn, WarnEvent

logger = logging.getLogger(__name__)

def on_turn(turn: Turn, record: SessionRecord) -> None:
    logger.info("Turn %s | $%.4f | session=%s", turn.turn_number, turn.cost_usd, record.session_id)

def on_trip(event: BreakerEvent, record: SessionRecord) -> None:
    logger.warning("Tripped [%s]: %s (session %s)", event.rule, event.trigger, record.session_id)
    # send_slack_alert(event, record)  # your integration here

def on_warn(event: WarnEvent, record: SessionRecord) -> None:
    logger.warning("Budget warning: %s (session %s)", event.trigger, record.session_id)

guard = Guard(
    agent_name="support-bot",
    max_cost=5.00,
    warn_pct=0.8,          # alert at 80% of max_cost
    on_turn=on_turn,
    on_warn=on_warn,
    on_trip=on_trip,
)
```

See [`examples/callbacks.py`](examples/callbacks.py) for a full example with logging and an optional Slack webhook (`SLACK_WEBHOOK_URL` env var).

### Export a Session

Dump a full session (turns, metadata, breaker event) as JSON for sharing or debugging:

```bash
agentguard export abc123def456
agentguard export abc123 --output session.json
```

---

## Terminal Output

**Successful session:**

```
+---------------- agentguard -- support-bot -- fdf2f69c9301 -----------------+
|   Turn 1  OK  510 tok  $0.0008  1200ms                                     |
|     -> user: "Refund order #4521"                                           |
|     <- tool_call: lookup_order({"order_id": "4521"})                        |
|                                                                             |
|   Turn 2  OK  380 tok  $0.0006  800ms                                      |
|     <- tool_call: process_refund({"order_id": "4521"})                      |
|                                                                             |
|   Turn 3  OK  290 tok  $0.0004  600ms                                      |
|     <- "Your refund of $49.99 has been processed."                          |
|                                                                             |
|   Completed -- 3 turns -- 1180 tok -- $0.0018                               |
+-----------------------------------------------------------------------------+
```

**Circuit breaker tripped:**

```
+------------ agentguard -- support-bot -- c699836f5882 ---------------------+
|   Turn 1  OK  150 tok  $0.0008  1100ms                                     |
|   Turn 2  OK  150 tok  $0.0008  900ms                                      |
|                                                                            |
|   == CIRCUIT BREAKER =========================                             |
|     Rule:    loop_detection                                                |
|     Cause:   search_kb called 3x with similar input                        |
|   ============================================                             |
|                                                                            |
|   Tripped -- 2 turns -- 300 tok -- $0.0015                                 |
+----------------------------------------------------------------------------+
```

---

## Supported Providers

AgentGuard has **zero LLM dependencies**. It wraps whatever client you already use:

- **OpenAI** — full extraction (tokens, cost, tool calls)
- **Anthropic** — full extraction (tokens, cost, tool calls)
- **Any provider** — generic extractor works with any dict-like or object response
- **NVIDIA NIM / DeepSeek** — works via OpenAI-compatible base_url

---

## Architecture

```
Your Agent Code
      |
      v
+-------------------------------------------+
|              Guard (wrapper)              |
|                                           |
|  +-------------+    +------------------+  |
|  |  Recorder   |    | Circuit Breaker  |  |
|  |  (black box)|    | (protection)     |  |
|  +------+------+    +--------+---------+  |
|         |                    |            |
|         v                    v            |
|  +-----------------------------------+   |
|  |       Storage (SQLite)             |   |
|  |  ~/.agentguard/agentguard.db       |   |
|  +----------------+------------------+   |
|                    |                      |
+--------------------+----------------------+
                     |
                     v
         +-----------------------+
         |   Dashboard (local)   |
         |   localhost:8585      |
         +-----------------------+
```

---

## License

MIT
