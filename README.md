# AgentGuard

[![CI](https://github.com/salikelbachir/agentguard/actions/workflows/ci.yml/badge.svg)](https://github.com/salikelbachir/agentguard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**The black box + circuit breaker for AI agents.**

Record every LLM turn. Trip on loops, budget overruns, and timeouts. Replay sessions locally. No cloud, no API keys.

```python
from agentguard import Guard

guard = Guard(agent_name="my-agent", max_cost=5.00, max_turns=20, max_tool_retries=3)

with guard.session() as session:
    response = session.call(client.chat.completions.create, model="gpt-4o", messages=messages)
```

---

## Features

- **Flight recorder** — input, output, tokens, cost, tool calls, latency → local SQLite
- **Circuit breaker** — budget, turns, loop detection, timeout, scope rules
- **CLI** — list, replay, export, stats
- **Dashboard** — browse sessions and cost charts at `localhost:8585`
- **Integrations** — OpenAI SDK, LangChain, FastAPI (optional extras)
- **Zero LLM deps** — wraps OpenAI, Anthropic, or any client you already use

---

## Installation

```bash
pip install agentguard
```

| Extra | Install | Includes |
|-------|---------|----------|
| Dashboard | `pip install agentguard[dashboard]` | Web UI + CLI dashboard |
| OpenAI | `pip install agentguard[openai]` | SDK wrapper |
| LangChain | `pip install agentguard[langchain]` | Callback handler |
| FastAPI | `pip install agentguard[fastapi]` | Per-request session dependency |

From source: `pip install -e ".[dashboard]"`

---

## Integrations

Pick your stack — full runnable examples live in [`examples/`](examples/).

| Stack | Quick pattern | Example |
|-------|---------------|---------|
| **Raw SDK** | `session.call(fn, ...)` | [`basic_openai.py`](examples/basic_openai.py) |
| **OpenAI SDK** | `guard_openai(guard, client)` | [`openai_guarded.py`](examples/openai_guarded.py) |
| **LangChain** | `guard_session(guard)` → callbacks | [`langchain_basic.py`](examples/langchain_basic.py) |
| **FastAPI** | `Depends(create_session_dependency())` | [`fastapi_agent.py`](examples/fastapi_agent.py) |
| **Callbacks** | `on_turn`, `on_warn`, `on_trip` | [`callbacks.py`](examples/callbacks.py) |

Streaming and async work on all paths via `session.stream()` / `session.acall()` / `session.astream()`.

---

## Circuit breaker rules

| Rule | Config | Catches |
|------|--------|---------|
| Budget | `max_cost`, `max_tokens` | Cost/token overrun |
| Budget warning | `warn_cost`, `warn_pct`, `on_warn` | Soft alert before hard trip |
| Turns | `max_turns` | Too many LLM calls |
| Loop | `max_tool_retries` | Repeated similar tool calls |
| Timeout | `timeout` | Session running too long |
| Scope | `allowed_tools`, `blocked_tools` | Unauthorized tools |

Trips raise `CircuitBreakerTripped` and save the session with full context. Custom rules: subclass `BaseRule` — see [`examples/`](examples/) or `agentguard/rules/`.

---

## CLI & dashboard

```bash
agentguard sessions                    # list sessions
agentguard sessions --meta env=staging # filter by metadata
agentguard replay <session_id>         # turn-by-turn replay
agentguard export <session_id>         # JSON export
agentguard stats                       # cost summary
agentguard dashboard                   # web UI → localhost:8585
```

Data stored at `~/.agentguard/agentguard.db` (override with `AGENTGUARD_DB_PATH`).

### Docker

```bash
docker build -t agentguard .
docker run -p 8585:8585 -v agentguard-data:/data agentguard
```

---

## Why AgentGuard?

Agents in production fail silently — they loop, burn tokens, and go off-scope. Enterprise observability tools need cloud accounts and API keys. AgentGuard is self-hosted recording + protection + dashboard in one install.

---

## License

MIT
